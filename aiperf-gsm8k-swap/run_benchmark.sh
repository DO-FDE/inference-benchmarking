#!/usr/bin/env bash
#
# run_benchmark.sh -- one self-bootstrapping entry point for the benchmark pack.
#
# Runs an aiperf concurrency sweep against an OpenAI-compatible endpoint.
# aiperf runs by DEFAULT. Two optional stages layer on top:
#
#   --prep-gsm8k        Swap aiperf's built-in corpus for GSM8K (coherent prompts
#                       -> representative speculative-decode acceptance).
#   --capture-accept    Also capture speculative-decode accept% and prefix-cache
#                       hit% by scraping /metrics around each concurrency point.
#                       Scrapes --url by default; override with --metrics-addrs
#                       only for fan-out endpoints (a router in front of N engines).
#
# The only real inputs are: endpoint URL, model name, and workload shape.
# Everything else -- dependency install, chmod, reachability checks -- is handled
# here so the script runs in any environment (bare host, container, k8s pod).
#
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- defaults ---------------------------------------------------------------
MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
TOKENIZER=""
URL="http://localhost:8000/"
URL_STRATEGY="round-robin"
API_KEY=""
ISL=120000
OSL=917
CACHE=90
CONCURRENCY="16,24,32,48,64,128"
GPUS=8
WARMUP=16
TTFT_P50=""
TTFT_P90=""
GATES=()   # each entry: NAME:ITL_P50_MS ; empty => harness defaults (claw:25, chat:66.7)
CORPUS="$HERE/gsm8k_corpus.txt"
METRICS_ADDRS=""
OUT_DIR="$HERE/results_$(date +%Y%m%d_%H%M%S)"
DO_PREP=false
DO_CAPTURE=false
SKIP_DEPS=false
SKIP_REACH=false
HARNESS="$HERE/run_aiperf_workload_shapes.sh"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  cat <<EOF

Usage:
  ./run_benchmark.sh --url http://ENDPOINT:PORT/ [options]

Core options:
  --model NAME          Served model id            (default: $MODEL)
  --tokenizer NAME      Tokenizer                  (default: --model)
  --url URL             Endpoint the sweep hits    (default: $URL)
                        Comma-separate multiple replica URLs to have the client
                        round-robin across them directly (bypassing any router),
                        e.g. --url http://p0:8000,http://p1:8000,...
  --url-strategy S      Client strategy when --url has multiple (default: round-robin)
  --api-key KEY         Bearer key                 (optional)
  --isl N               Total input tokens         (default: $ISL)
  --osl N               Output tokens              (default: $OSL)
  --cache PCT           Cached-prefix percent      (default: $CACHE)
  --concurrency LIST    Comma-separated grid       (default: $CONCURRENCY)
  --gpus N              Per-GPU divisor            (default: $GPUS)
  --warmup N            Warmup requests            (default: $WARMUP)

SLA gates (scored by parse_sweep; optional):
  --ttft-p50 MS         TTFT p50 target in ms      (default: off)
  --ttft-p90 MS         TTFT p90 target in ms      (default: off)
  --gate NAME:ITL_MS    ITL p50 gate, repeatable   (default: claw:25, chat:66.7)
                        e.g. --gate strict:9.17 --gate flex:14.28
  --out-dir PATH        Results directory          (default: results_<ts>)

Optional stages:
  --prep-gsm8k          Swap in GSM8K corpus before running (coherent prompts)
  --capture-accept      Capture accept% + cache-hit% around each point
  --metrics-addrs CSV   OPTIONAL. Only if the endpoint fans out across engines
                        whose /metrics differ from --url. Default: scrape --url.

Bootstrap controls:
  --skip-deps           Don't try to install aiperf/datasets (assume present)
  --skip-reach          Don't probe the endpoint before running
  -h, --help            Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --url|--endpoint) URL="$2"; shift 2;;
    --url-strategy) URL_STRATEGY="$2"; shift 2;;
    --api-key) API_KEY="$2"; shift 2;;
    --isl) ISL="$2"; shift 2;;
    --osl) OSL="$2"; shift 2;;
    --cache) CACHE="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --ttft-p50) TTFT_P50="$2"; shift 2;;
    --ttft-p90) TTFT_P90="$2"; shift 2;;
    --gate) GATES+=("$2"); shift 2;;
    --corpus) CORPUS="$2"; shift 2;;
    --metrics-addrs|--pod-ips) METRICS_ADDRS="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --prep-gsm8k) DO_PREP=true; shift;;
    --capture-accept) DO_CAPTURE=true; shift;;
    --skip-deps) SKIP_DEPS=true; shift;;
    --skip-reach) SKIP_REACH=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 2;;
  esac
done
[ -n "$TOKENIZER" ] || TOKENIZER="$MODEL"
mkdir -p "$OUT_DIR"

# assemble optional gate flags to forward to the harness
GATE_ARGS=()
[ -n "$TTFT_P50" ] && GATE_ARGS+=(--ttft-p50 "$TTFT_P50")
[ -n "$TTFT_P90" ] && GATE_ARGS+=(--ttft-p90 "$TTFT_P90")
for g in "${GATES[@]}"; do GATE_ARGS+=(--gate "$g"); done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---- 0. make helper scripts executable --------------------------------------
for s in run_aiperf_workload_shapes.sh prep_aiperf_corpus.sh capture_accept_cache.sh; do
  [ -f "$HERE/$s" ] && chmod +x "$HERE/$s" 2>/dev/null || true
done
[ -f "$HARNESS" ] || die "harness not found next to this script: $HARNESS
       Ensure run_aiperf_workload_shapes.sh and parse_sweep.py are in $HERE"

# ---- 1. dependency bootstrap ------------------------------------------------
# pip install helper: try several strategies so it works in venvs, system python,
# PEP-668 externally-managed envs, and rootless containers.
pip_install() {
  local pkg="$1"
  python3 -m pip install -q "$pkg" 2>/dev/null && return 0
  python3 -m pip install -q --user "$pkg" 2>/dev/null && return 0
  python3 -m pip install -q --break-system-packages "$pkg" 2>/dev/null && return 0
  python3 -m pip install -q --break-system-packages --user "$pkg" 2>/dev/null && return 0
  pip install -q "$pkg" 2>/dev/null && return 0
  return 1
}

ensure_python() {
  command -v python3 >/dev/null || die "python3 not found. Install Python 3.10+ and re-run."
  python3 -m pip --version >/dev/null 2>&1 || {
    log "pip missing; attempting ensurepip"
    python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
  }
}

ensure_aiperf() {
  command -v aiperf >/dev/null 2>&1 && return 0
  python3 -c "import aiperf" >/dev/null 2>&1 && return 0
  [ "$SKIP_DEPS" = true ] && die "aiperf not found and --skip-deps set."
  log "aiperf not found; installing..."
  pip_install aiperf || pip_install "aiperf --ignore-installed typing_extensions" || \
    die "could not install aiperf. Install it manually (pip install aiperf) or use --skip-deps."
  command -v aiperf >/dev/null 2>&1 || python3 -c "import aiperf" >/dev/null 2>&1 || \
    die "aiperf still not importable after install. Check pip output / PATH."
  log "aiperf ready"
}

ensure_datasets() {
  # only needed for --prep-gsm8k (building the GSM8K corpus)
  [ "$DO_PREP" = true ] || return 0
  python3 -c "import datasets" >/dev/null 2>&1 && return 0
  [ "$SKIP_DEPS" = true ] && die "datasets not found (needed for --prep-gsm8k) and --skip-deps set."
  log "datasets not found (needed for --prep-gsm8k); installing..."
  pip_install datasets || die "could not install datasets. pip install datasets, or drop --prep-gsm8k."
  log "datasets ready"
}

ensure_curl() {
  command -v curl >/dev/null 2>&1 && return 0
  log "WARNING: curl not found -- reachability probe and --capture-accept need it."
  [ "$DO_CAPTURE" = true ] && die "curl is required for --capture-accept. Install curl and re-run."
  SKIP_REACH=true
}

# ---- 2. reachability probe --------------------------------------------------
probe_endpoint() {
  [ "$SKIP_REACH" = true ] && { log "skipping reachability probe (--skip-reach)"; return 0; }
  command -v curl >/dev/null 2>&1 || { log "no curl; skipping probe"; return 0; }
  local base="${URL%%,*}"; base="${base%/}"; auth=()
  [ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
  log "probing endpoint: ${base}/v1/models"
  if curl -sf -m 10 "${auth[@]}" "${base}/v1/models" >/dev/null 2>&1 \
     || curl -sf -m 10 "${base}/health" >/dev/null 2>&1; then
    log "endpoint reachable"
    return 0
  fi
  echo "ERROR: cannot reach $URL from here." >&2
  echo "  - If this is a cluster-internal IP (ClusterIP / pod IP), run from inside" >&2
  echo "    the cluster, or set up a port-forward and point --url at localhost." >&2
  echo "  - If the endpoint needs auth, pass --api-key." >&2
  echo "  - To bypass this check and run anyway, add --skip-reach." >&2
  exit 1
}

probe_metrics() {
  # sanity-check that the metrics source exposes the counters, when capturing
  [ "$DO_CAPTURE" = true ] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  local src="${METRICS_ADDRS:-$URL}"
  local first="${src%%,*}"
  case "$first" in http://*|https://*) : ;; *) first="http://$first" ;; esac
  first="${first%/}"
  log "checking spec_decode counters at ${first}/metrics"
  local out
  out="$(curl -s -m 10 "${first}/metrics" 2>/dev/null | grep -E 'spec_decode_num_(draft|accepted)_tokens_total' | grep -v '^#' | head -2)"
  if [ -z "$out" ]; then
    echo "WARNING: no vllm:spec_decode_*_total counters found at ${first}/metrics." >&2
    echo "  Acceptance will report n/a. Causes: server has no draft model, /metrics" >&2
    echo "  not exposed there, or (fan-out endpoint) you must pass --metrics-addrs" >&2
    echo "  with the engines' own addresses. Continuing." >&2
  else
    log "spec_decode counters present"
  fi
}

# ---- run --------------------------------------------------------------------
echo "=============================================================="
echo " DSV4-Flash benchmark"
echo "   model       : $MODEL"
echo "   url         : $URL"
  case "$URL" in *,*) echo "   url-strategy : $URL_STRATEGY (client round-robin across replicas)";; esac
echo "   ISL/OSL     : $ISL / $OSL   (cached prefix ${CACHE}%)"
echo "   concurrency : $CONCURRENCY   gpus=$GPUS"
echo "   prep-gsm8k  : $DO_PREP"
echo "   capture     : $DO_CAPTURE${METRICS_ADDRS:+  (metrics: $METRICS_ADDRS)}"
echo "   out-dir     : $OUT_DIR"
echo "=============================================================="

ensure_python
ensure_curl
ensure_aiperf
ensure_datasets
probe_endpoint
probe_metrics

# ---- stage 1: optional GSM8K corpus swap ------------------------------------
if [ "$DO_PREP" = true ]; then
  echo; log "[prep] swapping in GSM8K corpus"
  [ -f "$HERE/prep_aiperf_corpus.sh" ] || die "prep_aiperf_corpus.sh not found in $HERE"
  ( cd "$HERE" && ./prep_aiperf_corpus.sh "$CORPUS" ) || die "prep stage failed"
fi

# ---- stage 2: the benchmark -------------------------------------------------
if [ "$DO_CAPTURE" = true ]; then
  echo; log "[run+capture] aiperf sweep with accept%/cache% capture"
  [ -f "$HERE/capture_accept_cache.sh" ] || die "capture_accept_cache.sh not found in $HERE"
  MODEL="$MODEL" URL="$URL" API_KEY="$API_KEY" \
  ${METRICS_ADDRS:+METRICS_ADDRS="$METRICS_ADDRS"} \
  ISL="$ISL" OSL="$OSL" CACHE="$CACHE" CONC="$CONCURRENCY" GPUS="$GPUS" WARMUP="$WARMUP" \
  GATE_ARGS_STR="${GATE_ARGS[*]}" \
  HARNESS="$HARNESS" OUTDIR="$OUT_DIR" \
  "$HERE/capture_accept_cache.sh"
else
  echo; log "[run] aiperf sweep (default)"
  API_ARGS=(); [ -n "$API_KEY" ] && API_ARGS=(--api-key "$API_KEY")
  ( cd "$OUT_DIR" && URL_STRATEGY="$URL_STRATEGY" "$HARNESS" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --url "$URL" "${API_ARGS[@]}" \
      --isl "$ISL" --osl "$OSL" --cache "$CACHE" \
      --concurrency "$CONCURRENCY" --gpus "$GPUS" --warmup "$WARMUP" \
      "${GATE_ARGS[@]}" \
      --skip-health-check )
  cp "$OUT_DIR"/aiperf_results/summary_ALL_*.csv "$OUT_DIR"/ 2>/dev/null || true
fi

echo; log "done. results in: $OUT_DIR"
ls -la "$OUT_DIR" 2>/dev/null | grep -E "\.csv$|accept_cache" || true