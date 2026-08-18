#!/usr/bin/env bash
#
# run_aiperf_workload_shapes.sh
#
# Concurrency sweep across the workload shapes, now across CACHE SCENARIOS.
# For each shape, the TOTAL input (8k / 64k / 256k) is split into a reused
# cached PREFIX + a fresh (incremental) remainder, per cache ratio:
#
#   ratio   prefix (cached)        fresh (incremental)     example @ 8k
#   0%      none                   full total              0     + 8000
#   75%     0.75 * total           0.25 * total            6000  + 2000
#   90%     0.90 * total           0.10 * total            7200  + 800
#
# The prefix is drawn from a pool of --num-prefix-prompts (default 8) reused
# across requests, so after warmup those tokens are cache-resident and the
# effective cache hit ratio approaches the target. The fresh remainder is the
# per-request non-cached input.
#
# ONE aiperf run per (shape x cache ratio); each run is scored against BOTH ITL
# gates (claw 25ms / chat 66.7ms) in the parse step -- the model is hit once per
# (shape,cache), not once per gate.
#
# ---------------------------------------------------------------------------
# SIMPLE MODE:
#   Instead of the full shape matrix you can describe ONE run inline:
#
#     --isl N            total input sequence length (the full prompt)
#     --osl N            output sequence length
#     --concurrency LIST concurrency grid
#     --cache PCT[,...]  cached-prefix percentage of --isl; prefix-len and the
#                        fresh input-sequence-len are computed automatically:
#                            prefix_len = isl * pct / 100
#                            fresh_len  = isl - prefix_len      (== synthetic ISL)
#
#   Example:
#     ./run_aiperf_workload_shapes.sh --model <id> \
#         --isl 8000 --osl 1000 --concurrency 1,2,4,8,16 --cache 90
#     # -> total 8000, cached prefix 7200, fresh 800, OSL 1000
#
#   --isl builds an ad-hoc shape and is combined with any explicit --shape
#   flags. TTFT gate targets for the --isl shape default to "off" (a large
#   sentinel) unless you pass --ttft-p50 / --ttft-p90.
# ---------------------------------------------------------------------------
# REMOTE / DEDICATED ENDPOINTS:
#   Point the sweep at a hosted endpoint with --url (alias --endpoint) plus
#   --api-key; both flow straight through to aiperf. Dedicated endpoints often
#   don't expose an unauthenticated /health, so the startup probe now (a) sends
#   the api key as a bearer token and (b) can be skipped with --skip-health-check
#   (or MAX_WAIT_SECONDS=0).
#
#   Example (DeepSeek-V4-Flash on a DO dedicated endpoint):
#     ./run_aiperf_workload_shapes.sh \
#         --model deepseek-ai/DeepSeek-V4-Flash \
#         --url https://2cpzxl-public-dedicated-inference.do-infra.ai/ \
#         --api-key do_dedicated_HRyzN \
#         --isl 5000 --osl 1000 --cache 100 \
#         --concurrency 16,32,64,128,256,512,1024 \
#         --warmup 3 --skip-health-check
# ---------------------------------------------------------------------------
#
# Workload shapes  (TOTAL ISL / OSL, TTFT SLA):
#   Shape 1:    8k / 1k    TTFT p50 < 2.5s   p90 < 5s
#   Shape 2:   64k / 1k    TTFT p50 < 8s     p90 < 15s
#   Shape 3:  256k / 1k    TTFT p50 < 30s    p90 < 70s
#
# ITL gates (OTPS-tier SLA thresholds, evaluated from the SAME run):
#   claw:  ITL p50 < 25ms     (>=40 OTPS)
#   chat:  ITL p50 < 66.7ms   (>=15 OTPS)
#
# THINKING: K2.7-Code is thinking-only; per the model card no `thinking` param
# is sent. claw/chat differ only by ITL gate (identical request; run once, score
# twice).
#
# IMPORTANT interpretation notes:
#   * REQUIRES server-side prefix caching enabled (e.g. vLLM --enable-prefix-caching)
#     or the "cached" prefix will just be prefilled every time (no hit).
#   * The contract's TTFT buckets are keyed on the INCREMENTAL (non-cached)
#     tokens. With caching, the fresh portion is smaller than the total, so a
#     cached run prefills less and should show LOWER TTFT than the 0% run at the
#     same total. The TTFT targets below stay tied to the shape's TOTAL label;
#     if you want to score against the incremental-token bucket instead, lower
#     the per-shape TTFT target for the cached runs.
#   * With caching, reported Input tok/s counts the FULL prompt (prefix+fresh),
#     but actual prefill compute is only the fresh part -- read throughput with
#     that in mind.
#   * OSL is fixed at 1k (total output incl. reasoning on a thinking-only model).
#
# SCOPE: synthetic fixed-ISL/OSL shapes characterize TTFT-by-input-size and
# ITL/throughput vs concurrency. They do NOT reproduce the real agentic claw
# workload -- measure that with the openclaw replay harness.
#
# Examples:
#   ./run_aiperf_workload_shapes.sh --model amd/Kimi-K2.7-Code-MXFP4 --port 8000 --gpus 8
#   ./run_aiperf_workload_shapes.sh --model <id> --gpus 8 --cache-ratios 75,90
#   ./run_aiperf_workload_shapes.sh --model <id> --gpus 8 \
#       --shape "s1_8k:8000:2500:5000:1,2,4,8,16,32" --gate "claw:25" --cache-ratios 90
#   ./run_aiperf_workload_shapes.sh --model <id> --gpus 8 \
#       --isl 8000 --osl 1000 --concurrency 1,2,4,8,16 --cache 90
#   ./run_aiperf_workload_shapes.sh --model <id> \
#       --url https://host/ --api-key KEY --isl 5000 --osl 1000 --cache 100 \
#       --concurrency 16,32,64,128,256 --skip-health-check
#
set -uo pipefail
###############################################################################
# Defaults (override via flags or env vars)
###############################################################################
MODEL="${MODEL:-}"                                  # served model id (REQUIRED)
TOKENIZER="${TOKENIZER:-}"                           # HF repo or local path; defaults to MODEL
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-true}"
PORT="${PORT:-8000}"                                 # vLLM 8000, SGLang 30000
BASE_URL="${BASE_URL:-}"                             # full endpoint URL (overrides PORT); --url / --endpoint
API_KEY="${API_KEY:-}"                               # bearer key for the endpoint (optional)
GPUS="${GPUS:-8}"                                    # per-GPU throughput divisor in the report
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-3600}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-15}"
SKIP_HEALTH_CHECK="${SKIP_HEALTH_CHECK:-false}"      # skip the /health probe (dedicated endpoints)
# Default concurrency grid; per-shape overrides live in the SHAPES entries.
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64,128,256,512}"
REQ_MULT="${REQ_MULT:-5}"                            # request-count per point = concurrency * REQ_MULT
OUTPUT_TOKENS_MEAN="${OUTPUT_TOKENS_MEAN:-1000}"     # OSL target, fixed across shapes
WARMUP_REQUESTS="${WARMUP_REQUESTS:-16}"             # should be >= NUM_PREFIX_PROMPTS to warm all prefixes
SEED="${SEED:-42}"
# Cache scenarios (percent of TOTAL input that is a reused cached prefix).
CACHE_RATIOS="${CACHE_RATIOS:-0,75,90}"
NUM_PREFIX_PROMPTS="${NUM_PREFIX_PROMPTS:-8}"        # pool size of reused prefixes (default 8)
# Simple-mode inputs (build a single ad-hoc shape from --isl).
ISL="${ISL:-}"                                       # total input sequence length (full prompt)
TTFT_P50="${TTFT_P50:-}"                             # TTFT p50 gate for the --isl shape (default: off)
TTFT_P90="${TTFT_P90:-}"                             # TTFT p90 gate for the --isl shape (default: off)
TTFT_OFF_SENTINEL="${TTFT_OFF_SENTINEL:-999999}"     # large value => TTFT gate effectively disabled
# Emit aiperf's native goodput metric (good-request fraction vs the ITL gates).
ENABLE_GOODPUT="${ENABLE_GOODPUT:-false}"
PARSER="${PARSER:-$(dirname "$0")/parse_sweep.py}"
# Workload shapes:  name:total_input_tokens:ttft_p50_ms:ttft_p90_ms[:concurrency_override]
# total_input_tokens is the FULL prompt (prefix+fresh); it is split by cache ratio.
SHAPES_DEFAULT=(
  "s1_8k:8000:2500:5000:1,2,4,8,16,32,64,128,256,512"
  "s2_64k:64000:8000:15000:1,2,4,8,16,32,64,128,256,512"
  "s3_256k:256000:30000:70000:1,2,4,8,16,32,64,128,256,512"
)
# ITL gates evaluated (from ONE run per shape,cache):  name:itl_p50_target_ms
GATES_DEFAULT=(
  "claw:25"
  "chat:66.7"
)
SHAPES=()
GATES=()
###############################################################################
# Parse CLI flags
###############################################################################
usage() {
  cat <<EOF
Usage: $0 [options]

  Core:
  --model NAME                 served model id (required)
  --tokenizer NAME             HF repo or local path (default: --model)
  --port N                     server port (vLLM 8000, SGLang 30000; default 8000)
  --url URL                    full endpoint URL (overrides --port); alias: --endpoint, --base-url
  --api-key KEY                bearer key for the endpoint (also used for the health probe)
  --skip-health-check          don't probe /health before running (dedicated endpoints)
  --gpus N                     GPU count for per-GPU division (default 8)
  --concurrency LIST           concurrency grid (default ${CONCURRENCY})
  --req-mult N                 request-count = concurrency * N (default ${REQ_MULT})
  --osl N                      output tokens / OSL target (default ${OUTPUT_TOKENS_MEAN})
  --warmup N                   warmup requests (default ${WARMUP_REQUESTS})

  Cache:
  --cache LIST                 alias for --cache-ratios: cached-prefix percentages of ISL.
                               prefix-len and fresh input-len are auto-computed.
  --cache-ratios LIST          cached-prefix percentages (default ${CACHE_RATIOS})
  --num-prefix-prompts N       reused-prefix pool size (default ${NUM_PREFIX_PROMPTS})

  Simple mode (build ONE ad-hoc shape):
  --isl N                      total input sequence length (full prompt)
  --ttft-p50 MS                TTFT p50 gate for the --isl shape (default: disabled)
  --ttft-p90 MS                TTFT p90 gate for the --isl shape (default: disabled)

  Advanced (full matrix):
  --shape n:total:t50:t90[:c]  add/override a shape (repeatable)
  --gate  n:itl_ms             add/override an ITL gate to score against (repeatable)
  --enable-goodput             add aiperf --goodput inter_token_latency:<gate> per gate
  -h, --help                   show this help

Examples:
  $0 --model <id> --isl 8000 --osl 1000 --concurrency 1,2,4,8,16 --cache 90
  $0 --model deepseek-ai/DeepSeek-V4-Flash \\
     --url https://host/ --api-key KEY \\
     --isl 5000 --osl 1000 --cache 100 \\
     --concurrency 16,32,64,128,256,512,1024 --warmup 3 --skip-health-check
EOF
  exit "${1:-0}"
}
while [ $# -gt 0 ]; do
  case "$1" in
    --model)             MODEL="$2"; shift 2;;
    --tokenizer)         TOKENIZER="$2"; shift 2;;
    --port)              PORT="$2"; shift 2;;
    --base-url|--url|--endpoint) BASE_URL="$2"; shift 2;;
    --api-key)           API_KEY="$2"; shift 2;;
    --skip-health-check) SKIP_HEALTH_CHECK="true"; shift 1;;
    --gpus)              GPUS="$2"; shift 2;;
    --concurrency)       CONCURRENCY="$2"; shift 2;;
    --req-mult)          REQ_MULT="$2"; shift 2;;
    --osl)               OUTPUT_TOKENS_MEAN="$2"; shift 2;;
    --isl)               ISL="$2"; shift 2;;
    --ttft-p50)          TTFT_P50="$2"; shift 2;;
    --ttft-p90)          TTFT_P90="$2"; shift 2;;
    --warmup)            WARMUP_REQUESTS="$2"; shift 2;;
    --cache|--cache-ratios) CACHE_RATIOS="$2"; shift 2;;
    --num-prefix-prompts) NUM_PREFIX_PROMPTS="$2"; shift 2;;
    --shape)             SHAPES+=("$2"); shift 2;;
    --gate)              GATES+=("$2"); shift 2;;
    --enable-goodput)    ENABLE_GOODPUT="true"; shift 1;;
    -h|--help)           usage 0;;
    *) echo "Unknown option: $1" >&2; usage 1;;
  esac
done
[ -z "$TOKENIZER" ] && TOKENIZER="$MODEL"
[ -z "$BASE_URL" ] && BASE_URL="http://localhost:${PORT}/"

# Simple mode: if --isl is given, build a single ad-hoc shape from it.
# It is COMBINED with any explicit --shape flags. TTFT gates default to "off".
if [ -n "$ISL" ]; then
  t50="${TTFT_P50:-$TTFT_OFF_SENTINEL}"
  t90="${TTFT_P90:-$TTFT_OFF_SENTINEL}"
  # concurrency override left empty => falls back to the global --concurrency grid
  SHAPES+=("isl${ISL}:${ISL}:${t50}:${t90}")
fi

[ ${#SHAPES[@]} -eq 0 ] && SHAPES=("${SHAPES_DEFAULT[@]}")
[ ${#GATES[@]}  -eq 0 ] && GATES=("${GATES_DEFAULT[@]}")
if [ -z "$MODEL" ]; then
  echo "ERROR: --model is required" >&2; usage 1
fi
IFS=',' read -ra CACHE_RATIO_LIST <<< "$CACHE_RATIOS"
###############################################################################
LOG_DIR="$(pwd)/aiperf_logs"
ARTIFACT_BASE_DIR="$(pwd)/aiperf_artifacts_shapes"
RESULTS_DIR="$(pwd)/aiperf_results"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$ARTIFACT_BASE_DIR" "$RESULTS_DIR"
MASTER_LOG="$LOG_DIR/master_shapes_${RUN_TS}.log"
SUMMARY_FILE="$LOG_DIR/summary_shapes_${RUN_TS}.log"
COMBINED_CSV="$RESULTS_DIR/summary_ALL_${RUN_TS}.csv"
TOKENIZER_EXTRA_ARGS=()
[ "$TOKENIZER_TRUST_REMOTE_CODE" = "true" ] && TOKENIZER_EXTRA_ARGS+=(--tokenizer-trust-remote-code)
API_KEY_ARGS=()
[ -n "$API_KEY" ] && API_KEY_ARGS+=(--api-key "$API_KEY")
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
gen_reqcounts() {
  local conc_csv="$1" mult="$2"
  local -a arr out=()
  IFS=',' read -ra arr <<< "$conc_csv"
  local c
  for c in "${arr[@]}"; do out+=( $(( c * mult )) ); done
  local IFS=,; echo "${out[*]}"
}
wait_for_endpoint() {
  local health_url="${BASE_URL%/}/health"
  local elapsed=0 status_code
  if [ "$SKIP_HEALTH_CHECK" = "true" ]; then
    log "Skipping health probe (--skip-health-check); assuming ${BASE_URL} is ready."
    return 0
  fi
  # authenticate the probe when an api key is set (dedicated endpoints reject anon)
  local -a curl_auth=()
  [ -n "$API_KEY" ] && curl_auth=(-H "Authorization: Bearer $API_KEY")
  log "Waiting for endpoint: ${health_url} (timeout ${MAX_WAIT_SECONDS}s, poll ${POLL_INTERVAL_SECONDS}s)..."
  if [ "$MAX_WAIT_SECONDS" -eq 0 ]; then log "MAX_WAIT_SECONDS=0, skipping health wait."; return 0; fi
  while true; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${curl_auth[@]}" "$health_url" 2>/dev/null || echo "000")
    [ "$status_code" = "200" ] && { log "Endpoint ready (HTTP 200) after ${elapsed}s."; return 0; }
    # Some hosted endpoints return 401/403/404 on /health but still serve chat.
    # Treat any real HTTP response (not 000) as "reachable" so we don't stall.
    if [ "$status_code" != "000" ] && [ "$elapsed" -ge "$POLL_INTERVAL_SECONDS" ]; then
      log "Endpoint reachable (HTTP ${status_code} on /health, not 200) after ${elapsed}s; proceeding. Use --skip-health-check to silence."
      return 0
    fi
    [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ] && { log "Timed out after ${elapsed}s (last ${status_code})."; return 1; }
    [ $((elapsed % 60)) -eq 0 ] && log "Still waiting... (${elapsed}s, last ${status_code})"
    sleep "$POLL_INTERVAL_SECONDS"; elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
  done
}
CHILD_PID=""
cleanup() {
  log "Received termination signal."
  [ -n "$CHILD_PID" ] && { log "Killing aiperf PID ${CHILD_PID}..."; kill "$CHILD_PID" 2>/dev/null; }
  log "=== Run aborted ==="; exit 1
}
trap cleanup SIGINT SIGTERM
log "=== aiperf workload-shape sweep started (PID $$) ==="
log "Model: $MODEL | Tokenizer: $TOKENIZER"
log "URL: $BASE_URL | api-key: $([ -n "$API_KEY" ] && echo set || echo none) | GPUs: $GPUS | OSL: $OUTPUT_TOKENS_MEAN | warmup: $WARMUP_REQUESTS | goodput: $ENABLE_GOODPUT"
[ -n "$ISL" ] && log "Simple mode: --isl ${ISL} (TTFT p50<=${TTFT_P50:-off} p90<=${TTFT_P90:-off}) conc=[${CONCURRENCY}]"
log "Shapes: ${SHAPES[*]}"
log "Cache ratios: ${CACHE_RATIOS}  | num-prefix-prompts: ${NUM_PREFIX_PROMPTS}"
log "Gates:  ${GATES[*]}  (one run per shape,cache; scored against each gate)"
log "Combined CSV: $COMBINED_CSV"
if ! wait_for_endpoint; then
  log "=== Aborting: endpoint never became ready ==="; exit 1
fi
combined_header_written=false
: > "$COMBINED_CSV"
for shape in "${SHAPES[@]}"; do
  IFS=':' read -r sname total_input ttft50 ttft90 conc_override <<< "$shape"
  conc_list="${conc_override:-$CONCURRENCY}"
  reqcounts="$(gen_reqcounts "$conc_list" "$REQ_MULT")"

  for ratio in "${CACHE_RATIO_LIST[@]}"; do
    # split total input into cached prefix + fresh (non-cached) remainder
    prefix_len=$(( total_input * ratio / 100 ))
    fresh_len=$(( total_input - prefix_len ))
    clabel="c${ratio}"

    PREFIX_ARGS=()
    if [ "$ratio" -eq 0 ]; then
      synthetic_mean="$total_input"           # no cache: full input is fresh
    else
      synthetic_mean="$fresh_len"
      PREFIX_ARGS=(--num-prefix-prompts "$NUM_PREFIX_PROMPTS" --prompt-prefix-length "$prefix_len")
    fi

    runname="${sname}_${clabel}"
    PROFILE_LOG="$LOG_DIR/profile_${runname}_${RUN_TS}.log"
    PROFILE_ARTIFACT_DIR="$ARTIFACT_BASE_DIR/profile_${runname}_${RUN_TS}"
    mkdir -p "$PROFILE_ARTIFACT_DIR"
    log "--- ${runname}: total=${total_input} prefix(cached)=${prefix_len} fresh=${synthetic_mean} OSL=${OUTPUT_TOKENS_MEAN} | TTFT<=${ttft50}/${ttft90}ms | conc=[${conc_list}] ---"

    GOODPUT_ARGS=()
    if [ "$ENABLE_GOODPUT" = "true" ]; then
      for g in "${GATES[@]}"; do
        IFS=':' read -r _gname gitl <<< "$g"
        GOODPUT_ARGS+=(--goodput "inter_token_latency:${gitl}")
      done
    fi

    # ONE run per (shape,cache). No `thinking` param (model is thinking-only).
    aiperf profile \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      "${TOKENIZER_EXTRA_ARGS[@]}" \
      --url "$BASE_URL" \
      "${API_KEY_ARGS[@]}" \
      --endpoint-type chat --streaming \
      --use-server-token-count \
      "${PREFIX_ARGS[@]}" \
      --synthetic-input-tokens-mean "$synthetic_mean" --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean "$OUTPUT_TOKENS_MEAN" --output-tokens-stddev 0 \
      --extra-inputs ignore_eos:true \
      --extra-inputs min_tokens:"$OUTPUT_TOKENS_MEAN" \
      --extra-inputs max_tokens:"$OUTPUT_TOKENS_MEAN" \
      "${GOODPUT_ARGS[@]}" \
      --warmup-request-count "$WARMUP_REQUESTS" \
      --sweep-type zip \
      --concurrency "$conc_list" \
      --request-count "$reqcounts" \
      --random-seed "$SEED" --ui simple \
      --output-artifact-dir "$PROFILE_ARTIFACT_DIR" \
      --profile-export-prefix "profile_${runname}" \
      > "$PROFILE_LOG" 2>&1 &
    CHILD_PID=$!; wait "$CHILD_PID"; status=$?; CHILD_PID=""

    if [ $status -ne 0 ]; then
      log "--- ${runname} FAILED (exit ${status}) -- see ${PROFILE_LOG} ---"
      echo "${runname}: FAILED (exit ${status})" >> "$SUMMARY_FILE"
      continue
    fi
    log "--- ${runname} COMPLETED OK ---"; echo "${runname}: OK" >> "$SUMMARY_FILE"

    if [ ! -f "$PARSER" ]; then
      log "ERROR: parser not found at $PARSER -- skipping parse for ${runname}."
      continue
    fi
    sweep_dir="$PROFILE_ARTIFACT_DIR"
    [ -d "$PROFILE_ARTIFACT_DIR/sweep_aggregate" ] && sweep_dir="$PROFILE_ARTIFACT_DIR/sweep_aggregate"

    # -------- score the SAME run against each ITL gate (no re-run) --------
    for g in "${GATES[@]}"; do
      IFS=':' read -r gname itl_target <<< "$g"
      per_csv="$RESULTS_DIR/summary_${runname}_${gname}_${RUN_TS}.csv"
      log "Scoring ${runname} against gate '${gname}' (ITL<=${itl_target}ms) -> ${per_csv}"
      if python3 "$PARSER" \
           --sweep-dir "$sweep_dir" \
           --gpus "$GPUS" \
           --osl-target "$OUTPUT_TOKENS_MEAN" \
           --ttft-p50-target "$ttft50" \
           --ttft-p90-target "$ttft90" \
           --itl-p50-target "$itl_target" \
           --profile-name "${runname}_${gname}" \
           --out-csv "$per_csv" 2>&1 | tee -a "$MASTER_LOG"; then
        if [ "$combined_header_written" = false ]; then
          cat "$per_csv" >> "$COMBINED_CSV"; combined_header_written=true
        else
          tail -n +2 "$per_csv" >> "$COMBINED_CSV"
        fi
      else
        log "WARN: parse failed for ${runname}_${gname}."
      fi
    done
  done
done
log "=== Combined results: $COMBINED_CSV ==="
log "=== Done. Pass/fail: ${SUMMARY_FILE} | Per-GPU CSVs: ${RESULTS_DIR} ==="