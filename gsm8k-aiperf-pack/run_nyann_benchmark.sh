#!/usr/bin/env bash
#
# run_nyann_benchmark.sh -- benchmark an OpenAI-compatible server with
# nyann-bench and report cache hit %, TPS, TTFT and ITL per concurrency point.
#
# TTFT / ITL / TPS come from nyann-bench's client-side recordings
# (`nyann-bench analyze --json`). Prefix-cache hit rate is not a client-side
# metric: this script snapshots the server's Prometheus /metrics endpoint
# (vllm:prefix_cache_queries / vllm:prefix_cache_hits, same counters
# scripts/report.py reads) before and after each point and reports the delta.
# Servers that do not expose those counters show "n/a".
#
# nyann-bench (v0.1.0) sends no Authorization header. When --api-key is set,
# a local streaming proxy is started that injects "Authorization: Bearer".
# The proxy adds client-side overhead; for precise ITL prefer a direct,
# unauthenticated endpoint.
#
# Usage:
#   ./run_nyann_benchmark.sh --model /models/Kimi-K3
#   ./run_nyann_benchmark.sh --model deepseek-ai/DeepSeek-V4-Flash-0731 \
#       --url "$API" --api-key "$API_KEY" \
#       --concurrency 160,320 --isl 120000 --osl 917 --duration 30m
#
# Run with --help for all options.
#
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL=""
URL="http://localhost:8000"
API_KEY=""
ISL=68000
OSL=350
CONCURRENCY="16"
DURATION="10m"
WARMUP="30s"
GPUS=8
CORPUS="$HERE/gsm8k_corpus.txt"
CHARS_PER_TOKEN=0          # 0 = auto-calibrate via the server's /tokenize
METRICS_URL=""             # default: URL without /v1, + /metrics
CACHE_SALT="fixed"         # fixed | random | off
OUT_DIR="$HERE/nyann_results_$(date +%Y%m%d_%H%M%S)"

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --model NAME            Model id the server serves             (required)
  --url URL               Server base URL, with or without /v1   (default: http://localhost:8000)
  --api-key KEY           Bearer key; enables the local auth proxy
  --concurrency LIST      Comma-separated, e.g. 16,32            (default: 16)
  --isl N                 Input tokens per request               (default: 68000)
  --osl N                 Max output tokens per request          (default: 350)
  --duration DUR          Measured time per point, Go syntax     (default: 10m)
  --warmup DUR            Warmup before measurement, 0 = none    (default: 30s)
  --gpus N                GPU count, for per-GPU throughput      (default: 8)
  --corpus PATH           Text corpus for prompts                (default: ./gsm8k_corpus.txt)
  --chars-per-token F     Override tokenizer calibration; needed when the
                          server has no /tokenize endpoint       (default: auto)
  --metrics-url URL       Prometheus /metrics endpoint to scrape for
                          cache hit %  (default: <url minus /v1>/metrics)
  --cache-salt MODE       fixed | random | off. "fixed" isolates this run's
                          prefix-cache namespace; "random" defeats caching
                          (useful as a 0%-cache baseline)        (default: fixed)
  --out-dir PATH          Results directory
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --isl) ISL="$2"; shift 2 ;;
        --osl) OSL="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --chars-per-token) CHARS_PER_TOKEN="$2"; shift 2 ;;
        --metrics-url) METRICS_URL="$2"; shift 2 ;;
        --cache-salt) CACHE_SALT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "ERROR: --model is required" >&2; usage; exit 2; }
[ -f "$CORPUS" ] || { echo "ERROR: corpus not found: $CORPUS" >&2; exit 1; }

NYANN="$(command -v nyann-bench || true)"
[ -n "$NYANN" ] || [ ! -x "$HOME/go/bin/nyann-bench" ] || NYANN="$HOME/go/bin/nyann-bench"
[ -n "$NYANN" ] || {
    echo "ERROR: nyann-bench not found. Install with:" >&2
    echo "  go install github.com/neuralmagic/nyann-bench/cmd/nyann-bench@latest" >&2
    exit 1
}

# Normalise URLs: nyann-bench wants .../v1, the metrics scrape wants the bare host.
BASE_URL="${URL%/}"
BASE_URL="${BASE_URL%/v1}"
TARGET_URL="$BASE_URL/v1"
[ -n "$METRICS_URL" ] || METRICS_URL="$BASE_URL/metrics"

auth_curl() {
    if [ -n "$API_KEY" ]; then
        curl -sf -m 15 -H "Authorization: Bearer $API_KEY" "$@"
    else
        curl -sf -m 15 "$@"
    fi
}

auth_curl "$TARGET_URL/models" >/dev/null 2>&1 || {
    echo "ERROR: no server responding at $TARGET_URL/models" >&2
    exit 1
}

mkdir -p "$OUT_DIR"

PROXY_PID=""
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# --- Auth proxy: nyann-bench v0.1.0 cannot send an Authorization header ------
if [ -n "$API_KEY" ]; then
    PROXY_PORT=18080
    while nc -z 127.0.0.1 "$PROXY_PORT" 2>/dev/null; do PROXY_PORT=$((PROXY_PORT + 1)); done
    cat > "$OUT_DIR/.auth_proxy.py" <<'PYEOF'
import http.client, os, sys, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

upstream = urllib.parse.urlparse(sys.argv[1])
port = int(sys.argv[2])
key = os.environ["NYANN_PROXY_KEY"]

class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def proxy(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        cls = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        conn = cls(upstream.netloc, timeout=21600)
        try:
            conn.request(self.command, upstream.path.rstrip("/") + self.path, body=body, headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": self.headers.get("Accept", "*/*"),
            })
            resp = conn.getresponse()
            self.send_response(resp.status)
            ct = resp.getheader("Content-Type")
            if ct:
                self.send_header("Content-Type", ct)
            # Close-delimited response body: lets us forward SSE chunks as they
            # arrive without re-implementing chunked transfer encoding.
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = proxy
    do_POST = proxy

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", port), Proxy).serve_forever()
PYEOF
    NYANN_PROXY_KEY="$API_KEY" python3 "$OUT_DIR/.auth_proxy.py" "$BASE_URL" "$PROXY_PORT" &
    PROXY_PID=$!
    sleep 1
    kill -0 "$PROXY_PID" 2>/dev/null || { echo "ERROR: auth proxy failed to start" >&2; exit 1; }
    TARGET_URL="http://127.0.0.1:$PROXY_PORT/v1"
    echo "auth proxy on $TARGET_URL -> $BASE_URL (adds overhead; ITL precision reduced)"
fi

# --- Cache-hit scrape helpers -------------------------------------------------
# Prints "queries hits" summed across vllm:(gpu_)?prefix_cache_{queries,hits}
# counters (with or without the _total suffix), or "0 0" when absent.
scrape_cache_counters() {
    auth_curl "$METRICS_URL" 2>/dev/null | python3 -c '
import re, sys
q = h = 0.0
pat = re.compile(r"^(vllm:(?:gpu_)?prefix_cache_(queries|hits)(?:_total)?)(\{[^}]*\})?\s+(\S+)")
for line in sys.stdin:
    m = pat.match(line)
    if not m:
        continue
    val = float(m.group(4))
    if m.group(2) == "queries":
        q += val
    else:
        h += val
print(f"{q} {h}")
' 2>/dev/null || echo "0 0"
}

case "$CACHE_SALT" in
    fixed)  SALT_JSON=", \"cache_salt\": {\"mode\": \"fixed\", \"value\": \"nyann-wrap-$(date +%s)\"}" ;;
    random) SALT_JSON=", \"cache_salt\": {\"mode\": \"random\"}" ;;
    off)    SALT_JSON="" ;;
    *) echo "ERROR: --cache-salt must be fixed, random or off" >&2; exit 2 ;;
esac

WARMUP_JSON=""
if [ "$WARMUP" != "0" ] && [ -n "$WARMUP" ]; then
    WARMUP_JSON="\"warmup\": {\"duration\": \"$WARMUP\", \"stagger\": true},"
fi

echo
echo "=== nyann-bench benchmark ==="
echo "  model       : $MODEL"
echo "  target      : $TARGET_URL"
echo "  metrics     : $METRICS_URL"
echo "  ISL/OSL     : $ISL / $OSL"
echo "  concurrency : $CONCURRENCY"
echo "  duration    : $DURATION per point (warmup $WARMUP)"
echo "  results     : $OUT_DIR"
echo

status_all=0
ROWS="$OUT_DIR/rows.jsonl"
: > "$ROWS"

for conc in $(echo "$CONCURRENCY" | tr ',' ' '); do
    art="$OUT_DIR/concurrency_${conc}"
    mkdir -p "$art"
    echo "--- concurrency $conc ($DURATION) ---"

    read -r q0 h0 <<< "$(scrape_cache_counters)"

    CONFIG=$(cat <<EOF
{
  $WARMUP_JSON
  "load": {"mode": "concurrent", "concurrency": $conc, "rampup": "10s", "duration": "$DURATION"},
  "workload": {
    "type": "corpus",
    "name": "gsm8k-wrap",
    "corpus_path": "$CORPUS",
    "isl": $ISL,
    "osl": $OSL,
    "turns": 1,
    "chars_per_token": $CHARS_PER_TOKEN$SALT_JSON
  }
}
EOF
)
    "$NYANN" generate \
        --target "$TARGET_URL" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$art" \
        > "$art/summary.json" 2> "$art/nyann.log"
    st=$?

    read -r q1 h1 <<< "$(scrape_cache_counters)"

    if [ $st -ne 0 ]; then
        echo "    FAILED (exit $st) -- see $art/nyann.log"
        status_all=1
        continue
    fi

    "$NYANN" analyze --dir "$art" --json > "$art/analysis.json" 2>/dev/null

    python3 - "$art/analysis.json" "$conc" "$q0" "$h0" "$q1" "$h1" "$GPUS" >> "$ROWS" <<'PYEOF'
import json, sys
path, conc, q0, h0, q1, h1, gpus = sys.argv[1:8]
s = json.load(open(path))
dq, dh = float(q1) - float(q0), float(h1) - float(h0)
row = {
    "concurrency": int(conc),
    "requests": s.get("successful_requests", 0),
    "errors": s.get("error_requests", 0),
    "cache_hit_pct": round(100.0 * dh / dq, 2) if dq > 0 else None,
    "out_tps": round(s.get("output_tokens_per_second", 0.0), 2),
    "out_tps_per_gpu": round(s.get("output_tokens_per_second", 0.0) / int(gpus), 2),
    "ttft_p50_ms": round(s.get("ttft_ms", {}).get("p50", 0.0), 1),
    "ttft_p90_ms": round(s.get("ttft_ms", {}).get("p90", 0.0), 1),
    "itl_p50_ms": round(s.get("itl_ms", {}).get("p50", 0.0), 2),
    "itl_p99_ms": round(s.get("itl_ms", {}).get("p99", 0.0), 2),
}
print(json.dumps(row))
PYEOF
    echo "    OK"
done

echo
echo "=== report ==="
python3 - "$ROWS" "$OUT_DIR/report.json" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("no successful points", file=sys.stderr)
    sys.exit(1)
hdr = (f"{'conc':>5} {'reqs':>6} {'err':>4} {'cache%':>7} {'TTFT p50':>9} {'TTFT p90':>9} "
       f"{'ITL p50':>8} {'ITL p99':>8} {'out tok/s':>10} {'out/s/gpu':>10}")
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda r: r["concurrency"]):
    cache = f"{r['cache_hit_pct']:.2f}" if r["cache_hit_pct"] is not None else "n/a"
    print(f"{r['concurrency']:>5} {r['requests']:>6} {r['errors']:>4} {cache:>7} "
          f"{r['ttft_p50_ms']:>9.1f} {r['ttft_p90_ms']:>9.1f} "
          f"{r['itl_p50_ms']:>8.2f} {r['itl_p99_ms']:>8.2f} "
          f"{r['out_tps']:>10.2f} {r['out_tps_per_gpu']:>10.2f}")
json.dump(rows, open(sys.argv[2], "w"), indent=2)
print(f"\nlatencies in ms; wrote {sys.argv[2]}")
if any(r["cache_hit_pct"] is None for r in rows):
    print("NOTE: cache% is n/a -- the server did not expose vllm prefix-cache")
    print("      counters at the metrics URL, or the scrape failed. If another")
    print("      workload shares the server, its traffic pollutes the delta.")
PYEOF

echo
echo "Results in $OUT_DIR"
exit $status_all
