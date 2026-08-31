#!/usr/bin/env bash
#
# run_cache_validation.sh — multi-turn, long-context aiperf profile for
# prefix-cache validation (DeepSeek-V4-Flash-0731 defaults).
#
# Simulates 64 concurrent users, each holding a long shared system prompt +
# per-user context, then driving ~32 conversation turns with inter-turn delay.
# Use this to exercise KV/prefix cache behaviour under sustained chat load.
#
# Usage:
#   export API="https://your-endpoint/"
#   export API_KEY="do_dedicated_..."
#   export MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"   # optional; this is default
#   ./run_cache_validation.sh
#
# Or pass flags:
#   ./run_cache_validation.sh --model <id> --url <url> --api-key <key>
#
# Run with --help for all options.
#
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash-0731}"
API="${API:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./aiperf_cache_validation}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-1800}"
# Fresh seed every run unless RANDOM_SEED / --random-seed is set.
RANDOM_SEED="${RANDOM_SEED:-}"
DRY_RUN=0

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --model NAME              Model id the server serves
                            (default: $MODEL or deepseek-ai/DeepSeek-V4-Flash-0731)
  --tokenizer NAME          HuggingFace tokenizer id
                            (default: $TOKENIZER or same as model)
  --url URL                 OpenAI-compatible base URL (default: $API or localhost:8000)
  --api-key KEY             Bearer API key (default: $API_KEY)
  --artifact-dir PATH       aiperf artifact output dir
                            (default: $ARTIFACT_DIR or ./aiperf_cache_validation)
  --benchmark-duration SEC  Measured duration in seconds (default: 1800)
  --random-seed N           RNG seed (default: fresh each run)
  --dry-run                 Print the aiperf command and exit
  -h, --help                Show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --url) API="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
        --benchmark-duration) BENCHMARK_DURATION="$2"; shift 2 ;;
        --random-seed) RANDOM_SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

command -v aiperf >/dev/null 2>&1 || {
    echo "ERROR: aiperf not found on PATH. Install with: pip install aiperf" >&2
    exit 1
}

[ -n "$API_KEY" ] || {
    echo "ERROR: --api-key or \$API_KEY is required" >&2
    usage
    exit 2
}

if [ -z "$RANDOM_SEED" ]; then
    RANDOM_SEED=$(( ($(date +%s) ^ $$ ^ RANDOM) % 2147483647 ))
fi

mkdir -p "$ARTIFACT_DIR"

echo "=== aiperf cache-validation profile ==="
echo "  model       : $MODEL"
echo "  tokenizer   : $TOKENIZER"
echo "  url         : $API"
echo "  duration    : ${BENCHMARK_DURATION}s"
echo "  artifact-dir: $ARTIFACT_DIR"
echo "  seed        : $RANDOM_SEED"
echo

set -- aiperf profile \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --url "$API" \
  --api-key "$API_KEY" \
  --endpoint-type chat \
  --streaming \
  --use-server-token-count \
  --user-centric-rate 0.4 \
  --num-users 64 \
  --conversation-num 64 \
  --conversation-turn-mean 6 \
  --conversation-turn-stddev 0 \
  --conversation-turn-delay-mean 30000 \
  --conversation-turn-delay-stddev 15000 \
  --shared-system-prompt-length 2000 \
  --user-context-prompt-length 112000 \
  --num-dataset-entries 64 \
  --isl 12700 \
  --isl-stddev 4000 \
  --osl 917 \
  --osl-stddev 300 \
  --extra-inputs ignore_eos:true \
  --benchmark-duration "$BENCHMARK_DURATION" \
  --random-seed "$RANDOM_SEED" \
  --server-metrics \
  --server-metrics-formats json parquet \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui simple \
  --export-level raw

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:\n'
    printf ' %q' "$@"
    printf '\n'
    exit 0
fi

exec "$@"
