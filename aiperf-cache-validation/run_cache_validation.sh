#!/usr/bin/env bash
#
# run_cache_validation.sh — multi-turn, long-context aiperf profile for
# prefix-cache validation (DeepSeek-V4-Flash-0731 defaults).
#
# Simulates concurrent users holding chat sessions against a long shared
# system prompt + per-user context, with human-like variability: think time
# varies per turn, completions stop naturally, and finished users are
# replaced by fresh (cache-cold) conversations.
#
# Usage:
#   export API="https://your-endpoint/"
#   export API_KEY="do_dedicated_..."
#   ./run_cache_validation.sh
#
# Reproduce the original fixed-length profile (32 uniform turns, pinned
# output length, no churn):
#   ./run_cache_validation.sh --turn-mean 32 --turn-stddev 0 \
#       --turn-delay-stddev 15000 --conversations 64 --dataset-entries 64 \
#       --ignore-eos
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

# --- Load shape (realism-tuned defaults; see README for rationale) ---------
RATE="${RATE:-1.6}"                  # --user-centric-rate (QPS across all users)
NUM_USERS="${NUM_USERS:-64}"         # --num-users
CONVERSATIONS="${CONVERSATIONS:-256}" # --conversation-num; > num-users => churn
TURN_MEAN="${TURN_MEAN:-8}"          # --conversation-turn-mean
# NOTE: keep 0 with --user-centric-rate. aiperf samples each session's turn
# count independently from the generated conversation's actual length, so a
# nonzero stddev triggers "num_turns (N) exceeds conversation length (M)"
# errors in the workers.
TURN_STDDEV="${TURN_STDDEV:-0}"      # --conversation-turn-stddev
TURN_DELAY_MEAN="${TURN_DELAY_MEAN:-30000}"     # ms think time between turns
TURN_DELAY_STDDEV="${TURN_DELAY_STDDEV:-25000}" # ms; high stddev ~ human variance
SHARED_SYSTEM_PROMPT="${SHARED_SYSTEM_PROMPT:-8000}"  # tokens, shared by all users
USER_CONTEXT="${USER_CONTEXT:-112000}"                # tokens, unique per user
DATASET_ENTRIES="${DATASET_ENTRIES:-256}"             # unique turn prompts

# --- Token lengths ----------------------------------------------------------
ISL="${ISL:-12700}"
ISL_STDDEV="${ISL_STDDEV:-4000}"
OSL="${OSL:-917}"
OSL_STDDEV="${OSL_STDDEV:-300}"
SEQ_DIST="${SEQ_DIST:-}"   # if set, replaces --isl/--osl with --sequence-distribution

# --- Behavior quirks --------------------------------------------------------
IGNORE_EOS=0               # 1 = pin output length (ignore_eos:true); default: natural stop
CANCEL_RATE="${CANCEL_RATE:-0}"   # % of requests aborted mid-stream (user gives up)
CANCEL_DELAY="${CANCEL_DELAY:-2}" # seconds streamed before the abort
GOODPUT="${GOODPUT:-}"     # e.g. "time_to_first_token:2000 inter_token_latency:50"

DRY_RUN=0

usage() {
    sed -n '2,23p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Endpoint options:
  --model NAME              Model id the server serves
  --tokenizer NAME          HuggingFace tokenizer id (default: same as model)
  --url URL                 OpenAI-compatible base URL
  --api-key KEY             Bearer API key (required)

Load shape:
  --rate QPS                Aggregate request rate across users     (default: 1.6)
  --num-users N             Concurrent simulated users              (default: 64)
  --conversations N         Unique conversations; keep > num-users
                            so fresh cache-cold sessions arrive     (default: 256)
  --turn-mean N             Mean turns per conversation             (default: 8)
  --turn-stddev N           Stddev of turns per conversation        (default: 0;
                            nonzero values crash aiperf's user-centric mode,
                            see README)
  --turn-delay-mean MS      Mean think time between turns           (default: 30000)
  --turn-delay-stddev MS    Stddev of think time                    (default: 25000)
  --shared-system-prompt N  Shared system prompt tokens             (default: 8000)
  --user-context N          Per-user context prompt tokens          (default: 112000)
  --dataset-entries N       Unique turn prompts in the dataset      (default: 256)

Token lengths:
  --isl N / --isl-stddev N  Per-turn input tokens                   (default: 12700 / 4000)
  --osl N / --osl-stddev N  Per-turn output token target            (default: 917 / 300)
  --seq-dist SPEC           Mixed ISL,OSL:percent pairs; replaces --isl/--osl.
                            e.g. "1500|500,300|150:60;12000|4000,900|300:30;80000|20000,1200|400:10"

Behavior quirks:
  --ignore-eos              Pin output length exactly (ignore_eos:true).
                            Default is natural stopping, which is realistic
                            but makes output token counts model-dependent.
  --cancel-rate PCT         Abort PCT% of requests mid-stream       (default: 0; try 5)
  --cancel-delay SEC        Seconds streamed before aborting        (default: 2)
  --goodput "K:V ..."       SLO pairs for goodput reporting,
                            e.g. "time_to_first_token:2000 inter_token_latency:50"

Run control:
  --artifact-dir PATH       aiperf artifact output dir (default: ./aiperf_cache_validation)
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
        --rate) RATE="$2"; shift 2 ;;
        --num-users) NUM_USERS="$2"; shift 2 ;;
        --conversations) CONVERSATIONS="$2"; shift 2 ;;
        --turn-mean) TURN_MEAN="$2"; shift 2 ;;
        --turn-stddev) TURN_STDDEV="$2"; shift 2 ;;
        --turn-delay-mean) TURN_DELAY_MEAN="$2"; shift 2 ;;
        --turn-delay-stddev) TURN_DELAY_STDDEV="$2"; shift 2 ;;
        --shared-system-prompt) SHARED_SYSTEM_PROMPT="$2"; shift 2 ;;
        --user-context) USER_CONTEXT="$2"; shift 2 ;;
        --dataset-entries) DATASET_ENTRIES="$2"; shift 2 ;;
        --isl) ISL="$2"; shift 2 ;;
        --isl-stddev) ISL_STDDEV="$2"; shift 2 ;;
        --osl) OSL="$2"; shift 2 ;;
        --osl-stddev) OSL_STDDEV="$2"; shift 2 ;;
        --seq-dist) SEQ_DIST="$2"; shift 2 ;;
        --ignore-eos) IGNORE_EOS=1; shift ;;
        --cancel-rate) CANCEL_RATE="$2"; shift 2 ;;
        --cancel-delay) CANCEL_DELAY="$2"; shift 2 ;;
        --goodput) GOODPUT="$2"; shift 2 ;;
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

# aiperf requires turn mean >= 2 for user-centric rate mode.
if [ "$TURN_MEAN" -lt 2 ]; then
    echo "ERROR: --turn-mean must be >= 2 (user-centric rate mode is multi-turn only)" >&2
    exit 2
fi

if [ "$TURN_STDDEV" != "0" ]; then
    echo "WARNING: --turn-stddev $TURN_STDDEV with user-centric rate mode can crash" >&2
    echo "         workers with 'num_turns (N) exceeds conversation length (M)':" >&2
    echo "         aiperf samples per-session turn counts independently from each" >&2
    echo "         generated conversation's actual length. Use 0 unless your aiperf" >&2
    echo "         version has fixed this." >&2
fi

if [ -z "$RANDOM_SEED" ]; then
    RANDOM_SEED=$(( ($(date +%s) ^ $$ ^ RANDOM) % 2147483647 ))
fi

mkdir -p "$ARTIFACT_DIR"

echo "=== aiperf cache-validation profile ==="
echo "  model        : $MODEL"
echo "  tokenizer    : $TOKENIZER"
echo "  url          : $API"
echo "  rate/users   : $RATE QPS across $NUM_USERS users"
echo "  conversations: $CONVERSATIONS x ${TURN_MEAN}+-${TURN_STDDEV} turns, think ${TURN_DELAY_MEAN}+-${TURN_DELAY_STDDEV} ms"
echo "  context      : system $SHARED_SYSTEM_PROMPT + per-user $USER_CONTEXT tokens"
if [ -n "$SEQ_DIST" ]; then
    echo "  lengths      : sequence distribution: $SEQ_DIST"
else
    echo "  lengths      : ISL ${ISL}+-${ISL_STDDEV} / OSL ${OSL}+-${OSL_STDDEV}"
fi
[ "$IGNORE_EOS" -eq 1 ] && echo "  output       : pinned (ignore_eos:true)" \
                        || echo "  output       : natural stopping"
[ "${CANCEL_RATE%.*}" != "0" ] && echo "  cancellation : ${CANCEL_RATE}% after ${CANCEL_DELAY}s"
echo "  duration     : ${BENCHMARK_DURATION}s"
echo "  artifact-dir : $ARTIFACT_DIR"
echo "  seed         : $RANDOM_SEED"
echo

CMD=(aiperf profile
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --tokenizer-trust-remote-code
  --url "$API"
  --api-key "$API_KEY"
  --endpoint-type chat
  --streaming
  --use-server-token-count
  --user-centric-rate "$RATE"
  --num-users "$NUM_USERS"
  --conversation-num "$CONVERSATIONS"
  --conversation-turn-mean "$TURN_MEAN"
  --conversation-turn-stddev "$TURN_STDDEV"
  --conversation-turn-delay-mean "$TURN_DELAY_MEAN"
  --conversation-turn-delay-stddev "$TURN_DELAY_STDDEV"
  --shared-system-prompt-length "$SHARED_SYSTEM_PROMPT"
  --user-context-prompt-length "$USER_CONTEXT"
  --num-dataset-entries "$DATASET_ENTRIES"
)

if [ -n "$SEQ_DIST" ]; then
    CMD+=(--sequence-distribution "$SEQ_DIST")
else
    CMD+=(--isl "$ISL" --isl-stddev "$ISL_STDDEV" --osl "$OSL" --osl-stddev "$OSL_STDDEV")
fi

[ "$IGNORE_EOS" -eq 1 ] && CMD+=(--extra-inputs ignore_eos:true)

if [ "${CANCEL_RATE%.*}" != "0" ]; then
    CMD+=(--request-cancellation-rate "$CANCEL_RATE" --request-cancellation-delay "$CANCEL_DELAY")
fi

[ -n "$GOODPUT" ] && CMD+=(--goodput "$GOODPUT")

CMD+=(
  --benchmark-duration "$BENCHMARK_DURATION"
  --random-seed "$RANDOM_SEED"
  --server-metrics
  --server-metrics-formats json parquet
  --artifact-dir "$ARTIFACT_DIR"
  --ui simple
  --export-level raw
)

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:\n'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
