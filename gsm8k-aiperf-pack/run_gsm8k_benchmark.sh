#!/usr/bin/env bash
#
# run_gsm8k_benchmark.sh -- benchmark a speculative-decoding server with
# coherent GSM8K prompts instead of aiperf's stock literary corpus.
#
# Uses aiperf's documented --input-file path: prompts are pre-built at their
# final length and handed to aiperf as a single_turn JSONL. Nothing inside the
# installed aiperf package is modified.
#
# Usage:
#   ./run_gsm8k_benchmark.sh --model /models/Kimi-K3
#   ./run_gsm8k_benchmark.sh --model /models/Kimi-K3 --concurrency 16,24 --isl 68000
#
# Run with --help for all options.
#
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL=""
TOKENIZER=""
URL="http://localhost:8000"
API_KEY="EMPTY"
ISL=68000
OSL=350
CACHE=93
NUM_PREFIX_PROMPTS=8
CONCURRENCY="16"
REQ_MULT=5
WARMUP=3
SEED=""   # empty = unique per run; set via --seed to pin
GPUS=8
OUT_DIR="$HERE/results_$(date +%Y%m%d_%H%M%S)"
CORPUS="$HERE/gsm8k_corpus.txt"
PROMPTS=""
KEEP_PROMPTS=false
TPS_TARGET=""
CATALOG_MODEL=""
CATALOG_URL="https://api.digitalocean.com/v2/gen-ai/models/catalog"

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --model PATH/NAME     Model id the server serves            (required)
  --tokenizer PATH      Tokenizer                             (default: --model)
  --url URL             Server base URL                       (default: http://localhost:8000)
  --api-key KEY         API key                               (default: EMPTY)
  --isl N               Total input tokens per request        (default: 68000)
  --osl N               Output tokens per request             (default: 350)
  --cache PCT           Percent of ISL that is a reused prefix(default: 93)
  --num-prefix-prompts N  Distinct prefixes in the pool       (default: 8)
  --concurrency LIST    Comma-separated, e.g. 16,24           (default: 16)
  --req-mult N          Requests per concurrency = N x conc   (default: 5)
  --warmup N            Warmup requests                       (default: 3)
  --seed N              Prefix-pool seed; tails use N+conc
                        (default: unique per run from time+pid)
  --gpus N              GPU count, for per-GPU throughput     (default: 8)
  --corpus PATH         GSM8K corpus text                     (default: ./gsm8k_corpus.txt)
  --prompts PATH        Reuse one prompt JSONL for every point (skips generation;
                        exact sequences replay and inflate cache hit rate)
  --keep-prompts        Do not delete generated prompt JSONLs on exit
  --out-dir PATH        Results directory
  --tps-target N        Target aggregate output tokens/s; the post-run economics
                        step reports how many nodes are needed to hit it
  --catalog-model ID    DigitalOcean catalog model_id to price against
                        (default: inferred from --model)
  --catalog-url URL     Pricing catalog endpoint
                        (default: DO gen-ai models catalog)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --isl) ISL="$2"; shift 2 ;;
        --osl) OSL="$2"; shift 2 ;;
        --cache) CACHE="$2"; shift 2 ;;
        --num-prefix-prompts) NUM_PREFIX_PROMPTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --req-mult) REQ_MULT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --prompts) PROMPTS="$2"; shift 2 ;;
        --keep-prompts) KEEP_PROMPTS=true; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --tps-target) TPS_TARGET="$2"; shift 2 ;;
        --catalog-model) CATALOG_MODEL="$2"; shift 2 ;;
        --catalog-url) CATALOG_URL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "ERROR: --model is required" >&2; usage; exit 2; }
[ -n "$TOKENIZER" ] || TOKENIZER="$MODEL"

# Fresh base seed every invocation so repeated runs do not replay the same
# prompts. Within a run, tails still vary by conc (SEED+conc). Pass --seed to pin.
if [ -z "$SEED" ]; then
    SEED=$(( ($(date +%s) ^ $$) & 0x7fffffff ))
fi

command -v aiperf >/dev/null || { echo "ERROR: aiperf not on PATH" >&2; exit 1; }

if ! curl -sf -m 10 "$URL/health" >/dev/null 2>&1 \
   && ! curl -sf -m 10 "$URL/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
    echo "ERROR: no server responding at $URL" >&2
    exit 1
fi

# Pre-flight: make sure the model resolves to a catalog entry for pricing
# BEFORE spending benchmark time. On a bad name, show close matches and ask
# the user to type a correct model_id (empty answer = run without pricing).
CATALOG_MATCH=""
while :; do
    lookup="${CATALOG_MODEL:-$MODEL}"
    resolved="$(python3 "$HERE/scripts/economics.py" --resolve "$lookup" \
                --catalog-url "$CATALOG_URL")"
    st=$?
    if [ $st -eq 0 ]; then
        CATALOG_MATCH="$resolved"
        break
    elif [ $st -eq 2 ]; then
        echo "WARNING: pricing catalog unreachable; \$/gpu/hr will be n/a" >&2
        break
    fi
    # st=3: no match; suggestions were already printed on stderr
    if [ -t 0 ]; then
        printf 'Catalog model_id to price against (empty = no pricing): '
        read -r ans
        if [ -z "$ans" ]; then
            echo "  continuing without pricing"
            break
        fi
        CATALOG_MODEL="$ans"
    else
        echo "WARNING: no catalog match for '$lookup' and stdin is not a tty;" >&2
        echo "         continuing without pricing (or set --catalog-model)" >&2
        break
    fi
done

mkdir -p "$OUT_DIR"

# Generated prompt files are large (~250KB/entry at ISL 68000); clean up unless asked not to.
GENERATED=()
cleanup() {
    if [ "$KEEP_PROMPTS" = false ] && [ ${#GENERATED[@]} -gt 0 ]; then
        rm -f "${GENERATED[@]}"
    fi
}
trap cleanup EXIT INT TERM

FIXED_PROMPTS="$PROMPTS"
if [ -z "$FIXED_PROMPTS" ]; then
    [ -f "$CORPUS" ] || {
        echo "ERROR: corpus not found: $CORPUS" >&2
        echo "Build it first:  python3 $HERE/scripts/make_gsm8k_corpus.py $CORPUS 4272850" >&2
        exit 1
    }
fi

echo
echo "=== benchmark ==="
echo "  model       : $MODEL"
if [ -n "$CATALOG_MATCH" ]; then
    echo "  pricing     : $(printf '%s' "$CATALOG_MATCH" | awk -F'\t' '{print $2" (model_id "$1")"}')"
fi
echo "  url         : $URL"
echo "  ISL/OSL     : $ISL / $OSL   (cached prefix ${CACHE}%)"
echo "  concurrency : $CONCURRENCY"
if [ -n "$FIXED_PROMPTS" ]; then
    echo "  prompts     : $FIXED_PROMPTS  (reused at every point)"
else
    echo "  prompts     : prefix seed $SEED, unique tails per conc (seed+conc)"
fi
echo "  results     : $OUT_DIR"
echo

status_all=0
for conc in $(echo "$CONCURRENCY" | tr ',' ' '); do
    reqs=$(( conc * REQ_MULT ))
    art="$OUT_DIR/concurrency_${conc}"
    mkdir -p "$art"
    echo "--- concurrency $conc ($reqs requests) ---"

    if [ -n "$FIXED_PROMPTS" ]; then
        PROMPTS="$FIXED_PROMPTS"
    else
        # Same prefix pool (--seed) at every point so a live server can keep
        # hitting ~cache%. New tail seed so later points are not exact replays.
        tail_seed=$((SEED + conc))
        PROMPTS="$art/prompts.jsonl"
        GENERATED+=("$PROMPTS")
        echo "    building prompts (prefix seed $SEED, tail seed $tail_seed)"
        python3 "$HERE/scripts/make_prompt_file.py" "$PROMPTS" \
            --corpus "$CORPUS" --tokenizer "$TOKENIZER" \
            --isl "$ISL" --cache "$CACHE" \
            --num-prefix-prompts "$NUM_PREFIX_PROMPTS" \
            --entries "$reqs" --seed "$SEED" --tail-seed "$tail_seed" || exit 1
    fi

    aiperf profile \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
        --url "$URL" --api-key "$API_KEY" \
        --endpoint-type chat --streaming \
        --use-server-token-count \
        --input-file "$PROMPTS" --custom-dataset-type single_turn \
        --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
        --extra-inputs ignore_eos:true \
        --extra-inputs min_tokens:"$OSL" \
        --extra-inputs max_tokens:"$OSL" \
        --warmup-request-count "$WARMUP" \
        --concurrency "$conc" --request-count "$reqs" \
        --random-seed "$SEED" --ui simple \
        --output-artifact-dir "$art" \
        --profile-export-prefix "profile_c${conc}" \
        > "$art/aiperf.log" 2>&1
    st=$?
    if [ $st -ne 0 ]; then
        echo "    FAILED (exit $st) -- see $art/aiperf.log"
        status_all=1
    else
        echo "    OK"
    fi
done

if [ -n "$FIXED_PROMPTS" ]; then
    prompt_source="$FIXED_PROMPTS"
    unique_tails=false
else
    prompt_source="per-conc (prefix seed $SEED, tails seed+conc)"
    unique_tails=true
fi
cat > "$OUT_DIR/run_meta.json" <<EOF
{"model": "$MODEL", "url": "$URL", "isl": $ISL, "osl": $OSL, "cache": $CACHE,
 "num_prefix_prompts": $NUM_PREFIX_PROMPTS, "concurrency": "$CONCURRENCY",
 "req_mult": $REQ_MULT, "warmup": $WARMUP, "seed": $SEED, "gpus": $GPUS,
 "prompt_source": "$prompt_source", "unique_tails_per_conc": $unique_tails,
 "corpus": "$CORPUS", "tps_target": ${TPS_TARGET:-null}}
EOF

echo
echo "=== report ==="
GPUS="$GPUS" python3 "$HERE/scripts/report.py" "$OUT_DIR" | tee "$OUT_DIR/report.txt"

echo
echo "=== economics ==="
GPUS="$GPUS" python3 "$HERE/scripts/economics.py" "$OUT_DIR" \
    --catalog-url "$CATALOG_URL" \
    ${CATALOG_MODEL:+--catalog-model "$CATALOG_MODEL"} \
    ${TPS_TARGET:+--tps-target "$TPS_TARGET"} \
    | tee "$OUT_DIR/economics.txt" \
    || echo "economics step failed (benchmark results are unaffected)"

echo
echo "Results in $OUT_DIR"
exit $status_all
