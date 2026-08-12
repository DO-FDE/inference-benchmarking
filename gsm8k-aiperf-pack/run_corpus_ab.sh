#!/usr/bin/env bash
#
# run_corpus_ab.sh -- measure what the GSM8K prompts are worth on YOUR server.
#
# Runs the same benchmark twice, changing only the prompt text: aiperf's stock
# literary corpus vs GSM8K word problems. Everything else -- server, flags, ISL,
# seed, concurrency -- is held fixed, so the difference is attributable to the
# prompt content.
#
# Arms are interleaved (stock, gsm8k, stock, gsm8k) rather than run in blocks,
# so slow server-side drift does not load onto one arm.
#
# NOTE: the stock arm requires temporarily replacing a data file inside the
# installed aiperf package. The original is backed up and restored on exit,
# including on Ctrl-C. A `kill -9` bypasses that restore; if it happens, run
# this script with --restore-only to put the original back.
#
# Usage:
#   ./run_corpus_ab.sh --model /models/Kimi-K3
#   ./run_corpus_ab.sh --model /models/Kimi-K3 --reps 3 --concurrency 16
#   ./run_corpus_ab.sh --restore-only
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
SEED=42
GPUS=8
REPS=2
CORPUS="$HERE/gsm8k_corpus.txt"
OUT_DIR="$HERE/ab_results_$(date +%Y%m%d_%H%M%S)"
RESTORE_ONLY=false

usage() {
    cat <<'EOF'
run_corpus_ab.sh -- A/B aiperf's stock corpus against GSM8K prompts.

  --model PATH/NAME     Model id the server serves            (required)
  --tokenizer PATH      Tokenizer                             (default: --model)
  --url URL             Server base URL                       (default: http://localhost:8000)
  --api-key KEY         API key                               (default: EMPTY)
  --isl N               Total input tokens                    (default: 68000)
  --osl N               Output tokens                         (default: 350)
  --cache PCT           Percent of ISL reused as prefix       (default: 93)
  --concurrency LIST    Comma-separated                       (default: 16)
  --reps N              Repeats per arm                       (default: 2)
  --req-mult N          Requests per concurrency = N x conc   (default: 5)
  --warmup N            Warmup requests                       (default: 3)
  --seed N              Random seed                           (default: 42)
  --gpus N              GPU count                             (default: 8)
  --corpus PATH         GSM8K corpus text                     (default: ./gsm8k_corpus.txt)
  --out-dir PATH        Results directory
  --restore-only        Restore the stock aiperf corpus and exit

Run at least 2 reps. Acceptance can vary run to run; a single pair of runs
cannot distinguish a real effect from noise.
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
        --reps) REPS="$2"; shift 2 ;;
        --req-mult) REQ_MULT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --restore-only) RESTORE_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# Locate the corpus asset inside the installed aiperf package.
ASSET=$(python3 - <<'PY'
from pathlib import Path
import aiperf.dataset.generator.prompt as p
print(Path(p.__file__).parent / p.DEFAULT_CORPUS_FILE)
PY
) || { echo "ERROR: could not locate aiperf's corpus asset. Is aiperf installed?" >&2; exit 1; }
[ -f "$ASSET" ] || { echo "ERROR: aiperf corpus asset not found at $ASSET" >&2; exit 1; }

BACKUP="$HERE/.aiperf_corpus_original.txt"

if [ "$RESTORE_ONLY" = true ]; then
    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$ASSET" && rm -rf ~/.cache/aiperf
        echo "restored stock corpus to $ASSET"
        md5sum "$ASSET"
    else
        echo "no backup at $BACKUP -- nothing to restore." >&2
        echo "Reinstall aiperf to get a clean corpus: pip install --force-reinstall aiperf" >&2
        exit 1
    fi
    exit 0
fi

[ -n "$MODEL" ] || { echo "ERROR: --model is required" >&2; usage; exit 2; }
[ -n "$TOKENIZER" ] || TOKENIZER="$MODEL"
[ -f "$CORPUS" ] || {
    echo "ERROR: corpus not found: $CORPUS" >&2
    echo "Build it:  python3 $HERE/scripts/make_gsm8k_corpus.py $CORPUS 4272850" >&2
    exit 1
}
command -v aiperf >/dev/null || { echo "ERROR: aiperf not on PATH" >&2; exit 1; }
if ! curl -sf -m 10 "$URL/health" >/dev/null 2>&1 \
   && ! curl -sf -m 10 "$URL/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
    echo "ERROR: no server responding at $URL" >&2
    exit 1
fi

# Back up the pristine corpus once, before anything is swapped.
[ -f "$BACKUP" ] || cp "$ASSET" "$BACKUP"

restore() {
    cp "$BACKUP" "$ASSET" 2>/dev/null && echo "[restored stock aiperf corpus]"
}
trap restore EXIT INT TERM

mkdir -p "$OUT_DIR"

prefix_len=$(( ISL * CACHE / 100 ))
fresh_len=$(( ISL - prefix_len ))

run_arm() {
    local arm="$1" corpus_file="$2" rep="$3"
    local arm_dir="$OUT_DIR/${arm}_rep${rep}"
    mkdir -p "$arm_dir"

    echo
    echo "==================== ${arm} rep${rep} ===================="
    cp "$corpus_file" "$ASSET"
    # aiperf caches the tokenised corpus keyed by tokenizer, not by content, so a
    # stale cache would silently replay the other arm's prompts.
    rm -rf ~/.cache/aiperf
    local md5; md5=$(md5sum "$ASSET" | cut -d' ' -f1)
    echo "corpus md5: $md5"

    for conc in $(echo "$CONCURRENCY" | tr ',' ' '); do
        local reqs=$(( conc * REQ_MULT ))
        local art="$arm_dir/concurrency_${conc}"
        mkdir -p "$art"
        echo "--- concurrency $conc ($reqs requests) ---"
        aiperf profile \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
            --url "$URL" --api-key "$API_KEY" \
            --endpoint-type chat --streaming \
            --use-server-token-count \
            --num-prefix-prompts "$NUM_PREFIX_PROMPTS" \
            --prompt-prefix-length "$prefix_len" \
            --synthetic-input-tokens-mean "$fresh_len" --synthetic-input-tokens-stddev 0 \
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
        [ $? -eq 0 ] && echo "    OK" || echo "    FAILED -- see $art/aiperf.log"
    done

    cat > "$arm_dir/run_meta.json" <<EOF
{"arm": "$arm", "rep": $rep, "corpus_md5": "$md5", "model": "$MODEL",
 "isl": $ISL, "osl": $OSL, "cache": $CACHE, "concurrency": "$CONCURRENCY", "gpus": $GPUS}
EOF
}

echo "A/B: stock aiperf corpus vs GSM8K"
echo "  aiperf asset : $ASSET"
echo "  backup       : $BACKUP"
echo "  reps         : $REPS   concurrency: $CONCURRENCY"
echo "  ISL/OSL      : $ISL / $OSL  (prefix $prefix_len + fresh $fresh_len)"
echo "  results      : $OUT_DIR"

for rep in $(seq 1 "$REPS"); do
    run_arm stock "$BACKUP" "$rep"
    run_arm gsm8k "$CORPUS" "$rep"
done

echo
echo "==================== comparison ===================="
GPUS="$GPUS" python3 "$HERE/scripts/compare_ab.py" "$OUT_DIR" | tee "$OUT_DIR/comparison.txt"
echo
echo "Results in $OUT_DIR"
