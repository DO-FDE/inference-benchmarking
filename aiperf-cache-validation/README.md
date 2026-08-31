# aiperf cache-validation profile

Reproducible multi-turn, long-context `aiperf profile` for exercising
prefix / KV cache behaviour under sustained chat load (DeepSeek-V4-Flash-0731
defaults).

## What it runs

64 concurrent users × 64 conversations, ~32 turns each, with:

| Knob | Value | Meaning |
|---|---:|---|
| `--user-centric-rate` | 1.6 | Arrival rate (users / s) |
| `--num-users` | 64 | Concurrent users |
| `--conversation-num` | 64 | Conversations in the dataset |
| `--conversation-turn-mean` | 32 | Turns per conversation |
| `--conversation-turn-delay-mean` | 30000 ms | Think time between turns (±15 s) |
| `--shared-system-prompt-length` | 2000 | Shared system prompt tokens |
| `--user-context-prompt-length` | 112000 | Per-user context tokens |
| `--isl` / `--osl` | 12700 ± 4000 / 917 ± 300 | Per-turn input / output |
| `--benchmark-duration` | 1800 s | Measured window (30 min) |
| `--random-seed` | fresh each run | Override with `--random-seed` / `$RANDOM_SEED` |

Streaming chat, server token counts, `ignore_eos:true`, and server metrics
exported as JSON + parquet under the artifact dir.

## Requirements

- [`aiperf`](https://github.com/ai-dynamo/aiperf) on `PATH` (`pip install aiperf`)
- Network access to an OpenAI-compatible chat endpoint
- HuggingFace access for the tokenizer (`deepseek-ai/DeepSeek-V4-Flash-0731`
  by default), or a local tokenizer checkout

## Quick start

```bash
cd aiperf-cache-validation

export API="https://your-endpoint/"
export API_KEY="do_dedicated_..."
# optional overrides:
# export MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
# export ARTIFACT_DIR="/root/aiperf_cache_validation"

./run_cache_validation.sh
```

Or with flags:

```bash
./run_cache_validation.sh \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --url "$API" \
  --api-key "$API_KEY" \
  --artifact-dir ./aiperf_cache_validation
```

Print the exact command without running it:

```bash
./run_cache_validation.sh --dry-run --url "$API" --api-key "$API_KEY"
```

## Raw command (copy-paste)

If you prefer not to use the wrapper:

```bash
aiperf profile \
  --model "$MODEL" \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tokenizer-trust-remote-code \
  --url "$API" \
  --api-key "$API_KEY" \
  --endpoint-type chat \
  --streaming \
  --use-server-token-count \
  --user-centric-rate 1.6 \
  --num-users 64 \
  --conversation-num 64 \
  --conversation-turn-mean 32 \
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
  --benchmark-duration 1800 \
  --random-seed "$RANDOM_SEED" \
  --server-metrics \
  --server-metrics-formats json parquet \
  --artifact-dir /root/aiperf_cache_validation \
  --ui simple \
  --export-level raw
```

Set `MODEL`, `API`, and `API_KEY` first. On a non-root host, change
`--artifact-dir` to a writable path (the wrapper defaults to
`./aiperf_cache_validation`).

## Output

Artifacts land in `--artifact-dir` (JSON + parquet server metrics, raw export).
A full run is ~30 minutes of measured load after aiperf finishes dataset setup.
