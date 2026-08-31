# aiperf cache-validation profile

## What it simulates

aiperf drives **64 simultaneous users**, each holding their own chat
conversation with the model. Every user carries a large personal context
(~112k tokens — think uploaded documents or imported history) plus a 2k-token
system prompt shared by everyone. A user sends a message (~12.7k tokens),
reads the streamed reply, thinks for ~30 seconds (with wide variance), and
sends a follow-up — for 8 turns per conversation.
When a user's conversation ends, a new user with a fresh (uncached) context
takes their place, so the server sees constant session churn, just like a
production chat service.

Because each turn resends the same per-user context and growing history, a
server with working prefix/KV caching should serve most prompt tokens from
cache. That cache-hit percentage is the number this profile measures.

Reproducible multi-turn, long-context `aiperf profile` for exercising
prefix / KV cache behaviour under sustained chat load (DeepSeek-V4-Flash-0731
defaults).

The defaults are tuned to mimic real user behaviour, not a lab pattern:
turn counts and think time vary per session, completions stop where the model
stops, and finished users are replaced by fresh cache-cold conversations. To
reproduce the original fixed-length profile instead, see
[Legacy profile](#legacy-profile) below.

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

Print the exact aiperf command without running it:

```bash
./run_cache_validation.sh --dry-run --url "$API" --api-key "$API_KEY"
```

## What it runs, flag by flag

Every aiperf flag the wrapper emits, and why it is set:

### Endpoint and protocol

| Flag | Default | What it does |
|---|---|---|
| `--model` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Model id sent in each request body. Must match what the server serves. |
| `--tokenizer` | same as model | HuggingFace tokenizer used to build prompts of the requested token lengths. Must tokenize like the server's model or lengths will be off. |
| `--tokenizer-trust-remote-code` | on | Allows the tokenizer's custom Python code to run — required for DeepSeek tokenizers. |
| `--url` | `$API` | OpenAI-compatible base URL. |
| `--api-key` | `$API_KEY` | Sent as `Authorization: Bearer`. Required by the wrapper. |
| `--endpoint-type chat` | fixed | Uses `/v1/chat/completions`, which is what real chat traffic hits. |
| `--streaming` | fixed | Requests SSE streaming, enabling TTFT and inter-token-latency measurement — the metrics users actually feel. |
| `--use-server-token-count` | fixed | Trusts the server's `usage` block for token counts (including `prompt_cache_read_tokens`, the number this profile exists to measure) instead of re-tokenizing client-side. |

### Load shape — who arrives, when

| Flag | Default | What it does |
|---|---|---|
| `--user-centric-rate` | 1.6 | Enables user-centric scheduling at 1.6 requests/s aggregate. Each user's turns are spaced `num_users / rate` seconds apart (here 64 / 1.6 = 40 s floor), independent of other users — like real users who don't coordinate. This mode is designed for KV-cache benchmarking; it cannot be combined with `--request-rate` or `--arrival-pattern`. |
| `--num-users` | 64 | Number of concurrent simulated users, i.e. how many distinct 112k contexts the cache must hold at once. |
| `--conversation-num` | 256 | Total unique conversations for the run. Set **above** `--num-users` so that when a user's conversation ends, a fresh cache-cold conversation takes its place — real services see constant session churn, and a benchmark where the same 64 sessions run forever overstates cache hit rates. |
| `--conversation-turn-mean` | 8 | Mean turns per conversation. Real chat sessions are mostly short, not uniform marathons. |
| `--conversation-turn-stddev` | 0 | Spread of turns per conversation. **Must stay 0 with `--user-centric-rate`**: aiperf samples each session's turn count independently from the generated conversation's actual length, so any nonzero stddev eventually raises `num_turns (N) exceeds conversation length (M)` in the workers. The wrapper warns if you set it. |
| `--conversation-turn-delay-mean` | 30000 ms | Mean think time between receiving a response and sending the next message. |
| `--conversation-turn-delay-stddev` | 25000 ms | Spread of think time. High relative to the mean on purpose: real think time is heavy-tailed (fast follow-ups and long pauses), and a wide normal distribution is the closest available approximation. This is also what stresses cache TTL — entries must survive the long pauses. |

### Context and dataset — what the cache has to hold

| Flag | Default | What it does |
|---|---|---|
| `--shared-system-prompt-length` | 2000 | Tokens of system prompt shared by *all* users. Cache-friendly: one copy serves everyone. Match to your production system prompt size. |
| `--user-context-prompt-length` | 112000 | Tokens of context unique to each user (documents, memory, history import), resent on every turn of that user's conversation. This is the payload whose caching is being validated. Requires `--num-dataset-entries`. |
| `--num-dataset-entries` | 256 | Unique user-turn prompts generated for the dataset. Kept high so turns don't replay verbatim — verbatim replays inflate cache hit rates in a way real traffic never does. |

### Token lengths — per-turn traffic

| Flag | Default | What it does |
|---|---|---|
| `--isl` / `--isl-stddev` | 12700 / 4000 | Per-turn user-message input tokens, normally distributed. Sits on top of the system prompt, user context, and accumulated conversation history. |
| `--osl` / `--osl-stddev` | 917 / 300 | Requested output tokens per turn (sets `max_tokens` per request), normally distributed. |
| `--sequence-distribution` | off (`--seq-dist` to enable) | Replaces the single ISL/OSL pair with a weighted mix, e.g. `"1500\|500,300\|150:60;12000\|4000,900\|300:30;80000\|20000,1200\|400:10"` = 60% quick questions, 30% working sessions, 10% long-document turns. The closest match to a production endpoint's heterogeneous traffic. |

### Behavior quirks — what real users do that benchmarks skip

| Flag | Default | What it does |
|---|---|---|
| `--extra-inputs ignore_eos:true` | **off** (enable with `--ignore-eos`) | When on, forces the model to generate the full `max_tokens` regardless of content. Realistic mode leaves it off so completions end where the model stops — output length becomes model-dependent (less comparable across models) but matches what users receive. Turn it on when you need exact token accounting for capacity math. |
| `--request-cancellation-rate` | 0 (enable with `--cancel-rate`, try 5) | Percentage of requests aborted mid-stream — users hitting stop or closing the tab. Exercises the server's cancellation path and KV cleanup, which pure-completion benchmarks never touch. |
| `--request-cancellation-delay` | 2 s (`--cancel-delay`) | How long a to-be-cancelled request streams before the abort. |
| `--goodput` | off (`--goodput "K:V ..."`) | Space-separated SLO pairs, e.g. `"time_to_first_token:2000 inter_token_latency:50"` (ms). The report then counts only requests meeting all SLOs — "requests a human would tolerate" rather than raw completions. |

### Run control and output

| Flag | Default | What it does |
|---|---|---|
| `--benchmark-duration` | 1800 s | Measured window (30 min). Long enough for cache warmup, churn, and think-time cycles to reach steady state. |
| `--random-seed` | fresh each run | Seeds prompt generation and length sampling. Fresh by default so repeated runs don't replay identical prompts; pin via `--random-seed` / `$RANDOM_SEED` for reproducible A/B comparisons. |
| `--server-metrics` + `--server-metrics-formats json parquet` | on | Scrapes the server's metrics endpoint during the run and exports it in both formats under the artifact dir. |
| `--artifact-dir` | `./aiperf_cache_validation` | Where all exports land. |
| `--ui simple` | fixed | Plain-text progress output, safe for logs and CI. |
| `--export-level raw` | fixed | Exports every per-request record, not just aggregates — needed to analyze cache-read distributions and per-turn behaviour after the run. |

## Reading the result

The headline number is **Overall Usage Prompt Cache Read %** in the LLM
metrics table. Interpretation guide from measured runs of this profile against
DeepSeek-V4-Flash-0731 (see repo history / benchmark reports):

- Dedicated endpoint, healthy cache: **~79–91%** overall, p50 cache read in the
  hundreds of thousands of tokens.
- Broken or non-sticky caching: **~28%** overall with p50 cache read near zero
  — the per-user context is being reprocessed every turn.

A p50 cache-read near zero with a decent average means only a minority of
requests (deep conversations) ever hit the cache: check for load balancing
without session affinity or aggressive eviction.

## Troubleshooting

**`ValueError: num_turns (N) exceeds conversation length (M)`** in worker
logs: you are running with a nonzero `--turn-stddev` (or an aiperf version
with the same scheduling behaviour). In user-centric rate mode, aiperf's
scheduler draws each session's turn count from the turn distribution
independently of the turn count the conversation was actually generated
with, so variance makes the two disagree. Set `--turn-stddev 0` (the
default) and rerun.

## Legacy profile

The original profile used uniform 32-turn conversations, no churn, and pinned
output lengths. Reproduce it with:

```bash
./run_cache_validation.sh \
  --url "$API" --api-key "$API_KEY" \
  --turn-mean 32 --turn-stddev 0 \
  --turn-delay-stddev 15000 \
  --conversations 64 --dataset-entries 64 \
  --ignore-eos \
  --random-seed 303
```

Note that at 32 turns the final turn's input approaches ~550k tokens
(2k system + 112k context + 32×12.7k turn inputs + accumulated outputs) —
make sure the model's context window allows it, or expect errors on deep turns.

## Examples

```bash
# Realistic default run
./run_cache_validation.sh --url "$API" --api-key "$API_KEY"

# Mixed workload: 60% short Q&A, 30% working sessions, 10% long-document turns
./run_cache_validation.sh --url "$API" --api-key "$API_KEY" \
  --seq-dist "1500|500,300|150:60;12000|4000,900|300:30;80000|20000,1200|400:10"

# With 5% user abandonment and SLO-based goodput reporting
./run_cache_validation.sh --url "$API" --api-key "$API_KEY" \
  --cancel-rate 5 --cancel-delay 2 \
  --goodput "time_to_first_token:2000 inter_token_latency:50"

# Pinned lengths for capacity math (exact token accounting, less realistic)
./run_cache_validation.sh --url "$API" --api-key "$API_KEY" --ignore-eos
```

## Requirements

- [`aiperf`](https://github.com/ai-dynamo/aiperf) on `PATH` (`pip install aiperf`)
- Network access to an OpenAI-compatible chat endpoint
- HuggingFace access for the tokenizer (`deepseek-ai/DeepSeek-V4-Flash-0731`
  by default), or a local tokenizer checkout

## Output

Artifacts land in `--artifact-dir`: aiperf's profile export (raw per-request
records), plus server metrics in JSON and parquet. A full run is ~30 minutes
of measured load after aiperf finishes dataset setup.
