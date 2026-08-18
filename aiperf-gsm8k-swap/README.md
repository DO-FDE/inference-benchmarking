# inference-benchmarking

Tools for benchmarking and characterizing LLM inference-serving performance.
This repo collects the scripts used to load-test served LLM endpoints (vLLM,
SGLang, dedicated/hosted endpoints), measure latency and throughput under
concurrency, score results against SLA targets, and — for speculative-decoding
servers — measure draft acceptance and prefix-cache hit rate.

One entry point, `run_benchmark.sh`, wraps the whole flow: it runs an aiperf
concurrency sweep by default, and two independent opt-in stages layer on top of
it (a coherent-prompt corpus swap, and fleet acceptance/cache capture).

## Layout

```
run_benchmark.sh                Main entry point. Runs aiperf by default; opt-in stages.
run_aiperf_workload_shapes.sh   Concurrency-sweep + SLA-scoring harness (aiperf).
parse_sweep.py                  Sweep export -> per-GPU summary CSV.
prep_aiperf_corpus.sh           Swap aiperf's stock corpus for GSM8K (coherent prompts).
capture_accept_cache.sh         Fleet accept% + cache-hit% capture around each point.
gsm8k_corpus.txt                GSM8K corpus (auto-built on first --prep-gsm8k run).
```

## What each piece does

- **`run_benchmark.sh`** — orchestrator. Resolves one endpoint + workload shape
  and drives the sweep. By default it calls the aiperf harness directly. With
  `--prep-gsm8k` it first swaps in the GSM8K corpus; with `--capture-accept` it
  runs the sweep through the capture wrapper so each concurrency point also
  yields acceptance and cache-hit numbers. The two flags are independent and
  additive.

- **`run_aiperf_workload_shapes.sh`** — the sweep harness. Drives `aiperf
  profile` across a concurrency grid for each workload shape and cache ratio
  (fraction of the input served from a reused cached prefix), against a local
  port or any remote/hosted endpoint. Each run is scored once per ITL gate
  (`claw` < 25 ms, `chat` < 66.7 ms) with no re-run. Supports a "simple mode"
  where you describe one ad-hoc shape inline with `--isl/--osl/--cache`.

- **`parse_sweep.py`** — post-processor. Converts an aiperf sweep export into a
  per-GPU summary CSV: per-GPU throughput, TTFT/ITL percentiles, identity
  validation, degeneracy detection, per-point SLA pass/fail, and the max
  concurrency under the ITL gate. The harness calls it automatically; it also
  runs standalone.

- **`prep_aiperf_corpus.sh`** — swaps aiperf's built-in prompt corpus
  (Shakespeare) for a GSM8K corpus, backing up the original once so the swap is
  reversible (`--restore`). Coherent prompts make speculative-decode acceptance
  representative of real traffic instead of understating it. Builds the corpus
  from the `openai/gsm8k` dataset on first run.

- **`capture_accept_cache.sh`** — brackets each concurrency point with a
  before/after scrape of one or more `/metrics` endpoints, summed and
  window-diffed, to compute speculative-decoding `accept%`, `accept length`, and
  prefix-cache `hit%`. Writes `accept_cache.csv` alongside the harness's SLA CSV.

## Requirements

The script is **self-bootstrapping** — it installs what it can and checks the
rest before running. You need only:

- **Python 3** and **`kubectl`/network access to the endpoint** (whatever reaches
  your server).
- A served, OpenAI-compatible chat endpoint (prefix caching enabled if you use
  non-zero cache ratios).

`run_benchmark.sh` handles the rest automatically:

- **`chmod +x`** on its helper scripts.
- **Installs `aiperf`** (and `datasets`, only when `--prep-gsm8k` is used) if
  missing — trying several pip strategies so it works in venvs, system Python,
  PEP-668 "externally managed" environments, and rootless containers. Skip with
  `--skip-deps` if you manage deps yourself.
- **Probes the endpoint** before running and, if it can't be reached, explains
  why (cluster-internal IP, missing auth, etc.). Skip with `--skip-reach`.
- **Checks `/metrics`** exposes the `spec_decode` counters when `--capture-accept`
  is set, warning early if acceptance would come back `n/a`.

For `--capture-accept` the server must expose `vllm:spec_decode_*` and
`vllm:prefix_cache_*` on `/metrics`; by default this is read from `--url`.

## Endpoints: how to point the sweep

The only things a run requires are the **endpoint URL**, the **model name**, and
the **workload shape** (`--isl/--osl/--cache/--concurrency`). Nothing about the
deployment matters — Docker, Kubernetes, a bare server, or a hosted endpoint are
all just a URL to aiperf.

`run_benchmark.sh` passes `--url` (and, if set, `--api-key`) straight through to
the harness, which forwards both to aiperf. Three common cases:

```bash
# 1. Local server on a port (no auth):
./run_benchmark.sh --url http://localhost:8000/

# 2. A direct IP/host (no auth) — e.g. a Service or a single server:
./run_benchmark.sh --url http://10.0.0.5:80/

# 3. A remote / dedicated / hosted endpoint behind a bearer key:
./run_benchmark.sh --url https://your-host/ --api-key YOUR_KEY
```

The health probe is skipped by default in this wrapper (many hosted endpoints
don't expose an unauthenticated `/health`); the sweep goes straight to
`aiperf profile`. When `--api-key` is set it is also used to authenticate the
sweep's requests.

## Quick start

Plain sweep (stock prompts, SLA-gate CSV) — a quick characterization:

```bash
./run_benchmark.sh --url http://localhost:8000/
```

Coherent prompts (recommended for any speculative-decoding server — makes the
measured acceptance/ITL representative of real traffic):

```bash
./run_benchmark.sh --url http://localhost:8000/ --prep-gsm8k
```

Coherent prompts + acceptance/cache capture (the full picture). For a single
endpoint, metrics are read from `--url` automatically — nothing extra:

```bash
./run_benchmark.sh --url http://localhost:8000/ --prep-gsm8k --capture-accept
```

Ad-hoc shape and remote endpoint:

```bash
./run_benchmark.sh --url https://host/ --api-key KEY \
  --model <served-id> --isl 8000 --osl 1000 --cache 90 \
  --concurrency 1,2,4,8,16,32
```

Run any script with `--help` for its full flag list.

## The three modes

| flags | prompts | outputs | use it for |
|---|---|---|---|
| *(none)* | aiperf stock | SLA-gate CSV | quick sweep; understates spec-decode |
| `--prep-gsm8k` | GSM8K (coherent) | SLA-gate CSV | representative throughput/latency |
| `--prep-gsm8k --capture-accept` | GSM8K (coherent) | SLA CSV **+** `accept_cache.csv` | full: adds accept% + cache-hit% |

`--capture-accept` works without `--prep-gsm8k` too — it just measures
acceptance on whatever prompts are active (stock prompts understate it).

## Options (run_benchmark.sh)

```
Core:
  --model NAME          Served model id (must match what the server serves)
  --tokenizer NAME      Tokenizer                          (default: --model)
  --url URL             Endpoint the sweep hits            (default: http://localhost:8000/)
  --api-key KEY         Bearer key for the endpoint        (optional; enables auth)
  --isl N               Total input tokens per request     (default: 120000)
  --osl N               Output tokens per request          (default: 917)
  --cache PCT           Percent of ISL served from a reused prefix (default: 90)
  --concurrency LIST    Comma-separated concurrency grid   (default: 16,24,32,48,64,128)
  --gpus N              Per-GPU throughput divisor         (default: 8)
  --warmup N            Warmup requests                    (default: 16)
  --out-dir PATH        Results directory                  (default: results_<timestamp>)
  --corpus PATH         GSM8K corpus file                  (default: ./gsm8k_corpus.txt)

Optional stages:
  --prep-gsm8k          Swap in the GSM8K corpus before running
  --capture-accept      Capture fleet accept% + cache-hit% around each point
  --metrics-addrs CSV   Comma-separated /metrics hosts to scrape (alias: --pod-ips)
  --metrics-port N      Port for the /metrics scrape        (default: 8000)

  -h, --help            Show help
```

`--gpus` only scales the throughput columns (aggregate ÷ N) for per-GPU
reporting; it does not change what is run. Set it to the number of GPUs actually
serving the endpoint so the per-GPU numbers are honest.

## --capture-accept: what to scrape

Acceptance and cache-hit come from the server's `/metrics` counters, diffed
across each point's window. Two rules make the numbers valid:

1. **Scrape the serving engine(s) directly, not a load balancer.** A round-robin
   Service (or any LB) will round-robin the `/metrics` GET too, so the counters
   you read won't correspond to the traffic that was served — producing
   impossible values (e.g. cache hit > 100%). Pass the direct address(es) of the
   engine(s) via `--metrics-addrs`. If the endpoint fans out across several
   engines, list all of them: the script sums the counters across all addresses
   and window-diffs the total, giving a correct fleet aggregate.

2. **Match the metrics port.** `--metrics-port` (default 8000) is the port the
   scrape hits on each address; set it to wherever the server exposes `/metrics`.

If the endpoint is a single server, `--metrics-addrs <that-host>` is all you
need. The benchmark `--url` and the scrape addresses are independent — the sweep
can go through one endpoint while metrics are read from the engine(s) behind it.

### Counter names

The capture matches these counter base names (a `_total` suffix is fine):

```
vllm:spec_decode_num_draft_tokens
vllm:spec_decode_num_accepted_tokens
vllm:spec_decode_num_drafts
vllm:prefix_cache_queries
vllm:prefix_cache_hits
```

Verify your server exposes them before the first capture run:

```bash
curl -s http://<engine-addr>:<metrics-port>/metrics \
  | grep -iE "spec_decode|prefix_cache" | grep -v '^#'
```

If a base name differs, edit the five `sum_counter` lines in
`capture_accept_cache.sh` to match; everything else is unaffected. If no
`spec_decode` counters appear, the server isn't running a draft model (or
doesn't export them) and acceptance will read `n/a` — throughput/latency still
work.

## SLA gates (TTFT and ITL)

`parse_sweep.py` scores each run against latency targets and reports pass/fail
per point. Set them from `run_benchmark.sh`:

- **`--ttft-p50 MS`** and **`--ttft-p90 MS`** — TTFT targets in milliseconds.
  Off by default (a large sentinel), so TTFT is reported but not gated unless you
  set them.
- **`--gate NAME:ITL_MS`** — an ITL p50 gate in milliseconds, named. Repeatable:
  pass it multiple times to score the same run against several ITL tiers in one
  pass (no re-run). If you pass none, the harness defaults apply
  (`claw:25`, `chat:66.7`).

Example — your workload's SLA (TTFT p50 < 4s / p90 < 8s, ITL strict 9.17ms with
a flex tier at 14.28ms):

```bash
./run_benchmark.sh --url http://ENDPOINT:80/ --prep-gsm8k \
  --isl 120000 --osl 917 --cache 90 --concurrency 16,24,32,48,64,128 --gpus 8 \
  --ttft-p50 4000 --ttft-p90 8000 \
  --gate strict:9.17 --gate flex:14.28
```

Each `--gate` produces its own scored CSV row set (the same measured run scored
against each ITL target), and the combined CSV carries the pass/fail columns and
the max concurrency under each ITL gate. TTFT targets apply to every row.

## Outputs

Everything lands in `--out-dir`:

- `summary_ALL_<ts>.csv` — per-GPU throughput, TTFT p50/p90, ITL p50/p90, and
  per-point SLA pass/fail, one row per (shape, cache, gate).
- `accept_cache.csv` (only with `--capture-accept`) — per concurrency:
  `accept_pct, accept_len, cache_hit_pct` plus the raw counter deltas.
- aiperf's own artifacts under the harness's working directories.

## Reading the metrics

- **accept %** — accepted draft tokens / drafted tokens. Per drafted token, so
  NOT comparable across different `num_speculative_tokens`. Use accept length
  for cross-config comparison.
- **accept length** — mean tokens committed per forward pass, ceiling N+1 for
  `num_speculative_tokens=N`. The right cross-config comparison.
- **cache_hit %** — prefix-cache hits / queries over the window. Should sit near
  `--cache` once prefixes are warm.
- **TTFT p90** — cold-start-dominated: the first `<concurrency>` requests each do
  a full-length prefill. Treat as cold-start-inclusive; compare like-for-like.
- **Run-to-run variance.** Cache-hit rate is state-dependent (the same config has
  measured very differently depending on what ran before), and acceptance can
  swing across repeats. Run >=2 reps, report a median, show the spread; restart
  the server between runs for a clean cache state.

## Notes

- `--prep-gsm8k` modifies aiperf's installed corpus asset (backed up once). Undo
  with `./prep_aiperf_corpus.sh --restore`.
- The wrapper skips the health probe and passes `--skip-health-check` to the
  harness, so dedicated endpoints that reject anonymous `/health` still run.
- `parse_sweep.py` can be run by hand on an existing sweep:
  `python3 parse_sweep.py --sweep-dir /path/to/sweep_aggregate --gpus 8 --osl-target 917 --itl-p50-target 25 --out-csv summary.csv`

## --capture-accept: where metrics come from

Acceptance and cache-hit are read from the server's `/metrics` counters, diffed
across each concurrency point's window.

- **Default (single endpoint):** metrics are scraped from `--url` itself. A
  single container, a single server, or a hosted endpoint needs nothing extra —
  just `--capture-accept`.
- **Fan-out endpoints (optional):** if `--url` is a router/load balancer in
  front of several engines, scraping the router is wrong — it round-robins the
  `/metrics` GET too, so the counters won't match the served traffic (you get
  impossible values like cache hit > 100%). In that case pass
  `--metrics-addrs` with the engines' own base URLs; the script sums the
  counters across them and window-diffs the total:

  ```bash
  --metrics-addrs http://engine1:8000,http://engine2:8000,http://engine3:8000
  ```

  Addresses may be full URLs (`http://host:port`) or bare `host:port`; `/metrics`
  is appended automatically.

### Counter names

The capture matches these exact counter names:

```
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_drafts_total
vllm:prefix_cache_queries_total
vllm:prefix_cache_hits_total
```

Verify your server before the first capture run:

```bash
curl -s <url>/metrics | grep -iE "spec_decode|prefix_cache" | grep -v '^#'
```

If a name differs (some builds use `vllm:gpu_prefix_cache_*`), edit the matching
`sum_counter` line in `capture_accept_cache.sh`. If no `spec_decode` counters
appear, the server isn't running a draft model and acceptance reads `n/a` —
throughput/latency still work.
