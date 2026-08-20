# GSM8K prompts for aiperf

Benchmark a speculative-decoding server with **coherent prompts** instead of
aiperf's stock literary corpus.

## Why

aiperf builds synthetic prompts by slicing `assets/shakespeare.txt`. At long
input lengths each request becomes a large block of Elizabethan prose containing
no task. A speculative draft model predicting the next token of arbitrary
Shakespeare is predicting against a near-unconstrained distribution, so draft
acceptance measures far lower than it does on real traffic — and acceptance is
what determines how much speculative decoding actually buys you.

Swapping in GSM8K grade-school word problems — coherent text with a task,
reasoning steps and an answer — measured on 8× MI355X, Kimi-K3 + DSpark draft
model, `num_speculative_tokens=3`, ISL 68000 / OSL 350 / 93% cached prefix,
concurrency 16:

| Prompts | Draft acceptance | Accept length | ITL P50 |
|---|---:|---:|---:|
| Stock literary corpus | 41.90 % | 2.26 | 38.18 ms |
| GSM8K | 61.32 % | 2.84 | 35.97 ms |

+19.4 points of acceptance, accept length 2.26 → 2.84 against a ceiling of 4.
Reproducible to within 0.65 points across repeats.

**This is a change to how you measure, not to what you run.** It makes the
benchmark representative of coherent traffic. It does not make the server faster
for a workload it was already serving. Read the numbers as "the stock corpus was
understating speculative decoding", not as a performance gain in the product.

## Requirements

- `aiperf` 0.11.0 on PATH
- Python 3.10+, `transformers`, `datasets`
- A running OpenAI-compatible inference server
- For acceptance metrics: the server must expose `vllm:spec_decode_*` counters
  on `/metrics`. Without them everything else still works; acceptance shows `n/a`.

## Quick start

```bash
./run_gsm8k_benchmark.sh --model /models/Kimi-K3 --concurrency 16,24
```

`gsm8k_corpus.txt` ships with this pack, so no HuggingFace access is needed. To
rebuild it yourself (md5 `3ff4dd69fc6b9afeb9ff315d68324ff2`):

```bash
python3 scripts/make_gsm8k_corpus.py gsm8k_corpus.txt 4272850
```

Output:

```
 conc  reqs     ISL  cache%  TTFT p50  TTFT p90  ITL p50  in/s/gpu  out/s/gpu  accept%  acc.len
   16    80   68088   80.11       878     18146    34.62    6901.2      35.48    59.60     2.79
      per-position acceptance %: 78.0, 57.6, 43.2
```

Everything lands in the results directory: `report.txt`, `report.json`,
`run_meta.json` recording the exact configuration, and aiperf's own artifacts.

After the report, a post-benchmark economics step prints the date, output TPS
per GPU and per node, and $/GPU/hr at DigitalOcean serverless token prices from
the [gen-ai model catalog](https://api.digitalocean.com/v2/gen-ai/models/catalog)
(cached-prefix input tokens billed at the cache-read rate). Pass
`--tps-target N` to also get the number of nodes needed to serve N output
tokens/s. The catalog model is inferred from `--model`; override with
`--catalog-model <model_id>` if the match is wrong. Results land in
`economics.txt` / `economics.json`. If the catalog is unreachable, the step
degrades to TPS-only and never fails the run.

The name is checked against the catalog **before** the benchmark starts: on no
match the script prints the closest catalog `model_id`s and asks you to type a
correct one (empty answer = run without pricing). Non-interactive runs skip the
prompt and continue without pricing.

```
      date  conc  out tok/s/gpu  out tok/s/node  $/gpu/hr  nodes@target
-----------------------------------------------------------------------
2026-08-20    16          35.48          283.84     21.58            18
```

## Measuring the gain on your own hardware

Don't take the numbers above on trust — the effect size depends on your model,
draft model and workload:

```bash
./run_corpus_ab.sh --model /models/Kimi-K3 --concurrency 16 --reps 2
```

Runs stock and GSM8K arms interleaved, changing only the prompt text, and prints
a comparison with per-run values and a pairwise win count.

> The stock arm temporarily replaces a data file **inside the installed aiperf
> package**. The original is backed up to `.aiperf_corpus_original.txt` and
> restored on exit including Ctrl-C. A `kill -9` bypasses that; recover with
> `./run_corpus_ab.sh --restore-only`. If you would rather never touch
> site-packages, use `run_gsm8k_benchmark.sh` only — it uses aiperf's documented
> `--input-file` path and modifies nothing.

## Two ways to feed GSM8K prompts

| | `run_gsm8k_benchmark.sh` (default) | `run_corpus_ab.sh` |
|---|---|---|
| Mechanism | `--input-file` + `--custom-dataset-type single_turn` | Swaps aiperf's corpus asset |
| Installed package | Untouched | One data file swapped, restored after |
| Measured acceptance (conc 16) | 57.5–59.6 % | 61.2–61.4 % |
| Use it for | Normal benchmarking | A/B against the stock corpus |

Both are well above the stock corpus's ~42%. The corpus-swap path reads slightly
higher because aiperf regenerates prompts internally per run, while the
`--input-file` path reuses one pre-built file; see `docs/METHOD.md`.

`--public-dataset spec_al_gsm8k` **cannot** substitute for either. It routes
through aiperf's public-dataset composer, which drops the prefix pool and the
synthetic ISL, so every request measures ISL ~142 instead of the length you
asked for.

## Reading the numbers

- **accept %** is per *drafted* token. Not comparable across different
  `num_speculative_tokens` values — use **accept length** (ceiling N+1) for that.
- **TTFT P90** is dominated by cold-start prefill. The first `<concurrency>`
  requests each do a full-length prefill (12–40 s vs 0.7–2 s steady-state) —
  roughly 18% of the sample at these settings. Raising `--warmup` does not remove
  it, because aiperf regenerates its prefix pool per run. Treat P90 as
  cold-start-inclusive, and compare it only against other runs measured the same way.
- **out tok/s/GPU** is aggregate and includes prefill, which is not
  speculatively decoded.
- **Prefix cache hit rate is state-dependent.** Back-to-back runs with the same
  prompts and seed can find prefixes still resident server-side — we have measured
  the same configuration at 80% and at 97% depending on what ran before. Cache hit
  rate strongly affects throughput, so compare runs at similar hit rates or
  restart the server between them.
- **Acceptance is not always stable.** At concurrency 24 we measured four GSM8K
  repeats at 44.3 / 58.1 / 59.1 / 60.4 %. Run at least 2 repeats, report a
  median, and show the spread.

## Files

```
run_gsm8k_benchmark.sh    Main entry point. No package modification.
run_corpus_ab.sh          Stock vs GSM8K A/B. --restore-only for recovery.
gsm8k_corpus.txt          The prompt corpus. Ships with the pack.
scripts/
  make_gsm8k_corpus.py    GSM8K -> corpus text
  make_prompt_file.py     corpus -> full-length prompt JSONL
  report.py               Summarise one results directory
  economics.py            Date / TPS / $ per GPU-hr / nodes-to-target from a report
  compare_ab.py           Compare A/B arms (medians + spread)
docs/
  METHOD.md               What is measured and why, and what we verified
  OFFLINE.md              Air-gapped and no-HF-access setup
  TUNING.md               Adapting ISL/OSL/cache to your own workload
```

## Support

Configuration lives in `run_meta.json` in every results directory — include it,
along with `report.txt`, in any question about a result.
