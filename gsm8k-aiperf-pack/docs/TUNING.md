# Adapting to your workload

The defaults (ISL 68000, OSL 350, 93% cached prefix) describe one specific
long-context workload. If yours differs, change them — the acceptance benefit
comes from prompt *coherence*, not from these particular numbers.

## Input and output length

```bash
./run_gsm8k_benchmark.sh --model /models/M --isl 8000 --osl 512
```

`--isl` is total input tokens per request; `--osl` is output tokens, pinned with
`min_tokens`/`max_tokens` and `ignore_eos` so every request generates exactly
that many. Pinning OSL is what makes ITL comparable across runs — without it,
requests that stop early skew the distribution.

## Cached prefix

```bash
./run_gsm8k_benchmark.sh --model /models/M --cache 50 --num-prefix-prompts 16
```

`--cache` is the percent of ISL that is a **reused prefix**, shared across
requests; the remainder is fresh text per request. `--num-prefix-prompts` is how
many distinct prefixes exist in the pool.

- **High cache % + few prefixes** — a chat or agent workload with a large shared
  system prompt or document context. Produces a high prefix cache hit rate.
- **`--cache 0`** — every request fully fresh. Worst case for prefill; use it to
  measure uncached prefill cost.
- **Many prefixes** — many distinct sessions. Raises KV cache pressure; if the
  pool exceeds what the server can hold, hit rate collapses. If your measured hit
  rate is far below `--cache`, that is usually why.

Cache hit rate strongly affects throughput, so **compare runs at similar hit
rates**. Back-to-back runs with identical prompts can find prefixes still
resident from the previous run — we have measured the same configuration at 80%
and 97% depending on what ran before. Restart the server between runs if you need
clean comparisons.

## Concurrency

```bash
./run_gsm8k_benchmark.sh --model /models/M --concurrency 8,16,24,32 --req-mult 5
```

Each point runs `concurrency x --req-mult` requests. Keep `--req-mult` at 5 or
higher: percentiles over fewer than ~40 requests are noisy, and the first
`<concurrency>` requests are cold-start prefills that skew TTFT P90 more the
smaller the sample.

Acceptance stability degrades at higher concurrency. At concurrency 16 our
repeats agreed within 0.65 points; at 24, four repeats spanned 44.3–60.4%. Repeat
any point you intend to quote.

## Using your own prompts instead of GSM8K

GSM8K is short-form arithmetic reasoning. If your traffic is code, long-form chat
or retrieval, measure with text that resembles it. The corpus is a plain text
file — one coherent passage per line, no blank lines:

```bash
python3 scripts/make_prompt_file.py prompts.jsonl \
    --corpus my_corpus.txt --tokenizer /models/M \
    --isl 68000 --cache 93 --num-prefix-prompts 8 --entries 120

./run_gsm8k_benchmark.sh --model /models/M --prompts prompts.jsonl
```

Size the corpus to at least `num_prefix_prompts x prefix_length + slack` tokens,
or windows will overlap heavily and prefixes will not be distinct. At ISL 68000
with 8 prefixes that is ~500K tokens minimum; the shipped corpus is ~1.17M.

To A/B your corpus against the stock one, pass it as `--corpus` to
`run_corpus_ab.sh`.

## Servers without speculative decoding

Everything works; acceptance columns show `n/a` and `report.py` prints a note.
Latency and throughput are still measured normally. Prompt coherence has little
effect on those without a draft model, so the A/B is mainly interesting when
speculative decoding is enabled.

## Non-vLLM servers

Latency and throughput come from aiperf and work against any OpenAI-compatible
endpoint. Acceptance and cache hit rate are read from Prometheus counters named
`vllm:spec_decode_*` and `vllm:prefix_cache_*` on `/metrics`; a server exposing
different names will report `n/a` for those columns. Adapt the metric names in
`scripts/report.py` (`counter_total` call sites) if yours differ.
