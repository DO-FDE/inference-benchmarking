# Method — what is measured, and what was verified

## The problem with the stock corpus

aiperf's synthetic prompt generator samples from a fixed corpus file,
`assets/shakespeare.txt`, inside the installed package
(`aiperf/dataset/generator/prompt.py`, `DEFAULT_CORPUS_FILE`). Prompts are built
by cutting a window of that corpus at a random offset.

At ISL 68,000 the result is a 68,000-token block of Elizabethan prose containing
no task, no question and no structure a model can latch onto. This matters
specifically for **speculative decoding**: the draft model proposes tokens that
the target model then accepts or rejects. Continuing arbitrary Shakespeare is
close to an unconstrained prediction problem, so the draft model's proposals are
rejected often, and measured acceptance is far below what the same
target/draft pair achieves on coherent traffic.

Acceptance is not a cosmetic metric. It determines how many tokens are committed
per forward pass, so it drives the ITL and throughput that speculative decoding
exists to improve. Benchmarking it on incoherent text understates the feature.

## What this pack changes

Prompts become GSM8K grade-school word problems, formatted as:

```
Problem. Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether?
Solution. Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips
altogether in April and May.
The answer is 72.
```

GSM8K's internal bookkeeping is stripped: calculator spans (`<<48/2=24>>`) and
the `#### 72` answer marker, neither of which is prose a model would emit. The
`Problem. / Solution. / The answer is N.` scaffold means a window cut at an
arbitrary offset still lands inside a recognisable task.

**Corpus size is held constant.** The output is trimmed to a character budget so
its token count matches the stock corpus within ~0.01% (1,165,305 vs 1,165,342
tokens under the Kimi-K3 tokenizer). Corpus length is therefore eliminated as an
explanation for any measured difference.

## Prompt shape is unchanged

Only the *text* changes. The workload shape — total ISL, output length, the
cached-prefix pool — is identical between arms:

```
--isl 68000 --cache 93  =>  prefix 63,240 tokens x 8 distinct  +  fresh 4,760 tokens
```

`--num-prefix-prompts 8` with `--prompt-prefix-length 63240` means "build 8
distinct prefixes and reuse them across requests", which is what produces the
~80% prefix cache hit rate. Both arms measure the same shape at the same seed.

## Two delivery mechanisms

### `--input-file` (default, `run_gsm8k_benchmark.sh`)

`make_prompt_file.py` builds prompts at their final length and writes them as a
`single_turn` JSONL. aiperf consumes it via `--input-file
--custom-dataset-type single_turn`, a documented CLI path. The cached-prefix
behaviour is reproduced in the generator rather than delegated to aiperf: N
distinct prefixes are drawn once and reused across entries. vLLM's prefix cache
keys on the token sequence, so it hits identically.

Nothing inside the installed aiperf package is modified. **This is the
recommended path for customer systems.**

### Corpus replacement (`run_corpus_ab.sh`)

The corpus asset inside site-packages is replaced for the duration of the run and
restored afterwards. Needed only for the A/B, because measuring the *stock* arm
means running aiperf's own synthetic generator against its own corpus.

Restore is on an EXIT/INT/TERM trap. `kill -9` bypasses it — recover with
`./run_corpus_ab.sh --restore-only`.

> aiperf caches the tokenised corpus keyed by **tokenizer, not by content**, so
> `~/.cache/aiperf` must be cleared when the corpus changes. Both scripts do this.
> Forgetting it silently replays the previous arm's prompts and produces a false
> null result.

### Why the two paths differ slightly

Measured at concurrency 16 on the same server:

| Path | Acceptance | Accept length |
|---|---:|---:|
| Corpus swap | 61.2 – 61.4 % | 2.84 |
| `--input-file` | 57.5 – 59.6 % | 2.72 – 2.79 |

Both are far above the stock corpus's ~42%, and the gap between them is small
relative to the effect being measured. The likely cause is prompt diversity:
aiperf regenerates its window offsets per run, while the `--input-file` path
reuses one pre-built file, so its fresh segments repeat across requests within a
run. We did not isolate this further — it does not affect the conclusion, but it
does mean **the two paths' absolute numbers should not be compared to each
other.** Pick one and stay on it.

## What was verified

Measured on 8× MI355X (gfx950), Kimi-K3 + Kimi-K3-DSpark,
`num_speculative_tokens=3`, ISL 68000 / OSL 350 / 93% cached prefix.

**Corpus construction**
- Corpus builder is deterministic: same input, same md5, across rebuilds.
- Token count within 0.003% of the stock corpus.
- Assembled prompts measure ISL 67,988–68,088 against a 68,000 target.
- Prompt tails land on completed solutions; heads may start mid-sentence, which
  is inherent to cutting a random window and behaves the same on the stock corpus.

**The acceptance effect** — interleaved arms (stock, GSM8K, stock, GSM8K),
identical server and flags, tokenizer cache cleared between arms, corpus md5
recorded per run:

| Concurrency | Stock | GSM8K | Delta | n per arm |
|---|---:|---:|---:|---:|
| 16 | 41.90 % | 61.32 % | +19.4 pts | 2 |
| 24 | 41.80 % (median) | 58.59 % (median) | +16.8 pts | 4 |

At concurrency 16 the arms are fully separated and repeats agree within 0.65
points. At concurrency 24 GSM8K wins 15 of 16 pairwise run comparisons; three of
four repeats land 58–60% with one outlier at 44.3%.

**Reproducibility of the published 08-07 figures** — our GSM8K arm against the
ATOM recipe's own table:

| Metric | conc 16 | conc 24 |
|---|---:|---:|
| Cache hit rate | +0.1 % | −0.1 % |
| TTFT P50 | −15.0 % | −12.6 % |
| TTFT P90 | −1.2 % | −0.6 % |
| ITL P50 | +12.3 % | +5.4 % |
| Throughput (in & out /s/GPU) | −5.1 % | −9.6 % |

Cache hit rate matching within 0.1% at both points indicates the workload shape
is genuinely equivalent. TTFT reproduces well. Throughput runs 5–10% low and ITL
5–12% high in the same direction at both points, which is consistent with a
systematic environment difference rather than run-to-run noise. Later repeats at
a warmer prefix cache (87–90% hit rate) reached 9,677–11,287 input tok/s/GPU,
bracketing the published figure — so cache warmth is the leading explanation, but
we did not isolate it with a controlled experiment.

## Known limitations

- **GSM8K is short-form arithmetic reasoning.** The defensible claim is "coherent
  prompts beat literary filler for measuring speculative decoding", not "you will
  see 61% on your workload". If your traffic is code or long-form chat, measure
  it — see `TUNING.md`.
- **Acceptance can be unstable at higher concurrency.** Always repeat.
- **Prefix cache hit rate is state-dependent** and strongly affects throughput.
  Compare runs at similar hit rates, or restart the server between them.
- **TTFT P90 is cold-start-dominated** and is not a steady-state latency number.
- **Absolute numbers do not transfer between the two delivery mechanisms.**
