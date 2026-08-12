# Offline and air-gapped setup

`make_gsm8k_corpus.py` downloads `openai/gsm8k` from the HuggingFace hub. On a
host without hub access, build the corpus elsewhere and copy the result.

## Build once, copy the artifact

`gsm8k_corpus.txt` is a plain UTF-8 text file (~4.3 MB) with no runtime
dependencies. Build it on any machine with network access:

```bash
pip install datasets
python3 scripts/make_gsm8k_corpus.py gsm8k_corpus.txt 4272850
```

Expected output:

```
loaded 8792 GSM8K items
wrote gsm8k_corpus.txt: 8312 problems used, 4272924 chars, 33248 lines
```

Copy `gsm8k_corpus.txt` to the air-gapped host next to the scripts. Nothing else
in the pack needs network access.

**The build is deterministic** — fixed seed, fixed character budget. The same
command produces a byte-identical file, so an md5 can be used to confirm a
transferred corpus matches the one a result was produced with. Record it:

```bash
md5sum gsm8k_corpus.txt
```

## Pre-populating a HuggingFace cache instead

If you would rather build on the target host, populate the cache from a
connected machine and transfer it:

```bash
# on the connected machine
python3 -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
tar czf hf_cache.tgz -C ~ .cache/huggingface

# on the air-gapped host
tar xzf hf_cache.tgz -C ~
export HF_HUB_OFFLINE=1
python3 scripts/make_gsm8k_corpus.py gsm8k_corpus.txt 4272850
```

## Tokenizer

`make_prompt_file.py` loads a tokenizer with `AutoTokenizer.from_pretrained`.
Pass a **local path** so nothing is fetched:

```bash
./run_gsm8k_benchmark.sh --model /models/Kimi-K3 --tokenizer /models/Kimi-K3
```

`--tokenizer` defaults to `--model`, so if the model is already a local path
there is nothing to do. If the server is addressed by a served-model *name*
rather than a path, pass `--tokenizer` explicitly:

```bash
./run_gsm8k_benchmark.sh --model my-served-name --tokenizer /models/Kimi-K3
```

Set `export HF_HUB_OFFLINE=1` to make any accidental fetch fail loudly rather
than hang.

## No `datasets` package

`datasets` is needed **only** to build the corpus. On a host that already has
`gsm8k_corpus.txt`, only `transformers` (for the tokenizer) and `aiperf` are
required.
