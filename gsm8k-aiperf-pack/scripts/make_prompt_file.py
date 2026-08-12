#!/usr/bin/env python3
"""Pre-build full-length prompts as an aiperf `single_turn` JSONL.

Alternative to replacing aiperf's corpus asset. Instead of changing what aiperf
samples from, the prompts are built here at their final length and handed to
aiperf through `--input-file` with `--custom-dataset-type single_turn`, a
documented CLI path. Nothing inside the installed aiperf package is touched.

The cached-prefix behaviour is reproduced here rather than delegated to aiperf.
`--num-prefix-prompts N` with `--prompt-prefix-length L` means "build N distinct
prefixes of L tokens and reuse them across requests", which is what this script
does. vLLM's prefix cache keys on the token sequence, so it hits the same way.

Usage:
    python3 make_prompt_file.py prompts.jsonl \
        --isl 68000 --cache 93 --num-prefix-prompts 8 --entries 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--corpus", type=Path, default=Path("gsm8k_corpus.txt"))
    ap.add_argument("--tokenizer", default="/models/Kimi-K3")
    ap.add_argument("--isl", type=int, default=68000)
    ap.add_argument("--cache", type=int, default=93, help="percent of ISL that is a reused prefix")
    ap.add_argument("--num-prefix-prompts", type=int, default=8)
    ap.add_argument("--entries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        print("Run make_gsm8k_corpus.py first.", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lines = [l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = tok.encode(" ".join(lines))
    n_corpus = len(corpus)

    prefix_len = args.isl * args.cache // 100
    fresh_len = args.isl - prefix_len
    print(
        f"corpus={n_corpus:,} tokens | prefix={prefix_len:,} x{args.num_prefix_prompts} "
        f"| fresh={fresh_len:,} x{args.entries}"
    )

    rng = random.Random(args.seed)

    def draw(n: int) -> list[int]:
        start = rng.randrange(n_corpus)
        end = start + n
        out = corpus[start:end]
        if end > n_corpus:
            out = out + corpus[: end - n_corpus]
        return out

    prefixes = [draw(prefix_len) for _ in range(args.num_prefix_prompts)]

    with args.out.open("w", encoding="utf-8") as fh:
        for i in range(args.entries):
            tokens = prefixes[i % args.num_prefix_prompts] + draw(fresh_len)
            fh.write(json.dumps({"text": tok.decode(tokens)}, ensure_ascii=False) + "\n")

    # decode/encode is not guaranteed length-preserving, so report what the
    # server will actually receive rather than the requested length.
    first = json.loads(args.out.read_text(encoding="utf-8").splitlines()[0])
    actual = len(tok.encode(first["text"]))
    print(f"wrote {args.out}: {args.entries} entries | first-entry ISL after round-trip = {actual:,}")
    if abs(actual - args.isl) > args.isl * 0.02:
        print(f"WARNING: ISL drifted more than 2% from the {args.isl:,} target", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
