#!/usr/bin/env python3
"""Build aiperf's prompt corpus out of GSM8K.

aiperf's plugin registry lists GSM8K among "the datasets the spec-decode
literature reports acceptance length against", but that path
(`--public-dataset spec_al_gsm8k`) goes through the public-dataset composer,
which neither honours `--synthetic-input-tokens-mean` nor applies the reused
prefix pool. A 68K-ISL / 93%-cached-prefix run can therefore only be produced
by the synthetic composer, whose corpus is fixed at `assets/shakespeare.txt`.

This script converts GSM8K into a drop-in corpus for that synthetic path, so
the prompt content is the dataset aiperf prescribes while the prompt shape
stays exactly as specified.

Question and solution text are taken verbatim. Only GSM8K's internal
annotations are removed: the `<<4*2=8>>` calculator spans and the `#### 42`
final-answer marker, which are dataset bookkeeping rather than prose a model
would emit. A minimal `Problem. / Solution.` scaffold is added so a window cut
at an arbitrary offset still lands inside a recognisable task.

`PromptGenerator._initialize_corpus` drops blank lines and joins the rest with
single spaces, so each line is written to read correctly space-joined.

Usage:
    python3 make_gsm8k_corpus.py gsm8k_corpus.txt 4272850
"""

from __future__ import annotations

import random
import re
import sys

CALC = re.compile(r"<<[^>]*>>")
FINAL = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def clean_answer(ans: str) -> tuple[str, str]:
    """Strip GSM8K bookkeeping, returning (reasoning, final answer)."""
    final = ""
    m = FINAL.search(ans)
    if m:
        final = m.group(1).strip()
    body = FINAL.sub("", ans)
    body = CALC.sub("", body)
    lines = [re.sub(r"\s+", " ", l).strip() for l in body.splitlines()]
    return " ".join(l for l in lines if l), final


def main() -> None:
    out_path = sys.argv[1]
    target_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 5_020_475

    from datasets import load_dataset

    items = []
    for split in ("test", "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for rec in ds:
            q = re.sub(r"\s+", " ", rec["question"]).strip()
            reasoning, final = clean_answer(rec["answer"])
            if q and reasoning:
                items.append((q, reasoning, final))

    # Fixed shuffle: a deterministic order that does not mirror the dataset's
    # own topic grouping, so successive windows are not correlated.
    random.Random(20260807).shuffle(items)
    print(f"loaded {len(items)} GSM8K items")

    lines: list[str] = []
    chars = 0
    used = 0
    for q, reasoning, final in items:
        if chars >= target_chars:
            break
        block = [f"Problem. {q}", "Solution."]
        block.append(reasoning if reasoning.endswith(".") else reasoning + ".")
        if final:
            block.append(f"The answer is {final}.")
        lines.extend(block)
        chars += sum(len(x) + 1 for x in block)
        used += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}: {used} problems used, {chars} chars, {len(lines)} lines")
    if used == len(items) and chars < target_chars:
        print(
            f"NOTE: GSM8K exhausted at {chars} chars, short of the {target_chars} target. "
            "Corpus is smaller than the stock one; prefix windows will overlap more."
        )


if __name__ == "__main__":
    main()
