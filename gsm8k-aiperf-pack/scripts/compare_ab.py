#!/usr/bin/env python3
"""Compare the stock-corpus and GSM8K arms of run_corpus_ab.sh.

Reports medians rather than means. Speculative-decode acceptance is occasionally
unstable run to run -- a single outlier moves a mean of 2-4 samples a long way
but barely moves a median. Full per-run values and the pairwise win count are
printed so the spread is visible rather than hidden behind a summary statistic.

Usage:
    python3 compare_ab.py ab_results_dir/
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import collect  # noqa: E402  (same-directory helper)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(root, "*_rep*"))):
        if not os.path.isdir(arm_dir):
            continue
        meta = {}
        mp = os.path.join(arm_dir, "run_meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
        base = os.path.basename(arm_dir)
        arm, _, rep = base.rpartition("_rep")
        for r in collect(arm_dir):
            r["arm"] = meta.get("arm", arm)
            r["rep"] = meta.get("rep", int(rep) if rep.isdigit() else 0)
            rows.append(r)

    if not rows:
        print(f"no results found under {root}", file=sys.stderr)
        return 1

    def g(v, spec, na="n/a"):
        return format(v, spec) if v is not None else na

    print("=" * 100)
    print("PER-RUN")
    print("=" * 100)
    hdr = (f"{'arm':>8} {'rep':>4} {'conc':>5} {'cache%':>7} {'TTFT p50':>9} {'TTFT p90':>9} "
           f"{'ITL p50':>8} {'out/s/gpu':>10} {'accept%':>8} {'acc.len':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["concurrency"], r["arm"], r["rep"])):
        print(f"{r['arm']:>8} {r['rep']:>4} {r['concurrency']:>5} {g(r['cache_hit'],'>7.2f')} "
              f"{r['ttft_p50']:>9.0f} {r['ttft_p90']:>9.0f} {r['itl_p50']:>8.2f} "
              f"{r['out_tps_gpu']:>10.2f} {g(r['accept'],'>8.2f')} {g(r['accept_len'],'>8.2f')}")

    print()
    print("=" * 100)
    print("COMPARISON  (medians)")
    print("=" * 100)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return st.median(vals) if vals else None

    for c in sorted({r["concurrency"] for r in rows}):
        S = [r for r in rows if r["arm"] == "stock" and r["concurrency"] == c]
        G = [r for r in rows if r["arm"] == "gsm8k" and r["concurrency"] == c]
        if not S or not G:
            continue
        print(f"\n--- concurrency {c}  (n={len(S)} stock, {len(G)} gsm8k) ---")
        h2 = f"{'metric':<22}{'stock':>12}{'gsm8k':>12}{'change':>14}"
        print(h2)
        print("-" * len(h2))
        for label, key, pct in [
            ("accept %", "accept", False),
            ("accept length", "accept_len", True),
            ("ITL p50 (ms)", "itl_p50", True),
            ("TTFT p50 (ms)", "ttft_p50", True),
            ("TTFT p90 (ms)", "ttft_p90", True),
            ("out tok/s/gpu", "out_tps_gpu", True),
            ("in tok/s/gpu", "in_tps_gpu", True),
            ("prefix cache %", "cache_hit", False),
        ]:
            s, gm = med([r[key] for r in S]), med([r[key] for r in G])
            if s is None or gm is None:
                print(f"{label:<22}{'n/a':>12}{'n/a':>12}{'':>14}")
                continue
            chg = (f"{100*(gm-s)/s:+.1f}%" if pct and s else f"{gm-s:+.2f} pts")
            print(f"{label:<22}{s:>12.2f}{gm:>12.2f}{chg:>14}")

        sa = [r["accept"] for r in S if r["accept"] is not None]
        ga = [r["accept"] for r in G if r["accept"] is not None]
        if len(sa) > 1 or len(ga) > 1:
            print()
            print(f"  stock accept runs : {', '.join(f'{v:.2f}' for v in sorted(sa))}")
            print(f"  gsm8k accept runs : {', '.join(f'{v:.2f}' for v in sorted(ga))}")
        if sa and ga:
            wins = sum(1 for a, b in itertools.product(ga, sa) if a > b)
            n = len(ga) * len(sa)
            print(f"  gsm8k > stock in {wins}/{n} pairwise run comparisons")
            if min(ga) > max(sa):
                print("  arms fully separated: every gsm8k run beat every stock run")
            else:
                print("  arms overlap: at least one run bucks the trend -- add reps "
                      "before quoting a single number")

    out = os.path.join(root, "comparison.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
