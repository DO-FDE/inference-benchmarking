#!/usr/bin/env python3
"""Summarise a run_gsm8k_benchmark.sh results directory.

Latency and throughput come from aiperf's own per-point summary JSON.
Speculative-decoding acceptance and prefix-cache hit rate come from aiperf's
per-point `/metrics` capture (`*_server_metrics.json`), whose counter totals are
already bracketed to that point's measurement window -- no external sampler and
no time-alignment needed.

Acceptance is reported two ways:

  overall accept %  -- accepted draft tokens / drafted tokens. This is per
                       drafted token, so it is NOT comparable across different
                       `num_speculative_tokens` settings.
  accept length     -- mean tokens committed per step, ceiling N+1. Compare this
                       across configurations instead.

Usage:
    python3 report.py results_dir/
    GPUS=8 python3 report.py results_dir/
"""

from __future__ import annotations

import glob
import json
import os
import sys

GPUS = int(os.environ.get("GPUS", "8"))
NS_PER_S = 1e9


def counter_total(metrics: dict, name: str, position: str | None = None) -> float | None:
    m = metrics.get(name)
    if not m:
        return None
    tot, found = 0.0, False
    for s in m.get("series", []):
        if position is not None and s.get("labels", {}).get("position") != position:
            continue
        v = s.get("stats", {}).get("total")
        if v is not None:
            tot += v
            found = True
    return tot if found else None


def collect(root: str) -> list[dict]:
    rows = []
    for cdir in sorted(glob.glob(os.path.join(root, "concurrency_*"))):
        if not os.path.isdir(cdir):
            continue
        summaries = sorted(
            p for p in glob.glob(os.path.join(cdir, "**", "*.json"), recursive=True)
            if os.path.basename(p) not in ("run_config.json", "inputs.json", "run_meta.json")
            and not p.endswith("_server_metrics.json")
        )
        smetrics = glob.glob(os.path.join(cdir, "**", "*_server_metrics.json"), recursive=True)
        jsonls = [p for p in glob.glob(os.path.join(cdir, "**", "*.jsonl"), recursive=True)
                  if "prompts" not in os.path.basename(p)]
        if not summaries or not jsonls:
            continue

        summary = None
        for p in summaries:
            d = json.load(open(p))
            if "time_to_first_token" in d:
                summary = d
                break
        if summary is None:
            continue

        recs = [json.loads(l) for l in open(jsonls[0]) if l.strip()]
        recs = [r for r in recs if r["metrics"].get("time_to_first_token")]
        if not recs:
            continue
        t0 = min(r["metadata"]["request_start_ns"] for r in recs)
        t1 = max(r["metadata"]["request_end_ns"] for r in recs)
        window = (t1 - t0) / NS_PER_S
        in_tok = sum(r["metrics"]["input_sequence_length"]["value"] for r in recs)
        out_tok = sum(r["metrics"]["output_sequence_length"]["value"] for r in recs)

        row = {
            "concurrency": int(os.path.basename(cdir).split("_")[-1]),
            "requests": len(recs),
            "isl": summary["input_sequence_length"]["avg"],
            "ttft_p50": summary["time_to_first_token"]["p50"],
            "ttft_p90": summary["time_to_first_token"]["p90"],
            "itl_p50": summary["inter_token_latency"]["p50"],
            "in_tps_gpu": in_tok / window / GPUS,
            "out_tps_gpu": out_tok / window / GPUS,
            "cache_hit": None, "accept": None, "accept_len": None, "pos": [],
        }

        if smetrics:
            M = json.load(open(smetrics[0])).get("metrics", {})
            q = counter_total(M, "vllm:prefix_cache_queries")
            h = counter_total(M, "vllm:prefix_cache_hits")
            dt = counter_total(M, "vllm:spec_decode_num_draft_tokens")
            at = counter_total(M, "vllm:spec_decode_num_accepted_tokens")
            nd = counter_total(M, "vllm:spec_decode_num_drafts")
            if q:
                row["cache_hit"] = 100.0 * h / q
            if dt:
                row["accept"] = 100.0 * at / dt
            if nd:
                row["accept_len"] = at / nd + 1.0
                i = 0
                while True:
                    p = counter_total(M, "vllm:spec_decode_num_accepted_tokens_per_pos",
                                      position=str(i))
                    if p is None:
                        break
                    row["pos"].append(100.0 * p / nd)
                    i += 1
        rows.append(row)
    return rows


def g(v, spec, na="n/a"):
    return format(v, spec) if v is not None else na


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = collect(root)
    if not rows:
        print(f"no aiperf results found under {root}", file=sys.stderr)
        return 1

    meta = {}
    mp = os.path.join(root, "run_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    if meta:
        print(f"model {meta.get('model')} | ISL {meta.get('isl')} OSL {meta.get('osl')} "
              f"| cached prefix {meta.get('cache')}% | GPUs {GPUS} | prompts: GSM8K")
        print()

    hdr = (f"{'conc':>5} {'reqs':>5} {'ISL':>7} {'cache%':>7} {'TTFT p50':>9} {'TTFT p90':>9} "
           f"{'ITL p50':>8} {'in/s/gpu':>9} {'out/s/gpu':>10} {'accept%':>8} {'acc.len':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["concurrency"]):
        print(f"{r['concurrency']:>5} {r['requests']:>5} {r['isl']:>7.0f} "
              f"{g(r['cache_hit'],'>7.2f')} {r['ttft_p50']:>9.0f} {r['ttft_p90']:>9.0f} "
              f"{r['itl_p50']:>8.2f} {r['in_tps_gpu']:>9.1f} {r['out_tps_gpu']:>10.2f} "
              f"{g(r['accept'],'>8.2f')} {g(r['accept_len'],'>8.2f')}")
        if r["pos"]:
            print(f"{'':>5} per-position acceptance %: "
                  + ", ".join(f"{p:.1f}" for p in r["pos"]))

    if not any(r["accept"] is not None for r in rows):
        print()
        print("NOTE: no speculative-decoding counters found. Either the server is not")
        print("      running a draft model, or it does not expose vllm:spec_decode_* on")
        print("      /metrics. Acceptance cannot be reported.")

    print()
    print("Reading these numbers:")
    print("  - accept %   is per drafted token; not comparable across different")
    print("               num_speculative_tokens. Use accept length for that.")
    print("  - TTFT P90   is dominated by cold-start prefill: the first <concurrency>")
    print("               requests each do a full-length prefill. Raising warmup does not")
    print("               remove this. Treat P90 as cold-start-inclusive.")
    print("  - out/s/gpu  is aggregate and includes prefill time, which is not")
    print("               speculatively decoded.")

    out = os.path.join(root, "report.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
