#!/usr/bin/env python3
"""Post-benchmark economics for a run_gsm8k_benchmark.sh results directory.

Reads report.json (written by report.py) and prints, per concurrency point:

  date        when the numbers were computed
  TPS         output tokens/s, per GPU and per node (node = GPUS gpus)
  $/gpu/hr    token revenue per GPU-hour at DigitalOcean serverless prices
              from the gen-ai model catalog. When the run captured a prefix
              cache hit rate, that fraction of input tokens is billed at the
              cache-read rate instead of the full input rate.
  nodes       nodes needed to serve --tps-target output tokens/s

Catalog pricing note: the API field is named *_price_per_million but the
values are dollars per single token (e.g. 0.00001425 = $14.25 per 1M).

Usage:
    GPUS=8 python3 economics.py results_dir/ \
        [--tps-target N] [--catalog-model ID] [--catalog-url URL]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

DEFAULT_CATALOG_URL = "https://api.digitalocean.com/v2/gen-ai/models/catalog"
GPUS = int(os.environ.get("GPUS", "8"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_catalog(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp).get("data", [])


def match_model(models: list[dict], wanted: str) -> dict | None:
    """Match a served model name/path against catalog model_id,
    hugging_face_id or display name. Exact (normalised) match wins;
    otherwise fall back to substring containment either way."""
    w = norm(os.path.basename(wanted.rstrip("/")))
    if not w:
        return None
    exact, partial = [], []
    for m in models:
        if not (m.get("pricing") or {}).get("output_price_per_million"):
            continue
        hf = m.get("hugging_face_id") or ""
        cands = {norm(m.get("model_id") or ""), norm(m.get("name") or ""),
                 norm(hf), norm(hf.split("/")[-1])}
        cands.discard("")
        if w in cands:
            exact.append(m)
        elif any(w in c or c in w for c in cands):
            partial.append(m)
    if exact:
        return exact[0]
    if partial:
        return partial[0]
    return None


def revenue_per_gpu_hr(row: dict, pricing: dict) -> dict:
    """Prices are $/token (see module docstring)."""
    in_price = pricing.get("input_price_per_million") or 0.0
    out_price = pricing.get("output_price_per_million") or 0.0
    cache_price = pricing.get("cache_read_input_price_per_million")

    hit = 0.0
    if cache_price is not None and row.get("cache_hit") is not None:
        hit = row["cache_hit"] / 100.0
    eff_in_price = (1.0 - hit) * in_price + hit * (cache_price or 0.0)

    in_hr = row["in_tps_gpu"] * eff_in_price * 3600.0
    out_hr = row["out_tps_gpu"] * out_price * 3600.0
    return {"input": in_hr, "output": out_hr, "total": in_hr + out_hr,
            "cache_hit_frac_billed": hit}


def resolve(catalog_url: str, wanted: str) -> int:
    """Pre-flight name check. Exit codes: 0 matched (prints 'model_id\\tname'
    on stdout), 2 catalog unreachable, 3 no match (suggestions on stderr)."""
    try:
        catalog = fetch_catalog(catalog_url)
    except Exception as e:
        print(f"catalog fetch failed: {e}", file=sys.stderr)
        return 2
    m = match_model(catalog, wanted)
    if m:
        print(f"{m['model_id']}\t{m['name']}")
        return 0

    import difflib
    priced = [c for c in catalog
              if (c.get("pricing") or {}).get("output_price_per_million")]
    by_norm = {}
    for c in priced:
        hf = c.get("hugging_face_id") or ""
        for cand in (c.get("model_id") or "", c.get("name") or "",
                     hf.split("/")[-1]):
            if cand:
                by_norm.setdefault(norm(cand), c["model_id"])
    close = difflib.get_close_matches(
        norm(os.path.basename(wanted.rstrip("/"))), list(by_norm), n=5, cutoff=0.5)
    sugg = sorted({by_norm[k] for k in close})
    print(f"no catalog model matched '{wanted}'", file=sys.stderr)
    if sugg:
        print("close matches (model_id): " + ", ".join(sugg), file=sys.stderr)
    else:
        print("no close matches; list ids with:\n"
              f"  curl -s {catalog_url} | "
              "python3 -c \"import json,sys; "
              "[print(m['model_id']) for m in json.load(sys.stdin)['data'] "
              "if m.get('pricing',{}).get('output_price_per_million')]\"",
              file=sys.stderr)
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("--tps-target", type=float, default=None,
                    help="target aggregate output tokens/s to size nodes for")
    ap.add_argument("--catalog-model", default=None,
                    help="catalog model_id to price against "
                         "(default: inferred from run_meta.json model)")
    ap.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    ap.add_argument("--resolve", metavar="NAME", default=None,
                    help="pre-flight: check NAME against the catalog and exit "
                         "(0 matched, 2 catalog unreachable, 3 no match)")
    args = ap.parse_args()

    if args.resolve is not None:
        return resolve(args.catalog_url, args.resolve)
    if not args.results_dir:
        ap.error("results_dir is required unless --resolve is given")

    report_path = os.path.join(args.results_dir, "report.json")
    if not os.path.exists(report_path):
        print(f"ERROR: {report_path} not found; run report.py first", file=sys.stderr)
        return 1
    rows = json.load(open(report_path))
    if not rows:
        print("ERROR: report.json is empty", file=sys.stderr)
        return 1

    meta = {}
    meta_path = os.path.join(args.results_dir, "run_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

    wanted = args.catalog_model or meta.get("model") or ""
    priced, pricing, price_err = None, None, None
    try:
        catalog = fetch_catalog(args.catalog_url)
        priced = match_model(catalog, wanted) if wanted else None
        if priced:
            pricing = priced["pricing"]
        else:
            price_err = f"no catalog model matched '{wanted}'"
    except Exception as e:  # network/JSON failure: degrade, don't abort the run
        price_err = f"catalog fetch failed: {e}"

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"date {date} | model {meta.get('model', wanted)} | GPUs/node {GPUS}")
    if pricing:
        cache_m = pricing.get("cache_read_input_price_per_million")
        print(f"pricing: {priced['name']} (model_id {priced['model_id']}) -- "
              f"in ${pricing['input_price_per_million'] * 1e6:.4g}/M, "
              f"out ${pricing['output_price_per_million'] * 1e6:.4g}/M"
              + (f", cache-read ${cache_m * 1e6:.4g}/M" if cache_m is not None else ""))
    else:
        print(f"pricing: unavailable ({price_err}); $/gpu/hr omitted")
        print("         (use --catalog-model to pick a catalog model_id explicitly)")
    if args.tps_target:
        print(f"tps target: {args.tps_target:,.0f} output tok/s")
    print()

    hdr = (f"{'date':>10} {'conc':>5} {'out tok/s/gpu':>14} {'out tok/s/node':>15} "
           f"{'$/gpu/hr':>9} {'nodes@target':>13}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for r in sorted(rows, key=lambda r: r["concurrency"]):
        node_tps = r["out_tps_gpu"] * GPUS
        rev = revenue_per_gpu_hr(r, pricing) if pricing else None
        nodes = (math.ceil(args.tps_target / node_tps)
                 if args.tps_target and node_tps > 0 else None)
        rev_s = f"{rev['total']:>9.2f}" if rev else f"{'n/a':>9}"
        nodes_s = f"{nodes:>13d}" if nodes is not None else f"{'n/a':>13}"
        print(f"{date:>10} {r['concurrency']:>5} {r['out_tps_gpu']:>14.2f} "
              f"{node_tps:>15.2f} {rev_s} {nodes_s}")
        results.append({
            "date": date,
            "concurrency": r["concurrency"],
            "in_tps_gpu": r["in_tps_gpu"],
            "out_tps_gpu": r["out_tps_gpu"],
            "out_tps_node": node_tps,
            "gpus_per_node": GPUS,
            "usd_per_gpu_hr": rev["total"] if rev else None,
            "usd_per_gpu_hr_input": rev["input"] if rev else None,
            "usd_per_gpu_hr_output": rev["output"] if rev else None,
            "cache_hit_frac_billed": rev["cache_hit_frac_billed"] if rev else None,
            "tps_target": args.tps_target,
            "nodes_needed": nodes,
        })

    if pricing or args.tps_target:
        print()
        print("Notes:")
    if pricing:
        print("  - $/gpu/hr = (input tok/s x input price + output tok/s x output price)")
        print("    x 3600, per GPU. Cached-prefix input tokens are billed at the")
        print("    cache-read rate using the measured prefix cache hit rate.")
    if args.tps_target:
        print("  - nodes@target = ceil(tps target / output tok/s per node); sized on")
        print("    output throughput at that concurrency.")

    out = {
        "date": date,
        "model": meta.get("model", wanted),
        "catalog_model": priced["model_id"] if priced else None,
        "catalog_url": args.catalog_url,
        "pricing_per_token": pricing,
        "pricing_error": price_err,
        "rows": results,
    }
    out_path = os.path.join(args.results_dir, "economics.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
