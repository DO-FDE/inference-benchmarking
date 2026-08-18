#!/usr/bin/env python3
"""
parse_sweep.py

Convert an aiperf multi-run "sweep_aggregate" export (JSON, with CSV as a
fallback) into the standard per-GPU summary table used in FDE benchmark reports,
now with per-shape / per-tier SLA pass/fail and a "max concurrency under the ITL
gate" determination.

Output columns (one row per concurrency point):
    Concurrency, Requests, Duration(s), Req/s,
    Input tok/s/GPU, Output tok/s/GPU, Total tok/s/GPU,
    TTFT P50(ms), TTFT P90(ms), ITL P50(ms), ITL P90(ms),
    TTFT P50 OK, TTFT P90 OK, ITL P50 OK, SLA Pass,
    ISL, OSL, Valid, Notes

KEY CONVENTIONS (learned the hard way -- do not change without reason):
  * The three throughput columns are PER-GPU = aggregate / NUM_GPUS.
    aiperf reports aggregate (whole-server) throughput; we divide by the
    GPU count to get per-GPU. Everything else (req/s, duration, latencies,
    ISL/OSL) is NOT divided.
  * Input tok/s/GPU = Total tok/s/GPU - Output tok/s/GPU.
  * Authoritative OSL is `output_sequence_length` (full count), NOT
    `output_token_count` / `reasoning_token_count` (these are PARTIAL counts
    that split reasoning vs content and must never be used for OSL). This
    matters for Claw runs (thinking enabled): reasoning tokens are part of OSL.
  * A concurrency point is flagged DEGENERATE and excluded from the curve if:
      - OSL collapsed (mean << target, e.g. < ~900 when target is 1000), or
      - recorded request count < expected requests (server dropped requests),
      - osl_mismatch_count is large.
    Degenerate points produce fake (inflated) throughput and must not be
    reported as real.
  * SLA pass/fail is only computed when a target is supplied. A point passes
    the ITL gate iff it is Valid AND ITL P50 < itl-p50-target. The reported
    "max concurrency under ITL" is the highest-concurrency point that passes.
    TTFT targets are checked the same way but are informational per point --
    TTFT rises with concurrency, so read it AT the max-concurrency point.

Usage:
    python3 parse_sweep.py --sweep-dir /path/to/sweep_aggregate \
        --gpus 8 --osl-target 1000 \
        --ttft-p50-target 2500 --ttft-p90-target 5000 --itl-p50-target 25 \
        --out-csv summary.csv [--profile-name s1_8k_claw]

    # or point directly at a json file:
    python3 parse_sweep.py --json /path/to/profile_s1_8k_claw.json --gpus 8 --out-csv summary.csv
"""

import argparse
import csv
import glob
import json
import os
import sys


# ---- identity / validation tolerances ----
RPS_TOL = 1e-2          # |reqs/dur - rps|
TOT_TOL = 5.0           # |(ISL+OSL)*rps - total_tok/s|
OUT_TOL = 2.0           # |OSL*rps - output_tok/s|


def _metric(m, key, field="avg"):
    """Safely pull metric[key][field] from an aiperf metrics dict."""
    v = m.get(key)
    if v is None:
        return None
    return v.get(field)


def _lt(value, target):
    """SLA helper: True if no target, or value is present and strictly under it."""
    if target is None:
        return True
    if value is None:
        return False
    return value < target


def _load_combos_from_json(path):
    """
    Return a list of per-combination dicts: {parameters:{...}, metrics:{...}}.

    Handles two aiperf schemas:
      * multi-run sweep export with top-level 'per_combination_metrics'
      * single-run export (the whole doc is one combination); we synthesize
        parameters from run_info.variation_values when present.
    """
    with open(path) as f:
        d = json.load(f)

    if "per_combination_metrics" in d:
        return d["per_combination_metrics"]

    # single-run schema: the doc itself is the metrics container
    params = {}
    ri = d.get("run_info", {})
    vv = ri.get("variation_values", {})
    if vv:
        params = {
            "concurrency": vv.get("phases.profiling.concurrency"),
            "requests": vv.get("phases.profiling.requests"),
        }
    # metrics live at top level in single-run schema
    return [{"parameters": params, "metrics": d}]


def _discover_jsons(sweep_dir):
    """
    Find the per-combination JSON files under a sweep_aggregate dir.
    Prefer a single aggregated file with per_combination_metrics; otherwise
    collect the per-concurrency profile_*.json files.
    """
    # 1) any json directly in sweep_dir that has per_combination_metrics
    cands = sorted(glob.glob(os.path.join(sweep_dir, "*.json")))
    for c in cands:
        try:
            with open(c) as f:
                d = json.load(f)
            if "per_combination_metrics" in d:
                return [c]
        except Exception:
            continue
    # 2) otherwise, recurse for per-concurrency profile_*.json
    rec = sorted(glob.glob(os.path.join(sweep_dir, "**", "profile_*.json"),
                           recursive=True))
    if rec:
        return rec
    # 3) fallback: every json we found
    if cands:
        return cands
    raise FileNotFoundError(f"No JSON files found under {sweep_dir}")


def build_rows(combos, gpus, osl_target,
               ttft_p50_target=None, ttft_p90_target=None, itl_p50_target=None):
    """
    Turn raw combos into validated summary rows.
    Returns (rows, notes) where rows is a list of dicts and notes is a list of
    human-readable validation/degeneracy messages.
    """
    rows = []
    notes = []
    for c in combos:
        p = c.get("parameters", {}) or {}
        m = c.get("metrics", {}) or {}

        conc = p.get("concurrency")
        reqs = p.get("requests")

        dur = _metric(m, "benchmark_duration")
        rps = _metric(m, "request_throughput")
        tot = _metric(m, "total_token_throughput")
        out = _metric(m, "output_token_throughput")
        isl = _metric(m, "input_sequence_length")
        osl = _metric(m, "output_sequence_length")
        osl_n = (m.get("output_sequence_length") or {}).get("count")
        mismatch = _metric(m, "osl_mismatch_count") or 0

        ttft = m.get("time_to_first_token", {}) or {}
        itl = m.get("inter_token_latency", {}) or {}

        # fall back for requests/conc if not in parameters
        if reqs is None:
            reqs = (m.get("request_count") or {}).get("avg")
        if conc is None:
            conc = osl_n  # last resort; usually present in parameters

        if None in (dur, rps, tot, out, isl, osl):
            notes.append(f"conc={conc}: missing core metrics, skipped")
            continue

        # ---- degeneracy checks ----
        degenerate = False
        reasons = []
        if osl < 0.9 * osl_target:
            degenerate = True
            reasons.append(f"OSL collapsed ({osl:.1f} << target {osl_target})")
        if reqs and osl_n and osl_n < 0.98 * reqs:
            degenerate = True
            reasons.append(f"only {osl_n}/{int(reqs)} requests recorded")
        if reqs and mismatch and mismatch > 0.1 * reqs:
            degenerate = True
            reasons.append(f"osl_mismatch_count={mismatch}")

        # ---- identity validation ----
        identity_ok = True
        if reqs and dur:
            if abs(reqs / dur - rps) > RPS_TOL:
                identity_ok = False
        if abs((isl + osl) * rps - tot) > TOT_TOL:
            identity_ok = False
        if abs(osl * rps - out) > OUT_TOL:
            identity_ok = False

        tot_g = tot / gpus
        out_g = out / gpus
        in_g = tot_g - out_g

        valid = bool(identity_ok and not degenerate)

        # ---- SLA pass/fail ----
        ttft_p50 = ttft.get("p50")
        ttft_p90 = ttft.get("p90")
        itl_p50 = itl.get("p50")
        ttft50_ok = _lt(ttft_p50, ttft_p50_target)
        ttft90_ok = _lt(ttft_p90, ttft_p90_target)
        itl50_ok = _lt(itl_p50, itl_p50_target)
        sla_pass = bool(valid and ttft50_ok and ttft90_ok and itl50_ok)

        row = {
            "Concurrency": int(conc) if conc is not None else None,
            "Requests": int(reqs) if reqs is not None else None,
            "Duration(s)": round(dur, 2),
            "Req/s": round(rps, 3),
            "Input tok/s/GPU": round(in_g, 1),
            "Output tok/s/GPU": round(out_g, 1),
            "Total tok/s/GPU": round(tot_g, 1),
            "TTFT P50(ms)": round(ttft_p50, 1) if ttft_p50 is not None else float("nan"),
            "TTFT P90(ms)": round(ttft_p90, 1) if ttft_p90 is not None else float("nan"),
            "ITL P50(ms)": round(itl_p50, 2) if itl_p50 is not None else float("nan"),
            "ITL P90(ms)": round(itl.get("p90"), 2) if itl.get("p90") is not None else float("nan"),
            "TTFT P50 OK": "" if ttft_p50_target is None else ("Y" if ttft50_ok else "N"),
            "TTFT P90 OK": "" if ttft_p90_target is None else ("Y" if ttft90_ok else "N"),
            "ITL P50 OK": "" if itl_p50_target is None else ("Y" if itl50_ok else "N"),
            "SLA Pass": "Y" if sla_pass else "N",
            "ISL": round(isl, 1),
            "OSL": round(osl, 1),
            "Valid": valid,
            "Notes": "; ".join(reasons) if reasons else ("identity check failed" if not identity_ok else ""),
        }
        rows.append(row)
        if degenerate:
            notes.append(f"conc={conc}: DEGENERATE -- {'; '.join(reasons)} (EXCLUDE from curve)")
        elif not identity_ok:
            notes.append(f"conc={conc}: identity check failed (review)")

    rows.sort(key=lambda r: (r["Concurrency"] is None, r["Concurrency"]))
    return rows, notes


# columns written to the final CSV (report cols, then SLA cols, then audit cols)
REPORT_COLS = [
    "Concurrency", "Requests", "Duration(s)", "Req/s",
    "Input tok/s/GPU", "Output tok/s/GPU", "Total tok/s/GPU",
    "TTFT P50(ms)", "TTFT P90(ms)", "ITL P50(ms)", "ITL P90(ms)",
]
SLA_COLS = ["TTFT P50 OK", "TTFT P90 OK", "ITL P50 OK", "SLA Pass"]
AUDIT_COLS = ["ISL", "OSL", "Valid", "Notes"]


def write_csv(rows, out_csv, profile_name=None, audit=True):
    cols = REPORT_COLS + SLA_COLS + (AUDIT_COLS if audit else [])
    if profile_name:
        cols = ["Profile"] + cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if profile_name:
                rr["Profile"] = profile_name
            w.writerow({k: rr.get(k, "") for k in cols})


def main():
    ap = argparse.ArgumentParser(description="Parse aiperf sweep_aggregate into per-GPU summary CSV with SLA pass/fail")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep-dir", help="path to sweep_aggregate dir (auto-discovers JSON)")
    src.add_argument("--json", help="path to a single aiperf JSON export")
    ap.add_argument("--gpus", type=int, required=True, help="GPU count for per-GPU division")
    ap.add_argument("--osl-target", type=int, default=1000, help="expected OSL (for degeneracy check)")
    ap.add_argument("--ttft-p50-target", type=float, default=None, help="TTFT p50 SLA target in ms")
    ap.add_argument("--ttft-p90-target", type=float, default=None, help="TTFT p90 SLA target in ms")
    ap.add_argument("--itl-p50-target", type=float, default=None, help="ITL p50 SLA target in ms (25 claw / 66.7 chat)")
    ap.add_argument("--out-csv", required=True, help="output CSV path")
    ap.add_argument("--profile-name", default=None, help="optional profile label, added as a column")
    ap.add_argument("--no-audit", action="store_true", help="omit ISL/OSL/Valid/Notes columns")
    args = ap.parse_args()

    if args.json:
        json_paths = [args.json]
    else:
        json_paths = _discover_jsons(args.sweep_dir)

    combos = []
    for jp in json_paths:
        combos.extend(_load_combos_from_json(jp))

    if not combos:
        print("ERROR: no combinations parsed", file=sys.stderr)
        sys.exit(1)

    rows, notes = build_rows(
        combos, args.gpus, args.osl_target,
        ttft_p50_target=args.ttft_p50_target,
        ttft_p90_target=args.ttft_p90_target,
        itl_p50_target=args.itl_p50_target,
    )
    write_csv(rows, args.out_csv, profile_name=args.profile_name, audit=not args.no_audit)

    # ---- console summary ----
    label = args.profile_name or "(sweep)"
    print(f"[{label}] wrote {len(rows)} rows -> {args.out_csv}  "
          f"(gpus={args.gpus}, osl_target={args.osl_target}, "
          f"ttft_p50<={args.ttft_p50_target}, ttft_p90<={args.ttft_p90_target}, itl_p50<={args.itl_p50_target})")
    for n in notes:
        print("  [!]", n)

    valid = [r for r in rows if r["Valid"]]
    if valid:
        peak = max(valid, key=lambda r: r["Total tok/s/GPU"])
        print(f"  peak valid throughput: {peak['Total tok/s/GPU']} tok/s/GPU at conc {peak['Concurrency']}")

    # Max concurrency under the ITL gate (the "concurrency as high as possible
    # with P50 ITL under target" question). Highest-concurrency VALID point that
    # meets the ITL p50 gate.
    if args.itl_p50_target is not None:
        under_gate = [r for r in valid if r["ITL P50 OK"] == "Y"]
        if under_gate:
            best = max(under_gate, key=lambda r: (r["Concurrency"] or -1))
            ttft_note = ""
            if args.ttft_p50_target is not None or args.ttft_p90_target is not None:
                ttft_note = (f" | TTFT p50={best['TTFT P50(ms)']}ms ({best['TTFT P50 OK']}) "
                             f"p90={best['TTFT P90(ms)']}ms ({best['TTFT P90 OK']})")
            print(f"  MAX concurrency under ITL {args.itl_p50_target}ms: "
                  f"conc={best['Concurrency']} "
                  f"(ITL p50={best['ITL P50(ms)']}ms, "
                  f"in={best['Input tok/s/GPU']}, out={best['Output tok/s/GPU']} tok/s/GPU)"
                  f"{ttft_note} | overall SLA Pass={best['SLA Pass']}")
        else:
            print(f"  MAX concurrency under ITL {args.itl_p50_target}ms: NONE "
                  f"(no valid point met the ITL gate -- server too slow even at lowest concurrency)")


if __name__ == "__main__":
    main()