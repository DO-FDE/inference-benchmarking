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


def _parse_server_metrics_file(path):
    """Parse ONE aiperf server_metrics.json -> dict of cache/accept metrics ({} on failure)."""
    try:
        with open(path) as f:
            sm = json.load(f)
    except Exception:
        return {}

    # Prefer the per-window 'rate' field over cumulative 'total': for a ratio
    # (hits/queries, accepted/drafted) the rate-based value is correct PER RUN
    # even when 'total' is cumulative-since-server-start across a sweep.
    want = {
        "prefix_cache_hits": None, "prefix_cache_queries": None,
        "spec_decode_num_accepted_tokens": None,
        "spec_decode_num_draft_tokens": None,
        "spec_decode_num_drafts": None,
    }
    per_pos = {}
    eps = (sm.get("data", {}) or {}).get("endpoint_summaries", {}) or {}
    for _url, summ in eps.items():
        for name, m in (summ.get("metrics", {}) or {}).items():
            base = name.replace("vllm:", "")
            for s in m.get("series", []):
                st = s.get("stats", {}) or {}
                lbl = s.get("labels", {}) or {}
                val = st.get("rate")
                if val is None:
                    val = st.get("total", st.get("sum"))
                if base == "spec_decode_num_accepted_tokens_per_pos" and "position" in lbl:
                    if val is not None:
                        per_pos[int(lbl["position"])] = per_pos.get(int(lbl["position"]), 0) + val
                    continue
                if base in want and val is not None:
                    want[base] = (want[base] or 0) + val

    out = {}
    H, Q = want["prefix_cache_hits"], want["prefix_cache_queries"]
    A = want["spec_decode_num_accepted_tokens"]
    D = want["spec_decode_num_draft_tokens"]
    N = want["spec_decode_num_drafts"]
    # Only emit physically-possible values. A mistimed/near-empty scrape can yield
    # hits>queries (cache >100%) or drafts~0 (accept 0% / len 1) -- treat as missing
    # rather than reporting garbage. Require a meaningful sample size too.
    MIN_Q = 1000.0   # need real cache traffic
    MIN_D = 100.0    # need real draft traffic
    if H is not None and Q and Q >= MIN_Q and 0.0 <= H <= Q:
        out["Cache Hit%"] = round(100.0 * H / Q, 2)
    if A is not None and D and D >= MIN_D and 0.0 <= A <= D:
        out["Accept%"] = round(100.0 * A / D, 2)
        if N:
            out["Accept Len"] = round(A / N + 1.0, 2)
    if per_pos and out.get("Accept%") is not None:
        out["Accept/Pos"] = "/".join(str(int(per_pos[k])) for k in sorted(per_pos))
    return out


def _build_server_metrics_map(search_root):
    """
    Scan a run tree for per-concurrency server_metrics and return
    {concurrency:int -> metrics dict}. aiperf writes several per point; we use
    ONLY the profiling-phase file ('.../phases/profiling/server_metrics.json'),
    because the warmup file and the top-level '*_server_metrics.json' hold
    partial or cumulative counters that yield wrong ratios (e.g. >100% cache).
    Falls back to any server_metrics.json only if no profiling-phase file exists.
    """
    import re as _re
    if not search_root or not os.path.isdir(search_root):
        return {}
    all_sm = glob.glob(os.path.join(search_root, "**", "server_metrics.json"), recursive=True)
    # split into profiling-phase files vs the rest
    prof = [p for p in all_sm if os.path.join("phases", "profiling") in p]
    other = [p for p in all_sm if p not in prof and "warmup" not in p]  # top-level, not warmup

    def _key(p):
        m = _re.search(r"concurrency_(\d+)__requests_\d+", p)
        return int(m.group(1)) if m else None

    result = {}
    for p in prof:  # profiling phase wins
        metrics = _parse_server_metrics_file(p)
        if metrics:
            result[_key(p)] = metrics
    for p in other:  # only fill gaps profiling didn't cover
        k = _key(p)
        if k not in result:
            metrics = _parse_server_metrics_file(p)
            if metrics:
                result[k] = metrics
    return result


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
               ttft_p50_target=None, ttft_p90_target=None, gates=None,
               server_metrics_map=None):
    """
    Turn raw combos into validated summary rows.
    gates: list of (name, itl_p50_ms) tuples -> one 'Gate:<name>' column each.
    server_metrics_map: {concurrency:int -> metrics dict}; matched per row. A
    single unkeyed entry (key None) applies to all rows when no per-conc match.
    """
    gates = gates or []
    smap = server_metrics_map or {}
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

        if reqs is None:
            reqs = (m.get("request_count") or {}).get("avg")
        if conc is None:
            conc = osl_n

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

        # ---- latency + gates ----
        ttft_p50 = ttft.get("p50")
        ttft_p90 = ttft.get("p90")
        itl_p50 = itl.get("p50")
        ttft50_ok = _lt(ttft_p50, ttft_p50_target)
        ttft90_ok = _lt(ttft_p90, ttft_p90_target)

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
        }

        # one pass/fail column per named ITL gate, all on this row
        all_gates_pass = True
        for gname, gitl in gates:
            gate_ok = bool(valid and ttft50_ok and ttft90_ok and _lt(itl_p50, gitl))
            row[f"Gate:{gname}"] = "Y" if gate_ok else "N"
            all_gates_pass = all_gates_pass and gate_ok
        if gates:
            row["SLA Pass"] = "Y" if all_gates_pass else "N"

        # merge server metrics (cache-hit / accept), matched by concurrency.
        sm = smap.get(int(conc) if conc is not None else None)
        if sm is None and None in smap:
            sm = smap[None]                       # explicit single-point entry
        if sm is None and len(smap) == 1:
            sm = next(iter(smap.values()))        # sole entry -> apply it
        if sm:
            for k in ("Cache Hit%", "Accept%", "Accept Len", "Accept/Pos"):
                if k in sm:
                    row[k] = sm[k]

        row.update({
            "ISL": round(isl, 1),
            "OSL": round(osl, 1),
            "Valid": valid,
            "Notes": "; ".join(reasons) if reasons else ("identity check failed" if not identity_ok else ""),
        })
        rows.append(row)
        if degenerate:
            notes.append(f"conc={conc}: DEGENERATE -- {'; '.join(reasons)} (EXCLUDE from curve)")
        elif not identity_ok:
            notes.append(f"conc={conc}: identity check failed (review)")

    rows.sort(key=lambda r: (r["Concurrency"] is None, r["Concurrency"]))
    return rows, notes


# columns written to the final CSV (report cols, then metrics, then gates, then audit)
REPORT_COLS = [
    "Concurrency", "Requests", "Duration(s)", "Req/s",
    "Input tok/s/GPU", "Output tok/s/GPU", "Total tok/s/GPU",
    "TTFT P50(ms)", "TTFT P90(ms)", "ITL P50(ms)", "ITL P90(ms)",
]
# server-derived metrics (present only if server_metrics.json was found)
SERVER_COLS = ["Cache Hit%", "Accept%", "Accept Len", "Accept/Pos"]
TTFT_OK_COLS = ["TTFT P50 OK", "TTFT P90 OK"]
AUDIT_COLS = ["ISL", "OSL", "Valid", "Notes"]


def write_csv(rows, out_csv, profile_name=None, audit=True, gate_names=None):
    gate_cols = [f"Gate:{g}" for g in (gate_names or [])]
    sla_col = ["SLA Pass"] if gate_cols else []
    # only include server cols that actually appear in the rows
    server_present = [c for c in SERVER_COLS if any(c in r for r in rows)]
    cols = (REPORT_COLS + server_present + TTFT_OK_COLS + gate_cols + sla_col
            + (AUDIT_COLS if audit else []))
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


def _parse_gate(s):
    # "name:itl_ms" -> (name, float ms)
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"gate must be NAME:ITL_MS, got '{s}'")
    name, ms = s.rsplit(":", 1)
    return (name, float(ms))


def main():
    ap = argparse.ArgumentParser(description="Parse aiperf sweep into per-GPU summary CSV with per-gate pass/fail + cache-hit/accept")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep-dir", help="path to sweep_aggregate dir (auto-discovers JSON)")
    src.add_argument("--json", help="path to a single aiperf JSON export")
    ap.add_argument("--gpus", type=int, required=True, help="GPU count for per-GPU division")
    ap.add_argument("--osl-target", type=int, default=1000, help="expected OSL (for degeneracy check)")
    ap.add_argument("--ttft-p50-target", type=float, default=None, help="TTFT p50 SLA target in ms")
    ap.add_argument("--ttft-p90-target", type=float, default=None, help="TTFT p90 SLA target in ms")
    # gates: repeatable NAME:ITL_MS. Back-compat: --itl-p50-target adds a gate named 'itl'.
    ap.add_argument("--gate", action="append", type=_parse_gate, default=[],
                    help="ITL p50 gate as NAME:ITL_MS, repeatable (e.g. --gate claw:25 --gate chat:66.7)")
    ap.add_argument("--itl-p50-target", type=float, default=None,
                    help="(compat) single ITL p50 target ms; added as gate 'itl'")
    ap.add_argument("--out-csv", required=True, help="output CSV path")
    ap.add_argument("--profile-name", default=None, help="optional profile label column")
    ap.add_argument("--no-audit", action="store_true", help="omit ISL/OSL/Valid/Notes columns")
    ap.add_argument("--no-server-metrics", action="store_true", help="don't read server_metrics.json")
    ap.add_argument("--artifacts-dir", default=None, help="explicit dir to scan for per-concurrency server_metrics.json")
    args = ap.parse_args()

    gates = list(args.gate)
    if args.itl_p50_target is not None:
        gates.append(("itl", args.itl_p50_target))
    if not gates:
        gates = [("claw", 25.0), ("chat", 66.7)]  # defaults
    gate_names = [g[0] for g in gates]

    if args.json:
        json_paths = [args.json]
    else:
        json_paths = _discover_jsons(args.sweep_dir)

    combos = []
    for jp in json_paths:
        combos.extend(_load_combos_from_json(jp))

    # build per-concurrency server-metrics map by scanning the run tree.
    server_metrics_map = {}
    if not args.no_server_metrics:
        search_roots = []
        if args.artifacts_dir:
            search_roots.append(args.artifacts_dir)
        if args.sweep_dir:
            # the artifacts tree usually sits beside aiperf_results, one level up
            search_roots.append(args.sweep_dir)
            search_roots.append(os.path.dirname(os.path.abspath(args.sweep_dir)))
        for jp in json_paths:
            search_roots.append(os.path.dirname(os.path.abspath(jp)))
        for root in search_roots:
            m = _build_server_metrics_map(root)
            if m:
                # merge, preferring keyed (per-conc) entries
                for k, v in m.items():
                    server_metrics_map.setdefault(k, v)
            if server_metrics_map:
                break

    if not combos:
        print("ERROR: no combinations parsed", file=sys.stderr)
        sys.exit(1)

    rows, notes = build_rows(
        combos, args.gpus, args.osl_target,
        ttft_p50_target=args.ttft_p50_target,
        ttft_p90_target=args.ttft_p90_target,
        gates=gates,
        server_metrics_map=server_metrics_map,
    )
    write_csv(rows, args.out_csv, profile_name=args.profile_name,
              audit=not args.no_audit, gate_names=gate_names)

    # ---- console summary ----
    label = args.profile_name or "(sweep)"
    gate_str = ", ".join(f"{n}<{ms}ms" for n, ms in gates)
    print(f"[{label}] wrote {len(rows)} rows -> {args.out_csv}  "
          f"(gpus={args.gpus}, gates: {gate_str})")
    if server_metrics_map:
        n_keyed = len([k for k in server_metrics_map if k is not None])
        sample = next(iter(server_metrics_map.values()))
        print(f"  server metrics: found for {n_keyed} concurrency point(s); "
              f"e.g. cache_hit={sample.get('Cache Hit%')}% accept={sample.get('Accept%')}% "
              f"accept_len={sample.get('Accept Len')}")
    for n in notes:
        print("  [!]", n)

    valid = [r for r in rows if r["Valid"]]
    if valid:
        peak = max(valid, key=lambda r: r["Total tok/s/GPU"])
        print(f"  peak valid throughput: {peak['Total tok/s/GPU']} tok/s/GPU at conc {peak['Concurrency']}")

    # max concurrency under each gate
    for gname, gms in gates:
        under = [r for r in valid if r.get(f"Gate:{gname}") == "Y"]
        if under:
            best = max(under, key=lambda r: (r["Concurrency"] or -1))
            print(f"  MAX concurrency under {gname} (ITL<{gms}ms): conc={best['Concurrency']} "
                  f"(ITL p50={best['ITL P50(ms)']}ms, in={best['Input tok/s/GPU']}, "
                  f"out={best['Output tok/s/GPU']} tok/s/GPU)")
        else:
            print(f"  MAX concurrency under {gname} (ITL<{gms}ms): NONE")


if __name__ == "__main__":
    main()