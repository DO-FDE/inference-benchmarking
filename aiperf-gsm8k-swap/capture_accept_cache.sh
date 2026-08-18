#!/usr/bin/env bash
# Fleet acceptance + cache-hit capture around the aiperf sweep.
#
# For each concurrency point: snapshot /metrics -> run the sweep point -> snapshot
# -> window-diff -> accept% / accept_len / cache_hit%. Writes accept_cache.csv.
#
# Metrics source:
#   By DEFAULT the server's /metrics is read from the SAME base URL the benchmark
#   hits (derived from $URL). That covers any single endpoint -- one container,
#   one server, a hosted endpoint -- with no extra input.
#   Only if the endpoint fans out across several engines whose /metrics differ
#   from the traffic URL (e.g. a router in front of N replicas) do you set
#   METRICS_ADDRS to a comma-separated list of engine base URLs to sum across.
set -uo pipefail

MODEL="${MODEL:?set MODEL}"
URL="${URL:?set URL (the benchmark endpoint)}"
ISL="${ISL:-120000}"; OSL="${OSL:-917}"; CACHE="${CACHE:-90}"
CONC="${CONC:-16,24,32,48,64,128}"; GPUS="${GPUS:-8}"; WARMUP="${WARMUP:-16}"
API_KEY="${API_KEY:-}"
# Metrics sources: default to the benchmark URL itself; override with METRICS_ADDRS.
# Accepts full base URLs (http://host:port) or bare host[:port]; /metrics is appended.
METRICS_ADDRS="${METRICS_ADDRS:-$URL}"
HARNESS="${HARNESS:-./run_aiperf_workload_shapes.sh}"
OUTDIR="${OUTDIR:-./results_accept}"
mkdir -p "$OUTDIR"

# normalize an address to a full http URL with no trailing slash
norm_url() {
  local a="$1"
  case "$a" in
    http://*|https://*) : ;;
    *) a="http://$a" ;;
  esac
  echo "${a%/}"
}

# sum a counter across all metrics addresses; arg1=exact metric name.
# Exact-name match: char after the name must be '{' or space, so we don't pick up
# _created (timestamp) or _per_pos (per-position breakdown) siblings.
sum_counter() {
  local metric="$1" total=0 v base
  for a in $(echo "$METRICS_ADDRS" | tr ',' ' '); do
    base="$(norm_url "$a")"
    v=$(curl -s ${API_KEY:+-H "Authorization: Bearer $API_KEY"} "${base}/metrics" 2>/dev/null \
        | awk -v m="$metric" '
            $0 !~ /^#/ {
              n=length(m)
              if (substr($0,1,n)==m) { c=substr($0,n+1,1); if (c=="{"||c==" ") s+=$2 }
            } END{print s+0}')
    total=$(python3 -c "print(${total}+${v:-0})")
  done
  echo "$total"
}

snapshot() {
  # echoes: D A N Q H  (draft_tokens, accepted_tokens, drafts, cache_queries, cache_hits)
  local D A N Q H
  D=$(sum_counter "vllm:spec_decode_num_draft_tokens_total")
  A=$(sum_counter "vllm:spec_decode_num_accepted_tokens_total")
  N=$(sum_counter "vllm:spec_decode_num_drafts_total")
  Q=$(sum_counter "vllm:prefix_cache_queries_total")
  H=$(sum_counter "vllm:prefix_cache_hits_total")
  echo "$D $A $N $Q $H"
}

echo "conc,accept_pct,accept_len,cache_hit_pct,d_draft,d_accept,d_drafts,d_qry,d_hits" > "$OUTDIR/accept_cache.csv"
echo "metrics source(s): $METRICS_ADDRS"

for c in $(echo "$CONC" | tr ',' ' '); do
  echo "=== concurrency $c: snapshot BEFORE ==="
  read D0 A0 N0 Q0 H0 <<< "$(snapshot)"
  echo "  before: D=$D0 A=$A0 N=$N0 Q=$Q0 H=$H0"

  echo "=== concurrency $c: run sweep point ==="
  API_ARGS=(); [ -n "$API_KEY" ] && API_ARGS=(--api-key "$API_KEY")
  GATE_ARGS=(); [ -n "${GATE_ARGS_STR:-}" ] && read -ra GATE_ARGS <<< "$GATE_ARGS_STR"
  MODEL="$MODEL" $HARNESS --model "$MODEL" --url "$URL" "${API_ARGS[@]}" \
    --isl "$ISL" --osl "$OSL" --cache "$CACHE" \
    --concurrency "$c" --gpus "$GPUS" --warmup "$WARMUP" "${GATE_ARGS[@]}" --skip-health-check \
    2>&1 | tail -5
  for f in aiperf_results/summary_ALL_*.csv; do
    [ -f "$f" ] && cp "$f" "$OUTDIR/summary_ALL_c${c}.csv" && break
  done
  rm -rf aiperf_results aiperf_artifacts_shapes 2>/dev/null || true

  echo "=== concurrency $c: snapshot AFTER ==="
  read D1 A1 N1 Q1 H1 <<< "$(snapshot)"
  echo "  after:  D=$D1 A=$A1 N=$N1 Q=$Q1 H=$H1"

  python3 - "$c" "$D0" "$A0" "$N0" "$Q0" "$H0" "$D1" "$A1" "$N1" "$Q1" "$H1" "$OUTDIR/accept_cache.csv" <<'PY'
import sys
c,D0,A0,N0,Q0,H0,D1,A1,N1,Q1,H1,out = sys.argv[1:]
f=float
dD,dA,dN,dQ,dH = f(D1)-f(D0), f(A1)-f(A0), f(N1)-f(N0), f(Q1)-f(Q0), f(H1)-f(H0)
acc  = 100*dA/dD if dD>0 else float('nan')
alen = dA/dN+1 if dN>0 else float('nan')
chr_ = 100*dH/dQ if dQ>0 else float('nan')
open(out,'a').write(f"{c},{acc:.2f},{alen:.2f},{chr_:.2f},{dD:.0f},{dA:.0f},{dN:.0f},{dQ:.0f},{dH:.0f}\n")
print(f"  conc {c}: accept={acc:.2f}% accept_len={alen:.2f} cache_hit={chr_:.2f}%  (dDraft={dD:.0f} dAccept={dA:.0f})")
PY
done

echo "======== accept_cache.csv ========"
cat "$OUTDIR/accept_cache.csv"

COMBINED="$OUTDIR/summary_ALL_combined.csv"; first=true
for f in $(ls -1 "$OUTDIR"/summary_ALL_c*.csv 2>/dev/null); do
  if $first; then head -1 "$f" > "$COMBINED"; first=false; fi
  tail -n +2 "$f" >> "$COMBINED"
done
[ -f "$COMBINED" ] && { echo "======== summary_ALL_combined.csv ========"; cat "$COMBINED"; }
