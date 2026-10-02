#!/usr/bin/env bash
# Run a shapes CSV across all GPUs, one process per case (both backends of a case run on the
# same GPU, so comparisons stay fair). One process per case avoids the intermittent stall of the
# Primus-Turbo multi-stream hipBLASLt loop in long-lived processes (see its source comments).
#   scripts/run_parallel.sh shapes/full.csv OUT.csv [extra tunemax_bench args]
# NGPU=1 runs sequentially on GPU 0. Running all GPUs at once skewed results (some cases 2-4x
# slower than on an idle node), so use NGPU=1 for numbers you report.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
shapes=$1; out=$2; shift 2
ngpu=$(ls -d /sys/class/kfd/kfd/topology/nodes/*/ 2>/dev/null | while read d; do grep -q "gfx_target_version [1-9]" $d/properties && echo x; done | wc -l)
ngpu=${NGPU:-$ngpu}
tmp=$(mktemp -d)
ids=$(tail -n +2 "$shapes" | cut -d, -f1)
worker() {
  local g=$1; shift
  local i=0
  for id in $ids; do
    if (( i % ngpu == g )); then
      HIP_VISIBLE_DEVICES=$g timeout 300 ./build/tunemax_bench --shapes "$shapes" --only $id \
        --out "$tmp/$id.csv" "$@" > "$tmp/$id.log" 2>&1 || echo "case $id failed/timeout on gpu $g" >&2
    fi
    i=$((i+1))
  done
}
for g in $(seq 0 $((ngpu-1))); do worker $g "$@" & done
wait
first=$(ls "$tmp"/*.csv | head -1); head -1 "$first" > "$out"
for f in $(ls "$tmp"/*.csv | sort -t/ -k3 -n); do tail -n +2 "$f"; done >> "$out"
echo "wrote $out ($(($(wc -l < "$out")-1)) rows) using $ngpu GPUs"
