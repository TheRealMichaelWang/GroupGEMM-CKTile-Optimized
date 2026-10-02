#!/usr/bin/env bash
# Run the benchmark on one of the shape sets. Same program, different CSV.
#
#   scripts/run.sh top3 [extra args]   # 3 largest shapes (quick check while tuning)
#   scripts/run.sh full [extra args]   # all 360 Primus-Turbo shapes
#   scripts/run.sh path/to/shapes.csv [extra args]
#
# Extra args go to tunemax_bench, e.g.
#   --dtype fp16                       (hipBLASLt's grouped kernels only exist for fp16 here)
#   --backends ck_tile                 (skip the baselines)
#   --only 59,71                       (TestIDs)
#   --warmup 20 --iters 100 --no-check --out results/foo.csv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set_name="${1:?usage: scripts/run.sh top3|full|FILE.csv [args...]}"
shift
case "$set_name" in
  top3|full) shapes="shapes/$set_name.csv" ;;
  *) shapes="$set_name" ;;
esac

cd "$ROOT"
mkdir -p results
exec ./build/tunemax_bench --shapes "$shapes" "$@"
