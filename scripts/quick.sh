#!/usr/bin/env bash
# quick tuning iteration: rebuild, show spills/occupancy, run CK only on top3
cd /workspace/cktile_tunemax
./scripts/build.sh > /tmp/quick_build.log 2>&1 || { grep -E "error" /tmp/quick_build.log | head -5; exit 1; }
grep -E "took" /tmp/quick_build.log
grep -E "VGPRs Spill|Occupancy|LDS Size" /tmp/quick_build.log | sed 's/.*remark: *//;s/ \[-R.*//' | sort | uniq -c | grep -v "Occupancy \[waves/SIMD\]: 8$" | tr '\n' ';'; echo
./build/tunemax_bench --shapes shapes/top3.csv ${QUICK_ONLY:+--only $QUICK_ONLY} --backends ck_tile --warmup 5 --iters 20 --out /tmp/quick.csv "$@" 2>&1 | grep -E "ck_tile|FAIL|ERROR" | awk '{print $2, $3, $4, $5, $6}'
