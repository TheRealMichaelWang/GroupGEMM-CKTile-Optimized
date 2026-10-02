#!/usr/bin/env bash
# thread trace of the CK kernel on Kimi-K2 (TestID 323), 1 CU; prints cycles per MFMA + summary
cd /workspace/cktile_tunemax; rm -rf /tmp/att
timeout 55 rocprofv3 --att --att-target-cu 1 --att-shader-engine-mask 0x1 --kernel-include-regex GroupedGemm -d /tmp/att -o a -- ./build/tunemax_bench --shapes shapes/top3.csv --only 323 --backends ck_tile --warmup 0 --iters 1 --no-check --out /tmp/x.csv > /tmp/att.log 2>&1
python3 - <<'PY'
import csv,glob
r=list(csv.DictReader(open(glob.glob("/tmp/att/stats_*.csv")[0])))
lat=sum(int(x['Latency'] or 0) for x in r); n=sum(int(x['Hitcount'] or 0) for x in r if x['Instruction'].startswith('v_mfma'))
print(f"cycles/MFMA = {lat/n:.2f}  (ideal 16)")
PY
python3 scripts/att_summary.py /tmp/att | sed -n 2,7p
