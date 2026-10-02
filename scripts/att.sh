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
python3 - <<'PY'
import csv,glob
r=list(csv.DictReader(open(glob.glob("/tmp/att/stats_*.csv")[0])))
for x in r:
    for k in ("Hitcount","Latency"): x[k]=int(x[k] or 0)
cnt={}
for x in r:
    if x['Instruction'].startswith('v_mfma'): cnt[x['Hitcount']]=cnt.get(x['Hitcount'],0)+1
h=max(cnt)  # hitcount of the main-loop MFMAs
loop=[x for x in r if x['Hitcount'] in (h,2*h)]
lo=min(int(x['Vaddr']) for x in loop); hi=max(int(x['Vaddr']) for x in loop)
nm=sum(x['Hitcount'] for x in loop if x['Instruction'].startswith('v_mfma'))
tiles=nm/14336
f=lambda c:sum(x['Latency'] for x in r if c(int(x['Vaddr'] or 0)))
nall=sum(x['Hitcount'] for x in r if x['Instruction'].startswith('v_mfma')); tiles=nall/14336
print(f"per wave-tile: total {f(lambda v:True)/tiles:.0f} (ideal 229376)  main loop {f(lambda v:lo<=v<=hi)/nm:.2f}/MFMA  outside main loop {f(lambda v:not(lo<=v<=hi))/tiles:.0f}")
PY
