#!/usr/bin/env bash
# usage: pmc.sh BACKEND ; profiles TestID 71, prints per-kernel-type counter sums
be=$1; rm -rf /tmp/pmc_$be
rocprofv3 --pmc ${PMC:-SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_VALU_MFMA_BUSY_CYCLES SQ_BUSY_CU_CYCLES GRBM_GUI_ACTIVE SQ_LDS_ADDR_CONFLICT SQ_WAIT_INST_LDS} -d /tmp/pmc_$be -o p --output-format csv -- ./build/tunemax_bench --shapes shapes/top3.csv --only 71 --backends $be --warmup 0 --iters 2 --no-check --out /tmp/pmc.csv > /dev/null 2>&1
python3 - "$be" <<'PY'
import csv,glob,sys,collections
f=glob.glob(f'/tmp/pmc_{sys.argv[1]}/**/*counter_collection.csv',recursive=True)
if not f: print("no counters"); sys.exit()
rows=list(csv.DictReader(open(f[0])))
agg=collections.defaultdict(float); kn=set()
for r in rows:
    if 'Cijk' in r['Kernel_Name'] or 'GroupedGemm' in r['Kernel_Name']:
        agg[r['Counter_Name']]+=float(r['Counter_Value']); kn.add(r['Kernel_Name'][:50])
print(sys.argv[1], kn)
for k,v in sorted(agg.items()): print(f"  {k:28s} {v:.4g}")
if agg.get('SQ_LDS_IDX_ACTIVE'): print(f"  bank conflict / LDS active = {agg['SQ_LDS_BANK_CONFLICT']/agg['SQ_LDS_IDX_ACTIVE']:.3f}")
if agg.get('SQ_BUSY_CU_CYCLES'): print(f"  MFMA busy / CU busy = {agg['SQ_VALU_MFMA_BUSY_CYCLES']/agg['SQ_BUSY_CU_CYCLES']:.3f}")
PY
