#!/usr/bin/env python3
"""Summarize a rocprofv3 --att stats CSV: stall cycles by instruction class + top stalls."""
import csv, glob, sys, collections
f = glob.glob(f"{sys.argv[1]}/stats_*.csv")[0]
rows = list(csv.DictReader(open(f)))
for r in rows:
    for k in ("Hitcount", "Latency", "Stall", "Idle"): r[k] = int(r[k] or 0)
lat = sum(r["Latency"] for r in rows); st = sum(r["Stall"] for r in rows)
print(f"total latency={lat:.3g} stall={st:.3g}")
cat = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    op = r["Instruction"].split()[0] if r["Instruction"] else "?"
    k = next((p for p in ["v_mfma", "ds_read", "ds_write", "buffer_load", "global_load", "s_barrier",
                          "s_waitcnt", "s_nop", "s_setprio", "v_", "s_"] if op.startswith(p)), op)
    cat[k][0] += r["Latency"]; cat[k][1] += r["Stall"]; cat[k][2] += r["Hitcount"]
for k, v in sorted(cat.items(), key=lambda x: -x[1][1])[:10]:
    print(f"  {k:12s} stall={v[1]:.3g} ({100*v[1]/max(st,1):.1f}% of stall) latency={v[0]:.3g} hits={v[2]}")
print("top stalls:")
for r in sorted(rows, key=lambda r: -r["Stall"])[:6]:
    print(f"  stall={r['Stall']:.3g} hits={r['Hitcount']} {r['Instruction'][:80]}")
