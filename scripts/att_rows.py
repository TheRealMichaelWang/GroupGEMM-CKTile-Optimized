#!/usr/bin/env python3
"""Per-MFMA-row cycle totals in the main loop of the last scripts/att.sh trace."""
import csv, glob
r = list(csv.DictReader(open(glob.glob("/tmp/att/stats_*.csv")[0])))
for x in r:
    for k in ("Hitcount", "Latency", "Stall"): x[k] = int(x[k] or 0)
r.sort(key=lambda x: int(x['Vaddr'] or 0))
cnt = {}
for x in r:
    if x['Instruction'].startswith('v_mfma'): cnt[x['Hitcount']] = cnt.get(x['Hitcount'], 0) + 1
H = max(cnt)
loop = [x for x in r if x['Hitcount'] in (H, 2 * H)]
n = acc = 0; notes = []
for x in loop:
    acc += x['Latency'] / H
    if not x['Instruction'].startswith('v_mfma') and x['Latency'] / H > 9:
        notes.append(f"{x['Instruction'].split(' ')[0]}:{x['Latency']/H:.0f}")
    if x['Instruction'].startswith('v_mfma'):
        n += 1
        if n % 16 == 0:
            print(f"row {n//16-1:2d}: {acc:5.0f} ({acc/16:.2f}/MFMA) {' '.join(notes)}"); acc = 0; notes = []
