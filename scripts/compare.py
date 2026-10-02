#!/usr/bin/env python3
"""Compare ck_tile vs another backend from one or more results CSVs (or a dir of per-case CSVs)."""
import csv, glob, math, os, sys
paths = []
for a in sys.argv[1:] or ["results"]:
    paths += glob.glob(os.path.join(a, "*.csv")) if os.path.isdir(a) else [a]
by = {}
for p in paths:
    for r in csv.DictReader(open(p)):
        by.setdefault(int(r["TestID"]), {})[r["Backend"]] = r
other = "hipblaslt_loop"
pairs = [(i, d["ck_tile"], d[other]) for i, d in sorted(by.items())
         if "ck_tile" in d and other in d and d["ck_tile"]["Check"] == "PASS"
         and d[other]["Check"] in ("PASS", "SKIP")]  # SKIP: baseline timed with --no-check
bad = [i for i, d in by.items() if any(x["Check"] not in ("PASS", "SKIP") for x in d.values())]
n = len(pairs)
wins = sum(float(c["TFLOPS"]) > float(h["TFLOPS"]) for _, c, h in pairs)
geo = math.exp(sum(math.log(float(h["Time_ms"]) / float(c["Time_ms"])) for _, c, h in pairs) / n)
tc = sum(float(c["Time_ms"]) for _, c, h in pairs); th = sum(float(h["Time_ms"]) for _, c, h in pairs)
mc = sum(float(c["TFLOPS"]) for _, c, h in pairs) / n; mh = sum(float(h["TFLOPS"]) for _, c, h in pairs) / n
print(f"{n} cases both PASS; ck_tile faster in {wins}/{n}; non-PASS cases: {sorted(bad)[:10]}")
print(f"geomean speedup ck_tile vs {other}: {geo:.3f}x ; total time {tc:.1f} ms vs {th:.1f} ms ({th/tc:.3f}x)")
print(f"mean TFLOPS: ck_tile {mc:.0f}  {other} {mh:.0f}")
losses = sorted(pairs, key=lambda t: float(t[1]["TFLOPS"]) / float(t[2]["TFLOPS"]))[:8]
print("worst cases for ck_tile (ck/other TFLOPS):")
for i, c, h in losses:
    print(f"  {i:3d} {c['Case']:28s} B={c['B']:>3} M={c['M']:>5} N={c['N']:>5} K={c['K']:>5}  "
          f"{float(c['TFLOPS']):6.0f} vs {float(h['TFLOPS']):6.0f}  ({float(c['TFLOPS'])/float(h['TFLOPS']):.2f})")
