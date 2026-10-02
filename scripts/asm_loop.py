#!/usr/bin/env python3
"""Compile the current CK config to device asm and summarize the bf16/no-pad kernel's loops.
Usage: scripts/asm_loop.py   (writes /tmp/asm/ck.s)"""
import re, subprocess, sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = "/workspace/rocm-libraries/projects/composablekernel/include"
os.makedirs("/tmp/asm", exist_ok=True)
cmd = ["/opt/rocm/bin/amdclang++", "-x", "hip", "--offload-arch=gfx950", "--cuda-device-only", "-S",
       "-std=c++20", "-O3", "-fno-offload-uniform-block", "-I", ck, "-I", f"{root}/common",
       "-I", f"{root}/ck_kernel", f"{root}/ck_kernel/ck_grouped_gemm.cpp", "-o", "/tmp/asm/ck.s"] + sys.argv[1:]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode:
    print("\n".join(l for l in r.stderr.split("\n") if "error" in l)[:2000]); sys.exit(1)
s = open("/tmp/asm/ck.s").read()
names = [(m.start(), m.group(1)) for m in re.finditer(r"^(_ZN7ck_tile6kentry\S*):", s, re.M)]
for i, (st, nm) in enumerate(names):
    if "DF16b" not in nm or "Lb1ELb1ELb1E" in nm:   # bf16, unpadded instance only
        continue
    body = s[st: names[i + 1][0] if i + 1 < len(names) else len(s)]
    meta = {k: re.search(rf"; {k}: (\d+)", body) for k in ["NumVgprs", "NumAgprs", "Occupancy", "ScratchSize"]}
    print("  ".join(f"{k}={v.group(1)}" for k, v in meta.items() if v))
    lines = body.split("\n")
    lab = {m.group(1): j for j, l in enumerate(lines) if (m := re.match(r"^(\.LBB\d+_\d+):", l))}
    for j, l in enumerate(lines):
        m = re.search(r"s_cbranch_\w+\s+(\.LBB\d+_\d+)", l)
        if m and m.group(1) in lab and lab[m.group(1)] < j:
            ops = [x.strip().split()[0] for x in lines[lab[m.group(1)] + 1: j + 1] if x.strip() and x.strip()[0] not in ";."]
            if not any(o.startswith("v_mfma") for o in ops):
                continue
            cnt = {}
            for o in ops:
                k = next((p for p in ["v_mfma", "ds_read", "ds_load", "ds_write", "ds_store", "buffer_load", "global_load",
                                      "s_barrier", "s_waitcnt", "v_accvgpr", "scratch", "s_nop", "v_", "s_"] if o.startswith(p)), o)
                cnt[k] = cnt.get(k, 0) + 1
            print(f"  loop {len(ops)} instrs:", dict(sorted(cnt.items(), key=lambda x: -x[1])))
    break
