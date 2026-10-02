import json,subprocess
d=json.loads(subprocess.run(["amd-smi","metric","-c","-p","--json"],capture_output=True,text=True).stdout)
d=d.get("gpu_data",d) if isinstance(d,dict) else d
best=None
for g in d:
    p=g.get("power",{}).get("socket_power",{}).get("value",0)
    c=g.get("clock",{}); gfx=c.get("gfx_0",c.get("gfx",{})).get("clk",{}).get("value")
    if best is None or (isinstance(p,(int,float)) and p>best[1]): best=(g.get("gpu"),p,gfx)
print(f"busiest gpu {best[0]}: {best[1]} W, gfx clk {best[2]} MHz")
