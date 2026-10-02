#!/usr/bin/env python3
"""Regenerate shapes/full.csv and shapes/top3.csv from Primus-Turbo's benchmark config.

Offline helper only (the benchmark itself is C++). It reads the model table and size
lists out of Primus-Turbo's benchmark/ops/training/config.py with `ast` instead of
importing it, so torch is not needed, and then repeats the loop in
`gen_grouped_gemm_test_cases()` / `_generate_moe_test_cases()` so the cases and their
order (TestID) match bench_grouped_gemm_turbo.py exactly.

Usage: scripts/gen_shapes.py [path/to/config.py]
"""

import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("/workspace/Primus-Turbo/benchmark/ops/training/config.py")
TOP_N = 3


def read_literals(config_path, names):
    tree = ast.parse(config_path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        sys.exit(f"{config_path}: could not find {sorted(missing)}")
    return found


def gen_cases(lits):
    cases = []
    for prefix, cfg in lits["MoEModelConfigs"].items():
        n_routed_experts = cfg["n_routed_experts"]
        inter, hidden = cfg["moe_intermediate_size"], cfg["hidden_size"]
        shapes = {f"{prefix}-GateUP": (2 * inter, hidden), f"{prefix}-Down": (hidden, inter)}
        for ep in lits["GROUPED_GEMM_EP_SIZE_LIST"]:
            if n_routed_experts % ep != 0:
                continue
            b = n_routed_experts // ep
            if b < 1:
                continue
            for m in lits["GROUPED_GEMM_M_SIZE_LIST"]:
                for name, (n, k) in shapes.items():
                    cases.append({"Case": name, "B": b, "M": m, "N": n, "K": k})
    for test_id, case in enumerate(cases, start=1):
        case["TestID"] = test_id
    return cases


def write_csv(path, cases):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["TestID", "Case", "B", "M", "N", "K"], lineterminator="\n")
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in w.fieldnames})
    print(f"wrote {len(cases):4d} cases -> {path.relative_to(ROOT)}")


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    lits = read_literals(
        config_path, {"MoEModelConfigs", "GROUPED_GEMM_EP_SIZE_LIST", "GROUPED_GEMM_M_SIZE_LIST"}
    )
    cases = gen_cases(lits)
    write_csv(ROOT / "shapes" / "full.csv", cases)
    # "Largest" = most FLOPs (2*B*M*N*K); ties keep the earlier TestID. Written in TestID order.
    top = sorted(cases, key=lambda c: (-c["B"] * c["M"] * c["N"] * c["K"], c["TestID"]))[:TOP_N]
    write_csv(ROOT / "shapes" / "top3.csv", sorted(top, key=lambda c: c["TestID"]))


if __name__ == "__main__":
    main()
