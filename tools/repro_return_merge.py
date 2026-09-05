"""Control-aware result merging: a ``return`` nested inside a loop must reach
the function exit carrying its OWN values, merged per slot with every other
return (docs/PLAN_CONTROL_AWARE_RESULT_MERGING.md).

Lowers the same two-output ``leaf`` shape as tools/audit_result_binding_
nested_return.py -- both returns inside ``while True:``, with DIFFERENT
values (``x * 2.0`` vs ``x``) -- all the way to repository SSA, then inspects
the lowered ``leaf``:

* exactly one ``Ret``, in the ``function_exit`` block;
* a ``Phi`` per output slot with ``binding == "return_merge"`` and one
  incoming per return site;
* branches stamped ``source_control == "return"``;
* no SSA result id defined more than once (the "last return wins" defect
  showed up as duplicate definitions of one id).
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

NESTED = """
def leaf(x, n):
    i = 0
    while True:
        i = i + 1
        if i > n:
            return x * 2.0, i
        if i > 100:
            return x, i


def tick(x, n):
    total = x * 0.0
    k = 0
    j = 0
    while j < 2:
        y, k = leaf(x, n)
        total = total + y
        j = j + 1
    return total, k
"""


def _contract():
    values = [
        {
            "function": "tick", "parameter": "x", "storage": "span",
            "dtype": "float64", "rank": 1, "shape": [4],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
        {
            "function": "tick", "parameter": "n", "storage": "scalar",
            "dtype": "int64", "rank": 0, "python_type": "builtins.int",
        },
    ]
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(
        {"records": {}, "bindings": [], "values": values}
    ).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")


def main() -> int:
    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            NESTED, "tick",
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            name="return_merge", runtime_closure_only=True,
            extraction_contract=_contract(),
        )
    except Exception as error:  # noqa: BLE001
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1500]}", flush=True)
        return 1
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    functions = getattr(module, "functions", {}) or {}
    ok = True
    for name, function in functions.items():
        if "leaf" not in name or "__planned_region_" in name:
            continue
        rets = [
            (block_name, [int(a.id) for a in instruction.args])
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret"}
        ]
        merges = [
            (block_name, [int(a.id) for a in instruction.args],
             int(instruction.res.id), dict(instruction.attributes or {}))
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Phi", "phi"}
            and (instruction.attributes or {}).get("binding") == "return_merge"
        ]
        return_edges = [
            (block_name, str(instruction.op))
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if (instruction.attributes or {}).get("source_control") == "return"
        ]
        counts = Counter(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        duplicates = sorted((k, v) for k, v in counts.items() if v > 1)
        print(f"function {name}")
        print(f"  blocks={list(function.blocks)}")
        print(f"  rets={rets}")
        print(f"  return_edges={return_edges}")
        for merge in merges:
            print(f"  return_merge={merge}")
        print(f"  duplicate_result_ids={duplicates}")
        if len(rets) != 1 or rets[0][0] != "function_exit":
            ok = False
            print("  !! expected exactly one Ret in function_exit")
        if not merges:
            ok = False
            print("  !! expected return_merge Phis")
        if duplicates:
            ok = False
            print("  !! duplicate SSA result ids")
    print("OK" if ok else "CHECK FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
