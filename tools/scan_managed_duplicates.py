"""Lower the managed balloon-tire program and scan for duplicate SSA results.

One SSA id must have exactly one defining instruction per function.  Every
duplicate found is printed with each producer and the three instructions
that precede it, plus the ``Ret``/``return_merge`` shape of the functions
that own control-aware result merging.  Exit status 2 on any duplicate.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.vehicle_python_compilation import (  # noqa: E402
    lower_balloon_tire_managed_python_ssa,
)


def main() -> int:
    t0 = time.time()
    lowered = lower_balloon_tire_managed_python_ssa(
        progress=lambda message: print(message, flush=True),
    )
    print(f"LOWERED OK in {time.time() - t0:.1f}s", flush=True)
    functions = dict(lowered.module.functions)
    print(f"functions={len(functions)}")
    duplicate_functions = 0
    for name, function in functions.items():
        counts = Counter(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        # A written mutable record field is caller-owned in/out storage whose
        # formal id every producer deliberately redefines (the C backend
        # publishes the last definition at exit).  Report those apart.
        inout_ids = {
            int(argument.id)
            for argument in function.args
            if bool((argument.accounting or {}).get("program_abi_mutable"))
            and bool((argument.accounting or {}).get("program_abi_field_written"))
        }
        redefined_inout = sorted(
            (k, v) for k, v in counts.items() if v > 1 and k in inout_ids
        )
        if redefined_inout:
            print(f"INOUT-REDEFINED {name[-70:]} {redefined_inout[:12]}")
        duplicates = sorted(
            (k, v) for k, v in counts.items() if v > 1 and k not in inout_ids
        )
        if not duplicates:
            continue
        duplicate_functions += 1
        print(f"DUPLICATE-RES {name[-70:]} {duplicates[:12]}")
        for duplicate_id, _count in duplicates[:6]:
            for block_name, block in function.blocks.items():
                for index, instruction in enumerate(block.instrs):
                    if instruction.res is None or int(instruction.res.id) != duplicate_id:
                        continue
                    attributes = {
                        key: value
                        for key, value in (instruction.attributes or {}).items()
                        if key in (
                            "binding", "incoming_blocks", "callee", "region_index",
                            "source_output_id", "source_control", "initial_value_id",
                        )
                    }
                    print(f"  producer[{duplicate_id}] {block_name}: {instruction.op} "
                          f"{[int(a.id) for a in instruction.args]} {attributes}")
                    for previous in block.instrs[max(0, index - 3):index]:
                        print(f"      before: {previous.op} {[int(a.id) for a in previous.args]} "
                              f"-> {None if previous.res is None else int(previous.res.id)} "
                              f"{(previous.attributes or {}).get('callee', '')}")
    print(f"duplicate functions={duplicate_functions}")
    for name, function in functions.items():
        if not any(key in name for key in (
            "step_with_dt_control_used", "run_superstep", "_propose_dt_pen", "pi_update",
        )) or "__planned_region_" in name:
            continue
        rets = [
            (block_name, [int(a.id) for a in instruction.args])
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret"}
        ]
        merges = [
            (block_name, [int(a.id) for a in instruction.args], int(instruction.res.id),
             (instruction.attributes or {}).get("incoming_blocks"))
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Phi", "phi"}
            and (instruction.attributes or {}).get("binding") == "return_merge"
        ]
        print(f"FN {name[-60:]} rets={rets} merges={merges}")
        region_calls = [
            (block_name, str((instruction.attributes or {}).get("callee", "")).rsplit("__", 1)[-1],
             [int(a.id) for a in instruction.args])
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op == "Call"
            and "__planned_region_" in str((instruction.attributes or {}).get("callee", ""))
        ]
        print(f"   region calls in emission order: {region_calls}")
    return 2 if duplicate_functions else 0


if __name__ == "__main__":
    raise SystemExit(main())
