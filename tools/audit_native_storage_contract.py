"""Report unresolved native array extents with their SSA definitions and uses.

This diagnostic is intentionally backend-adjacent rather than a one-off smoke
test: it identifies the exact producer, formal position, and call bindings for
every array base whose physical capacity is not statically known.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.ssa_storage_requirements import module_storage_requirements
from src.compiler.vehicle_python_compilation import lower_balloon_tire_python_ssa


def main() -> None:
    lowered = lower_balloon_tire_python_ssa(batch_size=8)
    emitted = emit_module(
        lowered.module,
        name="balloon_tire_native",
        outputs=lowered.outputs,
    )
    storage_requirements = module_storage_requirements(lowered.module)
    unresolved = sum(
        len(subroutine.dynamic_array_dimensions)
        for subroutine in emitted.subroutines
    )
    print(
        f"complete={emitted.complete} shortfalls={len(emitted.shortfalls)} "
        f"dynamic_arrays={unresolved}"
    )
    for subroutine in emitted.subroutines:
        if not subroutine.dynamic_array_dimensions:
            continue
        function = lowered.module.functions[subroutine.name]
        requirements = storage_requirements[subroutine.name]
        print(f"\n{subroutine.name}")
        for value_id, dimensions in subroutine.dynamic_array_dimensions:
            formal = next(
                (index for index, value in enumerate(function.args)
                 if int(value.id) == int(value_id)),
                None,
            )
            definition = None
            uses: list[str] = []
            for block_name, block in function.blocks.items():
                for instruction in block.instrs:
                    if instruction.res is not None and int(instruction.res.id) == int(value_id):
                        definition = (
                            str(block_name),
                            str(instruction.op),
                            tuple(int(arg.id) for arg in instruction.args),
                            dict(getattr(instruction, "attributes", {}) or {}),
                        )
                    if any(int(arg.id) == int(value_id) for arg in instruction.args):
                        uses.append(
                            f"{block_name}:{instruction.op}#"
                            f"{getattr(instruction.res, 'id', '-')}"
                        )
            print(
                f"  %{value_id} dimensions={dimensions} formal={formal} "
                f"requirement={requirements.get(int(value_id))!r}"
            )
            print(f"    definition={definition!r}")
            print(f"    uses={uses!r}")
            if formal is not None:
                for caller_name, caller in lowered.module.functions.items():
                    caller_requirements = storage_requirements[caller_name]
                    for caller_block_name, caller_block in caller.blocks.items():
                        for instruction in caller_block.instrs:
                            if (
                                str(instruction.op).lower() == "call"
                                and str(instruction.attributes.get("callee") or "")
                                == subroutine.name
                                and formal < len(instruction.args)
                            ):
                                actual = instruction.args[formal]
                                print(
                                    f"    caller={caller_name}:{caller_block_name} "
                                    f"actual=%{actual.id} "
                                    f"requirement={caller_requirements.get(int(actual.id))!r}"
                                )


if __name__ == "__main__":
    main()
