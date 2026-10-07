"""Every operand use in a lowered module is defined, and its definition
dominates it; every Phi incoming is defined on its predecessor's edge.

The two checks the compiler already owns, run over a WHOLE lowered module:

* ``fortran_c_shell._undefined_repository_ssa_operands`` -- an operand with no
  definition in its function (the full-native gate's inventory);
* ``ssa_self_check.check_definition_dominance`` -- an operand whose definitions
  cannot execute before the read, Phi operands judged on their incoming edge;

plus one the gate does not make: an ``initial_value_id`` (the version a Phi
names as standing before its merge) that names a value its function does not
define.  A stale one is a use waiting for a repair that cannot be made
(``repair_non_dominating_record_phi_uses`` has nothing to substitute).

Use as a library:

    from probe_module_dominance import module_dominance_violations
    violations = module_dominance_violations(system.module)   # {check: [...]}

or run it on the N-piece orbital dt system (cached pieces; never rebuilt):

    python -u tools/compiler_probes/probe_module_dominance.py [N]    # default 2

which lowers N pieces in link mode as ``lowered_system`` does, then checks the
module it returns (after emission).  Exit status is the violation count
(0 = clean).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT.parent / "engine_toy", ROOT / "examples", ROOT):
    sys.path.insert(0, str(_entry))


def stale_initial_values(module) -> list[dict]:
    """Instructions whose ``initial_value_id`` names an id the function does
    not define (not a formal, not any instruction's result)."""
    findings = []
    for name, function in module.functions.items():
        defined = {int(value.id) for value in function.args}
        defined.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attributes = instruction.attributes or {}
                initial = attributes.get("initial_value_id")
                if initial is not None and int(initial) not in defined:
                    findings.append({
                        "function": str(name), "block": str(block_name),
                        "index": int(index), "op": str(instruction.op),
                        "result": (
                            None if instruction.res is None
                            else int(instruction.res.id)
                        ),
                        "initial_value_id": int(initial),
                        "record_field": attributes.get("record_field"),
                    })
    return findings


def module_dominance_violations(module) -> dict[str, list]:
    """``{check: [violation, ...]}`` over every function of ``module``."""
    from src.compiler.fortran_c_shell import _undefined_repository_ssa_operands
    from src.compiler.ssa_self_check import check_definition_dominance

    return {
        "undefined_operand": [
            {key: item[key] for key in (
                "function", "block", "operation", "value_id", "callee",
            )}
            for item in _undefined_repository_ssa_operands(module)
        ],
        "definition_does_not_dominate_use": [
            str(finding) for finding in check_definition_dominance(module)
        ],
        "stale_initial_value_id": stale_initial_values(module),
    }


def main(argv: list[str]) -> int:
    pieces = int(argv[1]) if len(argv) > 1 else 2
    import importlib

    import src.compiler.fortran_c_shell as fcs  # noqa: F401  bind compiler to ROOT first
    import llvm_dt_system as lds
    probe = importlib.import_module("probe_orbital_craft_native_dt")
    from orbital_craft_machine import orbital_craft
    from orbital_jumper import ORBITAL_CHANNEL_NAMES

    paths = [path for _n, path in probe.cached_piece_paths(
        orbital_craft(), 1, 1, True)][:pieces]
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        print(f"MISSING piece(s) {missing}; never rebuilt here")
        return 2
    import tempfile

    system = lds.lowered_system(
        paths, directory=Path(tempfile.gettempdir()) / "module_dominance",
        piece_mode="link", channel_names=ORBITAL_CHANNEL_NAMES,
        progress=lambda message: None,
    )
    violations = module_dominance_violations(system.module)
    total = 0
    for check, items in violations.items():
        print(f"{check}: {len(items)}")
        for item in items:
            print(f"  {item}")
        total += len(items)
    print(f"{len(system.module.functions)} functions, {total} violation(s)")
    return total


if __name__ == "__main__":
    sys.exit(main(sys.argv))
