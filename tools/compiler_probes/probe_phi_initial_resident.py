"""A Phi's ``initial_value_id`` names a value its function defines.

The graph names the version that stood before a merge or a loop by its
planning spelling (a graph id, a planning alias of a region output, a
literal's graph node); the lowering holds that version as an SSA value whose id
is not always the spelling.  ``precompile_to_ssa`` wrote the spelling into the
Phi's ``initial_value_id`` at every carried-Phi writer (``conditional_carried``,
``loop_carried``, ``loop_result_port``, ``loop_latch_carried``,
``loop_continue_carried``), so the id named a value the function does not define
-- a use waiting for a repair that cannot be made (N=2 orbital dt system:
``run_superstep``, ``step_with_dt_control_used``, ``_apply_energy_sidechain``,
``exchange_time_bound``).

    python -u tools/compiler_probes/probe_phi_initial_resident.py

Part A (hand-built IR): the whole-program alias settlement
(``_apply_concorded_function_aliases``) rewrites a Phi's operands to the
planning alias's resident and left ``initial_value_id`` on the retired
spelling (N=2: the conditional-carried Phis of ``step_with_dt_control_used``
and ``_apply_energy_sidechain``); the initial is now a use one past the
operands, rebound through the same ``alias_application_concordance`` row.

Part C: ``_apply_energy_sidechain`` lowered twice (its specialized copy:
the planner's initial spelling was never produced; the entered version is the
non-writing arm's).

Part B, two real lowerings through ``lower_ast_source_to_ssa`` under the program
extraction contract (the audit's ``oscillator``: nested loops, a conditional
merge in a loop, a Heron loop, a ``for range`` series; and ``mapping``: a
``for name, limit in channels.items()`` loop).  Every Phi's ``initial_value_id``
must be a formal or an instruction result of its function, and is read back
from one ``phi_initial_binding`` row ``(function scope, Phi result)`` ->
``(spelled id, resident id)`` DERIVED from the resident's and the Phi's
``ssa_value`` cells; the spelling stays on the Phi as
``initial_spelled_value_id``.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def audit(case: str) -> None:
    import audit_identity_concordance as audit_cases
    from probe_module_dominance import module_dominance_violations
    from src.compiler.ssa_record_return_state import function_scope_of

    module = audit_cases.CASES[case]()
    stale = module_dominance_violations(module)["stale_initial_value_id"]
    for item in stale:
        print(f"  stale: {item}")
    check(f"{case}: no initial_value_id names an undefined id", not stale)

    try:
        from src.compiler.concordance_declarations import PHI_INITIAL_BINDING
    except ImportError:
        check(f"{case}: the phi_initial_binding page is declared", False)
        return
    book = module.metadata["identity_book"]
    page = book.page(PHI_INITIAL_BINDING)
    carried = unbound = unsourced = disagree = respelled = 0
    for function in module.functions.values():
        scope = function_scope_of(function)
        for block in function.blocks.values():
            for instruction in block.instrs:
                attributes = instruction.attributes or {}
                if (
                    instruction.op != "Phi"
                    or "initial_spelled_value_id" not in attributes
                ):
                    continue
                carried += 1
                row = (scope, int(instruction.res.id))
                fact = page.latest(row)
                if fact is None:
                    unbound += 1
                    continue
                spelled, resident = fact
                if int(attributes["initial_value_id"]) != int(resident):
                    disagree += 1
                if int(attributes["initial_spelled_value_id"]) != int(spelled):
                    disagree += 1
                if int(spelled) != int(resident):
                    respelled += 1
                ref = book.latest_ref(PHI_INITIAL_BINDING, row)
                if not book.edges_into(ref):
                    unsourced += 1
    print(f"  {case}: {carried} carried Phis, {respelled} whose resident is "
          f"not the spelling")
    check(f"{case}: every carried Phi has a phi_initial_binding row",
          carried > 0 and unbound == 0)
    check(f"{case}: initial_value_id / initial_spelled_value_id are read "
          "from the row", disagree == 0)
    check(f"{case}: every row has an inbound edge", unsourced == 0)


def alias_application() -> None:
    """Part A (hand-built IR): the planning alias that retires a Phi's
    initial spelling is applied to ``initial_value_id`` through the same
    ``alias_application_concordance`` page that rebinds the operands."""

    from src.compiler.fortran_c_shell import _apply_concorded_function_aliases
    from src.compiler.identity_concordance import (
        begin_identity_book, end_identity_book,
    )
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    resident = SSAValue(10, dtype="float64")
    other = SSAValue(11, dtype="float64")
    spelled = SSAValue(20, dtype="float64")
    merged = SSAValue(30, dtype="float64")
    phi = Instr("Phi", [spelled, other], merged, attributes={
        "binding": "conditional_carried", "initial_value_id": 20,
        "incoming_blocks": ("left", "right"),
    })
    unrelated = Instr("Phi", [other, other], SSAValue(31, dtype="float64"),
                      attributes={
        "binding": "conditional_carried", "initial_value_id": 11,
        "incoming_blocks": ("left", "right"),
    })
    function = Function(
        "step", [resident, other],
        {
            "entry": BasicBlock("entry", [
                Instr("CondBr", [other], None),
            ], successors=["left", "right"]),
            "left": BasicBlock("left", [Instr("Br", [], None)],
                               successors=["function_exit"]),
            "right": BasicBlock("right", [Instr("Br", [], None)],
                                successors=["function_exit"]),
            "function_exit": BasicBlock("function_exit", [
                phi, unrelated, Instr("Ret", [merged], None),
            ]),
        },
    )
    book, token = begin_identity_book()
    try:
        receipts = _apply_concorded_function_aliases(function, {20: 10})
        page = book.page("alias_application_concordance")
        row = ("step", "function_exit", 0, 2)
        print(f"  alias_application row {row} -> {page.latest(row)}")
        check("the operand follows the alias",
              [int(a.id) for a in phi.args] == [10, 11])
        check("the Phi's initial_value_id follows the alias",
              phi.attributes["initial_value_id"] == 10)
        check("it is read from one alias_application row, one past the operands",
              page.latest(row) == (20, 10, 10, "initial_names_resident",
                                   "function_exit"))
        check("an unrelated Phi's initial is untouched",
              unrelated.attributes["initial_value_id"] == 11)
        check("the receipts name the initial", any(
            receipt[-2] == "initial_names_resident" for receipt in receipts))
    finally:
        end_identity_book(token)


def sidechain_twice() -> None:
    """Part C: the real ``_apply_energy_sidechain`` called twice, so its
    specialized copy is lowered again.  The planner's conditional alias there
    is (65, 55, 64, 66): the pre-branch spelling 64 was never produced (the
    builder hands back a stand-in, ``NO_PRODUCER_AT_USE``) while the arm that
    did not rebind the name takes the first conditional's merge, 55.  The
    entered version is that arm's value."""

    import inspect

    from probe_module_dominance import module_dominance_violations
    from src.common.dt_system.dt_controller import _apply_energy_sidechain
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = REPO / "extraction_contracts" / "program_extraction.yaml"
    root = (
        "def root(dt, metrics, targets):\n"
        "    dt_tensor = AbstractTensor.tensor(dt)\n"
        "    dt_next = dt_tensor * 0.5\n"
        "    a = _apply_energy_sidechain(dt_next, dt_tensor, metrics, targets)\n"
        "    return _apply_energy_sidechain(a, dt_tensor, metrics, targets)\n"
    )
    base = ExtractionContract(contract).program_abi.receipt()
    policy = ExtractionContract(contract).with_program_abi({
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
        "values": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "\n\n".join((inspect.getsource(_apply_energy_sidechain), root)),
        "root", name="side",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=policy, progress=lambda message: None,
    )
    stale = module_dominance_violations(module)["stale_initial_value_id"]
    for item in stale:
        print(f"  stale: {item}")
    check("sidechain twice: no initial_value_id names an undefined id",
          not stale)


def main() -> int:
    alias_application()
    sidechain_twice()
    for case in ("oscillator", "mapping"):
        audit(case)
    print("FAILED: " + "; ".join(failures) if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
