"""An ``if`` arm whose only statement is a scalar record-field write.

Modelled on ``probe_branch_written_field.py`` (same record contract, same
``program_abi`` binding, same lowering entry) but the arm under test holds
NOTHING except the write::

    def step(m: Metrics, rejected: bool) -> Metrics:
        if bool(m.hard_failure):
            return m
        if rejected:
            m.hard_failure = True
        return m

Python executes the Store only when ``rejected`` is true.  The compiled
program must agree: the Store of ``hard_failure`` has to sit behind a
CondBr on the ``rejected`` formal.  The probe checks that structurally --
it removes the true edge of the CondBr whose predicate is the ``rejected``
formal and asserts the Store's block is then unreachable from entry.  A
Store still reachable executes on a path where ``rejected`` was never
tested true: the arm was dropped.

    python -u tools/compiler_probes/probe_scalar_write_only_arm.py

Diagnosis (read-only hooks, no compiler edits; every link below was
observed with the code's own traces ``TURING_DEBUG_CONTROL_OVERLAY`` and
``TURING_DEBUG_SEQUENCE_MUTATION`` and with a hook on the control
builder's ``lower``):

1. Reducer (``src/common/tensors/topological_reducer.py``, the ``ast.If``
   arm of ``reduce_statement``): correct.  The SetAttr posts a WRITTEN
   ``reducer_field_state`` cell and stamps it on the node as
   ``field_state_cell``; the ``if rejected`` merge posts a MERGED cell and
   a Phi carrying ``source_conditional_id`` (the inner ``if``),
   ``field_state_arms = (written cell, pre-branch cell, test cell)`` and
   ``field_state_cell = merged cell``.  The enclosing ``if bool(...)``
   merge then names that inner MERGED cell as its orelse arm.

2. Planner ``glsl_deployment_strategy._ordinary_conditional_control_programs``:
   THE DROP.  Retention is decided by ``body_regions`` / ``else_regions``
   / return or loop controls / callsites / ``result_aliases`` /
   ``predicate_regions and has_structural_branch_effect``.  A write-only
   arm has no numerical region and ``rejected`` is a bare parameter with
   no predicate region, so every term is false and the loop ``continue``s
   (``DEBUGIFMISS`` fires for ``if rejected: m.hard_failure = True``).
   The reducer's Phi for this conditional -- the record that says "this
   conditional owns a field-state merge" -- is scanned only AFTER that
   ``continue``, so it never counts as a reason to keep the conditional.
   No ConditionalBlock is built for the inner ``if``.
   (``_is_dispatch_metadata_node_impl`` is not involved: the SetAttr is
   admitted separately as a ScalarFieldWriteBlock, see 3.)

3. ``fortran_c_shell._class_surface_ssa_program``: correct.  The SetAttr
   is admitted as ``ScalarFieldWriteBlock(field_state_cell=<WRITTEN cell>)``.

4. ``fortran_c_shell._install_lexical_sequence_mutations``: WHERE THE
   STORE LANDS OUTSIDE ITS GUARD.  ``guarded`` for the write is
   ``((inner if, body), (outer if, orelse))``, innermost first.
   ``insert_in_conditional(root, inner, body)`` finds no block (step 2),
   ``insert_in_conditional(root, outer, orelse)`` succeeds, the loop
   ``break``s with ``inserted=True``, and the synthesis branch that
   rebuilds a dropped guard from the source control record never runs
   (``DEBUGMUTATION-GUARD`` does not fire).  The write is appended to the
   outer conditional's orelse SequenceBlock: control-program path
   ``root.sequence[1].orelse``.  In the flat shape (no enclosing
   conditional) ``guarded`` has one entry, insertion fails, and the same
   pass synthesizes a ConditionalBlock with ``carried_field_cells=()`` --
   the Store is guarded there, but the MERGED cell never gets a version.

5. ``precompile_to_ssa._ControlSSABuilder``: faithful to what it is given.
   ``lower(ScalarFieldWriteBlock)`` emits the Store into the current block
   (``if_false`` of the OUTER conditional) and publishes the WRITTEN
   cell's version.  ``lower_conditional`` for the outer ``if`` then asks
   ``_carried_field_arm`` for its orelse arm, the inner MERGED cell; no
   ConditionalBlock ever published a version there, so it records
   ``ARM_VERSION_MISSING`` and the lowering raises
   ``carried-field-arm-missing`` at ``root.sequence[1].orelse``.  Before
   the field-state pages existed this same placement produced a silent
   unguarded Store; today it is a hard error with the same cause.

The two records that disagree: the reducer's MERGED cell / Phi, which
says the write is the body arm of ``if rejected`` and that the outer
orelse ends at that merge; and the control program, which holds the write
directly under the outer conditional's orelse with no inner conditional
at all.

Fix at the identity (not implemented here): the planner's retention
predicate in ``_ordinary_conditional_control_programs`` must consume the
reducer's field-state merge -- a Phi with ``source_conditional_id`` equal
to this conditional and ``field_state_arms`` present IS a reason to build
its ConditionalBlock, carrying ``(written cell, pre-branch cell, merged
cell)`` as ``carried_field_cells``.  The control builder already consumes
those cells in ``lower_conditional`` / ``_carried_field_arm``; it is not
the place to fix.  ``_install_lexical_sequence_mutations`` would then find
the inner block and place the write in its body.  (Patching the placement
pass to synthesize the inner guard instead would keep the Store guarded
but still leave the MERGED cell without a version, as the flat shape
shows.)
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler import precompile_to_ssa as _precompile  # noqa: E402
from src.compiler.control_source import ScalarFieldWriteBlock  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
NAME = "scalar_write_only_arm"
FIELD = "hard_failure"
PREDICATE_PARAMETER = "rejected"

SOURCE = '''
from dataclasses import dataclass


@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0


def step(m: Metrics, rejected: bool) -> Metrics:
    if bool(m.hard_failure):
        return m
    if rejected:
        m.hard_failure = True
    return m
'''


def contract():
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Metrics": {
                "identity": f"{NAME}.Metrics",
                "fields": {
                    FIELD: {"storage": "scalar", "dtype": "bool",
                            "mutable": True},
                    "value": {"storage": "scalar", "dtype": "float64",
                              "mutable": True},
                },
            }},
            "bindings": [
                {"function": "step", "parameter": "m", "record": "Metrics"},
            ],
            "values": [
                {"function": "step", "parameter": PREDICATE_PARAMETER,
                 "storage": "scalar", "dtype": "bool", "rank": 0,
                 "python_type": "builtins.bool"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


# Read-only observation of where the control builder lowers the field
# write: the control-program path and the SSA block current at that
# moment.  The wrapper calls straight through; nothing is patched.
write_placements: list[tuple[str, str]] = []
_original_lower = _precompile._ControlSSABuilder.lower


def _observing_lower(self, block, *, path="root"):
    if isinstance(block, ScalarFieldWriteBlock):
        write_placements.append((path, self.current.name))
    return _original_lower(self, block, path=path)


_precompile._ControlSSABuilder.lower = _observing_lower


def lower():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            SOURCE, "step", name=NAME, python_bindings={},
            extraction_contract=contract(), runtime_closure_only=True,
        )


failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def formal_for_field(function, field: str):
    """The formal declared as the record field ``field`` (by ABI attributes)."""
    for value in function.args:
        accounting = value.accounting or {}
        if accounting.get("program_abi_field") == field:
            return value
    return None


def predicate_formal(function):
    """The scalar value formal that is not a record field projection."""
    for value in function.args:
        accounting = value.accounting or {}
        if (
            accounting.get("program_abi_storage") == "scalar"
            and "program_abi_field" not in accounting
            and "program_abi_record" not in accounting
        ):
            return value
    return None


def reachable(function, *, dropped_edges: set[tuple[str, str]]) -> set[str]:
    seen: set[str] = set()
    frontier = ["entry"]
    while frontier:
        name = frontier.pop()
        if name in seen or name not in function.blocks:
            continue
        seen.add(name)
        for successor in function.blocks[name].successors:
            if (name, successor) not in dropped_edges:
                frontier.append(successor)
    return seen


def main() -> int:
    try:
        module, _outputs, _exports = lower()
    except Exception as exc:  # noqa: BLE001 -- the defect surfaces here today
        print(f"lowering raised {type(exc).__name__}: {exc}")
        for path, block_name in write_placements:
            print(
                f"  the {FIELD!r} write was lowered at control path {path!r} "
                f"into SSA block {block_name!r}"
            )
        check(
            f"the program lowers (the write-only 'if {PREDICATE_PARAMETER}' "
            "arm must survive as a conditional)",
            False,
        )
        return 1

    check("the program lowers", True)
    step = next(iter(module.functions.values()))
    field_formal = formal_for_field(step, FIELD)
    predicate = predicate_formal(step)
    check(f"the {FIELD!r} field formal is declared", field_formal is not None)
    check(f"the {PREDICATE_PARAMETER!r} formal is declared", predicate is not None)
    if field_formal is None or predicate is None:
        return 1

    store_blocks = []
    guard_true_edges: set[tuple[str, str]] = set()
    for block in step.blocks.values():
        for instruction in block.instrs:
            if (
                instruction.op == "Store"
                and len(instruction.args) == 2
                and int(instruction.args[1].id) == int(field_formal.id)
            ):
                store_blocks.append(block.name)
            if (
                instruction.op == "CondBr"
                and instruction.args
                and int(instruction.args[0].id) == int(predicate.id)
            ):
                guard_true_edges.add(
                    (block.name, instruction.attributes["true_target"])
                )
    print(f"Store(s) of {FIELD!r} in block(s): {store_blocks}")
    print(f"CondBr(s) on {PREDICATE_PARAMETER!r}: {sorted(guard_true_edges)}")
    for path, block_name in write_placements:
        print(f"  write lowered at control path {path!r} into block {block_name!r}")
    check(f"exactly one Store of {FIELD!r}", len(store_blocks) == 1)
    check(f"a CondBr tests the {PREDICATE_PARAMETER!r} formal", bool(guard_true_edges))
    if store_blocks and guard_true_edges:
        still_reachable = reachable(step, dropped_edges=guard_true_edges)
        check(
            f"the Store of {FIELD!r} is unreachable once the true edge of the "
            f"CondBr on {PREDICATE_PARAMETER!r} is removed (it executes only "
            f"when {PREDICATE_PARAMETER} is true, as in Python)",
            store_blocks[0] not in still_reachable,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
