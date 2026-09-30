"""A record parameter returned inside a tuple from a guarded early return.

Repro (seconds).  A function taking a ``program_abi`` record parameter
with scalar fields, a guarded early ``return m, dt`` and a terminal
``return m, dt * 0.5``, is rejected by the full-native execution contract
with unaccounted formals::

    N formals but only M named or ABI-accounted; unnamed value ids [a, b]
    with accounting {a: {}, b: {}}

The same shape with ONE return site (``dt_controller.step_with_dt_control_used``:
``return metrics, dt_next, dt_tensor``) lowers, and is lowered here as the
control.  The probe FAILS while the defect is present and passes once the
guarded tuple return lowers to a Ret carrying the record's fields and the
scalar lane.

    python -u tools/compiler_probes/probe_record_in_tuple_return.py

Diagnosis (2026-09-30, read + read-only hooks; no compiler edits)
=================================================================

The record is not the trigger.  Observed lowering matrix on this tree
(same contract, same dataclass, same ``step(m, dt)``):

    guard + ``return m, dt`` / ``return m, dt*0.5``   REJECTED (this probe)
    guard + ``return dt, dt`` / ``return dt*0.5, dt`` REJECTED, same finding
    guard + ``return m`` / ``return m``               lowers, Ret = fields
    one site ``return m, dt*0.5``                     lowers, Ret = [field, lane]
    one site ``n = Metrics(...); return n, dt*0.5``   lowers (call-produced)
    ``if ...: dt = dt*2`` then one ``return m, dt*0.5`` lowers
    one site ``x = (m, dt*0.5); return x``            lowers (named aggregate)
    ``if/else`` assigning ``x = (m, ..)``; ``return x`` REJECTED (structural
                                                      output: the merge is a
                                                      phi, not an aggregate)

So the trigger is a tuple return reached through a control merge, and the
guard form is the compiler's own doing.  Chain, link by link:

1. ``fortran_c_shell._normalize_top_level_guard_returns`` (observed): for a
   compile target whose last statement is a ``Return`` and which has a
   return-only top-level guard, it rewrites every ``return <expr>`` into
   ``__turing_single_exit_result = <expr>`` and appends one
   ``return __turing_single_exit_result``.  ``result_assignment`` assigns
   the WHOLE ``ast.Tuple`` to one name.  The dt controller is never
   rewritten: its function ends in ``while True:`` and its single return
   is an ``ast.Tuple`` literal, which the reducer flattens per slot.

2. Reducer ``reduce_statement``, ``ast.Assign`` of an ``ast.Tuple``
   (observed in code, inferred for this run): each arm binds the name to
   an aggregate node (``producer_kind="aggregate"``, ``aggregate_kind=
   "tuple"``, ``aggregate_leaf_value_ids=(m, dt)`` / ``(m, dt*0.5)``).  No
   region produces such a node; it has no SSA instruction.

3. Reducer ``reduce_statement``, ``ast.Return`` (observed via the book):
   the returned expression is now an ``ast.Name``, so the site posts ONE
   ``return_site_slot`` row (index 0) whose value is the merge node of
   ``__turing_single_exit_result``, and ``return_site_container`` = VALUE.
   The two authored TUPLE sites with two slots each never reach the page.
   The record parameter is NOT a slot value here (it was in the one-site
   control, where ``record_return_layouts`` names it).

4. ``precompile_to_ssa`` conditional lowering (observed): the merge is one
   ``conditional_result`` Phi over the two arm values; each arm value is
   fetched with ``external_value``, which finds no producer and no region
   meta, builds ``SSAValue(id, dtype=None)`` with empty accounting and
   appends it to ``self.arguments`` -- the two aggregate nodes become
   formals.  The Ret carries the Phi result alone.

5. ``fortran_c_shell`` return-surface expansion (observed): the pass that
   publishes ``return total, (hub, angle), valid`` reads
   ``producer_kind``/``aggregate_leaf_value_ids`` off the OUTPUT node.  The
   merge node carries neither (only its two arms do), so no leaf is
   published and no ``record_return_layouts`` entry is written for ``m``.

6. ``ssa_self_check.check_formal_parity`` (observed): the two Phi operands
   sit in ``function.args`` with accounting ``{}``; no channel
   (``parameter_names``, ``storage_formals``, ``closure_formals``,
   ``parameter_member_formals``, ``program_abi_parameter``) can name them
   because they are not parameters at all.  The finding is correct.

The two records that disagree: the reducer's ``return_site_slot`` page
(one VALUE slot at the single exit, the merge node) versus the two arm
aggregates' ``aggregate_leaf_value_ids`` (two lanes each, one of them the
record parameter).  The merge has no lane-wise identity, so nothing
downstream can publish the lanes; the leaves surface as unproduced formals.

Fix at the identity (not implemented here): ``_normalize_top_level_guard_
returns.result_assignment`` must split a tuple return per lane, exactly as
``_normalize_direct_tail_recursion``'s ``ExitReturnRewriter`` already does
in the same file (``tuple_result_arity`` -> one ``__turing_..._{lane}``
name per element, final ``return (name_0, name_1, ...)``).  The final
``Return`` then holds an ``ast.Tuple`` again, the reducer posts one
``return_site_slot`` per lane with the record parameter as a slot value,
each lane merges as its own ``conditional_result`` Phi, and the record
lane expands through the existing ``record_return_layouts`` path -- the
path the one-site and record-only controls already take.  No new
machinery; the missing per-lane split is the whole defect.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.compiler.ssa_self_check as ssa_self_check  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import (  # noqa: E402
    FortranEmissionError,
    lower_ast_source_to_ssa,
)

CONTRACTS = REPO / "extraction_contracts"
NAME = "record_in_tuple_return"

HEAD = '''
from dataclasses import dataclass


@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0


def step(m: Metrics, dt: float):
'''
# The defect: a guarded early return and a terminal return, both tuples
# holding the record parameter.
DEFECT = HEAD + '''    if bool(m.hard_failure):
        return m, dt
    m.value = m.value * 0.25
    return m, dt * 0.5
'''
# The dt-controller shape: one return site holding the same tuple.
ONE_SITE = HEAD + '''    m.value = m.value * 0.25
    return m, dt * 0.5
'''

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def contract():
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Metrics": {
                "identity": f"{NAME}.Metrics",
                "fields": {
                    "hard_failure": {"storage": "scalar", "dtype": "bool",
                                     "mutable": True},
                    "value": {"storage": "scalar", "dtype": "float64",
                              "mutable": True},
                },
            }},
            "bindings": [
                {"function": "step", "parameter": "m", "record": "Metrics"},
            ],
            "values": [
                {"function": "step", "parameter": "dt",
                 "storage": "scalar", "dtype": "float64", "rank": 0,
                 "python_type": "builtins.float"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


class ParityHook:
    """Read-only: keep the module the full-native contract judged."""

    def __init__(self) -> None:
        self.module = None
        self.findings = ()
        self._original = ssa_self_check.check_formal_parity

    def __enter__(self):
        def hooked(module):
            findings = self._original(module)
            self.module = module
            self.findings = tuple(findings)
            return findings
        ssa_self_check.check_formal_parity = hooked
        return self

    def __exit__(self, *_exc):
        ssa_self_check.check_formal_parity = self._original
        return False


def lower(source: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            source, "step", name=NAME, python_bindings={},
            extraction_contract=contract(), runtime_closure_only=True,
        )


def step_function(module):
    for name, function in module.functions.items():
        if str(name).endswith("__step"):
            return function
    return None


def terminator(function):
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.op == "Ret":
                return instruction
    return None


def defined_ids(function):
    return {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if getattr(instruction, "res", None) is not None
    }


def main() -> int:
    # ---- the control: one return site, record parameter in the tuple -----
    module, _outputs, _exports = lower(ONE_SITE)
    function = step_function(module)
    check("one-site control lowers", function is not None)
    if function is not None:
        layouts = dict(function.metadata.get("record_return_layouts", ()))
        named = dict(function.metadata.get("parameter_names") or ())
        ret = terminator(function)
        record_id = named.get("m")
        check("one-site control: the record parameter is a return slot "
              "with a field layout",
              record_id is not None and int(record_id) in layouts)
        check("one-site control: Ret publishes the record field and the "
              "scalar lane", ret is not None and len(ret.args) == 2)

    # ---- the defect: guarded early return, same tuple --------------------
    print("-- guarded tuple return --")
    with ParityHook() as hook:
        try:
            module, _outputs, _exports = lower(DEFECT)
        except FortranEmissionError as exc:
            module = None
            message = str(exc)
            print(message[:700])
    if module is not None:
        function = step_function(module)
        layouts = dict(function.metadata.get("record_return_layouts", ()))
        named = dict(function.metadata.get("parameter_names") or ())
        ret = terminator(function)
        record_id = named.get("m")
        check("guarded tuple return lowers", True)
        check("guarded: the record parameter is a return slot with a field "
              "layout", record_id is not None and int(record_id) in layouts)
        check("guarded: Ret publishes both record fields and the scalar lane",
              ret is not None and len(ret.args) == 3)
        return 1 if failures else 0

    # Rejected: fail for the right reason, and say what the reason is.
    check("rejected by the full-native contract for unaccounted formals",
          "unaccounted_formals=({" in message)
    function = step_function(hook.module) if hook.module is not None else None
    check("the parity hook saw the judged module", function is not None)
    if function is None:
        return 1
    parity = [f for f in hook.findings if f.check == "formal_parity"]
    check("exactly one formal_parity finding, on step", len(parity) == 1)
    accounted_free = [
        int(argument.id) for argument in function.args
        if not (getattr(argument, "accounting", None) or {})
    ]
    ret = terminator(function)
    phi = None
    if ret is not None and len(ret.args) == 1:
        sole = int(ret.args[0].id)
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op == "Phi"
                    and int(instruction.res.id) == sole
                    and (instruction.attributes or {}).get("binding")
                    == "conditional_result"
                ):
                    phi = instruction
    check("Ret carries one value: the single-exit conditional_result Phi",
          phi is not None)
    if phi is not None:
        arm_ids = sorted(int(argument.id) for argument in phi.args)
        check("the unaccounted formals ARE the Phi's two arm values",
              sorted(accounted_free) == arm_ids)
        check("neither arm value is produced by any instruction "
              "(the tuple aggregates have no SSA producer)",
              not (set(arm_ids) & defined_ids(function)))
    check("no record_return_layouts entry names the record parameter "
          "(the one-site control had one)",
          not function.metadata.get("record_return_layouts"))
    single_exit = [
        name for name, _value in (function.metadata.get("value_names") or ())
        if str(name).startswith("__turing_single_exit_result")
    ]
    check("the single-exit rewrite ran (its result name is a value name)",
          bool(single_exit))
    # Every check above passing means the defect is exactly as diagnosed;
    # the probe still fails, because valid Python did not compile.
    failures.append(
        "DEFECT PRESENT: guarded tuple return with a record parameter is "
        "rejected (single-exit rewrite assigns the whole tuple to one name; "
        "the merge has no per-lane identity; the arm aggregates surface as "
        "unaccounted formals)"
    )
    for item in failures:
        print("FAIL " + item)
    return 1


if __name__ == "__main__":
    sys.exit(main())
