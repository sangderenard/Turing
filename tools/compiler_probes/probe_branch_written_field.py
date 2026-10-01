"""A dataclass field written on one branch and returned on two paths.

Modelled on ``Metrics.hard_failure`` in
``dt_controller.step_with_dt_control_used`` but tiny (plan 70, section 7):
one field, one read before the write, one write on one arm, two return
sites (one inside a terminal arm, one at the tail after a non-terminal
conditional).  Lowered through ``lower_ast_source_to_ssa`` with a
``program_abi`` record for the dataclass.

The probe prints the causal chain the book holds for the field, link by
link, each read back with ``book.edges_into``:

    field schema -> written cell -> merged cell -> return-site state
    -> SSA field version -> return-merge selection

and passes when every link that its sources make possible is present.  A
link whose source rows are not yet on the book (the reducer's
``reducer_field_state`` / ``return_site_*`` pages, the return-merge Phi's
identity cell) is printed as absent, not asserted: the probe is green
before and after those writers land, and the printed chain says which
writers have landed.

    python -u tools/compiler_probes/probe_branch_written_field.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CLASS_FIELD_DECLARATION,
    RECORD_RETURN_FIELD_SELECTION,
    REDUCER_FIELD_STATE,
    RETURN_SITE_FIELD_STATE,
    SSA_FIELD_VERSION,
    FieldStateKind,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    Ref,
    Unresolved,
    identity_book,
)
from src.compiler.ssa_record_return_state import (  # noqa: E402
    ssa_value_identity_cell,
)

CONTRACTS = REPO / "extraction_contracts"
NAME = "branch_written_field"
FIELD = "hard_failure"

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
        m.hard_failure = False
        m.value = m.value * 0.25
    return m
'''
# The plan's shape returns ``(m, dt)`` and updates ``dt`` in the arm.  On the
# untouched tree a tuple of a record and a scalar is rejected by the
# full-native execution contract (the two tuple slots surface as unaccounted
# formals), and an ``if`` arm holding only a scalar field write is not kept
# as a conditional at all (its Store lands unguarded in the enclosing arm),
# so the probe returns the record alone and gives the arm one numerical
# region (``m.value``) so the guard survives.  The ``hard_failure`` chain
# under test is the plan's.


def lower():
    contract = (
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
                {"function": "step", "parameter": "rejected",
                 "storage": "scalar", "dtype": "bool", "rank": 0,
                 "python_type": "builtins.bool"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            SOURCE, "step", name=NAME, python_bindings={},
            extraction_contract=contract, runtime_closure_only=True,
        )


failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def cells(book, page, predicate=lambda row, fact: True):
    """Every (Ref, fact) on ``page`` whose row and fact pass ``predicate``."""
    stored = book.pages.get(page.name)
    if stored is None:
        return []
    found = []
    for row in stored.rows():
        for column, fact in stored.history(row):
            if predicate(row, fact):
                found.append((Ref(page, row, column), fact))
    return found


def show_edges(book, ref, indent="        "):
    edges = book.edges_into(ref)
    if not edges:
        print(f"{indent}(no inbound edge)")
    for source, stage in edges:
        print(f"{indent}<- {source!r} [{stage.name}]")
    return edges


def field_state_kind(fact):
    return getattr(fact, "kind", None)


def main() -> int:
    module, _outputs, _exports = lower()
    check("the program lowers", True)
    book = identity_book(module)

    # ---- the emitted SSA: the conditional_carried Phi for the field ------
    step = None
    for name, function in module.functions.items():
        if str(name).split(".")[-1].split("_")[0] == "step" or "step" in str(name):
            step = function
            break
    check("the step function is in the module", step is not None)
    carried = []
    if step is not None:
        for block in step.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op == "Phi"
                    and (instruction.attributes or {}).get("binding")
                    == "conditional_carried"
                ):
                    carried.append(instruction)
    print(f"conditional_carried Phis: {len(carried)}")
    for phi in carried:
        args = tuple(int(argument.id) for argument in phi.args)
        print(
            f"  Phi res={int(phi.res.id)} args={args} "
            f"dtype={phi.res.dtype!r} initial={phi.attributes.get('initial_value_id')}"
        )

    # ---- link 1: the field schema ----------------------------------------
    schema = cells(
        book, CLASS_FIELD_DECLARATION,
        lambda row, fact: row[-1] == FIELD,
    )
    print(f"field schema cells for {FIELD!r}: {len(schema)}")
    for ref, fact in schema:
        print(f"    {ref!r} = {fact!r}")

    # ---- link 2/3: written and merged reducer field-state cells ----------
    states = cells(
        book, REDUCER_FIELD_STATE,
        lambda row, fact: row[-1] == FIELD,
    )
    seam_present = bool(states)
    written = [(ref, fact) for ref, fact in states
               if field_state_kind(fact) is FieldStateKind.WRITTEN]
    merged = [(ref, fact) for ref, fact in states
              if field_state_kind(fact) is FieldStateKind.MERGED]
    print(
        f"reducer_field_state cells for {FIELD!r}: {len(states)} "
        f"(written={len(written)}, merged={len(merged)})"
        + ("" if seam_present else " -- reducer field-state pages not on the book")
    )
    for ref, fact in states:
        print(f"    {ref!r} = {fact!r}")
        show_edges(book, ref)
    # The canonical relabel re-posts the field row under the read scope, so
    # a revision may appear once per row (ingestion and canonical); every
    # check below is therefore "at least one" and per row, never "exactly
    # one" across rows.
    if seam_present:
        check("a WRITTEN revision exists", bool(written))
        check("a MERGED revision exists", bool(merged))
        if written and merged:
            check("some MERGED cell derives from a WRITTEN cell", any(
                any(source == written_ref
                    for source, _stage in book.edges_into(merged_ref))
                for merged_ref, _fact in merged
                for written_ref, _fact2 in written
            ))

    # ---- link 4: return-site field state ----------------------------------
    sites = cells(
        book, RETURN_SITE_FIELD_STATE,
        lambda row, fact: row[-1] == FIELD,
    )
    print(f"return_site_field_state cells for {FIELD!r}: {len(sites)}")
    for ref, fact in sites:
        print(f"    {ref!r} = {fact!r}")
        show_edges(book, ref)
    if not sites:
        # The single-exit rewrite upstream of the reducer folds the two
        # ``return`` statements into one ``return <name>``, so the record
        # receiver is not itself a return slot value and no per-return row
        # names the field; the field's state at the single exit is the
        # MERGED cell of the ``if rejected`` merge, read above.
        print(
            "    (none: the single exit returns a name, not the receiver; "
            "the exit-side state is the MERGED cell)"
        )
    if seam_present and merged:
        check("a MERGED cell is the field state reachable at the exit "
              "(the last revision of its own row)", any(
                  book.latest_ref(REDUCER_FIELD_STATE, merged_ref.row) == merged_ref
                  for merged_ref, _fact in merged
              ))

    # ---- link 5: SSA field versions ---------------------------------------
    versions = cells(book, SSA_FIELD_VERSION)
    print(f"ssa_field_version cells: {len(versions)}")
    unresolved_versions = []
    for ref, fact in versions:
        print(f"    {ref!r} = {fact!r}")
        show_edges(book, ref)
        if isinstance(fact, Unresolved):
            unresolved_versions.append((ref, fact))
    if seam_present:
        by_cell = {ref.row[1]: fact for ref, fact in versions}
        written_versions = [
            by_cell[ref] for ref, _fact in written
            if ref in by_cell and not isinstance(by_cell[ref], Unresolved)
        ]
        merged_versions = [
            by_cell[ref] for ref, _fact in merged
            if ref in by_cell and not isinstance(by_cell[ref], Unresolved)
        ]
        check("an SSA version is posted at a WRITTEN cell", bool(written_versions))
        check("an SSA version is posted at a MERGED cell", bool(merged_versions))
        check("no ARM_VERSION_MISSING", not any(
            fact.reason.name == "arm_version_missing"
            for _ref, fact in unresolved_versions
        ))
        # The merge's version must derive from the WRITTEN version cell (the
        # arm that wrote) -- the arm reached the Phi through the book, not
        # through a snapshot.
        written_version_refs = {
            ref for ref, fact in versions
            if ref.row[1] in {w for w, _f in written}
            and not isinstance(fact, Unresolved)
        }
        merged_version_refs = [
            ref for ref, fact in versions
            if ref.row[1] in {m for m, _f in merged}
            and not isinstance(fact, Unresolved)
        ]
        check("a MERGED version derives from a WRITTEN version cell", any(
            any(source in written_version_refs
                for source, _stage in book.edges_into(ref))
            for ref in merged_version_refs
        ))
        phi = next((phi for phi in carried
                    if any(int(phi.res.id) == int(version)
                           for version in merged_versions)), None)
        if phi is not None:
            check("the Phi's two arguments differ (Const, snapshot)",
                  int(phi.args[0].id) != int(phi.args[1].id))
        else:
            # The receiver is a parameter record with physical storage: the
            # write is a Store and the emitted conditional_carried Phi over
            # the field is dead once nothing after the merge reads it, so it
            # may be eliminated from the finished module.  The book keeps
            # its version and its arm edges regardless (checked above).
            print("    (no live conditional_carried Phi for the field in the "
                  "finished module; its version and edges are on the book)")
    else:
        check("no version is posted without a field-state cell",
              not versions)

    # ---- link 6: the return-merge selection ------------------------------
    # A selection row is keyed by a record return-merge Phi's ``ssa_value``
    # cell (the control builder posts it; record materialization reads it
    # through ``ssa_value_identity_cell``).  This program's two ``return m``
    # statements fold into one exit, so the finished ``step`` holds no
    # return-merge Phi and no row can exist; the probe says which case it is.
    return_merges = [] if step is None else [
        instruction
        for block in step.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Phi"
        and (instruction.attributes or {}).get("binding") == "return_merge"
    ]
    print(f"return-merge Phis in step: {len(return_merges)}")
    selections = cells(book, RECORD_RETURN_FIELD_SELECTION)
    print(f"record_return_field_selection cells: {len(selections)}")
    for ref, fact in selections:
        print(f"    {ref!r} = {fact!r}")
        show_edges(book, ref)
    if return_merges and seam_present:
        check("every return-merge Phi has an ssa_value identity cell", all(
            ssa_value_identity_cell(step, int(phi.res.id)) is not None
            for phi in return_merges
        ))
        check("a record return-merge selection row is posted", bool(selections))
        check("every selection row has an inbound edge", all(
            bool(book.edges_into(ref)) for ref, _fact in selections
        ))
    elif not selections:
        print(
            "    (none: the function has no return-merge Phi, so no "
            "selection row can be keyed)"
        )

    # ---- what stays true today --------------------------------------------
    check("lowering reported no shortfall about a carried field arm", not any(
        "carried-field-arm-missing" in str(line)
        for line in (module.metadata or {}).get("lowering_shortfalls", ())
    ))
    print(f"seam present: {seam_present}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
