"""The control SSA builder's binding chain, read back from the book.

Plan 80 part B (step 5): one function with a scalar parameter, a name
rebound on one arm of a conditional, and a ``for`` loop carrying one name
with a ``break``.  Lowered through ``lower_ast_source_to_ssa`` like
``probe_branch_written_field.py``; every check below reads the book, never
the builder.

    python -u tools/compiler_probes/probe_control_binding_chain.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CARRIED_PORT_VALUE,
    CARRIED_SNAPSHOT,
    CELL_SET,
    CONTROL_VALUE_BINDING,
    DECLARED_PARAMETER,
    FUNCTION_OUTPUT,
    FUNCTION_PARAMETER,
    LOOP_CARRIED_BINDING,
    LOOP_CARRIED_ENTRY,
    LOOP_RESULT_PORT_BINDING,
    NAME_ARM_VERSION_MISSING,
    PHI_CONDITIONAL,
    PHI_LOOP_HEADER,
    SSA_VALUE,
    BindingKind,
    ControlBinding,
    ParameterDeclaration,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.id_space import MINTED, has_flag  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    EDGE_PAGE,
    MINT_PAGE,
    UNSOURCED_PAGE,
    Ref,
    Unresolved,
    identity_book,
)

CONTRACTS = REPO / "extraction_contracts"
NAME = "control_binding_chain"

SOURCE = '''
def step(k: int, flag: bool) -> int:
    total = k
    if flag:
        total = total + 1
    for i in range(4):
        total = total + i
        if total > 10:
            break
    return total
'''


def lower():
    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {}, "bindings": [],
            "values": [
                {"function": "step", "parameter": "k", "storage": "scalar",
                 "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
                {"function": "step", "parameter": "flag", "storage": "scalar",
                 "dtype": "bool", "rank": 0, "python_type": "builtins.bool"},
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


def rows(book, page, scope):
    stored = book.pages.get(page.name)
    return () if stored is None else stored.scope_rows(scope)


def latest(book, page, row):
    stored = book.pages.get(page.name)
    return None if stored is None else stored.latest(row)


def mint_of(book, ref):
    return book.mint_of(ref)


def operand_cells(book, ref):
    """The cells a mint was made from, expanding a ``cell_set`` operand."""
    mint = mint_of(book, ref)
    if mint is None:
        return ()
    _transform, operands = mint
    found = []
    for operand in operands:
        if operand.page is CELL_SET:
            found.extend(source for source, _stage in book.edges_into(operand))
        else:
            found.append(operand)
    return tuple(found)


def phis(function, binding):
    return [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Phi"
        and (instruction.attributes or {}).get("binding") == binding
    ]


def unsourced_groups(book):
    page = book.pages.get(UNSOURCED_PAGE.name)
    groups: Counter = Counter()
    if page is not None:
        for row in page.rows():
            groups[(row[0], row[2])] += 1
    return groups


def main() -> int:
    module, _outputs, _exports = lower()
    check("the program lowers", True)
    book = identity_book(module)
    step = next(
        (function for name, function in module.functions.items()
         if "step" in str(name) and function.metadata.get("control_ir")),
        None,
    )
    check("the control function is in the module", step is not None)
    if step is None:
        return 1
    scope = str(step.metadata.get("tensor_shape_concordance_scope"))
    print(f"control scope: {scope}")

    # ---- 1. bindings in the function scope -------------------------------
    binding_rows = rows(book, CONTROL_VALUE_BINDING, scope)
    kinds = Counter()
    unresolved = []
    for row in binding_rows:
        for _column, fact in book.pages[CONTROL_VALUE_BINDING.name].history(row):
            if isinstance(fact, ControlBinding):
                kinds[fact.kind.name] += 1
            elif isinstance(fact, Unresolved):
                unresolved.append((row, fact.reason.name))
    print(f"control_value_binding rows: {len(binding_rows)}; kinds: {dict(kinds)}")
    print(f"unresolved bindings: {unresolved}")
    check("bindings were posted for the function", bool(binding_rows))
    # The arm's ``total = total + 1`` is bound after the branch was entered:
    # a resolved revision (a region result or a control expression) stamped
    # after some ``carried_snapshot`` cell.
    snapshot_stamps = [
        book.stamp_of(book.latest_ref(CARRIED_SNAPSHOT, row))
        for row in rows(book, CARRIED_SNAPSHOT, scope)
    ]
    arm_bindings = [
        (row, column, fact)
        for row in binding_rows
        for column, fact in book.pages[CONTROL_VALUE_BINDING.name].history(row)
        if isinstance(fact, ControlBinding)
        and fact.kind in {BindingKind.CONTROL_EXPRESSION, BindingKind.REGION_RESULT}
        and snapshot_stamps
        and book.stamp_of(Ref(CONTROL_VALUE_BINDING, row, column)) > min(snapshot_stamps)
    ]
    print(f"arm bindings after the snapshot: {[(r[1], f.kind.name) for r, _c, f in arm_bindings]}")
    check("an arm binding is stamped inside the arm (after the snapshot)",
          bool(arm_bindings))
    check("no NAME_ARM_VERSION_MISSING", not any(
        reason == NAME_ARM_VERSION_MISSING.name for _row, reason in unresolved
    ))
    check("every binding revision has an inbound edge", all(
        book.edges_into(Ref(CONTROL_VALUE_BINDING, row, column))
        for row in binding_rows
        for column, _fact in book.pages[CONTROL_VALUE_BINDING.name].history(row)
    ))

    # ---- 2. the conditional Phi ------------------------------------------
    snapshots = rows(book, CARRIED_SNAPSHOT, scope)
    print(f"carried_snapshot rows: {len(snapshots)}")
    check("carried_snapshot rows were posted", bool(snapshots))
    conditional = phis(step, "conditional_carried")
    print(f"conditional_carried Phis: {len(conditional)}")
    for phi in conditional:
        ref = book.latest_ref(SSA_VALUE, (scope, int(phi.res.id)))
        mint = None if ref is None else mint_of(book, ref)
        # A join that reuses an arm's graph id is a NOVEL PHI_CONDITIONAL
        # mint; a join under the graph's own merge id is adopted, DERIVED
        # from the same operands.  Either way the operands are on the book.
        if mint is not None:
            cells = operand_cells(book, ref)
        else:
            cells = () if ref is None else tuple(
                source for source, _stage in book.edges_into(ref)
            )
        print(f"  Phi res={int(phi.res.id)} args={[int(a.id) for a in phi.args]} "
              f"mint={None if mint is None else mint[0].name} operands={cells}")
        check("the conditional Phi's row is on the book (NOVEL PHI_CONDITIONAL "
              "or adopted with edges)",
              ref is not None and (
                  (mint is not None and mint[0] is PHI_CONDITIONAL) or bool(cells)
              ))
        check("the Phi's operands include a carried_snapshot cell", any(
            cell.page is CARRIED_SNAPSHOT for cell in cells
        ))
        check("the Phi's two arguments differ",
              int(phi.args[0].id) != int(phi.args[1].id))
    check("a conditional_carried Phi was emitted", bool(conditional))

    # ---- 3. the loop ------------------------------------------------------
    header = phis(step, "loop_carried")
    check("a loop_carried header Phi was emitted", bool(header))
    for phi in header:
        ref = book.latest_ref(SSA_VALUE, (scope, int(phi.res.id)))
        mint = None if ref is None else mint_of(book, ref)
        check("the header Phi is NOVEL(PHI_LOOP_HEADER)",
              mint is not None and mint[0] is PHI_LOOP_HEADER)
    entries = rows(book, LOOP_CARRIED_ENTRY, scope)
    print(f"loop_carried_entry rows: {len(entries)}")
    check("loop_carried_entry rows were posted", bool(entries))
    check("loop_carried_entry derives from loop_carried_binding", all(
        any(source.page is LOOP_CARRIED_BINDING
            for source, _stage in book.edges_into(book.latest_ref(LOOP_CARRIED_ENTRY, row)))
        for row in entries
    ))
    ports = rows(book, CARRIED_PORT_VALUE, scope)
    print(f"carried_port_value rows: {len(ports)}")
    check("carried_port_value rows were posted", bool(ports))
    for row in ports:
        ref = book.latest_ref(CARRIED_PORT_VALUE, row)
        sources = [source for source, _stage in book.edges_into(ref)]
        print(f"  port {row[1]} -> {latest(book, CARRIED_PORT_VALUE, row)!r}; sources={sources}")
        check("the port derives from its loop_result_port_binding cell",
              any(source.page is LOOP_RESULT_PORT_BINDING for source in sources))
        check("the port derives from the exit Phi's ssa_value cell",
              any(source.page is SSA_VALUE for source in sources))

    # ---- 4. finish: outputs and parameters -------------------------------
    outputs = rows(book, FUNCTION_OUTPUT, scope)
    print(f"function_output rows: {[(row, latest(book, FUNCTION_OUTPUT, row)) for row in outputs]}")
    check("function_output slot 0 was posted", any(row[1] == 0 for row in outputs))
    named = tuple(
        (str(fact[0]), int(fact[1]))
        for row in outputs
        for fact in (latest(book, FUNCTION_OUTPUT, row),)
        if fact is not None and fact[0] is not None
    )
    check("metadata named_outputs equals the page-derived tuple",
          tuple(step.metadata.get("named_outputs") or ()) == named)
    parameters = rows(book, FUNCTION_PARAMETER, scope)
    page_parameters = tuple(
        (str(row[1]), int(latest(book, FUNCTION_PARAMETER, row))) for row in parameters
    )
    print(f"function_parameter rows: {page_parameters}")
    check("metadata parameter_names equals the page-derived tuple",
          tuple(step.metadata.get("parameter_names") or ()) == page_parameters)
    k_row = next((row for row in parameters if row[1] == "k"), None)
    check("function_parameter for k exists", k_row is not None)
    if k_row is not None:
        sources = [s for s, _ in book.edges_into(book.latest_ref(FUNCTION_PARAMETER, k_row))]
        used = [
            s for s in sources if s.page is DECLARED_PARAMETER
            and latest(book, DECLARED_PARAMETER, s.row) is ParameterDeclaration.USED
        ]
        check("k derives from declared_parameter USED", bool(used))

    # ---- 5. every MINTED id in the function has a mint record -----------
    mint_page = book.pages.get(MINT_PAGE.name)
    recorded = {row[1] for row in (mint_page.rows() if mint_page else ()) if row[1] is not None}
    ids = {int(v.id) for v in step.args}
    for block in step.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                ids.add(int(instruction.res.id))
    minted = sorted(i for i in ids if has_flag(i, MINTED))
    unrecorded = [i for i in minted if i not in recorded]
    # Ids the control builder minted have an ``ssa_value`` row under the
    # function scope; a MINTED id with no row was re-minted by a later pass
    # (the return-state freshening of steps 6-7) and is printed as residue.
    builder_minted = [
        i for i in minted if book.latest_ref(SSA_VALUE, (scope, i)) is not None
    ]
    builder_unrecorded = [i for i in builder_minted if i not in recorded]
    residue = [i for i in unrecorded if i not in builder_minted]
    definers = {
        int(instruction.res.id): (block.name, str(instruction.op))
        for block in step.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    print(f"MINTED ids in the function: {len(minted)}; minted by the control "
          f"builder: {len(builder_minted)}; builder mints without a record: "
          f"{builder_unrecorded}")
    print(f"later-pass mints without a record (steps 6-7 residue): "
          f"{[(i, definers.get(i)) for i in residue]}")
    check("every id the control builder minted has a mint record",
          bool(builder_minted) and not builder_unrecorded)

    # ---- 6. unsourced-fact groups by stage -------------------------------
    groups = unsourced_groups(book)
    by_stage = Counter()
    for (page_name, stage_name), count in groups.items():
        by_stage[stage_name] += count
    print(f"unsourced rows by stage: {dict(by_stage)}")
    for stage_name in by_stage:
        if stage_name.startswith("control_ssa") or stage_name in {"reducer_read", "operand_position"}:
            check(f"no unsourced rows under stage {stage_name}", False)
    print("unsourced rows by (page, stage):")
    for (page_name, stage_name), count in sorted(groups.items()):
        print(f"  {page_name} / {stage_name}: {count}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
