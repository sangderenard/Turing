"""The planner's structure as book rows: a literal callsite fold, end to end.

Plan 80, part A (step 4), section A7.  A callee ``advance(x, rollback=False)``
whose body is ``saved = x * 2.0 if rollback else 0.0; return x + saved`` and
a root ``step(x)`` that calls ``advance(x, True)`` once.  Lowered through
``lower_ast_source_to_ssa`` with ``x`` declared a scalar in ``program_abi``.

Reading only the book (``identity_book(module)``), the probe checks:

1. a ``planner_specialization`` row for formal ``rollback`` whose fact is
   ``SpecializationFact(True, LITERAL)``, with an inbound edge to the
   ``True`` Constant's identity cell and, through it, a ``source_span`` row
   whose kind is ``Constant``;
2. a ``source_control_specialization_concordance`` row for the ``IfExp``
   DERIVED from the ``proven_literal`` cell of its test, which is DERIVED
   from the row of (1) -- the planner literal reaches the folded control;
3. ``executable_node`` rows under one planning scope, every one DERIVED;
   each ``deployment_region`` row DERIVED from its members, and its members
   present as ``deployment_region_member`` rows;
4. ``hierarchy_value`` rows for ``advance``'s formal ``x`` and ``step``'s
   argument ``x`` share one ``hierarchy_global_value``;
5. a ``call_binding`` row for the call node DERIVED from ``advance``'s
   ``function_address`` cell, and the activation record DERIVED from it;
6. the ``unsourced-fact`` counts by stage for the six planner stages,
   printed as measured; the stages the landed writers source completely
   are asserted zero.

A link whose writer has not landed (the ``call_argument_operand`` edge of
the formal's hierarchy row, plan 80 A7 item 4) is printed as absent, not
asserted.

    python -u tools/compiler_probes/probe_planner_specialization_chain.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CALL_BINDING,
    CANONICAL_VALUE,
    DEPLOYMENT_REGION,
    DEPLOYMENT_REGION_MEMBER,
    EXECUTABLE_NODE,
    FUNCTION_ADDRESS,
    HIERARCHY_VALUE,
    NAME_BINDING,
    PLANNER_SPECIALIZATION_PAGE,
    PROVEN_LITERAL,
    SOURCE_CALLSITE_ACTIVATION,
    SOURCE_CONTROL_SPECIALIZATION,
    SOURCE_SPAN,
    BindingFact,
    SpecializationFact,
    SpecializationSource,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    Ref,
    Unresolved,
    identity_book,
)

CONTRACTS = REPO / "extraction_contracts"
NAME = "planner_specialization_chain"

SOURCE = '''
def advance(x: float, rollback: bool = False) -> float:
    saved = x * 2.0 if rollback else 0.0
    return x + saved


def step(x: float) -> float:
    return advance(x, True)
'''

PLANNER_STAGES = (
    "planner_specialization", "planner_structural_fold",
    "planner_dispatch_classification", "planner_region_carve",
    "planner_hierarchy", "planner_call_binding",
)
#: Stages whose landed writers name a source for every row in this program.
#: ``planner_structural_fold`` is printed, not asserted: its residue is the
#: round-1 ``structural_specialization_fixed_point`` record of a fold that
#: folded nothing and ran before the function table declared the function
#: (no ``function_address`` cell yet) -- the record has no cell to name.
ASSERTED_ZERO = (
    "planner_specialization", "planner_region_carve", "planner_call_binding",
)


def lower():
    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {}, "bindings": [],
            "values": [
                {"function": "step", "parameter": "x", "storage": "scalar",
                 "dtype": "float64", "rank": 0,
                 "python_type": "builtins.float"},
                {"function": "advance", "parameter": "x",
                 "storage": "scalar", "dtype": "float64", "rank": 0,
                 "python_type": "builtins.float"},
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
    stored = book.pages.get(page.name)
    if stored is None:
        return []
    found = []
    for row in stored.rows():
        for column, fact in stored.history(row):
            if predicate(row, fact):
                found.append((Ref(page, row, column), fact))
    return found


def reaches(book, ref, wanted, limit=12) -> Ref | None:
    """Breadth-first over inbound edges: the first cell on ``wanted``."""

    frontier = [ref]
    seen = set()
    for _depth in range(limit):
        next_frontier = []
        for current in frontier:
            for source, _stage in book.edges_into(current):
                if source.key in seen:
                    continue
                seen.add(source.key)
                if source.page.name == wanted.name:
                    return source
                next_frontier.append(source)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


def scope_label(scope) -> str:
    return str(scope[0]) if isinstance(scope, tuple) and scope else str(scope)


def main() -> int:
    module, _outputs, _exports = lower()
    check("the program lowers", True)
    book = identity_book(module)

    # ---- 1. the specialization row and its literal ---------------------
    rows = cells(
        book, PLANNER_SPECIALIZATION_PAGE,
        lambda row, fact: row[1] == "rollback"
        and isinstance(fact, SpecializationFact)
        and fact.value is True
        and fact.source is SpecializationSource.LITERAL,
    )
    print(f"planner_specialization rows for 'rollback' = True (LITERAL): {len(rows)}")
    for ref, fact in rows:
        print(f"    {ref!r} = {fact!r}")
        for source, stage in book.edges_into(ref):
            print(f"        <- {source!r} [{stage.name}]")
    check("a LITERAL True row exists for 'rollback'", bool(rows))
    latest_rows = [
        ref for ref, _fact in rows
        if book.latest_ref(PLANNER_SPECIALIZATION_PAGE, ref.row) == ref
    ]
    argument_cells = [
        source for ref in latest_rows
        for source, _stage in book.edges_into(ref)
        if source.page.name in {CANONICAL_VALUE.name, "ingestion_value"}
    ]
    check("the row has an inbound edge to an argument identity cell",
          bool(argument_cells))
    spans = [
        reaches(book, ref, SOURCE_SPAN) for ref in latest_rows
    ]
    span_kinds = {
        book.pages[SOURCE_SPAN.name].latest(span.row).kind
        for span in spans if span is not None
    }
    print(f"    source_span kinds reached: {sorted(span_kinds)}")
    check("the argument cell reaches a source_span row of kind Constant",
          "Constant" in span_kinds)

    # ---- 2. the folded IfExp -------------------------------------------
    controls = cells(book, SOURCE_CONTROL_SPECIALIZATION)
    print(f"source_control_specialization_concordance rows: {len(controls)}")
    literal_edges = []
    for ref, fact in controls:
        print(f"    {ref!r} = {fact!r}")
        for source, stage in book.edges_into(ref):
            print(f"        <- {source!r} [{stage.name}]")
            if source.page.name == PROVEN_LITERAL.name:
                literal_edges.append(source)
    check("a control row exists and none is Unresolved", bool(controls) and
          not any(isinstance(fact, Unresolved) for _ref, fact in controls))
    check("a control row is DERIVED from a proven_literal cell",
          bool(literal_edges))
    check("that proven_literal cell is DERIVED from the specialization row",
          any(
              source.page.name == PLANNER_SPECIALIZATION_PAGE.name
              for literal in literal_edges
              for source, _stage in book.edges_into(literal)
          ))

    # ---- 3. classification and regions ----------------------------------
    executable = cells(book, EXECUTABLE_NODE)
    by_scope: dict = {}
    for ref, fact in executable:
        by_scope.setdefault(ref.row[0], []).append(ref)
    print(f"executable_node rows: {len(executable)} under {len(by_scope)} planning scope(s)")
    check("executable_node rows exist", bool(executable))
    check("every executable_node row has an inbound edge", all(
        book.edges_into(ref) for ref, _fact in executable
    ))
    regions = cells(book, DEPLOYMENT_REGION)
    members = cells(book, DEPLOYMENT_REGION_MEMBER)
    print(f"deployment_region rows: {len(regions)}; members: {len(members)}")
    check("deployment_region rows exist", bool(regions))
    check("every region row derives from executable_node / identity cells", all(
        book.edges_into(ref) and all(
            source.page.name in {EXECUTABLE_NODE.name, CANONICAL_VALUE.name,
                                 "ingestion_value"}
            for source, _stage in book.edges_into(ref)
        )
        for ref, _fact in regions
    ))
    check("every region has member rows", all(
        any(member.row[0] == ref.row[0] and member.row[1] == ref.row[1]
            for member, _fact in members)
        for ref, _fact in regions
    ))

    # ---- 4. the hierarchy correlation ---------------------------------
    def value_cells_of(function: str, name: str) -> set:
        found = set()
        page = book.pages.get(NAME_BINDING.name)
        if page is None:
            return found
        for row in page.rows():
            if row[1] != name or row[2] != 0:
                continue
            if not scope_label(row[0]).startswith(f"lexical_reads:{function}"):
                continue
            fact = page.latest(row)
            if isinstance(fact, BindingFact):
                cell = book.latest_ref(CANONICAL_VALUE, (row[0], int(fact.value_id)))
                if cell is not None:
                    found.add(cell.key)
        return found

    def globals_of(value_keys: set) -> set:
        return {
            (ref.row[0], fact)
            for ref, fact in cells(book, HIERARCHY_VALUE)
            if any(source.key in value_keys for source, _stage in book.edges_into(ref))
        }

    advance_x = globals_of(value_cells_of("advance", "x"))
    step_x = globals_of(value_cells_of("step", "x"))
    print(f"hierarchy globals for advance.x: {sorted(map(repr, advance_x))}")
    print(f"hierarchy globals for step.x:    {sorted(map(repr, step_x))}")
    check("hierarchy_value rows exist for advance.x and step.x",
          bool(advance_x) and bool(step_x))
    check("advance.x and step.x share one hierarchy_global_value",
          bool(advance_x & step_x))
    print("    (the formal's call_argument_operand edge: writer not landed; absent)")

    # ---- 5. the call binding ------------------------------------------
    bindings = cells(book, CALL_BINDING)
    print(f"call_binding rows: {len(bindings)}")
    address_bound = []
    for ref, fact in bindings:
        print(f"    {ref!r} = {fact!r}")
        for source, stage in book.edges_into(ref):
            print(f"        <- {source!r} [{stage.name}]")
            if source.page.name == FUNCTION_ADDRESS.name and (
                str(source.row[0]).split(".")[-1] == "advance"
            ):
                address_bound.append(ref)
    check("a call_binding row derives from advance's function_address cell",
          bool(address_bound))
    activations = cells(book, SOURCE_CALLSITE_ACTIVATION)
    check("the activation record derives from a call_binding cell", any(
        source.page.name == CALL_BINDING.name
        for ref, _fact in activations
        for source, _stage in book.edges_into(ref)
    ))

    # ---- 6. unsourced facts by planner stage ---------------------------
    counts: Counter = Counter()
    for page, row, reason, stage in book.unsourced_rows():
        counts[stage.name] += 1
        if stage.name in PLANNER_STAGES:
            page_name = getattr(page, "name", page)
            print(f"    unsourced: {page_name} {row!r} ({reason.name})")
    print("unsourced facts by stage (planner stages):")
    for stage in PLANNER_STAGES:
        print(f"    {stage}: {counts.get(stage, 0)}")
    for stage in ASSERTED_ZERO:
        check(f"zero unsourced facts at stage {stage}", counts.get(stage, 0) == 0)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
