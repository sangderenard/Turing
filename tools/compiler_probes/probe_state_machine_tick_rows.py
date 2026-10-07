"""The marked state-machine shell (``StateMachineTick``) is on the book.

    python -u tools/compiler_probes/probe_state_machine_tick_rows.py

A tick used to be the one control block with no owner: ``control_block``
fell back to the program cell, its ``ssa_block`` rows (``state_case`` /
``state_next`` / ``state_merge``) were ``Unresolved(SSA_BLOCK_OWNER_UNROUTED)``,
the state was a name string looked up in a table, the case literal was a
string with no cell, and the merge had no Phi.

Part A (a real reduced graph, a real book, a hand-built tick): the tick names
its owner node and its state VALUE, and every row below derives from them.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CONTROL_BLOCK, CONTROL_OWNER_UNKNOWN, ControlBlockKind, SSA_BLOCK,
    SSA_BLOCK_OWNER_UNROUTED, SSABlockKind,
)
from src.compiler.control_source import (  # noqa: E402
    ControlProgram, ControlUniform, SequenceBlock, StateMachineTick,
    StatementBlock, post_control_rewrite,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    IdentityBook, Unresolved, Unsourced, begin_identity_book,
    end_identity_book,
)
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


# ----------------------------------------------------------------- fixture

#: ``s`` is the state selector, the ``if`` stands in for the dispatch
#: construct (a graph node with an identity cell), the two literals are the
#: case labels.  The reducer numbers the nodes; they are found by what they
#: are, never by a remembered number.
SOURCE = '''
def f(s, a, b):
    if s > 2:
        a = a + 1
    k = a + 0
    m = b + 1
    return k + m
'''


def reduced_graph(book):
    graphs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lower_ast_source_to_ssa(
            SOURCE, "f", name="tickrows",
            extraction_contract=REPO / "extraction_contracts"
            / "program_extraction.yaml",
            identity_book=book, resolved_process_graph_sink=graphs.append,
            stop_after_compilation_unit_plan=True,
            progress=lambda message: None,
        )
    entry = next(e for e in graphs[0].function_table if e.name == "f")
    return entry.graph


class Fixture:
    def __init__(self, book):
        self.graph = reduced_graph(book)
        nodes = self.graph.G.nodes(data=True)
        self.scope = tuple(self.graph.G.graph["lexical_read_scope"])
        by_label = {
            data.get("label"): node for node, data in nodes
            if data.get("type") == "Input"
        }
        self.state = by_label["s"]
        self.initial = by_label["a"]
        self.owner = next(
            node for node, data in nodes if data.get("type") == "If"
        )
        constants = sorted(
            (node for node, data in nodes
             if data.get("type") == "Constant"
             and (data.get("attributes") or {}).get("value") in (0, 1)),
            key=lambda node: (self.graph.G.nodes[node]["attributes"]["value"]),
        )
        self.literals = tuple(constants[:2])
        self.fresh = max(self.graph.G.nodes) + 10

    def cell(self, book, node):
        from src.compiler.concordance_declarations import CANONICAL_VALUE

        return book.latest_ref(CANONICAL_VALUE, (self.scope, int(node)))


def lower_tick(fixture, program, *, first=None):
    post_control_rewrite(fixture.graph, program)
    fresh = fixture.fresh
    return lower_control_program_to_ssa(
        program, function_name="tickrows__f",
        first_value_id=fresh + 100 if first is None else first,
        region_callees={0: "case_zero", 1: "case_one"},
        region_signatures={
            0: ((fixture.state,), (fresh,)),
            1: ((fixture.state,), (fresh + 1,)),
        },
        lexical_read_scope=fixture.scope,
    )


def sources_of(book, ref):
    return tuple(source for source, _stage in book.edges_into(ref))


def block_ref(book, function_name, label):
    """The latest ``ssa_block`` cell of ``label``: the row's scope is the
    lowering's own control scope (book-numbered), so it is matched, not
    guessed."""

    page = book.page("ssa_block")
    for row in page.rows():
        if row[1] == function_name and row[2] == label:
            return book.latest_ref(SSA_BLOCK, row)
    return None


def fact_at(book, page, ref):
    return book.page(page).latest(ref.row)


# ------------------------------------------------------------------ part A


def part_a():
    print("== part A: a hand-built tick on a reduced graph")
    book = IdentityBook()
    _book, token = begin_identity_book(book)
    fixture = Fixture(book)
    state_cell = fixture.cell(book, fixture.state)
    owner_cell = fixture.cell(book, fixture.owner)
    tick = StateMachineTick(
        "dispatch",
        (
            ("0", StatementBlock(("__scheduled_region_0__",))),
            ("1", StatementBlock(("__scheduled_region_1__",))),
        ),
        state_value_id=fixture.state,
        source_node_id=fixture.owner,
    )
    program = ControlProgram(
        SequenceBlock((tick,)), region_indices=(0, 1),
    )
    function, shortfalls = lower_tick(fixture, program)
    check("lowering has no shortfalls", shortfalls == ())

    row = (fixture.scope, ControlBlockKind.STATE_MACHINE_TICK, owner_cell)
    block_cell = book.latest_ref(CONTROL_BLOCK, row)
    check("control_block row is keyed by the owner node's cell",
          block_cell is not None)
    if block_cell is not None:
        sources = sources_of(book, block_cell)
        check("the row is DERIVED from the owner cell and the state value cell",
              owner_cell in sources and state_cell in sources)
        fact = fact_at(book, CONTROL_BLOCK, block_cell)
        check("the fact's predicate is the state value's cell",
              fact.predicate == state_cell)
    program_rows = [
        row for row in book.page(CONTROL_BLOCK).scope_rows(fixture.scope)
        if row[1] is ControlBlockKind.STATE_MACHINE_TICK
        and row[2] != owner_cell
    ]
    check("no program-cell fallback row for the tick", not program_rows)
    unknown = [
        row for row in book.page(CONTROL_BLOCK).scope_rows(fixture.scope)
        if isinstance(fact_at(book, CONTROL_BLOCK, book.latest_ref(
            CONTROL_BLOCK, row)), Unsourced)
    ]
    check("no control_block row is Unsourced(CONTROL_OWNER_UNKNOWN) for it",
          not any(row[1] is ControlBlockKind.STATE_MACHINE_TICK
                  for row in unknown))

    labels = {
        stem: [name for name in function.blocks
               if name == stem or name.startswith(stem + ".")]
        for stem in ("state_case", "state_next", "state_merge")
    }
    check("two case blocks, two next blocks, one merge block",
          [len(labels[s]) for s in ("state_case", "state_next", "state_merge")]
          == [2, 2, 1])
    unrouted = []
    for stem, names in labels.items():
        for name in names:
            ref = block_ref(book, function.name, name)
            fact = None if ref is None else fact_at(book, SSA_BLOCK, ref)
            routed = (
                isinstance(fact, SSABlockKind)
                and block_cell in sources_of(book, ref)
            )
            if not routed:
                unrouted.append((name, fact))
    check("every state_case/state_next/state_merge ssa_block row is routed "
          "to the tick's control_block cell", not unrouted)
    if unrouted:
        print("      unrouted:", unrouted)

    eq = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs if str(instruction.op) == "Eq"
        or getattr(instruction.op, "name", "") == "Eq"
    )
    check("the Eq reads the state value by id (no name lookup)",
          int(eq.args[0].id) == int(fixture.state))
    end_identity_book(token)
    return book


def part_a_control():
    """A tick with no owner (a hand-built coordinator switch) is no longer
    exempt: it is the worklist's, like any other ownerless block."""

    print("== part A control: an ownerless tick is flagged, not exempted")
    book = IdentityBook()
    _book, token = begin_identity_book(book)
    fixture = Fixture(book)
    tick = StateMachineTick(
        "dispatch",
        (
            ("0", StatementBlock(("__scheduled_region_0__",))),
            ("1", StatementBlock(("__scheduled_region_1__",))),
        ),
    )
    program = ControlProgram(
        SequenceBlock((tick,)), region_indices=(0, 1),
        uniforms=(ControlUniform("dispatch", fixture.state, "int"),),
    )
    function, shortfalls = lower_tick(fixture, program)
    check("lowering has no shortfalls", shortfalls == ())
    flagged = [
        item for item in book.unsourced_rows()
        if item[0] is CONTROL_BLOCK and item[1][1]
        is ControlBlockKind.STATE_MACHINE_TICK
    ]
    check("the ownerless tick posts Unsourced(CONTROL_OWNER_UNKNOWN)",
          len(flagged) == 1 and flagged[0][2] is CONTROL_OWNER_UNKNOWN)
    case = block_ref(book, function.name, "state_case")
    fact = None if case is None else fact_at(book, SSA_BLOCK, case)
    check("its case block reads Unresolved(SSA_BLOCK_OWNER_UNROUTED)",
          isinstance(fact, Unresolved)
          and fact.reason is SSA_BLOCK_OWNER_UNROUTED)
    end_identity_book(token)


def main() -> int:
    part_a()
    part_a_control()
    print()
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
