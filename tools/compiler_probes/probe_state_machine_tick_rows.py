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
    BindingKind, CARRIED_SNAPSHOT, CONTROL_BLOCK_PLACEMENT,
    CONTROL_VALUE_BINDING,
    SSA_BLOCK_OWNER_UNROUTED, SSA_VALUE, SSABlockKind,
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
from src.compiler.ssa_self_check import check_definition_dominance  # noqa: E402
from src.transmogrifier.ssa import IRModule  # noqa: E402

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
        # ``a`` is rebound by the ``if``: the reducer's merge node for it is
        # the tick's merged id, the arm's ``a + 1`` the rebinding arm's value.
        self.merged = next(
            node for node, data in nodes
            if data.get("type") == "Phi" and data.get("label") == "a"
        )
        self.arm_value = min(
            node for node, data in nodes
            if data.get("type") == "Add" and node > self.owner
        )

    def cell(self, book, node):
        from src.compiler.concordance_declarations import CANONICAL_VALUE

        return book.latest_ref(CANONICAL_VALUE, (self.scope, int(node)))


def lower_tick(fixture, program, *, first=None, outputs=None):
    post_control_rewrite(fixture.graph, program)
    fresh = fixture.fresh
    outputs = outputs or {0: (fresh,), 1: (fresh + 1,)}
    return lower_control_program_to_ssa(
        program, function_name="tickrows__f",
        first_value_id=fresh + 100 if first is None else first,
        region_callees={region: f"case_{region}" for region in outputs},
        region_signatures={
            region: ((fixture.state,), ids) for region, ids in outputs.items()
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
        case_value_ids=fixture.literals,
    )
    program = ControlProgram(
        SequenceBlock((tick,)), region_indices=(0, 1),
    )
    function, shortfalls = lower_tick(fixture, program)
    check("lowering has no shortfalls", shortfalls == ())
    literal_cells = tuple(fixture.cell(book, node) for node in fixture.literals)
    check("both case literals are graph nodes with identity cells",
          len(literal_cells) == 2 and all(c is not None for c in literal_cells))

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
        check("the fact names the case literal cells, in case order",
              fact.cases == literal_cells)
        check("the row is DERIVED from the case literal cells",
              all(cell in sources for cell in literal_cells))
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
        for index, name in enumerate(names):
            ref = block_ref(book, function.name, name)
            fact = None if ref is None else fact_at(book, SSA_BLOCK, ref)
            expected = (block_cell,) + (
                (literal_cells[index],) if stem != "state_merge" else ()
            )
            routed = (
                isinstance(fact, SSABlockKind)
                and all(cell in sources_of(book, ref) for cell in expected)
            )
            if not routed:
                unrouted.append((name, fact))
    check("every state_case/state_next/state_merge ssa_block row is routed "
          "to the tick's control_block cell, and each case/next row also to "
          "its case literal's cell", not unrouted)
    if unrouted:
        print("      unrouted:", unrouted)

    eq = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs if str(instruction.op) == "Eq"
        or getattr(instruction.op, "name", "") == "Eq"
    )
    check("the Eq reads the state value by id (no name lookup)",
          int(eq.args[0].id) == int(fixture.state))
    scope = block_ref(book, function.name, "state_merge").row[0]
    literal_value = book.latest_ref(SSA_VALUE, (scope, int(eq.args[1].id)))
    minted = None if literal_value is None else book.mint_of(literal_value)
    check("the Eq's literal operand is minted from the case literal's cell",
          minted is not None and minted[1] == (literal_cells[0],))
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


def part_a_merge():
    print("== part A merge: the arms' rebound scalar joins in one Phi")
    book = IdentityBook()
    _book, token = begin_identity_book(book)
    fixture = Fixture(book)
    state_cell = fixture.cell(book, fixture.state)
    merged_cell = fixture.cell(book, fixture.merged)
    initial_cell = fixture.cell(book, fixture.initial)
    tick = StateMachineTick(
        "dispatch",
        (
            ("0", StatementBlock(("__scheduled_region_0__",))),
            ("1", StatementBlock(("__scheduled_region_1__",))),
        ),
        state_value_id=fixture.state,
        source_node_id=fixture.owner,
        case_value_ids=fixture.literals,
        # case 0 rebinds ``a`` (its region publishes ``a + 1``); case 1 and
        # the no-default fall-through keep the entered version.
        carried_aliases=((
            (fixture.arm_value, fixture.initial),
            fixture.initial, fixture.initial, fixture.merged,
        ),),
    )
    program = ControlProgram(
        SequenceBlock((tick,)), region_indices=(0, 1),
    )
    function, shortfalls = lower_tick(
        fixture, program,
        outputs={0: (fixture.arm_value,), 1: (fixture.fresh,)},
    )
    check("lowering has no shortfalls", shortfalls == ())
    merge = function.blocks["state_merge"]
    phis = [i for i in merge.instrs if str(getattr(i.op, "name", i.op)) == "Phi"]
    check("state_merge holds one Phi for the carried scalar", len(phis) == 1)
    if not phis:
        end_identity_book(token)
        return
    phi = phis[0]
    check("the Phi is the N-way state join (3 incoming: case 0, case 1, "
          "no-case fall-through)",
          phi.attributes.get("binding") == "state_carried"
          and len(phi.args) == 3
          and len(phi.attributes["incoming_blocks"]) == 3)
    check("the Phi's incoming blocks each branch to the merge",
          all("state_merge" in function.blocks[name].successors
              for name in phi.attributes["incoming_blocks"]))
    check("the merged id is the graph's merge node", int(phi.res.id)
          == int(fixture.merged))
    scope = block_ref(book, function.name, "state_merge").row[0]
    merged_value = book.latest_ref(SSA_VALUE, (scope, int(phi.res.id)))
    sources = () if merged_value is None else sources_of(book, merged_value)
    arm_cell = book.latest_ref(SSA_VALUE, (scope, int(phi.args[0].id)))
    state_value = book.latest_ref(SSA_VALUE, (scope, int(fixture.state)))
    check("the merged value is DERIVED from the arm's value, the snapshot "
          "and the state",
          arm_cell in sources and state_value in sources
          and any(source.page.name == "carried_snapshot" for source in sources))
    row = (fixture.scope, ControlBlockKind.STATE_MACHINE_TICK,
           fixture.cell(book, fixture.owner))
    block_cell = book.latest_ref(CONTROL_BLOCK, row)
    fact = None if block_cell is None else fact_at(book, CONTROL_BLOCK, block_cell)
    check("the tick's control_block fact names the merged cell",
          fact is not None and fact.carried == (merged_cell,))
    snapshot = book.latest_ref(
        CARRIED_SNAPSHOT, (scope, fixture.cell(book, fixture.owner),
                           int(fixture.initial)))
    check("one carried_snapshot row, derived from the tick's cell",
          snapshot is not None
          and fixture.cell(book, fixture.owner) in sources_of(book, snapshot))
    binding = book.latest_ref(
        CONTROL_VALUE_BINDING, (scope, int(fixture.merged)))
    bound = None if binding is None else fact_at(
        book, CONTROL_VALUE_BINDING, binding)
    check("the merged id is bound CONDITIONAL_MERGE",
          bound is not None and bound.kind is BindingKind.CONDITIONAL_MERGE)
    findings = check_definition_dominance(IRModule({function.name: function}))
    check("every use is dominated by its definition (Phi edges included)",
          not [f for f in findings if "formal" not in str(f).lower()
               and int(getattr(f, "value_id", -1) or -1) in (
                   int(fixture.merged), int(fixture.arm_value))])
    end_identity_book(token)


def part_a_two_ticks():
    print("== part A two ticks: one function, two ticks, distinct rows")
    book = IdentityBook()
    _book, token = begin_identity_book(book)
    fixture = Fixture(book)
    second_owner = next(
        node for node, data in fixture.graph.G.nodes(data=True)
        if data.get("op") == "greater"
    )

    def tick(owner, first_region):
        return StateMachineTick(
            "dispatch",
            tuple(
                (str(position), StatementBlock(
                    (f"__scheduled_region_{first_region + position}__",)))
                for position in range(2)
            ),
            state_value_id=fixture.state, source_node_id=owner,
            case_value_ids=fixture.literals,
        )

    program = ControlProgram(
        SequenceBlock((tick(fixture.owner, 0), tick(second_owner, 2))),
        region_indices=(0, 1, 2, 3),
    )
    function, shortfalls = lower_tick(
        fixture, program,
        outputs={r: (fixture.fresh + r,) for r in range(4)},
    )
    check("lowering has no shortfalls", shortfalls == ())
    cells = [fixture.cell(book, node) for node in (fixture.owner, second_owner)]
    rows = [
        book.latest_ref(CONTROL_BLOCK, (
            fixture.scope, ControlBlockKind.STATE_MACHINE_TICK, cell))
        for cell in cells
    ]
    check("each tick has its own control_block row, keyed by its own cell",
          all(row is not None for row in rows) and rows[0] != rows[1])
    tick_rows = [
        row for row in book.page(CONTROL_BLOCK).scope_rows(fixture.scope)
        if row[1] is ControlBlockKind.STATE_MACHINE_TICK
    ]
    check("exactly two tick rows, both keyed by their own cells (neither "
          "fell back to the program cell and collided)",
          sorted(row[2].key for row in tick_rows)
          == sorted(cell.key for cell in cells))
    ordinals = []
    for row in rows:
        placement = None if row is None else fact_at(
            book, CONTROL_BLOCK_PLACEMENT, book.latest_ref(
                CONTROL_BLOCK_PLACEMENT, (fixture.scope, row)))
        ordinals.append(None if placement is None else placement.ordinal)
    check("the two ticks are placed at ordinals 0 and 1 of the root",
          ordinals == [0, 1])
    owners_of = {}
    for name in function.blocks:
        stem = name.split(".")[0]
        if stem not in ("state_case", "state_next", "state_merge"):
            continue
        ref = block_ref(book, function.name, name)
        sources = () if ref is None else sources_of(book, ref)
        owners_of[name] = tuple(
            index for index, row in enumerate(rows) if row in sources
        )
    check("ten distinct ssa_block labels, each derived from exactly one tick",
          len(owners_of) == 10
          and all(len(owner) == 1 for owner in owners_of.values()))
    check("each tick owns five of them (two case, two next, one merge)",
          sorted(owner[0] for owner in owners_of.values()) == [0] * 5 + [1] * 5)
    end_identity_book(token)


def main() -> int:
    part_a()
    part_a_control()
    part_a_merge()
    part_a_two_ticks()
    print()
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
