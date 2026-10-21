from src.compiler.identity_concordance import (
    CorrelationTable,
    Mode,
    Novel,
    begin_identity_book,
    concord_loop_scope_latch_residents,
    declare_loop_scope,
    end_identity_book,
    loop_scope_declarations,
    rebind_loop_scope_inner,
)
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGESTION,
    INGEST_SOURCE,
    LOOP_RESULT_USE_REBINDING,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def test_loop_inner_transition_preserves_declaration_and_publishes_resident():
    book, token = begin_identity_book()
    try:
        declare_loop_scope(
            "artifact__step__specialized_deadbeef",
            7,
            "loop_header",
            "loop_latch",
            "loop_exit",
            ((10, 11, 12, 100, 101),),
        )
        rebind_loop_scope_inner(
            "artifact__step__specialized_deadbeef",
            7,
            12,
            99,
            "complete_loop_latch_carried",
        )

        declarations = loop_scope_declarations(book, "artifact__step")
        assert declarations == [{
            "loop_node_id": 7,
            "boundary": ("loop_header", "loop_latch", "loop_exit"),
            "rebinds": [{
                "outer": 10,
                "carried": 11,
                "inner": 99,
                "declared_inner": 12,
            }],
        }]
        assert book.page("loop_scope").latest(
            ("step", 7, 10)
        ) == ("graph", 100, 101)
        assert book.page("loop_scope_inner_transition").latest(
            ("step", 7, 12)
        ) == (99, "complete_loop_latch_carried")
    finally:
        end_identity_book(token)


def test_completed_module_concords_latch_projection_with_declared_inner():
    outer = SSAValue(10, "float64")
    carried = SSAValue(11, "float64")
    declared_inner = SSAValue(12, "float64")
    resident_inner = SSAValue(99, "float64")
    phi = Instr(
        "Phi", [outer, resident_inner], carried,
        attributes={
            "incoming_blocks": ("entry", "loop_latch"),
            "binding": "loop_carried",
        },
    )
    function = Function("artifact__step__specialized_deadbeef", [], {
        "entry": BasicBlock("entry", [
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_header": BasicBlock("loop_header", [phi], successors=["loop_latch"]),
        "loop_latch": BasicBlock("loop_latch", [
            Instr("Copy", [declared_inner], resident_inner),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_exit": BasicBlock("loop_exit", [Instr("Ret", [carried], None)]),
    })
    module = IRModule({function.name: function})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        declare_loop_scope(
            function.name, 7,
            "loop_header", "loop_latch", "loop_exit",
            ((outer.id, carried.id, declared_inner.id, 100, 101),),
        )

        receipts = concord_loop_scope_latch_residents(module)

        assert receipts == ({
            "function": function.name,
            "loop_node_id": 7,
            "carried": carried.id,
            "declared_inner": declared_inner.id,
            "resident_inner": resident_inner.id,
            "reason": "completed_module_latch_projection",
        },)
        declaration = loop_scope_declarations(book, function.name)[0]
        assert declaration["rebinds"][0]["inner"] == resident_inner.id
        assert resident_inner.accounting["loop_scope_inner_transition"] == (
            declared_inner.id,
            resident_inner.id,
            "completed_module_latch_projection",
        )
    finally:
        end_identity_book(token)


def test_shared_loop_seed_retains_each_authored_binding_row():
    """Simultaneous future updates are not identified by one incumbent."""

    book, token = begin_identity_book()
    try:
        declare_loop_scope(
            "artifact__root", 7,
            "loop_header", "loop_latch", "loop_exit",
            (
                (10, 11, 12, 100, 101, ("cap",)),
                (10, 13, 14, 100, 102, ("last",)),
            ),
        )

        declaration = loop_scope_declarations(book, "artifact__root")[0]
        assert declaration["rebinds"] == [
            {
                "outer": 10, "carried": 11, "inner": 12,
                "declared_inner": 12, "source_bindings": ("cap",),
            },
            {
                "outer": 10, "carried": 13, "inner": 14,
                "declared_inner": 14, "source_bindings": ("last",),
            },
        ]
    finally:
        end_identity_book(token)


def test_sourced_loop_generations_advance_from_their_own_cells():
    book, token = begin_identity_book()
    try:
        cells = {}
        for value_id in (10, 11, 12):
            cells[value_id] = book.post(
                CANONICAL_VALUE, ("loop-scope-test", value_id),
                ("test-value", value_id),
                stage=INGESTION,
                provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )

        declare_loop_scope(
            "artifact__root", 7,
            "loop_header", "loop_latch", "loop_exit",
            ((10, 11, 12, 100, 101, ("value",)),),
            boundary_cells=(cells[10],),
            rebind_cells=lambda *value_ids: tuple(
                cells[value_id] for value_id in value_ids
            ),
        )

        declaration = loop_scope_declarations(book, "artifact__root")[0]
        assert declaration["rebinds"] == [{
            "outer": 10,
            "carried": 11,
            "inner": 12,
            "declared_inner": 12,
            "source_bindings": ("value",),
        }]
    finally:
        end_identity_book(token)


def test_completed_module_rebinds_declared_outer_read_to_header_phi():
    outer = SSAValue(10, "float64")
    carried = SSAValue(11, "float64")
    inner = SSAValue(12, "float64")
    call_result = SSAValue(13, "float64")
    phi = Instr(
        "Phi", [outer, inner], carried,
        attributes={
            "incoming_blocks": ("entry", "loop_latch"),
            "binding": "loop_carried",
        },
    )
    stale_call = Instr("Call", [outer], call_result)
    function = Function("artifact__root", [outer], {
        "entry": BasicBlock("entry", [
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_header": BasicBlock("loop_header", [
            phi,
            Instr("Br", [], None, attributes={"target": "loop_body"}),
        ], successors=["loop_body"]),
        "loop_body": BasicBlock("loop_body", [
            stale_call,
            Instr("Br", [], None, attributes={"target": "loop_latch"}),
        ], successors=["loop_latch"]),
        "loop_latch": BasicBlock("loop_latch", [
            Instr("Copy", [carried], inner),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_exit": BasicBlock("loop_exit", [Instr("Ret", [carried], None)]),
    })
    module = IRModule({function.name: function})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        publish_authored_ssa(module.functions)
        from src.compiler.ssa_record_return_state import ssa_value_identity_cell

        declare_loop_scope(
            function.name, 7,
            "loop_header", "loop_latch", "loop_exit",
            ((outer.id, carried.id, inner.id, 100, 101, ("row",)),),
            boundary_cells=(
                ssa_value_identity_cell(function, outer.id, book=book),
            ),
            rebind_cells=lambda *value_ids: tuple(
                ssa_value_identity_cell(function, value_id, book=book)
                for value_id in value_ids
            ),
        )

        receipts = concord_loop_scope_latch_residents(module)

        assert stale_call.args == [carried]
        assert any(
            receipt.get("reason") == "declared_loop_scope_outer_read"
            for receipt in receipts
        )
        row = (function.name, "loop_body", 0, 0)
        cell = book.latest_ref(LOOP_RESULT_USE_REBINDING, row)
        assert cell is not None
        assert book.page(LOOP_RESULT_USE_REBINDING).latest(row) == (
            outer.id,
            carried.id,
            ("declared_loop_scope", 7, "loop_header", "loop_latch"),
        )
        assert book.edges_into(cell)
    finally:
        end_identity_book(token)


def test_concordance_reports_shared_carried_identity_for_distinct_bindings():
    outer = SSAValue(10, "float64")
    carried = SSAValue(11, "float64")
    first_inner = SSAValue(12, "float64")
    second_inner = SSAValue(14, "float64")
    function = Function("artifact__root", [outer], {
        "entry": BasicBlock("entry", [
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_header": BasicBlock("loop_header", [
            Instr(
                "Phi", [outer, first_inner], carried,
                attributes={
                    "incoming_blocks": ("entry", "loop_latch"),
                    "binding": "loop_carried",
                },
            ),
            Instr("Br", [], None, attributes={"target": "loop_latch"}),
        ], successors=["loop_latch"]),
        "loop_latch": BasicBlock("loop_latch", [
            Instr("Copy", [outer], first_inner),
            Instr("Copy", [outer], second_inner),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], successors=["loop_header"]),
        "loop_exit": BasicBlock("loop_exit", [Instr("Ret", [carried], None)]),
    })
    module = IRModule({function.name: function})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        declare_loop_scope(
            function.name, 7,
            "loop_header", "loop_latch", "loop_exit",
            (
                (10, 11, 12, 100, 101, ("cap",)),
                (10, 11, 14, 100, 102, ("last",)),
            ),
        )

        findings = CorrelationTable.build(module).findings(module)
        assert any(
            finding.kind == "loop-scope-simultaneous-binding-collapse"
            for finding in findings
        )
    finally:
        end_identity_book(token)
