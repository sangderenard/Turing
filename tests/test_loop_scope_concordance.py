from src.compiler.identity_concordance import (
    begin_identity_book,
    concord_loop_scope_latch_residents,
    declare_loop_scope,
    end_identity_book,
    loop_scope_declarations,
    rebind_loop_scope_inner,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


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
