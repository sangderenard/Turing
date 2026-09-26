from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    concord_compiler_frame_formals,
    end_identity_book,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def test_hidden_formal_is_accounted_when_every_call_supplies_frame_storage():
    formal = SSAValue(3, "float64")
    callee = Function(
        "helper", [formal],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
        metadata={"parameter_names": (("authored", 99),)},
    )
    actual = SSAValue(20, "float64", accounting={
        "linked_call_frame_storage": "helper",
    })
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={"callee": "helper"}),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({"caller": caller, "helper": callee})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        receipts = concord_compiler_frame_formals(module)

        assert receipts == ({
            "function": "helper",
            "formal_id": 3,
            "sources": (("caller", 20),),
            "kind": "compiler_frame_storage",
        },)
        assert formal.accounting["compiler_frame_storage"] == "helper"
        assert book.page("formal_storage_resolution").latest(
            ("helper", 3)
        ) == ("compiler_frame_storage", (("caller", 20),))
        findings = CorrelationTable.build(module).findings(module)
        assert not any(
            finding.kind == "unaccounted-formal" for finding in findings
        )
    finally:
        end_identity_book(token)
