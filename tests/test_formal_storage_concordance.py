from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    concord_compiler_frame_formals,
    concord_program_abi_frame_transitions,
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


def test_program_abi_field_retires_its_provisional_frame_lease():
    field = SSAValue(3, "float64", accounting={
        "program_abi_record": "Metrics",
        "program_abi_parameter": "metrics",
        "program_abi_field": "pub_limits",
        "program_abi_storage": "scalar",
        "linked_call_frame_storage": "step_with_dt_control_used",
        "propagated_formal_id": 17,
    })
    function = Function(
        "run_superstep", [field],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
        metadata={
            "parameter_names": (("metrics", 99),),
            "storage_formals": ({
                "value_id": 3,
                "kind": "linked_call_frame_storage",
            },),
        },
    )
    module = IRModule({"run_superstep": function})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        before = CorrelationTable.build(module).findings(module)
        assert any(
            finding.kind == "conflicting-storage-claims"
            for finding in before
        )

        receipts = concord_program_abi_frame_transitions(module)

        assert len(receipts) == 1
        assert receipts[0]["formal_id"] == 3
        assert "linked_call_frame_storage" not in field.accounting
        assert "propagated_formal_id" not in field.accounting
        assert function.metadata["storage_formals"] == ()
        assert book.page("program_abi_frame_transition").latest(
            ("run_superstep", 3)
        ) == (
            "Metrics", "pub_limits",
            (("linked_call_frame_storage", "step_with_dt_control_used"),
             ("propagated_formal_id", 17)),
        )
        after = CorrelationTable.build(module).findings(module)
        assert not any(
            finding.kind == "conflicting-storage-claims"
            for finding in after
        )
    finally:
        end_identity_book(token)
