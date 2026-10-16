from src.compiler.identity_concordance import (
    CorrelationTable,
    Mode,
    Novel,
    begin_identity_book,
    concord_compiler_frame_formals,
    concord_program_abi_frame_transitions,
    end_identity_book,
)
from src.compiler.concordance_declarations import (
    CALL_RECORD,
    CONTROL_SSA,
    FORMAL_ACTUAL_OCCURRENCE,
    FRAME_BINDING,
    INGEST_SOURCE,
    SSA_VALUE,
    SSAValueFact,
    SSAValueOrigin,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSACallRecord,
    SSACallTable,
    SSAValue,
)


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


def test_planned_region_occurrence_derives_from_exact_call_record():
    """A graph feed without an SSA cell keeps its retained callsite source."""

    book, token = begin_identity_book()
    try:
        scope = "caller@control:0"
        source_row = (scope, 91)
        book.post(
            SSA_VALUE,
            source_row,
            SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=CONTROL_SSA,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        source_cell = book.latest_ref(SSA_VALUE, source_row)
        assert source_cell is not None

        call_table = SSACallTable(owner="module")
        call_table.assign(
            "caller",
            (SSACallRecord(
                caller="caller",
                callsite_id=91,
                callee_reference=None,
                callee_name="planned_region_0",
                callee_symbol="planned_region_0",
            ),),
            sources=(source_cell,),
            stage=FRAME_BINDING,
        )
        formal = SSAValue(0, "int64")
        actual = SSAValue(19, "int64")
        callee = Function(
            "planned_region_0",
            [formal],
            {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
            metadata={"parameter_names": (("feed", 0),)},
        )
        caller = Function(
            "caller",
            [],
            {"entry": BasicBlock("entry", [
                Instr("Call", [actual], None, attributes={
                    "callee": "planned_region_0",
                    "plan_callsite_id": 91,
                }),
                Instr("Ret", [], None),
            ])},
            metadata={"tensor_shape_concordance_scope": scope},
        )
        module = IRModule(
            {"caller": caller, "planned_region_0": callee},
            call_table=call_table,
            metadata={"identity_book": book},
        )

        concord_compiler_frame_formals(module)

        row = (
            "planned_region_0", 0, "caller", "entry", 0, 0, scope,
        )
        occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
        call_record = book.latest_ref(
            CALL_RECORD, (call_table.owner, "caller"),
        )
        assert occurrence is not None
        assert call_record is not None
        assert book.edges_into(occurrence) == ((call_record, FRAME_BINDING),)
        assert not [
            finding for finding in CorrelationTable.build(module).findings(module)
            if finding.kind == "unsourced-fact"
            and finding.function == FORMAL_ACTUAL_OCCURRENCE.name
        ]
    finally:
        end_identity_book(token)


def test_composition_does_not_duplicate_accepted_caller_occurrences():
    """An imported caller's exact occurrence remains on its accepted book."""

    accepted_book, accepted_token = begin_identity_book()
    scope = "accepted_caller@control:0"
    formal = SSAValue(0, "int64")
    actual = SSAValue(19, "int64")
    callee = Function(
        "planned_region_0",
        [formal],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
        metadata={"parameter_names": (("feed", 0),)},
    )
    caller = Function(
        "accepted_caller",
        [],
        {"entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={
                "callee": "planned_region_0",
            }),
            Instr("Ret", [], None),
        ])},
        metadata={"tensor_shape_concordance_scope": scope},
    )
    accepted_module = IRModule(
        {"accepted_caller": caller, "planned_region_0": callee},
        metadata={"identity_book": accepted_book},
    )
    accepted_row = (
        "planned_region_0", 0, "accepted_caller", "entry", 0, 0, scope,
    )
    try:
        accepted_book.post(
            SSA_VALUE,
            (scope, 19),
            SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=CONTROL_SSA,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        concord_compiler_frame_formals(accepted_module)
        accepted_occurrence = accepted_book.latest_ref(
            FORMAL_ACTUAL_OCCURRENCE, accepted_row,
        )
        assert accepted_occurrence is not None
        assert accepted_book.edges_into(accepted_occurrence)
    finally:
        end_identity_book(accepted_token)

    accepted_component = {
        "schema": "turing.accepted-ssa-component.v1",
        "artifact_sha256": "accepted-fixture",
        "unit_index": 1,
        "identity_book": accepted_book,
    }
    caller.metadata["accepted_ssa_component"] = accepted_component
    callee.metadata["accepted_ssa_component"] = accepted_component

    composing_book, composing_token = begin_identity_book()
    try:
        composed_module = IRModule(
            {"accepted_caller": caller, "planned_region_0": callee},
            metadata={"identity_book": composing_book},
        )
        concord_compiler_frame_formals(composed_module)

        assert composing_book.page(FORMAL_ACTUAL_OCCURRENCE).rows() == ()
        assert accepted_book.latest_ref(
            FORMAL_ACTUAL_OCCURRENCE, accepted_row,
        ) == accepted_occurrence
        assert accepted_book.edges_into(accepted_occurrence)
    finally:
        end_identity_book(composing_token)


def test_constructor_field_value_keeps_proven_compiler_frame_ownership():
    formal = SSAValue(3, "float64", accounting={
        "program_abi_record": "WorldContact",
        "program_abi_field": "penetration_m",
        "program_abi_storage": "scalar",
        "record_constructor_value": 171,
    })
    callee = Function(
        "resolve_pair", [formal],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
        metadata={"parameter_names": (("self", 99),)},
    )
    actual = SSAValue(20, "float64", accounting={
        "compiler_frame_storage": "caller",
    })
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={
                "callee": "resolve_pair",
            }),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({"caller": caller, "resolve_pair": callee})
    book, token = begin_identity_book()
    module.metadata["identity_book"] = book
    try:
        receipts = concord_compiler_frame_formals(module)

        assert receipts[0]["formal_id"] == 3
        assert formal.accounting["program_abi_field"] == "penetration_m"
        assert formal.accounting["compiler_frame_storage"] == "resolve_pair"
        assert callee.metadata["storage_formals"][0]["value_id"] == 3
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
