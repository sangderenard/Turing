import ast
import contextlib
import io
import inspect
import _pickle
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import sympy

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics, _scalar
from src.compiler.process_graph_function_linking import link_process_graph_functions
from src.compiler.fortran_c_shell import (
    _dispatch_region_completion_positions,
    _frame_binding_value_ids,
    _is_dead_conceptual_record_argument,
    _lower_planned_region_record_projection_captures,
    _linked_sequence_propagation_kind,
    _linked_frame_physical_shape,
    _same_declared_span_storage,
    _linked_frame_storage_role,
    _linked_frame_storage_owner,
    _monotonic_ssa_ids,
    _preferred_linked_field_candidates,
    _publish_concorded_output_identities,
    _concord_record_return_phi_inputs,
    _apply_concorded_function_aliases,
    _prune_dead_entry_field_aliases,
    _prune_unused_callee_formals,
    _rebind_linked_storage_alias,
    _reconcile_post_aggregate_record_results,
    _resolve_repeated_aggregate_output_positions,
    _retained_parameter_identity,
    _static_mapping_capacity_bounds,
    _undefined_repository_ssa_operands,
    lower_ast_source_to_ssa,
)
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator
from src.compiler.ssa_record_return_state import (
    repair_non_dominating_record_phi_uses,
)
from src.compiler.ssa_aggregate_abi import is_storage_view
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSACallRecord,
    SSARecordDescriptor,
    SSARecordFieldDescriptor,
    SSARecordFieldStorage,
    SSARecordTable,
    SSAValue,
)


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


# These fixtures exercise keyed records independently of the live dt span ABI.
from src.compiler.extraction_contract import ExtractionContract
import yaml
KEYED_CONTRACT = ExtractionContract(CONTRACT).with_program_abi(yaml.safe_load(
    (Path(__file__).parent / "fixtures/keyed_dt_record_abi.yaml").read_text()))



def test_duplicate_record_result_position_uses_first_physical_incumbent():
    resolved, receipts = _resolve_repeated_aggregate_output_positions(
        (307, 295, 295, 154, 155),
        {0: 367, 1: 1660, 3: 1661, 4: 1662},
    )

    assert resolved == {0: 367, 1: 1660, 2: 1660, 3: 1661, 4: 1662}
    assert receipts == ((
        2, 295, 1660,
        "exact_repeated_callee_result_identity",
        "incumbent_on_equal_priority",
    ),)


def test_frame_owner_uses_optional_program_abi_identity_before_local_id():
    common = {
        "program_abi_record": "package.STController",
        "program_abi_field": "dt_max",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
        "program_abi_optional_presence": True,
    }
    direct = SSAValue(95, "bool", accounting={
        **common, "program_abi_parameter": "self",
    })
    propagated = SSAValue(1784, "bool", accounting={
        **common, "program_abi_parameter": "ctrl",
        "linked_call_frame_storage": "update_dt_max",
    })
    payload = SSAValue(1637, "float64", accounting={
        "program_abi_record": "package.STController",
        "program_abi_field": "dt_max",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
        "program_abi_optional_payload": True,
    })
    propagated_payload = SSAValue(1776, "float64", accounting={
        "program_abi_record": "package.STController",
        "program_abi_parameter": "ctrl",
        "program_abi_field": "dt_max",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
        "linked_call_frame_storage": "step",
    })

    direct_owner = _linked_frame_storage_owner(
        direct, "package.STController.dt_max.__present", "pi_update", 95,
    )
    propagated_owner = _linked_frame_storage_owner(
        propagated, None, "update_dt_max", 37,
    )

    assert propagated_owner == direct_owner
    assert _linked_frame_storage_owner(
        payload, "package.STController.dt_max", "step", 1637,
    ) != direct_owner
    assert _linked_frame_storage_role(payload.accounting) == "payload"
    assert _linked_frame_storage_role(propagated_payload.accounting) == "payload"
    assert _linked_frame_storage_owner(
        payload, "package.STController.dt_max", "step", 1637,
    ) == _linked_frame_storage_owner(
        propagated_payload, None, "step", 1776,
    )
    ordinary = SSAValue(300, "float64", accounting={
        "program_abi_record": "package.State",
        "program_abi_field": "values",
    })
    # Ordinary fields and aggregate members still require a descriptor or a
    # callee-local identity; parameter accounting alone is insufficient to
    # merge distinct object instances.
    assert _linked_frame_storage_owner(
        ordinary, None, "first", 300,
    ) != _linked_frame_storage_owner(
        ordinary, None, "second", 300,
    )
    scalar = SSAValue(301, "float64", accounting={
        "program_abi_record": "package.STController",
        "program_abi_field": "acc",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
    })
    stale_singleton = SSAValue(302, "float64", (1,), accounting={
        **scalar.accounting,
        "linked_call_frame_storage": "step",
    })
    assert _linked_frame_physical_shape(stale_singleton) == ()
    assert _linked_frame_storage_owner(
        scalar, "package.STController.acc", "run", 301,
    ) == _linked_frame_storage_owner(
        stale_singleton, None, "step", 302,
    )
    table_handle = SSAValue(303, "int64", accounting={
        "program_abi_record": "package.Metrics",
        "program_abi_field": "error_channels",
        "program_abi_storage": "keyed",
        "program_abi_rank": 0,
    })
    assert _linked_frame_storage_owner(
        table_handle, None, "first", 303,
    ) != _linked_frame_storage_owner(
        table_handle, None, "second", 303,
    )


def test_linked_field_candidate_priority_retains_equal_incumbent():
    common = {
        "program_abi_record": "package.STController",
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_max",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
    }
    first_read_only = SSAValue(10, "float64", accounting=dict(common))
    second_read_only = SSAValue(11, "float64", accounting=dict(common))
    generated_writable = SSAValue(12, "float64", accounting={
        **common,
        "program_abi_field_written": True,
        "callsite_id": 7,
    })
    canonical_writable = SSAValue(13, "float64", accounting={
        **common,
        "program_abi_field_written": True,
    })

    assert _preferred_linked_field_candidates([
        first_read_only, second_read_only,
    ]) == [first_read_only]
    assert _preferred_linked_field_candidates([
        first_read_only, canonical_writable, generated_writable,
    ]) == [canonical_writable]
    first_span = SSAValue(14, "float64", (4,), accounting={
        **common,
        "program_abi_storage": "span",
    })
    second_span = SSAValue(15, "float64", (4,), accounting={
        **common,
        "program_abi_storage": "span",
        "program_abi_field_written": True,
    })
    assert _preferred_linked_field_candidates([
        first_span, second_span,
    ]) == [first_span, second_span]


def test_dead_entry_field_alias_pruning_preserves_live_and_distinct_storage():
    common = {
        "program_abi_record": "package.STController",
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_max",
        "program_abi_storage": "scalar",
        "program_abi_rank": 0,
    }
    canonical = SSAValue(120, "float64", accounting={
        **common,
        "program_abi_field_written": True,
    })
    dead_generated = SSAValue(386, "float64", accounting={
        **common,
        "callsite_id": 43,
        "linked_call_frame_storage": "run_superstep",
    })
    live_generated = SSAValue(435, "float64", accounting={
        **common,
        "callsite_id": 228,
        "linked_call_frame_storage": "run_superstep",
    })
    receipt_generated = SSAValue(436, "float64", accounting={
        **common,
        "callsite_id": 229,
        "linked_call_frame_storage": "run_superstep",
    })
    distinct_parameter = SSAValue(500, "float64", accounting={
        **common,
        "program_abi_parameter": "fallback_controller",
        "linked_call_frame_storage": "run_superstep",
    })
    root = Function("root", [
        canonical, dead_generated, live_generated, receipt_generated,
        distinct_parameter,
    ], {
        "entry": BasicBlock("entry", [Instr("Ret", [canonical, live_generated], None)]),
    })
    receipt = SSACallRecord(
        "root", 229, None, "callee", "callee",
        frame_bindings=((1, "caller_storage", 436),),
        resolution="native_call",
    )

    assert _prune_dead_entry_field_aliases({"root": root}, {"root": [receipt]}) == 1
    assert [argument.id for argument in root.args] == [120, 435, 436, 500]
    assert root.metadata["pruned_dead_entry_field_aliases"] == (386,)


def test_post_aggregate_passthrough_rebinds_record_and_following_frame():
    first = SSAValue(100, "float64")
    second = SSAValue(101, "float64")
    producer = Function("producer", [first, second], {
        "entry": BasicBlock("entry", [Instr("Ret", [first, second], None)]),
    }, metadata={"record_return_layouts": ((5, (100, 101)),)})
    actual = SSAValue(1, "float64")
    stale_first = SSAValue(10, "float64")
    stale_second = SSAValue(11, "float64")
    aggregate = SSAValue(12, "ssa.aggregate")
    produced = Instr("Call", [actual, actual], aggregate, attributes={
        "callee": "producer",
        "plan_callsite_id": 7,
        "result_convention": "ssa.aggregate",
        "native_result_contract": (
            (100, "float64", ()), (101, "float64", ()),
        ),
        "output_positions": (),
        "output_ids": (),
        "aggregate_output_passthrough_bindings": (
            (0, 10, 100, 1, "exact_callee_output_formal",
             "incumbent_on_equal_priority"),
            (1, 11, 101, 1, "exact_callee_output_formal",
             "incumbent_on_equal_priority"),
        ),
    })
    consumer_left = SSAValue(200, "float64")
    consumer_right = SSAValue(201, "float64")
    consumer = Function("consumer", [consumer_left, consumer_right], {
        "entry": BasicBlock("entry", []),
    })
    root = Function("root", [actual, stale_first, stale_second], {
        "entry": BasicBlock("entry", [produced]),
    })
    module = IRModule({
        "root": root, "producer": producer, "consumer": consumer,
    })
    module.record_tables["producer"] = SSARecordTable(records={
        5: SSARecordDescriptor(5, "Pair", (
            SSARecordFieldDescriptor(
                "left", "scalar", storage_identity="Pair.left",
                value_ids=(100,), dtype="float64",
            ),
            SSARecordFieldDescriptor(
                "right", "scalar", storage_identity="Pair.right",
                value_ids=(101,), dtype="float64",
            ),
        )),
    })
    module.record_tables["consumer"] = SSARecordTable(records={
        6: SSARecordDescriptor(6, "Pair", (
            SSARecordFieldDescriptor(
                "left", "scalar", storage_identity="Pair.left",
                value_ids=(200,), dtype="float64",
            ),
            SSARecordFieldDescriptor(
                "right", "scalar", storage_identity="Pair.right",
                value_ids=(201,), dtype="float64",
            ),
        )),
    })
    module.record_tables["root"] = SSARecordTable(records={
        20: SSARecordDescriptor(20, "Pair", (
            SSARecordFieldDescriptor(
                "left", "scalar", storage_identity="Pair.left",
                value_ids=(10,), dtype="float64",
            ),
            SSARecordFieldDescriptor(
                "right", "scalar", storage_identity="Pair.right",
                value_ids=(11,), dtype="float64",
            ),
        )),
    })
    module.call_table["root"] = (
        SSACallRecord(
            "root", 7, None, "producer", "producer",
            result_bindings=((5, 20),), resolution="native_call",
        ),
        SSACallRecord(
            "root", 8, None, "consumer", "consumer",
            argument_bindings=((20, 6),),
            frame_bindings=(
                (200, "caller_storage", 10),
                (201, "caller_storage", 11),
            ),
            resolution="native_call",
        ),
    )

    receipts = _reconcile_post_aggregate_record_results(module)

    assert receipts
    assert [
        field.value_ids
        for field in module.record_tables["root"].records[20].fields
    ] == [(1,), (1,)]
    assert module.call_table["root"][1].frame_bindings == (
        (200, "caller_storage", 1),
        (201, "caller_storage", 1),
    )
    assert all(item["tie_policy"] == "incumbent" for item in receipts)


def test_atomic_region_uses_its_completion_position_for_field_effect_order():
    graph = nx.DiGraph()
    early, middle, late = ast.parse(
        "first = 1\nsecond = 2\nthird = 3\n"
    ).body
    graph.add_node(10, expr_obj=early)
    graph.add_node(20, expr_obj=middle)
    graph.add_node(30, expr_obj=late)
    spanning = nx.DiGraph()
    spanning.graph["deployment_nodes"] = (10, 30)
    middle_only = nx.DiGraph()
    middle_only.graph["deployment_nodes"] = (20,)

    positions = _dispatch_region_completion_positions(
        graph,
        (SimpleNamespace(G=spanning), SimpleNamespace(G=middle_only)),
    )

    assert positions[0][:2] == (3, 0)
    assert positions[1][:2] == (2, 0)
    assert positions[1] < positions[0]


@pytest.mark.parametrize(("actual_record_id", "record_aliases", "keep_source_attribute", "keep_descriptor"), (
    (7, {}, True, True),
    (70, {7: 70}, True, True),
    (70, {7: 70}, False, True),
    (70, {7: 70}, False, False),
))
def test_planned_region_captures_proven_incumbent_record_scalar_once(
    actual_record_id, record_aliases, keep_source_attribute, keep_descriptor,
):
    owner_name = "controller"
    region_name = "controller__planned_region_0"
    dt = SSAValue(9, dtype="float64")
    record = SSAValue(actual_record_id, dtype="ssa.aggregate")
    clamp_events = SSAValue(124, dtype="int64")
    call_result = SSAValue(680, dtype="ssa.aggregate")
    call = Instr(
        "Call", [dt, record], call_result,
        attributes={
            "callee": region_name,
            "feed_ids": (9, actual_record_id),
            "feed_shapes": ((), ()),
            "feed_dtypes": ("float64", "ssa.aggregate"),
            "output_ids": (43, 311, 312),
            "result_convention": "ssa.aggregate",
        },
    )
    owner = Function(
        owner_name,
        [dt, record, clamp_events],
        {"entry": BasicBlock("entry", [call])},
        metadata={"value_aliases": {
            311: 124, 312: 124, **record_aliases,
        }},
    )

    region_dt = SSAValue(9, dtype="float64")
    region_record = SSAValue(7, dtype="ssa.aggregate")
    converted_dt = SSAValue(43, dtype="float64")
    first_projection = SSAValue(311, dtype="int64")
    second_projection = SSAValue(312, dtype="int64")
    region = Function(
        region_name,
        [region_dt, region_record],
        {"entry": BasicBlock("entry", [
            Instr("Cast", [region_dt], converted_dt),
            Instr(
                "getattr", [region_record], first_projection,
                attributes=({
                    "attribute": "clamp_events",
                    "initial_record_field_state": True,
                } if keep_source_attribute else {}),
            ),
            Instr(
                "getattr", [region_record], second_projection,
                attributes=({
                    "attribute": "clamp_events",
                    "initial_record_field_state": True,
                } if keep_source_attribute else {}),
            ),
            Instr("Ret", [converted_dt, first_projection, second_projection], None),
        ])},
        metadata={"source_region_integral": {
            "owner": owner_name,
            "capture_value_ids": (9, 7),
            "output_value_ids": (43, 311, 312),
        }},
    )
    records = SSARecordTable()
    if keep_descriptor:
        records.register(SSARecordDescriptor(
            7,
            "STController",
            fields=(SSARecordFieldDescriptor(
                "clamp_events",
                SSARecordFieldStorage.SCALAR,
                storage_identity="STController.clamp_events",
                value_ids=(124, 328, 126),
                dtype="int64",
            ),),
        ))
    functions = {owner_name: owner, region_name: region}

    assert _lower_planned_region_record_projection_captures(
        functions, {owner_name: records},
    ) == 2
    assert [value.id for value in region.args] == [9, 7, 124]
    assert [value.id for value in call.args] == [9, actual_record_id, 124]
    projections = region.blocks["entry"].instrs[1:3]
    assert [instruction.op for instruction in projections] == ["Cast", "Cast"]
    assert [instruction.args[0].id for instruction in projections] == [124, 124]
    assert region.metadata["source_region_integral"]["capture_value_ids"] == (
        9, 124,
    )
    assert region.metadata["lowered_record_projection_captures"] == (
        (311, "clamp_events" if keep_descriptor else "", 7, 124),
        (312, "clamp_events" if keep_descriptor else "", 7, 124),
    )

    assert _prune_unused_callee_formals(functions) == 1
    assert [value.id for value in region.args] == [9, 124]
    assert [value.id for value in call.args] == [9, 124]
    assert call.attributes["feed_ids"] == (9, 124)
    assert call.attributes["feed_shapes"] == ((), ())
    assert call.attributes["feed_dtypes"] == ("float64", "int64")


def test_planned_region_record_projection_reads_shared_concordance():
    from src.compiler.identity_concordance import (
        begin_identity_book,
        current_identity_book,
        end_identity_book,
    )

    owner_name = "controller"
    region_name = "controller__planned_region_31"
    dt = SSAValue(9, dtype="float64")
    actual_record = SSAValue(70, dtype="ssa.aggregate")
    resident = SSAValue(124, dtype="int64")
    call = Instr(
        "Call", [dt, actual_record], SSAValue(680, dtype="ssa.aggregate"),
        attributes={
            "callee": region_name,
            "feed_ids": (9, 70),
            "feed_shapes": ((), ()),
            "feed_dtypes": ("float64", "ssa.aggregate"),
            "output_ids": (43, 311, 312),
            "result_convention": "ssa.aggregate",
        },
    )
    owner = Function(
        owner_name, [dt, actual_record, resident],
        {"entry": BasicBlock("entry", [call])},
    )
    region_record = SSAValue(7, dtype="ssa.aggregate")
    first = SSAValue(311, dtype="int64")
    second = SSAValue(312, dtype="int64")
    region = Function(
        region_name, [SSAValue(9, dtype="float64"), region_record],
        {"entry": BasicBlock("entry", [
            Instr("getattr", [region_record], first,
                  attributes={"attribute": "clamp_events"}),
            Instr("getattr", [region_record], second,
                  attributes={"attribute": "clamp_events"}),
            Instr("Ret", [first, second], None),
        ])},
        metadata={"source_region_integral": {
            "owner": owner_name,
            "capture_value_ids": (9, 7),
            "output_value_ids": (311, 312),
        }},
    )
    records = SSARecordTable()
    records.register(SSARecordDescriptor(
        7, "STController",
        fields=(SSARecordFieldDescriptor(
            "clamp_events", SSARecordFieldStorage.SCALAR,
            storage_identity="STController.clamp_events",
            value_ids=(124,), dtype="int64",
        ),),
    ))

    _book, token = begin_identity_book()
    try:
        concordance = current_identity_book().page(
            "planning_value_concordance"
        )
        concordance.bind_alias(owner_name, 70, 7)
        concordance.bind_alias(owner_name, 311, 124)
        concordance.bind_alias(owner_name, 312, 124)

        assert _lower_planned_region_record_projection_captures(
            {owner_name: owner, region_name: region},
            {owner_name: records},
        ) == 2
    finally:
        end_identity_book(token)

    assert [instruction.op for instruction in region.blocks["entry"].instrs[:2]] == [
        "Cast", "Cast",
    ]
    assert [value.id for value in call.args] == [9, 70, 124]
    assert region.metadata["source_region_integral"]["capture_value_ids"] == (
        9, 124,
    )


def test_identity_audit_rejects_private_alias_snapshot():
    from src.compiler.identity_concordance import (
        IdentityBook,
        concordance_report,
    )

    function = Function(
        "owner", [SSAValue(1), SSAValue(2)],
        {"entry": BasicBlock("entry", [])},
        metadata={"value_aliases": {2: 1}},
    )
    module = IRModule({"owner": function})
    book = IdentityBook()
    module.metadata["identity_book"] = book

    assert "[alias-not-concorded] x1" in concordance_report(module)

    book.page("planning_value_concordance").bind_alias("owner", 2, 1)
    assert "alias-not-concorded" not in concordance_report(module)


def test_identity_audit_reads_source_field_identity_history():
    from src.compiler.identity_concordance import (
        IdentityBook,
        concordance_report,
    )

    module = IRModule({})
    book = IdentityBook()
    module.metadata["identity_book"] = book
    page = book.page("source_field_identity_concordance")
    page.set(("Sim", "registry"), 0, "RegistryA")
    page.set(("Sim", "registry"), 1, "RegistryB")

    report = concordance_report(module)
    assert "[source-field-identity-disagreement] x1" in report
    assert "field 'registry' changed identity" in report


def test_identity_audit_reads_callable_identity_history():
    from src.compiler.identity_concordance import (
        IdentityBook,
        concordance_report,
    )

    module = IRModule({})
    book = IdentityBook()
    module.metadata["identity_book"] = book
    page = book.page("callable_identity_concordance")
    page.set(("root", 17), 0, 3)
    page.set(("root", 17), 1, 4)

    report = concordance_report(module)
    assert "[callable-identity-disagreement] x1" in report
    assert "changed function-table address" in report


def test_output_alias_publication_uses_distinct_concordance_from_storage():
    from src.compiler.identity_concordance import (
        begin_identity_book,
        current_identity_book,
        end_identity_book,
    )

    function = Function(
        "step", [], {"entry": BasicBlock("entry", [])},
        metadata={"value_aliases": {376: 375}},
    )
    _book, token = begin_identity_book()
    try:
        page = current_identity_book().page("planning_value_concordance")
        page.bind_alias("step", 376, 375)

        aliases = _publish_concorded_output_identities(
            function, {376: 2305843010213698294},
        )

        assert aliases[376] == 2305843010213698294
        assert function.metadata["value_aliases"][376] == 375
        assert page.latest(("step", 376)) == 375
        assert current_identity_book().page(
            "output_identity_concordance"
        ).latest(("step", 376)) == 2305843010213698294
    finally:
        end_identity_book(token)


def test_planning_alias_refinement_records_concorded_transition():
    from src.compiler.fortran_c_shell import (
        _publish_concordant_function_aliases,
    )
    from src.compiler.identity_concordance import (
        begin_identity_book,
        current_identity_book,
        end_identity_book,
    )

    function = Function(
        "step", [], {"entry": BasicBlock("entry", [])},
        metadata={"value_aliases": {272: 100}},
    )
    _book, token = begin_identity_book()
    try:
        planning = current_identity_book().page(
            "planning_value_concordance"
        )
        planning.bind_alias("step", 272, 100)

        aliases = _publish_concordant_function_aliases(
            function, {272: 200},
            provenance="linked_call_record_projection",
        )

        assert aliases[272] == 200
        assert function.metadata["value_aliases"][272] == 200
        assert planning.latest(("step", 272)) == 200
        assert current_identity_book().page(
            "planning_alias_transition_concordance"
        ).latest(("step", 272)) == (
            100, 200, "linked_call_record_projection",
        )
    finally:
        end_identity_book(token)


def test_record_return_phi_revisit_rejects_merged_self_candidate():
    from src.compiler.identity_concordance import (
        begin_identity_book,
        current_identity_book,
        end_identity_book,
    )

    incoming = SSAValue(10, dtype="float64")
    result = SSAValue(20, dtype="float64")
    phi = Instr(
        "Phi", [incoming], result,
        attributes={
            "binding": "return_merge",
            "record_return_scalar": True,
            "record_field": "limits",
            "incoming_blocks": ("return_edge",),
        },
    )
    _book, token = begin_identity_book()
    try:
        selected = _concord_record_return_phi_inputs(
            "step", phi, [result],
        )

        assert selected == [incoming]
        assert current_identity_book().page(
            "record_return_phi_input_concordance"
        ).latest(("step", 20, "limits", 0, "return_edge")) == (
            20, 10, "merged_descriptor_self_candidate_rejected",
        )
    finally:
        end_identity_book(token)


def test_alias_application_retains_value_when_resident_is_future_return_phi():
    from src.compiler.identity_concordance import (
        begin_identity_book,
        current_identity_book,
        end_identity_book,
    )

    source = SSAValue(10, dtype="float64")
    condition = SSAValue(11, dtype="bool")
    merged = SSAValue(20, dtype="float64")
    use = Instr("Call", [source], None, attributes={"callee": "consume"})
    branch = Instr(
        "Br", [], None, attributes={"target": "function_exit"},
    )
    phi = Instr(
        "Phi", [source], merged,
        attributes={
            "binding": "return_merge",
            "incoming_blocks": ("body",),
        },
    )
    function = Function(
        "step", [source, condition],
        {
            "entry": BasicBlock(
                "entry", [Instr(
                    "Br", [], None, attributes={"target": "body"},
                )], successors=["body"],
            ),
            "body": BasicBlock(
                "body", [use, branch], successors=["function_exit"],
            ),
            "function_exit": BasicBlock(
                "function_exit", [phi, Instr("Ret", [merged], None)],
            ),
        },
    )
    _book, token = begin_identity_book()
    try:
        receipts = _apply_concorded_function_aliases(
            function, {10: 20},
        )

        assert use.args == [source]
        assert phi.args == [source]
        facts = current_identity_book().page(
            "alias_application_concordance"
        )
        assert facts.latest(("step", "body", 0, 0))[3] == (
            "resident_not_available_at_use"
        )
        assert facts.latest(("step", "function_exit", 0, 0))[3] == (
            "resident_not_available_at_use"
        )
        assert len(receipts) == 2
    finally:
        end_identity_book(token)


def test_record_phi_temporal_fallback_repairs_earlier_use_and_self_edge():
    initial = SSAValue(10, dtype="float64")
    result = SSAValue(20, dtype="float64")
    use = Instr("Call", [result], None, attributes={"callee": "consume"})
    phi = Instr(
        "Phi", [result], result,
        attributes={
            "record_field_phi": True,
            "initial_value_id": 10,
            "incoming_blocks": ("body",),
        },
    )
    function = Function(
        "step", [initial],
        {
            "entry": BasicBlock(
                "entry", [Instr("Br", [], None, attributes={"target": "body"})],
                successors=["body"],
            ),
            "body": BasicBlock(
                "body", [use, Instr(
                    "Br", [], None, attributes={"target": "function_exit"},
                )], successors=["function_exit"],
            ),
            "function_exit": BasicBlock(
                "function_exit", [phi, Instr("Ret", [result], None)],
            ),
        },
    )

    receipts = repair_non_dominating_record_phi_uses(function)

    assert use.args == [initial]
    assert phi.args == [initial]
    assert len(receipts) == 2


def test_public_source_compiler_reports_progress_by_default(monkeypatch, capsys):
    import src.compiler.fortran_c_shell as shell

    def fake_lower(*_args, progress, **_kwargs):
        progress("test phase")
        return "module", {}, ()

    monkeypatch.setattr(shell, "_lower_ast_source_to_ssa_impl", fake_lower)

    assert shell.lower_ast_source_to_ssa("pass", extraction_contract=CONTRACT) == ("module", {}, ())
    assert "[compiler] test phase" in capsys.readouterr().err


def test_dead_conceptual_record_recognition_uses_field_accounting_not_dtype():
    conceptual = SSAValue(7, dtype="float64")
    physical = SSAValue(
        7, dtype="float64",
        accounting={
            "program_abi_record": "Metrics",
            "program_abi_parameter": "metrics",
            "program_abi_field": "max_vel",
            "program_abi_storage": "scalar",
        },
    )

    assert _is_dead_conceptual_record_argument(conceptual, {7}, set())
    assert not _is_dead_conceptual_record_argument(physical, {7}, set())
    assert not _is_dead_conceptual_record_argument(conceptual, {7}, {7})


def test_unused_authored_record_handle_is_pruned_after_fields_expand():
    conceptual = SSAValue(
        2, dtype="float64",
        accounting={"program_abi_parameter": "metrics"},
    )
    physical = SSAValue(
        8, dtype="float64",
        accounting={
            "program_abi_parameter": "metrics",
            "program_abi_field": "max_vel",
            "program_abi_storage": "scalar",
        },
    )
    callee = Function(
        "consume", [conceptual, physical],
        {"entry": BasicBlock("entry", [Instr("Ret", [physical], None)])},
        metadata={
            "parameter_names": (("metrics", 2),),
            "parameter_record_abi": {"metrics": {"identity": "Metrics"}},
        },
    )
    call = Instr(
        "Call", [SSAValue(102), SSAValue(108, dtype="float64")], SSAValue(200),
        attributes={"callee": "consume", "callee_input_ids": (2, 8)},
    )
    caller = Function(
        "root", [], {"entry": BasicBlock("entry", [call])},
    )

    assert _prune_unused_callee_formals({"root": caller, "consume": callee}) == 1
    assert [value.id for value in callee.args] == [8]
    assert [value.id for value in call.args] == [108]
    assert call.attributes["callee_input_ids"] == (8,)


def test_dead_generated_entry_formal_needs_no_external_operand():
    authored = SSAValue(1, dtype="float64")
    identity_result = SSAValue(5, dtype="float64")
    root = Function(
        "root", [authored, identity_result],
        {"entry": BasicBlock("entry", [Instr("Ret", [authored], None)])},
        metadata={"parameter_names": (("value", 1),)},
    )

    assert _prune_dead_entry_field_aliases({"root": root}, {}) == 1
    assert [value.id for value in root.args] == [1]


def test_declared_span_storage_survives_distinct_call_frame_occurrences():
    first = SSARecordFieldDescriptor(
        "pub_exchange_time", SSARecordFieldStorage.SPAN,
        storage_identity="Metrics.pub_exchange_time", value_ids=(101,),
        dtype="float64",
    )
    second = SSARecordFieldDescriptor(
        "pub_exchange_time", SSARecordFieldStorage.SPAN,
        storage_identity="Metrics.pub_exchange_time", value_ids=(202,),
        dtype="float64",
    )
    scalar_version = SSARecordFieldDescriptor(
        "pub_exchange_time", SSARecordFieldStorage.SCALAR,
        storage_identity="Metrics.pub_exchange_time", value_ids=(202,),
        dtype="float64",
    )

    assert _same_declared_span_storage(first, second)
    assert not _same_declared_span_storage(first, scalar_version)


def _pursued_tail_retry(value):
    if value <= 0:
        return value
    return _pursued_tail_retry(value - 1)


def test_linker_rebinds_only_proven_ordered_views_of_freshened_storage():
    placeholder = SSAValue(7, dtype="ssa.aggregate")
    replacement = SSAValue(41, dtype="ssa.aggregate")
    ordered_view = SSAValue(
        7,
        dtype="float64",
        shape=(2, 4),
        accounting={
            "ssa_storage_alias": 7,
            "ssa_region_feed": (3, 0),
        },
    )
    unrelated_collision = SSAValue(7, dtype="float64", shape=(8,))

    rebound = _rebind_linked_storage_alias(
        ordered_view, placeholder, replacement,
    )

    assert rebound is not ordered_view
    assert rebound.id == replacement.id
    assert rebound.dtype == "float64"
    assert rebound.shape == (2, 4)
    assert rebound.accounting["ssa_storage_alias"] == replacement.id
    assert rebound.accounting["ssa_linked_storage_from"] == placeholder.id
    assert is_storage_view(rebound, replacement)
    assert _rebind_linked_storage_alias(
        unrelated_collision, placeholder, replacement,
    ) is unrelated_collision
    assert _rebind_linked_storage_alias(
        placeholder, placeholder, replacement,
    ) is replacement

    loop_carried_view = SSAValue(
        7,
        dtype="float64",
        accounting={
            "ssa_storage_alias": 7,
            "ssa_region_feed": (4, 1),
            "ssa_loop_carried_feed": 34,
        },
    )
    assert _rebind_linked_storage_alias(
        loop_carried_view, placeholder, replacement,
    ) is loop_carried_view


def test_linker_fresh_ids_ignore_integer_frame_payloads():
    assert _frame_binding_value_ids((
        (1, "caller_value", 7),
        (2, "caller_storage", 11),
        (3, "caller_alias", 13),
        (4, "caller_literal", 1_503_435_042_417),
        (5, "default_literal", 99),
    )) == (7, 11, 13)
    assert _monotonic_ssa_ids((3, 21, 1_503_435_042_417, "node")) == (
        3, 21,
    )


def test_linker_prefers_retained_parameter_over_structural_identity_history():
    assert _retained_parameter_identity(
        "dt",
        (10, 11),
        (("ref", 299), ("dt", 11), ("state", 32)),
    ) == 11


def test_sympy_process_graph_is_registered_before_python_dependency_pursuit():
    x = sympy.Symbol("x")
    symbolic = compile_sympy_equations((
        sympy.Eq(sympy.Symbol("out"), x + 1, evaluate=False),
    ), name="linked_math")
    root = ProcessGraph(materialize_memory=False)
    references = link_process_graph_functions(
        root, {"linked_math": symbolic.process_graph},
    )
    with contextlib.redirect_stdout(io.StringIO()):
        root.build_from_ast(
            ast.parse("def root(x):\n    return linked_math(x)\n"),
            resolve_unresolved_parents=True,
        )
    reduce_abstract_tensor_topology(root)

    callee = root.function_table.entry("linked_math")
    caller = root.function_table.entry("root").graph
    call = next(
        data for _node, data in caller.G.nodes(data=True)
        if data.get("op") == "Call"
    )
    assert callee.graph is symbolic.process_graph
    assert callee.metadata["source_language"] == "sympy"
    assert call["attributes"]["callee_ref"] == references["linked_math"]
    assert all(role != "callee" for _parent, role in call["parents"])
    assert tuple(callee.graph.G.graph["function_parameters"]) == tuple(
        symbolic.input_ids
    )


def test_direct_source_to_ssa_preserves_linked_sympy_function_without_fusion():
    x = sympy.Symbol("x")
    symbolic = compile_sympy_equations((
        sympy.Eq(sympy.Symbol("out"), x + 1, evaluate=False),
    ), name="linked_math")

    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(x):\n    return linked_math(x)\n",
        "root",
        linked_process_graphs={"linked_math": symbolic.process_graph},
        name="root_direct",
        extraction_contract=CONTRACT,
    )

    assert "root_direct__root" in module.functions
    assert "root_direct__linked_math" in module.functions
    assert any(
        name.startswith("root_direct__linked_math__planned_region_")
        for name in module.functions
    )
    assert outputs["root_direct__root"]
    assert exports == ("root_direct__root", "root_direct__linked_math")


def test_native_boundary_is_forwarded_through_shell_external_reference_abi():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "import _pickle\n"
        "def root(payload):\n"
        "    _pickle.loads(payload)\n"
        "    return 1\n",
        "root",
        name="native_boundary_forwarding",
        extraction_contract=CONTRACT,
    )

    shell_io = module.metadata["shell_io"]
    request, = shell_io["requirements"]["requests"]
    assert request == {
        "capability": "host_references",
        "optional": False,
        "attributes": {
            "execution": "shell_io.external_references",
            "shell_abi": "turing-shell-io-abi.external_references",
        },
    }
    assert shell_io["external_reference_plan_schema"] == (
        "turing.shell-external-reference-plan.v1"
    )
    plan, = shell_io["external_reference_plans"]
    assert plan["identity"] == "_pickle.loads"
    assert plan["loader"] == "existing_module"
    assert plan["symbol_resolution"] == "in_place"
    assert plan["external_domain"] == "host_system"
    assert plan["native_abi"] == "cpython-c-api"
    assert plan["runtime_owner"] == "shell"
    assert plan["shell_profiles"] == ("python", "cpython-c")
    assert plan["shell_abi"] == (
        "turing-shell-io-abi.external_references"
    )
    assert shell_io["external_reference_occurrence_schema"] == (
        "turing.shell-external-reference-occurrence.v1"
    )
    occurrence, = shell_io["external_reference_occurrences"]
    assert occurrence["identity"] == "_pickle.loads"
    assert occurrence["owner"] == "root"
    assert len(occurrence["argument_value_ids"]) == 1
    assert occurrence["result_value_id"] is not None
    assert occurrence["operations"] == ("resolve", "call", "release")
    assert occurrence["object_policy"] == "shell-owned-opaque-handles"
    external_calls = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("external_reference")
    ]
    call, = external_calls
    assert call.attributes["external_identity"] == "_pickle.loads"
    assert call.attributes["shell_abi"] == (
        "turing-shell-io-abi.external_references"
    )
    assert len(call.args) == 1
    assert call.res.dtype == "opaque_ref"
    assert [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("extraction_identity") == "_pickle.loads"
    ] == [call]


def test_repository_ssa_executes_pickle_through_the_recorded_shell_abi():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "import _pickle\n"
        "def root(payload):\n"
        "    return _pickle.loads(payload)\n",
        "root",
        name="native_pickle_execution",
        extraction_contract=CONTRACT,
    )
    function_name = "native_pickle_execution__root"
    function = module.functions[function_name]
    payload_id = int(function.args[0].id)
    expected = {"native-link": [3, 5, 8]}
    result = SSAReferenceEvaluator(module).run(
        function_name, {payload_id: _pickle.dumps(expected)}
    )
    assert result.returned == (expected,)
    assert result.values[int(outputs[function_name][0].id)] == expected


def test_repository_ssa_passes_file_object_handle_to_native_pickle_load():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "import _pickle\n"
        "def root(stream):\n"
        "    return _pickle.load(stream)\n",
        "root",
        name="native_pickle_file_execution",
        extraction_contract=CONTRACT,
    )
    function_name = "native_pickle_file_execution__root"
    function = module.functions[function_name]
    stream_id = int(function.args[0].id)
    expected = ("file-object-handle", {"generation": 144})
    result = SSAReferenceEvaluator(module).run(
        function_name,
        {stream_id: io.BytesIO(_pickle.dumps(expected))},
    )
    assert result.returned == (expected,)


def test_direct_source_to_ssa_preserves_all_linked_tuple_results():
    x = sympy.Symbol("x")
    symbolic = compile_sympy_equations((
        sympy.Eq(sympy.Symbol("incremented"), x + 1, evaluate=False),
        sympy.Eq(sympy.Symbol("doubled"), x * 2, evaluate=False),
    ), name="linked_pair")

    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root(x):\n    shifted = x + 3\n"
        "    incremented, doubled = linked_pair(shifted)\n"
        "    return incremented, doubled\n",
        "root",
        linked_process_graphs={"linked_pair": symbolic.process_graph},
        name="pair_direct",
        extraction_contract=CONTRACT,
    )

    root = module.functions["pair_direct__root"]
    returned = tuple(
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )[-1]
    assert len(returned) == 2
    assert len(outputs["pair_direct__root"]) == 2
    linked_call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        # The linked call may target a specialization of ``linked_pair``; it
        # is the one source-linked call, identified by that declaration.
        if instruction.op == "Call"
        and instruction.attributes.get("source_linked")
        and instruction.attributes.get("result_convention") == "ssa.aggregate"
    )
    callsite_id = int(linked_call.attributes["plan_callsite_id"])
    assert all(int(argument.id) != callsite_id for argument in root.args)


def test_direct_source_materializes_declared_record_parameter_fields():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root(state, value):\n    return value + state.dx\n",
        "root",
        name="record_parameter",
        extraction_contract=CONTRACT,
    )

    root = module.functions["record_parameter__root"]
    dx = next(
        value for value in root.args
        if (value.accounting or {}).get("program_abi_field") == "dx"
    )
    assert dx.dtype == "float64"
    assert outputs[root.name]
    record = module.record_tables[root.name].records[
        next(iter(module.record_tables[root.name].records))
    ]
    field = next(item for item in record.fields if item.name == "dx")
    assert field.value_ids == (dx.id,)
    assert field.writable is False


def test_record_field_assignment_is_a_real_inout_value():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def update(state, value):\n"
        "    state.last_wave_speed = value + 1.0\n"
        "    return state.last_wave_speed\n\n"
        "def root(state, value):\n"
        "    return update(state, value)\n",
        "root",
        name="record_field_inout",
        extraction_contract=CONTRACT,
    )

    update = module.functions["record_field_inout__update"]
    record = next(
        item for item in module.record_tables[update.name].records.values()
        if item.identity.endswith(".SymbolicFluidGridState")
    )
    field = next(
        item for item in record.fields if item.name == "last_wave_speed"
    )
    assert field.writable is True
    assert any(
        argument.id in field.value_ids
        and any(
            instruction.res is argument
            for block in update.blocks.values()
            for instruction in block.instrs
        )
        for argument in update.args
    )


def test_write_only_scalar_record_field_materializes_its_exact_value():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def update(state, value):\n"
        "    state.last_wave_speed = (value * value).sqrt()\n"
        "    return value\n\n"
        "def root(state, value):\n"
        "    return update(state, value)\n",
        "root",
        name="write_only_record_field_inout",
        extraction_contract=CONTRACT,
    )

    update = module.functions["write_only_record_field_inout__update"]
    record = next(
        item for item in module.record_tables[update.name].records.values()
        if item.identity.endswith(".SymbolicFluidGridState")
    )
    field = next(
        item for item in record.fields if item.name == "last_wave_speed"
    )
    formal = next(
        argument for argument in update.args
        if int(argument.id) in set(map(int, field.value_ids))
    )
    assert field.writable is True
    assert formal.accounting["program_abi_field_written"] is True
    assert formal.accounting["program_abi_field"] == "last_wave_speed"


def test_consecutive_scalar_record_writes_share_the_parameter_owner():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def update(state, value):\n"
        "    state.last_wave_speed = value + 1.0\n"
        "    state.last_height_violation = value + 2.0\n"
        "    return state.last_wave_speed + state.last_height_violation\n\n"
        "def root(state, value):\n"
        "    return update(state, value)\n",
        "root",
        name="consecutive_record_field_inout",
        extraction_contract=CONTRACT,
    )

    update = module.functions["consecutive_record_field_inout__update"]
    record = next(
        item for item in module.record_tables[update.name].records.values()
        if item.identity.endswith(".SymbolicFluidGridState")
    )
    fields = {
        field.name: field
        for field in record.fields
        if field.name in {"last_wave_speed", "last_height_violation"}
    }
    assert tuple(fields) == ("last_wave_speed", "last_height_violation")
    assert all(field.writable for field in fields.values())
    assert all(len(field.value_ids) == 1 for field in fields.values())
    for name, field in fields.items():
        arguments = [
            argument for argument in update.args
            if (argument.accounting or {}).get("program_abi_field") == name
        ]
        assert len(arguments) == 1
        assert int(arguments[0].id) == int(field.value_ids[0])


def test_callsite_specializes_generic_state_parameter_to_caller_record():
    """Concrete record identity crosses calls even when parameter names differ."""

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "ExactState": {
                    "identity": "tests.ExactState",
                    "fields": {
                        "metric": {
                            "storage": "scalar",
                            "dtype": "float64",
                            "mutable": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "*",
                "parameter": "material",
                "record": "ExactState",
            }],
            "values": [],
        })
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def mutate(material, value):\n"
        "    material.metric = value + 1.0\n"
        "    return material.metric\n\n"
        "def relay(state, value, operation):\n"
        "    return operation(state, value)\n\n"
        "def root(material, value):\n"
        "    return relay(material, value, mutate)\n",
        "root",
        name="generic_record_specialization",
        extraction_contract=contract,
    )

    relays = [
        (name, function)
        for name, function in module.functions.items()
        if "__relay__specialized_" in name
    ]
    assert len(relays) == 1
    relay_name, relay = relays[0]
    receipt = relay.metadata.get("parameter_record_abi", {}).get("state")
    assert receipt is not None
    assert receipt["identity"] == "tests.ExactState"
    record = next(
        item for item in module.record_tables[relay_name].records.values()
        if item.identity == "tests.ExactState"
    )
    field = next(item for item in record.fields if item.name == "metric")
    assert field.writable is True
    metric_arguments = [
        argument for argument in relay.args
        if (argument.accounting or {}).get("program_abi_field") == "metric"
    ]
    assert len(metric_arguments) == 1
    assert int(metric_arguments[0].id) == int(field.value_ids[0])


def test_optional_record_field_materializes_presence_and_payload_slots(tmp_path):
    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "Limits": {
                    "identity": "tests.Limits",
                    "fields": {
                        "floor": {
                            "storage": "scalar",
                            "dtype": "float64",
                            "optional": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "root",
                "parameter": "limits",
                "record": "Limits",
            }],
            "values": [],
        })
    )
    resolved = []
    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(limits):\n"
        "    if limits.floor is None:\n"
        "        return 1.0\n"
        "    return limits.floor\n",
        "root",
        name="optional_record_boundary",
        extraction_contract=contract,
        resolved_process_graph_sink=resolved.append,
    )

    from src.compiler.process_graph_value_ids import next_process_value_id

    resolved_root = next(
        entry.graph
        for entry in resolved[0].function_table
        if entry.name == "root"
    )
    reserved_id = next_process_value_id(resolved_root)
    assert reserved_id not in resolved_root.G

    root = module.functions["optional_record_boundary__root"]
    payloads = [
        argument for argument in root.args
        if (argument.accounting or {}).get("program_abi_optional_payload")
    ]
    presences = [
        argument for argument in root.args
        if (argument.accounting or {}).get("program_abi_optional_presence")
    ]
    assert len(payloads) == len(presences) == 1
    assert payloads[0].dtype == "float64"
    assert presences[0].dtype == "bool"
    assert presences[0].accounting["physical_dtype"] == "bool"
    assert presences[0].accounting["physical_dtype_provenance"] == (
        "program_abi_optional_presence"
    )
    assert presences[0].accounting["physical_dtype_tie_policy"] == "incumbent"
    record = next(
        descriptor
        for descriptor in module.record_tables[root.name].records.values()
        if descriptor.identity == "tests.Limits"
    )
    fields = {field.name: field for field in record.fields}
    assert fields["floor"].value_ids == (payloads[0].id,)
    assert fields["floor.__present"].value_ids == (presences[0].id,)

    from types import SimpleNamespace
    import numpy as np
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.vehicle_python_compilation import (
        VehiclePythonSSALowering,
        _managed_native_feeds_by_id,
    )

    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "optional_record_boundary", optimization="O0")
    lowered = VehiclePythonSSALowering(module, root.name, outputs, exports)
    result_id = outputs[root.name][0].id
    for value, expected in ((None, 1.0), (0.0, 0.0), (2.5, 2.5)):
        feeds = _managed_native_feeds_by_id(
            lowered, {"limits": SimpleNamespace(floor=value)},
        )
        execution = artifact.prepare_execution(feeds).run()
        assert np.asarray(execution.buffers[result_id]).item() == expected


def test_missing_aggregate_leaf_repair_is_exact_and_idempotent():
    from src.compiler.glsl_deployment_strategy import (
        _repair_missing_aggregate_leaf_projections,
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        type="Call",
        op="call",
        value_id=0,
        parents=[],
        children=[],
        attributes={
            "producer_kind": "aggregate",
            "aggregate_kind": "tuple",
            "aggregate_leaf_value_ids": (4, 6),
            "tensor_output_descriptors": (
                {"shape": (), "dtype": "float64"},
                {"shape": (), "dtype": "bool"},
            ),
        },
    )

    assert _repair_missing_aggregate_leaf_projections(graph) == 2
    leaves = tuple(
        graph.G.nodes[0]["attributes"]["aggregate_leaf_value_ids"]
    )
    assert len(leaves) == 2
    assert all(leaf in graph.G for leaf in leaves)
    assert [graph.G.nodes[leaf]["tensor"]["dtype"] for leaf in leaves] == [
        "float64", "bool",
    ]
    receipt = graph.G.nodes[0]["attributes"][
        "aggregate_leaf_republication"
    ]
    assert receipt["replacements"] == ((4, leaves[0]), (6, leaves[1]))
    assert receipt["tie_policy"] == "incumbent"
    assert _repair_missing_aggregate_leaf_projections(graph) == 0
    assert tuple(
        graph.G.nodes[0]["attributes"]["aggregate_leaf_value_ids"]
    ) == leaves


def test_mutable_optional_record_write_marks_presence_before_later_test(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.vehicle_python_compilation import (
        VehiclePythonSSALowering,
        _managed_native_feeds_by_id,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "State": {
                    "identity": "tests.State",
                    "fields": {
                        "limit": {
                            "storage": "scalar",
                            "dtype": "float64",
                            "optional": True,
                            "mutable": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "root",
                "parameter": "state",
                "record": "State",
            }],
            "values": [{
                "function": "root",
                "parameter": "value",
                "storage": "scalar",
                "dtype": "float64",
                "rank": 0,
                "python_type": "builtins.float",
            }],
        })
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(state, value):\n"
        "    state.limit = value\n"
        "    if state.limit is not None:\n"
        "        return state.limit\n"
        "    return -1.0\n",
        "root",
        name="mutable_optional_record_boundary",
        extraction_contract=contract,
    )

    root = module.functions["mutable_optional_record_boundary__root"]
    presence_writes = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("optional_presence_write")
    ]
    assert any(instruction.op == "Const" for instruction in presence_writes)
    assert any(instruction.op == "Store" for instruction in presence_writes)
    presence_arguments = [
        argument
        for argument in root.args
        if (argument.accounting or {}).get(
            "program_abi_optional_presence"
        )
    ]
    assert len(presence_arguments) == 1
    assert presence_arguments[0].accounting["physical_dtype"] == "bool"
    assert all(
        instruction.res is None
        or instruction.res is presence_arguments[0]
        or int(instruction.res.id) != int(presence_arguments[0].id)
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert root.metadata["optional_program_abi_materializations"] == ({
        "parameter": "state",
        "field": "limit",
        "payload_value_ids": tuple(
            argument.id
            for argument in root.args
            if (argument.accounting or {}).get(
                "program_abi_optional_payload"
            )
        ),
        "requested_presence_value_ids": tuple(
            argument.id
            for argument in root.args
            if (argument.accounting or {}).get(
                "program_abi_optional_presence"
            )
        ),
        "presence_value_ids": tuple(
            argument.id
            for argument in root.args
            if (argument.accounting or {}).get(
                "program_abi_optional_presence"
            )
        ),
        "mutable": True,
        "field_written": True,
        "presence_write_count": 1,
    },)

    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(
        tmp_path / "mutable_optional_record_boundary",
        optimization="O0",
    )
    lowered = VehiclePythonSSALowering(module, root.name, outputs, exports)
    result_id = outputs[root.name][0].id
    for initial, assigned in ((None, 2.5), (0.0, 3.5)):
        feeds = _managed_native_feeds_by_id(lowered, {
            "state": SimpleNamespace(limit=initial),
            "value": assigned,
        })
        execution = artifact.prepare_execution(feeds).run()
        assert np.asarray(execution.buffers[result_id]).item() == assigned


def test_loop_carried_record_callback_writes_caller_field_storage():
    """A loop-carried record is an alias, not a private callback record."""

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "ExactState": {
                    "identity": "tests.ExactState",
                    "fields": {
                        "metric": {
                            "storage": "scalar",
                            "dtype": "float64",
                            "mutable": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "*",
                "parameter": "material",
                "record": "ExactState",
            }],
            "values": [],
        })
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def mutate(material, value):\n"
        "    material.metric = value + 1.0\n"
        "    return material.metric\n\n"
        "def relay(state, value, operation):\n"
        "    index = 0\n"
        "    while index < 1:\n"
        "        operation(state, value)\n"
        "        index += 1\n"
        "    return state.metric\n\n"
        "def root(material, value):\n"
        "    relay(material, value, mutate)\n"
        "    return material.metric\n",
        "root",
        name="loop_carried_record_callback",
        extraction_contract=contract,
    )

    root = module.functions["loop_carried_record_callback__root"]
    metric_arguments = [
        argument for argument in root.args
        if (argument.accounting or {}).get("program_abi_field") == "metric"
    ]
    assert len(metric_arguments) == 1
    assert (metric_arguments[0].accounting or {}).get(
        "program_abi_field_written"
    ) is True


def test_direct_tail_recursion_lowers_to_control_loop_not_missing_call():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    if value <= 0:\n"
        "        return value\n"
        "    return root(value - 1)\n",
        "root",
        name="tail_retry",
        extraction_contract=CONTRACT,
    )

    root = module.functions["tail_retry__root"]
    instructions = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    ]
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == root.name
        for instruction in instructions
    )
    assert any(instruction.op in {"Br", "CondBr"} for instruction in instructions)
    retry_call_block = next(
        block
        for block in root.blocks.values()
        if any(
            instruction.op == "Call"
            and instruction.attributes.get("region_index") == 1
            for instruction in block.instrs
        )
    )
    assert not retry_call_block.name.startswith("unreachable")
    retry_call_index = next(
        index
        for index, instruction in enumerate(retry_call_block.instrs)
        if instruction.op == "Call"
        and instruction.attributes.get("region_index") == 1
    )
    continue_index = next(
        index
        for index, instruction in enumerate(retry_call_block.instrs)
        if instruction.op == "Br"
        and instruction.attributes.get("source_control") == "continue"
    )
    assert retry_call_index < continue_index
    assert any(
        instruction.op == "Ret" and instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_dependency_pursuit_normalizes_direct_tail_recursion_too():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    return _pursued_tail_retry(value)\n",
        "root",
        name="pursued_tail_retry",
        python_bindings={"_pursued_tail_retry": _pursued_tail_retry},
        extraction_contract=CONTRACT,
    )

    pursued = next(
        function
        for function in module.functions.values()
        if function.name.endswith("___pursued_tail_retry")
    )
    instructions = [
        instruction
        for block in pursued.blocks.values()
        for instruction in block.instrs
    ]
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == pursued.name
        for instruction in instructions
    )
    assert any(instruction.op == "Br" for instruction in instructions)
    assert any(
        instruction.op == "Ret" and instruction.args
        for instruction in instructions
    )
    assert not any(
        block.name.startswith("unreachable")
        and any(instruction.op == "Call" for instruction in block.instrs)
        for block in pursued.blocks.values()
    )


def test_direct_tail_recursion_publishes_tuple_result_lanes():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    if value <= 0:\n"
        "        return value, value + 1\n"
        "    return root(value - 1)\n",
        "root",
        name="tuple_tail_retry",
        extraction_contract=CONTRACT,
    )

    root = module.functions["tuple_tail_retry__root"]
    returned = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert tuple(root.metadata["source_output_value_ids"])
    assert len(tuple(root.metadata["source_output_value_ids"])) == 2
    assert returned.args


def test_multi_result_call_writes_the_record_fields_read_after_assignment():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def pair(value):\n"
        "    return value + 1.0, value + 2.0\n\n"
        "def update(state, value):\n"
        "    state.last_wave_speed, state.last_height_violation = pair(value)\n"
        "    return state.last_wave_speed + state.last_height_violation\n\n"
        "def root(state, value):\n"
        "    return update(state, value)\n",
        "root",
        name="record_field_multi_result_inout",
        extraction_contract=CONTRACT,
    )

    update = module.functions["record_field_multi_result_inout__update"]
    call = next(
        instruction
        for block in update.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "record_field_multi_result_inout__pair"
    )
    output_ids = tuple(map(int, call.attributes["output_ids"]))
    record = next(
        item for item in module.record_tables[update.name].records.values()
        if item.identity.endswith(".SymbolicFluidGridState")
    )
    fields = {
        field.name: field
        for field in record.fields
        if field.name in {"last_wave_speed", "last_height_violation"}
    }
    assert tuple(fields) == ("last_wave_speed", "last_height_violation")
    assert all(field.writable for field in fields.values())
    assert tuple(field.value_ids[0] for field in fields.values()) == output_ids
    assert all(len(field.value_ids) == 1 for field in fields.values())
    # The planned region that reads both fields back is the one region call
    # in ``update``; its ordinal is the planner's, not part of this contract.
    (region_call,) = (
        instruction
        for block in update.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("region_index") is not None
    )
    assert tuple(int(argument.id) for argument in region_call.args) == output_ids


def test_direct_source_lowers_declared_record_literal_and_bool_return():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0, dt_limit=value)\n"
        "    return value > 0.0 and value < 2.0, metrics\n",
        "root",
        name="record_literal",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["record_literal__root"]
    records = module.record_tables[root.name].records
    metrics = next(
        record for record in records.values()
        if record.identity.endswith(".Metrics")
    )
    assert {field.name for field in metrics.fields} >= {
        "max_vel", "max_flux", "div_inf", "mass_err", "hard_failure",
    }
    layouts = dict(root.metadata["record_return_layouts"])
    assert metrics.record_id in layouts
    assert len(outputs[root.name]) == 1 + len(layouts[metrics.record_id])
    assert any(
        instruction.op in {"LAnd", "Select"}
        and instruction.attributes.get("semantic_family") == "logical_and"
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_linked_record_result_expands_as_typed_sequence_row():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def make(value):\n"
        "    return Metrics(max_vel=value, max_flux=value, div_inf=value, "
        "mass_err=value, osc_flag=value, stiff_flag=value, sim_frame=value, "
        "proc_ms=value, dt_limit=value, "
        "error_channels={'residual': value}, hard_failure=value, "
        "advanced_dt=value, unresolved_report=[])\n\n"
        "def root(value):\n"
        "    rows: list[Metrics] = []\n"
        "    metrics = make(value)\n"
        "    rows.append(metrics)\n"
        "    return value\n",
        "root",
        name="linked_record_row_append",
        python_bindings={"Metrics": Metrics},
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["linked_record_row_append__root"]
    assert "unresolved_record_sequence_rows" not in root.metadata
    append = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
        == "append_child_copy"
    )
    helper = module.functions[append.attributes["callee"]]
    assert len(append.args) == len(helper.args)
    assert append.attributes["ssa_sequence_operation"] == "append_child_copy"
    destination = module.sequence_tables[root.name].by_id(1)
    assert destination.child_table_pool is not None
    assert destination.child_table_pool.handle_column == 16
    assert "ssa_deferred_record_row" not in append.attributes
    assert append.attributes["ssa_record_row_identity"] == "Metrics"


def test_looped_multi_result_record_append_reuses_authored_call_projection():
    source = "\n\n".join((
        "from src.common.tensors import AbstractTensor",
        inspect.getsource(_scalar),
        inspect.getsource(coerce_metrics),
        "def pair(metrics, value):\n"
        "    metrics = coerce_metrics(metrics)\n"
        "    return metrics, value + 1.0, value + 2.0\n",
        "def root(metrics, value, attempts):\n"
        "    rows: list[Metrics] = []\n"
        "    count = 0\n"
        "    total = 0.0\n"
        "    while count < attempts:\n"
        "        metrics, next_value, used_value = pair(metrics, value)\n"
        "        if metrics.max_vel > 0.0:\n"
        "            rows.append(metrics)\n"
        "        total = total + used_value\n"
        "        count = count + 1\n"
        "    return total\n",
    ))
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source,
        "root",
        name="looped_multi_result_record_append",
        python_bindings={"Metrics": Metrics},
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["looped_multi_result_record_append__root"]
    assert "unresolved_record_sequence_rows" not in root.metadata
    assert _undefined_repository_ssa_operands(module) == ()
    call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "looped_multi_result_record_append__pair"
    )
    call_record = next(
        record for record in module.call_table[root.name]
        if record.callee_symbol == "looped_multi_result_record_append__pair"
    )
    record_result_id = int(call_record.result_bindings[0][1])
    assert record_result_id in module.record_tables[root.name].records
    assert call.attributes["output_ids"]


def test_linked_returned_record_mapping_keeps_authored_capacity():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def make(value):\n"
        "    return Metrics(max_vel=value, max_flux=value, div_inf=value, "
        "mass_err=value, osc_flag=value, stiff_flag=value, sim_frame=value, "
        "proc_ms=value, dt_limit=value, "
        "error_channels={'residual': value}, "
        "hard_failure=value, advanced_dt=value, unresolved_report=[])\n\n"
        "def root(value):\n"
        "    metrics = make(value)\n"
        "    return metrics.error_channels.get('residual', 0.0)\n",
        "root",
        name="linked_returned_mapping_capacity",
        python_bindings={"Metrics": Metrics},
        extraction_contract=KEYED_CONTRACT,
    )

    make = module.functions["linked_returned_mapping_capacity__make"]
    root = module.functions["linked_returned_mapping_capacity__root"]
    make_bounds = dict(make.metadata["static_mapping_capacity_bounds"])
    assert 1 in make_bounds.values()

    assert module.sequence_tables[root.name].sequences


def test_exact_returned_record_sequence_is_propagation_source():
    columns = tuple(
        SSAValue(index, accounting={
            "program_abi_record": "Metrics",
            "program_abi_field": f"error_channels.{part}",
            "returned_record_storage": "advance",
        })
        for index, part in enumerate(("keys", "values"), start=1)
    )
    assert _linked_sequence_propagation_kind(columns) == (
        "exact_returned_record_storage"
    )
    assert _linked_sequence_propagation_kind((SSAValue(3),)) is None


def test_record_packaging_call_preserves_exact_mapping_capacity():
    graph = nx.DiGraph()
    mapping = ast.parse("{'first': 1.0, 'second': 2.0, 'third': 3.0}").body[
        0
    ].value
    call = ast.parse("Metrics(error_channels=channels)").body[0].value
    graph.add_node(1, type="Aggregate", op="aggregate", expr_obj=mapping)
    graph.add_node(
        2, type="Call", op="call", expr_obj=call,
        parents=((1, "kw:error_channels"),),
    )

    assert _static_mapping_capacity_bounds(graph) == {}
    assert _static_mapping_capacity_bounds(
        graph, nonmutating_call_ids=(2,),
    ) == {1: 3}


def test_record_return_call_refreshes_completed_physical_field_surface():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def child(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0, dt_limit=value)\n"
        "    return value > 0.0, metrics\n\n"
        "def root(value):\n"
        "    ok, metrics = child(value)\n"
        "    return ok, metrics\n",
        "root",
        name="record_return_call",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["record_return_call__root"]
    child = module.functions["record_return_call__child"]
    linked = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("source_linked")
        and instruction.attributes.get("callee") == child.name
    )
    child_outputs = next(
        instruction.args
        for block in child.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert linked.attributes["result_convention"] == "ssa.aggregate"
    assert len(linked.attributes["output_ids"]) == len(child_outputs)
    assert len(linked.args) == len(child.args)
    projections = {
        int(instruction.attributes["source_output_id"]): instruction.res
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Load"
        and instruction.attributes.get("source_output_id") is not None
    }
    for child_output, caller_output_id in zip(
        child_outputs, linked.attributes["output_ids"]
    ):
        assert projections[int(caller_output_id)].id == int(caller_output_id)
        assert projections[int(caller_output_id)].dtype == child_output.dtype
    record = next(iter(module.call_table[root.name]))
    assert record.resolution == "native_call"


def test_linked_scalar_call_result_inherits_callee_output_type():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def child():\n"
        "    return True\n\n"
        "def root():\n"
        "    return child()\n",
        "root",
        name="typed_scalar_call",
        extraction_contract=CONTRACT,
    )

    root = module.functions["typed_scalar_call__root"]
    child = module.functions["typed_scalar_call__child"]
    linked = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == child.name
    )

    assert outputs[child.name][0].dtype == "bool"
    assert linked.res.dtype == "bool"


def test_record_parameter_call_uses_fields_without_python_receiver_handle():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def coerce_metrics(value):\n"
        "    return value.max_vel\n\n"
        "def root(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0, dt_limit=value)\n"
        "    return coerce_metrics(metrics)\n",
        "root",
        name="record_parameter_call",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    callee = module.functions["record_parameter_call__coerce_metrics"]
    record_ids = set(module.record_tables[callee.name].records)
    assert not any(
        argument.id in record_ids
        and argument.dtype is None
        and not argument.shape
        and not argument.accounting
        for argument in callee.args
    )
    call_record = next(iter(
        module.call_table["record_parameter_call__root"]
    ))
    assert call_record.resolution == "native_call"
    linked = next(
        instruction
        for block in module.functions[
            "record_parameter_call__root"
        ].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == callee.name
    )
    assert len(linked.args) == len(callee.args)


def test_late_record_result_rebinds_aliased_fields_in_following_call_frame():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def make_metrics(value):\n"
        "    return Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0)\n\n"
        "def read_metrics(metrics):\n"
        "    return metrics.max_vel + metrics.max_flux\n\n"
        "def root(value):\n"
        "    metrics = make_metrics(value)\n"
        "    return read_metrics(metrics)\n",
        "root",
        name="late_record_alias_frame",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["late_record_alias_frame__root"]
    read = module.functions["late_record_alias_frame__read_metrics"]
    read_record = next(
        descriptor
        for descriptor in module.record_tables[read.name].records.values()
        if descriptor.identity.endswith(".Metrics")
    )
    fields = {field.name: field for field in read_record.fields}
    frame = next(
        record
        for record in module.call_table[root.name]
        if record.callee_symbol == read.name
    )
    physical = {
        int(callee_id): int(caller_id)
        for callee_id, kind, caller_id in frame.frame_bindings
        if kind == "caller_storage"
    }

    assert physical[fields["max_vel"].value_ids[0]] == physical[
        fields["max_flux"].value_ids[0]
    ]
    linked_read = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == read.name
    )
    max_vel_position = next(
        position
        for position, argument in enumerate(read.args)
        if (argument.accounting or {}).get("program_abi_field") == "max_vel"
    )
    max_flux_position = next(
        position
        for position, argument in enumerate(read.args)
        if (argument.accounting or {}).get("program_abi_field") == "max_flux"
    )
    assert linked_read.args[max_vel_position].id == physical[
        fields["max_vel"].value_ids[0]
    ]
    assert linked_read.args[max_flux_position].id == physical[
        fields["max_flux"].value_ids[0]
    ]
    assert linked_read.attributes["post_aggregate_frame_reconciled"]
    descriptor_receipts = root.metadata[
        "returned_record_descriptor_reconciliations"
    ]
    assert descriptor_receipts[0][-2:] == (
        "exact_aggregate_position", "incumbent_on_equal_priority",
    )
    aliases = set(map(int, root.metadata.get("value_aliases", {})))
    assert not any(
        int(argument.id) in aliases
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        for argument in instruction.args
    )
    assert max(
        int(value.id)
        for function in module.functions.values()
        for value in (
            *function.args,
            *(
                instruction.res
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        )
    ) < 1_000_000_000


def test_forwarded_record_parameter_reuses_only_demanded_caller_field_storage():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def child(state):\n"
        "    return state.dx\n\n"
        "def root(state):\n"
        "    return child(state)\n",
        "root",
        name="forwarded_record_parameter",
        extraction_contract=CONTRACT,
    )

    root = module.functions["forwarded_record_parameter__root"]
    child = module.functions["forwarded_record_parameter__child"]
    root_record = next(
        record for record in module.record_tables[root.name].records.values()
        if record.identity.endswith(".SymbolicFluidGridState")
    )
    child_record = next(
        record for record in module.record_tables[child.name].records.values()
        if record.identity.endswith(".SymbolicFluidGridState")
    )
    assert [field.name for field in root_record.fields] == ["dx"]
    root_dx = root_record.fields[0]
    child_dx = next(field for field in child_record.fields if field.name == "dx")
    linked = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == child.name
    )
    child_dx_index = next(
        index for index, argument in enumerate(child.args)
        if argument.id in child_dx.value_ids
    )
    assert linked.args[child_dx_index].id in root_dx.value_ids


def test_nested_record_forwarding_retries_outer_call_after_inner_linking():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def leaf(state):\n"
        "    return state.dx + 1.0\n\n"
        "def middle(state):\n"
        "    return leaf(state)\n\n"
        "def root(state):\n"
        "    return middle(state)\n",
        "root",
        name="nested_record_forwarding",
        extraction_contract=CONTRACT,
    )

    for name in (
        "nested_record_forwarding__middle",
        "nested_record_forwarding__root",
    ):
        function = module.functions[name]
        calls = tuple(module.call_table[name])
        assert len(calls) == 1
        assert calls[0].resolution == "native_call"
        assert not function.metadata.get("unresolved_call_diagnostics")
        assert any(
            instruction.op == "Call"
            for block in function.blocks.values()
            for instruction in block.instrs
        )
        assert outputs[name]


def test_post_loop_call_uses_the_exact_break_aware_loop_result_port():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def step(value):\n"
        "    return value, value, value\n\n"
        "def root(limit, value, boundaries):\n"
        "    total = 0.0\n"
        "    second = value\n"
        "    while total < limit:\n"
        "        current = value\n"
        "        for boundary in boundaries:\n"
        "            if boundary > total:\n"
        "                current = boundary - total\n"
        "                break\n"
        "        first, second, used = step(current)\n"
        "        if used <= 0:\n"
        "            break\n"
        "        total += used\n"
        "    return total, second\n",
        "root",
        name="break_aware_loop_call",
        extraction_contract=CONTRACT,
    )

    root = module.functions["break_aware_loop_call__root"]
    # ``step`` returns its argument, so the compiler folds the call through
    # its identity-return aliases; what follows the inner loop reads the
    # inner loop's break-aware result port directly.  That port merges the
    # no-break value -- ``value``, which ``current`` held before the loop,
    # NOT the outer loop's carried ``second`` that starts from the same
    # value -- with the break edge's ``boundary - total``.
    parameters = dict(root.metadata["parameter_names"])
    inner_exit = root.blocks["loop_exit"]
    loop_result = next(
        instruction for instruction in inner_exit.instrs
        if instruction.attributes.get("binding") == "loop_result_port"
    )
    assert len(loop_result.args) == 2
    assert int(loop_result.args[0].id) == int(parameters["value"])
    assert any(
        instruction.op == "Call"
        and any(
            int(argument.id) == int(loop_result.res.id)
            for argument in instruction.args
        )
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert not root.metadata.get("unresolved_call_diagnostics")
    assert not root.metadata.get("structural_output_shortfalls")


def test_structural_boolean_call_feed_is_materialized_before_linking():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def child(value, flag):\n"
        "    if flag:\n"
        "        return value\n"
        "    return -value\n\n"
        "def root(value, left, right):\n"
        "    return child(value, left or right)\n",
        "root",
        name="structural_call_feed",
        extraction_contract=CONTRACT,
    )

    root = module.functions["structural_call_feed__root"]
    calls = tuple(module.call_table[root.name])
    assert len(calls) == 1
    assert calls[0].resolution == "native_call"
    assert outputs[root.name]
    assert not root.metadata.get("unresolved_call_diagnostics")
    assert not root.metadata.get("structural_output_shortfalls")
    assert any(
        instruction.op == "LOr"
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_keyed_mapping_lowers_to_token_and_value_vectors():
    """A dict field is a length plus parallel key/value vectors, not a handle.

    The keys are words, so they lower to the repository's universal string
    tokens -- the same i64 identity a name hashed at run time produces -- which
    is why one shape serves a fixed key set and a dynamic one. As one opaque
    reference the mapping had no length to iterate and no slot to read, so
    every consumer of it was unresolvable at the backend.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return metrics.error_channels\n",
        "root",
        name="keyed_mapping",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["keyed_mapping__root"]
    slots = {
        (value.accounting or {}).get("program_abi_field"): value
        for value in root.args
        if (value.accounting or {}).get("program_abi_keyed_owner")
        == "error_channels"
    }
    assert set(slots) == {
        "error_channels.length",
        "error_channels.keys",
        "error_channels.values",
    }
    assert slots["error_channels.length"].dtype == "int64"
    # The key vector is token identities, never the words themselves.
    assert slots["error_channels.keys"].dtype == "int64"
    assert slots["error_channels.values"].dtype == "float64"
    assert slots["error_channels.length"].accounting["physical_dtype"] == "int64"
    assert slots["error_channels.keys"].accounting["physical_dtype"] == "int64"
    assert slots["error_channels.values"].accounting["physical_dtype"] == "float64"
    for name in ("error_channels.keys", "error_channels.values"):
        accounting = slots[name].accounting or {}
        assert accounting["program_abi_storage"] == "span"
        assert int(accounting["program_abi_rank"]) == 1
        assert accounting["physical_dtype_provenance"] == "program_abi_keyed_member"
        assert accounting["physical_dtype_tie_policy"] == "incumbent"

    record = module.record_tables[root.name].records[
        next(iter(module.record_tables[root.name].records))
    ]
    described = {
        field.name for field in record.fields
        if field.name.startswith("error_channels.")
    }
    assert described == set(slots)

    # The mapping keeps its own identity and names the three slots, so a
    # consumer still holding it can be resolved against them.
    mapping = next(
        value for value in root.args
        if (value.accounting or {}).get("program_abi_field")
        == "error_channels"
    )
    accounting = mapping.accounting or {}
    assert accounting["program_abi_storage"] == "keyed"
    assert accounting["program_abi_keyed_length"] == slots[
        "error_channels.length"
    ].id
    assert accounting["program_abi_keyed_keys"] == slots[
        "error_channels.keys"
    ].id
    assert accounting["program_abi_keyed_values"] == slots[
        "error_channels.values"
    ].id
    assert max(int(value.id) for value in root.args) < 1_000_000_000


def test_dynamic_dict_literal_is_populated_and_returned_with_its_record():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    return Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0, error_channels={'residual': value})\n",
        "root",
        name="dynamic_keyed_record_literal",
        python_bindings={"Metrics": Metrics},
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["dynamic_keyed_record_literal__root"]
    record = next(iter(module.record_tables[root.name].records.values()))
    keyed = {
        field.name: field
        for field in record.fields
        if field.name.startswith("error_channels.")
    }
    assert set(keyed) == {
        "error_channels.length",
        "error_channels.keys",
        "error_channels.values",
    }
    returned_ids = {int(value.id) for value in outputs[root.name]}
    assert {
        int(field.value_ids[0]) for field in keyed.values()
    }.issubset(returned_ids)
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("ssa_sequence_operation") == "add"
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert max(
        int(value.id)
        for value in (
            *root.args,
            *(
                instruction.res
                for block in root.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        )
    ) < 1_000_000_000


def test_linked_record_type_guard_prunes_terminal_normalization_tail():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0)\n"
        "    return coerce_metrics(metrics)\n",
        "root",
        name="linked_record_type_guard",
        python_bindings={
            "Metrics": Metrics,
            "coerce_metrics": coerce_metrics,
        },
        extraction_contract=CONTRACT,
    )

    callee = module.functions[
        "linked_record_type_guard__coerce_metrics"
    ]
    assert tuple(module.record_tables[callee.name].records) == ()
    returned = next(
        instruction.args
        for block in callee.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert len(returned) == 1


def test_present_physical_record_field_folds_identity_with_none():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def choose(state):\n"
        "    floor = state.dx if state.dx is not None else 1.0\n"
        "    return floor + 0.0\n\n"
        "def root(state):\n"
        "    return choose(state)\n",
        "root",
        name="physical_field_none_guard",
        extraction_contract=CONTRACT,
    )

    root = module.functions["physical_field_none_guard__choose"]
    floor_id = next(
        int(value_id)
        for name, value_id in root.metadata.get("value_names", ())
        if name == "floor"
    )
    floor_argument = next(
        argument for argument in root.args if int(argument.id) == floor_id
    )
    assert (floor_argument.accounting or {})["program_abi_field"] == "dx"


def test_selected_branch_value_replaces_external_merge_placeholder():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def choose(state, osc):\n"
        "    if state.dx is not None:\n"
        "        floor_t = state.dx + 0.0\n"
        "    value = state.dx\n"
        "    if osc:\n"
        "        value = value + floor_t\n"
        "    return value\n\n"
        "def root(state, osc):\n"
        "    return choose(state, osc)\n",
        "root",
        name="selected_branch_placeholder",
        extraction_contract=CONTRACT,
    )

    choose = module.functions["selected_branch_placeholder__choose"]
    floor_ids = tuple(
        int(value_id)
        for name, value_id in choose.metadata.get("value_names", ())
        if name == "floor_t"
    )
    assert floor_ids
    assert not any(
        int(argument.id) == floor_ids[-1]
        and not (argument.accounting or {}).get("program_abi_parameter")
        for argument in choose.args
    )


def test_mapping_iteration_walks_its_own_key_and_value_vectors():
    """``for k, v in d.items()`` indexes the mapping's declared slots.

    The loop lowering already walks an iterable as parallel columns, and a
    keyed mapping already *is* parallel key/value vectors, so the two only had
    to be recognised as the same thing. Before that the iterable and its second
    column were anonymous storage: the loop bound came from an opaque ``extent``
    call with nothing to measure, and neither projection named a slot, so every
    backend refused the whole comprehension.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        total = total + limit\n"
        "    return total\n",
        "root",
        name="mapping_iteration",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["mapping_iteration__root"]
    slot = {
        (value.accounting or {}).get("program_abi_field"): int(value.id)
        for value in root.args
        if (value.accounting or {}).get("program_abi_keyed_owner")
        == "error_channels"
    }
    instructions = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    ]

    # The opaque iterable extent is gone; the loop is bounded by the declared
    # length itself.
    assert not any(
        instruction.attributes.get("tensor_operation") == "extent"
        for instruction in instructions
    )
    condition = next(
        instruction for instruction in instructions
        if instruction.attributes.get("binding") == "loop_condition"
    )
    assert slot["error_channels.length"] in {
        int(argument.id) for argument in condition.args
    }

    # Each destructured column indexes its own vector: names from the token
    # vector, values from the value vector.
    projected = {
        int(instruction.args[0].id)
        for instruction in instructions
        if instruction.attributes.get("binding") == "projected_iterable"
        and instruction.op == "GetElementPtr"
    }
    assert projected == {
        slot["error_channels.keys"], slot["error_channels.values"],
    }

    # The anonymous iterable and its appended column are no longer arguments.
    assert not any(
        (value.accounting or {}).get("projected_row_source_id") is not None
        for value in root.args
    )


def test_defensive_mapping_items_reuse_declared_slots_and_key_tokens():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return {\n"
        "        str(name): channel\n"
        "        for name, channel in (metrics.error_channels or {}).items()\n"
        "    }\n",
        "root",
        name="defensive_mapping_iteration",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["defensive_mapping_iteration__root"]
    slots = {
        (value.accounting or {}).get("program_abi_keyed_part"): int(value.id)
        for value in root.args
        if (value.accounting or {}).get("program_abi_keyed_owner")
        == "error_channels"
    }
    instructions = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    ]
    projected = {
        int(instruction.attributes["projection"]): (
            int(instruction.args[0].id), int(instruction.res.id)
        )
        for instruction in instructions
        if instruction.op == "GetElementPtr"
        and instruction.attributes.get("binding") == "projected_iterable"
    }
    store = next(
        instruction for instruction in instructions
        if instruction.attributes.get("ssa_sequence_operation")
        == "table_store"
    )

    assert projected[0][0] == slots["keys"]
    assert projected[1][0] == slots["values"]
    key_load = next(
        instruction for instruction in instructions
        if instruction.op == "Load"
        and int(instruction.args[0].id) == projected[0][1]
    )
    assert int(store.args[-2].id) == int(key_load.res.id)
    assert not any(
        (value.accounting or {}).get("projected_row_source_id") is not None
        for value in root.args
    )
    assert any(
        receipt[-1] == "declared_keyed_mapping_iterable"
        for receipt in root.metadata["keyed_iterable_identity_receipts"]
    )
    assert any(
        receipt[-1] == "string_token_str_identity"
        for receipt in root.metadata["keyed_iterable_identity_receipts"]
    )


def test_comprehension_element_is_evaluated_inside_its_own_loop():
    """A generator's element expression is loop-owned work, not a prologue.

    A ``for`` statement claims its whole body subtree; a comprehension claimed
    only the element's root node, so every operand below it -- here the
    ``float`` cast -- was planned into a region scheduled before the loop and
    fed the target's pre-loop value.  The loop then loaded the real element
    into a value nothing read.  Both halves are silent: the program compiles
    and computes the wrong thing.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return any(\n"
        "        float(limit) > 1.0\n"
        "        for name, limit in metrics.error_channels.items()\n"
        "    )\n",
        "root",
        name="comprehension_element",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["comprehension_element__root"]
    body = next(
        block for name, block in root.blocks.items()
        if name.startswith("loop_body")
    )
    values_load = next(
        instruction for instruction in body.instrs
        if instruction.op == "Load"
        and instruction.attributes.get("binding") == "projected_iterable"
        and int(instruction.attributes.get("projection", -1)) == 1
    )
    element = int(values_load.res.id)

    # The projected value column is read by work in the same iteration ...
    consumers = [
        instruction for instruction in body.instrs
        if any(int(argument.id) == element for argument in instruction.args)
    ]
    assert consumers, "the loaded element has no consumer in the loop body"

    # ... and that work is the element expression itself, either retained as
    # direct loop-local control SSA or enclosed by its numerical region.
    direct = [instruction for instruction in consumers if instruction.op == "Gt"]
    region_calls = [
        instruction for instruction in consumers
        if instruction.op == "Call"
        and "planned_region" in str(instruction.attributes.get("callee", ""))
    ]
    assert direct or region_calls
    if region_calls:
        # The element expression may span several loop-body regions (the
        # cast in one, the comparison in the next); all of them run inside
        # the loop.
        region_ops = [
            instruction.op
            for call in body.instrs
            if call.op == "Call"
            and "planned_region" in str(call.attributes.get("callee", ""))
            for block in module.functions[
                str(call.attributes["callee"])
            ].blocks.values()
            for instruction in block.instrs
        ]
        assert "Gt" in region_ops

    # Nothing in the element expression is left as a pre-loop argument.
    entry = root.blocks["entry"]
    assert not any(
        instruction.op == "Call"
        and "planned_region" in str(instruction.attributes.get("callee", ""))
        for instruction in entry.instrs
    )


def test_comprehension_reduction_reads_the_collection_the_loop_publishes():
    """``any(...)`` consumes the loop's collection port, not the generator.

    A carried binding rewires its continuation onto its ``LoopResult``; a
    collection output did not.  The reduction kept naming the comprehension
    node, which the retained loop no longer produces, so it arrived as an
    anonymous frame slot while every published element went somewhere else.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return any(\n"
        "        float(limit) > 1.0\n"
        "        for name, limit in metrics.error_channels.items()\n"
        "    )\n",
        "root",
        name="comprehension_reduction",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["comprehension_reduction__root"]
    instructions = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    ]
    publication = next(
        instruction for instruction in instructions
        if instruction.op == "Call"
        and instruction.attributes.get("ssa_sequence_operation") == "append"
    )
    collection_id = int(publication.attributes["sequence_id"])

    def reduces(callee_name: str) -> bool:
        callee = module.functions.get(callee_name)
        return callee is not None and any(
            item.op == "any"
            for block in callee.blocks.values()
            for item in block.instrs
        )

    reduction_call = next(
        instruction for instruction in instructions
        if instruction.op == "Call"
        and reduces(str(instruction.attributes.get("callee", "")))
    )
    assert collection_id in {
        int(argument.id) for argument in reduction_call.args
    }


def test_declared_mapping_or_default_keeps_the_mapping():
    """``x or {}`` over a declared container selects, it does not combine.

    Python's ``or`` evaluates to one of its operands. For a boolean pair the
    combine and the selection are the same value, so the logical opcode still
    stands. For a declared mapping -- ``Metrics.error_channels`` is a dict --
    they are not the same at all: the combine yields a truth value and the
    mapping the field named is gone. That loss used to be invisible because the
    result was typed ``bool`` and nothing downstream could ask for the dict
    back.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return metrics.error_channels or {}\n",
        "root",
        name="reference_default",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["reference_default__root"]
    selection = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Select"
        and instruction.attributes.get("semantic_family") == "logical_or"
    ]
    assert len(selection) == 1
    mask, when_true, when_false = selection[0].args
    # Select(mask, when_true, when_false): `or` keeps the left operand when it
    # is truthy, so the mask and the true value are the same declared field.
    assert mask.id == when_true.id
    assert when_false.id != when_true.id
    assert (mask.accounting or {}).get("program_abi_field") == "error_channels"
    # A mapping keyed by words: length plus parallel token/value vectors.
    assert (mask.accounting or {}).get("program_abi_storage") == "keyed"
    # The old boolean combine is gone, and so is its `bool` result type.
    assert not any(
        instruction.op == "LOr"
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert selection[0].res.dtype != "bool"


def test_declared_mapping_or_empty_get_uses_resident_lookup_with_default():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root(metrics):\n"
        "    return float((metrics.error_channels or {}).get(\n"
        "        'dt_unresolved', 0.0\n"
        "    ))\n",
        "root",
        name="reference_default_get",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["reference_default_get__root"]
    lookups = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "lookup"
    ]
    assert outputs[root.name]
    assert len(lookups) == 1
    assert str(lookups[0].attributes.get("callee")).endswith(
        "_lookup_or_default"
    )
    helper = module.functions[str(lookups[0].attributes["callee"])]
    assert helper.args[0].accounting["physical_dtype"] == "int64"
    key_loads = [
        instruction
        for block in helper.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Load"
        and instruction.res is not None
        and instruction.res.dtype == "int64"
    ]
    assert key_loads
    assert any(
        instruction.op == "Const"
        and instruction.attributes.get("value") == 0.0
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert not root.metadata.get("structural_output_shortfalls")
    assert not root.metadata.get("unresolved_call_diagnostics")


def test_declared_mapping_lookup_owned_only_by_source_call_is_materialized():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def scalar(value):\n"
        "    return value\n\n"
        "def root(metrics):\n"
        "    channels = metrics.error_channels or {}\n"
        "    return scalar(channels['power_w'])\n",
        "root",
        name="reference_call_lookup",
        extraction_contract=KEYED_CONTRACT,
    )

    root = module.functions["reference_call_lookup__root"]
    lookups = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "lookup"
    ]
    assert len(lookups) == 1
    lookup = lookups[0]
    assert lookup.attributes["ssa_lookup_ownership"] == "source_call"
    assert lookup.attributes["ssa_lookup_owner_ids"]
    assert lookup.res.id not in {argument.id for argument in root.args}
    assert (
        lookup.res.id,
        "source_call",
        lookup.attributes["ssa_lookup_owner_ids"],
    ) in root.metadata["table_lookup_ownership_receipts"]
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "reference_call_lookup__scalar"
        and lookup.res.id in {argument.id for argument in instruction.args}
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_record_field_storage_identity_crosses_the_call_frame():
    """A declared span keeps its field identity into every callee it reaches.

    A callee's formal parameters are built before the record ABI is
    materialized, so a passed field used to arrive as an untyped scalar: the
    rank the contract declared was gone, and every address into the span
    became unresolvable at the backend. The caller's own argument binding is
    the exact carrier, so the identity travels the call frame rather than
    being re-derived from parameter names.
    """

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def inner(grid, i, j):\n"
        "    return grid[i][j]\n\n"
        "def root(state, i, j):\n"
        "    return inner(state.height, i, j)\n",
        "root",
        name="span_cross",
        extraction_contract=CONTRACT,
    )

    carried = {
        name: value
        for name, function in module.functions.items()
        for value in function.args
        if (value.accounting or {}).get("program_abi_field") == "height"
    }
    # The caller, the callee, and the callee's own planned region.
    assert set(carried) == {
        "span_cross__root",
        "span_cross__inner",
        "span_cross__inner__planned_region_0",
    }
    for name, value in carried.items():
        accounting = value.accounting or {}
        assert accounting["program_abi_storage"] == "span", name
        assert int(accounting["program_abi_rank"]) == 2, name
        assert value.dtype == "float64", name
        # The rank travels in the field identity. `shape` is the repository's
        # static element-count contract and these extents are only known at
        # call time, so naming symbolic axes there would corrupt every buffer
        # size derived from it.
        assert tuple(value.shape or ()) == (), name


def test_tensor_parameter_annotations_publish_dynamic_span_value_abi():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "import torch\n\n"
        "def root(left: torch.Tensor, right: torch.Tensor):\n"
        "    return left + right\n",
        "root",
        name="annotated_tensor_span",
        extraction_contract=ExtractionContract(CONTRACT),
    )

    root = module.functions["annotated_tensor_span__root"]
    assert tuple(root.metadata["authored_parameters"]) == ("left", "right")
    for formal in root.args[:2]:
        accounting = formal.accounting or {}
        assert accounting["program_abi_storage"] == "span"
        assert accounting["program_abi_rank"] == 1
        assert accounting["tensor_metadata_state"] == "dynamic"
        assert accounting["linked_parameter_provenance"] == (
            "exact_call_argument_value"
        )


def test_distinct_record_fields_do_not_label_one_generic_callee_formal():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def inner(grid, i, j):\n"
        "    return grid[i][j]\n\n"
        "def root(state, i, j):\n"
        "    return (inner(state.height, i, j)\n"
        "            + inner(state.momentum_x, i, j))\n",
        "root",
        name="ambiguous_span_cross",
        extraction_contract=CONTRACT,
    )

    inner = module.functions["ambiguous_span_cross__inner"]
    field_formals = [
        value for value in inner.args
        if (value.accounting or {}).get("program_abi_field") is not None
    ]
    assert field_formals == []
    assert inner.args[0].dtype == "float64"
    assert (inner.args[0].accounting or {})["program_abi_storage"] == "span"
    assert int((inner.args[0].accounting or {})["program_abi_rank"]) == 2


def test_exact_record_span_shape_reaches_callee_before_region_lowering():
    """Exact record extents must select callee tensor kernels pre-link."""

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "ExactState": {
                    "identity": "tests.ExactState",
                    "fields": {
                        "height": {
                            "storage": "span",
                            "dtype": "float64",
                            "rank": 2,
                            "shape": [2, 3],
                        },
                    },
                },
            },
            "bindings": [{
                "function": "root",
                "parameter": "state",
                "record": "ExactState",
            }],
            "values": [{
                "function": "root",
                "parameter": "indices",
                "storage": "span",
                "dtype": "int64",
                "rank": 1,
                "shape": [1],
                "python_type": (
                    "src.common.tensors.abstraction.AbstractTensor"
                ),
            }],
        })
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def inner(grid, indices):\n"
        "    view = grid.reshape((-1, 3))\n"
        "    return view.gather(indices, dim=0)\n\n"
        "def root(state, indices):\n"
        "    return inner(state.height, indices)\n",
        "root",
        name="exact_span_cross",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
    )

    carried = {
        name: value
        for name, function in module.functions.items()
        for value in function.args
        if (value.accounting or {}).get("program_abi_field") == "height"
    }
    assert "exact_span_cross__root" in carried
    assert any(
        name.startswith("exact_span_cross__inner__specialized_")
        for name in carried
    )
    assert all(value.shape == (2, 3) for value in carried.values())
    gathers = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "gather_values_double"
    ]
    assert len(gathers) == 1
    assert gathers[0].args[0].shape == (2, 3)


def test_indexed_record_field_write_keeps_one_resident_call_frame():
    """A post-IndexedStore field read is the mutated resident span."""

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "ExactState": {
                    "identity": "tests.ExactState",
                    "fields": {
                        "height": {
                            "storage": "span",
                            "dtype": "float64",
                            "rank": 2,
                            "shape": [2, 3],
                            "mutable": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "root",
                "parameter": "state",
                "record": "ExactState",
            }],
            "values": [{
                "function": "root",
                "parameter": "dt",
                "storage": "scalar",
                "dtype": "float64",
                "rank": 0,
                "python_type": "builtins.float",
            }],
        })
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def consume(grid):\n"
        "    return grid.sum()\n\n"
        "def root(state, dt):\n"
        "    state.height[:, 0] = dt\n"
        "    return consume(state.height)\n",
        "root",
        name="indexed_record_resident",
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
    )

    root = module.functions["indexed_record_resident__root"]
    height_arguments = [
        value for value in root.args
        if (value.accounting or {}).get("program_abi_field") == "height"
    ]
    assert len(height_arguments) == 1
    resident = height_arguments[0]
    assert (resident.accounting or {}).get("program_abi_field_written") is True
    assert not any(
        (value.accounting or {}).get("split_from_unproven_alias") is not None
        for value in root.args
    )
    indexed_assigns = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "index_assign_double"
    ]
    assert len(indexed_assigns) == 1
    assert indexed_assigns[0].args[0].id == resident.id
    value_count = indexed_assigns[0].args[-1]
    value_count_definition = next(
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is value_count
    )
    assert value_count_definition.attributes["constant"] == 1
    store_region = next(
        function.name
        for function in module.functions.values()
        if any(
            instruction is indexed_assigns[0]
            for block in function.blocks.values()
            for instruction in block.instrs
        )
    )
    root_callees = [
        str(instruction.attributes.get("callee") or "")
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    ]
    consume_index = next(
        index for index, callee in enumerate(root_callees)
        if "__consume" in callee
    )
    assert root_callees.index(store_region) < consume_index


def test_multi_result_call_does_not_move_before_indexed_record_write():
    """Call-frame temporary ids are not semantic pre-marker consumers."""

    from src.compiler.extraction_contract import (
        ExtractionContract,
        ProgramABIContract,
    )
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    contract = ExtractionContract(CONTRACT).with_program_abi(
        ProgramABIContract.from_mapping({
            "records": {
                "ExactState": {
                    "identity": "tests.ExactState",
                    "fields": {
                        "height": {
                            "storage": "span",
                            "dtype": "float64",
                            "rank": 2,
                            "shape": [2, 3],
                            "mutable": True,
                        },
                    },
                },
            },
            "bindings": [{
                "function": "root",
                "parameter": "state",
                "record": "ExactState",
            }],
            "values": [{
                "function": "root",
                "parameter": "dt",
                "storage": "scalar",
                "dtype": "float64",
                "rank": 0,
                "python_type": "builtins.float",
            }],
        })
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def pair(grid):\n"
        "    return grid + 1.0, grid + 2.0\n\n"
        "def root(state, dt):\n"
        "    state.height[:, 0] = dt\n"
        "    left, right = pair(state.height)\n"
        "    return left.sum() + right.sum()\n",
        "root",
        name="indexed_record_multi_result_order",
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
    )

    root = module.functions["indexed_record_multi_result_order__root"]
    calls = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    ]
    store_index = next(
        index for index, instruction in enumerate(calls)
        if "__planned_region_" in str(instruction.attributes.get("callee"))
        and any(
            nested.op == "Call"
            and nested.attributes.get("callee") == "index_assign_double"
            for block in module.functions[
                str(instruction.attributes["callee"])
            ].blocks.values()
            for nested in block.instrs
        )
    )
    pair_index = next(
        index for index, instruction in enumerate(calls)
        if "__pair" in str(instruction.attributes.get("callee"))
    )
    assert store_index < pair_index


def test_dependency_order_includes_canonical_parent_edges_absent_from_networkx():
    from src.compiler.glsl_deployment_strategy import _dependency_order

    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(1, type="Input", parents=())
    graph.G.add_node(4, type="Call", parents=((9, "after_write"),))
    graph.G.add_node(9, type="IndexedStore", parents=((1, "base"),))
    graph.G.add_edge(1, 9)

    order = _dependency_order(graph)
    assert order.index(9) < order.index(4)


def test_declared_record_span_is_a_planner_tensor_descriptor():
    from src.compiler.glsl_deployment_strategy import _tensor_descriptor

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["parameter_record_abi"] = {
        "state": {
            "identity": "tests.ExactState",
            "fields": {
                "height": {
                    "storage": "span",
                    "dtype": "float64",
                    "rank": 3,
                    "shape": [2, 3, 4],
                },
            },
        },
    }
    graph.G.add_node(
        1, type="Input", op="input",
        attributes={"binding_name": "state"},
    )
    graph.G.add_node(
        2, type="GetAttr", op="getattr",
        attributes={"attribute": "height"},
        parents=((1, "value"),),
    )

    assert _tensor_descriptor(graph, 2) == {
        "shape": (2, 3, 4),
        "dtype": "float64",
        "rank": 3,
    }


def test_single_aggregate_return_keeps_structured_tensor_descriptors():
    from src.compiler.glsl_deployment_strategy import (
        _propagate_callsite_tensor_specializations, _tensor_descriptor,
    )

    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(
            "def pack(values):\n"
            "    return {'values': values, 'label': 'wheel'}\n"
            "def root(values):\n"
            "    return pack(values)\n"
        ), resolve_unresolved_parents=True)
    reduce_abstract_tensor_topology(graph)
    caller = graph.function_table.entry("root").graph
    for _node, data in caller.G.nodes(data=True):
        if (data.get("attributes") or {}).get("binding_name") == "values":
            data["tensor"] = {"shape": (3,), "dtype": "float64"}

    _propagate_callsite_tensor_specializations(graph)

    node_id, call = next((node_id, data) for node_id, data in caller.G.nodes(data=True)
                         if data.get("op") == "Call")
    assert "tensor" not in call
    assert _tensor_descriptor(caller, node_id) is None
    descriptors = call["attributes"]["tensor_output_descriptors"]
    assert isinstance(descriptors[0], tuple)
    assert {"shape": (3,), "dtype": "float64", "rank": 1} in descriptors[0]


def test_callsite_shape_discards_padded_scalar_result_descriptors():
    from src.compiler.glsl_deployment_strategy import (
        _apply_callsite_tensor_descriptors,
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["identity_table"] = {"grid": (1,)}
    graph.G.add_node(
        1, type="Input", op="input",
        attributes={"binding_name": "grid"},
        tensor={"shape": (1, 1, 1), "dtype": "float64"},
    )
    graph.G.add_node(
        2, type="Call", op="min",
        attributes={"tensor_candidate": "min", "dim": 3},
        parents=((1, "operand"),),
        tensor={"shape": (1, 1, 1), "dtype": "float64"},
    )

    _apply_callsite_tensor_descriptors(
        graph, {"grid": {"shape": (8, 4, 128, 2), "dtype": "float64"}},
    )

    assert graph.G.nodes[1]["tensor"]["shape"] == (8, 4, 128, 2)
    assert "tensor" not in graph.G.nodes[2]


def test_shape_constant_waits_for_authoritative_callsite_descriptor():
    from src.compiler.glsl_deployment_strategy import (
        _apply_callsite_tensor_descriptors,
        _fold_callsite_structural_values,
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["identity_table"] = {"grid": (1,)}
    graph.G.add_node(
        1, type="Input", op="input",
        attributes={"binding_name": "grid"},
        parents=(),
    )
    graph.G.add_node(
        2, type="Call", op="min",
        attributes={"tensor_candidate": "min", "dim": 3},
        tensor={"shape": (1, 1, 1), "dtype": "float64"},
        parents=((1, "operand"),),
        expr_obj=ast.parse("grid.min(dim=3)", mode="eval").body,
    )
    graph.G.add_node(
        3, type="GetAttr", op="getattr",
        attributes={"attribute": "shape"},
        parents=((2, "value"),),
    )
    graph.G.add_node(
        4, type="Constant", op="const", constant=2,
        attributes={"value": 2}, parents=(),
    )
    graph.G.add_node(
        5, type="Indexed", op="indexed",
        parents=((3, "base"), (4, "index")),
    )
    graph.G.add_edges_from(((1, 2), (2, 3), (3, 5), (4, 5)))
    graph.roots = [5]

    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[5]["type"] == "Indexed"

    _apply_callsite_tensor_descriptors(
        graph, {"grid": {"shape": (8, 4, 128, 2), "dtype": "float64"}},
    )
    _fold_callsite_structural_values(graph)

    assert graph.G.nodes[5]["type"] == "Constant"
    assert graph.G.nodes[5]["constant"] == 128
    assert graph.G.nodes[5]["constant"] == 128


def test_declared_span_shape_folds_through_record_return_class():
    from src.compiler.glsl_deployment_strategy import (
        _fold_callsite_structural_values,
    )

    process = nx.DiGraph()
    process.graph["program_abi"] = {"records": {
        "Metrics": {
            "identity": "example.Metrics",
            "fields": {
                "pub_exchange_time": {
                    "storage": "span", "dtype": "float64", "shape": [2],
                },
            },
        },
    }}
    process.add_node(
        1, type="Call", op="Call",
        attributes={"result_class_ref": "Metrics"},
        parents=(), expr_obj=ast.parse("coerce(value)", mode="eval").body,
    )
    process.add_node(
        2, type="GetAttr", op="GetAttr",
        attributes={"attribute": "pub_exchange_time"}, parents=((1, "value"),),
        expr_obj=ast.parse("metrics.pub_exchange_time", mode="eval").body,
    )
    process.add_node(
        3, type="GetAttr", op="GetAttr",
        attributes={"attribute": "shape"}, parents=((2, "value"),),
        expr_obj=ast.parse("metrics.pub_exchange_time.shape", mode="eval").body,
    )
    process.add_node(
        4, type="Constant", op="const", constant=0,
        attributes={"value": 0}, parents=(),
    )
    process.add_node(
        5, type="Indexed", op="indexed",
        attributes={}, parents=((3, "base"), (4, "index")),
        expr_obj=ast.parse("metrics.pub_exchange_time.shape[0]", mode="eval").body,
    )
    process.add_edges_from(((1, 2), (2, 3), (3, 5), (4, 5)))
    graph = SimpleNamespace(G=process, roots=[5])

    _fold_callsite_structural_values(graph)

    assert process.nodes[5]["type"] == "Constant"
    assert process.nodes[5]["constant"] == 2


def test_tensor_item_capability_guard_is_structural():
    from src.compiler.glsl_deployment_strategy import (
        _fold_callsite_structural_values,
    )

    process = nx.DiGraph()
    process.graph["identity_table"] = {"value": (1,)}
    process.add_node(
        1, type="Input", op="input",
        attributes={"binding_name": "value"}, parents=(),
        tensor={"shape": (), "dtype": "float64"},
    )
    process.add_node(
        2, type="Constant", op="const", constant="item",
        attributes={"value": "item"}, parents=(),
    )
    process.add_node(
        3, type="Call", op="Call",
        attributes={"extraction_identity": "builtins.hasattr"},
        parents=((1, "arg:0"), (2, "arg:1")),
        expr_obj=ast.parse("hasattr(value, 'item')", mode="eval").body,
    )
    process.add_edges_from(((1, 3), (2, 3)))
    graph = SimpleNamespace(G=process, roots=[3])

    _fold_callsite_structural_values(graph)

    assert process.nodes[3]["type"] == "Constant"
    assert process.nodes[3]["constant"] is True


def test_returned_record_fields_feed_structural_call_argument():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def child(flag):\n"
        "    return flag\n\n"
        "def root(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0, osc_flag=True, hard_failure=False)\n"
        "    return child(metrics.osc_flag or metrics.hard_failure)\n",
        "root",
        name="record_structural_call_feed",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["record_structural_call_feed__root"]
    call = next(iter(module.call_table[root.name]))
    assert call.resolution == "native_call"
    assert outputs[root.name]
    assert not root.metadata.get("unresolved_call_diagnostics")
    assert not root.metadata.get("structural_output_shortfalls")
    assert any(
        instruction.op == "LOr"
        and instruction.attributes.get("call_feed") is True
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    boolop = next(
        instruction.res
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "LOr"
        and instruction.attributes.get("call_feed") is True
    )
    assert boolop is not None
    assert boolop.id not in {argument.id for argument in root.args}


def test_late_returned_record_with_sequence_publishes_scalar_field_to_caller():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def leaf(value):\n"
        "    return Metrics(max_vel=value, max_flux=value, div_inf=value, "
        "mass_err=value, hard_failure=value > 0.0, "
        "unresolved_report=[])\n\n"
        "def middle(value):\n"
        "    metrics = leaf(value)\n"
        "    return value, metrics\n\n"
        "def root(value):\n"
        "    advanced, metrics = middle(value)\n"
        "    return advanced + float(metrics.hard_failure)\n",
        "root",
        name="late_sequence_record_result",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["late_sequence_record_result__root"]
    call_record = next(iter(module.call_table[root.name]))
    returned_id = int(call_record.result_bindings[1][1])
    returned = module.record_tables[root.name].records[returned_id]
    fields = {field.name: field for field in returned.fields}

    assert root.metadata["late_returned_record_materializations"] == [(
        int(call_record.callsite_id),
        int(call_record.result_bindings[1][0]),
        returned_id,
    )]
    assert fields["unresolved_report"].sequence_id is not None
    hard_failure_id = int(fields["hard_failure"].value_ids[0])
    assert hard_failure_id not in {
        int(argument.id)
        for argument in root.args
        if not (argument.accounting or {})
    }
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("region_index") == 1
        and any(
            int(value.id) == hard_failure_id
            or int((value.accounting or {}).get(
                "call_input_conversion", (None, -1)
            )[1]) == hard_failure_id
            for value in instruction.args
        )
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_loop_carried_call_record_is_expanded_on_public_return():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def child(value):\n"
        "    metrics = Metrics(max_vel=value, max_flux=value, div_inf=0.0, "
        "mass_err=0.0)\n"
        "    return metrics, value\n\n"
        "def root(value):\n"
        "    last = None\n"
        "    result = value\n"
        "    index = 0\n"
        "    while index < 1:\n"
        "        last, result = child(value)\n"
        "        index += 1\n"
        "    return result, last\n",
        "root",
        name="loop_record_result",
        python_bindings={"Metrics": Metrics},
        extraction_contract=CONTRACT,
    )

    root = module.functions["loop_record_result__root"]
    returned = next(
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    layouts = dict(root.metadata["record_return_layouts"])
    assert len(layouts) == 1
    assert len(returned) == 1 + len(next(iter(layouts.values())))
    assert outputs[root.name] == tuple(returned)
    assert not root.metadata.get("unresolved_call_diagnostics")
    assert not any(
        instruction.attributes.get("plan_callsite_marker")
        for block in root.blocks.values()
        for instruction in block.instrs
    )


def test_specialized_function_argument_is_erased_from_runtime_frame():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def increment(value):\n"
        "    return value + 1\n\n"
        "def apply(value, operation):\n"
        "    return operation(value)\n\n"
        "def root(value):\n"
        "    return apply(value, increment)\n",
        "root",
        name="function_argument",
        extraction_contract=CONTRACT,
    )

    root = module.functions["function_argument__root"]
    call = next(iter(module.call_table[root.name]))
    assert call.resolution == "native_call"
    assert len(call.frame_bindings) == 1
    specialized = next(
        function for name, function in module.functions.items()
        if name.startswith("function_argument__apply__specialized_")
    )
    assert len(specialized.args) == 1
    assert outputs[root.name]


def test_callable_dataclass_field_preserves_function_identity():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "from dataclasses import dataclass\n"
        "from typing import Callable\n\n"
        "@dataclass\n"
        "class Law:\n"
        "    advance: Callable[[float], float]\n\n"
        "def increment(value):\n"
        "    return value + 1.0\n\n"
        "def root(value):\n"
        "    law = Law(increment)\n"
        "    return law.advance(value)\n",
        "root",
        name="callable_record_field",
        extraction_contract=CONTRACT,
    )

    root = module.functions["callable_record_field__root"]
    call = next(iter(module.call_table[root.name]))
    assert call.callee_symbol == "callable_record_field__increment"
    assert call.resolution == "native_call"
    assert outputs[root.name]

    page = module.metadata["identity_book"].page(
        "callable_identity_concordance"
    )
    rows = page.rows()
    assert len(rows) == 1
    assert tuple(fact for _column, fact in page.history(rows[0])) == (0, 0)


def _default_identity_child(value=None):
    value = 3.0 if value is None else value
    return value


def test_parameter_default_does_not_replace_later_same_name_ssa_value():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def root():\n"
        "    return child()\n",
        "root",
        name="default_identity_scope",
        python_bindings={"child": _default_identity_child},
        extraction_contract=CONTRACT,
    )

    root = module.functions["default_identity_scope__root"]
    record = next(iter(module.call_table[root.name]))
    default_bindings = tuple(
        binding for binding in record.frame_bindings
        if binding[1] == "default_literal"
    )
    assert default_bindings == ()
    callee = module.functions[record.callee_symbol]
    assert all(argument.id != 0 for argument in callee.args)
    assert outputs[root.name]


def test_specialized_dictionary_argument_has_no_runtime_argument_binding():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def child(geometry):\n"
        "    return dict(geometry)\n\n"
        "def root():\n"
        "    return child({'primitive': 'disc', 'radius': 2.0})\n",
        "root",
        name="specialized_dictionary",
        extraction_contract=CONTRACT,
    )

    root = module.functions["specialized_dictionary__root"]
    call = next(iter(module.call_table[root.name]))
    assert call.resolution == "native_call"
    assert call.argument_bindings == ()
    assert call.unresolved_frame_value_ids == ()
    assert call.result_bindings
    assert outputs[root.name]


def test_precompile_does_not_fold_inactive_module_definitions():
    module, outputs, _exports = lower_ast_source_to_ssa(
        "def helper():\n"
        "    return list(zip('xyz', range(3)))\n\n"
        "def root(value):\n"
        "    return value + 1\n",
        "root",
        name="catalogue_range",
        runtime_closure_only=True,
        extraction_contract=CONTRACT,
    )

    root = module.functions["catalogue_range__root"]
    assert outputs[root.name]
    assert not module.call_table.get(root.name)
    region_call = next(
        instruction for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Call"
    )
    region = module.functions[region_call.attributes["callee"]]
    assert any(
        instruction.op == "Add"
        for block in region.blocks.values()
        for instruction in block.instrs
    )
