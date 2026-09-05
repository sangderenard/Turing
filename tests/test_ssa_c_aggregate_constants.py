from __future__ import annotations

import numpy as np

from src.compiler.ssa_c_backend import (
    _publication_element_count,
    _publication_source_value,
    emit_ssa_module_to_c,
    emit_ssa_to_c,
    summarize_c_shortfalls,
    supported_scalar_operations,
    supported_tensor_operations,
)
from src.compiler.ssa_aggregate_abi import analyze_aggregate_abi


def test_c_publication_matches_exact_ret_id_before_position():
    omitted_inout = SSAValue(10, "bool")
    first_output = SSAValue(11, "float64")
    second_output = SSAValue(12, "int64")
    returned = (omitted_inout, first_output, second_output)

    assert _publication_source_value(first_output, 0, returned) is first_output
    assert _publication_source_value(second_output, 1, returned) is second_output


def test_c_converts_rankless_scalar_even_when_callee_indexes_one_element(
    tmp_path,
):
    source = SSAValue(0, "float64")
    zero = SSAValue(1, "int32")
    address = SSAValue(2, "ptr")
    loaded = SSAValue(3, "float64")
    helper = Function("rankless_scalar_helper", [source], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0}),
            Instr("GetElementPtr", [source, zero], address),
            Instr("Load", [address], loaded),
            Instr("Ret", [loaded], None),
        ]),
    })
    flag = SSAValue(10, "bool")
    result = SSAValue(11, "float64")
    root = Function("rankless_scalar_root", [flag], {
        "entry": BasicBlock("entry", [
            Instr("Call", [flag], result, attributes={"callee": helper.name}),
            Instr("Ret", [result], None),
        ]),
    }, metadata={"output_names": ("result",)})

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, helper.name: helper}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "double callarg" in artifact.source
    artifact.compile(tmp_path / "rankless_scalar")
    execution = artifact.prepare_execution({
        flag.id: np.array([True], dtype=np.bool_),
    }).run()
    assert execution.buffers[result.id].item() == 1.0


def test_c_converts_scalar_use_view_from_address_bound_storage(tmp_path):
    source = SSAValue(0, "float64")
    zero = SSAValue(1, "int32")
    address = SSAValue(2, "ptr")
    loaded = SSAValue(3, "float64")
    helper = Function("view_cast_helper", [source], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0}),
            Instr("GetElementPtr", [source, zero], address),
            Instr("Load", [address], loaded),
            Instr("Ret", [loaded], None),
        ]),
    })
    flag = SSAValue(10, "bool")
    float_use_of_flag = SSAValue(10, "float64")
    result = SSAValue(11, "float64")
    root = Function("storage_view_root", [flag], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [float_use_of_flag], result,
                attributes={"callee": helper.name},
            ),
            Instr("Ret", [result], None),
        ]),
    }, metadata={"output_names": ("result",)})

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, helper.name: helper}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "double callarg" in artifact.source
    artifact.compile(tmp_path / "storage_view_scalar")
    execution = artifact.prepare_execution({
        flag.id: np.array([True], dtype=np.bool_),
    }).run()
    assert execution.buffers[result.id].item() == 1.0
from src.compiler.tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa
from src.compiler.ir_identities import drop_dead_pure_structural_instructions
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    Instr,
    IRModule,
    SSAValue,
    SSATensorDescriptor,
    SSATensorTable,
)


def test_c_publishes_scalar_reduction_reusing_mutable_record_field(tmp_path):
    source = SSAValue(0, "float64", (4,))
    metric = SSAValue(1, "float64", (1,), accounting={
        "program_abi_parameter": "state",
        "program_abi_field": "metric",
        "program_abi_storage": "scalar",
        "program_abi_mutable": True,
        "program_abi_field_written": True,
    })
    reduced_metric = SSAValue(2, "float64")
    function = Function("record_reduction_inout", [source, metric], {
        "entry": BasicBlock("entry", [
            Instr("max", [source], reduced_metric),
        ]),
    }, metadata={"freshened_synthetic_value_ids": ((1, 2),)})

    artifact = emit_ssa_to_c(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "record_reduction_inout")
    execution = artifact.prepare_execution({
        source.id: np.asarray([-4.0, 7.5, 2.0, 1.0]),
        metric.id: np.asarray([0.0]),
    }).run()
    assert execution.buffers[metric.id].item() == 7.5


def test_aggregate_analysis_restores_sparse_inout_return_positions():
    state_metric = SSAValue(1, "float64")
    ok = SSAValue(2, "bool")
    divergence = SSAValue(3, "float64")
    callee = Function("sparse_record_callee", [state_metric], {
        "entry": BasicBlock("entry", [
            Instr("Ret", [
                ok, state_metric, state_metric, divergence,
            ], None),
        ]),
    })
    caller_metric = SSAValue(11, "float64")
    aggregate = SSAValue(12, "ssa.aggregate")
    caller_ok = SSAValue(13, "bool")
    caller_divergence = SSAValue(14, "float64")
    caller = Function("sparse_record_caller", [caller_metric], {
        "entry": BasicBlock("entry", [
            Instr("Call", [caller_metric], aggregate, attributes={
                "callee": callee.name,
                "result_convention": "ssa.aggregate",
                "output_ids": (caller_ok.id, caller_divergence.id),
                "output_positions": (0, 3),
                "callee_output_ids": (ok.id, divergence.id),
            }),
        ]),
    })

    analysis = analyze_aggregate_abi(
        IRModule({caller.name: caller, callee.name: callee})
    )

    assert analysis.calls[0].output_ids == (2, 1, 1, 3)


def test_module_c_artifact_carries_recovered_storage_shape():
    formal = SSAValue(0, "float64")
    shaped_view = SSAValue(0, "float64", (2, 3))
    function = Function("shaped_root", [formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [shaped_view], None)]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_shapes == ((2, 3),)


def test_module_c_owns_invocation_local_compiler_frame_storage():
    formal = SSAValue(
        0, "float64", accounting={"compiler_frame_storage": "root"}
    )
    shaped_view = SSAValue(0, "float64", (2, 3), accounting=formal.accounting)
    function = Function("owned_frame", [formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [shaped_view], None)]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_order == ()
    assert "double frame_0[6];" in artifact.source
    assert "static double frame_0[6];" not in artifact.source
    assert "memset(frame_0, 0, sizeof(frame_0));" in artifact.source


def test_module_c_emits_numeric_aggregate_constants_as_local_arrays():
    aggregate = SSAValue(0, "float64")
    index = SSAValue(1, "int64")
    address = SSAValue(2, "ptr")
    value = SSAValue(3, "float64")
    function = Function("aggregate_root", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], aggregate, attributes={"value": [1.0, [2, 3]]}),
            Instr("Const", [], index, attributes={"value": 1}),
            Instr("GetElementPtr", [aggregate, index], address),
            Instr("Load", [address], value),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert artifact.complete, artifact.shortfalls
    assert (
        "const double t0[] = {0x1.0000000000000p+0, "
        "0x1.0000000000000p+1, 0x1.8000000000000p+1};"
    ) in artifact.source


def test_module_c_loads_scalar_value_through_pointer_table_slot():
    table = SSAValue(0, "ptrptr_float64")
    index = SSAValue(1, "int64")
    address = SSAValue(2, "ptr")
    value = SSAValue(3, "float64")
    function = Function("pointer_table_scalar", [table], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [table, index], address),
            Instr("Load", [address], value),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert artifact.complete, artifact.shortfalls
    assert "double t3 = *((double *)(*((double * *)(t2))));" in artifact.source


def test_c_propagates_explicit_array_storage_type_across_call_edge():
    actual = SSAValue(
        0, "int64", (2,), accounting={"physical_dtype": "float64"}
    )
    formal = SSAValue(10, "int64", (2,))
    index = SSAValue(11, "int64")
    address = SSAValue(12, "ptr")
    loaded = SSAValue(13, "float64")
    helper = Function("index_helper", [formal], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [formal, index], address),
            Instr("Load", [address], loaded),
            Instr("Ret", [], None),
        ]),
    })
    root = Function("physical_index_root", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={"callee": helper.name}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, helper.name: helper}), root.name
    )

    assert artifact.complete, artifact.shortfalls
    assert "double *t12 = ((double *)(v10))" in artifact.source


def test_module_c_reports_non_numeric_aggregate_constants_without_crashing():
    aggregate = SSAValue(0, "float64")
    function = Function("aggregate_root", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], aggregate, attributes={"value": [object()]}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert not artifact.complete
    assert "require bool/int/float elements" in artifact.shortfalls[0].reason


def test_module_c_reports_invalid_none_const_without_guessing_zero():
    value = SSAValue(0, "int64")
    function = Function("none_root", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], value, attributes={"value": None}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert not artifact.complete
    assert "NoneValue operation" in artifact.shortfalls[0].reason


def test_module_c_decodes_imported_llvm_literals_and_integer_casts():
    one = SSAValue(0, "i32")
    widened = SSAValue(1, "int64")
    function = Function("llvm_literal_root", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], one, attributes={"llvm_literal": "i32 1"}),
            Instr("SExt", [one], widened),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({function.name: function}), function.name
    )

    assert artifact.complete, artifact.shortfalls
    assert "const int64_t t1 = (int64_t)((int32_t)(1));" in artifact.source


def test_c_shortfall_summary_separates_primary_gaps_from_operand_fallout():
    from src.compiler.ssa_c_backend import CEmissionShortfall

    summary = summarize_c_shortfalls((
        CEmissionShortfall("stack", "no module-lane C spelling in kernel"),
        CEmissionShortfall("operand", "%t3 is unavailable in kernel"),
        CEmissionShortfall("operand", "%t4 is unavailable in kernel"),
    ))

    assert summary == (
        "stack: no module-lane C spelling in kernel",
        "operand fallout: 2 unavailable use(s) in kernel",
    )


def test_tensor_stack_lowers_through_repository_pointer_array_abi():
    left = SSAValue(0, "float64", (2, 3))
    middle = SSAValue(1, "float64", (2, 3))
    right = SSAValue(2, "float64", (2, 3))
    dim = SSAValue(3, "int64")
    output = SSAValue(4, "float64", (2, 3, 3))
    function = Function("stack_root", [left, middle, right, output], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], dim, attributes={"value": -1}),
            Instr(
                "stack", [left, middle, right, dim], output,
                attributes={"tensor_operation": "stack"},
            ),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({function.name: function})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )
    artifact = emit_ssa_module_to_c(module, function.name)

    assert shortfalls == ()
    assert artifact.complete, artifact.shortfalls
    assert "double *t" in artifact.source
    assert "impl_stack_double(" in artifact.source


def test_c_emits_repository_scalar_and_structural_helper_closures():
    reference = c_backend_repository_ssa_reference()

    for root in (
        "binary_value",
        "binary_double",
        "unary_double",
        "reduce_dim_double",
        "where_double",
        "stack_double",
        "cat_double",
    ):
        artifact = emit_ssa_module_to_c(reference.module, root)
        assert artifact.complete, (root, artifact.shortfalls)


def test_c_binds_planned_aggregate_tensor_output_to_caller_storage(tmp_path):
    source = SSAValue(0, "float64", (2,))
    output = SSAValue(2, "float64", (2,))
    index = SSAValue(3, "int64")
    source_slot = SSAValue(4, "ptr")
    scalar = SSAValue(5, "float64")
    output_slot = SSAValue(6, "ptr")
    region = Function("planned_region", [source], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [source, index], source_slot),
            Instr("Load", [source_slot], scalar),
            Instr("GetElementPtr", [output, index], output_slot),
            Instr("Store", [scalar, output_slot], None),
        ]),
    })

    aggregate = SSAValue(10)
    aggregate_index = SSAValue(11, "int64")
    address = SSAValue(12, "ptr")
    projected = SSAValue(13, "float64", (2,))
    root = Function("root", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [source], aggregate,
                attributes={
                    "callee": "planned_region",
                    "result_convention": "ssa.aggregate",
                    "output_ids": (2,),
                },
            ),
            Instr("Const", [], aggregate_index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, aggregate_index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], projected),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, region.name: region}), root.name
    )

    assert artifact.complete, artifact.shortfalls
    assert "planned_region(double *v0, double *out2" in artifact.source
    assert "restrict" not in artifact.source
    assert "double callout13_0[2];" in artifact.source
    assert "static double callout13_0[2];" not in artifact.source
    assert "impl_planned_region(v0, callout13_0, extents);" in artifact.source
    artifact.compile(tmp_path / "aggregate_region")


def test_c_aggregate_projection_cannot_rebind_root_output_id():
    callee_output = SSAValue(2, "float64")
    callee = Function(
        "scalar_region", [],
        {"entry": BasicBlock("entry", [])},
        metadata={"named_outputs": (("value", callee_output.id),)},
    )
    aggregate = SSAValue(10, "ssa.aggregate")
    index = SSAValue(11, "int64")
    address = SSAValue(12, "ptr")
    root_output = SSAValue(13, "float64")
    root = Function(
        "projection_output_root", [],
        {"entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": callee.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (callee_output.id,),
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], root_output),
            Instr("Ret", [root_output], None),
        ])},
        metadata={"named_outputs": (("result", root_output.id),)},
    )

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, callee.name: callee}), root.name
    )

    assert artifact.complete, artifact.shortfalls
    assert "impl_scalar_region(out13" in artifact.source
    assert "*((double *)(out13)) = *((double *)(out" not in artifact.source


def test_c_aggregate_projection_cannot_rebind_unrelated_formal_id():
    callee_output = SSAValue(2, "float64")
    callee = Function(
        "scalar_region", [],
        {"entry": BasicBlock("entry", [])},
        metadata={"named_outputs": (("value", callee_output.id),)},
    )
    unrelated_formal = SSAValue(13, "bool")
    aggregate = SSAValue(10, "ssa.aggregate")
    index = SSAValue(11, "int64")
    first_address = SSAValue(12, "ptr")
    colliding_projection = SSAValue(13, "float64")
    root = Function("projection_formal_root", [unrelated_formal], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": callee.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (callee_output.id,),
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], first_address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [first_address], colliding_projection),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({
            root.name: root,
            callee.name: callee,
        }),
        root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "impl_scalar_region(&callout13_0);" in artifact.source
    assert "impl_scalar_region(v13);" not in artifact.source


def test_c_inout_projection_publishes_into_source_formal_storage():
    region_output = SSAValue(18, "float64")
    region = Function(
        "inout_scalar_region", [],
        {"entry": BasicBlock("entry", [])},
        metadata={"named_outputs": (("value", region_output.id),)},
    )
    formal = SSAValue(18, "float64")
    aggregate = SSAValue(20, "ssa.aggregate")
    index = SSAValue(21, "int64")
    address = SSAValue(22, "ptr")
    write_version = SSAValue(
        28,
        "float64",
        accounting={
            "source_value_id": 18,
            "ssa_inout_write_version": True,
        },
    )
    root = Function("inout_projection_root", [formal], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": region.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (region_output.id,),
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], write_version),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, region.name: region}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "impl_inout_scalar_region(v18);" in artifact.source
    assert "callout28_0" not in artifact.source


def test_c_linked_inout_projection_recognizes_exact_formal_object():
    region_output = SSAValue(18, "float64")
    region = Function(
        "linked_inout_scalar_region", [],
        {"entry": BasicBlock("entry", [])},
        metadata={"named_outputs": (("value", region_output.id),)},
    )
    formal_and_projection = SSAValue(18, "float64")
    aggregate = SSAValue(20, "ssa.aggregate")
    index = SSAValue(21, "int64")
    address = SSAValue(22, "ptr")
    root = Function("linked_inout_projection_root", [formal_and_projection], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": region.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (region_output.id,),
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr(
                "Load", [address], formal_and_projection,
                attributes={
                    "aggregate_index": 0,
                    "source_output_id": 18,
                },
            ),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, region.name: region}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "impl_linked_inout_scalar_region(v18);" in artifact.source
    assert "callout18_0" not in artifact.source


def test_c_sizes_stale_call_projection_from_propagated_callee_requirement():
    """A narrow call-site view must not under-allocate a wider callee write."""

    output = SSAValue(2, "float64", (8, 4))
    output_index = SSAValue(3, "int64")
    output_address = SSAValue(4, "ptr")
    zero = SSAValue(5, "float64")
    region = Function(
        "wide_region", [],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], output_index, attributes={"value": 31}),
            Instr("Const", [], zero, attributes={"value": 0.0}),
            Instr("GetElementPtr", [output, output_index], output_address),
            Instr("Store", [zero, output_address], None),
        ])},
        metadata={"named_outputs": (("wide", output.id),)},
    )
    aggregate = SSAValue(10, "ssa.aggregate")
    index = SSAValue(11, "int64")
    address = SSAValue(12, "ptr")
    stale_projection = SSAValue(13, "float64", (8, 1))
    root = Function("narrow_caller", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": region.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (output.id,),
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], stale_projection),
            Instr("Ret", [], None),
        ]),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=stale_projection.id,
        data_value_id=stale_projection.id,
        shape=stale_projection.shape,
        storage="temporary",
    ))
    module = IRModule(
        {root.name: root, region.name: region},
        tensor_tables={root.name: table},
    )

    artifact = emit_ssa_module_to_c(module, root.name)

    assert artifact.complete, artifact.shortfalls
    # Activation storage is zero-initialized (calloc) so a use-before-def
    # reads a deterministic 0.0 instead of run-varying heap garbage; the
    # propagated callee requirement still owns the 32-element size.
    assert "calloc(32, sizeof(*frame_storage_0))" in artifact.source


def test_c_scalar_ret_publication_does_not_copy_stale_view_extent():
    """A scalar local may share an id carrying a stale shaped occurrence."""

    returned = SSAValue(2, "float64", (8192,))

    assert _publication_element_count(returned, "&t2") == 1
    assert _publication_element_count(returned, "frame_storage_0") == 8192


def test_c_dispatches_call_encoded_extent_as_scalar_metadata():
    source = SSAValue(0, "float64", (3,))
    extent = SSAValue(1, "int", ())
    function = Function("call_encoded_extent", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call",
                [source],
                extent,
                attributes={
                    "tensor_operation": "extent",
                    "extent_kind": "dim",
                    "axis": 0,
                },
            ),
            Instr("Ret", [extent], None),
        ]),
    }, metadata={"named_outputs": (("extent", extent.id),)})
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=extent.id,
        data_value_id=extent.id,
        dtype="int",
        shape=(),
        storage="temporary",
        metadata_state="unresolved",
    ))

    artifact = emit_ssa_module_to_c(
        IRModule(
            {function.name: function},
            tensor_tables={function.name: table},
        ),
        function.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "frame_storage" not in artifact.source


def test_c_emits_tensor_clone_as_owned_copy():
    source = SSAValue(0, "float64", (3,))
    cloned = SSAValue(1, "float64", (3,))
    function = Function("tensor_clone", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call",
                [source],
                cloned,
                attributes={"tensor_operation": "clone"},
            ),
            Instr("Ret", [cloned], None),
        ]),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=cloned.id,
        data_value_id=cloned.id,
        dtype="float64",
        shape=(3,),
        storage="temporary",
    ))

    artifact = emit_ssa_module_to_c(
        IRModule(
            {function.name: function},
            tensor_tables={function.name: table},
        ),
        function.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert "memcpy(" in artifact.source


def test_c_preserves_physical_tensor_dtype_across_planned_region_output():
    source = SSAValue(0, "float64", (2,))
    mask = SSAValue(
        2, "bool", (2,), accounting={"physical_dtype": "float64"},
    )
    region = Function("mask_region", [source], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], mask, attributes={"value": [0, 1]}),
            Instr("Ret", [mask], None),
        ]),
    })

    aggregate = SSAValue(10, "ssa.aggregate")
    aggregate_index = SSAValue(11, "int64")
    address = SSAValue(12, "ptr")
    projected = SSAValue(13, "bool", (2,))
    root = Function("mask_root", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [source], aggregate,
                attributes={
                    "callee": region.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (2,),
                },
            ),
            Instr("Const", [], aggregate_index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, aggregate_index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], projected),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(
        IRModule({root.name: root, region.name: region}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    mask_lines = tuple(
        line for line in artifact.source.splitlines() if "mask_region" in line
    )
    assert (
        "mask_region(double *v0, double *out2"
        in artifact.source
    ), mask_lines
    assert "double callout13_0[2];" in artifact.source
    assert "static double callout13_0[2];" not in artifact.source
    assert "restrict" not in artifact.source


def test_c_materializes_aggregate_that_is_consumed_whole():
    first = SSAValue(20, "float64", (2,))
    second = SSAValue(21, "float64", (2,))
    producer = Function("produce_pair", [first, second], {
        "entry": BasicBlock("entry", [Instr("Ret", [first, second], None)]),
    })
    table_formal = SSAValue(30, "ptrptr_float64", (2,))
    table_index = SSAValue(31, "int32")
    table_address = SSAValue(32, "ptr")
    selected = SSAValue(33, "float64", (2,))
    consumer = Function("consume_pair", [table_formal], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], table_index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [table_formal, table_index], table_address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [table_address], selected),
            Instr("Ret", [], None),
        ]),
    })
    aggregate = SSAValue(40, "ssa.aggregate")
    first_address = SSAValue(41, "ptr")
    second_address = SSAValue(42, "ptr")
    caller_first = SSAValue(43, "float64", (2,))
    caller_second = SSAValue(44, "float64", (2,))
    root = Function("whole_aggregate_root", [first, second], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [first, second], aggregate,
                attributes={
                    "callee": producer.name,
                    "result_convention": "ssa.aggregate",
                    "output_ids": (43, 44),
                    "callee_output_ids": (20, 21),
                    "output_positions": (0, 1),
                    "output_slots": (0, 1),
                },
            ),
            Instr(
                "GetElementPtr", [aggregate], first_address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [first_address], caller_first),
            Instr(
                "GetElementPtr", [aggregate], second_address,
                attributes={"aggregate_index": 1},
            ),
            Instr("Load", [second_address], caller_second),
            Instr("Call", [aggregate], None, attributes={"callee": consumer.name}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(IRModule({
        root.name: root, producer.name: producer, consumer.name: consumer,
    }), root.name)

    assert artifact.complete, artifact.shortfalls
    assert "double *aggregate40[]" in artifact.source


def test_dead_frontend_slice_const_is_removed_without_a_structural_tag():
    dead = SSAValue(0)
    live = SSAValue(1, "int64")
    function = Function("root", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], dead, attributes={"value": slice(None)}),
            Instr("Const", [], live, attributes={"value": 2}),
            Instr("Ret", [live], None),
        ]),
    })

    assert drop_dead_pure_structural_instructions({function.name: function}) == 1
    assert [instruction.res for instruction in function.blocks["entry"].instrs] == [
        live, None,
    ]


def test_where_scalar_branch_is_materialized_before_repository_where(tmp_path):
    condition = SSAValue(0, "float64", (4,))
    when_true = SSAValue(1, "float64", (4,))
    scalar = SSAValue(2, "int32")
    output = SSAValue(3, "float64", (4,))
    function = Function("where_root", [condition, when_true, output], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], scalar, attributes={"value": 1}),
            Instr(
                "where", [condition, when_true, scalar], output,
                attributes={"tensor_operation": "where"},
            ),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({function.name: function})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )
    calls = [
        instruction.attributes.get("callee")
        for block in module.functions[function.name].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    ]
    artifact = emit_ssa_module_to_c(module, function.name)

    assert shortfalls == ()
    assert calls[-2:] == ["fill_double", "where_double"]
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "where_scalar")


def test_where_infers_result_shape_and_broadcasts_condition():
    condition = SSAValue(0, "float64", (2, 1))
    when_true = SSAValue(1, "float64", (2, 3))
    when_false = SSAValue(2, "float64", (2, 3))
    output = SSAValue(3, "float64")
    function = Function("where_broadcast", [condition, when_true, when_false], {
        "entry": BasicBlock("entry", [
            Instr(
                "where", [condition, when_true, when_false], output,
                attributes={"tensor_operation": "where"},
            ),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({function.name: function})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )
    instructions = module.functions[function.name].blocks["entry"].instrs
    calls = [
        instruction.attributes.get("callee")
        for instruction in instructions
        if instruction.op == "Call"
    ]
    where_call = next(
        instruction for instruction in instructions
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "where_double"
    )
    constants = {
        instruction.res.id: instruction.attributes.get("constant")
        for instruction in instructions
        if instruction.op == "Const" and instruction.res is not None
    }
    defined = {value.id for value in function.args}
    defined.update(
        instruction.res.id for instruction in instructions
        if instruction.res is not None
    )

    assert shortfalls == ()
    assert output.shape == (2, 3)
    assert module.tensor_tables[function.name].by_id(output.id).shape == (2, 3)
    assert module.tensor_tables[function.name].by_id(output.id).byte_size == 48
    assert calls[-2:] == ["broadcast_double", "where_double"]
    assert constants[where_call.args[-1].id] == 6
    assert all(
        argument.id in defined
        for instruction in instructions
        for argument in instruction.args
    )


def test_c_emits_static_extent_shape_rank_count_and_dimension():
    source = SSAValue(0, "float64", (2, 3))
    shape = SSAValue(1, "int32", (2,))
    rank = SSAValue(2, "int32")
    count = SSAValue(3, "int32")
    dimension = SSAValue(4, "int32")
    function = Function("extent_root", [source], {
        "entry": BasicBlock("entry", [
            Instr("extent", [source], shape, attributes={"extent_kind": "shape"}),
            Instr("extent", [source], rank, attributes={"extent_kind": "rank"}),
            Instr("extent", [source], count, attributes={"extent_kind": "element_count"}),
            Instr("extent", [source], dimension, attributes={"extent_kind": "dim", "axis": 1}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_module_to_c(IRModule({function.name: function}), function.name)

    assert artifact.complete, artifact.shortfalls
    assert "const int32_t t1[] = {2, 3};" in artifact.source


def test_c_measures_dynamic_shape_and_multi_axis_address(tmp_path):
    span = SSAValue(0, "float64", ("rows", "columns"))
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    element = SSAValue(4, "float64")
    function = Function("dynamic_grid", [span, row, column], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [span, row, column], address),
            Instr("Load", [address], element),
            Instr("Ret", [element], None),
        ]),
    }, metadata={"output_names": ("element",)})

    artifact = emit_ssa_to_c(IRModule({function.name: function}), function.name)

    assert artifact.complete, artifact.shortfalls
    assert artifact.extent_order == ((span.id, "dim", 1),)
    assert "restrict" not in artifact.source
    artifact.compile(tmp_path / "dynamic_grid")
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    execution = artifact.prepare_execution({
        span.id: values,
        row.id: np.array([2], dtype=np.int32),
        column.id: np.array([1], dtype=np.int32),
    }).run()
    assert execution.buffers[element.id].item() == 9.0


def test_c_propagates_dynamic_extents_through_internal_call(tmp_path):
    region_span = SSAValue(0, "float64", ("rows", "columns"))
    region_row = SSAValue(1, "int32")
    region_column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    element = SSAValue(4, "float64")
    region = Function("dynamic_region", [
        region_span, region_row, region_column,
    ], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [
                region_span, region_row, region_column,
            ], address),
            Instr("Load", [address], element),
            Instr("Ret", [element], None),
        ]),
    }, metadata={"llvm_return_dtype": "float64"})
    root_span = SSAValue(10, "float64", ("rows", "columns"))
    root_row = SSAValue(11, "int32")
    root_column = SSAValue(12, "int32")
    result = SSAValue(13, "float64")
    root = Function("dynamic_root", [root_span, root_row, root_column], {
        "entry": BasicBlock("entry", [
            Instr("Call", [root_span, root_row, root_column], result,
                  attributes={"callee": region.name}),
            Instr("Ret", [result], None),
        ]),
    }, metadata={"output_names": ("result",)})

    artifact = emit_ssa_to_c(
        IRModule({root.name: root, region.name: region}), root.name,
    )

    assert artifact.complete, artifact.shortfalls
    assert artifact.extent_order == ((root_span.id, "dim", 1),)
    artifact.compile(tmp_path / "dynamic_call")
    values = np.arange(15, dtype=np.float64).reshape(3, 5)
    execution = artifact.prepare_execution({
        root_span.id: values,
        root_row.id: np.array([1], dtype=np.int32),
        root_column.id: np.array([3], dtype=np.int32),
    }).run()
    assert execution.buffers[result.id].item() == 8.0


def test_c_preserves_integer_width_extension_and_wrapping(tmp_path):
    left = SSAValue(0, "int32")
    right = SSAValue(1, "int32")
    summed = SSAValue(2, "int32")
    widened = SSAValue(3, "int64")
    function = Function("integer_exact", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], summed),
            Instr("ZExt", [summed], widened),
            Instr("Ret", [summed, widened], None),
        ]),
    }, metadata={"output_names": ("summed", "widened")})

    artifact = emit_ssa_to_c(IRModule({function.name: function}), function.name)

    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_dtypes[-2:] == ("int32", "int64")
    artifact.compile(tmp_path / "integer_exact")
    execution = artifact.prepare_execution({
        left.id: np.array([2**31 - 1], dtype=np.int32),
        right.id: np.array([1], dtype=np.int32),
    }).run()
    assert execution.buffers[summed.id].item() == -(2**31)
    assert execution.buffers[widened.id].item() == 2**31


def test_c_module_lane_emits_preselected_scalar_aggregate_projection(tmp_path):
    selected_leaf = SSAValue(0, "int64")
    projected = SSAValue(1, "int")
    function = Function("scalar_projection", [selected_leaf], {
        "entry": BasicBlock("entry", [
            Instr("indexed", [selected_leaf], projected),
            Instr("Ret", [projected], None),
        ]),
    }, metadata={"output_names": ("projected",)})

    artifact = emit_ssa_to_c(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "scalar_projection", optimization="O0")
    execution = artifact.prepare_execution({
        selected_leaf.id: np.array([37], dtype=np.int64),
    }).run()
    assert execution.buffers[projected.id].item() == 37


def test_c_truncation_and_unsigned_comparison_use_bit_patterns(tmp_path):
    wide = SSAValue(0, "int64")
    one = SSAValue(1, "int32")
    narrow = SSAValue(2, "int32")
    widened = SSAValue(3, "int64")
    unsigned_less = SSAValue(4, "bool")
    function = Function("integer_bits", [wide, one], {
        "entry": BasicBlock("entry", [
            Instr("Trunc", [wide], narrow),
            Instr("ZExt", [narrow], widened),
            Instr("ULt", [narrow, one], unsigned_less),
            Instr("Ret", [narrow, widened, unsigned_less], None),
        ]),
    }, metadata={
        "output_names": ("narrow", "widened", "unsigned_less"),
    })

    artifact = emit_ssa_to_c(IRModule({function.name: function}), function.name)

    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "integer_bits")
    execution = artifact.prepare_execution({
        wide.id: np.array([2**32 - 1], dtype=np.int64),
        one.id: np.array([1], dtype=np.int32),
    }).run()
    assert execution.buffers[narrow.id].item() == -1
    assert execution.buffers[widened.id].item() == 2**32 - 1
    assert execution.buffers[unsigned_less.id].item() == 0


def test_c_module_lane_publishes_shared_native_capabilities():
    scalar = supported_scalar_operations()
    tensor = supported_tensor_operations()

    assert {"ZExt", "Trunc", "ULt", "UGe", "Exp", "Log"} <= scalar
    assert {"matmul", "stack", "scatter", "extent"} <= tensor
