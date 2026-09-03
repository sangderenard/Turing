from __future__ import annotations

from src.compiler.tensor_ssa_lowering import (
    legalize_aggregate_adapters,
    legalize_aggregate_output_views,
    propagate_repository_ssa_call_metadata,
    settle_repository_ssa_static_extent_operands,
    wire_repository_ssa_region_products,
    lower_tensor_calls_to_repository_ssa,
)
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


def test_materialized_call_operand_shape_wins_over_an_aliased_feed_view():
    """A semantic feed id may name several reshape views of one storage id."""

    actual = SSAValue(99, "float64", shape=(8, 41))
    conflicting_view = SSAValue(5, "float64", shape=(2, 4, 1, 1))
    call_result = SSAValue(100, "float64", shape=(8, 41))
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("UseView", [conflicting_view], None),
            Instr(
                "Call", [actual], call_result,
                attributes={
                    "callee": "owner__planned_region_0",
                    "feed_ids": (5,),
                },
            ),
            Instr("Ret", [call_result], None),
        ]),
    })
    formal = SSAValue(0)
    callee = Function("owner__planned_region_0", [formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [formal], None)]),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=formal.id,
        data_value_id=formal.id,
        shape=(1, 41),
        storage="input",
    ))
    module = IRModule(
        {caller.name: caller, callee.name: callee},
        tensor_tables={callee.name: table},
    )

    propagate_repository_ssa_call_metadata(module)

    assert formal.shape == actual.shape
    assert formal.dtype == actual.dtype
    assert table.by_id(formal.id).shape == actual.shape


def test_settled_formal_restamps_lowered_broadcast_shape_constants():
    source = SSAValue(0, "float64", (8, 4, 128, 2, 3))
    output = SSAValue(1, "float64", (8, 4, 128, 2, 3))
    source_shape = SSAValue(2, "int32", (5,))
    source_rank = SSAValue(3, "int32")
    output_shape = SSAValue(4, "int32", (5,))
    output_rank = SSAValue(5, "int32")
    function = Function("planned", [source], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], source_shape, attributes={
                "values": (8, 4, 128, 1, 3), "constant": None,
            }),
            Instr("Const", [], source_rank, attributes={"constant": 5}),
            Instr("Const", [], output_shape, attributes={
                "values": output.shape, "constant": None,
            }),
            Instr("Const", [], output_rank, attributes={"constant": 5}),
            Instr(
                "Call",
                [source, output, source_shape, source_rank,
                 output_shape, output_rank],
                output,
                attributes={"callee": "broadcast_double"},
            ),
        ]),
    })
    table = SSATensorTable()
    for value in (source, output):
        table.register(SSATensorDescriptor(
            tensor_id=value.id,
            data_value_id=value.id,
            shape=value.shape,
            storage="input" if value is source else "temporary",
        ))
    module = IRModule(
        {function.name: function}, tensor_tables={function.name: table},
    )

    assert settle_repository_ssa_static_extent_operands(module)
    assert function.blocks["entry"].instrs[0].attributes["values"] == (
        8, 4, 128, 2, 3,
    )


def test_bool_broadcast_keeps_double_backed_repository_storage():
    source = SSAValue(0, "bool", shape=(1,))
    requested_shape = SSAValue(1, "int32", shape=(1,))
    result = SSAValue(2, "bool", shape=(8,))
    function = Function("bool_broadcast", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Const", [], requested_shape,
                attributes={"values": (8,), "constant": None},
            ),
            Instr(
                "broadcast_to", [source, requested_shape], result,
                attributes={"tensor_operation": "broadcast_to"},
            ),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()

    assert source.accounting["physical_dtype"] == "float64"
    assert result.accounting["physical_dtype"] == "float64"
    descriptor = module.tensor_tables[function.name].by_id(result.id)
    assert descriptor is not None
    assert descriptor.byte_size == 8 * 8


def test_python_binary_min_max_keep_operands_in_repository_ssa():
    left = SSAValue(0, "float64")
    right = SSAValue(1, "float64")
    minimum = SSAValue(2, "float64")
    maximum = SSAValue(3, "float64")
    function = Function("python_min_max", [left, right], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [left, right], minimum,
                attributes={"tensor_operation": "min"},
            ),
            Instr(
                "Call", [left, right], maximum,
                attributes={"tensor_operation": "max"},
            ),
            Instr("Ret", [minimum, maximum], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()

    extrema = [
        instruction for instruction in function.blocks["entry"].instrs
        if instruction.op in {"Min", "Max"}
    ]
    assert [instruction.op for instruction in extrema] == ["Min", "Max"]
    assert [argument.id for argument in extrema[0].args] == [
        left.id, right.id,
    ]
    assert [argument.id for argument in extrema[1].args] == [
        left.id, right.id,
    ]


def test_indexed_store_versions_the_same_resident_arena_in_place():
    source = SSAValue(0, "float64", shape=(4,))
    value = SSAValue(1, "float64")
    result = SSAValue(2, "float64", shape=(4,))
    function = Function("resident_store", [source, value], {
        "entry": BasicBlock("entry", [
            Instr(
                "IndexedStore", [source, value], result,
                attributes={"basic_index_axes": (((1,), True),)},
            ),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()

    call = next(
        instruction
        for instruction in function.blocks["entry"].instrs
        if instruction.op == "Call"
    )
    assert call.attributes["callee"] == "index_assign_double"
    assert [argument.id for argument in call.args[:1]] == [source.id]
    descriptor = module.tensor_tables[function.name].by_id(result.id)
    assert descriptor is not None
    assert descriptor.alias_of == source.id
    assert not descriptor.owns_allocation


def test_fixed_integer_index_lowers_to_scalar_load_not_tensor_selection():
    source = SSAValue(0, "float64", shape=(20,))
    result = SSAValue(1, "float64")
    function = Function("scalar_index", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Indexed", [source], result,
                attributes={
                    "basic_index_axes": (((11,), True),),
                    "basic_index_source_shape": (20,),
                },
            ),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()

    instructions = function.blocks["entry"].instrs
    assert any(
        instruction.op == "Load" and instruction.res is result
        for instruction in instructions
    )
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == "index_select_double"
        for instruction in instructions
    )
    assert result.shape == ()


def test_prelowered_fixed_integer_selection_is_legalized_to_scalar_load():
    source = SSAValue(0, "float64", shape=(20,))
    result = SSAValue(1, "float64", shape=(1,))
    shape = SSAValue(2, "int32", shape=(1,))
    rank = SSAValue(3, "int32")
    axis = SSAValue(4, "int32")
    indices = SSAValue(5, "int32", shape=(1,))
    count = SSAValue(6, "int32")
    function = Function("prelowered_scalar_index", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call",
                [source, result, shape, rank, axis, indices, count],
                result,
                attributes={
                    "callee": "index_select_double",
                    "ssa_output_argument": 1,
                    "basic_index_axes": (((11,), True),),
                    "basic_index_source_shape": (20,),
                },
            ),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()

    instructions = function.blocks["entry"].instrs
    assert any(
        instruction.op == "Load" and instruction.res is result
        for instruction in instructions
    )
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == "index_select_double"
        for instruction in instructions
    )
    assert result.shape == ()


def test_slice_spelling_honors_normalized_dropped_axis_scalar_contract():
    source = SSAValue(0, "float64", shape=(20,))
    result = SSAValue(1, "float64", shape=(1,))
    function = Function("slice_spelled_scalar_index", [source], {
        "entry": BasicBlock("entry", [
            Instr(
                "Slice", [source], result,
                attributes={
                    "tensor_operation": "slice",
                    "basic_index_axes": (((13,), True),),
                    "basic_index_source_shape": (20,),
                },
            ),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()
    instructions = function.blocks["entry"].instrs
    assert any(
        instruction.op == "Load" and instruction.res is result
        for instruction in instructions
    )
    assert not any(instruction.op == "Call" for instruction in instructions)
    assert result.shape == ()


def test_scalar_arithmetic_discards_provisional_count_one_result_shape():
    left = SSAValue(0, "float64")
    right = SSAValue(1, "float64")
    result = SSAValue(2, "float64", shape=(1,))
    function = Function("stale_scalar_shape", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Sub", [left, right], result),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({function.name: function})

    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    ) == ()
    assert result.shape == ()
    assert not any(
        instruction.op == "extent"
        or instruction.attributes.get("callee") == "broadcast_double"
        for instruction in function.blocks["entry"].instrs
    )


def test_empty_call_operand_does_not_guess_between_reshape_views():
    actual = SSAValue(99)
    first_view = SSAValue(5, "float64", shape=(8, 1))
    second_view = SSAValue(5, "float64", shape=(2, 4, 1))
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("UseFirstView", [first_view], None),
            Instr("UseSecondView", [second_view], None),
            Instr(
                "Call", [actual], None,
                attributes={
                    "callee": "owner__planned_region_0",
                    "feed_ids": (5,),
                },
            ),
            Instr("Ret", [], None),
        ]),
    })
    formal = SSAValue(0)
    callee = Function("owner__planned_region_0", [formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })

    propagate_repository_ssa_call_metadata(IRModule({
        caller.name: caller,
        callee.name: callee,
    }))

    assert formal.shape == ()
    assert formal.dtype is None


def test_region_product_rewire_preserves_consumers_ordered_view_contract():
    aggregate = SSAValue(100, "ssa.aggregate")
    index = SSAValue(101, "int")
    address = SSAValue(102, "ptr")
    produced = SSAValue(5, "float64", shape=(8,))
    provisional = SSAValue(99, "float64", shape=(2, 4))
    producer_call = Instr(
        "Call", [], aggregate,
        attributes={"callee": "producer", "output_ids": (5,)},
    )
    consumer_call = Instr(
        "Call", [provisional], None,
        attributes={
            "callee": "consumer",
            "feed_ids": (5,),
            "feed_shapes": ((2, 4),),
            "feed_dtypes": ("float64",),
        },
    )
    caller = Function("caller", [], {
        "entry": BasicBlock("entry", [
            producer_call,
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [aggregate, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr(
                "Load", [address], produced,
                attributes={"aggregate_index": 0, "source_output_id": 5},
            ),
            consumer_call,
            Instr("Ret", [], None),
        ]),
    })

    assert wire_repository_ssa_region_products(IRModule({caller.name: caller}))
    assert consumer_call.args[0].id == produced.id
    assert consumer_call.args[0].shape == (2, 4)
    assert consumer_call.args[0] is not produced


def test_region_product_rewire_preserves_loop_carried_phi_feed():
    aggregate = SSAValue(100, "ssa.aggregate")
    index = SSAValue(101, "int")
    address = SSAValue(102, "ptr")
    repeated_seed = SSAValue(103, "float64")
    carried_phi = SSAValue(
        104,
        "float64",
        accounting={
            "ssa_storage_alias": 104,
            "ssa_loop_carried_feed": 5,
        },
    )
    producer_call = Instr(
        "Call", [], aggregate,
        attributes={"callee": "seed", "output_ids": (5,)},
    )
    consumer_call = Instr(
        "Call", [carried_phi], None,
        attributes={
            "callee": "consumer",
            "feed_ids": (5,),
            "feed_shapes": ((),),
            "feed_dtypes": ("float64",),
        },
    )
    caller = Function("caller", [], {
        "entry": BasicBlock("entry", [
            producer_call,
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [aggregate, index], address),
            Instr(
                "Load", [address], repeated_seed,
                attributes={"source_output_id": 5},
            ),
            consumer_call,
            Instr("Ret", [], None),
        ]),
    })

    assert not wire_repository_ssa_region_products(
        IRModule({caller.name: caller})
    )
    assert consumer_call.args[0] is carried_phi


def _aggregate_projection(
    aggregate, position, address_id, result, *, source_output_id=None,
):
    index = SSAValue(address_id, "int64")
    address = SSAValue(address_id + 1, "ptr")
    attributes = {"aggregate_index": position}
    if source_output_id is not None:
        attributes["source_output_id"] = source_output_id
    return (
        Instr("Const", [], index, attributes={"value": position}),
        Instr(
            "GetElementPtr", [aggregate, index], address,
            attributes={"aggregate_index": position},
        ),
        Instr("Load", [address], result, attributes=attributes),
    )


def test_materialized_region_output_descriptor_overrides_stale_projection_view():
    output_id = 1177
    computed = SSAValue(output_id, "float64", (8, 4, 128, 2, 3))
    producer = Function("owner__planned_region_0", [], {
        "entry": BasicBlock("entry", [Instr("Compute", [], computed)]),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=output_id,
        data_value_id=output_id,
        shape=(8, 4, 128, 2, 3),
        storage="temporary",
    ))
    aggregate = SSAValue(100, "ssa.aggregate")
    stale_projection = SSAValue(
        output_id, "ssa.aggregate", (8, 4, 128, 1, 3)
    )
    call = Instr(
        "Call", [], aggregate,
        attributes={"callee": producer.name, "output_ids": (output_id,)},
    )
    caller = Function("caller", [], {
        "entry": BasicBlock("entry", [
            call,
            *_aggregate_projection(
                aggregate, 0, 200, stale_projection,
                source_output_id=output_id,
            ),
        ]),
    })
    module = IRModule(
        {producer.name: producer, caller.name: caller},
        tensor_tables={producer.name: table},
    )

    propagate_repository_ssa_call_metadata(module)

    assert stale_projection.shape == (8, 4, 128, 2, 3)
    assert stale_projection.dtype == "float64"


def test_projection_only_aggregate_adapter_collapses_to_real_projections():
    first = SSAValue(10, "float64", (4,))
    second = SSAValue(11, "float64", (4,))
    producer_aggregate = SSAValue(
        100, "ssa.aggregate",
        accounting={"ssa_aggregate_outputs": (10, 11)},
    )
    producer_call = Instr(
        "Call", [], producer_aggregate,
        attributes={
            "callee": "producer",
            "plan_callsite_id": 50,
            "result_convention": "ssa.aggregate",
            "output_ids": (10, 11),
        },
    )
    formal = SSAValue(50, "float64")
    projected_first = SSAValue(20, "float64", (4,))
    projected_second = SSAValue(21, "float64", (4,))
    adapter = Function("owner__planned_region_0", [formal], {
        "entry": BasicBlock("entry", [
            *_aggregate_projection(formal, 0, 200, projected_first),
            *_aggregate_projection(formal, 1, 210, projected_second),
        ]),
    })
    adapter_actual = SSAValue(100, "float64", accounting={
        "ssa_storage_alias": 100,
        "ssa_linked_storage_from": 50,
    })
    adapter_aggregate = SSAValue(101, "ssa.aggregate")
    adapter_call = Instr(
        "Call", [adapter_actual], adapter_aggregate,
        attributes={
            "callee": adapter.name,
            "result_convention": "ssa.aggregate",
            "output_ids": (20, 21, 22),
        },
    )
    caller_first = SSAValue(30, "float64", (4,))
    caller_second_view = SSAValue(31, "float64", (2, 2))
    use_first = Instr("Use", [caller_first], None)
    use_second = Instr("Use", [caller_second_view], None)
    caller = Function("root", [], {
        "entry": BasicBlock("entry", [
            producer_call,
            *_aggregate_projection(
                producer_aggregate, 0, 300, first, source_output_id=10,
            ),
            *_aggregate_projection(
                producer_aggregate, 1, 310, second, source_output_id=11,
            ),
            adapter_call,
            *_aggregate_projection(adapter_aggregate, 0, 320, caller_first),
            *_aggregate_projection(
                adapter_aggregate, 2, 330, caller_second_view,
            ),
            use_first,
            use_second,
        ]),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=22, data_value_id=21, shape=(2, 2), storage="view",
        owns_allocation=False, allocation_owner=21, alias_of=21,
    ))
    module = IRModule(
        {"producer": Function("producer", [], {}), caller.name: caller,
         adapter.name: adapter},
        tensor_tables={adapter.name: table},
    )

    assert legalize_aggregate_adapters(module)

    assert adapter_call not in caller.blocks["entry"].instrs
    assert adapter.name not in module.functions
    assert use_first.args[0].id == first.id
    assert use_first.args[0] is not first
    assert use_second.args[0].id == second.id
    assert use_second.args[0].shape == (2, 2)
    assert not any(
        value.dtype == "ssa.aggregate" for value in caller.args
    )


def test_mixed_aggregate_adapter_keeps_only_computed_result_and_used_feed():
    producer_aggregate = SSAValue(100, "ssa.aggregate")
    producer_call = Instr(
        "Call", [], producer_aggregate,
        attributes={
            "callee": "producer",
            "plan_callsite_id": 50,
            "result_convention": "ssa.aggregate",
            "output_ids": (10, 11, 12),
            "output_positions": (0, 2, 4),
        },
    )
    produced = [SSAValue(value_id, "float64", (4,)) for value_id in (10, 11, 12)]
    formal = SSAValue(50, "float64")
    projected = [SSAValue(value_id, "float64", (4,)) for value_id in (20, 21, 22)]
    reduced = SSAValue(30, "float64")
    adapter = Function("owner__planned_region_1", [formal], {
        "entry": BasicBlock("entry", [
            *_aggregate_projection(formal, 0, 200, projected[0]),
            *_aggregate_projection(formal, 2, 210, projected[1]),
            *_aggregate_projection(formal, 4, 220, projected[2]),
            Instr("Reduce", [projected[0]], reduced),
        ]),
    })
    adapter_actual = SSAValue(100, "float64", accounting={
        "ssa_storage_alias": 100,
        "ssa_linked_storage_from": 50,
    })
    adapter_aggregate = SSAValue(101, "ssa.aggregate")
    adapter_call = Instr(
        "Call", [adapter_actual], adapter_aggregate,
        attributes={
            "callee": adapter.name,
            "result_convention": "ssa.aggregate",
            "output_ids": (21, 30),
        },
    )
    pass_value = SSAValue(40, "float64", (4,))
    computed_value = SSAValue(41, "float64")
    use_pass = Instr("Use", [pass_value], None)
    use_computed = Instr("Use", [computed_value], None)
    caller_instructions = [producer_call]
    for position, value in zip((0, 2, 4), produced):
        caller_instructions.extend(_aggregate_projection(
            producer_aggregate, position, 300 + position * 10, value,
            source_output_id=value.id,
        ))
    caller_instructions.extend((
        adapter_call,
        *_aggregate_projection(adapter_aggregate, 0, 400, pass_value),
        *_aggregate_projection(adapter_aggregate, 1, 410, computed_value),
        use_pass,
        use_computed,
    ))
    caller = Function("root", [], {
        "entry": BasicBlock("entry", caller_instructions),
    })
    module = IRModule({
        "producer": Function("producer", [], {}),
        caller.name: caller,
        adapter.name: adapter,
    })

    assert legalize_aggregate_adapters(module)

    assert [value.id for value in adapter.args] == [20]
    assert [value.id for value in adapter_call.args] == [10]
    assert adapter_call.attributes["output_ids"] == (30,)
    assert adapter_call.attributes["output_positions"] == (1,)
    assert use_pass.args[0].id == 11
    assert use_computed.args[0] is computed_value
    assert all(value.dtype != "ssa.aggregate" for value in caller.args)


def test_aggregate_output_views_share_the_callee_resident_value():
    computed = SSAValue(10, "float64", (8,))
    callee = Function("planned", [], {
        "entry": BasicBlock("entry", [Instr("Compute", [], computed)]),
    })
    table = SSATensorTable()
    for tensor_id, shape in ((20, (8, 1)), (21, (2, 4))):
        table.register(SSATensorDescriptor(
            tensor_id=tensor_id,
            data_value_id=10,
            shape=shape,
            storage="view",
            owns_allocation=False,
            allocation_owner=10,
            alias_of=10,
        ))
    aggregate = SSAValue(100, "ssa.aggregate")
    call = Instr(
        "Call", [], aggregate,
        attributes={
            "callee": callee.name,
            "result_convention": "ssa.aggregate",
            "output_ids": (20, 21),
        },
    )
    caller = Function("caller", [], {
        "entry": BasicBlock("entry", [call]),
    })
    module = IRModule(
        {callee.name: callee, caller.name: caller},
        tensor_tables={callee.name: table},
    )

    assert legalize_aggregate_output_views(module)
    assert call.attributes["output_ids"] == (20, 21)
    assert call.attributes["callee_output_ids"] == (10, 10)


def test_aggregate_output_view_of_formal_rebinds_to_caller_actual():
    formal = SSAValue(10, "float64", (8,))
    callee = Function("planned", [formal], {
        "entry": BasicBlock("entry", []),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=20,
        data_value_id=10,
        shape=(2, 4),
        storage="view",
        owns_allocation=False,
        allocation_owner=10,
        alias_of=10,
    ))
    actual = SSAValue(5, "float64", (8,))
    aggregate = SSAValue(100, "ssa.aggregate")
    call = Instr(
        "Call", [actual], aggregate,
        attributes={
            "callee": callee.name,
            "result_convention": "ssa.aggregate",
            "output_ids": (20,),
        },
    )
    projected = SSAValue(30, "float64", (2, 4))
    use = Instr("Use", [projected], None)
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            call,
            *_aggregate_projection(aggregate, 0, 200, projected),
            use,
        ]),
    })
    module = IRModule(
        {callee.name: callee, caller.name: caller},
        tensor_tables={callee.name: table},
    )

    assert legalize_aggregate_output_views(module)
    assert call.attributes["output_ids"] == ()
    assert call.attributes["callee_output_ids"] == ()
    assert use.args[0].id == actual.id
    assert use.args[0].shape == (2, 4)


def test_aggregate_output_view_keeps_independently_consumed_call_result():
    formal = SSAValue(10, "bool")
    callee = Function("planned", [formal], {
        "entry": BasicBlock("entry", []),
    })
    table = SSATensorTable()
    table.register(SSATensorDescriptor(
        tensor_id=20,
        data_value_id=10,
        shape=(),
        storage="view",
        owns_allocation=False,
        allocation_owner=10,
        alias_of=10,
    ))
    actual = SSAValue(5, "bool")
    direct_result = SSAValue(30, "bool")
    aggregate = SSAValue(100, "ssa.aggregate")
    call = Instr(
        "Call", [actual], aggregate,
        attributes={
            "callee": callee.name,
            "result_convention": "ssa.aggregate",
            "output_ids": (direct_result.id,),
            "callee_output_ids": (20,),
        },
    )
    projected = SSAValue(31, "bool")
    direct_use = Instr("LNot", [direct_result], SSAValue(32, "bool"))
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            call,
            *_aggregate_projection(aggregate, 0, 200, projected),
            direct_use,
        ]),
    })
    module = IRModule(
        {callee.name: callee, caller.name: caller},
        tensor_tables={callee.name: table},
    )

    assert legalize_aggregate_output_views(module)
    assert call.attributes["output_ids"] == (direct_result.id,)
    assert call.attributes["callee_output_ids"] == (formal.id,)
    assert direct_use.args[0] is direct_result
