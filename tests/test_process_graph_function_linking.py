import ast
import contextlib
import io
import _pickle
from pathlib import Path

import sympy

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics
from src.compiler.process_graph_function_linking import link_process_graph_functions
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator
from src.transmogrifier.graph.graph_express2 import ProcessGraph


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


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
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == "pair_direct__linked_pair"
        and instruction.attributes.get("result_convention") == "ssa.aggregate"
        for block in root.blocks.values()
        for instruction in block.instrs
    )


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
    )

    root = module.functions["break_aware_loop_call__root"]
    linked = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "break_aware_loop_call__step"
    )
    loop_result = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "loop_result_port"
        and instruction.res is linked.args[0]
    )

    assert loop_result.res.id == linked.args[0].id
    assert len(loop_result.args) == 2
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
        extraction_contract=CONTRACT,
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
    for name in ("error_channels.keys", "error_channels.values"):
        accounting = slots[name].accounting or {}
        assert accounting["program_abi_storage"] == "span"
        assert int(accounting["program_abi_rank"]) == 1

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
        extraction_contract=CONTRACT,
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
        extraction_contract=CONTRACT,
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
        extraction_contract=CONTRACT,
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
        region = module.functions[str(region_calls[0].attributes["callee"])]
        region_ops = [
            instruction.op
            for block in region.blocks.values()
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
        extraction_contract=CONTRACT,
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
        extraction_contract=CONTRACT,
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
