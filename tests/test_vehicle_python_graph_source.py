from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.compiler.vehicle_balloon_tire_program import (
    BALLOON_TIRE_VECTOR_SOURCE,
    balloon_tire_python_program,
)
from src.compiler.vehicle_native_graph_program import (
    VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE,
    vehicle_native_graph_python_program,
)


def test_balloon_tire_python_source_is_topology_vectorized():
    program = balloon_tire_python_program()
    compile(program.source, "<balloon-tire-vector>", "exec")
    assert program.constants["face_scatter"].shape == (
        3, program.vertex_count, program.face_count)
    assert program.constants["laplacian"].shape == (
        program.vertex_count, program.vertex_count)
    edge_count = program.constants["bending_weight"].shape[0]
    assert program.constants["bending_incidence"].shape == (
        edge_count, program.vertex_count)
    assert program.constants["bending_scatter"].shape == (
        program.vertex_count, edge_count)
    assert program.constants["vertex_area"].shape == (program.vertex_count,)
    assert "AbstractTensor.matmul" in program.source
    assert ".gather(" in program.source
    assert "for vertex in" not in program.source
    assert "for face in" not in program.source
    assert "#include" not in program.source
    assert "double " not in program.source


def test_balloon_tire_axis_is_specialized_from_wheel_identities_not_corners():
    for wheel_names in ((), ("only",), ("a", "b", "c", "d"),
                        tuple(f"wheel_{index}" for index in range(7))):
        program = balloon_tire_python_program(wheel_names)
        assert program.constants["wheel_input_indices"].shape == (
            len(wheel_names), 41)
        assert program.state_scalar_count == (
            len(wheel_names) * program.vertex_count * 6)
        assert len(program.output_names) == len(wheel_names) * 14
        assert all(
            name.startswith(f"{wheel}.")
            for wheel in wheel_names
            for name in program.output_names
            if name.startswith(f"{wheel}.")
        )


def test_balloon_tire_axis_rejects_duplicate_wheel_identities():
    import pytest

    with pytest.raises(ValueError, match="unique"):
        balloon_tire_python_program(("same", "same"))


def test_balloon_tire_initializer_executes_zero_one_and_many_wheel_axes():
    from src.common.tensors import AbstractTensor

    for wheel_names in ((), ("only",), tuple(f"wheel_{index}"
                                              for index in range(7))):
        program = balloon_tire_python_program(wheel_names)
        namespace = {"AbstractTensor": AbstractTensor}
        exec(program.source, namespace)
        state = AbstractTensor.tensor(np.zeros(
            (1, len(wheel_names), program.vertex_count, 6),
            dtype=np.float64))
        initialized = namespace["balloon_tire_vector_initialize"](
            AbstractTensor.tensor(program.constants["default_input"][None, :]),
            state,
            AbstractTensor.tensor(program.constants["wheel_input_indices"]),
            AbstractTensor.tensor(program.constants["rest"]),
        )
        assert initialized.shape == (
            1, len(wheel_names), program.vertex_count, 6)
        assert np.isfinite(initialized.data).all()


def test_closed_vehicle_python_source_has_all_parallel_axes():
    program = vehicle_native_graph_python_program()
    compile(program.source, "<vehicle-graph-vector>", "exec")
    assert program.vehicle_specific_c_lines == 0
    assert set(program.vector_axes) == {
        "batch", "wheel", "vertex", "face", "edge", "rig-point",
        "contact-surface", "xyz",
    }
    assert "vehicle_rig_points_vector" in program.source
    assert "vehicle_material_bank_vector" in program.source
    assert "vehicle_tire_recurrence" in program.source
    assert "vehicle_close_contact_graph" in program.source
    assert "#include" not in program.source
    assert "typedef " not in program.source
    assert "double " not in program.source


def test_material_graph_constants_replace_native_struct_tables():
    constants = vehicle_native_graph_python_program().constants
    assert constants is not None
    assert constants.node_reference.shape[1] == 3
    assert constants.node_structural_support_binding.shape == (
        constants.node_reference.shape[0], 4)
    assert constants.edge_nodes.shape[1] == 2
    assert constants.edge_geometry.shape[1] == 13
    assert constants.structural_support_edge_mask.shape == (
        4, constants.edge_nodes.shape[0])


def test_only_true_time_recurrence_remains_a_python_loop():
    numerical_lines = tuple(
        line.strip() for line in VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE.splitlines()
        if line.strip().startswith("for ")
    )
    assert numerical_lines == ("for microstep in range(microstep_count):",)
    assert "for " not in BALLOON_TIRE_VECTOR_SOURCE
    assert "tire_state, tire_output = balloon_tire_vector_step" in (
        VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE
    )


def test_vehicle_tire_recurrence_publishes_microstep_duration():
    assert "tire_inputs[:, 0] = outer_dt / microstep_count" in (
        VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE
    )


def test_stack_region_keeps_the_authored_list_element_producers(tmp_path):
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import (
        ExtractionContract, ProgramABIContract,
    )
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.tensor_ssa_lowering import (
        lower_tensor_calls_to_repository_ssa,
    )

    tree = ast.parse(VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE)
    graph_cross = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "graph_cross"
    )
    source = ast.unparse(ast.Module(body=[graph_cross], type_ignores=[]))
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi(ProgramABIContract.from_mapping({
        "records": {}, "bindings": [], "values": [
            {
                "function": "graph_cross", "parameter": parameter,
                "storage": "span", "dtype": "float64", "rank": 3,
                "shape": [8, 16, 3],
                "python_type": (
                    "src.common.tensors.abstraction.AbstractTensor"
                ),
            }
            for parameter in ("left", "right")
        ],
    }))
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source,
        "graph_cross",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        name="stack_leaf_producers",
    )

    regions = tuple(
        function for function in module.functions.values()
        if function.metadata.get("source_region_integral")
    )
    arithmetic = next(
        function for function in regions
        if len(function.metadata["source_region_integral"]["output_value_ids"])
        == 3
    )
    stack_region = next(
        function for function in regions
        if any(
            instruction.op == "stack"
            for block in function.blocks.values()
            for instruction in block.instrs
        )
    )
    stack = next(
        instruction
        for block in stack_region.blocks.values()
        for instruction in block.instrs
        if instruction.op == "stack"
    )

    assert stack.attributes["dim"] == -1
    assert tuple(value.id for value in stack.args[:-1]) == tuple(
        arithmetic.metadata["source_region_integral"]["output_value_ids"]
    )
    assert tuple(value.id for value in stack_region.args) == tuple(
        value.id for value in stack.args[:-1]
    )
    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )
    assert shortfalls == ()
    root = next(name for name in module.functions if name.endswith("__graph_cross"))
    returned = next(
        instruction.args[0]
        for block in module.functions[root].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert returned.shape == (8, 16, 3)
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "graph_cross")


def test_declared_rig_uses_tensor_indexing_and_native_call_outputs(tmp_path):
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import (
        ExtractionContract, ProgramABIContract,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    selected = {"graph_cross", "graph_norm", "vehicle_rig_points_vector"}
    tree = ast.parse(VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE)
    source = ast.unparse(ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ],
        type_ignores=[],
    ))
    shapes = {
        "body_position": (8, 3), "body_velocity": (8, 3),
        "attitude": (8, 3), "angular_velocity": (8, 3),
        "rig_points": (8, 16, 21),
    }
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi(ProgramABIContract.from_mapping({
        "records": {}, "bindings": [], "values": [
            {
                "function": "vehicle_rig_points_vector",
                "parameter": parameter,
                "storage": "span", "dtype": "float64",
                "rank": len(shape), "shape": list(shape),
                "python_type": (
                    "src.common.tensors.abstraction.AbstractTensor"
                ),
            }
            for parameter, shape in shapes.items()
        ],
    }))
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source,
        "vehicle_rig_points_vector",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name="declared_vehicle_rig",
    )
    callees = {
        instruction.attributes.get("callee")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    }
    assert "index_select_double" in callees
    assert "index_assign_double" in callees
    root = next(
        name for name in module.functions
        if name.endswith("__vehicle_rig_points_vector")
    )
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "declared_vehicle_rig")


def test_declared_material_bank_links_all_symbolic_outputs(tmp_path):
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import (
        ExtractionContract, ProgramABIContract,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.vehicle_mechanical_material import (
        compile_vehicle_member_material_ssa,
    )

    selected = {
        "graph_norm", "vehicle_member_material_vector",
        "vehicle_material_bank_vector",
    }
    tree = ast.parse(VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE)
    source = ast.unparse(ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ],
        type_ignores=[],
    ))
    shapes = {
        "node_positions": (8, 845, 3),
        "node_velocities": (8, 845, 3),
        "material_state": (8, 378, 9),
        "edge_nodes": (378, 2),
        "edge_geometry": (378, 13),
        "structural_support_edge_mask": (4, 378),
        "dt": (8,),
    }
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi(ProgramABIContract.from_mapping({
        "records": {}, "bindings": [], "values": [
            {
                "function": "vehicle_material_bank_vector",
                "parameter": parameter,
                "storage": "span", "dtype": "float64",
                "rank": len(shape), "shape": list(shape),
                "python_type": (
                    "src.common.tensors.abstraction.AbstractTensor"
                ),
            }
            for parameter, shape in shapes.items()
        ],
    }))
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source,
        "vehicle_material_bank_vector",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        linked_process_graphs={
            "vehicle_member_material_step": (
                compile_vehicle_member_material_ssa().process_graph
            ),
        },
        name="declared_material_bank",
    )
    root = next(
        name for name in module.functions
        if name.endswith("__vehicle_material_bank_vector")
    )
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "declared_material_bank")


def test_vehicle_entry_abi_uses_only_physical_values():
    from src.compiler.vehicle_python_compilation import (
        vehicle_python_compilation_inputs,
        vehicle_python_extraction_contract,
    )

    inputs = vehicle_python_compilation_inputs()
    assert all(not isinstance(value, tuple) for value in inputs.feeds.values())
    assert "material_parameters" not in inputs.feeds
    assert {
        "tire_previous_hub", "tire_previous_basis", "tire_previous_angle",
        "tire_previous_plane", "tire_wheel_input_indices", "tire_rest",
        "tire_face_vertices", "tire_face_rest", "tire_face_scatter",
        "tire_bending_incidence", "tire_bending_scatter",
        "tire_bending_weight", "tire_vertex_area", "tire_bead_mask",
        "tire_face_material",
    } <= inputs.feeds.keys()
    contract = vehicle_python_extraction_contract(inputs)
    declared = {
        binding.parameter
        for binding in contract.program_abi.values
        if binding.function == inputs.entrypoint
    }
    assert declared == set(inputs.feeds)


def test_vehicle_contract_requires_whole_program_native_closure():
    from src.compiler.vehicle_python_compilation import (
        VehiclePythonCompilationInputs,
        vehicle_python_extraction_contract,
    )

    inputs = VehiclePythonCompilationInputs(
        source="def main(value):\n    return value\n",
        entrypoint="main",
        feeds={"value": np.zeros((2, 3), dtype=np.float64)},
        linked_process_graphs={},
    )
    contract = vehicle_python_extraction_contract(inputs)

    assert contract.execution.host_runtime == "native"
    assert contract.execution.native_lowering == "required"
    assert contract.execution.dispatch_unit == "whole_program"
    assert contract.execution.unlowered_behavior == "reject"
    assert contract.execution.require_full_native is True
    assert contract.execution.python_callbacks == "reject"
    print_decision = contract.decide(print)
    assert print_decision.action.value == "reject"
    assert print_decision.rule_id == "execution:python_callbacks_reject"
    assert Path(contract.execution_overlay_path).name == (
        "vehicle_full_native_execution.yaml"
    )
    assert {
        binding.parameter for binding in contract.program_abi.values
    } == {"value"}


def test_active_native_build_does_not_import_legacy_vehicle_c():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "build_vehicle_native_teaser.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "src.compiler.vehicle_balloon_tire_native" not in imported
    assert "src.compiler.vehicle_native_deployment" not in imported
    assert "emit_vehicle_python_graph_c" in source


def test_vehicle_c_emitter_imports_its_shortfall_summarizer_in_scope():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "compiler" / "vehicle_python_compilation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    emitter = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_vehicle_python_graph_c"
    )
    imported = {
        alias.name
        for node in ast.walk(emitter)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {"emit_ssa_to_c", "summarize_c_shortfalls"} <= imported


def test_active_tire_authority_does_not_import_legacy_vehicle_c():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "compiler" / "vehicle_tire_authority.py").read_text(
        encoding="utf-8")
    assert "vehicle_balloon_tire_native" not in source
    assert "compile_native_balloon_tire_assembly" not in source
    assert "emit_balloon_tire_python_c" in source


def test_active_balloon_emitters_enter_through_repository_ssa():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "compiler" / "vehicle_python_compilation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("emit_balloon_tire_python_c", "emit_balloon_tire_python_llvm"):
        rendered = ast.unparse(functions[name])
        assert "lower_balloon_tire_python_ssa" in rendered
        assert "compile_balloon_tire_python_aot" not in rendered
        assert "lower_precompile_and_control_to_ssa" not in rendered


def test_managed_tire_uses_shared_strict_dt_and_exports_telemetry():
    from src.compiler.vehicle_python_compilation import (
        BALLOON_TIRE_MANAGED_SOURCE,
        balloon_tire_managed_python_compilation_inputs,
    )

    inputs = balloon_tire_managed_python_compilation_inputs(
        8, window_duration=1.0 / 120.0, dt_initial=1.0 / 120.0
    )
    material = inputs.feeds["material"]

    assert material.telemetry.shape == (20,)
    assert inputs.feeds["window_duration"] == 1.0 / 120.0
    assert inputs.feeds["dt_initial"] == 1.0 / 120.0
    assert "run_superstep(" in BALLOON_TIRE_MANAGED_SOURCE
    assert "allow_unresolved=False" in BALLOON_TIRE_MANAGED_SOURCE
    assert ".isfinite().all()" in BALLOON_TIRE_MANAGED_SOURCE
    assert "material.telemetry[16] = dt" in BALLOON_TIRE_MANAGED_SOURCE


def test_one_vehicle_program_materializes_semantic_eager_feed_dtypes():
    from src.compiler.vehicle_python_compilation import (
        vehicle_python_compilation_inputs,
    )

    program = vehicle_python_compilation_inputs(1)
    feeds = program.abstract_tensor_feeds()
    assert feeds["edge_nodes"].dtype.kind == "i"
    assert feeds["tire_wheel_input_indices"].dtype.kind == "i"
    assert feeds["tire_face_vertices"].dtype.kind == "i"
    assert feeds["tire_bead_mask"].dtype.kind == "b"
    assert feeds["tire_initialized"].dtype.kind == "b"
    assert program.source is vehicle_python_compilation_inputs(1).source
