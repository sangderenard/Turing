from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler import machine_targets


def _program(*steps, feeds=(1,), outputs=None):
    return FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=list(steps),
        outputs=outputs or {"result": steps[-1].result_id},
    )


def test_existing_backend_operator_lists_are_exposed_without_a_fifth_copy():
    inventories = {
        item.backend: item for item in machine_targets.operator_inventories()
    }

    assert set(inventories) == {"c", "glsl", "fortran", "llvm", "webgl"}
    assert len(inventories["c"].operations) == 40
    assert len(inventories["glsl"].operations) == 56
    assert len(inventories["fortran"].operations) == 57
    assert len(inventories["llvm"].operations) == 70
    assert len(inventories["webgl"].operations) == 44
    assert inventories["c"].operations <= inventories["llvm"].operations
    assert (
        inventories["webgl"].operations - {"tensor_from_list"}
        <= inventories["glsl"].operations
    )
    assert all(item.sources for item in inventories.values())


def test_glsl_fortran_and_webgl_join_the_machine_target_hub():
    names = {item.name for item in machine_targets.capabilities()}
    assert {"wasm", "glsl", "fortran", "webgl"} <= names


def test_closed_target_vocabularies_reject_unknown_ops_before_emission():
    unknown = _program(OpStep(0, "teleport", [1], {}, 2))

    assert not ({"wasm", "glsl", "fortran", "webgl"} & set(
        machine_targets.targets_for(unknown)
    ))


def test_webgl_prints_a_fragment_program_from_the_shared_glsl_expressions():
    program = _program(
        OpStep(0, "add", [1, 2], {}, 3),
        OpStep(1, "sin", [3], {}, 4),
        feeds=(1, 2),
    )

    artifact = machine_targets.emit(program, "webgl", name="browser_kernel")

    assert artifact.complete
    assert artifact.extension == ".frag.glsl"
    assert artifact.source.startswith("#version 300 es")
    assert "uniform sampler2D turing_feed_1;" in artifact.source
    assert "texelFetch(turing_feed_2, turing_coordinate, 0).r" in artifact.source
    assert "float v_3 = v_1 + v_2;" in artifact.source
    assert "float v_4 = sin(v_3);" in artifact.source
    assert "layout(location = 0) out vec4 turing_output_0;" in artifact.source
    assert artifact.api.to_mapping()["metadata"]["execution_model"] == (
        "fragment-raster"
    )


def test_webgl_names_non_fragment_layouts_and_oversized_mrt_as_shortfalls():
    bitwise = _program(OpStep(0, "bitand", [1, 2], {}, 3), feeds=(1, 2))
    multiple = _program(
        OpStep(0, "add", [1, 2], {}, 3),
        feeds=(1, 2),
        outputs={"left": 1, "sum": 3},
    )
    oversized = _program(
        OpStep(0, "add", [1, 2], {}, 3),
        feeds=(1, 2),
        outputs={f"plane_{index}": 3 for index in range(5)},
    )

    assert "webgl" not in machine_targets.targets_for(bitwise)
    artifact = machine_targets.emit(multiple, "webgl", name="two_outputs")
    assert artifact.complete
    assert "layout(location = 1) out vec4 turing_output_1;" in artifact.source
    artifact = machine_targets.emit(oversized, "webgl", name="five_outputs")
    assert not artifact.complete
    assert "between one and four" in artifact.shortfalls[0]


def test_fortran_and_desktop_glsl_print_the_same_numeric_program():
    program = _program(
        OpStep(0, "mul", [1], {"right_scalar": 2.0}, 2),
        OpStep(1, "cos", [2], {}, 3),
    )

    fortran = machine_targets.emit(program, "fortran", name="numeric_map")
    glsl = machine_targets.emit(program, "glsl", name="numeric_map")

    assert fortran.complete
    assert "module numeric_map" in fortran.source
    assert "cos(" in fortran.source
    assert fortran.api.to_mapping()["language"] == "fortran"
    assert glsl.complete
    assert glsl.source.startswith("#version 430")
    assert "cos(" in glsl.source
