"""Tests for the process-graph schedule table shell, against class modules
segmented by partition_reduced_program (see wasm_class_modules.py)."""

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.process_graph_shell import emit_process_graph_shell, schedule_table
from src.compiler.wasm_class_modules import partition_reduced_program


def _linear_program(n_steps: int, feed_id: int = 1) -> FusedProgram:
    steps = []
    previous = feed_id
    for i in range(n_steps):
        result_id = 100 + i
        steps.append(OpStep(
            step_id=i, op_name="mul", input_ids=[previous],
            attrs={"right_scalar": 2.0}, result_id=result_id,
        ))
        previous = result_id
    return FusedProgram(
        version=1, feeds={feed_id}, steps=steps, outputs={"result": previous},
    )


def _compile_two_function_specs(chunk_size=2):
    source = (
        "def helper(x):\n"
        "    return x * 2.0\n"
        "\n"
        "def kernel(a):\n"
        "    return helper(a) + 1.0\n"
    )
    aot = compile_ast_aot(
        source, "kernel", {"a": np.array([20.5])}, precompile_only=True,
    )
    program = getattr(
        aot.compiled_shell_program, "program", aot.compiled_shell_program
    )
    return partition_reduced_program(
        program, chunk_size=chunk_size, owner_name=aot.entrypoint,
    )


def test_aot_keeps_annotations_in_maps_and_out_of_runtime_dispatch():
    source = '''
ModuleValue: float = 3.0

class Scale:
    factor: float = 2.0

def kernel(value):
    result: float = value * 2.0
    return result
'''
    aot = compile_ast_aot(
        source,
        "kernel",
        {"value": np.array([4.0])},
        precompile_only=True,
    )

    schema = aot.map_ir["schema"]
    assert schema["module"]["annotations"][0]["name"] == "ModuleValue"
    assert schema["classes"][0]["members"][0]["name"] == "factor"
    function = next(
        item for item in schema["functions"] if item["identity"] == "kernel"
    )
    assert function["locals"][0]["name"] == "result"
    assert aot.function_outputs == ("result",)
    assert aot.function_parameters == ("value",)
    assert aot.identity_table["result"]
    assert aot.identity_table["result"][-1] == max(
        aot.identity_table["result"]
    )


def test_aot_records_and_validates_the_requested_bake_mode():
    source = (
        "def helper(x):\n"
        "    return x * 2.0\n"
        "\n"
        "def kernel(x):\n"
        "    return helper(x)\n"
    )
    one_shot = compile_ast_aot(
        source,
        "kernel",
        {"x": np.array([3.0])},
        precompile_only=True,
        bake_mode="one-shot",
        schedule_preference="ASAP",
    )

    assert one_shot.bake_mode == "one_shot"
    assert one_shot.schedule_preference == "asap"
    with pytest.raises(ValueError, match="one_shot.*whole_program"):
        compile_ast_aot(
            source,
            "kernel",
            {"x": np.array([3.0])},
            precompile_only=True,
            bake_mode="partial",
        )


def test_configured_parameter_constants_apply_before_graph_reduction():
    source = "def kernel(x, gain):\n    return x * gain\n"
    aot = compile_ast_aot(
        source,
        "kernel",
        {"x": np.array([3.0]), "gain": np.array([99.0])},
        precompile_only=True,
        constant_map={"gain": 2.0},
    )
    program = project_public_numerical_program(aot)

    assert aot.program_record_mode == "configured"
    assert aot.constant_map == {"gain": 2.0}
    assert len(program.feeds) == 1
    assert any(step.attrs.get("right_scalar") == 2.0 for step in program.steps)
    with pytest.raises(ValueError, match="asap.*alap"):
        compile_ast_aot(
            source,
            "kernel",
            {"x": np.array([3.0])},
            precompile_only=True,
            schedule_preference="middle",
        )


def test_schedule_table_is_built_fresh_from_real_compiled_ir():
    """Not a fixture: pulled from compile_ast_aot's own compiled_shell_program,
    cut into chunks, run through the real ProcessGraph/ILPScheduler, same as
    test_wasm_class_modules.py's scheduling tests."""

    specs = _compile_two_function_specs()
    table = schedule_table(specs)

    ids = {node["id"] for node in table["nodes"]}
    assert ids == {spec.module_name for spec in specs}

    root_node = next(n for n in table["nodes"] if n["is_root"])
    # The root is the process graph's owner -- named after the entrypoint
    # itself ("kernel"), not a chunk-numbered label.
    assert root_node["id"] == "kernel__" + str(
        next(s for s in specs if s.is_root).index
    )
    assert root_node["level"] == 0
    assert table["level_max"] == 0
    if len(specs) > 1:
        # Every prerequisite chunk sits behind the root, at a negative
        # level -- not counted up from an unrelated zero.
        assert table["level_min"] < 0
        for node in table["nodes"]:
            if not node["is_root"]:
                assert node["level"] < 0


def test_schedule_table_gives_a_single_node_its_own_row_and_column():
    specs = partition_reduced_program(
        _linear_program(3), chunk_size=10, owner_name="kernel",
    )
    table = schedule_table(specs)
    assert table["nodes"] == [
        {"id": "kernel__0", "level": 0, "group": 0, "is_root": True},
    ]
    assert table["edges"] == []
    assert table["level_min"] == 0
    assert table["level_max"] == 0
    assert table["groups"] == 1


def test_a_multi_chunk_program_lands_on_separate_rows():
    specs = partition_reduced_program(
        _linear_program(5), chunk_size=2, owner_name="kernel",
    )
    table = schedule_table(specs)
    levels = sorted(n["level"] for n in table["nodes"])
    assert levels == [-2, -1, 0]
    assert table["level_min"] == -2
    assert table["level_max"] == 0
    # A linear chain has no independent branches -- every chunk is in the
    # one weakly-connected component.
    assert table["groups"] == 1


def test_two_disjoint_call_trees_get_separate_groups():
    """The case the group definition (weakly-connected component) does
    distinguish: two entirely disjoint dependency graphs."""

    specs_a = partition_reduced_program(
        _linear_program(3, feed_id=1), chunk_size=10, owner_name="leaf_a",
    )
    specs_b = partition_reduced_program(
        _linear_program(3, feed_id=2), chunk_size=10, owner_name="leaf_b",
    )
    table = schedule_table([*specs_a, *specs_b])
    node_a = next(n for n in table["nodes"] if n["id"].startswith("leaf_a"))
    node_b = next(n for n in table["nodes"] if n["id"].startswith("leaf_b"))
    assert node_a["group"] != node_b["group"]
    assert table["groups"] == 2


def test_emit_process_graph_shell_embeds_the_schedule_as_json():
    specs = _compile_two_function_specs()
    table = schedule_table(specs)
    shell = emit_process_graph_shell(table, title="t")

    assert "<table" in shell.html
    assert "renderSchedule(SCHEDULE)" in shell.html
    for node in table["nodes"]:
        assert node["id"] in shell.html


def test_write_puts_the_page_on_disk(tmp_path):
    specs = _compile_two_function_specs()
    shell = emit_process_graph_shell(schedule_table(specs))
    path = shell.write(str(tmp_path / "out" / "index.html"))
    assert (tmp_path / "out" / "index.html").read_text(encoding="utf-8") == shell.html
    assert path == str(tmp_path / "out" / "index.html")
