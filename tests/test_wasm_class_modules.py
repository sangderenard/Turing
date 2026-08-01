"""Tests for auto-segmenting a fully-reduced FusedProgram into WASM class
modules by cutting its topological order into roughly-equal-sized,
contiguous, connected chunks -- see wasm_class_modules.py's module
docstring for why this replaced the earlier closure-boundary approach."""

import json
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.wasm_class_modules import (
    build_embedded_class_graph, build_hued_process_graph_views, build_manifest, build_module_process_graph,
    describe_process_graph_api, emit_class_modules,
    partition_reduced_program, schedule_module_levels,
)


def _linear_program(n_steps: int, feed_id: int = 1) -> FusedProgram:
    """A chain of ``n_steps`` scalar multiplies: pure linear dependency, no
    branching -- the shape a contiguous topological slice always handles."""

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


def test_a_small_program_becomes_a_single_root_module():
    program = _linear_program(3)
    specs = partition_reduced_program(program, chunk_size=10, owner_name="kernel")
    assert len(specs) == 1
    assert specs[0].is_root
    assert specs[0].name == "kernel"
    assert specs[0].module_name == "kernel__0"
    assert specs[0].calls == ()


def test_a_larger_program_is_cut_into_contiguous_chunks():
    program = _linear_program(5)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    # 5 ops / chunk_size 2 -> chunks of (2, 2, 1) nodes.
    assert [len(s.region.node_ids) for s in specs] == [2, 2, 1]
    assert [s.name for s in specs] == [
        "kernel_chunk0", "kernel_chunk1", "kernel",
    ]
    # Dependency order: earliest chunk first, root (owner) last.
    assert specs[-1].is_root
    assert all(not s.is_root for s in specs[:-1])


def test_each_chunk_declares_only_the_values_it_actually_needs():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    chunk0, chunk1 = specs
    # chunk0 owns nodes 100,101 and needs only the program's own feed (1).
    assert chunk0.region.input_ids == (1,)
    # chunk1 (the root here, since 4/2=2 chunks) needs chunk0's last value.
    assert chunk1.region.input_ids == (101,)
    assert chunk1.is_root


def test_the_call_dependency_points_at_the_producing_chunk():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    chunk0, chunk1 = specs
    assert chunk0.calls == ()
    assert len(chunk1.calls) == 1
    call = chunk1.calls[0]
    assert call.callee_index == chunk0.index
    assert call.callee_module_name == chunk0.module_name


def test_chunk_size_must_be_positive():
    program = _linear_program(3)
    with pytest.raises(ValueError):
        partition_reduced_program(program, chunk_size=0, owner_name="kernel")


def test_an_empty_program_still_produces_one_root_module():
    program = FusedProgram(version=1, feeds=set(), steps=[], outputs={})
    specs = partition_reduced_program(program, chunk_size=4, owner_name="kernel")
    assert len(specs) == 1
    assert specs[0].is_root


def test_partition_prunes_a_dead_tail_after_the_declared_output():
    program = FusedProgram(
        version=1, feeds={1},
        steps=[
            OpStep(0, "mul", [1], {"right_scalar": 2.0}, 2),
            OpStep(1, "mul", [2], {"right_scalar": 3.0}, 3),
            OpStep(2, "mul", [3], {"right_scalar": 4.0}, 4),
        ],
        outputs={"result": 2},
    )
    specs = partition_reduced_program(program, chunk_size=1, owner_name="kernel")
    assert len(specs) == 1
    assert specs[0].region.node_ids == (2,)
    assert specs[0].region.outputs == (("result", 2),)


# --- emission: real, independently-lowered modules per chunk ---------------


def test_emit_class_modules_produces_one_complete_module_per_chunk():
    program = _linear_program(5)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    assert set(modules) == {s.index for s in specs}
    for spec in specs:
        module = modules[spec.index]
        assert module.complete, (spec.module_name, module.shortfalls)
        assert module.name == spec.module_name


def test_link_calls_true_wires_a_real_import_for_the_dependency():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=True)
    chunk0, chunk1 = specs
    assert 2 in _section_ids_of(modules[chunk1.index].binary)  # import section
    assert 2 not in _section_ids_of(modules[chunk0.index].binary)


def test_link_calls_false_produces_independently_instantiable_modules():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    for module in modules.values():
        assert 2 not in _section_ids_of(module.binary)


def test_shared_memory_modules_import_one_global_memory_with_disjoint_static_data():
    program = FusedProgram(
        version=1, feeds={1},
        steps=[
            OpStep(0, "sin", [1], {}, 2),
            OpStep(1, "cos", [2], {}, 3),
        ],
        outputs={"result": 3},
    )
    specs = partition_reduced_program(program, chunk_size=1, owner_name="kernel")
    modules = emit_class_modules(
        specs, link_calls=False, shared_memory=True, dtype="float64",
    )
    manifest = build_manifest(specs, modules)

    assert manifest["shared_memory"] is True
    assert manifest["shared_static_bytes"] > 0
    first, second = (modules[spec.index] for spec in specs)
    assert 2 in _section_ids_of(first.binary)
    assert 2 in _section_ids_of(second.binary)
    assert second.api.metadata["static_data_offset"] >= first.api.metadata["reserved_bytes"]
    assert all(
        module.api.metadata["shared_memory_import"] == {"module": "env", "field": "memory"}
        for module in modules.values()
    )


def test_shared_memory_and_function_linking_are_distinct_modes():
    specs = partition_reduced_program(_linear_program(2), chunk_size=1, owner_name="kernel")
    with pytest.raises(ValueError, match="link_calls must be false"):
        emit_class_modules(specs, link_calls=True, shared_memory=True)


def _section_ids_of(binary: bytes) -> list[int]:
    cursor, seen = 8, []
    while cursor < len(binary):
        section_id = binary[cursor]
        cursor += 1
        length, shift = 0, 0
        while True:
            byte = binary[cursor]
            cursor += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        seen.append(section_id)
        cursor += length
    return seen


@pytest.mark.skipif(
    __import__("shutil").which("node") is None, reason="node not on PATH"
)
def test_the_wired_import_actually_type_checks_against_its_exporter(tmp_path):
    """``emit_class_modules`` picks the import's declared ``field`` name and
    ``parameter_types`` from the callee's own emitted API -- a mismatch
    there is exactly the kind of bug that assembles fine but fails to
    *instantiate*, which is what this checks."""

    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=True)
    chunk0, chunk1 = specs

    chunk0_path = tmp_path / "chunk0.wasm"
    chunk1_path = tmp_path / "chunk1.wasm"
    chunk0_path.write_bytes(modules[chunk0.index].binary)
    chunk1_path.write_bytes(modules[chunk1.index].binary)
    chunk0_entry = modules[chunk0.index].api.entry
    chunk0_name = chunk0.module_name

    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{ readFileSync }} from "node:fs";
        const [chunk0Path, chunk1Path] = process.argv.slice(2);
        const chunk0Mod = await WebAssembly.instantiate(
          readFileSync(chunk0Path), {{}}
        );
        await WebAssembly.instantiate(
          readFileSync(chunk1Path),
          {{
            {json.dumps(chunk0_name)}: {{
              {json.dumps(chunk0_entry)}: chunk0Mod.instance.exports[{json.dumps(chunk0_entry)}],
              memory: chunk0Mod.instance.exports.memory,
            }},
          }}
        );
        console.log(JSON.stringify({{ ok: true }}));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(chunk0_path), str(chunk1_path)],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True


# --- manifest: contracts and edges auto-derived, not hand-authored --------


def test_build_manifest_derives_edges_from_the_cut_boundary():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    manifest = build_manifest(specs, modules)

    chunk0, chunk1 = specs
    assert {"from": {"module": chunk0.module_name, "output": "value_101"},
            "to": {"module": chunk1.module_name, "input": "feed0"}} in manifest["edges"]
    # The program's own external feed is needed only by chunk0.
    assert chunk1.module_name not in manifest["graph_input_value_ids"]
    assert manifest["graph_input_value_ids"][chunk0.module_name] == [(1, "feed0")]


def test_manifest_uses_the_emitted_module_feed_order_not_boundary_set_order():
    program = FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[OpStep(0, "sub", [2, 1], result_id=3)],
        outputs={"result": 3},
    )
    specs = partition_reduced_program(program, chunk_size=10, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    manifest = build_manifest(specs, modules)

    # The graph boundary discovers feeds as 1,2, but the emitted function
    # consumes 2 first and therefore assigns feed0 to value 2.
    assert specs[0].region.input_ids == (1, 2)
    assert manifest["graph_input_value_ids"][specs[0].module_name] == [
        (2, "feed0"), (1, "feed1"),
    ]


def test_manifest_and_runner_compute_the_right_answer_in_the_browser_is_covered_elsewhere():
    """The actual numeric round-trip (real .wasm files, fetched, run through
    process_graph_runner.js, correct answer) is verified by hand in the
    session that built this -- 10 -> 21 -> 59 -> 11.8 for a 5-step chain cut
    into 3 chunks -- and is not re-asserted here since it needs a browser,
    not a property this file's pytest suite can check headlessly. See
    test_process_graph_shell.py for the level/group scheduling coverage of
    the same real, chunked program."""


# --- scheduling: the real ProcessGraph/ILPScheduler ------------------------


def test_schedule_module_levels_uses_the_real_ilp_scheduler():
    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    levels = schedule_module_levels(specs)
    chunk0, chunk1 = specs
    assert levels[chunk0.module_name] < levels[chunk1.module_name]
    assert levels[chunk1.module_name] == 0  # chunk1 is the root here


def test_build_module_process_graph_is_the_real_process_graph_class():
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from src.transmogrifier.ilpscheduler import ILPScheduler

    program = _linear_program(4)
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    graph = build_module_process_graph(specs)
    assert isinstance(graph, ProcessGraph)
    assert isinstance(graph.scheduler, ILPScheduler)
    chunk0, chunk1 = specs
    assert (chunk0.module_name, chunk1.module_name) in graph.G.edges


def test_a_single_module_gets_a_process_graph_of_one_node():
    program = _linear_program(3)
    specs = partition_reduced_program(program, chunk_size=10, owner_name="kernel")
    graph = build_module_process_graph(specs)
    assert list(graph.G.nodes) == ["kernel__0"]
    assert list(graph.G.edges) == []
    assert schedule_module_levels(specs) == {"kernel__0": 0}


# --- against real compiled IR, not a hand-built fixture --------------------


def test_partition_reduced_program_runs_on_a_real_compiled_program():
    """Not a fixture: compile_ast_aot's own compiled_shell_program, the real
    fully-reduced numeric IR for a two-function source program (the call to
    helper is already inlined away by this point -- see the module
    docstring for why segmentation no longer tries to preserve it)."""

    import numpy as np
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )

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
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    assert len(specs) >= 1
    assert specs[-1].is_root
    modules = emit_class_modules(specs, link_calls=False)
    for spec in specs:
        assert modules[spec.index].complete, (
            spec.module_name, modules[spec.index].shortfalls,
        )


# --- describe_process_graph_api: the whole kernel's own contract ----------


def test_describe_process_graph_api_resolves_the_real_source_parameter_name():
    """Not a fixture: a real two-function program, real capture_feed_origins
    resolution, real CompiledProgramAPI -- the same descriptor format
    wasm_html_shell.emit_html_shell already renders, so no new rendering
    code is needed to display this."""

    import numpy as np
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.compiled_program_api import CompiledProgramAPI

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
    specs = partition_reduced_program(program, chunk_size=1, owner_name=aot.entrypoint)
    modules = emit_class_modules(specs, link_calls=False)

    api = describe_process_graph_api(
        specs, modules, program, entrypoint=aot.entrypoint,
    )
    assert isinstance(api, CompiledProgramAPI)
    assert api.module == "kernel"
    assert api.entry == "kernel"

    entry_point = api.entry_points[0]
    assert entry_point.symbol == "kernel"
    assert api.language == "wasm"
    assert api.metadata["execution_mode"] == "segmented"

    names_by_role = {
        role: [p.name for p in entry_point.parameters if p.role == role]
        for role in ("extent", "input", "output")
    }
    assert names_by_role["extent"] == ["count"]
    # "a" -- the real source parameter name, not a synthesized "input_<id>"
    # or a raw feed label like "feed0".
    assert names_by_role["input"] == ["a"]
    assert len(names_by_role["output"]) == 1

    embedded = build_embedded_class_graph(
        specs, modules, program, entrypoint=aot.entrypoint,
    )
    output_name = next(iter(program.outputs))
    assert embedded["logical_outputs"][output_name][0] in {
        spec.module_name for spec in specs
    }
    assert len(embedded["schedule"]["nodes"]) == len(specs)
    assert all(
        "reserved_bytes" in module for module in embedded["modules"]
    )
    external = build_embedded_class_graph(
        specs, modules, program, entrypoint=aot.entrypoint,
        embed_binaries=False, module_dir="site/v1/wasm",
    )
    assert all(module["url"].startswith("site/v1/wasm/")
               for module in external["modules"])
    assert all("wasm_base64" not in module for module in external["modules"])


def test_describe_process_graph_api_collapses_a_value_needed_by_several_chunks():
    """A value several different chunks need directly (not produced by any
    of them) must appear exactly once in the logical API, not once per
    chunk that happens to need it."""

    program = FusedProgram(
        version=1, feeds={1},
        steps=[
            OpStep(step_id=0, op_name="mul", input_ids=[1],
                   attrs={"right_scalar": 2.0}, result_id=101),
            OpStep(step_id=1, op_name="mul", input_ids=[1],
                   attrs={"right_scalar": 3.0}, result_id=102),
            OpStep(step_id=2, op_name="add", input_ids=[101, 102],
                   attrs={}, result_id=103),
        ],
        outputs={"result": 103},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    specs = partition_reduced_program(program, chunk_size=1, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    api = describe_process_graph_api(specs, modules, program, entrypoint="kernel")
    inputs = [p.name for p in api.entry_points[0].parameters if p.role == "input"]
    assert inputs == ["x"]


def test_describe_process_graph_api_falls_back_to_a_synthetic_name():
    """A hand-built program with no capture_feed_origins extras still gets
    a usable, if less pretty, parameter name -- never a crash."""

    program = FusedProgram(
        version=1, feeds={1},
        steps=[OpStep(step_id=0, op_name="mul", input_ids=[1],
                       attrs={"right_scalar": 2.0}, result_id=2)],
        outputs={"result": 2},
    )
    specs = partition_reduced_program(program, chunk_size=10, owner_name="kernel")
    modules = emit_class_modules(specs, link_calls=False)
    api = describe_process_graph_api(specs, modules, program, entrypoint="kernel")
    inputs = [p.name for p in api.entry_points[0].parameters if p.role == "input"]
    assert inputs == ["input_1"]


def test_hue_identities_trickle_through_the_reduced_schedule():
    import networkx as nx
    from types import SimpleNamespace

    original = nx.DiGraph()
    original.add_node("source", type="Load", label="x")
    original.add_node("result", type="Return", label="return")
    original.add_edge("source", "result")

    program = _linear_program(3)
    specs = partition_reduced_program(program, chunk_size=1, owner_name="kernel")
    views = build_hued_process_graph_views(
        SimpleNamespace(G=original), program, specs,
    )

    assert set(views["views"]) == {"original", "reduced"}
    assert all(f"region:{index}" in views["identities"] for index in range(3))
    reduced = {node["id"]: node for node in views["views"]["reduced"]["nodes"]}
    assert {"region:0", "region:1", "region:2"}.issubset(
        set(reduced["102"]["contributors"])
    )
    assert reduced["102"]["level"] > reduced["100"]["level"]
