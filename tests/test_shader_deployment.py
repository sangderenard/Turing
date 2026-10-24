"""Deployable shader products (docs/NODUS_SHADER_DEPLOYMENT_REQUEST.md).

Each product directory must be consumable without joining in-memory objects
or reading declarations out of shader text: exact source and digest, the
logical component ABI, physical resources and launch controls derived from
it, target, provenance and structured shortfalls, under one identity.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.glsl_component_backend import (
    emit_glsl_compute_component,
    emit_glsl_fragment_component,
)
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.shader_deployment import (
    NODUS_REQUEST_SCHEMA,
    SCHEMA,
    SPIRV_PRODUCT_SCHEMA,
    publish_extracted_shader,
    publish_spirv_product,
    spirv_deployment,
)
from src.compiler.shader_extractor import ExtractedShader
from src.compiler.ssa_spirv_backend import (
    DEPLOYABLE_SHADER,
    LINKABLE_SPIRV_LIBRARY,
    emit_deployable_compute_shader,
    emit_function,
)
from src.compiler.ssa_webgpu_backend import emit_module
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


REPO = Path(__file__).resolve().parents[1]


def _program(meta=None):
    return FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[OpStep(0, "add", [1, 2], {}, 3)],
        outputs={"result": 3},
        meta=meta,
    )


def _wgsl(program, count=8):
    function, shortfalls = lower_fused_program_to_ssa(program, function_name="kernel")
    assert not shortfalls
    returned = next(
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
    )
    return emit_module(
        IRModule({"kernel": function}), name="kernel",
        outputs={"kernel": returned}, count=count,
    )


def _sums(directory):
    rows = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        rows[name] = digest
    return rows


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published(source_path):
    directory = source_path.parent
    manifest = json.loads(
        (directory / f"{source_path.stem}.shader.json").read_text(encoding="utf-8")
    )
    api = yaml.safe_load((directory / manifest["api"]).read_text(encoding="utf-8"))
    return directory, manifest, api, api["metadata"]["shader_deployment"]


def _assert_one_identity(directory, manifest, api, record, source_text):
    assert manifest["schema"] == NODUS_REQUEST_SCHEMA
    assert set(manifest) == {"schema", "source", "api", "targets"}
    source_path = directory / manifest["source"]
    assert source_path.read_bytes() == source_text.encode("utf-8")
    assert record["schema"] == SCHEMA
    assert record["source"] == {"file": manifest["source"], "sha256": _sha(source_path)}
    sums = _sums(directory)
    for name in (manifest["source"], manifest["api"], source_path.stem + ".shader.json"):
        assert sums[name] == _sha(directory / name)
    if record["component_abi"] is not None:
        component = api["metadata"]["component_abi"]
        assert record["component_abi"]["component_id"] == component["component_id"]
        assert record["component_id"] == component["component_id"]
        # The deployment view refers to the logical ABI; it never copies it.
        assert "ports" not in record
        slots = sorted(
            view["slot"] for resource in record["resources"]
            for view in resource["realizes"]
        )
        assert slots == [port["slot"] for port in component["ports"]]


def test_glsl_compute_product_publishes_the_arena_launch_contract(tmp_path):
    component = emit_glsl_compute_component(_program(), name="kernel", local_size=32)
    directory, manifest, api, record = _published(component.write(tmp_path))
    _assert_one_identity(directory, manifest, api, record, component.source)

    assert manifest["source"] == "kernel.comp"
    assert (record["language"], record["version"], record["stage"]) == ("glsl-430", "430", "compute")
    arena, slot_table, extent_table = record["resources"]
    assert (arena["kind"], arena["set"], arena["binding"], arena["layout"]) == (
        "storage-buffer", 0, 0, "std430",
    )
    # Every logical port is a view of the one arena at u_slot words, so the
    # logical backend_binding numbers are NOT descriptor bindings.
    for view in arena["realizes"]:
        assert view["byte_offset"] == {"control": "u_slot", "index": view["slot"], "scale": 4}
        assert view["rank"] == 1 and view["shape"] is None
    # The launch controls are read-only storage blocks at the next
    # contiguous bindings, never loose uniforms.
    assert [(item["name"], item["set"], item["binding"], item["access"])
            for item in (slot_table, extent_table)] == [
        ("TuringLaunchSlots", 0, 1, "read"), ("TuringLaunchExtents", 0, 2, "read"),
    ]
    assert "uniform" not in component.source
    launch = record["launch"]
    assert launch["local_size"] == [32, 1, 1]
    assert launch["element_count"]["control"] == "u_count"
    controls = {control["name"]: control for control in launch["controls"]}
    assert {name: (item["storage"], item["binding"], item["byte_offset"])
            for name, item in controls.items()} == {
        "u_slot": ("storage-buffer", 1, 0),
        "u_count": ("storage-buffer", 2, 0),
        "u_extent": ("storage-buffer", 2, 4),
    }
    assert controls["u_slot"]["length"] == len(api["metadata"]["component_abi"]["ports"])
    assert record["shortfalls"] == [] and record["complete"]
    assert [item["client"] for item in record["target"]["clients"]] == ["opengl", "vulkan"]


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="no glslangValidator")
def test_glsl_compute_source_compiles_for_every_client_it_claims(tmp_path):
    component = emit_glsl_compute_component(_program(), name="kernel")
    source = component.write(tmp_path)
    opengl = subprocess.run(["glslangValidator", str(source)], capture_output=True, text=True)
    assert opengl.returncode == 0, opengl.stdout
    vulkan = subprocess.run(
        ["glslangValidator", "-V", "--target-env", "vulkan1.1", str(source),
         "-o", str(tmp_path / "x.spv")],
        capture_output=True, text=True,
    )
    assert vulkan.returncode == 0, vulkan.stdout


def test_glsl_compute_ports_carry_the_dtype_the_source_reads():
    meta = {
        1: Meta(shape=(8,), dtype="int32", device="glsl"),
        2: Meta(shape=(8,), dtype="int32", device="glsl"),
        3: Meta(shape=(8,), dtype="int32", device="glsl"),
    }
    component = emit_glsl_compute_component(_program(meta), name="ints")
    assert "int(arena[u_slot[0]" in component.source
    assert [port.dtype for port in component.component_abi.ports] == ["i32", "i32", "i32"]


def test_glsl_compute_static_shapes_reach_the_port_views(tmp_path):
    component = emit_glsl_compute_component(
        _program(), name="bcast", feed_shapes={1: (2, 3), 2: (3,)},
    )
    _, _, _, record = _published(component.write(tmp_path))
    views = {view["value_id"]: view for view in record["resources"][0]["realizes"]}
    assert (views[1]["shape"], views[1]["byte_extent"]) == ([2, 3], 24)
    assert (views[2]["shape"], views[2]["byte_extent"]) == ([3], 12)
    assert (views[3]["shape"], views[3]["byte_extent"]) == ([2, 3], 24)
    assert record["launch"]["element_count"]["value"] == 6


def test_wgsl_product_bindings_and_dispatch_come_from_the_emitter(tmp_path):
    module = _wgsl(_program(), count=300)
    directory, manifest, api, record = _published(module.write(tmp_path))
    _assert_one_identity(directory, manifest, api, record, module.source)

    ports = api["metadata"]["component_abi"]["ports"]
    assert [(item["set"], item["binding"]) for item in record["resources"]] == [
        (0, port["backend_binding"]) for port in ports
    ]
    for resource in record["resources"]:
        assert f"@group(0) @binding({resource['binding']})" in module.source
        assert resource["byte_extent"] == 300 * 4
    assert record["launch"]["group_count"] == {
        "rule": "exact", "groups": list(module.launch_plan.groups),
    }
    assert record["launch"]["element_count"]["value"] == 300
    assert record["complete"]


def test_webgl_fragment_product_publishes_its_companion_vertex_stage(tmp_path):
    module = emit_glsl_fragment_component(_program(), name="present")
    directory, manifest, api, record = _published(module.write(tmp_path))
    _assert_one_identity(directory, manifest, api, record, module.source)

    (companion,) = record["companions"]
    assert companion["file"] == "present.vert"
    assert (directory / "present.vert").read_text(encoding="utf-8") == module.vertex_source
    assert _sums(directory)["present.vert"] == companion["sha256"]
    kinds = {resource["kind"] for resource in record["resources"]}
    assert kinds == {"sampled-texture", "draw-buffer"}
    assert "graphics-stage" in {item["code"] for item in record["shortfalls"]}


def test_extracted_shader_product_carries_its_provenance(tmp_path):
    source = "#version 300 es\nprecision highp float;\nout vec4 c;\nvoid main() { c = vec4(1.0); }\n"
    shader = ExtractedShader("fragment", source, "glCompileShader", "demo.py", 12, "FRAG")
    product = publish_extracted_shader(tmp_path, shader)
    directory, manifest, api, record = _published(product.source_path)
    _assert_one_identity(directory, manifest, api, record, source)

    assert manifest["source"] == "FRAG.frag"
    assert (record["version"], record["profile"]) == ("300", "es")
    assert record["provenance"] == {
        "emitter": "shader_extractor.extract_shader_compile_calls",
        "filename": "demo.py", "line": 12, "source_symbol": "FRAG",
        "compiler_call": "glCompileShader", "source_sha256": shader.sha256,
    }
    assert record["component_abi"]["record"] == "metadata.component_abi"
    (port,) = api["metadata"]["component_abi"]["ports"]
    assert (port["name"], port["role"], port["dtype"], port["transport"]) == (
        "c", "output", "f32x4", "draw-buffer",
    )
    assert [item["code"] for item in record["shortfalls"]] == ["graphics-stage"]


def _scalar(dtype="float32", op="add"):
    a, b, r = SSAValue(0, dtype, ()), SSAValue(1, dtype, ()), SSAValue(2, dtype, ())
    block = BasicBlock("entry", [Instr(op, [a, b], r), Instr("Ret", [], None)])
    return Function(f"{op}_{dtype}", [a, b], {"entry": block}), r


def test_spirv_library_and_deployable_shader_are_separate_products(tmp_path):
    function, result = _scalar()
    library = emit_function(function, outputs=[result])
    deployable = emit_deployable_compute_shader(function, outputs=[result])

    assert library.product_kind == LINKABLE_SPIRV_LIBRARY
    assert "OpEntryPoint" not in library.source
    assert "LinkageAttributes" in library.source
    assert deployable.product_kind == DEPLOYABLE_SHADER
    assert deployable.complete
    assert 'OpEntryPoint GLCompute %main "main" %gid' in deployable.source
    assert "OpExecutionMode %main LocalSize 64 1 1" in deployable.source
    assert "Linkage" not in deployable.source

    lib = json.loads(publish_spirv_product(tmp_path / "lib", library).manifest_path.read_text())
    dep = json.loads(publish_spirv_product(tmp_path / "dep", deployable).manifest_path.read_text())
    assert lib["schema"] == dep["schema"] == SPIRV_PRODUCT_SCHEMA
    assert (lib["product_kind"], lib["entrypoint"], lib["api"]) == (LINKABLE_SPIRV_LIBRARY, None, None)
    assert (dep["product_kind"], dep["entrypoint"]) == (DEPLOYABLE_SHADER, "main")
    assert dep["target_env"] == "vulkan1.1" and dep["spirv_version"] == "1.3"
    record = yaml.safe_load((tmp_path / "dep" / dep["api"]).read_text())["metadata"]["shader_deployment"]
    assert record["product_kind"] == DEPLOYABLE_SHADER
    assert [(item["set"], item["binding"], item["access"]) for item in record["resources"]] == [
        (0, 0, "read"), (0, 1, "read"), (0, 2, "read_write"),
    ]
    (control,) = record["launch"]["controls"]
    assert (control["name"], control["storage"], control["binding"]) == (
        "element_count", "storage-buffer", 3,
    )


def _spans(rank, *ids, dtype="float32"):
    return [
        SSAValue(i, dtype, (), accounting={"program_abi_rank": rank})
        for i in ids
    ]


def test_deployable_spirv_refuses_what_one_invocation_cannot_carry():
    # The library unrolls static shapes only: a runtime-sized span is
    # refused there, never read as its element 0.
    a, b, r = _spans(1, 0, 1, 2)
    runtime = Function("runtime", [a, b], {"entry": BasicBlock("entry", [
        Instr("add", [a, b], r), Instr("Ret", [], None),
    ])})
    library = emit_function(runtime, outputs=[r])
    assert any("runtime-sized span" in item.reason for item in library.shortfalls)

    # Element mode needs one rank, elementwise operations, span outputs.
    (x,) = _spans(1, 0)
    (y,) = _spans(2, 1)
    (z,) = _spans(2, 2)
    mixed = Function("mixed", [x, y], {"entry": BasicBlock("entry", [
        Instr("add", [x, y], z), Instr("Ret", [], None),
    ])})
    (u,) = _spans(1, 0)
    total = SSAValue(1, "float32", ())
    reduced = Function("reduced", [u], {"entry": BasicBlock("entry", [
        Instr("sum", [u], total, attributes={"axis": 0}), Instr("Ret", [], None),
    ])})
    (w,) = _spans(1, 0)
    first, second = SSAValue(1, "float32", ()), SSAValue(2, "float32", ())
    twice = Function("twice", [w], {"entry": BasicBlock("entry", [
        Instr("sum", [w], first), Instr("add", [first, first], second),
        Instr("Ret", [], None),
    ])})
    for function, outputs, reason in (
        (mixed, [z], "one rank"),
        (reduced, [total], "over an axis"),
        (twice, [second], "is a scalar beside"),
    ):
        refused = emit_deployable_compute_shader(function, outputs=outputs)
        assert not refused.complete and refused.source == ""
        assert any(reason in item.reason for item in refused.shortfalls), (
            refused.shortfalls
        )

    function, _result = _scalar()
    none = emit_deployable_compute_shader(function, outputs=[])
    assert not none.complete
    assert any("at least one result" in item.reason for item in none.shortfalls)


def _elementwise_runtime():
    """``(a + b) * s`` over runtime-sized rank-2 spans, ``s`` a scalar."""

    a, b, total, scaled = _spans(2, 0, 1, 3, 4)
    s = SSAValue(2, "float32", ())
    function = Function("runtime", [a, b, s], {"entry": BasicBlock("entry", [
        Instr("add", [a, b], total), Instr("mul", [total, s], scaled),
        Instr("Ret", [], None),
    ])})
    return function, [scaled]


def test_runtime_sized_elementwise_deploys_one_element_per_invocation():
    function, outputs = _elementwise_runtime()
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    record = spirv_deployment(shader).to_mapping()
    views = {item["name"]: item["realizes"][0] for item in record["resources"]}
    for name in ("feed_0", "feed_1", "output_0"):
        assert views[name]["rank"] == 2 and views[name]["shape"] is None
        assert views[name]["byte_extent"] == {
            "control": "element_count", "index": 0, "scale": 4,
        }
    assert views["feed_0"]["extents_from"] == "fed"
    assert views["output_0"]["extents_from"] == views["feed_0"]["slot"]
    # The scalar broadcasts: one element, read by every invocation.
    assert views["feed_2"]["rank"] == 0 and views["feed_2"]["byte_extent"] == 4
    element_count = record["launch"]["element_count"]
    assert element_count["rank"] == 2
    assert element_count["elements_of"] == views["feed_0"]["slot"]


def _instanced():
    """``(a + b) * s`` and ``a - b`` over (2, 3) instances, ``s`` scalar."""

    a, b = SSAValue(0, "float32", (2, 3)), SSAValue(1, "float32", (2, 3))
    s = SSAValue(2, "float32", ())
    total, scaled, difference = (
        SSAValue(3, "float32", (2, 3)), SSAValue(4, "float32", (2, 3)),
        SSAValue(5, "float32", (2, 3)),
    )
    function = Function("instanced", [a, b, s], {"entry": BasicBlock("entry", [
        Instr("add", [a, b], total), Instr("mul", [total, s], scaled),
        Instr("sub", [a, b], difference), Instr("Ret", [], None),
    ])})
    return function, [scaled, difference]


def test_deployable_spirv_states_instance_shapes_and_every_output():
    function, outputs = _instanced()
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    record = spirv_deployment(shader).to_mapping()
    views = {
        item["name"]: item["realizes"][0] for item in record["resources"]
    }
    assert [item["name"] for item in record["resources"]] == [
        "feed_0", "feed_1", "feed_2", "output_0", "output_1",
    ]
    for name in ("feed_0", "feed_1", "output_0", "output_1"):
        view = views[name]
        assert view["instance_shape"] == [2, 3]
        assert view["rank"] == 3 and view["shape"] is None
        assert view["byte_extent"] == {
            "control": "element_count", "index": 0, "scale": 4 * 6,
        }
    assert views["feed_2"]["instance_shape"] == []
    assert views["feed_2"]["byte_extent"]["scale"] == 4
    assert views["output_1"]["extents_from"] == views["output_0"]["extents_from"]


_HAVE_SPIRV_TOOLS = all(shutil.which(tool) for tool in ("spirv-as", "spirv-val"))


@pytest.mark.skipif(not _HAVE_SPIRV_TOOLS, reason="no spirv-as/spirv-val")
def test_deployable_spirv_validates_for_its_target_environment(tmp_path):
    function, result = _scalar()
    manifest = json.loads(publish_spirv_product(
        tmp_path, emit_deployable_compute_shader(function, outputs=[result]),
    ).manifest_path.read_text())
    assert manifest["complete"], manifest["shortfalls"]
    assert [tool["tool"] for tool in manifest["tools"]] == ["spirv-as", "spirv-val"]
    assert manifest["sha256"]["add_float32.spv"] == _sha(tmp_path / "add_float32.spv")


def _vulkan_probe(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None or not _HAVE_SPIRV_TOOLS:
        pytest.skip("needs cc, spirv-as and spirv-val")
    probe = tmp_path / "vulkan_compute_probe"
    built = subprocess.run(
        [compiler, "-O1", "-o", str(probe), str(REPO / "tools" / "vulkan_compute_probe.c"), "-lvulkan"],
        capture_output=True, text=True,
    )
    if built.returncode != 0:
        pytest.skip(f"no Vulkan development files: {built.stderr.strip()[:200]}")
    return probe


@pytest.mark.parametrize("dtype,op", [("float32", "add"), ("int32", "mul"), ("int32", "sub")])
def test_deployable_spirv_matches_python_when_run_from_its_record(tmp_path, dtype, op):
    probe = _vulkan_probe(tmp_path)
    function, result = _scalar(dtype, op)
    product = publish_spirv_product(
        tmp_path / "product", emit_deployable_compute_shader(function, outputs=[result]),
    )
    manifest = json.loads(product.manifest_path.read_text())
    record = product.deployment

    # Drive the dispatch only from the published record.
    count = 1000
    numpy_type = {"f32": np.float32, "i32": np.int32}
    rng = np.random.default_rng(7)
    buffers = {}
    feeds = []
    for resource in record["resources"]:
        dtype_np = numpy_type[resource["dtype"]]
        if resource["access"] == "read":
            values = (rng.standard_normal(count) * 1000).astype(dtype_np)
            feeds.append(values)
        else:
            # Past-the-end words must survive: the guard is the element count.
            values = np.full(count + 64, 12345, dtype=dtype_np)
        buffers[resource["binding"]] = values
    for control in record["launch"]["controls"]:
        assert control["derivation"] == [{"rule": "element_count"}]
        buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    group = record["launch"]["group_count"]["invocations_per_group"]
    groups = math.ceil(count / group) + 1  # one extra group exercises the guard
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        buffers[binding].tofile(path)
        paths.append(str(path))
    module = product.module_path
    assert module is not None and manifest["module"] == module.name
    ran = subprocess.run(
        [str(probe), str(module), str(groups), "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    output_resource = next(item for item in record["resources"] if item["access"] == "read_write")
    output = np.fromfile(paths[output_resource["binding"]], dtype=numpy_type[output_resource["dtype"]])
    with np.errstate(over="ignore"):
        expected = {"add": np.add, "mul": np.multiply, "sub": np.subtract}[op](*feeds)
    assert output[:count].tobytes() == expected.tobytes()
    assert (output[count:] == 12345).all()


def test_instanced_deployable_spirv_matches_python_on_vulkan(tmp_path):
    probe = _vulkan_probe(tmp_path)
    function, outputs = _instanced()
    product = publish_spirv_product(
        tmp_path / "product",
        emit_deployable_compute_shader(function, outputs=outputs),
    )
    record = product.deployment
    count = 300
    rng = np.random.default_rng(3)
    a = rng.standard_normal((count, 2, 3)).astype(np.float32)
    b = rng.standard_normal((count, 2, 3)).astype(np.float32)
    s = rng.standard_normal(count).astype(np.float32)
    fed = {0: a, 1: b, 2: s}
    buffers = {}
    for resource in record["resources"]:
        view = resource["realizes"][0]
        per_instance = math.prod(view["instance_shape"])
        if resource["access"] == "read":
            buffers[resource["binding"]] = fed[view["value_id"]]
        else:
            buffers[resource["binding"]] = np.full(
                (count + 64) * per_instance, 12345, dtype=np.float32,
            )
    (control,) = record["launch"]["controls"]
    buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    group = record["launch"]["group_count"]["invocations_per_group"]
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(product.module_path), str(math.ceil(count / group) + 1),
         "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    outputs_by_name = {
        item["name"]: np.fromfile(paths[item["binding"]], dtype=np.float32)
        for item in record["resources"] if item["access"] == "read_write"
    }
    expected = {
        "output_0": ((a + b) * s[:, None, None]).reshape(-1),
        "output_1": (a - b).reshape(-1),
    }
    for name, want in expected.items():
        got = outputs_by_name[name]
        np.testing.assert_allclose(got[: count * 6], want, rtol=1e-6)
        assert (got[count * 6:] == 12345).all()


def _reducing():
    """``sum``, ``prod``, ``max``, ``min`` and ``mean`` of one (2, 3) instance."""

    x = SSAValue(0, "float32", (2, 3))
    results = [SSAValue(1 + i, "float32", ()) for i in range(5)]
    ops = ("sum", "prod", "max", "min", "mean")
    function = Function("reducing", [x], {"entry": BasicBlock("entry", [
        *(Instr(op, [x], value) for op, value in zip(ops, results)),
        Instr("Ret", [], None),
    ])})
    return function, results


def _axis_reducing(axis):
    """``sum``/``prod``/``max``/``min``/``mean`` over ``axis`` of one (2, 3, 4)
    instance."""

    shape = (2, 3, 4)
    kept = shape[:axis % 3] + shape[axis % 3 + 1:]
    x = SSAValue(0, "float32", shape)
    results = [SSAValue(1 + i, "float32", kept) for i in range(5)]
    ops = ("sum", "prod", "max", "min", "mean")
    instructions = []
    for op, value in zip(ops, results):
        instr = Instr(op, [x], value)
        instr.attributes["axis"] = axis
        if op in ("max", "min"):
            instr.attributes["max_semantics"] = "reduce_axis"
        instructions.append(instr)
    function = Function("axis_reducing", [x], {"entry": BasicBlock("entry", [
        *instructions, Instr("Ret", [], None),
    ])})
    return function, results, kept


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_axis_reductions_over_instances_match_python_on_vulkan(tmp_path, axis):
    probe = _vulkan_probe(tmp_path)
    function, outputs, kept = _axis_reducing(axis)
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    product = publish_spirv_product(tmp_path / "product", shader)
    record = product.deployment
    count = 50
    lanes = math.prod(kept)
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((count, 2, 3, 4)) * 0.5 + 0.8).astype(np.float32)
    x[7, 1, 2, 3] = np.nan  # one lane of instance 7 reads NaN
    buffers = {}
    for resource in record["resources"]:
        if resource["access"] == "read":
            buffers[resource["binding"]] = x
        else:
            buffers[resource["binding"]] = np.full(
                (count + 8) * lanes, 12345, dtype=np.float32,
            )
    (control,) = record["launch"]["controls"]
    buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    group = record["launch"]["group_count"]["invocations_per_group"]
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(product.module_path), str(math.ceil(count / group) + 1),
         "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    got = {
        item["name"]: np.fromfile(paths[item["binding"]], dtype=np.float32)
        for item in record["resources"] if item["access"] == "read_write"
    }
    reduced_axis = 1 + axis % 3
    for position, op in enumerate(("sum", "prod", "max", "min", "mean")):
        want = getattr(np, op)(x, axis=reduced_axis).reshape(count, lanes)
        have = got[f"output_{position}"][:count * lanes].reshape(count, lanes)
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-6)
        assert (got[f"output_{position}"][count * lanes:] == 12345).all()


def test_reductions_over_instances_match_python_on_vulkan(tmp_path):
    probe = _vulkan_probe(tmp_path)
    function, outputs = _reducing()
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    product = publish_spirv_product(tmp_path / "product", shader)
    record = product.deployment
    count = 200
    rng = np.random.default_rng(5)
    x = rng.standard_normal((count, 2, 3)).astype(np.float32)
    # A NaN enters max/min and is never displaced, as numpy reports it.
    x[7, 1, 2] = np.nan
    buffers = {}
    for resource in record["resources"]:
        view = resource["realizes"][0]
        if resource["access"] == "read":
            buffers[resource["binding"]] = x
        else:
            assert view["instance_shape"] == []
            buffers[resource["binding"]] = np.full(count + 64, 12345, dtype=np.float32)
    (control,) = record["launch"]["controls"]
    buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    group = record["launch"]["group_count"]["invocations_per_group"]
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(product.module_path), str(math.ceil(count / group) + 1),
         "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    got = {
        item["name"]: np.fromfile(paths[item["binding"]], dtype=np.float32)
        for item in record["resources"] if item["access"] == "read_write"
    }
    flat = x.reshape(count, 6)
    expected = {
        "output_0": flat.sum(axis=1), "output_1": flat.prod(axis=1),
        "output_2": flat.max(axis=1), "output_3": flat.min(axis=1),
        "output_4": flat.mean(axis=1),
    }
    for name, want in expected.items():
        np.testing.assert_allclose(got[name][:count], want, rtol=1e-5, atol=1e-6)
        assert np.isnan(got[name][7])
        assert (got[name][count:] == 12345).all()


def _runtime_reduction(op, dtype="float32", start=False):
    """``op`` of one runtime-sized rank-2 span (``sum`` optionally from a
    scalar ``start`` feed)."""

    (x,) = _spans(2, 0, dtype=dtype)
    arguments = [x]
    if start:
        arguments.append(SSAValue(1, dtype, ()))
    result_dtype = "float32" if op == "mean" else dtype
    result = SSAValue(5, result_dtype, ())
    function = Function(f"reduce_{op}", arguments, {"entry": BasicBlock("entry", [
        Instr(op, arguments, result), Instr("Ret", [], None),
    ])})
    return function, [result]


def test_runtime_sized_reduction_deploys_as_one_looping_invocation():
    function, outputs = _runtime_reduction("sum", start=True)
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    assert shader.reduction == "sum" and "OpLoopMerge" in shader.source
    record = spirv_deployment(shader).to_mapping()
    views = {item["name"]: item["realizes"][0] for item in record["resources"]}
    assert views["feed_0"]["rank"] == 2 and views["feed_0"]["shape"] is None
    assert views["feed_0"]["byte_extent"] == {
        "control": "element_count", "index": 0, "scale": 4,
    }
    for name in ("feed_1", "output_0"):
        assert views[name]["rank"] == 0 and views[name]["shape"] == []
        assert views[name]["byte_extent"] == 4
    launch = record["launch"]
    assert launch["group_count"]["rule"] == "fixed"
    assert launch["group_count"]["groups"] == 1
    assert launch["element_count"]["elements_of"] == views["feed_0"]["slot"]
    assert launch["element_count"]["reduction"] == "sum"
    assert launch["element_count"]["minimum"] == 0
    for op in ("max", "min", "mean"):
        function, outputs = _runtime_reduction(op)
        shader = emit_deployable_compute_shader(function, outputs=outputs)
        assert spirv_deployment(shader).to_mapping()["launch"][
            "element_count"]["minimum"] == 1


_RUNTIME_REDUCTIONS = [
    ("sum", "float32", False), ("sum", "float32", True), ("prod", "float32", False),
    ("max", "float32", False), ("min", "float32", False), ("mean", "float32", False),
    ("all", "float32", False), ("any", "float32", False),
    ("sum", "int32", False), ("max", "int32", False),
]


@pytest.mark.parametrize("op,dtype,start", _RUNTIME_REDUCTIONS)
@pytest.mark.parametrize("extents", [(7, 5), (0, 3)])
def test_runtime_sized_reduction_matches_python_on_vulkan(
    tmp_path, op, dtype, start, extents,
):
    probe = _vulkan_probe(tmp_path)
    function, outputs = _runtime_reduction(op, dtype, start)
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    product = publish_spirv_product(tmp_path / "product", shader)
    record = product.deployment
    rng = np.random.default_rng(11)
    if dtype == "int32":
        x = rng.integers(-9, 9, extents).astype(np.int32)
    else:
        x = (rng.standard_normal(extents) * 0.9 + 0.1).astype(np.float32)
        if op in ("all", "any") and x.size:
            x[1, 2] = 0.0
        if op in ("max", "min") and x.size:
            x[3, 1] = np.nan
    sentinel = 12345
    fed = {0: x, 1: np.array([2.5 if dtype == "float32" else 2], dtype=dtype)}
    buffers = {}
    for resource in record["resources"]:
        view = resource["realizes"][0]
        if resource["access"] == "read":
            # A zero-length buffer cannot be bound; pad the empty span.
            fed_value = fed[view["value_id"]].reshape(-1)
            buffers[resource["binding"]] = (
                fed_value if fed_value.size else np.zeros(1, dtype=dtype)
            )
        else:
            result_dtype = "float32" if op == "mean" else dtype
            buffers[resource["binding"]] = np.full(4, sentinel, dtype=result_dtype)
    (control,) = record["launch"]["controls"]
    buffers[control["binding"]] = np.array([x.size], dtype=np.uint32)
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    # Two groups launched where the record asks for one: only invocation 0
    # folds, so the extra invocations change nothing.
    ran = subprocess.run(
        [str(probe), str(product.module_path), "2", "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    output = next(
        item for item in record["resources"] if item["access"] == "read_write"
    )
    got = np.fromfile(paths[output["binding"]], dtype="float32" if op == "mean" else dtype)
    assert (got[1:] == sentinel).all()
    flat = x.reshape(-1)
    if not flat.size and op in ("max", "min", "mean"):
        assert got[0] == sentinel  # no value: nothing written
        return
    if op == "sum":
        want = sum(flat.tolist(), fed[1][0].item() if start else 0)
    elif op == "prod":
        want = np.prod(flat.astype(np.float64))
    elif op in ("max", "min"):
        want = getattr(np, op)(flat)
    elif op == "mean":
        want = flat.astype(np.float64).mean()
    else:
        want = float(getattr(np, op)(flat))
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_glsl_arena_product_runs_on_vulkan_from_its_record(tmp_path, dtype):
    """The same GLSL arena source, compiled for Vulkan, launched only from
    the published record, must return numpy's answer."""

    if shutil.which("glslangValidator") is None:
        pytest.skip("no glslangValidator")
    probe = _vulkan_probe(tmp_path)
    meta = {value_id: Meta(shape=(1,), dtype=dtype, device="glsl") for value_id in (1, 2, 3)}
    component = emit_glsl_compute_component(_program(meta), name="arena", local_size=64)
    _, _, api, record = _published(component.write(tmp_path / "product"))
    spv = tmp_path / "arena.spv"
    compiled = subprocess.run(
        ["glslangValidator", "-V", "--target-env", "vulkan1.1",
         str(tmp_path / "product" / record["source"]["file"]), "-o", str(spv)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout

    count = 777
    numpy_type = {"f32": np.float32, "i32": np.int32}
    rng = np.random.default_rng(11)
    ports = api["metadata"]["component_abi"]["ports"]
    # The host chooses each port's arena word offset; the record says only
    # that u_slot[i] carries it. Leave gaps so a wrong offset shows.
    offsets = {port["slot"]: 5 + port["slot"] * (count + 3) for port in ports}
    arena_words = np.full(5 + len(ports) * (count + 3) + 64, 0xDEADBEEF, dtype=np.uint32)
    feeds = []
    for port in ports:
        if port["role"] == "feed":
            values = (rng.standard_normal(count) * 1000).astype(numpy_type[port["dtype"]])
            arena_words[offsets[port["slot"]]:offsets[port["slot"]] + count] = values.view(np.uint32)
            feeds.append(values)
    rule_values = {"element_count": count}
    buffers = {}
    for resource in record["resources"]:
        if resource["name"] == "Arena":
            buffers[resource["binding"]] = arena_words
            continue
        words = np.zeros(resource["byte_extent"] // 4, dtype=np.uint32)
        for control in record["launch"]["controls"]:
            if control["binding"] != resource["binding"]:
                continue
            for index, rule in enumerate(control["derivation"]):
                value = (
                    offsets[rule["port_slot"]] if rule["rule"] == "arena_word_offset"
                    else rule_values[rule["rule"]]
                )
                words[control["byte_offset"] // 4 + index] = value
        buffers[resource["binding"]] = words
    assert sorted(buffers) == list(range(len(buffers)))
    paths = []
    for binding in sorted(buffers):
        path = tmp_path / f"binding{binding}.bin"
        buffers[binding].tofile(path)
        paths.append(str(path))
    group = record["launch"]["group_count"]["invocations_per_group"]
    ran = subprocess.run(
        [str(probe), str(spv), str(math.ceil(count / group) + 1), "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    result = np.fromfile(paths[0], dtype=np.uint32)
    output = next(port for port in ports if port["role"] == "output")
    start = offsets[output["slot"]]
    with np.errstate(over="ignore"):
        expected = np.add(*feeds)
    assert result[start:start + count].tobytes() == expected.tobytes()
    assert (result[start + count:start + count + 3] == 0xDEADBEEF).all()


def test_runtime_sized_deployable_spirv_matches_python_on_vulkan(tmp_path):
    probe = _vulkan_probe(tmp_path)
    function, outputs = _elementwise_runtime()
    product = publish_spirv_product(
        tmp_path / "product",
        emit_deployable_compute_shader(function, outputs=outputs),
    )
    record = product.deployment
    rng = np.random.default_rng(5)
    # Extents the shader never saw: the host supplies their product.
    a = rng.standard_normal((17, 13)).astype(np.float32)
    b = rng.standard_normal((17, 13)).astype(np.float32)
    s = np.array([0.75], dtype=np.float32)
    count = a.size
    fed = {0: a, 1: b, 2: s}
    buffers = {}
    for resource in record["resources"]:
        view = resource["realizes"][0]
        if resource["access"] == "read":
            buffers[resource["binding"]] = fed[view["value_id"]]
        else:
            buffers[resource["binding"]] = np.full(count + 64, 12345, dtype=np.float32)
    (control,) = record["launch"]["controls"]
    buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    group = record["launch"]["group_count"]["invocations_per_group"]
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(product.module_path), str(math.ceil(count / group) + 1),
         "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    output = next(
        item for item in record["resources"] if item["access"] == "read_write"
    )
    got = np.fromfile(paths[output["binding"]], dtype=np.float32)
    np.testing.assert_allclose(got[:count], ((a + b) * s[0]).reshape(-1), rtol=1e-6)
    assert (got[count:] == 12345).all()


def _run_record(tmp_path, probe, product, fed, count, groups, written=None):
    """Bind ``fed`` (by value id) and a sentinel output, run, return the
    ``written`` (default ``count``) output elements."""

    written = count if written is None else written
    record = product.deployment
    buffers = {}
    for resource in record["resources"]:
        view = resource["realizes"][0]
        if resource["access"] == "read":
            buffers[resource["binding"]] = fed[view["value_id"]].reshape(-1)
        else:
            buffers[resource["binding"]] = np.full(count + 4, 12345, dtype=np.float32)
    for control in record["launch"]["controls"]:
        buffers[control["binding"]] = np.array([count], dtype=np.uint32)
    paths = []
    for binding in range(len(buffers)):
        path = tmp_path / f"binding{binding}.bin"
        np.ascontiguousarray(buffers[binding]).tofile(path)
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(product.module_path), str(groups), "1", "1", *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    output = next(item for item in record["resources"] if item["access"] == "read_write")
    got = np.fromfile(paths[output["binding"]], dtype=np.float32)
    assert (got[written:] == 12345).all()
    return got[:written]


def _same_floats(got, want):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return bool(np.all(
        (np.isnan(got) & np.isnan(want))
        | ((got == want) & (np.signbit(got) == np.signbit(want)))
    ))


_NAN = float("nan")


@pytest.mark.parametrize("semantics", ["scan", "reduce"])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("row", [
    [_NAN, 1.0, 5.0], [1.0, _NAN, 5.0], [1.0, 5.0, _NAN], [0.0, -0.0, 0.0],
])
def test_runtime_sized_max_follows_its_stamped_rule_on_vulkan(
    tmp_path, semantics, op, row,
):
    """``max(x)`` (scan: Python's builtin over the elements) and ``x.max()``
    (reduce: numpy's NaN propagation) fold differently on the device."""

    probe = _vulkan_probe(tmp_path)
    function, outputs = _runtime_reduction(op)
    function.blocks["entry"].instrs[0].attributes["max_semantics"] = semantics
    shader = emit_deployable_compute_shader(function, outputs=outputs)
    assert shader.complete, shader.shortfalls
    product = publish_spirv_product(tmp_path / "product", shader)
    x = np.array(row, dtype=np.float32).reshape(1, 3)
    (got,) = _run_record(tmp_path, probe, product, {0: x}, 3, 1, written=1)
    if semantics == "scan":
        want = (max if op == "max" else min)(x.reshape(-1).tolist())
    else:
        want = getattr(np, op)(x)
    assert _same_floats(got, want), (got, want)


@pytest.mark.parametrize("method", ["emax", "emin", "maximum", "minimum"])
def test_runtime_sized_elementwise_max_follows_its_name_on_vulkan(tmp_path, method):
    """``x.emax(y)`` is the builtin rule per element; ``x.maximum(y)``
    is numpy's."""

    probe = _vulkan_probe(tmp_path)
    a, b, r = _spans(1, 0, 1, 2)
    op = "max" if method in ("emax", "maximum") else "min"
    instr = Instr(op, [a, b], r)
    instr.attributes["max_semantics"] = "emax" if method.startswith("e") else "elementwise"
    function = Function(method, [a, b], {"entry": BasicBlock("entry", [
        instr, Instr("Ret", [], None),
    ])})
    shader = emit_deployable_compute_shader(function, outputs=[r])
    assert shader.complete, shader.shortfalls
    product = publish_spirv_product(tmp_path / "product", shader)
    x = np.array([_NAN, 1.0, 2.0, _NAN, 0.0, -0.0], dtype=np.float32)
    y = np.array([1.0, _NAN, 3.0, _NAN, -0.0, 0.0], dtype=np.float32)
    got = _run_record(tmp_path, probe, product, {0: x, 1: y}, 6, 1)
    if method.startswith("e"):
        pick = max if method == "emax" else min
        # x.emax(y) == max(x, y) per element: y replaces x only when better.
        want = [pick(float(p), float(q)) for p, q in zip(x, y)]
    else:
        want = getattr(np, method)(x, y)
    assert _same_floats(got, want), (got, want)
