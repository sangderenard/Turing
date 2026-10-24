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
    (arena,) = record["resources"]
    assert (arena["kind"], arena["set"], arena["binding"], arena["layout"]) == (
        "storage-buffer", 0, 0, "std430",
    )
    # Every logical port is a view of the one arena at u_slot words, so the
    # logical backend_binding numbers are NOT descriptor bindings.
    for view in arena["realizes"]:
        assert view["byte_offset"] == {"control": "u_slot", "index": view["slot"], "scale": 4}
        assert view["rank"] == 1 and view["shape"] is None
    launch = record["launch"]
    assert launch["local_size"] == [32, 1, 1]
    assert launch["element_count"]["control"] == "u_count"
    controls = {control["name"]: control for control in launch["controls"]}
    assert set(controls) == {"u_count", "u_extent", "u_slot"}
    assert controls["u_slot"]["length"] == len(api["metadata"]["component_abi"]["ports"])
    for name in controls:
        assert f"uniform uint {name}" in component.source
    assert [item["code"] for item in record["shortfalls"]] == ["vulkan-default-uniform-controls"]


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="no glslangValidator")
def test_glsl_compute_shortfall_matches_the_reference_compiler(tmp_path):
    component = emit_glsl_compute_component(_program(), name="kernel")
    source = component.write(tmp_path)
    opengl = subprocess.run(["glslangValidator", str(source)], capture_output=True, text=True)
    assert opengl.returncode == 0, opengl.stdout
    vulkan = subprocess.run(
        ["glslangValidator", "-V", str(source), "-o", str(tmp_path / "x.spv")],
        capture_output=True, text=True,
    )
    assert "non-opaque uniforms outside a block" in vulkan.stdout


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
    assert record["component_abi"] is None
    assert [item["code"] for item in record["shortfalls"]] == ["no-component-abi"]


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


def test_deployable_spirv_refuses_what_one_instance_per_invocation_cannot_carry():
    # A runtime-sized span states its rank, not its extents: one instance
    # per invocation cannot be laid out, and the scalar reading of it would
    # be element 0.
    a = SSAValue(0, "float32", (), accounting={"program_abi_rank": 1})
    b = SSAValue(1, "float32", (), accounting={"program_abi_rank": 1})
    r = SSAValue(2, "float32", (), accounting={"program_abi_rank": 1})
    runtime = Function("runtime", [a, b], {"entry": BasicBlock("entry", [
        Instr("add", [a, b], r), Instr("Ret", [], None),
    ])})
    refused = emit_deployable_compute_shader(runtime, outputs=[r])
    assert not refused.complete and refused.source == ""
    assert any("runtime-sized span" in item.reason for item in refused.shortfalls)
    library = emit_function(runtime, outputs=[r])
    assert any("runtime-sized span" in item.reason for item in library.shortfalls)

    function, _result = _scalar()
    none = emit_deployable_compute_shader(function, outputs=[])
    assert not none.complete
    assert any("at least one result" in item.reason for item in none.shortfalls)


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
