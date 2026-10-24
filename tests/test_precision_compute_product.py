"""The precision-core compute lane publishes its SSBO ABI as a product.

The kernels here are built by hand in the wrapper-plus-region shape
``emit_ssa_module_to_glsl_compute`` reads: a counted loop calling one
straight-line region per element. Every run below is driven only from the
published record (bindings, byte offsets, the element-count control, the
group rule), never from the artifact's own ``prepare_execution``, except
where that host path itself is under test.
"""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from src.compiler.ssa_glsl_compute_backend import emit_ssa_module_to_glsl_compute
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


REPO = Path(__file__).resolve().parents[1]
GLSLANG = shutil.which("glslangValidator")
COUNT = 777


def _core(baked: int | None = None) -> IRModule:
    """``y[i] = fma(x[i], c, d)`` with ``c``, ``d`` float64 coefficients."""

    def value(ident, dtype="float64"):
        return SSAValue(ident, dtype, ())

    x, y, n, c, d = value(1), value(2), value(3, "int"), value(4), value(5)
    zero, one, i, step, below = (
        value(10, "int"), value(11, "int"), value(12, "int"), value(13, "int"), value(14, "bool"),
    )
    entry = [
        Instr("Const", [], zero, attributes={"value": 0}),
        Instr("Const", [], one, attributes={"value": 1}),
    ]
    bound = n
    if baked is not None:
        bound = value(15, "int")
        entry.append(Instr("Const", [], bound, attributes={"value": int(baked)}))
    wrapper = Function("fma_core", [x, y, n, c, d], {
        "entry": BasicBlock("entry", [*entry, Instr("Br", [], None)]),
        "loop_header": BasicBlock("loop_header", [
            Instr("Phi", [zero, step], i),
            Instr("Lt", [i, bound], below),
            Instr("CondBr", [below], None),
        ]),
        "loop_body": BasicBlock("loop_body", [
            Instr("Call", [x, y, i, c, d], None, attributes={"callee": "fma_region"}),
            Instr("Br", [], None),
        ]),
        "loop_latch": BasicBlock("loop_latch", [Instr("Add", [i, one], step), Instr("Br", [], None)]),
        "loop_exit": BasicBlock("loop_exit", [Instr("Ret", [], None)]),
    })
    fx, fy, fi, fc, fd = value(21), value(22), value(23, "int"), value(24), value(25)
    at_x, loaded, fused, at_y = value(30), value(31), value(32), value(33)
    region = Function("fma_region", [fx, fy, fi, fc, fd], {"entry": BasicBlock("entry", [
        Instr("GetElementPtr", [fx, fi], at_x),
        Instr("Load", [at_x], loaded),
        Instr("Fma", [loaded, fc, fd], fused, attributes={"precision_section": True}),
        Instr("GetElementPtr", [fy, fi], at_y),
        Instr("Store", [fused, at_y], None),
        Instr("Ret", [], None),
    ])})
    return IRModule({"fma_region": region, "fma_core": wrapper})


def _artifact(baked=None):
    artifact = emit_ssa_module_to_glsl_compute(_core(baked), "fma_core")
    assert artifact.complete, artifact.shortfalls
    return artifact


def _feeds(count=COUNT):
    rng = np.random.default_rng(11)
    return {
        1: rng.standard_normal(count),
        "c": 1.0 / 3.0,
        "d": math.pi,
    }


def _assert_fma(result, feeds):
    """Each element is ``fma(x, c, d)`` rounded once, or ``x*c + d``.

    GLSL only requires every fma() consumed by a precise variable to round
    the same way, so a driver may round twice; the record says so with its
    ``fma-rounding`` shortfall. Anything else is a wrong answer.
    """

    from fractions import Fraction

    x = feeds[1]
    c, d = Fraction(feeds["c"]), Fraction(feeds["d"])
    once = np.array([float(Fraction(value) * c + d) for value in x])
    twice = x * feeds["c"] + feeds["d"]
    assert np.all((result == once) | (result == twice))
    assert (once != twice).any()  # the feeds do tell the two apart


@pytest.mark.parametrize("baked", [None, COUNT])
def test_component_abi_and_record_state_the_emitters_layout(baked):
    artifact = _artifact(baked)
    abi = artifact.component_abi
    record = artifact.deployment().to_mapping()

    assert [(port.name, port.role, port.value_id, port.backend_binding) for port in abi.ports] == [
        ("buffer_0", "feed", 1, 0),
        ("scalar_0", "feed", 4, 2),
        ("scalar_1", "feed", 5, 2),
        ("buffer_1", "output", 2, 1),
    ]
    assert {port.dtype for port in abi.ports} == {"f64"}
    resources = {resource["name"]: resource for resource in record["resources"]}
    assert (resources["Buffer0"]["binding"], resources["Buffer0"]["access"]) == (0, "read")
    assert (resources["Buffer1"]["binding"], resources["Buffer1"]["access"]) == (1, "read_write")
    assert resources["Buffer0"]["realizes"][0]["shape"] is None
    assert resources["Buffer0"]["realizes"][0]["extents_from"] == "fed"
    scalars = resources["ScalarFeed"]
    header = 8 if baked is None else 0
    assert scalars["binding"] == 2
    assert [view["byte_offset"] for view in scalars["realizes"]] == [header, header + 8]
    assert [view["slot"] for view in scalars["realizes"]] == [1, 2]
    assert scalars["members"][-1]["byte_offset"] == header
    assert "GL_ARB_gpu_shader_fp64" in scalars["extensions"]
    launch = record["launch"]
    if baked is None:
        (control,) = launch["controls"]
        assert (control["name"], control["binding"], control["byte_offset"]) == (
            "element_count", 2, 0,
        )
        assert control["derivation"] == [{"rule": "element_count", "value_id": 3}]
        assert launch["group_count"]["rule"] == "cover"
    else:
        assert launch["controls"] == []
        assert launch["group_count"] == {"rule": "exact", "groups": [math.ceil(COUNT / 64), 1, 1]}
        assert launch["element_count"]["value"] == COUNT
    assert [item["code"] for item in record["shortfalls"]] == ["fma-rounding"]


def test_published_precision_product_carries_one_identity(tmp_path):
    artifact = _artifact()
    source_path = artifact.write(tmp_path)
    manifest = json.loads((tmp_path / "fma_core.shader.json").read_text(encoding="utf-8"))
    api = yaml.safe_load((tmp_path / manifest["api"]).read_text(encoding="utf-8"))

    assert set(manifest) == {"schema", "source", "api", "targets"}
    assert source_path.read_text(encoding="utf-8") == artifact.source
    assert api["metadata"]["component_abi"]["component_id"] == "fma_core"
    record = api["metadata"]["shader_deployment"]
    assert record["component_abi"]["record"] == "metadata.component_abi"
    assert record["provenance"]["precision_sections"] is True


@pytest.mark.parametrize("baked", [None, COUNT])
def test_record_offsets_match_glslang_reflection(tmp_path, baked):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not installed")
    artifact = _artifact(baked)
    record = artifact.deployment().to_mapping()
    path = tmp_path / "core.comp"
    path.write_text(artifact.source, encoding="utf-8")
    reflected = subprocess.run(
        [GLSLANG, "-l", "-q", "--reflect-separate-buffers", str(path)],
        capture_output=True, text=True,
    )
    assert reflected.returncode == 0, reflected.stdout
    offsets, bindings = {}, {}
    section = None
    for line in reflected.stdout.splitlines():
        if line.endswith("reflection:"):
            section = line
        elif ":" in line and section:
            name, fields = line.split(":", 1)
            entry = dict(item.strip().split(" ", 1) for item in fields.split(",") if " " in item.strip())
            if section.startswith("Buffer variable"):
                offsets[name.strip()] = int(entry["offset"])
            elif section.startswith("Buffer block"):
                bindings[name.strip()] = int(entry["binding"])
    resources = {resource["name"]: resource for resource in record["resources"]}
    for name, resource in resources.items():
        assert bindings[name] == resource["binding"], name
    scalars = resources["ScalarFeed"]
    assert offsets["ScalarFeed.scalars"] == scalars["realizes"][0]["byte_offset"]
    if baked is None:
        assert offsets["ScalarFeed.element_count"] == record["launch"]["controls"][0]["byte_offset"]
    for target in ([], ["-V", "--target-env", "vulkan1.1", "-o", str(tmp_path / "core.spv")]):
        compiled = subprocess.run([GLSLANG, *target, str(path)], capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stdout


def _record_buffers(record, feeds, count):
    """Every storage buffer's bytes, built only from the published record."""

    values = {1: feeds[1], 4: feeds["c"], 5: feeds["d"], 3: count}
    data = {}
    for resource in record["resources"]:
        if resource["name"] == "ScalarFeed":
            block = bytearray(resource["byte_extent"])
            for view in resource["realizes"]:
                block[view["byte_offset"]: view["byte_offset"] + 8] = np.float64(
                    values[view["value_id"]]
                ).tobytes()
            for control in record["launch"]["controls"]:
                (rule,) = control["derivation"]
                block[control["byte_offset"]: control["byte_offset"] + 4] = np.uint32(
                    values[rule["value_id"]]
                ).tobytes()
            data[resource["binding"]] = bytes(block)
        elif resource["access"] == "read":
            data[resource["binding"]] = np.asarray(
                values[resource["realizes"][0]["value_id"]], dtype=np.float64,
            ).tobytes()
        else:
            data[resource["binding"]] = np.full(count + 64, -7.0).tobytes()
    return data


def _groups(record, count):
    rule = record["launch"]["group_count"]
    if rule["rule"] == "exact":
        return rule["groups"]
    # One extra group exercises the element-count guard.
    return [math.ceil(count / rule["invocations_per_group"]) + 1, 1, 1]


def _output(record, data):
    (resource,) = [item for item in record["resources"] if item["access"] == "read_write"]
    return np.frombuffer(data[resource["binding"]], dtype=np.float64)


@pytest.mark.parametrize("baked", [None, COUNT])
def test_precision_product_runs_on_gl_from_its_record(baked):
    from src.common.tensors.accelerator_backends.gl_context import (
        GLContextUnavailable,
        require_gl_context,
    )

    try:
        require_gl_context(min_version=(4, 3))
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")
    from OpenGL import GL

    artifact = _artifact(baked)
    record = artifact.deployment().to_mapping()
    feeds = _feeds()
    data = _record_buffers(record, feeds, COUNT)
    names = {}
    GL.glUseProgram(artifact._ensure_program())
    try:
        for binding, payload in data.items():
            name = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, name)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(payload), payload, GL.GL_DYNAMIC_COPY)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, name)
            names[binding] = name
        GL.glDispatchCompute(*_groups(record, COUNT))
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        (output,) = [item for item in record["resources"] if item["access"] == "read_write"]
        result = np.empty(len(data[output["binding"]]) // 8, dtype=np.float64)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, names[output["binding"]])
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, 0, result.nbytes, result.ctypes.data_as(ctypes.c_void_p),
        )
    finally:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glUseProgram(0)
        for name in names.values():
            GL.glDeleteBuffers(1, [name])

    _assert_fma(result[:COUNT], feeds)
    assert (result[COUNT:] == -7.0).all()


@pytest.mark.parametrize("baked", [None, COUNT])
def test_prepare_execution_feeds_the_coefficients_the_shader_reads(baked):
    """A baked count has no header; the coefficients start at byte 0."""

    from src.common.tensors.accelerator_backends.gl_context import (
        GLContextUnavailable,
        require_gl_context,
    )

    try:
        require_gl_context(min_version=(4, 3))
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")
    artifact = _artifact(baked)
    feeds = _feeds()
    execution = artifact.prepare_execution({
        1: feeds[1], 2: np.zeros(COUNT), 3: COUNT, 4: feeds["c"], 5: feeds["d"],
    })
    try:
        execution.run()
        result = execution.buffers[2].copy()
    finally:
        execution.close()

    _assert_fma(result, feeds)


@pytest.mark.parametrize("baked", [None, COUNT])
def test_precision_product_runs_on_vulkan_from_its_record(tmp_path, baked):
    compiler = shutil.which("cc")
    if compiler is None or GLSLANG is None:
        pytest.skip("needs cc and glslangValidator")
    probe = tmp_path / "vulkan_compute_probe"
    built = subprocess.run(
        [compiler, "-O1", "-o", str(probe), str(REPO / "tools" / "vulkan_compute_probe.c"), "-lvulkan"],
        capture_output=True, text=True,
    )
    if built.returncode != 0:
        pytest.skip(f"no Vulkan development files: {built.stderr.strip()[:200]}")
    artifact = _artifact(baked)
    record = artifact.deployment().to_mapping()
    assert {"client": "vulkan", "environment": "vulkan1.1", "spirv_version": "1.3"} in (
        record["target"]["clients"]
    )
    source = tmp_path / "core.comp"
    source.write_text(artifact.source, encoding="utf-8")
    module = tmp_path / "core.spv"
    compiled = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.1", str(source), "-o", str(module)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout
    feeds = _feeds()
    data = _record_buffers(record, feeds, COUNT)
    # The probe binds its buffer files at bindings 0..n-1 in order.
    assert sorted(data) == list(range(len(data)))
    paths = []
    for binding in range(len(data)):
        path = tmp_path / f"binding{binding}.bin"
        path.write_bytes(data[binding])
        paths.append(str(path))
    ran = subprocess.run(
        [str(probe), str(module), *map(str, _groups(record, COUNT)), *paths],
        capture_output=True, text=True,
    )
    if ran.returncode in (2, 3):
        pytest.skip(f"no usable Vulkan device: {ran.stderr.strip()[:200]}")
    assert ran.returncode == 0, ran.stderr
    (output,) = [item for item in record["resources"] if item["access"] == "read_write"]
    result = np.fromfile(paths[output["binding"]], dtype=np.float64)

    _assert_fma(result[:COUNT], feeds)
    assert (result[COUNT:] == -7.0).all()
