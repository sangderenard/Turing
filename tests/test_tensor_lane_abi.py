"""The tensor-lane ABI document (turing.wgsl-tensor-lane-abi.v1) is derived from
the emitted program and agrees with the WGSL text it sits beside.

Nothing here hand-writes a layout: every expectation is read from
WGSLTensorLaneProgram (feeds / published / function.args) or parsed out of the
shader source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from src.compiler import ssa_webgpu_backend as webgpu

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_graph_lens_webgpu", ROOT / "tools/build_graph_lens_webgpu.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tool
_spec.loader.exec_module(tool)

COUNT = 64
BINDING = re.compile(r"@binding\((\d+)\) var<storage, (read|read_write)> (\w+)")


@pytest.fixture(scope="module")
def lens():
    source = tool.LENS.read_text(encoding="utf-8")
    return tool.build_tensor_lane(
        source, tool.FUNCTION, tool.lens_values(COUNT), COUNT, "lens",
        tool.OUTPUT_NAMES, "lens.wgsl")


@pytest.fixture(scope="module")
def probe():
    source = "def f(x, b):\n    return x + b\n"
    values = [("x", (8, 2)), ("b", (2,))]
    return tool.build_tensor_lane(source, "f", values, 8, "lane_probe", ("y",), "probe.wgsl")


def _check_against_program_and_text(program, abi):
    text = program.source
    assert abi["schema"] == webgpu.TENSOR_LANE_ABI_SCHEMA == "turing.wgsl-tensor-lane-abi.v1"
    assert abi["shader_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert abi["count"] == program.count and abi["count_is_runtime"] is False
    assert abi["entry"] == "main" and "fn main(" in text
    assert abi["workgroup_size"] == list(program.module.launch_plan.workgroup_size)
    assert f"@workgroup_size({', '.join(map(str, abi['workgroup_size']))})" in text
    assert f"if (linear_index >= {abi['count']}u)" in text
    assert abi["scalar"] == "f32" and "f64" not in text

    # slots mirror program.feeds / program.published, in order
    assert len(abi["feed_slots"]) == len(program.feeds)
    for slot, feed in zip(abi["feed_slots"], program.feeds, strict=True):
        assert (slot["formal"], slot["component"], slot["uniform"], slot["shape"]) == (
            feed.formal, feed.component, feed.uniform, list(feed.shape))
        assert slot["dtype"] == "f32"
        assert slot["length_elements"] == (1 if feed.uniform else program.count)
    assert [s["slot"] for s in abi["feed_slots"]] == list(range(len(program.feeds)))
    assert len(abi["output_slots"]) == len(program.published)
    for slot, out in zip(abi["output_slots"], program.published, strict=True):
        assert (slot["output"], slot["component"], slot["shape"]) == (
            out.output, out.component, list(out.shape))
        assert slot["length_elements"] == program.count
    assert [s["slot"] for s in abi["output_slots"]] == list(range(len(program.published)))

    # the WGSL reads feed slot s at the declared binding/offset, uniforms at element 0
    arg_ids = [a.id for a in program.function.args]
    assert arg_ids == [f.value for f in program.feeds]
    bindings = abi["bindings"]
    declared = {name: (int(binding), access) for binding, access, name in BINDING.findall(text)}
    assert declared["outputs"] == (bindings["outputs"]["binding"], "read_write")
    if bindings["feed_layout"] == "packed":
        assert declared["feeds"] == (bindings["feeds"]["binding"], "read")
        reads = re.findall(r"let v_\d+: f32 = feeds\[(\d+)u \+ (0u|linear_index)\];", text)
        assert len(reads) == len(abi["feed_slots"])
        for (offset, index), slot in zip(reads, abi["feed_slots"], strict=True):
            assert int(offset) == slot["offset_elements"] == slot["slot"] * abi["count"]
            assert (index == "0u") == slot["uniform"]
            assert slot["binding"] == bindings["feeds"]["binding"]
        assert abi["feed_elements"] == abi["count"] * len(program.feeds)
    else:
        assert bindings["feed_layout"] == "separate" and abi["feed_elements"] is None
        for slot, binding, arg in zip(
                abi["feed_slots"], bindings["feeds"], program.function.args, strict=True):
            assert slot["binding"] == binding["binding"] == slot["slot"]
            assert declared[f"feed_{arg.id}"] == (slot["binding"], "read")
            assert f"feed_{arg.id}[{'0u' if slot['uniform'] else 'linear_index'}]" in text
            assert slot["offset_elements"] == 0
    writes = [int(n) for n in re.findall(r"outputs\[(\d+)u \+ linear_index\]", text)]
    assert sorted(writes) == [s["offset_elements"] for s in abi["output_slots"]]
    assert abi["output_elements"] == abi["count"] * len(program.published)
    assert json.loads(json.dumps(abi)) == abi  # plain JSON


def test_lens_abi_matches_program_and_wgsl(lens):
    program, abi = lens
    _check_against_program_and_text(program, abi)
    assert abi["function"] == "graph_lens" and abi["shader"] == "lens.wgsl"
    assert abi["bindings"]["feed_layout"] == "packed"
    assert [(s["name"], s["component"], s["uniform"]) for s in abi["feed_slots"]] == [
        ("P", 0, False), ("P", 1, False), ("F", 0, True), ("F", 1, True),
        ("H", 0, False), ("k", 0, True), ("R", 0, True), ("d", 0, True)]
    assert [(s["name"], s["component"]) for s in abi["output_slots"]] == [
        ("Q", 0), ("Q", 1), ("C", 0), ("A", 0)]
    shapes = {s["name"]: s["shape"] for s in abi["feed_slots"]}
    assert shapes == {"P": [64, 2], "F": [2], "H": [64], "k": [], "R": [], "d": []}


def test_other_lane_program_abi_uses_separate_feed_bindings(probe):
    program, abi = probe
    _check_against_program_and_text(program, abi)
    assert abi["bindings"]["feed_layout"] == "separate"
    assert abi["count"] == 8
    assert [(s["name"], s["component"], s["uniform"]) for s in abi["feed_slots"]] == [
        ("x", 0, False), ("x", 1, False), ("b", 0, True), ("b", 1, True)]
    assert [(s["name"], s["component"]) for s in abi["output_slots"]] == [("y", 0), ("y", 1)]
    assert abi["bindings"]["outputs"]["binding"] == len(program.feeds)


def test_incomplete_program_has_no_abi():
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    module, outputs, exports = tool.lower(
        "def f(x):\n    return x.sum(dim=0)\n", "f", [("x", (8, 2))], "lane_refusal")
    with scoped_work_contract(PRESETS["fast"]):
        program = webgpu.emit_tensor_lane_module(module, exports[0], outputs[exports[0]], count=8)
    assert not program.complete
    with pytest.raises(webgpu.WGSLEmissionError, match="complete program"):
        webgpu.emit_tensor_lane_abi(program, function="f", formal_names={"x": 0},
                                    output_names=("y",), outputs=outputs[exports[0]])


def test_build_tool_writes_wgsl_abi_and_reference(tmp_path, lens):
    abi = tool.build(tmp_path, COUNT)
    wgsl = (tmp_path / "lens.wgsl").read_text(encoding="utf-8")
    on_disk = json.loads((tmp_path / "lens.wgsl.abi.json").read_text(encoding="utf-8"))
    assert on_disk == abi
    assert hashlib.sha256(wgsl.encode("utf-8")).hexdigest() == on_disk["shader_sha256"]
    # value ids are ephemeral per lowering run (so the shader text and hash differ between
    # runs); the layout by name is not
    key = ("name", "component", "shape", "uniform", "offset_elements")
    for table in ("feed_slots", "output_slots"):
        assert [[s[k] for k in key if k in s] for s in on_disk[table]] ==                [[s[k] for k in key if k in s] for s in lens[1][table]]
    reference = json.loads((tmp_path / "reference.json").read_text(encoding="utf-8"))
    assert reference["count"] == COUNT
    for name, shape in reference["input_shapes"].items():
        total = 1
        for extent in shape:
            total *= extent
        values = reference["inputs"][name]
        assert (len(values) if shape else 1) == total
    for name in ("Q", "C", "A"):
        shape = reference["output_shapes"][name]
        assert shape == next(s["shape"] for s in on_disk["output_slots"] if s["name"] == name)
        assert len(reference["outputs"][name]) == COUNT * (shape[1] if len(shape) > 1 else 1)
    # A = 1 / (1 + d H) everywhere, C(0) = 0
    hops, d = reference["inputs"]["H"], reference["inputs"]["d"]
    for hop, alpha, radius in zip(hops, reference["outputs"]["A"], reference["outputs"]["C"], strict=True):
        assert alpha == pytest.approx(1.0 / (1.0 + d * hop), abs=1e-12)
        assert hop != 0 or radius == 0.0
