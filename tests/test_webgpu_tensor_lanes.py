"""A lowered AbstractTensor program (the graph lens) reaches WGSL as lanes.

``lower_ast_source_to_ssa`` plans a tensor program into a root plus
``planned_region_N`` functions of repository kernel Calls.  The WebGPU backend's
tensor-lane lowering inlines the regions, takes the leading extent as the lane
space (trailing small extents live inside a lane, a shared ``[2]`` operand is
read by every lane), resolves ``broadcast_double`` and the trailing-axis
``reduce_dim_double`` inside a lane, and emits one invocation per element.

No WGSL runtime or validator exists in this repository (the WGSL tests assert on
the emitted text), so the shader text is checked structurally here, and the
per-lane scalar program it is emitted from is run NATIVELY through the existing
LLVM path, once per lane, against the AbstractTensor reference.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import ssa_webgpu_backend as webgpu
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.compiler.work_contract import PRESETS, scoped_work_contract
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule

ROOT = Path(__file__).resolve().parents[1]
TENSOR = "src.common.tensors.abstraction.AbstractTensor"
LENS = ROOT / "src/rendering/graph_shaders/lens.py"


def _contract(function, values):
    rows = []
    for name, shape in values:
        if shape:
            rows.append({
                "function": function, "parameter": name, "storage": "span",
                "dtype": "float64", "rank": len(shape), "shape": list(shape),
                "python_type": TENSOR,
            })
        else:
            rows.append({
                "function": function, "parameter": name, "storage": "scalar",
                "dtype": "float64", "rank": 0, "python_type": "builtins.float",
            })
    return ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": rows})


def _lower(source, function, values, name):
    return lower_ast_source_to_ssa(
        source, function, name=name,
        extraction_contract=_contract(function, values),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False,
    )


def _emit(module, name, outputs, count):
    with scoped_work_contract(PRESETS["fast"]):
        return webgpu.emit_tensor_lane_module(module, name, outputs, count=count)


def _run_lanes_natively(program, module, name, arrays, tmp_path):
    """The per-lane scalar program, run once per lane through the LLVM path.

    ``arrays`` maps source parameter name -> numpy array.  Returns
    ``{(output value id, component): per-lane array}``."""

    ids = dict(module.functions[name].metadata["parameter_names"])
    by_id = {int(ids[key]): np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    function = program.function
    scalar = Function(
        function.name, function.args,
        {"entry": BasicBlock("entry", [
            *function.blocks["entry"].instrs, Instr("Ret", list(program.outputs), None),
        ])},
        metadata=dict(function.metadata),
    )
    artifact = emit_ssa_function_to_llvm(IRModule({name: scalar}), name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / "native")
    gathered = {(out.output, out.component): [] for out in program.published}
    for lane in range(program.count):
        feeds = {}
        for feed in program.feeds:
            source = by_id[feed.formal]
            if feed.uniform:
                feeds[feed.value] = np.array(source.reshape(-1)[feed.component])
            else:
                feeds[feed.value] = np.array(
                    source.reshape(program.count, -1)[lane, feed.component])
        execution = prepare_artifact_execution(native, feeds).run()
        for out in program.published:
            gathered[(out.output, out.component)].append(
                float(np.asarray(execution.buffers[out.value]).reshape(-1)[0]))
    return {key: np.array(values) for key, values in gathered.items()}


def _reference(source, function, arguments):
    scope = {"AbstractTensor": AbstractTensor}
    exec(source, scope)
    wrapped = [
        AbstractTensor.tensor(np.array(value)) if isinstance(value, np.ndarray) else value
        for value in arguments
    ]
    result = scope[function](*wrapped)
    return result if isinstance(result, tuple) else (result,)


def _assert_lanes_match(program, native, expected, count):
    """Published components, regrouped per source output, equal the reference."""
    order = []
    for out in program.published:
        if out.output not in order:
            order.append(out.output)
    assert len(order) == len(expected)
    for output, want in zip(order, expected, strict=True):
        components = sorted(c for (o, c) in native if o == output)
        got = np.stack([native[(output, c)] for c in components], axis=1)
        want = want.numpy().reshape(program.count, -1)
        assert got.shape == want.shape, (got.shape, want.shape)
        assert np.allclose(got, want, rtol=0, atol=1e-6), (output, got, want)


def _assert_wgsl_is_well_formed(program):
    source = program.source
    assert "@compute @workgroup_size(" in source and "fn main(" in source
    assert source.count("{") == source.count("}")
    declared = set()
    for line in source.splitlines():
        line = line.strip()
        match = re.match(r"let (v_\d+): f32 = (.*);$", line)
        if match:
            for used in re.findall(r"v_\d+", match.group(2)):
                assert used in declared, (line, used)
            declared.add(match.group(1))
    assert "f64" not in source
    slots = re.findall(r"outputs\[(\d+)u \+ linear_index\]", source)
    assert len(slots) == len(program.published) == len(set(slots))


PROBES = {
    "broadcast": ("def f(x, b):\n    return x + b\n", [("x", (8, 2)), ("b", (2,))]),
    "reduce": ("def f(x):\n    return x.sum(dim=1, keepdim=True)\n", [("x", (8, 2))]),
    "norm": (
        "def f(x, b):\n    d = x - b\n    return (d * d).sum(dim=1, keepdim=True).sqrt()\n",
        [("x", (8, 2)), ("b", (2,))],
    ),
}


@pytest.mark.parametrize("probe", sorted(PROBES))
def test_small_tensor_programs_reach_wgsl_and_match_the_reference(probe, tmp_path):
    source, values = PROBES[probe]
    module, outputs, exports = _lower(source, "f", values, "lane_probe")
    name = exports[0]
    program = _emit(module, name, outputs[name], 8)
    assert program.complete, program.shortfalls
    _assert_wgsl_is_well_formed(program)
    rng = np.random.default_rng(3)
    arrays = {key: rng.uniform(-2.0, 2.0, shape) for key, shape in values}
    native = _run_lanes_natively(program, module, name, arrays, tmp_path)
    expected = _reference(source, "f", [arrays[key] for key, _ in values])
    _assert_lanes_match(program, native, expected, 8)


def test_a_reduction_across_the_lane_axis_is_refused_by_name():
    module, outputs, exports = _lower(
        "def f(x):\n    return x.sum(dim=0)\n", "f", [("x", (8, 2))], "lane_refusal")
    name = exports[0]
    program = _emit(module, name, outputs[name], 8)
    assert not program.complete
    messages = [item.format() for item in program.shortfalls]
    assert any("reduction across the lane axis" in item for item in messages), messages


def test_graph_lens_reaches_wgsl_and_its_lanes_match_the_reference(tmp_path):
    authored = LENS.read_text(encoding="utf-8")
    count = 64
    values = [("P", (count, 2)), ("F", (2,)), ("H", (count,)),
              ("k", ()), ("R", ()), ("d", ())]
    module, outputs, exports = _lower(authored, "graph_lens", values, "lens")
    name = exports[0]
    program = _emit(module, name, outputs[name], count)
    assert program.complete, program.shortfalls
    _assert_wgsl_is_well_formed(program)

    # entry signature: one invocation per node; P is two component formals, F two
    # uniforms, H one lane formal, k/R/d uniforms; Q is two slots, C and A one each
    assert "@workgroup_size(" in program.source and "if (linear_index >= 64u)" in program.source
    ids = dict(module.functions[name].metadata["parameter_names"])
    layout = [(feed.formal, feed.component, feed.uniform) for feed in program.feeds]
    assert sorted(layout) == sorted([
        (ids["P"], 0, False), (ids["P"], 1, False),
        (ids["F"], 0, True), (ids["F"], 1, True),
        (ids["H"], 0, False),
        (ids["k"], 0, True), (ids["R"], 0, True), (ids["d"], 0, True),
    ])
    assert [(out.output, out.component) for out in program.published] == [
        (outputs[name][0].id, 0), (outputs[name][0].id, 1),
        (outputs[name][1].id, 0), (outputs[name][2].id, 0),
    ]
    assert "inline" not in program.source and "planned_region" not in program.source

    rng = np.random.default_rng(7)
    arrays = {
        "P": rng.uniform(-3.0, 3.0, (count, 2)), "F": np.array([0.25, -0.5]),
        "H": rng.integers(0, 6, count).astype(np.float64),
        "k": np.array(1.5), "R": np.array(2.0), "d": np.array(0.3),
    }
    native = _run_lanes_natively(program, module, name, arrays, tmp_path)
    expected = _reference(authored, "graph_lens", [
        arrays["P"], arrays["F"], arrays["H"], 1.5, 2.0, 0.3])
    _assert_lanes_match(program, native, expected, count)
