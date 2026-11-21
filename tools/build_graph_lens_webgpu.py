"""Publish the graph lens as a page-loadable WebGPU tensor-lane product.

Lowers src/rendering/graph_shaders/lens.py through the real pipeline (the
program extraction contract with an explicit program ABI), emits the WGSL
tensor-lane program, and writes beside it the lane ABI document derived from
that program, plus a reference.json: the inputs of a fixed small graph and the
outputs of the numpy-backed AbstractTensor function itself.

    python tools/build_graph_lens_webgpu.py OUT_DIR [--count 64]

writes OUT_DIR/lens.wgsl, OUT_DIR/lens.wgsl.abi.json, OUT_DIR/reference.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import ssa_webgpu_backend as webgpu
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.work_contract import PRESETS, scoped_work_contract

TENSOR = "src.common.tensors.abstraction.AbstractTensor"
LENS = ROOT / "src/rendering/graph_shaders/lens.py"
FUNCTION = "graph_lens"
OUTPUT_NAMES = ("Q", "C", "A")


def lens_values(count: int) -> list:
    return [("P", (count, 2)), ("F", (2,)), ("H", (count,)),
            ("k", ()), ("R", ()), ("d", ())]


def program_contract(function: str, values) -> ExtractionContract:
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


def lower(source: str, function: str, values, name: str):
    return lower_ast_source_to_ssa(
        source, function, name=name,
        extraction_contract=program_contract(function, values),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False,
    )


def build_tensor_lane(source: str, function: str, values, count: int, name: str,
                      output_names, shader: str):
    """(program, abi) for ``function`` of ``source`` over ``count`` lanes."""
    module, outputs, exports = lower(source, function, values, name)
    entry = exports[0]
    with scoped_work_contract(PRESETS["fast"]):
        program = webgpu.emit_tensor_lane_module(module, entry, outputs[entry], count=count)
    if not program.complete:
        raise RuntimeError("tensor-lane shortfalls: " + "; ".join(
            item.format() for item in program.shortfalls))
    abi = webgpu.emit_tensor_lane_abi(
        program, function=function,
        formal_names=dict(module.functions[entry].metadata["parameter_names"]),
        output_names=output_names, outputs=outputs[entry], shader=shader)
    return program, abi


def reference_case(source: str, count: int) -> dict:
    """Fixed deterministic inputs and the AbstractTensor function's outputs."""
    rng = np.random.default_rng(7)
    inputs = {
        "P": rng.uniform(-3.0, 3.0, (count, 2)), "F": np.array([0.25, -0.5]),
        "H": rng.integers(0, 6, count).astype(np.float64),
        "k": 1.5, "R": 2.0, "d": 0.3,
    }
    scope = {"AbstractTensor": AbstractTensor}
    exec(source, scope)
    wrapped = [AbstractTensor.tensor(np.array(v)) if isinstance(v, np.ndarray) else v
               for v in (inputs[n] for n in ("P", "F", "H", "k", "R", "d"))]
    result = scope[FUNCTION](*wrapped)
    flat = {name: np.asarray(value.numpy(), dtype=np.float64).reshape(-1).tolist()
            for name, value in zip(OUTPUT_NAMES, result, strict=True)}
    shapes = {name: list(np.asarray(value.numpy()).shape)
              for name, value in zip(OUTPUT_NAMES, result, strict=True)}
    return {
        "count": count,
        "note": "inputs are row-major flat; outputs are AbstractTensor (float64) results, row-major flat",
        "inputs": {k: (v.reshape(-1).tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in inputs.items()},
        "input_shapes": {k: list(np.shape(v)) for k, v in inputs.items()},
        "outputs": flat, "output_shapes": shapes,
    }


def build(directory: Path, count: int = 64) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = LENS.read_text(encoding="utf-8")
    program, abi = build_tensor_lane(
        source, FUNCTION, lens_values(count), count, "lens", OUTPUT_NAMES, "lens.wgsl")
    (directory / "lens.wgsl").write_text(program.source, encoding="utf-8", newline="\n")
    (directory / "lens.wgsl.abi.json").write_text(
        json.dumps(abi, indent=1) + "\n", encoding="utf-8", newline="\n")
    (directory / "reference.json").write_text(
        json.dumps(reference_case(source, count), indent=1) + "\n", encoding="utf-8", newline="\n")
    return abi


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args(argv)
    abi = build(Path(args.directory), args.count)
    print(f"wrote {args.directory}: lens.wgsl ({abi['shader_sha256'][:12]}), lens.wgsl.abi.json, reference.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
