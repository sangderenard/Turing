"""Build the pure-SymPy fluid stencil through four direct SSA backend lanes."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import sympy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compiler.ssa_c_backend import emit_ssa_function_to_c
from src.compiler.ssa_fortran_backend import (
    compile_module as compile_fortran_module,
    emit_module as emit_ssa_module_to_fortran,
    fortran_compiler,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact as compile_llvm_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
from src.compiler.symbolic_fluid_model import compile_symbolic_fluid_step
from src.compiler.fortran_toolchain import stage_fortran_runtime_dependencies
from src.compiler.work_contract import set_active_contract


def _uniform_inputs(compiled) -> dict[str, float]:
    values = {name: 0.0 for name in compiled.function.metadata["argument_names"]}
    for name in values:
        if name.startswith("height_"):
            values[name] = 1.0
    values.update({
        "dt": 0.01,
        "dx": 0.25,
        "gravity": 1.0,
        "minimum_height": 1.0e-4,
    })
    return values


def _run_llvm(compiled, artifact, feeds):
    execution = prepare_artifact_execution(
        artifact,
        {compiled.input_ids[name]: value for name, value in feeds.items()},
    ).run()
    return tuple(
        float(execution.buffers[compiled.output_ids[name]])
        for name in compiled.function.metadata["output_names"]
    )


def _run_wasm(artifact, wasm_path: Path, feeds):
    node = shutil.which("node")
    if node is None:
        return None, "Node.js is not installed"
    ordered = [feeds[name] for name in artifact.input_names]
    script = """
const fs = require('fs');
(async () => {
  const result = await WebAssembly.instantiate(fs.readFileSync(process.argv[1]), {});
  const memory = new Float64Array(result.instance.exports.memory.buffer);
  const inputs = JSON.parse(process.argv[2]);
  memory.set(inputs, 0);
  result.instance.exports.symbolic_fluid_step(0);
  console.log(JSON.stringify(Array.from(memory.slice(inputs.length, inputs.length + 11))));
})().catch(error => { console.error(error); process.exit(1); });
"""
    completed = subprocess.run(
        [node, "-e", script, str(wasm_path), json.dumps(ordered)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        return None, completed.stderr.strip()
    return tuple(map(float, json.loads(completed.stdout))), None


def _run_fortran(module, library_path: Path, feeds):
    entry = module.api.entry_point("symbolic_fluid_step")
    function = getattr(ctypes.CDLL(str(library_path)), entry.symbol)
    arguments = []
    keepalive = []
    outputs = []
    for parameter in entry.parameters:
        if parameter.dtype not in {"float64", "double", "f64"}:
            raise TypeError(f"unexpected Fortran ABI dtype {parameter.dtype!r}")
        if parameter.role == "input":
            value = ctypes.c_double(float(feeds[parameter.source_name]))
        elif parameter.role == "output":
            value = ctypes.c_double()
            outputs.append(value)
        else:
            raise TypeError(f"unexpected Fortran ABI role {parameter.role!r}")
        keepalive.append(value)
        arguments.append(value if parameter.passing == "value" else ctypes.byref(value))
    function.restype = None
    function(*arguments)
    return tuple(float(value.value) for value in outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("build/symbolic-fluid-backends"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # The four-lane agreement proof runs under ``deploy``: the shared
    # identity pass reduces every constant-exponent Pow once, before
    # emission, so all lanes receive identical SSA and none needs a private
    # spelling (scalar WASM has no pow instruction to fall back on).
    set_active_contract("deploy")
    compiled = compile_symbolic_fluid_step()
    function_name = compiled.function.name
    output_names = tuple(compiled.function.metadata["output_names"])
    feeds = _uniform_inputs(compiled)
    returned = compiled.function.blocks["entry"].instrs[-1].args

    (args.output / "equations.txt").write_text(
        "\n".join(sympy.sstr(equation) for equation in compiled.equations) + "\n",
        encoding="utf-8",
    )
    (args.output / "equations.srepr.txt").write_text(
        "\n".join(sympy.srepr(equation) for equation in compiled.equations) + "\n",
        encoding="utf-8",
    )
    (args.output / "repository_ssa.txt").write_text(
        "\n".join(map(str, compiled.instructions)) + "\n",
        encoding="utf-8",
    )

    c_artifact = emit_ssa_function_to_c(compiled.module, function_name)
    c_artifact.compile(args.output / "c")
    c_values = c_artifact.run(feeds)

    llvm_artifact = emit_ssa_function_to_llvm(compiled.module, function_name)
    compile_llvm_artifact(llvm_artifact, directory=args.output / "llvm")
    llvm_values = _run_llvm(compiled, llvm_artifact, feeds)

    fortran_artifact = emit_ssa_module_to_fortran(
        compiled.module,
        name="symbolic_fluid_fortran",
        outputs={function_name: returned},
    )
    (args.output / "fortran").mkdir(exist_ok=True)
    (args.output / "fortran" / "symbolic_fluid_step.f90").write_text(
        fortran_artifact.source, encoding="utf-8",
    )
    fortran_artifact.api.write(
        args.output / "fortran" / "symbolic_fluid_step.api.yaml"
    )
    fortran_values = None
    fortran_error = None
    compiler = fortran_compiler()
    if compiler is None:
        fortran_error = "no Fortran compiler was found"
    else:
        try:
            fortran_library = compile_fortran_module(
                fortran_artifact,
                directory=args.output / "fortran",
                standalone=True,
            )
        except Exception as static_error:
            try:
                fortran_library = compile_fortran_module(
                    fortran_artifact,
                    directory=args.output / "fortran",
                    standalone=False,
                )
                stage_fortran_runtime_dependencies(
                    compiler, fortran_library.parent,
                )
            except Exception as dynamic_error:
                fortran_error = (
                    f"static link failed: {static_error}; "
                    f"dynamic fallback failed: {dynamic_error}"
                )
            else:
                fortran_values = _run_fortran(
                    fortran_artifact, fortran_library, feeds,
                )
        else:
            fortran_values = _run_fortran(
                fortran_artifact, fortran_library, feeds,
            )

    wasm_artifact = emit_ssa_function_to_wasm(compiled.module, function_name)
    _wat_path, wasm_path = wasm_artifact.write(args.output / "wasm")
    wasm_values, wasm_error = _run_wasm(wasm_artifact, wasm_path, feeds)

    surface_contracts = {
        "c": c_artifact.output_surfaces,
        "llvm": llvm_artifact.output_surfaces,
        "fortran": fortran_artifact.api.metadata[
            "semantic_output_surfaces"
        ],
        "wasm": wasm_artifact.output_surfaces,
    }
    (args.output / "semantic_outputs.json").write_text(
        json.dumps({
            "publications": list(c_artifact.output_publications),
            "surface_contracts": surface_contracts,
        }, indent=2),
        encoding="utf-8",
    )

    results = {
        "schema": "turing.symbolic-fluid-backend-proof.v1",
        "source": "pure-sympy-equations",
        "canonical_ir": "repository-ssa",
        "fused_program_used": False,
        "outputs": output_names,
        "publications": list(c_artifact.output_publications),
        "surface_contracts": surface_contracts,
        "uniform_input": feeds,
        "lanes": {
            "c": {"emitted": c_artifact.complete, "executed": True, "values": c_values},
            "llvm": {"emitted": llvm_artifact.complete, "executed": True, "values": llvm_values},
            "fortran": {
                "emitted": fortran_artifact.complete,
                "compiler": compiler,
                "executed": fortran_values is not None,
                "values": fortran_values,
                "reason": fortran_error,
            },
            "wasm": {
                "emitted": wasm_artifact.complete,
                "executed": wasm_values is not None,
                "values": wasm_values,
                "reason": wasm_error,
            },
        },
    }
    executable_rows = [np.asarray(c_values), np.asarray(llvm_values)]
    if fortran_values is not None:
        executable_rows.append(np.asarray(fortran_values))
    if wasm_values is not None:
        executable_rows.append(np.asarray(wasm_values))
    results["executable_max_abs_difference"] = float(max(
        np.max(np.abs(row - executable_rows[0])) for row in executable_rows
    ))
    (args.output / "proof.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
