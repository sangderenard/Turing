"""Tensor ``clamp`` / ``clamp_min`` / ``clamp_max`` lower and run natively.

Program: ``x.clamp(min=0.0, max=1.0)`` refused "has no settled dtype"
(``_tensor_descriptor`` knew ``clamp`` as no operation), and ``x.clamp_max(1.0)``
lowered but reached the LLVM emitter as an operation with no repository
emission.  The repository's C backend defines each as ``maximum`` /
``minimum`` with the bound, so the plan expands them to the ``Max`` / ``Min``
folds the catalogued binary kernels already serve.  Each source is compiled
through ``lower_ast_source_to_ssa`` with the repository extraction contract and
run natively against the AbstractTensor reference.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

ROOT = Path(__file__).resolve().parents[1]
TENSOR = "src.common.tensors.abstraction.AbstractTensor"


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


def _lower(source, function, values, name="clamp_probe"):
    return lower_ast_source_to_ssa(
        source, function, name=name,
        extraction_contract=_contract(function, values),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False,
    )


EXPRESSIONS = [
    "x.clamp(min=-0.5, max=0.75)",
    "x.clamp(min=-0.5)",
    "x.clamp(max=0.75)",
    "x.clamp(-0.5, 0.75)",
    "x.clamp_min(-0.5)",
    "x.clamp_max(0.75)",
]
VALUES = np.array([-2.0, -0.5, -0.25, 0.0, 0.5, 0.75, 1.0, 3.0])


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_clamp_lowers_with_a_settled_dtype_and_emits_without_shortfall(expression):
    module, _, exports = _lower(
        f"def f(x):\n    return {expression}\n", "f", [("x", (8,))],
    )
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_clamp_native_result_matches_the_tensor_reference(expression, tmp_path):
    source = f"def f(x):\n    return {expression}\n"
    module, outputs, exports = _lower(source, "f", [("x", (8,))])
    name = exports[0]
    function = module.functions[name]
    native = compile_artifact(
        emit_ssa_function_to_llvm(module, name), directory=tmp_path / "native",
    )
    feeds = {dict(function.metadata["parameter_names"])["x"]: VALUES.copy()}
    execution = prepare_artifact_execution(native, feeds).run()
    actual = np.asarray(execution.buffers[outputs[name][0].id]).reshape(-1)
    scope = {"AbstractTensor": AbstractTensor}
    exec(source, scope)
    expected = scope["f"](AbstractTensor.tensor(VALUES.copy())).numpy().reshape(-1)
    assert np.allclose(actual, expected, rtol=0, atol=1e-6), (actual, expected)


LENS = ROOT / "src/rendering/graph_shaders/lens.py"


@pytest.mark.parametrize("spelling", [".clamp_max(1.0)", ".clamp(min=0.0, max=1.0)"])
def test_graph_lens_lowers_and_runs_natively_like_the_reference(spelling, tmp_path):
    authored = LENS.read_text(encoding="utf-8")
    assert ".clamp_max(1.0)" in authored
    source = authored.replace(".clamp_max(1.0)", spelling)
    values = [("P", (64, 2)), ("F", (2,)), ("H", (64,)),
              ("k", ()), ("R", ()), ("d", ())]
    module, outputs, exports = _lower(source, "graph_lens", values, name="lens")
    name = exports[0]
    function = module.functions[name]
    artifact = emit_ssa_function_to_llvm(module, name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / "native")
    rng = np.random.default_rng(7)
    P = rng.uniform(-3.0, 3.0, (64, 2))
    F = np.array([0.25, -0.5])
    H = rng.integers(0, 6, 64).astype(np.float64)
    k, R, d = 1.5, 2.0, 0.3
    ids = dict(function.metadata["parameter_names"])
    feeds = {ids["P"]: P.copy(), ids["F"]: F.copy(), ids["H"]: H.copy(),
             ids["k"]: np.array(k), ids["R"]: np.array(R), ids["d"]: np.array(d)}
    execution = prepare_artifact_execution(native, feeds).run()
    scope = {"AbstractTensor": AbstractTensor}
    exec(source, scope)
    expected = scope["graph_lens"](
        AbstractTensor.tensor(P.copy()), AbstractTensor.tensor(F.copy()),
        AbstractTensor.tensor(H.copy()), k, R, d,
    )
    for value, want in zip(outputs[name], expected, strict=True):
        got = np.asarray(execution.buffers[value.id]).reshape(-1)
        assert np.allclose(got, want.numpy().reshape(-1), rtol=0, atol=1e-6)
