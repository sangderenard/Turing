"""Scalar float ``math`` transcendentals follow the numeric contract.

* ``transcendentals`` is ``backend_intrinsic`` under ``compute`` and
  ``signal_cores`` under ``scientific``/``stochastic``; any other value
  refuses by name.
* ``signal_cores``: ``ir_identities.lower_transcendentals_to_signal_cores``
  rewrites exp, log, sin, cos, tan, atan, atan2, tanh and pow into their
  ``WideExpansionEmitter`` cores, binds the op's own result id to the
  collapsed double, and guards Python's exceptions with ``CondBr`` to
  ``turing_validation_error`` (``FLOAT_MATH_DOMAIN``, ``FLOAT_MATH_RANGE``,
  ``pow_expansion``'s rows).  A scalar op with no core (asin, acos, sinh,
  cosh) is refused by name.
* ``backend_intrinsic``: nothing is rewritten; each backend spells its own
  math library (LLVM's atan is a libm call, no longer refused).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _program(body: str) -> str:
    return (
        "import math\n\n"
        "def root(x: float, y: float) -> float:\n"
        f"    return 1.0 * ({body})\n"
    )


def _lower(source, mode=None):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_call_input_adapters import (
        active_numeric_mode, numeric_mode_scope,
    )

    with numeric_mode_scope(mode or active_numeric_mode()):
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="signal_core_route",
            extraction_contract=str(CONTRACT), progress=lambda _message: None,
        )
    root = module.functions[exports[0]]
    output = next(
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        for value in instruction.args
    )
    return module, exports[0], dict(root.metadata["parameter_names"]), output


def _ops(module):
    return [
        str(instruction.op)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    ]


def _receipt(module):
    return next(
        function.metadata["transcendental_pipeline"]
        for function in module.functions.values()
        if function.metadata.get("transcendental_pipeline")
    )


def _c_lane(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, output = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    return lambda **feeds: artifact.prepare_execution({
        parameters[name]: np.array([value])
        for name, value in feeds.items() if name in parameters
    }).run().buffers[output].reshape(-1)[0].item()


def _llvm_lane(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, output = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    return lambda **feeds: prepare_artifact_execution(artifact, {
        parameters[name]: np.array([value])
        for name, value in feeds.items() if name in parameters
    }).run().buffers[output].reshape(-1)[0].item()


LANES = pytest.mark.parametrize("lane", [_c_lane, _llvm_lane], ids=["c", "llvm"])


def test_contract_names_the_route_per_mode():
    from src.compiler import ssa_call_input_adapters as adapters

    assert adapters.transcendental_policy("compute") == "backend_intrinsic"
    assert adapters.transcendental_policy("scientific") == "signal_cores"
    assert adapters.transcendental_policy("stochastic") == "signal_cores"
    assert adapters.transcendental_policy() == "signal_cores"
    with adapters.numeric_mode_scope("compute"):
        assert adapters.transcendental_policy() == "backend_intrinsic"
    assert adapters.transcendental_policy() == "signal_cores"


def test_contract_refuses_any_other_transcendental_route(monkeypatch):
    from src.compiler import ssa_call_input_adapters as adapters

    monkeypatch.setitem(
        adapters._NUMERIC_CONVERSION_MODES, "compute",
        {**adapters._NUMERIC_CONVERSION_MODES["compute"],
         "transcendentals": "host_libm"},
    )
    with pytest.raises(ValueError, match="transcendentals must be one of"):
        adapters.transcendental_policy("compute")


#: (body, arguments, Python's value) -- arguments where the core is exact
#: and agrees with CPython bit for bit.
SIGNAL_CORE_CASES = [
    ("math.exp(x)", {"x": 0.5, "y": 0.0}),
    ("math.log(x)", {"x": 1.0000000000000002, "y": 0.0}),
    ("math.sin(x) + math.cos(y)", {"x": 1e22, "y": 3.0}),
    ("math.atan(x) + math.tanh(y)", {"x": 3.0, "y": -0.25}),
    ("x ** y + math.pow(y, x)", {"x": 1.5, "y": 2.5}),
]


@LANES
@pytest.mark.parametrize("body, arguments", SIGNAL_CORE_CASES)
def test_signal_cores_route_every_scalar_op_with_a_core(
    tmp_path, lane, body, arguments,
):
    lowered = _lower(_program(body))
    ops = set(_ops(lowered[0]))
    assert not ops & {"Exp", "Log", "Pow", "sin", "cos", "atan", "tanh"}
    assert _receipt(lowered[0])["policy"] == "signal_cores"
    namespace = {}
    exec(_program(body), namespace)
    assert lane(tmp_path, lowered)(**arguments) == namespace["root"](**arguments)


#: (body, arguments, the exception Python raises).
FAULT_CASES = [
    ("math.exp(x)", {"x": 710.0, "y": 0.0}, OverflowError),
    ("math.log(x)", {"x": 0.0, "y": 0.0}, ValueError),
    ("math.log(x)", {"x": -1.0, "y": 0.0}, ValueError),
    ("math.sin(x)", {"x": math.inf, "y": 0.0}, ValueError),
    ("x ** y", {"x": 0.0, "y": -1.0}, ZeroDivisionError),
    ("math.pow(x, y)", {"x": 0.0, "y": -1.0}, ValueError),
]


@pytest.mark.parametrize("body, arguments, raised", FAULT_CASES)
def test_signal_cores_fault_where_python_raises(
    tmp_path, body, arguments, raised,
):
    namespace = {}
    exec(_program(body), namespace)
    with pytest.raises(raised):
        namespace["root"](**arguments)
    run = _c_lane(tmp_path, _lower(_program(body)))
    with pytest.raises(raised):
        run(**arguments)


@pytest.mark.parametrize("body, op", [
    ("math.asin(x)", "asin"), ("math.sinh(x)", "sinh"),
])
def test_signal_cores_refuse_an_op_without_a_core(body, op):
    with pytest.raises(Exception, match=f"scalar float {op} has no signal core"):
        _lower(_program(body))


@LANES
def test_backend_intrinsic_leaves_math_to_the_backend(tmp_path, lane):
    body = "math.exp(x) + math.atan(y) + math.asin(0.25 * y)"
    lowered = _lower(_program(body), mode="compute")
    ops = set(_ops(lowered[0]))
    assert {"Exp", "atan", "asin"} <= ops
    receipt = _receipt(lowered[0])
    assert receipt["policy"] == "backend_intrinsic" and not receipt["lowered"]
    got = lane(tmp_path, lowered)(x=0.5, y=0.75)
    want = math.exp(0.5) + math.atan(0.75) + math.asin(0.1875)
    assert abs(got - want) <= 4 * math.ulp(want)
