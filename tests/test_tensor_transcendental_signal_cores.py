"""Elementwise tensor transcendentals follow the numeric contract too.

* ``signal_cores`` (``scientific``, the default): a float64 tensor
  ``exp``/``log``/``sin``/``cos``/``tan``/``atan``/``tanh``/``**`` becomes an
  SSA loop over its elements
  (``ir_identities.scalarize_elementwise_transcendentals``) whose body is the
  scalar op the signal cores route, so every element is the core's correctly
  rounded double -- no lowercase tensor op is left for C libm, LLVM's
  ``unary_double`` kernel or Fortran's elemental intrinsics -- and every
  element carries the core's fault guards.  An elementwise op with no core
  is refused by name.
* ``backend_intrinsic`` (``compute``): the tensor op is left to the backend.

Exactness is judged bit for bit against mpmath (the corpus judge allows
1e-12): the weighted sum below is evaluated in the same IEEE order natively
and in Python, so it is equal only when every element is the correctly
rounded double.
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath
import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _program(expression: str) -> str:
    return (
        "import numpy as np\n\n\n"
        "def root(x: float, y: float) -> float:\n"
        "    a = np.asarray([x, y, 0.75])\n"
        f"    e = {expression}\n"
        "    return e[0] + 2.0 * e[1] + 4.0 * e[2]\n"
    )


def _lower(source, mode=None):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_call_input_adapters import (
        active_numeric_mode, numeric_mode_scope,
    )

    with numeric_mode_scope(mode or active_numeric_mode()):
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="tensor_signal_core_route",
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


def _shaped_ops(module):
    return {
        str(instruction.op)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None and tuple(instruction.res.shape or ())
    }


def _receipts(module):
    return [
        function.metadata["transcendental_pipeline"]
        for function in module.functions.values()
        if function.metadata.get("transcendental_pipeline")
    ]


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

mpmath.mp.prec = 300


def _rounded(function, *arguments) -> float:
    """mpmath at 300 bits, rounded once to the nearest double."""

    return float(function(*(mpmath.mpf(value) for value in arguments)))


def _weighted(elements) -> float:
    return elements[0] + 2.0 * elements[1] + 4.0 * elements[2]


#: (expression, the elementwise op, mpmath's function, x, y).  The
#: arguments are hard: exp(192.9234597167931) is 1 ulp off in glibc
#: (c913), sin/cos(1e22) need the full 2/pi, log at 1 + ulp.  (No tensor
#: ``atan``/``atan2`` row: the frontend does not type ``np.arctan(a)`` --
#: "consumed as a host scalar" -- before any math runs.)
CASES = [
    ("np.exp(a)", "exp", mpmath.exp, 192.9234597167931, 534.707910443053),
    ("np.log(a)", "log", mpmath.log, 1.0000000000000002, 1e-300),
    ("np.sin(a)", "sin", mpmath.sin, 1e22, 3.0),
    ("np.cos(a)", "cos", mpmath.cos, 1e22, 1.5707963267948966),
    ("np.tan(a)", "tan", mpmath.tan, 1.5707963267948966, -0.5),
    ("np.tanh(a)", "tanh", mpmath.tanh, -0.25, 19.0),
    ("a ** 1.5", "Pow", lambda v: v ** mpmath.mpf(1.5), 1.5, 2.5),
]


@LANES
@pytest.mark.parametrize("expression, op, reference, x, y", CASES)
def test_signal_cores_route_each_element(
    tmp_path, lane, expression, op, reference, x, y,
):
    lowered = _lower(_program(expression))
    module = lowered[0]
    # No shaped transcendental is left for a backend's host math.
    assert not _shaped_ops(module) & {
        "exp", "log", "sin", "cos", "tan", "atan", "tanh", "pow",
        "Exp", "Log", "Pow",
    }
    receipts = _receipts(module)
    assert any(
        receipt["policy"] == "signal_cores"
        and receipt["elementwise"].get(op) == 1
        for receipt in receipts
    ), receipts
    assert not any(receipt["left_native"] for receipt in receipts)
    want = _weighted([_rounded(reference, value) for value in (x, y, 0.75)])
    assert lane(tmp_path, lowered)(x=x, y=y) == want


def test_exp_is_correctly_rounded_where_the_host_is_not():
    """The case the bit test is for: glibc's exp misses by one ulp here."""

    hard = 192.9234597167931
    assert math.exp(hard) != _rounded(mpmath.exp, hard)


#: (expression, x, y, the exception the scalar contract raises).
FAULT_CASES = [
    ("np.exp(a)", 1.0, 710.0, OverflowError),
    ("np.log(a)", 2.0, 0.0, ValueError),
    ("np.log(a)", -1.0, 2.0, ValueError),
    ("np.sin(a)", 0.5, math.inf, ValueError),
]


@pytest.mark.parametrize("expression, x, y, raised", FAULT_CASES)
def test_signal_cores_guard_every_element(tmp_path, expression, x, y, raised):
    run = _c_lane(tmp_path, _lower(_program(expression)))
    with pytest.raises(raised):
        run(x=x, y=y)
    # The same program on clean elements runs: the guard is per element.
    run(x=0.5, y=0.25)


@pytest.mark.parametrize("op", ["sinh", "asin", "cosh", "acos"])
def test_signal_cores_refuse_an_elementwise_op_without_a_core(op):
    """The frontend types none of these on a tensor yet (``np.sinh(a)``
    stops at "consumed as a host scalar"), so the shaped op is planted on a
    compute-mode lowering of ``np.exp(a)`` -- the same record with another
    catalogue name -- and the pass is run under ``scientific``."""

    from src.compiler.ir_identities import lower_transcendentals_to_signal_cores

    module = _lower(_program("np.exp(a)"), mode="compute")[0]
    planted = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) == "exp"
    ]
    assert len(planted) == 1
    planted[0].op = op
    with pytest.raises(
        ValueError, match=f"elementwise {op} has no signal core",
    ):
        lower_transcendentals_to_signal_cores(
            module.functions, mode="scientific",
        )


@LANES
def test_backend_intrinsic_leaves_tensor_math_to_the_backend(tmp_path, lane):
    lowered = _lower(_program("np.exp(a) + np.sin(a)"), mode="compute")
    module = lowered[0]
    assert {"exp", "sin"} <= _shaped_ops(module)
    receipts = _receipts(module)
    assert receipts and all(
        receipt["policy"] == "backend_intrinsic"
        and not receipt["elementwise"] and not receipt["lowered"]
        for receipt in receipts
    )
    assert {"exp", "sin"} <= {
        op for receipt in receipts for op in receipt["left_native"]
    }
    got = lane(tmp_path, lowered)(x=0.5, y=1.25)
    want = _weighted([
        math.exp(value) + math.sin(value) for value in (0.5, 1.25, 0.75)
    ])
    assert abs(got - want) <= 8 * math.ulp(want)
