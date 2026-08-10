from __future__ import annotations

import shutil

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.common.tensors.fused_ir import ordered_feed_ids
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.wasm_fidelity import verify_wasm_module, verify_wasm_source


# A trailing-axis reduction (N*K -> N): the flat run(count, ...) model cannot
# express it directly, so it exercises the nested reduction emitter and the
# NumPy oracle's matching min/sum lowering at once.
SOURCE = """
def reduce_kernel(px, ex):
    diff = px - ex
    dist = diff * diff
    nearest = dist.min(dim=-1)
    total = dist.sum(dim=-1)
    return total + nearest
"""


# Same feeds, same reduction structure, same output arity -- only the final
# combine differs, so it runs with the correct program's ABI yet computes a
# different number.
WRONG_SOURCE = """
def reduce_kernel(px, ex):
    diff = px - ex
    dist = diff * diff
    nearest = dist.min(dim=-1)
    total = dist.sum(dim=-1)
    return total * nearest
"""


def _reduction_module(source=SOURCE):
    named_feeds = {
        "px": np.asarray([[1.0], [4.0], [7.0], [10.0]]),
        "ex": np.asarray([[0.5, 2.5, 5.5]]),
    }
    aot = compile_ast_aot(
        source, "reduce_kernel", named_feeds, precompile_only=True, remove_loops=True
    )
    program = project_public_numerical_program(aot)
    module = emit_wasm_module(program, name="reduce_kernel", dtype="float64")
    origins = program.extras["capture_feed_origins"]
    feeds = {
        feed_id: named_feeds[origins[feed_id]["binding_name"]]
        for feed_id in ordered_feed_ids(program)
    }
    return module, program, feeds


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_emitted_reduction_wasm_matches_the_numpy_reference(tmp_path):
    module, program, feeds = _reduction_module()
    assert module.complete, module.shortfall_report()
    assert module.binary is not None

    proof = verify_wasm_module(
        module, program, feeds, tmp_path, entrypoint="reduce_kernel"
    )

    assert proof["passed"] is True
    assert proof["case_count"] == 3
    assert all(case["passed"] for case in proof["cases"])
    assert max(
        output["max_absolute_error"]
        for case in proof["cases"]
        for output in case["outputs"]
    ) == 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_fidelity_check_rejects_a_runnable_wrong_wasm_binary(tmp_path):
    module, program, feeds = _reduction_module()
    wrong_module, _wrong_program, _wrong_feeds = _reduction_module(WRONG_SOURCE)

    # Execute the arithmetically-wrong binary under the correct program's ABI
    # and reference: it runs cleanly yet disagrees numerically.
    class _Runnable:
        binary = wrong_module.binary
        api = module.api

    with pytest.raises(AssertionError, match="disagrees with the reference"):
        verify_wasm_module(
            _Runnable(), program, feeds, tmp_path, entrypoint="reduce_kernel"
        )


# --- operator coverage -----------------------------------------------------
# verify_wasm_source is the single-call path for exercising operators as
# extensively as wanted: give it a source snippet and named feeds and it
# compiles, emits, and compares the WebAssembly against the NumPy oracle.

_X = np.asarray([-2.0, -0.5, 0.25, 1.5, 3.0, 4.0])
_Y = np.asarray([1.0, 2.0, -1.5, 0.5, 3.0, -4.0])

_ELEMENTWISE_CASES = {
    "add": "def op(x, y):\n    return x + y\n",
    "sub": "def op(x, y):\n    return x - y\n",
    "mul": "def op(x, y):\n    return x * y\n",
    "truediv": "def op(x, y):\n    return x / y\n",
    "less_equal": "def op(x, y):\n    return (x <= y)\n",
    "greater": "def op(x, y):\n    return (x > y)\n",
    "less": "def op(x, y):\n    return (x < y)\n",
    "greater_equal": "def op(x, y):\n    return (x >= y)\n",
    "neg": "def op(x, y):\n    return -x + y - y\n",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("name", sorted(_ELEMENTWISE_CASES))
def test_elementwise_operator_matches_numpy(tmp_path, name):
    proof = verify_wasm_source(
        _ELEMENTWISE_CASES[name],
        "op",
        {"x": _X, "y": _Y},
        tmp_path,
    )
    assert proof["passed"] is True
    assert all(case["passed"] for case in proof["cases"])


# Bitwise ops require an integer working type. The compiler's own integer
# results default to int64 (a valuewise ``int(a) & int(b)`` is an unbounded
# Python int, materialised as int64), so int64 is the working type these
# programs actually compile to -- and the WebAssembly is emitted, executed in
# linear memory (i64.load/i64.store arrays), and read back through a
# BigInt64Array, then compared against the integer NumPy reference. ``y``
# doubles as the shift amount for shl/shr, so it stays small and non-negative
# (a negative shift count is undefined); ``x`` still carries negative
# two's-complement values so ``~`` and the arithmetic ``shr`` are exercised.
_XI = np.asarray([0b1010, 0b1100, 5, 255, -8, 123], dtype=np.int64)
_YI = np.asarray([0b0110, 3, 2, 1, 2, 7], dtype=np.int64)

_BITWISE_CASES = {
    "bitand": "def op(x, y):\n    return x & y\n",
    "bitor": "def op(x, y):\n    return x | y\n",
    "bitxor": "def op(x, y):\n    return x ^ y\n",
    "shl": "def op(x, y):\n    return x << y\n",
    "shr": "def op(x, y):\n    return x >> y\n",
    "invert": "def op(x, y):\n    return ~x + y - y\n",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("name", sorted(_BITWISE_CASES))
def test_bitwise_operator_matches_numpy_in_linear_memory(tmp_path, name):
    proof = verify_wasm_source(
        _BITWISE_CASES[name],
        "op",
        {"x": _XI, "y": _YI},
        tmp_path,
        dtype="int64",
    )
    assert proof["passed"] is True
    assert all(case["passed"] for case in proof["cases"])


# A view op (reshape/view/clone) is the same linear elements under a new shape.
# The WebAssembly backend lowers it to an identity of its operand, so the
# result is bit-identical to the un-reshaped bitwise result -- and the shape
# argument constants fall out as dead. The fidelity oracle compares flattened
# values, so the shape change itself is transparent to the check; what this
# proves is that the view lowers and runs rather than being refused.
@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_reshape_view_lowers_and_matches_numpy(tmp_path):
    proof = verify_wasm_source(
        "def op(x, y):\n    return (x & y).reshape(2, 3)\n",
        "op",
        {"x": _XI, "y": _YI},
        tmp_path,
        dtype="int64",
    )
    assert proof["passed"] is True
    assert all(case["passed"] for case in proof["cases"])


# (An int32 working-type end-to-end test is intentionally omitted: the
# compiler materialises every integer result as int64 -- a valuewise
# ``int(a) op int(b)`` is an unbounded Python int -- so an int32 *working
# type* would emit an i64 memory store of an i32 value. int32 remains a
# supported emission path for programs whose result Meta is genuinely int32;
# the front end simply does not produce one from these source expressions.)


# Reductions need a derived grid (a direct N*K feed cannot be sized by the
# count-based ABI), so each builds one from a row feed minus a kaxis feed.
_REDUCE_CASES = {
    "sum": "def op(px, ex):\n    g = (px - ex) * (px - ex)\n    return g.sum(dim=-1)\n",
    "min": "def op(px, ex):\n    g = (px - ex) * (px - ex)\n    return g.min(dim=-1)\n",
    "max": "def op(px, ex):\n    g = (px - ex) * (px - ex)\n    return g.max(dim=-1)\n",
    "mean": "def op(px, ex):\n    g = (px - ex) * (px - ex)\n    return g.mean(dim=-1)\n",
    "prod": "def op(px, ex):\n    g = (px - ex) * (px - ex)\n    return g.prod(dim=-1)\n",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("name", sorted(_REDUCE_CASES))
def test_axis_reduction_operator_matches_numpy(tmp_path, name):
    px = np.asarray([[1.0], [4.0], [7.0], [10.0]])
    ex = np.asarray([[0.5, 2.5, 5.5]])
    proof = verify_wasm_source(
        _REDUCE_CASES[name],
        "op",
        {"px": px, "ex": ex},
        tmp_path,
    )
    assert proof["passed"] is True
    assert all(case["passed"] for case in proof["cases"])

