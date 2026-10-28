"""Proven int intervals remove overflow checks, and only where they hold.

``_solve_int_ranges`` stamps each proven builtin ``int`` value with its
interval (``python_int_range``); a native op whose interval lies inside
int64 drops its ``__builtin_*_overflow`` / ``llvm.s*.with.overflow`` check.
A wrong interval would silently wrap, so each rule below is pinned twice:
the stamped interval is exactly Python's bound, and the native lanes compute
a value AT that bound exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
INT64_MAX = (1 << 63) - 1


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_range_proofs",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _instructions(module):
    for name, function in module.functions.items():
        for block in function.blocks.values():
            for instruction in block.instrs:
                yield name, instruction


def _stamped(module, op):
    """The ``python_int_range`` of every authored ``op`` instruction."""

    return sorted((
        tuple(instruction.attributes["python_int_range"])
        if instruction.attributes.get("python_int_range") is not None else None
        for _name, instruction in _instructions(module)
        if instruction.op == op
        and instruction.attributes.get("source_type") in {"BinOp", "AugAssign"}
    ), key=repr)


def _c_source(module, export):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    return artifact, artifact.source


def _feed(parameters, values):
    return {parameters[name]: np.array([value], dtype=np.int64) for name, value in values.items()}


def _fresh(tmp_path):
    """A new build directory for each native run of one test."""

    index = len(tuple(tmp_path.iterdir()))
    directory = tmp_path / f"run{index}"
    directory.mkdir()
    return directory


def _run_c(tmp_path, source, values):
    tmp_path = _fresh(tmp_path)
    module, export, parameters, outputs = _lowered(source)
    artifact, _text = _c_source(module, export)
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(_feed(parameters, values)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, source, values):
    tmp_path = _fresh(tmp_path)
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, _feed(parameters, values),
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _python(source, values):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](**values)


def _overflow_builtins(module, export, kinds=r"\w+"):
    return len(re.findall(
        rf"__builtin_(?:{kinds})_overflow", _c_source(module, export)[1],
    ))


RUNS = pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])

# --- 1. range() loop counters --------------------------------------------
#
# ``i`` of ``range(m)`` with ``m`` in [0, 999] lies in [0, 998]; the
# product at i = 998 is 9204925292780865450, inside int64 by 1.8e16.
RANGE_STOP = """
def root(n: int):
    m = n % 1000
    last = 0
    for i in range(m):
        last = i * 9223372036854775
    return last
"""

# A negative step: ``range(m, -1, -2)`` runs from m down to 0 or 1.
RANGE_STEP = """
def root(n: int):
    m = n % 1000
    last = 0
    for i in range(m, -1, -2):
        last = i * 9232604641496272
    return last
"""


def test_range_counter_interval_is_stamped():
    module, export, _parameters, _outputs = _lowered(RANGE_STOP)
    assert _stamped(module, "Mul") == [(0, 998 * 9223372036854775)]
    # The one ``__builtin_add_overflow`` left is the compiler's saturating
    # loop latch, not authored arithmetic.
    assert _overflow_builtins(module, export, "mul") == 0
    module, export, _parameters, _outputs = _lowered(RANGE_STEP)
    assert _stamped(module, "Mul") == [(0, 999 * 9232604641496272)]
    assert 999 * 9232604641496272 <= INT64_MAX
    assert _overflow_builtins(module, export, "mul") == 0


@RUNS
@pytest.mark.parametrize("source", [RANGE_STOP, RANGE_STEP], ids=["stop", "step"])
def test_range_counter_edge_is_exact(tmp_path, run, source):
    for n in (999, 998, 0):
        assert run(tmp_path, source, {"n": n}) == _python(source, {"n": n})


# --- 2. conditional expressions -------------------------------------------

IFEXP = """
def root(a: int):
    k = 4611686018427387903 if a > 0 else -4611686018427387904
    return k + k
"""


def test_conditional_expression_joins_its_arms():
    module, export, _parameters, _outputs = _lowered(IFEXP)
    assert _stamped(module, "Add") == [(-(1 << 63), INT64_MAX - 1)]
    assert _overflow_builtins(module, export) == 0


@RUNS
def test_conditional_expression_edges_are_exact(tmp_path, run):
    assert run(tmp_path, IFEXP, {"a": 1}) == INT64_MAX - 1
    assert run(tmp_path, IFEXP, {"a": 0}) == -(1 << 63)


# --- 3. callsite intervals reach callee formals ---------------------------
#
# ``step`` is called once with ``% 2**32`` / ``% 256`` limbs and once with
# the raw root inputs.  The copy for the proven callsite is proven; the
# other stays checked (its formal is reachable from an unproven root input).
CALLEE = """
def step(low: int, byte: int) -> int:
    return low * 2147483648 + byte


def root(x: int, y: int):
    a = step(x % 4294967296, y % 256)
    b = step(x, y)
    return a % 1000003 + b % 1000003
"""


def test_proven_callsite_interval_reaches_the_copy_formals():
    from src.compiler.ssa_llvm_backend import python_int_overflow_checked

    module, export, _parameters, _outputs = _lowered(CALLEE)
    edge = (4294967295 * 2147483648, 4294967295 * 2147483648 + 255)
    stamped = _stamped(module, "Add")
    assert (0, edge[1]) in stamped
    assert None in stamped, "the unproven callsite's copy must stay unproven"
    checked = [
        instruction for _name, instruction in _instructions(module)
        if instruction.op in {"Add", "Sub", "Mul"}
        and python_int_overflow_checked(instruction)
    ]
    assert len(checked) == 2
    assert _overflow_builtins(module, export) == len(checked)


@RUNS
def test_callee_formal_edge_is_exact(tmp_path, run):
    values = {"x": 4294967295, "y": 255}
    assert run(tmp_path, CALLEE, values) == _python(CALLEE, values)


@RUNS
def test_unproven_callsite_still_faults(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    with pytest.raises(NativeIntOverflow):
        run(tmp_path, CALLEE, {"x": 1 << 40, "y": 0})


# --- 4. % by a negative constant, // by a positive one --------------------
#
# ``len(xs)`` gets no interval: its true bound (2**60 - 1 items for a list)
# times an ordinary multiplier leaves int64, and a proven interval past
# int64 widens the value (corpus c271: ``len(a) * 10 + ...`` became an
# int128 the root ABI cannot return).  It stays a checked int64.

LEN = """
def root(n: int):
    xs = [k for k in range(n % 50)]
    return len(xs) * 10 + 1
"""

# ``a % -7`` lies in [-6, 0]; the product at -6 is -9223372036854775806.
MOD_NEGATIVE = """
def root(a: int):
    return (a % -7) * 1537228672809129301
"""

# ``(a % 2**40) // 3`` lies in [0, 366503875925]; times 25165824 the top is
# 9223372036854710200 <= INT64_MAX.
FLOORDIV = """
def root(a: int):
    return ((a % 1099511627776) // 3) * 25165824
"""


def test_mod_floordiv_and_len_intervals():
    module, export, _parameters, _outputs = _lowered(LEN)
    assert _stamped(module, "Add") == [None]
    module, export, _parameters, _outputs = _lowered(MOD_NEGATIVE)
    assert _stamped(module, "Mod") == [(-6, 0)]
    assert _stamped(module, "Mul") == [(-6 * 1537228672809129301, 0)]
    assert _overflow_builtins(module, export) == 0
    module, export, _parameters, _outputs = _lowered(FLOORDIV)
    assert _stamped(module, "FloorDiv") == [(0, 366503875925)]
    assert _stamped(module, "Mul") == [(0, 366503875925 * 25165824)]
    assert 366503875925 * 25165824 <= INT64_MAX
    assert _overflow_builtins(module, export) == 0


@RUNS
def test_mod_floordiv_and_len_edges_are_exact(tmp_path, run):
    assert run(tmp_path, LEN, {"n": 49}) == 49 * 10 + 1
    assert run(tmp_path, MOD_NEGATIVE, {"a": -6}) == -6 * 1537228672809129301
    assert run(tmp_path, MOD_NEGATIVE, {"a": 6}) == _python(MOD_NEGATIVE, {"a": 6})
    edge = 1099511627775
    assert run(tmp_path, FLOORDIV, {"a": edge}) == _python(FLOORDIV, {"a": edge})


# --- 5. a while test bounds the value it compares -------------------------

WHILE = """
def root(a: int):
    index = a % 2
    while index < 4:
        index += 1
    return index * 2305843009213693951
"""

# The guard does not bound a value an inner loop carries: ``i`` passes the
# outer test at 3 and the inner loop takes it to 6, whose product leaves
# int64.  It must stay checked and fault, never wrap.
NESTED = """
def root(n: int):
    i = n % 4
    while i < 4:
        for k in range(3):
            i = i + 1
    return i * 1844674407370955161
"""


def test_while_test_bounds_its_body():
    module, export, _parameters, _outputs = _lowered(WHILE)
    assert _stamped(module, "Add") == [(0, 4)]
    assert _stamped(module, "Mul") == [(0, 4 * 2305843009213693951)]
    assert _overflow_builtins(module, export) == 0
    module, _export, _parameters, _outputs = _lowered(NESTED)
    assert _stamped(module, "Mul") == [None]


@RUNS
def test_while_edges_are_exact(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    assert run(tmp_path, WHILE, {"a": 1}) == 4 * 2305843009213693951
    assert run(tmp_path, NESTED, {"n": 2}) == 5 * 1844674407370955161
    with pytest.raises(NativeIntOverflow):
        run(tmp_path, NESTED, {"n": 3})
