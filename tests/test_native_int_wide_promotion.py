"""Python ``int`` arithmetic over an operand proven past int64 stays exact.

Program: the string-core intern digest,
``d = (d * 1000003 + token + 7) % 4611686018427387847`` in a loop.  The
interval solver proves ``d * 1000003`` int128 (``d`` is a floor modulus),
``token`` is unproven.  Wrong record: the ``+`` kept the int64 class seed, so
the native add read the int128 product truncated to int64 and the digest
came back wrong on C and LLVM with no fault (seed 42, 20 rounds).  Rule: an
op no interval bounds computes in its operands' exact common width (int128
here), checked like int64 arithmetic is; a wider result declared into an
int64 slot is narrowed checked.  Past int128 the result still reports
``NativeIntOverflow`` until it continues as Precision limbs.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

MIX = '''
def mix(seed: int, n: int) -> int:
    d = 0
    for i in range(n):
        token = seed + i
        d = (d * 1000003 + token + 7) % 4611686018427387847
    return d


def root(seed: int, n: int):
    return mix(seed, n)
'''

# Every Python int rule at int128: ``>>`` floors, ``<<`` past 63 bits,
# ``//`` by zero raises, a negative shift count raises, ``abs`` and ``-``.
WIDE_RULES = '''
def f(a: int, b: int, c: int) -> int:
    d = a % 4611686018427387847
    w = d * 1000003 + b
    return ((w >> 3) + (w << c) // 7 + abs(-w) + w // b) % 1000000007


def root(a: int, b: int, c: int):
    return f(a, b, c)
'''

SINGLE = '''
def root(a: int, b: int):
    d = a % 4611686018427387847
    return (d * 1000003 + b) % 1000000007
'''

# ``d * d * d`` reaches 2**186: past int128, Python continues exactly.
PAST_INT128 = '''
def f(a: int, b: int) -> int:
    d = a % 4611686018427387847
    return (d * d * d + b) % 1000000007


def root(a: int, b: int):
    return f(a, b)
'''

# The coordinator's shrunken intern driver: an unannotated ``fold`` whose
# ``+`` read the int128 product truncated, so the stage-1 check tested the
# low bits (n=4 gave 2126811163820088809).
UNANNOTATED_FOLD = '''
from array import array


def fold(d, v):
    return (d * 1000003 + v + 7) % 4611686018427387847


class V:
    def __init__(self):
        self.epoch = 0
        self.created = array("Q")

    def add(self, x):
        self.epoch += 1
        self.created.append(self.epoch)
        return 0, x + len(self.created)


def root(n: int) -> int:
    v = V()
    d = 0
    for i in range(n % 5):
        status, t = v.add(i)
        d = fold(d, t + status)
    d = fold(d, n)
    return d
'''

# The entry returns the int128 sum itself: its public slot is int64.
WIDE_RETURN = '''
def f(a: int, b: int) -> int:
    d = a % 4611686018427387847
    return d * 1000003 + b


def root(a: int, b: int):
    return f(a, b)
'''

# Proven int128 ([0, 4.6e24]): it fits int64 for small ``a`` only.
PROVEN_WIDE_RETURN = '''
def f(a: int) -> int:
    d = a % 4611686018427387847
    return d * 1000003


def root(a: int, b: int):
    return f(a)
'''

# Proven wholly above int64: no input fits the public int64 return.
NEVER_FITS_RETURN = '''
def f(a: int) -> int:
    d = a % 1000
    return d * 1000003 + 1208925819614629174706176


def root(a: int):
    return f(a)
'''

BIG_D = 4611686018427387000


def _python(source, case):
    namespace: dict = {}
    exec(source, namespace)
    return namespace["root"](*case)


@functools.lru_cache(maxsize=None)
def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_wide_promotion",
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


def _feeds(parameters, case):
    return {
        parameters[name]: np.array([value], dtype=np.int64)
        for name, value in zip(parameters, case)
    }


def _run_c(tmp_path, source, cases):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for case in cases:
        execution = artifact.prepare_execution(_feeds(parameters, case)).run()
        yield int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, source, cases):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for case in cases:
        execution = prepare_artifact_execution(
            artifact, _feeds(parameters, case),
        ).run()
        yield int(execution.buffers[outputs[0]].reshape(-1)[0])


RUNS = pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])


@RUNS
def test_mix_hash_digest_matches_python(tmp_path, run):
    cases = [(7, 3), (42, 20), (0, 0), (1, 1), (3, 200), (2**62, 5)]
    got = list(run(tmp_path, MIX, cases))
    assert got == [_python(MIX, case) for case in cases]
    # Round 3 of seed 7 is the first product past int64; seed 42 diverged.
    assert 17000110000177 * 1000003 > 2**63
    assert got[1] == 4233164913185866953


@RUNS
def test_unannotated_fold_with_unproven_actual_matches_python(tmp_path, run):
    cases = [(0,), (1,), (3,), (4,), (9,)]
    got = list(run(tmp_path, UNANNOTATED_FOLD, cases))
    assert got == [_python(UNANNOTATED_FOLD, case) for case in cases]
    assert got[3] == 2126811163861969217


@RUNS
def test_int128_arithmetic_keeps_python_rules(tmp_path, run):
    from src.compiler.native_faults import NativeValueError, NativeZeroDivisionError

    cases = [(5, 3, 2), (BIG_D, 1, 20), (BIG_D, -9, 40), (BIG_D, 2**62, 1)]
    assert list(run(tmp_path, WIDE_RULES, cases)) == [
        _python(WIDE_RULES, case) for case in cases
    ]
    with pytest.raises(NativeZeroDivisionError):
        list(run(tmp_path, WIDE_RULES, [(3, 0, 1)]))
    with pytest.raises(NativeValueError):
        list(run(tmp_path, WIDE_RULES, [(3, 5, -1)]))


@RUNS
def test_single_function_wide_sum_matches_python(tmp_path, run):
    cases = [(5, 3), (BIG_D, 1), (BIG_D, -(2**63)), (BIG_D, 2**63 - 1)]
    assert list(run(tmp_path, SINGLE, cases)) == [
        _python(SINGLE, case) for case in cases
    ]


@RUNS
def test_past_int128_is_reported_not_wrapped(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    assert list(run(tmp_path, PAST_INT128, [(5, 3)])) == [
        _python(PAST_INT128, (5, 3))
    ]
    with pytest.raises(NativeIntOverflow):
        list(run(tmp_path, PAST_INT128, [(BIG_D, 1)]))


@RUNS
def test_wrap_policy_keeps_the_proven_width(tmp_path, run, monkeypatch):
    from src.compiler import ssa_call_input_adapters as adapters

    mode = adapters._DEFAULT_NUMERIC_MODE
    monkeypatch.setitem(
        adapters._NUMERIC_CONVERSION_MODES, mode,
        {**adapters._NUMERIC_CONVERSION_MODES[mode], "integer_arithmetic": "wrap"},
    )
    cases = [(7, 3), (42, 20)]
    assert list(run(tmp_path, MIX, cases)) == [_python(MIX, case) for case in cases]


@RUNS
def test_wide_return_is_narrowed_checked_into_int64(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    for position, source in enumerate((WIDE_RETURN, PROVEN_WIDE_RETURN)):
        directory = tmp_path / f"source{position}"
        cases = [(5, 3), (9000, -5), (0, -(2**63))]
        assert list(run(directory, source, cases)) == [
            _python(source, case) for case in cases
        ]
        with pytest.raises(NativeIntOverflow):
            list(run(directory / "big", source, [(BIG_D, 1)]))


def test_return_proven_outside_int64_is_refused_by_name():
    with pytest.raises(ValueError, match="wholly outside int64"):
        _lowered(NEVER_FITS_RETURN)
