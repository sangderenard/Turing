"""A loop-carried int whose update is wider than its seed.

Program: ``total = 0; while j < n: total = total * 7 + store.get(j)`` with
``store.values = array("Q")``.  The header is int64 (its seed), the update
adds an int64 to a uint64 and is proven to fit int128.  Wrong records: the
reserved backedge slot kept int64 while its producer wrote int128, so LLVM
refused the header Phi ("physical element i64 disagrees with incoming
i128") and C assigned the int128 to the int64 cell, dropping its high bits.
The update now enters the header through an explicit integer conversion in
the latch, range-checked under the numeric contract (``INT_OVERFLOW``), and
the C spelling of that check's int64 lower bound is a valid constant (it
read +2**63 and reported every value as overflowing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCE = '''
from array import array


class Store:
    def __init__(self):
        self.values = array("Q")

    def add(self, v):
        self.values.append(v)

    def get(self, i):
        return self.values[i]


def root(n: int):
    store = Store()
    i = 0
    while i < n:
        store.add(i * 3 + 1)
        i += 1
    total = 0
    j = 0
    while j < n:
        total = total * 7 + store.get(j)
        j += 1
    return total
'''

SMALL = (0, 1, 3, 9)
#: 7**n passes int64 near n = 23: Python keeps the int, native reports it.
LARGE = 30


def _python(n):
    namespace = {}
    exec(SOURCE, namespace)
    return namespace["root"](n)


@pytest.fixture(scope="module")
def lowered():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="carried_integer_width_narrowing",
        extraction_contract=contract, progress=lambda _message: None,
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


def _runner(backend, lowered, directory):
    module, export, parameters, outputs = lowered
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(directory)
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=directory)
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)

    def run(n):
        execution = prepare(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        return int(execution.buffers[outputs[0]].reshape(-1)[0])

    return run


def test_header_joins_through_an_explicit_conversion(lowered):
    module, export, _parameters, _outputs = lowered
    function = module.functions[export]
    produced = {
        int(instruction.res.id): instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.op != "Phi" or instruction.res is None:
                continue
            for argument in instruction.args:
                assert str(argument.dtype) == str(instruction.res.dtype), (
                    instruction.res.id, argument.id,
                )
    narrowing = [
        instruction for instruction in produced.values()
        if instruction.op == "Cast"
        and instruction.attributes.get("conversion_kind") == "integer_conversion"
        and str(instruction.args[0].dtype) == "int128"
        and str(instruction.res.dtype) == "int64"
    ]
    assert narrowing, "the int128 update enters the int64 header unconverted"


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_matches_python_and_reports_past_int64(tmp_path, lowered, backend):
    from src.compiler.native_faults import NativeIntOverflow

    run = _runner(backend, lowered, tmp_path / backend)
    for n in SMALL:
        assert run(n) == _python(n)
    assert _python(LARGE) >= 1 << 63
    with pytest.raises(NativeIntOverflow):
        run(LARGE)


def test_c_int64_lower_bound_is_a_signed_constant():
    from src.compiler.ssa_llvm_backend import integer_narrowing_c_check

    check = integer_narrowing_c_check("int128", "int64", "__int128", "x")
    assert "INT64_C(-9223372036854775808)" not in check
    assert "-INT64_C(9223372036854775807) - 1" in check
