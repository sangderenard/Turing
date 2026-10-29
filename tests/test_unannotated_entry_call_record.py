"""An unannotated entry formal is typed by its host's calls.

Program: ``def entry(seed, n)`` (the string-core differential drivers), fed
int arrays by the host.  The entry's caller is the host, so the host's calls
are the call records the called inventory reads (``root_calls=``); with them
the formals are exactly ``int`` and the program runs on C and LLVM as Python
does.  With no record the refusal names the missing record, not an unread
dtype.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def entry(seed, n):
    s = seed
    total = 0
    i = 0
    while i < n:
        s = (s * 1103515245 + 12345) % 2147483648
        total = (total * 31 + s % 1000) % 1000003
        i += 1
    return total
'''

CASES = [(0, 0), (1, 1), (7, 5), (12345, 40)]


def _python(seed, n):
    namespace = {}
    exec(SOURCE, namespace)
    return namespace["entry"](seed, n)


def _lowered(**kwargs):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "entry", name="unannotated_entry_call_record",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
        **kwargs,
    )
    entry = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in entry.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(entry.metadata["parameter_names"]), outputs


@pytest.fixture(scope="module")
def lowered():
    return _lowered(root_calls=[{"seed": seed, "n": n} for seed, n in CASES])


def _feeds(parameters, seed, n):
    return {
        parameters["seed"]: np.array([seed], dtype=np.int64),
        parameters["n"]: np.array([n], dtype=np.int64),
    }


def test_host_calls_type_the_formals_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for seed, n in CASES:
        execution = artifact.prepare_execution(_feeds(parameters, seed, n)).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(seed, n)


def test_host_calls_type_the_formals_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for seed, n in CASES:
        execution = prepare_artifact_execution(
            artifact, _feeds(parameters, seed, n)
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(seed, n)


def test_no_call_record_is_refused_by_name():
    with pytest.raises(ValueError, match="no call record reaches formal"):
        _lowered()
