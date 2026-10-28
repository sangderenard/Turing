"""A dict keyed by a 3- or 4-element tuple holds one row per distinct key.

Program: ``d[(1, n, 2)] = 5; d[(1, n + 1, 2)] = 6`` (string core
``key = (domain, hi, lo, count)``).  The static key-universe bound read only
the LAST component of a tuple subscript, so both stores counted as the one
literal key ``2``; the table got one row and the second insert faulted as
full (validation 75) on C and LLVM.  These programs pin the whole-key
bound, keys that differ only in their 3rd or 4th component, and uint64
fingerprint words at or above ``2**63``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"

MIDDLE = '''
def root(n: int) -> int:
    d = {}
    d[(1, n, 2)] = 5
    d[(1, n + 1, 2)] = 6
    return d.get((1, n, 2), 0) * 10 + len(d) * 100
'''

# hi and lo are uint64 words; n = 3887211255 puts hi above 2**63.
WIDE = '''
def root(n: int) -> int:
    hi = n % 4294967296 * (1 << 32) + 7
    lo = (n + 1) % 4294967296 * (1 << 32) + 9
    d = {}
    d[(1, hi, lo, 2)] = 5
    d[(1, hi, lo, 3)] = 6
    d[(1, hi, hi, 2)] = 7
    d[(1, hi, lo, 2)] = 8
    return (d.get((1, hi, lo, 2), 0) * 1000 + d.get((1, hi, lo, 3), 0) * 100
            + d.get((1, hi, hi, 2), 0) * 10 + d.get((1, lo, hi, 2), 0)
            + len(d) * 10000)
'''

LITERAL = '''
def root(n: int) -> int:
    d = {}
    d[(1, 2, 3)] = n
    d[(4, 5, 3)] = n + 1
    d[(4, 2, 3)] = n + 2
    d[(1, 2, 3)] = n + 3
    return d[(1, 2, 3)] + d[(4, 5, 3)] * 10 + d[(4, 2, 3)] * 100 + len(d) * 1000
'''

NAMED = '''
def root(n: int) -> int:
    hi = n % 4294967296 * (1 << 32) + 7
    lo = (n + 1) % 4294967296 * (1 << 32) + 9
    d = {}
    count = 1
    while count < 4:
        key = (1, hi, lo, count)
        d[key] = d.get(key, 0) + count
        count += 1
    key = (1, hi, lo, 2)
    d[key] = d.get(key, 0) + 9
    return len(d) * 100 + d[(1, hi, lo, 2)] * 10 + d.get((1, lo, hi, 2), 0)
'''

PROGRAMS = {"middle": MIDDLE, "wide": WIDE, "literal": LITERAL, "named": NAMED}
CASES = (0, 5, 3887211255, 4294967295)


def _lowered(source):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="tuple_dict_keys",
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


def _runner_c(source, directory):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(directory)
    return lambda n: int(artifact.prepare_execution(
        {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run().buffers[outputs[0]].reshape(-1)[0])


def _runner_llvm(source, directory):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=directory)
    return lambda n: int(prepare_artifact_execution(
        artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run().buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("program", sorted(PROGRAMS))
@pytest.mark.parametrize("runner", [_runner_c, _runner_llvm], ids=["c", "llvm"])
def test_tuple_keyed_dict_matches_python(tmp_path, runner, program):
    source = PROGRAMS[program]
    namespace = {}
    exec(source, namespace)
    run = runner(source, tmp_path)
    for n in CASES:
        assert run(n) == namespace["root"](n), n
