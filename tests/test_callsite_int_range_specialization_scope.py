"""Callsite copies that differ only by their actual's interval keep apart.

Shrunk from the string-core intern differential driver:
``validate_batch_envelope(1 << 64, ...)`` beside ``intern(1, ...)`` reaches
``_valid_domain(domain)`` with two proven intervals.  The numeric
specialization receipt left the interval out, so both copies shared one
numeric scope and the second formal range disagreed with the first
(``source_int_range_concordance disagreement``, recorded=(1, 2),
proposed=(2**64, 2**64)).  The interval is now part of the copy's identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def valid(domain):
    return domain == 1 or domain == 2


def check(domain, n):
    if not valid(domain):
        return 1
    return n


def root(seed: int, n: int) -> int:
    a = check(1 + seed % 2, n)
    b = check(BIG, n)
    c = check(True, n)
    return a * 100 + b * 10 + c
'''


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="callsite_int_range_scope",
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


def _feeds(parameters, seed, n):
    return {
        parameters["seed"]: np.array([seed], dtype=np.int64),
        parameters["n"]: np.array([n], dtype=np.int64),
    }


def _run_c(tmp_path, lowered, seed, n):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"c_{seed}_{n}")
    execution = artifact.prepare_execution(_feeds(parameters, seed, n)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, lowered, seed, n):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / f"llvm_{seed}_{n}")
    execution = prepare_artifact_execution(
        artifact, _feeds(parameters, seed, n)
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_copies_for_two_intervals_match_python(tmp_path, run):
    source = SOURCE.replace("BIG", "1 << 40")
    lowered = _lowered(source)
    for seed, n in ((0, 0), (1, 1), (2, 5)):
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n)


def test_interval_past_int64_lowers_and_names_its_overflow(tmp_path):
    from src.compiler.native_faults import NativeIntOverflow

    lowered = _lowered(SOURCE.replace("BIG", "1 << 64"))
    with pytest.raises(NativeIntOverflow):
        _run_c(tmp_path, lowered, 1, 1)
