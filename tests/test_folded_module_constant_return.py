"""A folded arm that returns a module constant keeps its own return.

Shrunk from the string-core intern differential driver:
``Vocabulary._empty_token(DOMAIN_TEXT)`` returned ``BYTES_EMPTY_TOKEN``
natively.  The callsite copy specialized on ``domain=1`` folded
``if domain == DOMAIN_TEXT`` to its body, but the body's ``return
TEXT_EMPTY_TOKEN`` names a module constant (no local binding, no authored
expression on its folded Constant), so the structural fold found no selected
value, kept the fallthrough return site, and that site overwrote the
selected one.  Silent wrong: native 2, Python 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

LITERAL_ARGUMENT = '''
TEXT_EMPTY_TOKEN = 1
BYTES_EMPTY_TOKEN = 2
DOMAIN_TEXT = 1


def empty(domain: int) -> int:
    if domain == DOMAIN_TEXT:
        return TEXT_EMPTY_TOKEN
    return BYTES_EMPTY_TOKEN


def root(seed: int, n: int) -> int:
    return empty(1) * 1000 + empty(seed % 3) * 10 + n
'''

CONSTANT_TEST = '''
TEXT_EMPTY_TOKEN = 1
BYTES_EMPTY_TOKEN = 2


def empty(domain: int) -> int:
    if TEXT_EMPTY_TOKEN == 1:
        return TEXT_EMPTY_TOKEN
    return BYTES_EMPTY_TOKEN


def root(seed: int, n: int) -> int:
    return empty(seed) * 1000 + n
'''

CASES = ((0, 0), (1, 1), (2, 3))


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="folded_module_constant_return",
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


@pytest.mark.parametrize("source", [LITERAL_ARGUMENT, CONSTANT_TEST], ids=["literal_argument", "constant_test"])
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_selected_module_constant_return_matches_python(tmp_path, run, source):
    lowered = _lowered(source)
    for seed, n in CASES:
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n), (seed, n)
