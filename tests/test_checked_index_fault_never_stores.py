"""A checked index's failing edge never touches the element.

Shrunk from the string-core intern differential driver (``intern_batch``
fills and returns ``[STATUS_OK] * item_count``).  A callee that stores into
a list it returns had its out-of-range edge rerouted into the in-range block
because its outputs were not scalars, so after the fault was recorded the
store still wrote past the list's storage: C corrupted the heap (``free():
invalid next size``) and killed the host process.  The failing edge now
leaves through the one ``Trap`` whatever the outputs are.  Each lane either
matches Python or names a validation fault; it never crashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def make(seed, n):
    xs = [0] * (n + 1)
    i = 0
    while i < n + 1:
        xs[i] = seed + i
        i += 1
    return 0, xs


def root(seed: int, n: int) -> int:
    env, xs = make(seed, n)
    return env * 1000 + xs[0] * 100 + len(xs)
'''


def _python(seed, n):
    namespace: dict = {}
    exec(compile(SOURCE, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="checked_index_fault_never_stores",
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


def _prepare_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    return lambda seed, n: int(
        artifact.prepare_execution(_feeds(parameters, seed, n)).run()
        .buffers[outputs[0]].reshape(-1)[0]
    )


def _prepare_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    return lambda seed, n: int(
        prepare_artifact_execution(artifact, _feeds(parameters, seed, n)).run()
        .buffers[outputs[0]].reshape(-1)[0]
    )


def test_failing_edge_of_every_checked_index_is_the_trap():
    module, export, _parameters, _outputs = _lowered()
    for function in module.functions.values():
        for block in function.blocks.values():
            calls = [
                instruction for instruction in block.instrs
                if instruction.op == "Call"
                and instruction.attributes.get("callee") == "turing_validation_error"
                and instruction.attributes.get("error_code") == 71
            ]
            if not calls:
                continue
            successors = [
                function.blocks[str(instruction.attributes["target"])]
                for instruction in block.instrs if instruction.op == "Br"
            ]
            assert successors and all(
                successor.instrs and successor.instrs[-1].op == "Trap"
                for successor in successors
            ), (function.name, block.name)


@pytest.mark.parametrize("prepare", [_prepare_c, _prepare_llvm], ids=["c", "llvm"])
def test_lane_matches_python_or_names_a_fault(tmp_path, prepare):
    run = prepare(tmp_path, _lowered())
    for seed, n in ((0, 0), (1, 1), (3, 5), (5, 40)):
        try:
            value = run(seed, n)
        except RuntimeError as error:
            assert "validation failed with code" in str(error), error
        else:
            assert value == _python(seed, n), (seed, n)
