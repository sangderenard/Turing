"""Callsite copies that differ only by a formal's capacity policy keep apart.

Shrunk from the string-core intern differential driver: ``intern_batch`` on
a display and again on a list grown by appends.  The two
``_validate_atom_span`` copies settled their ``atoms`` formal FIXED and
DYNAMIC, but carried one callsite contract, so both lowered under one symbol
and the FIXED caller bound the DYNAMIC view ("bound sequence views disagree
about dynamic capacity").  The settled capacity policy is now part of the
copy's identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("/home/user/nogodsnomasters/string_core")

pytestmark = pytest.mark.skipif(
    not (CORE / "token_orchard" / "core.py").exists(),
    reason="the Token Orchard string core is not checked out",
)

SOURCE = '''
from token_orchard.core import Vocabulary


def entry(seed: int, n: int) -> int:
    vocab = Vocabulary()
    env, statuses, tokens = vocab.intern_batch(1, [65, 66, 67], [0, 3])
    first = tokens[0]
    atoms = []
    j = 0
    while j < n:
        atoms.append(65 + (seed + j) % 3)
        j += 1
    env2, statuses2, tokens2 = vocab.intern_batch(1, atoms, [0, len(atoms)])
    second = tokens2[0]
    return first * 10000000 + second
'''

CASES = ((1, 1), (2, 3), (3, 7))


def _python(seed, n):
    import sys

    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    namespace: dict = {}
    exec(compile(SOURCE, "<program>", "exec"), namespace)
    return namespace["entry"](seed, n)


@pytest.fixture(scope="module")
def lowered():
    import sys

    # The program imports ``token_orchard``: the host that feeds it can.
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = (
        ExtractionContract(ROOT / "extraction_contracts" / "program_extraction.yaml")
        .with_sources(tuple(
            (f"token_orchard.{name}", CORE / "token_orchard" / f"{name}.py")
            for name in ("contract", "kernel", "core")
        ))
        .with_execution_file(
            ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"
        )
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "entry", name="callsite_dynamic_capacity",
        extraction_contract=contract, progress=lambda _message: None,
        root_calls=[{"seed": seed, "n": n} for seed, n in CASES],
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


def test_fixed_and_dynamic_copies_lower_under_two_symbols(lowered):
    module = lowered[0]
    symbols = {
        name for name in module.functions
        if "_validate_atom_span__specialized_" in str(name)
    }
    assert len(symbols) >= 2, symbols


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_both_copies_match_python(tmp_path, lowered, backend):
    module, export, parameters, outputs = lowered
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")

        def prepare(feeds):
            return prepare_artifact_execution(artifact, feeds)
    for seed, n in CASES:
        execution = prepare(_feeds(parameters, seed, n)).run()
        native = int(execution.buffers[outputs[0]].reshape(-1)[0])
        assert native == _python(seed, n), (seed, n)
