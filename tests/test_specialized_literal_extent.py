"""Each specialization of a method states the extent of its own literal.

Program: string core's ``intern`` beside a direct batch call --
``v.batch(n, [97, 98, 2], [0, 2, 3])`` in the root and
``self.batch(d, atoms, [0, len(atoms)])`` inside ``v.one``.  Wrong
record: the copy whose ``offsets`` formal folded to the literal
``[0, 2, 3]`` stated no extent over its call edge, so the other copy's
(2,) stood alone on the shared authored row and was restamped over this
copy's (3,); lowering refused it as a shape disagreement.  A formal a
copy folded to a literal sequence states that literal's extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

_CLASS = '''
class V:
    def __init__(self):
        self.base = 7

    def envelope(self, values, offsets):
        if len(offsets) < 1:
            return 3, 0
        return 0, len(offsets) - 1

    def batch(self, d, values, offsets):
        envelope_status, count = self.envelope(values, offsets)
        statuses = [0] * count
        tokens = [0] * count
        if envelope_status == 0:
            i = 0
            while i < count:
                statuses[i] = i % 2
                tokens[i] = values[offsets[i]] + d + self.base
                i += 1
        return envelope_status, statuses, tokens

    def one(self, d, atoms):
        env, statuses, tokens = self.batch(d, atoms, [0, len(atoms)])
        if env != 0:
            return env, 0
        return statuses[0], tokens[0]
'''

SOURCES = {
    "literal_and_runtime_offsets": _CLASS + '''

def root(n: int):
    v = V()
    env, statuses, tokens = v.batch(n, [97, 98, 2], [0, 2, 3])
    a = statuses[0]
    b = tokens[0]
    status, token = v.one(n, [97, 98])
    return env * 1000 + a * 100 + b * 10 + token + status * 7 + len(tokens)
''',
    "second_row_of_literal_copy": _CLASS + '''

def root(n: int):
    v = V()
    env, statuses, tokens = v.batch(n, [97, 98, 2], [0, 2, 3])
    status, token = v.one(n, [97, 98, 99])
    return env * 1000 + statuses[1] * 100 + tokens[1] * 10 + token + status + len(tokens)
''',
}

CASES = (0, 1, 7)


def _lower(source, name):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    return lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=contract, progress=lambda _message: None,
    )


@pytest.fixture(scope="module", params=sorted(SOURCES))
def lowered(request):
    source = SOURCES[request.param]
    module, _outputs, exports = _lower(source, f"literal_extent_{request.param}")
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return source, module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _check(run, source, parameters, outputs):
    namespace = {}
    exec(source, namespace)
    for n in CASES:
        expected = namespace["root"](n)
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
        got = run(feeds).buffers[outputs[0]].reshape(-1)[0].item()
        assert type(got) is type(expected) and got == expected, (n, got, expected)


def test_matches_python_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source, module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    _check(
        lambda feeds: artifact.prepare_execution(feeds).run(),
        source, parameters, outputs,
    )


def test_matches_python_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    _check(
        lambda feeds: prepare_artifact_execution(artifact, feeds).run(),
        source, parameters, outputs,
    )
