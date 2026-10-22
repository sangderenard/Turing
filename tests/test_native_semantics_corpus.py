"""Authored programs compiled to native C and executed against Python.

Each file in ``tests/native_semantics_corpus`` defines ``root`` and ends with
``# INPUTS [...]``, the argument sets it is run with.  The program is lowered
through the sanctioned entry, ``lower_ast_source_to_ssa``, under the
repository's extraction contract, emitted by the C module lane, compiled and
run; the native result must equal the Python result for every input.  The
lowered module must also leave the identity concordance with no finding.

The corpus records miscompiles that a clean audit did not catch: ``a / b`` on
two ints truncated, a structural ``Ret`` replaced the if-merge of a returned
name by the pre-``if`` loop result, a region fed a settled ``int64`` was cast
to its cut-time ``float64`` formal, and a list's arena became a double buffer
read back as float64.
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

CORPUS = Path(__file__).with_name("native_semantics_corpus")
PROGRAMS = sorted(CORPUS.glob("c*.py"))
CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

_RUNNER = r'''
import pickle, sys
import numpy as np
artifact, parameters, outputs, cases = pickle.load(open(sys.argv[1], "rb"))
for feeds, expected in cases:
    execution = artifact.prepare_execution({
        parameters[name]: np.array([value]) for name, value in feeds.items()
    }).run()
    got = [
        float(item)
        for value_id in outputs
        for item in execution.buffers[value_id].reshape(-1).tolist()
    ]
    want = [float(item) for item in expected]
    assert len(got) == len(want), (feeds, got, want)
    for left, right in zip(got, want):
        assert abs(left - right) <= 1e-12 * max(1.0, abs(right)), (feeds, got, want)
'''


def _inputs(source: str) -> list[dict]:
    return ast.literal_eval(source.split("# INPUTS", 1)[1].strip())


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda path: path.stem)
def test_native_program_matches_python(program, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source = program.read_text()
    namespace: dict = {}
    exec(compile(source, str(program), "exec"), namespace)
    cases = []
    for feeds in _inputs(source):
        expected = namespace["root"](**feeds)
        cases.append((feeds, list(expected) if isinstance(expected, tuple) else [expected]))

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"corpus_{program.stem}",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    findings = CorrelationTable.build(module).findings(module)
    assert findings == [], findings

    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    returns = [
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]
    assert len(returns) == 1
    outputs = [int(value.id) for value in returns[0]]
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    saved = tmp_path / "artifact.pkl"
    saved.write_bytes(pickle.dumps((artifact, parameters, outputs, cases)))
    result = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(saved)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unbounded_local_list_is_an_explicit_boundary():
    """A list grown by a data-bounded ``while`` has no proven capacity.

    It used to be given a one-element private arena whose ``full`` status
    nobody read, so ``len(xs)`` returned 1.  Dynamic growth is the declared
    unsupported boundary; emission must say so instead of computing.
    """
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source = (
        "def root(n: int) -> int:\n"
        "    xs = []\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        xs.append(i * 2)\n"
        "        i += 1\n"
        "    return len(xs)\n"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="corpus_unbounded_list",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    artifact = emit_ssa_to_c(module, exports[0])
    assert not artifact.complete
    assert any(
        "dynamic-growth-unavailable" in shortfall.reason
        for shortfall in artifact.shortfalls
    )


def test_guard_nested_in_a_loop_arm_is_placed_inside_that_arm():
    """``if i > 20 or not allow: raise`` inside ``if x == 6:`` in a loop.

    The loop composer placed the inner guard at the loop-body level after the
    outer conditional, where ``i > 20`` (computed only inside the arm) does not
    dominate it: audit ``use-not-dominated`` (``_validate_row``'s ``if item is
    NEW:`` arm in the bootstrap).  The guard belongs inside its arm.  The C
    module lane has no validation runtime, so this is checked by the audit.
    """
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable

    source = (
        "def root(n: int, allow: bool) -> int:\n"
        "    out = 0\n"
        "    for i in range(n):\n"
        "        x = (i * 3) % 7\n"
        "        if x == 6:\n"
        "            if i > 20 or not allow:\n"
        "                raise ValueError('bad')\n"
        "            out += 100\n"
        "            continue\n"
        "        if x > 10:\n"
        "            raise ValueError('never')\n"
        "        out += x\n"
        "    return out\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="corpus_nested_loop_guard",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    assert CorrelationTable.build(module).findings(module) == []
