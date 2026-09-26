"""Loop reads resolved by binding, checked by native execution.

Each program holds one value under two bindings around a loop, and each
miscompiled or was refused before its reads were resolved through the
identity book by the binding they read:

* ``if-alias-carried`` -- two carried bindings seeded from one value
  (``second = value; third = value``).  The lowering rebound the shared
  initial id to the first carried entry's header Phi only, and a direct
  ``if`` predicate leaf read that id, so ``third > 2.0`` tested ``second``
  (silent wrong answer).
* ``region-shared-feed`` -- ``second = value`` then ``second = second +
  value``: one captured value read as two bindings by one region, only one
  of them carried (refused: a region formal per value id cannot carry both).
* ``if-region-shared-feed`` -- the same shape under a conditional.
* ``while-bare-carried-test`` -- ``while go:`` with ``go`` rebound in the
  body.  The test was treated as overwritten-before-read and the latch
  re-ran the pre-loop condition region, testing the initial ``go`` forever
  (non-terminating native code).

The Python materializer reconstructs only the five-block counted loop, so
these run through the C backend; execution is in a subprocess under a time
bound so a non-terminating loop fails instead of hanging the suite.
"""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

_PROGRAMS = {
    "if-alias-carried": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, count):\n"
        "    second = value\n"
        "    third = value\n"
        "    for _ in range(count):\n"
        "        if third > 2.0:\n"
        "            second = second * 2.0\n"
        "        third = third + 1.0\n"
        "    return second\n",
        ((1.0, 3), (2.5, 2), (3.0, 0)),
    ),
    "region-shared-feed": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    second = value\n"
        "    while second < limit:\n"
        "        second = second + value\n"
        "    return second\n",
        ((1.0, 3.5), (2.0, 7.0), (5.0, 1.0)),
    ),
    "if-region-shared-feed": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, count):\n"
        "    second = value\n"
        "    for _ in range(count):\n"
        "        if value < second:\n"
        "            second = second + 1.0\n"
        "        else:\n"
        "            second = second + value\n"
        "    return second\n",
        ((1.0, 3), (2.0, 2), (1.5, 0)),
    ),
    "while-bare-carried-test": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    second = value\n"
        "    go = value < limit\n"
        "    while go:\n"
        "        second = second + 1.0\n"
        "        go = second < limit\n"
        "    return second\n",
        ((1.0, 4.0), (5.0, 2.0), (0.5, 3.0)),
    ),
}

_RUNNER = '''
import pickle, sys
import numpy as np
with open(sys.argv[1], "rb") as stream:
    artifact, feeds_by_probe, result_id = pickle.load(stream)
produced = []
for feeds in feeds_by_probe:
    result = artifact.prepare_execution({
        value_id: np.array([value], dtype=dtype)
        for value_id, (value, dtype) in feeds.items()
    }).run()
    produced.append(float(np.asarray(result.buffers[result_id]).reshape(-1)[0]))
print(repr(produced))
'''


@pytest.mark.parametrize("label", sorted(_PROGRAMS))
def test_loop_reads_compute_the_authored_answer_natively(label, tmp_path):
    source, probes = _PROGRAMS[label]
    namespace: dict = {}
    exec(compile(source, "<authored>", "exec"), namespace)
    authored = namespace["train"]
    prefix = "bindread_" + label.replace("-", "_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, "train", name=prefix, extraction_contract=CONTRACT,
        )
    root = module.functions[f"{prefix}__train"]
    parameters = dict(root.metadata["parameter_names"])
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / prefix)
    names = list(inspect.signature(authored).parameters)
    feeds_by_probe = [
        {
            parameters[name]: (
                value, "float64" if isinstance(value, float) else "int64",
            )
            for name, value in zip(names, probe)
            if name in parameters
        }
        for probe in probes
    ]
    payload = tmp_path / "artifact.pkl"
    payload.write_bytes(pickle.dumps(
        (artifact, feeds_by_probe, outputs[root.name][0].id)
    ))
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(payload)],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    produced = eval(completed.stdout.strip().splitlines()[-1])
    expected = [float(authored(*probe)) for probe in probes]
    assert produced == pytest.approx(expected, abs=1e-12)
