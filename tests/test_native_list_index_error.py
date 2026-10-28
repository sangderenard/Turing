import pathlib

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_faults import NativeIndexError
from src.compiler.ssa_c_backend import emit_ssa_to_c

CONTRACT = "extraction_contracts/program_extraction.yaml"

READ = "def root(n: int) -> int:\n    xs = [1, 2]\n    return xs[n]\n"
LITERAL = "def root(n: int) -> int:\n    xs = [1, 2, 3]\n    return xs[5] + n\n"
NESTED = (
    "def root(n: int) -> int:\n"
    "    g = [[1, 2, 3], [4, 5, 6]]\n"
    "    return g[n % 3][n % 4]\n"
)
HEAP = (
    "def root(n: int) -> int:\n"
    "    xs = []\n"
    "    for i in range(n):\n"
    "        xs.append(i * 2)\n"
    "    return xs[n - 1] + xs[-1] + xs[0]\n"
)
STORE = (
    "def root(n: int) -> int:\n"
    "    xs = [1, 2, 3]\n"
    "    xs[n] = 9\n"
    "    return xs[0] + xs[1] + xs[2]\n"
)
FLOAT_STORE = (
    "def root(n: int) -> float:\n"
    "    xs = [1.5, 2.5, 3.5]\n"
    "    xs[n] = 9.0\n"
    "    return xs[0] + xs[1] + xs[2]\n"
)
AUGMENTED = (
    "def root(n: int) -> int:\n"
    "    xs = [1, 2, 3]\n"
    "    xs[n] += 5\n"
    "    return xs[0] + xs[1] + xs[2]\n"
)
LOOP_STORE = (
    "def root(n: int) -> int:\n"
    "    xs = [0, 0, 0, 0]\n"
    "    for i in range(n):\n"
    "        xs[i] = i * 2\n"
    "    return xs[0] + xs[1] + xs[2] + xs[3]\n"
)

CASES = [
    (READ, (0, 1, -1, -2, 2, 5, -3)),
    (LITERAL, (0,)),
    (NESTED, (0, 1, 2, 3, 4, 5)),
    (HEAP, (0, 1, 3)),
    (STORE, (0, 2, -1, 3, 5, -4)),
    (FLOAT_STORE, (0, -1, 3)),
    (AUGMENTED, (0, 2, -1, 3)),
    (LOOP_STORE, (0, 3, 4, 5)),
]


@pytest.mark.parametrize("source,inputs", CASES)
def test_positional_access_matches_python_index_errors(tmp_path, source, inputs):
    """Out-of-range positions raise IndexError natively, never read or write
    past the list; in-range and negative positions give Python's answer."""
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="list_index_error", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    outputs = [
        i.args for b in root.blocks.values() for i in b.instrs if i.op == "Ret"
    ][0]
    artifact = emit_ssa_to_c(module, exports[0])
    artifact.compile(pathlib.Path(tmp_path) / "native")
    namespace: dict = {}
    exec(source, namespace)
    for n in inputs:
        try:
            expected = namespace["root"](n)
        except IndexError:
            with pytest.raises(NativeIndexError):
                artifact.prepare_execution(
                    {parameters[k]: np.array([n]) for k in parameters}
                ).run()
            continue
        execution = artifact.prepare_execution(
            {parameters[k]: np.array([n]) for k in parameters}
        ).run()
        assert execution.buffers[outputs[0].id].tolist() == [expected], n
