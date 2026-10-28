import pathlib

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_faults import NativeIndexError, NativeKeyError
from src.compiler.ssa_c_backend import emit_ssa_to_c

CONTRACT = "extraction_contracts/program_extraction.yaml"

LITERAL_ROWS = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2], 2: [3, 4]}\n"
    "    return d[2][1] + d[1][0] + len(d[1])\n"
)
RUNTIME_ROWS = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2, 3], 2: [4, n]}\n"
    "    return d[2][1] + len(d[1]) * 10 + d[1][2]\n"
)
RUNTIME_KEY = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2], 2: [3, n]}\n"
    "    return d[n - 2][0] + d[n - 3][1]\n"
)
RUNTIME_POSITION = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2], 2: [3, n]}\n"
    "    return d[1][n]\n"
)
ITERATE = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2], 2: [3, n]}\n"
    "    t = 0\n"
    "    for k in d:\n"
    "        for x in d[k]:\n"
    "            t += x * k\n"
    "    return t\n"
)
SUM_ROWS = (
    "def root(n: int) -> int:\n"
    "    d = {1: [1, 2, 3], 2: [4, n]}\n"
    "    return sum(d[1]) + sum(d[2])\n"
)
FLOAT_ROWS = (
    "def root(n: int) -> float:\n"
    "    d = {1: [1.5, 2.5], 2: [3.5, n * 0.5]}\n"
    "    return d[2][1] + d[1][0]\n"
)
EMPTY_ROW = (
    "def root(n: int) -> int:\n"
    "    d = {1: [], 2: [n]}\n"
    "    return len(d[1]) * 10 + len(d[2])\n"
)

CASES = [
    (LITERAL_ROWS, (0, 5)),
    (RUNTIME_ROWS, (0, 4, -3)),
    (RUNTIME_KEY, (3, 4, 5)),
    (RUNTIME_POSITION, (0, 1, 2, -1, -3)),
    (ITERATE, (0, 4)),
    (SUM_ROWS, (0, 4)),
    (FLOAT_ROWS, (0, 3)),
    (EMPTY_ROW, (0, 4)),
]


@pytest.mark.parametrize("source,inputs", CASES)
def test_dict_of_lists_matches_python(tmp_path, source, inputs):
    """A dict display of list displays keeps its rows in the child-table
    pool: reads, lengths, iteration and sums give Python's answer, and an
    absent key or out-of-range position raises KeyError or IndexError."""
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="dict_of_lists", extraction_contract=CONTRACT,
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
        feeds = {parameters[k]: np.array([n]) for k in parameters}
        try:
            expected = namespace["root"](n)
        except (IndexError, KeyError) as error:
            with pytest.raises(
                NativeIndexError if isinstance(error, IndexError)
                else NativeKeyError
            ):
                artifact.prepare_execution(feeds).run()
            continue
        execution = artifact.prepare_execution(feeds).run()
        assert execution.buffers[outputs[0].id].tolist() == [expected], n
