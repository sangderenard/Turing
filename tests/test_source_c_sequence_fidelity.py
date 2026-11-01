"""Authored control, sequence storage and call-frame identity through native C."""

import numpy as np
import pytest

from tests.test_source_ssa_synchronization import compile_source
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


CASES = {
    'loop': """\
def step(value: int, index: int) -> int:
    if index % 2 == 0:
        return value + index
    return value - index

def root(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        total = step(total, i)
        i += 1
    return total
""",
    'ring': """\
def root(n: int) -> int:
    rows = [0, 0, 0]
    i = 0
    while i < n:
        rows[i % 3] = i + 1
        i += 1
    return rows[0] * 100 + rows[1] * 10 + rows[2]
""",
    'alias': """\
def update(rows: list[int], index: int, value: int) -> int:
    rows[index] = value
    return rows[index]

def root(n: int) -> int:
    rows = [1, 2, 3]
    alias = rows
    result = update(alias, n % 3, n + 10)
    return rows[n % 3] + result
""",
    'pop': """\
def root(n: int) -> int:
    rows = [1, 2, 3]
    i = 0
    total = 0
    while i < n:
        rows.append(i + 10)
        total += rows.pop(0)
        i += 1
    return total * 100 + rows[-1]
""",
    'returned_alias': """\
def make(value: int) -> list[int]:
    return [value, value + 1, value + 2]

def change(rows: list[int], value: int) -> int:
    rows[-1] = value
    return rows[-1]

def root(n: int) -> int:
    rows = make(n)
    alias = rows
    value = change(alias, 17)
    return rows[-1] + value
""",
    'separate_returns': """\
def make(value: int) -> list[int]:
    return [value, value + 1, value + 2]

def change(rows: list[int], value: int) -> int:
    rows[-1] = value
    return rows[-1]

def root(n: int) -> int:
    first = make(n)
    second = make(n + 100)
    result = change(first, 17)
    return first[-1] + second[-1] * 100 + result
""",
    'returned_growth': """\
def make(value: int) -> list[int]:
    return [value, value + 1, value + 2]

def grow(rows: list[int], value: int) -> int:
    rows.append(value)
    return len(rows)

def root(n: int) -> int:
    rows = make(n)
    alias = rows
    count = grow(alias, n + 10)
    return len(rows) * 1000 + count * 100 + rows[-1]
""",
}


@pytest.mark.parametrize("name", CASES)
def test_authored_sequences_preserve_control_storage_and_results_in_c(name, tmp_path):
    source = CASES[name]
    reference = {}
    exec(source, reference)
    module = compile_source(source)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
    assert "unsourced: 0 fact(s), 0 identit(ies)" in report, report
    artifact = emit_ssa_module_to_c(module, "root__root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / name)
    input_id = module.functions["root__root"].args[0].id
    output_id, = [value_id for value_id, role in zip(artifact.buffer_order, artifact.buffer_roles)
                  if role == "output"]
    # Reuse the artifact: each invocation must own fresh local container state.
    for n in (0, 1, 2, 5, 8, 17, 0, 2):
        execution = artifact.prepare_execution({input_id: np.array([n], dtype=np.int64)}).run()
        assert execution.buffers[output_id].item() == reference["root"](n), (name, n)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
