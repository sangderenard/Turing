"""Set-to-list constructors own storage and use source iteration semantics."""
from src.compiler.identity_concordance import CorrelationTable
from tests.test_rebound_empty_list import _lowered, _python, _run_llvm, CASES

SOURCE = """
def root(seed, n):
    keys = {j % 3 for j in range(n)}
    values = list(keys)
    values.append(seed)
    return len(keys) * 100 + len(values)
"""

REPEAT = """
def root(seed, n):
    keys = {j % 3 for j in range(n)}
    total = 0
    for k in range(3):
        values = list(keys)
        values.append(seed)
        total += len(values)
    return len(keys) * 100 + total
"""


def test_set_snapshot_has_distinct_concordant_storage():
    module, export, *_ = _lowered(SOURCE)
    assert CorrelationTable.build(module).findings(module) == []
    descriptors = tuple(module.sequence_tables[export].sequences.values())
    keyed_columns = {column for d in descriptors if d.key_columns for column in d.column_value_ids}
    list_columns = {column for d in descriptors if not d.key_columns for column in d.column_value_ids}
    assert keyed_columns and list_columns
    assert keyed_columns.isdisjoint(list_columns)


def test_set_snapshot_matches_python(tmp_path):
    lowered = _lowered(SOURCE)
    for seed, n in CASES:
        assert _run_llvm(tmp_path, lowered, seed, n) == _python(SOURCE, seed, n)


def test_repeated_set_snapshot_matches_python(tmp_path):
    lowered = _lowered(REPEAT)
    for seed, n in CASES:
        assert _run_llvm(tmp_path, lowered, seed, n) == _python(REPEAT, seed, n)
