"""A comprehension continuation retargets the Phi's cached initial identity."""
from src.compiler.loop_composer import _retarget_cached_value_ids
from tests.test_rebound_empty_list import _lowered, _python, _run_llvm, CASES

SOURCE = '''
def root(seed, n):
    values = [j for j in range(n) if j > seed]
    if not values:
        values = list(range(n))
    return len(values)
'''


def test_retarget_preserves_phi_initial_identity():
    data = {"attributes": {"initial_value_id": 92, "value_source_id": 92}}
    _retarget_cached_value_ids(data, 92, (107,))
    assert data["attributes"] == {"initial_value_id": 107, "value_source_id": 107}


def test_comprehension_rebound_reaches_concordant_ssa():
    from src.compiler.identity_concordance import CorrelationTable
    module, *_ = _lowered(SOURCE)
    assert CorrelationTable.build(module).findings(module) == []


def test_bounded_comprehension_rebound_matches_python(tmp_path):
    source = SOURCE.replace("range(n)", "range(4)").replace("values = list(range(4))", "values = [seed, n]")
    lowered = _lowered(source)
    for seed, n in (*CASES, (10, 2)):
        assert _run_llvm(tmp_path, lowered, seed, n) == _python(source, seed, n)
