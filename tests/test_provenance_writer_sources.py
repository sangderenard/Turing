"""Real source producers must leave no migration worklist behind."""
from src.compiler.identity_concordance import CorrelationTable
from tools.audit_identity_concordance import _lower_mapping


def test_mapping_loop_producers_have_complete_sources():
    module = _lower_mapping()
    table = CorrelationTable.build(module)
    assert not table.findings(module)
    groups, identities = table._unsourced_worklist(module)
    assert not groups and not identities
