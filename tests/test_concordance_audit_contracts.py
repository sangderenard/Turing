"""Audit overlays preserve the program's already-declared helper ABI."""
from pathlib import Path
import runpy
import pytest
from src.compiler.identity_concordance import concordance_report
from src.compiler.identity_concordance import CorrelationTable


@pytest.mark.parametrize('declare_root_types', [False, True])
def test_controller_audit_preserves_existing_restore_type_contract(declare_root_types):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/audit_identity_concordance.py'))
    module = namespace['_lower_controller'](declare_root_types=declare_root_types)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    # The whole actual controller was still included, not replaced by a stub.
    assert any('pi_update' in name for name in module.functions)
    assert any('update_dt_max' in name for name in module.functions)


@pytest.mark.parametrize('case', ['energy', 'toplevel'])
def test_audit_has_no_undefined_values_or_unposted_minted_ids(case):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/audit_identity_concordance.py'))
    module = namespace['_lower_' + case]()
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    _, identities = CorrelationTable.build(module)._unsourced_worklist(module)
    assert identities == [], report
    if case == 'toplevel':
        assert any('distribution_override' in name for name in module.functions)
        assert any('_energy_time_limit' in name for name in module.functions)
        root = next(function for name, function in module.functions.items() if name.endswith('__root'))
        assert any(instruction.op == 'CondBr' for block in root.blocks.values()
                   for instruction in block.instrs)
