"""The audit mapping receives its real host type contract, without a default."""
import pytest
from src.compiler.called_inventory import MappingClass
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

SOURCE = '''def root(channels):
    total = 0.0
    for name, limit in channels.items():
        total = total + limit
    return total
'''


def test_identical_multi_parameter_mapping_contract_reaches_loop_value_type():
    module, _, _ = lower_ast_source_to_ssa(
        SOURCE, 'root', root_calls=[{'channels': MappingClass(str, float)}],
        extraction_contract='extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False)
    adds = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs if i.op == 'Add' and i.attributes.get('source_type') == 'BinOp']
    assert adds and all(i.res.dtype == 'float64' for i in adds)
    assert '0 finding(s)' in concordance_report(module)


def test_mapping_without_host_or_authored_value_contract_is_not_guessed():
    with pytest.raises(ValueError, match='no settled dtype'):
        lower_ast_source_to_ssa(SOURCE, 'root', extraction_contract='extraction_contracts/program_extraction.yaml',
                                progress=lambda _: None, progress_bars=False)
