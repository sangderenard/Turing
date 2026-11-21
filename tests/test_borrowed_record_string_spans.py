"""Borrowed record fields preserve independent ragged spans through native equality."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.string_table import string_token
from src.compiler.ssa_storage_requirements import is_compiler_owned_storage

@pytest.mark.parametrize('operator', ['==', '!='])
@pytest.mark.parametrize('layout', ['direct', 'nested', 'repeated'])
def test_record_string_fields_native(tmp_path, operator, layout):
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    source = f'''from dataclasses import dataclass
@dataclass
class Node:
    left: str
    right: str

def root(nodes: list[Node]):
    total = 0
    for node in nodes:
        if node.left {operator} node.right:
            total += 1
    return total
'''
    if layout == 'nested':
        source = source.replace('class Node:', 'class Inner:').replace('def root(', '@dataclass\nclass Node:\n    content: Inner\n\ndef root(')
        source = source.replace('node.left', 'node.content.left').replace('node.right', 'node.content.right')
    if layout == 'repeated':
        source = source.replace('            total += 1', f'            if node.left {operator} node.right:\n                total += 1')
    field_prefix = 'Node.content.' if layout == 'nested' else 'Node.'
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn = module.functions[exports[0]]
    seq = module.sequence_tables[exports[0]].sequences[dict(fn.metadata['parameter_names'])['nodes']]
    fields = {arg.accounting['program_abi_field']: arg for arg in fn.args if 'program_abi_field' in (arg.accounting or {})}
    assert len(fields) == 8
    assert all(not is_compiler_owned_storage(v) for v in fields.values())
    records = module.record_tables[exports[0]].records
    assert records
    assert all(record.instance_pool is None for record in records.values())
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / 'llvm')
    out = next(i.args[0].id for b in fn.blocks.values() for i in b.instrs if i.op == 'Ret' and not i.attributes.get('fault_exit'))
    for left, right, lh, rh in [
        (['ab', 'x', '', 'λ'], ['ac', 'x', '', 'β'], [0,1,2,3], [0,1,2,3]),
        (['master', '', 'λ'], ['λ', 'mast', ''], [2,0,1], [0,1,2]),
        (['a', 'bb', 'ccc'], ['bb', 'a', 'ccc', 'unused'], [1,0,2,0], [0,1,2,0]),
        ([''], [''], [0], [0]),
        ([], [], [], []),
    ]:
        feeds = {seq.column_value_ids[0]: np.array(lh, dtype=np.int64), seq.column_value_ids[1]: np.array(rh, dtype=np.int64),
                 seq.length_address_id: np.array([len(lh)], dtype=np.int64), seq.capacity_value_id: np.array([len(lh)], dtype=np.int64),
                 seq.status_address_id: np.array([0], dtype=np.int64)}
        for name, words in [('left', left), ('right', right)]:
            stride = max([1, *map(len, words)])
            values = {'column': [string_token(word[i]) if i<len(word) else 0 for word in words for i in range(stride)],
                      'lengths': list(map(len, words)), 'row_stride': [stride], 'capacity': [len(words)]}
            for member, data in values.items():
                feeds[fields[f'{field_prefix}{name}.{member}'].id] = np.array(data, dtype=np.int64)
        before = {key:value.copy() for key,value in feeds.items()}
        expected = sum((left[i] == right[j]) if operator=='==' else (left[i] != right[j]) for i,j in zip(lh,rh))
        for _ in range(2):
            result = prepare_artifact_execution(artifact, feeds).run()
            assert int(result.buffers[out].reshape(-1)[0]) == expected
            for key in feeds:
                np.testing.assert_array_equal(feeds[key], before[key])
