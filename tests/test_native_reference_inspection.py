"""Native inspection of existing keyed-record container descriptors."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {
            'Book': {'identity': 'metadata.Book', 'fields': {'objects': {
                'storage': 'keyed', 'dtype': 'int64', 'key_encoding': 'integer_identity',
                'value_record': 'Object',
            }}},
            'Object': {'identity': 'metadata.Object', 'fields': {
                'kind': {'storage': 'scalar', 'dtype': 'int64'},
                'members': {'storage': 'table', 'columns': [
                    {'name': 'key', 'dtype': 'opaque_ref'},
                    {'name': 'value', 'dtype': 'opaque_ref'},
                ]},
            }},
        },
        'bindings': [{'function': 'root', 'parameter': 'book', 'record': 'Book'}],
        'values': [
            {'function': 'root', 'parameter': 'handle', 'storage': 'reference',
             'python_type': 'builtins.object'},
            {'function': 'root', 'parameter': 'index', 'storage': 'scalar',
             'dtype': 'int64', 'python_type': 'builtins.int'},
        ],
    }).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')


@pytest.mark.parametrize('expression', ['len(item.members)', 'item.members[index][0]',
                                      'item.members[index][1]', 'iterate', 'combined'])
def test_native_keyed_record_member_inspection(expression, tmp_path):
    # This checks native traversal/zero-trip behavior. Exact reference payloads
    # are checked separately by both indexed-column cases above. Integer loop
    # accumulators still have a separate planner/result-type shortfall.
    result_expression = ('len(item.members),item.members[index][0],item.members[index][1]'
                         if expression == 'combined' else expression)
    body = ('    total=0.0\n    for key,member in item.members:\n'
            '        total=total+1.0\n    return total\n') if expression == 'iterate' else (
                '    return ' + result_expression + '\n')
    module, _, exports = lower_ast_source_to_ssa(
        'def root(book,handle,index):\n    item=book.objects[handle]\n' + body,
        'root', name='reference_inspection', extraction_contract=policy(),
        runtime_closure_only=True, progress=lambda _: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    from src.compiler.identity_concordance import concordance_report
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'inspect')
    facts = {
        'objects.length': np.array([3], dtype=np.int64),
        'objects.keys': np.array([2**60 + 3, 17, 29], dtype=np.int64),
        'objects.values': np.array([0, 1, 2], dtype=np.int64),
        'objects[].kind.column': np.array([1, 2, 6], dtype=np.int64),
        'objects[].members.lengths': np.array([2, 0, 1], dtype=np.int64),
        'objects[].members.row_stride': np.array([2], dtype=np.int64),
        'objects[].members.columns.key': np.array([2**60 + 7, 11, 0, 0, 13, 0], dtype=np.int64),
        'objects[].members.columns.value': np.array([2**60 + 19, 23, 0, 0, 31, 0], dtype=np.int64),
    }
    buffers = {}
    for value in function.args:
        field = value.accounting.get('program_abi_field')
        if field is not None:
            assert field in facts, (value.id, field)
            buffers[value.id] = facts[field]
    parameters = dict(function.metadata['parameter_names'])
    outputs = next(tuple(a.id for a in i.args) for b in function.blocks.values() for i in b.instrs if i.op == 'Ret')
    cases = [(2**60 + 3, 0, 2, 2**60 + 7, 2**60 + 19),
             (2**60 + 3, 1, 2, 11, 23), (29, 0, 1, 13, 31)]
    if expression.startswith('len') or expression == 'iterate':
        cases.append((17, 0, 0, None, None))
    for handle, index, length, key, member in cases:
        if parameters['handle'] in {v.id for v in function.args}:
            buffers[parameters['handle']] = np.array([handle], dtype=np.int64)
        if parameters.get('index') in {v.id for v in function.args}:
            buffers[parameters['index']] = np.array([index], dtype=np.int64)
        expected = ((length, key, member) if expression == 'combined' else
                    length if expression.startswith('len') else
                    length if expression == 'iterate' else key if expression.endswith('[0]') else member)
        execution = prepare_artifact_execution(native, buffers).run()
        actual = tuple(execution.buffers[output].item() for output in outputs)
        assert (actual if expression == 'combined' else actual[0]) == expected


def test_empty_dictionary_row_layout_comes_from_exact_runtime_store_operands():
    boundary = ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'root', 'parameter': 'key', 'storage': 'scalar',
             'dtype': 'int64', 'python_type': 'builtins.int'},
            {'function': 'root', 'parameter': 'value', 'storage': 'reference',
             'python_type': 'builtins.object'},
        ],
    }).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(
        'def root(key,value):\n    copied={}\n    copied[key]=value\n    return len(copied)\n',
        'root', name='empty_reference_rows', extraction_contract=boundary,
        runtime_closure_only=True, progress=lambda _: None, progress_bars=False,
    )
    descriptors = tuple(module.sequence_tables[exports[0]].sequences.values())
    assert len(descriptors) == 1
    assert descriptors[0].column_dtypes == ('int64', 'opaque_ref')
    assert descriptors[0].key_columns == (0,)
    assert descriptors[0].heap_owner_value_id is None
    # Step3 proves layout, not capacity/allocation/source constructor wiring.
    function = module.functions[exports[0]]
    assert any(i.attributes.get('binding') == 'ssa_sequence_length_value'
               for b in function.blocks.values() for i in b.instrs)


@pytest.mark.parametrize('other_dtype', [None, 'float64'])
def test_empty_mapping_does_not_guess_from_one_of_two_stores(other_dtype):
    import ast
    import networkx as nx
    from src.compiler.fortran_c_shell import _sequence_column_dtype_contracts
    graph = nx.DiGraph()
    graph.graph.update(identity_table={'key': (0,), 'value': (1,), 'other': (5,)},
        parameter_value_abi={'key': {'storage': 'scalar', 'dtype': 'int64'},
                             'value': {'storage': 'reference', 'dtype': 'opaque_ref'},
                             'other': {'storage': 'scalar', 'dtype': other_dtype}})
    graph.add_node(0, type='Input', value_id=0)
    graph.add_node(1, type='Input', value_id=1)
    graph.add_node(2, type='Dict', value_id=2, expr_obj=ast.parse('{}', mode='eval').body,
                   attributes={'aggregate_kind': 'dict'})
    graph.add_node(3, type='IndexedStore', value_id=3,
                   parents=[(2, 'base'), (0, 'index'), (1, 'value')])
    graph.add_node(5, type='Input', value_id=5)
    graph.add_node(6, type='IndexedStore', value_id=6,
                   parents=[(2, 'base'), (0, 'index'), (5, 'value')])
    assert 2 not in _sequence_column_dtype_contracts(graph, ((2, 'unique', 2, True),))


def test_detached_parameter_record_receipt_is_still_the_binding_authority():
    import networkx as nx
    from src.compiler.fortran_c_shell import _graph_record_abi_by_value
    graph = nx.DiGraph()
    record = {'identity': 'metadata.Object', 'fields': {'kind': {'storage': 'scalar', 'dtype': 'int64'}}}
    graph.graph.update(identity_table={'item': (7,)}, parameter_record_abi={'item': record})
    graph.add_node(7, value_id=7, type='Input', attributes={})
    assert _graph_record_abi_by_value(graph) == {7: record}


def test_record_schema_binding_does_not_guess_from_an_unqualified_name():
    import networkx as nx
    from src.compiler.fortran_c_shell import _graph_record_abi_by_value
    graph = nx.DiGraph()
    graph.graph['program_abi'] = {'records': {'Object': {'identity': 'metadata.Object', 'fields': {}}}}
    graph.add_node(7, value_id=7, type='Input', attributes={'program_abi_record_identity': 'Object'})
    with pytest.raises(ValueError, match='no unique ProgramABI schema'):
        _graph_record_abi_by_value(graph)
