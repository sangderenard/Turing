"""A handle's physical scalar layout is not its object's runtime type."""
import ast
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.compiler.glsl_deployment_strategy import _fold_callsite_structural_values


def kind_graph(target='tuple', *, declared=None, value_id=0, reference=True):
    process = nx.DiGraph()
    process.graph['function_name'] = 'linked_kind'
    process.graph['identity_table'] = {'item': (0,)}
    receipt = {'storage': 'reference' if reference else 'scalar',
               'dtype': 'opaque_ref' if reference else 'int64', 'rank': 0, 'shape': ()}
    if declared is not None:
        receipt['python_type'] = declared
    process.graph['linked_value_abi'] = {value_id: receipt}
    process.add_node(0, type='Input', op='input', value_id=value_id, parents=(),
        attributes={'binding_name': 'item', 'binding_kind': 'parameter'},
        tensor={'shape': (), 'dtype': receipt['dtype']})
    process.add_node(1, type='Call', op='Call', parents=((0, 'arg:0'),),
        attributes={}, expr_obj=ast.parse(f'isinstance(item,{target})', mode='eval').body)
    process.add_edge(0, 1, role='arg:0')
    return SimpleNamespace(G=process, roots=[1])


@pytest.mark.parametrize('target', ['dict', 'list', 'tuple', 'set', 'frozenset'])
def test_reference_kind_remains_runtime_not_a_false_scalar_test(target):
    graph = kind_graph(target)
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['type'] == 'Call'
    assert tuple(graph.G.nodes[1]['parents']) == ((0, 'arg:0'),)
    assert graph.G.has_edge(0, 1)


def test_linked_reference_fact_uses_value_identity_not_node_key():
    graph = kind_graph(value_id=103)
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['type'] == 'Call'


def test_declared_positive_reference_kind_still_folds():
    graph = kind_graph(declared='builtins.tuple')
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['attributes']['value'] is True


def test_declared_base_reference_does_not_prove_negative_subtype():
    graph = kind_graph(declared='builtins.object')
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['type'] == 'Call'


def test_scalar_integer_type_evidence_is_not_weakened():
    graph = kind_graph(reference=False)
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['attributes']['value'] is False


def test_reference_contract_overrules_weaker_tensor_type_hint():
    graph = kind_graph()
    graph.G.nodes[0]['tensor']['python_type'] = 'builtins.int'
    _fold_callsite_structural_values(graph)
    assert graph.G.nodes[1]['type'] == 'Call'


def test_forwarded_reference_identity_still_executes_natively(tmp_path):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    root = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(root / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': 'value',
        'storage': 'reference', 'python_type': 'builtins.object'}],
    }).with_execution_file(root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(
        'def identity(item):\n    return id(item)\ndef root(value):\n    return identity(value)\n',
        'root', extraction_contract=policy, progress=lambda _: None, progress_bars=False)
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    function = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path)
    parameter = dict(function.metadata['parameter_names'])['value']
    output = next(i.args[0].id for b in function.blocks.values() for i in b.instrs if i.op == 'Ret')
    for handle in (1, 17, 2**60 + 3):
        execution = prepare_artifact_execution(native, {parameter: np.array([handle], dtype=np.int64)}).run()
        assert execution.buffers[output].item() == handle


def test_recursive_collection_edges_survive_abi_settlement(monkeypatch):
    from tests.test_source_object_context import context
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import FortranEmissionError
    import src.compiler.ssa_object_context as objects
    root = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(root / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'root', 'parameter': 'value', 'storage': 'reference', 'python_type': 'builtins.object'},
            {'function': 'root', 'parameter': 'memo', 'storage': 'keyed', 'dtype': 'opaque_ref',
             'mutable': True, 'key_encoding': 'integer_identity', 'python_type': 'builtins.dict'},
        ],
    }).with_execution_file(root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    observations = []
    original = objects.source_object_collections
    def observe(graph, admitted):
        result = original(graph, admitted)
        if graph.graph.get('function_name') == 'visit':
            observations.append((dict(result), {
                node: tuple(data.get('parents') or ()) for node, data in graph.nodes(data=True)
                if (data.get('attributes') or {}).get('result_kind') == 'collection'
            }))
        return result
    monkeypatch.setattr(objects, 'source_object_collections', observe)
    source = ('def root(value,memo):\n    def visit(item):\n'
              '        identity=id(item)\n        if identity in memo:\n            return memo[identity]\n'
              '        if isinstance(item,tuple):\n'
              '            result=tuple(visit(child) for child in item)\n'
              '            memo[identity]=result\n            return result\n'
              '        return item\n    return visit(value)\n')
    try:
        lower_ast_source_to_ssa(source, 'root', extraction_contract=policy,
            native_object_context=context(), runtime_closure_only=True,
            progress=lambda _: None, progress_bars=False)
    except FortranEmissionError as error:
        # This test proves graph preservation, NOT recursive native execution.
        # Exact control ownership now materializes the recursive child. The
        # unresolved runtime kind predicate must remain a loud refusal.
        assert 'recursive invocation storage' not in str(error)
        assert "(10, 'call', 'call-result-unavailable')" in str(error)
    receipts, parents = observations[-1]
    assert len(receipts) == 1
    for node in receipts:
        assert {role for _, role in parents[node]} == {'value', 'control'}
