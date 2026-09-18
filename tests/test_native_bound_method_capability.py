"""A saved method capability retains its exact receiver through invocation."""
import pickle
import subprocess
import sys
import ast

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_getattr_method_capability_and_saved_invocation(tmp_path):
    source = '''
class State:
    def hint(self):
        return self.limit

def invoke(state):
    hint = getattr(state, 'hint', None)
    if callable(hint):
        return hint()
    return -999.0

def root(state):
    return invoke(state)
'''
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'records': {'State': {'identity': 'State', 'fields': {
            'limit': {'storage': 'scalar', 'dtype': 'float64', 'rank': 0},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'state', 'record': 'State'}],
        'values': [],
    })
    module, outputs, _ = lower_ast_source_to_ssa(source, 'root', name='saved_hint', extraction_contract=policy)
    root = module.functions['saved_hint__root']
    limit = next(value.id for value in root.args if value.accounting.get('program_abi_field') == 'limit')
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, limit, outputs[root.name][0].id)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, limit, output = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({limit: np.array([0.0])})
for value in (0.125, 0.0, -2.0, float('inf'), float('nan')):
    execution.buffers[limit][0] = value
    execution.run()
    actual = execution.buffers[output].item()
    assert actual == value or (np.isnan(actual) and np.isnan(value)), (actual, value)
''', str(payload)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_saved_graph_resolves_method_before_dependency_closure(monkeypatch):
    """An older snapshot must not strand a newly resolved method outside its plan."""
    from src.compiler import fortran_c_shell as shell
    from src.compiler import glsl_deployment_strategy as strategy

    source = '''
class State:
    def hint(self):
        return self.limit
def root(state):
    return state.hint()
'''
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {'State': {'identity': 'State', 'fields': {
            'limit': {'storage': 'scalar', 'dtype': 'float64', 'rank': 0},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'state', 'record': 'State'}],
        'values': [],
    })
    saved = []
    shell.lower_ast_source_to_ssa(source, 'root', extraction_contract=policy,
        stop_after_compilation_unit_plan=True, resolved_process_graph_sink=saved.append)
    graph = saved[0]
    references = set()
    for entry in graph.function_table:
        for _, node in entry.graph.G.nodes(data=True):
            expression = node.get('expr_obj')
            attrs = node.get('attributes') or {}
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute) and expression.func.attr == 'hint':
                reference = attrs.pop('method_ref', None)
                if reference is not None:
                    references.add(int(reference))
                attrs.pop('callee_ref', None)
    assert references

    class PlanInspected(Exception):
        pass

    def inspect_plan(deployment_graph, **kwargs):
        runtime = deployment_graph.G.graph['map_ir']['dependency_regions']['runtime']
        assert references.issubset(set(runtime))
        raise PlanInspected

    monkeypatch.setattr(strategy, 'strategize_shell_deployment', inspect_plan)
    with pytest.raises(PlanInspected):
        shell._lower_resolved_process_graph_deployment(graph, 'root')
