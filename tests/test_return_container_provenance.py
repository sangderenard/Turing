import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


@pytest.mark.parametrize('expression,kind', [('value', 'value'), ('(value,)', 'tuple'), ('[value]', 'list')])
def test_single_return_slot_retains_authored_container(expression, kind):
    graphs = []
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [{
            'function': 'root', 'parameter': 'value', 'storage': 'scalar',
            'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float',
        }],
    })
    lower_ast_source_to_ssa(
        f'def root(value):\n    return {expression}\n', 'root',
        extraction_contract=policy, resolved_process_graph_sink=graphs.append,
        stop_after_compilation_unit_plan=True,
    )
    root = next(entry.graph.G for entry in graphs[0].function_table
                if entry.graph.G.graph.get('function_name') == 'root')
    slots = root.graph['return_slot_values']
    kinds = root.graph['return_container_kinds']
    assert len(slots) == 1
    assert tuple(kinds) == tuple(slots)
    assert tuple(kinds.values()) == (kind,)
    assert len(next(iter(slots.values()))) == 1
