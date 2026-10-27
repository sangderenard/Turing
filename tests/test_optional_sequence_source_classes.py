"""Optional containers retain element facts without claiming presence."""
from pathlib import Path
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import current_identity_book
from src.common.tensors.topological_reducer import _source_numeric_scope

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('annotation,expected', [
    ('tuple[int, int] | None', 'builtins.int'),
    ('tuple[int, float] | None', None),
])
def test_optional_tuple_present_projection_and_merge_class(annotation, expected):
    source = f'''def optional_tuple(fingerprint: {annotation} = None) -> int:
    if fingerprint is None:
        hi, lo = (0, 0)
    else:
        hi, lo = fingerprint
    return hi + lo
'''
    captured = []
    def inspect(graph):
        entry = next(item for item in graph.function_table if item.name == 'optional_tuple')
        g = entry.graph.G
        scope = _source_numeric_scope(entry.graph)
        page = current_identity_book().page('source_value_class_concordance')
        for node, data in g.nodes(data=True):
            attrs = data.get('attributes') or {}
            if attrs.get('binding_name') == 'fingerprint':
                assert attrs['aggregate_kind'] == 'tuple'
                assert attrs.get('sequence_element_class_ref') == expected
                # Optionality is not erased by an always-tuple class claim.
                assert page.latest((scope, node)) is None
            if data.get('type') in {'Indexed', 'Phi', 'Add'}:
                fact = page.latest((scope, node))
                captured.append((data['type'], None if fact is None else fact[0]))
    lower_ast_source_to_ssa(
        source, 'optional_tuple', name='optional_tuple_class',
        extraction_contract=ROOT / 'extraction_contracts/program_extraction.yaml',
        resolved_process_graph_sink=inspect, stop_after_compilation_unit_plan=True,
        progress=lambda _: None, progress_bars=False,
    )
    assert {kind for kind, _ in captured} == {'Indexed', 'Phi', 'Add'}
    assert all(identity == expected for _, identity in captured)


def test_none_parameter_is_not_an_aggregate():
    def inspect(graph):
        entry = next(item for item in graph.function_table if item.name == 'none_value')
        node = next(data for _, data in entry.graph.G.nodes(data=True)
                    if (data.get('attributes') or {}).get('binding_name') == 'value')
        assert 'aggregate_kind' not in node['attributes']
        assert 'sequence_element_class_ref' not in node['attributes']
    lower_ast_source_to_ssa(
        'def none_value(value: None) -> bool:\n    return value is None\n', 'none_value',
        name='none_source_class',
        extraction_contract=ROOT / 'extraction_contracts/program_extraction.yaml',
        resolved_process_graph_sink=inspect, stop_after_compilation_unit_plan=True,
        progress=lambda _: None, progress_bars=False,
    )
