"""A destructuring loop target takes its class from the annotation's position."""
from pathlib import Path
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import current_identity_book
from src.common.tensors.topological_reducer import _source_numeric_scope

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('annotation,loop,expected', [
    ('list[tuple[int, float]]', 'a, b in ps', {'a': 'builtins.int', 'b': 'builtins.float'}),
    ('list[tuple[int, int]]', 'a, b in ps', {'a': 'builtins.int', 'b': 'builtins.int'}),
    ('dict[int, float]', 'a, b in ps.items()', {'a': 'builtins.int', 'b': 'builtins.float'}),
    ('Mapping[int, int]', 'a, b in ps.items()', {'a': 'builtins.int', 'b': 'builtins.int'}),
])
def test_loop_targets_take_annotated_position_class(annotation, loop, expected):
    source = f'''from typing import Mapping


def walk(ps: {annotation}) -> int:
    t = 0
    for {loop}:
        t += 1
    return t
'''
    seen = {}

    def inspect(graph):
        entry = next(item for item in graph.function_table if item.name == 'walk')
        g = entry.graph.G
        scope = _source_numeric_scope(entry.graph)
        page = current_identity_book().page('source_value_class_concordance')
        for node, data in g.nodes(data=True):
            attrs = data.get('attributes') or {}
            if attrs.get('binding_kind') == 'loop':
                fact = page.latest((scope, node))
                seen[attrs['binding_name']] = (
                    attrs.get('result_class_ref'), None if fact is None else fact[0],
                )

    lower_ast_source_to_ssa(
        source, 'walk', name='destructured_loop_target_class',
        extraction_contract=ROOT / 'extraction_contracts/program_extraction.yaml',
        resolved_process_graph_sink=inspect, stop_after_compilation_unit_plan=True,
        progress=lambda _: None, progress_bars=False,
    )
    assert seen == {name: (cls, cls) for name, cls in expected.items()}
