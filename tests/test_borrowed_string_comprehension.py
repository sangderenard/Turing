"""Source-only admission of exact local retained string comprehensions."""
from pathlib import Path
from dataclasses import replace

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.precompile_to_ssa import _ControlSSABuilder
from src.compiler.source_borrowed_record_spans import borrowed_record_string_layouts_from_book
from tests.owned_record_test_support import lower_before_owned_record_boundary


SOURCE = '''from dataclasses import dataclass
@dataclass
class Node:
    key: str
    layer: str
def root(nodes: list[Node]) -> int:
    out: list[str] = [node.key for node in nodes FILTER]
    return len(out)
'''


@pytest.mark.parametrize('condition', ['', 'if node.layer == "master"'])
def test_exact_comprehension_publication_reserves_child_layout(monkeypatch, condition):
    original = _ControlSSABuilder._lower_sequence_mutation_body
    checked = []

    def inspect(builder, mutation, *args, **kwargs):
        if mutation.operator == 'append' and mutation.object_contract_cell is not None:
            book = builder._book()
            sid = mutation.sequence_value_id
            declarations = [(sid, 'duplicates', 1, True)]
            survey = lambda: borrowed_record_string_layouts_from_book(
                book, builder.lexical_read_scope, declarations)
            assert sid in survey()
            publication = mutation.object_contract_cell
            page = book.page(publication.page)
            key = (publication.row, publication.column)
            original_fact = page.cells[key]
            try:
                for field in ('projected_argument_value_ids', 'source_argument_value_ids'):
                    page.cells[key] = tuple((name, (999999,) if name == field else value)
                                            for name, value in original_fact)
                    assert sid not in survey()
                page.cells[key] = original_fact + (('unexplained_argument', (999999,)),)
                assert sid not in survey()
            finally:
                page.cells[key] = original_fact
            assert sid in survey()
            from src.compiler.concordance_declarations import CANONICAL_VALUE
            from src.compiler.source_sequence_lifetime import _local_birth
            target = book.latest_ref(CANONICAL_VALUE, (builder.lexical_read_scope, sid))
            control, _, _, _ = _local_birth(book, target)
            control_page = book.page(control.page)
            control_key = (control.row, control.column)
            control_fact = control_page.cells[control_key]
            recorded, = control_fact.sequence_mutations
            try:
                for slot, value in ((3, recorded[2][0]), (4, 'unique'),
                                    (5, 'mapping_items'), (6, None)):
                    altered = list(recorded)
                    altered[slot] = value
                    control_page.cells[control_key] = replace(
                        control_fact, sequence_mutations=(tuple(altered),))
                    assert sid not in survey()
            finally:
                control_page.cells[control_key] = control_fact
            assert sid in survey()
            assert builder.sequence_descriptors[sid].child_table_pool is not None
            assert (mutation.predicate_expression is not None) == bool(condition)
            checked.append(sid)
        return original(builder, mutation, *args, **kwargs)

    monkeypatch.setattr(_ControlSSABuilder, '_lower_sequence_mutation_body', inspect)
    module, _, exports = lower_before_owned_record_boundary(
        SOURCE.replace('FILTER', condition), 'root',
        extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False,
    )
    assert checked
    calls = [op for block in module.functions[exports[0]].blocks.values()
             for op in block.instrs if op.op == 'Call'
             and op.attributes.get('ssa_sequence_operation') == 'append_child_copy']
    assert len(calls) == 1
    assert calls[0].attributes['sequence_id'] in checked


@pytest.mark.parametrize('backend', ['llvm', 'c'])
def test_filtered_comprehension_native_contents(tmp_path, backend):
    import numpy as np
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.source_sequence_child_lifetime import validated_sequence_child_storage
    from src.compiler.string_table import string_token
    source = SOURCE.replace('FILTER', 'if node.layer == "master"').replace(
        'return len(out)',
        "total = 0\n    for word in out:\n        total += len(word)\n"
        "        if word == '\u00e9':\n            total += 100\n    return total")
    module, _, exports = lower_before_owned_record_boundary(
        source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    function = module.functions[name]
    tables = module.sequence_tables[name].sequences
    owned = [sid for sid, d in tables.items() if d.child_table_pool is not None
             and d.heap_owner_value_id is not None]
    assert len(owned) == 1 and validated_sequence_child_storage(module, name, owned[0])
    assert CorrelationTable.build(module).findings(module) == []
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        emitted = emit_ssa_function_to_llvm(module, name)
        assert emitted.shortfalls == ()
        artifact = compile_artifact(emitted, directory=tmp_path / 'llvm')
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'c')
        prepare = artifact.prepare_execution
    root = tables[dict(function.metadata['parameter_names'])['nodes']]
    assert len(root.column_value_ids) == 2
    fields = {arg.accounting['program_abi_field']: arg.id for arg in function.args
              if 'program_abi_field' in (arg.accounting or {})}
    result = next(op.args[0].id for block in function.blocks.values() for op in block.instrs
                  if op.op == 'Ret' and not op.attributes.get('fault_exit'))
    for rows in ([], [('', 'master')], [('\u00e9', 'other'), ('excluded', 'masterx')],
                 [('\u00e9', 'master'), ('excluded'*20, 'other'), ('', 'master'), ('\u03bb', 'master')],
                 [('x'*129, 'master'), ('\u00e9', 'master'), ('\u00e9', 'other')]):
        feeds = {column: np.arange(len(rows), dtype=np.int64) for column in root.column_value_ids}
        feeds.update({root.length_address_id: np.array([len(rows)], dtype=np.int64),
                      root.capacity_value_id: np.array([len(rows)], dtype=np.int64),
                      root.status_address_id: np.array([0], dtype=np.int64)})
        for index, field in enumerate(('key', 'layer')):
            words = [row[index] for row in rows]
            stride = max([1, *map(len, words)])
            values = {'column': [string_token(word[i]) if i < len(word) else 0
                                 for word in words for i in range(stride)],
                      'lengths': list(map(len, words)), 'row_stride': [stride], 'capacity': [len(words)]}
            feeds.update({fields['Node.' + field + '.' + role]: np.array(data, dtype=np.int64)
                          for role, data in values.items()})
        selected = [key for key, layer in rows if layer == 'master']
        for _ in range(2):
            execution = prepare(feeds).run()
            assert int(execution.buffers[result].reshape(-1)[0]) == sum(map(len, selected)) + 100 * selected.count('\u00e9')
