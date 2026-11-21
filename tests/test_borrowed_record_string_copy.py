from dataclasses import replace
from pathlib import Path
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.precompile_to_ssa import _ControlSSABuilder, SSALoweringIncompleteError
from src.compiler.source_borrowed_record_spans import borrowed_record_string_view, borrowed_record_string_source
from tests.owned_record_test_support import lower_before_owned_record_boundary


SOURCE = '''from dataclasses import dataclass
@dataclass
class Node:
    key: str
def root(nodes: list[Node]) -> int:
    out = []
    for node in nodes:
        out.append(node.key)
    return len(out)
'''


def test_borrowed_record_string_has_child_copy_and_exact_view(monkeypatch):
    original = _ControlSSABuilder._lower_sequence_mutation_body
    checked = []

    def inspect(builder, mutation, *args, **kwargs):
        if mutation.operator == 'append':
            sid, = mutation.argument_value_ids
            assert borrowed_record_string_view(builder, sid)
            view = builder.sequence_descriptors[sid]
            pool, handle = builder.child_table_selections[sid]
            pool_key = next(key for key, value in builder.borrowed_record_field_pools.items()
                            if value == pool)
            for altered in (replace(view, column_dtypes=('float64',)),
                            replace(view, writable=True),
                            replace(view, key_columns=(0,)),
                            replace(view, column_shapes=((2,),))):
                builder.sequence_descriptors[sid] = altered
                assert borrowed_record_string_view(builder, sid) is None
            builder.sequence_descriptors[sid] = view
            for altered in (replace(pool, writable=True),
                            replace(pool, column_dtypes=('float64',)),
                            replace(pool, key_columns=(0,)),
                            replace(pool, status_value_id=pool.capacity_value_id)):
                builder.borrowed_record_field_pools[pool_key] = altered
                builder.child_table_selections[sid] = (altered, handle)
                assert borrowed_record_string_view(builder, sid) is None
            builder.borrowed_record_field_pools[pool_key] = pool
            builder.child_table_selections[sid] = (pool, handle)
            parent = builder.sequence_descriptors[pool_key[0]]
            builder.sequence_descriptors[pool_key[0]] = replace(parent, column_value_ids=view.column_value_ids)
            assert borrowed_record_string_view(builder, sid) is None
            builder.sequence_descriptors[pool_key[0]] = parent
            book = builder._book()
            evidence = borrowed_record_string_source(book, builder.lexical_read_scope, sid)
            control, _, _, _, projection, classes = evidence
            page = book.page(control.page)
            key = (control.row, control.column)
            original_fact = page.cells[key]
            projected = original_fact.projected_iterable_bindings
            for bad_projection in (projection[:3] + ('Node.wrong',) + projection[4:],
                                   projection[:5] + (999999, projection[6])):
                page.cells[key] = replace(original_fact, projected_iterable_bindings=tuple(
                    (*entry[:3], bad_projection) if entry[3] == projection else entry
                    for entry in projected))
                assert borrowed_record_string_view(builder, sid) is None
            page.cells[key] = original_fact
            class_cell = classes[0]
            class_page = book.page(class_cell.page)
            class_key = (class_cell.row, class_cell.column)
            class_fact = class_page.cells[class_key]
            class_page.cells[class_key] = ('builtins.int', *class_fact[1:])
            assert borrowed_record_string_view(builder, sid) is None
            class_page.cells[class_key] = class_fact
            assert borrowed_record_string_view(builder, sid)
            checked.append(sid)
        return original(builder, mutation, *args, **kwargs)

    monkeypatch.setattr(_ControlSSABuilder, '_lower_sequence_mutation_body', inspect)
    module, _, exports = lower_before_owned_record_boundary(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False,
    )
    assert checked
    function = module.functions[exports[0]]
    calls = [op for block in function.blocks.values() for op in block.instrs
             if op.op == 'Call' and op.attributes.get('ssa_sequence_operation')]
    copied = [op for op in calls if op.attributes['ssa_sequence_operation'] == 'append_child_copy']
    assert len(copied) == 1
    destination = module.sequence_tables[exports[0]].sequences[copied[0].attributes['sequence_id']]
    assert destination.child_table_pool is not None
    assert destination.heap_owner_value_id is not None
    assert not any(op.attributes['ssa_sequence_operation'] == 'append' for op in calls)


@pytest.mark.parametrize('body', [
    'def root(nodes: list[Node], out: list[str]) -> int:\n    for node in nodes:\n        out.append(node.key)\n    return len(out)\n',
    'def root(nodes: list[Node]) -> int:\n    out = []\n    out.append(42)\n    for node in nodes:\n        out.append(node.key)\n    return len(out)\n',
    'def root(nodes: list[Node]) -> int:\n    out = []\n    alias = out\n    for node in nodes:\n        out.append(node.key)\n        alias.append(42)\n    return len(out)\n',
])
def test_proven_runtime_span_never_falls_back_to_scalar_append(body):
    source = SOURCE[:SOURCE.index('def root')] + body
    with pytest.raises(SSALoweringIncompleteError, match='borrowed-string-child-copy'):
        lower_before_owned_record_boundary(source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
            progress=lambda _: None, progress_bars=False)


def test_scalar_record_field_keeps_scalar_append():
    module, _, exports = lower_before_owned_record_boundary(
        SOURCE.replace('key: str', 'key: int'), 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    calls = [op for block in module.functions[exports[0]].blocks.values() for op in block.instrs
             if op.op == 'Call' and op.attributes.get('ssa_sequence_operation')]
    assert any(op.attributes['ssa_sequence_operation'] == 'append' for op in calls)
    assert not any(op.attributes['ssa_sequence_operation'] == 'append_child_copy' for op in calls)


def test_repeated_field_reads_share_input_pool_not_source_identity():
    module, _, exports = lower_before_owned_record_boundary(
        SOURCE.replace('out.append(node.key)', 'out.append(node.key)\n        out.append(node.key)'),
        'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    calls = [op for block in module.functions[exports[0]].blocks.values() for op in block.instrs
             if op.op == 'Call' and op.attributes.get('ssa_sequence_operation') == 'append_child_copy']
    assert len(calls) == 2


def test_unannotated_string_consumer_does_not_compare_row_handles():
    source = SOURCE.replace('return len(out)', 'total = 0\n    for word in out:\n        total += len(word)\n    return total')
    with pytest.raises(SSALoweringIncompleteError, match='string-row-class'):
        lower_before_owned_record_boundary(source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
            progress=lambda _: None, progress_bars=False)


@pytest.mark.parametrize('backend', ['llvm', 'c'])
@pytest.mark.parametrize('comparison', ['==', '!='])
@pytest.mark.parametrize('needle', ['', '\u00e9'])
def test_copied_record_strings_native_contents(tmp_path, backend, comparison, needle):
    import numpy as np
    from src.compiler.source_sequence_child_lifetime import validated_sequence_child_storage
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.string_table import string_token
    source = SOURCE.replace('out = []', 'out: list[str] = []').replace(
        'return len(out)',
        "total = 0\n    for word in out:\n        total += len(word)\n"
        "        if word " + comparison + " " + repr(needle) + ":\n            total += 100\n    return total")
    module, _, exports = lower_before_owned_record_boundary(source, 'root',
        extraction_contract=ExtractionContract(Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    function = module.functions[name]
    tables = module.sequence_tables[name].sequences
    owned = [sid for sid, descriptor in tables.items()
             if descriptor.child_table_pool is not None and descriptor.heap_owner_value_id is not None]
    assert len(owned) == 1
    assert validated_sequence_child_storage(module, name, owned[0])
    parent_pool = tables[owned[0]].child_table_pool
    assert parent_pool.writable and parent_pool.status_value_id is not None
    views = [descriptor for descriptor in tables.values()
             if not descriptor.writable and descriptor.child_table_pool is None
             and descriptor.sequence_id != owned[0]]
    assert views and all(view.status_address_id is None for view in views)
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
    fields = {arg.accounting['program_abi_field']: arg.id for arg in function.args
              if 'program_abi_field' in (arg.accounting or {})}
    result = next(op.args[0].id for block in function.blocks.values() for op in block.instrs
                  if op.op == 'Ret' and not op.attributes.get('fault_exit'))
    for words in ([], [''], ['a', '\u00e9', '', 'alphabet'],
                  ['\u00e9x', '\u00e9', '\u00e9\u00e9', '\u00e8'],
                  ['x'*129, '\u00e9', '\u03bb\u03bb', 'z']):
        stride = max([1, *map(len, words)])
        feeds = {root.column_value_ids[0]: np.arange(len(words), dtype=np.int64),
                 root.length_address_id: np.array([len(words)], dtype=np.int64),
                 root.capacity_value_id: np.array([len(words)], dtype=np.int64),
                 root.status_address_id: np.array([0], dtype=np.int64)}
        values = {'column': [string_token(word[i]) if i < len(word) else 0
                             for word in words for i in range(stride)],
                  'lengths': list(map(len, words)), 'row_stride': [stride], 'capacity': [len(words)]}
        feeds.update({fields['Node.key.' + member]: np.array(data, dtype=np.int64)
                      for member, data in values.items()})
        for _ in range(2):
            execution = prepare(feeds).run()
            matches = sum((word == needle) if comparison == '==' else (word != needle) for word in words)
            assert int(execution.buffers[result].reshape(-1)[0]) == sum(map(len, words)) + 100 * matches


def test_resident_string_comparison_refuses_missing_class_and_wrong_layout(monkeypatch):
    import src.compiler.source_borrowed_record_spans as proofs
    original = _ControlSSABuilder._resident_string_control_comparison
    checked = []

    def inspect(builder, expression, operands, **kwargs):
        ids = [item.value_id for item in expression.operands]
        selected = [sid for sid in ids if sid is not None and sid in builder.child_table_selections]
        if expression.op in {'eq', 'ne'} and selected:
            sid = selected[0]
            descriptor = builder.sequence_descriptors[sid]
            for altered in (replace(descriptor, writable=True),
                            replace(descriptor, column_dtypes=('float64',)),
                            replace(descriptor, column_shapes=((1,),)),
                            replace(descriptor, key_columns=(0,)),
                            replace(descriptor, status_address_id=descriptor.capacity_value_id)):
                builder.sequence_descriptors[sid] = altered
                with pytest.raises(ValueError, match='incompatible or unbound glyph storage'):
                    original(builder, expression, operands, **kwargs)
            builder.sequence_descriptors[sid] = descriptor
            exact = proofs.exact_string_classes
            source = builder._canonical_cell(sid)
            with monkeypatch.context() as patch:
                patch.setattr(proofs, 'exact_string_classes',
                              lambda book, value: () if value == source else exact(book, value))
                with pytest.raises(ValueError, match='requires exact string classes'):
                    original(builder, expression, operands, **kwargs)
            checked.append(sid)
        return original(builder, expression, operands, **kwargs)

    monkeypatch.setattr(_ControlSSABuilder, '_resident_string_control_comparison', inspect)
    source = SOURCE.replace('out = []', 'out: list[str] = []').replace(
        'return len(out)', "total = 0\n    for word in out:\n        if word == 'a':\n            total += 1\n    return total")
    lower_before_owned_record_boundary(source, 'root', extraction_contract=ExtractionContract(
        Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    assert checked


@pytest.mark.parametrize('comparison', ['==', '!='])
def test_numeric_control_comparison_stays_scalar(tmp_path, comparison):
    import numpy as np
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    source = 'def root(a: int, b: int) -> int:\n    value = 3\n    if a ' + comparison + ' b:\n        value = 7\n    return value\n'
    module, _, exports = lower_before_owned_record_boundary(source, 'root', extraction_contract=ExtractionContract(
        Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    function = module.functions[exports[0]]
    assert not any(op.op == 'Call' and op.attributes.get('ssa_sequence_operation') == 'equal'
                   for block in function.blocks.values() for op in block.instrs)
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / 'llvm')
    names = dict(function.metadata['parameter_names'])
    result = next(op.args[0].id for block in function.blocks.values() for op in block.instrs
                  if op.op == 'Ret' and not op.attributes.get('fault_exit'))
    for a, b in ((0, 0), (1, 2), (-7, -7), (-7, 7)):
        execution = prepare_artifact_execution(artifact, {
            names['a']: np.array([a], dtype=np.int64), names['b']: np.array([b], dtype=np.int64),
        }).run()
        expected = 7 if ((a == b) if comparison == '==' else (a != b)) else 3
        assert int(execution.buffers[result].reshape(-1)[0]) == expected


@pytest.mark.parametrize('mutation', ['out.clear()', 'alias.clear()'])
def test_selected_string_view_cannot_survive_mutation(mutation):
    source = SOURCE.replace('out = []', 'out: list[str] = []').replace(
        'return len(out)', 'alias = out\n    total = 0\n    for word in out:\n        ' + mutation + '\n        total += len(word)\n    return total')
    with pytest.raises(SSALoweringIncompleteError, match='string-row-read-phase'):
        lower_before_owned_record_boundary(source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
            progress=lambda _: None, progress_bars=False)
