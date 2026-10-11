"""Paired/native aggregate producers consume shared concordance cells."""
import numpy as np
import pytest

from tests.test_native_object_contract_batch import typed_context, compile_source
from src.compiler.identity_concordance import identity_book, concordance_report, Derived, Mode
from src.compiler.concordance_declarations import NATIVE_OBJECT_SOURCE, PLANNER_HIERARCHY
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_llvm_backend import prepare_artifact_execution, emit_ssa_function_to_llvm


@pytest.mark.parametrize('expression', ['(key,item)', '(item,key)', '(key,(item,key))'])
def test_native_pair_generator_preserves_nested_and_shared_values(tmp_path, expression):
    scope = typed_context()
    source = f'def root(value):\n    return tuple({expression} for key,item in value.items())\n'
    module, function, native, release, output = compile_source(source, scope, tmp_path)
    book = identity_book(module)
    page = book.page(NATIVE_OBJECT_SOURCE)
    birth = next(i for b in function.blocks.values() for i in b.instrs
                 if i.attributes.get('source_object_constructor'))
    cell = birth.attributes['source_object_constructor']['contract_cell']
    assert page.scope_row_count(cell.row[0]) > 0
    assert book.latest_ref(NATIVE_OBJECT_SOURCE, cell.row) == cell
    assert book.edges_into(cell)
    for count in (0, 1, 3, 64):
        leaf = object()
        shared = [leaf]
        original = {f'key-{i}': shared for i in range(count)}
        if count:
            shared.append(original)  # An imported backedge, not a flattened tensor.
        expected = tuple(eval(expression, {}, {'key': key, 'item': item})
                         for key, item in original.items())
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            parameter = dict(function.metadata['parameter_names'])['value']
            invocation = boundary.bind_execution(native, function,
                {parameter: np.array([boundary.roots[0]], dtype=np.int64)})
            invocation.run()
            assert invocation.native_invocations == 1
            result, = boundary.export((int(invocation.buffers[output.id].item()),))
            assert type(result) is tuple and len(result) == count
            for row, expected_row in zip(result, expected, strict=True):
                assert type(row) is tuple
                if expression == '(key,(item,key))':
                    assert row[0] is expected_row[0]
                    assert row[1][0] is shared and row[1][1] is expected_row[0]
                else:
                    assert all(actual is wanted for actual, wanted in zip(row, expected_row, strict=True))
            if count:
                assert shared[-1] is original and shared[0] is leaf
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


@pytest.mark.parametrize('fault', ['copied_projection', 'superseded_constructor', 'missing_cell'])
def test_backend_reads_the_book_not_a_copied_pair_receipt(tmp_path, fault):
    scope = typed_context()
    source = 'def root(value):\n    return tuple((key,item) for key,item in value.items())\n'
    module, function, _, _, _ = compile_source(source, scope, tmp_path)
    birth = next(i for b in function.blocks.values() for i in b.instrs
                 if i.attributes.get('source_object_constructor'))
    receipt = birth.attributes['source_object_constructor']
    if fault == 'copied_projection':
        collection = next(i for b in function.blocks.values() for i in b.instrs
                          if i.attributes.get('source_object_collection'))
        collection.attributes['source_object_collection']['target_projection_ids'] = (0, 1)
    elif fault == 'missing_cell':
        receipt['contract_cell'] = None
    else:
        book = identity_book(module)
        cell = receipt['contract_cell']
        fact = dict(book.page(NATIVE_OBJECT_SOURCE).latest(cell.row))
        fact['member_value_ids'] = tuple(reversed(fact['member_value_ids']))
        book.post(NATIVE_OBJECT_SOURCE, cell.row, tuple(sorted(fact.items())),
                  stage=PLANNER_HIERARCHY, provenance=Derived((cell,)), mode=Mode.REVISE)
    assert 'source-object-contract' in concordance_report(module)
    with pytest.raises(ValueError):
        emit_ssa_function_to_llvm(module, function.name)


def test_native_mapping_update_consumes_paired_generator_once(tmp_path):
    scope = typed_context()
    source = ('def identity(item):\n    return item\n'
              'def root(value):\n    result={}\n'
              '    result.update((identity(key),identity(item)) for key,item in value.items())\n'
              '    return result\n')
    _, function, native, release, output = compile_source(source, scope, tmp_path)
    for count in (0, 1, 3, 64):
        leaf = object()
        shared = [leaf]
        original = {f'key-{i}': shared for i in range(count)}
        if count:
            shared.append(original)
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            parameter = dict(function.metadata['parameter_names'])['value']
            invocation = boundary.bind_execution(native, function,
                {parameter: np.array([boundary.roots[0]], dtype=np.int64)})
            invocation.run()
            result, = boundary.export((int(invocation.buffers[output.id].item()),))
            assert invocation.native_invocations == 1
            assert type(result) is dict and result is not original
            assert list(result) == list(original)
            assert all(item is shared for item in result.values())
            if count:
                assert shared[-1] is original
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()
