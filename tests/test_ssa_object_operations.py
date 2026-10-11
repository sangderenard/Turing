"""Checked object helper calls, compiled by our public source compiler."""
from pathlib import Path

import numpy as np
import pytest

from src.transmogrifier.ssa import SSAOpaqueObject, SSAValue, Function, BasicBlock, Instr, IRModule
from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_object_operations import (
    ObjectOperation, declare_object_helper, object_call,
    object_identity, object_same, validate_object_operations,
    object_kind_matches,
)
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)
from src.compiler.identity_concordance import concordance_report


ROOT = Path(__file__).resolve().parents[1]
KINDS = ('builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
         'builtins.frozenset', 'builtins.object')


def policy(mutable=False):
    return ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {
            'Book': {'identity': 'metadata.Book', 'fields': {'objects': {
                'storage': 'keyed', 'dtype': 'int64', 'key_encoding': 'integer_identity',
                'value_record': 'Object',
            }}},
            'Object': {'identity': 'metadata.Object', 'fields': {
                'kind': {'storage': 'scalar', 'dtype': 'int64', 'token_vocabulary': list(KINDS)},
                'members': {'storage': 'table', 'mutable': mutable, 'columns': [
                    {'name': 'key', 'dtype': 'opaque_ref'},
                    {'name': 'value', 'dtype': 'opaque_ref'},
                ]},
            }},
        },
        'bindings': [{'function': '*', 'parameter': 'book', 'record': 'Book'}],
        'values': [
            {'function': '*', 'parameter': 'handle', 'storage': 'reference',
             'python_type': 'builtins.object'},
            {'function': '*', 'parameter': 'value', 'storage': 'reference',
             'python_type': 'builtins.object'},
            {'function': '*', 'parameter': 'index', 'storage': 'scalar',
             'dtype': 'int64', 'python_type': 'builtins.int'},
        ],
    }).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')


def compile_helper(operation, mutable_members=None):
    body = {
        ObjectOperation.KIND: '    return item.kind\n',
        ObjectOperation.LENGTH: '    return len(item.members)\n',
        ObjectOperation.GET: '    return item.members[index][1]\n',
        ObjectOperation.SET: '    item.members[index][1]=value\n    return True\n',
    }[operation]
    source = ('def inspect(book,handle,index,value):\n    identity=id(handle)\n'
              '    item=book.objects[identity]\n' + body)
    module, _, exports = lower_ast_source_to_ssa(source, 'inspect', name='object_operations',
        extraction_contract=policy(mutable=(operation is ObjectOperation.SET
                                           if mutable_members is None else mutable_members)), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    helper = module.functions[exports[0]]
    parameters = dict(helper.metadata['parameter_names'])
    reference_id = parameters['handle']
    formal_ids = {v.id for v in helper.args}
    operand_ids = tuple(parameters[name] for name in ('index', 'value')
                        if parameters.get(name) in formal_ids)
    context_ids = tuple(v.id for v in helper.args if v.id not in {reference_id, *operand_ids})
    declare_object_helper(module, helper.name, operation, storage_identity='metadata.Book.objects',
        reference_formal_id=reference_id, context_formal_ids=context_ids, operand_formal_ids=operand_ids,
        kind_vocabulary=KINDS if operation is ObjectOperation.KIND else ())
    # The helper body goes through the public source entry. The caller below
    # is an SSA API test, NOT proof that source-level implicit object dispatch
    # or nested-record context forwarding through a source caller is solved.
    bindings = {v.id: v for v in helper.args}
    helper_output = next(i.args[0] for b in helper.blocks.values() for i in b.instrs if i.op == 'Ret')
    result = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), helper_output.dtype, helper_output.shape)
    lowered = object_call(module, helper.name,
        SSAOpaqueObject(bindings[reference_id], 'metadata.Book.objects'),
        context={key: bindings[key] for key in context_ids},
        operands={key: bindings[key] for key in operand_ids}, result=result)
    root = Function('object_operation_caller', list(helper.args),
        {'entry': BasicBlock('entry', [lowered, Instr('Ret', [result], None)])},
        metadata={'parameter_names': helper.metadata.get('parameter_names', ()),
                  'storage_formals': tuple({'value_id': key, 'kind': 'native-object-context'}
                                           for key in context_ids)})
    module.functions[root.name] = root
    return module, root, helper, lowered


def test_read_only_member_table_cannot_be_mutated():
    with pytest.raises(ValueError, match='read-only'):
        compile_helper(ObjectOperation.SET, mutable_members=False)


@pytest.mark.parametrize('accepted', [('builtins.dict',), ('builtins.dict', 'builtins.list')])
def test_native_kind_membership_uses_the_owner_vocabulary(accepted, tmp_path):
    module, root, helper, call = compile_helper(ObjectOperation.KIND)
    receipt = helper.metadata['opaque_object_helper']
    bindings = dict(zip(call.attributes['callee_input_ids'], call.args))
    result = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), 'bool')
    root.blocks['entry'].instrs = [*object_kind_matches(module, helper.name,
        SSAOpaqueObject(bindings[receipt['reference_formal_id']], receipt['storage_identity']),
        context={key: bindings[key] for key in receipt['context_formal_ids']},
        accepted=accepted, kind_result=call.res, result=result), Instr('Ret', [result], None)]
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'kind_match')
    facts = {'objects.length': [3], 'objects.keys': [2**60+3, 17, 29],
             'objects.values': [0, 1, 2], 'objects[].kind.column': [1, 2, 6]}
    buffers = {v.id: np.array(facts[v.accounting['program_abi_field']], dtype=np.int64)
               for v in root.args if v.accounting.get('program_abi_field') in facts}
    handle_id = dict(root.metadata['parameter_names'])['handle']
    for handle, kind in [(2**60+3, 'builtins.dict'), (17, 'builtins.list'), (29, 'builtins.object')]:
        buffers[handle_id] = np.array([handle], dtype=np.int64)
        assert prepare_artifact_execution(native, buffers).run().buffers[result.id].item() is (kind in accepted)


def test_backend_text_emission_keeps_checked_native_calls():
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_fortran_backend import emit_module
    for operation in (ObjectOperation.KIND, ObjectOperation.LENGTH, ObjectOperation.GET, ObjectOperation.SET):
        module, root, helper, call = compile_helper(operation)
        c = emit_ssa_module_to_c(module, root.name)
        assert c.complete, c.shortfalls
        fortran = emit_module(module, outputs={root.name: (call.res,)}, progress=lambda _: None)
        assert fortran.complete, fortran.shortfalls


def test_changed_callee_body_and_context_identities_invalidate_calls():
    module, root, helper, call = compile_helper(ObjectOperation.KIND)
    call.args[0] = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), call.args[0].dtype, call.args[0].shape)
    with pytest.raises(ValueError, match='stale context/reference'):
        validate_object_operations(module)
    assert 'opaque-object-contract' in concordance_report(module)


def test_native_result_identity_and_record_sequence_claim_are_audited():
    module, root, helper, call = compile_helper(ObjectOperation.GET)
    call.attributes['native_result_contract'] = ()
    with pytest.raises(ValueError, match='stale native result'):
        validate_object_operations(module)
    assert 'opaque-object-contract' in concordance_report(module)
    # A record field cannot keep naming a retired table descriptor.
    records = module.record_tables[helper.name]
    sequence_id = next(field.sequence_id for record in records.records.values()
                       for field in record.fields if field.sequence_id is not None)
    module.sequence_tables[helper.name].sequences.pop(sequence_id)
    assert 'record-sequence-unknown' in concordance_report(module)


@pytest.mark.parametrize('operation', [ObjectOperation.KIND, ObjectOperation.LENGTH,
                                       ObjectOperation.GET, ObjectOperation.SET])
@pytest.mark.parametrize('depth', [1, 2])
def test_source_calls_forward_complete_object_table_context(operation, depth, tmp_path):
    # Actual authored call composition through the public compiler entry,
    # not the manually assembled SSA caller used by the helper-API tests.
    body = {
        ObjectOperation.KIND: '    return item.kind\n',
        ObjectOperation.LENGTH: '    return len(item.members)\n',
        ObjectOperation.GET: '    return item.members[index][1]\n',
        ObjectOperation.SET: '    item.members[index][1]=value\n    return True\n',
    }[operation]
    source = ('def inspect(book,handle,index,value):\n    identity=id(handle)\n'
              '    item=book.objects[identity]\n' + body)
    target = 'inspect'
    if depth == 2:
        source += 'def relay(book,handle,index,value):\n    return inspect(book,handle,index,value)\n'
        target = 'relay'
    source += f'def root(book,handle,index,value):\n    return {target}(book,handle,index,value)\n'
    module, outputs, exports = lower_ast_source_to_ssa(source, 'root', name='object_context',
        extraction_contract=policy(operation is ObjectOperation.SET), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    root = module.functions[exports[0]]
    fields = {v.accounting.get('program_abi_field'): v for v in root.args
              if v.accounting.get('program_abi_field')}
    context_paths = ('objects[].members.columns.key', 'objects[].members.columns.value',
                     'objects[].members.lengths', 'objects[].members.row_stride')
    for path in context_paths:
        assert path in fields
        assert sum(v.accounting.get('program_abi_field') == path for v in root.args) == 1
        assert 'linked_call_frame_storage' not in fields[path].accounting
        assert 'returned_record_storage' not in fields[path].accounting
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source_calls')
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_fortran_backend import emit_module
    c = emit_ssa_module_to_c(module, root.name)
    assert c.complete, c.shortfalls
    fortran = emit_module(module, outputs=outputs, progress=lambda _: None)
    assert fortran.complete, fortran.shortfalls
    facts = {
        'objects.length': [3], 'objects.keys': [2**60+3, 17, 29], 'objects.values': [0, 1, 2],
        'objects[].kind.column': [1, 2, 6], 'objects[].members.lengths': [2, 0, 1],
        'objects[].members.row_stride': [2], 'objects[].members.columns.key': [7, 11, 0, 0, 13, 0],
        'objects[].members.columns.value': [2**60+19, 23, 0, 0, 31, 0],
    }
    arrays = {path: np.array(data, dtype=np.int64) for path, data in facts.items()}
    buffers = {v.id: arrays[v.accounting['program_abi_field']] for v in root.args
               if v.accounting.get('program_abi_field') in arrays}
    parameters = dict(root.metadata['parameter_names'])
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    cases = [(2**60+3, 0, 1, 2, 2**60+19), (29, 0, 6, 1, 31)]
    if operation in {ObjectOperation.KIND, ObjectOperation.LENGTH}:
        cases.append((17, 0, 2, 0, None))
    for handle, index, kind, length, member in cases:
        replacement = 2**60+43
        for name, value in [('handle', handle), ('index', index), ('value', replacement)]:
            if parameters.get(name) in {v.id for v in root.args}:
                buffers[parameters[name]] = np.array([value], dtype=np.int64)
        execution = prepare_artifact_execution(native, buffers).run()
        assert execution.buffers[output].item() == {
            ObjectOperation.KIND: kind, ObjectOperation.LENGTH: length,
            ObjectOperation.GET: member, ObjectOperation.SET: True}[operation]
        if operation is ObjectOperation.SET:
            row = 0 if handle == 2**60+3 else 2
            assert arrays['objects[].members.columns.value'][row*2+index] == replacement
        assert arrays['objects[].members.columns.key'].tolist() == facts['objects[].members.columns.key']
        assert arrays['objects[].members.lengths'].tolist() == facts['objects[].members.lengths']


@pytest.mark.parametrize('body, cases', [
    ('    if index:\n        result=23\n    else:\n        result=41\n',
     [(0, 41), (1, 23)]),
    ('    if index<=0:\n        result=23\n    else:\n        result=41\n',
     [(0, 23), (1, 41)]),
    ('    if index<=0:\n        result=23\n    else:\n'
     '        if index<2:\n            result=41\n        else:\n            result=59\n',
     [(0, 23), (1, 41), (2, 59)]),
    ('    result=5\n    if index<=0:\n        result=23\n    else:\n        result=41\n',
     [(0, 23), (1, 41)]),
])
def test_authored_assignment_merges_preserve_each_arm(body, cases, tmp_path):
    # New names use result merges, nested merges retain their arm ownership,
    # and an existing name still uses the incumbent carried-value path.
    module, _, exports = lower_ast_source_to_ssa('def root(index):\n' + body + '    return result\n',
        'root', name='assignment_merge', extraction_contract=policy(),
        runtime_closure_only=True, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'assignment_merge')
    parameter = dict(root.metadata['parameter_names'])['index']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    for index, expected in cases:
        assert prepare_artifact_execution(native, {
            parameter: np.array([index], dtype=np.int64),
        }).run().buffers[output].item() == expected


def test_recursive_keyed_view_shares_storage_and_returns_base_case(tmp_path):
    # Both arms first define result: there is no pre-branch version. Preserve
    # the reducer's exact result Phi AND settle storage before recursive lease.
    source = ('def inspect(book,handle,index):\n    if index<=0:\n'
              '        result=len(book.objects[id(handle)].members)\n'
              '    else:\n        result=inspect(book,handle,index-1)\n'
              '    return result\n'
              'def root(book,handle,index):\n    return inspect(book,handle,index)\n')
    module, outputs, exports = lower_ast_source_to_ssa(source, 'root', name='recursive_context',
        extraction_contract=policy(), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    recursive_calls = []
    for function in module.functions.values():
        for block in function.blocks.values():
            for call in block.instrs:
                if call.op == 'Call' and call.attributes.get('callee') == function.name:
                    recursive_calls.append((function, call))
    assert recursive_calls
    paths = ('objects.length', 'objects.keys', 'objects.values',
             'objects[].members.columns.key', 'objects[].members.columns.value',
             'objects[].members.lengths', 'objects[].members.row_stride')
    for function, call in recursive_calls:
        fields = {v.accounting.get('program_abi_field'): v for v in function.args
                  if v.accounting.get('program_abi_field')}
        bindings = dict(zip(call.attributes['callee_input_ids'], call.args))
        for path in paths:
            assert bindings[fields[path].id].id == fields[path].id
            assert 'linked_call_frame_storage' not in fields[path].accounting
        assert not any(v.accounting.get('program_abi_storage') == 'keyed'
                       for v in function.args)
        sequence = module.sequence_tables[function.name].by_id(fields['objects.keys'].id)
        assert sequence.column_value_ids == (fields['objects.keys'].id, fields['objects.values'].id)
        assert sequence.column_dtypes == ('int64', 'int64')
        assert sequence.length_address_id == sequence.capacity_value_id == fields['objects.length'].id
        local_storage = [i for b in function.blocks.values() for i in b.instrs
                         if i.attributes.get('binding') == 'recursive_invocation_storage']
        assert len(local_storage) == 1
        assert local_storage[0].res.dtype == 'int'
        assert not local_storage[0].res.accounting.get('sequence_arena')
        merges = [i for b in function.blocks.values() for i in b.instrs
                  if i.op == 'Phi' and i.attributes.get('binding') == 'conditional_result']
        assert len(merges) == 1
        assert merges[0].res.dtype == 'int64'
        assert all(v.dtype == 'int64' for v in merges[0].args)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'recursive_context')
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_fortran_backend import emit_module
    c = emit_ssa_module_to_c(module, root.name)
    # The C module lane still lacks the incumbent invocation-local zeros
    # spelling. Do not confuse that backend limitation with LLVM execution.
    assert not c.complete
    assert {finding.operation for finding in c.shortfalls} == {'zeros'}
    fortran = emit_module(module, outputs=outputs, progress=lambda _: None)
    assert fortran.complete, fortran.shortfalls
    facts = {'objects.length': [3], 'objects.keys': [2**60+3, 17, 29],
             'objects.values': [0, 1, 2], 'objects[].kind.column': [1, 2, 6],
             'objects[].members.lengths': [2, 0, 1], 'objects[].members.row_stride': [2],
             'objects[].members.columns.key': [7, 11, 0, 0, 13, 0],
             'objects[].members.columns.value': [2**60+19, 23, 0, 0, 31, 0]}
    buffers = {v.id: np.array(facts[v.accounting['program_abi_field']], dtype=np.int64)
               for v in root.args if v.accounting.get('program_abi_field') in facts}
    parameters = dict(root.metadata['parameter_names'])
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    for handle, length in [(2**60+3, 2), (17, 0), (29, 1)]:
        for depth in (0, 1, 7, 32):
            buffers[parameters['handle']] = np.array([handle], dtype=np.int64)
            buffers[parameters['index']] = np.array([depth], dtype=np.int64)
            assert prepare_artifact_execution(native, buffers).run().buffers[output].item() == length
    for value in root.args:
        path = value.accounting.get('program_abi_field')
        if path in facts:
            assert buffers[value.id].tolist() == facts[path]


@pytest.mark.parametrize('operation', [ObjectOperation.KIND, ObjectOperation.LENGTH,
                                       ObjectOperation.GET, ObjectOperation.SET])
def test_native_object_helpers_over_existing_descriptor_domain(operation, tmp_path):
    module, root, helper, call = compile_helper(operation)
    validate_object_operations(module)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / operation.value)
    facts = {
        'objects.length': np.array([3], dtype=np.int64),
        'objects.keys': np.array([2**60+3, 17, 29], dtype=np.int64),
        'objects.values': np.array([0, 1, 2], dtype=np.int64),
        'objects[].kind.column': np.array([1, 2, 6], dtype=np.int64),
        'objects[].members.lengths': np.array([2, 0, 1], dtype=np.int64),
        'objects[].members.row_stride': np.array([2], dtype=np.int64),
        'objects[].members.columns.key': np.array([7, 11, 0, 0, 13, 0], dtype=np.int64),
        'objects[].members.columns.value': np.array([2**60+19, 23, 0, 0, 31, 0], dtype=np.int64),
    }
    buffers = {v.id: facts[v.accounting['program_abi_field']] for v in root.args
               if v.accounting.get('program_abi_field') in facts}
    parameters = dict(root.metadata['parameter_names'])
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    cases = [(2**60+3, 0, 1, 2, 2**60+19), (29, 0, 6, 1, 31)]
    if operation in {ObjectOperation.KIND, ObjectOperation.LENGTH}:
        cases.append((17, 0, 2, 0, None))
    for handle, index, kind, length, member in cases:
        replacement = 2**60+43
        for name, value in [('handle', handle), ('index', index), ('value', replacement)]:
            if parameters.get(name) in {v.id for v in root.args}:
                buffers[parameters[name]] = np.array([value], dtype=np.int64)
        execution = prepare_artifact_execution(native, buffers).run()
        expected = {ObjectOperation.KIND: kind, ObjectOperation.LENGTH: length,
                    ObjectOperation.GET: member, ObjectOperation.SET: True}[operation]
        assert execution.buffers[output].item() == expected
        if operation is ObjectOperation.SET:
            row = 0 if handle == 2**60+3 else 2
            assert facts['objects[].members.columns.value'][row*2+index] == replacement
    assert all(i.op != 'PtrCast' for b in root.blocks.values() for i in b.instrs)


def test_wrong_context_domain_and_stale_helper_are_refused_and_audited():
    module, root, helper, call = compile_helper(ObjectOperation.KIND)
    call.attributes['opaque_object_operation'] = {**call.attributes['opaque_object_operation'],
                                                  'storage_identity': 'different.owner'}
    with pytest.raises(ValueError, match='matching live helper'):
        emit_ssa_function_to_llvm(module, root.name)
    assert 'opaque-object-contract' in concordance_report(module)
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_fortran_backend import emit_module
    for emit in [lambda: emit_ssa_module_to_c(module, root.name),
                 lambda: emit_module(module, progress=lambda _: None)]:
        with pytest.raises(ValueError, match='matching live helper'):
            emit()


def test_object_identity_is_not_dereference_or_a_payload_type_guess():
    reference = SSAValue(7, 'opaque_ref')
    object_ = SSAOpaqueObject(reference, 'owner')
    assert object_identity(object_, SSAValue(8, 'int64')).op == 'Cast'
    assert object_same(object_, object_, SSAValue(9, 'bool')).op == 'Eq'
    with pytest.raises(ValueError, match='scalar opaque_ref'):
        SSAOpaqueObject(SSAValue(7, 'ptr'), 'owner')
    with pytest.raises(ValueError, match='same admitted'):
        object_same(object_, SSAOpaqueObject(reference, 'other'), SSAValue(9, 'bool'))


@pytest.mark.parametrize('operation', ['identity', 'same'])
def test_native_identity_preserves_reference_bits(operation, tmp_path):
    left, right = SSAValue(0, 'opaque_ref'), SSAValue(1, 'opaque_ref')
    result = SSAValue(2, 'int64' if operation == 'identity' else 'bool')
    first, second = SSAOpaqueObject(left, 'owner'), SSAOpaqueObject(right, 'owner')
    instruction = (object_identity(first, result) if operation == 'identity'
                   else object_same(first, second, result))
    root = Function('identity', [left, right], {'entry': BasicBlock('entry', [
        instruction, Instr('Ret', [result], None)])})
    artifact = emit_ssa_function_to_llvm(IRModule(functions={root.name: root}), root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / operation)
    for other in (2**60+3, 2**60+5):
        execution = prepare_artifact_execution(native, {
            0: np.array([2**60+3], dtype=np.int64),
            1: np.array([other], dtype=np.int64),
        }).run()
        assert execution.buffers[result.id].item() == (
            2**60+3 if operation == 'identity' else other == 2**60+3)


def test_helper_roles_and_result_layout_are_not_guessed():
    handle, wrong_result = SSAValue(0, 'opaque_ref'), SSAValue(1, 'float64')
    helper = Function('bad_kind', [handle], {'entry': BasicBlock('entry', [
        Instr('Const', [], wrong_result, attributes={'value': 1.0}),
        Instr('Ret', [wrong_result], None)])})
    module = IRModule(functions={helper.name: helper})
    with pytest.raises(ValueError, match='cover each exact formal once'):
        declare_object_helper(module, helper.name, ObjectOperation.KIND,
            storage_identity='owner', reference_formal_id=handle.id,
            context_formal_ids=(handle.id,))
    with pytest.raises(ValueError, match='requires result'):
        declare_object_helper(module, helper.name, ObjectOperation.KIND,
            storage_identity='owner', reference_formal_id=handle.id)


@pytest.mark.parametrize('expression, accepted, expected', [
    ('{}', 'dict', True), ('{}', 'list', False), ('{}', 'object', True),
    ('[]', '(dict,list)', True), ('set()', 'dict', False),
])
def test_known_constructor_kind_is_not_lost_with_unfoldable_payload(expression, accepted, expected, tmp_path):
    # No reference domain is needed when the constructor already proves kind.
    source = f'def root():\n    value={expression}\n    return isinstance(value,{accepted})\n'
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='known_object_kind',
        extraction_contract=policy(), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'kind')
    root = module.functions[exports[0]]
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    assert prepare_artifact_execution(native, {}).run().buffers[output].item() is expected
