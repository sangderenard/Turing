"""Real public-entry regressions for structural uses and recursive edges."""
from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def native_policy(values):
    return ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': values}
    ).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')


@pytest.mark.parametrize('private', [False, True])
def test_native_reference_identity_keeps_its_authored_parameter(private, tmp_path):
    source = ('def leaf(identity):\n    return identity\n'
              'def root(value, unused):\n    return leaf(id(value))\n' if private else
              'def root(value, unused):\n    return id(value)\n')
    policy = native_policy([{'function': 'root', 'parameter': 'value',
                            'storage': 'reference', 'python_type': 'builtins.object'}])
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='native_identity', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None)
    root = module.functions[exports[0]]
    parameter_id = dict(root.metadata['parameter_names'])['value']
    assert 'unused' not in dict(root.metadata['parameter_names'])
    identity = [i for b in root.blocks.values() for i in b.instrs
                if i.attributes.get('binding') == 'opaque_reference_identity']
    assert len(identity) == 1
    assert identity[0].op == 'Cast'
    assert identity[0].args[0].dtype == 'opaque_ref'
    assert 'tensor_operation' not in identity[0].attributes
    assert identity[0].args[0].id == parameter_id
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    import numpy as np
    native = compile_artifact(artifact, directory=tmp_path / 'identity_native')
    output_id = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    # These are ABI stable handles, NOT claims of CPython object/address parity.
    for handle in (1, 4660, 2**60 + 3):
        execution = prepare_artifact_execution(native, {parameter_id: np.array([handle], dtype=np.uint64)}).run()
        assert execution.buffers[output_id].item() == handle
    from src.compiler.identity_concordance import CorrelationTable
    assert not [f for f in CorrelationTable.build(module).findings(module)
                if f.kind not in {'unsourced-fact', 'unsourced-identity'}]


@pytest.mark.parametrize('nested', [False, True])
def test_recursive_backedge_retains_its_source_call(nested, tmp_path):
    source = '''
def visit(n):
    if n <= 0.0:
        return 0.0
    return visit(n - 1) + 1

def root(n):
    return visit(n)
'''
    if nested:
        source = 'def root(n):\n' + ''.join('    ' + line + '\n' for line in source.splitlines()[:6]) + '    return visit(n)\n'
    policy = native_policy([{'function': '*', 'parameter': 'n',
                            'storage': 'scalar', 'dtype': 'float64', 'rank': 0,
                            'python_type': 'builtins.float'}])
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='native_recursion', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None)
    recursive = [f for f in module.functions.values()
                 if any(i.op == 'Call' and i.attributes.get('callee') == f.name
                        for b in f.blocks.values() for i in b.instrs)]
    assert recursive, 'the graph backedge must remain an executable recursive Call'
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    import numpy as np
    native = compile_artifact(artifact, directory=tmp_path / 'recursive_native')
    root = module.functions[exports[0]]
    parameter_id = dict(root.metadata['parameter_names'])['n']
    output_id = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    for n in (0, 1, 4, 12):
        execution = prepare_artifact_execution(native, {parameter_id: np.array([n], dtype=np.float64)}).run()
        assert execution.buffers[output_id].item() == n


def test_mutually_recursive_component_keeps_both_call_edges(tmp_path):
    source = '''
def left(n):
    if n <= 0.0:
        return 0.0
    return right(n - 1) + 1
def right(n):
    if n <= 0.0:
        return 0.0
    return left(n - 1) + 1
def root(n):
    return left(n)
'''
    policy = native_policy([{'function': '*', 'parameter': 'n', 'storage': 'scalar',
                            'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'}])
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='native_mutual',
                                               extraction_contract=policy,
                                               runtime_closure_only=True, progress=lambda _: None)
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'mutual_native')
    root = module.functions[exports[0]]
    parameter_id = dict(root.metadata['parameter_names'])['n']
    output_id = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    import numpy as np
    for n in (0, 1, 4, 11):
        execution = prepare_artifact_execution(native, {parameter_id: np.array([n], dtype=np.float64)}).run()
        assert execution.buffers[output_id].item() == n


@pytest.mark.parametrize('membership', [False, True])
def test_opaque_reference_without_object_operations_is_not_accepted_as_native(membership):
    from src.compiler.ssa_fortran_backend import FortranEmissionError
    source = ('def root(value):\n    return 1 in value\n' if membership else
              'def root(value):\n    return isinstance(value, dict)\n')
    policy = native_policy([{'function': 'root', 'parameter': 'value',
                            'storage': 'reference', 'python_type': 'builtins.object'}])
    with pytest.raises(FortranEmissionError, match='structural_outputs='):
        lower_ast_source_to_ssa(source, 'root', name='opaque_object_limit',
                                extraction_contract=policy, progress=lambda _: None)


@pytest.mark.parametrize('declared, accepted', [
    ('dict', 'dict'), ('list', '(dict, list)'), ('object', 'object'),
])
def test_reference_type_contract_proves_positive_predicate(declared, accepted, tmp_path):
    policy = native_policy([{'function': 'root', 'parameter': 'value',
                            'storage': 'reference', 'python_type': 'builtins.' + declared}])
    module, _, exports = lower_ast_source_to_ssa(
        'def root(value):\n    return isinstance(value, ' + accepted + ')\n',
        'root', name='proven_reference_type', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None)
    root = module.functions[exports[0]]
    assert not root.args  # No handle dereference is required for this proof.
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'proven_type')
    output_id = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    assert prepare_artifact_execution(native, {}).run().buffers[output_id].item() is True


@pytest.mark.parametrize('declared, accepted', [('builtins.dict', 'list'), ('custom.dict', 'dict')])
def test_unproved_reference_predicate_is_not_folded_false(declared, accepted):
    from src.compiler.ssa_fortran_backend import FortranEmissionError
    policy = native_policy([{'function': 'root', 'parameter': 'value',
                            'storage': 'reference', 'python_type': declared}])
    with pytest.raises(FortranEmissionError, match='structural_outputs='):
        lower_ast_source_to_ssa('def root(value):\n    return isinstance(value, ' + accepted + ')\n',
                                'root', name='unproved_reference_type', extraction_contract=policy,
                                runtime_closure_only=True, progress=lambda _: None)


def test_typed_recursive_workspace_is_not_propagated_into_its_signature(tmp_path):
    source = '''
def visit(n):
    parts = [0.0]
    if n <= 0.0:
        return 0.0
    child = visit(n - 1)
    return len(parts) + child
def root(n):
    return visit(n)
'''
    policy = native_policy([{'function': '*', 'parameter': 'n', 'storage': 'scalar',
                            'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'}])
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='typed_recursive_workspace', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None)
    local = [(f, i) for f in module.functions.values() for b in f.blocks.values()
             for i in b.instrs if i.attributes.get('binding') == 'recursive_invocation_storage']
    assert local
    for function, instruction in local:
        assert instruction.op == 'zeros'
        assert instruction.res not in function.args
        assert instruction.res.accounting['compiler_frame_storage'] == function.name
    from src.compiler.identity_concordance import CorrelationTable
    assert not [f for f in CorrelationTable.build(module).findings(module)
                if f.kind not in {'unsourced-fact', 'unsourced-identity'}]
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'typed_scratch')
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['n']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    import numpy as np
    for n in (0, 1, 4, 12, 1, 0):
        execution = prepare_artifact_execution(native, {parameter: np.array([n], dtype=np.float64)}).run()
        assert execution.buffers[output].item() == n


def test_recursive_private_scratch_refuses_signature_feedback_before_leasing():
    from src.compiler.ssa_fortran_backend import FortranEmissionError
    source = '''
def visit(n):
    parts = []
    if n <= 0.0:
        return 0.0
    parts.append(visit(n - 1))
    return parts[0]
def root(n):
    return visit(n)
'''
    policy = native_policy([{'function': '*', 'parameter': 'n', 'storage': 'scalar',
                            'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'}])
    with pytest.raises(FortranEmissionError, match='requires per-invocation native storage'):
        lower_ast_source_to_ssa(source, 'root', name='recursive_scratch_limit',
                                extraction_contract=policy, runtime_closure_only=True,
                                progress=lambda _: None)


@pytest.mark.parametrize('recursive', [False, True])
def test_indexed_write_keeps_one_resident_across_regions_and_recursion(recursive, tmp_path):
    source = '''
def root(n):
    parts = [0.0]
    parts[0] = n
    return parts[0]
'''
    if recursive:
        source = '''
def visit(n):
    parts = [0.0]
    if n <= 0.0:
        return 0.0
    parts[0] = n
    child = visit(n - 1)
    return parts[0] + child
def root(n):
    return visit(n)
'''
    policy = native_policy([{'function': '*', 'parameter': 'n', 'storage': 'scalar',
                            'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'}])
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='resident_mutation', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None)
    stores = [i for f in module.functions.values() for b in f.blocks.values()
              for i in b.instrs if i.op == 'Store' and not i.attributes.get('binding')]
    assert stores, 'the authored indexed effect must survive structural folding'
    from src.compiler.identity_concordance import CorrelationTable
    assert not [f for f in CorrelationTable.build(module).findings(module)
                if f.kind not in {'unsourced-fact', 'unsourced-identity'}]
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'resident_mutation')
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['n']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    import numpy as np
    for n in (0, 1, 4, 12, 1, 0):
        execution = prepare_artifact_execution(native, {parameter: np.array([n], dtype=np.float64)}).run()
        assert execution.buffers[output].item() == (n * (n + 1) / 2 if recursive else n)
