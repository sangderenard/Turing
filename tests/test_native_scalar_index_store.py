import pickle
import subprocess
import sys

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


@pytest.mark.parametrize('backend', ['c', 'llvm'])
def test_scalar_integer_stores_do_not_inherit_shared_kernel_buffer_rank(tmp_path, backend):
    """N4 pub_contract stores scalar zero through a pointer-taking kernel."""
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'bindings': [], 'values': [{
            'function': 'root', 'parameter': 'array', 'storage': 'span',
            'python_type': 'numpy.ndarray', 'dtype': 'float64',
            'rank': 1, 'shape': [4], 'mutable': True,
        }, {
            'function': 'root', 'parameter': 'value', 'storage': 'scalar',
            'dtype': 'int64', 'rank': 0, 'python_type': 'int',
        }],
    })
    module, _, exports = lower_ast_source_to_ssa(
        'def root(array, value):\n'
        '    array[0] = value\n'
        '    array[1] = value\n'
        '    array[2] = False\n'
        '    array[3] = value\n'
        '    return array\n',
        'root', name='scalar_integer_index_store', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    conversions = [i for fn in module.functions.values()
                   for block in fn.blocks.values() for i in block.instrs
                   if i.op == 'Cast' and i.res.accounting.get('call_input_conversion')]
    assert conversions
    for instruction in conversions:
        assert instruction.res.shape == ()
        assert int(instruction.res.accounting.get('ssa_call_rank', 0)) == 0
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
        artifact = emit_ssa_function_to_llvm(module, exports[0])
    else:
        artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    if backend == 'llvm':
        artifact = compile_artifact(artifact, directory=tmp_path / 'native')
    else:
        artifact.compile(tmp_path / 'native')
    parameters = dict(module.functions[exports[0]].metadata['parameter_names'])
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameters)))
    result = subprocess.run([sys.executable, '-u', '-c', '''
import pickle, sys
import numpy as np
artifact, parameters = pickle.load(open(sys.argv[1], 'rb'))
if hasattr(artifact, 'prepare_execution'):
    prepare = artifact.prepare_execution
else:
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
for value in (0, 1, -2, 2**40):
    execution = prepare({parameters['array']: np.full(4, 91.),
                         parameters['value']: np.array([value], dtype=np.int64)}).run()
    np.testing.assert_array_equal(execution.buffers[parameters['array']], [value, value, 0., value])
''', str(saved)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_indexed_unary_scalar_rank_survives_program_abi_settlement():
    """A scalar indexed read has rank zero even without static extents."""
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'bindings': [], 'values': [{
            'function': 'root', 'parameter': parameter, 'storage': 'span',
            'python_type': 'src.common.tensors.abstraction.AbstractTensor',
            'dtype': 'float64', 'rank': 1, 'shape': [3],
        } for parameter in ('values', 'output')],
    })
    module, _, exports = lower_ast_source_to_ssa(
        'def root(values, output):\n'
        '    for i in range(3):\n'
        '        output[i] = -values[i] + abs(values[i])\n'
        '    return output\n',
        'root', name='indexed_unary_scalar', extraction_contract=policy,
    )
    unary = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {'Neg', 'Abs', 'TensorNeg', 'TensorAbs'}
    ]
    assert {instruction.op for instruction in unary} == {'Neg', 'Abs'}
    for instruction in unary:
        for value in (*instruction.args, instruction.res):
            assert value.shape == ()
            assert int(value.accounting.get('program_abi_rank', 0)) == 0
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.shortfalls == ()


def test_repository_tensor_provider_preserves_scalar_index_assignment(tmp_path):
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [{
            'function': 'root', 'parameter': 'array', 'storage': 'span',
            'python_type': 'numpy.ndarray', 'dtype': 'float64',
            'rank': 1, 'shape': [2], 'mutable': True,
        }],
    })
    module, _, exports = lower_ast_source_to_ssa(
        'def root(array, value):\n    array[0] = value\n    return array\n',
        'root', name='scalar_index_store', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    parameters = dict(module.functions[exports[0]].metadata['parameter_names'])
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameters)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameters = pickle.load(open(sys.argv[1], 'rb'))
for value in (-3.5, 0.0, 12.0):
    execution = artifact.prepare_execution({
        parameters['array']: np.array([91.0, 42.0]),
        parameters['value']: np.array([value]),
    }).run()
    np.testing.assert_array_equal(execution.buffers[parameters['array']], [value, 42.0])
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
