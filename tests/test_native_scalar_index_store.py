import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


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
