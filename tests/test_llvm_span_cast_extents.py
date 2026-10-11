"""A Cast's data edge preserves runtime extents through internal call frames."""
import pickle
import subprocess
import sys

import pytest

from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact


def cast_module(rank=1, *, depth=1, source_bound=True):
    shape = tuple(f'axis_{axis}' for axis in range(rank))
    source = SSAValue(0, 'int64', shape)
    output = SSAValue(1, 'float64', shape)
    formal = SSAValue(10, 'int64', shape)
    converted = SSAValue(11, 'float64', shape)
    region = Function('region', [formal] if source_bound else [], {
        'entry': BasicBlock('entry', [Instr('Cast', [formal], converted),
                                     Instr('Ret', [converted], None)])})
    functions = {'region': region}
    callee = 'region'
    if depth == 2:
        middle_input, middle_output = SSAValue(20, 'int64', shape), SSAValue(21, 'float64', shape)
        middle = Function('middle', [middle_input], {
            'entry': BasicBlock('entry', [
                Instr('Call', [middle_input], middle_output, attributes={'callee': 'region'}),
                Instr('Ret', [middle_output], None)])})
        functions['middle'] = middle
        callee = 'middle'
    root = Function('root', [source], {
        'entry': BasicBlock('entry', [
            Instr('Call', [source] if source_bound else [], output, attributes={'callee': callee}),
            Instr('Ret', [output], None)])})
    functions['root'] = root
    return IRModule(functions)


@pytest.mark.parametrize('shape', [(6,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize('depth', [1, 2])
def test_native_span_cast_uses_all_public_axes_without_precision_specific_metadata(tmp_path, shape, depth):
    module = cast_module(len(shape), depth=depth)
    artifact = emit_ssa_function_to_llvm(module, 'root')
    assert artifact.complete, artifact.shortfalls
    assert artifact.extent_order == tuple((0, 'dim', axis) for axis in range(len(shape)))
    # The cast-only region must receive the vector, not just have slots added
    # to the public ABI while its internal LLVM prototype omits %extents.
    assert 'ptr %extents' in artifact.llvm_ir
    native = compile_artifact(artifact, directory=tmp_path / 'native')
    payload = tmp_path / 'native.pkl'
    payload.write_bytes(pickle.dumps((native, shape)))
    child = subprocess.run([sys.executable, '-u', '-c', '''
import pickle, sys
import numpy as np
from src.compiler.ssa_llvm_backend import prepare_artifact_execution
artifact, shape = pickle.load(open(sys.argv[1], 'rb'))
values = (np.arange(np.prod(shape), dtype=np.int64) - 5).reshape(shape)
for offset in (0, 2**40, -2**40):
    source = values + offset
    execution = prepare_artifact_execution(artifact, {0: source.copy()}, shapes={1: shape})
    execution.run()
    np.testing.assert_array_equal(execution.buffers[1], source.astype(np.float64))
print('SPAN_CAST=exact')
''', str(payload)], capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr


def test_missing_public_extent_origin_is_still_a_shortfall_not_a_scalar_standin():
    artifact = emit_ssa_function_to_llvm(cast_module(source_bound=False), 'root')
    assert not artifact.complete
    assert any(item.operation == 'Cast' and 'needs its extents' in item.reason
               for item in artifact.shortfalls)
