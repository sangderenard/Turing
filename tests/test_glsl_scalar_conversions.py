"""GLSL scalar constructors ``uint(x)`` and ``double(x)`` (issue 1b349683).

Before: both lowered to bare Python calls ``uint(v)``/``double(v)``, names
nothing binds; the planner made the unknown Call a third formal of ``main``
("unaccounted_formals") on the full-native path at every expansion depth.
Now: each is the builtin conversion of its kind under the annotated GLSL
width, and ``uint(int)`` is the same-width bitcast ``int_bits_to_uint``, so
it keeps the bit pattern and later uint arithmetic wraps modulo 2**32 as GLSL
defines it (issue cb85552d).
"""
import ast

import pytest

from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast


_UINT_CONVERSION = '''#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer UWords { uint uwords[]; };
layout(std430, binding = 1) buffer IWords { int iwords[]; };
void main() {
    uwords[0] = uint(iwords[0]);
    uwords[1] = uint(iwords[1]) + 1u;
    uint t = uint(iwords[2]);
    uwords[2] = t + 1u;
    uwords[3] = uint(iwords[3]) + 1u;
    uint w = uint(iwords[4]);
    uwords[4] = w + 2u;
}'''


def _run_native(source, uwords, iwords, *, depth_zero):
    import contextlib
    import tempfile
    from pathlib import Path
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.work_contract import (
        active_contract, scoped_work_contract, with_expansion_policy,
    )
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    root = Path(__file__).resolve().parents[1]
    span = {'function': '*', 'storage': 'span', 'rank': 1,
            'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {**span, 'parameter': 'uwords', 'dtype': 'uint32'},
        {**span, 'parameter': 'iwords', 'dtype': 'int32'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    scope = contextlib.nullcontext()
    if depth_zero:
        scope = scoped_work_contract(with_expansion_policy(active_contract(), {
            'inline_depth': 0, 'custom_version_depth': 0, 'unroll_depth': 0,
        }))
    with scope:
        module, _outputs, exports = lower_ast_source_to_ssa(
            ast.unparse(result.module), 'main', extraction_contract=contract,
            python_bindings={'AbstractTensor': AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            runtime_closure_only=True, progress_bars=False)
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    prepare_artifact_execution(
        artifact, {names['uwords']: uwords, names['iwords']: iwords}).run()


def test_uint_and_double_constructors_lower_to_bound_conversions():
    source = '''#version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer UWords { uint uwords[]; };
    layout(std430, binding = 1) buffer IWords { int iwords[]; };
    void main() {
        double d = double(iwords[0]);
        uwords[0] = uint(iwords[0]) + 1u;
    }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    text = ast.unparse(result.module)
    assert ' uint(' not in text and 'double(' not in text, text
    assert "'float64' = float(" in text, text
    # The int operand's bits are reinterpreted as uint32 (a BitCast).
    assert "int_bits_to_uint(iwords[" in text, text
    assert '4294967295' not in text, text


@pytest.mark.parametrize('depth_zero', [True, False], ids=['depth0', 'default'])
def test_uint_conversion_of_int_keeps_the_bit_pattern_on_native(depth_zero):
    import numpy as np
    # uwords[3] and uwords[4]: a uint add past 2**32 after the conversion
    # wraps (issue cb85552d); it used to fault 86 at the uint32 store.
    uwords = np.array([7, 7, 7, 7, 7], dtype=np.uint32)
    iwords = np.array([-1, 41, -2, -1, -1], dtype=np.int32)
    _run_native(_UINT_CONVERSION, uwords, iwords, depth_zero=depth_zero)
    assert uwords.tolist() == [4294967295, 42, 4294967295, 0, 1]


_DOUBLE_CONVERSION = '''#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer UWords { uint uwords[]; };
layout(std430, binding = 1) buffer IWords { int iwords[]; };
void main() {
    uwords[0] = uint(double(iwords[0]) + 1.0);
}'''


@pytest.mark.parametrize('depth_zero', [True, False], ids=['depth0', 'default'])
def test_double_conversion_is_float64_on_native(depth_zero):
    # 16777216 + 1 is exact in float64 and rounds back to 16777216 in
    # float32: the result shows the width the planner chose for double().
    import numpy as np
    uwords = np.array([7], dtype=np.uint32)
    iwords = np.array([16777216], dtype=np.int32)
    _run_native(_DOUBLE_CONVERSION, uwords, iwords, depth_zero=depth_zero)
    assert uwords.tolist() == [16777217]
