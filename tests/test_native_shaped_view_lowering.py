"""One storage identity, several shaped views, and integer-valued kernels.

Each check lowers authored source, emits and compiles native C, and compares
the executed result with the eager ``AbstractTensor`` reference.  A shape or
region-ownership mistake in these paths does not raise: it silently produces
a complete, compiling program that reads the wrong elements, so value
comparison against eager execution is the only honest gate.
"""

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_self_check import run_all

CONTRACT = 'extraction_contracts/program_extraction.yaml'


def _lower(source, name, shapes):
    policy = ExtractionContract(CONTRACT).with_program_abi({
        'bindings': [],
        'values': [
            {
                'function': 'root', 'parameter': parameter, 'storage': 'span',
                'dtype': 'float64', 'rank': len(shape), 'shape': list(shape),
                'python_type': 'src.common.tensors.abstraction.AbstractTensor',
            }
            for parameter, shape in shapes.items()
        ],
    })
    module, outputs, _exports = lower_ast_source_to_ssa(
        source, 'root', name=name, extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return module, module.functions[f'{name}__root']


def _execute(module, root, directory, feeds, reference):
    """Compile at -O0 and return the published buffers for every output."""

    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(directory, optimization='O0')
    names = dict(root.metadata['parameter_names'])
    outputs = [int(value_id) for _name, value_id in root.metadata['named_outputs']]
    packed = {names[parameter]: array.copy() for parameter, array in feeds.items()}
    for value_id, expected in zip(outputs, reference):
        packed[value_id] = np.full(np.shape(expected), np.nan)
    execution = artifact.prepare_execution(packed)
    execution.run()
    return [
        np.asarray(execution.buffers[value_id]).reshape(np.shape(expected))
        for value_id, expected in zip(outputs, reference)
    ]


def _region_calls(module, callee):
    return [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == 'Call'
        and instruction.attributes.get('callee') == callee
    ]


def test_reshaped_operand_keeps_its_view_extents_through_abi_settlement(tmp_path):
    """A reshape is a view of its source storage, not a restamp of it.

    ``b.reshape((-1, 1, 2))`` and ``b`` share one storage identity, so
    whole-module ABI propagation used to overwrite the view occurrence with
    the allocation owner's ``(4, 2)``.  The broadcast kernel then conformed
    the wrong axes and every consumer read misaligned elements.
    """

    module, root = _lower(
        'def root(a, b):\n    return a + b.reshape((-1, 1, 2))\n',
        'shaped_view', {'a': (4, 3, 2), 'b': (4, 2)},
    )
    broadcasts = _region_calls(module, 'broadcast_double')
    assert len(broadcasts) == 1
    assert tuple(broadcasts[0].args[0].shape) == (4, 1, 2)
    constants = {
        int(instruction.res.id): instruction.attributes
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == 'Const' and instruction.res is not None
    }
    source_extents = constants[int(broadcasts[0].args[2].id)]
    assert tuple(source_extents['values']) == (4, 1, 2)

    generator = np.random.default_rng(11)
    a = generator.standard_normal((4, 3, 2))
    b = generator.standard_normal((4, 2))
    expected = a + b.reshape((-1, 1, 2))
    actual, = _execute(
        module, root, tmp_path / 'shaped_view', {'a': a, 'b': b}, [expected],
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('spelling', ['a % b', 'a // b'])
def test_integer_valued_binary_operators_are_element_kernels(tmp_path, spelling):
    """``%`` and ``//`` are catalogued binary kernels, not scalar expressions.

    Without their opcode spellings both fell through to the scalar emitter,
    which computed element zero alone and left the result buffer untouched.
    Signed operands are included because the catalogue implements Python's
    floor semantics, not C truncation.
    """

    module, root = _lower(
        f'def root(a, b):\n    return {spelling}\n',
        'integer_ops', {'a': (6,), 'b': (6,)},
    )
    assert _region_calls(module, 'binary_double')

    a = np.array([7.0, -7.0, 7.0, -7.0, 5.5, -5.5])
    b = np.array([3.0, 3.0, -3.0, -3.0, 2.0, 2.0])
    reference = (
        AbstractTensor.get_tensor(a) % AbstractTensor.get_tensor(b)
        if spelling == 'a % b'
        else AbstractTensor.get_tensor(a) // AbstractTensor.get_tensor(b)
    )
    expected = np.asarray(reference.data)
    actual, = _execute(
        module, root, tmp_path / 'integer_ops', {'a': a, 'b': b}, [expected],
    )
    np.testing.assert_array_equal(actual, expected)


def test_dtype_spelling_does_not_evict_a_cast_from_its_region(tmp_path):
    """``to_dtype("int64")`` names a cast; its argument is not a data operand.

    Treating the dtype string as a non-numeric constant operand made the
    whole cast coordinator metadata, so its result became an unproduced
    region feed and read as zero.
    """

    module, root = _lower(
        'def root(a, b):\n'
        '    index = (a // b).to_dtype("int64")\n'
        '    return index * b + a\n',
        'dtype_spelling', {'a': (4, 3), 'b': (4, 3)},
    )
    assert not run_all(module)

    generator = np.random.default_rng(5)
    a = np.abs(generator.standard_normal((4, 3))) * 8.0 + 0.5
    b = np.abs(generator.standard_normal((4, 3))) + 0.5
    index = AbstractTensor.get_tensor(a) // AbstractTensor.get_tensor(b)
    expected = np.asarray(index.to_dtype('int64').data) * b + a
    actual, = _execute(
        module, root, tmp_path / 'dtype_spelling', {'a': a, 'b': b}, [expected],
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_short_circuit_reduction_operand_is_recovered_for_its_region(tmp_path):
    """A reduction reached only through ``and`` still has to be produced.

    ``pressure.isfinite().all()`` as the second operand of a short-circuit
    ``and`` is coordinator work no region owns, and structural recovery had
    no reduction case.  It therefore failed, its enclosing boolean chain
    failed with it, and the owner was left calling its own region with a
    value nothing defines -- the exact shape of the validator's undefined
    operand.
    """

    from src.compiler.fortran_c_shell import _undefined_repository_ssa_operands

    module, root = _lower(
        'def root(position, pressure):\n'
        '    finite = position.isfinite().all() and pressure.isfinite().all()\n'
        '    physical = bool(\n'
        '        finite\n'
        '        and position.abs().max() < 100.0\n'
        '        and pressure.min() >= 0.0\n'
        '    )\n'
        '    return physical\n',
        'short_circuit_reduction', {'position': (4, 3), 'pressure': (4, 3)},
    )
    assert _undefined_repository_ssa_operands(module) == ()
    reductions = [
        instruction.op
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op in {'all', 'any'}
    ]
    assert reductions, 'the recovered reduction must be produced by the owner'
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls


def test_captured_local_binds_to_its_enclosing_producer(tmp_path):
    """A capture is the enclosing value, contract included.

    ``blend`` reads ``scaled`` from its enclosing scope, so the caller must
    supply it. It is not an authored parameter, so the signature used to look
    as though it had grown a value no caller could name, and the callee had no
    contract for it at all -- which left its own result shapeless, so the whole
    tensor returned to the caller as a scalar occurrence and only one element
    was ever written.
    """

    module, root = _lower(
        'def root(a, b):\n'
        '    scaled = b * 2.0\n'
        '\n'
        '    def blend(query):\n'
        '        return query - scaled\n'
        '\n'
        '    return blend(a) + blend(a * 0.5)\n',
        'captured_local', {'a': (4, 3), 'b': (4, 3)},
    )
    assert not run_all(module)
    callee = next(
        function for name, function in module.functions.items()
        if '__blend__' in name and 'planned_region' not in name
    )
    captures = {
        entry['name']: int(entry['value_id'])
        for entry in callee.metadata['closure_formals']
    }
    assert captures.keys() == {'scaled'}
    assert dict(callee.metadata['parameter_names']).keys() == {'query'}

    generator = np.random.default_rng(29)
    a = generator.standard_normal((4, 3))
    b = generator.standard_normal((4, 3))
    scaled = b * 2.0
    expected = (a - scaled) + (a * 0.5 - scaled)
    actual, = _execute(
        module, root, tmp_path / 'captured_local', {'a': a, 'b': b}, [expected],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.xfail(
    reason='an enclosing parameter consumed only through a nested closure is '
           'dropped from the enclosing signature and replaced by one '
           'anonymous formal per callsite; reproduces identically at 3af7d206',
    strict=False,
)
def test_captured_enclosing_parameter_keeps_its_signature_slot(tmp_path):
    """Capturing a parameter must not delete it from the enclosing ABI.

    ``gain`` is an authored parameter of ``root`` read only inside ``blend``.
    The lowered ``root`` loses it and mints a fresh shapeless formal for each
    callsite instead, so no caller can supply the value and the compiled
    program reads uninitialized storage.
    """

    module, root = _lower(
        'def root(a, b, gain):\n'
        '    scaled = b * 2.0\n'
        '\n'
        '    def blend(query):\n'
        '        return (query - scaled) * gain\n'
        '\n'
        '    return blend(a) + blend(a * 0.5)\n',
        'captured_parameter', {'a': (4, 3), 'b': (4, 3), 'gain': (4, 3)},
    )
    names = dict(root.metadata['parameter_names'])
    assert set(root.metadata['authored_parameters']) == {'a', 'b', 'gain'}
    assert 'gain' in names

    generator = np.random.default_rng(31)
    a = generator.standard_normal((4, 3))
    b = generator.standard_normal((4, 3))
    gain = generator.standard_normal((4, 3))
    scaled = b * 2.0
    expected = (a - scaled) * gain + (a * 0.5 - scaled) * gain
    actual, = _execute(
        module, root, tmp_path / 'captured_parameter',
        {'a': a, 'b': b, 'gain': gain}, [expected],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.xfail(
    reason='declared integer output published as the double working '
           'representation when a tensor kernel also consumes it; the call '
           'edge has physical input adapters but no output adapter',
    strict=False,
)
def test_consumed_integer_output_is_published_in_its_declared_dtype(tmp_path):
    """A declared ``int64`` output must reach its caller as ``int64``.

    Returned alone, the cast writes ``int64_t`` storage and the exported
    buffer agrees.  Once a later tensor kernel also consumes the value its
    storage settles to ``double`` for the kernel, but the root wrapper still
    declares ``int64_t`` and the publication copies raw doubles, so the
    caller reads a reinterpreted bit pattern.
    """

    module, root = _lower(
        'def root(a, b):\n'
        '    index = (a // b).to_dtype("int64")\n'
        '    return index, index * 2.0\n',
        'integer_publication', {'a': (6,), 'b': (6,)},
    )
    a = np.array([7.0, -7.0, 7.0, 13.0, 5.5, 9.0])
    b = np.array([3.0, 3.0, -3.0, 4.0, 2.0, 2.0])
    quotient = AbstractTensor.get_tensor(a) // AbstractTensor.get_tensor(b)
    index = np.asarray(quotient.to_dtype('int64').data)
    actual, _scaled = _execute(
        module, root, tmp_path / 'integer_publication',
        {'a': a, 'b': b}, [index, index * 2.0],
    )
    np.testing.assert_array_equal(actual, index)


def test_sliced_call_result_is_owned_by_a_computing_region(tmp_path):
    """Only a literal integer index projects a call's outputs.

    A slice over a tensor call result is a numerical view some region must
    compute.  Classifying it as a call-boundary projection removed it from
    every region, leaving the caller with an unnamed formal nothing produces.
    """

    module, root = _lower(
        'def cross(left, right):\n'
        '    return AbstractTensor.stack([\n'
        '        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],\n'
        '        left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],\n'
        '        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],\n'
        '    ], dim=-1)\n'
        '\n'
        '\n'
        'def root(a, b, c):\n'
        '    moment = cross(a, b)\n'
        '    moment[:, :, 0:2] = moment[:, :, 0:2] + c\n'
        '    return moment.sum(dim=1)\n',
        'sliced_result', {'a': (4, 3, 3), 'b': (4, 3, 3), 'c': (4, 3, 2)},
    )
    assert not run_all(module)
    named = {int(value_id) for _name, value_id in root.metadata['parameter_names']}
    assert named == {int(value.id) for value in root.args} - {
        int(value.id) for value in root.args if value.accounting
    }

    generator = np.random.default_rng(17)
    a = generator.standard_normal((4, 3, 3))
    b = generator.standard_normal((4, 3, 3))
    c = generator.standard_normal((4, 3, 2))
    moment = np.cross(a, b)
    moment[:, :, 0:2] = moment[:, :, 0:2] + c
    expected = moment.sum(axis=1)
    actual, = _execute(
        module, root, tmp_path / 'sliced_result',
        {'a': a, 'b': b, 'c': c}, [expected],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
