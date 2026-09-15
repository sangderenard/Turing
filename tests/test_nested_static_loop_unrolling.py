"""Two static loops, nested, and the list index inside them.

Each loop here is individually unrollable. Nested, both are preserved as
coordinated recurrences, and the outer loop's Python-list index by its own
loop variable then has no lowering at all: the destructuring temporaries the
compiler created for ``r0, r1`` become formals with no producer.

See ``docs/COMPILER_INTERPRETATION_RULES.md`` section 9.
"""

import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_self_check import run_all

CONTRACT = 'extraction_contracts/program_extraction.yaml'

NESTED = '''
def root(inputs, depth):
    batch_count = inputs.shape[0]
    station_r = [inputs[:, 0].reshape((-1, 1)), inputs[:, 1].reshape((-1, 1)),
                 inputs[:, 2].reshape((-1, 1)), inputs[:, 3].reshape((-1, 1))]
    nodes = (0.0, 0.5384693101056831, -0.5384693101056831)
    area = depth * 0.0
    for segment in range(3):
        r0, r1 = station_r[segment], station_r[segment + 1]
        segment_area = depth * 0.0
        for node in nodes:
            t = (node + 1.0) / 2.0
            radius_here = (r0 + t * (r1 - r0)).reshape((batch_count, 1))
            segment_area = segment_area + radius_here
        area = area + segment_area
    return area
'''

SINGLE = '''
def root(inputs, depth):
    batch_count = inputs.shape[0]
    r0 = inputs[:, 0].reshape((-1, 1))
    r1 = inputs[:, 1].reshape((-1, 1))
    nodes = (0.0, 0.5384693101056831, -0.5384693101056831)
    segment_area = depth * 0.0
    for node in nodes:
        t = (node + 1.0) / 2.0
        radius_here = (r0 + t * (r1 - r0)).reshape((batch_count, 1))
        segment_area = segment_area + radius_here
    return segment_area
'''


def _lower(source, name):
    policy = ExtractionContract(CONTRACT).with_program_abi({
        'bindings': [],
        'values': [
            {
                'function': 'root', 'parameter': parameter, 'storage': 'span',
                'dtype': 'float64', 'rank': len(shape), 'shape': list(shape),
                'python_type': 'src.common.tensors.abstraction.AbstractTensor',
            }
            for parameter, shape in (('inputs', (8, 8)), ('depth', (8, 1)))
        ],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, 'root', name=name, extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return module, module.functions[f'{name}__root']


def test_a_single_static_loop_unrolls_and_leaves_no_control():
    """The baseline: one static loop evaporates completely."""

    module, root = _lower(SINGLE, 'single_static_loop')
    assert sorted(root.blocks) == ['entry']
    assert not run_all(module)


@pytest.mark.xfail(
    reason='a nested loop with one carried binding is protected as a '
           'coordinated recurrence and its owner with it, so the outer list '
           'index by a retained loop variable has no lowering and the '
           'destructuring temporaries become formals with no producer',
    strict=False,
)
def test_nested_static_loops_unroll_their_list_index():
    """Nesting alone must not turn two unrollable loops into retained ones."""

    module, root = _lower(NESTED, 'nested_static_loops')
    assert sorted(root.blocks) == ['entry']
    assert not run_all(module)


WRENCH = '''
def root(a, b):
    plane_point_q = b.reshape((2, 2, 1, 3))
    wrench_k = b[:, 0, 0].reshape((2, 1, 1))

    def wrench(query):
        radial = query[:, :, :, 0:2] - plane_point_q[:, :, :, 0:2]
        return wrench_k.reshape(wrench_k.shape[:2] + (1,)) * radial.sum(dim=-1)

    return wrench(a) + wrench(a * 0.5)
'''


@pytest.mark.xfail(
    reason='a capture whose producer was folded away keeps no entry in the '
           "caller's identity table and has no caller-side binding edge at "
           'graph level, so the callee has no contract for it and every shape '
           'expression over it escapes as an anonymous formal',
    strict=False,
)
def test_capture_of_a_folded_binding_still_has_a_contract():
    """``wrench_k.shape[:2]`` inside a closure over a reshaped local.

    This is the shape of the two ``_wrench_force`` specializations in the
    balloon tire. A capture the caller still names resolves; one whose
    producer was folded does not, because captures acquire their caller-side
    binding during call linking rather than in the source graph.
    """

    policy = ExtractionContract(CONTRACT).with_program_abi({
        'bindings': [],
        'values': [
            {
                'function': 'root', 'parameter': parameter, 'storage': 'span',
                'dtype': 'float64', 'rank': len(shape), 'shape': list(shape),
                'python_type': 'src.common.tensors.abstraction.AbstractTensor',
            }
            for parameter, shape in (('a', (2, 2, 4, 3)), ('b', (2, 2, 3)))
        ],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        WRENCH, 'root', name='folded_capture', extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    assert not run_all(module)
