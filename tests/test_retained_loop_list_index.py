"""A retained loop cannot index a Python list of tensors.

The escape is caused by *retention*, not by nesting. A loop is retained when it
is a coordinated recurrence -- more than one carried binding, or one carried
binding while nested -- and once retained, ``station[segment]`` indexes a
Python list of tensors by a runtime value, which has no lowering. The
destructuring temporaries the compiler created for ``r0, r1`` then become
formals with no producer.

This matters beyond the balloon tire. Carried state is what decides retention,
so any change that adds carried state to a controller moves its loops over the
multi-carried threshold and reproduces this defect there. Widening the unroll
rule would therefore be a fix that expires; making the retained case lower its
index is the one that does not.

See ``docs/COMPILER_INTERPRETATION_RULES.md`` section 10.
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

PREAMBLE = '''
def root(inputs, depth):
    station = [inputs[:, 0].reshape((-1, 1)), inputs[:, 1].reshape((-1, 1)),
               inputs[:, 2].reshape((-1, 1)), inputs[:, 3].reshape((-1, 1))]
'''

# One carried value, no nesting: the loop unrolls and the index becomes literal.
UNROLLED = PREAMBLE + '''    area = depth * 0.0
    for segment in range(3):
        area = area + station[segment] * 0.5
    return area
'''

# Two carried values: a coordinated recurrence, retained, and the index stays
# dynamic.  Nothing here is nested.
TWO_CARRIED = PREAMBLE + '''    area = depth * 0.0
    count = depth * 0.0
    for segment in range(3):
        r0, r1 = station[segment], station[segment + 1]
        area = area + (r0 + r1) * 0.5
        count = count + 1.0
    return area + count
'''

# One carried value in each of two nested loops: the inner is protected because
# it is nested, and the protection closes over its owner.  Same outcome by a
# different route.
NESTED = PREAMBLE + '''    nodes = (0.0, 0.5384693101056831, -0.5384693101056831)
    area = depth * 0.0
    for segment in range(3):
        r0, r1 = station[segment], station[segment + 1]
        segment_area = depth * 0.0
        for node in nodes:
            segment_area = segment_area + (r0 + node * (r1 - r0))
        area = area + segment_area
    return area
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
            for parameter, shape in (('inputs', (8, 4)), ('depth', (8, 1)))
        ],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, 'root', name=name, extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return module, module.functions[f'{name}__root']


def test_an_unrolled_loop_indexes_the_list_with_a_literal():
    """The baseline: one carried value, so the loop evaporates entirely."""

    module, root = _lower(UNROLLED, 'unrolled_list_index')
    assert sorted(root.blocks) == ['entry']
    assert not run_all(module)


@pytest.mark.xfail(
    reason='a retained loop indexing a Python list of tensors by its loop '
           'variable has no lowering, so the destructuring temporaries become '
           'formals with no producer; retention is what triggers it, and a '
           'second carried value is enough on its own',
    strict=False,
)
@pytest.mark.parametrize('source,name', [
    (TWO_CARRIED, 'two_carried_list_index'),
    (NESTED, 'nested_list_index'),
])
def test_a_retained_loop_can_index_its_list(source, name):
    """Retention must not make an authored list index unlowerable."""

    module, _root = _lower(source, name)
    assert not run_all(module)
