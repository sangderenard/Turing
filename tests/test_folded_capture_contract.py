"""A capture whose producer was folded away has no contract.

The capture contract lookup finds the enclosing value through the caller's
identity table. A binding whose producer was folded -- ``wrench_k =
inputs[:, 33].reshape(...)`` -- keeps no entry there, and captures have no
caller-side binding edge in the source graph at all: they acquire one during
call linking, later. Such a capture reaches its callee undescribed and every
shape expression over it stays unresolved.

See ``docs/COMPILER_INTERPRETATION_RULES.md`` section 6a.
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
    balloon tire, six unnamed formals each.
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
