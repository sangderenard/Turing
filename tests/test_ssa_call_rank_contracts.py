"""A shared native kernel buffer formal is not the actual's semantic shape."""
import pytest

from src.compiler.ssa_call_input_adapters import call_input_rank_updates
from src.transmogrifier.ssa import BasicBlock, Function, SSAValue


@pytest.mark.parametrize('shape', [(), (1,), (4,), (2, 3)])
def test_repository_pointer_formal_does_not_promote_or_overwrite_actual_rank(shape):
    kernel = Function('kernel', [], {'entry': BasicBlock('entry')},
                      metadata={'llvm_argument_names': ('values',)})
    actual = SSAValue(1, 'float64', shape, accounting={'physical_dtype': 'float64'})
    formal = SSAValue(582, 'ptr', (1,), accounting={
        'program_abi_storage': 'span', 'program_abi_rank': 1,
        'physical_dtype': 'float64',
    })
    assert call_input_rank_updates(kernel, actual, formal) == ()


def test_authored_internal_dynamic_span_still_propagates_rank():
    function = Function('authored', [], {})
    actual = SSAValue(1, 'float64', accounting={'program_abi_rank': 2})
    formal = SSAValue(2, 'float64')
    assert call_input_rank_updates(function, actual, formal) == ((formal, 2),)


def test_repository_by_value_formal_remains_scalar():
    function = Function('kernel', [], {}, metadata={'llvm_argument_names': ('value',)})
    assert call_input_rank_updates(function, SSAValue(1, 'float64', (1,)),
                                   SSAValue(2, 'float64'), by_value_kernel_formal=True) == ()
