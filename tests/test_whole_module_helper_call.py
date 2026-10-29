"""A helper call in the whole-module graph keeps its source authority.

``compile_ast_aot`` plans the whole-module graph, which executes the entry's
body, so the caller of ``g`` below is that module graph rather than ``f``'s
function graph.  Its numeric scope is book-minted (synthetic).  The call's
reachability row and the caller's numeric specialization must still be
read from, and derived from, real book cells.
"""
import contextlib
import io

import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


_SOURCE = '''
def g(observed, bit):
    return observed + bit

def f(x):
    return g(x, 4)
'''

_SOURCE_USED_RESULT = '''
def g(observed, bit):
    return observed + bit

def f(x):
    y = g(x, 4)
    return y * 2
'''


@pytest.mark.parametrize("source", [_SOURCE, _SOURCE_USED_RESULT])
def test_whole_module_helper_call_plans(source):
    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            source,
            "f",
            {"x": AT.get_tensor([1, 5], dtype="int64")},
            precompile_only=True,
            mutable_parameters=("x",),
        )
    assert compilation.control_shortfalls == ()
