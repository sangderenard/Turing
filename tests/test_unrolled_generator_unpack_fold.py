"""An unpacked generator over a static shape folds to its elements.

``lanes, capacity = (int(n) for n in x.shape)`` (the x86 read head's
``_peek``): loop evaporation publishes the generator's ordered element
producers, so each unpacked name is that element's static value and
``arange(lanes)`` has an extent.
"""
import contextlib
import io

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


_SOURCE = '''
from src.common.tensors.abstraction import AbstractTensor

def width(x):
    lanes, capacity = (int(n) for n in x.shape)
    return AbstractTensor.arange(lanes) * capacity

def f(x):
    return width(x) + 1
'''


def test_unpacked_generator_over_shape_folds_to_its_elements():
    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            _SOURCE,
            "f",
            {"x": AT.get_tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")},
            precompile_only=True,
            mutable_parameters=("x",),
        )
    assert compilation.control_shortfalls == ()
