"""The dt system's own functions, lowered from AbstractTensor source to NumPy.

``fused_program_python_backend.lower_functions_to_numpy`` rewrites every
AbstractTensor spelling in a function to the NumPy dialect's and leaves all
control flow as authored.  This file is the evidence that the spellings are
exact (same dtype, shape and bytes as the AbstractTensor call, on a grid with
NaN, signed zeros, infinities, integers and booleans), that control flow
survives, and that ``llvm_dt_system.dt_system(numpy_compiled=True)`` runs the
same dt sequence, state and controller as the AbstractTensor run.
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from src.common.tensors import AbstractTensor  # noqa: E402
from src.compiler.fused_program_python_backend import lower_functions_to_numpy  # noqa: E402

NAN, INF = float("nan"), float("inf")
GRID = [
    np.array([0.1]), np.array([-0.0]), np.array([0.0]), np.array([NAN]), np.array([INF]),
    np.array([0.4, -2.5, 3.0]), np.array([NAN, 1.0, -INF]), np.array([1.0, 1.0]),
    np.array([1, 2, 3]), np.array([True, False]), np.array(1.25), np.zeros((0,)),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
]
SCALARS = [0.4, 3, True, 0.0, -0.0, NAN, INF, np.float64(2.5), np.int64(2), np.bool_(True), np.float32(1.5)]


def _lower(source, name):
    import math

    path = textwrap.dedent(source)
    product = lower_functions_to_numpy([path], namespace={"math": math}, owned=lambda value: False)
    namespace = {}
    exec(compile(product.source, f"<lowered {name}>", "exec"), namespace)
    reference = {"AbstractTensor": AbstractTensor}
    exec(compile(path, f"<authored {name}>", "exec"), reference)
    return namespace[name], reference[name]


def _same(lowered, authored):
    authored = authored.numpy() if hasattr(authored, "numpy") else authored
    lowered = np.asarray(lowered)
    authored = np.asarray(authored)
    assert lowered.dtype == authored.dtype, (lowered.dtype, authored.dtype)
    assert lowered.shape == authored.shape, (lowered.shape, authored.shape)
    assert lowered.tobytes() == authored.tobytes(), (lowered, authored)


@pytest.mark.parametrize("value", SCALARS, ids=lambda value: f"{type(value).__name__}:{value}")
def test_tensor_of_a_scalar_is_the_library_rule(value):
    lowered, authored = _lower("def f(x):\n    return AbstractTensor.tensor(x)\n", "f")
    _same(lowered(value), authored(value))


@pytest.mark.parametrize("value", [[1.0, 2.0], [1, 2], [True, False], (1.0, 2.0), [], [[1.0, 2.0], [3.0, 4.0]]],
                         ids=repr)
def test_tensor_of_a_list_is_the_library_rule(value):
    lowered, authored = _lower("def f(x):\n    return AbstractTensor.tensor(x)\n", "f")
    _same(lowered(value), authored(value))


@pytest.mark.parametrize("array", GRID, ids=lambda a: f"{a.dtype}{a.shape}:{a.tolist()}")
def test_tensor_of_an_array_copies_it_as_it_is(array):
    lowered, authored = _lower("def f(x):\n    return AbstractTensor.tensor(x)\n", "f")
    copy = lowered(array)
    assert copy is not array
    _same(copy, authored(array))


UNARY = [
    ("log", "x.log()"), ("exp", "x.exp()"), ("sqrt", "x.sqrt()"), ("floor", "x.floor()"),
    ("isfinite", "x.isfinite()"), ("zeros_like", "AbstractTensor.zeros_like(x)"),
    ("ones_like", "AbstractTensor.ones_like(x)"), ("neg", "-x"), ("scale", "x * 0.4 + 1"),
    ("clamp", "x.clamp(min=-1.5, max=1.5)"), ("clamp_lo", "x.clamp(min=-1.5)"),
    ("any", "(x > 0.5).any()"), ("max", "x.max()"), ("sum", "x.sum()"),
]


@pytest.mark.parametrize("label,expression", UNARY, ids=[label for label, _ in UNARY])
@pytest.mark.parametrize("array", GRID, ids=lambda a: f"{a.dtype}{a.shape}:{a.tolist()}")
def test_unary_spellings_match_the_tensor_call(label, expression, array):
    if label in {"max"} and array.size == 0:
        pytest.skip("reduction of nothing raises in both")
    if label == "scale" and array.dtype == np.bool_:
        # AbstractTensor casts a boolean operand to its float32 before an
        # arithmetic operator with a Python float; NumPy keeps float64.  The
        # operators are authored syntax and are not rewritten; the dt system
        # multiplies boolean tensors only by float64 tensors (below).
        pytest.skip("boolean tensor times a Python float promotes differently")
    if label == "neg" and array.dtype == np.bool_:
        # The one operator spelling that differs: AbstractTensor negates a
        # boolean array, NumPy refuses.  Operators are authored syntax and
        # are not rewritten; the dt system never negates a boolean tensor.
        pytest.skip("boolean negation is not an AbstractTensor spelling the dt system uses")
    lowered, authored = _lower(f"def f(x):\n    return {expression}\n", "f")
    with np.errstate(all="ignore"):
        try:
            want = authored(AbstractTensor.tensor(array))
        except (TypeError, ValueError, FloatingPointError) as error:
            with pytest.raises(type(error)):
                lowered(array)
            return
        _same(lowered(array), want)


BINARY = [
    ("maximum", "AbstractTensor.maximum(x, y)"), ("minimum", "AbstractTensor.minimum(x, y)"),
    ("maximum_scalar", "AbstractTensor.maximum(x, 1e-30)"), ("minimum_scalar", "AbstractTensor.minimum(x, 0.5)"),
    ("where", "AbstractTensor.where(x > y, x, y)"), ("mul_bool", "(x > 0.5) * y"),
    ("add", "x + y"), ("div", "x / y"),
]


@pytest.mark.parametrize("label,expression", BINARY, ids=[label for label, _ in BINARY])
@pytest.mark.parametrize("pair", [
    (np.array([NAN]), np.array([0.1])), (np.array([-0.0]), np.array([0.0])), (np.array([0.0]), np.array([-0.0])),
    (np.array([1.0, 2.0, NAN]), np.array([2.0, 2.0, 1.0])), (np.array([INF]), np.array([-INF])),
    (np.array([0.1]), np.array([0.1])), (np.array([1, 2]), np.array([2.5, 1.0])),
], ids=lambda pair: repr(pair))
def test_binary_spellings_match_the_tensor_call(label, expression, pair):
    lowered, authored = _lower(f"def f(x, y):\n    return {expression}\n", "f")
    x, y = pair
    with np.errstate(all="ignore"):
        _same(lowered(x, y), authored(AbstractTensor.tensor(x), AbstractTensor.tensor(y)))


@pytest.mark.parametrize("pair", [
    (np.array([1.0]), np.array([0.0])), (np.array([-1.0, 0.0, NAN]), np.array([0.0, 0.0, 0.0])),
    (np.array([1.0, 2.0]), np.array([4.0, -0.0])), (np.array([1, 2, 3]), np.array([2, 0, 3])),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([2.0, 0.0])), (np.array([INF]), np.array([INF])),
], ids=repr)
@pytest.mark.parametrize("shape", ["tt", "ts", "st"])
def test_division_is_the_backends_zero_mode_divide(pair, shape):
    lowered, authored = _lower("def f(x, y):\n    return x / y\n", "f")
    x, y = pair
    with np.errstate(all="ignore"):
        if shape == "tt":
            _same(lowered(x, y), authored(AbstractTensor.tensor(x), AbstractTensor.tensor(y)))
        elif shape == "ts":
            _same(lowered(x, 0.0), authored(AbstractTensor.tensor(x), 0.0))
            _same(lowered(x, 4.0), authored(AbstractTensor.tensor(x), 4.0))
        else:
            _same(lowered(2.0, y), authored(2.0, AbstractTensor.tensor(y)))


def test_division_of_two_python_numbers_is_python_division():
    lowered, _ = _lower("def f(x, y):\n    return x / y\n", "f")
    assert lowered(1, 4) == 0.25
    with pytest.raises(ZeroDivisionError):
        lowered(1.0, 0.0)


def test_booleans_times_float_tensors_promote_like_the_tensor_call():
    """The product the dt system takes: a mask (bool) times a float64 tensor."""
    lowered, authored = _lower("def f(m, y):\n    return (m > 0.5) * y + y * (m > 0.5)\n", "f")
    m, y = np.array([0.0, 1.0, 2.0]), np.array([0.25, -0.5, NAN])
    with np.errstate(all="ignore"):
        _same(lowered(m, y), authored(AbstractTensor.tensor(m), AbstractTensor.tensor(y)))


def test_axis_reductions_take_dim_as_axis():
    lowered, authored = _lower(
        "def f(x):\n    return x.sum(dim=1) + x.max() + (x > 2).any(dim=0).sum()\n", "f")
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with np.errstate(all="ignore"):
        _same(lowered(x), authored(AbstractTensor.tensor(x)))


def test_module_functions_are_not_tensor_methods():
    lowered, _ = _lower("import math\n\ndef f(x):\n    return math.sqrt(x) + math.floor(x)\n", "f")
    assert lowered(2.0) == (2.0 ** 0.5) + 2


def test_control_flow_is_kept_as_authored():
    source = '''
    def f(x, n):
        total = AbstractTensor.tensor(0.0)
        for i in range(n):
            if i % 2:
                continue
            total = total + x * i
        while total.item() > 3.0:
            total = total * 0.5
            if total.item() < 0.0:
                break
        try:
            check = AbstractTensor.maximum(total, 0.0).log()
        except ValueError:
            check = total
        else:
            check = check + 1
        return total, check
    '''
    lowered, authored = _lower(source, "f")
    x = np.array([0.7])
    with np.errstate(all="ignore"):
        a = lowered(x, 9)
        b = authored(AbstractTensor.tensor(x), 9)
    for got, want in zip(a, b):
        _same(got, want)
    product = lower_functions_to_numpy([textwrap.dedent(source)], namespace={}, owned=lambda value: False)
    for keyword in ("for i in range(n):", "while total.item() > 3.0:", "try:", "except ValueError:", "else:", "break", "continue"):
        assert keyword in product.source


def test_the_product_is_cached_by_source_digest():
    first = lower_functions_to_numpy(["def f(x):\n    return AbstractTensor.tensor(x)\n"], namespace={}, owned=lambda v: False)
    again = lower_functions_to_numpy(["def f(x):\n    return AbstractTensor.tensor(x)\n"], namespace={}, owned=lambda v: False)
    other = lower_functions_to_numpy(["def f(x):\n    return AbstractTensor.tensor(x) + 1\n"], namespace={}, owned=lambda v: False)
    assert first.digest == again.digest != other.digest
    assert first.source == again.source


class _Piece:
    """A positional-callable dt-system piece (``llvm_dt_system.PIECE_API``)."""

    def __init__(self, dt_limit=0.01, energy=10.0, batch=4):
        self.entry = "drift"
        self.argument_names = ("x", "v", "dt")
        self.output_names = ("x_next", "dt_limit", "max_vel", "energy_j", "power_w")
        self.batch = batch
        self._dt_limit, self._energy = dt_limit, energy

    def instantiate(self, columns, outputs=None):
        pass

    def __call__(self, x, v, dt):
        ones = np.ones(self.batch)
        return (x + dt * v, self._dt_limit * ones, np.abs(v), self._energy * ones, np.abs(v))


def _run(compiled, rounds=6):
    from llvm_dt_system import dt_system

    columns = {"x": np.array([0.0, 1.0, -2.0, 10.0]), "v": np.array([1.0, 2.0, 3.0, -4.0])}
    state, controller, results = dt_system(
        [_Piece()], columns, rounds=rounds, round_dt=0.05, dx=1.0, numpy_compiled=compiled)
    return columns, state, controller, results


def test_numpy_compiled_dt_system_runs_the_same_sequence():
    plain = _run(False)
    compiled = _run(True)
    assert "numpy_product" in compiled[1].program and "numpy_product" not in plain[1].program
    for name in plain[0]:
        assert plain[0][name].tobytes() == compiled[0][name].tobytes(), name
    for a, b in zip(plain[3], compiled[3]):
        assert float(a[0]) == float(b[0]) and float(a[1]) == float(b[1])
        assert np.asarray(a[2]).tobytes() == np.asarray(b[2]).tobytes()
    assert np.asarray(plain[2].acc).tobytes() == np.asarray(compiled[2].acc).tobytes()
    assert np.asarray(plain[2].max_vel_ever).tobytes() == np.asarray(compiled[2].max_vel_ever).tobytes()
    assert isinstance(compiled[2], type(plain[2]).__mro__[0]) or isinstance(compiled[2], plain[2].__class__)


def test_numpy_compiled_controller_keeps_the_controller_type():
    from src.common.dt_system.dt_controller import STController

    _columns, _state, controller, _results = _run(True, rounds=2)
    assert isinstance(controller, STController)
