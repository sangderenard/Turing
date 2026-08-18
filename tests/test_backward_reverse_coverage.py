"""Check the reverses that were missing, on the tape, against two references.

The tape is the autograd system, so an operation without a registered reverse
is not slow or approximate -- it has no gradient at all, silently. These
operations were all already forward primitives (``ELEMENTWISE_UNARY`` carries
the trig and hyperbolic family, and the C, LLVM and Fortran lanes all lower
them) while having no entry in ``BACKWARD_RULES``.

Every rule here is checked twice: against the derivative written out by hand,
and against a central finite difference taken through the same forward. One
alone is not enough -- a rule transcribed wrongly from the maths agrees with
neither, but a rule that merely *reads* right agrees with the analytic form
while still disagreeing with what the forward actually computes.

The structural rules get non-uniform output weights on purpose. Under a plain
``.sum()`` every output position carries weight 1, so a reverse that scatters
gradient to the *wrong* positions still returns the right answer, and the
whole test passes while the rule is backwards.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.autograd import autograd
from src.common.tensors.backward import BACKWARD_REGISTRY
from src.common.tensors.backward_registry import (
    BACKWARD_RULES,
    pad_vjp,
    repeat_interleave_vjp,
    repeat_vjp,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations as T


def _taped_gradient(values, call, weights=None):
    """The gradient the tape produces for ``call`` at ``values``."""

    with autograd.forward_capture():
        x = T.tensor(values)
        x.requires_grad_(True)
        produced = call(x)
        loss = produced.sum() if weights is None else (produced * T.tensor(weights)).sum()
    (gradient,) = autograd.grad(loss, [x], allow_unused=False)
    return np.asarray(gradient.tolist(), dtype=np.float64)


def _finite_difference(values, call, weights=None, step=1e-6):
    """Central difference through the same forward, one entry at a time."""

    out = np.zeros_like(values)
    flat = out.reshape(-1)
    for index in range(values.size):
        up, down = values.copy(), values.copy()
        up.reshape(-1)[index] += step
        down.reshape(-1)[index] -= step
        with autograd.no_grad():
            def evaluate(sample):
                produced = call(T.tensor(sample))
                if weights is not None:
                    produced = produced * T.tensor(weights)
                return float(np.asarray(produced.sum().tolist()))

            flat[index] = (evaluate(up) - evaluate(down)) / (2 * step)
    return out


# -- the inverse-trigonometric and hyperbolic family ----------------------

_ELEMENTWISE = {
    "asin": (np.array([-0.6, -0.1, 0.2, 0.7]), lambda v: v.asin(),
             lambda x: 1.0 / np.sqrt(1 - x * x)),
    "acos": (np.array([-0.6, -0.1, 0.2, 0.7]), lambda v: v.acos(),
             lambda x: -1.0 / np.sqrt(1 - x * x)),
    "atan": (np.array([-2.0, -0.3, 0.4, 3.0]), lambda v: v.atan(),
             lambda x: 1.0 / (1 + x * x)),
    "sinh": (np.array([-1.2, -0.3, 0.4, 1.1]), lambda v: v.sinh(), np.cosh),
    "cosh": (np.array([-1.2, -0.3, 0.4, 1.1]), lambda v: v.cosh(), np.sinh),
    "asinh": (np.array([-1.2, -0.3, 0.4, 1.1]), lambda v: v.asinh(),
              lambda x: 1.0 / np.sqrt(x * x + 1)),
    "acosh": (np.array([1.4, 2.0, 3.0, 5.0]), lambda v: v.acosh(),
              lambda x: 1.0 / np.sqrt(x * x - 1)),
    "atanh": (np.array([-0.6, -0.1, 0.2, 0.7]), lambda v: v.atanh(),
              lambda x: 1.0 / (1 - x * x)),
}


@pytest.mark.parametrize("name", sorted(_ELEMENTWISE))
def test_the_trig_and_hyperbolic_family_has_a_reverse_that_is_correct(name):
    values, call, derivative = _ELEMENTWISE[name]

    assert name in BACKWARD_REGISTRY._methods, f"{name} has no registered reverse"

    taped = _taped_gradient(values, call)
    np.testing.assert_allclose(taped, derivative(values), rtol=0, atol=1e-9)
    np.testing.assert_allclose(taped, _finite_difference(values, call), rtol=0, atol=1e-7)


def test_the_composed_reciprocal_trig_operations_are_unblocked_by_the_family():
    """sech/csch/coth record sinh and cosh, so they had no gradient either."""

    values = np.array([0.4, 0.9, 1.3])
    taped = _taped_gradient(values, lambda v: v.coth())
    # d/dx coth(x) = -csch^2(x) = -1 / sinh^2(x)
    np.testing.assert_allclose(
        taped, -1.0 / np.sinh(values) ** 2, rtol=0, atol=1e-9
    )


# -- the structural reverses ----------------------------------------------

def _structural_cases():
    rng = np.random.default_rng(7)
    return [
        ("pad-1d", np.array([1.0, 2.0, 3.0]),
         lambda x: x.pad((1, 2), value=0.0), (6,)),
        # Asymmetric on both axes: a reverse that reads the pad tuple in the
        # wrong dimension order still produces the right shape here, and the
        # wrong numbers.
        ("pad-2d-asymmetric", rng.normal(size=(2, 3)),
         lambda x: x.pad((1, 0, 0, 2), value=0.0), (4, 4)),
        ("repeat-1d", np.array([1.0, 2.0, 3.0]),
         lambda x: x.repeat(2), (6,)),
        ("repeat-2d-dim0", rng.normal(size=(2, 3)),
         lambda x: x.repeat(3, dim=0), (6, 3)),
        ("repeat-2d-dim1", rng.normal(size=(2, 3)),
         lambda x: x.repeat(2, dim=1), (2, 6)),
        ("interleave-1d", np.array([1.0, 2.0, 3.0]),
         lambda x: x.repeat_interleave(2), (6,)),
        # Count differs from extent, so tiling and interleaving cannot be
        # confused for one another without the numbers changing.
        ("interleave-2d-dim0", rng.normal(size=(2, 3)),
         lambda x: x.repeat_interleave(4, dim=0), (8, 3)),
        ("interleave-2d-dim1", rng.normal(size=(2, 3)),
         lambda x: x.repeat_interleave(3, dim=1), (2, 9)),
    ]


@pytest.mark.parametrize(
    ("label", "values", "call", "out_shape"), _structural_cases(),
    ids=[case[0] for case in _structural_cases()],
)
def test_structural_reverses_land_gradient_on_the_right_positions(
    label, values, call, out_shape
):
    weights = np.random.default_rng(11).normal(size=out_shape)

    taped = _taped_gradient(values, call, weights)
    assert taped.shape == values.shape
    np.testing.assert_allclose(
        taped, _finite_difference(values, call, weights), rtol=0, atol=1e-6
    )


def test_tiling_and_interleaving_adjoints_are_genuinely_different():
    """The two differ only in which half of the split axis is summed."""

    source = T.tensor(np.zeros((2, 3)))
    gradient = T.tensor(np.arange(6.0).reshape(6, 1) * np.ones((1, 3)))

    tiled = np.asarray(repeat_vjp(gradient, source, 3, 0).tolist())
    interleaved = np.asarray(
        repeat_interleave_vjp(gradient, source, 3, 0).tolist()
    )
    assert tiled.shape == interleaved.shape == (2, 3)
    assert not np.allclose(tiled, interleaved)


# -- refusals -------------------------------------------------------------

def test_a_non_constant_pad_mode_refuses_rather_than_cropping():
    """reflect/replicate fold onto interior positions; a crop is not their adjoint."""

    source = T.tensor(np.array([1.0, 2.0, 3.0]))
    gradient = T.tensor(np.ones(6))
    with pytest.raises(NotImplementedError, match="scatter-add"):
        pad_vjp(gradient, source, (1, 2), "reflect")


def test_a_negative_pad_width_refuses_rather_than_guessing():
    source = T.tensor(np.array([1.0, 2.0, 3.0]))
    gradient = T.tensor(np.ones(2))
    with pytest.raises(NotImplementedError, match="forward crop"):
        pad_vjp(gradient, source, (-1, 0), "constant")


@pytest.mark.parametrize("count", [0, -2])
def test_a_non_positive_repeat_count_refuses(count):
    source = T.tensor(np.array([1.0, 2.0, 3.0]))
    gradient = T.tensor(np.ones(3))
    with pytest.raises(ValueError, match="positive count"):
        repeat_vjp(gradient, source, count, 0)
    with pytest.raises(ValueError, match="positive count"):
        repeat_interleave_vjp(gradient, source, count, 0)


# -- the registry itself --------------------------------------------------

def test_a_broad_program_leaves_no_unexplained_hole_on_the_tape():
    """``missing_backward_ops`` is only a useful signal if it is normally empty.

    Every recorded operation is classified as having a reverse, being a source
    that manufactures values, or being deliberately nondifferentiable. Anything
    else reports as "missing", so a report full of ops nobody intends to
    differentiate hides the one that matters. This exercises the linear
    algebra, the trig family, the structural moves and the index-producing
    operations together and expects nothing left over.
    """

    def value(raw):
        return T.tensor(np.asarray(raw, dtype=np.float64))

    spd = [[4.0, 1.0], [1.0, 3.0]]
    square = [[1.0, 2.0], [3.0, 5.0]]

    with autograd.forward_capture() as tape:
        T.eigh(value(spd))
        T.cholesky(value(spd))
        T.solve(value(square), value([1.0, 2.0]))
        T.det(value(square))
        T.norm(value([0.2, 0.5]))
        T.outer(value([0.2, 0.5]), value([0.2, 0.5]))
        T.einsum("ij,jk->ik", value(square), value(square))
        value([1.5, 2.5, 3.5]).index_select(0, value([0.0, 2.0]).long())
        value([0.2, 0.5]).asin().sinh().atanh().sech()
        value([1.0, 2.0, 3.0]).pad((1, 1), value=0.0).repeat(2).repeat_interleave(2)
        T.searchsorted(value([0.0, 1.0]), value([0.2, 0.9]))
        value([0.2, -0.5]).sign()
        value([0.2, 0.5]).softmax(dim=0).log()

    assert tape.missing_backward_ops() == []


def test_the_widening_and_narrowing_casts_are_deliberately_asymmetric():
    """One is an identity on the value; the other is piecewise constant."""

    from src.common.tensors.autograd import _INTENTIONALLY_NONDIFFERENTIABLE

    assert {"sitofp", "uitofp"} <= set(BACKWARD_REGISTRY._methods)
    assert {"long", "int", "fptosi", "fptoui", "int_trunc"} <= (
        _INTENTIONALLY_NONDIFFERENTIABLE
    )
    assert not {"sitofp", "uitofp"} & _INTENTIONALLY_NONDIFFERENTIABLE


def test_a_classified_operation_does_not_block_a_backward_capture():
    """``sign`` in the forward must not refuse a capture the tape can perform.

    ``capture_backward_program`` counted every op with no registered rule as
    missing, including the ones deliberately classified as nondifferentiable or
    as sources. That refused whole programs over a ``sign`` inside ``eigh`` or a
    ``zeros_like`` inside ``pad`` -- operations that contribute no backward step
    and that ``autograd.grad`` already walks straight past.
    """

    from src.common.tensors.abstraction import AbstractTensor as AT
    from src.common.tensors.abstract_nn.fused_program import (
        ProgramRunner,
        capture_backward_program,
    )
    from src.common.tensors.autograd import GradTape

    autograd.tape = GradTape()
    values = AT.tensor((0.3, 0.6, 0.9))
    values.requires_grad_(True)
    loss = (values.asin() * values.sign()).sum()

    captured = capture_backward_program(loss, (values,))
    assert captured.missing_backward == ()

    replayed = ProgramRunner(captured.program)(captured.feed_values)
    produced = np.asarray(replayed["grad_0"].tolist(), dtype=np.float64)

    autograd.tape = GradTape()
    again = AT.tensor((0.3, 0.6, 0.9))
    again.requires_grad_(True)
    (eager,) = autograd.grad(
        (again.asin() * again.sign()).sum(), [again], allow_unused=False
    )
    np.testing.assert_allclose(produced, np.asarray(eager.tolist()), atol=0)


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("pad", lambda x: x.pad((1, 1), value=0.0).sum()),
        ("repeat", lambda x: x.repeat(2).sum()),
        ("interleave", lambda x: x.repeat_interleave(3).sum()),
        ("hyperbolic", lambda x: x.sinh().tanh().atanh().sum()),
        ("reciprocal", lambda x: x.coth().sum()),
    ],
)
def test_the_new_reverses_replay_as_programs_and_not_only_eagerly(label, build):
    """A rule that only runs eagerly cannot reach the compiler.

    ``capture_backward_program`` records the backward pass and hands back a
    ``FusedProgram``, which is what a backend lowers. A reverse written out of
    operations the runner cannot execute captures and then dies at replay, so
    each rule is checked the whole way through rather than only under
    ``autograd.grad``.
    """

    from src.common.tensors.abstraction import AbstractTensor as AT
    from src.common.tensors.abstract_nn.fused_program import (
        ProgramRunner,
        capture_backward_program,
    )
    from src.common.tensors.autograd import GradTape

    sample = (0.4, 0.9, 1.3)

    autograd.tape = GradTape()
    values = AT.tensor(sample)
    values.requires_grad_(True)
    captured = capture_backward_program(build(values), (values,))
    assert captured.missing_backward == ()
    replayed = ProgramRunner(captured.program)(captured.feed_values)
    produced = np.asarray(replayed["grad_0"].tolist(), dtype=np.float64)

    autograd.tape = GradTape()
    again = AT.tensor(sample)
    again.requires_grad_(True)
    (eager,) = autograd.grad(build(again), [again], allow_unused=False)

    np.testing.assert_allclose(
        produced, np.asarray(eager.tolist(), dtype=np.float64), atol=0
    )


def test_a_genuinely_unknown_operator_still_refuses_a_capture():
    """Widening the accounting must not turn the guard off."""

    from src.common.tensors.abstraction import AbstractTensor as AT
    from src.common.tensors.abstract_nn.fused_program import (
        capture_backward_program,
    )
    from src.common.tensors.autograd import GradTape

    autograd.tape = GradTape()
    values = AT.tensor((1.0, 2.0, 3.0))
    values.requires_grad_(True)
    with autograd.no_grad():
        opaque = values * 1.0
    opaque.requires_grad_(True)
    autograd.capture_all = True
    try:
        autograd.record("test_unknown_operator", (values,), opaque)
    finally:
        autograd.capture_all = False
    loss = opaque.sum()

    with pytest.raises(RuntimeError, match="test_unknown_operator"):
        capture_backward_program(loss, (values,))


def test_every_new_rule_registered_and_declares_its_domain():
    """A rule that registers but states no domain is a trap for the next reader."""

    added = {
        "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh",
        "pad", "repeat", "repeat_interleave", "sitofp", "uitofp",
    }
    assert added <= set(BACKWARD_RULES)
    assert added <= set(BACKWARD_REGISTRY._methods)
    for name in sorted(added):
        rule = BACKWARD_RULES[name]
        assert rule["domain"], f"{name} states no domain"
        assert rule["python"]["body"], f"{name} has no executable body"
