import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor, tensor_identity
from src.common.tensors.autograd import GradTape, autograd


def test_forward_capture_honors_an_explicit_empty_tape():
    tape = GradTape()

    with autograd.forward_capture(tape) as active:
        assert active is tape
        with AbstractTensor.use_backend("numpy"):
            result = AbstractTensor.tensor([1.0]) + 1.0

    assert tensor_identity(result) in tape._nodes


def test_forward_capture_records_nondifferentiable_primitives():
    with autograd.forward_capture() as tape:
        with AbstractTensor.use_backend("numpy"):
            source = AbstractTensor.tensor([1.0, np.nan, np.inf])
            finite = source.isfinite()
            compared = source < 2.0
            remainder = source % 2.0

    captured_ops = [
        data["op"]
        for _, data in tape.graph.nodes(data=True)
        if data.get("kind") == "op"
    ]
    assert captured_ops.count("isfinite") == 1
    assert captured_ops.count("less") == 1
    assert captured_ops.index("isfinite") < captured_ops.index("less")
    assert tape.missing_backward_ops() == []
    assert tape.nondifferentiable_ops() == ["isfinite", "less"]
    assert finite.tolist() == [True, False, False]
    assert compared.tolist() == [True, False, False]
    assert remainder.shape == source.shape


@pytest.mark.parametrize("backend", ["numpy", "c"])
def test_smooth_canonical_unary_backward_is_connected(backend):
    autograd.tape = GradTape()
    values = np.asarray([0.5, 1.5, 3.0])
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor(values)
        source.requires_grad_(True)
        result = (
            source.exp() + source.log() + source.sqrt() + source.tanh()
        ).sum()
        gradient = autograd.grad(result, [source])[0]

    expected = (
        np.exp(values)
        + 1.0 / values
        + 0.5 / np.sqrt(values)
        + 1.0 - np.tanh(values) ** 2
    )
    np.testing.assert_allclose(gradient.tolist(), expected, rtol=1e-6)


def test_maximum_and_minimum_backward_are_connected():
    autograd.tape = GradTape()
    with AbstractTensor.use_backend("numpy"):
        left = AbstractTensor.tensor([1.0, 4.0, 3.0])
        right = AbstractTensor.tensor([2.0, 2.0, 3.0])
        left.requires_grad_(True)
        right.requires_grad_(True)
        result = (left.maximum(right) + left.minimum(right)).sum()
        left_grad, right_grad = autograd.grad(result, [left, right])

    np.testing.assert_allclose(left_grad.tolist(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(right_grad.tolist(), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("backend", ["numpy", "c"])
def test_reverse_arithmetic_uses_canonical_ops_and_backward(backend):
    autograd.tape = GradTape()
    values = np.asarray([1.0, 2.0, 4.0])
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor(values)
        source.requires_grad_(True)
        result = (2.0 * source + (5.0 - source) + 8.0 / source).sum()
        gradient = autograd.grad(result, [source])[0]

    expected = 1.0 - 8.0 / values**2
    np.testing.assert_allclose(gradient.tolist(), expected)


@pytest.mark.parametrize("backend", ["numpy", "c"])
def test_repeated_advanced_index_backward_accumulates(backend):
    autograd.tape = GradTape()
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor([1.0, 2.0, 3.0])
        source.requires_grad_(True)
        selected = source[np.asarray([2, 0, 2, 1, 2])]
        gradient = autograd.grad(selected.sum(), [source])[0]

    np.testing.assert_allclose(gradient.tolist(), [1.0, 1.0, 3.0])


@pytest.mark.parametrize("backend", ["numpy", "c"])
def test_tuple_advanced_index_forward_and_backward_share_policy(backend):
    autograd.tape = GradTape()
    values = np.arange(24.0).reshape(3, 4, 2)
    indices = np.asarray([3, 1, 3])
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor(values)
        source.requires_grad_(True)
        selected = source[:, indices, :]
        gradient = autograd.grad(selected.sum(), [source])[0]

    np.testing.assert_allclose(selected.tolist(), values[:, indices, :])
    expected = np.zeros_like(values)
    expected[:, 1, :] = 1.0
    expected[:, 3, :] = 2.0
    np.testing.assert_allclose(gradient.tolist(), expected)


@pytest.mark.parametrize("backend", ["numpy", "c", "torch"])
def test_axis_reduction_backward_restores_the_reduced_dimension(backend):
    autograd.tape = GradTape()
    weights = np.arange(8.0).reshape(2, 4) + 1.0
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor(np.arange(24.0).reshape(2, 3, 4))
        source.requires_grad_(True)
        reduced = source.sum(dim=1)
        loss = (reduced * AbstractTensor.tensor(weights)).sum()
        gradient = autograd.grad(loss, [source])[0]

    expected = np.broadcast_to(weights[:, None, :], (2, 3, 4))
    np.testing.assert_allclose(gradient.tolist(), expected)


def test_numpy_reduction_protocol_keywords_reach_abstract_tensor_methods():
    values = np.arange(24.0).reshape(2, 3, 4)
    with AbstractTensor.use_backend("numpy"):
        source = AbstractTensor.tensor(values)
        summed = np.sum(source, axis=1, keepdims=True)
        maximum = np.max(source, axis=2, keepdims=True)
        mean = np.mean(source, axis=0)

    np.testing.assert_allclose(summed.tolist(), values.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(maximum.tolist(), values.max(axis=2, keepdims=True))
    np.testing.assert_allclose(mean.tolist(), values.mean(axis=0))


@pytest.mark.parametrize("backend", ["numpy", "c"])
def test_mod_backward_matches_floor_quotient_semantics(backend):
    autograd.tape = GradTape()
    with AbstractTensor.use_backend(backend):
        left = AbstractTensor.tensor([5.5, -5.5, 3.0])
        right = AbstractTensor.tensor([2.0, 2.0, 4.0])
        left.requires_grad_(True)
        right.requires_grad_(True)
        result = (left % right).sum()
        left_grad, right_grad = autograd.grad(result, [left, right])

    np.testing.assert_allclose(left_grad.tolist(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(right_grad.tolist(), [-2.0, 3.0, 0.0])


def test_call_time_backward_override_wins_without_mutating_registry():
    autograd.tape = GradTape()
    with AbstractTensor.use_backend("numpy"):
        left = AbstractTensor.tensor([5.5, -5.5])
        right = AbstractTensor.tensor([2.0, 2.0])
        left.requires_grad_(True)
        right.requires_grad_(True)
        result = (left % right).sum()

        def numerical_hack(g, x, y):
            return AbstractTensor.zeros_like(x), AbstractTensor.zeros_like(y)

        left_grad, right_grad = autograd.grad(
            result,
            [left, right],
            backward_overrides={"mod": numerical_hack},
        )

    assert left_grad.tolist() == [0.0, 0.0]
    assert right_grad.tolist() == [0.0, 0.0]


def test_capture_tape_reports_carried_backward_override():
    def numerical_hack(g, x, y):
        return g, g

    with autograd.forward_capture(
        backward_overrides={"mod": numerical_hack}
    ) as tape:
        with AbstractTensor.use_backend("numpy"):
            source = AbstractTensor.tensor([1.0, 2.0])
            result = source % 0.75

    mod_nodes = [
        data for _, data in tape.graph.nodes(data=True)
        if data.get("kind") == "op" and data.get("op") == "mod"
    ]
    assert len(mod_nodes) == 1
    assert mod_nodes[0]["backward_status"] == "override"
    assert result.tolist() == [0.25, 0.5]
