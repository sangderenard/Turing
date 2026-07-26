import numpy as np
import pytest

from src.common.tensors.abstract_nn_graph_core import AbstractNNGraphCore
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend import CTensorOperations

def test_nn_graph_complete_and_cffi_forward_pass():
    segments = [
        {"label": "input", "nodes": ["i1", "i2"]},
        {"label": "hidden", "num_nodes": 2},
    ]
    segmap = AbstractNNGraphCore.create_segment_map(segments)
    graph = AbstractNNGraphCore()
    dummy = type("NN", (), {"segments": segmap, "in_nodes": ["i1", "i2"], "out_nodes": ["hidden_0", "hidden_1"]})
    graph._register_NN(dummy)
    assert graph.is_graph_complete()

    ops = CTensorOperations()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=float)
    expected = A @ B
    A_tensor = ops.tensor_from_list_(A.tolist(), ops.float_dtype_, None)
    B_tensor = ops.tensor_from_list_(B.tolist(), ops.float_dtype_, None)
    result = ops.matmul_(A_tensor, B_tensor)
    assert np.allclose(result.tolist(), expected.tolist())


def test_c_backend_stack_and_cat():
    ops = CTensorOperations()
    t1 = ops.tensor_from_list_([[1.0, 2.0], [3.0, 4.0]], ops.float_dtype_, None)
    t2 = ops.tensor_from_list_([[5.0, 6.0], [7.0, 8.0]], ops.float_dtype_, None)

    stacked = ops.stack_([t1, t2], dim=0)
    assert stacked.shape == (2, 2, 2)
    assert stacked.tolist() == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]

    cat = ops.cat_([t1, t2], dim=0)
    assert cat.shape == (4, 2)
    assert cat.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]


def test_c_backend_is_selectable_through_abstract_tensor():
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = (values * 2.0 + 1.0).sqrt()
        product = values @ AbstractTensor.tensor([[2.0], [1.0]])
        column_mean = values.mean(dim=0, keepdim=True)

    assert type(values).__name__ == "CTensorOperations"
    assert np.allclose(result.tolist(), np.sqrt([[3.0, 5.0], [7.0, 9.0]]))
    assert product.tolist() == [[4.0], [10.0]]
    assert column_mean.shape == (1, 2)
    assert column_mean.tolist() == [[2.0, 3.0]]


@pytest.mark.parametrize("dim", (0, 1))
def test_c_backend_mean_and_composition_match_numpy(dim):
    source = np.arange(24.0).reshape(2, 3, 4)
    ops = CTensorOperations()
    tensor = ops.tensor_from_list_(source.tolist(), ops.float_dtype_, None)

    assert np.allclose(ops.mean_(tensor, dim).tolist(), source.mean(axis=dim))
    stacked = ops.stack_([tensor, tensor], dim=dim)
    assert np.allclose(
        stacked.tolist(), np.stack([source, source], axis=dim)
    )
    concatenated = ops.cat_([tensor, tensor], dim=dim)
    assert np.allclose(
        concatenated.tolist(), np.concatenate([source, source], axis=dim)
    )


def test_c_backend_numeric_cast_keeps_safe_double_storage():
    ops = CTensorOperations()
    tensor = ops.tensor_from_list_(
        [-2.8, 1.9, 3.0], ops.float_dtype_, None
    )

    converted = ops.to_dtype_(tensor, "int64")

    assert converted.tolist() == [-2.0, 1.0, 3.0]
    assert converted.size == 3


def test_c_backend_shape_unary_and_reduction_families_match_numpy():
    source = np.array(
        [[[-2.5, -1.0, 0.0], [0.5, 1.5, 3.0]]],
        dtype=np.float64,
    )
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor(source)
        transposed = values.squeeze(0).transpose(0, 1)
        flattened = transposed.flatten()
        exponential = flattened.exp()
        totals = values.sum(dim=2, keepdim=True)
        products = values.prod(dim=1)
        minimum = values.min(dim=2)
        cumulative = values.cumsum(dim=2)
        argmin = values.argmin(dim=2, keepdim=True)
        softmax = values.softmax(dim=2)

    assert transposed.shape == (3, 2)
    assert np.allclose(transposed.tolist(), source.squeeze(0).T)
    assert flattened.shape == (6,)
    assert np.allclose(exponential.tolist(), np.exp(source.squeeze(0).T.flat))
    assert np.allclose(totals.tolist(), source.sum(axis=2, keepdims=True))
    assert np.allclose(products.tolist(), source.prod(axis=1))
    assert np.allclose(minimum.tolist(), source.min(axis=2))
    assert np.allclose(cumulative.tolist(), source.cumsum(axis=2))
    assert np.allclose(argmin.tolist(), source.argmin(axis=2, keepdims=True))
    shifted = source - source.max(axis=2, keepdims=True)
    expected_softmax = np.exp(shifted) / np.exp(shifted).sum(
        axis=2, keepdims=True
    )
    assert np.allclose(softmax.tolist(), expected_softmax)


def test_c_backend_comparisons_masks_and_clamp_match_numpy():
    source = np.array([[-2.0, 0.25], [1.5, 4.0]])
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor(source)
        mask = values > 0.0
        selected = AbstractTensor.where(mask, values, -1.0)
        clipped = values.clamp(-0.5, 2.0)
        finite = (values / 0.0).isfinite()

    assert mask.tolist() == [[0.0, 1.0], [1.0, 1.0]]
    assert selected.tolist() == [[-1.0, 0.25], [1.5, 4.0]]
    assert clipped.tolist() == [[-0.5, 0.25], [1.5, 2.0]]
    assert finite.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_c_backend_broadcasting_supports_linear_layer_composition():
    source = np.arange(12.0).reshape(4, 3)
    weights = np.array([[0.5, -1.0], [1.0, 0.25], [-0.5, 2.0]])
    bias = np.array([[0.75, -0.25]])
    with AbstractTensor.use_backend("c"):
        x = AbstractTensor.tensor(source)
        w = AbstractTensor.tensor(weights)
        b = AbstractTensor.tensor(bias)
        result = (x @ w + b).clamp_min(0.0)
        expanded = b.expand(2, 4, 2)

    expected = np.maximum(source @ weights + bias, 0.0)
    assert np.allclose(result.tolist(), expected)
    assert expanded.shape == (2, 4, 2)
    assert np.allclose(expanded.tolist(), np.broadcast_to(bias, (2, 4, 2)))


def test_c_backend_repeat_gather_and_mask_primitives_match_numpy():
    source = np.arange(12.0).reshape(3, 4)
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor(source)
        repeated_elements = values.repeat_interleave(2, dim=1)
        tiled = values.repeat(2, dim=0)
        gathered = values.index_select(1, [3, 1])
        selected = values.boolean_mask_select(values > 6.0)

    assert np.allclose(
        repeated_elements.tolist(), np.repeat(source, 2, axis=1)
    )
    assert np.allclose(tiled.tolist(), np.tile(source, (2, 1)))
    assert np.allclose(gathered.tolist(), np.take(source, [3, 1], axis=1))
    assert np.allclose(selected.tolist(), source[source > 6.0])
