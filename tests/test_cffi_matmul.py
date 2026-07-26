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
