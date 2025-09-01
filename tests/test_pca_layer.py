import numpy as np

from src.common.tensors.abstract_nn.pca_layer import PCATransformLayer
from src.common.tensors.abstraction import AbstractTensor as AT


def test_pca_layer_reduces_dimension():
    rng = np.random.default_rng(0)
    data = AT.get_tensor(rng.standard_normal((10, 5)))
    layer = PCATransformLayer(n_components=3)
    out = layer.forward(data)
    assert out.shape == (10, 3)
