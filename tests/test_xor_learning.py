import pytest

from src.common.tensors.abstract_nn import Linear, Model, Tanh, Sigmoid, BCEWithLogitsLoss, MSELoss, Adam, train_loop, set_seed
from src.common.tensors.abstract_nn.utils import from_list_like
from src.common.tensors.abstraction import AbstractTensor

@pytest.mark.parametrize("loss_type", ["bce", "mse"])
def test_xor_learns(loss_type):
    ops = AbstractTensor.get_tensor(faculty=None)
    set_seed(0)
    X = from_list_like([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=ops)
    X = X * 2.0 - 1.0
    Y = from_list_like([[0.0], [1.0], [1.0], [0.0]], like=ops)
    model = Model(
        layers=[Linear(2, 8, like=ops, init="xavier"), Linear(8, 1, like=ops, init="xavier")],
        activations=[Tanh(), Sigmoid() if loss_type == "mse" else None],
    )
    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = MSELoss() if loss_type == "mse" else BCEWithLogitsLoss()
    # Large log_every suppresses internal printing during tests
    losses, _ = train_loop(model, loss_fn, opt, X, Y, epochs=2000, log_every=10000)
    assert losses[-1] < 0.1
