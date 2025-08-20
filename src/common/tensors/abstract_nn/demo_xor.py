from ..abstraction import AbstractTensor
from . import (
    Linear,
    Model,
    Tanh,
    Sigmoid,
    BCEWithLogitsLoss,
    MSELoss,
    Adam,
    train_loop,
    set_seed,
)
from .utils import from_list_like


def run_bce_demo(ops):
    set_seed(0)
    X = from_list_like(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=ops
    )
    Y = from_list_like([[0.0], [1.0], [1.0], [0.0]], like=ops)
    model = Model(
        layers=[Linear(2, 8, like=ops), Linear(8, 1, like=ops)],
        activations=[Tanh(), None],
    )
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = BCEWithLogitsLoss()
    print("-- Recipe 1: BCEWithLogits --")
    train_loop(model, loss_fn, opt, X, Y, epochs=500, log_every=100)


def run_mse_demo(ops):
    set_seed(0)
    X = from_list_like(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=ops
    )
    Y = from_list_like([[0.0], [1.0], [1.0], [0.0]], like=ops)
    model = Model(
        layers=[Linear(2, 8, like=ops), Linear(8, 1, like=ops)],
        activations=[Tanh(), Sigmoid()],
    )
    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = MSELoss()
    print("-- Recipe 2: MSE --")
    train_loop(model, loss_fn, opt, X, Y, epochs=500, log_every=100)


if __name__ == "__main__":
    ops = AbstractTensor.get_tensor(faculty=None)
    run_bce_demo(ops)
    run_mse_demo(ops)

