"""An ordinary training program, written like any user of abstract_nn would.

Nothing here knows about translation. It builds a model, trains it on XOR
with Adam, and prints the loss. This file is the INPUT to the translator --
the program that crosses to nodus -- so it stays exactly as a person would
write it.
"""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn import (
    Adam,
    Linear,
    Model,
    MSELoss,
    Sigmoid,
    Tanh,
    set_seed,
)
from src.common.tensors.abstract_nn.utils import from_list_like


def build_dataset(like):
    X = from_list_like(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=like
    )
    X = X * 2.0 - 1.0
    Y = from_list_like([[0.0], [1.0], [1.0], [0.0]], like=like)
    return X, Y


def build_model(like):
    return Model(
        layers=[
            Linear(2, 8, like=like, init="xavier"),
            Linear(8, 1, like=like, init="xavier"),
        ],
        activations=[Tanh(), Sigmoid()],
    )


def train(steps=200, lr=1e-2):
    like = AbstractTensor.get_tensor()
    set_seed(0)
    X, Y = build_dataset(like)
    model = build_model(like)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    for step in range(1, steps + 1):
        prediction = model.forward(X)
        loss = loss_fn.forward(prediction, Y)
        gradient = loss_fn.backward(prediction, Y)
        gradients = model.backward(gradient)
        parameters = optimizer.step(model.parameters(), gradients)
        model.assign_parameters(parameters)
        if step == 1 or step % 50 == 0:
            print(f"step {step}: loss {float(loss):.6f}")
    return model, float(loss)


if __name__ == "__main__":
    train()
