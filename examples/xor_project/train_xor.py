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


def build_model(like) -> Model:
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
        loss.backward()
        parameters = model.parameters()
        gradients = [parameter.grad for parameter in parameters]
        updated = optimizer.step(parameters, gradients)
        model.layers[0].W = updated[0]
        model.layers[0].b = updated[1]
        model.layers[1].W = updated[2]
        model.layers[1].b = updated[3]
        if step == 1 or step % 50 == 0:
            print(f"step {step}: loss {float(loss):.6f}")
    return model, float(loss)


if __name__ == "__main__":
    train()
