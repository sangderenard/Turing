"""Two-row tanh VJP: the second row has a different exact derivative."""

import numpy as np


def forward_and_vjp(x, weight, bias, gain, route, seed):
    branches = np.tanh(x @ weight + bias)
    prediction = (branches * gain) @ route
    branch_gradient = (seed @ route.T) * gain * (1 - branches * branches)
    return prediction, {
        "dendrite_weight": x.T @ branch_gradient,
        "dendrite_bias": branch_gradient.sum(axis=0, keepdims=True),
        "dendrite_gain": ((seed @ route.T) * branches).sum(
            axis=0, keepdims=True,
        ),
    }
