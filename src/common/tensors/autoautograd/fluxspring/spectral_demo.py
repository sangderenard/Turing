# -*- coding: utf-8 -*-
"""Simple demo showcasing spectral band targets routed through linear layers.

The demo synthesises a buffer containing three sine bands, extracts bandpower
features with :func:`compute_metrics`, and trains a small stack of linear layers
(two before and after a middle routing layer) to reproduce those targets.
"""
from __future__ import annotations

import numpy as np
import torch

from .spectral_readout import compute_metrics


def main() -> None:
    fs = 44100
    N = 2048
    t = np.arange(N) / fs
    buffer = (
        np.sin(2 * np.pi * 440 * t)
        + 0.5 * np.sin(2 * np.pi * 880 * t)
        + 0.2 * np.sin(2 * np.pi * 1760 * t)
    )
    config = {
        "tick_hz": fs,
        "win_len": N,
        "window_fn": "hann",
        "metrics": {
            "bands": [[400, 500], [850, 910], [1700, 1800]],
        },
    }
    m = compute_metrics(buffer, config)
    target = torch.tensor(m["bandpower"].data, dtype=torch.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.Linear(8, 8),  # first double layer
        torch.nn.Linear(8, 8),  # middle routing layer
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 3),  # second double layer
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        out = model(target)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()

    print("Target bandpower:", target)
    print("Network output:", model(target).detach())


if __name__ == "__main__":
    main()
