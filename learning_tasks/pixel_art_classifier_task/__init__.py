"""Pixel art classification task.

Generates noisy versions of simple pixel art shapes and asks the model to
classify which shape was chosen.  The noise shares the spectral magnitude of
the clean shape but has randomized phase, providing different textures for each
shape.
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Dict, Tuple

import numpy as np
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.losses import CrossEntropyLoss
from learning_tasks.loss_composer import LossComposer
from learning_tasks.pixel_shapes import SHAPES, SHAPE_NAMES

NUM_LOGITS = len(SHAPE_NAMES)


def _spectral_noise(img: np.ndarray) -> np.ndarray:
    """Generate noise sharing ``img``'s magnitude spectrum."""
    fft = np.fft.fftn(img)
    mag = np.abs(fft)
    phase = np.exp(1j * np.random.uniform(0.0, 2 * np.pi, fft.shape))
    return np.fft.ifftn(mag * phase).real


def _corrupt_noise(img: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Return ``img`` with a fraction of pixels replaced by random noise.

    Parameters
    ----------
    img:
        The image to corrupt.
    p:
        Probability that each pixel will be replaced by a random value in
        ``[0, 1]``.
    """
    mask = np.random.rand(*img.shape) < p
    noise = np.random.rand(*img.shape)
    corrupted = img.copy()
    corrupted[mask] = noise[mask]
    return corrupted


def pump_queue(
    q: Queue,
    grid_shape: Tuple[int, int],
    channels: int,
    *,
    stop_event: threading.Event | None = None,
    delay: float = 0.0,
    noise_mode: str = "spectral",
) -> None:
    """Continuously fill ``q`` with classification samples.

    Parameters
    ----------
    noise_mode:
        ``"spectral"`` (default) adds phase-randomized spectral noise sharing
        the target's magnitude spectrum. ``"corrupt"`` randomly replaces a
        fraction of pixels with uniform noise.
    """
    while stop_event is None or not stop_event.is_set():
        idx = int(np.random.randint(len(SHAPE_NAMES)))
        name = SHAPE_NAMES[idx]
        shape = SHAPES[name][None, ...]
        if noise_mode == "corrupt":
            inp = _corrupt_noise(shape[0])[None, ...]
        else:
            inp = shape + _spectral_noise(shape[0])
        tgt = shape
        category: Dict[str, int | str] = {"label": idx, "name": name}
        q.put((inp, tgt, category))
        if delay:
            time.sleep(delay)


def build_loss_composer(C: int, num_logits: int = NUM_LOGITS) -> LossComposer:
    """Return a :class:`LossComposer` for the classifier task."""
    ce_loss = CrossEntropyLoss()
    AT = AbstractTensor
    composer = LossComposer()

    def cat_target(_tgt, cats):
        return AT.get_tensor(np.array([c["label"] for c in cats]))

    def ce(pred, tgt, _cats):
        return ce_loss(pred.reshape(len(tgt), num_logits), tgt)

    composer.add(slice(0, num_logits), cat_target, ce)
    return composer
