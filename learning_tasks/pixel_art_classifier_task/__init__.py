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


def pump_queue(
    q: Queue,
    grid_shape: Tuple[int, int],
    channels: int,
    *,
    stop_event: threading.Event | None = None,
    delay: float = 0.0,
) -> None:
    """Continuously fill ``q`` with classification samples."""
    while stop_event is None or not stop_event.is_set():
        idx = int(np.random.randint(len(SHAPE_NAMES)))
        name = SHAPE_NAMES[idx]
        shape = SHAPES[name][None, ...]
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
