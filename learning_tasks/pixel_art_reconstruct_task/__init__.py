"""Pixel art reconstruction from spectral noise."""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Dict, Tuple

import numpy as np
from learning_tasks.loss_composer import LossComposer
from learning_tasks.pixel_shapes import SHAPES, SHAPE_NAMES

NUM_LOGITS = 0


def _spectral_noise(img: np.ndarray) -> np.ndarray:
    """Generate noise with the magnitude spectrum of ``img``."""
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
    """Continuously fill ``q`` with reconstruction samples."""
    while stop_event is None or not stop_event.is_set():
        idx = int(np.random.randint(len(SHAPE_NAMES)))
        name = SHAPE_NAMES[idx]
        shape = SHAPES[name][None, ...]
        inp = _spectral_noise(shape[0])[None, ...]
        tgt = shape
        category: Dict[str, int | str] = {"label": idx, "name": name}
        q.put((inp, tgt, category))
        if delay:
            time.sleep(delay)


def build_loss_composer(C: int, num_logits: int = NUM_LOGITS) -> LossComposer:
    """Return a :class:`LossComposer` for the reconstruction task."""
    composer = LossComposer()

    def mse(pred, tgt, _cats):
        return ((pred - tgt) ** 2).mean()

    composer.add(slice(0, C), lambda tgt, _cats: tgt, mse)
    return composer
