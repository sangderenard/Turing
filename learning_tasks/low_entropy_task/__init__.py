"""Low entropy learning task generator.

This module supplies a queue pumping helper used by the Riemann demo.
It continually generates tuples of ``(input, target, category)`` where:

* ``input``  – random Gaussian fields with distinct spectral content.
* ``target`` – low-entropy variants of a shared base field.
* ``category`` – dictionary carrying both the low-entropy offset and a
  spectrum category.  The ``"spectrum"`` entry is intended for classifier
  logits in the demo, while ``"offset"`` tracks the generation index.
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Dict, Tuple
import numpy as np


def _random_spectral_gaussian(shape: Tuple[int, ...]) -> np.ndarray:
    """Return Gaussian noise with a randomized spectrum."""
    arr = np.random.randn(*shape)
    fft = np.fft.fftn(arr)
    phase = np.angle(fft)
    mag = np.random.randn(*shape)
    fft_rand = mag * np.exp(1j * phase)
    return np.fft.ifftn(fft_rand).real


def _low_entropy_variant(base: np.ndarray, seed: int) -> np.ndarray:
    """Generate a low-entropy variant of ``base`` using simple shifts."""
    x = np.roll(base, shift=1, axis=-1)
    if (seed // 2) % 2 == 0:
        x = np.flip(x, axis=-2)
    x += np.random.normal(scale=0.01, size=x.shape)
    return x


def pump_queue(
    q: Queue,
    grid_shape: Tuple[int, int, int],
    channels: int,
    *,
    stop_event: threading.Event | None = None,
    num_spectrum_categories: int = 3,
    delay: float = 0.0,
) -> None:
    """Continuously fill ``q`` with learning samples.

    Parameters
    ----------
    q : Queue
        Destination queue receiving ``(input, target, category)`` tuples.
    grid_shape : tuple
        Spatial dimensions for each sample.
    channels : int
        Number of channels per sample.
    stop_event : threading.Event, optional
        When set, terminates the pumping loop.
    num_spectrum_categories : int
        Number of distinct spectral categories to sample from.
    delay : float
        Optional sleep duration between generated samples.
    """
    base_target = _random_spectral_gaussian((channels, *grid_shape))
    seed = 0
    while stop_event is None or not stop_event.is_set():
        spectrum_cat = int(np.random.randint(num_spectrum_categories))
        inp = _random_spectral_gaussian((channels, *grid_shape))
        tgt = _low_entropy_variant(base_target, seed)
        category: Dict[str, int] = {"offset": seed, "spectrum": spectrum_cat}
        q.put((inp, tgt, category))
        seed += 1
        if delay:
            time.sleep(delay)
