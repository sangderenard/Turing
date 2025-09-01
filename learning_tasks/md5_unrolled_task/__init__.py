"""MD5 depth-unrolled learning task generator.

This module generates training samples where the model observes the MD5 state
at every step of the algorithm.  Each sample is represented as a pair of 3‑D
bit‑plane tensors:

* ``inputs``  – message block bit‑planes concatenated with the state prior to
  each supervised step.
* ``targets`` – state bit‑planes after each supervised step.

The helper :func:`pump_queue` continuously feeds a queue with random samples for
integration with the Riemann demo's training loop.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from queue import Queue
from typing import Dict, List, Tuple

import numpy as np

# MD5 constants
S = [7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 + [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4
K = [int(abs(math.sin(i + 1)) * (1 << 32)) & 0xFFFFFFFF for i in range(64)]


def _left_rotate(x: int, c: int) -> int:
    return ((x << c) | (x >> (32 - c))) & 0xFFFFFFFF


def _md5_pad(msg_bytes: bytes) -> bytes:
    ml = (8 * len(msg_bytes)) & 0xFFFFFFFFFFFFFFFF
    msg = msg_bytes + b"\x80"
    while (len(msg) % 64) != 56:
        msg += b"\x00"
    msg += struct.pack("<Q", ml)
    return msg


def words_to_bitplanes(words: List[int]) -> np.ndarray:
    """Convert a sequence of 32‑bit words to a flat bit‑plane array."""
    arr = np.array(words, dtype=np.uint32).reshape(-1, 1)
    bits = ((arr >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8)
    return bits.reshape(-1)


def md5_with_states(msg_bytes: bytes) -> Tuple[bytes, List[Tuple[int, int, int, int]], List[int]]:
    """Return digest, per‑step states and message words for a single block."""
    msg = _md5_pad(msg_bytes)
    A0, B0, C0, D0 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
    all_step_states: List[List[Tuple[int, int, int, int]]] = []
    first_block_words: List[int] | None = None

    for i in range(0, len(msg), 64):
        block = msg[i : i + 64]
        M = list(struct.unpack("<16I", block))
        if first_block_words is None:
            first_block_words = M
        A, B, C, D = A0, B0, C0, D0
        step_states: List[Tuple[int, int, int, int]] = []

        for j in range(64):
            if 0 <= j <= 15:
                F = (B & C) | (~B & D)
                g = j
            elif 16 <= j <= 31:
                F = (D & B) | (~D & C)
                g = (5 * j + 1) % 16
            elif 32 <= j <= 47:
                F = B ^ C ^ D
                g = (3 * j + 5) % 16
            else:
                F = C ^ (B | ~D)
                g = (7 * j) % 16

            F = (F + A + K[j] + M[g]) & 0xFFFFFFFF
            A, D, C = D, C, B
            B = (B + _left_rotate(F, S[j])) & 0xFFFFFFFF
            step_states.append((A, B, C, D))

        A0 = (A0 + A) & 0xFFFFFFFF
        B0 = (B0 + B) & 0xFFFFFFFF
        C0 = (C0 + C) & 0xFFFFFFFF
        D0 = (D0 + D) & 0xFFFFFFFF
        all_step_states.append(step_states)

    digest = struct.pack("<4I", A0, B0, C0, D0)
    assert first_block_words is not None
    return digest, all_step_states[0], first_block_words


def make_sample(msg_bytes: bytes, supervise_every: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Build input/target tensors for ``msg_bytes``.

    Parameters
    ----------
    msg_bytes : bytes
        Message to hash.  Must result in a single 512‑bit MD5 block.
    supervise_every : int
        Take every ``n``‑th step for supervision.
    """
    digest, step_states, message_words = md5_with_states(msg_bytes)
    A0, B0, C0, D0 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
    prev_states = [(A0, B0, C0, D0)] + step_states[:-1]
    step_states = step_states[::supervise_every]
    prev_states = prev_states[::supervise_every]

    tgt = np.stack([words_to_bitplanes(s) for s in step_states], axis=-1)
    prev = np.stack([words_to_bitplanes(s) for s in prev_states], axis=-1)
    msg_bits = words_to_bitplanes(message_words)
    msg_bits = np.tile(msg_bits[:, None], (1, tgt.shape[1]))
    inp = np.concatenate([msg_bits, prev], axis=0)
    category: Dict[str, int | str] = {"message_hex": msg_bytes.hex(), "supervise_every": supervise_every}
    return inp, tgt, category


def pump_queue(
    q: Queue,
    grid_shape: Tuple[int, int, int],
    channels: int,
    *,
    stop_event: threading.Event | None = None,
    supervise_every: int = 1,
    delay: float = 0.0,
) -> None:
    """Continuously fill ``q`` with MD5 learning samples."""
    while stop_event is None or not stop_event.is_set():
        msg_len = int(np.random.randint(1, 56))
        msg_bytes = np.random.bytes(msg_len)
        inp, tgt, category = make_sample(msg_bytes, supervise_every=supervise_every)
        q.put((inp, tgt, category))
        if delay:
            time.sleep(delay)
