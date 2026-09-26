"""Renderer-neutral geometry frame transport for the shell mailbox.

The producer owns geometry preparation. The consumer receives an immutable
packet, so a later simulation update cannot change a frame awaiting display.
This module defines the wire format; it does not claim backend emission support.
"""

from dataclasses import dataclass
import struct

import numpy as np


# Little endian, no pointers: identity, viewport and element counts followed by
# tightly packed arrays. Mesh/line rows are xyz+rgba; points add pixel diameter.
HEADER = struct.Struct("<8sQQQ7I")
MAGIC = b"TURGEOM1"


@dataclass(frozen=True)
class GeometryFrame:
    generation: int
    attempt: int
    revision: int
    viewport: tuple[int, int]
    mvp: np.ndarray
    vertices: np.ndarray
    indices: np.ndarray
    lines: np.ndarray
    points: np.ndarray
    overlay: str

    def encode(self) -> bytes:
        """Copy one producer frame into the mailbox's immutable payload."""
        arrays = (
            np.asarray(self.mvp, dtype="<f4"),
            np.asarray(self.vertices, dtype="<f4"),
            np.asarray(self.indices),
            np.asarray(self.lines, dtype="<f4"),
            np.asarray(self.points, dtype="<f4"),
        )
        mvp, vertices, indices, lines, points = arrays
        if mvp.shape != (4, 4):
            raise ValueError("geometry MVP must be 4 by 4")
        for array, columns in ((vertices, 7), (lines, 7), (points, 8)):
            if array.ndim != 2 or array.shape[1] != columns:
                raise ValueError(f"geometry rows must have {columns} columns")
        if indices.ndim != 1 or indices.dtype.kind not in "ui":
            raise ValueError("triangle indices must be a flat integer array")
        if indices.size and (indices.min() < 0 or indices.max() >= len(vertices)):
            raise ValueError("triangle index outside vertex buffer")
        text = self.overlay.encode("utf-8")
        payload = HEADER.pack(
            MAGIC, self.generation, self.attempt, self.revision,
            *self.viewport, len(vertices), len(indices), len(lines), len(points), len(text),
        ) + b"".join((
            mvp.tobytes(), vertices.tobytes(), indices.astype("<u4").tobytes(),
            lines.tobytes(), points.tobytes(), text,
        ))
        # The decoder also validates incoming native packets. Share its checks
        # rather than accepting frames from Python that native hosts reject.
        decode_geometry_frame(payload)
        return payload


def decode_geometry_frame(payload: bytes) -> GeometryFrame:
    """Validate every span before exposing read-only array views to a renderer."""
    payload = bytes(payload)
    if len(payload) < HEADER.size:
        raise ValueError("truncated geometry header")
    magic, generation, attempt, revision, width, height, nv, ni, nl, np_, nt = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("unsupported geometry packet version")
    if not width or not height or ni % 3 or nl % 2:
        raise ValueError("invalid viewport or incomplete geometry primitive")
    expected = HEADER.size + 64 + 28 * nv + 4 * ni + 28 * nl + 32 * np_ + nt
    if len(payload) != expected:
        raise ValueError("geometry payload size does not match descriptor")
    offset = HEADER.size

    def array(dtype, count, shape):
        nonlocal offset
        value = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += value.nbytes
        return value

    mvp = array("<f4", 16, (4, 4))
    vertices = array("<f4", nv * 7, (nv, 7))
    indices = array("<u4", ni, (ni,))
    lines = array("<f4", nl * 7, (nl, 7))
    points = array("<f4", np_ * 8, (np_, 8))
    if any(not np.isfinite(a).all() for a in (mvp, vertices, lines, points)):
        raise ValueError("geometry contains non-finite values")
    if ni and indices.max() >= nv:
        raise ValueError("triangle index outside vertex buffer")
    for rows in (vertices, lines, points):
        if np.any(rows[:, 3:7] < 0) or np.any(rows[:, 3:7] > 1):
            raise ValueError("geometry colors must be normalized RGBA")
    if np.any(points[:, 7] <= 0):
        raise ValueError("point diameter must be positive")
    return GeometryFrame(
        generation, attempt, revision, (width, height), mvp, vertices,
        indices, lines, points, payload[offset:].decode("utf-8"),
    )
