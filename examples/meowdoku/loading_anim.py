"""[LOADING] The loading animation: a REDUCED, engine-free recording of a fountain, the store that holds it, and its validator.

The animation is not a second capture path.  The page's state memory (``web/recording.js``: the compute runtime's recorded
window of packed points) is FILTERED -- positions only, frames and grains decimated, quantized -- into the format below
(``web/loading_record.js`` does the filtering in the browser and uploads the result; ``web/loading_anim.js`` is the JS twin of
this file's codec).  The loading panel (``web/loading_panel.js``) then plays it with a tiny WebGL2 shader and no engine.

Format ``MLA1`` (little endian; every array is BYTE-PLANAR -- all low bytes, then all high bytes -- so gzip finds the zeros)::

    magic "MLA1" | u8 version=1 | u8 fps | u16 n grains | u16 frames F | u16 seam S | u16 outline polylines P
    f32 x0, x1, y0, y1                       the box the u16 positions are quantized into
    n   u16 radius                           (in the same units as positions: radius / max(x1-x0, y1-y0) * 65535)
    F   u16 angle                            the case's view angle per frame: (angle mod 2 pi) / 2 pi * 65536
    P x { u16 k | k x (u16 x, u16 y) }       the chamber outline polylines
    frame 0: x[n] u16, y[n] u16 (planar bytes);  frames 1..F-1: x, y as the wrapped i16 DELTA from the previous frame

The loop: the last ``S`` frames are an overlap.  The period is ``L = F - S``; playing position ``p`` in ``[0, L)`` shows frame ``p``,
cross-faded with frame ``L + p`` while ``p < S`` (see :func:`seam`).  The recording ends settled in the pose it started (an even
number of turns), so the cross-fade is between two resting piles.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
from pathlib import Path

import numpy as np

MAGIC = b"MLA1"
VERSION = 1
MAX_BYTES = 1 << 20                 # an uploaded animation is at most 1 MiB raw ("well under 1 MB" is the target; this is the cap)
MAX_GRAINS = 4096
MAX_FRAMES = 4096
MAX_POLYLINES = 8
MAX_POINTS = 4096
HEADER = struct.Struct("<4sBBHHHH4f")       # magic, version, fps, n, frames, seam, polylines, bbox


class AnimError(ValueError):
    """The bytes are not a valid animation (the message says which rule)."""


def _planar(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a, dtype="<u2")
    return (a & 0xFF).astype(np.uint8).tobytes() + (a >> 8).astype(np.uint8).tobytes()


def _unplanar(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8, count=2 * count).astype(np.uint16)
    return b[:count] | (b[count:] << 8)


def encode(anim: dict) -> bytes:
    """``anim``: fps, seam, bbox (x0, x1, y0, y1), radius u16[n], angle u16[F], outline [u16 (k, 2)], x u16[F, n], y u16[F, n]."""
    x, y = np.asarray(anim["x"], dtype=np.uint16), np.asarray(anim["y"], dtype=np.uint16)
    frames, n = x.shape
    outline = [np.asarray(p, dtype=np.uint16).reshape(-1, 2) for p in anim.get("outline", [])]
    out = [HEADER.pack(MAGIC, VERSION, int(anim["fps"]), n, frames, int(anim["seam"]), len(outline),
                       *[float(v) for v in anim["bbox"]])]
    out.append(_planar(np.asarray(anim["radius"], dtype=np.uint16)))
    out.append(_planar(np.asarray(anim["angle"], dtype=np.uint16)))
    for poly in outline:
        out.append(struct.pack("<H", len(poly)) + _planar(poly[:, 0]) + _planar(poly[:, 1]))
    out.append(_planar(x[0]) + _planar(y[0]))
    for f in range(1, frames):
        out.append(_planar((x[f] - x[f - 1]).astype(np.uint16)) + _planar((y[f] - y[f - 1]).astype(np.uint16)))
    return b"".join(out)


def decode(data: bytes) -> dict:
    """The inverse of :func:`encode`, validating every length: raises :class:`AnimError` on anything that is not exactly one animation."""
    if len(data) > MAX_BYTES:
        raise AnimError(f"larger than the {MAX_BYTES}-byte cap")
    if len(data) < HEADER.size:
        raise AnimError("shorter than the header")
    magic, version, fps, n, frames, seam, polylines, *bbox = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise AnimError("not an MLA1 animation")
    if not (1 <= fps <= 120) or not (1 <= n <= MAX_GRAINS) or not (2 <= frames <= MAX_FRAMES) or polylines > MAX_POLYLINES:
        raise AnimError("header field out of range")
    if seam * 2 > frames:
        raise AnimError("the seam is longer than half the animation")
    x0, x1, y0, y1 = bbox
    if not all(math.isfinite(v) for v in bbox) or not (x1 > x0 and y1 > y0):
        raise AnimError("bad bounding box")
    at = HEADER.size

    def take(count: int) -> np.ndarray:
        nonlocal at
        if at + 2 * count > len(data):
            raise AnimError("truncated")
        arr = _unplanar(data[at:at + 2 * count], count)
        at += 2 * count
        return arr

    radius, angle = take(n), take(frames)
    outline = []
    for _ in range(polylines):
        if at + 2 > len(data):
            raise AnimError("truncated")
        (k,) = struct.unpack_from("<H", data, at)
        at += 2
        if k > MAX_POINTS:
            raise AnimError("outline polyline too long")
        outline.append(np.stack([take(k), take(k)], axis=1))
    x = np.zeros((frames, n), dtype=np.uint16)
    y = np.zeros((frames, n), dtype=np.uint16)
    x[0], y[0] = take(n), take(n)
    for f in range(1, frames):
        x[f] = x[f - 1] + take(n)
        y[f] = y[f - 1] + take(n)
    if at != len(data):
        raise AnimError("trailing bytes")
    return {"fps": fps, "seam": seam, "bbox": tuple(bbox), "radius": radius, "angle": angle, "outline": outline, "x": x, "y": y}


def seam(p: int, frames: int, overlap: int) -> tuple[int, int, float]:
    """Playing position ``p`` of a loop of ``frames - overlap`` -> (frame a, frame b, weight of b).  Weight 0: frame a alone."""
    period = frames - overlap
    p %= period
    if p < overlap:
        w = (p + 0.5) / overlap
        return p + period, p, w
    return p, p, 0.0


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def pack(raw: bytes) -> bytes:
    """The stored/served form: the validated MLA1 stream through OUR octet codec (LZ77 + canonical Huffman, ``octet_codec``)."""
    from src.common.tensors.compression.octet_codec import compress_octets
    return compress_octets(raw)


def unpack(packet: bytes) -> bytes:
    from src.common.tensors.compression.octet_codec import decompress_octets
    return decompress_octets(packet)


class Store:
    """``build/meowdoku/loading_anim.mla`` (the octet-codec packet of the animation) and ``loading_anim.json`` (its numbers).
    Written once, only when valid; an upload never replaces it."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path, self.meta_path = self.directory / "loading_anim.mla", self.directory / "loading_anim.json"
        self._lock = threading.Lock()

    def info(self) -> dict | None:
        """{"digest", "raw_bytes", "bytes" (the packet), "frames", "grains", "fps"} of the stored animation, or None."""
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.path.stat().st_size == meta["bytes"]:
                return meta
        except (OSError, ValueError, KeyError):
            pass
        return None

    def read(self) -> bytes | None:
        return self.path.read_bytes() if self.info() else None

    def put(self, raw: bytes) -> dict:
        """Store a new animation if (and only if) there is none; raises :class:`AnimError` (invalid) or ``FileExistsError`` (present)."""
        anim = decode(raw)                                         # raises on anything invalid
        with self._lock:
            if self.info() is not None:
                raise FileExistsError(str(self.path))
            self.directory.mkdir(parents=True, exist_ok=True)
            packet = pack(raw)
            if unpack(packet) != raw:                              # the codec must give back exactly what was validated
                raise RuntimeError("octet codec round trip failed")
            meta = {"digest": digest(packet), "raw_bytes": len(raw), "bytes": len(packet),
                    "frames": int(anim["x"].shape[0]), "grains": int(anim["x"].shape[1]), "fps": int(anim["fps"])}
            tmp = self.path.with_suffix(".mla.tmp")
            tmp.write_bytes(packet)
            os.replace(tmp, self.path)
            self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return meta
