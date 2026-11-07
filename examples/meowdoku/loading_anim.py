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
import hmac
import json
import math
import os
import re
import struct
import threading
import time
from pathlib import Path

import numpy as np

MAGIC = b"MLA1"
VERSION = 1
MAX_BYTES = 1 << 20                 # an uploaded animation is at most 1 MiB raw ("well under 1 MB" is the target; this is the cap)
MAX_GRAINS = 4096
MAX_FRAMES = 4096
MAX_POLYLINES = 8
MAX_POINTS = 4096
TEMP_SECONDS = 3 * 24 * 60 * 60       # ungraduated fountain captures live for three days
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


#: [STUDIO] the library's caps: an upload that would pass either is refused (nothing is evicted without the owner)
MAX_ENTRIES = 24
MAX_LIBRARY_BYTES = 32 << 20
MAX_TEMP_ENTRIES = 8
MAX_TEMP_BYTES = 8 << 20
MAX_PARAMS_BYTES = 2048
DIGEST_RE = re.compile(r"^[0-9a-f]{16}$")


class LibraryFull(RuntimeError):
    """The library is at its entry or byte cap."""


def clean_params(params) -> dict:
    """The recording parameters kept as metadata: flat, short keys, number / bool / short string values, at most MAX_PARAMS_BYTES of JSON."""
    out = {}
    if isinstance(params, dict):
        for k, v in list(params.items())[:40]:
            if isinstance(k, str) and len(k) <= 40 and isinstance(v, (bool, int, float, str)) and not (isinstance(v, float) and not math.isfinite(v)):
                out[k] = v[:80] if isinstance(v, str) else v
    return out if len(json.dumps(out)) <= MAX_PARAMS_BYTES else {}


def owner_authorize(tls: bool, forwarded_proto: str | None, loopback_direct: bool, supplied: str | None, token: str) -> tuple[bool, int, str]:
    """[STUDIO] Making an entry active or deleting one changes what every player sees: the owner's lifecycle admin token (a constant-time
    compare), over HTTPS only.  HTTPS means this socket is TLS, or a proxy says it terminated TLS (X-Forwarded-Proto: https), or the caller
    is on this machine's loopback with no proxy in between (nothing crosses a network)."""
    if not (tls or (forwarded_proto or "").strip().lower() == "https" or loopback_direct):
        return False, 403, "owner actions need HTTPS"
    if not token or not supplied or not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
        return False, 401, "bad owner token"
    return True, 200, "ok"


class Library:
    """[STUDIO] ``<dir>/loading_library/<digest>.mla`` (the octet-codec packet) + ``<digest>.json`` (date, parameters, bytes, frames) and
    ``active.json`` (``{"digest": ...}``: the entry players get from /api/loading_anim).  The pre-library single file
    ``<dir>/loading_anim.mla`` + ``.json`` is MIGRATED in as the active entry the first time a library without an active pointer opens
    (the old files are left where they are).  ``info()`` / ``read()`` keep the old store's shape for the active entry."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.root = self.directory / "loading_library"
        self.pointer = self.root / "active.json"
        self.legacy_path, self.legacy_meta = self.directory / "loading_anim.mla", self.directory / "loading_anim.json"
        self._lock = threading.RLock()
        self.migrate()

    # -- files
    def _files(self, digest_: str) -> tuple[Path, Path]:
        if not DIGEST_RE.match(digest_ or ""):
            raise KeyError(digest_)
        return self.root / f"{digest_}.mla", self.root / f"{digest_}.json"

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _set_active(self, digest_: str | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._atomic(self.pointer, json.dumps({"digest": digest_}).encode("utf-8"))

    def migrate(self) -> None:
        with self._lock:
            if self.pointer.exists():
                return
            try:
                meta = json.loads(self.legacy_meta.read_text(encoding="utf-8"))
                if self.legacy_path.stat().st_size != meta["bytes"] or not DIGEST_RE.match(meta["digest"]):
                    return
                packet = self.legacy_path.read_bytes()
                decode(unpack(packet))                              # only a valid animation is migrated
            except (OSError, ValueError, KeyError, AnimError):
                return
            self.root.mkdir(parents=True, exist_ok=True)
            mla, js = self._files(meta["digest"])
            when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.legacy_path.stat().st_mtime))
            self._atomic(mla, packet)
            self._atomic(js, json.dumps({**meta, "date": when, "params": {"migrated": True}}).encode("utf-8"))
            self._set_active(meta["digest"])

    # -- reading
    def meta(self, digest_: str) -> dict | None:
        try:
            mla, js = self._files(digest_)
            meta = json.loads(js.read_text(encoding="utf-8"))
            if meta.get("expires_at") and meta["expires_at"] <= time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()):
                return None
            return meta if mla.stat().st_size == meta["bytes"] else None
        except (KeyError, OSError, ValueError):
            return None

    def active_digest(self) -> str | None:
        try:
            digest_ = json.loads(self.pointer.read_text(encoding="utf-8")).get("digest")
        except (OSError, ValueError):
            return None
        return digest_ if digest_ and self.meta(digest_) else None

    def entries(self) -> list[dict]:
        """Every entry's metadata, newest first, each with ``active``."""
        self.prune_expired()
        active, out = self.active_digest(), []
        if self.root.is_dir():
            for js in self.root.glob("*.json"):
                if js.stem != "active" and (meta := self.meta(js.stem)):
                    out.append({**meta, "active": meta["digest"] == active})
        return sorted(out, key=lambda m: (m.get("date", ""), m["digest"]), reverse=True)

    def prune_expired(self) -> None:
        """Delete expired candidates and their octet packets during normal library reads/writes."""
        with self._lock:
            if not self.root.is_dir():
                return
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for js in self.root.glob("*.json"):
                if js.stem == "active" or not DIGEST_RE.match(js.stem):
                    continue
                try:
                    meta = json.loads(js.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if meta.get("expires_at") and meta["expires_at"] <= now:
                    for path in self._files(js.stem):
                        path.unlink(missing_ok=True)

    def caps(self) -> dict:
        entries = self.entries()
        pending = [e for e in entries if e.get("expires_at")]
        return {"max_entries": MAX_ENTRIES, "max_bytes": MAX_LIBRARY_BYTES, "entries": len(entries), "used_bytes": sum(e["bytes"] for e in entries),
                "max_temporary_entries": MAX_TEMP_ENTRIES, "max_temporary_bytes": MAX_TEMP_BYTES,
                "temporary_entries": len(pending), "temporary_bytes": sum(e["bytes"] for e in pending)}

    def info(self) -> dict | None:
        """The ACTIVE entry's metadata ({"digest", "raw_bytes", "bytes" (the packet), "frames", "grains", "fps", "date", "params"}), or None."""
        digest_ = self.active_digest()
        return self.meta(digest_) if digest_ else None

    def read(self, digest_: str | None = None) -> bytes | None:
        """The packet of ``digest_`` (default: the active entry), or None."""
        digest_ = digest_ or self.active_digest()
        try:
            return self._files(digest_)[0].read_bytes() if digest_ and self.meta(digest_) else None
        except (KeyError, OSError):
            return None

    # -- writing
    def put(self, raw: bytes, params=None, *, temporary: bool = False) -> dict:
        """Validate, pack and keep an animation.  The first one into a library with no active entry becomes active; a later one is only
        added.  The same animation twice is one entry.  Raises :class:`AnimError` (invalid) or :class:`LibraryFull`."""
        anim = decode(raw)
        with self._lock:
            self.prune_expired()
            packet = pack(raw)
            if unpack(packet) != raw:                              # the codec must give back exactly what was validated
                raise RuntimeError("octet codec round trip failed")
            digest_ = digest(packet)
            if self.meta(digest_):
                if not temporary and self.meta(digest_).get("expires_at"):
                    return {**self.graduate(digest_), "active": digest_ == self.active_digest(), "duplicate": True}
                return {**self.meta(digest_), "active": digest_ == self.active_digest(), "duplicate": True}
            caps = self.caps()
            if temporary:
                pending = [e for e in self.entries() if e.get("expires_at")]
                if len(pending) >= MAX_TEMP_ENTRIES or sum(e["bytes"] for e in pending) + len(packet) > MAX_TEMP_BYTES:
                    raise LibraryFull("the temporary capture shelf is full; wait for expiry or graduate a capture")
            if caps["entries"] >= MAX_ENTRIES or caps["used_bytes"] + len(packet) > MAX_LIBRARY_BYTES:
                raise LibraryFull(f"the library is full ({caps['entries']}/{MAX_ENTRIES} entries, {caps['used_bytes']}/{MAX_LIBRARY_BYTES} bytes): delete one first")
            self.root.mkdir(parents=True, exist_ok=True)
            meta = {"digest": digest_, "raw_bytes": len(raw), "bytes": len(packet), "frames": int(anim["x"].shape[0]),
                    "grains": int(anim["x"].shape[1]), "fps": int(anim["fps"]), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "params": clean_params(params)}
            if temporary:
                meta["expires_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + TEMP_SECONDS))
            mla, js = self._files(digest_)
            self._atomic(mla, packet)
            self._atomic(js, json.dumps(meta).encode("utf-8"))
            became = self.active_digest() is None
            if became and not temporary:
                self._set_active(digest_)
            return {**meta, "active": became and not temporary}

    def graduate(self, digest_: str) -> dict:
        """Keep a candidate permanently; activation remains a separate owner action."""
        with self._lock:
            meta = self.meta(digest_)
            if meta is None:
                raise KeyError(digest_)
            if "expires_at" in meta:
                meta = {k: v for k, v in meta.items() if k != "expires_at"}
                self._atomic(self._files(digest_)[1], json.dumps(meta).encode("utf-8"))
            return meta

    def activate(self, digest_: str) -> dict:
        with self._lock:
            meta = self.meta(digest_)
            if meta is None:
                raise KeyError(digest_)
            if meta.get("expires_at"):
                raise KeyError(digest_)
            self._set_active(digest_)
            return meta

    def delete(self, digest_: str) -> dict:
        """Remove an entry (the active one too: players then get the procedural animation until one is made active)."""
        with self._lock:
            meta = self.meta(digest_)
            if meta is None:
                raise KeyError(digest_)
            was_active = digest_ == self.active_digest()
            for path in self._files(digest_):
                path.unlink(missing_ok=True)
            if was_active:
                self._set_active(None)
            return {**meta, "was_active": was_active}


Store = Library                                                    # the name the server and the tests used before the library
