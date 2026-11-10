"""Bounded, read-only native housing transport. Never starts an exposure.

The record supplies a mount envelope, not a replacement camera render mesh:
native face/material ownership has not been exported by this endpoint.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import threading

from .marble_camera_mount import camera_mount_source
from .marble_lens_contract import SPECTRAL_ROOT, apply_lens_selection

MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CACHE_BYTES = 24 * 1024 * 1024
MAX_CACHE_ENTRIES = 2
MAX_HOUSING_TRIANGLES = 100000
_LOCK = threading.Lock()
_CACHE: OrderedDict[str, bytes] = OrderedDict()


class HousingError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise HousingError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _camera(payload, spectral_root):
    if not isinstance(payload, dict) or set(payload) != {"camera"} or not isinstance(payload["camera"], dict):
        raise HousingError("Housing request must contain one explicit camera object")
    source = payload["camera"]
    camera = {}
    for name in ("eye_m", "target_m"):
        vector = source.get(name)
        if not isinstance(vector, list) or len(vector) != 3:
            raise HousingError(f"camera.{name} must be an explicit three-element array")
        camera[name] = [_number(x, f"camera.{name}", -10, 10) for x in vector]
    if camera["eye_m"] == camera["target_m"]:
        raise HousingError("Camera eye and target must differ")
    camera["aperture_radius_m"] = _number(source.get("aperture_radius_m", .005), "aperture_radius_m", 1e-5, .02)
    camera["aperture_rotation_radians"] = _number(source.get("aperture_rotation_radians", 0), "aperture_rotation_radians", -7, 7)
    blades = source.get("aperture_blades", 0)
    if type(blades) is not int or not 0 <= blades <= 16:
        raise HousingError("aperture_blades must be an integer in [0, 16]")
    camera["aperture_blades"] = blades
    if source.get("lens_selection") is not None:
        camera["focus_distance_m"] = _number(source.get("focus_distance_m"), "focus_distance_m", .001, 10)
        camera = apply_lens_selection(camera, source["lens_selection"], spectral_root)
    return camera


def housing_response(body: bytes, spectral_root: Path = SPECTRAL_ROOT) -> bytes:
    """Return bounded serialized JSON; concurrent cache misses fail without queueing."""
    if not isinstance(body, bytes) or not 0 < len(body) <= MAX_REQUEST_BYTES:
        raise HousingError("Housing request must be between 1 and 4096 bytes", 413)
    try:
        payload = json.loads(body)
        camera = _camera(payload, spectral_root)
    except (ValueError, TypeError, UnicodeError) as error:
        if isinstance(error, HousingError):
            raise
        raise HousingError(str(error)) from None
    root = Path(spectral_root).resolve()
    # A source edit must not return a cached geometry record from older code.
    sources = ("camera_preset.py", "camera_frame.py", "scene_builder.py", "studio_photo_geometry.py")
    try:
        revisions = [(name, (root / "camera_designer" / name).stat().st_mtime_ns) for name in sources]
    except OSError:
        raise HousingError("Canonical native camera geometry is unavailable", 503) from None
    key = hashlib.sha256(json.dumps([str(root), revisions, camera], sort_keys=True, allow_nan=False).encode()).hexdigest()
    if not _LOCK.acquire(blocking=False):
        raise HousingError("Native camera housing is busy. Retry after the current geometry request finishes.", 503)
    try:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        try:
            record = camera_mount_source(camera, root)
        except (ImportError, OSError) as error:
            raise HousingError("Canonical native camera geometry is unavailable: " + str(error), 503) from None
        if not 0 < record["triangle_count"] <= MAX_HOUSING_TRIANGLES:
            raise HousingError("Native camera housing exceeds the bounded geometry contract", 422)
        record["capabilities"] = {"mountEnvelope": True, "renderMesh": False,
                                  "reason": "Native face/material ownership is not exported by this mount endpoint"}
        result = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(result) > MAX_RESPONSE_BYTES:
            raise HousingError("Native camera housing exceeds the 16 MiB response limit", 422)
        _CACHE[key] = result
        while len(_CACHE) > MAX_CACHE_ENTRIES or sum(map(len, _CACHE.values())) > MAX_CACHE_BYTES:
            _CACHE.popitem(last=False)
        return result
    finally:
        _LOCK.release()
