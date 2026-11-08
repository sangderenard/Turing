"""Saved marble photo bridge to Spectral's separate OpenGL worker process.

MRP1 request body (all integers/floats little endian):
  magic[4], config_bytes:u32, triangles:u32, reserved:u32,
  config JSON, positions:f32[triangles,9], normals:f32[triangles,9],
  material_ids:u32[triangles].
The positions/normals/IDs are the exact final Studio faces in normalized
radius units. The server verifies the saved object's geometry_sha256 before
launching a photo and takes PBR16 rows from that immutable saved object.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

MAX_UPLOAD_BYTES = 48 * 1024 * 1024
MAX_TRIANGLES = 500_000
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^preview-[0-9]{6}\.png$")
_HDR = re.compile(r"^exposure-[0-9]{6}\.npz$")


class PhotoError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _finite_vectors(value: object, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise PhotoError(f"{name} must be a {length}-element array")
    try:
        result = [float(x) for x in value]
    except (ValueError, TypeError):
        raise PhotoError(f"{name} must contain numbers") from None
    if not all(math.isfinite(x) and abs(x) <= 10 for x in result):
        raise PhotoError(f"{name} must contain finite scene coordinates")
    return result


def _number(value: object, low: float, high: float, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise PhotoError(f"{name} must be numeric") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise PhotoError(f"{name} is outside [{low}, {high}]")
    return result


def _config(value: dict, saved: dict) -> tuple[dict, float]:
    if value.get("object_digest") != saved["digest"]:
        raise PhotoError("photo request does not name the saved object")
    seconds = _number(value.get("exposure_seconds", 180), 1, 3600, "exposure_seconds")
    transport = str(value.get("transport", "fixed:1"))
    if transport not in ("fixed:1", "fixed:3"):
        raise PhotoError("transport must be fixed:1 or fixed:3")
    camera_in = value.get("camera", {})
    checker_in = value.get("checker", {})
    if not isinstance(camera_in, dict) or not isinstance(checker_in, dict):
        raise PhotoError("camera and checker must be objects")
    resolution = int(_number(camera_in.get("resolution", 256), 128, 2048, "resolution"))
    if resolution not in (128, 256, 512, 768, 1024, 2048):
        raise PhotoError("resolution must be 128, 256, 512, 768, 1024, or 2048")
    orbit_frames = int(_number(value.get("orbit_frames", 1), 1, 32, "orbit_frames"))
    if orbit_frames not in (1, 8, 16, 32):
        raise PhotoError("orbit_frames must be 1, 8, 16, or 32")
    camera = {
        "resolution": resolution,
        "eye_m": _finite_vectors(camera_in.get("eye_m", [0, 0, .07]), 3, "camera.eye_m"),
        "target_m": _finite_vectors(camera_in.get("target_m", [0, 0, .18]), 3, "camera.target_m"),
        "fov_degrees": _number(camera_in.get("fov_degrees", 36), 5, 100, "camera.fov_degrees"),
        "aperture_radius_m": _number(camera_in.get("aperture_radius_m", .002), 1e-5, .02, "camera.aperture_radius_m"),
        "focus_distance_m": _number(camera_in.get("focus_distance_m", .11), .02, 10, "camera.focus_distance_m"),
        "aperture_blades": int(_number(camera_in.get("aperture_blades", 0), 0, 16, "camera.aperture_blades")),
        "aperture_rotation_radians": _number(camera_in.get("aperture_rotation_radians", 0), -7, 7, "camera.aperture_rotation_radians"),
    }
    checker = {
        "center_m": _finite_vectors(checker_in.get("center_m", [0, 0, .30]), 3, "checker.center_m"),
        "width_m": _number(checker_in.get("width_m", .15), .02, 1, "checker.width_m"),
        "checks": int(_number(checker_in.get("checks", 4), 2, 32, "checker.checks")),
        "normal_sign": int(_number(checker_in.get("normal_sign", -1), -1, 1, "checker.normal_sign")),
    }
    if checker["normal_sign"] not in (-1, 1):
        raise PhotoError("checker.normal_sign must be -1 or 1")
    lights_in = value.get("lights", [{"pos": [.14, .10, .17], "dir": [-.14, -.10, .13],
                                     "power": 1.0, "label": "preview-stage"}])
    if not isinstance(lights_in, list) or not 1 <= len(lights_in) <= 8:
        raise PhotoError("lights must contain 1..8 sources")
    lights = []
    for source in lights_in:
        if not isinstance(source, dict):
            raise PhotoError("each light must be an object")
        lights.append({
            "pos": _finite_vectors(source.get("pos"), 3, "light.pos"),
            "dir": _finite_vectors(source.get("dir"), 3, "light.dir"),
            "power": _number(source.get("power"), .01, 100, "light.power"),
            "label": str(source.get("label", "preview-stage"))[:80],
        })
    physical = saved["physical"]
    scale = _number(physical.get("radius-mm"), .1, 200, "saved physical radius-mm") / 1000
    config = {
        "object_digest": saved["digest"],
        "geometry_sha256": saved["geometry_sha256"],
        "exposure_seconds": seconds,
        "orbit_frames": orbit_frames,
        "transport": transport,
        "scale_m": scale,
        "object_center_m": _finite_vectors(value.get("object_center_m", [0, 0, .18]), 3, "object_center_m"),
        "camera": camera, "checker": checker, "lights": lights,
    }
    return config, seconds


def decode_upload(raw: bytes, saved_objects) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float]:
    if len(raw) < 16 or raw[:4] != b"MRP1":
        raise PhotoError("expected MRP1 saved-mesh photo payload")
    config_size, count, reserved = struct.unpack_from("<III", raw, 4)
    if reserved or not 2 <= config_size <= 8192 or not 1 <= count <= MAX_TRIANGLES:
        raise PhotoError("invalid MRP1 header")
    expected = 16 + config_size + count * (9 * 4 + 9 * 4 + 4)
    if len(raw) != expected:
        raise PhotoError("MRP1 payload length does not match its triangle count")
    try:
        request = json.loads(raw[16:16 + config_size])
    except (UnicodeError, ValueError):
        raise PhotoError("invalid MRP1 config JSON") from None
    if not isinstance(request, dict) or not _DIGEST.fullmatch(str(request.get("object_digest", ""))):
        raise PhotoError("a saved object digest is required")
    try:
        saved = saved_objects.get(request["object_digest"])
    except (KeyError, FileNotFoundError, ValueError):
        raise PhotoError("saved marble object is unavailable", 404) from None
    pos_start = 16 + config_size
    norm_start = pos_start + count * 36
    ids_start = norm_start + count * 36
    positions = np.frombuffer(raw, dtype="<f4", count=count * 9, offset=pos_start).reshape(count, 9)
    normals = np.frombuffer(raw, dtype="<f4", count=count * 9, offset=norm_start).reshape(count, 9)
    ids = np.frombuffer(raw, dtype="<u4", count=count, offset=ids_start)
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise PhotoError("saved mesh contains nonfinite coordinates or normals")
    if np.any(np.abs(positions) > 100) or np.any(np.abs(normals) > 2):
        raise PhotoError("saved mesh coordinates or normals are outside their radius units")
    # The browser's geometrySha256 serializes each face as its nine position
    # floats, nine vertex-normal floats, then one unsigned material ID.
    packed = np.empty(count, dtype=[("positions", "<f4", (9,)),
                                    ("normals", "<f4", (9,)), ("id", "<u4")])
    packed["positions"] = positions
    packed["normals"] = normals
    packed["id"] = ids
    if hashlib.sha256(packed.tobytes()).hexdigest() != saved["geometry_sha256"]:
        raise PhotoError("uploaded triangles differ from the saved marble object")
    table = saved["materials"]["tables"]["pbr"]
    if table["stride"] != 16:
        raise PhotoError("saved material table is not PBR16")
    pbr = np.asarray(table["data"], dtype=np.float32).reshape(-1, 16)
    if np.any(ids >= len(pbr)):
        raise PhotoError("saved triangle material ID exceeds the PBR table")
    config, seconds = _config(request, saved)
    return positions.copy().reshape(count, 3, 3), ids.astype(np.int32), pbr, config, seconds


class PhotoJobs:
    """One active GPU exposure per server; digest and subprocess own its state."""

    def __init__(self, directory: Path, spectral_root: Path):
        self.directory = Path(directory)
        self.spectral_root = Path(spectral_root)
        self.lock = threading.Lock()
        self.active_digest: str | None = None
        self.process: subprocess.Popen | None = None
        self._sweep_stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_animations, daemon=True, name="marble-photo-expiry")
        self._sweeper.start()

    def _sweep_animations(self) -> None:
        while not self._sweep_stop.is_set():
            with self.lock:
                self._expire_animations_locked()
            self._sweep_stop.wait(60)

    def _expire_animations_locked(self) -> None:
        if not self.directory.exists():
            return
        for directory in self.directory.iterdir():
            if not directory.is_dir() or not _DIGEST.fullmatch(directory.name):
                continue
            path = directory / "status.json"
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                expires = float(status.get("animation_expires_at", 0))
                if status.get("animation") == "orbit.avi" and 0 < expires <= time.time():
                    (directory / "orbit.avi").unlink(missing_ok=True)
                    status.pop("animation", None)
                    status["animation_expired"] = True
                    temp = directory / "status.expiry.tmp"
                    temp.write_text(json.dumps(status), encoding="utf-8")
                    os.replace(temp, path)
            except (OSError, ValueError, TypeError):
                continue

    def _stop_locked(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def start(self, raw: bytes, saved_objects) -> dict:
        vertices, ids, pbr, config, seconds = decode_upload(raw, saved_objects)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="marble-photo-", dir=self.directory) as temporary:
            staging = Path(temporary)
            payload = staging / "payload.npz"
            np.savez_compressed(payload, vertices=vertices, material_ids=ids, pbr=pbr)
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(payload.read_bytes() + canonical).hexdigest()
            (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.lock:
                if self.active_digest == digest:
                    status = self._status_locked(digest)
                    if status["state"] in ("preparing", "exposing", "encoding", "complete"):
                        return status
                self._stop_locked()
                self.active_digest = None
                directory = self.directory / digest
                directory.mkdir(exist_ok=True)
                os.replace(payload, directory / "payload.npz")
                os.replace(staging / "config.json", directory / "config.json")
                status = {"digest": digest, "state": "preparing", "image": None,
                          "object_digest": config["object_digest"],
                          "exposure_seconds": seconds}
                (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
                command = [sys.executable, "-m", "camera_designer.marble_preview_worker",
                           "--payload", str(directory / "payload.npz"),
                           "--config", str(directory / "config.json"),
                           "--out", str(directory), "--seconds", str(seconds),
                           "--native-module-dir", str(self.spectral_root),
                           "--avi-writer-root", str(Path(__file__).resolve().parents[2])]
                log = open(directory / "worker.log", "ab")
                try:
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    self.process = subprocess.Popen(command, cwd=self.spectral_root,
                                                    stdout=log, stderr=subprocess.STDOUT,
                                                    creationflags=flags)
                except OSError as error:
                    raise PhotoError(f"Spectral worker could not start: {error}", 503) from error
                finally:
                    log.close()
                self.active_digest = digest
                return status

    def _status_locked(self, digest: str) -> dict:
        path = self.directory / digest / "status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"digest": digest, "state": "preparing", "image": None}
        if status.get("state") == "error" and not status.get("digest"):
            return {"digest": digest, "state": "error", "image": None,
                    "error": str(status.get("error", "Spectral worker failed"))}
        if status.get("digest") != digest:
            raise PhotoError("photo status digest mismatch", 503)
        if digest != self.active_digest and status.get("state") != "complete":
            raise PhotoError("photo digest is no longer active", 404)
        process = self.process if digest == self.active_digest else None
        if process is not None and process.poll() is not None and status.get("state") not in ("error", "complete", "canceled"):
            return {"digest": digest, "state": "error", "image": status.get("image"),
                    "frames": status.get("frames", []),
                    "error": f"Spectral worker exited with code {process.returncode} before finishing the photo"}
        if status.get("state") == "encoding":
            temporary = self.directory / digest / "orbit.avi.tmp"
            try:
                status["encoding_bytes"] = temporary.stat().st_size
            except OSError:
                status["encoding_bytes"] = 0
        return status

    def status(self, digest: str) -> dict:
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        with self.lock:
            return self._status_locked(digest)

    def image(self, digest: str, name: str) -> bytes:
        if not _DIGEST.fullmatch(digest) or not _IMAGE.fullmatch(name):
            raise PhotoError("invalid photo image path", 400)
        with self.lock:
            self._status_locked(digest)
            # A newer frame may publish between the status poll and the
            # browser's image fetch. Every numbered frame under this digest
            # represents the same saved scene, so older frames remain valid.
            try:
                return (self.directory / digest / name).read_bytes()
            except OSError:
                raise PhotoError("photo frame is unavailable", 404) from None

    def hdr(self, digest: str, name: str) -> bytes:
        if not _DIGEST.fullmatch(digest) or not _HDR.fullmatch(name):
            raise PhotoError("invalid HDR exposure path", 400)
        with self.lock:
            status = self._status_locked(digest)
            available = {frame.get("hdr") for frame in status.get("frames", [])}
            if status.get("hdr") != name and name not in available:
                raise PhotoError("HDR exposure is unavailable", 404)
            try:
                return (self.directory / digest / name).read_bytes()
            except OSError:
                raise PhotoError("HDR exposure is unavailable", 404) from None

    def animation(self, digest: str) -> bytes:
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        with self.lock:
            self._expire_animations_locked()
            status = self._status_locked(digest)
            if status.get("animation") != "orbit.avi" or time.time() >= float(status.get("animation_expires_at", 0)):
                raise PhotoError("orbit animation has expired or is unavailable", 404)
            try:
                return (self.directory / digest / "orbit.avi").read_bytes()
            except OSError:
                raise PhotoError("orbit animation is unavailable", 404) from None

    def cancel(self, digest: str) -> dict:
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        with self.lock:
            if self.active_digest != digest:
                raise PhotoError("photo digest is no longer active", 404)
            self._stop_locked()
            self.active_digest = None
            (self.directory / digest / "orbit.avi").unlink(missing_ok=True)
            path = self.directory / digest / "status.json"
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                current["state"] = "canceled"
                current.pop("animation", None)
                temp = path.with_suffix(".cancel.tmp")
                temp.write_text(json.dumps(current), encoding="utf-8")
                os.replace(temp, path)
            except (OSError, ValueError):
                pass
            return {"digest": digest, "state": "canceled", "image": None}

    def close(self) -> None:
        self._sweep_stop.set()
        with self.lock:
            self._stop_locked()
            self.active_digest = None
        self._sweeper.join(timeout=2)
