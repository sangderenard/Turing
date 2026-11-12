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
import itertools
import importlib.util
from functools import lru_cache
import json
import math
import uuid
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
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .marble_lens_contract import FOV_MIN_DEGREES, PUPIL_RADIUS_MAX_M, FOCUS_MIN_M, FOCUS_MAX_M, FOCUS_SEMANTICS, SPECTRAL_ROOT, apply_lens_selection

from .marble_camera_back import apply_camera_back

MAX_UPLOAD_BYTES = 48 * 1024 * 1024
MAX_TRIANGLES = 500_000
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^preview-[0-9]{6}\.png$")
_MARKER = re.compile(r"^marker-[0-9]{6}\.png$")
_SCENE_RESPONSE = re.compile(r"^light-scene-response-[0-9a-f]{32}\.bin$")
_FIELD = re.compile(r"^light-field-[0-9a-f]{32}\.bin$")
_FIELD_ESTIMATOR = "track-length-throughput-v1"
_FIELD_ROOT_ESTIMATOR = "track-length-root-sweep-v1"
_SURFACE_TRANSPORT = "studio-pbr-mixture-v1"
_FIELD_INTERNAL_MEDIA = dict(version="canonical-beer-segment-integral-v1",
    scope="actual valid interior and exterior traced segments", entry_packet=True,
    integration="per-recording-cell integral of canonical Beer transmittance",
    optical_authority="installed medium/region/dye helper", invalid_ownership="excluded",
    external_extinction="unsupported; request rejected")
_FIELD_AXES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
_ICO_A, _ICO_B = .5257311121191336, .85065080835204
_FIELD_ICOSAHEDRON = [[0, _ICO_A, _ICO_B], [0, -_ICO_A, _ICO_B], [0, _ICO_A, -_ICO_B], [0, -_ICO_A, -_ICO_B],
    [_ICO_A, _ICO_B, 0], [-_ICO_A, _ICO_B, 0], [_ICO_A, -_ICO_B, 0], [-_ICO_A, -_ICO_B, 0],
    [_ICO_B, 0, _ICO_A], [_ICO_B, 0, -_ICO_A], [-_ICO_B, 0, _ICO_A], [-_ICO_B, 0, -_ICO_A]]
_HDR = re.compile(r"^exposure-[0-9]{6}\.npz$")
# Exact float64 luminance row generated from the IEC primaries/D65 matrix in
# Spectral marble_color_profile.SRGB_TO_XYZ. Keep percentile and pixel output
# parity; the former red coefficient ending in 104 differed by several ULPs.
_PROFILED_LUMA = np.array([.2126390058715103, .7151686787677559, .0721923153607337])
# Same reasoning as marble_preview_worker.py's _retouch_filename: a reused
# filename let the browser treat two different retouches as the same image
# and never redraw. Reserved band keeps this from ever colliding with a
# real per-epoch preview filename.
_retouch_seq = itertools.count()


class _LeasedResponseStream:
    """Keep a published immutable snapshot protected throughout HTTP streaming."""
    def __init__(self, stream, directory, name, lock, atomic):
        self.stream, self.directory, self.name = stream, directory, name
        self.lock, self.atomic = lock, atomic
        self.token = uuid.uuid4().hex
        self.touched = 0.
        self.closed = False

    def touch_locked(self, release=False):
        path = (self.directory / self.name).with_suffix('.leases.json')
        if path.is_symlink(): raise ValueError('invalid scene response lease')
        try:
            if path.stat().st_size > 16384: raise ValueError('oversized scene response lease')
            data = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            data = dict(version=1, name=self.name, transfers={})
        transfers = data.get('transfers')
        now = time.time()
        if (data.get('version') != 1 or data.get('name') != self.name or not isinstance(transfers, dict)
                or len(transfers) > 8 or any(not re.fullmatch(r'[0-9a-f]{32}', key)
                or type(value) not in (int, float) or not math.isfinite(value) for key, value in transfers.items())):
            raise ValueError('invalid scene response leases')
        transfers = {key:value for key,value in transfers.items() if value > now and key != self.token}
        if not release:
            if len(transfers) >= 8: raise ValueError('scene response transfer capacity reached')
            transfers[self.token] = now + 600
        data['transfers'] = transfers
        self.atomic(path, json.dumps(data, separators=(',', ':'), allow_nan=False).encode())
        self.touched = time.monotonic()

    def read(self, size=-1):
        if time.monotonic() - self.touched >= 60:
            with self.lock(self.directory): self.touch_locked()
        return self.stream.read(size)

    def fileno(self): return self.stream.fileno()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self):
        if self.closed: return
        self.closed = True
        self.stream.close()
        try:
            with self.lock(self.directory): self.touch_locked(release=True)
        except (OSError, ValueError, TimeoutError):
            pass  # An expired protective lease is safe; never remove another transfer.


def _retouch_filename() -> str:
    return f"preview-{900000 + (next(_retouch_seq) % 100000):06d}.png"



class PhotoError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _validate_vector_bundle_metadata(field, config):
    """The scalar prefix is unchanged; every v2 section owns a contiguous span."""
    def require(condition):
        if not condition:
            raise PhotoError("invalid light field vector bundle metadata", 503)

    require(isinstance(field.get("sha256"), str) and bool(_DIGEST.fullmatch(field["sha256"])))
    require(isinstance(field.get("scene_sha256"), str) and bool(_DIGEST.fullmatch(field["scene_sha256"])))
    require(isinstance(config.get("geometry_sha256"), str)
            and bool(_DIGEST.fullmatch(config["geometry_sha256"]))
            and field.get("geometry_sha256") == config["geometry_sha256"])
    materials = field.get("materials")
    require(isinstance(materials, dict) and type(materials.get("count")) is int
            and 1 <= materials["count"] <= 65536)
    offset = math.prod(field["dimensions"]) * 12
    directional = field.get("directional")
    require(isinstance(directional, dict))
    internal = config.get("light_field_internal_media", False)
    require(type(internal) is bool)
    if internal:
        receipt = directional.get("internal_media")
        require(field["schema"] == "studio.forward-field.v6"
                and isinstance(receipt, dict) and receipt == _FIELD_INTERNAL_MEDIA
                and receipt.get("entry_packet") is True)
    else:
        require("internal_media" not in directional)
    dims = directional.get("dimensions")
    normalized = field["schema"] in ("studio.forward-field.v4", "studio.forward-field.v5", "studio.forward-field.v6")
    require(isinstance(dims, list) and len(dims) == 3
            and all(type(n) is int and 1 <= n <= (64 if normalized else 32) for n in dims))
    if normalized:
        require(dims == [dims[0]] * 3 and dims[0] in (32, 64)
                and config.get("light_field_grid_size") == dims[0]
                and type(config.get("light_field_grid_size")) is int
                and config.get("light_field_schema") == field["schema"]
                and config.get("light_field_estimator") == _FIELD_ROOT_ESTIMATOR)
    require(directional.get("format") == "rgb-direction-f32le"
            and directional.get("estimator") == (_FIELD_ROOT_ESTIMATOR if normalized else _FIELD_ESTIMATOR)
            and directional.get("direction_space") == "world"
            and directional.get("layout") == "zyx-bin-channel"
            and directional.get("channels") == ["R", "G", "B", "mean_propagation_x",
                                                "mean_propagation_y", "mean_propagation_z"])
    bins = 6
    if field["schema"] in ("studio.forward-field.v3", "studio.forward-field.v4", "studio.forward-field.v5", "studio.forward-field.v6"):
        bins = directional.get("bin_count")
        require(type(bins) is int and bins in (6, 12))
        require(config.get("light_field_stencil", bins) == bins)
        require(directional.get("stencil") == ("axes6-v1" if bins == 6 else "icosahedron12-v1"))
        basis = directional.get("bin_directions")
        expected = _FIELD_AXES if bins == 6 else _FIELD_ICOSAHEDRON
        require(isinstance(basis, list) and len(basis) == bins)
        for row, wanted in zip(basis, expected):
            require(isinstance(row, list) and len(row) == 3
                    and all(type(v) in (int, float) and math.isfinite(v) and abs(v - w) <= 1e-7
                            for v, w in zip(row, wanted)))
        weights = directional.get("weight_from_linear_rgb")
        # Inverse of the producer's existing fixed3 CIE/D65 RGB conversion.
        expected_weights = [1.4400948563066378, .7787163904320042, .781188753261357]
        require(isinstance(weights, list) and len(weights) == 3
                and all(type(v) in (int, float) and math.isfinite(v) and abs(v - w) <= 1e-7
                        for v, w in zip(weights, expected_weights)))
        require(directional.get("bins") == (["+X", "-X", "+Y", "-Y", "+Z", "-Z"] if bins == 6
                                            else [f"I{i}" for i in range(12)]))
    else:
        require(directional.get("bins") == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"])
    sections = [(directional, math.prod(dims) * bins * 6 * 4)]
    for key, stride, fmt in (("triangles", 16, "u32le"), ("sources", 48, "f32le"),
                             ("materials", 64, "f32le"), ("source_pbr16", 64, "f32le")):
        section = field.get(key)
        require(isinstance(section, dict))
        count = section.get("count")
        require(type(count) is int and 0 <= count <= MAX_TRIANGLES
                and section.get("stride_bytes") == stride and type(section.get("stride_bytes")) is int
                and section.get("format") == fmt)
        sections.append((section, count * stride))
    require(field["triangles"].get("channels") == ["triangle_id", "material_id", "flags", "class_mask"])
    bits = field["triangles"].get("class_bits")
    require(isinstance(bits, dict) and bits == {"volumetric": 1, "reflectance": 2, "glass": 4}
            and all(type(v) is int for v in bits.values()))
    require(field["sources"].get("layout") == "baked-source-records")
    if normalized:
        sampling = directional.get("root_sampling")
        require(isinstance(sampling, dict))
        require(sampling.get("version") == "per-complete-pump-inverse-replay-v1"
                and sampling.get("incomplete_dispatch") is False)
        for key in ("source_root_count", "completed_root_sweeps", "recorded_root_jobs",
                    "normalization_divisor", "excluded_partial_root_jobs"):
            require(type(sampling.get(key)) is int and 0 <= sampling[key] <= 2**53 - 1)
        roots, sweeps = sampling["source_root_count"], sampling["completed_root_sweeps"]
        require(roots > 0 and roots == field["sources"]["count"] and sweeps > 0
                and sampling["recorded_root_jobs"] >= roots * sweeps
                and sampling["normalization_divisor"] == sweeps)
        if field["schema"] == "studio.forward-field.v6":
            require(type(directional.get("quantization_divisor")) is int
                    and directional["quantization_divisor"] == 16777216
                    and directional.get("quantization_version") == "fixed24-track-length-v1")
            populations = directional.get("source_populations")
            settings = config.get("emitter_sampling")
            require(isinstance(populations, dict) and isinstance(settings, dict)
                    and settings.get("refresh_each_sweep") is True)
            require(populations.get("version") == "native-angular-refresh-mean-v1"
                    and populations.get("generator") == "native-emissive-triangle-roots-v1")
            for key in ("ray_budget", "base_seed", "first_index", "last_index", "count", "latest_seed"):
                require(type(populations.get(key)) is int and 0 <= populations[key] <= 2**53 - 1)
            require(populations["ray_budget"] == settings.get("ray_budget")
                    and populations["base_seed"] == settings.get("seed")
                    and populations["count"] == sweeps
                    and populations["last_index"] - populations["first_index"] + 1 == sweeps
                    and populations["latest_seed"] == (populations["base_seed"] + populations["last_index"]) % 2147483648)
            for key in ("latest_sources_sha256", "rolling_sha256"):
                require(isinstance(populations.get(key), str) and bool(_DIGEST.fullmatch(populations[key])))
    require(materials.get("layout") == "native-pbr16")
    lineage = field["source_pbr16"]
    ids = lineage.get("material_ids")
    require(lineage.get("layout") == "authored-pbr16" and isinstance(ids, list)
            and len(ids) == lineage["count"]
            and all(type(i) is int and 0 <= i < materials["count"] for i in ids))
    for section, size in sections:
        require(type(section.get("offset_bytes")) is int and section["offset_bytes"] == offset
                and type(section.get("byte_length")) is int and section["byte_length"] == size)
        offset += size
    require(field["byte_length"] == offset)
    if field["schema"] == "studio.forward-field.v5" or (field["schema"] == "studio.forward-field.v6" and config.get("light_field_region") == "subject"):
        receipt = config.get("light_field_subject_region")
        require(isinstance(receipt, dict) and config.get("light_field_region") == "subject")
        # The shared producer helper derived this receipt from verified uploaded
        # geometry before job hashing. Never accept request-authored ROI bounds.
        for key in ("dimensions", "bounds_min_m", "bounds_max_m", "cell_size_m", "margin_m", "region"):
            require(key in receipt and directional.get(key) == receipt[key])
        require(type(directional.get("cell_size_m")) in (int, float)
                and directional["cell_size_m"] == config.get("light_field_cell_size_m")
                and .00005 <= directional["cell_size_m"] <= .01
                and type(directional.get("margin_m")) in (int, float)
                and directional["margin_m"] == config.get("light_field_margin_m"))
        region = directional["region"]
        require(isinstance(region, dict) and region.get("kind") == "subject"
                and region.get("subject_geometry_sha256") == config["geometry_sha256"])
        require(isinstance(region.get("world_geometry_sha256"), str)
                and bool(_DIGEST.fullmatch(region["world_geometry_sha256"])))
        arrays = [directional["bounds_min_m"], directional["bounds_max_m"],
                  region.get("actual_bounds_min_m"), region.get("actual_bounds_max_m")]
        require(all(isinstance(a, list) and len(a) == 3
                    and all(type(v) in (int, float) and math.isfinite(v) for v in a) for a in arrays))
        lo, hi, actual_lo, actual_hi = arrays
        pitch = directional["cell_size_m"]
        require(type(region.get("effective_margin_m")) in (int, float)
                and region["effective_margin_m"] == max(pitch, directional["margin_m"]))
        require(all(a < b and c <= d and a < c and d < b
                    and math.isclose(b - a, n * pitch, rel_tol=1e-12, abs_tol=1e-15)
                    for a, b, c, d, n in zip(lo, hi, actual_lo, actual_hi, dims)))
        for key, expected in (("padding_min_m", [c - a for a, c in zip(lo, actual_lo)]),
                              ("padding_max_m", [b - d for b, d in zip(hi, actual_hi)])):
            padding = region.get(key)
            require(isinstance(padding, list) and len(padding) == 3
                    and all(type(v) in (int, float) and math.isfinite(v)
                            and math.isclose(v, w, rel_tol=1e-12, abs_tol=1e-15)
                            and v + 1e-15 >= region["effective_margin_m"]
                            for v, w in zip(padding, expected)))


def _validate_vector_bundle_payload(stream, field):
    """Verify the same opened immutable file that the HTTP handler will stream."""
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    if digest.hexdigest() != field["sha256"]:
        raise PhotoError("light field content hash mismatch", 503)
    if field["schema"] == "studio.forward-field.v6":
        source = field["sources"]
        stream.seek(source["offset_bytes"])
        digest = hashlib.sha256(stream.read(source["byte_length"]))
        if digest.hexdigest() != field["directional"]["source_populations"]["latest_sources_sha256"]:
            raise PhotoError("light field latest source population hash mismatch", 503)
    # Bound peak allocation independently of the field dimensions.
    for offset, size in ((0, field["triangles"]["offset_bytes"]),
                         (field["sources"]["offset_bytes"],
                          field["sources"]["byte_length"] + field["materials"]["byte_length"]
                          + field["source_pbr16"]["byte_length"])):
        stream.seek(offset)
        while size:
            chunk = stream.read(min(size, 1024 * 1024))
            if not chunk or not np.isfinite(np.frombuffer(chunk, dtype="<f4")).all():
                raise PhotoError("invalid light field vector samples", 503)
            size -= len(chunk)
    triangles = field["triangles"]
    stream.seek(triangles["offset_bytes"])
    start = 0
    while start < triangles["count"]:
        count = min(16384, triangles["count"] - start)
        rows = np.frombuffer(stream.read(count * 16), dtype="<u4").reshape(count, 4)
        if (not np.array_equal(rows[:, 0], np.arange(start, start + count))
                or np.any(rows[:, 1] >= field["materials"]["count"])
                or np.any(rows[:, 3] & np.uint32(0xfffffff8))
                or np.any((rows[:, 3] & 1) == 0)):
            raise PhotoError("invalid light field triangle identities or classifications", 503)
        start += count
    stream.seek(0)


@lru_cache(maxsize=4)
def _directional_contract(root):
    path = Path(root) / "camera_designer" / "studio_directional_field.py"
    spec = importlib.util.spec_from_file_location("studio_source_directional_contract", path)
    if spec is None or spec.loader is None or not path.is_file():
        raise PhotoError("Spectral subject-field support is unavailable", 503)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def _tile_contract(root):
    path = Path(root) / "camera_designer" / "marble_tiles.py"
    spec = importlib.util.spec_from_file_location("studio_source_tile_contract", path)
    if spec is None or spec.loader is None or not path.is_file():
        raise PhotoError("Spectral tiled-photo support is unavailable", 503)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _film_response(value: object) -> dict:
    """Versioned boundary shared with Spectral's film_response contract.

    Centres are wavelengths in nm. Sigma is log2-frequency octaves, not a
    nanometre bandwidth. These are custom peak-normalized sensor curves.
    """
    if (not isinstance(value, dict) or set(value) != {"version", "centers_nm", "sigmas_oct"}
            or type(value.get("version")) is not int or value["version"] != 1):
        raise PhotoError("film_response must contain version 1, centers_nm and sigmas_oct")
    result = {"version": 1}
    for key, low, high in (("centers_nm", 380, 780), ("sigmas_oct", .01, 1.0)):
        values = value[key]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v) for v in values)):
            raise PhotoError(f"film_response.{key} must contain three finite numbers in [{low}, {high}]")
        result[key] = [float(v) for v in values]
    return result


# Sequence entries are camera overrides, not projections or sensor windows.
_CAMERA_POSE_FIELDS = frozenset((
    "eye_m", "target_m", "fov_degrees", "aperture_radius_m", "focus_distance_m",
    "aperture_blades", "aperture_rotation_radians", "lens_selection",
    "camera_back", "back_reference_fov_degrees",
))


def _camera_config(camera_in: dict, resolution: int, spectral_root: Path) -> dict:
    """Use the same bounded analytical controls for a still or every pose."""
    camera = {
        "resolution": resolution,
        "eye_m": _finite_vectors(camera_in.get("eye_m", [0, 0, .07]), 3, "camera.eye_m"),
        "target_m": _finite_vectors(camera_in.get("target_m", [0, 0, .18]), 3, "camera.target_m"),
        "fov_degrees": _number(camera_in.get("fov_degrees", 36), .001 if camera_in.get("camera_back") is not None else FOV_MIN_DEGREES, 100, "camera.fov_degrees"),
        "aperture_radius_m": _number(camera_in.get("aperture_radius_m", .002), 1e-5, PUPIL_RADIUS_MAX_M, "camera.aperture_radius_m"),
        "focus_distance_m": _number(camera_in.get("focus_distance_m", .11), FOCUS_MIN_M, FOCUS_MAX_M, "camera.focus_distance_m"),
        "aperture_blades": int(_number(camera_in.get("aperture_blades", 0), 0, 16, "camera.aperture_blades")),
        "aperture_rotation_radians": _number(camera_in.get("aperture_rotation_radians", 0), -7, 7, "camera.aperture_rotation_radians"),
    }
    if "width" in camera_in or "height" in camera_in:
        camera.update(width=camera_in["width"], height=camera_in["height"])
    try:
        camera = apply_lens_selection(camera, camera_in.get("lens_selection"), spectral_root)
    except (ValueError, OSError) as error:
        raise PhotoError(str(error)) from None
    return camera


def _config(value: dict, saved: dict, spectral_root: Path = SPECTRAL_ROOT) -> tuple[dict, float]:
    if value.get("object_digest") != saved["digest"]:
        raise PhotoError("photo request does not name the saved object")
    seconds = _number(value.get("exposure_seconds", 30), 0.1, 3600, "exposure_seconds")
    substeps = value.get("exposure_substeps", math.ceil(seconds * 60))
    if (type(substeps) is not int or substeps < math.ceil(seconds * 10 - 1e-9)
            or substeps > 216000):
        raise PhotoError("exposure_substeps must give dt <= 0.1 s and total at most 216000")
    iso = _number(value.get("exposure_iso", 100), 25, 6400, "exposure_iso")
    clip = _number(value.get("highlight_clip_percent", 25), 0, 25, "highlight_clip_percent")
    transport = str(value.get("transport", "fixed:1"))
    if transport not in ("fixed:1", "fixed:3", "fixed:4", "fixed:8"):
        raise PhotoError("transport must be fixed:1, fixed:3, fixed:4 or fixed:8")
    if "surface_transport" in value and value["surface_transport"] != _SURFACE_TRANSPORT:
        raise PhotoError("unsupported surface_transport; expected studio-pbr-mixture-v1")
    receive_light_field = value.get("receive_light_field", False)
    if type(receive_light_field) is not bool:
        raise PhotoError("receive_light_field must be a boolean")
    if receive_light_field and transport != "fixed:3":
        raise PhotoError("receive_light_field requires fixed:3 transport")
    film_response = None
    if transport in ("fixed:3", "fixed:4", "fixed:8"):
        film_response = _film_response(value.get("film_response", {
            "version": 1, "centers_nm": [650, 550, 450], "sigmas_oct": [.06, .06, .06]}))
    elif "film_response" in value:
        raise PhotoError("film_response requires fixed:3 RGB transport")
    gamma = _number(value.get("gamma", 1.0), 0.2, 5.0, "gamma")
    bounce_value = value.get("max_bounces", 4)
    max_bounces = _number(bounce_value, 1, 200, "max_bounces")
    if isinstance(bounce_value, bool) or not max_bounces.is_integer():
        raise PhotoError("max_bounces must be a whole number")
    max_bounces = int(max_bounces)
    # Explicit legacy counts remain manual. New clients choose Auto explicitly.
    bounce_mode = value.get("bounce_mode", "manual" if "max_bounces" in value else "auto")
    if bounce_mode not in ("auto", "manual"):
        raise PhotoError("bounce_mode must be auto or manual")
    # A policy cannot rewrite the authenticated saved geometry. Spectral applies
    # the cap to its final scene after eligible analytic boundaries are resolved.
    triangle_budget = value.get("triangle_budget", 250_000)
    if type(triangle_budget) is not int or not 1000 <= triangle_budget <= 1_000_000:
        raise PhotoError("triangle_budget must be a whole number from 1000 to 1000000")
    sphere_coverage = value.get("sphere_coverage", "mesh")
    if sphere_coverage not in ("mesh", "exact"):
        raise PhotoError("sphere_coverage must be mesh or exact")
    camera_in = value.get("camera", {})
    checker_in = value.get("checker", {})
    if not isinstance(camera_in, dict) or not isinstance(checker_in, dict):
        raise PhotoError("camera and checker must be objects")
    resolution = camera_in.get("resolution", 256)
    if type(resolution) is not int or not 128 <= resolution <= 32768:
        raise PhotoError("resolution must be an integer from 128 to 32768")
    custom_size = "width" in camera_in or "height" in camera_in
    oversampling = value.get("oversampling", 1)
    if type(oversampling) is not int or not 1 <= oversampling <= 4:
        raise PhotoError("oversampling must be an integer multiplier from 1 to 4")
    tiled = "tiled" in value or "samples_per_output_pixel" in value or oversampling > 1 or custom_size or resolution not in (128,256,512,768,1024,2048,4096)
    if tiled:
        try:
            _tile_contract(str(spectral_root)).tile_plan(value)
        except ValueError as error:
            raise PhotoError(str(error)) from error
    orbit_frames = value.get("orbit_frames", 1)
    if type(orbit_frames) is not int or orbit_frames not in (1, 8, 16, 32):
        raise PhotoError("orbit_frames must be 1, 8, 16, or 32")
    poses_in = value.get("camera_poses")
    lanes_in = value.get("sensor_lanes")
    if "camera_poses" in value:
        if not isinstance(poses_in, list) or not 1 <= len(poses_in) <= 32:
            raise PhotoError("camera_poses must contain 1..32 camera overrides")
        if orbit_frames != 1 or tiled or "live" in value or value.get("client_intersections", False):
            raise PhotoError("camera_poses cannot be combined with orbit, tiles, live view or client intersections")
        if "projection" in camera_in:
            raise PhotoError("camera_poses use only the existing analytical camera projection")
    if "sensor_lanes" in value:
        if (not isinstance(lanes_in, list) or not 1 <= len(lanes_in) <= 32
                or lanes_in[0] != {}):
            raise PhotoError("sensor_lanes must contain 1..32 camera overrides, starting with the main camera")
        if (poses_in is not None or transport not in ("fixed:3", "fixed:4", "fixed:8") or orbit_frames != 1 or tiled or "live" in value
                or value.get("client_intersections", False) or receive_light_field
                or "exposure_state_frames" not in value):
            raise PhotoError("sensor_lanes require one timed, untiled RGB fixture exposure")
    if receive_light_field and (tiled or orbit_frames != 1 or poses_in is not None or lanes_in is not None):
        raise PhotoError("receive_light_field requires one non-tiled camera without camera_poses")
    camera = _camera_config(camera_in, resolution, spectral_root)
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
        shape = str(source.get("shape", "bare")).lower()
        if shape not in ("bare", "scrim", "grid", "snoot"):
            raise PhotoError("light.shape must be bare, scrim, grid, or snoot")
        filter_in = source.get("filter_rgb")
        filter_rgb = (_finite_vectors(filter_in, 3, "light.filter_rgb")
                     if filter_in is not None else None)
        if filter_rgb is not None and (min(filter_rgb) < 0 or max(filter_rgb) > 1):
            raise PhotoError("light.filter_rgb components must be in [0, 1]")
        lights.append({
            "pos": _finite_vectors(source.get("pos"), 3, "light.pos"),
            "dir": _finite_vectors(source.get("dir"), 3, "light.dir"),
            "power": _number(source.get("power"), .01, 100, "light.power"),
            "label": str(source.get("label", "preview-stage"))[:80],
            "shape": shape,
            "filter_rgb": filter_rgb,
        })
    # Real emissive fixture geometry (the Studio light housings + glow
    # panels, already in real photo-metre coordinates): a small, separate,
    # NOT geometry-verified mesh -- it's live UI state (the fixture rig),
    # not the permanent saved object, so it gets its own bounded validation
    # rather than the saved-object's SHA256 check. Appended into the scene
    # the same way the saved marble mesh is, so Spectral's own emissive-
    # material detection lights the photo from the real fixtures.
    fixtures = None
    fixtures_in = value.get("fixtures")
    if fixtures_in is not None:
        if not isinstance(fixtures_in, dict):
            raise PhotoError("fixtures must be an object")
        verts_in = fixtures_in.get("vertices")
        ids_in = fixtures_in.get("material_ids")
        pbr_in = fixtures_in.get("pbr")
        if not isinstance(verts_in, list) or len(verts_in) % 9 != 0:
            raise PhotoError("fixtures.vertices must be a flat array of 9 floats per triangle")
        n_tri = len(verts_in) // 9
        if n_tri > triangle_budget:
            raise PhotoError("fixtures.vertices exceeds the selected triangle_budget")
        try:
            verts = [float(v) for v in verts_in]
        except (ValueError, TypeError):
            raise PhotoError("fixtures.vertices must contain numbers") from None
        if not all(math.isfinite(v) and abs(v) <= 10 for v in verts):
            raise PhotoError("fixtures.vertices must contain finite scene coordinates")
        if not isinstance(ids_in, list) or len(ids_in) != n_tri:
            raise PhotoError("fixtures.material_ids must have one entry per triangle")
        if not isinstance(pbr_in, list) or len(pbr_in) % 16 != 0 or not pbr_in:
            raise PhotoError("fixtures.pbr must be a flat array of 16 floats per material")
        n_mat = len(pbr_in) // 16
        if n_mat > 64:
            raise PhotoError("fixtures.pbr exceeds 64 materials")
        try:
            mat_ids = [int(i) for i in ids_in]
            pbr_vals = [float(v) for v in pbr_in]
        except (ValueError, TypeError):
            raise PhotoError("fixtures.material_ids/pbr must contain numbers") from None
        if any(i < 0 or i >= n_mat for i in mat_ids):
            raise PhotoError("fixtures.material_ids is outside the supplied pbr table")
        if not all(math.isfinite(v) for v in pbr_vals):
            raise PhotoError("fixtures.pbr must contain finite values")
        fixtures = {"vertices": verts, "material_ids": mat_ids, "pbr": pbr_vals}
        emissive_ids = fixtures_in.get("emissive_material_ids")
        if emissive_ids is not None:
            if (not isinstance(emissive_ids, list) or len(emissive_ids) > n_mat
                    or any(type(i) is not int or i < 0 or i >= n_mat for i in emissive_ids)
                    or len(set(emissive_ids)) != len(emissive_ids)):
                raise PhotoError("invalid fixtures.emissive_material_ids")
            fixtures["emissive_material_ids"] = emissive_ids
        shop_ids = fixtures_in.get("shop_material_ids")
        trigger_emitters = fixtures_in.get("trigger_emitters")
        if shop_ids is not None or trigger_emitters is not None:
            if (not isinstance(shop_ids, list) or not isinstance(trigger_emitters, list)
                    or len(shop_ids) > n_mat or len(trigger_emitters) > n_mat
                    or any(type(i) is not int or i not in (emissive_ids or []) for i in shop_ids)
                    or len(set(shop_ids)) != len(shop_ids)):
                raise PhotoError("invalid timed Shop emitter identities")
            seen_materials, seen_receivers = set(), set()
            for item in trigger_emitters:
                if (not isinstance(item, dict) or set(item) != {"material", "receiver", "gain"}
                        or type(item["material"]) is not int or item["material"] not in shop_ids
                        or item["material"] in seen_materials or not isinstance(item["receiver"], str)
                        or not re.fullmatch(r"shop:[0-9]+", item["receiver"])
                        or item["receiver"] in seen_receivers
                        or isinstance(item["gain"], bool) or not isinstance(item["gain"], (int, float))
                        or not math.isfinite(item["gain"]) or not 0 <= item["gain"] <= 1e6):
                    raise PhotoError("invalid timed Shop trigger emitter")
                seen_materials.add(item["material"]);seen_receivers.add(item["receiver"])
            fixtures["shop_material_ids"] = shop_ids
            fixtures["trigger_emitters"] = trigger_emitters
        if (value.get("light_field_information") == "evaluated-responses-v1"
                and "response_lineage" in fixtures_in):
            lineage = fixtures_in["response_lineage"]
            if (not isinstance(lineage, dict) or set(lineage) != {"version", "parts"}
                    or lineage.get("version") != "studio-response-lineage-v1"
                    or not isinstance(lineage.get("parts"), list)
                    or len(lineage["parts"]) > n_tri):
                raise PhotoError("invalid fixtures.response_lineage")
            parts, cursor, sources = [], 0, set()
            for part in lineage["parts"]:
                if (not isinstance(part, dict) or set(part) != {"source_id", "start", "count"}
                        or not isinstance(part["source_id"], str) or not 1 <= len(part["source_id"]) <= 256
                        or part["source_id"] in sources
                        or type(part["start"]) is not int or part["start"] != cursor
                        or type(part["count"]) is not int or not 0 < part["count"] <= n_tri - cursor):
                    raise PhotoError("fixture response lineage must uniquely cover submitted triangles in order")
                sources.add(part["source_id"])
                cursor += part["count"]
                parts.append(dict(part))
            if cursor != n_tri:
                raise PhotoError("fixture response lineage must cover all submitted triangles")
            fixtures["response_lineage"] = {"version": "studio-response-lineage-v1", "parts": parts}
        if "emitter_profiles" in fixtures_in:
            try:
                import sys
                from pathlib import Path
                root=Path(__file__).resolve().parents[3]/'spectral-analyzer'
                if str(root) not in sys.path:sys.path.insert(0,str(root))
                from camera_designer.studio_named_emitters import validate_named_emitter_refs
                validated=validate_named_emitter_refs(fixtures_in['emitter_profiles'],n_mat)
                fixtures['emitter_profiles']={'version':1,'records':list(validated.values())}
            except (ValueError,TypeError,KeyError) as error:
                raise PhotoError(str(error)) from None
        if trigger_emitters and not {item["material"] for item in trigger_emitters}.issubset(
                {item["material"] for item in fixtures.get("emitter_profiles", {}).get("records", [])}):
            raise PhotoError("triggered Shop light needs a named emitter source")
        if "material_engines" in fixtures_in:
            from .scene_material_engines import validate_scene_engines
            try:
                fixtures["material_engines"] = validate_scene_engines(fixtures_in["material_engines"], verts, mat_ids, pbr_vals)
            except (ValueError, TypeError, KeyError) as error:
                raise PhotoError(str(error)) from None
    physical = saved["physical"]
    scale = _number(physical.get("radius-mm"), .1, 200, "saved physical radius-mm") / 1000
    object_center_m = _finite_vectors(value.get("object_center_m", [0, 0, .18]), 3, "object_center_m")
    # The real Studio room (floor + back wall), not the small generic
    # checker_target plane: same geometry marble_studio.js's checkerboard()
    # draws (floor_y=-1.02, wall_z=-3.1, half-extent=4, in Studio's own
    # view-space units), converted with the same ratio Studio's photo
    # bridge already uses for light positions (photoLightsFromStudio's
    # 0.11/3.45), then recentred on the saved object like everything else.
    _studio_scale = 0.11 / 3.45
    room = {
        "floor_y": -1.02 * _studio_scale + object_center_m[1],
        "wall_z": -3.1 * _studio_scale + object_center_m[2],
        "extent": 4.0 * _studio_scale,
        "checks": 8,
    }
    if type(value.get("camera_visible", True)) is not bool:
        raise PhotoError("camera_visible must be boolean")
    config = {
        "object_digest": saved["digest"],
        "geometry_sha256": saved["geometry_sha256"],
        "exposure_seconds": seconds,
        "exposure_substeps": substeps,
        "orbit_frames": orbit_frames,
        "transport": transport,
        **({"optical_outside_policy":"game-transparent-outside-authored-v1"} if transport=="fixed:8" else {}),
        "gamma": gamma,
        "max_bounces": max_bounces,
        "bounce_mode": bounce_mode,
        "triangle_budget": triangle_budget,
        "sphere_coverage": sphere_coverage,
        # Server-owned scene convention is part of the immutable job digest.
        # New requests must not reuse images whose demo camera stayed at origin.
        "camera_geometry_version": "native-housing-behind-thin-lens-v1",
        "camera_visible": value.get("camera_visible", True),
        "camera_focus_semantics": FOCUS_SEMANTICS,
        "scale_m": scale,
        "object_center_m": object_center_m,
        "camera": camera, "checker": checker, "lights": lights,
        # "room" is disabled for now: a real floor+wall corner with the
        # checker_target's calibration-target reflectance (0.92 white /
        # 0.006 black) traps and amplifies light between the two facing
        # planes under path-traced multi-bounce transport, producing a
        # runaway-bright washed-out image (confirmed via worker.log:
        # display_white ~5.5e5, versus O(1) for a normal exposure). Needs
        # realistic matte-studio reflectance tuning offline before
        # re-enabling, not live against production.
        # "room": room,
    }
    # Omitted and false preserve the pre-existing canonical job digest.
    if "surface_transport" in value:
        if transport not in ("fixed:3", "fixed:4", "fixed:8") or fixtures is None:
            raise PhotoError("surface_transport requires RGB transport and authored fixtures")
        config["surface_transport"] = _SURFACE_TRANSPORT
    if receive_light_field:
        config["receive_light_field"] = True
        # The field estimator changes the requested product, so it must change
        # the canonical job identity rather than reuse a completed fog field.
        estimator = value.get("light_field_estimator", _FIELD_ESTIMATOR)
        if estimator not in (_FIELD_ESTIMATOR, _FIELD_ROOT_ESTIMATOR):
            raise PhotoError("unsupported light_field_estimator")
        normalized = estimator == _FIELD_ROOT_ESTIMATOR
        config["light_field_estimator"] = estimator
        stencil = value.get("light_field_stencil", 12)
        if type(stencil) is not int or stencil not in (6, 12):
            raise PhotoError("light_field_stencil must be 6 or 12")
        config["light_field_stencil"] = stencil
        grid = value.get("light_field_grid_size", 64 if normalized else 32)
        if type(grid) is not int or grid not in ((32, 64) if normalized else (32,)):
            raise PhotoError("light_field_grid_size requires 32, or 64 with track-length-root-sweep-v1")
        # Old tabs omit the estimator and must retain their v3 reader and job
        # identity. The new UI explicitly requests normalized v4 and its grid.
        if normalized:
            config["light_field_grid_size"] = grid
        config["light_field_schema"] = "studio.forward-field.v4" if normalized else "studio.forward-field.v3"
        if "light_field_region" in value:
            if value["light_field_region"] != "subject" or not normalized:
                raise PhotoError("light_field_region requires subject and track-length-root-sweep-v1")
            pitch = value.get("light_field_cell_size_m", .0005)
            margin = value.get("light_field_margin_m", .001)
            if type(pitch) not in (int, float) or not math.isfinite(pitch) or not .00005 <= pitch <= .01:
                raise PhotoError("subject light_field_cell_size_m must be between 0.00005 and 0.01")
            if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= .032:
                raise PhotoError("light_field_margin_m must be finite and between 0 and 0.032")
            config.update(light_field_region="subject", light_field_cell_size_m=pitch,
                          light_field_margin_m=margin, light_field_schema="studio.forward-field.v5")
    if tiled:
        config["tiled"] = value.get("tiled", {"tile_size": 1024})
        config["oversampling"] = oversampling
        if "samples_per_output_pixel" in value:
            config["samples_per_output_pixel"] = value["samples_per_output_pixel"]
    participation=value.get("client_intersections",False)
    if type(participation) is not bool:
        raise PhotoError("client_intersections must be a boolean")
    if participation:
        if transport!="fixed:3" or orbit_frames!=1:
            raise PhotoError("Client intersections require RGB transport and one camera")
        config["client_intersections"]=True
    if film_response is not None:
        config["film_response"] = film_response
        # Server-owned profile revision participates in the job digest.
        config["output_color_profile"] = "marble-fixed3-cie1931-d65-v1"
    if "exposure_iso" in value or "highlight_clip_percent" in value:
        config["exposure_iso"] = iso
        config["highlight_clip_percent"] = clip
    if fixtures is not None:
        config["fixtures"] = fixtures
    if "exposure_state_frames" in value:
        raw_frames = value["exposure_state_frames"]
        if (fixtures is None or transport not in ("fixed:3", "fixed:4", "fixed:8") or tiled or orbit_frames != 1
                or "live" in value or participation or receive_light_field
                or not isinstance(raw_frames, list) or not 1 <= len(raw_frames) <= 4096
                or "emissive_material_ids" not in fixtures):
            raise PhotoError("timed Studio exposure requires one ordinary RGB fixture photograph")
        from camera_designer.studio_named_emitters import validate_named_emitter_refs
        base = np.asarray(fixtures["pbr"], np.float32).reshape(-1, 16)
        fixed = np.ones(base.shape, bool)
        fixed[fixtures["emissive_material_ids"], 8:11] = False
        initial_refs = validate_named_emitter_refs(fixtures.get("emitter_profiles"), len(base))
        if not set(initial_refs).issubset(fixtures["emissive_material_ids"]):
            raise PhotoError("timed emitter profile has no prepared material identity")
        frames = []
        cursor = 0.0
        for raw_frame in raw_frames:
            if (not isinstance(raw_frame, dict) or
                    set(raw_frame) not in ({"t0", "t1", "pbr", "emitter_profiles"},
                                           {"t0", "t1", "pbr", "emitter_profiles", "vertices", "material_ids"})):
                raise PhotoError("invalid timed Studio frame")
            t0, t1 = raw_frame["t0"], raw_frame["t1"]
            if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (t0, t1))
                    or abs(t0 - cursor) > 1e-5 or not cursor < t1 <= seconds + 1e-5):
                raise PhotoError("timed Studio frames must cover contiguous positive intervals")
            tick = seconds / substeps
            if (abs(t0 / tick - round(t0 / tick)) > 1e-4
                    or abs(t1 / tick - round(t1 / tick)) > 1e-4):
                raise PhotoError("timed Studio frames must end on exposure substep boundaries")
            try:
                current = np.asarray(raw_frame["pbr"], np.float32).reshape(base.shape)
                refs = validate_named_emitter_refs(raw_frame["emitter_profiles"], len(base))
            except (ValueError, TypeError, OverflowError) as error:
                raise PhotoError(f"invalid timed Studio light state: {error}") from None
            if (not np.isfinite(current).all() or not np.array_equal(current[fixed], base[fixed])
                    or set(refs) != set(initial_refs)
                    or any((refs[k]["profile"], refs[k]["revision"]) !=
                           (initial_refs[k]["profile"], initial_refs[k]["revision"]) for k in refs)):
                raise PhotoError("timed Studio frame changed non-light scene state")
            geometry = {}
            if "vertices" in raw_frame:
                try:
                    moving = np.asarray(raw_frame["vertices"], np.float32).reshape(-1, 3, 3)
                    moving_ids = raw_frame["material_ids"]
                except (ValueError, TypeError, OverflowError):
                    raise PhotoError("invalid timed Studio geometry slice") from None
                if (len(moving) > triangle_budget or not np.isfinite(moving).all()
                        or np.any(np.abs(moving) > 10) or not isinstance(moving_ids, list)
                        or len(moving_ids) != len(moving)
                        or any(type(i) is not int or i < 0 or i >= n_mat for i in moving_ids)):
                    raise PhotoError("invalid timed Studio geometry slice")
                geometry = {"vertices": moving.ravel().tolist(), "material_ids": moving_ids}
            frames.append({"t0": float(t0), "t1": float(t1), "pbr": current.ravel().tolist(),
                           "emitter_profiles": {"version": 1, "records": list(refs.values())}, **geometry})
            cursor = float(t1)
        if abs(cursor - seconds) > 1e-5:
            raise PhotoError("timed Studio frames must cover the full shutter interval")
        config["exposure_state_frames"] = frames
    if "camera_trigger_graph" in value:
        if "exposure_state_frames" not in config or not fixtures.get("trigger_emitters"):
            raise PhotoError("camera trigger graph needs timed trigger emitters")
        graph = value["camera_trigger_graph"]
        if not isinstance(graph, dict) or len(json.dumps(graph)) > 65536:
            raise PhotoError("camera trigger graph must be a bounded RoutingGraph object")
        try:
            from routing_engine import RoutingGraph
            config["camera_trigger_graph"] = RoutingGraph.from_dict(graph).to_dict()
        except (TypeError, ValueError, KeyError) as error:
            raise PhotoError(f"invalid camera trigger graph: {error}") from None
    if "camera_trigger_generators" in value:
        if "camera_trigger_graph" not in config:
            raise PhotoError("camera trigger generators need a routed trigger graph")
        try:
            from camera_software.trigger_generators import validate_generators
            generators = list(validate_generators(value["camera_trigger_generators"]))
            if any(item["key"] not in config["camera_trigger_graph"]["nodes"] for item in generators):
                raise ValueError("generator source is missing from the routed trigger graph")
            config["camera_trigger_generators"] = generators
        except (TypeError, ValueError, KeyError) as error:
            raise PhotoError(f"invalid camera trigger generators: {error}") from None
    if fixtures and fixtures.get("trigger_emitters"):
        timing = value.get("camera_trigger_timing", {"shop_cutoff_lead_s": .001, "flash_duration_s": .010})
        if not isinstance(timing, dict) or set(timing) != {"shop_cutoff_lead_s", "flash_duration_s"}:
            raise PhotoError("camera trigger timing needs cutoff lead and flash duration")
        lead = _number(timing["shop_cutoff_lead_s"], 1e-6, 1.0, "shop_cutoff_lead_s")
        pulse = _number(timing["flash_duration_s"], 1e-6, 1.0, "flash_duration_s")
        if lead + pulse >= seconds:
            raise PhotoError("camera trigger timing must fit inside the shutter interval")
        config["camera_trigger_timing"] = {"shop_cutoff_lead_s": lead, "flash_duration_s": pulse}
    elif "camera_trigger_timing" in value:
        raise PhotoError("camera trigger timing needs timed trigger emitters")
    if "live" in value:
        from .marble_live import live_settings
        if transport != "fixed:3" or orbit_frames != 1:
            raise PhotoError("live rays require RGB transport and one camera")
        settings = live_settings(value["live"], camera, client_intersections=participation)
        session = value["live"].get("session_id", "")
        if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f]{32}", session):
            raise PhotoError("invalid live view session")
        settings["session_id"] = session
        config["live"] = settings
    if "emitter_sampling" in value:
        sampling = value["emitter_sampling"]
        keys = {"version", "ray_budget", "seed", "refresh_each_sweep"}
        if not isinstance(sampling, dict) or set(sampling) != keys:
            raise PhotoError("emitter_sampling requires version, ray_budget, seed, and refresh_each_sweep")
        if type(sampling["version"]) is not int or sampling["version"] != 1:
            raise PhotoError("unsupported emitter_sampling version")
        if type(sampling["ray_budget"]) is not int or not 128 <= sampling["ray_budget"] <= 65536:
            raise PhotoError("emitter_sampling ray_budget must be an integer between 128 and 65536")
        if type(sampling["seed"]) is not int or not 0 <= sampling["seed"] <= 2147483647:
            raise PhotoError("emitter_sampling seed must be an integer between 0 and 2147483647")
        if type(sampling["refresh_each_sweep"]) is not bool:
            raise PhotoError("emitter_sampling refresh_each_sweep must be a boolean")
        if transport != "fixed:3" or fixtures is None:
            raise PhotoError("emitter_sampling requires fixed:3 and authored fixtures")
        if sampling["refresh_each_sweep"]:
            if ("live" not in config or not receive_light_field
                    or config.get("light_field_estimator") != _FIELD_ROOT_ESTIMATOR):
                raise PhotoError("emitter_sampling refresh requires live root-sweep light field")
            if sampling["ray_budget"] > config["live"]["ray_batch_size"]:
                raise PhotoError("emitter_sampling ray_budget exceeds live ray_batch_size")
        # Immutable source sampling affects the product identity. Rounded actual
        # root capacity is checked by the existing CPU scene admission next.
        config["emitter_sampling"] = dict(sampling)
        if sampling["refresh_each_sweep"]:
            config["light_field_schema"] = "studio.forward-field.v6"
    internal = value.get("light_field_internal_media", False)
    if type(internal) is not bool:
        raise PhotoError("light_field_internal_media must be a boolean")
    if internal:
        if config.get("light_field_schema") != "studio.forward-field.v6":
            raise PhotoError("light_field_internal_media requires refreshed v6 light field")
        if any(key in value for key in ("volume_extinction", "volume_extinction_sigma_t_m_inv", "volume_extinction_bounds_m")):
            raise PhotoError("light_field_internal_media cannot be combined with external volume_extinction")
        config["light_field_internal_media"] = True
    if "light_field_information" in value:
        if value["light_field_information"] != "evaluated-responses-v1":
            raise PhotoError("unsupported light_field_information")
        if (not receive_light_field or config.get("light_field_schema") != "studio.forward-field.v6"
                or not internal):
            raise PhotoError("evaluated response information requires an explicit refreshed internal v6 field request")
        config["light_field_information"] = "evaluated-responses-v1"
    # Optics use the complete image, never a tile's temporary allocation.
    # Live's established shader contract uses the displayed viewport aspect.
    aspect = config["live"]["aspect"] if "live" in config else camera.get("width", resolution) / camera.get("height", resolution)
    try:
        config["camera"] = apply_camera_back(camera, camera_in.get("camera_back"), aspect,
                                             camera_in.get("back_reference_fov_degrees"))
    except ValueError as error:
        raise PhotoError(str(error)) from None
    if poses_in is not None:
        cameras = []
        for index, pose in enumerate(poses_in):
            if not isinstance(pose, dict) or set(pose) - _CAMERA_POSE_FIELDS:
                raise PhotoError(f"camera_poses[{index}] has unsupported camera fields")
            pose_in = {**camera_in, **pose}
            # An explicit FOV is a new reference, not the inherited back's
            # already-cropped field. A removed back restores its reference.
            if "fov_degrees" in pose and "back_reference_fov_degrees" not in pose:
                pose_in.pop("back_reference_fov_degrees", None)
            elif pose.get("camera_back", True) is None and "fov_degrees" not in pose:
                pose_in["fov_degrees"] = camera_in.get("back_reference_fov_degrees", camera["fov_degrees"])
            try:
                resolved = _camera_config(pose_in, resolution, spectral_root)
                if float(np.linalg.norm(np.asarray(resolved["target_m"], np.float32) - np.asarray(resolved["eye_m"], np.float32))) <= 1e-6:
                    raise ValueError("camera eye and target must differ at sensor precision")
                resolved = apply_camera_back(resolved, pose_in.get("camera_back"), aspect,
                                             pose_in.get("back_reference_fov_degrees"))
            except ValueError as error:
                raise PhotoError(f"camera_poses[{index}]: {error}") from None
            cameras.append(resolved)
        config["camera_poses"] = cameras
    if lanes_in is not None:
        lanes = [config["camera"]]
        for index, pose in enumerate(lanes_in[1:], 1):
            if not isinstance(pose, dict) or set(pose) - (_CAMERA_POSE_FIELDS | {"sensor_transport", "sensor_output"}):
                raise PhotoError(f"sensor_lanes[{index}] has unsupported camera fields")
            lane_transport = pose.get("sensor_transport", "fixed:3")
            if lane_transport not in ("fixed:1", "fixed:3", "fixed:4", "fixed:8"):
                raise PhotoError(f"sensor_lanes[{index}] has unsupported sensor transport")
            if pose.get("sensor_output", "rgb") not in ("rgb", "mono"):
                raise PhotoError(f"sensor_lanes[{index}] has unsupported sensor output")
            pose_in = {**camera_in, **{key: item for key, item in pose.items() if key not in ("sensor_transport", "sensor_output")}}
            if "fov_degrees" in pose and "back_reference_fov_degrees" not in pose:
                pose_in.pop("back_reference_fov_degrees", None)
            elif pose.get("camera_back", True) is None and "fov_degrees" not in pose:
                pose_in["fov_degrees"] = camera_in.get("back_reference_fov_degrees", camera["fov_degrees"])
            try:
                resolved = _camera_config(pose_in, resolution, spectral_root)
                resolved = apply_camera_back(resolved, pose_in.get("camera_back"), aspect,
                                             pose_in.get("back_reference_fov_degrees"))
            except ValueError as error:
                raise PhotoError(f"sensor_lanes[{index}]: {error}") from None
            if lane_transport != "fixed:3":
                resolved["sensor_transport"] = lane_transport
            if pose.get("sensor_output") == "mono":
                resolved["sensor_output"] = "mono"
            lanes.append(resolved)
        config["sensor_lanes"] = lanes
    if "calibration_regions" in value:
        maps = value["calibration_regions"]
        names = {"blackCenter", "gray18", "whiteRing", "blackOuter",
                 "red", "green", "blue", "yellow"}
        if (lanes_in is None or not isinstance(maps, list) or len(maps) != len(config["sensor_lanes"])):
            raise PhotoError("calibration regions need one map per sensor lane")
        checked = []
        for lane, item in enumerate(maps):
            if (not isinstance(item, dict) or set(item) != {"version", "width", "height", "regions"}
                    or item["version"] != 1 or item["width"] != resolution or item["height"] != resolution
                    or not isinstance(item["regions"], dict) or set(item["regions"]) != names):
                raise PhotoError(f"calibration lane {lane} has an invalid sparse map")
            regions = {}
            total = 0
            for name in sorted(names):
                runs = item["regions"][name]
                if not isinstance(runs, list) or len(runs) > 8192:
                    raise PhotoError("calibration region has too many runs")
                last_end = -1
                regions[name] = []
                for pair in runs:
                    if (not isinstance(pair, list) or len(pair) != 2 or
                            any(type(n) is not int for n in pair)):
                        raise PhotoError("calibration run must have integer start and length")
                    start, length = pair
                    if (start < 0 or length < 1 or start <= last_end or
                            start + length > resolution * resolution or
                            start // resolution != (start + length - 1) // resolution):
                        raise PhotoError("calibration run leaves its sensor row or overlaps")
                    last_end = start + length - 1
                    total += length
                    regions[name].append([start, length])
            if total > resolution * resolution:
                raise PhotoError("calibration map selects more pixels than the sensor owns")
            checked.append({"version": 1, "width": resolution, "height": resolution,
                            "regions": regions})
        config["calibration_regions"] = checked
    return config, seconds


def decode_upload(raw: bytes, saved_objects, spectral_root: Path = SPECTRAL_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise PhotoError("marble photo upload exceeds byte limit", 413)
    if len(raw) < 16 or raw[:4] != b"MRP1":
        raise PhotoError("expected MRP1 saved-mesh photo payload")
    config_size, count, reserved = struct.unpack_from("<III", raw, 4)
    # Fixture triangles follow the explicit scene budget rather than a hidden
    # 5000-triangle cap. Keep the independent 2 MiB serialized-config bound:
    # this bounds JSON parsing/allocation before the geometry policy is read.
    # Whole upload stays within 48 MiB, including saved geometry; the final
    # combined marble + fixture count is checked by preflight_triangle_budget.
    if reserved or not 2 <= config_size <= 2 * 1024 * 1024 or not 1 <= count <= MAX_TRIANGLES:
        raise PhotoError("invalid MRP1 header")
    expected = 16 + config_size + count * (9 * 4 + 9 * 4 + 4)
    if len(raw) != expected:
        raise PhotoError("MRP1 payload length does not match its triangle count")
    try:
        request = json.loads(raw[16:16 + config_size])
    except (UnicodeError, ValueError):
        raise PhotoError("invalid MRP1 config JSON") from None
    if not isinstance(request, dict):
        raise PhotoError("invalid photo request")
    snapshot = request.get("object_snapshot")
    if snapshot is not None:
        # Exposure ownership needs an immutable validated source, not a library
        # entry. Keep the exact same object validator, digest and mesh proof.
        # This job-local source never changes autosave or fills the library.
        from .marble_objects import validate_object, source_digest, MAX_BYTES
        if "live" in request:
            raise PhotoError("exposure snapshots are for still photographs only")
        if request.get("object_digest") is not None:
            raise PhotoError("choose a saved object or an exposure snapshot, not both")
        try:
            raw_snapshot = json.dumps(snapshot, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if len(raw_snapshot) > MAX_BYTES:
                raise ValueError("exposure snapshot exceeds object size limit")
            validate_object(snapshot)
            digest = source_digest(snapshot)
        except (ValueError, TypeError, OverflowError) as error:
            raise PhotoError(f"invalid exposure snapshot: {error}") from error
        saved = {**snapshot, "digest": digest}
        request = {**request, "object_digest": digest}
    else:
        if not _DIGEST.fullmatch(str(request.get("object_digest", ""))):
            raise PhotoError("a saved object digest or exposure snapshot is required")
        try:
            saved = saved_objects.get(request["object_digest"])
        except (KeyError, FileNotFoundError, ValueError):
            raise PhotoError("saved marble object is unavailable", 404) from None
    imported_packet = None
    if saved.get("generator") == "meowdoku-imported-object-v1":
        try:
            imported_packet = saved_objects.validate_source(saved)
        except (ValueError, AttributeError, KeyError) as error:
            raise PhotoError(f"imported source unavailable: {error}", 404) from error
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
    if imported_packet is not None and packed.tobytes() != imported_packet.tobytes():
        raise PhotoError("uploaded triangles differ from the imported source asset")
    if hashlib.sha256(packed.tobytes()).hexdigest() != saved["geometry_sha256"]:
        raise PhotoError("uploaded triangles differ from the saved marble object")
    table = saved["materials"]["tables"]["pbr"]
    if table["stride"] != 16:
        raise PhotoError("saved material table is not PBR16")
    pbr = np.asarray(table["data"], dtype=np.float32).reshape(-1, 16)
    if np.any(ids >= len(pbr)):
        raise PhotoError("saved triangle material ID exceeds the PBR table")
    config, seconds = _config(request, saved, spectral_root)
    if "authoring" in saved["materials"]:
        from .marble_material_authoring import validate_material_authoring
        try:
            validate_material_authoring(saved["materials"]["authoring"],len(pbr),spectral_root)
        except (ValueError,TypeError,KeyError) as error:
            raise PhotoError(str(error)) from error
        config["material_authoring"]=saved["materials"]["authoring"]
    if snapshot is not None:
        config["exposure_snapshot"] = snapshot
    # Only the immutable saved design may author medium adjacency. The request
    # cannot replace it, just as it cannot replace the verified material table.
    from .marble_media import saved_boundary_media
    from .marble_region_media import saved_region_media
    try:
        region_media = saved_region_media(saved, positions.reshape(count,3,3),ids,pbr)
        boundary_media = (None if region_media is not None else
                          saved_boundary_media(saved, positions.reshape(count, 3, 3), ids, pbr))
    except (ValueError, KeyError, TypeError) as exc:
        raise PhotoError(str(exc)) from exc
    if region_media is not None:
        config["region_media"] = region_media
        if "canonical_volume_optics" in saved["design"]:
            config["canonical_volume_optics"]=saved["design"]["canonical_volume_optics"]
        else:
            config["optical_media"] = saved["design"]["optical_media"]
        config["boundary_transport"] = "studio-region-media-v1"
    if boundary_media is not None:
        config["boundary_media"] = boundary_media
        config["boundary_transport"] = "studio-explicit-media-v1"
        if "optical_media" in saved["design"]:
            # Only immutable, digest-covered optical definitions enter preflight.
            # Spectral validates spectrum/units and projects to its native n/k.
            config["optical_media"] = saved["design"]["optical_media"]
    from .marble_spheres import saved_analytic_spheres, preflight_triangle_budget
    try:
        spheres = saved_analytic_spheres(saved, positions.reshape(count, 3, 3), ids, pbr, config)
        if spheres is not None:
            config["analytic_spheres"] = spheres
        preflight_triangle_budget(positions.reshape(count, 3, 3), config, spheres)
        if config.get("light_field_region") == "subject":
            receipt = _directional_contract(str(spectral_root)).subject_field_region(positions.reshape(count, 3, 3), config)
            config["light_field_subject_region"] = receipt
            config["light_field_grid_size"] = receipt["dimensions"][0]
    except (ValueError, KeyError, TypeError) as exc:
        raise PhotoError(str(exc)) from exc
    return positions.copy().reshape(count, 3, 3), ids.astype(np.int32), pbr, config, seconds


def _display_params(params: dict) -> dict:
    """Bounded display controls; the recorded sensor profile is not changed."""
    normalization = params.get("normalization", "auto99")
    tone_curve = params.get("tone_curve", "exponential")
    if normalization not in ("auto99", "manual", "percentile", "specular"):
        raise PhotoError("retouch.normalization must be auto99, manual, percentile or specular", 400)
    if tone_curve not in ("exponential", "linear"):
        raise PhotoError("retouch.tone_curve must be exponential or linear", 400)
    display_look = params.get("display_look", "reference")
    if display_look not in ("reference", "color_priority"):
        raise PhotoError("retouch.display_look must be reference or color_priority", 400)
    if normalization == "specular":
        display_look = "reference"  # the explicitly requested white-clipping preset
        tone_curve = "linear"
    if isinstance(params.get("normalization_white"), bool):
        raise PhotoError("retouch.normalization_white must be a number, not a boolean", 400)
    try:
        white = float(params.get("normalization_white", 1.0))
    except (TypeError, ValueError, OverflowError):
        raise PhotoError("retouch.normalization_white must be a number", 400) from None
    if not math.isfinite(white) or not 1e-6 <= white <= 1e12:
        raise PhotoError("retouch.normalization_white must be finite in [1e-6,1e12]", 400)
    result = {"normalization": normalization, "normalization_white": white, "tone_curve": tone_curve, "display_look": display_look}
    enabled = params.get("firefly_filter", False)
    if type(enabled) is not bool:
        raise PhotoError("retouch.firefly_filter must be a boolean", 400)
    result["firefly_filter"] = enabled
    for name, low, high, default in (("saturation", 0., 2., 1.),
                                     ("vibrance", -1., 1., 0.),
                                     ("black_point", 0., .5, 0.),
                                     ("firefly_threshold", 2., 64., 8.),
                                     ("firefly_strength", 0., 1., 1.),
                                     ("normalization_percentile", 1., 100.,
                                      95. if normalization == "specular" else 99.)):
        value = params.get(name, default)
        if isinstance(value, bool):
            raise PhotoError(f"retouch.{name} must be a number, not a boolean", 400)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise PhotoError(f"retouch.{name} must be a number", 400) from None
        if not math.isfinite(value) or not low <= value <= high:
            raise PhotoError(f"retouch.{name} must be finite in [{low},{high}]", 400)
        result[name] = value
    return result


def _display_white_reference(linear: np.ndarray, params: dict) -> float:
    controls = _display_params(params)
    if controls["normalization"] == "manual":
        return controls["normalization_white"]
    luminance = np.maximum(np.asarray(linear, np.float64) @ _PROFILED_LUMA, 0.0)
    positive = luminance[luminance > 0]
    percentile = 99. if controls["normalization"] == "auto99" else controls["normalization_percentile"]
    return max(float(np.percentile(positive, percentile)) if len(positive) else 0.0, 1e-6)


def _display_metadata(linear: np.ndarray, params: dict) -> dict:
    controls = _display_params(params)
    white = _display_white_reference(linear, controls)
    result = dict(controls, effective_white=white,
        tone_formula="1-exp(-2*Y/white)" if controls["tone_curve"] == "exponential" else "clip(Y/white,0,1)",
        transfer="standard sRGB once, followed by optional artistic gamma/intensity",
        gamut_mapping="neutral-axis compression at mapped luminance",
        white_units="relative profiled linear luminance before artistic gains",
        black_point_formula="x=max((Y/white-black_point)/(1-black_point),0); tone uses x",
        chroma_formula="chroma *= saturation*(1+vibrance*(1-C)); C=clip((maxRGB-minRGB)/maxAbsRGB,0,1)",
        creative_controls="Artistic linear-sRGB adjustments; not calibration or perceptually uniform saturation",
        firefly_method="if interior luminance exceeds threshold times the maximum of 8 neighbors, blend toward that maximum by strength; RGB ratios preserved; after gains, before tone",
        firefly_warning="May reduce genuine isolated glints or sharp highlight peaks. Brightest-neighbor comparison protects connected detail; borders, raw HDR and white reference stay unchanged.",
        effective_percentile=(None if controls["normalization"] == "manual" else
                              99. if controls["normalization"] == "auto99" else controls["normalization_percentile"]),
        specular_clip_mode=controls["normalization"] == "specular",
        highlight_policy=("Linear normalized luminance at or above white is clipped to display white; lower values remain graded."
                          if controls["tone_curve"] == "linear" else "Exponential luminance shoulder approaches display white."))
    if controls["black_point"] != 0:
        result["tone_formula"] = ("1-exp(-2*x)" if controls["tone_curve"] == "exponential"
                                  else "clip(x,0,1)")
    if result["display_look"] == "color_priority":
        result["gamut_mapping"] = "scene-linear negative-channel neutral compression; ratio-preserving maximum-channel shoulder"
        result["tone_formula"] = "map maxRGB/white with selected curve, scale all channels together after black point"
        result["highlight_policy"] = "Color-priority custom look preserves positive RGB ratios through the shoulder; displayed luminance may be lower. Not Canon or ACES."
    return result


def _suppress_fireflies(rgb, controls, luma_weights):
    """Same non-destructive display-only spatial cap as Spectral's helper."""
    if (not controls["firefly_filter"] or controls["firefly_strength"] == 0
            or rgb.ndim != 3 or min(rgb.shape[:2]) < 3):
        return rgb
    y = np.maximum(rgb @ luma_weights, 0.0)
    center = y[1:-1, 1:-1]
    neighbor_max = np.zeros_like(center)
    h, w = center.shape
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                np.maximum(neighbor_max, y[dy:dy+h, dx:dx+w], out=neighbor_max)
    isolated = center > neighbor_max * controls["firefly_threshold"]
    ratio = np.divide(neighbor_max, center, out=np.ones_like(center),
                      where=isolated & (center > 0))
    scale = (1.0 - controls["firefly_strength"]) + controls["firefly_strength"] * ratio
    filtered = rgb.copy()
    filtered[1:-1, 1:-1] *= scale[..., None]
    return filtered


def color_priority_map(rgb, controls, white, luma_weights):
    """Optional custom display look; never changes sensor/profile measurements.

    The old fixed-Y gamut projection makes bright green pale and cancels a
    saturation increase once a channel reaches the display boundary. Here
    negative channels are first compressed toward neutral in scene-linear RGB;
    one scalar shoulder then maps the maximum channel. Positive RGB ratios
    survive that shoulder. This trades displayed luminance for color, not a
    colorimetric/Canon/ACES rendering intent. Neutral ramps remain unchanged.
    """
    y = np.maximum(rgb @ luma_weights, 0.0)
    chroma = rgb - y[..., None]
    spread = rgb.max(axis=-1) - rgb.min(axis=-1)
    magnitude = np.abs(rgb).max(axis=-1)
    colorfulness = np.clip(np.divide(spread, magnitude,
        out=np.zeros_like(spread), where=magnitude > 0), 0.0, 1.0)
    factor = controls["saturation"] * (1.0 + controls["vibrance"] * (1.0 - colorfulness))
    chroma = chroma * factor[..., None]
    low = chroma.min(axis=-1)
    compression = np.minimum(1.0, np.divide(y, -low,
        out=np.ones_like(y), where=low < 0))
    positive = np.maximum(y[..., None] + compression[..., None] * chroma, 0.0)
    normalized_y = y / white
    black_adjusted = np.maximum((normalized_y - controls["black_point"]) /
                                (1.0 - controls["black_point"]), 0.0)
    positive *= np.divide(black_adjusted, normalized_y,
        out=np.zeros_like(y), where=normalized_y > 0)[..., None] / white
    peak = positive.max(axis=-1)
    mapped_peak = (-np.expm1(-2.0 * peak) if controls["tone_curve"] == "exponential"
                   else np.clip(peak, 0.0, 1.0))
    return np.clip(positive * np.divide(mapped_peak, peak,
        out=np.zeros_like(peak), where=peak > 0)[..., None], 0.0, 1.0)


def _retonemap_linear(linear: np.ndarray, params: dict, *, colorimetric: bool = False) -> np.ndarray:
    """The same re-tonemap formula marble_preview_worker.py's _render_retouch()
    applies to the live in-memory texture -- duplicated here (not imported
    across the turing/spectral-analyzer package boundary) for the case where
    the job's worker process has already exited and there is no live GPU
    texture left to read, only the one persisted final HDR file.
    """
    luma_weights = _PROFILED_LUMA if colorimetric else [.2126, .7152, .0722]
    luma = np.sum(linear * luma_weights, axis=2)
    if colorimetric:
        luma = np.maximum(luma, 0.0)
    nonzero = luma[luma > 0]
    base_white = max(float(np.percentile(nonzero, 99)) if len(nonzero) else 0.0, 1e-6)
    controls = _display_params(params) if colorimetric else None
    if colorimetric:
        base_white = _display_white_reference(linear, controls)
    exposure = float(params.get("exposure", 0.0))
    gamma = max(1.0e-3, float(params.get("gamma", 1.0)))
    red = float(params.get("red", 1.0))
    green = float(params.get("green", 1.0))
    blue = float(params.get("blue", 1.0))
    white_balance = float(params.get("whitebalance", 0.0))
    tint = float(params.get("tint", 0.0))
    intensity = max(0.0, float(params.get("intensity", 1.0)))
    exp_mul = 2.0 ** exposure
    wb = 2.0 ** white_balance
    tn = 2.0 ** tint
    gain = np.asarray([red * exp_mul * wb, green * exp_mul / tn, blue * exp_mul / wb], np.float64)
    k = 2.0 / base_white
    if colorimetric:
        # Same display-only recipe as Spectral render_linear_srgb(). Input
        # here is the separately persisted profiled linear sRGB, never raw
        # sensor channels. Do not apply the sensor color matrix again.
        rgb = np.asarray(linear, np.float64) * gain
        rgb = _suppress_fireflies(rgb, controls, luma_weights)
        luminance = np.maximum(rgb @ luma_weights, 0.0)
        target_y = (-np.expm1(-luminance * k) if controls["tone_curve"] == "exponential"
                    else np.clip(luminance / base_white, 0.0, 1.0))
        if controls["black_point"] != 0:
            normalized = np.maximum((luminance / base_white - controls["black_point"]) /
                                    (1.0 - controls["black_point"]), 0.0)
            target_y = (-np.expm1(-2.0 * normalized) if controls["tone_curve"] == "exponential"
                        else np.clip(normalized, 0.0, 1.0))
        scaled = rgb * np.divide(target_y, luminance, out=np.zeros_like(luminance),
                                where=luminance > 0)[..., None]
        chroma = scaled - target_y[..., None]
        if controls["saturation"] != 1.0 or controls["vibrance"] != 0.0:
            spread = scaled.max(axis=-1) - scaled.min(axis=-1)
            magnitude = np.abs(scaled).max(axis=-1)
            colorfulness = np.clip(np.divide(spread, magnitude,
                out=np.zeros_like(spread), where=magnitude > 0), 0.0, 1.0)
            factor = controls["saturation"] * (1.0 + controls["vibrance"] * (1.0 - colorfulness))
            chroma = chroma * factor[..., None]
        low, high = chroma.min(axis=-1), chroma.max(axis=-1)
        saturation = np.ones_like(target_y)
        saturation = np.minimum(saturation, np.divide(target_y, -low,
            out=np.ones_like(target_y), where=low < 0))
        saturation = np.minimum(saturation, np.divide(1.0 - target_y, high,
            out=np.ones_like(target_y), where=high > 0))
        mapped = np.clip(target_y[..., None] + saturation[..., None] * chroma, 0.0, 1.0)
    else:
        mapped = 1.0 - np.exp(-np.maximum(linear * gain, 0.0) * k)
    if colorimetric and controls["display_look"] == "color_priority":
        mapped = color_priority_map(rgb, controls, base_white, luma_weights)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped,
                    1.055 * np.power(np.maximum(mapped, 0.0), 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0) ** (1.0 / gamma)
    srgb = np.clip(srgb * intensity, 0.0, 1.0)
    return np.rint(srgb * 255).astype(np.uint8) if colorimetric else np.uint8(srgb * 255)


class PhotoJobs:
    """One active GPU exposure per server; digest and subprocess own its state."""

    def __init__(self, directory: Path, spectral_root: Path, *, users_directory: Path | None = None):
        self.directory = Path(directory)
        self.users_directory = Path(users_directory) if users_directory is not None else self.directory / "users"
        self.spectral_root = Path(spectral_root)
        self.lock = threading.Lock()
        self.client_bridge = None
        self.active_digest: str | None = None
        self.process: subprocess.Popen | None = None
        self._sweep_stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_animations, daemon=True, name="marble-photo-expiry")
        self._sweeper.start()

    def job_directory(self, digest: str) -> Path:
        """Resolve an admitted job inside its owner's private content area.

        Jobs created before account-scoped storage retain their original path.
        The owner index is written at admission and is never inferred from a
        client-supplied digest or path.
        """
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        legacy = self.directory / digest
        if legacy.exists():
            return legacy
        from . import job_library
        for owner in job_library.owner_keys(self.directory, digest):
            kind, _, identity = owner.partition(":")
            if kind == "account" and identity.isascii() and identity.isdecimal():
                return self.users_directory / "account" / identity / "assets" / "marble_photos" / digest
            if kind == "browser" and _DIGEST.fullmatch(identity):
                return self.users_directory / "browser" / identity / "assets" / "marble_photos" / digest
        return legacy

    def _sweep_animations(self) -> None:
        while not self._sweep_stop.is_set():
            with self.lock:
                self._expire_animations_locked()
                self._cleanup_client_bridge_locked()
            self._sweep_stop.wait(60)

    def _expire_animations_locked(self) -> None:
        if not self.directory.exists():
            return
        legacy = (p for p in self.directory.iterdir() if p.is_dir() and _DIGEST.fullmatch(p.name))
        scoped = self.users_directory.glob("*/*/assets/marble_photos/*")
        for directory in (*legacy, *scoped):
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

    def renew_participant_live(self, digest, owner):
        """Renew only an owned active cooperative Live lease, never its scene."""
        from . import job_library
        with self.lock:
            if (digest != self.active_digest or self.process is None
                    or self.process.poll() is not None
                    or not job_library.allowed(self.directory, digest, (owner,))):
                return
            directory = self.job_directory(digest)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            if config.get("client_intersections") is not True or "live" not in config:
                return
            path = directory / "live-control.json"
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("expires_at", 0) > time.time() + 20:
                return
            from .marble_live import LEASE_SECONDS
            current["expires_at"] = time.time() + LEASE_SECONDS
            temporary = directory / "live-control.json.tmp"
            temporary.write_text(json.dumps(current), encoding="utf-8")
            for attempt in range(100):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 99:
                        raise PhotoError("live lease file is temporarily busy; retry", 503) from None
                    time.sleep(.01)

    def _cleanup_client_bridge_locked(self):
        if self.client_bridge is not None and self.active_digest and self.process is not None and self.process.poll() is not None:
            self.client_bridge.close(self.active_digest)

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
        if self.client_bridge is not None and self.active_digest:
            self.client_bridge.close(self.active_digest)

    def _validate_geometry(self, payload: Path, config_path: Path) -> dict:
        """Run the exact worker CPU admission before touching the GPU owner."""
        command = [sys.executable, "-m", "camera_designer.studio_photo_geometry",
                   str(payload), str(config_path)]
        try:
            result = subprocess.run(command, cwd=self.spectral_root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=8,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired as error:
            raise PhotoError("Geometry validation exceeded 8 seconds; the current exposure was left running", 503) from error
        except OSError as error:
            raise PhotoError(f"Geometry validation could not start: {error}", 503) from error
        receipt = None
        for line in reversed(result.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and "ok" in candidate:
                receipt = candidate
                break
        if not receipt or not receipt.get("ok") or result.returncode:
            raise PhotoError(receipt.get("error", "Geometry validation failed") if receipt
                             else "Geometry validation failed before starting an exposure", 400)
        return receipt["geometry_stats"]

    def start(self, raw: bytes, saved_objects, *, owners=(), client_bridge=None, bridge_url=None, bridge_ca=None) -> dict:
        vertices, ids, pbr, config, seconds = decode_upload(raw, saved_objects, self.spectral_root)
        if config.get("client_intersections") and (not owners or client_bridge is None or not bridge_url):
            raise PhotoError("Authenticated client work bridge is unavailable",503)
        self.directory.mkdir(parents=True, exist_ok=True)
        tile_plan = None
        if "tiled" in config:
            contract = _tile_contract(str(self.spectral_root))
            tile_plan = contract.tile_plan(config)
            try:
                contract.require_tile_disk(self.directory, tile_plan)
            except ValueError as error:
                raise PhotoError(str(error), 507) from error
        with tempfile.TemporaryDirectory(prefix="marble-photo-", dir=self.directory) as temporary:
            staging = Path(temporary)
            payload = staging / "payload.npz"
            np.savez_compressed(payload, vertices=vertices, material_ids=ids, pbr=pbr)
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(payload.read_bytes() + canonical).hexdigest()
            (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
            geometry_stats = self._validate_geometry(payload, staging / "config.json")
            from meowdoku import job_library
            collation = job_library.collation_manifest(digest, config,
                tile_contract=_tile_contract(str(self.spectral_root)) if tile_plan else None)
            job_library.write_collation(staging, collation)
            with self.lock:
                from meowdoku import job_library
                # Client-authored live.session_id is a scene continuity marker,
                # never permission to replace another identity's active worker.
                if (self.active_digest and self.process is not None and self.process.poll() is None
                        and not job_library.allowed(self.directory, self.active_digest, owners)):
                    raise PhotoError("another identity owns the active GPU job", 403)
                # A live toggle must never replace a running photo (nor may a
                # new photo silently end a live session). Different tabs also
                # cannot steal each other's live GPU owner. Revisions from one
                # live session may replace only that session's own worker.
                if self.process is not None and self.process.poll() is None and self.active_digest:
                    active_config = json.loads((self.job_directory(self.active_digest) / "config.json").read_text(encoding="utf-8"))
                    active_live, requested_live = active_config.get("live"), config.get("live")
                    if bool(active_live) != bool(requested_live):
                        raise PhotoError("GPU is busy: stop the live view or finish the photograph before switching modes", 409)
                    if active_live and active_live["session_id"] != requested_live["session_id"]:
                        raise PhotoError("Another tab owns the live ray view", 409)
                    if not active_live and self.active_digest != digest:
                        raise PhotoError("GPU is busy: finish or explicitly cancel the current photograph before starting another", 409)
                from meowdoku import job_library
                try:
                    job_library.claim(self.directory, digest, owners)
                except PermissionError as error:
                    raise PhotoError(str(error), 403) from error
                if self.active_digest == digest:
                    status = self._status_locked(digest)
                    if (status["state"] in ("preparing", "exposing", "encoding", "complete", "live") or
                            (self.process is not None and self.process.poll() is None and status["state"] not in ("error","canceled"))):
                        if config.get("client_intersections") and self.client_bridge is not None and self.process is not None and self.process.poll() is None:
                            status={**status,"client_join":self.client_bridge.join(digest,owners[0])}
                        return status
                self._stop_locked()
                self.active_digest = None
                directory = self.job_directory(digest)
                directory.mkdir(parents=True, exist_ok=True)
                os.replace(payload, directory / "payload.npz")
                os.replace(staging / "config.json", directory / "config.json")
                if collation is not None:
                    os.replace(staging / "collation.json", directory / "collation.json")
                if "live" in config:
                    (directory / "live-control.json").write_text(json.dumps({
                        "revision": 0, "camera": config["camera"], "live": config["live"],
                        "expires_at": time.time() + config["live"]["lease_seconds"],
                    }), encoding="utf-8")
                status = {"digest": digest, "state": "preparing", "image": None,
                          "object_digest": config["object_digest"],
                          "exposure_seconds": seconds, "geometry_stats": geometry_stats,
                          **({"tiled": tile_plan} if tile_plan else {})}
                (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
                command = [sys.executable, "-m", "camera_designer.marble_preview_worker",
                           "--payload", str(directory / "payload.npz"),
                           "--config", str(directory / "config.json"),
                           "--out", str(directory), "--seconds", str(seconds),
                           "--native-module-dir", str(self.spectral_root),
                           "--avi-writer-root", str(Path(__file__).resolve().parents[2])]
                worker_env=None
                client_join=None
                if config.get("client_intersections"):
                    from meowdoku.client_ray_relay import RelayError
                    try:
                        secret=client_bridge.register(digest,owners[0])
                        self.client_bridge=client_bridge
                        worker_env=dict(os.environ,MEOWDOKU_RAY_BRIDGE_URL=bridge_url,
                                        MEOWDOKU_RAY_BRIDGE_JOB=digest,MEOWDOKU_RAY_BRIDGE_SECRET=secret,
                                        MEOWDOKU_RAY_BRIDGE_CA=str(bridge_ca or ""))
                        client_join=client_bridge.join(digest,owners[0])
                    except RelayError as error:
                        raise PhotoError(str(error),error.status) from error
                log = open(directory / "worker.log", "ab")
                try:
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    self.process = subprocess.Popen(command, cwd=self.spectral_root,
                                                    stdout=log, stderr=subprocess.STDOUT,
                                                    creationflags=flags,env=worker_env)
                except OSError as error:
                    if client_join is not None:client_bridge.close(digest)
                    raise PhotoError(f"Spectral worker could not start: {error}", 503) from error
                finally:
                    log.close()
                self.active_digest = digest
                return {**status,"client_join":client_join} if client_join is not None else status

    def _status_locked(self, digest: str, *, retained_artifact: bool = False) -> dict:
        self._cleanup_client_bridge_locked()
        directory = self.job_directory(digest)
        if not directory.is_dir():
            raise PhotoError("photo job is unavailable", 404)
        path = directory / "status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"digest": digest, "state": "preparing", "image": None}
        if status.get("state") == "error" and not status.get("digest"):
            return {"digest": digest, "state": "error", "image": None,
                    "error": str(status.get("error", "Spectral worker failed"))}
        if status.get("digest") != digest:
            raise PhotoError("photo status digest mismatch", 503)
        if digest != self.active_digest and status.get("state") != "complete" and not retained_artifact:
            raise PhotoError("photo digest is no longer active", 404)
        process = self.process if digest == self.active_digest else None
        if process is not None and process.poll() is not None and status.get("state") not in ("error", "complete", "canceled"):
            return {**status, "digest": digest, "state": "error", "image": status.get("image"),
                    "frames": status.get("frames", []),
                    "error": f"Spectral worker exited with code {process.returncode} before finishing the photo"}
        if status.get("state") == "encoding":
            temporary = self.job_directory(digest) / "orbit.avi.tmp"
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

    def open_artifact(self, digest: str, name: str, *, hdr: bool = False, lane: int = 0):
        """Open a verified artifact; callers stream it after existing ownership authorization."""
        marker = not hdr and bool(_MARKER.fullmatch(name))
        pattern = _HDR if hdr else _MARKER if marker else _IMAGE
        if not _DIGEST.fullmatch(digest) or not pattern.fullmatch(name) or type(lane) is not int or not 0 <= lane < 32:
            raise PhotoError("invalid photo artifact path", 400)
        with self.lock:
            status = self._status_locked(digest, retained_artifact=True)
            if lane:
                advertised = next((item for item in status.get("sensor_lanes", [])
                    if isinstance(item, dict) and item.get("index") == lane), None)
                expected = f"lane-{lane:02d}/{name}"
                field = "hdr" if hdr else "marker" if marker else "image"
                if advertised is None or advertised.get(field) != expected:
                    raise PhotoError("sensor lane artifact is unavailable", 404)
                directory = self.job_directory(digest) / f"lane-{lane:02d}"
                path = directory / name
                if directory.is_symlink() or path.is_symlink() or path.resolve().parent != directory.resolve():
                    raise PhotoError("sensor lane artifact is unavailable", 404)
                try:
                    return path.open("rb")
                except OSError:
                    raise PhotoError("sensor lane artifact is unavailable", 404) from None
            if digest != self.active_digest and status.get("state") != "complete":
                # A canceled/interrupted camera sequence keeps finished frames.
                # Only recorded child results survive this read-only exception;
                # a last progressive preview never becomes a completed frame.
                field = "hdr" if hdr else "marker" if marker else "image"
                if name not in {frame.get(field) for frame in status.get("frames", [])} and not (marker and status.get("marker") == name):
                    raise PhotoError("completed child artifact is unavailable", 404)
            if hdr:
                available = {frame.get("hdr") for frame in status.get("frames", [])}
                if status.get("hdr") != name and name not in available:
                    raise PhotoError("HDR exposure is unavailable", 404)
            if marker and status.get("marker") != name:
                raise PhotoError("spectral marker is unavailable", 404)
            try:
                return (self.job_directory(digest) / name).open("rb")
            except OSError:
                raise PhotoError("photo artifact is unavailable", 404) from None

    def _open_scene_response(self, digest, name):
        with self.lock:
            status = self._status_locked(digest)
            if status.get('state') not in ('exposing', 'live', 'complete'):
                raise PhotoError('scene response is unavailable', 404)
            directory = self.job_directory(digest)
            try:
                config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
                revision = (json.loads((directory / 'live-control.json').read_text(encoding='utf-8'))['revision']
                            if 'live' in config else 0)
                current = status.get('scene_responses')
                if (config.get('receive_light_field') is not True
                        or config.get('light_field_information') != 'evaluated-responses-v1'
                        or not isinstance(current, dict) or current.get('digest') != digest
                        or type(revision) is not int or revision < 0 or current.get('live_revision') != revision):
                    raise PhotoError('scene response is unavailable', 404)
                root = str(self.spectral_root)
                if root not in sys.path: sys.path.insert(0, root)
                from camera_designer.studio_scene_responses import validate_scene_response_snapshot
                from camera_designer.studio_scene_response_publish import response_lock, _atomic
                path = directory / name
                sidecar = path.with_suffix('.scene.json')
                if (directory / 'light-response.lock').is_symlink():
                    raise PhotoError('scene response lock is unavailable', 503)
                with response_lock(directory):
                    if (path.is_symlink() or sidecar.is_symlink() or path.resolve().parent != directory.resolve()
                            or sidecar.stat().st_size > 8 * 1024 * 1024):
                        raise PhotoError('scene response is unavailable', 404)
                    receipt = json.loads(sidecar.read_text(encoding='utf-8'))
                    metadata = receipt.get('metadata')
                    length = receipt.get('byte_length')
                    if (receipt.get('schema') != 'studio-scene-responses-v1' or receipt.get('name') != name
                            or receipt.get('digest') != digest or type(receipt.get('live_revision')) is not int
                            or receipt['live_revision'] != revision or type(receipt.get('sequence')) is not int
                            or receipt['sequence'] < 0 or type(length) is not int or not 0 <= length <= 64 * 1024 * 1024
                            or receipt.get('application_ready') is not False
                            or not isinstance(metadata, dict) or metadata.get('application_ready') is not False
                            or metadata.get('scene_revision') != current.get('metadata', {}).get('scene_revision')
                            or not isinstance(metadata.get('scene_revision'), str) or not metadata['scene_revision']
                            or receipt.get('sha256') != metadata.get('payload_sha256')
                            or length != metadata.get('payload_bytes')):
                        raise PhotoError('invalid scene response receipt', 503)
                    stream = path.open('rb')
                    try:
                        if os.fstat(stream.fileno()).st_size != length:
                            raise ValueError('scene response length mismatch')
                        # Validation is serialized under the job lock and bounded to64MiB.
                        payload = stream.read(length)
                        validate_scene_response_snapshot(metadata, payload)
                        del payload
                        stream.seek(0)
                        leased = _LeasedResponseStream(stream, directory, name, response_lock, _atomic)
                        leased.touch_locked()
                        return leased
                    except Exception:
                        stream.close()
                        raise
            except PhotoError: raise
            except FileNotFoundError:
                raise PhotoError('scene response is unavailable', 404) from None
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, ImportError, TimeoutError) as error:
                raise PhotoError('invalid scene response artifact: ' + str(error), 503) from None

    def open_field(self, digest: str, name: str):
        """Stream only the opted-in job's current, revision-matched field."""
        if _DIGEST.fullmatch(digest) and _SCENE_RESPONSE.fullmatch(name):
            return self._open_scene_response(digest, name)
        if not _DIGEST.fullmatch(digest) or not _FIELD.fullmatch(name):
            raise PhotoError("invalid light field artifact path", 400)
        with self.lock:
            status = self._status_locked(digest)
            if status.get("state") not in ("exposing", "live", "complete"):
                raise PhotoError("light field is unavailable", 404)
            directory = self.job_directory(digest)
            try:
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                revision = (json.loads((directory / "live-control.json").read_text(encoding="utf-8"))["revision"]
                            if "live" in config else 0)
            except (OSError, ValueError, KeyError, TypeError):
                raise PhotoError("light field configuration is unavailable", 404) from None
            field = status.get("light_field")
            if not isinstance(config, dict) or config.get("receive_light_field") is not True or not isinstance(field, dict):
                raise PhotoError("light field is unavailable", 404)
            if "surface_transport" in config:
                if config["surface_transport"] != _SURFACE_TRANSPORT or field.get("surface_transport") != _SURFACE_TRANSPORT:
                    raise PhotoError("light field surface_transport receipt mismatch", 503)
            elif "surface_transport" in field:
                raise PhotoError("unexpected light field surface_transport receipt", 503)
            if (field.get("name") != name or field.get("digest") != digest
                    or field.get("live_revision") != revision):
                raise PhotoError("light field is stale or unavailable", 404)
            if config.get("light_field_schema") in ("studio.forward-field.v4", "studio.forward-field.v5", "studio.forward-field.v6") and field.get("schema") != config["light_field_schema"]:
                raise PhotoError("light field estimator schema mismatch", 503)
            dimensions = field.get("dimensions")
            lo, hi = field.get("bounds_min_m"), field.get("bounds_max_m")
            vector_bundle = field.get("schema") in ("studio.forward-field.v2", "studio.forward-field.v3", "studio.forward-field.v4", "studio.forward-field.v5", "studio.forward-field.v6")
            if (field.get("schema") not in ("studio.forward-field.v1", "studio.forward-field.v2", "studio.forward-field.v3", "studio.forward-field.v4", "studio.forward-field.v5", "studio.forward-field.v6")
                    or field.get("format") != "rgb-f32le"
                    or field.get("layout") != "zyx-interleaved-rgb"
                    or field.get("color_space") != "linear-srgb"
                    or type(revision) is not int or type(field.get("live_revision")) is not int or revision < 0
                    or type(field.get("sequence")) is not int or field["sequence"] < 0
                    or not isinstance(dimensions, list) or len(dimensions) != 3
                    or any(type(n) is not int or not 1 <= n <= 256 for n in dimensions)
                    or type(field.get("byte_length")) is not int
                    or not 0 < field["byte_length"] <= 256 * 1024 * 1024
                    or (not vector_bundle and field["byte_length"] != math.prod(dimensions) * 12)
                    or not isinstance(lo, list) or not isinstance(hi, list) or len(lo) != 3 or len(hi) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in lo + hi)
                    or any(a >= b for a, b in zip(lo, hi))):
                raise PhotoError("invalid light field metadata", 503)
            if vector_bundle:
                _validate_vector_bundle_metadata(field, config)
            path = directory / name
            try:
                # Do not let an advertised filename redirect to another job or arbitrary file.
                if path.is_symlink() or path.resolve().parent != directory.resolve():
                    raise PhotoError("light field is unavailable", 404)
                stream = path.open("rb")
                if os.fstat(stream.fileno()).st_size != field["byte_length"]:
                    stream.close()
                    raise PhotoError("light field length mismatch", 503)
                if vector_bundle:
                    try:
                        _validate_vector_bundle_payload(stream, field)
                    except Exception:
                        stream.close()
                        raise
                return stream
            except OSError:
                raise PhotoError("light field is unavailable", 404) from None

    def image(self, digest: str, name: str) -> bytes:
        with self.open_artifact(digest, name) as stream:
            return stream.read()

    def hdr(self, digest: str, name: str) -> bytes:
        with self.open_artifact(digest, name, hdr=True) as stream:
            return stream.read()

    def animation(self, digest: str) -> bytes:
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        with self.lock:
            self._expire_animations_locked()
            status = self._status_locked(digest)
            if status.get("animation") != "orbit.avi" or time.time() >= float(status.get("animation_expires_at", 0)):
                raise PhotoError("orbit animation has expired or is unavailable", 404)
            try:
                return (self.job_directory(digest) / "orbit.avi").read_bytes()
            except OSError:
                raise PhotoError("orbit animation is unavailable", 404) from None

    def retouch(self, digest: str, params: dict) -> dict:
        # While the job's own worker is still running, this just hands off
        # the small parameter set for its snapshot loop to pick up against
        # the linear radiance it already holds live in GPU memory. Once
        # that process has exited (job complete), there is no live texture
        # left to read -- re-tonemap the one persisted final HDR file
        # directly here instead, synchronously, so Apply still works on an
        # already-finished photo, not only mid-exposure.
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        # Match the controls and direct preview contract before writing the
        # live worker handoff. Merely finite values (exposure=2000) overflow
        # exponentiation in the snapshot loop and must never reach it.
        allowed = {"exposure": (-6, 6), "gamma": (.2, 3),
                   "red": (0, 3), "green": (0, 3), "blue": (0, 3),
                   "whitebalance": (-3, 3), "tint": (-3, 3), "intensity": (0, 3)}
        clean = _display_params(params)
        spectral_look = params.get("spectral_look", "compressed")
        if spectral_look not in ("compressed", "nir_red", "uv_violet", "visible_reference"):
            raise PhotoError("retouch.spectral_look must be a supported eight-band view", 400)
        clean["spectral_look"] = spectral_look
        for key in allowed:
            if key in params:
                try:
                    clean[key] = float(params[key])
                except (TypeError, ValueError, OverflowError):
                    raise PhotoError(f"retouch.{key} must be a number", 400) from None
                if not math.isfinite(clean[key]):
                    raise PhotoError(f"retouch.{key} must be finite", 400)
                low, high = allowed[key]
                if not low <= clean[key] <= high:
                    raise PhotoError(f"retouch.{key} is outside [{low},{high}]", 400)
        result = {"ok": True}
        with self.lock:
            status = self._status_locked(digest)
            if status.get("tiled"):
                raise PhotoError("Tiled HDR retouch is not available in the interactive editor; download the HDR or set gamma before exposing", 409)
            directory = self.job_directory(digest)
            temp = directory / "retouch.json.tmp"
            temp.write_text(json.dumps(clean), encoding="utf-8")
            os.replace(temp, directory / "retouch.json")
            running = (status.get("state") not in ("complete", "error", "canceled")
                       and self.active_digest == digest and self.process is not None
                       and self.process.poll() is None)
            if not running and status.get("hdr"):
                hdr_path = directory / status["hdr"]
                try:
                    with np.load(hdr_path) as data:
                        colorimetric = "linear_srgb_exposure" in data and "color_profile_json" in data
                        if not colorimetric and "linear_rgb_exposure" not in data:
                            raise PhotoError("monochrome HDR retouch requires the monochrome calibration viewer", 409)
                        field = "linear_srgb_exposure" if colorimetric else "linear_rgb_exposure"
                        linear = np.asarray(data[field], np.float64)
                        if "linear_spectral_exposure" in data:
                            root = str(self.spectral_root)
                            if root not in sys.path: sys.path.insert(0, root)
                            from camera_designer.studio_spectral_display import project_spectral_exposure
                            linear, _ = project_spectral_exposure(
                                np.asarray(data["linear_spectral_exposure"], np.float64), clean["spectral_look"])
                except OSError as error:
                    raise PhotoError("saved HDR exposure is unavailable", 404) from error
                display = _retonemap_linear(linear, clean, colorimetric=colorimetric)
                previous = status.get("retouched_image")
                filename = _retouch_filename()
                temp_png = directory / (filename + ".tmp")
                png_info = None
                if colorimetric:
                    png_info = PngInfo()
                    png_info.add(b"sRGB", b"\x00")
                Image.fromarray(display, mode="RGB").save(temp_png, format="PNG", pnginfo=png_info)
                os.replace(temp_png, directory / filename)
                status["retouched_image"] = filename
                status["retouch_params"] = clean
                if colorimetric:
                    status["retouch_display_transform"] = _display_metadata(linear, clean)
                    if "spectral_look" in clean and status.get("transport", "").startswith("fixed:8"):
                        status["retouch_display_transform"]["spectral_look"] = clean["spectral_look"]
                else:
                    warning = "Legacy HDR has no recorded sensor color profile; new normalization, tone, saturation, vibrance, black-point and firefly controls are not applied."
                    status["retouch_display_transform"] = {"legacy": True, "warning": warning}
                    result["warning"] = warning
                temp_status = directory / "status.retouch.tmp"
                temp_status.write_text(json.dumps(status), encoding="utf-8")
                os.replace(temp_status, directory / "status.json")
                if previous and previous != filename:
                    (directory / previous).unlink(missing_ok=True)
        return result

    def cancel(self, digest: str, *, live_only: bool = False) -> dict:
        if not _DIGEST.fullmatch(digest):
            raise PhotoError("invalid photo digest", 400)
        with self.lock:
            if live_only:
                try:
                    config = json.loads((self.job_directory(digest) / "config.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise PhotoError("live view is unavailable", 404) from None
                if "live" not in config:
                    raise PhotoError("this job is a photograph, not a live ray view", 409)
                if self.active_digest != digest:
                    return {"digest": digest, "state": "canceled", "image": None}
            if self.active_digest != digest:
                raise PhotoError("photo digest is no longer active", 404)
            self._stop_locked()
            self.active_digest = None
            directory = self.job_directory(digest)
            # These are internal scratch stores, not finished PNG/HDR artifacts.
            # Reclaim the disk-backed full sensor only after its owner has exited.
            for name in ("tile-sensor.npy", "tile-output.npy", "tile-srgb.npy", "tile-luma.bin"):
                (directory / name).unlink(missing_ok=True)
            (directory / "orbit.avi").unlink(missing_ok=True)
            path = self.job_directory(digest) / "status.json"
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
