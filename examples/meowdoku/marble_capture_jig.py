"""Geodesic view jobs for the existing Studio photograph renderer/catalog.

This is an immutable capture plan, not a new renderer, scheduler, or light
field. Each job uses the existing one-frame photo configuration. The caller
must prepare the suspended subject/rig and execute these jobs through the
existing GPU owner. Subject/stage geometry, light positions and materials do
not rotate. The native physical camera housing follows each camera position.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from src.cells.softbody.geometry.geodesic import geodesic_sphere
from .marble_lens_contract import SPECTRAL_ROOT


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _spectral(root):
    root = str(Path(root))
    if root not in sys.path:
        sys.path.insert(0, root)


def _vec(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be three finite metres")
    return array


def geodesic_capture_plan(config, *, frequency=1, spectral_root=SPECTRAL_ROOT):
    """Expand an already normalized photo config into 12, 42 or 92 views.

    Capture radius, focus, aperture and FOV come from the explicit current
    camera. Only its eye position changes. This deliberately does not move a
    subject off its table or invent an invisible support/transparent matte.
    """
    if type(frequency) is not int or frequency not in (1, 2, 3):
        raise ValueError("Geodesic frequency must be 1, 2, or 3")
    if not isinstance(config, dict) or config.get("transport") != "fixed:3":
        raise ValueError("Geodesic publication requires existing RGB photo transport")
    if config.get("live") or config.get("client_intersections"):
        raise ValueError("Geodesic capture uses ordinary exclusive one-frame photo jobs")
    for name in ("object_digest", "geometry_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(config.get(name, ""))):
            raise ValueError(f"Exact saved {name} is required")
    base = deepcopy(config)
    _canonical(base)  # Reject NaN/Infinity before making any plan.
    camera = base.get("camera", {})
    center = _vec(base.get("object_center_m"), "Subject center")
    target = _vec(camera.get("target_m"), "Camera target")
    eye = _vec(camera.get("eye_m"), "Camera eye")
    if not np.array_equal(target, center):
        raise ValueError("Geodesic capture must explicitly aim at the subject center")
    distance = float(np.linalg.norm(eye - center))
    radius = float(base.get("scale_m", 0))
    fov = float(camera.get("fov_degrees", 0))
    if not 0 < radius < distance or not 5 <= fov <= 100:
        raise ValueError("Capture distance must be outside the physical subject")
    if math.asin(radius / distance) >= math.radians(fov / 2):
        raise ValueError("The nominal spherical radius must fit inside every view")
    if not 1e-5 <= float(camera.get("aperture_radius_m", 0)) <= .02:
        raise ValueError("The existing physical pupil must be explicit")
    if not math.isfinite(float(camera.get("focus_distance_m", 0))) or camera["focus_distance_m"] <= 0:
        raise ValueError("The existing physical focus must be explicit")
    if camera.get("resolution") not in (128, 256, 512, 768, 1024, 2048, 4096):
        raise ValueError("Use a supported existing photo resolution")
    _spectral(spectral_root)
    from camera_software.render_assets import LightFieldCondition, stable_asset_key
    from camera_designer.camera_frame import sensor_camera_basis

    mesh = geodesic_sphere(frequency, radius=1., topology="tri")
    directions = np.asarray(mesh["verts"], dtype=np.float64)
    # The stable rig key includes all authored scene state, with camera pose
    # removed. Fixed source energy/materials are therefore part of identity.
    rig = deepcopy(base)
    rig["camera"].pop("eye_m", None)
    rig["camera"].pop("target_m", None)
    rig["orbit_frames"] = 1
    # Direction varies per condition; distance must remain part of catalog
    # identity or photographs from different camera spheres overwrite each other.
    rig["capture_distance_m"] = distance
    rig_key = stable_asset_key("studio-fixed-capture-rig", rig)
    jobs = []
    for index, direction in enumerate(directions):
        view = deepcopy(base)
        view["orbit_frames"] = 1
        view["camera"]["eye_m"] = (center + direction * distance).tolist()
        view["camera"]["target_m"] = center.tolist()
        # These are the actual native sensor basis and vertical-view rule.
        _, right, up, forward = sensor_camera_basis(
            view["camera"]["eye_m"], view["camera"]["target_m"])
        condition = LightFieldCondition(
            azimuth_deg=math.degrees(math.atan2(direction[0], direction[2])),
            elevation_deg=math.degrees(math.asin(float(direction[1]))),
            light_rig=rig_key, material_variant=base["geometry_sha256"], frame=index)
        jobs.append({"index": index, "direction": direction.tolist(),
                     "condition": asdict(condition), "condition_key": condition.condition_key,
                     "camera_sha256": _sha(_canonical(view["camera"])),
                     "basis": {"right": right.tolist(), "up": up.tolist(), "forward": forward.tolist()},
                     "config_sha256": _sha(_canonical(view)), "config": view})
    plan = {"schema": "turing.studio-geodesic-capture.v1", "frequency": frequency,
            "object_digest": base["object_digest"], "geometry_sha256": base["geometry_sha256"],
            "radius_m": radius, "capture_distance_m": distance, "center_m": center.tolist(),
            "geodesic_source": "src.cells.softbody.geometry.geodesic.geodesic_sphere",
            "view_faces": np.asarray(mesh["faces"]).tolist(), "rig_key": rig_key, "jobs": jobs,
            "execution": "existing one-frame PhotoJobs and marble_preview_worker",
            "suspension_verified": False, "product_ready": False,
            "lighting": {"policy": "immutable explicit capture configuration",
                         "requested_product": "dedicated fixed ideal jig",
                         "dedicated_ideal_jig_verified": False,
                         "game_time_of_day_consumed": False, "game_added_lights_consumed": False},
            "observer_camera": {"required": True, "native_visibility_filter_verified": False,
                                "selection": "exact source-object identities; no material-wide X-ray",
                                "scope": "primary camera segment only; full scene after first real interaction",
                                "table_light_paths_preserved": True,
                                "table_geometry_deleted": False,
                                "shadow_receiver_product_ready": False},
            "source_mesh_bounds_verified": False, "view_occlusion_verified": False,
            "radius_semantics": "nominal saved-object scale; not a measured bound on all authored vertices",
            "remaining": ["prepare and validate suspended capture jig",
                          "construct and validate dedicated ideal lighting, not incidental live room lighting",
                          "implement observer-only camera and primary-only source-object filter without changing light paths",
                          "capture separate table shadow-receiver texture with actual transport",
                          "verify actual mesh framing and clear camera/scene paths for every view",
                          "execute each original ray-traced photo job",
                          "bind object coverage/background and projective sampling",
                          "install view-dependent presentation on existing projectile"]}
    plan["plan_sha256"] = _sha(_canonical(plan))
    return plan


def record_completed_view(plan, index, directory, catalog, *, spectral_root=SPECTRAL_ROOT):
    """Register an actually completed existing photo in the existing catalog.

    This does not promote an ordinary image to an object alpha/light field.
    Config/payload job identity and the original finite, profiled HDR contract
    are checked. Saved geometry+normal admission remains PhotoJobs' authority:
    its payload no longer contains the source normals needed to repeat that
    source hash. No job is launched and no rendered bytes are changed.
    """
    checked = deepcopy(plan)
    claimed = checked.pop("plan_sha256", None)
    if plan.get("schema") != "turing.studio-geodesic-capture.v1" or _sha(_canonical(checked)) != claimed:
        raise ValueError("Capture plan identity changed")
    if type(index) is not int or not 0 <= index < len(plan["jobs"]):
        raise ValueError("Invalid capture view index")
    job = plan["jobs"][index]
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if _sha(_canonical(config)) != job["config_sha256"]:
        raise ValueError("Completed photo does not match this geodesic view")
    payload = (directory / "payload.npz").read_bytes()
    digest = _sha(payload + _canonical(config))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or status.get("digest") != digest or status.get("total_frames") != 1:
        raise ValueError("The matching single-frame photograph is not complete")
    frames = status.get("frames", [])
    if len(frames) != 1 or frames[0].get("index") != 0:
        raise ValueError("A finished single-frame result is required")
    frame = frames[0]
    paths = {}
    for key in ("image", "hdr"):
        name = frame.get(key)
        pattern = r"preview-[0-9]{6,12}\.png" if key == "image" else r"exposure-[0-9]{6,12}\.npz"
        if not isinstance(name, str) or Path(name).name != name or not re.fullmatch(pattern, name):
            raise ValueError("Invalid completed artifact name")
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError("Completed artifact is unavailable")
        paths[key] = path
    resolution = config["camera"]["resolution"]
    _spectral(spectral_root)
    from camera_designer.film_response import normalize_film_response
    from camera_designer.marble_color_profile import build_output_profile
    response = normalize_film_response(config.get("film_response"))
    _, expected_profile = build_output_profile(response)
    with np.load(paths["hdr"], allow_pickle=False) as hdr:
        required = {"linear_srgb_exposure", "linear_rgb_exposure", "color_profile_json",
                    "film_response_version", "film_centers_nm", "film_sigmas_oct"}
        if not required.issubset(hdr.files):
            raise ValueError("Completed HDR lacks the original worker's profile provenance")
        linear = hdr["linear_srgb_exposure"]
        sensor = hdr["linear_rgb_exposure"]
        if (any(array.shape != (resolution, resolution, 3)
                or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()
                for array in (linear, sensor))):
            raise ValueError("Completed linear HDR is not a finite full view")
        try:
            profile = json.loads(str(hdr["color_profile_json"].item()))
            matches = (hdr["film_response_version"].item() == response["version"]
                       and np.array_equal(hdr["film_centers_nm"], response["centers_nm"])
                       and np.array_equal(hdr["film_sigmas_oct"], response["sigmas_oct"]))
        except (TypeError, ValueError):
            raise ValueError("Completed HDR profile metadata is invalid") from None
        if not matches or profile != expected_profile or status.get("color_profile") != profile:
            raise ValueError("Completed HDR profile differs from its camera/film response")
    from PIL import Image
    with Image.open(paths["image"]) as image:
        if image.format != "PNG" or image.mode not in ("RGB", "RGBA"):
            raise ValueError("Completed preview must be the original RGB PNG product")
        if image.size != (resolution, resolution):
            raise ValueError("Completed preview dimensions differ")
        image.verify()
    samples = status.get("sensor_sites", 0)
    if type(samples) is not int or samples < 0:
        raise ValueError("Completed sensor-site count is invalid")
    from camera_software.render_assets import RenderedAssetRecord
    from camera_software.display_scene import DisplayProductKind
    record = RenderedAssetRecord(
        target_key="studio-object:" + plan["object_digest"], condition_key=job["condition_key"],
        product_kind=DisplayProductKind.IMAGE, linear_path=str(paths["hdr"]),
        preview_path=str(paths["image"]), manifest_path=str(directory / "config.json"),
        width=resolution, height=resolution, samples=samples,
        metadata={"capture_plan_sha256": claimed, "view_index": index, "job_digest": digest,
                  "camera_sha256": job["camera_sha256"], "geometry_sha256": plan["geometry_sha256"],
                  "hdr_sha256": _sha(paths["hdr"].read_bytes()),
                  "preview_sha256": _sha(paths["image"].read_bytes()),
                  "linear_exposure": "existing profiled linear_srgb_exposure; not calibrated radiance",
                  "color_profile": profile,
                  "geometry_admission": "original PhotoJobs authenticated upload, not recomputed from the normal-free payload",
                  "object_coverage": "not yet isolated", "view_dependent_product_ready": False})
    catalog.record(record)
    return record
