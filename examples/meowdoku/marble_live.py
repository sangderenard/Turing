"""Validation and control-file handoff for the opt-in rolling ray view."""
import json
import math
import os
import time

from .marble_photo import PhotoError, _number, _finite_vectors, _DIGEST, _camera_movements
from .marble_lens_contract import FOV_MIN_DEGREES, PUPIL_RADIUS_MAX_M, FOCUS_MIN_M, FOCUS_MAX_M, apply_lens_selection

from .marble_camera_back import apply_camera_back

LEASE_SECONDS = 30
MAX_HISTORY_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_RAY_BATCH_SIZE = 8192
MIN_RAY_BATCH_SIZE = 128
MAX_RAY_BATCH_SIZE = 65536
# Requested server contribution boundary, expressed in the UI's longest-edge
# pixels. This is NOT an enabled resolution or a client-work capability flag.
CLIENT_REQUIRED_ABOVE_EDGE = 612

def live_settings(value, camera, *, client_intersections=False):
    if not isinstance(value, dict):
        raise PhotoError("live settings must be an object")
    samples = value.get("history_samples", 16)
    if type(samples) is not int or samples < 1:
        raise PhotoError("live history must be a positive integer")
    ray_batch_size = value.get("ray_batch_size", DEFAULT_RAY_BATCH_SIZE)
    if type(ray_batch_size) is not int or not MIN_RAY_BATCH_SIZE <= ray_batch_size <= MAX_RAY_BATCH_SIZE:
        raise PhotoError("live ray_batch_size must be an integer between 128 and 65536")
    aspect = _number(value.get("aspect", 1), .1, 10, "live.aspect")
    edge = camera["resolution"]
    if type(edge) is int and edge > CLIENT_REQUIRED_ABOVE_EDGE and not client_intersections:
        raise PhotoError("Live resolution above 612 longest-edge pixels requires enabled client intersection sharing. Server-only Live supports 128, 256, and 512. No excess work was assigned to the server.")
    if edge not in ((128,256,512,768,1024,2048,4096) if client_intersections else (128,256,512)):
        raise PhotoError("unsupported live resolution for the selected participation mode")
    # Match the browser's Math.round at half-pixel aspect boundaries.
    width, height = ((edge, max(16, math.floor(edge / aspect + .5))) if aspect >= 1
                     else (max(16, math.floor(edge * aspect + .5)), edge))
    layers = 3  # live is always the fixed:3 sensor contract
    checkpoint = value.get("checkpoint", False)
    if type(checkpoint) is not bool:
        raise PhotoError("live checkpoint must be a boolean")
    memory = width * height * layers * (4 * samples + (32 if checkpoint else 8))
    if memory > MAX_HISTORY_BYTES:
        raise PhotoError("ray history exceeds 2 GiB; lower history length or resolution")
    display = value.get("display", {})
    if not isinstance(display, dict):
        raise PhotoError("live display settings must be an object")
    normalization = display.get("normalization", "auto99")
    if normalization not in ("auto99", "manual", "percentile", "specular"):
        raise PhotoError("live normalization must be auto99, manual, percentile, or specular")
    display_look = display.get("display_look", "reference")
    if display_look not in ("reference", "color_priority"):
        raise PhotoError("live.display_look must be reference or color_priority")
    if normalization == "specular":
        display_look = "reference"
    tone = display.get("tone_curve", "exponential")
    if tone not in ("exponential", "linear"):
        raise PhotoError("live tone_curve must be exponential or linear")
    percentile = display.get("normalization_percentile", 95 if normalization == "specular" else 99)
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise PhotoError("live.normalization_percentile must be a number")
    try:
        percentile = float(percentile)
    except (ValueError, OverflowError):
        raise PhotoError("live.normalization_percentile must be finite") from None
    if not math.isfinite(percentile) or not 1 <= percentile <= 100:
        raise PhotoError("live.normalization_percentile must be between 1 and 100")
    # Live snapshots use the same display transform as completed photographs.
    # Previously these fields were silently dropped here, so the existing color
    # and isolated-highlight controls could never affect the live image.
    adjustments = {}
    for name, low, high, default in (
        ("exposure", -6, 6, 0), ("whitebalance", -3, 3, 0),
        ("saturation", 0, 2, 1), ("vibrance", -1, 1, 0),
        ("black_point", 0, .5, 0), ("firefly_threshold", 2, 64, 8),
        ("firefly_strength", 0, 1, 1),
    ):
        value = display.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PhotoError("live." + name + " must be a number")
        try:
            adjustments[name] = _number(value, low, high, "live." + name)
        except OverflowError:
            raise PhotoError("live." + name + " must be finite") from None
    enabled = display.get("firefly_filter", False)
    if type(enabled) is not bool:
        raise PhotoError("live.firefly_filter must be a boolean")
    adjustments["firefly_filter"] = enabled
    display = dict(**adjustments, display_look=display_look, tint=_number(display.get("tint", 0), -3, 3, "live.tint"),
                   normalization=normalization,
                   normalization_percentile=percentile,
                   tone_curve="linear" if normalization == "specular" else tone,
                   normalization_white=_number(display.get("normalization_white", 1), 1e-6, 1e12, "live.normalization_white"))
    return dict(history_samples=samples, aspect=aspect, width=width, height=height,
                history_bytes=memory, lease_seconds=LEASE_SECONDS, display=display,
                checkpoint=checkpoint, ray_batch_size=ray_batch_size)


def update_live(jobs, digest, value):
    # All camera/control writes serialize with start/cancel. Only the owning
    # live digest can be changed. This path never stops a still photograph.
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise PhotoError("invalid live digest", 400)
    with jobs.lock:
        status = jobs._status_locked(digest)
        if digest != jobs.active_digest or jobs.process is None or jobs.process.poll() is not None:
            raise PhotoError("live ray view is no longer active", 409)
        directory = jobs.job_directory(digest)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if "live" not in config:
            raise PhotoError("this job is a photograph, not a live ray view", 409)
        path = directory / "live-control.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        revision = value.get("revision", current["revision"])
        if type(revision) is not int or revision < 0 or revision > 2147483647:
            raise PhotoError("invalid live camera revision")
        if revision < current["revision"]:
            raise PhotoError("stale live camera revision", 409)
        next_control = dict(current)
        if revision > current["revision"]:
            camera = dict(current["camera"])
            updates = value.get("camera", {})
            if not isinstance(updates, dict):
                raise PhotoError("live camera must be an object")
            for key in ("eye_m", "target_m"):
                if key in updates:
                    camera[key] = _finite_vectors(updates[key], 3, "camera." + key)
            for key, bounds in {"fov_degrees": (.001 if updates.get("camera_back", camera.get("camera_back")) is not None else FOV_MIN_DEGREES,100), "focus_distance_m": (FOCUS_MIN_M,FOCUS_MAX_M),
                                "aperture_radius_m": (1e-5,PUPIL_RADIUS_MAX_M)}.items():
                if key in updates:
                    camera[key] = _number(updates[key], *bounds, "camera." + key)
            if any(key in updates for key in ("tilt_shift", "lens_tilt")):
                camera.update(_camera_movements({**camera, **updates}))
            try:
                selection = updates.get("lens_selection", camera.get("lens_selection"))
                camera = (apply_lens_selection(camera, selection, jobs.spectral_root) if selection is not None
                          else apply_lens_selection(camera, None))
            except (ValueError, OSError) as error:
                raise PhotoError(str(error)) from None
            if sum((a-b)**2 for a,b in zip(camera["eye_m"],camera["target_m"])) < 1e-12:
                raise PhotoError("live camera eye and target must differ")
            if not isinstance(value.get("live", {}), dict):
                raise PhotoError("live settings must be an object")
            settings = live_settings({**current["live"], **value.get("live", {})}, camera, client_intersections=config.get("client_intersections") is True)
            sampling = config.get("emitter_sampling", {})
            if sampling.get("refresh_each_sweep") and settings["ray_batch_size"] < sampling["ray_budget"]:
                raise PhotoError("live ray_batch_size cannot be below emitter_sampling ray_budget")
            # Resizing the sensor is a new immutable scene job, not a control edit.
            if (settings["width"], settings["height"]) != (current["live"]["width"], current["live"]["height"]):
                raise PhotoError("live viewport size changed; restart the live view", 409)
            try:
                camera = apply_camera_back(camera, updates.get("camera_back", camera.get("camera_back")),
                                           settings["aspect"], updates.get("back_reference_fov_degrees"))
            except ValueError as error:
                raise PhotoError(str(error)) from None
            next_control.update(camera=camera, live=settings, revision=revision)
        next_control["expires_at"] = time.time() + LEASE_SECONDS
        temporary = directory / "live-control.json.tmp"
        temporary.write_text(json.dumps(next_control), encoding="utf-8")
        # The GL worker reads this file every 50 ms. Windows can briefly deny
        # delete sharing; preserve the complete temp file and retry atomically.
        for attempt in range(100):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 99:
                    raise PhotoError("live control file is temporarily busy; retry", 503) from None
                time.sleep(.01)
        return {"ok": True, "revision": next_control["revision"]}
