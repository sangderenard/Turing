"""Source-derived controls for Studio's existing analytical camera component.

No compound renderer is installed here. Authored LensSpec profiles retain
their limitations; calibrated compound entries remain read-only references.
"""
from __future__ import annotations

import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

FOCUS_MIN_M = .001  # Numerical/application domain, not a physical lens limit.
FOCUS_MAX_M = 10.
FOCAL_MAX_M = .5
FOV_MIN_DEGREES = math.degrees(2 * math.atan(.024 / (2 * FOCAL_MAX_M)))
PUPIL_RADIUS_MAX_M = .03125
SENSOR_HEIGHT_M = .024  # Explicit reference format for focal length -> FOV.
SELECTION_SCHEMA = "studio-analytical-lens-selection-v1"
FOCUS_SEMANTICS = "axial-planar-focus-min-1mm-v1"
SPECTRAL_ROOT = Path(__file__).resolve().parents[3] / "spectral-analyzer"
CATALOG = Path(__file__).with_name("lens_contract_catalog.json")


@lru_cache(maxsize=4)
def _source_module(root: str):
    path = Path(root) / "camera_software" / "lens_contracts.py"
    spec = importlib.util.spec_from_file_location("studio_source_lens_contracts", path)
    if spec is None or spec.loader is None:
        raise ValueError("source lens-contract generator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lens_catalog(spectral_root: Path = SPECTRAL_ROOT) -> dict:
    source = _source_module(str(spectral_root))
    current = source.export_collection(Path(spectral_root) / "configs" / "lenses")
    if current["failures"]:
        raise ValueError("source lens profile validation failed: " + str(current["failures"]))
    references = []
    if CATALOG.exists():
        references = [c for c in json.loads(CATALOG.read_text(encoding="utf-8"))["contracts"]
                      if c["kind"] != "authored-thin-profile"]
    # Return the useful control/provenance subset, not every calibration ray.
    return {"schema": "studio-lens-catalog-v1", "focus_semantics": FOCUS_SEMANTICS,
            "custom_focus_minimum_m": FOCUS_MIN_M, "custom_focus_maximum_m": FOCUS_MAX_M,
            "reference_sensor_height_m": SENSOR_HEIGHT_M,
            "pupil_radius_maximum_m": PUPIL_RADIUS_MAX_M,
            "profiles": current["contracts"],
            "references": [{"key": c["key"], "kind": c["kind"], "source": c["source"],
                            "cardinal": c["cardinal"], "calibration": c["calibration"]["status"],
                            "preset_fit": c.get("preset_fit"), "selectable": False,
                            "reason": "Reference only: compound-state shader adapter and full-field calibration are not installed."}
                           for c in references]}


def _finite(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def apply_lens_selection(camera: dict, selection, spectral_root: Path = SPECTRAL_ROOT) -> dict:
    """Validate source identity and derive FOV/pupil from authoritative settings."""
    camera = dict(camera)
    for name in ("lens_selection", "lens_calibration", "lens_reference_sensor_height_m", "focus_semantics"):
        camera.pop(name, None)
    if selection is None:
        return camera
    if not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unsupported analytical lens selection")
    if set(selection) != {"schema", "key", "source_sha256", "focal_length_m", "f_number", "sensor_height_m"}:
        raise ValueError("analytical lens selection has unexpected or missing fields")
    f = _finite(selection["focal_length_m"], "focal_length_m", .001, FOCAL_MAX_M)
    n = _finite(selection["f_number"], "f_number", 1., 64.)
    if selection["sensor_height_m"] != SENSOR_HEIGHT_M:
        raise ValueError("this camera contract uses a 24 mm reference sensor height")
    focus = _finite(camera["focus_distance_m"], "focus_distance_m", FOCUS_MIN_M, FOCUS_MAX_M)
    key = selection["key"]
    if key is None:
        if selection["source_sha256"] is not None:
            raise ValueError("a custom analytical lens has no source profile hash")
        basis = "custom analytical camera, no physical prescription"
    else:
        profiles = lens_catalog(spectral_root)["profiles"]
        contract = next((c for c in profiles if c["key"] == key), None)
        if contract is None or contract["source_sha256"] != selection["source_sha256"]:
            raise ValueError("lens profile changed or is unavailable; reload the lens collection")
        source = _source_module(str(spectral_root))
        source.profile_shader_parameters(contract, focal_length_m=f, focus_distance_m=focus,
                                         f_number=n, sensor_width_m=SENSOR_HEIGHT_M,
                                         sensor_height_m=SENSOR_HEIGHT_M)
        basis = "authored analytical profile; distortion, vignetting and spectral transmission not calibrated/applied"
    fov = math.degrees(2 * math.atan(SENSOR_HEIGHT_M / (2 * f)))
    radius = f / (2 * n)
    if not FOV_MIN_DEGREES <= fov <= 100:
        raise ValueError("focal length exceeds this renderer's 500 mm / 100 degree field-of-view range")
    if not 1e-5 <= radius <= PUPIL_RADIUS_MAX_M:
        raise ValueError("aperture exceeds this renderer's 31.25 mm pupil-radius limit; increase f-number")
    camera.update(fov_degrees=fov, aperture_radius_m=radius,
                  lens_selection=dict(selection), lens_calibration=basis,
                  lens_reference_sensor_height_m=SENSOR_HEIGHT_M,
                  focus_semantics=FOCUS_SEMANTICS)
    return camera
