"""Pure native camera-housing export for support geometry. No service endpoint.

Uses the same preset, rigid frame and complete geometry builder as photo
preflight. It neither launches an exposure nor changes a camera or material.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
from .marble_lens_contract import SPECTRAL_ROOT


def camera_mount_source(camera: dict, spectral_root: Path = SPECTRAL_ROOT) -> dict:
    if not isinstance(camera, dict):
        raise ValueError("Camera mount export requires an explicit camera pose")
    checked = {}
    for key in ("eye_m", "target_m"):
        value = np.asarray(camera.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Camera mount export needs finite {key}")
        checked[key] = value.tolist()
    if checked["eye_m"] == checked["target_m"]:
        raise ValueError("Camera mount eye and target must differ")
    for key, default in (("aperture_radius_m", .005), ("aperture_rotation_radians", 0)):
        checked[key] = float(camera.get(key, default))
        if not np.isfinite(checked[key]):
            raise ValueError(f"Camera mount export needs finite {key}")
    blades = camera.get("aperture_blades", 0)
    if type(blades) is not int or blades < 0 or checked["aperture_radius_m"] < 0:
        raise ValueError("Invalid native pupil settings")
    checked["aperture_blades"] = blades
    root = str(Path(spectral_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from camera_designer.camera_preset import thin_lens_housing_preset
    from camera_designer.camera_frame import thin_lens_housing_frame
    from camera_designer.scene_builder import build_gpu_scene
    from camera_designer.studio_photo_geometry import CAMERA_GEOMETRY_VERSION
    preset = thin_lens_housing_preset(checked)
    frame = thin_lens_housing_frame(preset, checked["eye_m"], checked["target_m"])
    scene = build_gpu_scene(preset, [], scene_object=None,
                            wavelengths_um=[.450, .550, .650], camera_frame=frame)
    vertices = np.asarray(scene[3], dtype=np.float32).reshape(-1, 3)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Native camera housing has no finite geometry")
    return {"schema": "native-camera-housing-mount.v1", "camera": checked,
            "geometry_version": CAMERA_GEOMETRY_VERSION,
            "geometry_sha256": hashlib.sha256(vertices.astype('<f4').tobytes()).hexdigest(),
            "triangle_count": len(vertices) // 3, "housingVerticesM": vertices.tolist(),
            "source": "thin_lens_housing_preset + thin_lens_housing_frame + build_gpu_scene"}
