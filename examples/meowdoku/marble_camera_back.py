"""Physical back geometry for the existing rectilinear analytical camera.

The back is center-cropped to the complete output aspect, never stretched.
Pixel count and tile layout are deliberately not inputs to this contract.
This retains the renderer's reference-focal-length/axial-focus approximation;
it does not claim finite-conjugate sensor extension or a calibrated lens.
"""
import math

BACK_SCHEMA = "studio-camera-back-v1"
BACK_PRESETS = {
    "micro_four_thirds": (.0173, .013),
    "aps_c": (.0236, .0157),
    "full_frame": (.036, .024),
    "medium_44x33": (.044, .033),
    "film_6x6": (.056, .056),
    "large_4x5": (.120, .095),
}


def _number(value, name, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def camera_back(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"schema", "key", "width_m", "height_m", "fit"}:
        raise ValueError("camera_back must contain schema, key, width_m, height_m and fit")
    if value["schema"] != BACK_SCHEMA or value["fit"] != "center_crop":
        raise ValueError("unsupported camera back schema or fit")
    key = value["key"]
    if not isinstance(key, str) or key not in (*BACK_PRESETS, "custom"):
        raise ValueError("unknown camera back")
    width = _number(value["width_m"], "back width (m)", .001, .3)
    height = _number(value["height_m"], "back height (m)", .001, .3)
    if key != "custom" and (width, height) != BACK_PRESETS[key]:
        raise ValueError("camera back dimensions disagree with its preset")
    return dict(schema=BACK_SCHEMA, key=key, width_m=width, height_m=height, fit="center_crop")


def sensor_gate(back, aspect=1.):
    aspect = _number(aspect, "output aspect", .01, 100.)
    back = camera_back(back)
    if back is None:
        return dict(width_m=.024 * aspect, height_m=.024, output_aspect=aspect, fit="reference_height")
    height = min(back["height_m"], back["width_m"] / aspect)
    return dict(width_m=height * aspect, height_m=height, output_aspect=aspect, fit="center_crop")


def apply_camera_back(camera, back, aspect=1., reference_fov=None):
    """Resolve only optics. Call with the full image aspect, never a tile's."""
    result = dict(camera)
    back = camera_back(back)
    reference = reference_fov if reference_fov is not None else camera.get("back_reference_fov_degrees", camera.get("fov_degrees"))
    for key in ("camera_back", "sensor_gate", "back_reference_fov_degrees"):
        result.pop(key, None)
    if back is None:
        if "back_reference_fov_degrees" in camera:
            selected = camera.get("lens_selection")
            result["fov_degrees"] = (math.degrees(2 * math.atan(.024 / (2 * selected["focal_length_m"])))
                                     if selected else reference)
        return result
    selection = camera.get("lens_selection")
    if selection is not None:
        focal = selection["focal_length_m"]  # Validated by apply_lens_selection first.
        reference = math.degrees(2 * math.atan(.024 / (2 * focal)))
    else:
        reference = _number(reference, "back reference field of view", .001, 100.)
        focal = .024 / (2 * math.tan(math.radians(reference) / 2))
    gate = sensor_gate(back, aspect)
    fov = math.degrees(2 * math.atan(gate["height_m"] / (2 * focal)))
    if not .001 <= fov <= 100:
        raise ValueError("camera back exceeds the 100 degree analytical field; increase focal length or reduce the back")
    result.update(camera_back=back, sensor_gate=gate, back_reference_fov_degrees=reference, fov_degrees=fov)
    return result
