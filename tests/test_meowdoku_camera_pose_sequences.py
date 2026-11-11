"""Explicit sequence admission reuses the still-camera optics authorities."""
import copy
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import marble_photo as photos
from meowdoku.marble_camera_back import apply_camera_back
from meowdoku.marble_lens_contract import apply_lens_selection, SELECTION_SCHEMA

SPEC = Path(os.environ.get("STUDIO_TEST_SPECTRAL_ROOT", Path(__file__).resolve().parents[2] / "spectral-analyzer"))
SAVED = {"digest": "a" * 64, "geometry_sha256": "b" * 64, "physical": {"radius-mm": 25}}
BACK = {"schema": "studio-camera-back-v1", "key": "full_frame", "width_m": .036, "height_m": .024, "fit": "center_crop"}


def config(**options):
    return photos._config({"object_digest": SAVED["digest"], "transport": "fixed:3", **options}, SAVED, SPEC)[0]


def test_pose_overrides_are_resolved_ordered_and_do_not_mutate_input():
    poses = [{"eye_m": [.02, .01, .07]}, {"target_m": [.01, .02, .18], "fov_degrees": 50, "focus_distance_m": .2}]
    original = copy.deepcopy(poses)
    result = config(camera={"resolution": 512, "aperture_radius_m": .004}, camera_poses=poses)
    assert poses == original and result["orbit_frames"] == 1
    assert len(result["camera_poses"]) == 2
    first, second = result["camera_poses"]
    assert first["eye_m"] == poses[0]["eye_m"] and first["target_m"] == [0, 0, .18]
    assert second["eye_m"] == [0, 0, .07] and second["target_m"] == poses[1]["target_m"]
    assert second["fov_degrees"] == 50 and second["focus_distance_m"] == .2
    assert all(p["resolution"] == 512 and p["aperture_radius_m"] == .004 for p in result["camera_poses"])
    assert "camera_poses" not in config()
    assert len(config(camera_poses=[{}] * 32)["camera_poses"]) == 32


@pytest.mark.parametrize("poses", [None, {}, [], [{}] * 33, [None], [{"projection": "fisheye"}], [{"width": 1024}],
    [{"eye_m": [0, 0]}], [{"eye_m": [0, 0, float("nan")]}], [{"eye_m": [0, 0, .18]}],
    [{"eye_m": [0, 0, .18000001]}], [{"fov_degrees": 101}], [{"focus_distance_m": 0}],
    [{"aperture_radius_m": .1}], [{"lens_selection": {}}], [{"camera_back": {}}]])
def test_bad_sequence_fails_before_admission(poses):
    with pytest.raises(photos.PhotoError):
        config(camera_poses=poses)


@pytest.mark.parametrize("options", [{"orbit_frames": 8}, {"oversampling": 2}, {"tiled": {"tile_size": 128}},
    {"live": {}}, {"client_intersections": True}, {"camera": {"projection": "orthographic"}}])
def test_sequence_does_not_alias_other_job_modes(options):
    with pytest.raises(photos.PhotoError):
        config(camera_poses=[{}, {}], **options)


@pytest.mark.parametrize("frames", [True, 1.5, 8.1, "8"])
def test_orbit_count_does_not_silently_truncate(frames):
    with pytest.raises(photos.PhotoError, match="orbit_frames"):
        config(orbit_frames=frames)


def test_lens_and_back_resolve_once_per_pose():
    lens = {"schema": SELECTION_SCHEMA, "key": None, "source_sha256": None,
            "focal_length_m": .05, "f_number": 4, "sensor_height_m": .024}
    result = config(camera={"camera_back": BACK}, camera_poses=[{"lens_selection": lens}, {"fov_degrees": 48}])
    first, second = result["camera_poses"]
    expected = apply_camera_back(apply_lens_selection(result["camera"], lens, SPEC), BACK, 1.)
    assert first == expected
    assert second["back_reference_fov_degrees"] == 48 and second["sensor_gate"]["output_aspect"] == 1.
    # A browser has already resolved the back once. Reapplying it must keep
    # the reference field, except when a pose deliberately replaces that FOV.
    resolved = result["camera"]
    again = config(camera=resolved, camera_poses=[{}, {"fov_degrees": 48}, {"camera_back": None}])["camera_poses"]
    assert again[0] == resolved
    assert again[1] == second
    assert again[2]["fov_degrees"] == resolved["back_reference_fov_degrees"]
    assert "camera_back" not in again[2]
