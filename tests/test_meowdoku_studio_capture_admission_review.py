"""Regression probes for complete actual Studio capture admission, CPU only."""
import copy

import pytest

from tests.test_meowdoku_studio_scene_prepare import capture
from meowdoku.studio_scene_prepare import StudioVisibleBake


@pytest.mark.parametrize("mutation", [
    lambda s: s["subject"].pop("faces"),
    lambda s: s["subject"].__setitem__("faces", []),
    lambda s: s["subject"]["ray"]["vertices"].__setitem__(0, 999),
    lambda s: s["subject"]["ray"]["material_ids"].__setitem__(0, 999),
    lambda s: s["subject"]["faces"][0].__setitem__(3, 999),
    lambda s: s["subject"]["faces"][0].__setitem__(4, []),
    lambda s: s["subject"]["model"].__setitem__(0, -1),
    lambda s: s["subject"]["model"].__setitem__(0, 2),
    lambda s: s["subject"]["model"].__setitem__(1, .1),
    lambda s: s["subject"]["model"].__setitem__(3, 1),
    lambda s: s["ray"]["normals"].__setitem__(0, 999),
    lambda s: s["ray"].pop("normals"),
], ids=["subject-missing", "subject-empty", "subject-ray-vertices", "subject-ray-materials",
        "subject-material", "subject-normals", "reflection", "nonuniform-scale", "shear",
        "nonaffine", "static-normal-drift", "static-normal-missing"])
def test_inconsistent_complete_capture_is_not_admitted(capture, mutation):
    changed = copy.deepcopy(capture)
    mutation(changed["state"])
    with pytest.raises(ValueError):
        StudioVisibleBake.from_capture(changed)


def test_actual_complete_capture_still_admits(capture):
    baked = StudioVisibleBake.from_capture(capture)
    assert len(baked.mesh.triangles) == len(capture["state"]["ray"]["material_ids"])
