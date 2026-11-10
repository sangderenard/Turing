"""Exposure policy admission, saved-mesh identity and active-job safety; no GPU/server."""
import json

import numpy as np
import pytest

from .test_meowdoku_marble_photo import photos, photo_payload, _wire, _FakeProcess


SAVED = {"digest": "a" * 64, "geometry_sha256": "b" * 64, "physical": {"radius-mm": 25}}


def request(live=False, **options):
    value = {"object_digest": SAVED["digest"], "transport": "fixed:3", **options}
    if live:
        value["live"] = {"session_id": "c" * 32, "history_samples": 16, "aspect": 1, "checkpoint": True}
    return value


@pytest.mark.parametrize("live", [False, True])
def test_geometry_policy_defaults_and_boundaries(live):
    config, _ = photos._config(request(live), SAVED)
    assert config["triangle_budget"] == 250_000
    assert config["sphere_coverage"] == "mesh"
    for budget in (1000, 250_000, 1_000_000):
        for coverage in ("mesh", "exact"):
            config, _ = photos._config(request(live, triangle_budget=budget,
                sphere_coverage=coverage, max_bounces=37, bounce_mode="auto"), SAVED)
            assert (config["triangle_budget"], config["sphere_coverage"]) == (budget, coverage)
            assert (config["max_bounces"], config["bounce_mode"]) == (37, "auto")
            if live:
                assert config["live"]["checkpoint"] is True
                assert config["live"]["history_samples"] == 16


@pytest.mark.parametrize("value", [999, 1_000_001, 1000.5, 1000.0, True, False, None,
                                 "250000", float("nan"), float("inf"), [], {}])
@pytest.mark.parametrize("live", [False, True])
def test_geometry_budget_rejects_invalid_values(value, live):
    with pytest.raises(photos.PhotoError, match="triangle_budget"):
        photos._config(request(live, triangle_budget=value), SAVED)


@pytest.mark.parametrize("value", ["analytic", "Mesh", "", None, True, 1, [], {}])
@pytest.mark.parametrize("live", [False, True])
def test_geometry_coverage_rejects_invalid_values(value, live):
    with pytest.raises(photos.PhotoError, match="sphere_coverage"):
        photos._config(request(live, sphere_coverage=value), SAVED)


@pytest.mark.parametrize("coverage", ["mesh", "exact"])
def test_policy_does_not_filter_or_bypass_saved_hash(photo_payload, coverage):
    store, digest, vertices, normals, ids = photo_payload
    options = {"sphere_coverage": coverage, "triangle_budget": 1000}
    if coverage == "mesh":
        got_vertices, got_ids, _, config, _ = photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)
        np.testing.assert_array_equal(got_vertices, vertices)
        np.testing.assert_array_equal(got_ids, ids)
        assert config["triangle_budget"] == 1000
        assert config["sphere_coverage"] == coverage
    else:
        # A legacy saved mesh does not acquire an analytic claim from the new
        # selector. Kernel admission authenticates the saved sphere metadata.
        with pytest.raises(photos.PhotoError, match="saved analytic sphere metadata"):
            photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)
    damaged = vertices.copy()
    damaged[0, 0, 0] += .1
    with pytest.raises(photos.PhotoError, match="differ from the saved"):
        photos.decode_upload(_wire(digest, damaged, normals, ids, **options), store)


@pytest.mark.parametrize("live", [False, True])
def test_invalid_policy_never_cancels_current_job(tmp_path, photo_payload, monkeypatch, live):
    store, digest, vertices, normals, ids = photo_payload
    monkeypatch.setattr(photos.subprocess, "Popen", _FakeProcess)
    _FakeProcess.instances.clear()
    preflight_calls = []
    reject_preflight = False

    def preflight(self, payload, config_path):
        preflight_calls.append(self.process)
        config = json.loads(config_path.read_text())
        if reject_preflight:
            raise photos.PhotoError("Scene needs 24450 triangles; exposure budget is 1000. Raise the budget or simplify the saved design.")
        return {"sphere_coverage": config["sphere_coverage"], "triangle_budget": config["triangle_budget"],
                "input_triangles": 24450, "effective_triangles": 24450, "analytic_spheres": 0}

    monkeypatch.setattr(photos.PhotoJobs, "_validate_geometry", preflight)
    jobs = photos.PhotoJobs(tmp_path / "jobs", tmp_path)
    options = {"transport": "fixed:3"}
    if live:
        options["live"] = {"session_id": "c" * 32, "history_samples": 16, "aspect": 1}
    try:
        started = jobs.start(_wire(digest, vertices, normals, ids, **options), store)
        process = jobs.process
        assert preflight_calls == [None]
        assert started["geometry_stats"]["effective_triangles"] == 24450
        for invalid in ({"triangle_budget": 999}, {"triangle_budget": 1000.5},
                        {"sphere_coverage": "analytic"}, {"sphere_coverage": "exact"}):
            with pytest.raises(photos.PhotoError):
                jobs.start(_wire(digest, vertices, normals, ids, **options, **invalid), store)
            assert jobs.active_digest == started["digest"]
            assert jobs.process is process and process.poll() is None
            assert len(_FakeProcess.instances) == 1
        assert preflight_calls == [None], "invalid fields and unsupported exact fail before scene preflight"
        reject_preflight = True
        with pytest.raises(photos.PhotoError, match="Scene needs 24450 triangles"):
            jobs.start(_wire(digest, vertices, normals, ids, **options, triangle_budget=1000), store)
        assert preflight_calls == [None, process], "final-scene preflight runs before any current worker is stopped"
        assert jobs.active_digest == started["digest"]
        assert jobs.process is process and process.poll() is None
        assert len(_FakeProcess.instances) == 1
        stored = json.loads((jobs.directory / started["digest"] / "config.json").read_text())
        assert stored["triangle_budget"] == 250_000 and stored["sphere_coverage"] == "mesh"
    finally:
        jobs.close()


@pytest.mark.parametrize("live", [False, True])
def test_fixture_mesh_uses_selected_budget_above_old_5000_cap(live):
    # Real stand geometry must be admitted, never truncated to the old cap.
    n = 6500
    fixture = {"vertices": [0.0, 0.0, 0.0, .1, 0.0, 0.0, 0.0, .1, 0.0] * n,
               "material_ids": [0] * n, "pbr": [0.0] * 16}
    config, _ = photos._config(request(live, fixtures=fixture, triangle_budget=n), SAVED)
    assert len(config["fixtures"]["vertices"]) == 9 * n
    assert len(config["fixtures"]["material_ids"]) == n
    with pytest.raises(photos.PhotoError, match="selected triangle_budget"):
        photos._config(request(live, fixtures=fixture, triangle_budget=n - 1), SAVED)


def test_fixture_config_still_has_independent_byte_bound():
    import struct
    # Header admission precedes JSON parsing or allocations for geometry.
    raw = b"MRP1" + struct.pack("<III", 2 * 1024 * 1024 + 1, 1, 0)
    with pytest.raises(photos.PhotoError, match="invalid MRP1 header"):
        photos.decode_upload(raw, None)


def test_direct_decode_enforces_whole_upload_byte_bound(monkeypatch):
    monkeypatch.setattr(photos, "MAX_UPLOAD_BYTES", 16)
    with pytest.raises(photos.PhotoError, match="upload exceeds byte limit") as error:
        photos.decode_upload(b"MRP1" + b"0" * 13, None)
    assert error.value.status == 413
