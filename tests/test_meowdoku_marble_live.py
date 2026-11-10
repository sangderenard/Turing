"""Bounded live controls, revision ordering and still-job isolation."""
import json
import threading
import time
from types import SimpleNamespace
import pytest
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/"examples")]
from meowdoku.marble_live import live_settings, update_live
from meowdoku.marble_photo import PhotoError


def test_bounded_ring_memory_and_rectangular_view():
    value=live_settings({"history_samples":16,"aspect":2},{"resolution":256})
    assert (value["width"],value["height"],value["history_bytes"]) == (256,128,7077888)
    for samples in (0,-1,1.5,True):
        with pytest.raises(PhotoError):live_settings({"history_samples":samples},{"resolution":256})
    assert live_settings({"history_samples":10920},{"resolution":128})["history_bytes"]<=2**31
    assert live_settings({"history_samples":674,"checkpoint":True},{"resolution":512})["history_bytes"]==2145386496
    with pytest.raises(PhotoError):live_settings({"history_samples":10921},{"resolution":128})
    with pytest.raises(PhotoError):live_settings({"history_samples":675,"checkpoint":True},{"resolution":512})
    with pytest.raises(PhotoError):live_settings({}, {"resolution":1024})


@pytest.mark.parametrize("mode,percentile,tone", [("auto99",99,"exponential"),("manual",99,"exponential"),
    ("percentile",99,"exponential"),("specular",95,"linear")])
def test_live_display_modes_validate_and_forward_without_changing_history(mode,percentile,tone):
    base=live_settings({"checkpoint":True},{"resolution":128})
    selected=live_settings({"checkpoint":True,"display":{"normalization":mode}},{"resolution":128})
    assert {k:v for k,v in selected.items() if k!="display"}=={k:v for k,v in base.items() if k!="display"}
    assert selected["display"]["normalization_percentile"]==percentile
    assert selected["display"]["tone_curve"]==tone
    for value in (1,94.3,100):
        display=live_settings({"display":{"normalization":mode,"normalization_percentile":value,
            "tone_curve":"exponential"}},{"resolution":128})["display"]
        assert display["normalization_percentile"]==value
        assert display["tone_curve"]==tone


@pytest.mark.parametrize("bad", [True,"95",None,.99,100.01,float("nan"),float("inf"),10**1000])
def test_live_invalid_percentile_rejected(bad):
    with pytest.raises(PhotoError,match="normalization_percentile"):
        live_settings({"display":{"normalization":"specular","normalization_percentile":bad}},{"resolution":128})


def test_live_specular_still_validates_requested_tone():
    with pytest.raises(PhotoError,match="tone_curve"):
        live_settings({"display":{"normalization":"specular","tone_curve":"bogus"}},{"resolution":128})


def fake_jobs(tmp_path, *, live=True):
    digest="a"*64
    directory=tmp_path/digest;directory.mkdir()
    camera={"resolution":128,"eye_m":[0,0,.07],"target_m":[0,0,.18],
            "fov_degrees":18.,"focus_distance_m":.11,"aperture_radius_m":.002}
    settings=live_settings({},camera)
    (directory/"config.json").write_text(json.dumps({"live":settings} if live else {}))
    (directory/"live-control.json").write_text(json.dumps({"camera":camera,"live":settings,"revision":0,"expires_at":0}))
    return SimpleNamespace(directory=tmp_path,lock=threading.Lock(),active_digest=digest,
        process=SimpleNamespace(poll=lambda:None),_status_locked=lambda digest:{}),digest,directory


def test_updates_are_latest_revision_only_and_heartbeats_do_not_reset(tmp_path):
    jobs,digest,directory=fake_jobs(tmp_path)
    result=update_live(jobs,digest,{"revision":2,"camera":{"eye_m":[.1,0,.07]},"live":{"history_samples":8}})
    assert result=={"ok":True,"revision":2}
    control=json.loads((directory/"live-control.json").read_text())
    assert control["camera"]["eye_m"]==[.1,0,.07]
    assert control["live"]["history_samples"]==8
    assert time.time()+28<control["expires_at"]<=time.time()+30
    with pytest.raises(PhotoError,match="stale"):update_live(jobs,digest,{"revision":1})
    update_live(jobs,digest,{"revision":2,"camera":{"eye_m":[99,99,99]}})
    again=json.loads((directory/"live-control.json").read_text())
    assert again["camera"]==control["camera"]


def test_live_control_cannot_touch_still_or_other_job(tmp_path):
    jobs,digest,directory=fake_jobs(tmp_path,live=False)
    with pytest.raises(PhotoError,match="photograph"):update_live(jobs,digest,{"revision":1})
    with pytest.raises(PhotoError,match="invalid live digest"):update_live(jobs,"../escape",{})
    jobs.active_digest="b"*64
    with pytest.raises(PhotoError,match="no longer active"):update_live(jobs,digest,{})


def test_invalid_control_preserves_prior_state(tmp_path):
    jobs,digest,directory=fake_jobs(tmp_path)
    before=(directory/"live-control.json").read_bytes()
    for value in ({"revision":True},{"revision":1,"camera":{"eye_m":[0,0,.18]}},
                  {"revision":1,"live":{"aspect":2}},{"revision":1,"live":[]},
                  {"revision":1,"live":{"display":{"tint":4}}},
                  {"revision":1,"live":{"display":{"normalization":"bad"}}},
                  {"revision":1,"live":{"display":{"normalization":"specular","normalization_percentile":True}}},
                  {"revision":1,"live":{"display":{"normalization":"percentile","normalization_percentile":float("nan")}}},
                  {"revision":1,"live":{"display":{"normalization":"specular","tone_curve":"bad"}}},
                  {"revision":1,"live":{"checkpoint":1}}):
        with pytest.raises(PhotoError):update_live(jobs,digest,value)
        assert (directory/"live-control.json").read_bytes()==before


def test_start_never_cancels_photo_or_other_live_tab(tmp_path, monkeypatch):
    import numpy as np
    from meowdoku import marble_photo as photos
    jobs=photos.PhotoJobs(tmp_path, tmp_path/"spectral")
    existing="b"*64
    directory=tmp_path/existing;directory.mkdir()
    (directory/"config.json").write_text("{}")
    fake_process=SimpleNamespace(poll=lambda:None)
    jobs.process=fake_process;jobs.active_digest=existing
    config={"live":{"session_id":"a"*32}}
    monkeypatch.setattr(photos,"decode_upload",lambda *a:(np.zeros((1,3,3),np.float32),np.zeros(1,np.int32),np.zeros((1,16),np.float32),config,3600))
    monkeypatch.setattr(photos.PhotoJobs,"_validate_geometry",lambda *a:{})
    try:
        with pytest.raises(PhotoError,match="GPU is busy"):jobs.start(b"ignored",None)
        assert jobs.process is fake_process and jobs.active_digest==existing
        (directory/"config.json").write_text(json.dumps({"live":{"session_id":"b"*32}}))
        with pytest.raises(PhotoError,match="Another tab"):jobs.start(b"ignored",None)
        assert jobs.process is fake_process
        config.clear()
        with pytest.raises(PhotoError,match="GPU is busy"):jobs.start(b"ignored",None)
        (directory/"config.json").write_text("{}")
        with pytest.raises(PhotoError,match="photograph"):jobs.cancel(existing,live_only=True)
        assert jobs.process is fake_process
    finally:
        jobs.process=None;jobs.close()


def test_control_retries_windows_reader_sharing_collision(tmp_path,monkeypatch):
    from meowdoku import marble_live
    jobs,digest,directory=fake_jobs(tmp_path)
    original=marble_live.os.replace;calls=[]
    def replace(source,target):
        calls.append((source,target))
        if len(calls)<3:raise PermissionError("temporary reader")
        return original(source,target)
    monkeypatch.setattr(marble_live.os,"replace",replace)
    monkeypatch.setattr(marble_live.time,"sleep",lambda seconds:None)
    assert update_live(jobs,digest,{"revision":1})["revision"]==1
    assert len(calls)==3


@pytest.mark.parametrize("key,value", [("exposure", -2.), ("whitebalance", .75),
    ("saturation", 1.2), ("vibrance", .35), ("black_point", .03),
    ("firefly_filter", True), ("firefly_threshold", 4.), ("firefly_strength", .5)])
def test_live_color_and_cleanup_controls_reach_display_only(tmp_path, key, value):
    jobs, digest, directory = fake_jobs(tmp_path)
    before = json.loads((directory / "live-control.json").read_text())
    update_live(jobs, digest, {"revision": 1, "live": {"display": {key: value}}})
    after = json.loads((directory / "live-control.json").read_text())
    assert after["camera"] == before["camera"]
    assert {k:v for k,v in after["live"].items() if k != "display"} == {k:v for k,v in before["live"].items() if k != "display"}
    assert after["live"]["display"][key] == value


@pytest.mark.parametrize("key", ["exposure", "whitebalance", "saturation", "vibrance",
    "black_point", "firefly_threshold", "firefly_strength"])
@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf"), 10**1000])
def test_invalid_live_color_controls_reject_without_mutation(tmp_path, key, value):
    jobs, digest, directory = fake_jobs(tmp_path)
    before = (directory / "live-control.json").read_bytes()
    with pytest.raises(PhotoError):
        update_live(jobs, digest, {"revision": 1, "live": {"display": {key: value}}})
    assert (directory / "live-control.json").read_bytes() == before


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_live_filter_requires_boolean(value):
    with pytest.raises(PhotoError):
        live_settings({"display": {"firefly_filter": value}}, {"resolution": 128})
