"""Saved mesh proof and bounded Spectral subprocess route contract."""
import hashlib
import http.client
import json
import struct
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import marble_objects as objects  # noqa: E402
from meowdoku import marble_photo as photos  # noqa: E402
from meowdoku import server  # noqa: E402


def _saved_object(vertices, normals, material_ids):
    count = len(material_ids)
    packed = np.empty(count, dtype=[("positions", "<f4", (9,)),
                                    ("normals", "<f4", (9,)), ("id", "<u4")])
    packed["positions"] = vertices.reshape(count, 9)
    packed["normals"] = normals.reshape(count, 9)
    packed["id"] = material_ids
    pbr = np.zeros((2, 16), np.float32)
    pbr[:, 0:3] = .9
    pbr[:, 6] = 1.52
    pbr[0, 5] = 1.0
    pbr[1, 7] = 1.0
    tables = {name: {"stride": 16, "data": pbr.reshape(-1).tolist()}
              for name in ("pbr", "phong", "enamel", "texstack")}
    return {"v": 1, "name": "Photo source", "generator": "meowdoku-marble-object-v1",
            "design": {"base-shape": "marble", "kind": "cat_eye", "seed": 5},
            "materials": {"index": {"glass": 0, "eye": 1},
                          "extra": {"coated": 1, "paintA": 1, "paintB": 1},
                          "tables": tables},
            "physical": {"radius-mm": 25},
            "derived": {"massKg": .02},
            "geometry_sha256": hashlib.sha256(packed.tobytes()).hexdigest()}


def _wire(digest, vertices, normals, material_ids, **options):
    config = json.dumps({"object_digest": digest, "exposure_seconds": 30, **options}).encode()
    count = len(material_ids)
    return (b"MRP1" + struct.pack("<III", len(config), count, 0) + config +
            vertices.astype("<f4").tobytes() + normals.astype("<f4").tobytes() +
            material_ids.astype("<u4").tobytes())


@pytest.fixture()
def photo_payload(tmp_path):
    vertices = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                         [[0, 0, -1], [0, 1, 0], [1, 0, 0]]], np.float32)
    normals = np.tile(np.array([[[0, 0, 1]] * 3], np.float32), (2, 1, 1))
    ids = np.array([0, 1], np.uint32)
    store = objects.Store(tmp_path / "saved")
    digest = store.put(_saved_object(vertices, normals, ids))["digest"]
    return store, digest, vertices, normals, ids


def test_mrp1_matches_saved_geometry_and_pbr(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    wire = _wire(digest, vertices, normals, ids, camera={"resolution": 1024}, transport="fixed:3",
                 exposure_seconds=1800, orbit_frames=8)
    decoded_vertices, decoded_ids, pbr, config, seconds = photos.decode_upload(wire, store)
    np.testing.assert_array_equal(decoded_vertices, vertices)
    np.testing.assert_array_equal(decoded_ids, ids)
    assert pbr.shape == (2, 16) and config["scale_m"] == .025 and seconds == 1800
    assert config["camera"]["resolution"] == 1024 and config["orbit_frames"] == 8
    assert config["transport"] == "fixed:3"
    short_wire = _wire(digest, vertices, normals, ids, exposure_seconds=0.25)
    assert photos.decode_upload(short_wire, store)[4] == 0.25
    with pytest.raises(photos.PhotoError, match="exposure_seconds"):
        photos.decode_upload(_wire(digest, vertices, normals, ids, exposure_seconds=0.01), store)
    altered = bytearray(wire)
    altered[-4] = 0  # change the second face's material ID
    with pytest.raises(photos.PhotoError, match="differ from the saved"):
        photos.decode_upload(bytes(altered), store)
    with pytest.raises(photos.PhotoError, match="length"):
        photos.decode_upload(wire[:-1], store)


def test_timed_studio_light_frames_keep_one_scene_and_contiguous_shutter(photo_payload):
    store,digest,vertices,normals,ids=photo_payload
    base=[0.0]*16;base[3]=1.0;base[7]=1.0
    fixture={"vertices":[0,0,.2,.01,0,.2,0,.01,.2],"material_ids":[0],
             "pbr":base,"emissive_material_ids":[0],
             "emitter_profiles":{"version":1,"records":[]}}
    first=base.copy();first[8:11]=[1.,.5,0.]
    second=base.copy();second[8:11]=[0.,0.,0.]
    frames=[{"t0":0,"t1":.5,"pbr":first,"emitter_profiles":fixture["emitter_profiles"]},
            {"t0":.5,"t1":2,"pbr":second,"emitter_profiles":fixture["emitter_profiles"]}]
    options=dict(transport="fixed:3",exposure_seconds=2,fixtures=fixture,exposure_state_frames=frames)
    config=photos.decode_upload(_wire(digest,vertices,normals,ids,**options),store)[3]
    assert [frame["t1"]-frame["t0"] for frame in config["exposure_state_frames"]]==[.5,1.5]
    bad=[dict(frame) for frame in frames];bad[1]["t0"]=.6
    with pytest.raises(photos.PhotoError,match="contiguous"):
        photos.decode_upload(_wire(digest,vertices,normals,ids,**{**options,"exposure_state_frames":bad}),store)
    bad=[dict(frame) for frame in frames];bad[1]["pbr"]=second.copy();bad[1]["pbr"][0]=.8
    with pytest.raises(photos.PhotoError,match="non-light"):
        photos.decode_upload(_wire(digest,vertices,normals,ids,**{**options,"exposure_state_frames":bad}),store)


def test_simultaneous_sensor_lanes_share_one_timed_shutter_and_visibility(photo_payload):
    store,digest,vertices,normals,ids=photo_payload
    base=[0.0]*16;base[3]=base[7]=1.0;base[8:11]=[1.,1.,1.]
    fixture={'vertices':[0,0,.2,.01,0,.2,0,.01,.2],
             'material_ids':[0],'pbr':base,'emissive_material_ids':[0],
             'emitter_profiles':{'version':1,'records':[]}}
    frame={'t0':0,'t1':.1,'pbr':base,'emitter_profiles':fixture['emitter_profiles']}
    options=dict(transport='fixed:3',exposure_seconds=.1,fixtures=fixture,
        exposure_state_frames=[frame],camera_visible=False,
        sensor_lanes=[{}, {'eye_m':[.1,0,.07], 'sensor_transport':'fixed:1'},
                      {'eye_m':[-.1,0,.07]}])
    config=photos.decode_upload(_wire(digest,vertices,normals,ids,**options),store)[3]
    assert config['camera_visible'] is False
    assert len(config['sensor_lanes'])==3
    assert config['sensor_lanes'][0]==config['camera']
    assert [lane['eye_m'][0] for lane in config['sensor_lanes']]==[0,.1,-.1]
    assert config['sensor_lanes'][1]['sensor_transport']=='fixed:1'
    names=('blackCenter','gray18','whiteRing','blackOuter','red','green','blue','yellow')
    sparse={'version':1,'width':128,'height':128,
            'regions':{name:([[128,3]] if name=='gray18' else []) for name in names}}
    mapped=photos.decode_upload(_wire(digest,vertices,normals,ids,
        **{**options,'camera':{'resolution':128},'calibration_regions':[sparse]*3}),store)[3]
    assert mapped['calibration_regions'][1]['regions']['gray18']==[[128,3]]
    bad={**sparse,'regions':{**sparse['regions'],'gray18':[[127,3]]}}
    with pytest.raises(photos.PhotoError,match='sensor row'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,
            **{**options,'camera':{'resolution':128},'calibration_regions':[bad]*3}),store)
    with pytest.raises(photos.PhotoError,match='starting with the main camera'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,
            **{**options,'sensor_lanes':[{'eye_m':[0,0,.07]}]}),store)


def test_sensor_lane_artifacts_require_advertised_lane_and_stay_inside_job(tmp_path):
    digest='a'*64
    lane_dir=tmp_path/'lane-01';lane_dir.mkdir()
    (lane_dir/'preview-000001.png').write_bytes(b'lane image')
    (lane_dir/'exposure-000001.npz').write_bytes(b'lane hdr')
    status={'state':'complete','sensor_lanes':[{'index':1,
        'image':'lane-01/preview-000001.png','hdr':'lane-01/exposure-000001.npz'}]}
    jobs=object.__new__(photos.PhotoJobs)
    jobs.lock=threading.Lock();jobs.active_digest=digest
    jobs.job_directory=lambda value:tmp_path
    jobs._status_locked=lambda value,**kwargs:status
    with jobs.open_artifact(digest,'preview-000001.png',lane=1) as stream:
        assert stream.read()==b'lane image'
    with jobs.open_artifact(digest,'exposure-000001.npz',hdr=True,lane=1) as stream:
        assert stream.read()==b'lane hdr'
    with pytest.raises(photos.PhotoError,match='unavailable'):
        jobs.open_artifact(digest,'preview-000001.png',lane=2)
    with pytest.raises(photos.PhotoError,match='invalid photo artifact path'):
        jobs.open_artifact(digest,'../preview-000001.png',lane=1)


def test_timed_shop_trigger_metadata_keeps_named_source_identity(photo_payload):
    store,digest,vertices,normals,ids=photo_payload
    sys.path.insert(0,str(ROOT.parent/'spectral-analyzer'))
    from camera_designer.studio_named_emitters import named_emitter
    definition,_=named_emitter('fluorescent_cool_white')
    pbr=[0.0]*16;pbr[3]=pbr[7]=1.0
    emitter={'material':0,'profile':'fluorescent_cool_white','revision':definition['revision'],
             'gain':0.0,'filter':[1,1,1]}
    fixture={'vertices':[0,0,.2,.01,0,.2,0,.01,.2],'material_ids':[0],
             'pbr':pbr,'emissive_material_ids':[0],'shop_material_ids':[0],
             'trigger_emitters':[{'material':0,'receiver':'shop:0','gain':20.0}],
             'emitter_profiles':{'version':1,'records':[emitter]}}
    frames=[{'t0':0,'t1':2,'pbr':pbr,'emitter_profiles':fixture['emitter_profiles']}]
    config=photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',
        exposure_seconds=2,fixtures=fixture,exposure_state_frames=frames),store)[3]
    assert config['fixtures']['trigger_emitters'][0]['gain']==20.0
    bad={**fixture,'trigger_emitters':[{'material':0,'receiver':'other','gain':20.0}]}
    with pytest.raises(photos.PhotoError,match='trigger emitter'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',
            exposure_seconds=2,fixtures=bad,exposure_state_frames=frames),store)


class _FakeProcess:
    instances = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_photo_routes_reuse_cancel_and_guard_image(tmp_path, photo_payload, monkeypatch):
    store, saved_digest, vertices, normals, ids = photo_payload
    monkeypatch.setattr(photos.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs, "_validate_geometry", lambda self, payload, config: {
        "sphere_coverage": "mesh", "triangle_budget": 250000,
        "input_triangles": len(vertices), "effective_triangles": len(vertices), "analytic_spheres": 0,
    })
    _FakeProcess.instances.clear()
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}))
    sessions = server.Sessions(web, None)
    sessions.access.lookup = lambda url: None  # local HTTP test must not contact a public IP echo service
    sessions.marble_objects = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    cookie = ""
    def request(method, path, body=b"", content_type="application/octet-stream"):
        nonlocal cookie
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(method, path, body=body, headers={"Content-Type": content_type, "Cookie": cookie})
        response = conn.getresponse()
        # Requests from one browser retain its ownership cookie.
        cookie = "; ".join(v.split(";")[0] for k,v in response.getheaders() if k.lower()=="set-cookie") or cookie
        data = response.read()
        status = response.status
        conn.close()
        return status, data

    try:
        wire = _wire(saved_digest, vertices, normals, ids)
        status, body = request("POST", "/api/marble_photo", wire)
        started = json.loads(body)
        assert status == 202 and started["state"] == "preparing"
        digest = started["digest"]
        own_jobs = json.loads(request("GET", "/api/marble_photo/jobs")[1])
        assert [entry["digest"] for entry in own_jobs["entries"]] == [digest]
        assert own_jobs["queue_available"] is False
        stranger = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        stranger.request("GET", "/api/marble_photo/jobs")
        response = stranger.getresponse()
        assert json.loads(response.read())["entries"] == []
        stranger.close()
        stranger = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        stranger.request("POST", "/api/marble_photo/cancel", json.dumps({"digest":digest}), {"Content-Type":"application/json"})
        response = stranger.getresponse()
        assert response.status == 403
        response.read(); stranger.close()
        assert len(_FakeProcess.instances) == 1
        assert "--seconds" in _FakeProcess.instances[0].command
        live_wire = _wire(saved_digest, vertices, normals, ids, transport="fixed:3",
                          live={"session_id":"a"*32,"history_samples":8,"aspect":2})
        assert request("POST", "/api/marble_photo", live_wire)[0] == 409
        assert len(_FakeProcess.instances) == 1  # live never displaces this photo
        assert request("POST", "/api/marble_photo/live", json.dumps({"digest":digest,"stop":True}).encode(),
                       "application/json")[0] == 409
        assert _FakeProcess.instances[0].returncode is None

        assert request("POST", "/api/marble_photo", wire)[0] == 202
        assert len(_FakeProcess.instances) == 1  # same saved scene reuses its job
        status, body = request("GET", f"/api/marble_photo?digest={digest}")
        assert status == 200 and json.loads(body)["state"] == "preparing"
        directory = sessions.marble_photos.job_directory(digest)
        assert directory.is_relative_to(sessions.state.users_dir)
        assert not (sessions.marble_photos.directory / digest).exists()
        _FakeProcess.instances[0].returncode = 0
        status, body = request("GET", f"/api/marble_photo?digest={digest}")
        assert status == 200 and json.loads(body)["state"] == "error"
        _FakeProcess.instances[0].returncode = None
        (directory / "preview-000120.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
        (directory / "status.json").write_text(json.dumps({
            "digest": digest, "state": "exposing", "image": "preview-000120.png"}))
        status, body = request("GET", f"/api/marble_photo/image?digest={digest}&name=preview-000120.png")
        assert status == 200 and body.startswith(b"\x89PNG")
        (directory / "orbit.avi.tmp").write_bytes(b"RIFFprogress")
        (directory / "status.json").write_text(json.dumps({
            "digest": digest, "state": "encoding", "image": "preview-000120.png",
            "frames": [{"index": 0, "image": "preview-000120.png"}]}))
        status, body = request("GET", f"/api/marble_photo?digest={digest}")
        assert status == 200 and json.loads(body)["encoding_bytes"] == len(b"RIFFprogress")
        assert request("GET", f"/api/marble_photo/image?digest={digest}&name=../payload.npz")[0] == 400
        (directory / "exposure-000120.npz").write_bytes(b"linear-rgb-hdr")
        (directory / "orbit.avi").write_bytes(b"RIFFtestAVI ")
        (directory / "status.json").write_text(json.dumps({
            "digest": digest, "state": "complete", "image": "preview-000120.png",
            "hdr": "exposure-000120.npz", "frames": [{"index": 0, "image": "preview-000120.png",
            "hdr": "exposure-000120.npz"}], "animation": "orbit.avi", "animation_expires_at": photos.time.time() + 3600}))
        status, body = request("GET", f"/api/marble_photo/hdr?digest={digest}&name=exposure-000120.npz")
        assert status == 200 and body == b"linear-rgb-hdr"
        status, body = request("GET", f"/api/marble_photo/animation?digest={digest}")
        assert status == 200 and body.startswith(b"RIFF")
        expired = json.loads((directory / "status.json").read_text())
        expired["animation_expires_at"] = photos.time.time() - 1
        (directory / "status.json").write_text(json.dumps(expired))
        assert request("GET", f"/api/marble_photo/animation?digest={digest}")[0] == 404
        assert not (directory / "orbit.avi").exists()
        assert json.loads((directory / "status.json").read_text())["animation_expired"]
        assert request("GET", f"/api/marble_photo/hdr?digest={digest}&name=../payload.npz")[0] == 400
        status, body = request("POST", "/api/marble_photo/cancel", json.dumps({"digest": digest}).encode(),
                               "application/json")
        assert status == 200 and json.loads(body)["state"] == "canceled"
        assert _FakeProcess.instances[0].returncode == -15
        assert request("GET", f"/api/marble_photo?digest={digest}")[0] == 404
        code, body = request("POST", "/api/marble_photo", live_wire)
        assert code == 202
        live_digest = json.loads(body)["digest"]
        assert len(_FakeProcess.instances) == 2
        live_dir = sessions.marble_photos.job_directory(live_digest)
        control = json.loads((live_dir / "live-control.json").read_text())
        assert control["live"]["history_samples"] == 8 and control["live"]["width"] == 256
        assert request("POST", "/api/marble_photo", wire)[0] == 409  # photo cannot silently replace live
        code, _ = request("POST", "/api/marble_photo/live", json.dumps({"digest":live_digest,"revision":1,
            "camera":{"eye_m":[.1,0,.07]},"live":{"history_samples":4}}).encode(), "application/json")
        assert code == 200
        control = json.loads((live_dir / "live-control.json").read_text())
        assert control["revision"] == 1 and control["live"]["history_samples"] == 4
        assert request("POST", "/api/marble_photo/live", json.dumps({"digest":live_digest,"revision":0}).encode(),
                       "application/json")[0] == 409
        assert request("POST", "/api/marble_photo/live", json.dumps({"digest":live_digest,"stop":True}).encode(),
                       "application/json")[0] == 200
        assert _FakeProcess.instances[1].returncode == -15

    finally:
        sessions.marble_photos.close()
        httpd.shutdown()
        httpd.server_close()

def test_matching_live_marker_does_not_transfer_ownership(tmp_path,photo_payload,monkeypatch):
    store,digest,vertices,normals,ids=photo_payload
    monkeypatch.setattr(photos.subprocess,'Popen',_FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs,'_validate_geometry',lambda *args:{})
    jobs=photos.PhotoJobs(tmp_path/'jobs',tmp_path)
    wire=_wire(digest,vertices,normals,ids,transport='fixed:3',live={'session_id':'f'*32,'history_samples':8,'aspect':2})
    try:
        first=jobs.start(wire,store,owners=['account:1'])
        private=jobs.job_directory(first['digest'])
        assert private == tmp_path/'jobs'/'users'/'account'/'1'/'assets'/'marble_photos'/first['digest']
        assert (private/'config.json').is_file()
        assert not (jobs.directory/first['digest']).exists()
        process=jobs.process
        altered=_wire(digest,vertices,normals,ids,transport='fixed:3',live={'session_id':'f'*32,'history_samples':16,'aspect':2})
        with pytest.raises(photos.PhotoError,match='another identity'):
            jobs.start(altered,store,owners=['account:2'])
        assert jobs.process is process and process.poll() is None
        assert jobs.active_digest==first['digest']
    finally:jobs.close()

def test_client_job_secret_environment_and_cleanup(tmp_path,photo_payload,monkeypatch):
    from meowdoku.client_ray_process_bridge import ClientRayProcessBridge
    from meowdoku.client_ray_relay import RelayError
    store,digest,vertices,normals,ids=photo_payload
    monkeypatch.setattr(photos.subprocess,'Popen',_FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs,'_validate_geometry',lambda *args:{})
    jobs=photos.PhotoJobs(tmp_path/'jobs',tmp_path);bridge=ClientRayProcessBridge()
    wire=_wire(digest,vertices,normals,ids,transport='fixed:3',camera={'resolution':128},client_intersections=True)
    try:
        result=jobs.start(wire,store,owners=['account:1'],client_bridge=bridge,bridge_url='https://127.0.0.1:9000/api/client_rays/engine',bridge_ca=tmp_path/'ca.crt')
        env=jobs.process.kwargs['env'];secret=env['MEOWDOKU_RAY_BRIDGE_SECRET']
        assert result['client_join']['credential']!=secret
        assert env['MEOWDOKU_RAY_BRIDGE_JOB']==result['digest']
        config=(jobs.job_directory(result['digest'])/'config.json').read_text()
        assert secret not in config and result['client_join']['credential'] not in config
        jobs.process.returncode=0
        jobs.status(result['digest'])
        with pytest.raises(RelayError):bridge.join(result['digest'],'account:1')
    finally:jobs.close()

@pytest.mark.parametrize('options',[{'camera':{'resolution':8192}}, {'transport':'rgb'}, {'orbit_frames':2}])
def test_client_job_rejects_unsupported_mode(photo_payload,options):
    store,digest,vertices,normals,ids=photo_payload
    base={'transport':'fixed:3','camera':{'resolution':128},'client_intersections':True,**options}
    with pytest.raises(photos.PhotoError):photos.decode_upload(_wire(digest,vertices,normals,ids,**base),store)

@pytest.mark.parametrize('resolution',[128,256,512,768,1024,2048,4096])
def test_cooperative_still_admits_all_exposure_sizes(photo_payload,resolution):
    store,digest,vertices,normals,ids=photo_payload
    config=photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',camera={'resolution':resolution},client_intersections=True),store)[3]
    assert config['client_intersections'] is True and config['camera']['resolution']==resolution


@pytest.mark.parametrize('resolution',[128,256,512,768,1024,2048,4096])
def test_cooperative_live_admits_all_exposure_sizes(photo_payload,resolution):
    store,digest,vertices,normals,ids=photo_payload
    config=photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',camera={'resolution':resolution},client_intersections=True,live={'session_id':'a'*32,'history_samples':8,'aspect':1}),store)[3]
    assert config['client_intersections'] is True and config['live']['width']==resolution


def test_participant_live_renewal_preserves_scene_and_owner(tmp_path,photo_payload,monkeypatch):
    import json,time
    from meowdoku.client_ray_process_bridge import ClientRayProcessBridge
    store,digest,vertices,normals,ids=photo_payload
    monkeypatch.setattr(photos.subprocess,'Popen',_FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs,'_validate_geometry',lambda *args:{})
    jobs=photos.PhotoJobs(tmp_path/'jobs',tmp_path);bridge=ClientRayProcessBridge()
    wire=_wire(digest,vertices,normals,ids,transport='fixed:3',camera={'resolution':128},client_intersections=True,live={'session_id':'a'*32,'history_samples':8})
    try:
        result=jobs.start(wire,store,owners=['account:1'],client_bridge=bridge,bridge_url='https://127.0.0.1:9000/api/client_rays/engine',bridge_ca=tmp_path/'ca.crt')
        path=jobs.job_directory(result['digest'])/'live-control.json'
        before=json.loads(path.read_text());before['expires_at']=time.time()-5
        path.write_text(json.dumps(before))
        jobs.renew_participant_live(result['digest'],'account:2')
        assert json.loads(path.read_text())==before
        jobs.renew_participant_live('f'*64,'account:1')
        assert json.loads(path.read_text())==before
        jobs.process.returncode=0
        jobs.renew_participant_live(result['digest'],'account:1')
        assert json.loads(path.read_text())==before
        jobs.process.returncode=None
        jobs.renew_participant_live(result['digest'],'account:1')
        after=json.loads(path.read_text())
        assert after['expires_at']>time.time()
        assert {k:v for k,v in after.items() if k!='expires_at'}=={k:v for k,v in before.items() if k!='expires_at'}
        jobs.renew_participant_live(result['digest'],'account:1')
        assert json.loads(path.read_text())==after
        config_path=jobs.job_directory(result['digest'])/'config.json'
        original=json.loads(config_path.read_text())
        for config in ({**original,'client_intersections':False}, {k:v for k,v in original.items() if k!='live'}):
            config_path.write_text(json.dumps(config))
            path.write_text(json.dumps(before))
            jobs.renew_participant_live(result['digest'],'account:1')
            assert json.loads(path.read_text())==before
    finally:jobs.close()


def test_cooperative_4k_history_budget_is_not_bypassed(photo_payload):
    store,digest,vertices,normals,ids=photo_payload
    with pytest.raises(photos.PhotoError,match='2 GiB'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',camera={'resolution':4096},client_intersections=True,live={'session_id':'a'*32,'history_samples':16,'aspect':1}),store)
