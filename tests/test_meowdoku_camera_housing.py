"""Bounded CPU housing endpoint, with real native geometry and no exposures."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
from examples.meowdoku import marble_camera_housing as housing
from examples.meowdoku import server

ROOT = Path(__file__).resolve().parents[1]
SPECTRAL = ROOT.parent / 'spectral-analyzer'
CAMERA = {'eye_m': [.3, .4, .1], 'target_m': [0, 0, .18], 'aperture_radius_m': .005}


@pytest.fixture(autouse=True)
def empty_cache():
    housing._CACHE.clear()
    yield
    housing._CACHE.clear()


@pytest.mark.parametrize('body,status', [
    (b'', 413), (b'x' * 4097, 413), (b'{', 400), (b'[]', 400),
    (b'{"camera":{}}', 400),
    (json.dumps({'camera': {**CAMERA, 'aperture_blades': 1000000}}).encode(), 400),
    (json.dumps({'camera': {**CAMERA, 'eye_m': [0, 0, 11]}}).encode(), 400),
    (json.dumps({'camera': {**CAMERA, 'aperture_radius_m': float('nan')}}).encode(), 400),
    (json.dumps({'camera': {**CAMERA, 'eye_m': [10 ** 1000, 0, 0]}}).encode(), 400),
])
def test_invalid_requests_never_build(monkeypatch, body, status):
    monkeypatch.setattr(housing, 'camera_mount_source', lambda *a: pytest.fail('invalid request reached native builder'))
    with pytest.raises(housing.HousingError) as error:
        housing.housing_response(body, SPECTRAL)
    assert error.value.status == status


def test_busy_and_native_response_bounds(monkeypatch):
    body = json.dumps({'camera': CAMERA}).encode()
    housing._LOCK.acquire()
    try:
        with pytest.raises(housing.HousingError, match='busy') as error:
            housing.housing_response(body, SPECTRAL)
        assert error.value.status == 503
    finally:
        housing._LOCK.release()
    monkeypatch.setattr(housing, 'camera_mount_source', lambda *a: {'triangle_count': housing.MAX_HOUSING_TRIANGLES + 1})
    with pytest.raises(housing.HousingError, match='bounded geometry'):
        housing.housing_response(body, SPECTRAL)
    monkeypatch.setattr(housing, 'camera_mount_source', lambda *a: {'triangle_count': 1, 'oversized': 'x' * 200})
    monkeypatch.setattr(housing, 'MAX_RESPONSE_BYTES', 100)
    with pytest.raises(housing.HousingError, match='response limit'):
        housing.housing_response(body, SPECTRAL)
    assert not housing._CACHE


def test_cache_is_bounded_and_pose_keyed(monkeypatch):
    calls = []
    def source(camera, root):
        calls.append(camera)
        return {'triangle_count': 1, 'camera': camera}
    monkeypatch.setattr(housing, 'camera_mount_source', source)
    body = json.dumps({'camera': CAMERA}).encode()
    first = housing.housing_response(body, SPECTRAL)
    assert housing.housing_response(body, SPECTRAL) is first
    assert len(calls) == 1
    for x in [.5, .6, .7]:
        housing.housing_response(json.dumps({'camera': {**CAMERA, 'eye_m': [x, .4, .1]}}).encode(), SPECTRAL)
    assert len(calls) == 4 and len(housing._CACHE) == 2
    assert sum(map(len, housing._CACHE.values())) <= housing.MAX_CACHE_BYTES


def test_lens_selection_uses_authoritative_pupil(monkeypatch):
    selection = {'schema': 'studio-analytical-lens-selection-v1', 'key': None,
                 'source_sha256': None, 'focal_length_m': .050,
                 'f_number': 8, 'sensor_height_m': .024}
    camera = {**CAMERA, 'lens_selection': selection, 'focus_distance_m': .4}
    # A mismatched client radius must not override the same authoritative
    # analytical lens selection used by the native photo request validator.
    monkeypatch.setattr(housing, 'camera_mount_source', lambda camera, root: {'triangle_count': 1, 'camera': camera})
    record = json.loads(housing.housing_response(json.dumps({'camera': camera}).encode(), SPECTRAL))
    assert record['camera']['aperture_radius_m'] == .050 / 16


def test_real_housing_http_transport_and_guards(tmp_path, monkeypatch):
    if not (SPECTRAL / 'camera_designer/camera_frame.py').exists():
        pytest.skip('Canonical native camera frame unavailable')
    web = tmp_path / 'build' / 'meowdoku' / 'web'
    web.mkdir(parents=True)
    (web / 'boards.json').write_text(json.dumps({'boards': [], 'levels': {}}))
    sessions = server.Sessions(web, None)
    sessions.access.lookup = lambda url: None
    sessions.marble_photos.spectral_root = SPECTRAL
    monkeypatch.setattr(sessions.marble_photos, 'submit', lambda *a, **kw: pytest.fail('housing started an exposure'), raising=False)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    def request(body, extra=None):
        connection = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=20)
        connection.request('POST', '/api/marble_camera/housing', body,
                           {'Content-Type': 'application/json', **(extra or {})})
        response = connection.getresponse()
        data = response.read()
        status = response.status
        assert response.getheader('Cache-Control') == 'no-store'
        connection.close()
        return status, data
    try:
        wire = json.dumps({'camera': CAMERA}).encode()
        status, data = request(wire)
        assert status == 200
        record = json.loads(data)
        assert record['schema'] == 'native-camera-housing-mount.v1'
        assert record['triangle_count'] == len(record['housingVerticesM']) // 3 > 1000
        assert record['capabilities'] == {'mountEnvelope': True, 'renderMesh': False,
            'reason': 'Native face/material ownership is not exported by this mount endpoint'}
        assert record['camera']['eye_m'] == CAMERA['eye_m']
        assert request(wire)[1] == data
        assert request(b'x' * 4097)[0] == 413
        assert request(b'{')[0] == 400
        sessions.access.config()['policy']['lan'] = 'play'
        assert request(wire)[0] == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
