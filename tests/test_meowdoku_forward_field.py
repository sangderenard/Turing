"""Opt-in forward-field request, artifact identity, and owner route regression."""
import hashlib
import http.client
import json

import pytest

from tests.test_meowdoku_marble_photo import photo_payload, _wire
from tests.test_meowdoku_access import tiered, _get
from meowdoku import marble_photo as photos, job_library, server

NAME = 'light-field-' + '1' * 32 + '.bin'


def metadata(digest, **changes):
    return dict(schema='studio.forward-field.v1', digest=digest, name=NAME,
                live_revision=0, sequence=1, dimensions=[2, 2, 2],
                bounds_min_m=[-1, -1, -1], bounds_max_m=[1, 1, 1],
                format='rgb-f32le', layout='zyx-interleaved-rgb', color_space='linear-srgb',
                byte_length=96, **changes)


def test_opt_in_preserves_default_config_and_changes_digest(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    def config(**options):
        return photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3', **options), store)[3]
    old, disabled, enabled = config(), config(receive_light_field=False), config(receive_light_field=True)
    assert old == disabled and 'receive_light_field' not in old
    assert enabled.pop('receive_light_field') is True and enabled == old
    canonical = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert canonical(old) != canonical(config(receive_light_field=True))


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_opt_in_is_strict_boolean(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='boolean'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, receive_light_field=value), store)


@pytest.mark.parametrize('options', [{'transport': 'fixed:1'}, {'transport': 'fixed:3', 'orbit_frames': 8},
    {'transport': 'fixed:3', 'tiled': {'tile_size': 1024}}, {'transport': 'fixed:3', 'camera_poses': [{}]}])
def test_opt_in_requires_single_rgb_non_tiled_camera(photo_payload, options):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='receive_light_field'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, receive_light_field=True, **options), store)


@pytest.fixture
def field_job(tmp_path):
    jobs = photos.PhotoJobs(tmp_path, photos.SPECTRAL_ROOT)
    digest = 'a' * 64
    directory = tmp_path / digest
    directory.mkdir()
    config = {'receive_light_field': True}
    status = {'digest': digest, 'state': 'complete', 'light_field': metadata(digest)}
    (directory / NAME).write_bytes(bytes(96))
    def save():
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'status.json').write_text(json.dumps(status))
    save()
    yield jobs, digest, directory, config, status, save
    jobs.close()


def test_exact_artifact_only_and_streaming(field_job):
    jobs, digest, directory, config, status, save = field_job
    with jobs.open_field(digest, NAME) as stream:
        assert stream.read() == bytes(96)
    with pytest.raises(photos.PhotoError, match='path'):
        jobs.open_field(digest, '../' + NAME)
    wrong = 'light-field-' + '2' * 32 + '.bin'
    (directory / wrong).write_bytes(bytes(96))
    with pytest.raises(photos.PhotoError, match='stale'):
        jobs.open_field(digest, wrong)
    config.clear(); save()
    with pytest.raises(photos.PhotoError, match='unavailable'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('key,value', [('digest', 'b' * 64), ('dimensions', [2, 2, True]),
    ('dimensions', [257, 2, 2]), ('byte_length', 95), ('bounds_max_m', [-1, 1, 1]),
    ('bounds_min_m', [float('nan'), -1, -1]), ('sequence', True), ('live_revision', True),
    ('format', 'other'), ('layout', 'xyz'), ('color_space', 'srgb')])
def test_bad_metadata_fails_closed(field_job, key, value):
    jobs, digest, directory, config, status, save = field_job
    status['light_field'][key] = value; save()
    with pytest.raises(photos.PhotoError):
        jobs.open_field(digest, NAME)


def test_live_revision_and_retention_fail_closed(field_job):
    jobs, digest, directory, config, status, save = field_job
    config['live'] = {}; status['state'] = 'live'; jobs.active_digest = digest
    (directory / 'live-control.json').write_text(json.dumps({'revision': 1})); save()
    with pytest.raises(photos.PhotoError, match='stale'):
        jobs.open_field(digest, NAME)
    status['light_field']['live_revision'] = 1; save()
    with jobs.open_field(digest, NAME) as stream:
        assert len(stream.read()) == 96
    status['state'] = 'yielded'; save()
    with pytest.raises(photos.PhotoError, match='unavailable'):
        jobs.open_field(digest, NAME)
    status['state'] = 'live'; save()
    jobs.active_digest = None
    with pytest.raises(photos.PhotoError, match='no longer active'):
        jobs.open_field(digest, NAME)
    jobs.active_digest = digest; status['state'] = 'canceled'; save()
    with pytest.raises(photos.PhotoError, match='unavailable'):
        jobs.open_field(digest, NAME)


def test_truncated_field_rejected(field_job):
    jobs, digest, directory, config, status, save = field_job
    (directory / NAME).write_bytes(b'x')
    with pytest.raises(photos.PhotoError, match='length'):
        jobs.open_field(digest, NAME)


def test_field_http_owner_and_capability(tiered):
    port, set_tier, sessions = tiered
    set_tier('shader')
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', '/api/marble_photo/jobs'); response = conn.getresponse(); response.read()
    cookies = [v.split(';')[0] for k, v in response.getheaders() if k.lower() == 'set-cookie']; conn.close()
    headers = {'Cookie': '; '.join(cookies)}
    token = next(v.split('=', 1)[1] for v in cookies if v.startswith(server.COOKIE + '='))
    owner = 'browser:' + hashlib.sha256(token.encode()).hexdigest()
    for char in 'abc':
        digest = char * 64
        directory = sessions.marble_photos.directory / digest; directory.mkdir(parents=True)
        (directory / 'config.json').write_text(json.dumps({'receive_light_field': True}))
        (directory / 'status.json').write_text(json.dumps({'digest': digest, 'state': 'complete', 'light_field': metadata(digest)}))
        (directory / NAME).write_bytes(bytes(96))
    job_library.claim(sessions.marble_photos.directory, 'a' * 64, [owner])
    job_library.claim(sessions.marble_photos.directory, 'b' * 64, ['account:999'])
    path = '/api/marble_photo/field?name=' + NAME + '&digest='
    assert _get(port, path + 'a' * 64, headers)[0] == 200
    assert _get(port, path + 'a' * 64)[0] == 403
    assert _get(port, path + 'b' * 64, headers)[0] == 403
    assert _get(port, path + 'c' * 64, headers)[0] == 403
    set_tier('edit')  # Even a submitter cannot inherit an unclaimed field.
    assert _get(port, path + 'c' * 64, headers)[0] == 403
    sessions.authz_policy.path.write_text(json.dumps({'tier_caps': {'edit': ['view_site']}}))
    sessions.authz_policy._stamp = object()
    assert _get(port, path + 'a' * 64, headers)[0] == 403


def test_live_field_http_full_bytes_and_dimensions(tiered):
    import struct
    port, set_tier, sessions = tiered
    set_tier('edit')
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', '/api/marble_photo/jobs'); response = conn.getresponse(); response.read()
    cookies = [v.split(';')[0] for k, v in response.getheaders() if k.lower() == 'set-cookie']; conn.close()
    headers = {'Cookie': '; '.join(cookies)}
    token = next(v.split('=', 1)[1] for v in cookies if v.startswith(server.COOKIE + '='))
    owner = 'browser:' + hashlib.sha256(token.encode()).hexdigest()
    digest = 'e' * 64
    directory = sessions.marble_photos.directory / digest; directory.mkdir(parents=True)
    field = metadata(digest)
    field.update(dimensions=[2, 3, 4], byte_length=2 * 3 * 4 * 12, live_revision=7)
    payload = struct.pack('<72f', *(i - 10.5 for i in range(72)))
    (directory / NAME).write_bytes(payload)
    (directory / 'config.json').write_text(json.dumps({'receive_light_field': True, 'live': {}}))
    (directory / 'live-control.json').write_text(json.dumps({'revision': 7}))
    (directory / 'status.json').write_text(json.dumps({'digest': digest, 'state': 'live', 'live_revision': 7, 'light_field': field}))
    sessions.marble_photos.active_digest = digest
    job_library.claim(sessions.marble_photos.directory, digest, [owner])
    assert _get(port, '/api/marble_photo?digest=' + digest, headers)[1]['light_field']['dimensions'] == [2, 3, 4]
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', '/api/marble_photo/field?digest=' + digest + '&name=' + NAME, headers=headers)
    response = conn.getresponse()
    assert response.status == 200
    assert response.getheader('Content-Type') == 'application/octet-stream'
    assert int(response.getheader('Content-Length')) == len(payload)
    assert response.read() == payload
    conn.close()
