"""Guest photo GETs are separate from rendering, sharing and private ownership."""
import hashlib
import http.client
import json
import time

import pytest

from tests.test_meowdoku_access import tiered, _get
from meowdoku import authz, job_library, server


def test_read_only_tiers_do_not_gain_submission():
    for tier in ('shader', 'play', 'dev'):
        caps = authz.capabilities(authz.Principal('guest', None, None, 'dns', tier), authz.DEFAULT_POLICY)
        assert 'photo_read' in caps
        assert not {'photo_submit', 'share_live', 'studio_write'} & caps


def test_guest_loading_page_photo_ownership_and_mutations(tiered, monkeypatch):
    port, set_tier, sessions = tiered
    set_tier('shader')
    from meowdoku import marble_lens_contract
    monkeypatch.setattr(marble_lens_contract, 'lens_catalog', lambda root: {'lenses': []})
    assert _get(port, '/api/marble_photo/lenses') == (200, {'lenses': []})
    # Keep the browser cookie from the list request to seed one browser-owned photo.
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    c.request('GET', '/api/marble_photo/jobs')
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())['entries'] == []
    cookies = [v.split(';')[0] for k, v in r.getheaders() if k.lower() == 'set-cookie']
    c.close()
    headers = {'Cookie': '; '.join(cookies)}
    token = next(v.split('=', 1)[1] for v in cookies if v.startswith(server.COOKIE + '='))
    owner = 'browser:' + hashlib.sha256(token.encode()).hexdigest()
    own, other, legacy = (c * 64 for c in 'abc')
    for digest in (own, other, legacy):
        directory = sessions.marble_photos.directory / digest
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
        (directory / 'status.json').write_text(json.dumps({
            'digest': digest, 'state': 'complete', 'image': 'preview-000001.png',
            'hdr': 'exposure-000001.npz', 'animation': 'orbit.avi',
            'animation_expires_at': time.time() + 3600}))
        for name in ('preview-000001.png', 'exposure-000001.npz', 'orbit.avi'):
            (directory / name).write_bytes(b'fixture')
    job_library.claim(sessions.marble_photos.directory, own, [owner])
    job_library.claim(sessions.marble_photos.directory, other, ['account:999'])
    who = _get(port, '/api/whoami', headers)[1]
    assert who['role'] == 'guest' and who['signed_in'] is False and who['shader']
    assert 'photo_read' in who['capabilities'] and 'photo_submit' not in who['capabilities']
    assert _get(port, '/api/marble_photo/jobs', headers)[1]['entries'][0]['digest'] == own
    for suffix in ('', '/image', '/hdr', '/animation'):
        name = 'exposure-000001.npz' if suffix == '/hdr' else 'preview-000001.png'
        path = '/api/marble_photo' + suffix + '?name=' + name + '&digest='
        assert _get(port, path + own, headers)[0] == 200
        assert _get(port, path + own)[0] == 403  # another browser
        assert _get(port, path + other, headers)[0] == 403
        assert _get(port, path + legacy, headers)[0] == 403  # not a public marker
    assert _get(port, '/api/marble_photo/jobs')[1]['entries'] == []
    for path in ('/api/marble_photo', '/api/marble_photo/cancel', '/api/marble_photo/live',
                 '/api/marble_photo/retouch', '/api/marble_photo/film_preview', '/api/loading_anim'):
        assert _get(port, path, headers, method='POST', body=b'{}')[0] == 403
    assert _get(port, '/api/ready', headers)[1].keys() == {'loading_anim'}
    assert 'SHADER-PAGE' in _get(port, '/', headers)[1]
    # A policy can revoke read access even though the routing allow-list admits GET.
    sessions.authz_policy.path.write_text(json.dumps({'tier_caps': {'shader': ['view_site']}}))
    sessions.authz_policy._stamp = object()
    assert _get(port, '/api/marble_photo/jobs', headers)[0] == 403
    assert _get(port, '/api/marble_photo?digest=' + own, headers)[0] == 403
    sessions.marble_photos.close()


def test_legacy_read_requires_existing_privilege(tmp_path):
    assert job_library.allowed(tmp_path, 'a' * 64, ['browser:one'])
    assert not job_library.allowed(tmp_path, 'a' * 64, ['browser:one'], allow_legacy=False)

