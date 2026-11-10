"""Owner isolation, immutable source binding and existing photo-path integration."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'examples')]
from meowdoku import obj_assets, marble_objects, marble_photo
from meowdoku.accounts import Accounts
from tests.test_meowdoku_marble_photo import _saved_object, _wire

ROOT = Path(__file__).resolve().parents[1]
SPECTRAL = ROOT.parent / 'spectral-analyzer'
OBJ = 'v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n'

@pytest.fixture
def library(tmp_path):
    accounts = Accounts(tmp_path / 'accounts.db')
    alice = accounts.signup('alice', 'test-password-alice')['id']
    bob = accounts.signup('bobby', 'test-password-bobby')['id']
    return obj_assets.Store(tmp_path / 'assets', accounts, SPECTRAL), alice, bob


def snapshot(store, owner):
    digest = store.put({'name': 'Triangle', 'obj': OBJ}, owner)['digest']
    asset = store.get(digest, owner)
    packet = obj_assets.subject_packet(asset, 0)
    value = _saved_object(packet['positions'], packet['normals'], packet['id'])
    value['generator'] = 'meowdoku-imported-object-v1'
    value['design'] = {'base-shape': 'obj', 'kind': 'clear', 'source_asset': digest,
                       'material_mode': 'studio-shell', 'shell': 'glass'}
    value['derived'] = {'massKg': None, 'massExact': False, 'validForRoll': False,
                        'centerM': None, 'inertiaKgM2': None, 'components': [],
                        'principalMoments': [], 'mechanicsNote': 'Imported mesh mechanics have not been certified'}
    return value, packet


def test_private_immutable_and_public_access(library):
    store, alice, bob = library
    value = {'name': 'Triangle', 'obj': OBJ}
    digest = store.put(value, alice)['digest']
    assert store.put(value, alice)['duplicate']
    assert store.entries() == [] and store.entries(bob) == []
    for user in (None, bob):
        with pytest.raises(obj_assets.AssetError) as exc:
            store.source(digest, user)
        assert exc.value.status == 404
    other = store.put(value, bob)['digest']
    assert other != digest
    assert store.source(digest, alice) == OBJ.encode()
    with pytest.raises(obj_assets.AssetError):
        store.publish(digest, bob, attribution='credit')
    store.publish(digest, alice, attribution='Test triangle by Alice')
    assert store.get(digest)['readOnly'] and not store.get(digest)['owned']
    assert store.source(digest) == OBJ.encode()


def test_imported_snapshots_and_photo_bind_source(library, tmp_path):
    assets, alice, bob = library
    value, packet = snapshot(assets, alice)
    objects = marble_objects.Store(tmp_path / 'objects', assets).for_account(alice)
    digest = objects.put(value)['digest']
    assert objects.entries()[0]['massKg'] is None
    assert objects.for_account(bob).entries() == []
    with pytest.raises(marble_objects.ObjectError):
        objects.for_account(bob).get(digest)
    wire = _wire(None, packet['positions'], packet['normals'], packet['id'], object_snapshot=value)
    assert marble_photo.decode_upload(wire, objects)[3]['exposure_snapshot'] == value
    with pytest.raises(marble_photo.PhotoError):
        marble_photo.decode_upload(wire, objects.for_account(bob))
    bad = json.loads(json.dumps(value)); bad['derived']['massKg'] = 1
    with pytest.raises(marble_objects.ObjectError, match='certified'):
        objects.put(bad)
    changed = packet.copy(); changed['positions'][0, 0] += .1
    bad = json.loads(json.dumps(value)); bad['geometry_sha256'] = hashlib.sha256(changed.tobytes()).hexdigest()
    with pytest.raises(marble_objects.ObjectError, match='source asset'):
        objects.put(bad)
    with pytest.raises(marble_photo.PhotoError, match='source asset'):
        marble_photo.decode_upload(_wire(None, changed['positions'], changed['normals'], changed['id'], object_snapshot=bad), objects)


@pytest.mark.parametrize('body', [ {'name':'bad','obj':OBJ,'visibility':'public'}, {'name':'bad','obj':'mtllib ../../secret\n'+OBJ}, {'name':'bad','obj':'v 0 0 0\nv 1 0 0\nf 1 2 3'}, {'name':'bad','obj':'v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3'} ])
def test_invalid_or_external_sources_refused(library, body):
    store, alice, _ = library
    with pytest.raises(obj_assets.AssetError): store.put(body, alice)
    assert store.entries(alice) == []


def test_javascript_packet_matches_python(library, tmp_path):
    store, alice, _ = library
    value, packet = snapshot(store, alice)
    asset = store.get(value['design']['source_asset'], alice)
    path = tmp_path / 'asset.json'; path.write_text(json.dumps(asset))
    script = """
import fs from 'node:fs';
import {registerObjAsset,importedObjectGeometry} from './examples/meowdoku/web/marble_imported_object.js';
import {geometrySha256} from './examples/meowdoku/web/marble_object.js';
const asset=registerObjAsset(JSON.parse(fs.readFileSync(process.argv[1],'utf8')));
console.log(await geometrySha256(importedObjectGeometry({source_asset:asset.digest,material_mode:'studio-shell'},0)));
"""
    result = subprocess.run(['node', '--input-type=module', '-e', script, str(path)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15)
    assert result.stdout.strip() == hashlib.sha256(packet.tobytes()).hexdigest()

def test_http_library_isolation_and_bounded_upload(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    from meowdoku import server
    web = tmp_path / 'web'; web.mkdir()
    (web / 'boards.json').write_text('{"boards":[],"levels":{}}')
    sessions = server.Sessions(web, None)
    alice = sessions.accounts.signup('alice', 'test-password-alice')['id']
    bob = sessions.accounts.signup('bobby', 'test-password-bobby')['id']
    tokens = {user: sessions.accounts.create_session(user) for user in (alice, bob)}
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    def request(method, path, user=None, body=None, extra=None):
        headers = {'Content-Type': 'application/json'}
        if user is not None: headers['Cookie'] = server.ACCOUNT_COOKIE + '=' + tokens[user]
        headers.update(extra or {})
        connection = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=5)
        connection.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        response = connection.getresponse(); raw = response.read(); status = response.status
        connection.close()
        return status, raw
    try:
        assert request('POST', '/api/obj_assets', body={'name':'T','obj':OBJ})[0] == 401
        status, raw = request('POST', '/api/obj_assets', alice, {'name':'T','obj':OBJ})
        assert status == 201, raw
        digest = json.loads(raw)['digest']
        for user in (None, bob):
            assert json.loads(request('GET', '/api/obj_assets', user)[1]) == {'entries':[]}
            for suffix in ('', '/source', '/mtl'):
                assert request('GET', '/api/obj_assets/' + digest + suffix, user)[0] == 404
        assert request('GET', '/api/obj_assets/' + digest + '/source', alice) == (200, OBJ.encode())
        assert json.loads(request('GET', '/api/account/assets', alice)[1])['obj_assets'][0]['owned']
        sessions.obj_assets.publish(digest, alice, attribution='Alice test fixture')
        assert json.loads(request('GET', '/api/obj_assets')[1])['entries'][0]['readOnly']
        assert request('GET', '/api/obj_assets/' + digest + '/source') == (200, OBJ.encode())
        assert json.loads(request('GET', '/api/obj_assets/' + digest)[1])['geometry']['vertices']
        assert request('POST', '/api/obj_assets', body={'name':'Guest', 'obj':OBJ})[0] == 401
        assert request('POST', '/api/obj_assets', alice, {'name':'T','obj':OBJ}, {'Content-Length': str(obj_assets.MAX_BYTES+1)})[0] == 413
    finally:
        httpd.shutdown(); httpd.server_close(); sessions.marble_photos._sweep_stop.set()

def test_guest_public_source_and_photo_admission_without_asset_write(library, tmp_path):
    assets, alice, _ = library
    value, packet = snapshot(assets, alice)
    source = value['design']['source_asset']
    objects = marble_objects.Store(tmp_path / 'objects', assets)
    owner = objects.for_account(alice)
    saved_digest = owner.put(value)['digest']
    guest = objects.for_account(None)
    inline = _wire(None, packet['positions'], packet['normals'], packet['id'], object_snapshot=value)
    retained = _wire(saved_digest, packet['positions'], packet['normals'], packet['id'])
    # Both paths must fail while the real owner's source is private.
    for wire in (inline, retained):
        with pytest.raises(marble_photo.PhotoError) as exc:
            marble_photo.decode_upload(wire, guest)
        assert exc.value.status == 404
    with pytest.raises(marble_objects.ObjectError): guest.validate_source(value)
    assets.publish(source, alice, attribution='Isolated Alice triangle fixture')
    before = {p.name: p.read_bytes() for p in assets.directory.iterdir()}
    assert guest.validate_source(value).tobytes() == packet.tobytes()
    assert guest.get(saved_digest)['design']['source_asset'] == source
    assert guest.entries()[0]['digest'] == saved_digest
    assert assets.source(source, None) == OBJ.encode()
    for wire in (inline, retained):
        positions, ids, _, config, _ = marble_photo.decode_upload(wire, guest)
        np.testing.assert_array_equal(positions.reshape(-1, 9), packet['positions'])
        np.testing.assert_array_equal(ids, packet['id'])
        assert config['geometry_sha256'] == value['geometry_sha256']
    with pytest.raises(obj_assets.AssetError) as exc:
        assets.put({'name':'Unauthorized guest copy', 'obj':OBJ}, None)
    assert exc.value.status == 401
    with pytest.raises(obj_assets.AssetError) as exc:
        assets.publish(source, None, attribution='Guest replacement credit')
    assert exc.value.status == 403
    assert {p.name: p.read_bytes() for p in assets.directory.iterdir()} == before
