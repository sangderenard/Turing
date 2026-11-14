"""Static DM transport tests. Synthetic PNG is a validation fixture, not a render claim."""
import base64
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

from PIL import Image
import pytest
from meowdoku.living_data_static import StaticWorlds
from meowdoku.living_data_cursor import CursorError
from tests.test_living_data_query import model
from tests.test_meowdoku_profile_dashboard import server, _cookie, _call, OUTSIDE

BASE = '/api/living-data/dm/worlds'


def registration():
    return {'map': model(), 'source_sha256': 'a' * 64,
            'renderer_identity': 'test-transport-only', 'request_id': 'attach'}


def render_command():
    return {'request_id': 'frame', 'operation': 'render-phong', 'position': [1, 2, 3],
            'facing': [0, 0, -1], 'width': 64, 'height': 64}


def result():
    buf = io.BytesIO(); Image.new('RGB', (64, 64)).save(buf, format='PNG')
    return {'source_sha256': 'a' * 64, 'renderer_identity': 'test-transport-only',
            'position': [1, 2, 3], 'facing': [0, 0, -1], 'width': 64, 'height': 64,
            'png_base64': base64.b64encode(buf.getvalue()).decode(),
            'actual_camera': {'position': [1, 2, 3], 'facing': [0, 0, -1], 'width': 64, 'height': 64}}


def test_http_static_inspection_render_transport_and_ownership(server):
    h = _cookie(server, 'staticdm', 'operator')
    other = _cookie(server, 'staticother', 'operator')
    assert _call(server.port, BASE, 'POST', OUTSIDE, registration())[0] == 403
    assert _call(server.port, BASE, 'POST', {}, registration())[0] == 403          # a guest never holds dev_ui (owner, 2026-11-13)
    status, _, world = _call(server.port, BASE, 'POST', h, registration())
    assert status == 201
    url = BASE + '/' + world['world_id']
    assert 'host_token' not in _call(server.port, url, headers=h)[2]
    assert _call(server.port, url, headers=other)[0] == 404
    inspect = {'operation': 'inspect', 'request_id': 'look', 'at': 'r'}
    status, _, receipt = _call(server.port, url + '/commands', 'POST', h, inspect)
    assert status == 200 and receipt['result']['object']['kind'] == 'courtyard'
    assert 'physical' in receipt['result']['nodes'][0]['identity']
    assert _call(server.port, url + '/commands', 'POST', h, {'operation': 'teleport', 'request_id': 'no'})[0] == 400
    status, _, queued = _call(server.port, url + '/commands', 'POST', h, render_command())
    assert status == 202 and queued['state'] == 'queued'
    assert _call(server.port, url + '/host', 'POST', h, {'host_token': '\u2603', 'action': 'poll'})[0] == 403
    poll = {'host_token': world['host_token'], 'action': 'poll'}
    assert _call(server.port, url + '/host', 'POST', other, poll)[0] == 404
    status, _, job = _call(server.port, url + '/host', 'POST', h, poll)
    assert status == 200 and job['request_id'] == 'frame'
    complete = {'host_token': world['host_token'], 'action': 'complete', 'request_id': 'frame', 'result': result()}
    bad = {**complete, 'result': {**result(), 'source_sha256': 'b' * 64}}
    assert _call(server.port, url + '/host', 'POST', h, bad)[0] == 409
    assert _call(server.port, url + '/host', 'POST', h, complete)[0] == 200
    status, _, final = _call(server.port, url + '/commands/frame', headers=h)
    assert status == 200 and final['state'] == 'complete' and final['result']['png_sha256']
    assert final['result']['simulation'] is False
    assert _call(server.port, url + '/commands', 'POST', h, render_command())[2]['replayed'] is True
    assert _call(server.port, url + '/close', 'POST', other, {})[0] == 404
    assert _call(server.port, url + '/close', 'POST', h, {})[0] == 200
    assert _call(server.port, url, headers=h)[0] == 404
    assert server.sessions.tables == {}


def test_timeout_bounds_and_invalid_pixels(monkeypatch):
    from meowdoku import living_data_static as mod
    now = [0.0]; service = StaticWorlds(clock=lambda: now[0])
    world = service.register(1, registration()); wid = world['world_id']; token = world['host_token']
    for position in ([float('nan'), 0, 0], [10**400, 0, 0], [True, 0, 0]):
        with pytest.raises(CursorError): service.command(1, wid, {**render_command(), 'position': position})
    service.command(1, wid, render_command())
    with pytest.raises(CursorError): service.command(1, wid, {**render_command(), 'request_id': 'second'})
    service.host(1, wid, {'action': 'poll', 'host_token': token})
    bad = result(); bad['png_base64'] = base64.b64encode(base64.b64decode(bad['png_base64'])[:33]).decode()
    with pytest.raises(CursorError):
        service.host(1, wid, {'action': 'complete', 'host_token': token, 'request_id': 'frame', 'result': bad})
    now[0] = 61
    assert service.receipt(1, wid, 'frame')['state'] == 'failed'
    monkeypatch.setattr(mod, 'MAX_COMMANDS', 1)
    with pytest.raises(CursorError) as error:
        service.command(1, wid, {'operation': 'inspect', 'request_id': 'look', 'at': 'r'})
    assert error.value.status == 429
    now[0] = 901
    with pytest.raises(CursorError) as error: service.status(1, wid)
    assert error.value.status == 404


def test_valid_png_chunks_with_corrupt_pixel_data_rejected():
    import struct, zlib
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    broken = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
              + chunk(b'IDAT', b'not-zlib') + chunk(b'IEND', b''))
    service = StaticWorlds(); world = service.register(1, registration()); wid = world['world_id']
    service.command(1, wid, render_command())
    service.host(1, wid, {'action': 'poll', 'host_token': world['host_token']})
    with pytest.raises(CursorError):
        service.host(1, wid, {'action': 'complete', 'host_token': world['host_token'], 'request_id': 'frame',
                             'result': {**result(), 'png_base64': base64.b64encode(broken).decode()}})
    assert service.receipt(1, wid, 'frame')['state'] == 'rendering'


def test_animation_and_wrong_actual_camera_rejected():
    service = StaticWorlds(); world = service.register(1, registration()); wid = world['world_id']
    service.command(1, wid, render_command())
    service.host(1, wid, {'action': 'poll', 'host_token': world['host_token']})
    envelope = {'action': 'complete', 'host_token': world['host_token'], 'request_id': 'frame'}
    wrong = result(); wrong['actual_camera']['facing'] = [0, 0, 1]
    with pytest.raises(CursorError): service.host(1, wid, {**envelope, 'result': wrong})
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG', save_all=True,
                                        append_images=[Image.new('RGB', (64, 64), 'blue')], duration=100)
    animated = {**result(), 'png_base64': base64.b64encode(buffer.getvalue()).decode()}
    with pytest.raises(CursorError): service.host(1, wid, {**envelope, 'result': animated})


def test_static_layout_keeps_authored_story_descriptions():
    payload = registration()
    payload['map']['regions'][0]['buildings'][0]['rooms'] = [{
        'identity': 'story-room', 'kind': 'room', 'name': 'copper_kettle',
        'metaphor': 'illuminated page', 'parameters': [],
        'dependencies': [{'relationship': 'contained-by', 'target': 'b'}],
        'implied_code': [{'operation': 'describe', 'source': 'A copper kettle hums.',
                         'executable': False, 'dialect': 'prose', 'explanation': 'A copper kettle hums.'}]}]
    service = StaticWorlds(); world = service.register(1, payload)
    receipt = service.command(1, world['world_id'], {'request_id': 'story', 'operation': 'inspect',
                                                   'layer': 'layout', 'at': 'story-room'})
    assert receipt['result']['object']['implied_code'][0]['source'] == 'A copper kettle hums.'
    assert receipt['result']['object']['metaphor'] == 'illuminated page'


def test_optical_request_is_carried_and_requires_matching_result():
    service = StaticWorlds()
    world = service.register(1, registration()); wid = world['world_id']
    command = {**render_command(), 'optics': {'autofocus': True, 'f_stop': 4}}
    service.command(1, wid, command)
    work = service.host(1, wid, {'action': 'poll', 'host_token': world['host_token']})
    assert work['camera']['optics'] == {'autofocus': True, 'f_stop': 4, 'sensor_height_m': .024, 'focus_distance_m': 3}
    payload = result()
    complete = {'action': 'complete', 'host_token': world['host_token'], 'request_id': 'frame', 'result': payload}
    with pytest.raises(CursorError): service.host(1, wid, complete)
    payload['actual_camera']['optics'] = {**work['camera']['optics'], 'focus_distance_m': 2,
        'measured_rays': 9, 'coverage': .7, 'contrast': 20, 'fov_degrees': 2 * __import__('math').atan(.70) * 180 / __import__('math').pi,
        'method': 'studio-gaussian-rangefinder-phong-contrast', 'depth_mode': 'opaque'}
    assert service.host(1, wid, complete)['state'] == 'complete'
    with pytest.raises(CursorError): service.command(1, wid, {**command, 'optics': {'f_stop': 8}})


@pytest.mark.parametrize('optics', [{'f_stop': 0}, {'autofocus': 'yes'}, {'focus_distance_m': -1}, {'sensor_height_m': float('inf')}, {'unknown': True}])
def test_optical_bad_request_is_refused(optics):
    service = StaticWorlds(); world = service.register(1, registration())
    with pytest.raises(CursorError): service.command(1, world['world_id'], {**render_command(), 'optics': optics})


def foyer_registration():
    return {**registration(), 'request_id': 'foyer', 'foyer': {'start_position': [1, 2, 3], 'start_facing': [0, 0, -1]}}


def test_foyer_one_shared_world_queues_one_picture_per_visitor():
    service = StaticWorlds()
    with pytest.raises(CursorError) as closed:
        service.foyer_status()
    assert closed.value.status == 503
    world = service.register(1, foyer_registration())
    with pytest.raises(CursorError):                                   # one foyer at a time
        service.register(2, {**foyer_registration(), 'request_id': 'other'})
    status = service.foyer_status()
    assert status['demo'] and status['start'] == {'position': [1, 2, 3], 'facing': [0, 0, -1]} and 'host_token' not in status
    asked = service.foyer_command('browser:a', render_command())
    assert asked['state'] == 'queued' and asked['request_id'] == 'frame' and 'world_id' not in asked
    with pytest.raises(CursorError) as busy:                           # one picture at a time per visitor
        service.foyer_command('browser:a', {**render_command(), 'request_id': 'frame2'})
    assert busy.value.status == 409
    assert service.foyer_command('browser:b', render_command())['state'] == 'queued'   # same id, another visitor
    with pytest.raises(CursorError):                                   # a visitor reads only its own receipts
        service.foyer_receipt('browser:c', 'frame')
    from meowdoku.living_data_query import QueryError
    with pytest.raises(QueryError):                                    # 64..640 px a side; the route answers 400
        service.foyer_command('browser:c', {**render_command(), 'request_id': 'big', 'width': 1024})
    with pytest.raises(CursorError):
        service.foyer_command('browser:c', {'request_id': 'scene', 'operation': 'export-scene'})
    token, wid = world['host_token'], world['world_id']
    work = service.host(1, wid, {'action': 'poll', 'host_token': token})
    assert work['request_id'] not in ('frame', None)                   # the host sees an internal id, never a visitor's
    service.host(1, wid, {'action': 'complete', 'host_token': token, 'request_id': work['request_id'], 'result': result()})
    done = service.foyer_receipt('browser:a', 'frame')
    assert done['state'] == 'complete' and done['result']['png_sha256'] and done['demo']
    assert service.foyer_receipt('browser:b', 'frame')['state'] == 'queued'
    for n in range(7):                                                  # eight renders queued or in flight across visitors
        service.foyer_command(f'browser:q{n}', {**render_command(), 'request_id': f'q{n}'})
    with pytest.raises(CursorError) as full:
        service.foyer_command('browser:late', {**render_command(), 'request_id': 'late'})
    assert full.value.status == 429
    looked = service.foyer_command('browser:late', {'request_id': 'look', 'operation': 'inspect'})
    assert looked['state'] == 'complete' and looked['result']['object']


def test_http_foyer_needs_library_manage_to_host_and_nothing_to_visit(server):
    h = _cookie(server, 'foyerhost', 'operator')                      # a LAN account: the edit tier gives it library_manage
    remote = _cookie(server, 'remotehost', 'operator', OUTSIDE)
    assert _call(server.port, '/api/foyer')[0] == 503
    assert _call(server.port, BASE, 'POST', remote, foyer_registration())[0] == 403    # no library_manage from outside
    status, _, world = _call(server.port, BASE, 'POST', h, foyer_registration())
    assert status == 201
    for headers in ({}, OUTSIDE):                                       # a visitor with no account, on the LAN and outside
        status, _, greeting = _call(server.port, '/api/foyer', 'GET', headers)
        assert status == 200 and greeting['demo'] and 'man foyer' in greeting['greeting']
    status, cookies, asked = _call(server.port, '/api/foyer/commands', 'POST', OUTSIDE, render_command())
    assert status == 202 and asked['state'] == 'queued'
    # with no account the visitor is its browser session: the cookie the service set, as a browser or the MCP keeps it
    visitor = {**OUTSIDE, 'Cookie': '; '.join(c.split(';')[0] for c in cookies)}
    assert _call(server.port, '/api/foyer/commands/frame', 'GET', OUTSIDE)[0] == 404     # another session sees nothing
    url = BASE + '/' + world['world_id'] + '/host'
    status, _, work = _call(server.port, url, 'POST', h, {'host_token': world['host_token'], 'action': 'poll'})
    assert status == 200 and work['request_id'] != 'frame'
    complete = {'host_token': world['host_token'], 'action': 'complete', 'request_id': work['request_id'], 'result': result()}
    assert _call(server.port, url, 'POST', h, complete)[0] == 200
    status, _, receipt = _call(server.port, '/api/foyer/commands/frame', 'GET', visitor)
    assert status == 200 and receipt['state'] == 'complete' and base64.b64decode(receipt['result']['png_base64'])[:4] == b'\x89PNG'
