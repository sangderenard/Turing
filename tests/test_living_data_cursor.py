"""Real isolated HTTP calls: graph traversal, never a physical world simulation."""
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import pytest

from meowdoku.living_data_cursor import CursorStore, CursorError
from tests.test_living_data_query import model
from tests.test_meowdoku_profile_dashboard import server, _cookie, _call, OUTSIDE

BASE = '/api/living-data/sessions'


def karate_map():
    # Same canonical source as the existing repository demo. Never its fallback.
    from src.demos import filtered_poisson_karate_demo as demo
    assert demo._HAVE_NX, 'The actual Karate Club fixture requires NetworkX'
    graph = demo.nx.karate_club_graph()
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (34, 78)
    nodes = [{'identity': f'karate:{n}', 'kind': 'region',
              'name': f'Member {n} ({data["club"]})', 'buildings': []}
             for n, data in graph.nodes(data=True)]
    tracks = [{'source': f'karate:{a}', 'target': f'karate:{b}', 'relationship': 'friendship'}
              for a, b in graph.edges()]
    return graph, {'schema': 'abstract-ui-div-map-v0', 'identity': 'karate',
                   'kind': 'world', 'name': "Zachary's Karate Club", 'regions': nodes, 'tracks': tracks}


def test_karate_club_real_rest_walk(server):
    graph, m = karate_map()
    h = _cookie(server, 'karatewalker', 'operator')
    create = {'map': m, 'at': 'karate:0', 'request_id': 'create-karate'}
    status, _, state = _call(server.port, BASE, 'POST', h, create)
    assert status == 201 and state['revision'] == 0
    sid = state['session_id']; url = BASE + '/' + sid
    assert len(state['exits']) == graph.degree(0)
    assert {int(e['to'].split(':')[1]) for e in state['exits']} == set(graph.neighbors(0))
    assert 'Member 0' in state['text']
    transcript = [{'method': 'POST', 'path': BASE, 'status': status, 'at': state['at']['identity'], 'revision': state['revision']}]
    bad = {'to': 'karate:33', 'relationship': 'friendship', 'expected_revision': 0, 'request_id': 'not-an-edge'}
    assert _call(server.port, url + '/move', 'POST', h, bad)[0] == 409
    for revision, target in enumerate([8, 33, 8, 0]):
        exit = next(e for e in state['exits'] if e['to'] == f'karate:{target}')
        move = {**exit, 'expected_revision': revision, 'request_id': f'move-{revision}'}
        status, _, state = _call(server.port, url + '/move', 'POST', h, move)
        assert status == 200 and state['revision'] == revision + 1
        assert state['at']['identity'] == f'karate:{target}'
        assert state['simulation_advanced'] is False and state['frame'] is None
        transcript.append({'method': 'POST', 'path': '/sessions/{id}/move', 'status': status,
                           'at': state['at']['identity'], 'revision': state['revision']})
        replay_status, _, replay = _call(server.port, url + '/move', 'POST', h, move)
        assert replay_status == 200 and replay['replayed'] and replay['revision'] == state['revision']
        assert len(state['exits']) == graph.degree(target)
        assert {int(e['to'].split(':')[1]) for e in state['exits']} == set(graph.neighbors(target))
    assert _call(server.port, url, headers=h)[2]['revision'] == 4
    assert _call(server.port, BASE, 'POST', h, create)[2]['replayed'] is True
    # No rest call creates/moves a catalog Table or starts a physics clock.
    assert server.sessions.tables == {}
    print('\nKarate REST receipt:', transcript)
    assert _call(server.port, url + '/close', 'POST', h, {'expected_revision': 4})[0] == 200
    assert _call(server.port, url, headers=h)[0] == 404


def test_http_ownership_admission_and_conflict(server):
    payload = {'map': model(), 'at': 'r', 'request_id': 'new'}
    assert _call(server.port, BASE, 'POST', OUTSIDE, payload)[0] == 403
    assert _call(server.port, BASE, 'POST', {}, payload)[0] == 401
    h = _cookie(server, 'cursorowner', 'operator')
    other = _cookie(server, 'cursorother', 'operator')
    status, _, item = _call(server.port, BASE, 'POST', h, payload)
    assert status == 201
    url = BASE + '/' + item['session_id']
    assert _call(server.port, url, headers=other)[0] == 404
    assert _call(server.port, url + '?offset=-1', headers=h)[0] == 400
    assert _call(server.port, url + '?offset=0&offset=1', headers=h)[0] == 400
    movement = {'to': 'b', 'relationship': 'contained-by', 'direction': 'incoming',
                'expected_revision': 0, 'request_id': 'move'}
    assert _call(server.port, url + '/move', 'POST', other, movement)[0] == 404
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_call, server.port, url + '/move', 'POST', h,
                                    {**movement, 'request_id': f'race-{i}'}) for i in range(2)]
    assert sorted(f.result()[0] for f in futures) == [200, 409]
    assert _call(server.port, url, headers=h)[2]['revision'] == 1
    assert _call(server.port, url + '/close', 'POST', h, {'expected_revision': 0})[0] == 409


def test_store_immutable_input_retries_expiry_and_capacity(monkeypatch):
    from meowdoku import living_data_cursor as mod
    now = [0.0]; store = CursorStore(clock=lambda: now[0])
    m = model(); payload = {'map': m, 'at': 'r', 'request_id': 'new'}
    item = store.create(1, payload); sid = item['session_id']
    m['regions'][0]['name'] = 'mutated externally'
    assert store.status(1, sid)['at']['name'] == 'R'
    with pytest.raises(CursorError): store.create(1, payload)
    monkeypatch.setattr(mod, 'MAX_SESSIONS', 1)
    with pytest.raises(CursorError) as error: store.create(2, payload)
    assert error.value.status == 503
    movement = {'to': 'b', 'relationship': 'contained-by', 'direction': 'incoming',
                'expected_revision': 0, 'request_id': 'm'}
    store.move(1, sid, movement)
    with pytest.raises(CursorError) as error: store.move(1, sid, {**movement, 'to': 'w'})
    assert error.value.status == 409
    now[0] = 3601
    with pytest.raises(CursorError) as error: store.status(1, sid)
    assert error.value.status == 404
    assert store.create(2, payload)['revision'] == 0
