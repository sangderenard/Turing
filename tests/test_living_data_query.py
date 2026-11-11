"""No renderer or replacement simulation: canonical graph query and HTTP admission."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku.living_data_query import GraphView, QueryError, query
from tests.test_meowdoku_profile_dashboard import server, _cookie, _call, OUTSIDE


def model():
    return {'schema': 'abstract-ui-div-map-v0', 'identity': 'w', 'kind': 'world',
            'regions': [{'identity': 'r', 'kind': 'region', 'name': 'R',
                         'position': {'column': 0, 'row': 0, 'width': 1, 'height': 1},
                         'dependencies': [{'relationship': 'contained-by', 'target': 'w'}],
                         'buildings': [{'identity': 'b', 'kind': 'building', 'type_name': None,
                                        'dependencies': [{'relationship': 'contained-by', 'target': 'r'}],
                                        'rooms': []}]}],
            'tracks': [{'source': 'b', 'target': 'outside', 'relationship': 'imports'}],
            'world': {'objects': [{'identity': 'r', 'kind': 'courtyard',
                                  'dependencies': [{'relationship': 'contained-by', 'target': 'physical'}]}]}}


def test_existing_layers_keep_same_identity_without_conflating_relations():
    m = model(); before = copy.deepcopy(m)
    layout = query({'map': m, 'at': 'r', 'radius': 2})
    assert layout['at']['kind'] == 'region'
    assert layout['at']['layout_position']['row'] == 0
    assert {x['identity'] for x in layout['nodes']} == {'w', 'b', 'outside'}
    assert next(x for x in layout['nodes'] if x['identity'] == 'outside')['known'] is False
    world = query({'map': m, 'at': 'r', 'layer': 'world'})
    assert world['at']['kind'] == 'courtyard'
    assert world['nodes'][0]['identity'] == 'physical'
    assert world['capabilities'] == {'graph_query': True, 'advance': False, 'render': False}
    assert all(e['traversability'] == 'unknown' for e in layout['edges'])
    assert m == before


def test_horizon_pagination_cycles_and_zero():
    m = model();m['tracks'].append({'source': 'b', 'target': 'r', 'relationship': 'calls'})
    assert query({'map': m, 'at': 'r', 'radius': 0})['nodes'] == []
    page = query({'map': m, 'at': 'r', 'radius': 2, 'limit': 1})
    assert page['next_offset'] == 1
    assert query({'map': m, 'at': 'r', 'radius': 2, 'limit': 1, 'offset': 1})['nodes'] != page['nodes']
    assert query({'map': m, 'at': 'r', 'radius': 2})['total'] == 3


@pytest.mark.parametrize('patch', [{'radius': True}, {'radius': 5}, {'limit': 0}, {'offset': -1},
                                 {'at': 'missing'}, {'layer': 'mixed'}, {'actor': 'x'}, {'limit': float('nan')}])
def test_invalid_query(patch):
    with pytest.raises(QueryError):
        query({'map': model(), 'at': 'r', **patch})


def test_duplicate_and_bad_reference_rejected():
    m = model();m['regions'].append(m['regions'][0])
    with pytest.raises(QueryError): GraphView(m)
    m = model();m['tracks'][0]['target'] = []
    with pytest.raises(QueryError): GraphView(m)


def test_limits(monkeypatch):
    from meowdoku import living_data_query as q
    monkeypatch.setattr(q, 'MAX_NODES', 2)
    with pytest.raises(QueryError): GraphView(model())
    monkeypatch.setattr(q, 'MAX_NODES', 100)
    monkeypatch.setattr(q, 'MAX_EDGES', 1)
    with pytest.raises(QueryError): GraphView(model())


def test_http_query_and_admission(server):
    path = '/api/living-data/query'; payload = {'map': model(), 'at': 'r'}
    assert _call(server.port, path, 'POST', OUTSIDE, payload)[0] == 403
    assert _call(server.port, path, 'POST', {}, payload)[0] == 401
    member = _cookie(server, 'querymember', 'member')
    assert _call(server.port, path, 'POST', {**member, **OUTSIDE}, payload)[0] == 403
    operator = _cookie(server, 'queryoperator', 'operator')
    status, _, body = _call(server.port, path, 'POST', operator, payload)
    assert status == 200 and body['at']['identity'] == 'r'
    assert _call(server.port, path, 'POST', {**operator, 'Content-Type': 'text/plain'}, payload)[0] == 415
    assert _call(server.port, path, 'POST', operator, {**payload, 'radius': 9})[0] == 400


def test_edge_pagination_does_not_lose_parallel_relationships():
    m = model()
    m['tracks'] = [{'source': 'r', 'target': 'b', 'relationship': f'rel-{i}'} for i in range(300)]
    first = query({'map': m, 'at': 'r'})
    second = query({'map': m, 'at': 'r', 'edge_offset': first['next_edge_offset']})
    assert len(first['edges']) + len(second['edges']) == first['edge_total'] == 302
    assert second['next_edge_offset'] is None


def test_edges_between_node_pages_are_not_lost():
    m = model()
    first = query({'map': m, 'at': 'w', 'radius': 2, 'limit': 1})
    assert {n['identity'] for n in first['nodes']} == {'r'}
    assert any(e['source'] == 'b' and e['target'] == 'r' for e in first['edges'])
    second = query({'map': m, 'at': 'w', 'radius': 2, 'limit': 1, 'offset': 1})
    assert first['edges'] == second['edges']


def test_large_integer_layout_coordinate_is_not_float_coerced(server):
    m = model();m['regions'][0]['position']['row'] = 10 ** 400
    assert query({'map': m, 'at': 'r'})['at']['layout_position']['row'] == 10 ** 400
    cookie = _cookie(server, 'largecoordinate', 'operator')
    status, _, result = _call(server.port, '/api/living-data/query', 'POST', cookie, {'map': m, 'at': 'r'})
    assert status == 200 and result['at']['layout_position']['row'] == 10 ** 400
