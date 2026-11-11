"""Account-owned cursors over declared graph edges; never a physics runtime."""
from copy import deepcopy
import hashlib
import json
import secrets
import threading
import time

from meowdoku.living_data_query import GraphView, QueryError, _integer, _text

MAX_SESSIONS = 16
TTL_SECONDS = 3600
MAX_RECEIPTS = 64


class CursorError(QueryError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _fields(body, allowed):
    if not isinstance(body, dict) or set(body) - allowed:
        raise CursorError('unexpected request fields')


def _request(body):
    request_id = _text(body.get('request_id'), 'request_id')
    if len(request_id) > 128:
        raise CursorError('request_id is limited to 128 characters')
    try:
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError, OverflowError) as error:
        raise CursorError('request must be finite JSON data') from error
    return request_id, digest


class CursorStore:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.sessions = {}

    def _prune(self):
        now = self.clock()
        for identity in [k for k, v in self.sessions.items() if now - v['touched'] >= TTL_SECONDS]:
            del self.sessions[identity]

    def _owned(self, owner, identity):
        self._prune()
        item = self.sessions.get(identity)
        if item is None or item['owner'] != owner:
            raise CursorError('graph session not found', 404)
        return item

    def _status(self, item, offset=0):
        _integer(offset, 0, 32768, 'offset')
        graph, at = item['graph'], item['at']
        choices = []
        for edge in graph.adj.get(at, []):
            forward = edge['source'] == at
            target = edge['target'] if forward else edge['source']
            # References outside the supplied graph can be narrated, not entered.
            if target not in graph.nodes:
                continue
            choices.append({'to': target, 'relationship': edge['relationship'],
                            'direction': 'outgoing' if forward else 'incoming'})
        page = choices[offset:offset + 64]
        node = graph.node(at)
        label = node.get('name') or at
        directions = '; '.join(f"{c['direction']} {c['relationship']} to {c['to']}" for c in page)
        return {'schema': 'living-data-cursor.v1', 'mode': 'graph-cursor',
                'session_id': item['id'], 'revision': item['revision'], 'at': node,
                'layer': graph.layer, 'exits': page, 'exit_total': len(choices),
                'next_exit_offset': offset + 64 if offset + 64 < len(choices) else None,
                'text': f"You are at {label} ({at}). " +
                        (f"Graph exits: {directions}." if page else 'No graph exits on this page.'),
                'semantics': 'Moves change only this graph cursor. No simulation time or physical state changes.',
                'simulation_advanced': False, 'frame': None}

    def create(self, owner, body):
        _fields(body, {'map', 'at', 'layer', 'request_id'})
        request_id, digest = _request(body)
        with self.lock:
            self._prune()
            for item in self.sessions.values():
                if item['owner'] == owner:
                    if item['create_request'] == request_id and item['create_digest'] == digest:
                        return dict(deepcopy(item['created_receipt']), replayed=True)
                    if item['create_request'] == request_id:
                        raise CursorError('request_id already used with different content', 409)
                    raise CursorError('close your current graph session before creating another', 409)
            if len(self.sessions) >= MAX_SESSIONS:
                raise CursorError('graph session capacity reached', 503)
            graph = GraphView(body.get('map'), body.get('layer', 'layout'))
            at = _text(body.get('at'), 'at')
            if at not in graph.nodes:
                raise CursorError('at must name a known graph node')
            for identity in graph.nodes:
                graph.node(identity)  # Validate every summary before admitting state.
            # Retain only graph summary columns, never uploaded programs/binaries.
            keys = ('identity', 'kind', 'name', 'member_kind', 'type_name', 'position')
            graph.nodes = {k: deepcopy({f: n[f] for f in keys if f in n}) for k, n in graph.nodes.items()}
            item = {'id': secrets.token_hex(16), 'owner': owner, 'graph': graph, 'at': at,
                    'revision': 0, 'touched': self.clock(), 'receipts': {},
                    'create_request': request_id, 'create_digest': digest}
            receipt = self._status(item)
            item['created_receipt'] = deepcopy(receipt)
            self.sessions[item['id']] = item
            return receipt

    def status(self, owner, identity, offset=0):
        with self.lock:
            item = self._owned(owner, identity)
            result = self._status(item, offset)
            item['touched'] = self.clock()
            return result

    def move(self, owner, identity, body):
        _fields(body, {'to', 'relationship', 'direction', 'expected_revision', 'request_id'})
        request_id, digest = _request(body)
        target = _text(body.get('to'), 'to')
        relationship = _text(body.get('relationship'), 'relationship')
        direction = body.get('direction', 'outgoing')
        if direction not in ('outgoing', 'incoming'):
            raise CursorError('direction must be outgoing or incoming')
        revision = _integer(body.get('expected_revision'), 0, 2**53 - 1, 'expected_revision')
        with self.lock:
            item = self._owned(owner, identity)
            prior = item['receipts'].get(request_id)
            if prior:
                if prior[0] != digest:
                    raise CursorError('request_id already used with different content', 409)
                return dict(deepcopy(prior[1]), replayed=True)
            if revision != item['revision']:
                raise CursorError('stale revision; read the current graph session', 409)
            graph, at = item['graph'], item['at']
            source, destination = (at, target) if direction == 'outgoing' else (target, at)
            if target not in graph.nodes or not any(
                e['source'] == source and e['target'] == destination and e['relationship'] == relationship
                for e in graph.adj.get(at, [])
            ):
                raise CursorError('no such declared graph exit', 409)
            item['at'] = target
            item['revision'] += 1
            item['touched'] = self.clock()
            result = self._status(item)
            item['receipts'][request_id] = (digest, deepcopy(result))
            while len(item['receipts']) > MAX_RECEIPTS:
                del item['receipts'][next(iter(item['receipts']))]
            return result

    def close(self, owner, identity, body):
        _fields(body, {'expected_revision'})
        revision = _integer(body.get('expected_revision'), 0, 2**53 - 1, 'expected_revision')
        with self.lock:
            item = self._owned(owner, identity)
            if revision != item['revision']:
                raise CursorError('stale revision; read the current graph session', 409)
            del self.sessions[identity]
            return {'closed': True, 'session_id': identity}
