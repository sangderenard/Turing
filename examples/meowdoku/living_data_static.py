"""Owner-scoped REST transport for a registered static Living Data/Phong host.

No simulation, renderer, source executor, URL loader or new permission role.
The browser host owns the original renderer; uploaded metadata is not an
independent server attestation of that host's source or output.
"""
from copy import deepcopy
import base64
import hashlib
import hmac
import json
import io
import math
import re
import secrets
import struct
import threading
import time

from meowdoku.living_data_query import GraphView, QueryError, _integer, _text
from meowdoku.living_data_cursor import CursorError, _fields

MAX_WORLDS = 4
MAX_COMMANDS = 32
MAX_PNG = 2 * 1024 * 1024
TTL = 900
RENDER_TIMEOUT = 60
RID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
# [FOYER] the one shared demo world (owner, 2026-11-13): registered by a resident host holding library_manage, walked and
# photographed by anyone in the foyer (guests, guest agents, interns).  Nothing in it is a visitor's: no world of their own,
# no edits, no scene export; a picture is one queued render-phong on the shared host, at most one per visitor.
FOYER_PENDING = 8                  # renders queued or in flight across all visitors
FOYER_RECEIPTS = 64                # visitor receipts kept; the oldest finished ones go first
FOYER_MAX_SIDE = 640               # a visitor's picture is at most 640 px a side
FOYER_GREETING = ("This is a demo: one shared world anyone in the foyer may walk and photograph. Nothing here is yours; "
                  "an admin makes intern accounts (man foyer).")
SHA = re.compile(r'^[0-9a-f]{64}$')


def _digest(body):
    try:
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError, OverflowError) as error:
        raise CursorError('finite JSON data required') from error


def _rid(value):
    if not isinstance(value, str) or not RID.fullmatch(value):
        raise CursorError('request_id must contain 1-64 letters, digits, underscores or hyphens')
    return value


def _vector(value, facing=False):
    if not isinstance(value, list) or len(value) != 3 or any(
        type(x) not in (int, float) or abs(x) > 1e6 or not math.isfinite(x) for x in value
    ):
        raise CursorError('camera vectors require three finite numbers within +/-1000000')
    if facing and math.hypot(*value) < 1e-12:
        raise CursorError('camera facing cannot be zero')
    return list(value)


def _optics(value):
    _fields(value, {'autofocus', 'f_stop', 'sensor_height_m', 'focus_distance_m'})
    result = {'autofocus': value.get('autofocus', True), 'f_stop': value.get('f_stop', 8),
              'sensor_height_m': value.get('sensor_height_m', .024),
              'focus_distance_m': value.get('focus_distance_m', 3)}
    if type(result['autofocus']) is not bool:
        raise CursorError('autofocus must be boolean')
    for key, low, high in [('f_stop', .5, 128), ('sensor_height_m', .001, .3),
                           ('focus_distance_m', .041, 128)]:
        x = result[key]
        if type(x) not in (int, float) or not math.isfinite(x) or not low <= x <= high:
            raise CursorError(f'{key} outside optical bounds')
    return result


def _scene_snapshot(value, world):
    """Admit the complete native draw data, not a marble/OBJ substitution.

    This is an owner-host declaration. It is never executable and is not a
    ray job: native celestial shading has no physical emitter/medium receipt.
    Keep the original stream/tables; RGB-to-material selection is part of the
    native vertex shader and must not silently become a different material map.
    """
    if not isinstance(value, dict):
        raise CursorError('realized scene snapshot required')
    if len(json.dumps(value, allow_nan=False).encode()) > 3 * 1024 * 1024:
        raise CursorError('scene snapshot exceeds 3 MiB admission limit', 413)
    if (value.get('schema') != 'living-data-realized-mesh.v1' or
            value.get('coordinate_space') != 'world' or value.get('up_axis') != 'Y' or
            value.get('stride') != 9 or value.get('world_identity') != world['root'] or
            value.get('source_sha256') != world['source'] or
            value.get('renderer_identity') != world['renderer']):
        raise CursorError('scene source or coordinate contract mismatch', 409)
    if value.get('physics_started') is not False or value.get('simulation_advanced') is not False:
        raise CursorError('static scene cannot contain advancing simulation')
    vertices = value.get('vertices')
    if (not isinstance(vertices, list) or not vertices or len(vertices) % 27 or
            len(vertices) > 200000 * 27 or any(type(x) not in (int, float) or
            not math.isfinite(x) for x in vertices)):
        raise CursorError('finite interleaved native triangles required')
    colors = value.get('material_colors')
    if (not isinstance(colors, list) or not 1 <= len(colors) <= 16 or
            any(not isinstance(c, list) or len(c) != 3 or any(type(x) not in (int, float)
            or not math.isfinite(x) for x in c) for c in colors)):
        raise CursorError('native material palette required')
    tables = value.get('materials')
    if not isinstance(tables, dict):
        raise CursorError('native material tables required')
    for name, stride in [('pbr', 16), ('phong', 8), ('enamel', 8), ('texstack', 16), ('thermal_k', 0)]:
        data = tables.get(name)
        size = len(colors) * stride if stride else 1
        if (not isinstance(data, list) or len(data) != size or any(type(x) not in (int, float)
                or not math.isfinite(x) for x in data)):
            raise CursorError('invalid native material table: ' + name)
    registry = value.get('world')
    objects = registry.get('objects') if isinstance(registry, dict) else None
    if not isinstance(objects, list) or not objects:
        raise CursorError('native object registry required')
    if _digest(registry) != world['world_registry_digest']:
        raise CursorError('scene registry differs from registered world', 409)
    owners = {}
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('identity'), str):
            raise CursorError('invalid native object identity')
        identity = obj['identity']
        if identity in owners or identity not in world['graph'].nodes or obj != world['graph'].nodes[identity]:
            raise CursorError('scene object differs from registered world', 409)
        owners[identity] = obj
    spans, overlays, parts = (value.get(key) for key in ('identity_spans', 'overlays', 'semantic_part_spans'))
    if any(not isinstance(x, list) for x in (spans, overlays, parts)):
        raise CursorError('native ownership spans required')
    def span_range(span):
        if not isinstance(span, dict):
            raise CursorError('invalid native span')
        first, count = span.get('firstVertex'), span.get('vertexCount')
        if (type(first) is not int or type(count) is not int or first < 0 or count < 0 or
                first % 3 or count % 3 or first + count > len(vertices) // 9):
            raise CursorError('invalid native triangle span')
        return first, count
    owned = {}
    for span in spans:
        span_range(span)
        identity = span.get('identity')
        if not isinstance(identity, str) or identity not in owners or identity in owned:
            raise CursorError('missing or duplicate span owner')
        owned[identity] = span
    kinds = set()
    for span in overlays:
        span_range(span)
        kind = span.get('kind')
        if not isinstance(kind, str) or kind not in ('navigation-routes', 'portal-tubes', 'portal-splats') or kind in kinds:
            raise CursorError('invalid overlay provenance')
        kinds.add(kind)
    covered = 0
    for first, count in sorted(span_range(s) for s in spans + overlays):
        if count:
            if first != covered:
                raise CursorError('scene ownership has gaps or overlap')
            covered += count
    if covered != len(vertices) // 9:
        raise CursorError('scene ownership omits triangles')
    for part in parts:
        first, count = span_range(part)
        parent_id = part.get('objectIdentity')
        parent = owned.get(parent_id) if isinstance(parent_id, str) else None
        if (parent is None or not isinstance(part.get('identity'), str) or not part['identity'] or
                first < parent['firstVertex'] or first + count > parent['firstVertex'] + parent['vertexCount']):
            raise CursorError('invalid semantic part ownership')
    # The digest binds every supplied field, including material data and raw
    # coordinates. No normalizing, re-extruding, centering or shell substitution.
    return {'snapshot': deepcopy(value), 'scene_sha256': _digest(value),
            'ray_available': False, 'ray_unavailable_reasons': [
                'Native celestial shader has no physical light-source contract',
                'Native scene has no certified volume/boundary ownership contract',
                'PhotoJobs admits saved Studio subjects, not general world scenes']}


class StaticWorlds:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.worlds = {}

    def _prune(self):
        now = self.clock()
        for key in [k for k, w in self.worlds.items() if now - w['touched'] >= TTL]:
            del self.worlds[key]

    def _owned(self, owner, identity):
        self._prune()
        world = self.worlds.get(identity)
        if world is None or world['owner'] != owner:
            raise CursorError('static world not found', 404)
        return world

    def _public(self, world):
        return {'world_id': world['id'], 'mode': 'static-living-data',
                'root': world['graph'].nodes[world['root']]['identity'],
                'source_sha256': world['source'], 'renderer_identity': world['renderer'],
                'map_sha256': world['map_digest'],
                'authority': 'registered browser-host snapshot; not server-attested live physics',
                'capabilities': ['inspect', 'render-phong', 'export-scene'], 'simulation': False,
                'scene_sha256': world.get('scene', {}).get('scene_sha256'),
                'ray_available': False}

    def register(self, owner, body):
        _fields(body, {'map', 'source_sha256', 'renderer_identity', 'request_id', 'foyer'})
        foyer = body.get('foyer')
        if foyer is not None:              # the caller's route already required library_manage for this
            _fields(foyer, {'start_position', 'start_facing'})
            foyer = {'start_position': _vector(foyer.get('start_position')), 'start_facing': _vector(foyer.get('start_facing'), True)}
        request_id = _rid(body.get('request_id')); digest = _digest(body)
        source = body.get('source_sha256')
        if not isinstance(source, str) or not SHA.fullmatch(source):
            raise CursorError('source_sha256 must identify the selected static source')
        renderer = _text(body.get('renderer_identity'), 'renderer_identity')
        with self.lock:
            self._prune()
            for world in self.worlds.values():
                if world['owner'] == owner:
                    if world['request_id'] == request_id and world['digest'] == digest:
                        return dict(self._public(world), host_token=world['token'], replayed=True)
                    raise CursorError('close your registered static world before replacing it', 409)
            if foyer is not None and self._foyer() is not None:
                raise CursorError('the foyer already has a resident host; close it first', 409)
            if len(self.worlds) >= MAX_WORLDS:
                raise CursorError('static world capacity reached', 503)
            # Only data graph views; do not instantiate uploaded worker or WASM.
            graph = GraphView(body.get('map'), 'world')
            layout = GraphView(body.get('map'), 'layout')
            for projection in (graph, layout):
                for identity in projection.nodes:
                    projection.node(identity)
            # Drop program-bearing root containers; keep actual object records.
            root = body['map']['identity']
            graph.nodes = {k: deepcopy(n) if k != root else
                           {f: deepcopy(n[f]) for f in ('identity', 'kind', 'name') if f in n}
                           for k, n in graph.nodes.items()}
            layout.nodes = {k: {f: deepcopy(n[f]) for f in
                              ('identity', 'kind', 'name', 'member_kind', 'type_name', 'position',
                               'module', 'source_kind', 'metaphor', 'parameters', 'dependencies',
                               'interaction', 'implied_code') if f in n}
                            for k, n in layout.nodes.items()}
            world = {'id': secrets.token_hex(16), 'owner': owner, 'token': secrets.token_urlsafe(32),
                     'graph': graph, 'layout': layout, 'root': root, 'source': source,
                     'renderer': renderer, 'map_digest': _digest(body['map']), 'commands': {}, 'touched': self.clock(),
                     'request_id': request_id, 'digest': digest,
                     'world_registry_digest': _digest(body['map'].get('world', {}))}
            if foyer is not None:
                world.update(foyer=foyer, visitors={})
            self.worlds[world['id']] = world
            return dict(self._public(world), host_token=world['token'])

    def status(self, owner, identity):
        with self.lock:
            world = self._owned(owner, identity)
            return self._public(world)

    def _receipt(self, world, entry):
        if entry['state'] in ('queued', 'rendering') and self.clock() >= entry['deadline']:
            entry.update(state='failed', result={'error': 'static render host timed out'})
        return {**self._public(world), 'request_id': entry['id'], 'operation': entry['operation'],
                'state': entry['state'], 'result': deepcopy(entry.get('result'))}

    def command(self, owner, identity, body):
        _fields(body, {'request_id', 'operation', 'at', 'layer', 'radius', 'limit', 'offset',
                       'edge_offset', 'position', 'facing', 'width', 'height', 'optics'})
        request_id = _rid(body.get('request_id')); digest = _digest(body)
        operation = body.get('operation')
        if operation not in ('inspect', 'render-phong', 'export-scene'):
            raise CursorError('supported DM commands: inspect, render-phong, export-scene')
        with self.lock:
            world = self._owned(owner, identity)
            old = world['commands'].get(request_id)
            if old:
                if old['digest'] != digest:
                    raise CursorError('request_id reused with different command', 409)
                return dict(self._receipt(world, old), replayed=True)
            if len(world['commands']) >= MAX_COMMANDS:
                raise CursorError('command receipt capacity reached; close and reattach world', 429)
            entry = self._enter(world, body)
            world['touched'] = self.clock()
            return self._receipt(world, entry)

    def _enter(self, world, body, queue=False):
        """Build, record and return one command entry of ``world`` (caller holds the lock and has checked replay/capacity)."""
        request_id, digest, operation = body['request_id'], _digest(body), body.get('operation')
        entry = {'id': request_id, 'digest': digest, 'operation': operation,
                 'state': 'complete', 'deadline': self.clock() + RENDER_TIMEOUT}
        if operation == 'export-scene':
            _fields(body, {'request_id', 'operation'})
            if 'scene' not in world:
                raise CursorError('host has not admitted its realized scene snapshot', 409)
            entry['result'] = world['scene']
        elif operation == 'inspect':
            _fields(body, {'request_id', 'operation', 'at', 'layer', 'radius', 'limit', 'offset', 'edge_offset'})
            layer = body.get('layer', 'world')
            if layer not in ('world', 'layout'):
                raise CursorError('layer must be world or layout')
            graph = world['graph'] if layer == 'world' else world['layout']
            at = body.get('at', world['root'])
            result = graph.query(at, radius=body.get('radius', 1), limit=body.get('limit', 64),
                                 offset=body.get('offset', 0), edge_offset=body.get('edge_offset', 0))
            # Analytic data comes from the exact registered object record.
            result['object'] = deepcopy(graph.nodes[at])
            result['text'] = f"{result['at'].get('name') or at}: {result['total']} declared graph references within {result['radius']} hops."
            if len(json.dumps(result).encode()) > MAX_PNG:
                raise CursorError('analytic object response exceeds 2 MiB', 413)
            entry['result'] = result
        else:
            _fields(body, {'request_id', 'operation', 'position', 'facing', 'width', 'height', 'optics'})
            camera = {'position': _vector(body.get('position')), 'facing': _vector(body.get('facing'), True),
                      'width': _integer(body.get('width', 512), 64, 1024, 'width'),
                      'height': _integer(body.get('height', 384), 64, 1024, 'height')}
            if 'optics' in body:
                camera['optics'] = _optics(body['optics'])
            if not queue:                      # [FOYER] the shared world queues instead, bounded by foyer_command
                for pending in world['commands'].values():
                    self._receipt(world, pending)
                    if pending['state'] in ('queued', 'rendering'):
                        raise CursorError('a render is already pending for this world', 409)
            entry.update(state='queued', camera=camera)
        world['commands'][request_id] = entry
        return entry

    def receipt(self, owner, identity, request_id):
        _rid(request_id)
        with self.lock:
            world = self._owned(owner, identity)
            entry = world['commands'].get(request_id)
            if entry is None:
                raise CursorError('command not found', 404)
            return self._receipt(world, entry)

    def host(self, owner, identity, body):
        _fields(body, {'host_token', 'action', 'request_id', 'result', 'scene'})
        with self.lock:
            world = self._owned(owner, identity)
            token = body.get('host_token')
            if not isinstance(token, str) or not token.isascii() or not hmac.compare_digest(token, world['token']):
                raise CursorError('invalid static host token', 403)
            if body.get('action') == 'scene':
                _fields(body, {'host_token', 'action', 'scene'})
                admitted = _scene_snapshot(body.get('scene'), world)
                previous = world.get('scene')
                if previous and previous['scene_sha256'] != admitted['scene_sha256']:
                    raise CursorError('scene changed; close and register a new world', 409)
                world['scene'] = admitted
                world['touched'] = self.clock()
                return {**self._public(world), **{k: v for k, v in admitted.items() if k != 'snapshot'}}
            if body.get('action') == 'poll':
                _fields(body, {'host_token', 'action'})
                if 'foyer' in world:           # [FOYER] a resident host's polling keeps the shared world open
                    world['touched'] = self.clock()
                for entry in world['commands'].values():
                    self._receipt(world, entry)
                    if entry['state'] in ('queued', 'rendering'):
                        entry['state'] = 'rendering'
                        return {'request_id': entry['id'], 'camera': deepcopy(entry['camera']),
                                'source_sha256': world['source'], 'renderer_identity': world['renderer']}
                return {'request_id': None}
            if body.get('action') != 'complete':
                raise CursorError('host action must be scene, poll or complete')
            request_id = _rid(body.get('request_id')); entry = world['commands'].get(request_id)
            if not entry or entry['operation'] != 'render-phong':
                raise CursorError('render command not found', 404)
            self._receipt(world, entry)
            if entry['state'] != 'rendering':
                raise CursorError('render command is not in flight', 409)
            result = body.get('result')
            if not isinstance(result, dict):
                raise CursorError('render result required')
            if set(result) == {'error'}:
                error = _text(result['error'], 'error')
                entry.update(state='failed', result={'error': error})
                return self._receipt(world, entry)
            _fields(result, {'png_base64', 'width', 'height', 'source_sha256', 'renderer_identity', 'position', 'facing', 'actual_camera'})
            for field in ('source_sha256', 'renderer_identity'):
                if result.get(field) != world['source' if field == 'source_sha256' else 'renderer']:
                    raise CursorError('render source identity mismatch', 409)
            _vector(result.get('position'))
            _vector(result.get('facing'), True)
            _integer(result.get('width'), 64, 1024, 'width')
            _integer(result.get('height'), 64, 1024, 'height')
            for field in ('position', 'facing', 'width', 'height'):
                if result.get(field) != entry['camera'][field]:
                    raise CursorError('render camera receipt mismatch', 409)
            actual = result.get('actual_camera')
            if actual is None:
                raise CursorError('actual camera receipt required')
            if actual is not None:
                _fields(actual, {'position', 'facing', 'width', 'height', 'optics'})
                requested = entry['camera'].get('optics')
                resolved = actual.get('optics')
                if requested is not None:
                    _fields(resolved, {'autofocus', 'f_stop', 'sensor_height_m', 'focus_distance_m',
                                      'measured_rays', 'coverage', 'contrast', 'method', 'depth_mode', 'fov_degrees'})
                    _optics({k: resolved.get(k) for k in requested})
                    if any(resolved.get(k) != requested[k] for k in ('autofocus', 'f_stop', 'sensor_height_m')):
                        raise CursorError('resolved optics differ from requested optics', 409)
                    if not requested['autofocus'] and resolved['focus_distance_m'] != requested['focus_distance_m']:
                        raise CursorError('manual focus differs from request', 409)
                    if resolved.get('method') != 'studio-gaussian-rangefinder-phong-contrast' or resolved.get('depth_mode') != 'opaque':
                        raise CursorError('unsupported optical receipt')
                    if resolved.get('fov_degrees') != 2 * math.atan(.70) * 180 / math.pi:
                        raise CursorError('focus receipt changed native field of view', 409)
                    _integer(resolved.get('measured_rays'), 1 if requested['autofocus'] else 0, 9, 'measured_rays')
                    if requested['autofocus']:
                        for key in ('coverage', 'contrast'):
                            x = resolved.get(key)
                            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                                raise CursorError('invalid focus measurement')
                        if resolved['coverage'] > 1 or resolved['contrast'] <= 0:
                            raise CursorError('invalid focus objective')
                elif resolved is not None:
                    raise CursorError('unsolicited optical result', 409)
                _vector(actual.get('position')); _vector(actual.get('facing'), True)
                if any(actual.get(k) != entry['camera'][k] for k in ('position', 'width', 'height')):
                    raise CursorError('rendered camera position or dimensions mismatch', 409)
                length = math.hypot(*entry['camera']['facing'])
                # Direction normalization is renderer transport, not a physics law.
                if any(not math.isclose(a, b / length, rel_tol=1e-12, abs_tol=1e-12)
                       for a, b in zip(actual['facing'], entry['camera']['facing'])):
                    raise CursorError('rendered camera direction mismatch', 409)
            encoded = result.get('png_base64')
            if not isinstance(encoded, str) or len(encoded) > (MAX_PNG * 4 // 3 + 4):
                raise CursorError('PNG exceeds bounded result size', 413)
            try:
                png = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise CursorError('invalid PNG encoding') from error
            if len(png) > MAX_PNG or len(png) < 33 or png[:8] != b'\x89PNG\r\n\x1a\n' or png[12:16] != b'IHDR':
                raise CursorError('PNG header required')
            if struct.unpack('>II', png[16:24]) != (result['width'], result['height']):
                raise CursorError('PNG dimensions do not match camera receipt')
            # Pillow is an existing repository dependency. Decode only after
            # the bounded IHDR matches the requested <=1024x1024 dimensions.
            try:
                from PIL import Image
                with Image.open(io.BytesIO(png)) as image:
                    if image.format != 'PNG' or image.size != (result['width'], result['height']):
                        raise ValueError('unexpected image format or dimensions')
                    image.verify()
                with Image.open(io.BytesIO(png)) as image:
                    if getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('only single-frame PNG is supported')
                    image.load()
            except (OSError, ValueError, SyntaxError) as error:
                raise CursorError('invalid PNG image') from error
            entry.update(state='complete', result={**deepcopy(result),
                         'png_sha256': hashlib.sha256(png).hexdigest(), 'simulation': False})
            world['touched'] = self.clock()
            return self._receipt(world, entry)

    # ---- [FOYER] the shared demo world, for anyone in the foyer --------------------------------------------------
    def _foyer(self):
        self._prune()
        return next((w for w in self.worlds.values() if 'foyer' in w), None)

    def _foyer_world(self):
        world = self._foyer()
        if world is None:
            raise CursorError('the foyer is closed: no resident host is attached', 503)
        return world

    def _foyer_public(self, world):
        return {'foyer': True, 'demo': True, 'greeting': FOYER_GREETING, 'root': world['graph'].nodes[world['root']]['identity'],
                'source_sha256': world['source'], 'renderer_identity': world['renderer'], 'map_sha256': world['map_digest'],
                'start': {'position': list(world['foyer']['start_position']), 'facing': list(world['foyer']['start_facing'])},
                'capabilities': ['inspect', 'render-phong'], 'max_side': FOYER_MAX_SIDE, 'simulation': False}

    def foyer_status(self):
        with self.lock:
            return self._foyer_public(self._foyer_world())

    def _foyer_receipt(self, world, entry, request_id):
        receipt = self._receipt(world, entry)
        for private in ('world_id',):
            receipt.pop(private, None)
        return {**receipt, 'request_id': request_id, 'demo': True}

    def foyer_command(self, visitor, body):
        """One visitor's inspect or picture on the shared world.  ``visitor`` is the server's identity key for the caller."""
        _fields(body, {'request_id', 'operation', 'at', 'layer', 'radius', 'limit', 'offset', 'edge_offset',
                       'position', 'facing', 'width', 'height'})
        request_id = _rid(body.get('request_id'))
        if body.get('operation') not in ('inspect', 'render-phong'):
            raise CursorError('the foyer offers inspect and render-phong')
        if body.get('operation') == 'render-phong':
            for side in ('width', 'height'):
                if side in body:
                    _integer(body[side], 64, FOYER_MAX_SIDE, side)
        with self.lock:
            world = self._foyer_world()
            mine = world['visitors'].setdefault(visitor, {})
            if request_id in mine:
                entry = world['commands'].get(mine[request_id])
                if entry is None:
                    raise CursorError('that receipt has been forgotten', 410)
                if entry['visitor_digest'] != _digest(body):
                    raise CursorError('request_id reused with different command', 409)
                return dict(self._foyer_receipt(world, entry, request_id), replayed=True)
            if body.get('operation') == 'render-phong':
                pending = [e for e in world['commands'].values() if e['operation'] == 'render-phong'
                           and self._receipt(world, e)['state'] in ('queued', 'rendering')]
                if any(e.get('visitor') == visitor for e in pending):
                    raise CursorError('your last picture is still being taken; ask for its receipt', 409)
                if len(pending) >= FOYER_PENDING:
                    raise CursorError('the foyer camera is busy; try again in a moment', 429)
            self._foyer_evict(world)
            internal = secrets.token_hex(16)                   # the host sees only this, never a visitor's request id
            command = dict(body, request_id=internal)
            # The shared world's own command path: one render in flight is relaxed to the foyer queue above.
            entry = self._enter(world, command, queue=True)
            entry.update(visitor=visitor, visitor_digest=_digest(body))
            mine[request_id] = internal
            world['touched'] = self.clock()
            return self._foyer_receipt(world, entry, request_id)

    def foyer_receipt(self, visitor, request_id):
        _rid(request_id)
        with self.lock:
            world = self._foyer_world()
            internal = world['visitors'].get(visitor, {}).get(request_id)
            entry = world['commands'].get(internal) if internal else None
            if entry is None:
                raise CursorError('receipt not found', 404)
            return self._foyer_receipt(world, entry, request_id)

    def _foyer_evict(self, world):
        finished = [k for k, e in world['commands'].items() if self._receipt(world, e)['state'] in ('complete', 'failed')]
        while len(world['commands']) >= FOYER_RECEIPTS and finished:
            gone = world['commands'].pop(finished.pop(0))
            mine = world['visitors'].get(gone.get('visitor'), {})
            for key in [k for k, v in mine.items() if v == gone['id']]:
                del mine[key]
        if len(world['commands']) >= FOYER_RECEIPTS:
            raise CursorError('the foyer camera is busy; try again in a moment', 429)

    def close(self, owner, identity):
        with self.lock:
            self._owned(owner, identity)
            del self.worlds[identity]
            return {'closed': True, 'world_id': identity}
