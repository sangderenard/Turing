"""Wavefront serialization of the native Living Data renderer's realized mesh.

This module does not generate geometry or evaluate the authored world. Its only
input is livingDataStaticHost.meshSnapshot(), captured after native realization.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.parse import quote


LOSSES = [
    'OBJ contains a static triangle snapshot, not executable graph, physics, animation, or persistence behavior.',
    'World-space transforms are baked into vertices and normals. Authored transforms and graph identities remain in the manifest.',
    'MTL carries albedo, Phong ambient/specular/shininess, emission, IOR and opacity. Pr/Pm are optional PBR extensions.',
    'Transmission uses qualitative illum 7; MTL opacity and lighting are not equivalent to native optical/shader semantics.',
    'Full PBR, enamel, texture-stack, thermal bindings and original vertex colours remain in the manifest, not portable OBJ shading.',
    'OBJ has no unit declaration. Coordinates, scale, handedness and Y-up orientation are unchanged from the native mesh.',
    'No UVs are invented. No materials, cameras, lights or behavior are reconstructed from names or layout boxes.',
]


def _numbers(value, label, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise ValueError(f'{label} has an invalid shape')
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in value):
        raise ValueError(f'{label} contains a non-finite or nonnumeric value')
    return value


def _range(span, vertex_count):
    start, count = span.get('firstVertex'), span.get('vertexCount')
    if any(type(v) is not int or v < 0 or v % 3 for v in (start, count)) or start + count > vertex_count:
        raise ValueError('Identity/overlay/semantic span is not an in-bounds triangle range')
    return range(start // 3, (start + count) // 3)


def _name(prefix, identity):
    if not isinstance(identity, str) or not identity:
        raise ValueError('Every object, group and overlay needs a nonempty identity')
    return prefix + quote(identity, safe='._-:')


def _f32(number):
    return struct.unpack('<f', struct.pack('<f', number))[0]


def _material_index(color, palette):
    # The native adapter has flat vMatId, so the last (provoking) vertex owns
    # the face material. Mirror its Float32 nearest-palette, first-tie rule.
    def distance(candidate):
        delta = [_f32(a - b) for a, b in zip(color, candidate)]
        return _f32(_f32(_f32(delta[0] ** 2) + _f32(delta[1] ** 2)) + _f32(delta[2] ** 2))
    return min(range(len(palette)), key=lambda index: distance(palette[index]))


def export_obj(snapshot, *, stem='living-data'):
    """Return OBJ, MTL and a lossless source manifest without changing geometry."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', stem):
        raise ValueError('Export stem must be a simple ASCII filename')
    if snapshot.get('schema') != 'living-data-realized-mesh.v1' or snapshot.get('stride') != 9:
        raise ValueError('Expected the native interleaved9 realized mesh snapshot')
    if snapshot.get('coordinate_space') != 'world' or snapshot.get('up_axis') != 'Y':
        raise ValueError('Expected native world-space Y-up coordinates')
    if snapshot.get('physics_started') is not False or snapshot.get('simulation_advanced') is not False:
        raise ValueError('Export requires a static host snapshot')
    world = snapshot.get('world', {})
    identity = snapshot.get('world_identity')
    if not isinstance(identity, str) or not identity or world.get('root') != identity:
        raise ValueError('World identity does not match the native registry root')
    if snapshot.get('renderer_identity') != identity + '/shader-programs/pluck-phong':
        raise ValueError('Expected the native world Pluck Phong renderer identity')
    if not re.fullmatch(r'[a-f0-9]{64}', snapshot.get('source_sha256', '')):
        raise ValueError('Expected a native source SHA256 receipt')
    objects = world.get('objects', [])
    object_ids = [node.get('identity') for node in objects]
    if not objects or any(not isinstance(key, str) or not key for key in object_ids) or len(set(object_ids)) != len(object_ids):
        raise ValueError('Native registry object identities must be nonempty and unique')
    vertices = _numbers(snapshot.get('vertices'), 'vertices')
    if not vertices or len(vertices) % 27 or len(vertices) > 27 * 500000:
        raise ValueError('Expected 1..500000 complete native triangles')
    vertex_count, triangle_count = len(vertices) // 9, len(vertices) // 27
    palette = snapshot.get('material_colors')
    if not isinstance(palette, list) or not 1 <= len(palette) <= 16:
        raise ValueError('Expected 1..16 native material colours')
    for color in palette:
        _numbers(color, 'material colour', 3)
    materials = snapshot.get('materials', {})
    for member, width in [('pbr', 16), ('phong', 8), ('enamel', 8), ('texstack', 16)]:
        _numbers(materials.get(member), member, len(palette) * width)
    owners, groups, mappings = [None] * triangle_count, [[] for _ in range(triangle_count)], []
    for key, prefix, identity_key in [('identity_spans', 'object_', 'identity'), ('overlays', 'overlay_', 'kind')]:
        for span in snapshot.get(key, []):
            if key == 'identity_spans' and span.get('identity') not in object_ids:
                raise ValueError('Mesh span owner is absent from the native object registry')
            if key == 'overlays' and span.get('kind') not in {'navigation-routes', 'portal-tubes', 'portal-splats'}:
                raise ValueError('Unsupported native overlay kind')
            name = _name(prefix, span.get(identity_key))
            mappings.append({'obj_name': name, 'source_kind': key, 'source': span})
            for triangle in _range(span, vertex_count):
                if owners[triangle] is not None:
                    raise ValueError('Object and overlay spans overlap')
                owners[triangle] = name
    if any(owner is None for owner in owners):
        raise ValueError('Object and overlay spans do not cover every triangle')
    for span in snapshot.get('semantic_part_spans', []):
        name = _name('part_', span.get('identity'))
        owner_name = _name('object_', span.get('objectIdentity'))
        mappings.append({'obj_name': name, 'source_kind': 'semantic_part_spans', 'source': span})
        for triangle in _range(span, vertex_count):
            if owners[triangle] != owner_name:
                raise ValueError('Semantic part range is outside its matching object owner')
            if name not in groups[triangle]:
                groups[triangle].append(name)
    fmt = lambda values: ' '.join(format(float(v), '.17g') for v in values)
    obj = ['# Native Living Data realized mesh. World coordinates and Y-up unchanged.',
           f'# Full source state and format limitations: {stem}.manifest.json', f'mtllib {stem}.mtl', 's off']
    for offset in range(0, len(vertices), 9):
        obj.append('v ' + fmt(vertices[offset:offset + 3]))
    for offset in range(0, len(vertices), 9):
        obj.append('vn ' + fmt(vertices[offset + 3:offset + 6]))
    face_materials = []
    previous = None
    for triangle in range(triangle_count):
        first = triangle * 3
        material = _material_index(vertices[(first + 2) * 9 + 6:(first + 3) * 9], palette)
        face_materials.append(material)
        state = owners[triangle], tuple(groups[triangle]), material
        if previous is None or state[0] != previous[0]:
            obj.append('o ' + state[0])
        if previous is None or state[1] != previous[1]:
            obj.append('g' + (' ' + ' '.join(state[1]) if state[1] else ''))
        if previous is None or material != previous[2]:
            obj.append(f'usemtl native_material_{material:02d}')
        obj.append('f ' + ' '.join(f'{i}//{i}' for i in range(first + 1, first + 4)))
        previous = state
    mtl = ['# Native material projection. Full arrays retained in companion manifest.',
           '# Pr/Pm are optional extensions; MTL is not a native-shader equivalence.']
    for index in range(len(palette)):
        pbr = materials['pbr'][index * 16:(index + 1) * 16]
        phong = materials['phong'][index * 8:(index + 1) * 8]
        mtl.extend([f'newmtl native_material_{index:02d}', 'Kd ' + fmt(pbr[:3]),
                    'Ka ' + fmt([phong[0] * x for x in pbr[:3]]),
                    'Ks ' + fmt([phong[1]] * 3), 'Ns ' + fmt([phong[2]]),
                    'Ke ' + fmt(pbr[8:11]), 'Ni ' + fmt([pbr[6]]), 'd ' + fmt([pbr[7]]),
                    'Pr ' + fmt([pbr[3]]), 'Pm ' + fmt([pbr[4]]),
                    f'illum {7 if pbr[5] > 0 else 2}', ''])
    obj_text, mtl_text = '\n'.join(obj) + '\n', '\n'.join(mtl) + '\n'
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    manifest = {'schema': 'living-data-obj-export.v1', 'format': 'Wavefront OBJ + MTL',
                'provenance_scope': 'Host-supplied snapshot; structural identity validation is not independent source attestation.',
                'source_snapshot_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
                'vertices': vertex_count, 'triangles': triangle_count,
                'name_encoding': 'prefix plus reversible UTF-8 percent encoding of source identity',
                'names': mappings, 'face_material_ids': face_materials, 'losses': LOSSES,
                'source': snapshot, 'files': {stem + '.obj': hashlib.sha256(obj_text.encode()).hexdigest(),
                                             stem + '.mtl': hashlib.sha256(mtl_text.encode()).hexdigest()}}
    return {'obj': obj_text, 'mtl': mtl_text, 'manifest': manifest}


def write_obj_bundle(snapshot, directory, *, stem='living-data'):
    bundle = export_obj(snapshot, stem=stem)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for extension, content in [('obj', bundle['obj']), ('mtl', bundle['mtl']),
                               ('manifest.json', json.dumps(bundle['manifest'], ensure_ascii=False, indent=2, allow_nan=False) + '\n')]:
        path = directory / f'{stem}.{extension}'
        path.write_text(content, encoding='utf-8', newline='\n')
        files.append(str(path))
    return {'files': files, 'vertices': bundle['manifest']['vertices'],
            'triangles': bundle['manifest']['triangles'],
            'source_snapshot_sha256': bundle['manifest']['source_snapshot_sha256']}
