"""Native mesh -> Wavefront -> parsed triangles, names and normals round trip."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from meowdoku.living_data_obj import export_obj, write_obj_bundle
CAPTURE = ROOT / 'build/dot-storybook-static-phong-20261110/realized-mesh.json'
SPECTRAL_LOADER = ROOT.parent / 'spectral-analyzer/camera_designer/obj_loader.py'


def fixture():
    identity = 'world:test'
    owner = 'room:a b/\u2603\nusemtl injected'
    part = owner + '/surface:floor'
    return {'schema': 'living-data-realized-mesh.v1', 'stride': 9,
            'coordinate_space': 'world', 'up_axis': 'Y', 'physics_started': False,
            'simulation_advanced': False, 'world_identity': identity, 'source_sha256': '1' * 64,
            'renderer_identity': identity + '/shader-programs/pluck-phong',
            'world': {'root': identity, 'objects': [{'identity': owner, 'name': 'Snow room',
                       'transform': {'position': [10, 2, 4]}, 'semantic_parts': [{'identity': part}]}]},
            'vertices': [10., 2., 4., 0., 0., 1., 1., 0., 0.,
                         11., 2., 4., 0., 0., 1., 1., 0., 0.,
                         10., 3., 4., 0., 0., 1., 0., 1., 0.],
            'material_colors': [[1., 0., 0.], [0., 1., 0.]],
            'materials': {'pbr': [.2, .3, .4, .5, .6, .7, 1.5, .8, 1., 2., 3., 0., 0., 0., 0., 0.] * 2,
                          'phong': [.1, .2, 32., 0., .3, .4, .5, 0.] * 2,
                          'enamel': [0.] * 16, 'texstack': [0.] * 32},
            'identity_spans': [{'identity': owner, 'firstVertex': 0, 'vertexCount': 3}],
            'semantic_part_spans': [{'identity': part, 'objectIdentity': owner, 'firstVertex': 0, 'vertexCount': 3}],
            'overlays': []}


def parse_obj(text):
    positions, normals, faces = [], [], []
    owner, groups, material = None, (), None
    for line in text.splitlines():
        token = line.split()
        if not token:
            continue
        if token[0] == 'v':
            positions.append(list(map(float, token[1:])))
        elif token[0] == 'vn':
            normals.append(list(map(float, token[1:])))
        elif token[0] == 'o':
            owner = token[1]
        elif token[0] == 'g':
            groups = tuple(token[1:])
        elif token[0] == 'usemtl':
            material = token[1]
        elif token[0] == 'f':
            assert len(token) == 4
            indices = [tuple(map(int, item.split('//'))) for item in token[1:]]
            faces.append({'positions': [positions[v - 1] for v, n in indices],
                          'normals': [normals[n - 1] for v, n in indices],
                          'owner': owner, 'groups': groups, 'material': material})
    return faces


def assert_round_trip(snapshot):
    result = export_obj(snapshot)
    faces = parse_obj(result['obj'])
    source = np.array(snapshot['vertices']).reshape(-1, 3, 9)
    assert np.array_equal([face['positions'] for face in faces], source[:, :, :3])
    assert np.array_equal([face['normals'] for face in faces], source[:, :, 3:6])
    for mapping in result['manifest']['names']:
        span = mapping['source']
        start, end = span['firstVertex'] // 3, (span['firstVertex'] + span['vertexCount']) // 3
        for face in faces[start:end]:
            if mapping['source_kind'] == 'semantic_part_spans':
                assert mapping['obj_name'] in face['groups']
            else:
                assert mapping['obj_name'] == face['owner']
    assert result['manifest']['source'] == snapshot
    assert len(faces) == result['manifest']['triangles']
    return result, faces


def test_exact_positions_normals_names_transform_and_last_vertex_material(tmp_path):
    snapshot = fixture()
    before = deepcopy(snapshot)
    result, faces = assert_round_trip(snapshot)
    assert faces[0]['material'] == 'native_material_01'  # Flat native provoking vertex.
    assert '%0Ausemtl%20injected' in faces[0]['owner']
    assert '\nusemtl injected\n' not in result['obj']
    assert 'Ni 1.5\n' in result['mtl']
    assert snapshot == before
    receipt = write_obj_bundle(snapshot, tmp_path, stem='storybook')
    assert len(receipt['files']) == 3
    assert json.loads((tmp_path / 'storybook.manifest.json').read_text(encoding='utf-8'))['source'] == snapshot


def test_overlays_are_exported_without_fabricated_object_identity():
    snapshot = fixture()
    snapshot['vertices'] *= 2
    snapshot['overlays'] = [{'kind': 'portal-tubes', 'firstVertex': 3, 'vertexCount': 3}]
    _, faces = assert_round_trip(snapshot)
    assert faces[1]['owner'] == 'overlay_portal-tubes'
    assert faces[1]['groups'] == ()


@pytest.mark.parametrize('mutate', [
    lambda s: s['vertices'].__setitem__(0, float('nan')),
    lambda s: s['vertices'].pop(),
    lambda s: s['identity_spans'].clear(),
    lambda s: s['identity_spans'].append(deepcopy(s['identity_spans'][0])),
    lambda s: s['identity_spans'][0].update(firstVertex=1),
    lambda s: s['identity_spans'][0].update(identity='absent'),
    lambda s: s['semantic_part_spans'][0].update(objectIdentity='absent'),
    lambda s: s['overlays'].append({'kind': 'invented', 'firstVertex': 0, 'vertexCount': 0}),
    lambda s: s.update(up_axis='Z'),
    lambda s: s.update(renderer_identity='fake'),
    lambda s: s.update(world_identity='different'),
    lambda s: s.update(physics_started=True),
])
def test_invalid_or_incomplete_native_input_fails_closed(mutate):
    snapshot = fixture()
    mutate(snapshot)
    with pytest.raises(ValueError):
        export_obj(snapshot)


@pytest.mark.skipif(not CAPTURE.exists(), reason='Actual browser Storybook mesh has not been captured')
def test_actual_storybook_all_triangles_normals_objects_and_groups_round_trip():
    snapshot = json.loads(CAPTURE.read_text(encoding='utf-8'))
    assert len(snapshot['world']['objects']) == 33
    assert snapshot['world_identity'].endswith('StorybookEnvironment')
    result, faces = assert_round_trip(snapshot)
    assert len({face['owner'] for face in faces}) == 33
    assert result['manifest']['vertices'] == len(snapshot['vertices']) // 9


@pytest.mark.skipif(not CAPTURE.exists() or not SPECTRAL_LOADER.exists(), reason='Native Storybook capture and existing Spectral importer required')
def test_actual_storybook_existing_spectral_importer_accepts_obj_mtl():
    snapshot = json.loads(CAPTURE.read_text(encoding='utf-8'))
    result = export_obj(snapshot)
    spec = importlib.util.spec_from_file_location('existing_spectral_obj_loader', SPECTRAL_LOADER)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    vertices, material_ids, pbr, names = loader.load_obj(result['obj'], result['mtl'])
    native = np.array(snapshot['vertices']).reshape(-1, 9)[:, :3]
    # Existing importer deliberately normalizes imported objects. Verify that
    # documented normalization, rather than claiming it preserves world scale.
    center = (native.min(axis=0) + native.max(axis=0)) * .5
    radius = np.linalg.norm(native - center, axis=1).max()
    expected = ((native - center) / radius).astype(np.float32).reshape(-1, 3, 3)
    assert np.array_equal(vertices, expected)
    assert [names[index] for index in material_ids] == [f'native_material_{i:02d}' for i in result['manifest']['face_material_ids']]
    for name, row in zip(names, pbr):
        index = int(name.rsplit('_', 1)[1])
        source = np.array(snapshot['materials']['pbr'][index * 16:(index + 1) * 16], dtype=np.float32)
        assert np.array_equal(row[[0, 1, 2, 6, 7, 8, 9, 10]], source[[0, 1, 2, 6, 7, 8, 9, 10]])
