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
    old = config()
    assert old == config(receive_light_field=False)
    legacy = dict(old, receive_light_field=True, light_field_estimator='track-length-throughput-v1',
                  light_field_schema='studio.forward-field.v3', light_field_stencil=12)
    assert config(receive_light_field=True) == legacy
    assert config(receive_light_field=True, light_field_estimator='track-length-throughput-v1') == legacy
    def modern(**options):
        return config(receive_light_field=True, light_field_estimator='track-length-root-sweep-v1', **options)
    expected = dict(legacy, light_field_schema='studio.forward-field.v4',
                    light_field_estimator='track-length-root-sweep-v1', light_field_grid_size=64)
    assert modern() == expected
    assert modern(light_field_schema='arbitrary-client-schema') == expected
    canonical = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert canonical(legacy) != canonical(modern())
    assert canonical(modern(light_field_grid_size=32)) != canonical(modern())
    assert canonical(modern(light_field_stencil=6)) != canonical(modern())
    assert config(light_field_grid_size=64, light_field_estimator='track-length-root-sweep-v1') == old
    with pytest.raises(photos.PhotoError, match='light_field_grid_size'):
        config(receive_light_field=True, light_field_grid_size=64)


@pytest.mark.parametrize('value', [None, True, [], {}, 'fog-deposit-v1', 'other'])
def test_unknown_estimator_request_fails_closed(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='light_field_estimator'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3', receive_light_field=True,
                                   light_field_estimator=value), store)


@pytest.mark.parametrize('value', [None, True, '64', 64.0, 0, 16, 128])
def test_grid_request_rejects_unsupported_values(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='light_field_grid_size'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3', receive_light_field=True,
                                   light_field_grid_size=value), store)


@pytest.mark.parametrize('value', [None, True, '12', 12.0, 0, 8, 13])
def test_stencil_request_rejects_unsupported_values(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='light_field_stencil'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3', receive_light_field=True,
                                   light_field_stencil=value), store)


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


@pytest.mark.parametrize('version', [1, 2])
def test_live_field_http_full_bytes_and_dimensions(tiered, version):
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
    if version == 2:
        field, payload = vector_metadata(digest)
        field['live_revision'] = 7
    (directory / NAME).write_bytes(payload)
    (directory / 'config.json').write_text(json.dumps({'receive_light_field': True, 'live': {}, 'geometry_sha256': 'd' * 64}))
    (directory / 'live-control.json').write_text(json.dumps({'revision': 7}))
    (directory / 'status.json').write_text(json.dumps({'digest': digest, 'state': 'live', 'live_revision': 7, 'light_field': field}))
    sessions.marble_photos.active_digest = digest
    job_library.claim(sessions.marble_photos.directory, digest, [owner])
    assert _get(port, '/api/marble_photo?digest=' + digest, headers)[1]['light_field']['dimensions'] == field['dimensions']
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', '/api/marble_photo/field?digest=' + digest + '&name=' + NAME, headers=headers)
    response = conn.getresponse()
    assert response.status == 200
    assert response.getheader('Content-Type') == 'application/octet-stream'
    assert int(response.getheader('Content-Length')) == len(payload)
    assert response.read() == payload
    conn.close()


def vector_metadata(digest):
    import struct
    payload = bytes(96 + 144) + struct.pack('<4I', 0, 0, 0, 1) + bytes(48 + 64 + 64)
    value = metadata(digest)
    value.update(schema='studio.forward-field.v2', byte_length=len(payload),
                 scene_sha256='c' * 64, geometry_sha256='d' * 64,
                 materials=dict(offset_bytes=304, byte_length=64, count=1, stride_bytes=64,
                     format='f32le', layout='native-pbr16'),
                 source_pbr16=dict(offset_bytes=368, byte_length=64, count=1, stride_bytes=64,
                     format='f32le', layout='authored-pbr16', material_ids=[0]),
                 sha256=hashlib.sha256(payload).hexdigest(),
                 directional=dict(offset_bytes=96, byte_length=144, dimensions=[1, 1, 1],
                     estimator='track-length-throughput-v1',
                     direction_space='world',
                     bins=['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
                     channels=['R', 'G', 'B', 'mean_propagation_x', 'mean_propagation_y', 'mean_propagation_z'],
                     format='rgb-direction-f32le', layout='zyx-bin-channel'),
                 triangles=dict(offset_bytes=240, byte_length=16, count=1, stride_bytes=16,
                     class_bits={'volumetric': 1, 'reflectance': 2, 'glass': 4},
                     format='u32le', channels=['triangle_id', 'material_id', 'flags', 'class_mask']),
                 sources=dict(offset_bytes=256, byte_length=48, count=1, stride_bytes=48,
                     format='f32le', layout='baked-source-records'))
    return value, payload


@pytest.fixture
def vector_job(field_job):
    jobs, digest, directory, config, status, save = field_job
    status['light_field'], payload = vector_metadata(digest)
    config['geometry_sha256'] = 'd' * 64
    (directory / NAME).write_bytes(payload)
    save()
    return field_job


def test_vector_bundle_exact_bytes_and_digest(vector_job):
    jobs, digest, directory, config, status, save = vector_job
    with jobs.open_field(digest, NAME) as stream:
        assert stream.read() == vector_metadata(digest)[1]
    payload = bytearray((directory / NAME).read_bytes()); payload[0] = 1
    (directory / NAME).write_bytes(payload)
    with pytest.raises(photos.PhotoError, match='hash'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('section,key,value', [
    ('directional', 'offset_bytes', 0), ('directional', 'byte_length', 140),
    ('directional', 'dimensions', [33, 1, 1]), ('directional', 'dimensions', [True, 1, 1]),
    ('directional', 'bins', ['+X']), ('directional', 'channels', ['R']),
    ('directional', 'estimator', None), ('directional', 'estimator', 'fog-deposit-v1'),
    ('directional', 'direction_space', None), ('directional', 'direction_space', 'view'),
    ('directional', 'layout', 'xyz'), ('triangles', 'count', True),
    ('triangles', 'count', 500001), ('triangles', 'stride_bytes', 12),
    ('triangles', 'offset_bytes', 236), ('sources', 'offset_bytes', 252),
    ('sources', 'format', 'u32le'), ('sources', 'count', -1),
    ('materials', 'offset_bytes', 300), ('materials', 'byte_length', 60),
    ('materials', 'stride_bytes', 32), ('materials', 'layout', 'phong'),
    ('triangles', 'class_bits', {'volumetric': True, 'reflectance': 2, 'glass': 4}),
    ('source_pbr16', 'offset_bytes', 364), ('source_pbr16', 'byte_length', 60),
    ('source_pbr16', 'layout', 'native-pbr16'), ('source_pbr16', 'material_ids', []),
    ('source_pbr16', 'material_ids', [True]), ('source_pbr16', 'material_ids', [1]),
    ('source_pbr16', 'material_ids', [-1])])
def test_vector_bundle_sections_fail_closed(vector_job, section, key, value):
    jobs, digest, directory, config, status, save = vector_job
    status['light_field'][section][key] = value; save()
    with pytest.raises(photos.PhotoError, match='metadata'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('offset', [0, 96, 256, 304, 368])
def test_vector_bundle_nonfinite_payload_rejected_even_with_matching_hash(vector_job, offset):
    import struct
    jobs, digest, directory, config, status, save = vector_job
    payload = bytearray((directory / NAME).read_bytes())
    struct.pack_into('<f', payload, offset, float('nan'))
    (directory / NAME).write_bytes(payload)
    status['light_field']['sha256'] = hashlib.sha256(payload).hexdigest(); save()
    with pytest.raises(photos.PhotoError, match='samples'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('column,value', [(0, 1), (1, 1), (3, 8), (3, 2)])
def test_vector_bundle_invalid_triangle_record(vector_job, column, value):
    import struct
    jobs, digest, directory, config, status, save = vector_job
    payload = bytearray((directory / NAME).read_bytes())
    struct.pack_into('<I', payload, 240 + 4 * column, value)
    (directory / NAME).write_bytes(payload)
    status['light_field']['sha256'] = hashlib.sha256(payload).hexdigest(); save()
    with pytest.raises(photos.PhotoError, match='triangle'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('key,value', [('geometry_sha256', 'e' * 64), ('scene_sha256', ''),
    ('sha256', ''), ('materials', {'count': True}), ('materials', {'count': 0})])
def test_vector_bundle_scene_identity_and_material_metadata(vector_job, key, value):
    jobs, digest, directory, config, status, save = vector_job
    status['light_field'][key] = value; save()
    with pytest.raises(photos.PhotoError, match='metadata'):
        jobs.open_field(digest, NAME)


def test_vector_bundle_live_revision_and_opt_out_fail_closed(vector_job):
    jobs, digest, directory, config, status, save = vector_job
    config['live'] = {}; status['state'] = 'live'; jobs.active_digest = digest
    (directory / 'live-control.json').write_text(json.dumps({'revision': 1})); save()
    with pytest.raises(photos.PhotoError, match='stale'):
        jobs.open_field(digest, NAME)

    status['light_field']['live_revision'] = 1; save()
    with jobs.open_field(digest, NAME) as stream:
        assert stream.read() == vector_metadata(digest)[1]
    config['receive_light_field'] = False; save()
    with pytest.raises(photos.PhotoError, match='unavailable'):
        jobs.open_field(digest, NAME)


@pytest.mark.parametrize('ids', [[], [0, 0]])
def test_vector_bundle_empty_and_duplicate_authored_lineage(vector_job, ids):
    jobs, digest, directory, config, status, save = vector_job
    field = status['light_field']
    field['source_pbr16'].update(count=len(ids), byte_length=64 * len(ids), material_ids=ids)
    payload = (directory / NAME).read_bytes()[:368] + bytes(64 * len(ids))
    field.update(byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (directory / NAME).write_bytes(payload); save()
    with jobs.open_field(digest, NAME) as stream:
        assert stream.read() == payload


@pytest.mark.parametrize('bins', [6, 12])
def test_v3_stencil_sections_and_basis(vector_job, bins):
    jobs, digest, directory, config, status, save = vector_job
    field = status['light_field']; field['schema'] = 'studio.forward-field.v3'
    config['light_field_stencil'] = bins
    section = field['directional']
    section.update(bin_count=bins, stencil='axes6-v1' if bins == 6 else 'icosahedron12-v1',
        bin_directions=photos._FIELD_AXES if bins == 6 else photos._FIELD_ICOSAHEDRON,
        bins=['+X', '-X', '+Y', '-Y', '+Z', '-Z'] if bins == 6 else [f'I{i}' for i in range(12)],
        weight_from_linear_rgb=[1.4400948563066378, .7787163904320042, .781188753261357])
    old_payload = (directory / NAME).read_bytes()
    extra = (bins - 6) * 24
    payload = old_payload[:240] + bytes(extra) + old_payload[240:]
    section['byte_length'] += extra
    for key in ('triangles', 'sources', 'materials', 'source_pbr16'):
        field[key]['offset_bytes'] += extra
    field.update(byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (directory / NAME).write_bytes(payload); save()
    with jobs.open_field(digest, NAME) as stream:
        assert stream.read() == payload
    import copy
    for key, value in [('bin_count', True), ('bin_count', 13), ('stencil', 'unknown'),
        ('bin_directions', [[0, 0, 0]] * bins), ('weight_from_linear_rgb', [1, 1, 1]),
        ('weight_from_linear_rgb', [float('nan'), 1, 1]), ('byte_length', 1)]:
        original = copy.deepcopy(section[key]); section[key] = value; save()
        with pytest.raises(photos.PhotoError): jobs.open_field(digest, NAME)
        section[key] = original


@pytest.mark.parametrize('grid', [32, 64])
@pytest.mark.parametrize('version', [4, 5, 6])
def test_v4_grid_exact_payload_and_config_identity(vector_job, grid, version):
    jobs, digest, directory, config, status, save = vector_job
    field = status['light_field']; field['schema'] = 'studio.forward-field.v4'
    config.update(light_field_schema='studio.forward-field.v4', light_field_grid_size=grid,
                  light_field_estimator='track-length-root-sweep-v1', light_field_stencil=12)
    section = field['directional']
    section.update(dimensions=[grid] * 3, bin_count=12, stencil='icosahedron12-v1',
        estimator='track-length-root-sweep-v1', bin_directions=photos._FIELD_ICOSAHEDRON,
        bins=[f'I{i}' for i in range(12)],
        root_sampling=dict(version='per-complete-pump-inverse-replay-v1', source_root_count=1,
            completed_root_sweeps=2, recorded_root_jobs=3, normalization_divisor=2,
            excluded_partial_root_jobs=0, incomplete_dispatch=False),
        weight_from_linear_rgb=[1.4400948563066378, .7787163904320042, .781188753261357])
    if version in (5, 6):
        import numpy as np
        config.update(light_field_schema='studio.forward-field.v5', light_field_region='subject',
                      light_field_cell_size_m=.0005, light_field_margin_m=.001, scale_m=1., object_center_m=[0, 0, 0])
        edge = .005 if grid == 32 else .012
        vertices = np.array([[[-edge, -edge, -edge], [edge, edge, edge], [0, 0, 0]]], dtype=np.float32)
        receipt = photos._directional_contract(str(photos.SPECTRAL_ROOT)).subject_field_region(vertices, config)
        assert receipt['dimensions'] == [grid] * 3
        config['light_field_subject_region'] = receipt
        field['schema'] = 'studio.forward-field.v5'
        section.update(receipt)
    if version == 6:
        field['schema'] = config['light_field_schema'] = 'studio.forward-field.v6'
        section['quantization_divisor'] = 16777216
        section['quantization_version'] = 'fixed24-track-length-v1'
        config['emitter_sampling'] = dict(version=1, ray_budget=4096, seed=42, refresh_each_sweep=True)
        section['source_populations'] = dict(version='native-angular-refresh-mean-v1',
            generator='native-emissive-triangle-roots-v1', ray_budget=4096, base_seed=42,
            first_index=0, last_index=1, count=2, latest_seed=43,
            latest_sources_sha256=hashlib.sha256(bytes(48)).hexdigest(), rolling_sha256='f'*64)
    old_payload = (directory / NAME).read_bytes()
    section['byte_length'] = grid**3 * 12 * 6 * 4
    payload = old_payload[:96] + bytes(section['byte_length']) + old_payload[240:]
    extra = section['byte_length'] - 144
    for key in ('triangles', 'sources', 'materials', 'source_pbr16'):
        field[key]['offset_bytes'] += extra
    field.update(byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (directory / NAME).write_bytes(payload); save()
    with jobs.open_field(digest, NAME) as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == field['sha256']
    if version == 6:
        config['light_field_internal_media'] = True
        with pytest.raises(photos.PhotoError, match='metadata'):
            photos._validate_vector_bundle_metadata(field, config)
        section['internal_media'] = dict(photos._FIELD_INTERNAL_MEDIA)
        photos._validate_vector_bundle_metadata(field, config)
        for key, value in [('entry_packet', 1), ('version', 'fog'), ('invalid_ownership', 'included'),
                           ('integration', 'constant-throughput')]:
            original = section['internal_media'][key]; section['internal_media'][key] = value
            with pytest.raises(photos.PhotoError, match='metadata'):
                photos._validate_vector_bundle_metadata(field, config)
            section['internal_media'][key] = original
        config['light_field_internal_media'] = False
        with pytest.raises(photos.PhotoError, match='metadata'):
            photos._validate_vector_bundle_metadata(field, config)
        del section['internal_media']; del config['light_field_internal_media']
    config['light_field_grid_size'] = 64 if grid == 32 else 32; save()
    with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
    config['light_field_grid_size'] = grid
    import copy
    for key, value in [('dimensions', [grid, grid, grid//2]), ('dimensions', [128]*3),
        ('estimator', 'track-length-throughput-v1')]:
        original = copy.deepcopy(section[key]); section[key] = value; save()
        with pytest.raises(photos.PhotoError): jobs.open_field(digest, NAME)
        section[key] = original
    sampling = section['root_sampling']
    for key, value in [('version', 'unknown'), ('source_root_count', 0), ('source_root_count', 2),
        ('completed_root_sweeps', 0), ('completed_root_sweeps', True), ('recorded_root_jobs', 1),
        ('normalization_divisor', 3), ('normalization_divisor', 2.0),
        ('excluded_partial_root_jobs', -1), ('incomplete_dispatch', True), ('incomplete_dispatch', 0)]:
        original = sampling[key]; sampling[key] = value; save()
        with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
        sampling[key] = original
    if version in (5, 6):
        for key, value in [('bounds_min_m', [-1, -1, -1]), ('cell_size_m', .001), ('margin_m', .002),
                           ('region', dict(section['region'], subject_geometry_sha256='f' * 64)),
                           ('region', dict(section['region'], padding_min_m=[0, 0, 0])),
                           ('region', dict(section['region'], padding_max_m=[float('nan')] * 3))]:
            original = section[key]; section[key] = value; save()
            with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
            section[key] = original
    if version == 6:
        section['quantization_version'] = 'unknown'; save()
        with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
        section['quantization_version'] = 'fixed24-track-length-v1'
        for value in (None, True, 65535, 16777216.0):
            section['quantization_divisor'] = value; save()
            with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
        section['quantization_divisor'] = 16777216
        populations = section['source_populations']
        for key, value in [('count', 1), ('count', True), ('last_index', 3), ('latest_seed', 42),
                           ('generator', 'alternate'), ('rolling_sha256', ''), ('base_seed', -1)]:
            original = populations[key]; populations[key] = value; save()
            with pytest.raises(photos.PhotoError, match='metadata'): jobs.open_field(digest, NAME)
            populations[key] = original
        populations['latest_sources_sha256'] = 'e'*64; save()
        with pytest.raises(photos.PhotoError, match='population hash'): jobs.open_field(digest, NAME)


def test_v4_request_cannot_accept_legacy_field(vector_job):
    jobs, digest, directory, config, status, save = vector_job
    config['light_field_schema'] = 'studio.forward-field.v4'; save()
    with pytest.raises(photos.PhotoError, match='schema mismatch'): jobs.open_field(digest, NAME)


@pytest.mark.parametrize('margin', [0, .001, .003])
def test_subject_pitch_admission_uses_verified_geometry(photo_payload, margin):
    from tests.test_meowdoku_marble_photo import _saved_object
    store, digest, vertices, normals, ids = photo_payload
    saved = _saved_object(vertices, normals, ids); saved['physical']['radius-mm'] = 5
    digest = store.put(saved)['digest']
    config = photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3',
        receive_light_field=True, light_field_estimator='track-length-root-sweep-v1',
        light_field_region='subject', light_field_cell_size_m=.0005, light_field_margin_m=margin,
        light_field_subject_region={'bounds_min_m': [999]*3}), store)[3]
    receipt = config['light_field_subject_region']
    assert config['light_field_schema'] == 'studio.forward-field.v5'
    assert receipt['cell_size_m'] == .0005 and receipt['margin_m'] == margin
    assert receipt['region']['subject_geometry_sha256'] == config['geometry_sha256']
    assert receipt['region']['effective_margin_m'] == max(.0005, margin)
    assert config['light_field_grid_size'] in (32, 64)
    assert receipt['bounds_min_m'] != [999]*3


def test_subject_pitch_capacity_rejects_without_coarsening(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='maximum is64'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3',
            receive_light_field=True, light_field_estimator='track-length-root-sweep-v1',
            light_field_region='subject', light_field_cell_size_m=.0005), store)


@pytest.mark.parametrize('key,value', [('light_field_margin_m', -.001), ('light_field_margin_m', True),
    ('light_field_margin_m', .033), ('light_field_margin_m', float('nan')),
    ('light_field_cell_size_m', .00001), ('light_field_cell_size_m', True), ('light_field_region', 'stage')])
def test_subject_request_rejects_invalid_options(photo_payload, key, value):
    store, digest, vertices, normals, ids = photo_payload
    options = dict(transport='fixed:3', receive_light_field=True,
        light_field_estimator='track-length-root-sweep-v1', light_field_region='subject')
    options[key] = value
    with pytest.raises(photos.PhotoError):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)


@pytest.mark.parametrize('pitch', [.00005, .001, .01])
def test_subject_pitch_supported_range(photo_payload, pitch):
    from tests.test_meowdoku_marble_photo import _saved_object
    store, digest, vertices, normals, ids = photo_payload
    saved = _saved_object(vertices, normals, ids); saved['physical']['radius-mm'] = .1
    digest = store.put(saved)['digest']
    config = photos.decode_upload(_wire(digest, vertices, normals, ids, transport='fixed:3',
        receive_light_field=True, light_field_estimator='track-length-root-sweep-v1',
        light_field_region='subject', light_field_cell_size_m=pitch, light_field_margin_m=0), store)[3]
    assert config['light_field_subject_region']['cell_size_m'] == pitch



def emitter_request():
    return dict(transport='fixed:3', receive_light_field=True,
        light_field_estimator='track-length-root-sweep-v1', camera={'resolution': 128},
        fixtures={'vertices': [0, 0, 0, .01, 0, 0, 0, .01, 0], 'material_ids': [0], 'pbr': [0]*16},
        live={'session_id': '1'*32, 'ray_batch_size': 8192},
        emitter_sampling={'version': 1, 'ray_budget': 4096, 'seed': 42, 'refresh_each_sweep': True})


def test_emitter_sampling_preserved_in_immutable_job_config(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    def config(options): return photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)[3]
    options = emitter_request(); first = config(options)
    assert first['emitter_sampling'] == options['emitter_sampling']
    options['emitter_sampling']['seed'] = 43
    assert config(options) != first
    del options['emitter_sampling']
    assert 'emitter_sampling' not in config(options)


@pytest.mark.parametrize('key,value', [('version', True), ('version', 2), ('ray_budget', True),
    ('ray_budget', 127), ('ray_budget', 65537), ('seed', -1), ('seed', 2147483648),
    ('seed', 4.2), ('refresh_each_sweep', 1), ('interaction_target', [0, 0, 0])])
def test_emitter_sampling_invalid_levers_fail_closed(photo_payload, key, value):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request(); options['emitter_sampling'][key] = value
    with pytest.raises(photos.PhotoError, match='emitter_sampling'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)


@pytest.mark.parametrize('change', ['no_live', 'no_field', 'no_fixtures', 'legacy', 'short_batch'])
def test_emitter_refresh_requires_supported_job(photo_payload, change):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request()
    if change == 'no_live': del options['live']
    elif change == 'no_field': options['receive_light_field'] = False
    elif change == 'no_fixtures': del options['fixtures']
    elif change == 'legacy': del options['light_field_estimator']
    elif change == 'short_batch': options['live']['ray_batch_size'] = 128
    with pytest.raises(photos.PhotoError, match='emitter_sampling'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)



def test_live_batch_update_cannot_starve_configured_source_population(tmp_path):
    from tests.test_meowdoku_marble_live import fake_jobs
    from meowdoku.marble_live import update_live
    jobs, digest, directory = fake_jobs(tmp_path)
    config = json.loads((directory/'config.json').read_text())
    config['emitter_sampling'] = emitter_request()['emitter_sampling']
    (directory/'config.json').write_text(json.dumps(config))
    before = (directory/'live-control.json').read_bytes()
    with pytest.raises(photos.PhotoError, match='emitter_sampling'):
        update_live(jobs, digest, {'revision': 1, 'live': {'ray_batch_size': 128}})
    assert (directory/'live-control.json').read_bytes() == before
    assert update_live(jobs, digest, {'revision': 1, 'live': {'ray_batch_size': 8192}})['ok']



def test_internal_media_explicit_opt_in_and_cache_identity(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    def config(options): return photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)[3]
    options = emitter_request(); old = config(options)
    assert config(dict(options, light_field_internal_media=False)) == old
    enabled = config(dict(options, light_field_internal_media=True))
    assert enabled == dict(old, light_field_internal_media=True)
    assert enabled['light_field_schema'] == 'studio.forward-field.v6'


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_internal_media_requires_strict_boolean(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request(); options['light_field_internal_media'] = value
    with pytest.raises(photos.PhotoError, match='boolean'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)


@pytest.mark.parametrize('change', ['no_refresh', 'no_sampling', 'volume_extinction',
    'volume_extinction_sigma_t_m_inv', 'volume_extinction_bounds_m'])
def test_internal_media_unsupported_combinations_rejected(photo_payload, change):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request(); options['light_field_internal_media'] = True
    if change == 'no_refresh': options['emitter_sampling']['refresh_each_sweep'] = False
    elif change == 'no_sampling': del options['emitter_sampling']
    else: options[change] = {}
    with pytest.raises(photos.PhotoError, match='light_field_internal_media'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)



def test_surface_transport_explicit_changes_immutable_config_only_when_selected(photo_payload):
    transport = 'fixed:3'
    store, digest, vertices, normals, ids = photo_payload
    def config(**options):
        return photos.decode_upload(_wire(digest, vertices, normals, ids, transport=transport, fixtures=emitter_request()['fixtures'], **options), store)[3]
    original = config()
    assert 'surface_transport' not in original
    selected = config(surface_transport='studio-pbr-mixture-v1')
    assert selected == dict(original, surface_transport='studio-pbr-mixture-v1')
    encode = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    assert encode(selected) != encode(original)


@pytest.mark.parametrize('value', [None, True, False, 1, [], {}, 'legacy', 'studio-pbr-mixture-v2'])
def test_surface_transport_rejects_unsupported_values(photo_payload, value):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='surface_transport'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, surface_transport=value), store)


@pytest.mark.parametrize('options', [dict(transport='fixed:1', fixtures=emitter_request()['fixtures']),
                                      dict(transport='fixed:3')])
def test_surface_transport_requires_supported_source_contract(photo_payload, options):
    store, digest, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match='surface_transport requires'):
        photos.decode_upload(_wire(digest, vertices, normals, ids,
            surface_transport='studio-pbr-mixture-v1', **options), store)


@pytest.mark.parametrize('vector', [False, True])
def test_surface_transport_receipt_matches_immutable_job(field_job, vector):
    jobs, digest, directory, config, status, save = field_job
    if vector:
        status['light_field'], payload = vector_metadata(digest)
        config['geometry_sha256'] = 'd'*64
        (directory/NAME).write_bytes(payload)
    field = status['light_field']; config['surface_transport'] = 'studio-pbr-mixture-v1'; save()
    with pytest.raises(photos.PhotoError, match='receipt mismatch'): jobs.open_field(digest, NAME)
    for bad in (None, True, {}, 'old'):
        field['surface_transport'] = bad; save()
        with pytest.raises(photos.PhotoError, match='receipt mismatch'): jobs.open_field(digest, NAME)
    field['surface_transport'] = 'studio-pbr-mixture-v1'; save()
    with jobs.open_field(digest, NAME) as stream: assert stream.read()
    del config['surface_transport']; save()
    with pytest.raises(photos.PhotoError, match='unexpected'): jobs.open_field(digest, NAME)





def test_response_information_is_explicit_opt_in_only(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    def config(options): return photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)[3]
    options = emitter_request(); options['light_field_internal_media'] = True
    old = config(options)
    assert old['light_field_schema'] == 'studio.forward-field.v6'
    assert 'light_field_information' not in old
    new = config(dict(options, light_field_information='evaluated-responses-v1'))
    assert new == dict(old, light_field_information='evaluated-responses-v1')
    # A schema name alone cannot activate additional collection or transport.
    assert config(dict(options, light_field_schema='studio.forward-field.v7')) == old


@pytest.mark.parametrize('change', ['missing_internal', 'missing_sampling', 'missing_field', 'wrong_value'])
def test_response_information_requires_full_explicit_contract(photo_payload, change):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request(); options.update(light_field_internal_media=True, light_field_information='evaluated-responses-v1')
    if change == 'missing_internal': options['light_field_internal_media'] = False
    elif change == 'missing_sampling': del options['emitter_sampling']
    elif change == 'missing_field': options['receive_light_field'] = False
    else: options['light_field_information'] = True
    with pytest.raises(photos.PhotoError): photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)


def test_response_lineage_is_opt_in_and_preserved(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    def config(options): return photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)[3]
    options = emitter_request()
    ordinary = config(options)
    lineage = {'version': 'studio-response-lineage-v1', 'parts': [{'source_id': 'fixture-1', 'start': 0, 'count': 1}]}
    options['fixtures']['response_lineage'] = lineage
    assert config(options) == ordinary
    options.update(light_field_internal_media=True, light_field_information='evaluated-responses-v1')
    assert config(options)['fixtures']['response_lineage'] == lineage
    options['light_field_information'] = 'spectral-arrivals-v1'
    with pytest.raises(photos.PhotoError): config(options)


@pytest.mark.parametrize('parts', [[], [{'source_id':'x','start':1,'count':1}],
    [{'source_id':'x','start':0,'count':True}], [{'source_id':'','start':0,'count':1}],
    [{'source_id':'x','start':0,'count':2}], [{'source_id':'x','start':0,'count':1,'extra':1}]])
def test_response_lineage_rejects_invalid_cover(photo_payload, parts):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request()
    options.update(light_field_internal_media=True, light_field_information='evaluated-responses-v1')
    options['fixtures']['response_lineage'] = {'version':'studio-response-lineage-v1', 'parts':parts}
    with pytest.raises(photos.PhotoError):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store)


SCENE_NAME = 'light-scene-response-' + '3' * 32 + '.bin'


def scene_receipt(digest):
    import sys
    import numpy as np
    if str(photos.SPECTRAL_ROOT) not in sys.path: sys.path.insert(0, str(photos.SPECTRAL_ROOT))
    from camera_designer.studio_scene_responses import SceneResponseStore
    store = SceneResponseStore(scene_revision='scene-a', memory_budget_bytes=32768, lineage_provider=lambda: __import__('camera_designer.studio_response_lineage', fromlist=['NativeResponseLineage']).NativeResponseLineage(0).snapshot())
    store.consume_scalar_cells({'scene_revision':'scene-a', 'scalar_clear_epoch':0, 'snapshot_id':0, 'cell_offset':0, 'total_cells':1, 'final_slab':True}, np.arange(4,dtype='<u4').reshape(1,1,4))
    manifest, payload = store.snapshot()
    return dict(schema='studio-scene-responses-v1', digest=digest, live_revision=0, sequence=1,
        name=SCENE_NAME, byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        metadata=manifest, application_ready=False), payload


@pytest.fixture
def scene_job(field_job):
    jobs, digest, directory, config, status, save = field_job
    receipt, payload = scene_receipt(digest)
    config['light_field_information'] = 'evaluated-responses-v1'
    status['scene_responses'] = receipt
    def publish():
        (directory / SCENE_NAME).write_bytes(payload)
        (directory / SCENE_NAME).with_suffix('.scene.json').write_text(json.dumps(receipt))
        save()
    publish()
    return jobs, digest, directory, config, status, receipt, payload, publish


def test_scene_response_stream_lease_and_old_snapshot(scene_job):
    jobs,digest,directory,config,status,receipt,payload,publish = scene_job
    status['scene_responses'] = dict(receipt, name='light-scene-response-'+'4'*32+'.bin', sequence=2)
    publish()
    lease = (directory / SCENE_NAME).with_suffix('.leases.json')
    with jobs.open_field(digest,SCENE_NAME) as stream:
        assert len(json.loads(lease.read_text())['transfers']) == 1
        assert stream.read() == payload
    assert json.loads(lease.read_text())['transfers'] == {}
    del config['light_field_information']; publish()
    with pytest.raises(photos.PhotoError): jobs.open_field(digest,SCENE_NAME)


@pytest.mark.parametrize('bad', ['hash','length','ready','revision','scene','sections','traversal'])
def test_scene_response_rejects_corrupt_or_stale(scene_job,bad):
    jobs,digest,directory,config,status,receipt,payload,publish = scene_job
    if bad=='hash': receipt['sha256']='f'*64
    elif bad=='length': receipt['byte_length']+=1
    elif bad=='ready': receipt['metadata']['application_ready']=True
    elif bad=='revision': receipt['live_revision']=1
    elif bad=='scene':
        status['scene_responses']=dict(receipt,metadata=dict(receipt['metadata'],scene_revision='scene-b'))
    elif bad=='sections': receipt['metadata']['sections'].reverse()
    publish()
    if bad=='traversal': name='../'+SCENE_NAME
    else: name=SCENE_NAME
    with pytest.raises(photos.PhotoError): jobs.open_field(digest,name)


@pytest.mark.parametrize("header_failure", [None, "send_header", "end_headers"])
def test_scene_response_http_uses_existing_owner_route(tiered, monkeypatch, header_failure):
    port,set_tier,sessions=tiered
    set_tier('shader')
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    conn.request('GET','/api/marble_photo/jobs');response=conn.getresponse();response.read()
    cookies=[v.split(';')[0] for k,v in response.getheaders() if k.lower()=='set-cookie'];conn.close()
    token=next(v.split('=',1)[1] for v in cookies if v.startswith(server.COOKIE+'='))
    headers={'Cookie':'; '.join(cookies)}
    digest='f'*64;directory=sessions.marble_photos.directory/digest;directory.mkdir(parents=True)
    receipt,payload=scene_receipt(digest)
    (directory/'config.json').write_text(json.dumps({'receive_light_field':True,'light_field_information':'evaluated-responses-v1'}))
    (directory/'status.json').write_text(json.dumps({'digest':digest,'state':'complete','scene_responses':receipt}))
    (directory/SCENE_NAME).write_bytes(payload)
    (directory/SCENE_NAME).with_suffix('.scene.json').write_text(json.dumps(receipt))
    job_library.claim(sessions.marble_photos.directory,digest,['browser:'+hashlib.sha256(token.encode()).hexdigest()])
    path='/api/marble_photo/field?digest='+digest+'&name='+SCENE_NAME
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    if header_failure:
        from http.server import BaseHTTPRequestHandler
        original=getattr(BaseHTTPRequestHandler,header_failure)
        def fail_header(handler,*args,**kwargs):
            if handler.path.startswith('/api/marble_photo/field?'):
                raise ConnectionResetError('simulated header disconnect')
            return original(handler,*args,**kwargs)
        monkeypatch.setattr(BaseHTTPRequestHandler,header_failure,fail_header)
        conn.request('GET',path,headers=headers)
        with pytest.raises((http.client.RemoteDisconnected, ConnectionError)):
            conn.getresponse()
        conn.close()
        import time
        lease=(directory/SCENE_NAME).with_suffix('.leases.json')
        for _ in range(100):
            if lease.exists() and json.loads(lease.read_text())['transfers']=={}:break
            time.sleep(.01)
        assert json.loads(lease.read_text())['transfers']=={}
        return
    conn.request('GET',path,headers=headers);response=conn.getresponse()
    assert response.status==200
    assert int(response.getheader('Content-Length'))==len(payload)
    assert response.read()==payload;conn.close()
    assert _get(port,path)[0]==403


def test_scene_response_payload_hash_and_transfer_limit(scene_job):
    import time
    jobs,digest,directory,config,status,receipt,payload,publish=scene_job
    path=directory/SCENE_NAME
    path.write_bytes(bytes([payload[0]^1])+payload[1:])
    with pytest.raises(photos.PhotoError,match='hash'):jobs.open_field(digest,SCENE_NAME)
    publish()
    path.with_suffix('.leases.json').write_text(json.dumps(dict(version=1,name=SCENE_NAME,
        transfers={format(i,'032x'):time.time()+600 for i in range(8)})))
    with pytest.raises(photos.PhotoError,match='capacity'):jobs.open_field(digest,SCENE_NAME)


def test_scene_publisher_backpressure_respects_active_api_transfers(field_job):
    from types import SimpleNamespace
    jobs,digest,directory,config,status,save=field_job
    receipt,payload=scene_receipt(digest)
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher
    publisher=SceneResponsePublisher(directory,digest,interval=0)
    store=SimpleNamespace(snapshot=lambda:(receipt['metadata'],payload))
    config['light_field_information']='evaluated-responses-v1'
    streams=[]
    try:
        for i in range(3):
            item=publisher.publish(store)
            status['scene_responses']=item;save()
            streams.append(jobs.open_field(digest,item['name']))
        latest=publisher.latest
        assert publisher.publish(store)==latest
        assert len(list(directory.glob('light-scene-response-*.bin')))==3
        assert streams[0].read()==payload
        streams[0].close()
        assert publisher.publish(store)['name']!=latest['name']
        assert len(list(directory.glob('light-scene-response-*.bin')))==3
    finally:
        for stream in streams:stream.close()


@pytest.mark.parametrize('protected', ['none','lease','status','latest','wrong_digest','wrong_hash'])
def test_scene_retired_artifacts_reconcile_only_verified_owned(field_job,protected):
    import time
    jobs,digest,directory,config,status,save=field_job
    receipt,payload=scene_receipt(digest)
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher,response_lock
    target=directory/SCENE_NAME
    retired=target.with_suffix('.scene.retired')
    target.write_bytes(payload)
    if protected=='wrong_digest':receipt['digest']='b'*64
    if protected=='wrong_hash':receipt['sha256']='0'*64
    retired.write_text(json.dumps(receipt))
    publisher=SceneResponsePublisher(directory,digest)
    if protected=='latest':publisher.latest=receipt
    if protected=='status':status['scene_responses']=receipt;save()
    if protected=='lease':target.with_suffix('.leases.json').write_text(json.dumps(dict(version=1,name=SCENE_NAME,transfers={'1'*32:time.time()+600})))
    with response_lock(directory):room=publisher._room_locked()
    if protected=='none':
        assert room and not target.exists() and not retired.exists()
    else:
        assert target.exists() and retired.exists()
        if protected.startswith('wrong_'):assert not room


def test_scene_failed_retirement_stays_counted_and_recovers(field_job,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    jobs,digest,directory,config,status,save=field_job
    receipt,payload=scene_receipt(digest)
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher,response_lock
    publisher=SceneResponsePublisher(directory,digest,interval=0)
    store=SimpleNamespace(snapshot=lambda:(receipt['metadata'],payload))
    items=[publisher.publish(store) for _ in range(3)]
    victim=directory/items[0]['name'];original=Path.unlink
    def blocked(path,*args,**kwargs):
        if path==victim:raise PermissionError('simulated interrupted retirement')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path,'unlink',blocked)
        assert publisher.publish(store)==items[-1]
        assert victim.with_suffix('.scene.retired').exists()
        assert publisher.publish(store)==items[-1]
        assert len(list(directory.glob('light-scene-response-*.bin')))==3
    new=publisher.publish(store)
    assert new['name']!=items[-1]['name']
    assert not victim.exists() and not victim.with_suffix('.scene.retired').exists()
    assert len(list(directory.glob('light-scene-response-*.bin')))==3


def test_scene_unowned_orphan_binary_blocks_without_deletion(field_job):
    jobs,digest,directory,config,status,save=field_job
    scene_receipt(digest)
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher,response_lock
    orphan=directory/SCENE_NAME;orphan.write_bytes(b'unowned')
    publisher=SceneResponsePublisher(directory,digest)
    with response_lock(directory):assert not publisher._room_locked()
    assert orphan.read_bytes()==b'unowned'


def native_partition_store():
    import numpy as np
    scene_receipt('a'*64)  # Load the same CPU-only Spectral contract used by the API.
    from camera_designer.studio_scene_responses import SceneResponseStore
    from camera_designer.studio_response_lineage import NativeResponseLineage
    from camera_designer.film_response import normalize_film_response
    lineage=NativeResponseLineage(1)
    store=SceneResponseStore(scene_revision='physical',memory_budget_bytes=1024*1024,
        lineage_provider=lineage.snapshot,bounds=[[0,0,0],[1,1,1]],dimensions=[32,32,32])
    rows=np.zeros((3,128),dtype='<u4')
    rows[:,0:2]=[5,1];rows[:,5]=[0,0xffffffff,0]
    rows[:,30]=1;rows[:,32]=0x80000001;rows[:,64]=0x40000002
    rows[:,18:24]=np.array([.1,.2,.3,.4,.5,.6],np.float32).view(np.uint32)
    rows[:,24:30]=np.array([0,1,0,1,0,0],np.float32).view(np.uint32)
    rows[:,112:114]=[2,0];rows[1:,112]=0
    metadata=dict(scene_revision='physical',stream_id='native',sequence=0,source_sha256='source',
        transport_scene_sha256='transport',film_response=normalize_film_response(),primitive_identity_offsets=[112,113])
    store.consume_forward(metadata,rows)
    return store,rows


@pytest.mark.parametrize('change', ['available','count','capacity','evicted','boolean_count','identity','scheme','absent'])
def test_api_validates_optional_native_partition_receipt(field_job,change):
    jobs,digest,directory,config,status,save=field_job
    store,rows=native_partition_store()
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher
    receipt=SceneResponsePublisher(directory,digest).publish(store)
    config['light_field_information']='evaluated-responses-v1';status['scene_responses']=receipt
    partitions=receipt['metadata']['retention']['partitions']
    if change=='available':partitions['native_only_application_available']=True
    elif change=='count':partitions['native_only_retained_entries']+=1
    elif change=='capacity':partitions['native_only_capacity']+=1
    elif change=='evicted':partitions['native_only_evicted_entries']+=1
    elif change=='boolean_count':partitions['native_only_context_discarded_entries']=True
    elif change=='identity':
        partitions['native_only_retained_entries']-=1;partitions['surface_retained_entries']+=1
    elif change=='scheme':partitions['scheme']='other'
    else:del receipt['metadata']['retention']['partitions']
    (directory/receipt['name']).with_suffix('.scene.json').write_text(json.dumps(receipt));save()
    if change=='absent':
        with jobs.open_field(digest,receipt['name']) as stream:assert len(stream.read())==receipt['byte_length']
    else:
        with pytest.raises(photos.PhotoError,match='retention'):jobs.open_field(digest,receipt['name'])


def test_native_partition_publisher_to_existing_http_is_bit_exact(tiered):
    import numpy as np
    port,set_tier,sessions=tiered;set_tier('shader')
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    conn.request('GET','/api/marble_photo/jobs');response=conn.getresponse();response.read()
    cookies=[v.split(';')[0] for k,v in response.getheaders() if k.lower()=='set-cookie'];conn.close()
    token=next(v.split('=',1)[1] for v in cookies if v.startswith(server.COOKIE+'='))
    headers={'Cookie':'; '.join(cookies)}
    digest='d'*64;directory=sessions.marble_photos.directory/digest;directory.mkdir(parents=True)
    store,rows=native_partition_store()
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher
    receipt=SceneResponsePublisher(directory,digest).publish(store)
    (directory/'config.json').write_text(json.dumps({'receive_light_field':True,'light_field_information':'evaluated-responses-v1'}))
    (directory/'status.json').write_text(json.dumps({'digest':digest,'state':'complete','scene_responses':receipt}))
    job_library.claim(sessions.marble_photos.directory,digest,['browser:'+hashlib.sha256(token.encode()).hexdigest()])
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    conn.request('GET','/api/marble_photo/field?digest='+digest+'&name='+receipt['name'],headers=headers)
    response=conn.getresponse();assert response.status==200
    payload=response.read();conn.close()
    assert payload==(directory/receipt['name']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==receipt['sha256']
    section=receipt['metadata']['sections'][1]
    np.testing.assert_array_equal(np.frombuffer(payload,'<u4',offset=section['offset'],count=384).reshape(3,128),rows)
    index=np.frombuffer(payload,'<u4',count=60).reshape(3,20)
    assert list(index[:,4])==[0,0,0] and list(index[:,11])==[1,0,1]
    assert list(index[:,6].view('<i4'))==[2,0,0]
    assert receipt['metadata']['retention']['partitions']['native_only_retained_entries']==3
    assert receipt['application_ready'] is False


def test_absent_information_opt_in_creates_no_scene_assets(field_job):
    jobs,digest,directory,config,status,save=field_job
    before=set(directory.iterdir())
    with jobs.open_field(digest,NAME) as stream:assert stream.read()==bytes(96)
    with pytest.raises(photos.PhotoError):jobs.open_field(digest,SCENE_NAME)
    assert set(directory.iterdir())==before
    assert not list(directory.glob('light-scene-response-*'))
    assert not (directory/'light-response.lock').exists()
