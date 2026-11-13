"""CPU-only camera-pair request and existing artifact-route contracts."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.test_meowdoku_marble_photo import photo_payload, _wire
from tests.test_meowdoku_forward_field import emitter_request
from meowdoku import marble_photo as photos


@pytest.fixture
def spectral_root():
    # Isolated cross-repository verification can name the real producer checkout.
    # No modules or production defaults are replaced by a stand-in.
    return Path(os.environ.get('STUDIO_SPECTRAL_TEST_ROOT', str(photos.SPECTRAL_ROOT)))


def request_config(photo_payload, spectral_root, **changes):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request()
    options.update(light_field_internal_media=True, light_field_information='evaluated-responses-v1')
    options.update(changes)
    return photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store, spectral_root)[3]


def test_interval_mode_is_explicit_and_preserves_completed_default(photo_payload, spectral_root):
    default = request_config(photo_payload, spectral_root)
    assert 'light_field_camera_records' not in default
    selected = request_config(photo_payload, spectral_root, light_field_camera_records='interval-responses-v2')
    assert selected == dict(default, light_field_camera_records='interval-responses-v2')


@pytest.mark.parametrize('mode', [None, True, 1, {}, [], 'interval-responses-v1', 'completed-samples-v1'])
def test_unknown_camera_record_mode_is_rejected(photo_payload, spectral_root, mode):
    with pytest.raises(photos.PhotoError, match='interval camera records'):
        request_config(photo_payload, spectral_root, light_field_camera_records=mode)


def test_interval_mode_cannot_enable_response_information(photo_payload, spectral_root):
    store, digest, vertices, normals, ids = photo_payload
    options = emitter_request()
    options.update(light_field_internal_media=True, light_field_camera_records='interval-responses-v2')
    with pytest.raises(photos.PhotoError, match='explicit evaluated response information'):
        photos.decode_upload(_wire(digest, vertices, normals, ids, **options), store, spectral_root)


@pytest.fixture
def pair_job(tmp_path, spectral_root):
    sys.path.insert(0, str(spectral_root))
    from camera_designer.studio_response_lineage import NativeResponseLineage
    from camera_designer.studio_scene_responses import SceneResponseStore
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher
    if not hasattr(SceneResponseStore, 'consume_camera_pairs'):
        pytest.fail('Versioned delivery test requires the matching real Spectral camera-pair producer checkout')
    digest = 'a'*64
    directory = tmp_path / digest
    directory.mkdir()
    lineage = NativeResponseLineage(0)
    store = SceneResponseStore(scene_revision='b'*64, memory_budget_bytes=1024*1024,
        lineage_provider=lambda: lineage.snapshot())
    intervals = np.zeros((1,128), dtype='<u4')
    intervals[0,4] = 0xffffffff
    intervals[0,6] = 3
    intervals[0,11] = 0x7fc12345  # Actual miss distance remains unknown.
    intervals[0,12:15] = np.array([1,0,0], np.float32).view(np.uint32)
    intervals[0,40] = 7
    intervals[0,126] = 2
    responses = np.zeros((1,128), dtype='<u4')
    responses[0,0] = 0x7fc01234
    metadata = dict(version='native-camera-intervals-v1', response_schema='native-camera-response-copies-v1',
        complete_dispatch_observation=True, application_ready=False, row_count=1,
        scene_revision='b'*64, stream_id='actual-camera', sequence=0,
        source_sha256='c'*64, transport_scene_sha256='d'*64)
    store.consume_camera_pairs(metadata, intervals, responses)
    receipt = SceneResponsePublisher(directory,digest).publish(store)
    payload = (directory / receipt['name']).read_bytes()
    config = dict(receive_light_field=True, light_field_information='evaluated-responses-v1',
        light_field_camera_records='interval-responses-v2')
    status = dict(digest=digest, state='complete', scene_responses=receipt)
    def save():
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'status.json').write_text(json.dumps(status))
        (directory / receipt['name']).with_suffix('.scene.json').write_text(json.dumps(receipt))
    save()
    jobs = photos.PhotoJobs(tmp_path, spectral_root)
    try:
        yield jobs, digest, directory, config, status, receipt, payload, save
    finally:
        jobs.close()


def test_real_v2_publisher_uses_existing_immutable_artifact_and_lease(pair_job):
    jobs,digest,directory,config,status,receipt,payload,save = pair_job
    assert receipt['schema'] == receipt['metadata']['schema'] == 'studio-scene-responses-v2'
    assert receipt['metadata']['sections'][2]['stride'] == 1024
    lease = (directory / receipt['name']).with_suffix('.leases.json')
    with jobs.open_field(digest, receipt['name']) as stream:
        assert len(json.loads(lease.read_text())['transfers']) == 1
        assert stream.read() == payload
    assert json.loads(lease.read_text())['transfers'] == {}
    del config['light_field_camera_records'];save()
    with pytest.raises(photos.PhotoError):
        jobs.open_field(digest, receipt['name'])


@pytest.mark.parametrize('bad', ['mixed_schema','unknown_schema','layout','codec','contract','direction','ready','hash'])
def test_v2_artifact_route_rejects_mixed_or_corrupt_contract(pair_job, bad):
    import hashlib
    jobs,digest,directory,config,status,receipt,payload,save = pair_job
    if bad == 'mixed_schema': receipt['schema'] = 'studio-scene-responses-v1'
    elif bad == 'unknown_schema': receipt['schema'] = receipt['metadata']['schema'] = 'studio-scene-responses-v3'
    elif bad == 'layout': receipt['metadata']['sections'][2]['stride'] = 1536
    elif bad == 'codec': receipt['metadata']['families']['3']['codec'] = 'native-camera-response-pairs-v1'
    elif bad == 'contract': receipt['metadata']['camera_pair_contract']['response_offset_words'] = 0
    elif bad == 'ready': receipt['metadata']['application_ready'] = True
    elif bad == 'hash': receipt['sha256'] = 'e'*64
    elif bad == 'direction':
        raw = bytearray(payload)
        np.frombuffer(raw, '<u4')[12] = 0  # Outer attachment no longer matches actual interval.
        (directory / receipt['name']).write_bytes(raw)
        receipt['sha256'] = receipt['metadata']['payload_sha256'] = hashlib.sha256(raw).hexdigest()
    save()
    with pytest.raises(photos.PhotoError):
        jobs.open_field(digest, receipt['name'])
