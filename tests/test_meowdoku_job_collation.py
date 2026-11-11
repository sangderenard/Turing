"""Collation is persisted metadata over the existing owned photo worker."""
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tests.test_meowdoku_marble_photo import photo_payload, _wire, _FakeProcess
from meowdoku import job_library as lib, marble_photo as photos

SPEC = Path(os.environ.get('STUDIO_TEST_SPECTRAL_ROOT', Path(__file__).resolve().parents[2] / 'spectral-analyzer'))
DIGEST = 'a' * 64


def setup_job(tmp_path, config, status, *, manifest=True):
    directory = tmp_path / DIGEST
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'status.json').write_text(json.dumps({'digest': DIGEST, **status}))
    contract = photos._tile_contract(str(SPEC)) if 'tiled' in config else None
    if manifest:
        lib.write_collation(directory, lib.collation_manifest(DIGEST, config, tile_contract=contract))
    lib.claim(tmp_path, DIGEST, ['account:1'])
    jobs = SimpleNamespace(directory=tmp_path, spectral_root=SPEC, lock=threading.Lock(),
                           active_digest=DIGEST, process=SimpleNamespace(poll=lambda: None),
                           _status_locked=lambda digest: json.loads((directory / 'status.json').read_text()))
    return jobs, directory


def entry(jobs):
    return lib.entries(jobs, ['account:1'])['entries'][0]


def test_four_tiles_are_one_owned_parent_and_restart_is_interrupted(tmp_path):
    config = {'transport': 'fixed:3', 'camera': {'resolution': 1024},
              'oversampling': 2, 'tiled': {'tile_size': 1024}}
    jobs, directory = setup_job(tmp_path, config, {'state': 'exposing',
        'tiled': {'completed_tiles': 1, 'active_tile': {'index': 1}}})
    value = entry(jobs)
    group = value['collation']
    assert (group['completed'], group['total'], group['kind']) == (1, 4, 'tiles')
    assert group['dimensions'] == {'width': 2048, 'height': 2048, 'output_width': 1024,
                                    'output_height': 1024, 'oversampling': 2}
    assert [child['state'] for child in group['children']] == ['complete', 'exposing', 'pending', 'pending']
    assert group['children'][3]['window'] == {'x': 1024, 'y': 1024, 'width': 1024, 'height': 1024}
    assert all(child['result_kind'] == 'assembled_image' and 'image' not in child for child in group['children'])
    assert value['can_cancel']
    assert lib.entries(jobs, ['account:2'], [DIGEST])['entries'] == []
    before = (directory / 'collation.json').read_bytes()
    assert entry(jobs) == value  # Reading/replay never schedules or duplicates a child.
    assert (directory / 'collation.json').read_bytes() == before
    jobs.active_digest = None
    jobs.process = None
    interrupted = entry(jobs)
    assert interrupted['state'] == interrupted['collation']['state'] == 'interrupted'
    assert [child['state'] for child in interrupted['collation']['children']] == ['complete'] + ['interrupted'] * 3
    assert not interrupted['can_cancel']
    assert not lib.entries(jobs, ['account:1'])['queue_available']


@pytest.mark.parametrize('source', ['orbit', 'explicit'])
def test_pose_frames_have_stable_membership_and_only_real_results(tmp_path, source):
    poses = [{'eye_m': [i, 0, .07], 'target_m': [0, 0, .18], 'fov_degrees': 18} for i in range(8)]
    config = {'camera_poses': poses} if source == 'explicit' else {'orbit_frames': 8}
    jobs, directory = setup_job(tmp_path, config, {'state': 'exposing', 'frame_index': 2, 'frames': [
        {'index': 0, 'image': 'preview-000001.png', 'hdr': 'exposure-000001.npz'},
        {'index': 1, 'image': 'preview-000002.png', 'hdr': None}]})
    for filename in ('preview-000001.png', 'exposure-000001.npz', 'preview-000002.png'):
        (directory / filename).write_bytes(b'retained result')
    group = entry(jobs)['collation']
    assert group['source'] == source and group['kind'] == 'camera_pose_sequence'
    assert group['completed'] == 2 and group['total'] == 8
    assert [child['state'] for child in group['children'][:4]] == ['complete', 'complete', 'exposing', 'pending']
    if source == 'explicit':
        assert group['children'][3]['pose'] == poses[3]
    else:
        assert group['children'][3]['pose'] == {'path': 'orbit', 'frame_index': 3, 'frame_count': 8}
    (directory / 'exposure-000001.npz').unlink()
    assert entry(jobs)['collation']['completed'] == 1
    jobs.active_digest = None
    assert entry(jobs)['state'] == 'interrupted'
    assert entry(jobs)['collation']['children'][1]['image'] == 'preview-000002.png'


def test_complete_parent_requires_complete_children_and_final_tile_result(tmp_path):
    config = {'transport': 'fixed:3', 'camera': {'resolution': 1024},
              'oversampling': 2, 'tiled': {'tile_size': 1024}}
    jobs, directory = setup_job(tmp_path, config, {'state': 'complete', 'tiled': {'completed_tiles': 4},
        'image': 'preview-000000.png', 'hdr': 'exposure-000000.npz'})
    assert entry(jobs)['state'] == 'error'
    for filename in ('preview-000000.png', 'exposure-000000.npz'):
        (directory / filename).write_bytes(b'final result')
    assert entry(jobs)['state'] == 'complete'
    status = json.loads((directory / 'status.json').read_text())
    status['tiled']['completed_tiles'] = 3
    (directory / 'status.json').write_text(json.dumps(status))
    assert entry(jobs)['state'] == 'error'


def test_legacy_config_is_read_without_backfill_and_corrupt_manifest_is_visible(tmp_path):
    jobs, directory = setup_job(tmp_path, {'orbit_frames': 8}, {'state': 'exposing', 'frames': []}, manifest=False)
    assert entry(jobs)['collation']['manifest_source'] == 'saved_config'
    assert not (directory / 'collation.json').exists()
    (directory / 'collation.json').write_text('{"private":"never reveal"}')
    value = entry(jobs)
    assert value['state'] == 'error' and 'collation' not in value
    assert 'private' not in str(value) and 'never reveal' not in str(value)


@pytest.mark.parametrize('state', ['canceled', 'interrupted', 'error'])
def test_retained_child_artifacts_readable_without_reviving_parent(tmp_path, state):
    directory = tmp_path / DIGEST
    directory.mkdir()
    (directory / 'status.json').write_text(json.dumps({'digest': DIGEST, 'state': state,
        'image': 'preview-000002.png', 'frames': [{'index': 0, 'image': 'preview-000001.png', 'hdr': 'exposure-000001.npz'}]}))
    for filename in ('preview-000001.png', 'exposure-000001.npz', 'preview-000002.png', 'exposure-000002.npz'):
        (directory / filename).write_bytes(b'finished child')
    jobs = photos.PhotoJobs(tmp_path, SPEC)
    try:
        assert jobs.image(DIGEST, 'preview-000001.png') == b'finished child'
        assert jobs.hdr(DIGEST, 'exposure-000001.npz') == b'finished child'
        for filename, method in [('preview-000002.png', jobs.image), ('exposure-000002.npz', jobs.hdr)]:
            with pytest.raises(photos.PhotoError, match='completed child'):
                method(DIGEST, filename)
        assert jobs.active_digest is None and jobs.process is None
    finally:
        jobs.close()


def test_admission_persists_manifest_before_worker_and_cancel_preserves_receipts(tmp_path, photo_payload, monkeypatch):
    store, digest, vertices, normals, ids = photo_payload
    monkeypatch.setattr(photos.PhotoJobs, '_validate_geometry', lambda *a: {})
    seen = []
    def launch(command, **kwargs):
        path = Path(command[command.index('--out') + 1])
        seen.append(json.loads((path / 'collation.json').read_text()))
        return _FakeProcess(command, **kwargs)
    monkeypatch.setattr(photos.subprocess, 'Popen', launch)
    jobs = photos.PhotoJobs(tmp_path / 'jobs', SPEC)
    try:
        result = jobs.start(_wire(digest, vertices, normals, ids, orbit_frames=8), store, owners=['account:1'])
        assert len(seen) == 1 and seen[0]['source'] == 'orbit' and len(seen[0]['children']) == 8
        assert seen[0]['parent_digest'] == result['digest']
        jobs.start(_wire(digest, vertices, normals, ids, orbit_frames=8), store, owners=['account:1'])
        assert len(seen) == 1
        directory = jobs.directory / result['digest']
        (directory / 'preview-000001.png').write_bytes(b'frame')
        status = {'digest': result['digest'], 'state': 'exposing', 'frame_index': 1,
                  'frames': [{'index': 0, 'image': 'preview-000001.png', 'hdr': None}]}
        (directory / 'status.json').write_text(json.dumps(status))
        jobs.cancel(result['digest'])
        value = lib.entries(jobs, ['account:1'])['entries'][0]
        assert value['state'] == 'canceled' and value['collation']['completed'] == 1
        assert [child['state'] for child in value['collation']['children']] == ['complete'] + ['canceled'] * 7
        assert jobs.image(result['digest'], 'preview-000001.png') == b'frame'
    finally:
        jobs.close()
