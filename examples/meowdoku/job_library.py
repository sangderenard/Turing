"""Ownership/index metadata only. Scheduling stays with the engine/Nodus host."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import time
from contextlib import contextmanager

try:
    from meowdoku import schema_version as _schema_version
except ImportError:      # loaded by path or run as a script: take the sibling file
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("meowdoku_schema_version", Path(__file__).with_name("schema_version.py"))
    _schema_version = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_schema_version)


@contextmanager
def connect(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(directory/'job-library.sqlite',timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS job_owners (digest TEXT PRIMARY KEY, owners TEXT NOT NULL, created REAL NOT NULL)')
    _schema_version.declare(db,'job_library',1)
    try:
        with db:
            yield db
    finally:
        db.close()


def claim(directory,digest,owners):
    """First owner wins. Multiple keys are same authenticated submitting identity."""
    if not owners: return
    with connect(directory) as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT owners FROM job_owners WHERE digest=?',(digest,)).fetchone()
        if row is not None:
            if not set(json.loads(row[0])).intersection(owners):
                raise PermissionError('job belongs to another identity')
        else:
            db.execute('INSERT INTO job_owners VALUES (?,?,?)',(digest,json.dumps(sorted(set(owners))),time.time()))


def allowed(directory,digest,owners,*,allow_legacy=True):
    with connect(directory) as db:
        row=db.execute('SELECT owners FROM job_owners WHERE digest=?',(digest,)).fetchone()
    # Legacy unindexed jobs keep their existing access semantics, but are not
    # enumerated to strangers. New read-only callers must pass allow_legacy=False.
    # No guessing/importing ownership from a digest.
    return allow_legacy if row is None else bool(set(json.loads(row[0])).intersection(owners))


def collation_manifest(digest, config, *, tile_contract=None):
    """Describe the existing worker's children, never schedule independent jobs.

    Membership is immutable and bound to the exact admitted config. Progress
    remains owned by the worker's status.json, not a second job-state ledger.
    """
    if config.get('live'):
        return None
    children = []
    if 'tiled' in config:
        if tile_contract is None:
            raise ValueError('The existing Spectral tile contract is required')
        plan = tile_contract.tile_plan(config)
        kind, source = 'tiles', 'sensor_windows'
        children = [{'index': i, 'kind': 'tile', 'window': window}
                    for i, window in enumerate(tile_contract.tile_windows(plan))]
        dimensions = {k: plan[k] for k in ('width', 'height', 'output_width', 'output_height', 'oversampling')}
    else:
        poses = config.get('camera_poses')
        count = len(poses) if poses is not None else config.get('orbit_frames', 1)
        if count == 1 and poses is None:
            return None
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError('Invalid camera sequence size')
        kind, source = 'camera_pose_sequence', 'explicit' if poses is not None else 'orbit'
        camera_fields = ('eye_m', 'target_m', 'fov_degrees', 'aperture_radius_m',
                         'focus_distance_m', 'aperture_blades', 'aperture_rotation_radians')
        for i in range(count):
            pose = ({k: poses[i][k] for k in camera_fields if k in poses[i]} if poses is not None else
                    {'path': 'orbit', 'frame_index': i, 'frame_count': count})
            children.append({'index': i, 'kind': 'camera_pose', 'pose': pose})
        dimensions = {}
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'version': 1, 'parent_digest': digest, 'config_sha256': hashlib.sha256(canonical).hexdigest(),
            'kind': kind, 'source': source, 'dimensions': dimensions, 'children': children}


def write_collation(directory, manifest):
    """Write admission metadata beside the staged config, before worker birth."""
    if manifest is None:
        return
    path = Path(directory) / 'collation.json'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(directory, name, kind):
    pattern = r'preview-[0-9]{6}\.png' if kind == 'image' else r'exposure-[0-9]{6}\.npz'
    return name if isinstance(name, str) and re.fullmatch(pattern, name) and (directory / name).is_file() else None


def _collation(jobs, digest, config, status):
    """Project actual worker receipts onto immutable parent/child membership."""
    directory = jobs.directory / digest
    contract = None
    if 'tiled' in config:
        from .marble_photo import _tile_contract
        contract = _tile_contract(str(jobs.spectral_root))
    expected = collation_manifest(digest, config, tile_contract=contract)
    if expected is None:
        return None
    path = directory / 'collation.json'
    # Old jobs already retain their admitted config. Read that plan without
    # silently mutating files or adopting a new ownership claim while listing.
    source = 'admission_manifest' if path.exists() else 'saved_config'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else expected
    if manifest != expected:
        raise ValueError('Collation manifest differs from the admitted configuration')
    total = len(manifest['children'])
    state = status.get('state', 'unknown')
    children = []
    completed = 0
    if manifest['kind'] == 'tiles':
        tile_status = status.get('tiled', {})
        completed = tile_status.get('completed_tiles', 0)
        if type(completed) is not int or not 0 <= completed <= total:
            raise ValueError('Invalid completed tile count')
        active = tile_status.get('active_tile')
        active_index = active.get('index') if isinstance(active, dict) else None
        frames = {}
    else:
        frames = {}
        for frame in status.get('frames', []):
            index = frame.get('index')
            if type(index) is not int or not 0 <= index < total or index in frames:
                raise ValueError('Invalid completed camera frame index')
            frames[index] = frame
        active_index = status.get('frame_index')
    for member in manifest['children']:
        index = member['index']
        image = hdr = None
        if manifest['kind'] == 'tiles':
            done = index < completed
        else:
            frame = frames.get(index, {})
            image = _artifact(directory, frame.get('image'), 'image')
            hdr = _artifact(directory, frame.get('hdr'), 'hdr')
            done = bool(image and (not frame.get('hdr') or hdr))
            completed += int(done)
        if done:
            child_state = 'complete'
        elif state in ('interrupted', 'error', 'canceled'):
            child_state = state
        elif index == active_index and state in ('preparing', 'exposing'):
            child_state = state
        else:
            child_state = 'pending'
        child = {**member, 'id': f'{digest}:{index}', 'state': child_state,
                 'result_kind': 'assembled_image' if manifest['kind'] == 'tiles' else 'frame'}
        if image:
            child['image'] = image
        if hdr:
            child['hdr'] = hdr
        children.append(child)
    if state == 'complete' and (completed != total or
            (manifest['kind'] == 'tiles' and not (_artifact(directory, status.get('image'), 'image')
                                                and _artifact(directory, status.get('hdr'), 'hdr')))):
        state = 'error'
    return {'version': 1, 'kind': manifest['kind'], 'source': manifest['source'],
            'manifest_source': source, 'state': state, 'completed': completed, 'total': total,
            'dimensions': manifest['dimensions'], 'children': children,
            'result_kind': 'assembled_image' if manifest['kind'] == 'tiles' else 'frame_set'}


def entries(jobs,owners,account_digests=()):
    with connect(jobs.directory) as db:
        rows=db.execute('SELECT digest,owners,created FROM job_owners ORDER BY created DESC').fetchall()
    selected={d:created for d,stored,created in rows if set(json.loads(stored)).intersection(owners)}
    indexed={d:set(json.loads(stored)) for d,stored,_ in rows}
    for d in account_digests:
        if d not in indexed or indexed[d].intersection(owners):
            selected.setdefault(d,0)
    output=[]
    with jobs.lock:
        for digest,created in sorted(selected.items(),key=lambda item:item[1],reverse=True)[:256]:
            if not __import__('re').fullmatch('[a-f0-9]{64}',digest):continue
            directory=jobs.directory/digest
            try:
                config=json.loads((directory/'config.json').read_text())
                status=json.loads((directory/'status.json').read_text())
                if status.get('digest',digest)!=digest:continue
                if digest==jobs.active_digest:
                    status=jobs._status_locked(digest)
                elif status.get('state') not in ('complete','error','canceled'):
                    status={**status,'state':'interrupted','error':'Worker is no longer owned by this server; no automatic restart.'}
                # Explicit UI fields only; never expose payload paths or owner keys.
                entry={k:status[k] for k in ('state','image','error','frames','hdr','exposure_seconds','geometry_stats','client_work_profile','client_work_state') if k in status}
                try:
                    collation = _collation(jobs, digest, config, status)
                    if collation is not None:
                        entry['collation'] = collation
                        if collation['state'] != status.get('state'):
                            entry.update(state=collation['state'], error='Collated job is missing completed child results.')
                except (OSError, ValueError, KeyError, TypeError):
                    entry.update(state='error', error='Collation metadata is unavailable or inconsistent.')
                output.append({**entry,'digest':digest,'created_at':created,
                    'kind':'live' if config.get('live') else 'photo',
                    'client_intersections':config.get('client_intersections') is True,
                    'can_cancel':digest==jobs.active_digest and jobs.process is not None and jobs.process.poll() is None})
            except (OSError,ValueError):continue
    return {'entries':output,'scheduling':'single_gpu','queue_available':False,'live_yield':False,
            'limit':256,'truncated':len(selected)>256}
