"""Ownership/index metadata only. Scheduling stays with the engine/Nodus host."""
from __future__ import annotations
import json
from pathlib import Path
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
                output.append({**entry,'digest':digest,'created_at':created,
                    'kind':'live' if config.get('live') else 'photo',
                    'client_intersections':config.get('client_intersections') is True,
                    'can_cancel':digest==jobs.active_digest and jobs.process is not None and jobs.process.poll() is None})
            except (OSError,ValueError):continue
    return {'entries':output,'scheduling':'single_gpu','queue_available':False,'live_yield':False,
            'limit':256,'truncated':len(selected)>256}
