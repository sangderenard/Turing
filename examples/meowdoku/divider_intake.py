"""Divider intake: the web-facing front door of source_divider (the page is web/divider.html, the endpoints are server.py [DIVIDER]).

Nothing here compiles, imports, executes or fetches anything.  An upload is validated, written under the job directory, and handed to
source_divider's own job creation (``intake`` Intake + ``submit``, which plans by AST triage and queues the attempts).

    accept_upload(paths, owner_key, filename, data)   one .py (<= 512 KiB) or one .zip (<= 20 MiB compressed) -> job id
    accept_repo_link(paths, owner_key, url, ref)      validation ONLY; records a job in state ``awaiting_fetch``.  The server makes no
                                                      outbound request.  The operator-side fetch step (source_divider.intake_git with
                                                      allow_network=True, then submit) is NOT BUILT; this module never calls it.
    list_jobs / get_job / cancel_job                  owner-only views (an admin sees every job); another owner's job is None.
    run_pending(paths, worker_id, ...)                an explicit hook that runs source_divider.work.  Never called by the server
                                                      unless ``sessions.divider_hook`` is set (off by default).

Zip rule: the whole archive is REFUSED (nothing kept) for an absolute or drive path, a ``..`` part, a symlink or other non-regular
member, more than 2,000 entries, more than 100 MiB declared uncompressed, a compression ratio over 100x, or a kept member over 4 MiB
(source_divider.MAX_FILE).  Members whose suffix is not in KEPT_SUFFIXES are skipped and counted.  The files are written under
``<jobs_dir>/divider/<job-id>/src/``; the .py files are then also stored through source_divider's Intake into
``<jobs_dir>/<job-id>/source/``, which is where source_divider.plan reads them.
"""
from __future__ import annotations

import json
import re
import shutil
import stat
import sys
import threading
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import source_divider as sd                # noqa: E402
from meowdoku.state_paths import StatePaths              # noqa: E402

MAX_PY_BYTES = 512 * 1024
MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_ENTRIES = 2000
MAX_UNCOMPRESSED = 100 * 1024 * 1024
MAX_RATIO = 100
MAX_MEMBER = sd.MAX_FILE                                 # 4 MiB per kept member
KEPT_SUFFIXES = (".py", ".pyi", ".toml", ".cfg", ".txt", ".md", ".json", ".yaml", ".yml")
MAX_QUEUED_PER_ACCOUNT = 5
MAX_PER_DAY = 20
ACTIVE_STATES = ("awaiting_fetch", "intaken", "planned", "running")
REPO_RE = re.compile(r"https://github\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?")
REF_RE = re.compile(r"[A-Za-z0-9._/-]{1,100}")
JOB_ID_RE = re.compile(r"[0-9a-f]{12}")

_LOCK = threading.Lock()


class DividerError(Exception):
    """A refusal with an HTTP status (400 bad input, 413 too large, 429 limit)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ---------------------------------------------------------------------------------------------------------------- validation
def validate_repo(url, ref=None) -> tuple:
    """-> (canonical url, ref or None).  Validation only; DividerError otherwise.  No network."""
    if not isinstance(url, str) or len(url) > 200:
        raise DividerError("repo must be https://github.com/<owner>/<repo>")
    match = REPO_RE.fullmatch(url)
    if match is None or match.group("repo") in (".", "..") or match.group("repo").startswith("."):
        raise DividerError("repo must be https://github.com/<owner>/<repo> (optionally ending .git)")
    if ref in (None, ""):
        ref = None
    elif (not isinstance(ref, str) or REF_RE.fullmatch(ref) is None or ".." in ref or ref.startswith(("/", "-"))
          or ref.endswith(("/", ".lock"))):
        raise DividerError("ref must match [A-Za-z0-9._/-]{1,100} (no '..', no leading '-' or '/')")
    return f"https://github.com/{match.group('owner')}/{match.group('repo')}", ref


def _upload_name(filename) -> str:
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:100]
    if not name or name.startswith("."):
        raise DividerError("X-Filename is required (.py or .zip)")
    return name


def _member_path(name: str) -> str:
    rel = sd._safe_relative(name)
    if rel is None or ":" in rel or any(ord(c) < 32 for c in rel):
        raise DividerError(f"unsafe path in archive: {name!r}")
    return rel


# ----------------------------------------------------------------------------------------------------------------- extraction
def extract_zip(data: bytes, src: Path) -> dict:
    """Safe extraction of ``data`` into ``src``; returns {kept: [rel], skipped: n}.  Any violation raises DividerError (the caller
    removes the job dir)."""
    if len(data) > MAX_ZIP_BYTES:
        raise DividerError("archive exceeds 20 MiB", 413)
    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile:
        raise DividerError("not a valid zip archive")
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            raise DividerError(f"archive has {len(infos)} entries (cap {MAX_ENTRIES})")
        declared = sum(i.file_size for i in infos)
        if declared > MAX_UNCOMPRESSED:
            raise DividerError("archive declares more than 100 MiB uncompressed")
        if declared > MAX_RATIO * max(len(data), 1):
            raise DividerError(f"compression ratio over {MAX_RATIO}x")
        kept, skipped, seen, total = [], 0, set(), 0
        root = src.resolve()
        for info in infos:
            mode = info.external_attr >> 16
            if info.is_dir():
                _member_path(info.filename.rstrip("/\\") or info.filename)
                continue
            rel = _member_path(info.filename)
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):    # no type bits = a plain writestr entry
                raise DividerError(f"not a regular file: {info.filename!r}")
            if not rel.lower().endswith(KEPT_SUFFIXES):
                skipped += 1
                continue
            if rel.lower() in seen:
                raise DividerError(f"duplicate path in archive: {rel!r}")
            seen.add(rel.lower())
            with zf.open(info) as member:
                blob = member.read(MAX_MEMBER + 1)
            if len(blob) > MAX_MEMBER:
                raise DividerError(f"{rel!r} exceeds the {MAX_MEMBER // (1024 * 1024)} MiB per-file cap")
            total += len(blob)
            if total > MAX_UNCOMPRESSED:
                raise DividerError("extracted bytes exceed 100 MiB")
            dest = src / rel
            if root not in dest.resolve().parents:
                raise DividerError(f"path escapes the job directory: {rel!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            kept.append(rel)
    return {"kept": kept, "skipped": skipped}


# --------------------------------------------------------------------------------------------------------------------- limits
def _check_limits(db, owner_key: str, now: float) -> None:
    active = db.execute("SELECT COUNT(*) FROM divider_jobs WHERE owner_key=? AND state IN (%s)" % ",".join("?" * len(ACTIVE_STATES)),
                        (owner_key, *ACTIVE_STATES)).fetchone()[0]
    if active >= MAX_QUEUED_PER_ACCOUNT:
        raise DividerError(f"at most {MAX_QUEUED_PER_ACCOUNT} queued jobs per account", 429)
    recent = db.execute("SELECT COUNT(*) FROM divider_jobs WHERE owner_key=? AND created > ?", (owner_key, now - 86400)).fetchone()[0]
    if recent >= MAX_PER_DAY:
        raise DividerError(f"at most {MAX_PER_DAY} jobs per day", 429)


def _new_id(jobs_dir: Path) -> str:
    while True:
        job_id = uuid.uuid4().hex[:12]
        if not (jobs_dir / job_id).exists() and not (jobs_dir / "divider" / job_id).exists():
            return job_id


# ----------------------------------------------------------------------------------------------------------------------- accept
def accept_upload(paths, owner_key: str, filename: str, data: bytes, backends=("wasm",)) -> str:
    """One .py (<= 512 KiB) or one .zip (<= 20 MiB) -> a source_divider job (planned, attempts queued).  Returns the job id.
    DividerError on refusal; on any failure nothing stays on disk and no row is written."""
    state = StatePaths.coerce(paths)
    name = _upload_name(filename)
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix not in ("py", "zip"):
        raise DividerError("upload a .py file or a .zip archive")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise DividerError("empty upload")
    if suffix == "py" and len(data) > MAX_PY_BYTES:
        raise DividerError("a .py upload is capped at 512 KiB", 413)
    if suffix == "zip" and len(data) > MAX_ZIP_BYTES:
        raise DividerError("a .zip upload is capped at 20 MiB", 413)
    owner_key = str(owner_key)
    db = sd.connect(state.accounts_db)
    try:
        with _LOCK:
            _check_limits(db, owner_key, time.time())
            job_id = _new_id(state.jobs_dir)
            area = state.jobs_dir / "divider" / job_id
            src = area / "src"
            try:
                src.mkdir(parents=True)
                if suffix == "py":
                    (src / name).write_bytes(bytes(data))
                    report = {"kept": [name], "skipped": 0}
                else:
                    report = extract_zip(bytes(data), src)
                pys = [r for r in report["kept"] if r.lower().endswith(".py")]
                if not pys:
                    raise DividerError("no .py files in the upload")
                intake = sd._new_job(state, job_id, f"upload:{name}")
                for rel in pys:
                    sd._store(intake, rel, (src / rel).read_bytes())
                intake.skipped.extend([("(non-source)", report["skipped"])] if report["skipped"] else [])
                (area / "upload.json").write_text(json.dumps({"filename": name, "bytes": len(data), **report}), encoding="utf-8")
                sd.submit(db, state, intake, owner_key=owner_key, backends=backends, label=name)
            except BaseException:
                shutil.rmtree(area, ignore_errors=True)
                shutil.rmtree(state.jobs_dir / job_id, ignore_errors=True)
                raise
            return job_id
    finally:
        db.close()


def accept_repo_link(paths, owner_key: str, url, ref=None) -> str:
    """Validate and record a repository link as a job in state ``awaiting_fetch`` (no pieces).  NO FETCH: the server makes no outbound
    request, and the operator-side fetch step is not built."""
    state = StatePaths.coerce(paths)
    url, ref = validate_repo(url, ref)
    owner_key = str(owner_key)
    db = sd.connect(state.accounts_db)
    try:
        with _LOCK:
            now = time.time()
            _check_limits(db, owner_key, now)
            job_id = _new_id(state.jobs_dir)
            area = state.jobs_dir / "divider" / job_id
            area.mkdir(parents=True)
            (area / "repo.json").write_text(json.dumps({"url": url, "ref": ref, "fetched": False}), encoding="utf-8")
            origin = f"repo:{url}" + (f"@{ref}" if ref else "")
            db.execute("INSERT INTO divider_jobs VALUES (?,?,?,?,?,?,?)",
                       (job_id, owner_key, url.rsplit("/", 1)[-1], origin, now, "awaiting_fetch", "wasm"))
            return job_id
    finally:
        db.close()


# ------------------------------------------------------------------------------------------------------------------------ views
def _visible(row, owner_key: str, admin: bool) -> bool:
    return row is not None and (admin or row["owner_key"] == str(owner_key))


def _first_line(text, limit=200):
    lines = str(text or "").strip().splitlines()
    return (lines[0] if lines else "")[:limit] or None


def _job_row(db, row) -> dict:
    counts = {r[0]: r[1] for r in db.execute("SELECT state, COUNT(*) FROM divider_attempts WHERE job_id=? GROUP BY state", (row["job_id"],))}
    pieces = db.execute("SELECT COUNT(*) FROM divider_pieces WHERE job_id=?", (row["job_id"],)).fetchone()[0]
    return {"job_id": row["job_id"], "label": row["label"], "origin": row["origin"], "state": row["state"], "created": row["created"],
            "backends": [b for b in (row["backends"] or "").split(",") if b], "pieces": pieces, "attempts": counts,
            "active": row["state"] in ACTIVE_STATES}


def list_jobs(paths, owner_key: str, admin: bool = False, limit: int = 100) -> list:
    state = StatePaths.coerce(paths)
    db = sd.connect(state.accounts_db)
    try:
        if admin:
            rows = db.execute("SELECT * FROM divider_jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM divider_jobs WHERE owner_key=? ORDER BY created DESC LIMIT ?", (str(owner_key), limit)).fetchall()
        out = []
        for row in rows:
            item = _job_row(db, row)
            if admin:
                item["owner"] = row["owner_key"]
            out.append(item)
        return out
    finally:
        db.close()


def _piece_state(states: list) -> str:
    for wanted in ("complete", "leased", "queued"):
        if wanted in states:
            return wanted
    return states[0] if states else "none"


def get_job(paths, job_id: str, owner_key: str, admin: bool = False):
    """The job plus its pieces (state, attempts per backend with the lease generation as the attempt count), or None when the job is
    missing or belongs to someone else (the server answers 404 for both)."""
    state = StatePaths.coerce(paths)
    if not JOB_ID_RE.fullmatch(str(job_id)):
        return None
    db = sd.connect(state.accounts_db)
    try:
        row = db.execute("SELECT * FROM divider_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not _visible(row, owner_key, admin):
            return None
        job = _job_row(db, row)
        attempts = {}
        for a in db.execute("SELECT * FROM divider_attempts WHERE job_id=? ORDER BY piece_id, backend", (job_id,)):
            attempts.setdefault(a["piece_id"], []).append(
                {"backend": a["backend"], "state": a["state"], "attempts": a["lease_generation"] or 0, "stage": a["stage"],
                 "error_type": a["error_type"], "error": _first_line(a["error"]), "seconds": a["seconds"]})
        job["pieces_detail"] = [
            {"piece_id": p["piece_id"], "file": p["file"], "names": p["names"], "deps": p["deps"], "nodes": p["nodes"],
             "annotated": p["annotated"], "attempts": attempts.get(p["piece_id"], []),
             "state": _piece_state([a["state"] for a in attempts.get(p["piece_id"], [])])}
            for p in sd.pieces(db, job_id)]
        if row["state"] == "awaiting_fetch":
            job["note"] = "recorded, not fetched: this server makes no outbound request and the operator-side fetch step is not built"
        area = state.jobs_dir / "divider" / job_id
        for key, name in (("upload", "upload.json"), ("repo", "repo.json")):
            try:
                job[key] = json.loads((area / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if "upload" in job:
            job["upload"] = {"kept": len(job["upload"].get("kept", [])), "skipped": job["upload"].get("skipped", 0)}
        return job
    finally:
        db.close()


def cancel_job(paths, job_id: str, owner_key: str, admin: bool = False):
    """Cancel an active job: queued attempts become ``blocked`` (error_type cancelled); a leased attempt finishes on its own.  None
    when not visible; DividerError 409 when the job is no longer active."""
    state = StatePaths.coerce(paths)
    if not JOB_ID_RE.fullmatch(str(job_id)):
        return None
    db = sd.connect(state.accounts_db)
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT * FROM divider_jobs WHERE job_id=?", (job_id,)).fetchone()
            if not _visible(row, owner_key, admin):
                db.execute("ROLLBACK")
                return None
            if row["state"] not in ACTIVE_STATES:
                db.execute("ROLLBACK")
                raise DividerError(f"the job is {row['state']}", 409)
            db.execute("UPDATE divider_attempts SET state='blocked', error_type='cancelled', error='cancelled by the owner', updated=? "
                       "WHERE job_id=? AND state='queued'", (time.time(), job_id))
            db.execute("UPDATE divider_jobs SET state='cancelled' WHERE job_id=?", (job_id,))
            db.execute("COMMIT")
        except DividerError:
            raise
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
    finally:
        db.close()
    return get_job(paths, job_id, owner_key, admin)


# ------------------------------------------------------------------------------------------------------------------------- hook
def run_pending(paths, worker_id: str = "divider-hook", max_attempts: int = 1, **kwargs) -> int:
    """EXPLICIT, off by default: claim and run up to ``max_attempts`` queued attempts through source_divider.work (which runs each in
    a subprocess with a hard timeout).  The server calls this only when ``sessions.divider_hook`` is set; nothing here starts a thread."""
    state = StatePaths.coerce(paths)
    db = sd.connect(state.accounts_db)
    try:
        return sd.work(db, state, worker_id, max_attempts, **kwargs)
    finally:
        db.close()
