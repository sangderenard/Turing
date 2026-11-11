"""Source divider: divide a Python source into pieces, try every piece on any combination of backends, recompose.

A job is one intaken source tree (``<state>/jobs/<job_id>/source/``, .py files only).  It is never imported or executed here, only
read as text and ``ast.parse``d.  Pipeline:

  intake_file / intake_zip / intake_text / intake_git / intake_dream  ->  source tree (a dream document's python
                blocks become one file each; its other blocks are carried into the record unchanged)
  plan          per file, ``project_compilation_product.authored_compilation_growth_plan`` (cheap AST TRIAGE, not the resolved
                FunctionTable plan: the real cut decisions are made later by the compiler) -> pieces, dependencies first.  A
                piece is one unit (an SCC of the call graph): file, qualified names, dependency piece ids, AST node count,
                ``annotated`` (every formal annotated; an unannotated piece is not attempted: verdict ``needs-annotation``).
                Dependencies are resolved within one file only; cross-file imports are not followed in this slice.
  queue         ``divider_attempts`` (one row per piece x backend) in accounts.db.  ``claim_next`` hands out ONE attempt at a
                time under BEGIN IMMEDIATE, dependencies first, with a lease generation + deadline.  THIS QUEUE IS WHAT A NODUS
                COMPILE HOST WILL DRAW FROM LATER: it claims with its own worker_id, runs the attempt anywhere, and reports back
                with ``complete`` / ``refuse`` (matching worker + generation) or loses the lease to ``reclaim_expired``.
  run_attempt   one attempt in a SUBPROCESS with a hard timeout: ``partition_authored_source`` (exact lexical closure of one call)
                -> ``lower_ast_source_to_ssa`` (the sanctioned entry, program extraction contract) -> backend emitter.  A hang
                or crash never takes the worker down.
  work          claim one, run, record, repeat.  Single in flight.
  recompose     ``summary`` (verdict table) and ``write_recomposition`` (recomposition.json: per authored function ``python`` or
                its available lowerings).  NO CODE IS REWRITTEN in this slice; the file is the data a later rewrite would read.
  goal          ``write_goal_document`` (goal.goal, a ``speaktome.goal-document.v1``): the units LINKED.  Each unit
                holds its own text only; ``link`` rows name what it takes from other units; ``api`` / ``build`` / ``abi``
                tables say what it offers, which lowerings exist and how they are called; the document's ``target``
                table says for which language every unit is built (the goal achieved).
  record        ``write_dream_record`` (record.dream, a ``turing.dream-document.v1``): the job's enduring record.  One
                python block per piece holding its exact lexical closure, so every part builds on its own; one block per
                completed lowering in that backend's language, byte for byte; refusals on the piece's header.  The
                record can be given back to ``intake_dream``: dream in, dream out.

Backends (SOURCE emission only, no native toolchain build here): ``wasm`` (binary + api), ``js`` (module + api), ``c``, ``llvm``,
``fortran`` through the same emitter calls the Nodus row compiler uses.  Success is state ``complete`` with the note
``emitted-not-built`` for c/llvm/fortran; an incomplete emission is ``refused`` with the emitter's named shortfall.  Only products
that carry an ``.api.yaml`` can be catalogued in lowering_library (family ``divider``); c/llvm artifacts carry none, so those are
recorded as attempts but not catalogued.  An unimportable emitter is ``refused`` with error_type ``backend-unavailable``.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import lowering_library as lowering_mod   # noqa: E402
from meowdoku import schema_version as _schema_version  # noqa: E402
from meowdoku.state_paths import StatePaths             # noqa: E402

SCHEMA_RECOMPOSITION = "turing.divider-recomposition.v1"
SCHEMA_DREAM_RECORD = "turing.divider-dream-record.v1"
# The language a backend's product is written in, as a dream block names it. wasm is a binary: its block is carried
# byte for byte under the latin-1 encoding, which maps every byte to one character and back.
DREAM_LANGUAGE = {"js": "javascript", "c": "c", "llvm": "llvm", "fortran": "fortran", "wasm": "wasm"}
SCHEMA_PLAN = "turing.divider-plan-triage.v1"
BACKENDS = ("wasm", "js", "c", "llvm", "fortran")
NOT_BUILT = ("c", "llvm", "fortran")
MAX_FILES, MAX_TOTAL, MAX_FILE = 2000, 64 * 1024 * 1024, 4 * 1024 * 1024
DEFAULT_TIMEOUT_S = 90.0
EXTRACTION_CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"
FAILED = ("refused", "error", "needs-annotation", "blocked")

DDL = (
    "CREATE TABLE IF NOT EXISTS divider_jobs (job_id TEXT PRIMARY KEY, owner_key TEXT NOT NULL, label TEXT, origin TEXT, "
    "created REAL, state TEXT, backends TEXT)",
    "CREATE TABLE IF NOT EXISTS divider_pieces (job_id TEXT, piece_id INTEGER, file TEXT, names TEXT, deps TEXT, nodes INTEGER, "
    "annotated INTEGER, ord INTEGER, PRIMARY KEY (job_id, piece_id))",
    "CREATE TABLE IF NOT EXISTS divider_attempts (job_id TEXT, piece_id INTEGER, backend TEXT, state TEXT CHECK(state IN "
    "('queued','leased','complete','refused','error','needs-annotation','blocked')), lease_owner TEXT, lease_generation INTEGER, "
    "lease_deadline REAL, stage TEXT, error_type TEXT, error TEXT, seconds REAL, product_sha256 TEXT, api_path TEXT, updated REAL, "
    "PRIMARY KEY (job_id, piece_id, backend))",
)


def connect(db_path) -> sqlite3.Connection:
    """Own connection to accounts.db (autocommit; transactions are explicit).  Declares schema module ``source_divider`` 1."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "source_divider", 1)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    return db


# ------------------------------------------------------------------------------------------------------------------ intake
class IntakeError(ValueError):
    pass


@dataclass
class Intake:
    job_id: str
    job_dir: Path
    source_dir: Path
    origin: str
    kept: list = field(default_factory=list)
    skipped: list = field(default_factory=list)       # [(name, reason)]


def _new_job(state, job_id=None, origin=""):
    state = StatePaths.coerce(state)
    job_id = job_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise IntakeError(f"bad job id {job_id!r}")
    job_dir = state.jobs_dir / job_id
    source = job_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    return Intake(job_id, job_dir, source, origin)


def _safe_relative(name: str) -> str | None:
    """The member name as a safe relative posix path, or None (absolute, drive, empty, or any '..' part)."""
    n = str(name).replace("\\", "/")
    if not n or n.startswith("/") or re.match(r"^[A-Za-z]:", n):
        return None
    parts = PurePosixPath(n).parts
    if not parts or any(part in ("..", "") for part in parts):
        return None
    return "/".join(parts)


def _store(intake: Intake, rel: str, data: bytes) -> None:
    dest = (intake.source_dir / rel)
    root = intake.source_dir.resolve()
    if root not in dest.resolve().parents:
        raise IntakeError(f"path escapes the source tree: {rel!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    intake.kept.append(rel)


def intake_text(state, name: str, text: str, *, job_id=None) -> Intake:
    intake = _new_job(state, job_id, f"text:{name}")
    rel = _safe_relative(name if str(name).endswith(".py") else f"{name}.py")
    if rel is None:
        raise IntakeError(f"unsafe name {name!r}")
    data = text.encode("utf-8")
    if len(data) > MAX_FILE:
        raise IntakeError("text exceeds the per-file cap")
    _store(intake, rel, data)
    return intake


DREAM_PRODUCT_ROLES = ("lowering", "job-record")      # blocks an earlier pass of the divider wrote; never taken as source


def intake_dream(state, path, *, job_id=None) -> Intake:
    """A dream document as the source of a job.

    Every python block becomes one source file ``<block id>.py``; a block is then divided on its own, because pieces
    are cut within one file. Blocks in other languages are not divided: the document is kept as ``intake.dream`` and
    ``write_dream_record`` carries those blocks into the record unchanged. Blocks that an earlier pass of the divider
    produced (role lowering or job-record) are dropped, so a record can be fed back in.
    """
    from src.compiler.dream_document import DreamDocumentError, parse_dream_document
    path = Path(path)
    if not path.is_file():
        raise IntakeError(f"not a file: {path}")
    if path.stat().st_size > MAX_TOTAL:
        raise IntakeError("the dream document exceeds the total cap")
    raw = path.read_bytes()
    try:
        document = parse_dream_document(raw)
    except DreamDocumentError as error:
        raise IntakeError(f"not a dream document: {error}") from error
    intake = _new_job(state, job_id, f"dream:{path.name}")
    for block in document.blocks:
        role = block.decorations.get("role")
        if role in DREAM_PRODUCT_ROLES:
            intake.skipped.append((block.identity, f"product of an earlier pass ({role})"))
            continue
        if block.language != "python":
            if block.identity == "record" or block.identity.startswith("piece-"):
                raise IntakeError(f"block {block.identity!r} uses an identity the record reserves")
            intake.skipped.append((block.identity, f"carried unchanged ({block.language})"))
            continue
        rel = _safe_relative(f"{block.identity}.py")
        if rel is None or "/" in rel:
            raise IntakeError(f"block identity {block.identity!r} cannot name a source file")
        data = block.payload.encode("utf-8")
        if len(data) > MAX_FILE:
            raise IntakeError(f"block {block.identity!r} exceeds the per-file cap")
        _store(intake, rel, data)
    if not intake.kept:
        raise IntakeError("the dream document has no python block to divide")
    (intake.job_dir / "intake.dream").write_bytes(raw)
    return intake


def intake_file(state, path, *, job_id=None) -> Intake:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise IntakeError(f"not a regular file: {path}")
    if path.suffix != ".py":
        raise IntakeError(f"not a .py file: {path.name}")
    if path.stat().st_size > MAX_FILE:
        raise IntakeError("file exceeds the per-file cap")
    intake = _new_job(state, job_id, f"file:{path.name}")
    _store(intake, path.name, path.read_bytes())
    return intake


def intake_zip(state, path, *, job_id=None) -> Intake:
    """Safe extraction: members with absolute/drive/'..' names or symlink mode are skipped (recorded with a reason); only .py are
    kept (others skipped 'not-python'); > 2000 members or > 64 MiB declared total is an IntakeError; a member over 4 MiB (by the
    bytes actually read, not the header) is skipped 'too-large'."""
    with zipfile.ZipFile(path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES:
            raise IntakeError(f"archive has {len(infos)} files (cap {MAX_FILES})")
        if sum(i.file_size for i in infos) > MAX_TOTAL:
            raise IntakeError(f"archive declares more than {MAX_TOTAL} bytes")
        intake = _new_job(state, job_id, f"zip:{Path(path).name}")
        total = 0
        for info in infos:
            rel = _safe_relative(info.filename)
            if rel is None:
                intake.skipped.append((info.filename, "unsafe-path"))
            elif stat.S_ISLNK(info.external_attr >> 16):
                intake.skipped.append((info.filename, "symlink"))
            elif not rel.endswith(".py"):
                intake.skipped.append((info.filename, "not-python"))
            else:
                with zf.open(info) as member:
                    data = member.read(MAX_FILE + 1)
                if len(data) > MAX_FILE:
                    intake.skipped.append((info.filename, "too-large"))
                    continue
                total += len(data)
                if total > MAX_TOTAL:
                    raise IntakeError("extracted bytes exceed the total cap")
                _store(intake, rel, data)
    return intake


def intake_git(state, url: str, *, allow_network: bool = False, job_id=None, timeout_s: float = 60.0) -> Intake:
    """REFUSES unless ``allow_network=True``: cloning is an operator action.  https URLs only; ``git clone --depth 1`` into the job
    dir with a 60 s timeout, then everything but regular .py files is deleted (and recorded skipped)."""
    if not allow_network:
        raise IntakeError("intake_git refuses: cloning is an operator action; pass allow_network=True explicitly")
    if not re.fullmatch(r"https://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+", str(url)) or " " in url:
        raise IntakeError("intake_git accepts https URLs only")
    intake = _new_job(state, job_id, f"git:{url}")
    clone = intake.job_dir / "clone"
    proc = subprocess.run(["git", "clone", "--depth", "1", "--", url, str(clone)], capture_output=True, text=True,
                          timeout=timeout_s)
    if proc.returncode != 0:
        raise IntakeError(f"git clone failed: {proc.stderr.strip().splitlines()[-1:] or proc.returncode}")
    count = total = 0
    for p in sorted(clone.rglob("*")):
        rel = p.relative_to(clone).as_posix()
        if ".git" in p.relative_to(clone).parts or p.is_dir():
            continue
        if p.is_symlink():
            intake.skipped.append((rel, "symlink"))
        elif p.suffix != ".py":
            intake.skipped.append((rel, "not-python"))
        elif p.stat().st_size > MAX_FILE:
            intake.skipped.append((rel, "too-large"))
        else:
            count, total = count + 1, total + p.stat().st_size
            if count > MAX_FILES or total > MAX_TOTAL:
                raise IntakeError("repository exceeds the file or size cap")
            _store(intake, rel, p.read_bytes())
    shutil.rmtree(clone, ignore_errors=True)
    return intake


# ------------------------------------------------------------------------------------------------------------------- plan
def _function_nodes(tree: ast.AST) -> dict:
    """qualified name -> (node, parent_is_class), matching the authored catalogue's spelling (C.m, f.<locals>.g)."""
    found = {}

    def walk(body, prefix, in_class):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{node.name}"
                found[q] = (node, in_class)
                walk(node.body, f"{q}.<locals>.", False)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.", True)

    walk(tree.body, "", False)
    return found


def _annotated(node, in_class: bool) -> bool:
    a = node.args
    formals = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs) + [x for x in (a.vararg, a.kwarg) if x]
    if in_class and (a.posonlyargs or a.args):
        formals.remove((a.posonlyargs or a.args)[0])           # self / cls
    return all(f.annotation is not None for f in formals)


def _read_source(path: Path):
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def plan(db, state, job_id: str) -> list[dict]:
    """Triage every .py file into pieces and queue one attempt per piece x backend (idempotent: an existing plan is returned).

    This is ``authored_compilation_growth_plan``: a cheap AST triage, NOT the resolved plan; written to ``plan.json`` as such."""
    from src.compiler.project_compilation_product import authored_compilation_growth_plan
    state = StatePaths.coerce(state)
    job = dict(db.execute("SELECT * FROM divider_jobs WHERE job_id=?", (job_id,)).fetchone() or {})
    if not job:
        raise KeyError(f"no such job {job_id!r}")
    existing = pieces(db, job_id)
    if existing:
        return existing
    job_dir = state.jobs_dir / job_id
    source_dir = job_dir / "source"
    backends = [b for b in job["backends"].split(",") if b]
    rows, files, next_id, ordinal = [], [], 0, 0
    for path in sorted(source_dir.rglob("*.py")):
        rel = path.relative_to(source_dir).as_posix()
        text = _read_source(path)
        if text is None:
            files.append({"file": rel, "skipped": "not-utf8"})
            continue
        try:
            tree = ast.parse(text)
            triage = authored_compilation_growth_plan(text)
        except (SyntaxError, ValueError, RecursionError) as error:
            files.append({"file": rel, "skipped": f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"})
            continue
        nodes = _function_nodes(tree)
        base = next_id
        ids = {}
        for index, unit in enumerate(triage["units"]):
            ids[index] = base + index
        for index, unit in enumerate(triage["units"]):
            names = list(unit["qualified_names"])
            ann = all(_annotated(*nodes[n]) for n in names if n in nodes)
            rows.append({"piece_id": ids[index], "file": rel, "names": names,
                         "deps": [ids[d] for d in unit["dependency_units"]], "nodes": int(unit["source_nodes"]),
                         "annotated": ann, "ord": ordinal})
            ordinal += 1
        next_id = base + len(triage["units"])
        files.append({"file": rel, "units": len(triage["units"]), "triage_source_sha256": triage["source_sha256"]})
    now = time.time()
    db.execute("BEGIN IMMEDIATE")
    try:
        for r in rows:
            db.execute("INSERT OR IGNORE INTO divider_pieces VALUES (?,?,?,?,?,?,?,?)",
                       (job_id, r["piece_id"], r["file"], json.dumps(r["names"]), json.dumps(r["deps"]), r["nodes"],
                        int(r["annotated"]), r["ord"]))
            for backend in backends:
                db.execute("INSERT OR IGNORE INTO divider_attempts (job_id,piece_id,backend,state,stage,updated) "
                           "VALUES (?,?,?,?,?,?)",
                           (job_id, r["piece_id"], backend, "queued" if r["annotated"] else "needs-annotation",
                            None if r["annotated"] else "plan", now))
        db.execute("UPDATE divider_jobs SET state='planned' WHERE job_id=?", (job_id,))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    (job_dir / "plan.json").write_text(json.dumps(
        {"schema": SCHEMA_PLAN, "kind": "triage", "note": "authored_compilation_growth_plan is cheap AST triage, not the resolved "
         "FunctionTable plan; cross-file imports are not followed", "files": files, "pieces": rows}, indent=1), encoding="utf-8")
    _refresh_job(db, job_id)
    return pieces(db, job_id)


def pieces(db, job_id: str) -> list[dict]:
    out = []
    for r in db.execute("SELECT * FROM divider_pieces WHERE job_id=? ORDER BY ord, piece_id", (job_id,)):
        d = dict(r)
        d["names"], d["deps"], d["annotated"] = json.loads(d["names"]), json.loads(d["deps"]), bool(d["annotated"])
        out.append(d)
    return out


def submit(db, state, intake: Intake, *, owner_key: str, backends=("wasm",), label: str | None = None) -> str:
    """Register an intaken tree as a job, plan it and queue its attempts.  Returns the job id."""
    backends = tuple(backends)
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown or not backends:
        raise ValueError(f"unknown backends {unknown!r}; supported {BACKENDS}")
    db.execute("INSERT INTO divider_jobs VALUES (?,?,?,?,?,?,?)",
               (intake.job_id, owner_key, label, intake.origin, time.time(), "intaken", ",".join(backends)))
    (intake.job_dir / "intake.json").write_text(json.dumps({"kept": intake.kept, "skipped": intake.skipped}, indent=1),
                                                encoding="utf-8")
    plan(db, state, intake.job_id)
    return intake.job_id


# ------------------------------------------------------------------------------------------------------------------- queue
def _refresh_job(db, job_id):
    open_ = db.execute("SELECT COUNT(*) FROM divider_attempts WHERE job_id=? AND state IN ('queued','leased')",
                       (job_id,)).fetchone()[0]
    db.execute("UPDATE divider_jobs SET state=? WHERE job_id=? AND state IN ('planned','running','done')",
               ("running" if open_ else "done", job_id))


def claim_next(db, worker_id: str, seconds: float, now: float | None = None) -> dict | None:
    """Lease ONE queued attempt (dependencies first), or None.  Dependencies on the same backend must be out of queued/leased;
    a dependency that was refused/errored does not block: the attempt is still made and ``dep_refused`` lists those piece ids
    (also written into the stage note)."""
    now = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        candidates = db.execute(
            "SELECT a.*, p.deps, p.names, p.file, p.ord FROM divider_attempts a JOIN divider_pieces p "
            "ON p.job_id=a.job_id AND p.piece_id=a.piece_id JOIN divider_jobs j ON j.job_id=a.job_id "
            "WHERE a.state='queued' ORDER BY j.created, p.ord, a.piece_id, a.backend").fetchall()
        for c in candidates:
            states = {}
            for dep in json.loads(c["deps"]):
                row = db.execute("SELECT state FROM divider_attempts WHERE job_id=? AND piece_id=? AND backend=?",
                                 (c["job_id"], dep, c["backend"])).fetchone()
                if row:
                    states[dep] = row["state"]
            if any(s in ("queued", "leased") for s in states.values()):
                continue
            refused = sorted(d for d, s in states.items() if s in FAILED)
            generation = (c["lease_generation"] or 0) + 1
            stage = "leased" + (f" dep_refused={refused}" if refused else "")
            db.execute("UPDATE divider_attempts SET state='leased', lease_owner=?, lease_generation=?, lease_deadline=?, stage=?, "
                       "updated=? WHERE job_id=? AND piece_id=? AND backend=?",
                       (worker_id, generation, now + seconds, stage, now, c["job_id"], c["piece_id"], c["backend"]))
            db.execute("UPDATE divider_jobs SET state='running' WHERE job_id=? AND state IN ('planned','done')", (c["job_id"],))
            db.execute("COMMIT")
            return {"job_id": c["job_id"], "piece_id": c["piece_id"], "backend": c["backend"], "generation": generation,
                    "deadline": now + seconds, "worker_id": worker_id, "names": json.loads(c["names"]), "file": c["file"],
                    "dep_refused": refused}
        db.execute("COMMIT")
        return None
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


def _finish(db, state_name, job_id, piece_id, backend, worker_id, generation, *, stage=None, error_type=None, error=None,
            seconds=None, product_sha256=None, api_path=None) -> bool:
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT stage FROM divider_attempts WHERE job_id=? AND piece_id=? AND backend=? AND state='leased' "
                         "AND lease_owner=? AND lease_generation=?", (job_id, piece_id, backend, worker_id, generation)).fetchone()
        if row is None:
            db.execute("ROLLBACK")
            return False
        note = re.search(r" dep_refused=\[[^\]]*\]", row["stage"] or "")
        if note and stage is not None:
            stage = stage + note.group(0)
        db.execute("UPDATE divider_attempts SET state=?, lease_deadline=NULL, stage=?, error_type=?, error=?, seconds=?, "
                   "product_sha256=?, api_path=?, updated=? WHERE job_id=? AND piece_id=? AND backend=?",
                   (state_name, stage if stage is not None else row["stage"], error_type, error, seconds, product_sha256,
                    api_path, time.time(), job_id, piece_id, backend))
        _refresh_job(db, job_id)
        db.execute("COMMIT")
        return True
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


def complete(db, job_id, piece_id, backend, worker_id, generation, **fields) -> bool:
    """Record success; False (nothing written) unless the worker and lease generation match the live lease."""
    return _finish(db, "complete", job_id, piece_id, backend, worker_id, generation, **fields)


def refuse(db, job_id, piece_id, backend, worker_id, generation, *, state="refused", **fields) -> bool:
    """Record a refusal (``state`` refused|error|blocked); same worker + generation rule as ``complete``."""
    if state not in ("refused", "error", "blocked"):
        raise ValueError(state)
    return _finish(db, state, job_id, piece_id, backend, worker_id, generation, **fields)


def reclaim_expired(db, now: float | None = None) -> int:
    """Requeue leased attempts past their deadline (the generation is kept, so the old worker's report is refused)."""
    now = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    cur = db.execute("UPDATE divider_attempts SET state='queued', lease_owner=NULL, lease_deadline=NULL, stage='reclaimed', "
                     "updated=? WHERE state='leased' AND lease_deadline < ?", (now, now))
    db.execute("COMMIT")
    return cur.rowcount


# ---------------------------------------------------------------------------------------------------------------- attempts
def _stem(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_") or "entry"


def _first_line(text, limit=300) -> str:
    lines = str(text).strip().splitlines()
    return (lines[0] if lines else "")[:limit]


def _attempt_entry(source: str, name: str, backend: str, out: Path, mark) -> dict:
    """In the child: one entry, one backend.  Returns an entry record; raises with ``mark`` having set the stage."""
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import partition_authored_source

    stem = _stem(name)
    mark("partition")
    compile_source, _retained = partition_authored_source(source, name)
    mark("ssa-lowering")
    module, outputs, exports = lower_ast_source_to_ssa(
        compile_source, name, name=stem, extraction_contract=ExtractionContract(EXTRACTION_CONTRACT), progress_bars=False)
    entry = exports[0]
    mark("emission")
    out.mkdir(parents=True, exist_ok=True)
    api_path, product = None, None
    try:
        if backend == "wasm":
            from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core as emit
        elif backend == "js":
            from src.compiler.ssa_javascript_backend import emit_ssa_module_to_javascript as emit
        elif backend == "c":
            from src.compiler.ssa_c_backend import emit_ssa_module_to_c as emit
        elif backend == "llvm":
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm as emit
        elif backend == "fortran":
            from src.compiler.ssa_fortran_backend import emit_module as emit
        else:
            raise ImportError(f"no emitter for {backend}")
    except ImportError as error:
        return {"name": name, "state": "refused", "stage": "emission", "error_type": "backend-unavailable",
                "error": _first_line(error)}
    if backend == "wasm":
        artifact = emit(module, entry, entry_name=entry)
    elif backend == "js":
        artifact = emit(module, entry)
    elif backend in ("c", "llvm"):
        artifact = emit(module, entry, entry_name=stem)
    else:
        artifact = emit(module, name=stem, outputs=outputs, extra_roots=tuple(exports))
    shortfalls = tuple(getattr(artifact, "shortfalls", ()) or ())
    if not bool(getattr(artifact, "complete", not shortfalls)):
        return {"name": name, "state": "refused", "stage": "emission", "error_type": "EmissionShortfall",
                "error": _first_line(f"{len(shortfalls)} shortfall(s): {shortfalls[0] if shortfalls else ''}")}
    if backend == "wasm":
        product = out / f"{stem}.wasm"
        product.write_bytes(artifact.binary)
    elif backend == "js":
        product = out / f"{stem}.mjs"
        product.write_text(artifact.source, encoding="utf-8")
    elif backend == "c":
        product = out / f"{stem}.c"
        product.write_text(str(artifact.source), encoding="utf-8")
    elif backend == "llvm":
        product = out / f"{stem}.ll"
        product.write_text(str(artifact.llvm_ir), encoding="utf-8")
    else:
        product = Path(artifact.write(out))
    api = getattr(artifact, "api", None)
    if api is not None:
        api_path = out / f"{stem}.api.yaml"
        api.write(api_path)
    return {"name": name, "state": "complete", "stage": "emission", "entry": entry,
            "product": product.name, "api": api_path.name if api_path else None,
            "sha256": hashlib.sha256(product.read_bytes()).hexdigest(),
            "note": "emitted-not-built" if backend in NOT_BUILT else None}


def _attempt_main(spec_path: str) -> int:
    """Child process: run every entry of one piece on one backend, write ``attempt.json`` (and stage.txt as it goes)."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(spec["out_dir"])
    stage_file = out / "stage.txt"
    source = Path(spec["source_file"]).read_text(encoding="utf-8-sig")
    entries, current = [], {"stage": "start"}

    def mark(stage):
        current["stage"] = stage
        stage_file.write_text(stage, encoding="utf-8")

    for name in spec["names"]:
        try:
            entries.append(_attempt_entry(source, name, spec["backend"], out / _stem(name) if len(spec["names"]) > 1 else out,
                                          mark))
            if len(spec["names"]) > 1:
                for key in ("product", "api"):
                    if entries[-1].get(key):
                        entries[-1][key] = f"{_stem(name)}/{entries[-1][key]}"
        except BaseException as error:            # the record is the point: type, first line, stage where it raised
            entries.append({"name": name, "state": "error", "stage": current["stage"], "error_type": type(error).__name__,
                            "error": _first_line(error)})
    (out / "attempt.json").write_text(json.dumps({"backend": spec["backend"], "entries": entries}, indent=1), encoding="utf-8")
    return 0


def _kill_tree(proc) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    proc.kill()


def run_attempt(job: dict, piece: dict, backend: str, out_dir, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run one piece on one backend in a subprocess with a hard timeout (killed on timeout: state ``error``, ``Timeout``).

    ``job`` needs ``source_dir``; ``piece`` needs ``file`` and ``names``.  Returns a result dict: state (complete|refused|error),
    stage, error_type, error, seconds, product_sha256, api_path (relative to ``out_dir``'s job), entries, note."""
    out = Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir()
    spec = {"source_file": str(Path(job["source_dir"]) / piece["file"]), "names": list(piece["names"]), "backend": backend,
            "out_dir": str(out)}
    spec_path = work / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    started = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "examples")]))
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_attempt", str(spec_path)], cwd=str(work),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    result = {"entries": []}
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        stage = (out / "stage.txt").read_text(encoding="utf-8") if (out / "stage.txt").exists() else "start"
        result.update(state="error", stage=stage, error_type="Timeout", error=f"killed after {timeout_s:g} s",
                      seconds=time.time() - started)
        return result
    result["seconds"] = time.time() - started
    attempt = out / "attempt.json"
    if proc.returncode != 0 or not attempt.exists():
        stage = (out / "stage.txt").read_text(encoding="utf-8") if (out / "stage.txt").exists() else "start"
        result.update(state="error", stage=stage, error_type="ChildCrash",
                      error=_first_line(f"exit {proc.returncode}: {(output or '').strip().splitlines()[-1:] or ''}"))
        return result
    entries = json.loads(attempt.read_text(encoding="utf-8"))["entries"]
    result["entries"] = entries
    bad = next((e for e in entries if e["state"] != "complete"), None)
    if bad is not None:
        result.update(state="refused" if bad["state"] == "refused" else "error", stage=bad["stage"],
                      error_type=bad["error_type"], error=bad["error"])
        return result
    digest = (entries[0]["sha256"] if len(entries) == 1
              else hashlib.sha256("".join(e["sha256"] for e in entries).encode()).hexdigest())
    apis = [e["api"] for e in entries if e.get("api")]
    result.update(state="complete", stage="emission", product_sha256=digest, api_path=apis[0] if apis else None,
                  note=entries[0].get("note"))
    return result


def _register(state, job_id, piece, backend, entries, out_dir):
    """Catalogue each complete entry that carries an api.yaml in lowering_library (family 'divider'); returns core ids."""
    state = StatePaths.coerce(state)
    cores = []
    for e in entries:
        if e["state"] != "complete" or not e.get("api"):
            continue
        api = Path(out_dir) / e["api"]
        record = lowering_mod.describe_product(api, state.root)
        product = Path(out_dir) / e["product"]
        record = dataclasses.replace(
            record, family="divider", name=f"{piece['file']}:{e['name']}", source_function=e["name"],
            core_id=f"divider:{e['sha256'][:32]}", product_path=product.relative_to(state.root).as_posix(),
            product_sha256=e["sha256"], verdict="built")
        lowering_mod.register(state.accounts_db, record)
        lowering_mod.store_blob(state.accounts_db, product)
        cores.append(record.core_id)
    return cores


def work(db, state, worker_id: str, max_attempts: int | None = None, *, runner=None, timeout_s: float = DEFAULT_TIMEOUT_S,
         lease_seconds: float | None = None) -> int:
    """Claim one, run, record, repeat (single in flight); stops when nothing is claimable or after ``max_attempts``.
    ``runner`` defaults to ``run_attempt``; returns the number of attempts processed."""
    state = StatePaths.coerce(state)
    runner = runner or run_attempt
    done = 0
    while max_attempts is None or done < max_attempts:
        reclaim_expired(db)
        claim = claim_next(db, worker_id, lease_seconds or (timeout_s + 30))
        if claim is None:
            break
        job_dir = state.jobs_dir / claim["job_id"]
        piece = next(p for p in pieces(db, claim["job_id"]) if p["piece_id"] == claim["piece_id"])
        out = job_dir / "products" / str(claim["piece_id"]) / claim["backend"]
        try:
            result = runner({"job_id": claim["job_id"], "source_dir": str(job_dir / "source")}, piece, claim["backend"], out,
                            timeout_s)
        except Exception as error:
            result = {"state": "error", "stage": "runner", "error_type": type(error).__name__, "error": _first_line(error),
                      "seconds": None, "entries": []}
        key = (claim["job_id"], claim["piece_id"], claim["backend"], worker_id, claim["generation"])
        fields = dict(stage=result.get("stage"), seconds=result.get("seconds"), product_sha256=result.get("product_sha256"),
                      api_path=(str((out / result["api_path"]).relative_to(job_dir).as_posix())
                                if result.get("api_path") else None))
        if result["state"] == "complete":
            if result.get("note"):
                fields["stage"] = f"{fields['stage']} ({result['note']})"
            ok = complete(db, *key, **fields)
        else:
            ok = refuse(db, *key, state=result["state"], error_type=result.get("error_type"), error=result.get("error"), **fields)
        if ok and result.get("entries"):
            _register(state, claim["job_id"], piece, claim["backend"], result["entries"], out)
        done += 1
    return done


# -------------------------------------------------------------------------------------------------------------- recompose
def _rank(backend):
    order = tuple(lowering_mod.BACKEND_ORDER)
    return order.index(backend) if backend in order else len(order)


def summary(db, job_id: str) -> dict:
    """Per piece: the verdict per backend, the best lowering (lowering_library backend order), or why it stays Python."""
    job = db.execute("SELECT * FROM divider_jobs WHERE job_id=?", (job_id,)).fetchone()
    if job is None:
        raise KeyError(job_id)
    attempts = {}
    for r in db.execute("SELECT * FROM divider_attempts WHERE job_id=?", (job_id,)):
        attempts.setdefault(r["piece_id"], {})[r["backend"]] = dict(r)
    out, counts = [], {}
    for p in pieces(db, job_id):
        per = {}
        for backend, a in sorted(attempts.get(p["piece_id"], {}).items()):
            per[backend] = {k: a[k] for k in ("state", "stage", "error_type", "error", "seconds", "product_sha256", "api_path")}
            counts[a["state"]] = counts.get(a["state"], 0) + 1
        good = sorted((b for b, v in per.items() if v["state"] == "complete"), key=_rank)
        if good:
            reason = None
        elif not p["annotated"]:
            reason = "needs-annotation"
        else:
            parts = [f"{b}: {v['state']}" + (f" {v['error_type']}@{v['stage']}" if v["error_type"] else "") for b, v in per.items()]
            reason = "; ".join(parts) or "no attempts"
        out.append({**p, "attempts": per, "best": good[0] if good else None, "stays_python": not good, "python_reason": reason})
    return {"job": dict(job), "pieces": out, "attempt_counts": counts,
            "lowered_pieces": sum(1 for p in out if p["best"]), "python_pieces": sum(1 for p in out if p["stays_python"])}


def write_recomposition(db, state, job_id: str) -> Path:
    """``<job>/recomposition.json`` (``turing.divider-recomposition.v1``): per authored function ``python`` or its lowerings with
    sha256 and api path.  Data only: no source is rewritten in this slice."""
    state = StatePaths.coerce(state)
    job_dir = state.jobs_dir / job_id
    functions = []
    for p in summary(db, job_id)["pieces"]:
        for name in p["names"]:
            lowered = []
            for backend in sorted(p["attempts"], key=_rank):
                record = job_dir / "products" / str(p["piece_id"]) / backend / "attempt.json"
                if not record.exists():
                    continue
                for e in json.loads(record.read_text(encoding="utf-8"))["entries"]:
                    if e["name"] == name and e["state"] == "complete":
                        base = job_dir / "products" / str(p["piece_id"]) / backend
                        lowered.append({"backend": backend, "sha256": e["sha256"], "note": e.get("note"),
                                        "product": (base / e["product"]).relative_to(job_dir).as_posix(),
                                        "api": (base / e["api"]).relative_to(job_dir).as_posix() if e.get("api") else None})
            functions.append({"file": p["file"], "name": name, "piece_id": p["piece_id"],
                              "python": not lowered, "reason": None if lowered else p["python_reason"], "lowerings": lowered})
    path = job_dir / "recomposition.json"
    path.write_text(json.dumps({"schema": SCHEMA_RECOMPOSITION, "job_id": job_id, "rewrites_source": False,
                                "functions": functions}, indent=1), encoding="utf-8")
    return path


def _dream_value(text, limit=240) -> str:
    """A fact as one restricted-ASCII header value (the scanner accepts nothing else in a header)."""
    flat = " ".join(str(text).split())
    flat = "".join(ch if 0x20 <= ord(ch) < 0x7F else "?" for ch in flat).replace("@end*/", "@end*?")
    return flat[:limit].strip()


def _buildable_part(text: str, names) -> tuple:
    """The text of one piece that builds on its own: the exact lexical closure of its functions.

    ``partition_authored_source`` (the cut ``run_attempt`` compiles) keeps every retained line at its original line
    number and blanks the rest.  The closures of the piece's names are joined line by line, and the blank lines before
    the first and after the last retained line are dropped.  Returns (payload, first line, how it was cut)."""
    from src.compiler.project_compilation_product import partition_authored_source
    lines = None
    for name in names:
        try:
            closure, _retained = partition_authored_source(text, name)
        except (ValueError, SyntaxError, RecursionError) as error:
            return "", 0, f"no closure: {type(error).__name__}: {_first_line(error)}"
        mine = closure.splitlines()
        if lines is None:
            lines = mine
        else:
            if len(mine) > len(lines):
                lines += [""] * (len(mine) - len(lines))
            for index, line in enumerate(mine):
                if line.strip():
                    lines[index] = line
    kept = [index for index, line in enumerate(lines or []) if line.strip()]
    if not kept:
        return "", 0, "no closure: nothing was retained"
    return "\n".join(lines[kept[0]:kept[-1] + 1]), kept[0] + 1, "exact lexical closure"


def write_dream_record(db, state, job_id: str) -> Path:
    """``<job>/record.dream`` (``turing.dream-document.v1``): the enduring record of the job, part by part.

    One python block per piece, in the divider's order (dependencies first).  A block holds the exact lexical closure
    of its piece, the same cut ``run_attempt`` compiles, so EVERY PART BUILDS ON ITS OWN: nothing in a block refers to
    another block's text.  Its ``outputs`` are the functions the piece is for and its ``inputs`` those of the pieces it
    depends on (their definitions are inside the closure; the ports record the division).  After each piece, one
    block per lowering that completed, in the language of that backend, holding the emitted product byte for byte
    (``lowers`` names the piece).  What was tried and refused is written on the piece's header.  When the job came
    from a dream document, its blocks in other languages follow, unchanged.  A last ``record`` block states the job
    as data.  The file is read back by ``load_dream_document``, which checks every block against its ``sha256``
    sentinel, and can be given back to ``intake_dream``.
    """
    from types import MappingProxyType
    from src.compiler.dream_document import (DreamBlock, DreamDocument, load_dream_document, parse_dream_document,
                                             render_dream_document)
    state = StatePaths.coerce(state)
    job_dir = state.jobs_dir / job_id
    came_in = job_dir / "intake.dream"
    source_document = parse_dream_document(came_in.read_bytes()) if came_in.is_file() else None
    report = summary(db, job_id)
    owner = {}                                          # piece id -> its names, for the inputs of the pieces after it
    for p in report["pieces"]:
        owner[p["piece_id"]] = list(p["names"])
    sources, blocks, lowered_blocks = {}, [], 0
    for p in report["pieces"]:
        identity = f"piece-{p['piece_id']}"
        if p["file"] not in sources:
            text = _read_source(job_dir / "source" / p["file"]) or ""
            sources[p["file"]] = (text, hashlib.sha256(text.encode("utf-8")).hexdigest())
        text, file_digest = sources[p["file"]]
        payload, first_line, cut_note = _buildable_part(text, p["names"])
        facts = {"file": _dream_value(p["file"]), "source_sha256": file_digest, "first_line": str(first_line),
                 "cut": _dream_value(cut_note), "ast_nodes": str(p["nodes"]), "annotated": "yes" if p["annotated"] else "no",
                 "depends": ", ".join(f"piece-{d}" for d in p["deps"]) or "none",
                 "verdict": f"lowered ({p['best']})" if p["best"] else "stays python",
                 "role": "part"}
        if source_document is not None:
            facts["source_block"] = _dream_value(PurePosixPath(p["file"]).stem)
        if p["python_reason"]:
            facts["reason"] = _dream_value(p["python_reason"])
        for backend, attempt in sorted(p["attempts"].items()):
            said = attempt["state"] + (f" {attempt['error_type']}@{attempt['stage']}" if attempt["error_type"] else "")
            if attempt["error"]:
                said += f": {attempt['error']}"
            facts[f"attempt.{backend}"] = _dream_value(said)
        inputs = tuple(name for d in p["deps"] for name in owner.get(d, ()))
        blocks.append(DreamBlock(identity=identity, kind="segment", language="python", payload=payload, encoding="utf-8",
                                 inputs=inputs, outputs=tuple(p["names"]), decorations=MappingProxyType(facts)))
        for backend in sorted(p["attempts"], key=_rank):
            base = job_dir / "products" / str(p["piece_id"]) / backend
            attempt_file = base / "attempt.json"
            if backend not in DREAM_LANGUAGE or not attempt_file.exists():
                continue
            for e in json.loads(attempt_file.read_text(encoding="utf-8"))["entries"]:
                product = base / str(e.get("product") or "")
                if e["state"] != "complete" or not product.is_file():
                    continue
                raw = product.read_bytes()
                binary = backend == "wasm"
                try:
                    body = raw.decode("latin-1" if binary else "utf-8")
                except UnicodeDecodeError:
                    binary, body = True, raw.decode("latin-1")
                lowered = {"role": "lowering", "lowers": identity, "backend": backend, "entry": _dream_value(e["name"]),
                           "product": _dream_value(product.name), "carried": "bytes" if binary else "text"}
                if e.get("note"):
                    lowered["note"] = _dream_value(e["note"])
                blocks.append(DreamBlock(identity=f"{identity}.{backend}.{_stem(e['name'])}", kind="segment",
                                         language=DREAM_LANGUAGE[backend], payload=body,
                                         encoding="latin-1" if binary else "utf-8", inputs=(e["name"],),
                                         decorations=MappingProxyType(lowered)))
                lowered_blocks += 1
    carried, dropped_groups, parallel = [], [], ()
    if source_document is not None:
        carried = [b for b in source_document.blocks
                   if b.language != "python" and b.decorations.get("role") not in DREAM_PRODUCT_ROLES]
        blocks.extend(carried)                           # not divided: exactly as they came in
        names = {b.identity for b in carried}
        parallel = tuple(g for g in source_document.parallel if set(g.members) <= names)
        dropped_groups = [g.identity for g in source_document.parallel if not set(g.members) <= names]
    job = report["job"]
    stated = {"schema": SCHEMA_DREAM_RECORD, "job_id": job_id, "label": job.get("label"), "origin": job.get("origin"),
              "created": job.get("created"), "backends": [b for b in str(job.get("backends") or "").split(",") if b],
              "pieces": len(report["pieces"]), "lowered_pieces": report["lowered_pieces"],
              "python_pieces": report["python_pieces"], "lowering_blocks": lowered_blocks,
              "attempt_counts": report["attempt_counts"], "written": time.time(),
              "files": {name: digest for name, (_t, digest) in sorted(sources.items())},
              "came_from_dream": source_document is not None, "carried_blocks": [b.identity for b in carried],
              # a parallel group whose members were divided no longer names blocks of this document
              "parallel_groups_dropped": dropped_groups}
    blocks.append(DreamBlock(identity="record", kind="segment", language="json", encoding="utf-8",
                             payload=json.dumps(stated, indent=1, sort_keys=True),
                             decorations=MappingProxyType({"role": "job-record"})))
    path = job_dir / "record.dream"
    scratch = path.with_suffix(".dream.tmp")
    scratch.write_bytes(render_dream_document(DreamDocument(tuple(blocks), parallel)))
    load_dream_document(scratch)                        # the record must read back, hashes and all, before it replaces the last one
    os.replace(scratch, path)
    return path


def _top_defined(node) -> set:
    """The module-level names one top-level statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    return {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}


def _own_text(closure: str, own: set, provided: set) -> tuple:
    """A unit's own text, cut from its closure: its definitions and what THEY reach, without what other units provide.

    The closure (the compiler's cut) holds the unit's definitions, the definitions of everything it calls, and the
    module-level statements any of those need.  Here the top-level statements are walked from the unit's own
    definitions by the names they reference: a statement is kept when a kept statement names something it defines.
    A definition another unit provides is never kept; it becomes a link.  So a constant only a callee uses leaves
    with the callee, and a constant both use stays in each.  ``own`` and ``provided`` are module-level names.
    Returns (text, names kept, names the kept statements reference); the closure unchanged if it does not parse."""
    try:
        body = ast.parse(closure).body
    except (SyntaxError, ValueError, RecursionError):
        return closure, set(own), set()
    defined = [_top_defined(node) for node in body]
    kept = {index for index, names in enumerate(defined) if names & own}
    kept |= {index for index, node in enumerate(body) if isinstance(node, ast.ImportFrom) and node.module == "__future__"}
    referenced = set()

    def absorb(index):
        referenced.update(n.id for n in ast.walk(body[index]) if isinstance(n, ast.Name))
    for index in kept:
        absorb(index)
    grew = True
    while grew:
        grew = False
        for index, names in enumerate(defined):
            if index in kept or not (names & referenced):
                continue
            if isinstance(body[index], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and names & provided:
                continue                             # another unit provides it: a link, not a copy
            kept.add(index)
            absorb(index)
            grew = True
    lines = closure.splitlines()
    parts = []
    for index in sorted(kept):
        node = body[index]
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        parts.append("\n".join(lines[first - 1:node.end_lineno]))
    kept_names = set().union(*(defined[index] for index in kept)) if kept else set()
    return "\n\n".join(parts), kept_names, referenced


def _api_rows(nodes: dict, names) -> list:
    """The authored API of a unit: one row per formal and one for the result of each entry, annotations as written."""
    rows = []
    for name in names:
        if name not in nodes:
            continue
        node = nodes[name][0]
        a = node.args
        formals = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs) + [x for x in (a.vararg, a.kwarg) if x]
        for formal in formals:
            rows.append({"entry": name, "name": formal.arg, "role": "parameter",
                         "type": ast.unparse(formal.annotation) if formal.annotation is not None else ""})
        rows.append({"entry": name, "name": "return", "role": "result",
                     "type": ast.unparse(node.returns) if node.returns is not None else ""})
    return rows


def write_goal_document(db, state, job_id: str) -> Path:
    """``<job>/goal.goal`` (``speaktome.goal-document.v1``): the job's units, linked.

    The record (``record.dream``) holds every part with what it calls copied inside it, which is what lets a part be
    built alone.  The Goal document is the other form: each unit holds only its OWN definitions; what it needs from
    another unit is a row of its ``link`` table; its ``api`` table is the authored signature of each entry; its
    ``build`` table lists the lowerings that exist (by product hash and by their block in the record) and its ``abi``
    table how each is called, from the product's own api description when the backend wrote one.  The document's
    ``target`` table says, per language, whether every unit is built for it: that is the goal achieved.  Nothing is
    compiled, linked or run here; the links are the ones the triage already resolved (``deps``), by piece id."""
    from types import MappingProxyType
    from meowdoku import rest_api
    goal = rest_api.goal
    if goal is None:
        raise RuntimeError("the SpeakToMe checkout has no speaktome_goal package")
    state = StatePaths.coerce(state)
    job_dir = state.jobs_dir / job_id
    report = summary(db, job_id)
    owner = {p["piece_id"]: list(p["names"]) for p in report["pieces"]}
    sources, units = {}, []
    for p in report["pieces"]:
        identity = f"piece-{p['piece_id']}"
        if p["file"] not in sources:
            text = _read_source(job_dir / "source" / p["file"]) or ""
            try:
                nodes = _function_nodes(ast.parse(text))
            except (SyntaxError, ValueError, RecursionError):
                nodes = {}
            sources[p["file"]] = (text, nodes)
        text, nodes = sources[p["file"]]
        closure, first_line, cut_note = _buildable_part(text, p["names"])
        provided = {name: d for d in p["deps"] for name in owner.get(d, ())}
        top = lambda qualified: qualified.split(".", 1)[0]          # C.m and f.<locals>.g live in C and f
        own, kept_names, used = _own_text(closure, {top(n) for n in p["names"]}, {top(n) for n in provided})
        links = []
        for name, d in provided.items():
            if top(name) in kept_names:             # its definition could not be separated: still inside this text
                links.append({"name": name, "unit": f"piece-{d}", "entry": name, "how": "carried"})
            elif top(name) in used:                 # the text only names it
                links.append({"name": name, "unit": f"piece-{d}", "entry": name, "how": "reference"})
        builds, abi = [], []
        for backend in sorted(p["attempts"], key=_rank):
            base = job_dir / "products" / str(p["piece_id"]) / backend
            attempt_file = base / "attempt.json"
            if backend not in DREAM_LANGUAGE or not attempt_file.exists():
                continue
            for e in json.loads(attempt_file.read_text(encoding="utf-8"))["entries"]:
                if e["state"] != "complete":
                    continue
                target = DREAM_LANGUAGE[backend]
                builds.append({"target": target, "entry": e["name"], "symbol": str(e.get("entry") or ""),
                               "product": str(e.get("product") or ""), "sha256": e["sha256"],
                               "record": f"{identity}.{backend}.{_stem(e['name'])}", "note": str(e.get("note") or "")})
                described = base / str(e.get("api") or "")
                if e.get("api") and described.is_file():
                    try:
                        import yaml
                        api_doc = yaml.safe_load(described.read_text(encoding="utf-8")) or {}
                    except Exception:               # no yaml here, or an unreadable description: the build is still listed
                        api_doc = {}
                    for point in api_doc.get("entry_points") or []:
                        for formal in point.get("parameters") or []:
                            abi.append({"target": target, "entry": e["name"], "name": str(formal.get("name") or ""),
                                        "role": str(formal.get("role") or ""), "dtype": str(formal.get("dtype") or ""),
                                        "passing": str(formal.get("passing") or ""),
                                        "source": str(formal.get("source_name") or "")})
        facts = {"file": p["file"], "first_line": str(first_line), "cut": cut_note,
                 "annotated": "yes" if p["annotated"] else "no",
                 "verdict": f"lowered ({p['best']})" if p["best"] else "stays python"}
        if p["python_reason"]:
            facts["reason"] = p["python_reason"]
        for backend, attempt in sorted(p["attempts"].items()):
            said = attempt["state"] + (f" {attempt['error_type']}@{attempt['stage']}" if attempt["error_type"] else "")
            facts[f"attempt.{backend}"] = said
        units.append(goal.Unit(identity=identity, language="python", text=own, state="lowered" if builds else "source",
                               facts=MappingProxyType(facts), api=goal._rows(_api_rows(nodes, p["names"])),
                               link=goal._rows(links), build=goal._rows(builds), abi=goal._rows(abi)))
    job = report["job"]
    record = job_dir / "record.dream"
    stated = {"source": f"divider job {job_id}" + (f" ({job.get('origin')})" if job.get("origin") else ""),
              "units_record": f"record.dream sha256 {hashlib.sha256(record.read_bytes()).hexdigest()}" if record.is_file() else "",
              "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if job.get("label"):
        stated["label"] = str(job["label"])
    document = goal.GoalDocument(f"job-{job_id}", tuple(units), MappingProxyType(stated))
    path = job_dir / "goal.goal"
    scratch = path.with_suffix(".goal.tmp")
    scratch.write_bytes(goal.render(document))
    goal.load(scratch)                                  # it must read back, hashes and tables, before it replaces the last one
    os.replace(scratch, path)
    return path


# -------------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_attempt":
        return _attempt_main(argv[1])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--state-dir", required=True)
        return sp

    sp = common(sub.add_parser("submit"))
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument("--file")
    group.add_argument("--zip")
    group.add_argument("--text")
    group.add_argument("--git")
    group.add_argument("--dream", help="a dream document; its python blocks are divided, the rest carried into the record")
    sp.add_argument("--name", default="source.py", help="file name for --text")
    sp.add_argument("--allow-network", action="store_true", help="required for --git (operator action)")
    sp.add_argument("--backends", default="wasm")
    sp.add_argument("--owner", required=True)
    sp.add_argument("--label")
    for name in ("plan", "status", "recompose"):
        common(sub.add_parser(name)).add_argument("job")
    wp = common(sub.add_parser("work"))
    wp.add_argument("--once", action="store_true")
    wp.add_argument("--loop", action="store_true")
    wp.add_argument("--worker", default=f"worker-{os.getpid()}")
    wp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    state = StatePaths(args.state_dir)
    db = connect(state.accounts_db)
    if args.command == "submit":
        if args.file:
            intake = intake_file(state, args.file)
        elif args.zip:
            intake = intake_zip(state, args.zip)
        elif args.git:
            intake = intake_git(state, args.git, allow_network=args.allow_network)
        elif args.dream:
            intake = intake_dream(state, args.dream)
        else:
            intake = intake_text(state, args.name, args.text)
        job = submit(db, state, intake, owner_key=args.owner, backends=[b for b in args.backends.split(",") if b], label=args.label)
        print(job)
    elif args.command == "plan":
        for p in plan(db, state, args.job):
            print(f"{p['piece_id']:3d} ord={p['ord']:3d} {p['file']}: {','.join(p['names'])} deps={p['deps']} nodes={p['nodes']} "
                  f"annotated={p['annotated']}")
    elif args.command == "work":
        n = work(db, state, args.worker, 1 if args.once and not args.loop else None, timeout_s=args.timeout)
        print(f"processed {n} attempt(s)")
    elif args.command == "status":
        s = summary(db, args.job)
        print(f"job {args.job} state={s['job']['state']} lowered={s['lowered_pieces']} python={s['python_pieces']} "
              f"attempts={s['attempt_counts']}")
        for p in s["pieces"]:
            print(f"{p['piece_id']:3d} {','.join(p['names'])}: best={p['best']} {p['python_reason'] or ''}")
    elif args.command == "recompose":
        print(write_recomposition(db, state, args.job))
        print(write_dream_record(db, state, args.job))
        print(write_goal_document(db, state, args.job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
