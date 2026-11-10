"""Lowering library: the lowerings (wasm, wgsl, c, llvm, ...) of one computational core, catalogued in SQLite.

Fed only from what each product declares about itself: the ``turing-compiled-program-api-v1`` YAML (read with
``compiled_program_api.load_api``) and the JSON the writer pairs with it.  Nothing is inferred from emitted source.

Pairing (the writers' own rules):
  * wasm lane: ``<stem>.api.yaml`` beside ``<stem>.wasm`` and ``<stem>.abi.json`` (same stem).
  * WGSL lane: ``<stem>.api.yaml`` beside ``<stem>.wgsl``; the per-size record is in a ``*.manifest.json`` of the same
    directory (``fountain.manifest.json``, ``fountain_wheel.manifest.json``, ...) under ``sizes[tag]``, matched by the
    record's declared ``wgsl`` filename.

Family (the directory-declared lane), from the JSON schema + stem prefix the writers use:
  ``turing.mobile-wasm-abi.v1`` -> entity (entity_motion); a manifest record (``turing.fountain-products.*``) or any stem
  starting ``fountain`` -> fountain; ``turing.graph-buffer-abi.v1`` with stem ``law``/``pilot`` -> policy, other -> graph.
core_id: ``logical:<logical_abi_digest>`` if the API metadata declares one, else ``<family>:<source_digest>``, else
``api:<sha256 of the api yaml bytes>``.  Verdict is 'built' (product file exists) or 'described'; never 'verified'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    from meowdoku import schema_version as _schema_version
except ImportError:      # loaded by path or run as a script: take the sibling file
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("meowdoku_schema_version", Path(__file__).with_name("schema_version.py"))
    _schema_version = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_schema_version)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORES = ("CREATE TABLE IF NOT EXISTS lowering_cores (core_id TEXT PRIMARY KEY, family TEXT NOT NULL, name TEXT NOT NULL, "
         "source_function TEXT, logical_abi_digest TEXT, source_digest TEXT, extraction_contract_sha256 TEXT, "
         "first_seen REAL NOT NULL)")
LOWERINGS = ("CREATE TABLE IF NOT EXISTS lowerings (core_id TEXT NOT NULL REFERENCES lowering_cores(core_id), "
             "backend TEXT NOT NULL, contract TEXT, entry TEXT NOT NULL, api_path TEXT NOT NULL, abi_path TEXT, "
             "product_path TEXT, product_sha256 TEXT, abi_schema TEXT, arch TEXT, verdict TEXT NOT NULL, "
             "scanned REAL NOT NULL, PRIMARY KEY (core_id, backend, api_path))")
BLOBS = ("CREATE TABLE IF NOT EXISTS lowering_blobs (product_sha256 TEXT PRIMARY KEY, kind TEXT NOT NULL, "
         "byte_count INTEGER NOT NULL, bytes BLOB NOT NULL, stored REAL NOT NULL)")
LEASES = ("CREATE TABLE IF NOT EXISTS lowering_leases (lease_id INTEGER PRIMARY KEY AUTOINCREMENT, "
          "product_sha256 TEXT NOT NULL REFERENCES lowering_blobs(product_sha256), core_id TEXT NOT NULL, "
          "backend TEXT NOT NULL, account_key TEXT NOT NULL, purpose TEXT NOT NULL, generation INTEGER NOT NULL, "
          "issued REAL NOT NULL, deadline REAL NOT NULL, returned REAL)")
BACKEND_ORDER = ("c", "llvm", "fortran", "wasm", "wgsl", "spirv", "cpython")


@dataclass
class ProductRecord:
    core_id: str
    family: str
    name: str
    backend: str
    entry: str
    api_path: str
    verdict: str
    source_function: str | None = None
    logical_abi_digest: str | None = None
    source_digest: str | None = None
    extraction_contract_sha256: str | None = None
    contract: str | None = None
    abi_path: str | None = None
    product_path: str | None = None
    product_sha256: str | None = None
    abi_schema: str | None = None
    arch: str | None = None


@contextmanager
def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(CORES)
        db.execute(LOWERINGS)
        db.execute(BLOBS)
        db.execute(LEASES)
        _schema_version.declare(db, "lowering_library", 1)
        with db:
            yield db
    finally:
        db.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(value):
    return None if value is None else (value if isinstance(value, str) else json.dumps(value, sort_keys=True))


def describe_product(api_yaml_path, root=None) -> ProductRecord:
    """Describe one product.  Uses ``load_api`` when the document is the v1 schema, else plain ``yaml.safe_load``."""
    import yaml
    api_yaml_path = Path(api_yaml_path)
    raw = api_yaml_path.read_bytes()
    try:
        from src.compiler.compiled_program_api import load_api
        api = load_api(api_yaml_path)
    except ValueError:
        api = yaml.safe_load(raw.decode("utf-8")) or {}
    meta = dict(api.get("metadata") or {})
    name, backend, entry = str(api.get("module") or ""), str(api.get("language") or ""), str(api.get("entry") or "")
    stem = api_yaml_path.name[: -len(".api.yaml")] if api_yaml_path.name.endswith(".api.yaml") else api_yaml_path.stem
    here = api_yaml_path.parent
    abi_json, abi_path, wgsl_record, manifest_schema = {}, None, None, None
    wgsl = here / f"{stem}.wgsl"
    if wgsl.exists():
        for manifest in sorted(here.glob("*.manifest.json")):
            doc = _load(manifest) or {}
            for record in (doc.get("sizes") or {}).values():
                if isinstance(record, dict) and record.get("wgsl") == wgsl.name:
                    wgsl_record, abi_path, manifest_schema = record, manifest, doc.get("schema")
                    break
            if wgsl_record:
                break
    elif (here / f"{stem}.abi.json").exists():
        abi_path = here / f"{stem}.abi.json"
        abi_json = _load(abi_path) or {}
    source = wgsl_record or abi_json
    abi_ref = (wgsl_record or {}).get("wgsl_abi") or abi_json
    source_digest = source.get("source_digest")
    contract = abi_ref.get("contract") if isinstance(abi_ref, dict) else None
    schema = manifest_schema if wgsl_record else abi_json.get("schema")
    if schema == "turing.mobile-wasm-abi.v1":
        family = "entity"
    elif wgsl_record is not None or stem.startswith("fountain"):
        family = "fountain"
    elif schema == "turing.graph-buffer-abi.v1":
        family = "policy" if stem in ("law", "pilot") else "graph"
    else:
        family = "api"
    product = next((p for p in (here / f"{stem}.wasm", wgsl) if p.exists()), None)
    logical = meta.get("logical_abi_digest")
    if logical:
        core_id = f"logical:{logical}"
    elif source_digest and family != "api":
        core_id = f"{family}:{source_digest}"
    else:
        core_id = f"api:{hashlib.sha256(raw).hexdigest()}"

    def rel(p):
        if p is None:
            return None
        return Path(p).relative_to(root).as_posix() if root else Path(p).as_posix()

    return ProductRecord(
        core_id=core_id, family=family, name=name, backend=backend, entry=entry, api_path=rel(api_yaml_path),
        verdict="built" if product else "described", source_function=meta.get("source_function"),
        logical_abi_digest=logical, source_digest=source_digest,
        extraction_contract_sha256=meta.get("extraction_contract_sha256"), contract=_text(contract),
        abi_path=rel(abi_path), product_path=rel(product), product_sha256=_sha256(product) if product else None,
        abi_schema=schema, arch=_text(meta.get("arch") or (abi_ref.get("arch") if isinstance(abi_ref, dict) else None)))


def register(db_path, record: ProductRecord) -> None:
    now = time.time()
    with connect(db_path) as db:
        db.execute("INSERT OR IGNORE INTO lowering_cores VALUES (?,?,?,?,?,?,?,?)",
                   (record.core_id, record.family, record.name, record.source_function, record.logical_abi_digest,
                    record.source_digest, record.extraction_contract_sha256, now))
        db.execute("INSERT OR REPLACE INTO lowerings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (record.core_id, record.backend, record.contract, record.entry, record.api_path, record.abi_path,
                    record.product_path, record.product_sha256, record.abi_schema, record.arch, record.verdict, now))


def store_blob(db_path, product_path) -> str:
    """Store the product file's bytes content-addressed (idempotent); returns its sha256."""
    product_path = Path(product_path)
    data = product_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with connect(db_path) as db:
        db.execute("INSERT OR IGNORE INTO lowering_blobs VALUES (?,?,?,?,?)",
                   (sha, product_path.suffix.lstrip("."), len(data), data, time.time()))
    return sha


def blob_bytes(db_path, product_sha256) -> bytes | None:
    with connect(db_path) as db:
        row = db.execute("SELECT bytes FROM lowering_blobs WHERE product_sha256=?", (product_sha256,)).fetchone()
    return None if row is None else bytes(row["bytes"])


def scan(root, db_path, store_blobs=True) -> list[ProductRecord]:
    root = Path(root)
    records = [describe_product(p, root) for p in sorted(root.rglob("*.api.yaml"))]
    for record in records:
        register(db_path, record)
        if store_blobs and record.product_path:
            store_blob(db_path, root / record.product_path)
    return records


def cores(db_path) -> list[dict]:
    with connect(db_path) as db:
        return [dict(r) for r in db.execute("SELECT * FROM lowering_cores ORDER BY family, name, core_id")]


def lowerings(db_path, core_id=None, backend=None) -> list[dict]:
    where, args = [], []
    for column, value in (("core_id", core_id), ("backend", backend)):
        if value is not None:
            where.append(f"{column}=?")
            args.append(value)
    sql = "SELECT * FROM lowerings" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY core_id, backend, api_path"
    with connect(db_path) as db:
        return [dict(r) for r in db.execute(sql, args)]


def versions(db_path, core_id, backend) -> list[dict]:
    """All lowerings of that core/backend, newest scanned first; each distinct product_sha256 is one version."""
    with connect(db_path) as db:
        return [dict(r) for r in db.execute(
            "SELECT * FROM lowerings WHERE core_id=? AND backend=? ORDER BY scanned DESC, api_path", (core_id, backend))]


def _select(db, purpose, capabilities, prefer, contract):
    family, name = purpose
    caps = frozenset(capabilities)
    rank = {b: i for i, b in enumerate(tuple(prefer) + BACKEND_ORDER)}
    rows = [dict(r) for r in db.execute(
        "SELECT l.* FROM lowerings l JOIN lowering_cores c ON c.core_id=l.core_id "
        "JOIN lowering_blobs b ON b.product_sha256=l.product_sha256 "
        "WHERE c.family=? AND c.name=? AND l.verdict='built'", (family, name))]
    rows = [r for r in rows if r["backend"] in caps and (contract is None or r["contract"] == contract)]
    rows.sort(key=lambda r: (rank.get(r["backend"], len(rank)), -r["scanned"]))
    return rows[0] if rows else None


def select_lowering(db_path, purpose, capabilities, *, prefer=(), contract=None) -> dict | None:
    """Pick the lowering of a computational purpose that the caller can execute.

    ``purpose`` is ``(family, name)``: a core's computational purpose IS its declared family + name.  ``capabilities``
    is the set of backend names the caller can execute; it is the only capability interpreted here (role/tier/account
    rights belong to the server authz layer, which passes in only what the account may execute).  Order: ``prefer``
    backends first, then BACKEND_ORDER, then newest scanned.  Only built lowerings whose blob is stored qualify.
    Returns None when nothing fits; never falls back to a backend outside ``capabilities``.
    """
    with connect(db_path) as db:
        return _select(db, purpose, capabilities, prefer, contract)


@dataclass
class Lease:
    lease_id: int
    product_sha256: str
    core_id: str
    backend: str
    generation: int
    deadline: float
    kind: str
    byte_count: int
    account_key: str = ""


def checkout(db_path, purpose, capabilities, account_key, seconds, *, prefer=(), contract=None) -> Lease | None:
    """Select (see select_lowering) and lease a version under BEGIN IMMEDIATE; fetch bytes with ``blob_bytes``.

    Each lease gets a fresh per-blob generation (max existing + 1).  Capabilities beyond backends are not interpreted.
    """
    db = sqlite3.connect(Path(db_path), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN IMMEDIATE")
        chosen = _select(db, purpose, capabilities, prefer, contract)
        if chosen is None:
            db.execute("ROLLBACK")
            return None
        sha, now = chosen["product_sha256"], time.time()
        blob = db.execute("SELECT kind, byte_count FROM lowering_blobs WHERE product_sha256=?", (sha,)).fetchone()
        gen = db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM lowering_leases WHERE product_sha256=?",
                         (sha,)).fetchone()[0]
        cur = db.execute("INSERT INTO lowering_leases (product_sha256,core_id,backend,account_key,purpose,generation,"
                         "issued,deadline) VALUES (?,?,?,?,?,?,?,?)",
                         (sha, chosen["core_id"], chosen["backend"], account_key, f"{purpose[0]}:{purpose[1]}", gen,
                          now, now + seconds))
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return Lease(cur.lastrowid, sha, chosen["core_id"], chosen["backend"], gen, now + seconds, blob["kind"],
                 blob["byte_count"], account_key)


def checkin(db_path, lease_id, account_key, generation) -> bool:
    """Return a lease: succeeds once, only when account_key and generation match and it is not already returned."""
    with connect(db_path) as db:
        cur = db.execute("UPDATE lowering_leases SET returned=? WHERE lease_id=? AND account_key=? AND generation=? "
                         "AND returned IS NULL", (time.time(), lease_id, account_key, generation))
        return cur.rowcount == 1


def _leases(db_path, where="", args=()) -> list[Lease]:
    with connect(db_path) as db:
        return [Lease(r["lease_id"], r["product_sha256"], r["core_id"], r["backend"], r["generation"], r["deadline"],
                      r["kind"], r["byte_count"], r["account_key"]) for r in db.execute(
            "SELECT l.*, b.kind, b.byte_count FROM lowering_leases l JOIN lowering_blobs b "
            "ON b.product_sha256=l.product_sha256 " + where + " ORDER BY l.lease_id", args)]


def expired(db_path, now=None) -> list[Lease]:
    """Unreturned leases past their deadline (eligible for reassignment); nothing is deleted."""
    return _leases(db_path, "WHERE l.returned IS NULL AND l.deadline < ?", (time.time() if now is None else now,))


def outstanding(db_path, product_sha256) -> int:
    """Open (unreturned) leases of one blob: the contention signal."""
    with connect(db_path) as db:
        return db.execute("SELECT COUNT(*) FROM lowering_leases WHERE product_sha256=? AND returned IS NULL",
                          (product_sha256,)).fetchone()[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "store"):
        sp = sub.add_parser(name)
        sp.add_argument("root")
        sp.add_argument("--db", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--db", required=True)
    v = sub.add_parser("versions")
    v.add_argument("core_id")
    v.add_argument("backend")
    v.add_argument("--db", required=True)
    co = sub.add_parser("checkout")
    co.add_argument("--db", required=True)
    co.add_argument("--purpose", required=True, help="family:name")
    co.add_argument("--caps", required=True, help="comma-separated backends")
    co.add_argument("--account", required=True)
    co.add_argument("--seconds", type=float, required=True)
    co.add_argument("--prefer", default="")
    ci = sub.add_parser("checkin")
    ci.add_argument("lease_id", type=int)
    ci.add_argument("--db", required=True)
    ci.add_argument("--account", required=True)
    ci.add_argument("--generation", type=int, required=True)
    lz = sub.add_parser("leases")
    lz.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    if args.command in ("scan", "store"):
        found = scan(args.root, args.db)
        print(f"registered {len(found)} products")
    elif args.command == "versions":
        for row in versions(args.db, args.core_id, args.backend):
            print(f"{row['product_sha256']}  {row['scanned']:.0f}  {row['api_path']}")
        return 0
    elif args.command == "checkout":
        family, _, name = args.purpose.partition(":")
        lease = checkout(args.db, (family, name), frozenset(filter(None, args.caps.split(","))), args.account,
                         args.seconds, prefer=tuple(filter(None, args.prefer.split(","))))
        print(lease if lease else "no executable lowering")
        return 0 if lease else 1
    elif args.command == "checkin":
        ok = checkin(args.db, args.lease_id, args.account, args.generation)
        print("returned" if ok else "refused")
        return 0 if ok else 1
    elif args.command == "leases":
        for r in _leases(args.db):
            print(r)
        return 0
    for row in lowerings(args.db):
        print(f"{row['core_id']}  {row['backend']:<6} {row['verdict']:<9} {row['entry']}  {row['api_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
