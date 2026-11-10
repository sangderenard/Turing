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


def scan(root, db_path) -> list[ProductRecord]:
    root = Path(root)
    records = [describe_product(p, root) for p in sorted(root.rglob("*.api.yaml"))]
    for record in records:
        register(db_path, record)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--db", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    if args.command == "scan":
        found = scan(args.root, args.db)
        print(f"registered {len(found)} products")
    for row in lowerings(args.db):
        print(f"{row['core_id']}  {row['backend']:<6} {row['verdict']:<9} {row['entry']}  {row['api_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
