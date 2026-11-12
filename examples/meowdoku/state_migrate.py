"""Copy the legacy Meowdoku state (build/meowdoku beside web/) into a StatePaths root, then verify it.  Run in an Apache stop window.

The legacy tree is NEVER modified or deleted: ``copy`` only reads it.  Destination layout and legacy names both come from state_paths.StatePaths
(legacy_at(base) -> StatePaths(root)); this module owns no path convention of its own.

  plan(legacy_base, root)  -> list[Step]   one Step per file (directories are expanded), so the receipt names every file
  execute(plan, mode, root=...) -> receipt  mode "copy" | "verify" | "copy-verify"; the receipt is also written to <root>/run/state-migration-<ts>.json
  python state_migrate.py --legacy-base build/meowdoku --state-dir srv/state [--dry-run] [--mode copy|verify|copy-verify]

SQLite files (accounts.db, job-library.sqlite) go through the sqlite3 backup API so WAL contents are included; everything else is a byte copy.
copy refuses (MigrationError, nothing written) when a destination exists and differs; an identical destination is skipped (idempotent).
verify: files by size and sha256; sqlite by PRAGMA integrity_check == ok on both sides and equal row counts of every table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from state_paths import StatePaths  # noqa: E402


class MigrationError(Exception):
    pass


@dataclass(frozen=True)
class Step:
    kind: str        # "file" | "sqlite"
    category: str    # sql | content | canvas | secrets | logs | run
    src: Path
    dst: Path


def _walk(src_dir: Path):
    if src_dir.is_dir():
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                yield p


def plan(legacy_base, root) -> list[Step]:
    old, new = StatePaths.legacy_at(Path(legacy_base)), StatePaths(Path(root))
    steps: list[Step] = []

    def add(kind, category, src, dst):
        if Path(src).is_file():
            steps.append(Step(kind, category, Path(src), Path(dst)))

    def add_dir(category, src_dir, dst_dir, skip=lambda p: False):
        for p in _walk(Path(src_dir)):
            if not skip(p):
                steps.append(Step("file", category, p, Path(dst_dir) / p.relative_to(src_dir)))

    # sql: databases through the backup API, policy files beside them
    add("sqlite", "sql", old.accounts_db, new.accounts_db)
    add("sqlite", "sql", old.job_library_db, new.job_library_db)
    add("file", "sql", old.access_json, new.access_json)
    add("file", "sql", old.sql_dir / "authz_policy.json", new.sql_dir / "authz_policy.json")
    # content
    for name in ("profiles.json", "profiles.dev.json", "tipjar.json", "lichess_token.txt", "loading_anim.json"):
        add("file", "content", old.content_dir / name, new.content_dir / name)
    for p in sorted(old.content_dir.glob("*.mla")):
        add("file", "content", p, new.content_dir / p.name)
    skip_db = lambda p: p.name.startswith("job-library.sqlite")      # the database travels by the backup API, not as a byte copy
    add_dir("content", old.marble_objects_dir, new.marble_objects_dir)
    add_dir("content", old.marble_photos_dir, new.marble_photos_dir, skip_db)
    add_dir("content", old.users_dir, new.users_dir)
    add_dir("content", old.obj_assets_dir, new.obj_assets_dir)
    add_dir("content", old.shares_dir, new.shares_dir)
    add_dir("content", old.tickets_dir, new.tickets_dir)
    # canvas
    add_dir("canvas", old.canvas_dir, new.canvas_dir)
    # secrets (operator_credentials holds operator, console and windows-service secrets)
    add_dir("secrets", old.tls_dir, new.tls_dir)
    for p in sorted(old.secrets_dir.glob("admin_token*.txt")):
        add("file", "secrets", p, new.secrets_dir / p.name)
    add_dir("secrets", old.operator_credentials_dir, new.operator_credentials_dir)
    # logs
    add_dir("logs", old.logs_dir, new.logs_dir)
    add("file", "logs", old.log_ring, new.log_ring)
    for pat in ("live-*.std*.log", "dev_*_std*.log"):
        for p in sorted(old.root.glob(pat)):
            add("file", "logs", p, new.logs_dir / p.name)
    # run: pid files, per-port session dirs and launch json
    for d in sorted(old.root.glob("sessions*")):
        add_dir("run", d, new.run_dir / d.name)
    for p in sorted(old.root.glob("launch*.json")) + sorted(old.root.glob("console-*.pid")) + [old.root / "last_exit.json"]:
        add("file", "run", p, new.run_dir / p.name)
    return steps


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tables(db: sqlite3.Connection) -> dict:
    names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: db.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}


def _sqlite_facts(path: Path) -> dict:
    db = sqlite3.connect(str(path), timeout=10)
    try:
        return {"integrity": db.execute("PRAGMA integrity_check").fetchone()[0], "rows": _tables(db)}
    finally:
        db.close()


def _backup(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".migrating")
    if tmp.exists():
        tmp.unlink()
    s = sqlite3.connect(str(src), timeout=10)
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    os.replace(tmp, dst)


def _same(step: Step) -> bool:
    if step.kind == "sqlite":
        a, b = _sqlite_facts(step.src), _sqlite_facts(step.dst)
        return a == b and b["integrity"] == "ok"
    return step.src.stat().st_size == step.dst.stat().st_size and sha256(step.src) == sha256(step.dst)


def _copy(step: Step) -> str:
    if step.dst.exists():
        if _same(step):
            return "skipped-identical"
        raise MigrationError(f"REFUSED: destination exists and differs: {step.dst} (source {step.src}); nothing is overwritten")
    if step.kind == "sqlite":
        _backup(step.src, step.dst)
    else:
        step.dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = step.dst.with_name(step.dst.name + ".migrating")
        shutil.copy2(step.src, tmp)
        os.replace(tmp, step.dst)
    return "copied"


def _verify(step: Step) -> dict:
    if not step.dst.exists():
        return {"ok": False, "why": "destination missing"}
    if step.kind == "sqlite":
        a, b = _sqlite_facts(step.src), _sqlite_facts(step.dst)
        ok = b["integrity"] == "ok" and a["integrity"] == "ok" and a["rows"] == b["rows"]
        return {"ok": ok, "integrity": b["integrity"], "rows_src": a["rows"], "rows_dst": b["rows"],
                **({} if ok else {"why": "integrity or row counts differ"})}
    ss, ds = step.src.stat().st_size, step.dst.stat().st_size
    hs, hd = sha256(step.src), sha256(step.dst)
    ok = ss == ds and hs == hd
    return {"ok": ok, **({} if ok else {"why": f"size {ss}/{ds} sha256 {hs[:12]}/{hd[:12]}"})}


def execute(steps, mode="copy", root=None, restrict=True) -> dict:
    """Run the plan; returns the receipt and writes it under <root>/run/.  restrict=True runs operators' _restrict on <root>/secrets after a copy."""
    if mode not in ("copy", "verify", "copy-verify"):
        raise ValueError("mode must be copy, verify or copy-verify")
    if root is None:
        raise ValueError("execute needs root= (the StatePaths root the plan was made for)")
    steps = list(steps)
    new = StatePaths(Path(root))
    results: dict[Step, dict] = {s: {} for s in steps}
    restrict_report = None
    if "copy" in mode:
        for s in steps:                                  # refuse up front: no partial copy when any destination differs
            if s.dst.exists() and not _same(s):
                raise MigrationError(f"REFUSED: destination exists and differs: {s.dst} (source {s.src}); nothing was written")
        new.ensure(restrict=False)
        for s in steps:
            results[s]["action"] = _copy(s)
        if restrict and any(s.category == "secrets" for s in steps):
            if str(HERE / "deploy") not in sys.path:
                sys.path.append(str(HERE / "deploy"))
            import operators as operators_mod
            restrict_report = operators_mod._restrict(new.secrets_dir)
    verdict = None
    if "verify" in mode:
        for s in steps:
            results[s]["verify"] = _verify(s)
        verdict = all(results[s]["verify"]["ok"] for s in steps)
    files = []
    for s in steps:
        e = {"category": s.category, "kind": s.kind, "src": str(s.src), "dst": str(s.dst),
             "rel": s.dst.relative_to(new.root).as_posix(), **results[s]}
        if s.dst.exists():
            e["size"], e["sha256"] = s.dst.stat().st_size, sha256(s.dst)
        files.append(e)
    receipt = {"mode": mode, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": str(new.root),
               "count": len(files), "verify_ok": verdict, "secrets_restrict": restrict_report, "files": files}
    new.run_dir.mkdir(parents=True, exist_ok=True)
    path = new.run_dir / f"state-migration-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    receipt["receipt_path"] = str(path)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--legacy-base", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--mode", default="copy-verify", choices=["copy", "verify", "copy-verify"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    steps = plan(a.legacy_base, a.state_dir)
    if a.dry_run:
        for s in steps:
            print(f"{s.kind:6} {s.category:8} {s.src} -> {s.dst}")
        print(f"{len(steps)} step(s); dry run, nothing written")
        return 0
    try:
        r = execute(steps, a.mode, root=a.state_dir)
    except MigrationError as error:
        print(str(error))
        return 2
    print(f"{r['count']} file(s); mode {r['mode']}; verify_ok={r['verify_ok']}")
    print(f"receipt: {r['receipt_path']}")
    return 0 if r["verify_ok"] in (None, True) else 1


if __name__ == "__main__":
    sys.exit(main())
