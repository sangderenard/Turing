"""Local snapshot backups of the Meowdoku state root; safe to run while the servers are live.

  python state_backup.py snapshot --state-dir ROOT --out BACKUP_ROOT [--keep 14] [--include-logs] [--include-secrets]
  python state_backup.py verify   --snapshot DIR
  python state_backup.py restore  --snapshot DIR --state-dir ROOT [--dry-run] [--into-empty]
  python state_backup.py list     --out BACKUP_ROOT

Layout comes from state_paths.StatePaths; sha256 / sqlite helpers come from state_migrate.  Databases (sql/*.db, *.sqlite) travel through the
sqlite3 backup API (consistent while live) then PRAGMA integrity_check; other sql/ files, content/, products/, jobs/, canvas/ are copied file by
file (a file that vanishes or is locked is skipped and recorded).  secrets/ only with --include-secrets (snapshot dir then restricted with the
same icacls step as operators.py, or the snapshot is refused and removed); logs only with --include-logs; run/ never.
restore never overwrites: it refuses when any target file it would write already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from state_paths import StatePaths  # noqa: E402
from state_migrate import sha256, _sqlite_facts  # noqa: E402

TOOL_VERSION = "1"
STAMP = re.compile(r"^\d{8}T\d{6}Z$")
DB_SUFFIXES = (".db", ".sqlite")


class BackupError(Exception):
    pass


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_out(state_dir: Path, out: Path):
    if _inside(out, state_dir):
        raise BackupError(f"REFUSED: --out {out} is inside the state dir {state_dir}")
    probe = out
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        r = subprocess.run(["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise BackupError(f"REFUSED: cannot check whether --out is inside a git tree ({type(error).__name__})")
    if r.returncode == 0 and r.stdout.strip() == "true":
        # A path the enclosing repository IGNORES is not tracked and cannot be committed by accident (the state root itself sits in one).
        ig = subprocess.run(["git", "-C", str(probe), "check-ignore", "-q", str(out)], capture_output=True, text=True, timeout=30)
        if ig.returncode != 0:
            raise BackupError(f"REFUSED: --out {out} is inside a git working tree and not ignored by it")


def _backup_db(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(str(src), timeout=10)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def _walk(d: Path):
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                yield p


def _plan(paths: StatePaths, include_logs: bool, include_secrets: bool):
    """(kind, src, rel) for every candidate file."""
    items = []
    for p in _walk(paths.sql_dir):
        if p.name.endswith(("-wal", "-shm", "-journal")):
            continue
        items.append(("sqlite" if p.suffix.lower() in DB_SUFFIXES else "file", p, p.relative_to(paths.root).as_posix()))
    dirs = [paths.content_dir, paths.products_dir, paths.jobs_dir, paths.canvas_dir]
    if include_logs:
        dirs.append(paths.logs_dir)
    if include_secrets:
        dirs.append(paths.secrets_dir)
    for d in dirs:
        for p in _walk(d):
            items.append(("file", p, p.relative_to(paths.root).as_posix()))
    return items


def _restrict(directory: Path) -> str:
    if str(HERE / "deploy") not in sys.path:
        sys.path.append(str(HERE / "deploy"))
    import operators as operators_mod
    return operators_mod._restrict(directory)


def snapshot(state_dir, out, keep=14, include_logs=False, include_secrets=False) -> dict:
    t0 = time.time()
    state_dir, out = Path(state_dir).resolve(), Path(out).resolve()
    if not state_dir.is_dir():
        raise BackupError(f"REFUSED: state dir {state_dir} does not exist")
    _check_out(state_dir, out)
    paths = StatePaths(state_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final, work = out / stamp, out / f".partial-{stamp}"
    if final.exists() or work.exists():
        raise BackupError(f"REFUSED: snapshot {stamp} already exists; wait a second")
    work.mkdir()
    restrict_report = None
    try:
        if include_secrets:
            restrict_report = _restrict(work)
            if "restricted to the current user" not in restrict_report and "chmod 700" not in restrict_report:
                raise BackupError(f"REFUSED: could not restrict the snapshot directory ({restrict_report})")
        files, skipped = [], []
        for kind, src, rel in _plan(paths, include_logs, include_secrets):
            dst = work / rel
            try:
                if kind == "sqlite":
                    _backup_db(src, dst)
                    facts = _sqlite_facts(dst)
                    if facts["integrity"] != "ok":
                        raise BackupError(f"integrity_check failed on the copy of {rel}: {facts['integrity']}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            except BackupError:
                raise
            except FileNotFoundError:
                skipped.append({"rel": rel, "why": "vanished"})
                dst.unlink(missing_ok=True)
                continue
            except (OSError, sqlite3.Error) as error:
                skipped.append({"rel": rel, "why": f"{type(error).__name__}: {error}"})
                dst.unlink(missing_ok=True)
                continue
            e = {"rel": rel, "kind": kind, "size": dst.stat().st_size, "sha256": sha256(dst)}
            if kind == "sqlite":
                e["integrity"], e["rows"] = facts["integrity"], facts["rows"]
            files.append(e)
        manifest = {"tool_version": TOOL_VERSION, "source_root": str(state_dir), "created_utc": stamp,
                    "include_logs": include_logs, "include_secrets": include_secrets, "secrets_restrict": restrict_report,
                    "files": files, "skipped": skipped, "duration_s": round(time.time() - t0, 3)}
        (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        os.replace(work, final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    removed = _retain(out, keep, keep_name=stamp)
    manifest["snapshot"], manifest["removed"] = str(final), removed
    print(f"snapshot {final} files={len(files)} dbs={sum(1 for f in files if f['kind'] == 'sqlite')} skipped={len(skipped)} "
          f"secrets={'yes' if include_secrets else 'no'} removed_old={len(removed)} {manifest['duration_s']}s")
    return manifest


def _snapshots(out: Path):
    if not out.is_dir():
        return []
    return sorted(d for d in out.iterdir() if d.is_dir() and STAMP.match(d.name) and (d / "manifest.json").is_file())


def _retain(out: Path, keep: int, keep_name: str) -> list:
    removed = []
    snaps = _snapshots(out)
    for d in snaps[:max(0, len(snaps) - max(1, keep))]:
        if d.name != keep_name:
            shutil.rmtree(d)
            removed.append(d.name)
    return removed


def verify(snapshot_dir) -> dict:
    d = Path(snapshot_dir)
    mf = d / "manifest.json"
    if not mf.is_file():
        raise BackupError(f"no manifest.json in {d}")
    m = json.loads(mf.read_text(encoding="utf-8"))
    bad = []
    for e in m["files"]:
        p = d / e["rel"]
        if not p.is_file():
            bad.append({"rel": e["rel"], "why": "missing"})
        elif p.stat().st_size != e["size"] or sha256(p) != e["sha256"]:
            bad.append({"rel": e["rel"], "why": "size or sha256 differs"})
        elif e["kind"] == "sqlite":
            f = _sqlite_facts(p)
            if f["integrity"] != "ok" or f["rows"] != e["rows"]:
                bad.append({"rel": e["rel"], "why": "integrity or row counts differ"})
    return {"ok": not bad, "checked": len(m["files"]), "bad": bad}


def restore(snapshot_dir, state_dir, dry_run=False, into_empty=False) -> dict:
    d, root = Path(snapshot_dir), Path(state_dir)
    v = verify(d)
    if not v["ok"]:
        raise BackupError(f"REFUSED: snapshot does not verify: {v['bad']}")
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if into_empty and root.exists() and any(root.iterdir()):
        raise BackupError(f"REFUSED: --into-empty but {root} is not empty")
    existing = [e["rel"] for e in m["files"] if (root / e["rel"]).exists()]
    live_dbs = [p.name for p in (root / "sql").glob("*") if p.suffix.lower() in DB_SUFFIXES] if (root / "sql").is_dir() else []
    if existing or live_dbs:
        raise BackupError(f"REFUSED: target already holds state ({(existing or live_dbs)[:5]}); restoring over a live root is a manual act "
                          f"(see deploy/BACKUPS_2026-11-10.md)")
    if dry_run:
        print(f"dry run: would restore {len(m['files'])} file(s) from {d} into {root}")
        return {"dry_run": True, "count": len(m["files"])}
    for e in m["files"]:
        dst = root / e["rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".restoring")
        shutil.copy2(d / e["rel"], tmp)
        os.replace(tmp, dst)
    if m.get("include_secrets") and (root / "secrets").is_dir():
        _restrict(root / "secrets")
    bad = [e["rel"] for e in m["files"]
           if sha256(root / e["rel"]) != e["sha256"] or (e["kind"] == "sqlite" and _sqlite_facts(root / e["rel"])["integrity"] != "ok")]
    print(f"restored {len(m['files'])} file(s) into {root}; verify_ok={not bad}")
    return {"dry_run": False, "count": len(m["files"]), "ok": not bad, "bad": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--keep", type=int, default=14)
    s.add_argument("--include-logs", action="store_true")
    s.add_argument("--include-secrets", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--snapshot", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--snapshot", required=True)
    r.add_argument("--state-dir", required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--into-empty", action="store_true")
    ls = sub.add_parser("list")
    ls.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "snapshot":
            snapshot(a.state_dir, a.out, a.keep, a.include_logs, a.include_secrets)
        elif a.cmd == "verify":
            res = verify(a.snapshot)
            print(f"verify ok={res['ok']} checked={res['checked']} bad={res['bad']}")
            return 0 if res["ok"] else 1
        elif a.cmd == "restore":
            res = restore(a.snapshot, a.state_dir, a.dry_run, a.into_empty)
            return 0 if res.get("dry_run") or res["ok"] else 1
        else:
            for d in _snapshots(Path(a.out)):
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                print(f"{d.name}  files={len(m['files'])} skipped={len(m['skipped'])} secrets={m['include_secrets']}")
    except BackupError as error:
        print(str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
