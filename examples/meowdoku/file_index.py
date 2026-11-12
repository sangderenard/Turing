"""A flat, searchable index of every file in the core: one SQL row per file, every metric on the row.

The cleanup account keeps it (``build`` refuses anyone else) so that what is in the core can be found and filtered
without walking it: which files are large, old, untracked, binary, never committed, of one kind, under one folder.

    python file_index.py --policy deploy/branch_policy.json --db <state>/sql/file_index.db build --as svc-pale-rider
    python file_index.py --policy ... --db ... search --q shader --ext glsl --order -bytes --limit 20
    python file_index.py --policy ... --db ... summary

One table, ``files``, keyed by (repository, path). No tree, no joins.

    repository, path     the policy's name of the repository, and the path inside its own checkout
    name, ext, dir, depth
    status               tracked | modified | untracked | ignored
    is_dir               1 for an ignored directory, which is one row and is never entered
    bytes, mtime         from the file system (NULL for an ignored directory)
    lines, binary        counted for text files up to 4 MiB; binary is what Git says of the file
    blob, mode           Git's id and mode of the tracked content
    commits, first_commit_at, last_commit_at, last_commit
                         from one pass over the history of what is checked out
    indexed_at

What is listed comes from Git (``git ls-files``), never from a recursive walk: an ignored directory such as a build
tree or an environment is one row. Unchanged files (same size and time as the last index) keep their counted lines,
so a second build reads only what changed. The index lives in its own database beside the accounts database and can
be rebuilt from nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

MAX_COUNTED_BYTES = 4 * 1024 * 1024
STATUSES = ("tracked", "modified", "untracked", "ignored")
COLUMNS = ("repository", "path", "name", "ext", "dir", "depth", "status", "is_dir", "bytes", "mtime", "lines", "binary",
           "blob", "mode", "commits", "first_commit_at", "last_commit_at", "last_commit", "indexed_at")
ORDERS = ("path", "name", "ext", "bytes", "mtime", "lines", "commits", "last_commit_at", "first_commit_at", "depth")
DDL = (
    "CREATE TABLE IF NOT EXISTS files (repository TEXT NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL, ext TEXT NOT NULL, "
    "dir TEXT NOT NULL, depth INTEGER NOT NULL, status TEXT NOT NULL, is_dir INTEGER NOT NULL DEFAULT 0, bytes INTEGER, "
    "mtime REAL, lines INTEGER, binary INTEGER, blob TEXT, mode TEXT, commits INTEGER, first_commit_at REAL, "
    "last_commit_at REAL, last_commit TEXT, indexed_at REAL NOT NULL, PRIMARY KEY(repository, path)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS files_ext ON files(ext)",
    "CREATE INDEX IF NOT EXISTS files_bytes ON files(bytes)",
    "CREATE INDEX IF NOT EXISTS files_mtime ON files(mtime)",
    "CREATE INDEX IF NOT EXISTS files_status ON files(status)",
    "CREATE INDEX IF NOT EXISTS files_name ON files(name)",
    "CREATE TABLE IF NOT EXISTS file_index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)


class IndexError_(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def connect(db_path):
    db = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    for statement in DDL:
        db.execute(statement)
    return db


def _git(path, *args) -> list:
    """NUL-separated output of one Git command, as a list."""
    done = subprocess.run(["git", "-C", str(path), "-c", "core.quotepath=off", *args], capture_output=True)
    if done.returncode != 0:
        raise IndexError_(f"git {args[0]} failed in {path}: {done.stderr.decode('utf-8', 'replace').strip()[-200:]}", 500)
    return [part for part in done.stdout.decode("utf-8", "replace").split("\0") if part]


def _history(path) -> dict:
    """path -> [commits, first time, last time, last commit], from one pass over what is checked out."""
    done = subprocess.run(["git", "-C", str(path), "-c", "core.quotepath=off", "log", "--format=%x01%H%x09%ct", "--name-only",
                           "--no-renames", "-z"], capture_output=True)
    seen = {}
    if done.returncode != 0:
        return seen
    for chunk in done.stdout.decode("utf-8", "replace").split("\x01")[1:]:
        head, _, names = chunk.partition("\n")
        commit, _, stamp = head.strip("\0").partition("\t")
        when = float(stamp.strip("\0\n") or 0)
        for name in names.split("\0"):
            name = name.strip("\n")
            if not name:
                continue
            row = seen.get(name)
            if row is None:
                seen[name] = [1, when, when, commit]          # newest first: the first sight is the last commit
            else:
                row[0] += 1
                row[1] = when
    return seen


def _count_lines(file, size) -> int | None:
    if size > MAX_COUNTED_BYTES:
        return None
    try:
        with open(file, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def scan(name, path, previous, now) -> list:
    """The rows of one repository's own checkout."""
    root = Path(path)
    history = _history(root)
    binary = {}
    for line in _git(root, "ls-files", "--eol", "-z"):
        info, _, file = line.partition("\t")
        binary[file] = 1 if "w/-text" in info or ("w/" not in info and "i/-text" in info) else 0
    changed = {entry[3:] for entry in _git(root, "status", "--porcelain", "-z", "--untracked-files=no") if len(entry) > 3}
    listed = []
    for line in _git(root, "ls-files", "-s", "-z"):
        info, _, file = line.partition("\t")
        mode, blob, _stage = info.split(" ")
        listed.append((file, "modified" if file in changed else "tracked", mode, blob))
    listed += [(file, "untracked", None, None) for file in _git(root, "ls-files", "--others", "--exclude-standard", "-z")]
    listed += [(file, "ignored", None, None)
               for file in _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z")]
    rows, taken = [], set()
    for file, status, mode, blob in listed:
        if file in taken:
            continue
        taken.add(file)
        is_dir = 1 if file.endswith("/") else 0
        clean = file.rstrip("/")
        base = clean.rsplit("/", 1)[-1]
        ext = "" if is_dir or "." not in base.lstrip(".") else base.rsplit(".", 1)[-1].lower()
        size = mtime = lines = None
        if not is_dir:
            try:
                stat = os.stat(root / clean)
                size, mtime = stat.st_size, stat.st_mtime
            except OSError:
                pass
        is_binary = binary.get(file)
        old = previous.get((name, clean))
        if old is not None and old[0] == size and old[1] == mtime:
            lines = old[2]
        elif size is not None and status != "ignored" and not is_binary:
            lines = _count_lines(root / clean, size)
        past = history.get(clean) or [None, None, None, None]
        rows.append((name, clean, base, ext, clean.rsplit("/", 1)[0] if "/" in clean else "", clean.count("/"), status, is_dir,
                     size, mtime, lines, is_binary, blob, mode, past[0], past[1], past[2], past[3], now))
    return rows


def build(policy, db_path, *, actor, now=None, log=print) -> dict:
    """Replace the index with what the core holds now. Only the policy's cleanup account does this."""
    if actor != policy["cleaner"]:
        raise IndexError_(f"only {policy['cleaner']} keeps the file index; this run acts as {actor or 'nobody'}", 403)
    now = time.time() if now is None else now
    db = connect(db_path)
    try:
        previous = {(r[0], r[1]): (r[2], r[3], r[4]) for r in db.execute("SELECT repository, path, bytes, mtime, lines FROM files")}
        counts = {}
        for name, declared in sorted(policy["repositories"].items()):
            if not Path(declared["path"], ".git").exists():
                continue
            rows = scan(name, declared["path"], previous, now)
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM files WHERE repository = ?", (name,))
            db.executemany(f"INSERT INTO files ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})", rows)
            db.execute("COMMIT")
            counts[name] = len(rows)
            log(f"[index] {name}: {len(rows)} entries")
        known = "', '".join(counts)
        db.execute(f"DELETE FROM files WHERE repository NOT IN ('{known}')" if counts else "DELETE FROM files")
        meta = {"indexed_at": now, "indexed_by": actor, "entries": sum(counts.values()), "repositories": counts,
                "seconds": round(time.time() - now, 1) if now <= time.time() else 0}
        db.execute("INSERT INTO file_index_meta VALUES ('last', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                   (json.dumps(meta),))
        return meta
    finally:
        db.close()


def _number(value, what):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise IndexError_(f"{what} must be a number") from None


def search(db, *, q=None, repository=None, ext=None, status=None, under=None, min_bytes=None, max_bytes=None,
           newer_than=None, older_than=None, binary=None, order="path", limit=100, offset=0) -> dict:
    """Filter the flat set. ``order`` is a column of ORDERS, with a leading ``-`` for descending."""
    where, values = [], []
    if q:
        where.append("path LIKE ? ESCAPE '\\'")
        values.append("%" + str(q).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    for column, value in (("repository", repository), ("status", status)):
        if value:
            where.append(f"{column} = ?")
            values.append(str(value))
    if status and status not in STATUSES:
        raise IndexError_("status must be one of " + ", ".join(STATUSES))
    if ext is not None and ext != "":
        where.append("ext = ?")
        values.append(str(ext).lower().lstrip("."))
    if under:
        where.append("(dir = ? OR dir LIKE ? ESCAPE '\\')")
        folder = str(under).strip("/")
        values += [folder, folder.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"]
    for column, sign, value, what in (("bytes", ">=", min_bytes, "min_bytes"), ("bytes", "<=", max_bytes, "max_bytes"),
                                      ("mtime", ">=", newer_than, "newer_than"), ("mtime", "<=", older_than, "older_than")):
        number = _number(value, what)
        if number is not None:
            where.append(f"{column} {sign} ?")
            values.append(number)
    if binary is not None and binary != "":
        where.append("binary = ?")
        values.append(1 if str(binary).lower() in ("1", "true", "yes") else 0)
    descending = str(order).startswith("-")
    column = str(order).lstrip("-")
    if column not in ORDERS:
        raise IndexError_("order must be one of " + ", ".join(ORDERS) + ", with - in front for descending")
    limit = max(1, min(int(_number(limit, "limit") or 100), 500))
    offset = max(0, int(_number(offset, "offset") or 0))
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total, size = db.execute(f"SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM files{clause}", values).fetchone()
    rows = db.execute(f"SELECT {', '.join(COLUMNS)} FROM files{clause} ORDER BY {column} IS NULL, {column} "
                      f"{'DESC' if descending else 'ASC'}, repository, path LIMIT ? OFFSET ?", values + [limit, offset]).fetchall()
    return {"total": total, "bytes": size, "files": [dict(zip(COLUMNS, row)) for row in rows],
            "next_offset": offset + len(rows) if offset + len(rows) < total else None}


def summary(db) -> dict:
    last = db.execute("SELECT value FROM file_index_meta WHERE key = 'last'").fetchone()
    return {"last": json.loads(last[0]) if last else None,
            "by_status": [{"repository": r[0], "status": r[1], "entries": r[2], "bytes": r[3]} for r in db.execute(
                "SELECT repository, status, COUNT(*), COALESCE(SUM(bytes), 0) FROM files GROUP BY repository, status ORDER BY 1, 2")],
            "largest_kinds": [{"ext": r[0], "entries": r[1], "bytes": r[2]} for r in db.execute(
                "SELECT ext, COUNT(*), COALESCE(SUM(bytes), 0) FROM files GROUP BY ext ORDER BY 3 DESC LIMIT 20")]}


def http_route(self, path, *, db_path):
    """``GET /api/files/search`` and ``GET /api/files/summary``: the index, read-only (dev_ui, signed in)."""
    from urllib.parse import parse_qs
    self._set_cookie = []
    if self._principal().user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "dev_ui" not in self._caps():
        return self._json({"error": "this needs dev_ui"}, 403)
    if not Path(db_path).is_file():
        return self._json({"error": "the file index has not been built yet"}, 503)
    query = parse_qs(self.path.partition("?")[2])
    allowed = {"q", "repository", "ext", "status", "under", "min_bytes", "max_bytes", "newer_than", "older_than", "binary",
               "order", "limit", "offset"}
    if set(query) - allowed or any(len(v) != 1 for v in query.values()):
        return self._json({"error": "unexpected search query"}, 400)
    db = connect(db_path)
    try:
        if path == "/api/files/summary":
            return self._json(summary(db))
        if path != "/api/files/search":
            return self._json({"error": "not found"}, 404)
        return self._json(search(db, **{key: values[0] for key, values in query.items()}))
    except IndexError_ as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--db", required=True, help="the file index database")
    parser.add_argument("--as", dest="actor", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("summary")
    find = sub.add_parser("search")
    for name in ("q", "repository", "ext", "status", "under", "min-bytes", "max-bytes", "newer-than", "older-than", "binary",
                 "order", "limit", "offset"):
        find.add_argument("--" + name, default=None)
    args = parser.parse_args(argv)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import branch_gate
        policy = branch_gate.load_policy(args.policy)
        if args.command == "build":
            print(json.dumps(build(policy, args.db, actor=args.actor)))
            return 0
        db = connect(args.db)
        try:
            if args.command == "summary":
                print(json.dumps(summary(db), indent=1))
            else:
                given = {k: v for k, v in vars(args).items() if k not in ("policy", "db", "actor", "command") and v is not None}
                found = search(db, **given)
                for row in found["files"]:
                    print(f"{row['bytes'] if row['bytes'] is not None else '-':>12} {row['status']:9} {row['repository']}/{row['path']}")
                print(f"{found['total']} entries, {found['bytes']} bytes")
        finally:
            db.close()
        return 0
    except (IndexError_, OSError, ValueError) as error:
        print(f"file_index: {getattr(error, 'message', error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
