"""One schema-version gate for every module that creates tables in accounts.db (and sibling sqlite files).

Each module calls ``declare(conn, "<module>", <version>)`` right after its own DDL. A stored version greater than the code's
refuses (``SchemaVersionError``) so an older checkout never touches a newer database; a lower stored version is updated (the
module's own DDL owns its forward migration); absent is inserted. Rule: one writer (module) per table.
"""
from __future__ import annotations

import time
from pathlib import Path

SCHEMA_TABLE_DDL = ("CREATE TABLE IF NOT EXISTS schema_version (module TEXT PRIMARY KEY, version INTEGER NOT NULL, "
                    "updated_at REAL NOT NULL, writer TEXT NOT NULL)")


class SchemaVersionError(RuntimeError):
    def __init__(self, module, stored, version):
        self.module, self.stored, self.version = module, stored, version
        super().__init__(f"schema_version: module {module!r} is at version {stored} in the database but this code declares {version}; "
                         "refusing to run an older checkout against a newer database")


def _git_head() -> str:
    """Head of the repo root from .git/HEAD (no subprocess), 12 chars, or 'unknown'."""
    try:
        for root in Path(__file__).resolve().parents:
            git = root / ".git"
            if git.is_file():       # worktree: 'gitdir: <path>'
                text = git.read_text(encoding="utf-8").strip()
                if text.startswith("gitdir:"):
                    git = Path(text[7:].strip())
                    if not git.is_absolute():
                        git = (root / git).resolve()
            if git.is_dir():
                head = (git / "HEAD").read_text(encoding="utf-8").strip()
                if head.startswith("ref:"):
                    ref = head[4:].strip()
                    for base in (git, *([git.parent.parent] if (git / "commondir").exists() else [])):
                        p = base / ref
                        if p.is_file():
                            return p.read_text(encoding="utf-8").strip()[:12]
                    common = (git / "commondir")
                    if common.exists():
                        cd = (git / common.read_text(encoding="utf-8").strip()).resolve()
                        p = cd / ref
                        if p.is_file():
                            return p.read_text(encoding="utf-8").strip()[:12]
                    packed = (git / "packed-refs")
                    if packed.is_file():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if line.endswith(" " + ref):
                                return line.split()[0][:12]
                    return "unknown"
                return head[:12] or "unknown"
    except OSError:
        pass
    return "unknown"


def declare(conn, module: str, version: int, writer: str | None = None) -> None:
    conn.execute(SCHEMA_TABLE_DDL)
    row = conn.execute("SELECT version FROM schema_version WHERE module = ?", (module,)).fetchone()
    stored = None if row is None else int(row[0])
    if stored is not None and stored > version:
        raise SchemaVersionError(module, stored, version)
    if stored == version:
        return
    w = writer or _git_head()
    if stored is None:
        conn.execute("INSERT INTO schema_version (module, version, updated_at, writer) VALUES (?, ?, ?, ?)", (module, version, time.time(), w))
    else:
        conn.execute("UPDATE schema_version SET version = ?, updated_at = ?, writer = ? WHERE module = ?", (version, time.time(), w, module))
    conn.commit()


def report(conn) -> list:
    conn.execute(SCHEMA_TABLE_DDL)
    return [tuple(r) for r in conn.execute("SELECT module, version, updated_at, writer FROM schema_version ORDER BY module")]
