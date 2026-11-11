"""Read-only access to the repository world maps under <state content dir>/worldmap/ (server.py [WORLDMAP], web/worldmap.html).

Layout (written by another tool, never by this module):  worldmap/index.json, worldmap/<owner>__<name>/<commit>/map.json.
The only request-derived parts are a repo dir name and a 40-hex commit; both are validated and the resolved file must stay inside the worldmap dir.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_DIR_RE = re.compile(r"^[A-Za-z0-9._-]+__[A-Za-z0-9._-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class WorldMapError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message, self.status = message, status


def root(paths) -> Path:
    return Path(paths.content_dir) / "worldmap"


def index(paths) -> dict:
    """{"repos": [...]}; a missing index is an empty list; a bare-list index file is accepted."""
    file = root(paths) / "index.json"
    if not file.is_file():
        return {"repos": []}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise WorldMapError("index.json unreadable", 500)
    repos = data.get("repos") if isinstance(data, dict) else data
    return {"repos": [r for r in repos if isinstance(r, dict)] if isinstance(repos, list) else []}


def map_path(paths, repo_dir: str, commit: str) -> Path:
    if not isinstance(repo_dir, str) or not REPO_DIR_RE.match(repo_dir) or repo_dir.startswith(".") or ".." in repo_dir:
        raise WorldMapError("bad repo name", 400)
    if not isinstance(commit, str) or not COMMIT_RE.match(commit):
        raise WorldMapError("bad commit", 400)
    base = root(paths).resolve()
    target = (base / repo_dir / commit / "map.json").resolve()
    if base not in target.parents:
        raise WorldMapError("bad path", 400)
    if not target.is_file():
        raise WorldMapError("no such map", 404)
    return target
