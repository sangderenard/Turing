"""Account-owned published pages shared by the stable and development hosts.

The Accounts database is the common identity and publication authority. Uploaded
pages are retained as bounded public copies in that database. Compiler bundles
remain in the measured account workspace that produced them; only a trusted
builder may register one. Neither an HTTP body nor a browser path names a host
file. A release marker controls both feed membership and direct reads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid

from meowdoku import schema_version

SCHEMA_VERSION = 1
PAGE_ID = re.compile(r"[0-9a-f]{32}\Z")
BUNDLE_URL = re.compile(r"/site/programs/([a-z0-9][a-z0-9-]{0,63})/versions/([A-Za-z0-9._-]{1,80})/(.+)\Z")
MAX_UPLOAD = 512 * 1024
MAX_PAGES_PER_AUTHOR = 32
MAX_AUTHOR_UPLOAD = 4 * 1024 * 1024
STAGES = frozenset(("dev", "release"))


class PageError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def connect(path):
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    schema_version.declare(db, "authored_pages", SCHEMA_VERSION)
    db.execute("""CREATE TABLE IF NOT EXISTS authored_pages (
        id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
        title TEXT NOT NULL, description TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('upload','bundle')),
        stage TEXT NOT NULL CHECK(stage IN ('dev','release')),
        content BLOB, content_type TEXT,
        workspace_id TEXT, bundle_root TEXT, bundle_route TEXT, bundle_inventory TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        CHECK((kind='upload')=(content IS NOT NULL)),
        CHECK((kind='bundle')=(bundle_root IS NOT NULL)),
        CHECK((kind='bundle')=(bundle_route IS NOT NULL)))""")
    db.execute("CREATE INDEX IF NOT EXISTS authored_pages_feed ON authored_pages(stage,updated_at DESC,id DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS authored_pages_owner ON authored_pages(owner_id,updated_at DESC,id DESC)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS authored_pages_bundle ON authored_pages(bundle_root) WHERE bundle_root IS NOT NULL")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS authored_pages_bundle_route ON authored_pages(bundle_route) WHERE bundle_route IS NOT NULL")
    db.commit()
    return db


def _owner(db, actor):
    if type(actor) is not int or actor <= 0 or db.execute("SELECT 1 FROM users WHERE id=?", (actor,)).fetchone() is None:
        raise PageError("sign in to publish a page", 401)


def _metadata(title, description):
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120 or "\x00" in title:
        raise PageError("title must contain 1–120 characters")
    if not isinstance(description, str) or len(description) > 500 or "\x00" in description:
        raise PageError("description must contain at most 500 characters")
    return title.strip(), description.strip()


def _counts(db, actor):
    return db.execute("SELECT COUNT(*),COALESCE(SUM(LENGTH(content)),0) FROM authored_pages WHERE owner_id=?", (actor,)).fetchone()


def upload(db, actor, *, title, description, content, content_type):
    """Explicit public copy; encrypted private files never enter this API."""
    _owner(db, actor)
    title, description = _metadata(title, description)
    if content_type not in ("text/html", "text/plain") or type(content) is not bytes or not 1 <= len(content) <= MAX_UPLOAD:
        raise PageError("page must be HTML or plain text, at most 512 KiB", 413)
    try:
        content.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise PageError("page must be UTF-8") from None
    db.execute("BEGIN IMMEDIATE")
    try:
        count, used = _counts(db, actor)
        if count >= MAX_PAGES_PER_AUTHOR or used + len(content) > MAX_AUTHOR_UPLOAD:
            raise PageError("page publication quota reached", 429)
        page_id, now = uuid.uuid4().hex, time.time()
        db.execute("INSERT INTO authored_pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (page_id, actor, title, description, "upload", "dev", content, content_type,
                    None, None, None, None, now, now))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return {"id": page_id, "stage": "dev", "kind": "upload"}


def _bundle_inventory(directory):
    supplied = Path(directory).absolute()
    root = supplied.resolve(strict=True)
    if supplied != root or not root.is_dir() or root.is_symlink():
        raise PageError("bundle directory unavailable")
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema") != "turing-program-bundle-v1"
            or not isinstance(manifest.get("page"), dict) or manifest["page"].get("path") != "index.html"):
        raise PageError("expected a completed Turing site bundle")
    inventory = {}
    if not isinstance(manifest.get("artifacts"), list):
        raise PageError("invalid bundle inventory")
    for item in manifest["artifacts"]:
        if not isinstance(item, dict):
            raise PageError("invalid bundle inventory")
        path, digest, size = item.get("path"), item.get("sha256"), item.get("bytes")
        if (not isinstance(path, str) or not 1 <= len(path) <= 240 or path.startswith("/")
                or any(part in ("", ".", "..") or part.endswith((" ", "."))
                       or any(ord(char) < 32 or char in '\\:<>"|?*' for char in part)
                       or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
                       or part.casefold() == ".git" for part in path.split("/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int or size < 0 or path in inventory):
            raise PageError("invalid bundle inventory")
        candidate = root.joinpath(*path.split("/"))
        if candidate.resolve() != candidate or not candidate.is_file() or candidate.is_symlink():
            raise PageError("bundle contains redirected files")
        info = candidate.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != size:
            raise PageError("bundle inventory does not match files")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise PageError("bundle inventory does not match files")
        inventory[path] = (size, digest)
    if "index.html" not in inventory:
        raise PageError("bundle has no page")
    if root.parent.name != "versions" or root.parent.parent.parent.name != "programs" or root.parent.parent.parent.parent.name != "site":
        raise PageError("bundle has no Turing site route")
    slug, version = root.parent.parent.name, root.name
    route = f"/site/programs/{slug}/versions/{version}/"
    if not BUNDLE_URL.fullmatch(route + "index.html") or manifest["page"].get("url") != route + "index.html":
        raise PageError("bundle page route does not match its immutable directory")
    return root, route, inventory, manifest


def register_bundle(db, actor, workspace_id, directory):
    """Trusted builder entry: bind a completed bundle to its measured workspace owner.

    The HTTP layer does not expose this function. The caller must have finished
    the site_bundle build in the existing account workspace and reconciled its
    measured usage before registration.
    """
    _owner(db, actor)
    row = db.execute("SELECT w.owner_id,w.state,w.measured_bytes,v.root FROM workspace_leases w JOIN workspace_volumes v USING(volume_id) WHERE w.workspace_id=?", (workspace_id,)).fetchone()
    if row is None or row["owner_id"] != actor or row["state"] != "active":
        raise PageError("owned active workspace required", 404)
    base = Path(row["root"]).resolve(strict=True)
    root, route, inventory, manifest = _bundle_inventory(directory)
    if not root.is_relative_to(base / workspace_id):
        raise PageError("bundle must be inside the account workspace")
    if row["measured_bytes"] < sum(size for size, _ in inventory.values()) + (root / "bundle.json").stat().st_size:
        raise PageError("bundle bytes have not been measured in the account workspace")
    title = str(manifest.get("program", {}).get("title", ""))
    title, description = _metadata(title, "")
    db.execute("BEGIN IMMEDIATE")
    try:
        previous = db.execute("SELECT id,stage,owner_id FROM authored_pages WHERE bundle_root=?", (str(root),)).fetchone()
        if previous is not None:
            if previous["owner_id"] != actor:
                raise PageError("bundle already belongs to another author", 409)
            db.commit()
            return {"id": previous["id"], "stage": previous["stage"], "kind": "bundle"}
        if db.execute("SELECT 1 FROM authored_pages WHERE bundle_route=?", (route,)).fetchone():
            raise PageError("bundle route already belongs to another page; use an account-specific slug", 409)
        if _counts(db, actor)[0] >= MAX_PAGES_PER_AUTHOR:
            raise PageError("page publication quota reached", 429)
        page_id, now = uuid.uuid4().hex, time.time()
        db.execute("INSERT INTO authored_pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (page_id, actor, title, description, "bundle", "dev", None, None,
                    workspace_id, str(root), route, json.dumps(inventory, separators=(",", ":")), now, now))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return {"id": page_id, "stage": "dev", "kind": "bundle"}


def stage(db, actor, page_id, value):
    _owner(db, actor)
    if not isinstance(page_id, str) or not PAGE_ID.fullmatch(page_id) or value not in STAGES:
        raise PageError("invalid page or stage")
    fields = "owner_id,kind,stage,updated_at,bundle_root,bundle_route,bundle_inventory,workspace_id"
    row = db.execute(f"SELECT {fields} FROM authored_pages WHERE id=?", (page_id,)).fetchone()
    if row is None or row["owner_id"] != actor:
        raise PageError("page not found", 404)
    if value == "release" and row["kind"] == "bundle":
        try:
            root, route, inventory, _ = _bundle_inventory(row["bundle_root"])
        except (PageError, OSError, ValueError, json.JSONDecodeError):
            raise PageError("bundle changed since registration", 409) from None
        stored = {key: tuple(item) for key, item in json.loads(row["bundle_inventory"]).items()}
        if str(root) != row["bundle_root"] or route != row["bundle_route"] or inventory != stored:
            raise PageError("bundle changed since registration", 409)
    db.execute("BEGIN IMMEDIATE")
    try:
        current = db.execute(f"SELECT {fields} FROM authored_pages WHERE id=?", (page_id,)).fetchone()
        if current is None or tuple(current) != tuple(row):
            raise PageError("page changed; refresh before retrying", 409)
        if value == "release" and row["kind"] == "bundle":
            held = db.execute("SELECT state FROM workspace_leases WHERE workspace_id=? AND owner_id=?",
                              (row["workspace_id"], actor)).fetchone()
            if held is None or held["state"] != "active":
                raise PageError("bundle workspace is unavailable", 409)
        db.execute("UPDATE authored_pages SET stage=?,updated_at=? WHERE id=?", (value, time.time(), page_id))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return {"id": page_id, "stage": value}


def listing(db, *, site, actor=None, author=None, mine=False, limit=50):
    if site not in STAGES or type(limit) is not int or not 1 <= limit <= 100:
        raise PageError("invalid feed request")
    where, args = [], []
    if mine:
        _owner(db, actor)
        where.append("p.owner_id=?")
        args.append(actor)
    elif site == "release":
        where.append("p.stage='release'")
    else:
        where.append("(p.stage='release' OR p.owner_id=?)")
        args.append(actor or -1)
    if author is not None:
        where.append("u.username=? COLLATE NOCASE")
        args.append(author)
    rows = db.execute("SELECT p.id,p.title,p.description,p.kind,p.stage,p.created_at,p.updated_at,u.username AS author "
                      "FROM authored_pages p JOIN users u ON u.id=p.owner_id WHERE " + " AND ".join(where) +
                      " ORDER BY p.updated_at DESC,p.id DESC LIMIT ?", (*args, limit)).fetchall()
    return [dict(row) for row in rows]


def read(db, page_id, *, site, actor=None, path="index.html"):
    if not isinstance(page_id, str) or not PAGE_ID.fullmatch(page_id):
        raise PageError("page not found", 404)
    row = db.execute("SELECT p.*,u.username AS author FROM authored_pages p JOIN users u ON u.id=p.owner_id WHERE p.id=?", (page_id,)).fetchone()
    if row is None or (site == "release" and row["stage"] != "release") or (site == "dev" and row["stage"] != "release" and row["owner_id"] != actor):
        raise PageError("page not found", 404)
    if row["kind"] == "upload":
        if path != "index.html":
            raise PageError("page asset not found", 404)
        return bytes(row["content"]), row["content_type"] + "; charset=utf-8"
    inventory = json.loads(row["bundle_inventory"])
    if path not in inventory or not isinstance(path, str) or "\\" in path:
        raise PageError("page asset not found", 404)
    root = Path(row["bundle_root"])
    target = root.joinpath(*path.split("/"))
    try:
        if target.resolve(strict=True) != target or not target.is_file() or target.is_symlink():
            raise PageError("page asset unavailable", 503)
        data = target.read_bytes()
    except OSError:
        raise PageError("page asset unavailable", 503) from None
    size, digest = inventory[path]
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise PageError("page asset unavailable", 503)
    import mimetypes
    return data, mimetypes.guess_type(path)[0] or "application/octet-stream"


def read_bundle_url(db, url, *, site, actor=None):
    """Resolve the absolute URLs emitted by the existing Turing site bundle."""
    match = BUNDLE_URL.fullmatch(url)
    if match is None:
        raise PageError("page asset not found", 404)
    route = url[:match.start(3)]
    row = db.execute("SELECT id FROM authored_pages WHERE bundle_route=?", (route,)).fetchone()
    if row is None:
        raise PageError("page asset not found", 404)
    return read(db, row["id"], site=site, actor=actor, path=match[3])
