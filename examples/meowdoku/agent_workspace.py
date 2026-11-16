"""Shared operator documents and account-owned agent scratch text.

Documents are indexed by parent and fetched separately so an MCP orientation
never needs to load a whole personality tree. Scratch is deliberately separate
from the browser's end-to-end encrypted user-file attachments.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from http.cookies import SimpleCookie
from pathlib import Path
from meowdoku import schema_version

DOC_KINDS = frozenset({"orientation", "prompt", "personality", "guidance", "trouble", "scenario"})


def _limit(name, default, ceiling):
    try:
        value = int(os.environ.get(name, default))
        return value if 1 <= value <= ceiling else default
    except ValueError:
        return default


MAX_PROMPTS_PER_OPERATOR = _limit("MEOWDOKU_OPERATOR_PROMPTS", 32, 256)
MAX_DOCS = _limit("MEOWDOKU_AGENT_DOCUMENTS", 512, 4096)
MAX_DOC_BYTES = _limit("MEOWDOKU_AGENT_DOCUMENT_BYTES", 64 * 1024, 256 * 1024)
MAX_SCRATCH_FILES = _limit("MEOWDOKU_SCRATCH_FILES", 128, 1024)
MAX_SCRATCH_FILE_BYTES = _limit("MEOWDOKU_SCRATCH_FILE_BYTES", 128 * 1024, 256 * 1024)
MAX_SCRATCH_BYTES = _limit("MEOWDOKU_SCRATCH_BYTES", 8 * 1024 * 1024, 64 * 1024 * 1024)
TEMP_COOKIE = "meowdoku_scooby_files"
MAX_TEMP_BYTES = 2048


class WorkspaceError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    schema_version.declare(db, "agent_workspace", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_documents ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, kind TEXT NOT NULL, "
               "parent_id TEXT REFERENCES agent_documents(id), title TEXT NOT NULL, body TEXT NOT NULL, "
               "contexts TEXT NOT NULL DEFAULT '[]', updated REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_documents_parent ON agent_documents(parent_id,kind,title)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_documents_owner_kind ON agent_documents(owner_id,kind)")
    db.execute("CREATE TABLE IF NOT EXISTS agent_scratch ("
               "owner_id INTEGER NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL, "
               "PRIMARY KEY(owner_id,name))")
    return db


def _text(value, label, maximum=MAX_DOC_BYTES):
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
        raise WorkspaceError(label + " is too large or invalid")
    return value


def _name(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 120 or any(c in value for c in "\\/\r\n\0"):
        raise WorkspaceError("invalid scratch name")
    return value


def index(db):
    roots = [dict(row) for row in db.execute(
        "SELECT id,owner_id,kind,parent_id,title,updated FROM agent_documents WHERE parent_id IS NULL ORDER BY kind,title")]
    return {"roots": roots, "kinds": sorted(DOC_KINDS),
            "quotas": {"prompts_per_operator": MAX_PROMPTS_PER_OPERATOR, "documents": MAX_DOCS,
                       "document_bytes": MAX_DOC_BYTES},
            "orientation": "Read the index, then open only the branches and documents needed for the present task. "
                           "Scenario context ids are ordered from broad to specific."}


def children(db, parent_id=None, kind=None):
    if kind is not None and kind not in DOC_KINDS:
        raise WorkspaceError("invalid document kind")
    if parent_id is not None and db.execute("SELECT 1 FROM agent_documents WHERE id=?", (parent_id,)).fetchone() is None:
        raise WorkspaceError("parent document not found", 404)
    sql = "SELECT id,owner_id,kind,parent_id,title,updated FROM agent_documents WHERE parent_id IS ?"
    args = [parent_id]
    if kind:
        sql += " AND kind=?"; args.append(kind)
    return {"documents": [dict(row) for row in db.execute(sql + " ORDER BY kind,title", args)]}


def document(db, doc_id):
    row = db.execute("SELECT * FROM agent_documents WHERE id=?", (doc_id,)).fetchone()
    if row is None:
        raise WorkspaceError("document not found", 404)
    return {**dict(row), "contexts": json.loads(row["contexts"])}


def _contexts(db, values):
    if not isinstance(values, list) or len(values) > 64 or any(not isinstance(v, str) for v in values):
        raise WorkspaceError("scenario contexts must be a list of at most 64 document ids")
    if len(values) != len(set(values)):
        raise WorkspaceError("scenario contexts must be distinct")
    for value in values:
        if db.execute("SELECT 1 FROM agent_documents WHERE id=?", (value,)).fetchone() is None:
            raise WorkspaceError("scenario context document not found", 404)
    return json.dumps(values, separators=(",", ":"))


def save_document(db, actor_id: int, data: dict):
    kind = data.get("kind")
    if kind not in DOC_KINDS:
        raise WorkspaceError("invalid document kind")
    title = _text(data.get("title"), "title", 160).strip()
    body = _text(data.get("body", ""), "document")
    if not title:
        raise WorkspaceError("title is required")
    doc_id = data.get("id")
    parent = data.get("parent_id")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise WorkspaceError("invalid parent id")
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT owner_id,kind FROM agent_documents WHERE id=?", (doc_id,)).fetchone() if doc_id else None
        if doc_id and old is None:
            raise WorkspaceError("document not found", 404)
        if not doc_id:
            if db.execute("SELECT COUNT(*) FROM agent_documents").fetchone()[0] >= MAX_DOCS:
                raise WorkspaceError("document quota reached", 429)
            if kind == "prompt" and db.execute("SELECT COUNT(*) FROM agent_documents WHERE owner_id=? AND kind='prompt'",
                                               (actor_id,)).fetchone()[0] >= MAX_PROMPTS_PER_OPERATOR:
                raise WorkspaceError("operator prompt quota reached", 429)
            doc_id = secrets.token_hex(16)
        elif kind == "prompt" and old["kind"] != "prompt" and db.execute(
                "SELECT COUNT(*) FROM agent_documents WHERE owner_id=? AND kind='prompt'", (old["owner_id"],)).fetchone()[0] >= MAX_PROMPTS_PER_OPERATOR:
            raise WorkspaceError("operator prompt quota reached", 429)
        if parent:
            cursor = parent
            while cursor:
                if cursor == doc_id:
                    raise WorkspaceError("document tree cannot contain a cycle")
                row = db.execute("SELECT parent_id FROM agent_documents WHERE id=?", (cursor,)).fetchone()
                if row is None:
                    raise WorkspaceError("parent document not found", 404)
                cursor = row[0]
        contexts = _contexts(db, data.get("contexts", [])) if kind == "scenario" else "[]"
        if doc_id in json.loads(contexts):
            raise WorkspaceError("scenario cannot include itself")
        if old:
            db.execute("UPDATE agent_documents SET kind=?,parent_id=?,title=?,body=?,contexts=?,updated=? WHERE id=?",
                       (kind, parent, title, body, contexts, time.time(), doc_id))
        else:
            db.execute("INSERT INTO agent_documents VALUES (?,?,?,?,?,?,?,?)",
                       (doc_id, actor_id, kind, parent, title, body, contexts, time.time()))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return document(db, doc_id)


def delete_document(db, doc_id):
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("SELECT 1 FROM agent_documents WHERE id=?", (doc_id,)).fetchone() is None:
            raise WorkspaceError("document not found", 404)
        if db.execute("SELECT 1 FROM agent_documents WHERE parent_id=?", (doc_id,)).fetchone():
            raise WorkspaceError("remove child documents first", 409)
        for row in db.execute("SELECT contexts FROM agent_documents WHERE kind='scenario'"):
            if doc_id in json.loads(row[0]):
                raise WorkspaceError("document is used by a scenario", 409)
        db.execute("DELETE FROM agent_documents WHERE id=?", (doc_id,)); db.commit()
    except BaseException:
        db.rollback(); raise
    return {"deleted": doc_id}


def scratch_list(db, owner_id):
    rows = [dict(row) for row in db.execute(
        "SELECT name,length(CAST(body AS BLOB)) AS bytes,updated FROM agent_scratch WHERE owner_id=? ORDER BY name", (owner_id,))]
    return {"files": rows, "quota": {"files": MAX_SCRATCH_FILES, "file_bytes": MAX_SCRATCH_FILE_BYTES,
                                     "total_bytes": MAX_SCRATCH_BYTES}, "used_bytes": sum(row["bytes"] for row in rows)}


def scratch_read(db, owner_id, name):
    row = db.execute("SELECT body,updated FROM agent_scratch WHERE owner_id=? AND name=?", (owner_id, _name(name))).fetchone()
    if row is None:
        raise WorkspaceError("scratch file not found", 404)
    return {"name": name, "body": row["body"], "updated": row["updated"]}


def scratch_write(db, owner_id, name, body):
    name = _name(name); body = _text(body, "scratch file", MAX_SCRATCH_FILE_BYTES)
    db.execute("BEGIN IMMEDIATE")
    try:
        used, count = db.execute("SELECT COALESCE(SUM(length(CAST(body AS BLOB))),0),COUNT(*) FROM agent_scratch WHERE owner_id=?",
                                 (owner_id,)).fetchone()
        prior = db.execute("SELECT length(CAST(body AS BLOB)) FROM agent_scratch WHERE owner_id=? AND name=?",
                           (owner_id, name)).fetchone()
        if (count >= MAX_SCRATCH_FILES and prior is None) or used - (prior[0] if prior else 0) + len(body.encode()) > MAX_SCRATCH_BYTES:
            raise WorkspaceError("scratch quota reached", 429)
        db.execute("INSERT INTO agent_scratch VALUES (?,?,?,?) ON CONFLICT(owner_id,name) DO UPDATE SET body=excluded.body,updated=excluded.updated",
                   (owner_id, name, body, time.time()))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return scratch_read(db, owner_id, name)


def scratch_delete(db, owner_id, name):
    name = _name(name)
    if db.execute("DELETE FROM agent_scratch WHERE owner_id=? AND name=?", (owner_id, name)).rowcount == 0:
        raise WorkspaceError("scratch file not found", 404)
    return {"deleted": name}


def temp_files(cookie_header, owner_id):
    import base64
    import binascii
    jar = SimpleCookie()
    try:
        jar.load(cookie_header or "")
        raw = jar[TEMP_COOKIE].value if TEMP_COOKIE in jar else ""
        value = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))) if raw else {}
        files = value.get("files") if isinstance(value, dict) and value.get("owner") == owner_id else None
        if isinstance(files, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
            return files
    except (ValueError, KeyError, UnicodeError, binascii.Error):
        pass
    return {}


def temp_cookie(owner_id, files, secure=True):
    import base64
    raw = json.dumps({"owner": owner_id, "files": files}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_TEMP_BYTES:
        raise WorkspaceError("temporary cookie filesystem is full", 429)
    value = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return TEMP_COOKIE + "=" + value + "; Path=/api/scratch; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")
