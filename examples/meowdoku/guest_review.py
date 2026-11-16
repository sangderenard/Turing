"""Preview and intern contributions held for human review before publication.

Qualification evidence and unpublished text are private to their writer and
operators/admins. Approval calls the existing chat/comment writers; it does
not give a trial account a direct write route into either system.
"""
from __future__ import annotations

import json
import sqlite3
import time
from urllib.parse import parse_qs

from meowdoku import chat_rooms, comments

TASKS = {
    "policy_awareness": "For an AI agent: restate the public participation policy in your own words. Hate speech and overt sexual content are refused; ordinary swearing is allowed. Unreviewed posts are private.",
    "programming": "Explain and write a Python function that returns the first repeated item in a list, or None. Include an example and its result. Code is reviewed as text and never executed here.",
    "syllogism": "All silver keys open the east door. No east-door key opens the vault. Can a silver key open the vault? Explain the inference.",
    "game_report": "Try a SpeakToMe game, then write an essay-style experience report about what happened, what worked, and what was confusing. If the interface was not navigable, describe that attempt instead.",
    "diff_explanation": "Explain this diff in plain language and identify a possible risk: - if user: allow() / + if user and verified: allow().",
    "llama_conversation": "Talk with the hosted Llama agent for several turns if reachable. Report the room, turn count, and what you learned; if blocked, describe the failed navigation.",
}
POLICY = ("Reject hate speech and overtly sexual content. Ordinary swear words alone are allowed. "
          "Assess context, targeting, and quoted material. Never publish a pending guest post automatically.")
MAX_PENDING = 10
MAX_DAILY = 20


class GuestReviewError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def _connect(path):
    db = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS guest_actor_kind (
            user_id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('human','agent')));
        CREATE TABLE IF NOT EXISTS guest_test_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, task TEXT NOT NULL,
            response TEXT NOT NULL, created REAL NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            reviewer_id INTEGER, reason TEXT NOT NULL DEFAULT '');
        CREATE INDEX IF NOT EXISTS guest_test_by_writer ON guest_test_evidence(user_id, id);
        CREATE TABLE IF NOT EXISTS guest_qualification (
            user_id INTEGER PRIMARY KEY, qualified_at REAL NOT NULL, reviewer_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS guest_post_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            target TEXT NOT NULL CHECK(target IN ('chat','comment')), room TEXT, kind TEXT, ref TEXT,
            reply_to INTEGER, body TEXT NOT NULL, created REAL NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending', reviewer_id INTEGER, reason TEXT NOT NULL DEFAULT '',
            published_id INTEGER);
        CREATE INDEX IF NOT EXISTS guest_post_by_writer ON guest_post_queue(user_id, id);
    """)
    return db


def _body(handler):
    if (handler.headers.get("Content-Type") or "").split(";", 1)[0].lower() != "application/json":
        raise GuestReviewError("JSON only", 415)
    body = handler._body_json(8192)
    if body is None:
        return None
    if not isinstance(body, dict):
        raise GuestReviewError("expected an object")
    return body


def _writer(handler):
    principal = handler._principal()
    if principal.user_id is None:
        raise GuestReviewError("start a guest preview first", 401)
    if principal.role not in ("trial", "intern"):
        raise GuestReviewError("this route is for trial and intern contributors", 403)
    return principal


def _moderator(handler):
    principal = handler._principal()
    if principal.role not in ("admin", "operator"):
        raise GuestReviewError("an operator or admin reviews guest contributions", 403)
    return principal


def _evidence(db, user_id, body):
    task, response, kind = body.get("task"), body.get("response"), body.get("actor_kind")
    if task not in TASKS or not isinstance(response, str) or not 20 <= len(response.strip()) <= 5000:
        raise GuestReviewError("choose a task and give 20 to 5000 characters of evidence")
    if kind not in ("agent", "human"):
        raise GuestReviewError("actor_kind must be agent or human")
    existing = db.execute("SELECT kind FROM guest_actor_kind WHERE user_id = ?", (user_id,)).fetchone()
    if existing and existing[0] != kind:
        raise GuestReviewError("actor_kind is fixed for this preview", 409)
    if kind == "human" and task == "policy_awareness":
        raise GuestReviewError("the policy awareness oath is for agents", 400)
    attempts = db.execute("SELECT COUNT(*) FROM guest_test_evidence WHERE user_id = ? AND task = ?", (user_id, task)).fetchone()[0]
    if attempts >= 3:
        raise GuestReviewError("three attempts per task", 429)
    db.execute("INSERT OR IGNORE INTO guest_actor_kind VALUES (?, ?)", (user_id, kind))
    cursor = db.execute("INSERT INTO guest_test_evidence(user_id,task,response,created) VALUES (?,?,?,?)",
                        (user_id, task, response.strip(), time.time()))
    return {"id": cursor.lastrowid, "state": "pending", "task": task}


def _qualified(db, principal):
    return principal.role == "intern" or db.execute(
        "SELECT 1 FROM guest_qualification WHERE user_id = ?", (principal.user_id,)).fetchone() is not None


def _validate_target(db, principal, body, kinds, caps):
    target, text = body.get("target"), body.get("body")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
        raise GuestReviewError("body must be 1 to 4000 characters")
    if target == "chat":
        room = body.get("room")
        if not isinstance(room, str) or not chat_rooms.ROOM.fullmatch(room):
            raise GuestReviewError("invalid room")
        chat_rooms.ensure(db)
        if not chat_rooms.may_write(db, room, user_id=principal.user_id, caps=caps):
            raise GuestReviewError("this room is not open for your posts", 403)
        return target, room, None, None, None, text.strip()
    if target == "comment":
        kind, ref = comments._post(body.get("kind"), body.get("ref"))
        if kind not in kinds:
            raise GuestReviewError("no such comment thread", 404)
        who = {"user_id": principal.user_id, "username": principal.username, "caps": caps}
        allowed = kinds[kind].may_write(db, who, ref)
        if allowed is not True:
            raise GuestReviewError(allowed if isinstance(allowed, str) else "thread is closed", 403)
        reply_to = body.get("reply_to")
        comments.ensure(db)
        if reply_to is not None and (type(reply_to) is not int or db.execute(
                "SELECT 1 FROM cm_comments WHERE id = ? AND kind = ? AND ref = ?",
                (reply_to, kind, ref)).fetchone() is None):
            raise GuestReviewError("reply_to must name a comment in that thread")
        return target, None, kind, ref, reply_to, text.strip()
    raise GuestReviewError("target must be chat or comment")


def _submit(db, principal, body, kinds, caps):
    if not _qualified(db, principal):
        raise GuestReviewError("complete several tests and await qualification review", 403)
    now = time.time()
    pending = db.execute("SELECT COUNT(*) FROM guest_post_queue WHERE user_id = ? AND state = 'pending'", (principal.user_id,)).fetchone()[0]
    daily = db.execute("SELECT COUNT(*) FROM guest_post_queue WHERE user_id = ? AND created > ?", (principal.user_id, now - 86400)).fetchone()[0]
    if pending >= MAX_PENDING or daily >= MAX_DAILY:
        raise GuestReviewError("guest contribution limit reached", 429)
    target, room, kind, ref, reply_to, text = _validate_target(db, principal, body, kinds, caps)
    cursor = db.execute("INSERT INTO guest_post_queue(user_id,target,room,kind,ref,reply_to,body,created) VALUES (?,?,?,?,?,?,?,?)",
                        (principal.user_id, target, room, kind, ref, reply_to, text, now))
    return {"id": cursor.lastrowid, "state": "pending", "public": False}


def _own_status(db, principal):
    tests = [{"id": r[0], "task": r[1], "state": r[2], "reason": r[3], "untrusted": {"response": r[4]}}
             for r in db.execute("SELECT id,task,state,reason,response FROM guest_test_evidence WHERE user_id = ? ORDER BY id DESC LIMIT 30", (principal.user_id,))]
    posts = [{"id": r[0], "target": r[1], "room": r[2], "kind": r[3], "ref": r[4],
              "state": r[5], "reason": r[6], "published_id": r[7], "untrusted": {"body": r[8]}}
             for r in db.execute("SELECT id,target,room,kind,ref,state,reason,published_id,body FROM guest_post_queue WHERE user_id = ? ORDER BY id DESC LIMIT 30", (principal.user_id,))]
    return {"qualified": _qualified(db, principal), "tests": tests, "posts": posts}


def _review_queue(db):
    tests = [{"id": r[0], "user_id": r[1], "username": r[2], "task": r[3], "actor_kind": r[4],
              "untrusted": {"response": r[5]}}
             for r in db.execute("SELECT e.id,e.user_id,u.username,e.task,k.kind,e.response FROM guest_test_evidence e "
                                 "JOIN users u ON u.id=e.user_id JOIN guest_actor_kind k ON k.user_id=e.user_id "
                                 "WHERE e.state='pending' ORDER BY e.id LIMIT 100")]
    posts = [{"id": r[0], "user_id": r[1], "username": r[2], "target": r[3], "room": r[4],
              "kind": r[5], "ref": r[6], "reply_to": r[7], "untrusted": {"body": r[8]}}
             for r in db.execute("SELECT p.id,p.user_id,u.username,p.target,p.room,p.kind,p.ref,p.reply_to,p.body "
                                 "FROM guest_post_queue p JOIN users u ON u.id=p.user_id "
                                 "WHERE p.state='pending' ORDER BY p.id LIMIT 100")]
    return {"policy": POLICY, "tests": tests, "posts": posts}


def _decide_test(db, moderator, body):
    state = body.get("decision")
    if state not in ("accepted", "rejected") or type(body.get("id")) is not int:
        raise GuestReviewError("decision must be accepted or rejected, with a test id")
    reason = body.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 500:
        raise GuestReviewError("reason is too long")
    row = db.execute("UPDATE guest_test_evidence SET state=?,reviewer_id=?,reason=? WHERE id=? AND state='pending' RETURNING id,user_id,task",
                     (state, moderator.user_id, reason, body["id"])).fetchone()
    if row is None:
        raise GuestReviewError("no pending test with that id", 404)
    return {"id": row[0], "user_id": row[1], "task": row[2], "state": state}


def _qualify(db, moderator, body):
    user_id = body.get("user_id")
    if type(user_id) is not int:
        raise GuestReviewError("user_id required")
    row = db.execute("SELECT kind FROM guest_actor_kind WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        raise GuestReviewError("no guest evidence for this user", 404)
    accepted = {r[0] for r in db.execute("SELECT task FROM guest_test_evidence WHERE user_id=? AND state='accepted'", (user_id,))}
    if len(accepted) < 3 or (row[0] == "agent" and "policy_awareness" not in accepted):
        raise GuestReviewError("accept at least three distinct tests, including policy awareness for an agent", 409)
    db.execute("INSERT OR REPLACE INTO guest_qualification VALUES (?,?,?)", (user_id, time.time(), moderator.user_id))
    return {"user_id": user_id, "qualified": True}


def _decide_post(db, moderator, body, kinds, caps_for):
    state = body.get("decision")
    if state not in ("approved", "rejected") or type(body.get("id")) is not int:
        raise GuestReviewError("decision must be approved or rejected, with a post id")
    if state == "approved" and body.get("policy_confirmed") is not True:
        raise GuestReviewError("confirm the content policy before publication")
    reason = body.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 500:
        raise GuestReviewError("reason is too long")
    row = db.execute("SELECT p.user_id,u.username,p.target,p.room,p.kind,p.ref,p.reply_to,p.body "
                     "FROM guest_post_queue p JOIN users u ON u.id=p.user_id WHERE p.id=? AND p.state='pending'",
                     (body["id"],)).fetchone()
    if row is None:
        raise GuestReviewError("no pending post with that id", 404)
    user_id, username, target, room, kind, ref, reply_to, text = row
    published_id = None
    if state == "approved":
        account = {"id": user_id, "username": username}
        temporary = db.execute("SELECT expires_at FROM temporary_accounts WHERE user_id=?", (user_id,)).fetchone()
        if temporary is not None:
            if temporary[0] <= time.time():
                raise GuestReviewError("the preview has expired; reject this post", 409)
            account["temporary"] = True
        caps = frozenset(caps_for(account))
        if target == "chat":
            published_id = chat_rooms.post(db, room, user_id=user_id, username=username, body=text, caps=caps)["id"]
        else:
            who = {"user_id": user_id, "username": username, "caps": caps}
            if kind not in kinds or kinds[kind].may_write(db, who, ref) is not True:
                raise GuestReviewError("the comment target is no longer open", 409)
            published_id = comments.post(db, kind, ref, user_id=user_id, username=username, body=text,
                                         reply_to=reply_to)["id"]
    db.execute("UPDATE guest_post_queue SET state=?,reviewer_id=?,reason=?,published_id=? WHERE id=? AND state='pending'",
               (state, moderator.user_id, reason, published_id, body["id"]))
    return {"id": body["id"], "state": state, "published_id": published_id}


def http_route(handler, path, method, *, accounts_db, kinds, caps_for):
    db = _connect(accounts_db)
    try:
        if method == "GET" and path == "/api/guest/tasks":
            return handler._json({"tasks": TASKS, "qualification": "A reviewer accepts at least three distinct tasks; agents include policy awareness.",
                                  "posting": "Qualified preview guests and interns submit text for private review before publication."})
        if method == "GET" and path == "/api/guest/status":
            return handler._json(_own_status(db, _writer(handler)))
        if method == "GET" and path == "/api/guest/review":
            _moderator(handler)
            return handler._json(_review_queue(db))
        if method != "POST":
            raise GuestReviewError("not found", 404)
        body = _body(handler)
        if body is None:
            return
        if path == "/api/guest/tests":
            principal = _writer(handler)
            if principal.role != "trial":
                raise GuestReviewError("an intern is already qualified", 409)
            return handler._json({"evidence": _evidence(db, principal.user_id, body)}, 202)
        if path == "/api/guest/submissions":
            principal = _writer(handler)
            return handler._json({"submission": _submit(db, principal, body, kinds, frozenset(handler._caps()))}, 202)
        # Existing intern chat/comment tools keep their familiar entry points,
        # but their text takes the same pending path instead of publishing.
        if path == "/api/comments" or path.startswith("/api/chat/rooms/"):
            principal = _writer(handler)
            if principal.role != "intern":
                raise GuestReviewError("use guest_post_submit", 403)
            if path == "/api/comments":
                queued = {"target": "comment", "kind": body.get("kind"), "ref": body.get("ref"),
                          "reply_to": body.get("reply_to"), "body": body.get("body")}
            else:
                parts = path[len("/api/chat/rooms/"):].split("/")
                if len(parts) != 1 or body.get("sticker") is not None:
                    raise GuestReviewError("only text room posts enter public review", 400)
                queued = {"target": "chat", "room": parts[0], "body": body.get("body")}
            return handler._json({"submission": _submit(db, principal, queued, kinds, frozenset(handler._caps()))}, 202)
        moderator = _moderator(handler)
        db.execute("BEGIN IMMEDIATE")
        try:
            if path == "/api/guest/review/test":
                answer = _decide_test(db, moderator, body)
            elif path == "/api/guest/review/qualify":
                answer = _qualify(db, moderator, body)
            elif path == "/api/guest/review/post":
                answer = _decide_post(db, moderator, body, kinds, caps_for)
            else:
                raise GuestReviewError("not found", 404)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        return handler._json(answer)
    except (GuestReviewError, comments.CommentError, chat_rooms.ChatError) as error:
        return handler._json({"error": str(error)}, getattr(error, "status", 400))
    finally:
        db.close()
