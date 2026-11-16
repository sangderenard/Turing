"""Research cases and development briefs for signed-in accounts.

Research copies are intentionally readable to case participants. Private file bytes remain
in user_files; only text explicitly decrypted and submitted by its owner enters a case.
Chat and threaded comments stay in their existing owners.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from meowdoku import authored_pages, schema_version

SCHEMA_VERSION = 1
CASE_ID = re.compile(r"[0-9a-f]{32}\Z")
AXES = ("clarity", "confidence", "momentum", "trust", "inclusion", "coordination")
PROGRESS_AXES = AXES[:3]
SOCIAL_AXES = AXES[3:]
MAX_BODY = 60_000
MAX_DOCS = 20
VOTE_SECONDS = 24 * 3600
DDL = (
    """CREATE TABLE IF NOT EXISTS research_cases (
        id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
        title TEXT NOT NULL, goal TEXT NOT NULL, ticket_ids TEXT NOT NULL,
        phase TEXT NOT NULL CHECK(phase IN ('research','development')),
        invariants TEXT NOT NULL, active_user_id INTEGER REFERENCES users(id),
        revision INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS research_docs (
        case_id TEXT NOT NULL REFERENCES research_cases(id), user_id INTEGER NOT NULL REFERENCES users(id),
        title TEXT NOT NULL, body TEXT NOT NULL, source_kind TEXT NOT NULL,
        source_ref TEXT, updated_at REAL NOT NULL, PRIMARY KEY(case_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS research_brief (
        case_id TEXT NOT NULL REFERENCES research_cases(id), user_id INTEGER NOT NULL REFERENCES users(id),
        position INTEGER NOT NULL, PRIMARY KEY(case_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS research_affirmations (
        case_id TEXT NOT NULL REFERENCES research_cases(id), user_id INTEGER NOT NULL REFERENCES users(id),
        phrase TEXT NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY(case_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS research_surveys (
        case_id TEXT NOT NULL REFERENCES research_cases(id), user_id INTEGER NOT NULL REFERENCES users(id),
        axis TEXT NOT NULL, value INTEGER NOT NULL CHECK(value BETWEEN 1 AND 5), updated_at REAL NOT NULL,
        PRIMARY KEY(case_id,user_id,axis))""",
    """CREATE TABLE IF NOT EXISTS research_proposals (
        id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES research_cases(id),
        target_user_id INTEGER NOT NULL REFERENCES users(id), proposer_id INTEGER NOT NULL REFERENCES users(id),
        base_revision INTEGER NOT NULL, state TEXT NOT NULL CHECK(state IN ('open','passed','failed')),
        deadline REAL NOT NULL, created_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS research_eligible (
        proposal_id TEXT NOT NULL REFERENCES research_proposals(id), user_id INTEGER NOT NULL REFERENCES users(id),
        PRIMARY KEY(proposal_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS research_ballots (
        proposal_id TEXT NOT NULL REFERENCES research_proposals(id), user_id INTEGER NOT NULL REFERENCES users(id),
        choice TEXT NOT NULL CHECK(choice IN ('new','current')), at REAL NOT NULL,
        PRIMARY KEY(proposal_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS research_events (
        id INTEGER PRIMARY KEY, case_id TEXT NOT NULL REFERENCES research_cases(id),
        actor_id INTEGER REFERENCES users(id), kind TEXT NOT NULL,
        before_json TEXT, after_json TEXT, at REAL NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS research_events_case ON research_events(case_id,id)",
)

class ResearchError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

def bounded(value, name, maximum, *, minimum=1):
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum or "\x00" in value:
        raise ResearchError(f"{name} must be {minimum} to {maximum} characters")
    return value.strip()

def exact(body, fields):
    if not isinstance(body, dict) or set(body) != set(fields):
        raise ResearchError("unexpected research fields")

class Service:
    def __init__(self, accounts):
        self.accounts = accounts
        with accounts.lock:
            rows = accounts._conn.execute("PRAGMA database_list").fetchall()
        filename = next((row[2] for row in rows if row[1] == "main"), None)
        if not filename:
            raise ResearchError("file-backed Accounts database required", 503)
        self.path = Path(filename).resolve(strict=True)

    @contextmanager
    def transaction(self, token):
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise ResearchError("sign in to use research", 401)
        account = self.accounts.user_for_token(token)
        if account is None:
            raise ResearchError("sign in to use research", 401)
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            schema_version.declare(db, "ticket_research", SCHEMA_VERSION)
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
                schema_version.declare(db, "ticket_research", SCHEMA_VERSION)
            for statement in DDL:
                db.execute(statement)
            current = db.execute("SELECT user_id FROM sessions WHERE token=? AND expires_at>=?",
                                 (token, time.time())).fetchone()
            if current is None or current["user_id"] != account["id"]:
                raise ResearchError("sign in to use research", 401)
            admin = db.execute("SELECT 1 FROM account_roles WHERE user_id=? AND role='admin'",
                               (account["id"],)).fetchone() is not None
            yield db, account["id"], admin
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, case_id, actor, kind, before, after, now=None):
        db.execute("INSERT INTO research_events(case_id,actor_id,kind,before_json,after_json,at) VALUES(?,?,?,?,?,?)",
                   (case_id, actor, kind, json.dumps(before) if before is not None else None,
                    json.dumps(after) if after is not None else None, time.time() if now is None else now))

    @staticmethod
    def case(db, case_id):
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
            raise ResearchError("research case not found", 404)
        row = db.execute("SELECT * FROM research_cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise ResearchError("research case not found", 404)
        return row

    @staticmethod
    def entered(db, case_id, actor):
        if db.execute("SELECT 1 FROM research_affirmations WHERE case_id=? AND user_id=?",
                      (case_id, actor)).fetchone() is None:
            raise ResearchError("enter an affirmation phrase to join this case", 403)

    @staticmethod
    def authors_brief(db, case_id, actor, owner, admin):
        return admin or actor == owner or db.execute(
            "SELECT 1 FROM research_brief WHERE case_id=? AND user_id=?", (case_id, actor)).fetchone() is not None

    def list_cases(self, token):
        with self.transaction(token) as (db, actor, admin):
            rows = db.execute(
                "SELECT c.id,c.title,c.goal,c.ticket_ids,c.phase,c.revision,c.updated_at,u.username AS author "
                "FROM research_cases c JOIN users u ON u.id=c.owner_id ORDER BY c.updated_at DESC LIMIT 100").fetchall()
            return {"cases": [{**dict(row), "ticket_ids": json.loads(row["ticket_ids"])} for row in rows]}

    def create(self, token, body):
        exact(body, ("title", "goal", "ticket_ids"))
        title = bounded(body["title"], "title", 120)
        goal = bounded(body["goal"], "goal", 4000)
        ids = body["ticket_ids"]
        if not isinstance(ids, list) or len(ids) > 10 or len(set(map(str, ids))) != len(ids) or any(
                not isinstance(value, str) or not CASE_ID.fullmatch(value) for value in ids):
            raise ResearchError("ticket_ids must contain at most ten distinct issue ticket IDs")
        with self.transaction(token) as (db, actor, admin):
            for ticket_id in ids:
                row = db.execute("SELECT owner_id FROM issue_tickets WHERE id=?", (ticket_id,)).fetchone()
                assigned = db.execute("SELECT 1 FROM issue_ticket_assignees WHERE ticket_id=? AND user_id=?",
                                      (ticket_id, actor)).fetchone() if row else None
                if row is None or not (admin or row["owner_id"] == actor or assigned):
                    raise ResearchError("linked ticket is unavailable", 404)
            case_id, now = uuid.uuid4().hex, time.time()
            db.execute("INSERT INTO research_cases VALUES(?,?,?,?,?,'research','[]',NULL,1,?,?)",
                       (case_id, actor, title, goal, json.dumps(ids), now, now))
            self.event(db, case_id, actor, "case_created", None, {"title": title, "goal": goal, "ticket_ids": ids}, now)
            return {"id": case_id, "phase": "research"}

    def affirmation(self, token, case_id, body):
        exact(body, ("phrase",))
        phrase = bounded(body["phrase"], "affirmation", 160)
        with self.transaction(token) as (db, actor, admin):
            self.case(db, case_id)
            previous = db.execute("SELECT phrase FROM research_affirmations WHERE case_id=? AND user_id=?",
                                  (case_id, actor)).fetchone()
            old = previous["phrase"] if previous else None
            if old != phrase:
                now = time.time()
                db.execute("INSERT INTO research_affirmations VALUES(?,?,?,?) ON CONFLICT(case_id,user_id) "
                           "DO UPDATE SET phrase=excluded.phrase,updated_at=excluded.updated_at",
                           (case_id, actor, phrase, now))
                self.event(db, case_id, actor, "affirmation", old, phrase, now)
            return {"phrase": phrase}

    @staticmethod
    def _settle(db, row):
        if row is None or row["state"] != "open":
            return
        total = db.execute("SELECT COUNT(*) FROM research_eligible WHERE proposal_id=?", (row["id"],)).fetchone()[0]
        ballots = db.execute("SELECT choice,COUNT(*) AS n FROM research_ballots WHERE proposal_id=? GROUP BY choice",
                             (row["id"],)).fetchall()
        counts = {item["choice"]: item["n"] for item in ballots}
        if sum(counts.values()) < total and time.time() < row["deadline"]:
            return
        passed = counts.get("new", 0) >= counts.get("current", 0)
        case = db.execute("SELECT active_user_id,revision FROM research_cases WHERE id=?", (row["case_id"],)).fetchone()
        if db.execute(
                "SELECT 1 FROM research_docs WHERE case_id=? AND user_id=?",
                (row["case_id"], row["target_user_id"])).fetchone() is None:
            passed = False
        state = "passed" if passed else "failed"
        db.execute("UPDATE research_proposals SET state=? WHERE id=?", (state, row["id"]))
        if passed:
            db.execute("UPDATE research_cases SET active_user_id=?,revision=revision+1,updated_at=? WHERE id=?",
                       (row["target_user_id"], time.time(), row["case_id"]))
        Service.event(db, row["case_id"], None, "vote_closed",
                      {"active_user_id": case["active_user_id"]},
                      {"active_user_id": row["target_user_id"] if passed else case["active_user_id"],
                       "state": state, "new": counts.get("new", 0), "current": counts.get("current", 0)})

    def snapshot(self, token, case_id):
        with self.transaction(token) as (db, actor, admin):
            case = self.case(db, case_id)
            pending = db.execute("SELECT * FROM research_proposals WHERE case_id=? AND state='open' "
                                 "ORDER BY created_at DESC LIMIT 1", (case_id,)).fetchone()
            self._settle(db, pending)
            case = self.case(db, case_id)
            entry = db.execute("SELECT phrase FROM research_affirmations WHERE case_id=? AND user_id=?",
                               (case_id, actor)).fetchone()
            basic = {**dict(case), "ticket_ids": json.loads(case["ticket_ids"]),
                     "invariants": json.loads(case["invariants"])}
            if entry is None:
                return {"case": basic, "requires_affirmation": True}
            users = db.execute("SELECT u.id,u.username,a.phrase,d.title AS document_title,d.body AS document_body,"
                               "d.source_kind,d.updated_at AS document_updated "
                               "FROM users u LEFT JOIN account_roles r ON r.user_id=u.id "
                               "LEFT JOIN research_affirmations a ON a.case_id=? AND a.user_id=u.id "
                               "LEFT JOIN research_docs d ON d.case_id=? AND d.user_id=u.id "
                               "WHERE COALESCE(r.role,'member')!='service' ORDER BY u.username COLLATE NOCASE",
                               (case_id, case_id)).fetchall()
            brief = [item[0] for item in db.execute(
                "SELECT user_id FROM research_brief WHERE case_id=? ORDER BY position", (case_id,))]
            pending = db.execute("SELECT * FROM research_proposals WHERE case_id=? AND state='open' "
                                 "ORDER BY created_at DESC LIMIT 1", (case_id,)).fetchone()
            proposal = None
            if pending:
                tallies = {item["choice"]: item["n"] for item in db.execute(
                    "SELECT choice,COUNT(*) AS n FROM research_ballots WHERE proposal_id=? GROUP BY choice",
                    (pending["id"],))}
                mine = db.execute("SELECT choice FROM research_ballots WHERE proposal_id=? AND user_id=?",
                                  (pending["id"], actor)).fetchone()
                eligible = db.execute("SELECT 1 FROM research_eligible WHERE proposal_id=? AND user_id=?",
                                      (pending["id"], actor)).fetchone() is not None
                proposal = {"id": pending["id"], "target_user_id": pending["target_user_id"],
                            "deadline": pending["deadline"], "new": tallies.get("new", 0),
                            "current": tallies.get("current", 0), "my_vote": mine["choice"] if mine else None,
                            "eligible": eligible}
            surveys = {}
            for axis in AXES:
                values = db.execute("SELECT value,COUNT(*) AS n FROM research_surveys "
                                    "WHERE case_id=? AND axis=? GROUP BY value ORDER BY value",
                                    (case_id, axis)).fetchall()
                mine = db.execute("SELECT value FROM research_surveys WHERE case_id=? AND user_id=? AND axis=?",
                                  (case_id, actor, axis)).fetchone()
                surveys[axis] = {"counts": {str(row["value"]): row["n"] for row in values},
                                 "mine": mine["value"] if mine else None}
            events = [dict(row) for row in db.execute(
                "SELECT e.id,e.kind,e.before_json,e.after_json,e.at,u.username AS actor FROM research_events e "
                "LEFT JOIN users u ON u.id=e.actor_id WHERE e.case_id=? ORDER BY e.id DESC LIMIT 80", (case_id,))]
            return {"case": basic, "requires_affirmation": False, "my_id": actor,
                    "participants": [dict(row) for row in users], "brief_user_ids": brief,
                    "can_edit_brief": bool(self.authors_brief(db,case_id,actor,case["owner_id"],admin)),
                    "proposal": proposal, "surveys": surveys, "events": events}

    def document(self, token, case_id, body):
        exact(body, ("title", "body", "source_kind", "source_ref"))
        title = bounded(body["title"], "document title", 120)
        kind, ref = body["source_kind"], body["source_ref"]
        if kind not in ("text", "file_copy", "page") or (kind == "text" and ref is not None):
            raise ResearchError("invalid document source")
        with self.transaction(token) as (db, actor, admin):
            case = self.case(db, case_id)
            self.entered(db, case_id, actor)
            content = body["body"]
            if kind == "page":
                if not isinstance(ref, str) or not CASE_ID.fullmatch(ref):
                    raise ResearchError("invalid page reference")
                owner = db.execute("SELECT owner_id FROM authored_pages WHERE id=?", (ref,)).fetchone()
                if owner is None or owner["owner_id"] != actor:
                    raise ResearchError("select a page from your own inventory", 404)
                try:
                    data, _ = authored_pages.read(db, ref, site="dev", actor=actor)
                    content = data.decode("utf-8", "strict")
                except (authored_pages.PageError, UnicodeError, OSError):
                    raise ResearchError("page cannot be copied as text", 400) from None
            elif kind == "file_copy":
                if not isinstance(ref, str) or not CASE_ID.fullmatch(ref) or db.execute(
                        "SELECT 1 FROM user_files WHERE id=? AND owner_id=?",
                        (ref,actor)).fetchone() is None:
                    raise ResearchError("select your own encrypted file", 404)
            if not isinstance(content, str) or not 1 <= len(content.strip()) <= MAX_BODY or "\x00" in content:
                raise ResearchError("document text must be 1 to 60000 characters")
            old = db.execute("SELECT title,body,source_kind,source_ref FROM research_docs "
                             "WHERE case_id=? AND user_id=?", (case_id,actor)).fetchone()
            now = time.time()
            db.execute("INSERT INTO research_docs VALUES(?,?,?,?,?,?,?) ON CONFLICT(case_id,user_id) DO UPDATE "
                       "SET title=excluded.title,body=excluded.body,source_kind=excluded.source_kind,"
                       "source_ref=excluded.source_ref,updated_at=excluded.updated_at",
                       (case_id,actor,title,content,kind,ref,now))
            self.event(db,case_id,actor,"document",dict(old) if old else None,
                       {"title":title,"source_kind":kind,"source_ref":ref,"body":content},now)
            if case["active_user_id"] is None:
                db.execute("UPDATE research_cases SET active_user_id=?,revision=revision+1,updated_at=? WHERE id=?",
                           (actor,now,case_id))
                self.event(db,case_id,actor,"initial_tab",None,{"active_user_id":actor},now)
            return {"ok":True}

    def invariants(self, token, case_id, body):
        exact(body, ("invariants", "revision"))
        items = body["invariants"]
        if not isinstance(items,list) or len(items)>30 or any(
                not isinstance(item,str) or not 1<=len(item.strip())<=400 for item in items):
            raise ResearchError("use at most 30 invariants of 1 to 400 characters")
        items = [item.strip() for item in items]
        if len(set(items))!=len(items) or type(body["revision"]) is not int:
            raise ResearchError("invalid invariant update")
        with self.transaction(token) as (db,actor,admin):
            case=self.case(db,case_id)
            self.entered(db,case_id,actor)
            if case["phase"]=="development" and not self.authors_brief(db,case_id,actor,case["owner_id"],admin):
                raise ResearchError("brief author or admin required",403)
            if body["revision"]!=case["revision"]:
                raise ResearchError("case changed; refresh first",409)
            old=json.loads(case["invariants"])
            if old==items:
                return {"revision":case["revision"]}
            now=time.time()
            db.execute("UPDATE research_cases SET invariants=?,revision=revision+1,updated_at=? WHERE id=?",
                       (json.dumps(items),now,case_id))
            self.event(db,case_id,actor,"invariants",old,items,now)
            return {"revision":case["revision"]+1}

    def brief(self, token, case_id, body):
        exact(body, ("user_ids", "revision"))
        ids=body["user_ids"]
        if not isinstance(ids,list) or len(ids)>MAX_DOCS or len(set(map(str,ids)))!=len(ids) or any(
                type(value) is not int or value<1 for value in ids) or type(body["revision"]) is not int:
            raise ResearchError("invalid brief tabs")
        with self.transaction(token) as (db,actor,admin):
            case=self.case(db,case_id)
            self.entered(db,case_id,actor)
            if case["phase"]=="development" and not self.authors_brief(db,case_id,actor,case["owner_id"],admin):
                raise ResearchError("brief author or admin required",403)
            if body["revision"]!=case["revision"]:
                raise ResearchError("case changed; refresh first",409)
            available={row[0] for row in db.execute("SELECT user_id FROM research_docs WHERE case_id=?",(case_id,))}
            if not set(ids)<=available:
                raise ResearchError("brief tabs must name contributed documents")
            old=[row[0] for row in db.execute(
                "SELECT user_id FROM research_brief WHERE case_id=? ORDER BY position",(case_id,))]
            if old==ids:
                return {"revision":case["revision"]}
            now=time.time()
            db.execute("DELETE FROM research_brief WHERE case_id=?",(case_id,))
            db.executemany("INSERT INTO research_brief VALUES(?,?,?)",
                           [(case_id,user_id,pos) for pos,user_id in enumerate(ids)])
            db.execute("UPDATE research_cases SET revision=revision+1,updated_at=? WHERE id=?",(now,case_id))
            self.event(db,case_id,actor,"brief_tabs",old,ids,now)
            return {"revision":case["revision"]+1}

    def promote(self, token, case_id, body):
        exact(body, ("revision",))
        with self.transaction(token) as (db,actor,admin):
            case=self.case(db,case_id)
            self.entered(db,case_id,actor)
            if case["owner_id"]!=actor and not admin:
                raise ResearchError("research author or admin required",403)
            if case["phase"]!="research" or case["revision"]!=body["revision"]:
                raise ResearchError("case changed; refresh first",409)
            if not json.loads(case["invariants"]) or db.execute(
                    "SELECT 1 FROM research_brief WHERE case_id=?",(case_id,)).fetchone() is None:
                raise ResearchError("add invariants and design document tabs before development")
            now=time.time()
            db.execute("UPDATE research_cases SET phase='development',revision=revision+1,updated_at=? WHERE id=?",
                       (now,case_id))
            self.event(db,case_id,actor,"promote",{"phase":"research"},{"phase":"development"},now)
            return {"phase":"development","revision":case["revision"]+1}

    def propose(self, token, case_id, body):
        exact(body, ("target_user_id", "revision"))
        target,version=body["target_user_id"],body["revision"]
        if type(target) is not int or type(version) is not int:
            raise ResearchError("invalid tab proposal")
        with self.transaction(token) as (db,actor,admin):
            case=self.case(db,case_id)
            self.entered(db,case_id,actor)
            if case["revision"]!=version:
                raise ResearchError("case changed; refresh first",409)
            if case["active_user_id"]==target or db.execute(
                    "SELECT 1 FROM research_docs WHERE case_id=? AND user_id=?",
                    (case_id,target)).fetchone() is None:
                raise ResearchError("select another contributed document")
            pending=db.execute("SELECT * FROM research_proposals WHERE case_id=? AND state='open'",
                               (case_id,)).fetchone()
            self._settle(db,pending)
            if pending and db.execute("SELECT state FROM research_proposals WHERE id=?",
                                      (pending["id"],)).fetchone()[0]=="open":
                raise ResearchError("a tab vote is already open",409)
            now=time.time()
            proposal_id=uuid.uuid4().hex
            db.execute("INSERT INTO research_proposals VALUES(?,?,?,?,?,'open',?,?)",
                       (proposal_id,case_id,target,actor,version,now+VOTE_SECONDS,now))
            voters=[row[0] for row in db.execute(
                "SELECT a.user_id FROM research_affirmations a JOIN users u ON u.id=a.user_id "
                "LEFT JOIN account_roles r ON r.user_id=u.id WHERE a.case_id=? "
                "AND COALESCE(r.role,'member')!='service'",(case_id,))]
            db.executemany("INSERT INTO research_eligible VALUES(?,?)",
                           [(proposal_id,user_id) for user_id in voters])
            db.execute("INSERT INTO research_ballots VALUES(?,?, 'new', ?)",(proposal_id,actor,now))
            self.event(db,case_id,actor,"vote_open",None,{"target_user_id":target,"deadline":now+VOTE_SECONDS},now)
            self.event(db,case_id,actor,"vote_cast",None,{"choice":"new","proposal_id":proposal_id},now)
            self._settle(db,db.execute("SELECT * FROM research_proposals WHERE id=?",(proposal_id,)).fetchone())
            return {"id":proposal_id,"deadline":now+VOTE_SECONDS}

    def vote(self, token, case_id, body):
        exact(body, ("choice",))
        choice=body["choice"]
        if choice not in ("new","current"):
            raise ResearchError("vote new or current")
        with self.transaction(token) as (db,actor,admin):
            self.case(db,case_id)
            self.entered(db,case_id,actor)
            row=db.execute("SELECT * FROM research_proposals WHERE case_id=? AND state='open' "
                           "ORDER BY created_at DESC LIMIT 1",(case_id,)).fetchone()
            self._settle(db,row)
            if row is None or db.execute("SELECT state FROM research_proposals WHERE id=?",
                                         (row["id"],)).fetchone()[0]!="open":
                raise ResearchError("no open tab vote",409)
            if db.execute("SELECT 1 FROM research_eligible WHERE proposal_id=? AND user_id=?",
                          (row["id"],actor)).fetchone() is None:
                raise ResearchError("this vote began before you joined",403)
            old=db.execute("SELECT choice FROM research_ballots WHERE proposal_id=? AND user_id=?",
                           (row["id"],actor)).fetchone()
            now=time.time()
            db.execute("INSERT INTO research_ballots VALUES(?,?,?,?) ON CONFLICT(proposal_id,user_id) "
                       "DO UPDATE SET choice=excluded.choice,at=excluded.at",(row["id"],actor,choice,now))
            if old is None or old["choice"]!=choice:
                self.event(db,case_id,actor,"vote_cast",old["choice"] if old else None,
                           {"choice":choice,"proposal_id":row["id"]},now)
            self._settle(db,row)
            return {"ok":True}

    def survey(self, token, case_id, body):
        exact(body, ("axis", "value"))
        axis,value=body["axis"],body["value"]
        if axis not in AXES or type(value) is not int or not 1<=value<=5:
            raise ResearchError("choose a survey axis and a value from 1 to 5")
        with self.transaction(token) as (db,actor,admin):
            self.case(db,case_id)
            self.entered(db,case_id,actor)
            old=db.execute("SELECT value FROM research_surveys WHERE case_id=? AND user_id=? AND axis=?",
                           (case_id,actor,axis)).fetchone()
            if old is None or old["value"]!=value:
                now=time.time()
                db.execute("INSERT INTO research_surveys VALUES(?,?,?,?,?) ON CONFLICT(case_id,user_id,axis) "
                           "DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                           (case_id,actor,axis,value,now))
                self.event(db,case_id,actor,"survey:"+axis,old["value"] if old else None,value,now)
            return {"axis":axis,"value":value}

def comment_kind():
    from meowdoku import comments
    def row(db, ref):
        if not isinstance(ref,str) or not CASE_ID.fullmatch(ref):
            raise comments.CommentError("research case not found",404)
        found=db.execute("SELECT id FROM research_cases WHERE id=?",(ref,)).fetchone()
        if found is None:
            raise comments.CommentError("research case not found",404)
    def may(db, who, ref):
        row(db,ref)
        return db.execute("SELECT 1 FROM research_affirmations WHERE case_id=? AND user_id=?",
                          (ref,who["user_id"])).fetchone() is not None
    def tell(db, ref, comment):
        return [r[0] for r in db.execute(
            "SELECT u.username FROM research_affirmations a JOIN users u ON u.id=a.user_id "
            "WHERE a.case_id=?",(ref,))]
    def label(db, ref):
        found=db.execute("SELECT title FROM research_cases WHERE id=?",(ref,)).fetchone()
        return found[0] if found else "Research"
    return comments.Kind(may_read=may,may_write=may,tell=tell,label=label)
