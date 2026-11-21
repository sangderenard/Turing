"""[GUIDANCE] Cached, reduced working guidance for stateless agents (case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 4).

A stateless agent starts every call empty, so its whole orientation is handed to it each time. A reducer, a fresh claude or
codex instance on a non-operator account with a large context, reduces that orientation for it:
- the sources are what the agent itself would read: the shared basics and its profile's documents, its decision graph and
  its declared capacity (sources), never its memory, messages or anything private;
- the reduction is cached per profile in agent_guidance_cache, with the digest of the sources it was made from;
- it is served only while that digest equals the current one and before its drop time, so a changed document invalidates
  it at once, and each copy also expires at a random time around DROP_SECONDS, so the reductions refresh at different times;
- stale or missing, the orientation serves the full documents and queues a request (agent_guidance_requests).
The full documents stay readable by every agent, unchanged (workspace_orientation full=1). A reduction never edits them.

Who reduces is declared, never inferred from a name: an account whose own profile declares the job "reduce" (REDUCE_JOB,
an ordinary profile job on the workspace page). Its job documents are the reducer's own instructions.

    prepare(db)                                   the tables, under schema module agent_guidance 1
    sources(db, profile_id)                       {"digest", "text", "budget_bytes", ...}: what is reduced
    served(db, profile_id, now)                   the cached reduction while it is current, else None
    request(db, profile_id, now)                  queue a reduce request (idempotent)
    is_reducer(db, user_id)                       the account's profile declares the reduce job
    claim(db, user_id, now)                       a reducer takes the oldest waiting request, with its sources
    submit(db, user_id, data, now)                a reducer returns one reduction, citing the digest it read
    fail(db, user_id, data, now)                  a reducer reports that one request could not be reduced
    status(db)                                    every cached reduction and waiting request (operators, reducers)
"""
from __future__ import annotations

import hashlib
import json
import random
import time

from meowdoku import agent_workspace, schema_version

REDUCE_JOB = "reduce"
DROP_SECONDS = 6 * 3600               # a reduction lives DROP_SECONDS x uniform(DROP_SPREAD) after it was made
DROP_SPREAD = (0.5, 1.5)
LEASE_SECONDS = 15 * 60               # a claim not answered by then may be taken by another reducer
MAX_CLAIM = 1                         # one at a time: a slow reduction never lets a later claim lapse
# The reduction's size, from the agent's declared context: a quarter of it, at about four bytes a token, so the rest is
# left for memory, the conversation and the answer. Never below MIN_BUDGET nor above MAX_BUDGET bytes.
CONTEXT_SHARE = 0.25
BYTES_PER_TOKEN = 4
MIN_BUDGET, MAX_BUDGET = 1024, 32 * 1024
HEAD = ("Your working guidance follows: a reduction of your documents, made for you by a reducer. Your documents stay "
        "available in full and are the authority where this reduction is unclear: workspace_orientation full=1.")


class GuidanceError(agent_workspace.WorkspaceError):
    pass


def prepare(db):
    """The tables. Called by agent_workspace.connect, outside any transaction (declare commits)."""
    schema_version.declare(db, "agent_guidance", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_guidance_cache (profile_id TEXT PRIMARY KEY "
               "REFERENCES agent_profiles(id) ON DELETE CASCADE, source_digest TEXT NOT NULL, body TEXT NOT NULL, "
               "made_by INTEGER NOT NULL, made_at REAL NOT NULL, drop_at REAL NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS agent_guidance_requests (profile_id TEXT PRIMARY KEY "
               "REFERENCES agent_profiles(id) ON DELETE CASCADE, requested_at REAL NOT NULL, claimed_by INTEGER, "
               "claimed_at REAL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '')")


# ---------------------------------------------------------------- the sources
def budget_bytes(declared):
    tokens = declared["context_tokens"] if declared else MIN_BUDGET // BYTES_PER_TOKEN
    return max(MIN_BUDGET, min(MAX_BUDGET, int(tokens * CONTEXT_SHARE * BYTES_PER_TOKEN)))


def sources(db, profile_id):
    """What the profile's agent itself reads, as one text for the reducer, and the digest that identifies it.

    The digest is the sha256 of every source document's id, updated time and body, the decision graph's places and routes
    (each with its updated time) and the capacity row, so any change to any of them changes it.
    """
    from meowdoku import agent_decisions
    agent_workspace.profile(db, profile_id)                           # 404 for an unknown profile
    documents = agent_workspace.orientation_documents(db, profile_id)
    declared = agent_workspace.capacity(db, profile_id)
    graph = agent_decisions.graph_of(db, profile_id)
    places, routes = agent_decisions._places(db, graph), agent_decisions._routes(db, graph)
    digest = hashlib.sha256(json.dumps({
        "documents": [[d["id"], d["updated"], d["body"]] for d in documents],
        "graph": [graph, places, routes],
        "capacity": [declared[k] for k in ("state", "context_tokens", "tools", "session", "note")] if declared else None,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    parts = [agent_workspace._capacity_words(declared).capitalize() + "."]
    parts += [f"# {d['title']}\n{d['body']}" for d in documents]
    if places:
        lines = [f"# Your decision graph ({graph}): places, what may be done at each, and the routes between them"]
        for place in places:
            acts = ", ".join(f"{a['tool']} ({a['for']})" if a.get("for") else a["tool"] for a in place["acts"])
            lines.append(f"## {place['title']} [{place['key']}]{' (start)' if place['start'] else ''}\n{place['body']}" +
                         (f"\nActs: {acts}" if acts else ""))
            lines += [f"- if {r['condition']}: go to {r['target']}" for r in routes if r["source"] == place["key"]]
        parts.append("\n".join(lines))
    return {"profile_id": profile_id, "digest": digest, "text": "\n\n".join(parts), "budget_bytes": budget_bytes(declared),
            "documents": [{"id": d["id"], "title": d["title"]} for d in documents], "graph": graph}


# ---------------------------------------------------------------- serving
def served(db, profile_id, now=None):
    """The cached reduction, only while its digest is the current digest and now is before its drop time."""
    now = time.time() if now is None else now
    row = db.execute("SELECT source_digest,body,made_by,made_at,drop_at FROM agent_guidance_cache WHERE profile_id=?",
                     (profile_id,)).fetchone()
    if row is None or now >= row["drop_at"] or row["source_digest"] != sources(db, profile_id)["digest"]:
        return None
    return dict(row)


def request(db, profile_id, now=None):
    """Queue a reduce request; an existing one is kept (and keeps its claim)."""
    db.execute("INSERT OR IGNORE INTO agent_guidance_requests(profile_id,requested_at) VALUES (?,?)",
               (profile_id, time.time() if now is None else now))


def working_guidance(db, profile_id, now=None):
    """(the reduction's text with its heading, the cache row) when current; else (None, None) and a request is queued."""
    found = served(db, profile_id, now)
    if found is None:
        request(db, profile_id, now)
        return None, None
    return HEAD + "\n\n" + found["body"], found


# ---------------------------------------------------------------- the reducer
def _own_profile(db, user_id):
    return agent_workspace._profile_for(db, user_id)


def is_reducer(db, user_id):
    profile_id = _own_profile(db, user_id)
    return bool(profile_id and db.execute("SELECT 1 FROM agent_profile_jobs WHERE profile_id=? AND job_key=?",
                                          (profile_id, REDUCE_JOB)).fetchone())


def _require_reducer(db, user_id):
    if not is_reducer(db, user_id):
        raise GuidanceError(f"only an account whose profile declares the job {REDUCE_JOB!r} reduces guidance", 403)


def claim(db, user_id, now=None):
    """The oldest waiting requests (unclaimed, or claimed past LEASE_SECONDS; MAX_CLAIM of them), taken by this reducer, each
    with its sources, and
    the reducer's own instructions: its reduce job's documents."""
    _require_reducer(db, user_id)
    now = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        rows = db.execute("SELECT profile_id FROM agent_guidance_requests WHERE claimed_by IS NULL OR claimed_by=? "
                          "OR claimed_at<? ORDER BY requested_at LIMIT ?",
                          (user_id, now - LEASE_SECONDS, MAX_CLAIM)).fetchall()
        taken = []
        for (profile_id,) in rows:
            db.execute("UPDATE agent_guidance_requests SET claimed_by=?, claimed_at=?, attempts=attempts+1 WHERE profile_id=?",
                       (user_id, now, profile_id))
            found = sources(db, profile_id)
            title = db.execute("SELECT title FROM agent_profiles WHERE id=?", (profile_id,)).fetchone()[0]
            taken.append({**found, "title": title})
        db.commit()
    except BaseException:
        db.rollback(); raise
    instructions = agent_workspace.resolve_profile(db, _own_profile(db, user_id), REDUCE_JOB)["preamble"]
    return {"requests": taken, "instructions": instructions, "lease_seconds": LEASE_SECONDS}


def _claimed(db, user_id, profile_id, now):
    row = db.execute("SELECT claimed_by,claimed_at FROM agent_guidance_requests WHERE profile_id=?", (profile_id,)).fetchone()
    if row is None or row["claimed_by"] != user_id or row["claimed_at"] < now - LEASE_SECONDS:
        raise GuidanceError("this request is not claimed by you (claim it again)", 409)


def submit(db, user_id, data, now=None, rng=random.uniform):
    """Cache one reduction. It must cite the digest of the sources as they are now, and fit the profile's budget."""
    _require_reducer(db, user_id)
    now = time.time() if now is None else now
    profile_id, digest = data.get("profile_id"), data.get("source_digest")
    if not isinstance(profile_id, str) or not isinstance(digest, str):
        raise GuidanceError("profile_id and source_digest are required")
    body = agent_workspace._text(data.get("body"), "guidance", MAX_BUDGET).strip()
    if not body:
        raise GuidanceError("the guidance is empty")
    db.execute("BEGIN IMMEDIATE")
    try:
        _claimed(db, user_id, profile_id, now)
        current = sources(db, profile_id)
        if digest != current["digest"]:
            raise GuidanceError("the sources changed since you read them; claim the request again", 409)
        if len(body.encode("utf-8")) > current["budget_bytes"]:
            raise GuidanceError(f"the guidance is {len(body.encode('utf-8'))} bytes; this profile's budget is "
                                f"{current['budget_bytes']}", 413)
        drop_at = now + DROP_SECONDS * rng(*DROP_SPREAD)
        db.execute("INSERT INTO agent_guidance_cache VALUES (?,?,?,?,?,?) ON CONFLICT(profile_id) DO UPDATE SET "
                   "source_digest=excluded.source_digest, body=excluded.body, made_by=excluded.made_by, "
                   "made_at=excluded.made_at, drop_at=excluded.drop_at", (profile_id, digest, body, user_id, now, drop_at))
        db.execute("DELETE FROM agent_guidance_requests WHERE profile_id=?", (profile_id,))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return {"profile_id": profile_id, "source_digest": digest, "made_at": now, "drop_at": drop_at}


def fail(db, user_id, data, now=None):
    """Record why a claimed request could not be reduced, and let it go for the next claim."""
    _require_reducer(db, user_id)
    now = time.time() if now is None else now
    profile_id = data.get("profile_id")
    error = agent_workspace._text(data.get("error", ""), "error", 1000).strip() or "unspecified"
    db.execute("BEGIN IMMEDIATE")
    try:
        _claimed(db, user_id, profile_id, now)
        db.execute("UPDATE agent_guidance_requests SET claimed_by=NULL, claimed_at=NULL, error=? WHERE profile_id=?",
                   (error, profile_id))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return {"profile_id": profile_id, "error": error}


def status(db, now=None):
    """Every cached reduction (current or not, and why) and every waiting request; bodies are left out."""
    now = time.time() if now is None else now
    cached = []
    for row in db.execute("SELECT c.profile_id,p.title,c.source_digest,c.made_by,c.made_at,c.drop_at,length(c.body) AS bytes "
                          "FROM agent_guidance_cache c JOIN agent_profiles p ON p.id=c.profile_id ORDER BY p.title"):
        item = dict(row)
        item["current"] = ("dropped" if now >= item["drop_at"] else
                           "sources changed" if item["source_digest"] != sources(db, item["profile_id"])["digest"] else "yes")
        cached.append(item)
    waiting = [dict(row) for row in db.execute(
        "SELECT r.profile_id,p.title,r.requested_at,r.claimed_by,r.claimed_at,r.attempts,r.error FROM agent_guidance_requests r "
        "JOIN agent_profiles p ON p.id=r.profile_id ORDER BY r.requested_at")]
    return {"cached": cached, "requests": waiting}
