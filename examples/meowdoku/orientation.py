"""Agent orientation: every markdown file of the core's repositories in one sequential causal order, as a map the Living Data
cursor walks, with filters and a reading position kept per account in SQL.

    documents(db, policy, user_id=, username=)   (ordered documents, notes) of every policy repository's default branch
    state(db, policy, user_id=, username=)       the account's filters, position and counts
    update(db, policy, body, user_id=, username=) change filters, set the position, or step through the order
    page(db, policy, user_id=, username=, offset=)  the filtered order as an abstract-ui-div-map-v0 map for the cursor
    http_route(self, path, method, *, policy, accounts_db)

The order ("links, history ties", owner 2026-11-13) is a depth-first walk of markdown links, taken in the order they appear
in each file. It starts at each repository's AGENTS.md and then its README.md, in branch-policy order, so the workspace map
comes first. A document no walk reaches starts a walk of its own, oldest first by the commit that added it; path order
breaks equal times. A link that climbs out of its repository's folder into another policy repository's folder is followed
there, so the workspace AGENTS.md leads straight into speaktome/AGENTS.md.

A filter keeps the order and shows the matching subsequence: a document's place in the walk does not change when it is
filtered. Every read goes through core_git: its declared operations, the core_git capability, and one git_calls row per call.
A repository's documents are kept in memory per commit, so a request costs one log call per repository unless a default
branch moved.
"""
from __future__ import annotations

import json
import posixpath
import threading
import time
from pathlib import Path

from meowdoku import core_git

PAGE = 500              # documents per map page (about 470 KB); far inside living_data_query MAX_NODES and MAX_EDGES
ROOT = "orientation"
STARTS = ("AGENTS.md", "README.md")
DDL = ("CREATE TABLE IF NOT EXISTS orientation_state (user_id INTEGER PRIMARY KEY, filters TEXT NOT NULL, at TEXT, "
       "updated_at REAL NOT NULL)")
# A document's kind, the first rule that holds, by its lower-cased path and file name.
KINDS = (
    ("agents", lambda path, name: name.startswith("agents")),
    ("readme", lambda path, name: name.startswith("readme")),
    ("report", lambda path, name: any(part in path for part in ("experience_reports/", "guestbook/", "docs/hazards/"))),
    ("handoff", lambda path, name: "handoff" in name),
    ("doc", lambda path, name: True),
)
FILTERS = ("repository", "prefix", "kind", "keyword")
STEPS = ("first", "next", "previous", "last")

_cache: dict = {}       # (repository, commit) -> (paths, links, added)
_lock = threading.Lock()


class OrientationError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def identity(repository: str, path: str) -> str:
    return f"doc:{repository}/{path}"


def kind_of(path: str) -> str:
    lowered = path.lower()
    name = lowered.rsplit("/", 1)[-1]
    return next(kind for kind, holds in KINDS if holds(lowered, name))


def _git(db, policy, who, repository, operation, **arguments) -> dict:
    return core_git.run(db, policy, user_id=who[0], username=who[1], repository=repository, operation=operation,
                        arguments=arguments)


def _read(db, policy, who, repository) -> tuple:
    """(commit, paths, links, added, truncated) of one repository's default branch."""
    default = policy["repositories"][repository]["default"]
    head = _git(db, policy, who, repository, "log", ref=default, count=1)["output"].split("\t", 1)[0].strip()
    with _lock:
        held = _cache.get((repository, head))
    if held is not None:
        return (head, *held)
    tree = _git(db, policy, who, repository, "tree", ref=head)
    paths = [p for p in tree["output"].splitlines() if p.lower().endswith(".md")]
    grep = _git(db, policy, who, repository, "markdown_links", ref=head)
    links: dict = {}
    for line in grep["output"].splitlines():
        if not line.startswith(head + ":"):
            continue
        path, _, rest = line[len(head) + 1:].partition(":")
        target = rest.partition(":")[2][2:]                     # "](target.md" -> "target.md"
        links.setdefault(path, []).append(target)
    log = _git(db, policy, who, repository, "markdown_added", ref=head)
    added: dict = {}
    for block in log["output"].split("\x00")[1:]:               # newest first: the last time seen is the first add
        lines = [line for line in block.splitlines() if line.strip()]
        if lines and lines[0].strip().isdigit():
            for path in lines[1:]:
                added[path] = int(lines[0])
    truncated = any(r["truncated"] for r in (tree, grep, log))
    held = (paths, links, added, truncated)
    with _lock:
        _cache[(repository, head)] = held
    return (head, *held)


def _resolve(policy, roots, repository, path, target):
    """The (repository, path) a link names, or None for a web address, an anchor-only link or a path outside the core."""
    if "://" in target or target.lower().startswith("mailto:"):
        return None
    if target.startswith("/"):
        joined = posixpath.normpath(target.lstrip("/"))
    else:
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
    inside = posixpath.normpath(roots[repository] + "/" + joined)   # may climb out of this repository into another one
    # the most specific policy repository whose folder holds the target
    found, depth = None, -1
    for name, root in roots.items():
        if (inside + "/").lower().startswith(root.lower() + "/") and root.count("/") > depth:
            found, depth = name, root.count("/")
    if found is None:
        return None
    return found, inside[len(roots[found]) + 1:]


def documents(db, policy, *, user_id, username) -> tuple:
    """(documents in the causal order, notes). A document is a dict: identity, repository, path, kind, ref, added, links."""
    who = (user_id, username)
    roots = {name: str(Path(d["path"])).replace("\\", "/").rstrip("/") for name, d in policy["repositories"].items()}
    docs, notes = {}, {"truncated": [], "unavailable": []}
    for repository in policy["repositories"]:
        try:
            head, paths, links, added, truncated = _read(db, policy, who, repository)
        except core_git.GitError as error:
            if error.status in (404, 503):
                notes["unavailable"].append(repository)
                continue
            raise
        if truncated:
            notes["truncated"].append(repository)
        for path in paths:
            docs[identity(repository, path)] = {"identity": identity(repository, path), "repository": repository,
                                                "path": path, "kind": kind_of(path), "ref": head,
                                                "added": added.get(path), "raw_links": links.get(path, [])}
    for doc in docs.values():
        targets = []
        for target in doc.pop("raw_links"):
            found = _resolve(policy, roots, doc["repository"], doc["path"], target)
            if found is not None and identity(*found) in docs and identity(*found) not in targets \
                    and identity(*found) != doc["identity"]:
                targets.append(identity(*found))
        doc["links"] = targets
    seen, ordered = set(), []

    def walk(start):
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ordered.append(docs[current])
            stack.extend(reversed([t for t in docs[current]["links"] if t not in seen]))

    for repository in policy["repositories"]:
        for start in STARTS:
            if identity(repository, start) in docs:
                walk(identity(repository, start))
    rest = sorted((d for d in docs.values() if d["identity"] not in seen),
                  key=lambda d: (d["added"] is None, d["added"] or 0, d["repository"], d["path"]))
    for doc in rest:
        walk(doc["identity"])
    for position, doc in enumerate(ordered, 1):
        doc["order"] = position
    return ordered, notes


def _filters(policy, raw) -> dict:
    if not isinstance(raw, dict) or set(raw) - set(FILTERS):
        raise OrientationError("filters are " + ", ".join(FILTERS))
    clean = {}
    for key, value in raw.items():
        if value in (None, ""):
            continue
        if not isinstance(value, str) or len(value) > 200:
            raise OrientationError(f"{key} must be text of at most 200 characters")
        if key == "repository" and value not in policy["repositories"]:
            raise OrientationError("repository must be one of " + ", ".join(policy["repositories"]))
        if key == "kind" and value not in [k for k, _ in KINDS]:
            raise OrientationError("kind must be one of " + ", ".join(k for k, _ in KINDS))
        clean[key] = value
    return clean


def _matches(doc, filters) -> bool:
    return ((filters.get("repository") in (None, doc["repository"]))
            and doc["path"].lower().startswith(filters.get("prefix", "").lower().lstrip("/"))
            and filters.get("kind") in (None, doc["kind"])
            and filters.get("keyword", "").lower() in doc["path"].lower())


def _stored(db, user_id) -> tuple:
    db.execute(DDL)
    row = db.execute("SELECT filters, at FROM orientation_state WHERE user_id = ?", (user_id,)).fetchone()
    return (json.loads(row[0]), row[1]) if row else ({}, None)


def _store(db, user_id, filters, at) -> None:
    db.execute(DDL)
    db.execute("INSERT INTO orientation_state (user_id, filters, at, updated_at) VALUES (?,?,?,?) "
               "ON CONFLICT(user_id) DO UPDATE SET filters = excluded.filters, at = excluded.at, "
               "updated_at = excluded.updated_at", (user_id, json.dumps(filters, sort_keys=True), at, time.time()))


def _view(db, policy, user_id, username):
    ordered, notes = documents(db, policy, user_id=user_id, username=username)
    filters, at = _stored(db, user_id)
    matching = [d for d in ordered if _matches(d, filters)]
    index = next((i for i, d in enumerate(matching) if d["identity"] == at), None)
    return ordered, notes, filters, matching, index


def _public(doc) -> dict:
    return {k: doc[k] for k in ("identity", "repository", "path", "kind", "ref", "added", "order")}


def _summary(ordered, notes, filters, matching, index) -> dict:
    here = matching[index] if index is not None else None
    return {"filters": filters, "at": here["identity"] if here else None, "document": _public(here) if here else None,
            "position": index + 1 if index is not None else None, "matching": len(matching), "total": len(ordered),
            "next": matching[index + 1]["identity"] if index is not None and index + 1 < len(matching) else None,
            "previous": matching[index - 1]["identity"] if index else None,
            "repositories": sorted({d["repository"] for d in ordered}), "kinds": [k for k, _ in KINDS],
            "steps": list(STEPS), "truncated": notes["truncated"], "unavailable": notes["unavailable"]}


def state(db, policy, *, user_id, username) -> dict:
    view = _view(db, policy, user_id, username)
    out = _summary(*view)
    out["say"] = (f"Document {out['position']} of {out['matching']} matching ({out['total']} in all)." if out["position"]
                  else f"{out['matching']} of {out['total']} documents match; no position yet (step first).")
    return out


def update(db, policy, body, *, user_id, username) -> dict:
    """``{filters?: {...} | clear?: true, at?: identity | step?: first|next|previous|last}``."""
    if not isinstance(body, dict) or set(body) - {"filters", "clear", "at", "step"}:
        raise OrientationError("give filters or clear, and at or step")
    if "filters" in body and body.get("clear"):
        raise OrientationError("give filters or clear, not both")
    if "at" in body and "step" in body:
        raise OrientationError("give at or step, not both")
    ordered, notes, filters, matching, index = _view(db, policy, user_id, username)
    _, at = _stored(db, user_id)
    if body.get("clear"):
        filters = {}
    elif "filters" in body:
        filters = _filters(policy, body["filters"])
    matching = [d for d in ordered if _matches(d, filters)]
    index = next((i for i, d in enumerate(matching) if d["identity"] == at), None)
    if "at" in body:
        index = next((i for i, d in enumerate(matching) if d["identity"] == body["at"]), None)
        if index is None:
            raise OrientationError("that document is not in the current order (check the filters)", 404)
    elif "step" in body:
        if body["step"] not in STEPS:
            raise OrientationError("step must be one of " + ", ".join(STEPS))
        if not matching:
            raise OrientationError("no document matches the filters", 404)
        step = body["step"]
        if step == "first" or (index is None and step == "next"):
            index = 0
        elif step == "last" or (index is None and step == "previous"):
            index = len(matching) - 1
        elif step == "next":
            index = min(index + 1, len(matching) - 1)
        else:
            index = max(index - 1, 0)
    _store(db, user_id, filters, matching[index]["identity"] if index is not None else at)
    out = _summary(ordered, notes, filters, matching, index)
    out["say"] = (f"Document {out['position']} of {out['matching']}: {out['document']['repository']}/{out['document']['path']}."
                  if out["position"] else f"{out['matching']} of {out['total']} documents match.")
    return out


def page(db, policy, *, user_id, username, offset=None) -> dict:
    """One page of the filtered order as a map: the root holds a place per repository, each repository holds its
    documents, ``next`` runs through the order (its reverse is previous), and ``links-to`` is a document's own links."""
    ordered, notes, filters, matching, index = _view(db, policy, user_id, username)
    if offset is None:
        offset = (index // PAGE) * PAGE if index is not None else 0
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or (offset and offset >= len(matching)):
        raise OrientationError("offset is outside the order")
    shown = matching[offset:offset + PAGE]
    on_page = {d["identity"] for d in shown}
    regions, tracks, objects = [], [], []      # the map's own identity is the root place
    for repository in sorted({d["repository"] for d in shown}, key=list(policy["repositories"]).index):
        regions.append({"identity": f"{ROOT}/{repository}", "kind": "repository", "name": repository, "buildings": []})
        tracks.append({"source": f"{ROOT}/{repository}", "target": ROOT, "relationship": "contained-by"})
    for i, doc in enumerate(shown):
        name = f"{doc['repository']}/{doc['path']}"
        regions.append({"identity": doc["identity"], "kind": doc["kind"], "name": name, "buildings": []})
        tracks.append({"source": doc["identity"], "target": f"{ROOT}/{doc['repository']}", "relationship": "contained-by"})
        if i + 1 < len(shown):
            tracks.append({"source": doc["identity"], "target": shown[i + 1]["identity"], "relationship": "next"})
        tracks.extend({"source": doc["identity"], "target": t, "relationship": "links-to"} for t in doc["links"] if t in on_page)
        objects.append({**_public(doc), "name": name, "position": offset + i + 1})
    if shown:
        tracks.append({"source": ROOT, "target": shown[0]["identity"], "relationship": "begins"})
    at = matching[index]["identity"] if index is not None and matching[index]["identity"] in on_page else ROOT
    summary = _summary(ordered, notes, filters, matching, index)
    return {"map": {"schema": "abstract-ui-div-map-v0", "identity": ROOT, "kind": "orientation", "name": "Agent orientation",
                    "regions": regions, "tracks": tracks, "world": {"objects": objects}},
            "at": at, "offset": offset, "next_offset": offset + PAGE if offset + PAGE < len(matching) else None,
            "previous_offset": max(offset - PAGE, 0) if offset else None, **{k: summary[k] for k in (
                "filters", "position", "matching", "total", "truncated", "unavailable")},
            "say": f"Documents {offset + 1 if shown else 0} to {offset + len(shown)} of {len(matching)} matching, in causal order."}


def http_route(self, path, method, *, policy, accounts_db):
    """Agent orientation over REST, for accounts that hold core_git and dev_ui (the reads are core_git's).

        GET  /api/orientation                   the filters, the position and the counts
        POST /api/orientation                   {filters?: {repository?, prefix?, kind?, keyword?} | clear?: true,
                                                 at?: <document identity> | step?: first|next|previous|last}
        GET  /api/orientation/map?offset=N      one page of the filtered order as a map for /api/living-data/sessions
    """
    import sqlite3
    from urllib.parse import parse_qs
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    caps = self._caps()
    if "core_git" not in caps or "dev_ui" not in caps:
        return self._json({"error": "this needs core_git and dev_ui"}, 403)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    who = {"user_id": principal.user_id, "username": principal.username}
    try:
        query = parse_qs(self.path.partition("?")[2], keep_blank_values=True)
        if method == "GET" and path == "/api/orientation":
            if query:
                raise OrientationError("this takes no query")
            return self._json(state(db, policy, **who))
        if method == "GET" and path == "/api/orientation/map":
            if set(query) - {"offset"} or any(len(v) != 1 for v in query.values()):
                raise OrientationError("the map takes only offset")
            offset = int(query["offset"][0]) if "offset" in query else None
            return self._json(page(db, policy, offset=offset, **who))
        if method == "POST" and path == "/api/orientation":
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return self._json({"error": "JSON only"}, 415)
            body = self._body_json(4096)
            if body is None:
                return
            return self._json(update(db, policy, body, **who))
        return self._json({"error": "not found"}, 404)
    except ValueError:
        return self._json({"error": "offset must be a whole number"}, 400)
    except (OrientationError, core_git.GitError) as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
