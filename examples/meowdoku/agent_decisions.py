"""[DECISIONS] Each agent's decision graph (case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 3).

Not a binary tree but a graph of places (owner, 2026-11-17: "a graph decision tree, you can perform operation or act and
have routes from and to where you are"). A place is a situation:
- its text says what the situation is and why;
- its acts are what may be done there without moving: tool names, each with what it is for;
- its routes lead to other places, each labelled with the condition that takes it. Any place may have any number of
  routes out and in, back routes included.
At each place the agent chooses among a few concrete acts and routes, not among everything it knows: the contextual
reduction to one decision.

The graph is served as a map of the form the cursor verbs already walk (abstract-ui-div-map-v0, like the manual and the
orientation): enter map=decisions, where, go relationship=<condition>, examine. A route's condition is its relationship.
Each account's place is kept, so a stateless agent resumes where it was. Operators edit the graphs; a character may be
assigned a graph, and every other account walks DEFAULT_GRAPH.

    prepare(db)                                   the tables, under schema module agent_decisions 1
    graph_for(db, user_id)                        the graph an account walks
    listing(db, graph)                            every place, route and assignment of a graph (operators)
    save_place, delete_place, save_route, delete_route, assign      operator edits
    as_map(db, user_id)                           the account's graph as a map, and where it stands
    move(db, user_id, at)                         keep the account's place (a place key or identity)
    here(db, user_id)                             the account's place, its acts and its routes out and in
    where_text(db, user_id)                       the same as the lines that end a stateless agent's orientation
"""
from __future__ import annotations

import json
import re
import time

from meowdoku import agent_workspace, schema_version

DEFAULT_GRAPH = "line"
PREFIX = "decisions:"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")          # a graph's name and a place's key
TOOL = re.compile(r"[a-z][a-z0-9_]{0,79}")
MAX_PLACES = 256
MAX_ROUTES = 2048
MAX_ACTS = 32
MAX_BODY = 16 * 1024


class DecisionError(agent_workspace.WorkspaceError):
    pass


def prepare(db):
    """The tables. Called by agent_workspace.connect, outside any transaction (declare commits)."""
    schema_version.declare(db, "agent_decisions", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_decision_places (graph TEXT NOT NULL, key TEXT NOT NULL, "
               "title TEXT NOT NULL, body TEXT NOT NULL, acts TEXT NOT NULL, start INTEGER NOT NULL DEFAULT 0, "
               "updated REAL NOT NULL, PRIMARY KEY(graph, key))")
    db.execute("CREATE TABLE IF NOT EXISTS agent_decision_routes (graph TEXT NOT NULL, source TEXT NOT NULL, "
               "target TEXT NOT NULL, condition TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(graph, source, condition), "
               "FOREIGN KEY(graph, source) REFERENCES agent_decision_places(graph, key) ON DELETE CASCADE, "
               "FOREIGN KEY(graph, target) REFERENCES agent_decision_places(graph, key) ON DELETE CASCADE)")
    db.execute("CREATE TABLE IF NOT EXISTS agent_decision_assignments (profile_id TEXT PRIMARY KEY "
               "REFERENCES agent_profiles(id) ON DELETE CASCADE, graph TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS agent_decision_at (user_id INTEGER NOT NULL, graph TEXT NOT NULL, "
               "key TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(user_id, graph))")


# ---------------------------------------------------------------- reading
def _name(value, label):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise DecisionError(f"{label} must be 1 to 64 lower-case letters, digits or dashes")
    return value


def identity(graph, key):
    return f"{PREFIX}{graph}/{key}"


def _key(graph, value):
    """A place key from a key or an identity of this graph."""
    if isinstance(value, str) and value.startswith(identity(graph, "")):
        value = value[len(identity(graph, "")):]
    return _name(value, "place")


def graph_for(db, user_id):
    profile_id = agent_workspace._profile_for(db, user_id)
    row = db.execute("SELECT graph FROM agent_decision_assignments WHERE profile_id=?", (profile_id,)).fetchone() \
        if profile_id else None
    return row[0] if row else DEFAULT_GRAPH


def _places(db, graph):
    return [{**dict(row), "acts": json.loads(row["acts"]), "start": bool(row["start"])} for row in db.execute(
        "SELECT key,title,body,acts,start,updated FROM agent_decision_places WHERE graph=? ORDER BY start DESC, key", (graph,))]


def _routes(db, graph):
    return [dict(row) for row in db.execute(
        "SELECT source,target,condition,updated FROM agent_decision_routes WHERE graph=? ORDER BY source, condition", (graph,))]


def listing(db, graph):
    graph = _name(graph, "graph")
    return {"graph": graph, "places": _places(db, graph), "routes": _routes(db, graph),
            "assigned": [dict(row) for row in db.execute(
                "SELECT a.profile_id, p.title FROM agent_decision_assignments a JOIN agent_profiles p ON p.id=a.profile_id "
                "WHERE a.graph=? ORDER BY p.title", (graph,))],
            "graphs": [row[0] for row in db.execute("SELECT DISTINCT graph FROM agent_decision_places ORDER BY graph")]}


def _at(db, user_id, graph, places):
    """The account's kept place, else the graph's start place, else its first."""
    keys = {p["key"] for p in places}
    row = db.execute("SELECT key FROM agent_decision_at WHERE user_id=? AND graph=?", (user_id, graph)).fetchone()
    if row and row[0] in keys:
        return row[0]
    return places[0]["key"] if places else None


def _view(db, user_id):
    graph = graph_for(db, user_id)
    places, routes = _places(db, graph), _routes(db, graph)
    if not places:
        raise DecisionError(f"the decision graph {graph!r} has no places yet", 404)
    return graph, places, routes, _at(db, user_id, graph, places)


def _here(graph, places, routes, at):
    by_key = {p["key"]: p for p in places}
    place = by_key[at]
    return {"graph": graph, "at": identity(graph, at), "key": at, "title": place["title"], "body": place["body"],
            "acts": place["acts"],
            "routes": [{"condition": r["condition"], "to": identity(graph, r["target"]), "title": by_key[r["target"]]["title"]}
                       for r in routes if r["source"] == at],
            "from": [{"condition": r["condition"], "from": identity(graph, r["source"]), "title": by_key[r["source"]]["title"]}
                     for r in routes if r["target"] == at]}


def here(db, user_id):
    return _here(*_view(db, user_id))


def as_map(db, user_id):
    graph, places, routes, at = _view(db, user_id)
    root = PREFIX + graph
    regions, tracks, objects = [], [], []
    for place in places:
        ident = identity(graph, place["key"])
        regions.append({"identity": ident, "kind": "place", "name": place["title"], "buildings": []})
        tracks.append({"source": ident, "target": root, "relationship": "contained-by"})
        view = _here(graph, places, routes, place["key"])
        objects.append({"identity": ident, "kind": "place", "name": place["title"], "text": place["body"],
                        "acts": place["acts"], "routes": view["routes"], "from": view["from"], "start": place["start"]})
    tracks.extend({"source": identity(graph, r["source"]), "target": identity(graph, r["target"]),
                   "relationship": r["condition"]} for r in routes)
    return {"map": {"schema": "abstract-ui-div-map-v0", "identity": root, "kind": "decisions",
                    "name": f"Decision graph {graph}", "regions": regions, "tracks": tracks, "world": {"objects": objects}},
            "at": identity(graph, at), "graph": graph,
            "say": f"The decision graph {graph}: {len(places)} places, {len(routes)} routes; you are at {dict((p['key'], p['title']) for p in places)[at]}."}


def move(db, user_id, at):
    graph, places, routes, _ = _view(db, user_id)
    key = _key(graph, at)
    if key not in {p["key"] for p in places}:
        raise DecisionError("that place is not in your decision graph", 404)
    db.execute("INSERT INTO agent_decision_at(user_id,graph,key,updated) VALUES (?,?,?,?) ON CONFLICT(user_id,graph) "
               "DO UPDATE SET key=excluded.key, updated=excluded.updated", (user_id, graph, key, time.time()))
    return _here(graph, places, routes, key)


def where_text(db, user_id):
    """Where the account stands, for the end of a stateless agent's orientation; empty when its graph has no places."""
    try:
        view = here(db, user_id)
    except DecisionError:
        return ""
    lines = [f"# Where you are: {view['title']} (decision graph {view['graph']})", view["body"].strip()]
    if view["acts"]:
        lines.append("Here you can: " + "; ".join(f"{act['tool']} ({act['for']})" for act in view["acts"]) + ".")
    if view["routes"]:
        lines.append("Routes from here: " + "; ".join(f"when {r['condition']}, go to {r['title']}" for r in view["routes"]) + ".")
    lines.append("Walk it with enter map=decisions, where, go relationship=<condition> and examine.")
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- operator edits
def _acts(value):
    if not isinstance(value, list) or len(value) > MAX_ACTS:
        raise DecisionError(f"acts must be a list of at most {MAX_ACTS}")
    out = []
    for act in value:
        if not isinstance(act, dict) or set(act) - {"tool", "for"}:
            raise DecisionError("each act is {tool, for}")
        tool, why = act.get("tool"), act.get("for", "")
        if not isinstance(tool, str) or not TOOL.fullmatch(tool):
            raise DecisionError("an act's tool is a tool name: lower-case letters, digits and underscores")
        if not isinstance(why, str) or len(why) > 300 or "\n" in why:
            raise DecisionError("an act's 'for' is one line of at most 300 characters")
        out.append({"tool": tool, "for": why.strip()})
    return out


def save_place(db, data):
    graph, key = _name(data.get("graph", DEFAULT_GRAPH), "graph"), _name(data.get("key"), "place key")
    title = agent_workspace._text(data.get("title"), "place title", 160).strip()
    body = agent_workspace._text(data.get("body", ""), "place text", MAX_BODY)
    if not title:
        raise DecisionError("a place needs a title")
    acts, start = _acts(data.get("acts", [])), data.get("start", False)
    if not isinstance(start, bool):
        raise DecisionError("start is true or false")
    db.execute("BEGIN IMMEDIATE")
    try:
        exists = db.execute("SELECT 1 FROM agent_decision_places WHERE graph=? AND key=?", (graph, key)).fetchone()
        if not exists and db.execute("SELECT COUNT(*) FROM agent_decision_places WHERE graph=?", (graph,)).fetchone()[0] >= MAX_PLACES:
            raise DecisionError("place quota reached", 429)
        if start:
            db.execute("UPDATE agent_decision_places SET start=0 WHERE graph=?", (graph,))
        db.execute("INSERT INTO agent_decision_places(graph,key,title,body,acts,start,updated) VALUES (?,?,?,?,?,?,?) "
                   "ON CONFLICT(graph,key) DO UPDATE SET title=excluded.title, body=excluded.body, acts=excluded.acts, "
                   "start=excluded.start, updated=excluded.updated",
                   (graph, key, title, body, json.dumps(acts), int(start), time.time()))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return listing(db, graph)


def delete_place(db, data):
    graph, key = _name(data.get("graph", DEFAULT_GRAPH), "graph"), _name(data.get("key"), "place key")
    if db.execute("DELETE FROM agent_decision_places WHERE graph=? AND key=?", (graph, key)).rowcount == 0:
        raise DecisionError("place not found", 404)
    db.execute("DELETE FROM agent_decision_at WHERE graph=? AND key=?", (graph, key))
    return listing(db, graph)                                # its routes in and out went with it (ON DELETE CASCADE)


def save_route(db, data):
    graph = _name(data.get("graph", DEFAULT_GRAPH), "graph")
    source, target = _name(data.get("source"), "route source"), _name(data.get("target"), "route target")
    condition = data.get("condition")
    if not isinstance(condition, str) or not condition.strip() or len(condition) > 200 or "\n" in condition:
        raise DecisionError("a route's condition is one line of 1 to 200 characters")
    condition = condition.strip()
    if condition == "contained-by":
        raise DecisionError("contained-by is the map's own relationship")
    for key in (source, target):
        if not db.execute("SELECT 1 FROM agent_decision_places WHERE graph=? AND key=?", (graph, key)).fetchone():
            raise DecisionError(f"place {key!r} is not in graph {graph!r}", 404)
    if db.execute("SELECT COUNT(*) FROM agent_decision_routes WHERE graph=?", (graph,)).fetchone()[0] >= MAX_ROUTES:
        raise DecisionError("route quota reached", 429)
    db.execute("INSERT INTO agent_decision_routes(graph,source,target,condition,updated) VALUES (?,?,?,?,?) "
               "ON CONFLICT(graph,source,condition) DO UPDATE SET target=excluded.target, updated=excluded.updated",
               (graph, source, target, condition, time.time()))
    return listing(db, graph)


def delete_route(db, data):
    graph, source = _name(data.get("graph", DEFAULT_GRAPH), "graph"), _name(data.get("source"), "route source")
    condition = data.get("condition")
    if db.execute("DELETE FROM agent_decision_routes WHERE graph=? AND source=? AND condition=?",
                  (graph, source, condition)).rowcount == 0:
        raise DecisionError("route not found", 404)
    return listing(db, graph)


def assign(db, data):
    """A character walks the named graph; graph null returns it to DEFAULT_GRAPH."""
    profile_id = data.get("profile_id")
    agent_workspace.profile(db, profile_id)                    # 404 for an unknown character
    if data.get("graph") is None:
        db.execute("DELETE FROM agent_decision_assignments WHERE profile_id=?", (profile_id,))
        return {"profile_id": profile_id, "graph": DEFAULT_GRAPH}
    graph = _name(data.get("graph"), "graph")
    db.execute("INSERT INTO agent_decision_assignments(profile_id,graph) VALUES (?,?) ON CONFLICT(profile_id) "
               "DO UPDATE SET graph=excluded.graph", (profile_id, graph))
    return {"profile_id": profile_id, "graph": graph}


def change(db, data):
    """The operator route's actions."""
    if not isinstance(data, dict):
        raise DecisionError("object required")
    actions = {"place": save_place, "delete_place": delete_place, "route": save_route, "delete_route": delete_route,
               "assign": assign}
    action = actions.get(data.get("action"))
    if action is None:
        raise DecisionError("action must be place, delete_place, route, delete_route or assign")
    return action(db, data)
