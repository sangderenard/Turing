"""[DECISIONS] Each agent's decision graph: places with acts, routes to and from them (case 46d6ad2d, stage 3)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import agent_decisions as D  # noqa: E402
from meowdoku import agent_workspace as W  # noqa: E402
from meowdoku import living_data_query  # noqa: E402

DAY = 1_794_960_000.0


def _world(tmp_path):
    path = tmp_path / "accounts.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, display_name TEXT, created_at REAL)")
    raw.execute("CREATE TABLE temporary_accounts (user_id INTEGER PRIMARY KEY, expires_at REAL)")
    raw.executemany("INSERT INTO users VALUES (?,?,'x',?,0)", [(1, "boss", "boss"), (7, "llama", "llama"), (8, "codex", "codex")])
    raw.commit(); raw.close()
    db = W.connect(path)
    D.save_place(db, {"key": "arrived", "title": "Arrived", "body": "You have just started.", "start": True,
                      "acts": [{"tool": "pending", "for": "see what waits for you"}]})
    D.save_place(db, {"key": "messages", "title": "Messages waiting", "body": "Someone wrote to you.",
                      "acts": [{"tool": "say", "for": "answer"}, {"tool": "acknowledge", "for": "mark handled"}]})
    D.save_place(db, {"key": "triage", "title": "Ticket triage", "body": "Pick the worst problem."})
    D.save_route(db, {"source": "arrived", "target": "messages", "condition": "a message is waiting"})
    D.save_route(db, {"source": "arrived", "target": "triage", "condition": "nothing is waiting"})
    D.save_route(db, {"source": "messages", "target": "arrived", "condition": "every message is answered"})
    D.save_route(db, {"source": "triage", "target": "messages", "condition": "a message arrives"})
    return db


def test_a_place_holds_its_acts_and_its_routes_out_and_in(tmp_path):
    db = _world(tmp_path)
    view = D.here(db, 7)
    assert view["key"] == "arrived", "an account starts at the start place"
    assert [a["tool"] for a in view["acts"]] == ["pending"]
    assert {r["condition"]: r["title"] for r in view["routes"]} == {"a message is waiting": "Messages waiting",
                                                                    "nothing is waiting": "Ticket triage"}
    there = D.move(db, 7, "messages")
    assert [a["tool"] for a in there["acts"]] == ["say", "acknowledge"]
    assert {r["condition"] for r in there["from"]} == {"a message is waiting", "a message arrives"}, "routes in, from anywhere"
    assert [r["condition"] for r in there["routes"]] == ["every message is answered"], "and back out"
    assert D.here(db, 7)["key"] == "messages", "the place is kept"
    assert D.here(db, 8)["key"] == "arrived", "per account"


def test_the_map_is_one_the_cursor_walks(tmp_path):
    db = _world(tmp_path)
    found = D.as_map(db, 7)
    view = living_data_query.GraphView(found["map"])               # the cursor's own validation of the form
    assert found["at"] == D.identity("line", "arrived")
    assert len(view.nodes) >= 3
    relationships = {(t["source"], t["relationship"]) for t in found["map"]["tracks"]}
    assert (D.identity("line", "arrived"), "a message is waiting") in relationships
    place = next(o for o in found["map"]["world"]["objects"] if o["identity"] == D.identity("line", "messages"))
    assert place["text"] == "Someone wrote to you." and [a["tool"] for a in place["acts"]] == ["say", "acknowledge"]


def test_deleting_a_place_takes_its_routes_and_a_kept_place_falls_back_to_start(tmp_path):
    db = _world(tmp_path)
    D.move(db, 7, D.identity("line", "triage"))
    D.delete_place(db, {"key": "triage"})
    listing = D.listing(db, "line")
    assert all("triage" not in (r["source"], r["target"]) for r in listing["routes"])
    assert D.here(db, 7)["key"] == "arrived"


@pytest.mark.parametrize("call,data,error", [
    ("save_place", {"key": "Bad Key", "title": "x"}, "place key"),
    ("save_place", {"key": "ok", "title": "x", "acts": [{"tool": "Rm -rf", "for": ""}]}, "tool name"),
    ("save_place", {"key": "ok", "title": ""}, "needs a title"),
    ("save_route", {"source": "arrived", "target": "nowhere", "condition": "x"}, "not in graph"),
    ("save_route", {"source": "arrived", "target": "triage", "condition": "two\nlines"}, "one line"),
    ("save_route", {"source": "arrived", "target": "triage", "condition": "contained-by"}, "contained-by"),
])
def test_only_well_formed_places_and_routes_are_kept(tmp_path, call, data, error):
    db = _world(tmp_path)
    with pytest.raises(D.DecisionError, match=error):
        getattr(D, call)(db, data)


def test_moving_outside_the_graph_is_refused(tmp_path):
    db = _world(tmp_path)
    with pytest.raises(D.DecisionError) as refused:
        D.move(db, 7, "elsewhere")
    assert refused.value.status == 404


def test_a_character_may_walk_its_own_graph(tmp_path):
    db = _world(tmp_path)
    profile = W.save_profile(db, {"title": "Llama", "individual": "llama", "document_ids": []})
    D.save_place(db, {"graph": "chat", "key": "listening", "title": "Listening", "body": "Wait for a line addressed to you."})
    D.assign(db, {"profile_id": profile["id"], "graph": "chat"})
    assert D.graph_for(db, 7) == "chat" and D.here(db, 7)["key"] == "listening"
    assert D.graph_for(db, 8) == "line", "every other account walks the default graph"
    D.assign(db, {"profile_id": profile["id"], "graph": None})
    assert D.graph_for(db, 7) == "line"


def test_a_stateless_agent_is_told_where_it_stands_last(tmp_path):
    db = _world(tmp_path)
    profile = W.save_profile(db, {"title": "Llama", "individual": "llama", "document_ids": []})
    assert "Where you are" not in W.orientation(db, 7, now=DAY)["preamble"], "not declared: no place lines"
    W.save_capacity(db, 1, {"id": profile["id"], "state": "stateless", "context_tokens": 4096, "tools": "runner",
                            "session": "none"})
    D.move(db, 7, "messages")
    preamble = W.orientation(db, 7, now=DAY + 86400)["preamble"]
    assert preamble.startswith(W.CAPACITY_LINES["stateless"])
    tail = preamble[preamble.index("# Where you are"):]
    assert "Messages waiting" in tail and "say (answer)" in tail and "when every message is answered, go to Arrived" in tail
    W.save_capacity(db, 1, {"id": profile["id"], "state": "stateful", "context_tokens": 4096, "tools": "runner",
                            "session": "none"})
    assert "Where you are" not in W.orientation(db, 7, now=DAY + 2 * 86400)["preamble"], "a stateful agent walks it itself"


def test_the_decisions_are_their_own_schema_module(tmp_path):
    db = _world(tmp_path)
    assert dict(db.execute("SELECT module, version FROM schema_version"))["agent_decisions"] == 1
