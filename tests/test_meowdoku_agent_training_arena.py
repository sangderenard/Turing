"""Arena queue, source visibility, and held-open agent pipe contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "examples" / "meowdoku" / "deploy"), str(ROOT / "tests")]

from meowdoku import agent_training_arena as arena  # noqa: E402
from meowdoku import agent_workspace as workspace  # noqa: E402
from meowdoku import authz  # noqa: E402
from test_meowdoku_server_authz import PUBLIC_XFF, _call, _signup, server  # noqa: E402,F401
from arena_seat import serve  # noqa: E402


def specification():
    return {"source": {"kind": "ticket", "id": "ticket-123"}, "goal": "Make a useful Research result",
            "target": {"agent": "codex", "stateful": True}, "judge": "claude", "training_mode": True,
            "runtime": {"turn_seconds": 600, "model": "test model", "parameters": {"temperature": 0}},
            "teams": [{"name": "Solo", "agents": ["codex"], "stateful": True},
                      {"name": "Pair", "agents": ["claude", "codex"], "stateful": False}]}


def receipt():
    return {"teams": [{"name": name, "scores": {axis: 2 for axis in arena.AXES},
                        "system_prompt": "prompt", "first_message": "hello", "response": "answer",
                        "pieces": {piece: {"scores": {axis: 2 for axis in arena.AXES}, "reasons": {}}
                                   for piece in ("prompt", "message", "response")},
                        "test_evidence": []} for name in ("Solo", "Pair")],
            "source": {"kind": "ticket", "id": "ticket-123"}, "goal": "Make a useful Research result",
            "runtime": specification()["runtime"], "training_mode": True}


def test_queue_is_durable_and_claim_token_controls_completion(tmp_path):
    db = workspace.connect(tmp_path / "accounts.db")
    run = arena.submit(db, 7, specification())
    assert run["status"] == "queued" and arena.listing(db, 8)["runs"] == []
    claim = arena.claim(db, 7)["run"]
    assert claim["id"] == run["id"] and arena.claim(db, 7)["run"] is None
    with pytest.raises(arena.ArenaError, match="claim"):
        arena.complete(db, 7, {"id": run["id"], "claim_token": "wrong", "result": receipt(), "error": ""})
    wrong = receipt(); wrong["goal"] = "different goal"
    with pytest.raises(arena.ArenaError, match="match"):
        arena.complete(db, 7, {"id": run["id"], "claim_token": claim["claim_token"],
                               "result": wrong, "error": ""})
    done = arena.complete(db, 7, {"id": run["id"], "claim_token": claim["claim_token"],
                                  "result": receipt(), "error": ""})
    assert done["status"] == "done"
    assert arena.by_source(db, "ticket", "ticket-123")["runs"][0]["result"]["teams"][0]["scores"]["code_elegance"] == 2
    assert arena.by_source(db, "research", "ticket-123")["runs"] == []
    db.close()


@pytest.mark.parametrize("change", [
    lambda s: s.update(teams=[{"name": "One", "agents": ["codex"], "stateful": True}]),
    lambda s: s["teams"][0].update(agents=["codex", "claude", "codex"]),
    lambda s: s["teams"][1].update(name="Solo"),
    lambda s: s["target"].update(stateful="yes"),
    lambda s: s["runtime"].update(turn_seconds=0),
])
def test_invalid_arena_specifications_are_rejected(change):
    spec = specification()
    change(spec)
    with pytest.raises(arena.ArenaError):
        arena.validate(spec)


def test_seat_keeps_stateful_process_and_restarts_on_new_prompt():
    opened = []
    class Fake:
        def __init__(self, system):
            self.system = system; self.calls = []; self.closed = False
            opened.append(self)
        def ask(self, text, seconds):
            self.calls.append((text, seconds))
            return f"{self.system}: {text}"
        def close(self): self.closed = True
    output = []
    commands = [{"op":"start","system":"prompt one"}, {"op":"ask","text":"first","seconds":60,"stateful":True},
                {"op":"ask","text":"second","seconds":60,"stateful":True},
                {"op":"start","system":"prompt two"}, {"op":"ask","text":"fresh","seconds":60,"stateful":False}]
    serve("codex", (json.dumps(x) for x in commands), output.append, factory=Fake)
    assert [x["op"] for x in output] == ["ready","started","answer","answer","started","answer"]
    assert len(opened) == 2 and opened[0].calls == [("first",60),("second",60)]
    assert opened[0].closed and opened[1].closed


def test_arena_route_requires_operator_and_write_header(server):
    operator = _signup(server.port, "arena_operator")
    member = _signup(server.port, "arena_member")
    authz.grant_role(server.sessions.accounts_db, "arena_operator", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator}
    assert _call(server.port, "/api/agent-training-arena", headers={**PUBLIC_XFF, "Cookie": member})[0] == 403
    assert _call(server.port, "/api/agent-training-arena", headers=headers)[0] == 200
    assert _call(server.port, "/api/agent-training-arena", "POST", headers, specification())[0] == 403


def test_completed_attempt_is_visible_only_to_source_reader(server):
    operator = _signup(server.port, "arena_ticket_owner")
    stranger = _signup(server.port, "arena_ticket_stranger")
    authz.grant_role(server.sessions.accounts_db, "arena_ticket_owner", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator}
    ticket_body = {"title": "A real ticket", "description": "Observed failure", "diagnostics": {},
                   "client_key": "arena-ticket-key-1"}
    status, created, _, _ = _call(server.port, "/api/issues", "POST",
                                  {**headers, "X-Meowdoku-Issue": "1"}, ticket_body)
    assert status == 201, created
    spec = specification(); spec["source"]["id"] = created["id"]
    status, queued, _, _ = _call(server.port, "/api/agent-training-arena", "POST",
                                 {**headers, "X-Meowdoku-Arena": "1"}, spec)
    assert status == 200, queued
    db = workspace.connect(server.sessions.accounts_db)
    owner_id = db.execute("SELECT id FROM users WHERE username='arena_ticket_owner'").fetchone()[0]
    claim = arena.claim(db, owner_id)["run"]
    result = receipt(); result["source"]["id"] = created["id"]
    arena.complete(db, owner_id, {"id": claim["id"], "claim_token": claim["claim_token"],
                                  "result": result, "error": ""})
    db.close()
    path = f"/api/agent-training-arena/by-source?kind=ticket&id={created['id']}"
    status, visible, _, _ = _call(server.port, path, headers=headers)
    assert status == 200 and visible["runs"][0]["id"] == queued["id"]
    assert _call(server.port, path, headers={**PUBLIC_XFF, "Cookie": stranger})[0] == 404
