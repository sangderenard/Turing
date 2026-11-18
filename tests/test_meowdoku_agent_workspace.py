"""Shared operator tree and private agent scratch, over temporary state only."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import agent_workspace as work  # noqa: E402
from meowdoku import authz  # noqa: E402
from meowdoku import server as srv  # noqa: E402
from test_meowdoku_server_authz import PUBLIC_XFF, _call, _signup, server  # noqa: E402,F401


def test_shared_tree_scenario_order_and_prompt_quota(tmp_path, monkeypatch):
    db = work.connect(tmp_path / "accounts.db")
    monkeypatch.setattr(work, "MAX_PROMPTS_PER_OPERATOR", 1)
    root = work.save_document(db, 11, {"kind": "personality", "title": "Character", "body": "Read only when relevant"})
    leaf = work.save_document(db, 12, {"kind": "personality", "title": "More specific", "body": "Narrow context", "parent_id": root["id"]})
    assert [row["id"] for row in work.children(db, root["id"])["documents"]] == [leaf["id"]]
    assert "body" not in work.index(db)["roots"][0]
    scenario = work.save_document(db, 11, {"kind": "scenario", "title": "A scene", "body": "Act", "contexts": [root["id"], leaf["id"]]})
    assert work.document(db, scenario["id"])["contexts"] == [root["id"], leaf["id"]]
    work.save_document(db, 11, {"kind": "prompt", "title": "Orientation prompt", "body": "Read index"})
    with pytest.raises(work.WorkspaceError, match="quota"):
        work.save_document(db, 11, {"kind": "prompt", "title": "Second", "body": "No"})
    with pytest.raises(work.WorkspaceError, match="cycle"):
        work.save_document(db, 12, {**leaf, "parent_id": leaf["id"]})
    with pytest.raises(work.WorkspaceError, match="used by a scenario"):
        work.delete_document(db, leaf["id"])
    db.close()


def test_scratch_is_per_account_and_temp_cookie_is_bounded(tmp_path):
    db = work.connect(tmp_path / "accounts.db")
    work.scratch_write(db, 11, "notes", "private")
    assert work.scratch_read(db, 11, "notes")["body"] == "private"
    with pytest.raises(work.WorkspaceError, match="not found"):
        work.scratch_read(db, 12, "notes")
    cookie = work.temp_cookie(19, {"map": "one"})
    assert work.temp_files(cookie, 19) == {"map": "one"}
    assert work.temp_files(cookie, 20) == {}
    with pytest.raises(work.WorkspaceError, match="full"):
        work.temp_cookie(19, {"big": "a" * work.MAX_TEMP_BYTES})
    db.close()


def test_operator_api_shares_documents_but_scratch_remains_private(server):
    first = _signup(server.port, "operator_one")
    second = _signup(server.port, "operator_two")
    member = _signup(server.port, "ordinary")
    authz.grant_role(server.sessions.accounts_db, "operator_one", "operator", "test")
    authz.grant_role(server.sessions.accounts_db, "operator_two", "operator", "test")
    headers = lambda cookie: {**PUBLIC_XFF, "Cookie": cookie, "X-Meowdoku-Workspace": "1"}
    assert _call(server.port, "/api/agent-workspace", headers=headers(member))[0] == 403
    status, item, _, _ = _call(server.port, "/api/agent-workspace/documents", "POST", headers(first),
                               {"action": "save", "kind": "guidance", "title": "Shared", "body": "first"})
    assert status == 200, item
    item["action"] = "save"; item["body"] = "second"
    status, updated, _, _ = _call(server.port, "/api/agent-workspace/documents", "POST", headers(second), item)
    assert status == 200 and updated["body"] == "second"
    assert _call(server.port, "/api/agent-workspace/document/" + item["id"], headers=headers(first))[1]["body"] == "second"
    assert _call(server.port, "/api/agent-workspace", headers={"Host": "127.0.0.1", "Cookie": first})[0] == 403
    assert _call(server.port, "/api/scratch", "POST", headers(first),
                 {"action": "write", "name": "idea", "body": "one"})[0] == 200
    assert _call(server.port, "/api/scratch/file?name=idea", headers=headers(second))[0] == 404
    assert _call(server.port, "/api/scratch", "POST", headers(member),
                 {"action": "write", "name": "mine", "body": "member space"})[0] == 200


def test_temporary_preview_scratch_stays_in_its_cookie(server):
    preview = server.sessions.accounts.create_preview()
    token = server.sessions.accounts.create_session(preview["id"])
    account_cookie = srv.ACCOUNT_COOKIE + "=" + token
    headers = {**PUBLIC_XFF, "Cookie": account_cookie, "X-Meowdoku-Workspace": "1"}
    status, out, cookies, _ = _call(server.port, "/api/scratch", "POST", headers,
                                    {"action": "write", "name": "clue", "body": "a temporary note"})
    assert status == 200 and out["storage"] == "cookie", out
    saved = next(c.split(";", 1)[0] for c in cookies if c.startswith(work.TEMP_COOKIE + "="))
    assert _call(server.port, "/api/scratch/file?name=clue", headers={**headers, "Cookie": account_cookie + "; " + saved})[1]["body"] == "a temporary note"
    assert _call(server.port, "/api/scratch/file?name=clue", headers=headers)[0] == 404
    assert _call(server.port, "/api/agent-workspace", headers=headers)[0] == 403


def test_titled_profiles_job_changes_and_operator_boundary(server):
    operator = _signup(server.port, "profile_operator")
    other = _signup(server.port, "profile_person")
    authz.grant_role(server.sessions.accounts_db, "profile_operator", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator, "X-Meowdoku-Workspace": "1"}
    def post(path, body):
        status, result, _, _ = _call(server.port, path, "POST", headers, body)
        assert status == 200, result
        return result
    base = post("/api/agent-workspace/documents", {"action": "save", "kind": "prompt", "title": "Core", "body": "base"})
    changed = post("/api/agent-workspace/documents", {"action": "save", "kind": "prompt", "title": "Job core", "body": "changed"})
    extra = post("/api/agent-workspace/documents", {"action": "save", "kind": "guidance", "title": "Extra", "body": "added"})
    profile = post("/api/agent-workspace/profiles", {"action": "save", "title": "The character",
                   "individual": "profile_person", "document_ids": [base["id"]]})
    second = post("/api/agent-workspace/profiles", {"action": "save", "title": "Another character",
                  "individual": "profile_person", "document_ids": [extra["id"]]})
    assert second["id"] != profile["id"]
    job = post("/api/agent-workspace/jobs", {"action": "save", "profile_id": profile["id"],
               "job_key": "guide", "title": "A guide", "replacements": {base["id"]: changed["id"]},
               "additions": [extra["id"]]})
    resolved = _call(server.port, "/api/agent-workspace/resolve?profile=" + profile["id"] + "&job=guide",
                     headers=headers)[1]
    assert [row["id"] for row in resolved["documents"]] == [changed["id"], extra["id"]]
    assert "changed" in resolved["preamble"] and "added" in resolved["preamble"]
    assert _call(server.port, "/api/agent-workspace/profile/" + profile["id"],
                 headers={**PUBLIC_XFF, "Cookie": other})[0] == 403
    assert _call(server.port, "/api/agent-workspace/resolve?profile=" + profile["id"],
                 headers={**PUBLIC_XFF, "Cookie": other})[0] == 403
    assert _call(server.port, "/api/agent-workspace/documents", "POST", headers,
                 {"action": "delete", "id": base["id"]})[0] == 409
    assert _call(server.port, "/api/agent-workspace/job/" + job["id"], headers=headers)[1]["job_key"] == "guide"


def test_memory_ids_transfer_between_permanent_accounts(server):
    first = _signup(server.port, "memory_first")
    second = _signup(server.port, "memory_second")
    operator = _signup(server.port, "memory_operator")
    authz.grant_role(server.sessions.accounts_db, "memory_operator", "operator", "test")
    headers = lambda cookie: {**PUBLIC_XFF, "Cookie": cookie, "X-Meowdoku-Workspace": "1"}
    status, memory, _, _ = _call(server.port, "/api/memory", "POST", headers(first),
                                 {"action": "write", "title": "Carry me", "body": "a remembered fact"})
    assert status == 200, memory
    assert len(memory["id"]) == 32
    path = "/api/memory/file/" + memory["id"]
    assert _call(server.port, path, headers=headers(second))[0] == 404
    assert _call(server.port, "/api/memory", "POST", headers(second),
                 {"action": "transfer", "id": memory["id"], "to": "memory_second"})[0] == 403
    status, moved, _, _ = _call(server.port, "/api/memory", "POST", headers(operator),
                                {"action": "transfer", "id": memory["id"], "to": "memory_second"})
    assert status == 200 and moved["id"] == memory["id"]
    assert _call(server.port, path, headers=headers(first))[0] == 404
    assert _call(server.port, path, headers=headers(second))[1]["body"] == "a remembered fact"


def test_imported_briefs_are_idempotent_and_roundtable_is_job_specific(server):
    operator = _signup(server.port, "brief_operator")
    _signup(server.port, "claude")
    _signup(server.port, "codex")
    _signup(server.port, "gemini")
    _signup(server.port, "llama")
    authz.grant_role(server.sessions.accounts_db, "brief_operator", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator}
    first = _call(server.port, "/api/agent-workspace/profiles", headers=headers)[1]["profiles"]
    second = _call(server.port, "/api/agent-workspace/profiles", headers=headers)[1]["profiles"]
    assert len(first) == len(second) == 4
    claude = next(row for row in first if row["individual"] == "claude")
    base = _call(server.port, "/api/agent-workspace/resolve?profile=" + claude["id"], headers=headers)[1]
    table = _call(server.port, "/api/agent-workspace/resolve?profile=" + claude["id"] + "&job=roundtable",
                  headers=headers)[1]
    assert "Every run, first" in base["preamble"]
    assert "Every run, first" not in table["preamble"]
    assert "one seat at a round table" in table["preamble"]
    assert _call(server.port, "/api/agent-workspace/profiles", "POST",
                 {**headers, "X-Meowdoku-Workspace": "1"},
                 {"action": "delete", "id": claude["id"]})[0] == 200
    assert len(_call(server.port, "/api/agent-workspace/profiles", headers=headers)[1]["profiles"]) == 3


def test_profile_launcher_prepares_stable_memory_id(server):
    _signup(server.port, "spawn_person")
    _signup(server.port, "another_person")
    script = ROOT / "examples" / "meowdoku" / "deploy" / "profile_prompt.py"
    memory_id = "a" * 32
    command = [sys.executable, str(script), "--accounts-db", str(server.sessions.accounts_db),
               "--individual", "spawn_person", "--ensure-memory", memory_id]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert subprocess.run([*command[:5], "another_person", *command[6:]], capture_output=True).returncode == 1
    db = work.connect(server.sessions.accounts_db)
    try:
        row = db.execute("SELECT u.id FROM users u WHERE username='spawn_person'").fetchone()
        assert work.memory_read(db, row[0], memory_id)["body"] == ""
    finally:
        db.close()


def test_character_instances_attach_memory_and_llama_batch_claims(server):
    operator = _signup(server.port, "batch_operator")
    llama = _signup(server.port, "llama")
    outsider = _signup(server.port, "batch_outsider")
    authz.grant_role(server.sessions.accounts_db, "batch_operator", "operator", "test")
    headers = lambda cookie: {**PUBLIC_XFF, "Cookie": cookie, "X-Meowdoku-Workspace": "1"}
    def post(path, body, cookie=operator):
        return _call(server.port, path, "POST", headers(cookie), body)
    prompt = post("/api/agent-workspace/documents", {"action": "save", "kind": "prompt",
                  "title": "Llama character", "body": "You are a cartographer."})[1]
    character = post("/api/agent-workspace/profiles", {"action": "save", "title": "Map maker",
                      "individual": "llama", "document_ids": [prompt["id"]]})[1]
    first = post("/api/agent-workspace/instances", {"action": "save", "profile_id": character["id"],
                  "title": "North map maker"})[1]
    second = post("/api/agent-workspace/instances", {"action": "save", "profile_id": character["id"],
                   "title": "South map maker"})[1]
    assert first["memory_id"] != second["memory_id"]
    assert len(_call(server.port, "/api/agent-workspace/instances", headers=headers(operator))[1]["instances"]) == 2
    assert post("/api/agent-workspace/profiles", {"action": "delete", "id": character["id"]})[0] == 409
    assert post("/api/memory", {"action": "transfer", "id": first["memory_id"],
                                "to": "batch_outsider"})[0] == 409
    status, run, _, _ = post("/api/agent-runs", {"instance_id": first["id"], "prompt": "Draw the north coast"})
    assert status == 200 and run["status"] == "queued", run
    assert post("/api/agent-runs/claim", {"limit": 4}, outsider)[0] == 403
    status, claimed, _, _ = post("/api/agent-runs/claim", {"limit": 4}, llama)
    assert status == 200 and len(claimed["runs"]) == 1, claimed
    work_item = claimed["runs"][0]
    assert work_item["instance_id"] == first["id"]
    assert "cartographer" in work_item["preamble"]
    assert work_item["prompt"] == "Draw the north coast"
    assert post("/api/agent-runs/complete", {"id": run["id"], "claim_token": "bad",
                                             "response": "wrong"}, llama)[0] == 409
    status, done, _, _ = post("/api/agent-runs/complete", {"id": run["id"],
                            "claim_token": work_item["claim_token"], "response": "A coastline", "error": ""}, llama)
    assert status == 200 and done["status"] == "done" and done["response"] == "A coastline"
    assert _call(server.port, "/api/agent-runs/" + run["id"], headers=headers(outsider))[0] == 403
    destination = post("/api/agent-workspace/profiles", {"action": "save", "title": "Human cartographer",
                        "individual": "batch_outsider", "document_ids": [prompt["id"]]})[1]
    status, moved, _, _ = post("/api/agent-workspace/instances", {"action": "transfer", "id": first["id"],
                            "target_profile_id": destination["id"]})
    assert status == 200 and moved["id"] == first["id"] and moved["memory_id"] == first["memory_id"]
    assert moved["individual"] == "batch_outsider"
    assert _call(server.port, "/api/memory/file/" + first["memory_id"], headers=headers(outsider))[0] == 200


def test_an_operator_declares_how_a_character_is_run(server):
    """[CAPACITY] case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 2: action capacity on the profiles route."""
    operator = _signup(server.port, "capacity_operator")
    agent = _signup(server.port, "capacity_agent")
    authz.grant_role(server.sessions.accounts_db, "capacity_operator", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator, "X-Meowdoku-Workspace": "1"}
    status, character, _, _ = _call(server.port, "/api/agent-workspace/profiles", "POST", headers,
                                    {"action": "save", "title": "Run me", "individual": "capacity_agent", "document_ids": []})
    assert status == 200 and character["capacity"] is None
    declare = {"action": "capacity", "id": character["id"], "state": "stateless", "context_tokens": 4096,
               "tools": "runner", "session": "none", "note": ""}
    status, saved, _, _ = _call(server.port, "/api/agent-workspace/profiles", "POST", headers, declare)
    assert status == 200 and saved["capacity"]["state"] == "stateless"
    assert _call(server.port, "/api/agent-workspace/profiles", "POST", headers, {**declare, "state": "maybe"})[0] == 400
    assert _call(server.port, "/api/agent-workspace/profiles", "POST",
                 {**PUBLIC_XFF, "Cookie": agent, "X-Meowdoku-Workspace": "1"}, declare)[0] == 403
    mine = _call(server.port, "/api/agent-workspace/orientation", headers={**PUBLIC_XFF, "Cookie": agent})[1]
    assert mine["capacity"]["state"] == "stateless" and mine["preamble"].startswith("You are run stateless")


def test_any_account_walks_its_own_decision_graph_and_only_operators_edit_it(server):
    """[DECISIONS] case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 3."""
    operator = _signup(server.port, "graph_operator")
    walker = _signup(server.port, "graph_walker")
    authz.grant_role(server.sessions.accounts_db, "graph_operator", "operator", "test")
    headers = {**PUBLIC_XFF, "Cookie": operator, "X-Meowdoku-Workspace": "1"}
    mine = {**PUBLIC_XFF, "Cookie": walker, "X-Meowdoku-Workspace": "1"}
    for place in ({"key": "arrived", "title": "Arrived", "start": True, "acts": [{"tool": "pending", "for": "see"}]},
                  {"key": "triage", "title": "Triage"}):
        assert _call(server.port, "/api/agent-workspace/decisions", "POST", headers, {"action": "place", **place})[0] == 200
    route = {"action": "route", "source": "arrived", "target": "triage", "condition": "nothing is waiting"}
    assert _call(server.port, "/api/agent-workspace/decisions", "POST", headers, route)[0] == 200
    assert _call(server.port, "/api/agent-workspace/decisions", "POST", mine, route)[0] == 403, "a walker cannot edit"
    status, found, _, _ = _call(server.port, "/api/agent-workspace/decisions/map", headers=mine)
    assert status == 200 and found["at"] == "decisions:line/arrived" and found["map"]["kind"] == "decisions"
    status, moved, _, _ = _call(server.port, "/api/agent-workspace/decisions/at", "POST", mine, {"at": "decisions:line/triage"})
    assert status == 200 and moved["key"] == "triage"
    assert _call(server.port, "/api/agent-workspace/decisions/at", headers=mine)[1]["key"] == "triage"
    assert _call(server.port, "/api/agent-workspace/decisions/at", "POST", {**PUBLIC_XFF, "Cookie": walker},
                 {"at": "arrived"})[0] == 403, "a write needs the workspace header"
    assert _call(server.port, "/api/agent-workspace/decisions/map", headers=PUBLIC_XFF)[0] == 401
