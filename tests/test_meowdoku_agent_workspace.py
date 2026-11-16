"""Shared operator tree and private agent scratch, over temporary state only."""
from __future__ import annotations

import sys
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
