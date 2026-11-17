"""[PRIVATE] Private values are tokens wherever they are shown; only operators and admins (private_read) read what a token stands for."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import access as A  # noqa: E402
from meowdoku import authz  # noqa: E402
from meowdoku import logring as L  # noqa: E402
from meowdoku import private_record as P  # noqa: E402
from meowdoku import server as srv  # noqa: E402
from meowdoku.secure_records import RecordCipher  # noqa: E402


@pytest.fixture
def record(tmp_path, monkeypatch):
    rec = P.Record(P.load_key(tmp_path / "secrets" / "private_hash.key"), tmp_path / "sql" / "private_record.db",
                   RecordCipher(tmp_path / "secrets" / "records.key"))
    monkeypatch.setattr(P, "RECORD", rec)
    return rec


def test_a_token_is_keyed_stable_and_recorded_on_the_server(tmp_path, record):
    token = record.token("ip", "203.0.113.7")
    assert token.startswith("ip~") and len(token) == 15 and "203" not in token
    assert record.token("ip", "203.0.113.7") == token, "the same value is the same visitor"
    other = P.Record(P.load_key(tmp_path / "other.key"))
    assert other.token("ip", "203.0.113.7") != token, "another state's key gives another token"
    assert P.load_key(tmp_path / "secrets" / "private_hash.key") == record.key, "the key is made once, never replaced"
    got = record.resolve([token, "ip~000000000000", "203.0.113.7"])
    assert list(got) == [token] and got[token]["value"] == "203.0.113.7" and got[token]["seen"] == 2
    assert record.token("ip", "-") == "-" and record.token("ua", "") == "-"
    assert b"203.0.113.7" not in record.db_path.read_bytes(), "the record holds the value sealed"


def test_an_unconfigured_module_still_hashes_and_records_nothing():
    rec = P.Record(b"k" * 32)
    token = rec.token("email", "cat@example.org")
    assert token.startswith("email~") and rec.resolve([token]) == {}


def test_redact_replaces_addresses_and_leaves_ordinary_text(record):
    text = ("ok user=cat client=198.51.100.23 via ::1 and 2001:db8::8a2e:370:7334 mail cat.owner+x@example.co.uk "
            "at 12:30:45, std::vector, Chrome/120.0.6099.109, v1.2, 999.1.1.1, seq=10.5")
    out = record.redact(text)
    for raw in ("198.51.100.23", "::1", "2001:db8::8a2e:370:7334", "cat.owner+x@example.co.uk"):
        assert raw not in out, raw
        assert (record.token("email", raw) if "@" in raw else record.token("ip", raw)) in out
    for kept in ("12:30:45", "std::vector", "Chrome/120.0.6099.109", "v1.2", "999.1.1.1", "seq=10.5"):
        assert kept in out, kept
    assert record.redact(out) == out, "a token passes through unchanged"
    assert record.redact_json({"peer": "10.0.0.5", "n": 3, "log": [{"text": "x 10.0.0.5"}]}) == {
        "peer": record.token("ip", "10.0.0.5"), "n": 3, "log": [{"text": f"x {record.token('ip', '10.0.0.5')}"}]}


def test_the_ring_never_holds_a_raw_address_even_from_a_legacy_file(tmp_path, record):
    legacy = tmp_path / "ring.jsonl"
    legacy.write_text(json.dumps({"seq": 1, "ts": 1.0, "src": "login", "text": "ok user=cat client=192.0.2.9"}) + "\n",
                      encoding="utf-8")
    ring = L.Ring(cap_bytes=10000, path=legacy, key_path=tmp_path / "secrets" / "records.key")
    ring.add("[client-error] 192.0.2.10: boom")
    texts = [e["text"] for e in ring.fetch(0)["entries"]]
    assert texts == [f"ok user=cat client={record.token('ip', '192.0.2.9')}",
                     f"[client-error] {record.token('ip', '192.0.2.10')}: boom"]


def P_(role, tier="edit", origin="lan", user_id=7):
    return authz.Principal(role=role, user_id=user_id, username="x", origin=origin, tier=tier)


def test_private_read_is_the_operator_and_admin_roles_and_nothing_else():
    policy = authz.DEFAULT_POLICY
    for role in ("operator", "admin"):
        for tier in authz.TIERS:
            assert "private_read" in authz.capabilities(P_(role, tier), policy), (role, tier)
    for role in ("dev", "member", "service", "intern", "guest"):
        for tier in authz.TIERS:
            assert "private_read" not in authz.capabilities(P_(role, tier), policy), (role, tier)
    generous = {"tier_caps": {t: list(authz.CAPS) for t in authz.TIERS}, "role_caps": {r: list(authz.CAPS) for r in authz.ROLES}}
    assert "private_read" not in authz.capabilities(P_("dev"), generous), "no policy file confers it on a developer"
    assert "private_read" not in authz.capabilities(P_("guest", user_id=None), generous)


def _server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.access = A.Access(None)
    sessions.access.config()["policy"].update({"lan": "edit"})          # the most generous origin tier
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cookies = {}
    for name, role in (("dev_cat", "dev"), ("op_cat", "operator")):
        uid = sessions.accounts.ensure_user(name, "correcthorse1")
        if role == "dev":                                                  # dev is a permission group on a member account
            conn = authz._connect(sessions.accounts_db)
            conn.execute("INSERT INTO account_permission_grants(user_id,group_name,granted_by,created_at) VALUES(?,?,?,?)",
                         (uid, "dev", 0, 0.0))
            conn.commit(); conn.close()
        else:
            authz.grant_role(sessions.accounts_db, name, role, "test")
        cookies[role] = {"Cookie": f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(uid)}"}
    return httpd, sessions, cookies


def _get(httpd, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    return r.status, r.read()


def test_a_developer_reads_tokens_and_cannot_resolve_them(tmp_path, record, monkeypatch):
    monkeypatch.setattr(L, "RING", L.Ring(cap_bytes=10000))
    httpd, sessions, cookies = _server(tmp_path)
    try:
        L.note_login("ok", "cat", "198.51.100.40")
        sessions.tickets.path.mkdir(parents=True, exist_ok=True)              # a ticket filed before tokens existed
        (sessions.tickets.path / "1-abcdef0123.json").write_text(json.dumps({
            "id": "1-abcdef0123", "created": 1, "updated": 1, "status": "open", "peer": "198.51.100.41", "tier": "dev",
            "log_seq": 1, "server_log": [{"text": "ok user=cat client=198.51.100.42"}], "report": {"note": "mine is a@b.org"}}),
            encoding="utf-8")
        status, body = _get(httpd, "/api/log", cookies["dev"])
        assert status == 403 and b"198.51.100." not in body, "the server log is for operators and admins"
        for path in ("/api/tickets", "/api/tickets?id=1-abcdef0123"):
            status, body = _get(httpd, path, cookies["dev"])
            assert status == 200, (path, body)
            assert b"198.51.100." not in body and b"a@b.org" not in body, (path, body)
            assert b"ip~" in body, path
        token = record.token("ip", "198.51.100.40")
        status, body = _get(httpd, f"/api/private/resolve?tokens={token}", cookies["dev"])
        assert status == 403 and b"private_read" in body and b"198.51.100." not in body
        status, body = _get(httpd, f"/api/private/resolve?tokens={token}", cookies["operator"])
        assert status == 403 and b"HTTPS required" in body, "an operator passes the capability gate; the address needs HTTPS"
    finally:
        httpd.shutdown(); httpd.server_close()
