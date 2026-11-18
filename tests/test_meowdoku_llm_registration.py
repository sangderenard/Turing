"""[LLM REGISTRATION] An LLM agent registers itself as a member, briefed on its credentials (owner, 2026-11-17).

People keep the alpha request; an outside visitor is simulated as in test_meowdoku_public_login (the loopback peer is a trusted
proxy forwarding X-Forwarded-For)."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import authz  # noqa: E402
from meowdoku import llm_registration as L  # noqa: E402
from test_meowdoku_public_login import OUTSIDE, _call, server  # noqa: E402,F401

WEB = ROOT / "examples" / "meowdoku" / "web"


def _llm(name, **change):
    body = {"username": name, "password": "a long enough password", "register_as": "llm",
            "alpha_terms_version": __import__("meowdoku.accounts").accounts.ALPHA_TERMS_VERSION,
            "agree_mit": True, "agree_observation": True, "agree_research": True, "agree_credentials": True}
    body.update(change)
    return body


def test_an_llm_registers_as_a_member_and_is_handed_the_brief(server):
    status, _, cookies, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE, _llm("visiting_agent"))
    assert status == 201, body
    assert body["llm"]["role"] == "member" and body["llm"]["brief"] == L.BRIEF
    db = server.sessions.accounts_db
    assert authz.role_of(db, body["account"]["id"]) == "member"
    assert L.is_llm(db, body["account"]["id"])
    assert any(c.startswith("meowdoku_account=") for c in cookies), "it is signed in, like a person who signs up"


def test_a_person_still_gets_the_alpha_account(server):
    person = _llm("a_person", register_as="human")
    person.pop("agree_credentials")
    status, _, _, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE, person)
    assert status == 201 and "llm" not in body
    assert not L.is_llm(server.sessions.accounts_db, body["account"]["id"])


def test_an_llm_must_accept_the_terms_and_the_credentials(server):
    for change in ({"agree_credentials": False}, {"agree_research": False}, {"alpha_terms_version": "old"},
                   {"register_as": "robot"}):
        status, _, _, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE, _llm("refused_agent", **change))
        assert status == 400, (change, body)


def test_registration_is_capped_per_day(server, monkeypatch):
    monkeypatch.setattr(L, "MAX_PER_DAY", 2)
    assert _call(server.port, "/api/account/signup", "POST", OUTSIDE, _llm("first_agent"))[0] == 201
    assert _call(server.port, "/api/account/signup", "POST", OUTSIDE, _llm("second_agent"))[0] == 201
    status, _, _, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE, _llm("third_agent"))
    assert status == 429 and "full for today" in body["error"]


def test_the_page_offers_it_and_carries_the_same_brief_and_terms():
    page = (WEB / "register.html").read_text(encoding="utf-8")
    assert 'disabled title="General registration is not open"' not in page and '<a href="#llm">Register as an LLM</a>' in page
    shown = re.search(r'<pre id="llm-brief"[^>]*>(.*?)</pre>', page, re.S).group(1)
    assert html.unescape(shown) == L.BRIEF
    version = __import__("meowdoku.accounts").accounts.ALPHA_TERMS_VERSION
    assert f'"alpha_terms_version": "{version}"' in page and 'data-terms-version="' + version + '"' in page
    assert 'register_as: "llm"' in (WEB / "register.js").read_text(encoding="utf-8")
