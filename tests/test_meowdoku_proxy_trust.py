"""Trusted-proxy origin resolution (examples/meowdoku/proxy_trust.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import proxy_trust as pt  # noqa: E402


def test_direct_peer_spoofed_header_ignored():
    h = {"X-Forwarded-For": "10.0.0.1", "X-Forwarded-Proto": "https"}
    assert pt.client_origin("198.51.100.7", h) == ("198.51.100.7", "http", False)


def test_trusted_loopback_uses_forwarded_client():
    h = {"X-Forwarded-For": "203.0.113.9, 127.0.0.1", "X-Forwarded-Proto": "https"}
    assert pt.client_origin("127.0.0.1", h) == ("203.0.113.9", "https", True)


def test_chain_picks_rightmost_untrusted():
    h = {"X-Forwarded-For": "1.1.1.1, 203.0.113.9, 198.51.100.2, 127.0.0.1"}
    assert pt.client_origin("::1", h)[0] == "198.51.100.2"
    assert pt.client_origin("127.0.0.1", h, trusted={"127.0.0.1", "198.51.100.2"})[0] == "203.0.113.9"


def test_missing_header_from_trusted_peer():
    assert pt.client_origin("127.0.0.1", {}) == ("127.0.0.1", "http", False)


def test_scheme_defaults_http():
    assert pt.client_origin("127.0.0.1", {"X-Forwarded-For": "203.0.113.9"}) == ("203.0.113.9", "http", True)


def test_route_cookie_format():
    assert pt.route_cookie("a").startswith("meowdoku_route=a;")
    with pytest.raises(ValueError):
        pt.route_cookie("a;b")


def test_load_trusted_proxies(tmp_path):
    p = tmp_path / "proxy_trust.json"
    p.write_text(json.dumps({"trusted_proxies": ["10.1.1.1"]}))
    try:
        assert pt.load_trusted_proxies(p) == {"10.1.1.1"}
        assert pt.load_trusted_proxies(tmp_path / "none.json") == {"127.0.0.1", "::1"}
    finally:
        pt.load_trusted_proxies(tmp_path / "none.json")
