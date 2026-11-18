"""[ACCESS] A client's own Host header never raises its tier (case d56a1d9d843b447c87aa607a78e36069, 2026-11-17).

Before: a visitor that was not on the LAN and wrote an address (127.0.0.1, or any other) in its Host header, or no Host at all,
was origin "ip", whose default tier was dev (dev_ui). Apache passes the Host through, so anyone could ask for the dev tier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import access as A  # noqa: E402
from meowdoku import authz  # noqa: E402

PUBLIC = A.parse_ip("173.195.26.104")
HOSTS = ["127.0.0.1", "127.0.0.1:8766", "localhost", "[::1]", "10.0.0.5", "173.195.26.104", "2130706433", None, "",
         "stichka.duckdns.org"]


@pytest.mark.parametrize("host", HOSTS)
def test_a_remote_peer_gets_the_public_tier_whatever_its_host_says(tmp_path, host):
    access = A.Access(tmp_path / "access.json")
    access._found, access._found_at = PUBLIC, 1e12
    origin, tier = access.decide("8.8.8.8", host)
    assert origin in ("ip", "dns") and tier == "shader"
    principal = authz.Principal("member", 5, "someone", origin, tier)
    assert "dev_ui" not in authz.capabilities(principal, authz.DEFAULT_POLICY)


def test_a_policy_cannot_set_the_bare_address_above_the_public_name(tmp_path):
    path = tmp_path / "access.json"
    path.write_text(json.dumps({"policy": {"ip": "dev", "dns": "play"}}), encoding="utf-8")
    access = A.Access(path)
    assert access.decide("8.8.8.8", "127.0.0.1") == ("ip", "play"), "held to the name's tier"
    assert access.decide("8.8.8.8", "stichka.duckdns.org") == ("dns", "play")
    path.write_text(json.dumps({"policy": {"ip": "dev", "dns": "dev"}}), encoding="utf-8")
    assert access.decide("8.8.8.8", "127.0.0.1") == ("ip", "dev"), "only by raising the public name with it"


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "192.168.1.20", "173.195.26.104"])
def test_the_lan_comes_only_from_the_peer(tmp_path, peer):
    access = A.Access(tmp_path / "access.json")
    access._found, access._found_at = PUBLIC, 1e12
    assert access.decide(peer, "anything.example.org") == ("lan", "edit")
