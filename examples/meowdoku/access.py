"""Access tiers: who is asking, by where they came from, decides what they may see and do.

ORIGIN of a request (``classify``), from the client address and the Host header ONLY.  The client address is the socket peer, unless
the peer is a trusted reverse proxy (proxy_trust.client_origin, loopback by default): then it is the proxy's X-Forwarded-For client.
From any other peer ``X-Forwarded-For`` / ``Forwarded`` / ``X-Real-IP`` are NEVER read: a client can write anything in a header.

    lan   the peer is loopback, private (10/8, 172.16/12, 192.168/16), link-local (169.254/16, fe80::/10), IPv6 ULA (fc00::/7), or the
          server's OWN public IP.  That last one is the NAT hairpin: a phone on the home Wi-Fi that opens the site through the public
          address reaches the router, which loops it back, and the peer shows up as the public IP itself, not as 192.168.x.x.
    dns   not LAN, and the Host header is a hostname (the visitor used the name).
    ip    not LAN, and the Host header is an IP literal (the visitor used the raw address); also a missing Host.

TIER of an origin: ``build/meowdoku/access.json`` (gitignored with build/; read again whenever its mtime changes, so flipping a tier
needs no restart).  A missing or unreadable file means the defaults.

    {
      "policy": {"lan": "edit", "ip": "dev", "dns": "shader"},
      "public_ip": null,
      "public_ip_service": "https://api.ipify.org",
      "public_ip_ttl_seconds": 3600
    }

``policy`` maps an origin to a tier (omitted origins keep the default; an unknown tier name is ignored and logged):

    edit     the owner's tools: studio activation/delete, the lifecycle trigger, loading-animation upload, the process-wide learner
             and Lichess controls, the dev panels and debug flags on the page.
    dev      sees the development options and controls on the page, read-only (the server refuses the owner-only calls).
    play     the normal site.
    shader   a FULLSCREEN SHADER ONLY page (shader.html: the loading panel's animation full window, no controls).  The server answers
             every request with that page, or 403 for APIs (only /api/whoami, a reduced /api/ready and /api/loading_anim are served).

Defaults, in origin order the user's triple (edit / development options / fullscreen shader only): lan -> edit, ip -> dev, dns -> shader.
A bare IP is findable by ordinary developer means, so whoever reaches it sees the development options (read-only: dev cannot call the
owner endpoints, and edit actions need the tier AND their tokens from anywhere); the hostname is the public face and gets the shader only.
Set "dns" to "play" to open the real site to the name, or any origin to any tier.

``public_ip`` is the server's own public address(es) for the hairpin rule: a string or a list of strings.  When it is null the address is
asked of ``public_ip_service`` (default https://api.ipify.org, which answers the caller's address as plain text) once at startup in a
background thread, cached for ``public_ip_ttl_seconds`` and refreshed the same way.  The lookup FAILS SOFT: until it answers (or if it never
does) the hairpin phone is classified "ip", which is "dev" by default; nothing blocks and nothing is guessed.  Set ``public_ip`` to skip the
lookup entirely.

``require_tier(handler, "edit")`` is the one gate the server calls.  A tier is an ADDITIONAL requirement: owner endpoints also keep
their token checks (lifecycle admin token), because a tier says where the caller came from, not that they are the owner.
"""
from __future__ import annotations

import ipaddress
import json
import sys
import threading
import time
import urllib.request
from http import HTTPStatus
from pathlib import Path

ORIGINS = ("lan", "dns", "ip")
TIERS = ("shader", "play", "dev", "edit")
RANK = {"shader": 0, "play": 1, "dev": 2, "edit": 3}
DEFAULT_POLICY = {"lan": "edit", "dns": "shader", "ip": "dev"}
DEFAULT_SERVICE = "https://api.ipify.org"
DEFAULT_TTL = 3600
RETRY_SECONDS = 60

LAN_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "fe80::/10", "fc00::/7"))

# The shader tier's whole world: these files (the page, the panel script that draws the animation, its decoder) and three small APIs.
SHADER_FILES = ("/shader.html", "/loading_panel.js", "/loading_anim.js", "/octet_codec.js", "/access.js")
SHADER_PAGE = "/shader.html"
# Public sign-in: the only other things the shader tier may reach (server.py _shader_request / _login_post).  Sign-up stays closed.
SHADER_LOGIN_FILES = ("/login.html", "/login.js", "/messenger-widget.js", "/site-nav.js")   # the widget renders nothing without the message capability
SHADER_LOGIN_GETS = ("/api/account/me",)
SHADER_LOGIN_POSTS = ("/api/account/login", "/api/account/logout")
# GET routing only: the photo handlers still require photo_read and check ownership.
PHOTO_READ_APIS = ("/api/marble_photo/jobs", "/api/marble_photo/lenses", "/api/marble_photo",
                   "/api/marble_photo/image", "/api/marble_photo/hdr", "/api/marble_photo/field", "/api/marble_photo/animation")
# /api/reviews/standing answers accepted true/false for one commit id and how to get a review; the review-gate
# check on GitHub asks it from outside with no account (branch_review.http_route). It reads no identity and no row's text.
SHADER_APIS = ("/api/whoami", "/api/ready", "/api/loading_anim", "/api/reviews/standing") + PHOTO_READ_APIS
# [GUEST] sockets a visitor with no account may open (server.py _kiosk_gate; the REST entries are declared in speaktome_api.catalog):
# the game table's input socket and a viewer's push socket, and, by prefix, the federation bridge whose peer grant is the authority.
GUEST_SOCKETS = ("/ws", "/ws/spectate")
CREDENTIAL_SOCKETS = ("/ws/contract/",)

# /api/action types that act on the whole process (the shared learner, the Lichess client, the pilot weights), not on the caller's table.
EDIT_ACTIONS = frozenset({"demo_learning", "arena_save", "arena_stop", "arena_resume", "lichess_pause", "lichess_resume", "pilot_arch"})


def parse_ip(text: str):
    """An ``ipaddress`` object for a textual peer address (IPv4-mapped IPv6 unwrapped, zone id dropped), or None."""
    try:
        addr = ipaddress.ip_address(str(text).split("%", 1)[0])
    except ValueError:
        return None
    return addr.ipv4_mapped if getattr(addr, "ipv4_mapped", None) else addr


def host_name(host_header: str | None) -> str:
    """The host part of a Host header with port and IPv6 brackets removed (lower case, no trailing dot)."""
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    if host.count(":") == 1:                       # name:port or v4:port (a bare IPv6 literal has several colons)
        host = host.split(":", 1)[0]
    return host.rstrip(".")


def host_is_ip(host_header: str | None) -> bool:
    """True when the Host is an IP literal.  A name whose last label is all digits (``2130706433``, ``1.2.3``) is no hostname: browsers read
    it as a number, so it counts as an address; so does an empty Host."""
    name = host_name(host_header)
    if not name:
        return True
    if parse_ip(name) is not None:
        return True
    return not any(ch.isalpha() for ch in name.rsplit(".", 1)[-1])


def classify(peer: str, host_header: str | None, public_ips=()) -> str:
    """lan | dns | ip, from the socket peer and the Host header.  Nothing else about the request is consulted."""
    addr = parse_ip(peer)
    if addr is not None:
        if any(addr in net for net in LAN_NETWORKS) or any(addr == p for p in public_ips):
            return "lan"
    return "ip" if host_is_ip(host_header) else "dns"


def tier_of(origin: str, policy: dict) -> str:
    return policy.get(origin) or DEFAULT_POLICY[origin]


def load_config(path: Path | None) -> dict:
    """The file's settings merged over the defaults; bad entries are ignored and said so."""
    out = {"policy": dict(DEFAULT_POLICY), "public_ip": None, "public_ip_service": DEFAULT_SERVICE, "public_ip_ttl_seconds": DEFAULT_TTL}
    if path is None or not path.exists():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as error:
        print(f"[access] {path}: unreadable ({error}); using the defaults", file=sys.stderr, flush=True)
        return out
    for origin, tier in (raw.get("policy") or {}).items():
        if origin in ORIGINS and tier in TIERS:
            out["policy"][origin] = tier
        else:
            print(f"[access] ignoring policy {origin!r}: {tier!r}", file=sys.stderr, flush=True)
    given = raw.get("public_ip")
    if isinstance(given, str):
        given = [given]
    if isinstance(given, list):
        out["public_ip"] = [a for a in (parse_ip(g) for g in given) if a is not None]
    if isinstance(raw.get("public_ip_service"), str) and raw["public_ip_service"]:
        out["public_ip_service"] = raw["public_ip_service"]
    if isinstance(raw.get("public_ip_ttl_seconds"), (int, float)) and raw["public_ip_ttl_seconds"] > 0:
        out["public_ip_ttl_seconds"] = float(raw["public_ip_ttl_seconds"])
    return out


def echo_lookup(url: str, timeout: float = 5.0):
    """Ask a public echo service for our address (plain text body).  Returns an ip object or None; never raises."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "meowdoku-access"}), timeout=timeout) as reply:
            return parse_ip(reply.read(128).decode("ascii", "replace").strip())
    except Exception:  # noqa: BLE001 - fail soft: no network, a dead service, a bad body
        return None


class Access:
    """The policy file and the server's own public address, for one server."""

    def __init__(self, path: Path | None, lookup=echo_lookup, clock=time.time):
        self.path, self.lookup, self.clock = path, lookup, clock
        self._stamp, self._config = None, load_config(None)
        self._lock = threading.Lock()
        self._found, self._found_at, self._tried_at, self._busy = None, 0.0, -1e18, False

    # ---- policy (re-read when the file changes)
    def config(self) -> dict:
        try:
            stamp = self.path.stat().st_mtime_ns if self.path is not None else None
        except OSError:
            stamp = None
        with self._lock:
            if stamp != self._stamp:
                self._stamp, self._config = stamp, load_config(self.path)
            return self._config

    # ---- the public address
    def public_ips(self) -> tuple:
        cfg = self.config()
        if cfg["public_ip"] is not None:
            return tuple(cfg["public_ip"])
        now = self.clock()
        with self._lock:
            stale = self._found is None or now - self._found_at > cfg["public_ip_ttl_seconds"]
            if stale and not self._busy and now - self._tried_at > RETRY_SECONDS:
                self._busy, self._tried_at = True, now
                threading.Thread(target=self._refresh, args=(cfg["public_ip_service"],), name="public-ip", daemon=True).start()
            return (self._found,) if self._found is not None else ()

    def _refresh(self, url: str) -> None:
        found = self.lookup(url)
        with self._lock:
            if found is not None:
                self._found, self._found_at = found, self.clock()
            self._busy = False

    def start(self) -> None:
        """Begin the startup lookup (non-blocking) and say what was chosen."""
        cfg = self.config()
        if cfg["public_ip"] is not None:
            print(f"[access] public IP from config: {[str(a) for a in cfg['public_ip']]}; policy {cfg['policy']}", flush=True)
        else:
            self.public_ips()
            print(f"[access] public IP from the echo service {cfg['public_ip_service']} (cached {cfg['public_ip_ttl_seconds']:.0f}s, fails soft); "
                  f"policy {cfg['policy']}", flush=True)

    # ---- the decision
    def decide(self, peer: str, host_header: str | None) -> tuple[str, str]:
        origin = classify(peer, host_header, self.public_ips())
        return origin, tier_of(origin, self.config()["policy"])


def at_least(tier: str, need: str) -> bool:
    return tier != "shader" and RANK[tier] >= RANK[need]


def require_tier(handler, need: str) -> bool:
    """True when the request's tier reaches ``need``; otherwise a 403 is sent and False returned.  Call it first, keep the token checks after."""
    handler._classify()
    if at_least(handler._tier, need):
        return True
    handler.close_connection = True              # the refused request's body is unread
    handler._set_cookie = []
    handler._json({"error": f"this needs the {need} tier"}, HTTPStatus.FORBIDDEN)
    return False
