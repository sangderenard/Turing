"""Trusted-proxy handling for running Meowdoku behind Apache (see deploy/APACHE_FRONT_2026-11-10.md).

STATUS: standalone module; NOT yet wired into server.py / access.py / lifecycle.py (those are owned by other work).

``client_origin(peer_ip, headers) -> (client_ip, scheme, via_proxy)``

* peer NOT in TRUSTED_PROXIES: every forwarding header is ignored, client_ip = peer_ip, scheme "http", via_proxy False.
  A header from an untrusted peer is never believed (today's stance: a client can write anything in a header).
* peer trusted AND X-Forwarded-For present: client = the RIGHTMOST address in X-Forwarded-For that is not itself a trusted proxy
  (the last untrusted hop; everything left of it is client-supplied and unverified).  scheme from X-Forwarded-Proto
  ("https" or "http", else "http").  via_proxy True.
* peer trusted, no X-Forwarded-For: client = the peer itself, via_proxy False (a local direct caller), scheme "http".
  If every hop in X-Forwarded-For is trusted, the leftmost is used.

TRUSTED_PROXIES defaults to {"127.0.0.1", "::1"}; ``load_trusted_proxies(path)`` reads a JSON list (or {"trusted_proxies": [...]})
from build/meowdoku/proxy_trust.json.  A missing or unreadable file means the default.

``route_cookie(instance_route)`` gives the ``meowdoku_route=<route>`` Set-Cookie value matching Apache's
``stickysession=MEOWDOKU_ROUTE|meowdoku_route``.

INTEGRATION SPEC (pending in the live tree):
  * access.classify receives ``client_ip`` instead of the socket peer when ``via_proxy`` is True (Host header handling unchanged).
  * server.py's Secure-cookie decision (``_is_secure``, now TLS or X-Forwarded-Proto: https) uses ``scheme`` from client_origin, so
    the header is honoured only from a trusted peer.
  * lifecycle.admin_authorize and the loading-library owner path KEEP refusing any request carrying forwarding headers.  Admin stays
    loopback-direct, i.e. NOT through Apache (the vhost also excludes /api/admin from ProxyPass).
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

DEFAULT_TRUSTED_PROXIES = frozenset({"127.0.0.1", "::1"})
TRUSTED_PROXIES = set(DEFAULT_TRUSTED_PROXIES)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "build" / "meowdoku" / "proxy_trust.json"
ROUTE_COOKIE = "meowdoku_route"


def _norm(ip: str) -> str:
    ip = (ip or "").strip().strip("[]")
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return str(a)


def load_trusted_proxies(path: Optional[Path] = None) -> set:
    """Read the trusted set from JSON; missing/invalid file -> the default.  Replaces TRUSTED_PROXIES in place."""
    p = Path(path) if path is not None else CONFIG_PATH
    result = set(DEFAULT_TRUSTED_PROXIES)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("trusted_proxies")
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            result = {_norm(x) for x in data}
    except (OSError, ValueError):
        pass
    TRUSTED_PROXIES.clear()
    TRUSTED_PROXIES.update(result)
    return TRUSTED_PROXIES


def _header(headers: Mapping[str, str], name: str) -> str:
    getter = getattr(headers, "get", None)
    v = getter(name) if getter else None
    if v is None:
        low = name.lower()
        for k in headers:
            if str(k).lower() == low:
                return headers[k]
    return v or ""


def client_origin(peer_ip: str, headers: Mapping[str, str],
                  trusted: Optional[Iterable[str]] = None) -> Tuple[str, str, bool]:
    peer = _norm(peer_ip)
    trust = {_norm(x) for x in (TRUSTED_PROXIES if trusted is None else trusted)}
    if peer not in trust:
        return peer, "http", False
    xff = _header(headers, "X-Forwarded-For")
    hops = [_norm(h) for h in xff.split(",") if h.strip()]
    if not hops:
        return peer, "http", False
    client = next((h for h in reversed(hops) if h not in trust), hops[0])
    proto = _header(headers, "X-Forwarded-Proto").split(",")[0].strip().lower()
    return client, ("https" if proto == "https" else "http"), True


def route_cookie(instance_route: str) -> str:
    route = str(instance_route)
    if not route or any(c in route for c in ' ;,="\r\n'):
        raise ValueError(f"invalid route name: {route!r}")
    return f"{ROUTE_COOKIE}={route}; Path=/; HttpOnly; SameSite=Lax"
