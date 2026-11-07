"""Share links and the view-only live stream (the logic behind server.py's /api/share and /api/spectate).

SHARE STORE.  A configured view is a small JSON document the page builds (``web/share.js``).  When it fits in the URL
fragment nothing reaches the server; when it does not, the page POSTs it here and links ``#share=<id>``.  The id is the hash
of the canonical JSON (content addressed: the same view is the same id, a store of it is idempotent), the file lives under
``build/meowdoku/shares/`` (gitignored with the rest of ``build/``), and the size and the count are capped.  Whatever is
stored is scrubbed first: credential-ish keys never persist, whatever the page sent.

SPECTATE.  "Share live view" mints a viewer token for the owner's session.  The token opens exactly one thing: the read
endpoint ``/api/spectate/watch``.  The owner's session cookie (HttpOnly) is the only credential for the owner's table, the
input endpoints and the owner's spectate controls; a request that presents the viewer header to any of those is refused
(``VIEWER_HEADER``), and the token is not a session id, so presenting it as the cookie reaches nothing of the owner's.
Tokens are ``secrets.token_urlsafe``, revoked by the owner (``stop``) and dead when the owner's session expires.  At most
``VIEWER_CAP`` viewers are active per session (a viewer is active while it polled within ``ACTIVE_SECONDS``).
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

VIEWER_HEADER = "X-Meowdoku-Viewer"
SHARE_MAX_BYTES = 64 * 1024                  # one stored view
SHARE_MAX_FILES = 5000                       # the whole store
VIEWER_CAP = 16                              # active viewers per shared session
ACTIVE_SECONDS = 15.0                        # a viewer that has not polled for this long no longer counts
SHARE_TTL_SECONDS = 12 * 3600                # a live share dies at the latest then, with or without its session
STATE_CACHE_SECONDS = 0.25                   # the owner's state is serialised at most this often however many watch it
LIVE_MAX_BYTES = 16 * 1024                   # the owner's pushed page-side inputs
ID_RE = re.compile(r"^[0-9a-f]{20}$")
# Keys the page must never put into a shared view (also dropped here, whatever arrives): credentials, identity, location.
SECRET_KEY_RE = re.compile(r"token|secret|passw|credential|auth|cookie|session|api[_-]?key|lichess|player|geo", re.I)
# The only localStorage keys a view carries: the page's own ``meowdoku.*`` / ``meowdoku_*`` keys.
SHARED_KEY_RE = re.compile(r"^meowdoku[._][A-Za-z0-9_.\-]{1,80}$")
# The owner's table state also carries the owner's profile; a viewer gets the game and none of it.
PRIVATE_STATE_KEYS = ("player", "record", "inventory", "award")


def scrub(view: Any) -> dict[str, Any]:
    """The view with every credential-ish storage key removed and non-shareable keys dropped."""
    if not isinstance(view, dict):
        raise ValueError("a shared view is a JSON object")
    out = dict(view)
    ls = out.get("ls")
    if ls is not None:
        if not isinstance(ls, dict):
            raise ValueError("ls must be an object")
        out["ls"] = {k: v for k, v in ls.items()
                     if isinstance(k, str) and isinstance(v, str) and SHARED_KEY_RE.match(k) and not SECRET_KEY_RE.search(k)}
    return out


def canonical(view: dict[str, Any]) -> str:
    return json.dumps(view, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ShareError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class ShareStore:
    """Content-addressed, size-capped store of shared views."""

    def __init__(self, directory: Path, max_bytes: int = SHARE_MAX_BYTES, max_files: int = SHARE_MAX_FILES):
        self.dir, self.max_bytes, self.max_files = Path(directory), max_bytes, max_files
        self.lock = threading.Lock()

    def put(self, view: Any) -> str:
        text = canonical(scrub(view))
        data = text.encode("utf-8")
        if len(data) > self.max_bytes:
            raise ShareError(f"this view is {len(data)} bytes; the limit is {self.max_bytes}", 413)
        sid = hashlib.sha256(data).hexdigest()[:20]
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{sid}.json"
            if not path.exists():
                if sum(1 for _ in self.dir.glob("*.json")) >= self.max_files:
                    raise ShareError("the share store is full", 507)
                path.write_bytes(data)
        return sid

    def get(self, sid: str) -> dict[str, Any]:
        if not ID_RE.match(sid or ""):
            raise ShareError("unknown share", 404)
        path = self.dir / f"{sid}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            raise ShareError("unknown share", 404) from None


class _Share:
    def __init__(self, owner: str, token: str, now: float):
        self.owner, self.token, self.created = owner, token, now
        self.live: dict[str, Any] = {}
        self.live_rev = 0
        self.viewers: dict[str, dict[str, float]] = {}      # vid -> {seen, since, bytes}
        self.state_text = ""
        self.state_rev = ""
        self.state_at = -1e9


class Spectate:
    """owner session token -> its one live share; viewer token -> that share."""

    def __init__(self, cap: int = VIEWER_CAP, clock: Callable[[], float] = time.time):
        self.cap, self.clock = cap, clock
        self.lock = threading.Lock()
        self.by_token: dict[str, _Share] = {}
        self.by_owner: dict[str, _Share] = {}

    # -- the owner's side
    def start(self, owner: str) -> str:
        with self.lock:
            share = self.by_owner.get(owner)
            if share is None:
                share = _Share(owner, secrets.token_urlsafe(24), self.clock())
                self.by_owner[owner], self.by_token[share.token] = share, share
            return share.token

    def stop(self, owner: str) -> bool:
        with self.lock:
            return self._drop(self.by_owner.get(owner))

    drop_owner = stop            # the owner's session ended

    def _drop(self, share: _Share | None) -> bool:
        if share is None:
            return False
        self.by_token.pop(share.token, None)
        self.by_owner.pop(share.owner, None)
        return True

    def push_live(self, owner: str, live: Any) -> int:
        if not isinstance(live, dict):
            raise ShareError("live must be an object")
        live = scrub(live)
        if len(canonical(live)) > LIVE_MAX_BYTES:
            raise ShareError("the live inputs are too large", 413)
        with self.lock:
            share = self.by_owner.get(owner)
            if share is None:
                raise ShareError("not sharing", 409)
            if live != share.live:
                share.live, share.live_rev = live, share.live_rev + 1
            return share.live_rev

    def status(self, owner: str) -> dict[str, Any]:
        with self.lock:
            share = self.by_owner.get(owner)
            if share is None:
                return {"sharing": False}
            now = self.clock()
            self._prune(share, now)
            return {"sharing": True, "token": share.token, "cap": self.cap, "viewers": len(share.viewers),
                    "per_viewer": [self._viewer_stats(v, now) for v in share.viewers.values()]}

    # -- the viewer's side
    def _prune(self, share: _Share, now: float) -> None:
        for vid in [k for k, v in share.viewers.items() if now - v["seen"] > ACTIVE_SECONDS]:
            del share.viewers[vid]

    @staticmethod
    def _viewer_stats(v: dict[str, float], now: float) -> dict[str, float]:
        span = max(now - v["since"], 1e-6)
        return {"bytes": int(v["bytes"]), "seconds": round(span, 2), "bytes_per_s": round(v["bytes"] / span, 1)}

    def watch(self, token: str, vid: str | None, have_rev: str | None, state_fn: Callable[[], dict[str, Any]],
              owner_alive: Callable[[str], bool] = lambda owner: True, have_live: int | None = None) -> dict[str, Any]:
        """One poll.  ``state_fn`` returns the owner's table state (already free of private keys); the full state is sent
        only when its revision differs from ``have_rev``, the pushed page-side inputs only when ``have_live`` is not the current one."""
        with self.lock:
            share = self.by_token.get(token or "")
            now = self.clock()
            if share is None:
                raise ShareError("this live view is not shared (any more)", 410)
            if now - share.created > SHARE_TTL_SECONDS or not owner_alive(share.owner):
                self._drop(share)
                raise ShareError("the shared session has ended", 410)
            self._prune(share, now)
            viewer = share.viewers.get(vid or "")
            if viewer is None:
                if len(share.viewers) >= self.cap:
                    raise ShareError(f"this live view already has {self.cap} viewers", 429)
                vid = secrets.token_urlsafe(9)
                viewer = share.viewers[vid] = {"seen": now, "since": now, "bytes": 0.0}
            viewer["seen"] = now
            refresh = now - share.state_at > STATE_CACHE_SECONDS
        if refresh:                     # outside the registry lock: the owner's table lock may be held by a tick
            text = json.dumps(state_fn(), separators=(",", ":"))
            with self.lock:
                if now - share.state_at > STATE_CACHE_SECONDS:
                    rev = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                    share.state_text, share.state_at = text, now
                    share.state_rev = rev
        with self.lock:
            out: dict[str, Any] = {"vid": vid, "live_rev": share.live_rev, "state_rev": share.state_rev, "viewers": len(share.viewers)}
            if have_live != share.live_rev:
                out["live"] = share.live
            if have_rev != share.state_rev:
                out["state"] = json.loads(share.state_text)
            return out

    def account(self, token: str, vid: str, nbytes: int) -> None:
        with self.lock:
            share = self.by_token.get(token)
            viewer = share.viewers.get(vid) if share else None
            if viewer is not None:
                viewer["bytes"] += nbytes

    def viewer_stats(self, token: str, vid: str) -> dict[str, float] | None:
        with self.lock:
            share = self.by_token.get(token)
            viewer = share.viewers.get(vid) if share else None
            return self._viewer_stats(viewer, self.clock()) if viewer else None


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    """The owner's state as a viewer may see it: the game, without the owner's profile."""
    return {k: v for k, v in state.items() if k not in PRIVATE_STATE_KEYS}
