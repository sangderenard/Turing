"""Server lifecycle: announce a restart, save every session on the way down, come back in the same configuration.

    python -m meowdoku.lifecycle restart --in 60      # announce: pages count down, then the server saves and exits (75)
    python -m meowdoku.lifecycle cancel | status      # [--port 8766]
    python -m meowdoku.lifecycle supervise            # parent loop: spawn server.py from launch.json, relaunch it

THE FLOW.  ``restart`` posts to ``/api/admin/lifecycle`` (localhost callers only, and the token in
``<state>/admin_token.txt``, which is never printed).  Pages learn of it from ``GET /api/lifecycle`` (web/lifecycle.js shows the
countdown).  At T-0 the server writes every live session to ``<state>/sessions/<sha256(cookie)>.json`` and exits 75; the
supervisor relaunches the argv/env recorded in ``<state>/launch.json``; a request that presents a cookie whose hash has a file
rebuilds that session's Table lazily (so the restart is not slower for sessions nobody uses).  The page keeps its own copy of the
last dump (localStorage ``meowdoku.session_copy``) and re-seeds a server that lost the file through ``POST /api/lifecycle/restore``.

WHAT A TABLE NEEDS TO RESTORE (``dump_table`` / ``restore_table``; the dump carries no cookie, player or token):
  exact  meowdoku (board id, palette, cats, X marks, mice, hearts, moves, hints, power-up stock; replayed through the dt-system law),
         wordle (seed, dictionary, length, guesses, entry), chess / checkers (pieces, turn, castling rights, en passant),
         fountain / fountain3d (the server-side configuration, replayed through the game's own ``do`` actions; particles live on the page)
  fresh  every real-time entity game (the dt-system span has no clean dump): a new game, same controls (autopilot, hero, aspect);
         chess_vs_net / chess_arena / lichess (a network search, a learner, a remote game): a new game
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATE_DIR = ROOT / "build" / "meowdoku"
DEFAULT_PORT = 8766
EXIT_RESTART = 75                      # EX_TEMPFAIL: a scheduled exit the supervisor relaunches at once
DUMP_VERSION = 1
DUMP_MAX_BYTES = 64 * 1024
AUTOSAVE_SECONDS = 30.0
SESSION_FILE_TTL = 24 * 3600
MAX_ANNOUNCE_SECONDS = 3600
SECRET_KEY_RE = re.compile(r"token|secret|passw|credential|auth|cookie|session|api[_-]?key|lichess|player|geo", re.I)
ENV_KEEP_RE = re.compile(r"^(MEOWDOKU_|TURING_|NODUS_|LLVM_|PYTHONPATH$|PYTHONUTF8$|PYTHONIOENCODING$)")
ENV_SECRET_RE = re.compile(r"token|secret|passw|credential|api[_-]?key", re.I)
DUMP_KEYS = {"v", "kind", "mode", "new", "game", "chess_clock"}
EXACT_GAMES = ("meowdoku", "wordle", "chess", "checkers", "fountain", "fountain3d")
SAFE_WORD = re.compile(r"^[A-Za-z0-9_.\- ]{1,40}$")


class LifecycleError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- paths and identity
def state_paths(state_dir: Path, port: int) -> dict[str, Path]:
    """The default port keeps the plain names; any other port gets its own files so servers on one machine never meet."""
    suffix = "" if port == DEFAULT_PORT else f"_{port}"
    return {"sessions": state_dir / f"sessions{suffix}", "token": state_dir / f"admin_token{suffix}.txt",
            "launch": state_dir / f"launch{suffix}.json"}


def cookie_hash(cookie: str) -> str:
    """The file name of a session: the raw cookie never reaches the disk."""
    return hashlib.sha256(cookie.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strip_secrets(value: Any) -> Any:
    """Recursively drop credential-ish keys (the rule share.py applies to a shared view)."""
    if isinstance(value, dict):
        return {k: strip_secrets(v) for k, v in value.items() if isinstance(k, str) and not SECRET_KEY_RE.search(k)}
    if isinstance(value, list):
        return [strip_secrets(v) for v in value]
    return value


def _int(value: Any, lo: int, hi: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or not lo <= int(value) <= hi:
        raise LifecycleError(f"{what} must be an integer in {lo}..{hi}")
    return int(value)


def _num(value: Any, lo: float, hi: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
        raise LifecycleError(f"{what} must be a number in {lo}..{hi}")
    return float(value)


def _ints(values: Any, lo: int, hi: int, what: str, limit: int = 4096) -> list[int]:
    if not isinstance(values, list) or len(values) > limit:
        raise LifecycleError(f"{what} must be a list of at most {limit}")
    return [_int(v, lo, hi, what) for v in values]


# --------------------------------------------------------------------------- dump
def _new_payload(table) -> dict[str, Any]:
    """The /api/new payload that rebuilds this game, reduced to the keys it may carry."""
    src = getattr(table, "new_payload", None) or {"game": getattr(table.game, "game", "")}
    out: dict[str, Any] = {"game": str(src.get("game", ""))}
    for key, kind in (("id", str), ("palette", str), ("dictionary", str), ("clock", str)):
        if isinstance(src.get(key), str) and SAFE_WORD.match(src[key]):
            out[key] = src[key]
    for key in ("seed", "length", "parallel"):
        if isinstance(src.get(key), (int, float)) and not isinstance(src.get(key), bool):
            out[key] = int(src[key])
    if isinstance(src.get("aspect"), (int, float)) and 0.0 <= float(src["aspect"]) <= 10.0:
        out["aspect"] = float(src["aspect"])
    return out


def dump_table(table) -> dict[str, Any] | None:
    """The restorable state of one session's Table, or None when it has no game.  Call under ``table.lock``."""
    game = table.game
    if game is None:
        return None
    kind = str(getattr(game, "game", ""))
    dump: dict[str, Any] = {"v": DUMP_VERSION, "kind": kind, "mode": "exact" if kind in EXACT_GAMES else "fresh",
                            "new": _new_payload(table), "game": {}, "chess_clock": getattr(table, "chess_clock", None)}
    if kind == "meowdoku":
        s = game.session
        cat, mark = [int(i) for i in (s.cat.reshape(-1) > 0).nonzero()[0]], [int(i) for i in (s.x_mark.reshape(-1) > 0).nonzero()[0]]
        dump["game"] = {"cats": cat, "marks": mark, "mice": sorted(int(i) for i in game.mouse_cells), "hearts": int(s.hearts),
                        "moves": int(game.moves), "hints_used": int(game.hints_used), "stock": {k: int(v) for k, v in game.stock.items()},
                        "bank_start": {k: int(v) for k, v in table.bank_start.items()},
                        "free_start": {k: int(v) for k, v in table.free_start.items()}}
    elif kind == "wordle":
        dump["game"] = {"guesses": list(game.guesses), "entry": game.entry, "gave_up": bool(game.gave_up)}
    elif kind in ("chess", "checkers"):
        dump["game"] = {"pieces": [[int(c), k, side, game.ids.get(c, "")] for c, (k, side) in sorted(game.pieces.items())],
                        "turn": game.turn, "moves": int(game.moves),
                        "rights": sorted(int(i) for i in game.rights) if hasattr(game, "rights") else None,
                        "ep": game.ep if hasattr(game, "ep") else None}
    elif kind in ("fountain", "fountain3d"):
        image = None
        if getattr(game, "image", None) is not None and game.image.size <= 16 * 1024:
            import base64
            image = base64.b64encode(game.image.astype("uint8").tobytes()).decode("ascii")
        dump["game"] = {"size": game.size, "mix": game.mix, "mode": game.mode, "speed": game.speed, "strata": game.strata,
                        "params": {k: float(v) for k, v in game.params.items()}, "shape": game.shape, "fluid": game.fluid,
                        "inlet": getattr(game, "inlet", None), "run": game.run, "aspect": game.aspect, "phase": game.phase,
                        "epoch": int(game.epoch), "has_image": bool(game.has_image), "image": image}
    else:
        dump["game"] = {"autopilot": bool(getattr(game, "autopilot", False)), "hero": getattr(game, "hero", None)}
    return dump


def dump_rev(dump: dict[str, Any]) -> str:
    return hashlib.sha1(canonical(dump).encode("utf-8")).hexdigest()[:12]


def validate_dump(dump: Any) -> dict[str, Any]:
    """A dump from the page (or the disk) as the server will accept it: bounded, versioned, no credential-ish keys, no strangers."""
    if not isinstance(dump, dict):
        raise LifecycleError("a session copy is a JSON object")
    if len(canonical(dump)) > DUMP_MAX_BYTES:
        raise LifecycleError("this session copy is too large", 413)
    clean = {k: v for k, v in strip_secrets(dump).items() if k in DUMP_KEYS}
    if clean.get("v") != DUMP_VERSION:
        raise LifecycleError("unknown session copy version")
    if not isinstance(clean.get("kind"), str) or not re.match(r"^[a-z0-9_]{1,32}$", clean["kind"]):
        raise LifecycleError("bad game kind")
    if clean.get("mode") not in ("exact", "fresh") or not isinstance(clean.get("new"), dict) or not isinstance(clean.get("game"), dict):
        raise LifecycleError("malformed session copy")
    return clean


# --------------------------------------------------------------------------- restore
def restore_table(table, dump: Any) -> str:
    """Rebuild ``table``'s game from a dump (validated here, whoever sent it).  Returns "exact" or "fresh"."""
    dump = validate_dump(dump)
    new, data, kind = dump["new"], dump["game"], dump["kind"]
    if new.get("game") != kind:
        raise LifecycleError("the copy's game does not match its own start")
    with table.lock:
        if isinstance(dump.get("chess_clock"), str) and SAFE_WORD.match(dump["chess_clock"]):
            table.chess_clock = dump["chess_clock"]
        try:
            table.new(_new_payload_from(new))
        except (KeyError, ValueError, TypeError) as error:
            raise LifecycleError(f"cannot restart {kind}: {error}") from None
        game = table.game
        if dump["mode"] == "fresh" or kind not in EXACT_GAMES:
            if data.get("autopilot") is True and hasattr(game, "set_autopilot"):
                game.set_autopilot(True)
            if isinstance(data.get("hero"), str) and hasattr(game, "set_hero"):
                game.set_hero(data["hero"])
            return "fresh"
        {"meowdoku": _restore_meowdoku, "wordle": _restore_wordle, "chess": _restore_board, "checkers": _restore_board,
         "fountain": _restore_fountain, "fountain3d": _restore_fountain}[kind](table, game, data)
        return "exact"


def _new_payload_from(new: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in new.items() if k in ("game", "id", "palette", "dictionary", "clock", "seed", "length", "parallel", "aspect")}
    return out


def _restore_meowdoku(table, game, data: dict[str, Any]) -> None:
    n2 = game.n * game.n
    cats, marks = _ints(data.get("cats", []), 0, n2 - 1, "cats", n2), _ints(data.get("marks", []), 0, n2 - 1, "marks", n2)
    mice = set(_ints(data.get("mice", []), 0, n2 - 1, "mice", n2)) & set(marks)
    hearts = _int(data.get("hearts", 3), 0, 9, "hearts")
    solution = [i for i in range(n2) if float(game.board["solution"][i]) == 0.0]
    lost_hearts = max(0, 3 - hearts)

    def wrong_moves(count: int) -> None:       # a wrong placement lands nothing and costs a heart, by the law itself
        for _ in range(count):
            game.session.place(*divmod(solution[0], game.n))
    if hearts > 0:
        wrong_moves(lost_hearts)
    for i in cats:
        game.session.place(*divmod(i, game.n))
    for i in marks:
        if i not in cats:
            game.session.toggle_x(*divmod(i, game.n))
    if hearts == 0:
        wrong_moves(3)
    game.mouse_cells = mice
    game.moves = _int(data.get("moves", 0), 0, 100000, "moves")
    game.hints_used = _int(data.get("hints_used", 0), 0, 100000, "hints_used")
    stock = data.get("stock", {})
    for key in game.stock:
        if key in stock:
            game.stock[key] = _int(stock[key], 0, 99, f"stock.{key}")
    for attr in ("bank_start", "free_start"):
        saved = data.get(attr, {})
        setattr(table, attr, {k: _int(saved[k], 0, 10**6, attr) for k in game.stock if isinstance(saved, dict) and k in saved})


def _restore_wordle(table, game, data: dict[str, Any]) -> None:
    guesses = data.get("guesses", [])
    if not isinstance(guesses, list) or len(guesses) > 60:
        raise LifecycleError("guesses must be at most 60 words")
    kept: list[str] = []
    for word in guesses:
        if not isinstance(word, str) or len(word) != game.length or not word.isalpha() or not word.islower():
            raise LifecycleError("a guess is not a word of the right length")
        kept.append(word)
        if word == game.answer:
            break
    game.guesses, game.moves = kept, len(kept)
    game.won = bool(kept) and kept[-1] == game.answer
    game.gave_up = bool(data.get("gave_up")) and not game.won
    entry = data.get("entry", "")
    game.entry = "".join(ch for ch in entry.lower() if ch.isalpha())[:game.length] if isinstance(entry, str) and not game.over else ""


def _restore_board(table, game, data: dict[str, Any]) -> None:
    from meowdoku.s2_titles.classic.games import PIECE_VALUES
    pieces, ids, seen = {}, {}, set()
    rows = data.get("pieces", [])
    if not isinstance(rows, list) or len(rows) > 64:
        raise LifecycleError("at most 64 pieces")
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise LifecycleError("malformed piece")
        cell, kind, side, pid = _int(row[0], 0, 63, "cell"), row[1], row[2], row[3]
        if cell in seen or kind not in PIECE_VALUES or side not in game.sides or not isinstance(pid, str) or not SAFE_WORD.match(pid):
            raise LifecycleError("malformed piece")
        seen.add(cell)
        pieces[cell], ids[cell] = (kind, side), pid
    if data.get("turn") not in game.sides:
        raise LifecycleError("bad turn")
    game.pieces, game.ids, game.turn, game.selected, game.history = pieces, ids, data["turn"], None, []
    game.moves = _int(data.get("moves", 0), 0, 100000, "moves")
    if hasattr(game, "rights") and data.get("rights") is not None:
        game.rights = set(_ints(data["rights"], 0, 63, "rights", 64))
        game.ep = None if data.get("ep") is None else _int(data["ep"], 0, 63, "ep")


def _restore_fountain(table, game, data: dict[str, Any]) -> None:
    """Through the game's own ``do`` actions, so every value is validated as a page's action would be."""
    from meowdoku.fountain import FLUIDS
    size = data.get("size")
    if not isinstance(size, str):
        raise LifecycleError("bad size")
    try:
        game.do({"type": "fountain_size", "size": size})
        game.do({"type": "fountain_run", "run": data.get("run")})
        shape, fluid = data.get("shape"), data.get("fluid")
        if isinstance(shape, str):
            game.do({"type": "fountain_choice", "name": "shape", "value": shape})
        if isinstance(data.get("inlet"), str) and hasattr(game, "inlet"):
            game.do({"type": "fountain_choice", "name": "inlet", "value": data["inlet"]})
        if fluid in FLUIDS:
            game.do({"type": "fountain_choice", "name": "fluid", "value": fluid})
        params = data.get("params", {})
        if not isinstance(params, dict) or len(params) > 200:
            raise LifecycleError("bad params")
        for name, value in params.items():
            game.do({"type": "fountain_param", "name": str(name), "value": _num(value, -1e12, 1e12, name)})
        if fluid in FLUIDS or fluid == "custom":         # (the params above marked it custom: the saved name wins)
            game.fluid = fluid
        game.do({"type": "fountain_mode", "mode": _int(data.get("mode", 2), 0, 64, "mode")})
        game.do({"type": "fountain_speed", "speed": _int(data.get("speed", 1), 0, 1000, "speed")})
        game.do({"type": "fountain_strata", "strata": _num(data.get("strata", game.strata), 0.0, 1e6, "strata")})
        game.mix = _num(data.get("mix", 0.5), 0.0, 1.0, "mix")
        if data.get("aspect") is not None:
            game.aspect = _num(data["aspect"], 0.1, 10.0, "aspect")
        if data.get("has_image"):
            if isinstance(data.get("image"), str) and len(data["image"]) <= 24 * 1024:
                game.do({"type": "fountain_image", "rgb": data["image"]})
            else:
                game.do({"type": "fountain_image", "count": game.n})
        if data.get("phase") == "replay" and game.has_image:
            game.phase = "replay"
        game.epoch = _int(data.get("epoch", 1), 1, 10**9, "epoch")     # the page's run is untouched: the same epoch
    except (KeyError, ValueError, TypeError) as error:
        if isinstance(error, LifecycleError):
            raise
        raise LifecycleError(f"cannot restore the fountain: {error}") from None


# --------------------------------------------------------------------------- the store
class SessionStore:
    """build/meowdoku/sessions/<sha256(cookie)>.json: the dump, the time it was saved, and nothing that names the player."""

    def __init__(self, directory: Path, clock=time.time):
        self.dir, self.clock = Path(directory), clock
        self.lock = threading.Lock()
        self.last_rev: dict[str, str] = {}

    def save(self, cookie: str, dump: dict[str, Any]) -> bool:
        name, rev = cookie_hash(cookie), dump_rev(dump)
        with self.lock:
            if self.last_rev.get(name) == rev:
                return False
            self.dir.mkdir(parents=True, exist_ok=True)
            text = canonical({"saved_at": self.clock(), "dump": dump})
            tmp = self.dir / f"{name}.tmp"
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.dir / f"{name}.json")
            self.last_rev[name] = rev
        return True

    def load(self, cookie: str) -> dict[str, Any] | None:
        path = self.dir / f"{cookie_hash(cookie)}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return validate_dump(record["dump"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def prune(self) -> int:
        removed = 0
        for path in self.dir.glob("*.json") if self.dir.exists() else ():
            if self.clock() - path.stat().st_mtime > SESSION_FILE_TTL:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


# --------------------------------------------------------------------------- admin
def admin_authorize(client_ip: str, host_header: str, forwarded: bool, supplied: str | None, token: str) -> tuple[bool, int, str]:
    """Localhost callers with the token only.  A request that came through a proxy (it carries forwarding headers, or a Host that is
    not a loopback name) is refused even from 127.0.0.1: a proxy's own address is loopback and says nothing about the caller."""
    if client_ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        return False, 403, "admin is localhost only"
    host = (host_header or "").strip()
    host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
    if forwarded or host not in ("127.0.0.1", "localhost", "[::1]"):
        return False, 403, "admin is localhost only (not through a proxy)"
    if not supplied or not token or not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
        return False, 401, "bad admin token"
    return True, 200, "ok"


def load_or_make_token(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if len(text) >= 32:
            return text
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    return token


class Lifecycle:
    """The running server's side: the announcement, the timer, the save, the exit."""

    def __init__(self, sessions, state_dir: Path, port: int, table_factory, clock=time.time):
        self.sessions, self.port, self.clock, self.table_factory = sessions, port, clock, table_factory
        self.paths = state_paths(Path(state_dir), port)
        self.store = SessionStore(self.paths["sessions"], clock)
        self.token = load_or_make_token(self.paths["token"])
        self.boot = secrets.token_hex(6)
        self.restart_at: float | None = None
        self.exit_code = 0
        self.server = None
        self.cond = threading.Condition()
        self.stopping = False
        self.restored = 0
        self.store.prune()

    # -- the announcement
    def announce(self, seconds: float) -> dict[str, Any]:
        if not 0 <= seconds <= MAX_ANNOUNCE_SECONDS:
            raise LifecycleError(f"seconds must be 0..{MAX_ANNOUNCE_SECONDS}")
        with self.cond:
            self.restart_at = self.clock() + seconds
            self.cond.notify_all()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        with self.cond:
            self.restart_at = None
            self.cond.notify_all()
        return self.status()

    def status(self) -> dict[str, Any]:
        at = self.restart_at
        now = self.clock()
        return {"boot": self.boot, "now": now, "restart_at": at, "seconds": None if at is None else max(0.0, at - now),
                "sessions": len(self.sessions.tables), "restored": self.restored}

    def run(self) -> None:
        """The timer thread: wake for an announcement's deadline, and autosave in between (a crash then loses little)."""
        next_save = self.clock() + AUTOSAVE_SECONDS
        while True:
            with self.cond:
                if self.stopping:
                    return
                now = self.clock()
                wake = min(next_save, self.restart_at if self.restart_at is not None else next_save)
                if wake > now:
                    self.cond.wait(timeout=min(wake - now, 1.0))
                    continue
                due = self.restart_at is not None and now >= self.restart_at
            if due:
                self.exit(EXIT_RESTART)
                return
            self.save_all()
            next_save = self.clock() + AUTOSAVE_SECONDS

    def exit(self, code: int) -> None:
        self.exit_code = code
        self.save_all()
        (self.paths["sessions"].parent / "last_exit.json").write_text(json.dumps({"at": self.clock(), "code": code, "boot": self.boot}), encoding="utf-8")
        print(f"[meowdoku] lifecycle: saved {len(list(self.paths['sessions'].glob('*.json')))} session file(s); exiting {code}", flush=True)
        if self.server is not None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    # -- sessions
    def save_all(self) -> int:
        saved = 0
        with self.sessions.lock:
            tables = [(token, entry[0]) for token, entry in self.sessions.tables.items()]
        for token, table in tables:
            if not table.lock.acquire(timeout=2.0):
                continue
            try:
                dump = dump_table(table)
            finally:
                table.lock.release()
            if dump is not None and self.store.save(token, dump):
                saved += 1
        return saved

    def restore_for(self, token: str):
        """A Table for a cookie that has a saved session (None: no file, or it would not restore)."""
        dump = self.store.load(token)
        if dump is None:
            return None
        table = self.table_factory()
        try:
            how = restore_table(table, dump)
        except Exception as error:      # noqa: BLE001 (a session that will not restore starts fresh)
            print(f"[meowdoku] lifecycle: session {cookie_hash(token)[:8]} did not restore ({type(error).__name__}: {error})", flush=True)
            return None
        self.restored += 1
        print(f"[meowdoku] lifecycle: session {cookie_hash(token)[:8]} restored ({dump['kind']}, {how})", flush=True)
        return table

    def snapshot_for(self, token: str | None, have: str | None) -> dict[str, Any]:
        """What GET /api/lifecycle?snap=1 adds: this cookie's dump when it changed (looked up without touching its idle clock)."""
        with self.sessions.lock:
            entry = self.sessions.tables.get(token or "")
        if entry is None:
            return {}
        table = entry[0]
        if not table.lock.acquire(timeout=0.2):
            return {}
        try:
            dump = dump_table(table)
        finally:
            table.lock.release()
        if dump is None:
            return {"snap_rev": ""}
        rev = dump_rev(dump)
        return {"snap_rev": rev} if rev == have else {"snap_rev": rev, "snap": dump}

    def request_json(self, body: dict[str, Any]) -> dict[str, Any]:
        op = body.get("op")
        if op == "restart":
            return self.announce(float(body.get("in", 60)))
        if op == "cancel":
            return self.cancel()
        if op == "status":
            return self.status()
        raise LifecycleError("unknown op")


def record_launch(paths: dict[str, Path], port: int, host: str, web_dir: Path) -> dict[str, Any]:
    """The effective launch configuration, so the supervisor can start the same server again."""
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    env = {k: v for k, v in os.environ.items() if ENV_KEEP_RE.match(k) and not ENV_SECRET_RE.search(k)}
    record = {"argv": argv, "python": sys.executable, "cwd": os.getcwd(), "port": port, "host": host, "web_dir": str(web_dir),
              "env": env, "pid": os.getpid(), "recorded_at": time.time()}
    paths["launch"].parent.mkdir(parents=True, exist_ok=True)
    paths["launch"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------------- supervisor
def supervise(launch_file: Path, *, max_backoff: float = 30.0, relaunch_delay: float = 0.5, max_runs: int | None = None,
              out=print, sleep=time.sleep, clock=time.time) -> int:
    """Spawn the recorded server; relaunch after a scheduled exit (75) at once and after a crash with backoff.
    A clean exit (0: Ctrl-C, or an operator stop) ends the supervisor.  The child writes its own stdout here."""
    runs, backoff = 0, 1.0
    while max_runs is None or runs < max_runs:
        try:
            cfg = json.loads(Path(launch_file).read_text(encoding="utf-8"))
            argv, cwd = [str(a) for a in cfg["argv"]], cfg.get("cwd") or None
            env = {**os.environ, **{str(k): str(v) for k, v in cfg.get("env", {}).items()}}
        except (OSError, ValueError, KeyError) as error:
            out(f"[supervisor] no usable {launch_file} ({type(error).__name__}: {error}); retry in {backoff:.0f}s")
            sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue
        runs += 1
        started = clock()
        out(f"[supervisor] run {runs}: {' '.join(argv)}  (cwd {cwd})")
        try:
            code = subprocess.call(argv, cwd=cwd, env=env)
        except KeyboardInterrupt:
            out("[supervisor] interrupted")
            return 0
        lived = clock() - started
        if code == 0:
            out(f"[supervisor] child exited 0 after {lived:.1f}s: stopping")
            return 0
        if code == EXIT_RESTART:
            out(f"[supervisor] scheduled exit after {lived:.1f}s: relaunching in {relaunch_delay:.1f}s")
            backoff = 1.0
            sleep(relaunch_delay)
            continue
        backoff = 1.0 if lived > 60 else min(backoff * 2, max_backoff)
        out(f"[supervisor] crash (exit {code}) after {lived:.1f}s: relaunching in {backoff:.0f}s")
        sleep(backoff)
    return 0


# --------------------------------------------------------------------------- command line
def _post_admin(port: int, state_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    import urllib.error
    import urllib.request
    token = state_paths(state_dir, port)["token"].read_text(encoding="utf-8").strip()      # never printed
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/admin/lifecycle", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "X-Meowdoku-Admin": token})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"refused ({error.code}): {error.read().decode('utf-8', 'replace')[:200]}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meowdoku.lifecycle", description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    sub = parser.add_subparsers(dest="cmd", required=True)
    restart = sub.add_parser("restart", help="announce a restart in N seconds")
    restart.add_argument("--in", dest="seconds", type=float, default=60.0)
    sub.add_parser("cancel")
    sub.add_parser("status")
    sup = sub.add_parser("supervise", help="run the server from launch.json, relaunching it")
    sup.add_argument("--launch-file", default=None)
    sup.add_argument("--max-backoff", type=float, default=30.0)
    sup.add_argument("server_args", nargs="*", help="first run only: server.py arguments, when no launch.json exists yet")
    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir)
    if args.cmd == "supervise":
        launch = Path(args.launch_file) if args.launch_file else state_paths(state_dir, args.port)["launch"]
        if not launch.exists():
            script = HERE / "server.py"
            launch.parent.mkdir(parents=True, exist_ok=True)
            launch.write_text(json.dumps({"argv": [sys.executable, str(script), "--port", str(args.port), *args.server_args],
                                          "cwd": os.getcwd(), "env": {}}), encoding="utf-8")
        return supervise(launch, max_backoff=args.max_backoff, out=lambda line: print(time.strftime("%H:%M:%S ") + line, flush=True))
    body = {"op": args.cmd, **({"in": args.seconds} if args.cmd == "restart" else {})}
    print(json.dumps(_post_admin(args.port, state_dir, body)))
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
    raise SystemExit(main())
