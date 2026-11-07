"""The Meowdoku web server: the Python state machines behind the WebGL page.

    python examples/meowdoku/server.py --port 8766 --bind 0.0.0.0

Serves ``build/meowdoku/web`` (the exported page, shaders, assets and the
prebaked board database) and a JSON API over the games in ``games.py``:

    GET  /api/catalog                      -> games, Meowdoku levels and boards
    POST /api/new     {"game": "meowdoku", "id": "<board id>"}
                      {"game": "chess"|"checkers"} | {"game": "wordle", "seed": 3}
    POST /api/action  {"type": "cell", "r", "c", "button"} | {"type": "text", "text"}
                      | {"type": "submit"|"backspace"|"undo"|"reset"|"give_up"}
                      | {"type": "tick", "move_axis", "fire"}  (free-board games)
                      | {"type": "autopilot", "on": true|false}  (entity games: the trained policy
                        in ``autopilot.py`` supplies each tick's controls; state field ``autopilot``)
                      | {"type": "worldwalk_goto"|"worldwalk_stop"|"worldwalk_tool"|"worldwalk_place"
                        |"worldwalk_here", ...}  (World Walk: click to walk, build tiles; see ``WorldWalk``)
    GET  /api/state
    POST /api/profile {"name": "..."}      -> pick this session's player profile
    GET  /api/profile                      -> the profile, and the current game's top scores

Finished games (``Game.result()``) are recorded on the session's profile
(``profiles.py``); every state carries ``player`` and that game's ``record``.

Every call returns the game's state (see ``games.py``); the page animates
the difference between states.  Meowdoku runs the dt-system law through
``MeowdokuSession``; the server never renders anything.
"""
from __future__ import annotations

import argparse
import os
import json
import secrets
import struct
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Checkers, Chess, ChessArena, ChessVsNet, LichessGame, MeowdokuGame, Wordle  # noqa: E402
from meowdoku.invaders import SpaceInvaders  # noqa: E402
from meowdoku.scroller import BulletScroller  # noqa: E402
from meowdoku.delve import Delve  # noqa: E402
from meowdoku.platformer import Platformer  # noqa: E402
from meowdoku.sidescroller import SideScroller  # noqa: E402
from meowdoku.tilequest import TileQuest  # noqa: E402
from meowdoku.boardwalk import Boardwalk  # noqa: E402
from meowdoku.worldwalk import WorldWalk  # noqa: E402
from meowdoku.fountain import FountainGame  # noqa: E402
from meowdoku.fountain_game import FountainGame as Fountain3DGame  # noqa: E402  (the 3-D fountain: its own law, products and page controls)
from meowdoku.marble_madness import MarbleMadness  # noqa: E402
from meowdoku.purrline import Purrline  # noqa: E402
from meowdoku.mobroad import Mobroad  # noqa: E402  (the road-defence mob shooter: the fountain law as the mob, an entity law for the shooters)
from meowdoku.profiles import Profiles  # noqa: E402
from meowdoku.realtime_socket import SnapshotEncoder, accept_key, frame, read_frame  # noqa: E402
from meowdoku import share as share_mod  # noqa: E402  (share links and the view-only live stream)
from meowdoku import lifecycle as lifecycle_mod  # noqa: E402  (restart announcements, session save/restore, launch record)
from meowdoku import logring as logring_mod  # noqa: E402  [LOG] (size-capped text log ring behind /api/log)
from meowdoku import startup as startup_mod  # noqa: E402  [LOADING] (bind first; per-product readiness)
from meowdoku import loading_anim as anim_mod  # noqa: E402  [LOADING] (the reduced loading animation's store and validator)
from meowdoku import access as access_mod  # noqa: E402  [ACCESS] (origin -> tier: lan/dns/ip -> edit/dev/play/shader)

WEB_DIR = ROOT / "build" / "meowdoku" / "web"
PRODUCT_WAIT_SECONDS = 90       # [LOADING] how long a request for a building product file waits before "503, retry"
PROFILES = ROOT / "build" / "meowdoku" / "profiles.json"


class Table:
    """One game at a time, picked from the catalog."""

    def __init__(self, web_dir: Path, profiles: Profiles | None = None):
        self.web_dir = web_dir
        self.profiles = profiles
        self.player: str | None = None       # this session's profile name
        self.player_token: str | None = None # the browser's long-lived player token (cookie)
        self.last_award: dict | None = None  # points from the last finished game
        self.bank_start: dict = {}           # a puzzle's banked power-ups at its start
        self.free_start: dict = {}           # ... and its free ones (stock + ongoing extras)
        self.recorded = None                 # the game object whose result is on the profile
        self.lock = threading.RLock()
        database = json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))
        self.boards = {board["id"]: board for board in database["boards"]}
        self.levels = database["levels"]
        self.game = None
        self.rt_seq = 0                      # [LIVEVIEW] real-time ticks landed on this table (a viewer socket sends a snapshot when it moves)
        self.chess_clock = None              # the time control the player last picked (chess against the network)

    def catalog(self) -> dict:
        return {
            "games": ["meowdoku", "chess", "checkers", "wordle", "chess_vs_net", "chess_arena", "lichess", "space_invaders", "bullet_hell", "platformer", "delve", "side_scroller", "tilequest", "boardwalk", "worldwalk", "fountain", "fountain3d", "marble_madness", "tipjar", "purrline", "mobroad"],
            "meowdoku": {"levels": self.levels,
                         "boards": [{"id": b["id"], "level": b["level"], "type": b["type"], "n": b["n"]} for b in self.boards.values()]},
        }

    def new(self, payload: dict) -> dict:
        kind = str(payload.get("game", "meowdoku"))
        # A streak game left unsolved breaks its success streak.
        previous = self.game
        if (self.profiles is not None and self.player and previous is not None
                and getattr(previous, "game", None) in ("meowdoku", "wordle")
                and self.recorded is not previous and getattr(previous, "moves", 0) > 0):
            self.profiles.break_streak(self.player, previous.game)
        # Games are views; the learner and the Lichess client are process-wide
        # background workers that a mode switch never stops.
        if kind == "meowdoku":
            self.game = MeowdokuGame(self.boards[str(payload["id"])], **({"palette": str(payload["palette"])} if payload.get("palette") else {}))
            # The player's banked power-ups and any ongoing extras join the tray.
            if self.profiles is not None and self.player:
                bank, bonus = self.profiles.start_puzzle(self.player)
                self.bank_start = {k: int(bank.get(k, 0)) for k in self.game.stock}
                self.free_start = {k: self.game.stock[k] + int(bonus.get(k, 0)) for k in self.game.stock}
                for k in self.game.stock:
                    self.game.stock[k] = self.free_start[k] + self.bank_start[k]
        elif kind == "chess":
            self.game = Chess("chess")
        elif kind == "checkers":
            self.game = Checkers("checkers")
        elif kind == "wordle":
            self.game = Wordle(int(payload.get("seed", 0)), str(payload.get("dictionary") or "builtin"),
                               int(payload.get("length") or 5))
        elif kind == "chess_arena":
            self.game = ChessArena(parallel=int(payload.get("parallel", 4)), lock=self.lock)   # the learner is one per process; 4 searched games
        elif kind == "chess_vs_net":
            self.game = ChessVsNet(clock=payload.get("clock") or self.chess_clock)
            self.game.lock = self.lock       # the network's search lands its move under this table's lock
        elif kind == "lichess":
            self.game = LichessGame()
        elif kind == "space_invaders":
            self.game = SpaceInvaders(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "bullet_hell":
            # The player starts at the level after the highest they have defeated.
            start = 1 + (self.profiles.level(self.player, "bullet_hell") if self.profiles else 0)
            self.game = BulletScroller(aspect=float(payload.get("aspect") or 0.0), level=start)
        elif kind == "platformer":
            self.game = Platformer(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "delve":
            self.game = Delve(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "side_scroller":
            self.game = SideScroller(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "tilequest":
            self.game = TileQuest(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "boardwalk":
            self.game = Boardwalk(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "worldwalk":
            self.game = WorldWalk(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "fountain":
            self.game = FountainGame(aspect=float(payload.get("aspect") or 0.0) or None)       # the page runs the law; the server holds phase, seed, image
        elif kind == "fountain3d":
            self.game = Fountain3DGame(aspect=float(payload.get("aspect") or 0.0) or None)
        elif kind == "marble_madness":
            self.game = MarbleMadness(aspect=float(payload.get("aspect") or 0.0) or None)
        elif kind == "tipjar":
            from meowdoku import tipjar            # the watch-only Bitcoin tip jar on the 3-D fountain (build/meowdoku/tipjar.json; else a DEMO)
            self.game = tipjar.TipJarGameClass()(aspect=float(payload.get("aspect") or 0.0) or None)
        elif kind == "purrline":
            self.game = Purrline(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "mobroad":
            self.game = Mobroad(aspect=float(payload.get("aspect") or 0.0), size=str(payload.get("size") or "4k"))
        else:
            raise ValueError(f"unknown game {kind!r}")
        self.new_payload = dict(payload)        # LIFECYCLE: what rebuilds this game after a restart (lifecycle.dump_table)
        self.apply_hero()
        return self.state()

    def apply_hero(self) -> None:
        """A platformer-style game draws the player's chosen character (the profile's ``avatar`` setting)."""
        if hasattr(self.game, "set_hero") and self.profiles is not None and self.player:
            avatar = ((self.profiles.inventory(self.player) or {}).get("settings") or {}).get("avatar")
            if avatar:
                self.game.set_hero(avatar)

    def action(self, payload: dict) -> dict:
        if self.game is None:
            raise ValueError("no game: POST /api/new first")
        kind = str(payload.get("type"))
        from meowdoku import idle_trainer
        idle_trainer.touch()                            # play is on: the background trainer waits
        if kind == "cell":
            self.game.cell(int(payload["r"]), int(payload["c"]), str(payload.get("button", "primary")))
        elif kind == "text":
            self.game.text(str(payload.get("text", "")))
        elif kind == "tick":
            if hasattr(self.game, "control"):
                self.game.control(payload)          # the held controls ride on the tick
            if hasattr(self.game, "tick"):
                self.game.tick()
        elif kind.startswith("fountain_") or kind.startswith("marble_") or kind.startswith("worldwalk_") or kind.startswith("delve_"):
            if not hasattr(self.game, "do"):
                raise ValueError("this game takes no such action")
            self.game.do(payload)
        elif kind == "clock":
            if not isinstance(self.game, ChessVsNet):
                raise ValueError("only chess against the network has a clock")
            self.chess_clock = str(payload.get("name", ""))
            self.game.set_clock(self.chess_clock)           # a new time control restarts the game
        elif kind == "autopilot":
            if not hasattr(self.game, "set_autopilot"):
                raise ValueError("this game has no autopilot")
            self.game.set_autopilot(bool(payload.get("on")))    # the compiled policy plays; the page keeps drawing
        elif kind == "pilot_arch":
            from meowdoku import autopilot, pilot_decl
            name = getattr(self.game, "game", None)
            if name not in pilot_decl.DECLS:
                raise ValueError("this game has no declared autopilot networks")
            pilot_decl.set_arch(name, str(payload.get("arch")))          # token net or simple graph net, per game
            autopilot._PILOT_WEIGHTS.pop(name, None)
            if self.game.autopilot:
                self.game.set_autopilot(True)                             # re-attach the chosen network's weights
        elif kind == "demo_learning":
            from meowdoku import autopilot
            if getattr(self.game, "game", None) not in autopilot.FEATURE_FUNCTIONS:
                raise ValueError("this game has no autopilot to teach")
            autopilot.DEMO_LEARNING[self.game.game] = bool(payload.get("on"))
            autopilot._save_learning()
        elif kind == "layout":
            if hasattr(self.game, "layout"):
                self.game.layout(float(payload.get("aspect") or 0.0))
        elif kind == "move":
            if not hasattr(self.game, "uci_move") or getattr(self.game, "input", "") == "watch":
                raise ValueError("this game takes no typed moves")
            self.game.uci_move(payload.get("uci", ""))
        elif kind in ("arena_save", "arena_stop", "arena_resume"):
            if not isinstance(self.game, ChessArena):
                raise ValueError("not the chess arena")
            {"arena_save": self.game.learner.save, "arena_stop": self.game.learner.stop,
             "arena_resume": self.game.learner.resume}[kind]()
        elif kind in ("lichess_pause", "lichess_resume"):
            if not isinstance(self.game, LichessGame) or self.game.client is None:
                raise ValueError("Lichess mode is not connected")
            if kind == "lichess_pause":
                self.game.client.pause_at_idle()
            else:
                self.game.client.resume()
        elif kind == "hint":
            self.game.hint()
        elif kind == "power":
            power = str(payload.get("kind", ""))
            self.game.power(power)
            # Free power-ups go first; past them, each one used spends the bank.
            if self.profiles is not None and self.player and power in self.bank_start:
                used = self.free_start[power] + self.bank_start[power] - self.game.stock[power]
                self.profiles.set_bank(self.player, power, self.bank_start[power] - max(0, used - self.free_start[power]))
        elif kind == "buy":
            if self.profiles is None or not self.player:
                raise ValueError("no player")
            self.profiles.buy(self.player, str(payload.get("item", "")))
        elif kind == "settings":
            if self.profiles is not None and self.player and isinstance(payload.get("values"), dict):
                self.profiles.save_settings(self.player, payload["values"])
            if hasattr(self.game, "set_hero") and isinstance(payload.get("values"), dict) and "avatar" in payload["values"]:
                self.game.set_hero(payload["values"]["avatar"])
        elif kind == "place":                                        # [MOBROAD] a tower dropped on the field: the game decides validity and cost
            if not hasattr(self.game, "place"):
                raise ValueError("this game has no placeables")
            self.game.place(payload)
        elif kind == "palette":
            self.game.set_palette(str(payload.get("name", "")))
        elif kind == "mode":
            if not hasattr(self.game, "set_mode"):
                raise ValueError("this game has no modes")
            self.game.set_mode(str(payload.get("name", "")), bool(payload.get("on")))
        elif kind in ("submit", "backspace", "undo", "reset", "give_up"):
            getattr(self.game, kind)()
            if kind in ("undo", "reset"):
                self.recorded = None if self.game.result() is None else self.recorded
        else:
            raise ValueError(f"unknown action {kind!r}")
        return self.state()

    def state(self) -> dict:
        st = self.game.state() if self.game is not None else {"game": None}
        self._record()
        if self.profiles is not None and self.game is not None:
            st["player"] = self.player
            st["record"] = self.profiles.best(self.player, self.game.game)
            st["inventory"] = self.profiles.inventory(self.player)
            st["award"] = self.last_award
        return st

    def _record(self) -> None:
        """Put a finished game's result on the player's profile, once."""
        if self.profiles is None or self.player is None or self.game is None or self.recorded is self.game:
            return
        result = self.game.result()
        if result is not None:
            self.profiles.record(self.player, self.game.game, result)
            self.last_award = self.profiles.award(self.player, self.game.game, result)
            if result.get("won") and result.get("level"):
                self.profiles.defeated(self.player, self.game.game, int(result["level"]))
            self.recorded = self.game

    def set_player(self, payload: dict) -> dict:
        self.player = self.profiles.ensure(str(payload.get("name", "")))
        if self.player_token:
            self.profiles.bind(self.player_token, self.player)     # this browser is now that player
        return self.profile()

    def profile(self) -> dict:
        game = self.game.game if self.game is not None else None
        return {"player": self.player, "profile": self.profiles.profile(self.player),
                "game": game, "leaderboard": self.profiles.leaderboard(game) if game else []}


# Browsers scope cookies by host, not port: two servers on one host (live 8766
# and dev 8767) sharing one cookie name overwrite each other's session token on
# every request, so each sees "no game" and the page restarts in a loop.  The
# default port keeps the historical name; any other port gets its own.
COOKIE = "meowdoku_session"
PLAYER_COOKIE = "meowdoku_player"      # the long-lived player identity (per port, like COOKIE)
PLAYER_COOKIE_SECONDS = 2 * 365 * 24 * 3600
SESSION_IDLE_SECONDS = 2 * 3600


class Sessions:
    """session token -> that client's own Table; idle ones expire."""

    def __init__(self, web_dir: Path, profiles: Profiles | None = None):
        self.web_dir = web_dir
        self.profiles = profiles
        self.lock = threading.Lock()
        self.tables: dict[str, list] = {}       # token -> [Table, last_seen]
        self.shares = share_mod.ShareStore(web_dir.parent / "shares")     # build/meowdoku/shares (gitignored with build/)
        self.spectate = share_mod.Spectate()                              # owner session -> its view-only live share
        self.share_hits: dict[str, list] = {}                             # client address -> recent share POST times
        boards = json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))
        self.board_count = len(boards["boards"])
        self.lifecycle = None                   # LIFECYCLE: set by main(); restores a saved session for a cookie that has a file
        self.readiness = startup_mod.Readiness()          # [LOADING] what is built, building, pending (set up by main(); an empty one never holds a request)
        self.anim = anim_mod.Store(web_dir.parent)        # [LOADING] build/meowdoku/loading_anim.mla
        self.anim_busy = threading.Lock()
        self.anim_hits: dict[str, list] = {}
        self.access = access_mod.Access(web_dir.parent / "access.json")   # [ACCESS] build/meowdoku/access.json (gitignored); main() may point it elsewhere
        self.anim_misses: dict[str, list] = {}            # [STUDIO] wrong owner tokens per address

    def get(self, token: str | None) -> tuple[str, Table, bool]:
        now = time.time()
        with self.lock:
            for key in [k for k, (_, seen) in self.tables.items() if now - seen > SESSION_IDLE_SECONDS]:
                del self.tables[key]
                self.spectate.drop_owner(key)       # a session that ended takes its viewer token with it
            fresh = token not in self.tables
            if fresh and token and self.lifecycle is not None:       # LIFECYCLE: a cookie with a saved session gets it back
                restored = self.lifecycle.restore_for(token)
                if restored is not None:
                    self.tables[token] = [restored, now]
                    fresh = False
            if fresh:
                token = secrets.token_urlsafe(24)
                self.tables[token] = [Table(self.web_dir, self.profiles), now]
            entry = self.tables[token]
            entry[1] = now
            return token, entry[0], fresh


def make_handler(sessions: Sessions):
    web_dir = sessions.web_dir

    class Handler(SimpleHTTPRequestHandler):
        # Keep-alive: a real-time page ticks ~30 times a second, and a new TCP
        # connection per tick (HTTP/1.0) piles closed sockets up until Windows
        # refuses (ERR_NO_BUFFER_SPACE).  Every response carries a
        # Content-Length (_json, the static files, send_error), which is what
        # HTTP/1.1 persistence needs.
        protocol_version = "HTTP/1.1"

        def handle(self):
            # A kept-alive connection the browser drops (a closed tab, a phone
            # going to sleep) is the normal end of a connection, not an error.
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def log_message(self, format, *args):  # noqa: A002 - quiet static and polling logs
            line = str(args[0]) if args else ""
            if "/api/" in line and "/api/log" not in line and "/api/action" not in line and "/api/state" not in line and "/api/lifecycle?" not in line:   # LIFECYCLE: the poll is quiet
                sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _table(self) -> Table:
            from http.cookies import SimpleCookie
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            token, table, fresh = sessions.get(jar[COOKIE].value if COOKIE in jar else None)
            self._token = token
            self._set_cookie = [f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax"] if fresh else []
            # The player: the browser's long-lived token names its profile; a
            # browser without one gets a token and a generated name.
            profiles = sessions.profiles
            if profiles is not None and table.player_token is None:
                held = jar[PLAYER_COOKIE].value if PLAYER_COOKIE in jar else None
                name = profiles.by_token(held)
                if name is None:
                    held = secrets.token_urlsafe(24)
                    name = profiles.generated_name()
                    profiles.bind(held, name)
                    self._set_cookie.append(f"{PLAYER_COOKIE}={held}; Path=/; Max-Age={PLAYER_COOKIE_SECONDS}; HttpOnly; SameSite=Lax")
                table.player_token, table.player = held, name
            return table

        # ---- [ACCESS] origin and tier, from the socket peer and the Host header only (never a forwarding header: there is no proxy)
        def _classify(self) -> None:
            self._origin, self._tier = sessions.access.decide(self.client_address[0], self.headers.get("Host"))

        def _shader_request(self, path: str, method: str) -> bool:
            """The shader tier gets the shader page and its few assets, nothing else.  True when the request was answered here."""
            if method != "GET":
                self.close_connection = True
                self._set_cookie = []
                self._json({"error": "shader tier"}, HTTPStatus.FORBIDDEN)
                return True
            if path in access_mod.SHADER_APIS:
                return False                                   # served below (/api/ready is reduced for this tier)
            if path.startswith("/api/") or path.startswith("/ws"):
                self._set_cookie = []
                self._json({"error": "shader tier"}, HTTPStatus.FORBIDDEN)
                return True
            if path not in access_mod.SHADER_FILES:
                self.path = access_mod.SHADER_PAGE                # any other page address is the shader page
            super().do_GET()
            return True

        def _log_get(self) -> None:                               # [LOG]
            from urllib.parse import parse_qs
            if not access_mod.require_tier(self, "dev"):
                return
            if self._is_viewer():
                return self._refuse_viewer()
            q = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                since, limit = int(q.get("since", ["0"])[0]), int(q.get("limit", ["500"])[0])
            except ValueError:
                return self._json({"error": "since and limit are integers"}, HTTPStatus.BAD_REQUEST)
            self._set_cookie = []
            self._json(logring_mod.RING.fetch(since, limit, set(q["src"][0].split(",")) if q.get("src") else None))

        def _whoami(self) -> None:
            self._set_cookie = []
            self._json({"origin": self._origin, "tier": self._tier,
                        "dev": access_mod.at_least(self._tier, "dev"), "edit": access_mod.at_least(self._tier, "edit"),
                        "shader": self._tier == "shader"})

        def do_HEAD(self):  # noqa: N802
            self._classify()
            if self._tier == "shader" or self.path.startswith("/api/") or self.path.startswith("/ws"):
                self.send_response(HTTPStatus.FORBIDDEN); self.send_header("Content-Length", "0"); self.end_headers()
                return
            return super().do_HEAD()

        def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for cookie in getattr(self, "_set_cookie", None) or ():
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            # A malformed request (e.g. TLS bytes on this http port) is answered
            # before the request line is parsed, so there may be no path yet.
            if not getattr(self, "path", "").startswith("/api/"):
                self.send_header("Cache-Control", "no-cache")          # the page is edited live
            super().end_headers()

        def _socket(self):
            """``/ws``: this session's real-time channel (realtime_socket.py).
            Each text frame is one tick's input; each answer is a binary
            snapshot delta, preceded by a ``meta`` text frame when one is due."""
            key = self.headers.get("Sec-WebSocket-Key")
            if self._is_viewer() or "viewer=" in self.path or "mdv." in (self.headers.get("Sec-WebSocket-Protocol") or ""):
                return self._refuse_viewer()           # [LIVEVIEW] a viewer token never opens the owner's (input) socket
            if not key or "websocket" not in (self.headers.get("Upgrade") or "").lower():
                return self._json({"error": "expected a WebSocket upgrade"}, HTTPStatus.BAD_REQUEST)
            table = self._table()
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key(key))
            self.end_headers()
            self.wfile.flush()
            encoder = SnapshotEncoder()
            try:
                while True:
                    message = read_frame(self.rfile)
                    if message is None or message[0] == 0x8:
                        break
                    received = time.perf_counter()
                    opcode, payload = message
                    if opcode == 0x9:
                        self.wfile.write(frame(0xA, payload)); continue
                    if opcode != 0x1:
                        continue
                    try:
                        request = json.loads(payload.decode("utf-8"))
                        with table.lock:
                            game = table.game
                            if game is None or not hasattr(game, "session"):
                                raise ValueError("no game: POST /api/new first")
                            from meowdoku import idle_trainer
                            idle_trainer.touch()          # a realtime tick: play is on
                            game.control(request)
                            game.tick()
                            table.rt_seq += 1          # [LIVEVIEW]
                            echo = request.get("ct")
                            # (the timing exchange is in the game's virtual wall: the wall integrated by its clock speed)
                            stamp = game.virtual(received) if hasattr(game, "virtual") else received
                            binary, meta = encoder.encode(game, table.state(), echo=echo if isinstance(echo, (int, float)) else None,
                                                          received=stamp)
                        if meta is not None:
                            self.wfile.write(frame(0x1, json.dumps({"meta": meta}).encode("utf-8")))
                        self.wfile.write(frame(0x2, binary))
                    except Exception as error:  # the page shows the refusal
                        self.wfile.write(frame(0x1, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode("utf-8")))
            except (ConnectionError, OSError):
                pass
            finally:
                self.close_connection = True

        def _spectate_socket(self) -> None:
            """[LIVEVIEW] ``/ws/spectate``: the viewer's READ-ONLY push socket.  The owner's page gets one snapshot delta per tick
            over ``/ws``; this sends a viewer the same frames (its own SnapshotEncoder over the owner's game, so a late viewer
            starts on a keyframe) whenever the owner's table has ticked (``Table.rt_seq``).  The token rides the WebSocket
            subprotocol (``mdv.<token>``; a browser cannot set a header, and it stays out of the URL and the log).  The socket
            NEVER reads a message's content: any data frame from the viewer closes the socket (1008).  The registry's heartbeat
            (keep_stream) closes it the moment the token is revoked, expires, or its owner's session ends."""
            key = self.headers.get("Sec-WebSocket-Key")
            protocols = [p.strip() for p in (self.headers.get("Sec-WebSocket-Protocol") or "").split(",")]
            token = next((p[4:] for p in protocols if p.startswith("mdv.")), "")
            if not key or "websocket" not in (self.headers.get("Upgrade") or "").lower():
                return self._json({"error": "expected a WebSocket upgrade"}, HTTPStatus.BAD_REQUEST)
            alive = lambda o: o in sessions.tables           # noqa: E731
            try:
                owner, vid = sessions.spectate.open_stream(token, alive)
            except share_mod.ShareError as error:
                return self._share_error(error)
            self._set_cookie = []
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key(key))
            self.send_header("Sec-WebSocket-Protocol", "mdv." + token)
            self.end_headers()
            self.wfile.flush()
            wlock, done = threading.Lock(), threading.Event()

            def send(opcode: int, payload: bytes) -> int:
                with wlock:
                    self.wfile.write(frame(opcode, payload))
                return len(payload) + 4

            def listen() -> None:               # a viewer may only ping and close; every data frame ends the socket unread
                try:
                    while not done.is_set():
                        head = self.rfile.read(2)
                        if len(head) < 2:
                            break
                        op, length = head[0] & 0x0F, head[1] & 0x7F
                        if op in (0x8, 0x9, 0xA) and length < 126:
                            if head[1] & 0x80:
                                mask = self.rfile.read(4)
                            data = self.rfile.read(length)
                            if op == 0x8:
                                break
                            if op == 0x9:
                                send(0xA, data)
                            continue
                        send(0x8, struct.pack(">H", 1008) + b"view-only: this socket accepts no inputs")
                        sessions.spectate_refused = getattr(sessions, "spectate_refused", 0) + 1
                        break
                except (ConnectionError, OSError, ValueError):
                    pass
                finally:
                    done.set()
            threading.Thread(target=listen, name="spectate-listen", daemon=True).start()
            encoder, last_seq, last_at, sent_bytes, reason = SnapshotEncoder(), -1, 0.0, 0, 1000
            try:
                while not done.is_set():
                    if not sessions.spectate.keep_stream(token, vid, sent_bytes, alive):
                        reason = 4410
                        break
                    sent_bytes = 0
                    entry = sessions.tables.get(owner)
                    table = entry[0] if entry else None
                    now = time.perf_counter()
                    if table is not None and table.rt_seq != last_seq and now - last_at >= 0.03:
                        with table.lock:
                            last_seq, game = table.rt_seq, table.game
                            if game is not None and hasattr(game, "session"):
                                stamp = game.virtual(now) if hasattr(game, "virtual") else now
                                binary, meta = encoder.encode(game, table.state(), echo=None, received=stamp)
                            else:
                                binary = meta = None
                        if binary is not None:
                            if meta is not None:
                                sent_bytes += send(0x1, json.dumps({"meta": meta}).encode("utf-8"))
                            sent_bytes += send(0x2, binary)
                            last_at = now
                    done.wait(0.01)
                if not done.is_set():
                    send(0x8, struct.pack(">H", reason) + b"the shared session has ended")
            except (ConnectionError, OSError):
                pass
            finally:
                done.set()
                sessions.spectate.close_stream(token, vid)
                self.close_connection = True

        # ---- share links and the view-only stream (share.py).  A request that presents the viewer header is a viewer's:
        # it may read the catalog, the static page and /api/spectate/watch, and nothing else.
        def _is_viewer(self) -> bool:
            return self.headers.get(share_mod.VIEWER_HEADER) is not None

        def _refuse_viewer(self) -> None:
            self._set_cookie = []
            self.close_connection = True      # the refused request's body is unread: the connection must not carry on
            self._json({"error": "this is a view-only link: it cannot send inputs or controls"}, HTTPStatus.FORBIDDEN)

        def _share_error(self, error) -> None:
            self._set_cookie = []
            if getattr(error, "status", 0) == 413:
                self.close_connection = True  # a too-large body was not read
            self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))

        def _watch(self) -> None:
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            token = self.headers.get(share_mod.VIEWER_HEADER) or ""
            vid, rev, lrev = (query.get(k, [None])[0] for k in ("vid", "rev", "lrev"))

            # [LIVEVIEW] "nostate=1" (a socket carries the game) is honoured only while the viewer holds the owner's CURRENT game
            # (its game name and id, ``gk``): a switch to another game still sends the state.
            want_state = True
            if query.get("nostate", ["0"])[0] == "1":
                try:
                    game = sessions.tables[sessions.spectate.by_token[token].owner][0].game
                    want_state = query.get("gk", [""])[0] != f"{getattr(game, 'game', '')}|{getattr(game, 'id', '')}"
                except (KeyError, AttributeError):
                    want_state = True

            def owner_state():
                owner = sessions.spectate.by_token[token].owner
                table = sessions.tables[owner][0]
                with table.lock:
                    return share_mod.public_state(table.state())
            try:
                payload = sessions.spectate.watch(token, vid, rev, owner_state, owner_alive=lambda o: o in sessions.tables,
                                                  have_live=int(lrev) if lrev and lrev.lstrip("-").isdigit() else None,
                                                  want_state=want_state,
                                                  have_key=int(query["kseq"][0]) if query.get("kseq", [""])[0].isdigit() else None)
            except KeyError:
                return self._share_error(share_mod.ShareError("the shared session has ended", 410))
            except share_mod.ShareError as error:
                return self._share_error(error)
            stats = sessions.spectate.viewer_stats(token, payload["vid"])
            if stats:
                payload["stats"] = stats                    # this viewer's bytes so far, per second (before this reply)
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sessions.spectate.account(token, payload["vid"], len(body) + 180)     # + the response headers
            self._set_cookie = []
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _share_rate_ok(self) -> bool:
            now, hits = time.time(), sessions.share_hits.setdefault(self.client_address[0], [])
            hits[:] = [t for t in hits if now - t < 60]
            if len(hits) >= 60:
                return False
            hits.append(now)
            return True

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            self._classify()                                      # [ACCESS]
            if self._tier == "shader" and self._shader_request(path, "GET"):
                return
            if path == "/api/log":                                # [LOG] the text ring: dev tier and above, never a view-only link
                return self._log_get()
            if path == "/api/whoami":                             # [ACCESS] the page learns its own tier here
                return self._whoami()
            if path == "/api/lifecycle":                         # LIFECYCLE: the countdown, and this session's restorable copy
                return self._lifecycle_get()
            if path == "/api/ready":                              # [LOADING] per-product readiness, polled by the page (loading_panel.js)
                return self._ready()
            if path == "/api/loading_anim":
                return self._anim_get()
            if path == "/api/loading_library":                    # [STUDIO] the library listing: anyone may view
                return self._library_get()
            if path == "/api/spectate/watch":
                return self._watch()
            if path == "/ws/spectate":                           # [LIVEVIEW] the viewer's read-only push socket
                return self._spectate_socket()
            if path.startswith("/api/share/"):
                try:
                    return self._json(sessions.shares.get(path[len("/api/share/"):]))
                except share_mod.ShareError as error:
                    return self._share_error(error)
            if self._is_viewer() and (path.startswith("/ws") or (path.startswith("/api/") and not path.startswith("/api/catalog"))):
                return self._refuse_viewer()
            if self.path.startswith("/ws"):
                return self._socket()
            if self.path.startswith("/api/state"):
                table = self._table()
                with table.lock:
                    return self._json(table.state())
            if self.path.startswith("/api/catalog"):
                table = self._table()
                return self._json(table.catalog())
            if self.path.startswith("/api/profile"):
                table = self._table()
                with table.lock:
                    return self._json(table.profile())
            if not path.startswith("/api/") and not sessions.readiness.wait_for_path(path, PRODUCT_WAIT_SECONDS):
                # [LOADING] a product file that is still building: wait for it (above); past the limit, "try again", never a 404 into a broken page
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.send_header("Retry-After", "5"); self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return super().do_GET()

        # ---- [LOADING] readiness and the loading animation
        def _ready(self) -> None:
            self._set_cookie = []
            info = sessions.anim.info()
            out = sessions.readiness.snapshot() if self._tier != "shader" else {}      # [ACCESS] the shader tier sees only the animation
            out["loading_anim"] = ({"present": True, "url": f"/api/loading_anim?v={info['digest']}", **info} if info
                                   else {"present": False, "upload": True})
            self._json(out)

        def _anim_get(self) -> None:
            """The ACTIVE animation; with ``?v=<digest>`` that library entry (any visitor may view one: the studio plays them)."""
            from urllib.parse import parse_qs, urlparse
            self._set_cookie = []
            wanted = (parse_qs(urlparse(self.path).query).get("v") or [None])[0]
            if wanted:
                info = sessions.anim.meta(wanted)
                body = sessions.anim.read(wanted) if info else None
            else:
                info = sessions.anim.info()
                body = sessions.anim.read() if info else None
            if body is None:
                return self._json({"error": "no loading animation recorded yet" if not wanted else "no such library entry"}, HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", f'"{info["digest"]}"')
            # the URL carries the digest: immutable.  Without one the answer is whatever is active now (the owner can change it).
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if wanted else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _anim_post(self, length: int) -> None:
            """Accept a loading animation (the raw MLA1 stream the page filtered from its state memory): only when none exists, only a
            valid stream under the size cap, from a same-origin script (the custom header forces a CORS preflight no one answers),
            a few tries per address, one compression at a time.  The octet codec packs it; the packet is what is stored."""
            self._set_cookie = []
            def refuse(message, status):
                self.close_connection = True                       # the body may be unread
                return self._json({"error": message}, status)
            if not access_mod.require_tier(self, "edit"):          # [ACCESS] recording the animation is an owner tool
                return
            if self.headers.get("X-Meowdoku-Loading") != "1":
                return refuse("missing the loading-animation header", HTTPStatus.FORBIDDEN)
            if length <= 0 or length > anim_mod.MAX_BYTES:
                return refuse(f"the animation must be 1..{anim_mod.MAX_BYTES} bytes", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            now, hits = time.time(), sessions.anim_hits.setdefault(self.client_address[0], [])
            hits[:] = [t for t in hits if now - t < 600]
            if len(hits) >= 5:
                return refuse("too many uploads from this address", HTTPStatus.TOO_MANY_REQUESTS)
            hits.append(now)
            raw = self.rfile.read(length)
            if not sessions.anim_busy.acquire(blocking=False):
                return self._json({"error": "an animation is being stored"}, HTTPStatus.CONFLICT)
            try:
                # [STUDIO] a save goes into the LIBRARY; it becomes the one players get only when there is none (activating needs the owner)
                from urllib.parse import unquote
                try:
                    params = json.loads(unquote(self.headers.get("X-Meowdoku-Params") or "")[:anim_mod.MAX_PARAMS_BYTES] or "{}")
                except ValueError:
                    params = {}
                return self._json({"stored": sessions.anim.put(raw, params)}, HTTPStatus.CREATED)
            except anim_mod.AnimError as error:
                return self._json({"error": f"not a valid animation: {error}"}, HTTPStatus.BAD_REQUEST)
            except anim_mod.LibraryFull as error:
                return self._json({"error": str(error)}, HTTPStatus.INSUFFICIENT_STORAGE)
            finally:
                sessions.anim_busy.release()

        # ---- [STUDIO] the library listing (anyone), and the owner's two verbs: make active, delete
        def _library_get(self) -> None:
            self._set_cookie = []
            self._json({"entries": sessions.anim.entries(), "active": sessions.anim.active_digest(), "caps": sessions.anim.caps()})

        def _library_post(self, path: str, length: int) -> None:
            """``/api/loading_library/activate`` and ``/delete`` with ``{"digest": ...}``: what every player sees changes, so the owner's
            lifecycle admin token (header ``X-Meowdoku-Admin``, never a URL; constant-time compare) over HTTPS only, a few wrong tries per
            address.  The server never prints or sends the token."""
            self._set_cookie = []
            def refuse(message, status):
                self.close_connection = True
                return self._json({"error": message}, status)
            lc = sessions.lifecycle
            if lc is None:
                return refuse("owner actions are not available on this server", HTTPStatus.SERVICE_UNAVAILABLE)
            if length <= 0 or length > 1024:
                return refuse("expected {\"digest\": ...}", HTTPStatus.BAD_REQUEST)
            import ssl
            forwarded = any(self.headers.get(h) for h in ("X-Forwarded-For", "Forwarded", "X-Real-IP", "Via"))
            host = (self.headers.get("Host") or "").strip()
            host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
            direct = self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1") and not forwarded and host in ("127.0.0.1", "localhost", "[::1]")
            now, misses = time.time(), sessions.anim_misses.setdefault(self.client_address[0], [])
            misses[:] = [t for t in misses if now - t < 600]
            if len(misses) >= 5:
                return refuse("too many wrong tokens from this address", HTTPStatus.TOO_MANY_REQUESTS)
            # a LOCAL (LAN-origin) visitor is the owner's own network: no token for the studio (the owner asked for it open to creative use);
            # remote visitors still need the token
            self._classify()
            if getattr(self, "_origin", None) == "lan":
                ok, status, why = True, 200, ""
            else:
                ok, status, why = anim_mod.owner_authorize(isinstance(self.connection, ssl.SSLSocket), self.headers.get("X-Forwarded-Proto"), direct,
                                                           self.headers.get("X-Meowdoku-Admin"), lc.token)
            if not ok:
                if status == 401:
                    misses.append(now)
                return refuse(why, status)
            try:
                digest = str(json.loads(self.rfile.read(length) or b"{}").get("digest", ""))
                done = sessions.anim.activate(digest) if path.endswith("/activate") else sessions.anim.delete(digest)
            except KeyError:
                return self._json({"error": "no such library entry"}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError, AttributeError):
                return self._json({"error": "expected {\"digest\": ...}"}, HTTPStatus.BAD_REQUEST)
            print(f"[meowdoku] loading library: {path.rsplit('/', 1)[1]} {done['digest']} from {self.client_address[0]}", flush=True)
            return self._json({"ok": True, "entry": done, "active": sessions.anim.active_digest()})

        # ---- LIFECYCLE (lifecycle.py): the announcement every page reads, the page's copy re-seeding a session, and the admin trigger.
        def _lifecycle_get(self) -> None:
            from http.cookies import SimpleCookie
            from urllib.parse import parse_qs, urlparse
            lc = sessions.lifecycle
            self._set_cookie = []
            out = lc.status()
            query = parse_qs(urlparse(self.path).query)
            if query.get("snap") and not self._is_viewer():
                jar = SimpleCookie(self.headers.get("Cookie", ""))
                out.update(lc.snapshot_for(jar[COOKIE].value if COOKIE in jar else None, (query.get("have") or [None])[0]))
            self._json(out)

        def _lifecycle_post(self, path: str, length: int) -> None:
            lc = sessions.lifecycle
            if path == "/api/admin/lifecycle":
                if not access_mod.require_tier(self, "edit"):      # [ACCESS] the tier is in addition to the token below
                    return
                forwarded = any(self.headers.get(h) for h in ("X-Forwarded-For", "Forwarded", "X-Real-IP", "Via"))
                ok, status, why = lifecycle_mod.admin_authorize(self.client_address[0], self.headers.get("Host", ""), forwarded,
                                                                self.headers.get("X-Meowdoku-Admin"), lc.token)
                if not ok:
                    self.close_connection = True
                    return self._json({"error": why}, status)
                try:
                    return self._json(lc.request_json(json.loads(self.rfile.read(length) or b"{}")))
                except (ValueError, TypeError) as error:
                    return self._json({"error": str(error)}, getattr(error, "status", 400))
            if length > 2 * lifecycle_mod.DUMP_MAX_BYTES:
                self.close_connection = True
                return self._json({"error": "this session copy is too large"}, 413)
            try:
                dump = json.loads(self.rfile.read(length) or b"{}").get("dump")
                table = self._table()
                lifecycle_mod.restore_table(table, dump)
                with table.lock:
                    return self._json(table.state())
            except (ValueError, TypeError, AttributeError) as error:
                return self._json({"error": f"{type(error).__name__}: {error}"}, getattr(error, "status", 400))

        def do_POST(self):  # noqa: N802
            path0 = self.path.split("?", 1)[0]
            self._classify()                                      # [ACCESS]
            if self._tier == "shader":
                self._shader_request(path0, "POST")
                return
            if path0 == "/api/clientlog":                      # CLIENT ERRORS: a page's uncaught error, printed here (phones have no console)
                length = min(int(self.headers.get("Content-Length") or 0), 4096)
                try:
                    note = json.loads(self.rfile.read(length) or b"{}")
                    line = " | ".join(str(note.get(k, ""))[:600] for k in ("kind", "msg", "where", "stack", "ua"))
                except (ValueError, TypeError):
                    line = "(unreadable)"
                print(f"[client-error] {self.client_address[0]}: {line}", flush=True)
                self.send_response(204); self.end_headers()
                return
            if path0 == "/api/loading_anim" and not self._is_viewer():
                return self._anim_post(int(self.headers.get("Content-Length") or 0))
            if path0 in ("/api/loading_library/activate", "/api/loading_library/delete") and not self._is_viewer():     # [STUDIO]
                if not access_mod.require_tier(self, "edit"):          # [ACCESS] the tier is in addition to the owner token checked inside
                    return
                return self._library_post(path0, int(self.headers.get("Content-Length") or 0))
            if path0 in ("/api/admin/lifecycle", "/api/lifecycle/restore") and not self._is_viewer():
                return self._lifecycle_post(path0, int(self.headers.get("Content-Length") or 0))
            if self._is_viewer():
                return self._refuse_viewer()                  # every POST (new, action, profile, share, spectate) is an owner's
            length = int(self.headers.get("Content-Length") or 0)
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/share") or path.startswith("/api/spectate"):
                if length > 2 * share_mod.SHARE_MAX_BYTES:
                    return self._share_error(share_mod.ShareError("too large", 413))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                    if path == "/api/share":
                        if not self._share_rate_ok():
                            raise share_mod.ShareError("too many shares from this address", 429)
                        return self._json({"id": sessions.shares.put(body.get("view"))})
                    self._table()                                # the owner's cookie session names the owner
                    if path == "/api/spectate":
                        op = body.get("op")
                        if op == "start":
                            sessions.spectate.start(self._token)
                        elif op == "stop":
                            sessions.spectate.stop(self._token)
                        elif op != "status":
                            raise share_mod.ShareError("unknown op")
                        return self._json(sessions.spectate.status(self._token))
                    if path == "/api/spectate/key":              # [LIVEVIEW] the owner's fountain position keyframe (base64 MFK1)
                        import base64
                        return self._json({"seq": sessions.spectate.push_key(self._token, base64.b64decode(str(body.get("key", "")), validate=True))})
                    if path == "/api/spectate/live":
                        return self._json({"rev": sessions.spectate.push_live(self._token, body.get("live"))})
                except (ValueError, TypeError) as error:
                    return self._share_error(error)
                return self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path.startswith("/api/action") and str(payload.get("type")) in access_mod.EDIT_ACTIONS and not access_mod.require_tier(self, "edit"):
                return                                            # [ACCESS] process-wide controls (learner, Lichess, pilot weights)
            table = self._table()
            try:
                with table.lock:
                    if self.path.startswith("/api/new"):
                        return self._json(table.new(payload))
                    if self.path.startswith("/api/action"):
                        return self._json(table.action(payload))
                    if self.path.startswith("/api/profile"):
                        return self._json(table.set_player(payload))
            except Exception as error:  # the page shows the refusal
                return self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.BAD_REQUEST)
            return self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

    return Handler


def install_shell(web_dir: Path) -> None:
    """The mobile reference app's wasm loader, which app.js imports: copied before the server listens (a file copy), because the page
    cannot start without it."""
    import shutil
    shell = ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js"
    shutil.copyfile(shell, web_dir / "turing-wasm-shell.js")


def prepare_predictor(web_dir: Path, ctx=None) -> None:
    """The page's ship predictor: ``entity_motion.damped_axis`` lowered to a
    WebAssembly core (rebuilt only when its source changes)."""
    from meowdoku.entity_motion import build_predictor
    try:
        build_predictor(web_dir)
        print("[meowdoku] engine laws for the page: every entity_motion.PREDICTORS core (motion, travel, ..., shot, climb, project, heading, gate, ballistic) ready", flush=True)
    except Exception as error:  # the page plays without prediction
        print(f"[meowdoku] ship predictor NOT built ({type(error).__name__}: {error}); "
              "free-board games run unpredicted", flush=True)
        if ctx is not None:
            ctx.failed("predictor", f"{type(error).__name__}: {error}"[:300])


def _settle_family(ctx, family: str, wasm: bool) -> None:
    """[LOADING] A fountain family's build finished: its GPU product and (unless MEOWDOKU_FOUNTAIN_WASM=0) its CPU fallbacks are ready."""
    if ctx is None:
        return
    if ctx.readiness.state(family) not in startup_mod.SETTLED:
        ctx.ready(family)
    (ctx.ready if wasm else ctx.skipped)(f"{family}_cpu", "" if wasm else "MEOWDOKU_FOUNTAIN_WASM=0")


def _fail_family(ctx, family: str, error) -> None:
    if ctx is None:
        return
    detail = f"{type(error).__name__}: {error}"[:300]
    for name in (family, f"{family}_cpu"):
        if ctx.readiness.state(name) not in startup_mod.SETTLED:
            ctx.failed(name, detail)


def prepare_fountain(web_dir: Path, ctx=None) -> None:
    """The fountain's compute product (the page's generic compute runtime loads it) (``fountain_build.build``):
    the generated WebGPU compute shader and the wasm product for each size, and
    the manifest holding the compiler's deployment verdicts.  Rebuilt only when
    the law or the emitters change."""
    from meowdoku import fountain_build
    try:
        started = time.perf_counter()
        def gpu_ready(manifest):
            routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
            print(f"[meowdoku] fountain WebGPU products ready ({routes}; {time.perf_counter() - started:.1f}s); "
                  f"CPU fallbacks follow in the background", flush=True)
            if ctx is not None:
                ctx.ready("fountain", "WebGPU products")            # [LOADING] the page can run on a WebGPU browser now
        # MEOWDOKU_FOUNTAIN_WASM=0 skips the CPU (wasm) fallbacks entirely: WebGPU products only (a rushed deploy)
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        fountain_build.build(web_dir, wasm=wasm, on_gpu_ready=gpu_ready)
        print(f"[meowdoku] fountain CPU fallbacks {'ready' if wasm else 'skipped (MEOWDOKU_FOUNTAIN_WASM=0)'} "
              f"({time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain", error)


def prepare_fountain_wheel(web_dir: Path, ctx=None) -> None:
    """The 2-D fountain's hinged-body (waterwheel) products, beside the default ones (``fountain_build.build_wheel``)."""
    from meowdoku import fountain_build
    try:
        started = time.perf_counter()
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        manifest = fountain_build.build_wheel(web_dir, wasm=wasm)
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain wheel products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain_wheel", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain wheel products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain_wheel", error)


def prepare_fountain_purrline(web_dir: Path, ctx=None) -> None:
    """The 2-D fountain's purrline sand variant (the occupancy grid and the moving boxes; ``fountain_build.build_purrline``), built after the
    default and wheel products and independent of them: WebGPU products first, the CPU fallbacks too unless MEOWDOKU_FOUNTAIN_WASM=0."""
    from meowdoku import fountain_build
    try:
        started = time.perf_counter()
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        manifest = fountain_build.build_purrline(web_dir, wasm=wasm)
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain purrline products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain_purrline", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain purrline products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain_purrline", error)


def prepare_fountain_mobroad(web_dir: Path, ctx=None) -> None:
    """The mob road's variant of the 2-D fountain (the kill segments, the goal line, the crush colour; ``fountain_build.build_mobroad``), built after the
    default, wheel and purrline products and independent of them: WebGPU products first, the CPU fallbacks too unless MEOWDOKU_FOUNTAIN_WASM=0."""
    from meowdoku import fountain_build
    try:
        started = time.perf_counter()
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        manifest = fountain_build.build_mobroad(web_dir, wasm=wasm)
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain mobroad products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain_mobroad", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain mobroad products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain_mobroad", error)


def prepare_fountain3d(web_dir: Path, ctx=None) -> None:
    """The 3-D fountain's compute product, beside the 2-D one and independent of it (``fountain3d_build.build``)."""
    from meowdoku import fountain3d_build
    try:
        started = time.perf_counter()
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        manifest = fountain3d_build.build(web_dir, wasm=wasm, variant="flat")      # the box and polygons first: the common case
        flat = time.perf_counter() - started
        print(f"[meowdoku] fountain3d flat products ready ({flat:.1f}s); the SDF products follow", flush=True)
        _settle_family(ctx, "fountain3d_flat", wasm)
        manifest = fountain3d_build.build(web_dir, wasm=wasm, variant="sdf")
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain3d products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain3d", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain3d products NOT built ({type(error).__name__}: {error})", flush=True)
        for family in ("fountain3d_flat", "fountain3d"):
            _fail_family(ctx, family, error)


def prepare_fountain3d_wheel(web_dir: Path, ctx=None) -> None:
    """The 3-D fountain's hinged-body (waterwheel) products, after the flat and SDF sets (``fountain3d_build.build(variant="wheel")``)."""
    from meowdoku import fountain3d_build
    try:
        started = time.perf_counter()
        wasm = os.environ.get("MEOWDOKU_FOUNTAIN_WASM", "1") != "0"
        manifest = fountain3d_build.build(web_dir, sizes=fountain3d_build.WHEEL_SIZES, wasm=wasm, variant="wheel")
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain3d wheel products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain3d_wheel", wasm)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain3d wheel products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain3d_wheel", error)


def prepare_fountain3d_toy(web_dir: Path, ctx=None) -> None:
    """The shared oil toy's extruded SDF chamber and free hinged wheel."""
    from meowdoku import fountain3d_build
    try:
        started = time.perf_counter()
        wasm = False  # Startup builds the toy's WebGPU product only; a future fallback/idle path must request Wasm explicitly.
        manifest = fountain3d_build.build(web_dir, sizes=fountain3d_build.WHEEL_SIZES, wasm=wasm, variant="toy")
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain3d toy products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain3d_toy", wasm)
    except Exception as error:
        print(f"[meowdoku] fountain3d toy products NOT built ({type(error).__name__}: {error})", flush=True)
        _fail_family(ctx, "fountain3d_toy", error)


def prepare_policy(web_dir: Path, ctx=None) -> None:
    """The autopilot policy network as a WebAssembly core beside the engine
    laws (``policy_wasm.build_policy``): ``policy.wasm`` + ``policy.abi.json``,
    with the live weights as ``policy.weights.json`` for the page to feed.
    The MLP architecture only: the token network does not lower yet."""
    from meowdoku import autopilot
    from meowdoku.policy_wasm import build_policy
    for name in autopilot.GAMES:
        if autopilot.GAME_ARCH.get(name) != "mlp":
            continue
        try:
            params = autopilot.live_params(name)
            if params is None:
                continue
            build_policy(web_dir, params)
            print(f"[meowdoku] autopilot policy for the page ({name}): policy.wasm + policy.weights.json ready", flush=True)
        except Exception as error:  # the page plays without it
            print(f"[meowdoku] autopilot policy NOT built for {name} ({type(error).__name__}: {error})", flush=True)
            if ctx is not None:
                ctx.failed("policy", f"{name}: {type(error).__name__}: {error}"[:300])


def prepare_graph_products(web_dir: Path, ctx=None) -> None:
    """The bullet-hell law and the token pilot as buffer-table WebAssembly
    products (``graph_wasm.py``): ``law.wasm`` + ``law.abi.json`` and
    ``pilot.wasm`` + ``pilot.abi.json`` + ``pilot.weights.json``, rebuilt only
    when their source digest changes.  The page loads them
    (``window.meowdokuDebug.lawWasm`` / ``.pilotWasm``); nothing pilots or
    predicts with them yet."""
    from meowdoku import graph_wasm
    for label, build in (("law", graph_wasm.build_law), ("pilot", graph_wasm.build_pilot)):
        try:
            started = time.perf_counter()
            abi = build(web_dir)
            print(f"[meowdoku] page {label} graph product ready ({len(abi['order'])} buffers, "
                  f"{time.perf_counter() - started:.1f}s)", flush=True)
        except Exception as error:  # the page plays without it
            print(f"[meowdoku] page {label} graph product NOT built ({type(error).__name__}: {error})", flush=True)
            if ctx is not None:
                ctx.failed("graph", f"{label}: {type(error).__name__}: {error}"[:300])


def predictor_stems() -> tuple[str, ...]:
    from meowdoku.entity_motion import PREDICTORS
    return tuple(p[3] for p in PREDICTORS)


def prepare_autopilot(args, ctx=None) -> None:
    from meowdoku import autopilot
    print(f"[meowdoku] autopilot weights loaded: {', '.join(autopilot.preload()) or 'none (train with autopilot.py)'}", flush=True)
    autopilot.load_learning()
    from meowdoku import idle_trainer
    if not args.no_idle_train:
        idle_trainer.start_default()                    # studies recorded play whenever the server is idle


def prepare_lanes(web_dir: Path, args) -> dict:
    """[LOADING] The prepare steps as lanes (startup.Lanes): each lane runs its steps in order on its own thread, the lanes side by side.
    ``laws``: the page's engine laws first (the free-board games need them), then the inspection-only policy and graph products;
    ``fountain``: the fountain families as they always ran (GPU products before CPU fallbacks within each); ``autopilot``: the weights."""
    step = lambda products, fn: (products, fn)                                  # noqa: E731
    both = lambda family: (family, f"{family}_cpu")                             # noqa: E731
    return {
        "laws": [step(("predictor",), lambda c: prepare_predictor(web_dir, c)),
                 step(("policy",), lambda c: prepare_policy(web_dir, c)),
                 step(("graph",), lambda c: prepare_graph_products(web_dir, c))],
        "fountain": [step(both("fountain"), lambda c: prepare_fountain(web_dir, c)),
                     step(both("fountain_wheel"), lambda c: prepare_fountain_wheel(web_dir, c)),
                     step(both("fountain_purrline"), lambda c: prepare_fountain_purrline(web_dir, c)),
                     step(both("fountain_mobroad"), lambda c: prepare_fountain_mobroad(web_dir, c)),
                     step(both("fountain3d_flat") + both("fountain3d"), lambda c: prepare_fountain3d(web_dir, c)),
                     step(both("fountain3d_wheel"), lambda c: prepare_fountain3d_wheel(web_dir, c)),
                     step(both("fountain3d_toy"), lambda c: prepare_fountain3d_toy(web_dir, c))],
        "autopilot": [step(("autopilot",), lambda c: prepare_autopilot(args))],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--web-dir", default=str(WEB_DIR))
    parser.add_argument("--profiles", default=str(PROFILES),
                        help="profile/score file; a second (dev) server keeps its own so it never clobbers the live one")
    parser.add_argument("--tls-cert", default=None,
                        help="PEM certificate: serve https/wss (browsers expose WebGPU only to secure pages)")
    parser.add_argument("--tls-key", default=None, help="PEM private key for --tls-cert")
    parser.add_argument("--access", default=None,
                        help="access tier policy file (default <web-dir>/../access.json; see access.py)")
    parser.add_argument("--no-wasm", action="store_true",
                        help="build the fountains' GPU products only (skip the slow wasm fallback compiles)")
    parser.add_argument("--no-idle-train", action="store_true",
                        help="do not run the background trainer that studies recorded play when the server is idle")
    args = parser.parse_args(argv)
    web_dir = Path(args.web_dir)
    if not (web_dir / "boards.json").exists():
        print(f"no export at {web_dir}; run examples/meowdoku/web_export.py first", flush=True)
        return 1
    # [LOADING] BIND FIRST.  Nothing below the listening socket waits on a compile: the prepare steps run in background lanes
    # (startup.py), each product reporting its state to /api/ready; a game's page starts when ITS products are ready (the Meowdoku
    # puzzle, chess, ... need none: playable at once) and a request for a file that is still building waits for it.
    install_shell(web_dir)
    if args.no_wasm:
        os.environ["MEOWDOKU_FOUNTAIN_WASM"] = "0"                   # the same switch as MEOWDOKU_FOUNTAIN_WASM=0: WebGPU products only
    global COOKIE, PLAYER_COOKIE
    if args.port != 8766:
        COOKIE = f"meowdoku_session_{args.port}"
        PLAYER_COOKIE = f"meowdoku_player_{args.port}"
    logring_mod.RING = logring_mod.Ring(path=web_dir.parent / "log_ring.jsonl"); logring_mod.tee_stdout()   # [LOG]
    sessions = Sessions(web_dir, Profiles(Path(args.profiles)))
    if args.access:
        sessions.access = access_mod.Access(Path(args.access))
    sessions.access.start()                                          # [ACCESS] policy file + the server's own public IP (background lookup)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(sessions))
    lc = lifecycle_mod.Lifecycle(sessions, web_dir.parent, args.port, lambda: Table(web_dir, sessions.profiles))     # LIFECYCLE
    lc.server, sessions.lifecycle = server, lc
    sessions.readiness.predictor_stems = predictor_stems()
    lanes = prepare_lanes(web_dir, args)
    startup_mod.Lanes(sessions.readiness).declare(lanes)            # every product is visible as "pending" from the first request
    lifecycle_mod.record_launch(lc.paths, args.port, args.bind, web_dir)
    threading.Thread(target=lc.run, name="lifecycle", daemon=True).start()
    scheme = "http"
    if args.tls_cert:
        import ssl
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        # The handshake runs on the first read, in the request's own thread,
        # so a slow or refusing client never stalls the accept loop.
        server.socket = context.wrap_socket(server.socket, server_side=True,
                                            do_handshake_on_connect=False)
        scheme = "https"
    print(f"[meowdoku] serving {web_dir} at {scheme}://{args.bind}:{args.port}/  ({sessions.board_count} prebaked boards; chess, checkers, wordle)", flush=True)
    startup_mod.Lanes(sessions.readiness).start(lanes)               # [LOADING] the builds begin once the socket is accepting
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lc.stopping = True
        if not lc.exit_code:
            lc.save_all()                                   # LIFECYCLE: a Ctrl-C keeps the sessions too
        server.server_close()
    return lc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
