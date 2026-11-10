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
    POST /api/account/signup {"username","password"}  -> create a real login, own this browser's content
    POST /api/account/login  {"username","password"}  -> real login (accounts.py; separate from /api/profile)
    POST /api/account/logout {}
    GET  /api/account/me                   -> the logged-in account, or null
    GET  /api/account/assets               -> this account's owned marble designs/photos + migrated profile

Finished games (``Game.result()``) are recorded on the session's profile
(``profiles.py``); every state carries ``player`` and that game's ``record``.

Every call returns the game's state (see ``games.py``); the page animates
the difference between states.  Meowdoku runs the dt-system law through
``MeowdokuSession``; the server never renders anything.
"""
from __future__ import annotations

import argparse
import base64
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
ROOT = HERE.parents[1]
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
from meowdoku import tickets as tickets_mod  # noqa: E402  [TICKETS] persisted client/server diagnostic reports
from meowdoku import startup as startup_mod  # noqa: E402  [LOADING] (bind first; per-product readiness)
from meowdoku import loading_anim as anim_mod  # noqa: E402  [LOADING] (the reduced loading animation's store and validator)
from meowdoku import obj_assets as obj_assets_mod
from meowdoku import marble_objects as marble_objects_mod  # noqa: E402  [MARBLE] immutable saved designs
from meowdoku import marble_photo as marble_photo_mod  # noqa: E402  [MARBLE PHOTO] Spectral worker process
from meowdoku import access as access_mod  # noqa: E402  [ACCESS] (origin -> tier: lan/dns/ip -> edit/dev/play/shader)
from meowdoku import login_guard as login_guard_mod  # noqa: E402  [LOGIN] brute-force limits for /api/account/login
from meowdoku import accounts as accounts_mod  # noqa: E402  [ACCOUNTS] real password login, SQLite-backed, separate from profiles.py
from meowdoku import proxy_trust as proxy_trust_mod  # noqa: E402  [PROXY] client origin/scheme from a trusted reverse proxy only
from meowdoku import authz as authz_mod  # noqa: E402  [AUTHZ] roles + capabilities over the origin tiers
from meowdoku import lowering_library as lowering_mod  # noqa: E402  [LOWERINGS] catalogued, leased lowerings of each core
from meowdoku import math_jobs as math_jobs_mod         # noqa: E402  [MATHLAB] the math-lab job table and its one worker
from meowdoku.state_paths import StatePaths  # noqa: E402  [STATE] every state path (--state-dir; default = the old build/meowdoku layout)

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
ACCOUNT_COOKIE = "meowdoku_account"    # a real login session token (accounts.py) -- distinct from
                                       # both COOKIE (game-table) and PLAYER_COOKIE (passwordless name)


class FederationUnavailable(Exception):
    """[FEDERATE] this server fronts no Nodus instance (or its canvas is missing); the message is the refusal reason."""


class Sessions:
    """session token -> that client's own Table; idle ones expire."""

    def __init__(self, web_dir: Path, profiles: Profiles | None = None, state: StatePaths | None = None):
        self.web_dir = web_dir
        self.profiles = profiles
        self.state = state or StatePaths.legacy_at(web_dir.parent)      # [STATE] no --state-dir: everything beside web/, as before
        self.lock = threading.Lock()
        self.tables: dict[str, list] = {}       # token -> [Table, last_seen]
        self.shares = share_mod.ShareStore(self.state.shares_dir)     # build/meowdoku/shares (gitignored with build/)
        self.spectate = share_mod.Spectate()                              # owner session -> its view-only live share
        self.share_hits: dict[str, list] = {}                             # client address -> recent share POST times
        boards = json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))
        self.board_count = len(boards["boards"])
        self.lifecycle = None                   # LIFECYCLE: set by main(); restores a saved session for a cookie that has a file
        self.readiness = startup_mod.Readiness()          # [LOADING] what is built, building, pending (set up by main(); an empty one never holds a request)
        self.anim = anim_mod.Store(self.state.content_dir)        # [LOADING] build/meowdoku/loading_anim.mla
        self.marble_objects = marble_objects_mod.Store(self.state.marble_objects_dir)
        self.marble_photos = marble_photo_mod.PhotoJobs(self.state.marble_photos_dir, ROOT.parent / "spectral-analyzer")
        from meowdoku.client_ray_process_bridge import ClientRayProcessBridge
        self.client_ray_bridge = ClientRayProcessBridge()
        self.tickets = tickets_mod.Store(self.state.tickets_dir)
        self.anim_busy = threading.Lock()
        self.anim_hits: dict[str, list] = {}
        self.access = access_mod.Access(self.state.access_json)   # [ACCESS] build/meowdoku/access.json (gitignored); main() may point it elsewhere
        self.anim_misses: dict[str, list] = {}            # [STUDIO] wrong owner tokens per address
        self.accounts = accounts_mod.Accounts(self.state.accounts_db)   # [ACCOUNTS] real password login
        self.accounts_db = self.state.accounts_db
        self.mathlab_lock = threading.Lock()
        self.mathlab_worker = None                                      # [MATHLAB] set by main() unless --no-mathlab
        self.authz_policy = authz_mod.Policy.beside(self.accounts_db)           # [AUTHZ] authz_policy.json beside accounts.db
        self.instance = ""                                                      # [INSTANCE] --instance NAME (account_instances binding)
        self.route = ""                                                         # [PROXY] --route NAME: balancer stickiness cookie value
        self.obj_assets = obj_assets_mod.Store(self.state.obj_assets_dir, self.accounts, ROOT.parent / "spectral-analyzer")
        self.marble_objects.obj_assets = self.obj_assets

    # ---- [FEDERATE begin] what this instance promises a federating peer (see _contract_socket)
    federation_port: int | None = None          # --federation-port: the local Nodus host's 127.0.0.1 contract port (not read from the supervisor yet)
    federation_canvas: str | None = None        # --federation-canvas: the running Nodus instance's canvas file

    def federation_identity(self) -> dict:
        """{"graph_definition_digest", "api_digests", "port"} of the Nodus instance this server fronts: sha256 of its canvas file and the
        sorted ``module:entry`` pairs of every ``*.api.yaml`` under web_dir (federation_peer.identity_digests, the same function the peer
        runs).  Raises FederationUnavailable with a reason when no instance is configured (no --federation-port/--federation-canvas)."""
        from meowdoku import federation_peer
        if not self.federation_port or not self.federation_canvas:
            raise FederationUnavailable("no Nodus instance is configured for federation (--federation-port and --federation-canvas)")
        if not Path(self.federation_canvas).is_file():
            raise FederationUnavailable("the configured Nodus canvas file is missing")
        return {**federation_peer.identity_digests(self.federation_canvas, self.web_dir), "port": int(self.federation_port)}
    # ---- [FEDERATE end]

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

        def _job_identity(self):
            import hashlib
            table = self._table()
            account = self._account()
            # A shared browser is not permission to access another login's jobs.
            # Anonymous jobs stay browser-owned; migration is not implicit.
            keys = (["account:" + str(account["id"])] if account is not None else
                    ["browser:" + hashlib.sha256((table.player_token or self._token).encode()).hexdigest()])
            return keys, account

        def _job_authorize(self, digest):
            from meowdoku import job_library
            keys, account = self._job_identity()
            if not job_library.allowed(sessions.marble_photos.directory, digest, keys):
                raise marble_photo_mod.PhotoError("job belongs to another identity", 403)

        def _is_secure(self) -> bool:
            import ssl
            if not hasattr(self, "_scheme"):
                self._classify()
            return isinstance(self.connection, ssl.SSLSocket) or self._scheme == "https"

        def _account(self) -> dict | None:
            """The logged-in account for this request (accounts.py), or None.
            Reads ACCOUNT_COOKIE only -- never issues one itself; that only
            happens on a successful /api/account/signup or /api/account/login."""
            from http.cookies import SimpleCookie
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            token = jar[ACCOUNT_COOKIE].value if ACCOUNT_COOKIE in jar else None
            return sessions.accounts.user_for_token(token)

        def _set_account_cookie(self, token: str) -> None:
            if not getattr(self, "_set_cookie", None):
                self._set_cookie = []
            secure = "; Secure" if self._is_secure() else ""
            self._set_cookie.append(
                f"{ACCOUNT_COOKIE}={token}; Path=/; Max-Age={accounts_mod.SESSION_SECONDS}; HttpOnly; SameSite=Lax{secure}")

        def _clear_account_cookie(self) -> None:
            if not getattr(self, "_set_cookie", None):
                self._set_cookie = []
            self._set_cookie.append(f"{ACCOUNT_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")

        # ---- [ACCESS] origin and tier, from the client address and the Host header.  The client address is the socket peer, except when the
        # peer is a trusted proxy (proxy_trust.py), whose X-Forwarded-For names the real client; a header from any other peer is never read.
        def _classify(self) -> None:
            client_ip, self._scheme, self._via_proxy = proxy_trust_mod.client_origin(self.client_address[0], self.headers)
            self._client_ip = client_ip
            self._origin, self._origin_tier = sessions.access.decide(client_ip, self.headers.get("Host"))
            self._tier = self._origin_tier
            self._principal_cache = None
            if self._origin_tier == "shader":                    # [LOGIN] a signed-in account lifts the shader lock; its caps still come from its role
                account = self._account()
                lift = {"member": "play", "service": "play", "operator": "dev", "admin": "edit"}
                if account is not None:
                    self._tier = lift.get(authz_mod.role_of(sessions.accounts_db, account["id"]), "shader")

        # ---- [AUTHZ] the request's principal and its capabilities (authz.py); built lazily after _classify
        def _principal(self):
            if getattr(self, "_principal_cache", None) is None:
                self._principal_cache = authz_mod.principal_from(self._account(), self._origin, self._tier, sessions.accounts_db)
            return self._principal_cache

        def _caps(self) -> frozenset:
            return authz_mod.capabilities(self._principal(), sessions.authz_policy.get())

        def _require(self, cap: str) -> bool:
            """True when the request holds ``cap``; otherwise a 403 is sent and False returned (same contract as access.require_tier)."""
            self._classify()
            try:
                authz_mod.require(self._principal(), cap, sessions.authz_policy.get())
            except authz_mod.AuthzError as error:
                self.close_connection = True
                self._set_cookie = []
                self._json({"error": error.message}, error.status)
                return False
            return True

        def _body_json(self, limit: int = 4096):
            """The request body as a JSON object, or None after a 400 was sent."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= limit:
                    raise ValueError("body size is invalid")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                return body
            except (ValueError, TypeError) as error:
                self.close_connection = True
                self._set_cookie = []
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return None

        def _login_post(self) -> None:
            """[LOGIN] POST /api/account/login: JSON only, rate-limited per client/username (login_guard.py), uniform failure body."""
            self._set_cookie = []
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                self.close_connection = True
                return self._json({"error": "expected application/json"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > 1024:
                return self._json({"error": "expected {\"username\", \"password\"}"}, HTTPStatus.BAD_REQUEST)
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("expected an object")
                username, password = str(body.get("username", "")), str(body.get("password", ""))
            except (ValueError, TypeError):
                return self._json({"error": "expected {\"username\", \"password\"}"}, HTTPStatus.BAD_REQUEST)
            client = self._client_ip
            db = login_guard_mod.connect(sessions.accounts_db)
            try:
                wait = login_guard_mod.check(db, client, username)
                if wait:
                    out = json.dumps({"error": "too many attempts", "retry_after": int(wait)}).encode("utf-8")
                    self.send_response(HTTPStatus.TOO_MANY_REQUESTS)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Retry-After", str(int(wait)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                try:
                    user = sessions.accounts.login(username, password)
                except accounts_mod.AccountError:                 # wrong password, unknown user and a malformed name look the same
                    login_guard_mod.record(db, client, username, False)
                    time.sleep(0.25)
                    return self._json({"error": "wrong username or password"}, HTTPStatus.UNAUTHORIZED)
                login_guard_mod.record(db, client, username, True)
            finally:
                db.close()
            self._bind_instance(user["id"])
            self._set_account_cookie(sessions.accounts.create_session(user["id"]))
            from http.cookies import SimpleCookie
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            held = jar[PLAYER_COOKIE].value if PLAYER_COOKIE in jar else None
            name = sessions.profiles.by_token(held) if held else None
            if name:
                sessions.accounts.claim(user["id"], "profile", name)
            return self._json({"account": user})

        def _shader_request(self, path: str, method: str) -> bool:
            """The shader tier gets the shader page and its few assets, nothing else.  True when the request was answered here."""
            if method == "POST" and path in access_mod.SHADER_LOGIN_POSTS:
                return False                                   # [LOGIN] served by do_POST's login/logout branches
            if method == "POST" and path == "/api/account/signup":
                self.close_connection = True
                self._set_cookie = []
                self._json({"error": "sign-up is closed here"}, HTTPStatus.FORBIDDEN)
                return True
            if method != "GET":
                self.close_connection = True
                self._set_cookie = []
                self._json({"error": "shader tier"}, HTTPStatus.FORBIDDEN)
                return True
            if path in access_mod.SHADER_APIS or path in access_mod.SHADER_LOGIN_GETS:
                return False                                   # served below (/api/ready is reduced for this tier)
            if path.startswith("/api/") or path.startswith("/ws"):
                self._set_cookie = []
                self._json({"error": "shader tier"}, HTTPStatus.FORBIDDEN)
                return True
            if path not in access_mod.SHADER_FILES and path not in access_mod.SHADER_LOGIN_FILES:
                self.path = access_mod.SHADER_PAGE                # any other page address is the shader page
            super().do_GET()
            return True

        def _log_get(self) -> None:                               # [LOG]
            from urllib.parse import parse_qs
            if not self._require("dev_ui"):
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

        def _instance_of(self, user_id) -> str | None:
            """The service instance a user is bound to (account_instances, written by bind below / deploy/apache_supervisor.py)."""
            if user_id is None:
                return sessions.instance or None
            import sqlite3
            conn = sqlite3.connect(str(sessions.accounts_db), timeout=10)
            try:
                row = conn.execute("SELECT instance FROM account_instances WHERE user_id = ?", (user_id,)).fetchone()
            except sqlite3.OperationalError:
                row = None
            finally:
                conn.close()
            return row[0] if row else (sessions.instance or None)

        def _bind_instance(self, user_id: int) -> None:
            if not sessions.instance:
                return
            import sqlite3
            authz_mod._connect(sessions.accounts_db).close()          # authz.py owns the account_instances DDL
            conn = sqlite3.connect(str(sessions.accounts_db), timeout=10)
            try:
                conn.execute("INSERT OR IGNORE INTO account_instances (user_id, instance) VALUES (?, ?)", (user_id, sessions.instance))
                conn.commit()
            finally:
                conn.close()

        def _user_list(self) -> list:
            import sqlite3
            conn = sqlite3.connect(str(sessions.accounts_db), timeout=10)
            try:
                try:
                    rows = conn.execute("SELECT u.username, i.instance FROM users u LEFT JOIN account_instances i ON i.user_id = u.id "
                                        "ORDER BY u.username").fetchall()
                except sqlite3.OperationalError:
                    rows = conn.execute("SELECT username, NULL FROM users ORDER BY username").fetchall()
                return [{"username": r[0], "instance": r[1]} for r in rows]
            finally:
                conn.close()

        def _admin_list(self) -> list:
            import sqlite3
            authz_mod._connect(sessions.accounts_db).close()          # make sure account_roles exists
            conn = sqlite3.connect(str(sessions.accounts_db), timeout=10)
            try:
                return [{"username": r[0], "granted_by": r[1], "created_at": r[2]} for r in conn.execute(
                    "SELECT u.username, r.granted_by, r.created_at FROM account_roles r JOIN users u ON u.id = r.user_id "
                    "WHERE r.role = 'admin' ORDER BY u.username")]
            finally:
                conn.close()

        def _lowering_blob(self, sha: str) -> None:
            """Stream a stored lowering's bytes to the account that holds an open, unexpired lease on that exact blob."""
            if not self._require("studio_write"):
                return
            self._set_cookie = []
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                return self._json({"error": "expected a sha256"}, HTTPStatus.BAD_REQUEST)
            key = self._job_identity()[0][0]
            leases = lowering_mod._leases(sessions.accounts_db, "WHERE l.returned IS NULL AND l.product_sha256 = ? AND l.account_key = ? "
                                          "AND l.deadline > ?", (sha, key, time.time()))
            data = lowering_mod.blob_bytes(sessions.accounts_db, sha) if leases else None
            if data is None:
                return self._json({"error": "no open lease for that blob"}, HTTPStatus.FORBIDDEN)
            ctype = {"wasm": "application/wasm", "json": "application/json", "wgsl": "text/plain; charset=utf-8",
                     "c": "text/plain; charset=utf-8", "ll": "text/plain; charset=utf-8"}.get(leases[0].kind, "application/octet-stream")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _whoami(self) -> None:
            self._set_cookie = []
            caps = self._caps()
            self._json({"origin": self._origin, "tier": self._tier,
                        "dev": "dev_ui" in caps, "edit": "owner_tools" in caps,
                        "shader": self._tier == "shader", "role": self._principal().role, "capabilities": sorted(caps),
                        "instance": sessions.instance or None,
                        "signed_in": self._principal().user_id is not None, "username": self._principal().username,
                        "origin_tier": self._origin_tier})

        # ---- [MATHLAB] POST/GET /api/mathlab/jobs...: the math lab's job endpoints (math_jobs.py; the page is web/mathlab.html).
        # Every endpoint needs compile_submit (members and up; no tier grants it).  A job is visible to its owner only (404 otherwise).
        _MATHLAB_FILES = ("law.at.py", "law.numpy.py", "law.abi.json", "law.wgsl.abi.json", "law.api.yaml")
        _MATHLAB_ATTACHMENTS = ("law.numpy.py", "law.at.py")
        _MATHLAB_BROWSER_LIMIT = 4096

        def _mathlab_served_names(self, directory: Path) -> set:
            """The files a job may serve: the fixed list plus every file products.json names, never the native library or piece."""
            names = set(self._MATHLAB_FILES)
            try:
                record = json.loads((directory / "products.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            native = record.get("targets", {}).get("native", {}) if isinstance(record, dict) else {}
            refused = {str(native.get(k)) for k in ("file", "piece_file") if native.get(k)}
            listed = [v for v in (record.get("files") or {}).values()] if isinstance(record, dict) else []
            for target in (record.get("targets") or {}).values() if isinstance(record, dict) else ():
                listed += [target.get(k) for k in ("file", "abi_file", "api_file") if isinstance(target, dict)]
            for name in listed:
                if isinstance(name, str) and name not in refused and not name.startswith("native/") \
                        and not name.endswith((".dll", ".so", ".dylib", ".piece")):
                    names.add(name)
            names.add("products.json")
            names.add("profile.server.json")
            names.add("profile.browser.json")
            return {n for n in names if "/" not in n and "\\" not in n and not n.endswith((".dll", ".so", ".dylib", ".piece"))}

        def _mathlab_get(self, path: str) -> None:
            import re as _re
            from urllib.parse import unquote
            if not self._require("compile_submit"):
                return
            self._set_cookie = []
            owner = self._job_identity()[0][0]
            self._set_cookie = []
            parts = [unquote(x) for x in path[len("/api/mathlab/"):].split("/")]
            db = math_jobs_mod.connect(sessions.accounts_db)
            try:
                if parts == ["jobs"]:
                    return self._json({"jobs": math_jobs_mod.list_jobs(db, owner)})
                if len(parts) >= 2 and parts[0] == "jobs" and _re.fullmatch(r"[0-9a-f]{16}", parts[1] or ""):
                    job = math_jobs_mod.get_job(db, sessions.state, parts[1], owner)
                    if job is None:
                        return self._json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
                    if len(parts) == 2:
                        return self._json(job)
                    if len(parts) == 4 and parts[2] == "file":
                        directory = math_jobs_mod.job_dir(sessions.state, parts[1])
                        name = parts[3]
                        target = directory / name
                        if name not in self._mathlab_served_names(directory) or not target.is_file():
                            return self._json({"error": "no such file"}, HTTPStatus.NOT_FOUND)
                        data = target.read_bytes()
                        ctype = ("application/wasm" if name.endswith(".wasm") else "application/json" if name.endswith(".json")
                                 else "text/plain; charset=utf-8")
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "no-store")
                        if name in self._MATHLAB_ATTACHMENTS:
                            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                        self.end_headers()
                        self.wfile.write(data)
                        return
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            finally:
                db.close()

        def _mathlab_post(self, path0: str) -> None:
            import math as _math
            import re as _re
            if not self._require("compile_submit"):
                return
            owner = self._job_identity()[0][0]
            self._set_cookie = []
            body = self._body_json(8192)
            if body is None:
                return
            db = math_jobs_mod.connect(sessions.accounts_db)
            try:
                if path0 == "/api/mathlab/jobs":
                    text, targets = body.get("text"), body.get("targets")
                    if not isinstance(text, str) or not (targets is None or isinstance(targets, list)):
                        return self._json({"error": "expected {text: string, targets?: [names]}"}, HTTPStatus.BAD_REQUEST)
                    try:
                        result = math_jobs_mod.submit(db, sessions.state, owner, text, targets)
                    except math_jobs_mod.LimitError as error:
                        return self._json({"error": error.message}, error.status)
                    except ValueError as error:
                        return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                    if sessions.mathlab_worker is not None:
                        sessions.mathlab_worker.wake()
                    return self._json(result, HTTPStatus.UNPROCESSABLE_ENTITY if result["state"] == "rejected" else HTTPStatus.OK)
                match = _re.fullmatch(r"/api/mathlab/jobs/([0-9a-f]{16})/browser_profile", path0)
                if match:
                    job = math_jobs_mod.get_job(db, sessions.state, match.group(1), owner)
                    if job is None:
                        return self._json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
                    if job["state"] != "complete":
                        return self._json({"error": "the job has not completed"}, HTTPStatus.CONFLICT)

                    def finite(key, low=0.0, optional=False):
                        value = body.get(key)
                        if value is None and optional:
                            return None
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not _math.isfinite(value) or value < low:
                            raise ValueError(f"{key} must be a finite number >= {low}")
                        return float(value)
                    try:
                        target = body.get("target")
                        if target not in ("wasm", "wgsl"):
                            raise ValueError("target must be wasm or wgsl")
                        rows = finite("rows", 1.0)
                        if rows > 1e9:
                            raise ValueError("rows is out of range")
                        entry = {"target": target, "rows": int(rows), "ns_per_row": finite("ns_per_row"),
                                 "max_abs_err": finite("max_abs_err", optional=True),
                                 "user_agent_hint": str(body.get("user_agent_hint") or "")[:200], "recorded": time.time()}
                    except ValueError as error:
                        return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                    with sessions.mathlab_lock:
                        path = math_jobs_mod.job_dir(sessions.state, match.group(1)) / "profile.browser.json"
                        try:
                            held = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                        except (OSError, ValueError):
                            held = {}
                        results = [r for r in held.get("results", []) if r.get("target") != target] + [entry]
                        path.write_text(json.dumps({"schema": "turing.math-lab-browser-profile.v1", "results": results}, indent=1),
                                        encoding="utf-8")
                    return self._json({"ok": True, "results": results})
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            finally:
                db.close()

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
            cookies = getattr(self, "_set_cookie", None) or ()
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
            if cookies and sessions.route:
                self.send_header("Set-Cookie", proxy_trust_mod.route_cookie(sessions.route))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            # A malformed request (e.g. TLS bytes on this http port) is answered
            # before the request line is parsed, so there may be no path yet.
            if not getattr(self, "path", "").startswith("/api/"):
                path = getattr(self, "path", "").split("?", 1)[0]
                self.send_header("Cache-Control", "no-store" if path in
                                 ("/marble_studio.html", "/marble_studio.js", "/marble_drive_input.js", "/marble_chase_camera.js", "/marble_studio_scene.js", "/marble_studio_transition.js", "/marble_studio_phong.js", "/marble_studio_dof.js", "/marble_camera_toolbox.js", "/marble_camera_toolbox_mount.js", "/marble_spot_sample.js", "/marble_rangefinder.js", "/marble_rangefinder_panel.js", "/marble_camera_binding.js", "/marble_object.js",
                                  "/marble_mesh.js", "/dice_mesh.js", "/marble_photo.js", "/marble_live.js", "/marble_downloads.js", "/marble_autosave.js", "/marble_object_snapshot.js", "/marble_imported_object.js", "/marble_geometry_policy.js", "/marble_film_response.js", "/marble_film_science.js",
                                  "/marble_room.js", "/marble_studio_lighting.js", "/marble_light_strengths.js", "/marble_backdrop.js",
                                  "/marble_glass_preview.js", "/marble_materials.js", "/marble_studio_surfaces.js",
                                  "/marble_studio_table.js", "/marble_studio_scale.js", "/marble_table_lights.js", "/marble_bubbles.js",
                                  "/marble_studio_gobos.js", "/marble_lighting_profiles.js", "/particle_styles.js",
                                  "/marble_optical_authoring.js", "/marble_studio_optical.js", "/marble_optical_diagnostic.js", "/marble_lens_controls.js")
                                 else "no-cache")  # actively edited pages must revalidate
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

        # ---- [FEDERATE begin] ``/ws/contract/<prefix>``: a remote Nodus instance joins this instance's loopback contract port.
        def _contract_socket(self, prefix: str) -> None:
            """Bridge one WebSocket to the local contract port (STATE_AND_FEDERATION_2026-11-10.md section 4, phase 1).

            CREDENTIAL.  The peer GRANT (authz.peer_grants; grant_id + secret) IS the credential -- service to service, no login cookie.
            The route still sits behind the normal access tier: a shader-tier origin is refused by do_GET's shader gate before this
            runs, and any other origin must hold ``view_site``.  Viewer links never reach it.  The grant names the prefix, the Nodus role
            (host=participant 1 / engine=producer 2), the frame kinds the peer may send, an expiry, and the local SERVICE account whose
            id becomes Hello.id (the peer cannot choose it).
            FLOW.  1) the peer's first message is a JSON promise {grant_id, secret, direction, graph_definition_digest, api_digests};
            2) the grant is verified, then the peer's digests must equal ``Sessions.federation_identity()`` (the local instance: canvas
            sha256 + api.yaml module:entry pairs); the answer is {"ok":true,"hello_id":N} or {"ok":false,"reason":...} and a refusal
            closes (1008); 3) the server connects to 127.0.0.1:<--federation-port>, writes the Hello itself and relays.  Binary messages
            only, exactly 72 bytes each; a frame whose kind is outside the grant's allowed_ops (or Hello, or malformed) closes 1008.
            Either side closing closes the other."""
            import socket
            import hmac
            if not self._require("view_site"):
                return
            key = self.headers.get("Sec-WebSocket-Key")
            if not key or "websocket" not in (self.headers.get("Upgrade") or "").lower():
                return self._json({"error": "expected a WebSocket upgrade"}, HTTPStatus.BAD_REQUEST)
            self._set_cookie = []
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key(key))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            wlock, done, upstream = threading.Lock(), threading.Event(), None

            def send(opcode: int, payload: bytes) -> None:
                with wlock:
                    self.wfile.write(frame(opcode, payload))
                    self.wfile.flush()

            def close(code: int, reason: str = "") -> None:
                try:
                    send(0x8, struct.pack(">H", code) + reason.encode("utf-8")[:100])
                except (OSError, ValueError):
                    pass

            def refuse(reason: str, code: int = 1008) -> None:
                try:
                    send(0x1, json.dumps({"ok": False, "reason": reason}).encode("utf-8"))
                except (OSError, ValueError):
                    pass
                close(code)

            def read(limit: int):
                """The next data message (opcode, payload) within ``limit`` bytes, answering pings; None on close/EOF; "big" past the limit."""
                while True:
                    head = self.rfile.read(2)
                    if len(head) < 2:
                        return None
                    fin, op, length = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x7F
                    if length == 126:
                        length = struct.unpack(">H", self.rfile.read(2))[0]
                    elif length == 127:
                        length = struct.unpack(">Q", self.rfile.read(8))[0]
                    if length > limit or not fin:
                        return "big"
                    mask = self.rfile.read(4) if head[1] & 0x80 else b""
                    data = self.rfile.read(length)
                    if len(data) < length:
                        return None
                    if mask:
                        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
                    if op == 0x8:
                        return None
                    if op == 0x9:
                        send(0xA, data)
                        continue
                    if op == 0xA:
                        continue
                    return op, data

            try:
                first = read(16384)
                if first is None:
                    return
                if first == "big" or first[0] not in (0x1, 0x2):
                    return refuse("the first message must be the JSON promise (at most 16 KiB)")
                try:
                    promise = json.loads(first[1].decode("utf-8"))
                    direction = str(promise["direction"])
                    grant = authz_mod.verify_peer(sessions.accounts_db, promise["grant_id"], promise["secret"], prefix, direction)
                except (ValueError, KeyError, TypeError):
                    return refuse("malformed promise")
                if grant is None:
                    print(f"[federation] refused grant for prefix {prefix!r} ({self.client_address[0]})", flush=True)
                    return refuse("grant refused")
                try:
                    local = sessions.federation_identity()
                except FederationUnavailable as error:
                    return refuse(str(error))
                mismatched = []
                if not hmac.compare_digest(str(promise.get("graph_definition_digest")), str(local["graph_definition_digest"])):
                    mismatched.append("graph_definition_digest")
                if promise.get("api_digests") != local["api_digests"]:
                    mismatched.append("api_digests")
                if mismatched:
                    return refuse("promise does not match this instance: " + ", ".join(mismatched) + " differ")
                allowed = {int(k) for k in grant["allowed_ops"].split(",") if k}
                try:
                    upstream = socket.create_connection(("127.0.0.1", int(local["port"])), timeout=5.0)
                    upstream.settimeout(None)
                    # the bridge's Hello (kind 5: v0 role, v1 id, v2 version 2); the peer never sends one
                    upstream.sendall(struct.pack("<II8Q", 0x31434E4E, 5, authz_mod.DIRECTION_ROLE[grant["direction"]],
                                                 int(grant["service_user_id"]), 2, 0, 0, 0, 0, 0))
                except OSError:
                    return refuse("the local contract port is unreachable", 1011)
                send(0x1, json.dumps({"ok": True, "hello_id": int(grant["service_user_id"])}).encode("utf-8"))
                print(f"[federation] {grant['peer']} joined {prefix!r} as {grant['direction']} (id {grant['service_user_id']})", flush=True)

                def downstream() -> None:                # contract port -> peer, whole 72-byte frames
                    buffer = bytearray()
                    try:
                        while not done.is_set():
                            data = upstream.recv(72 - len(buffer))
                            if not data:
                                break
                            buffer.extend(data)
                            if len(buffer) == 72:
                                send(0x2, bytes(buffer))
                                buffer.clear()
                    except (OSError, ValueError):
                        pass
                    finally:
                        done.set()
                        close(1000)
                        try:
                            self.connection.shutdown(socket.SHUT_RD)      # unblock the peer-side read
                        except OSError:
                            pass
                threading.Thread(target=downstream, name="federation-down", daemon=True).start()
                while not done.is_set():
                    message = read(72)
                    if message is None:
                        break
                    if message == "big":
                        close(1009, "frames are exactly 72 bytes")
                        break
                    op, data = message
                    kind = struct.unpack("<II", data[:8])[1] if len(data) >= 8 else 0
                    if op != 0x2 or len(data) != 72 or struct.unpack("<I", data[:4])[0] != 0x31434E4E or kind not in allowed:
                        close(1008, f"frame kind {kind} is outside this grant")
                        break
                    upstream.sendall(data)
            except (ConnectionError, OSError, ValueError, struct.error):
                pass
            finally:
                done.set()
                if upstream is not None:
                    try:
                        upstream.close()
                    except OSError:
                        pass
                self.close_connection = True
        # ---- [FEDERATE end]

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
            if path == "/api/tickets":
                if not self._require("dev_ui"):
                    return
                if self._is_viewer():
                    return self._refuse_viewer()
                from urllib.parse import parse_qs
                ticket_id = (parse_qs(self.path.partition("?")[2]).get("id") or [None])[0]
                try:
                    return self._json({"ticket": sessions.tickets.get(ticket_id)} if ticket_id else {"tickets": sessions.tickets.list()})
                except (ValueError, FileNotFoundError):
                    return self._json({"error": "ticket not found"}, HTTPStatus.NOT_FOUND)
            if path == "/api/whoami":                             # [ACCESS] the page learns its own tier here
                return self._whoami()
            if path.startswith("/api/mathlab/"):                  # [MATHLAB]
                return self._mathlab_get(path)
            if path == "/api/lifecycle":                         # LIFECYCLE: the countdown, and this session's restorable copy
                return self._lifecycle_get()
            if path == "/api/ready":                              # [LOADING] per-product readiness, polled by the page (loading_panel.js)
                return self._ready()
            if path == "/api/loading_anim":
                return self._anim_get()
            if path == "/api/loading_library":                    # [STUDIO] the library listing: anyone may view
                return self._library_get()
            if path == "/api/marble_photo/jobs":
                if self._is_viewer():
                    return self._refuse_viewer()
                if not self._require("photo_submit"):
                    return
                from meowdoku import job_library
                keys, account = self._job_identity()
                digests = sessions.accounts.assets_for_user(account["id"], "marble_photo") if account else ()
                return self._json(job_library.entries(sessions.marble_photos, keys, digests))
            if path == "/api/marble_photo/lenses":
                if not self._require("photo_submit"):
                    return
                from meowdoku.marble_lens_contract import lens_catalog
                try:
                    return self._json(lens_catalog(sessions.marble_photos.spectral_root))
                except (ValueError, OSError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            if path in ("/api/marble_photo", "/api/marble_photo/image", "/api/marble_photo/hdr", "/api/marble_photo/animation"):
                if not self._require("photo_submit"):
                    return
                from urllib.parse import parse_qs
                query = parse_qs(self.path.partition("?")[2])
                digest = (query.get("digest") or [""])[0]
                try:
                    self._job_authorize(digest)
                    if path == "/api/marble_photo":
                        return self._json(sessions.marble_photos.status(digest))
                    name = (query.get("name") or [""])[0]
                    body = (sessions.marble_photos.hdr(digest, name) if path == "/api/marble_photo/hdr"
                            else sessions.marble_photos.animation(digest) if path == "/api/marble_photo/animation"
                            else sessions.marble_photos.image(digest, name))
                except marble_photo_mod.PhotoError as error:
                    return self._json({"error": str(error)}, error.status)
                self._set_cookie = []
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "video/x-msvideo" if path == "/api/marble_photo/animation" else
                                 "application/octet-stream" if path == "/api/marble_photo/hdr" else "image/png")
                if path in ("/api/marble_photo/hdr", "/api/marble_photo/animation"):
                    self.send_header("Content-Disposition", f'attachment; filename="{name if path.endswith("/hdr") else "marble-orbit.avi"}"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/obj_assets" or path.startswith("/api/obj_assets/"):
                account = self._account()
                account_id = account["id"] if account else None
                try:
                    if path == "/api/obj_assets":
                        return self._json({"entries": sessions.obj_assets.entries(account_id)})
                    parts = path[len("/api/obj_assets/"):].split("/")
                    if len(parts) == 1:
                        return self._json(sessions.obj_assets.get(parts[0], account_id))
                    if len(parts) != 2 or parts[1] not in ("source", "mtl"):
                        return self._json({"error": "OBJ asset unavailable"}, HTTPStatus.NOT_FOUND)
                    body = sessions.obj_assets.source(parts[0], account_id, mtl=parts[1] == "mtl")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                except obj_assets_mod.AssetError as error:
                    return self._json({"error": str(error)}, error.status)
            if path == "/api/marble_objects":
                account = self._account()
                return self._json({"entries": sessions.marble_objects.for_account(account["id"] if account else None).entries()})
            if path == "/api/marble_preview_spec":
                # The Studio drive uses the same compiled 4k SDF toy course and
                # contact law as Marble Madness in an independent 1–16 marble run.
                # Opening Studio does not replace the user's game.
                from urllib.parse import parse_qs
                try:
                    count = int((parse_qs(self.path.partition("?")[2]).get("count") or ["1"])[0])
                except ValueError:
                    return self.send_error(400, "invalid marble count")
                if not 1 <= count <= 16:
                    return self.send_error(400, "marble count must be 1–16")
                preview = MarbleMadness()
                spec = preview.state()["compute"]
                spec["uniforms"]["emit_rate"] = 0.16 if count > 1 else 1.0e9
                # The existing start-column contract initializes age. IDs above
                # count stay parked for this run; the product and contact law stay
                # the same 4k compiled toy rather than pretending N is a new size.
                ages = [0.0] * count + [-1.0e9] * (preview.n - count)
                spec["start"] = {"age_f32": base64.b64encode(struct.pack(f"<{preview.n}f", *ages)).decode("ascii")}
                spec["studio_count"] = count
                return self._json(spec)
            if path.startswith("/api/marble_objects/"):
                try:
                    account = self._account()
                    return self._json(sessions.marble_objects.for_account(account["id"] if account else None).get(path.rsplit("/", 1)[1]))
                except (KeyError, FileNotFoundError):
                    return self._json({"error": "no such marble object"}, HTTPStatus.NOT_FOUND)
                except marble_objects_mod.ObjectError as error:
                    return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.INTERNAL_SERVER_ERROR))
            if path == "/api/account/admin/list":
                if not self._require("account_manage"):
                    return
                self._set_cookie = []
                return self._json({"admins": self._admin_list(), "users": self._user_list()})
            if path == "/api/lowerings":
                if not self._require("view_site"):
                    return
                self._set_cookie = []
                by_core: dict = {}
                for low in lowering_mod.lowerings(sessions.accounts_db):
                    by_core.setdefault(low["core_id"], []).append(low)
                return self._json({"cores": [{**core, "lowerings": by_core.get(core["core_id"], [])}
                                             for core in lowering_mod.cores(sessions.accounts_db)]})
            if path.startswith("/api/lowerings/blob/"):
                return self._lowering_blob(path.rsplit("/", 1)[1])
            if path == "/api/account/me":
                self._set_cookie = []
                account = self._account()
                role = authz_mod.role_of(sessions.accounts_db, account["id"]) if account else "guest"
                return self._json({"account": account, "role": role,
                                   "instance": self._instance_of(account["id"] if account else None)})
            if path == "/api/account/assets":
                self._set_cookie = []
                account = self._account()
                if account is None:
                    return self._json({"error": "not logged in"}, HTTPStatus.UNAUTHORIZED)
                marble_object_digests = sessions.accounts.assets_for_user(account["id"], "marble_object")
                marble_objects = []
                for digest in marble_object_digests:
                    try:
                        entry = sessions.marble_objects.for_account(account["id"]).get(digest)
                        marble_objects.append({"digest": digest, "name": entry.get("name"),
                                               "shape": entry.get("design", {}).get("base-shape")})
                    except (KeyError, FileNotFoundError, marble_objects_mod.ObjectError):
                        continue
                marble_photo_digests = sessions.accounts.assets_for_user(account["id"], "marble_photo")
                marble_photos = []
                for digest in marble_photo_digests:
                    try:
                        from meowdoku import job_library
                        if not job_library.allowed(sessions.marble_photos.directory, digest, ["account:" + str(account["id"])]):
                            continue
                        marble_photos.append({"digest": digest, **sessions.marble_photos.status(digest)})
                    except marble_photo_mod.PhotoError:
                        continue
                profile_names = sessions.accounts.assets_for_user(account["id"], "profile")
                profile = sessions.profiles.profile(profile_names[0]) if profile_names else None
                return self._json({"account": account, "marble_objects": marble_objects,
                                   "marble_photos": marble_photos, "profile": profile,
                                   "obj_assets": sessions.obj_assets.entries(account["id"])})
            if path == "/api/spectate/watch":
                return self._watch()
            if path == "/ws/spectate":                           # [LIVEVIEW] the viewer's read-only push socket
                return self._spectate_socket()
            if path.startswith("/api/share/"):
                try:
                    return self._json(sessions.shares.get(path[len("/api/share/"):]))
                except share_mod.ShareError as error:
                    return self._share_error(error)
            if path.startswith("/ws/contract/") and not self._is_viewer():      # [FEDERATE] a peer's contract-port bridge (grant-authenticated)
                return self._contract_socket(path[len("/ws/contract/"):])
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

        def _anim_post(self, length: int, *, temporary: bool = False) -> None:
            """Accept a filtered MLA1 stream from Studio or a Fountain capture. Candidate uploads are temporary and never activate;
            permanent Studio uploads can activate only when the library has no active entry. The custom header forces a CORS
            preflight, and all uploads share the size cap, per-address rate limit, and single compressor."""
            self._set_cookie = []
            def refuse(message, status):
                self.close_connection = True                       # the body may be unread
                return self._json({"error": message}, status)
            if not self._require("play" if temporary else "owner_tools"):
                return
            if self.headers.get("X-Meowdoku-Loading") != "1":
                return refuse("missing the loading-animation header", HTTPStatus.FORBIDDEN)
            if length <= 0:
                return refuse("the animation body is empty", HTTPStatus.BAD_REQUEST)
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
                return self._json({"stored": sessions.anim.put(raw, params, temporary=temporary)}, HTTPStatus.CREATED)
            except anim_mod.AnimError as error:
                return self._json({"error": f"not a valid animation: {error}"}, HTTPStatus.BAD_REQUEST)
            except anim_mod.LibraryFull as error:
                return self._json({"error": str(error)}, HTTPStatus.INSUFFICIENT_STORAGE)
            finally:
                sessions.anim_busy.release()

        # ---- [STUDIO] the library listing (anyone), and the owner's activation, graduation, deletion
        def _library_get(self) -> None:
            self._set_cookie = []
            self._json({"entries": sessions.anim.entries(), "active": sessions.anim.active_digest(), "caps": sessions.anim.caps()})

        def _library_post(self, path: str, length: int) -> None:
            """``/api/loading_library/activate``, ``/graduate`` and ``/delete`` with ``{"digest": ...}``: library state changes, so the owner's
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
            if getattr(self, "_origin", None) == "lan" and "library_manage" in self._caps():
                ok, status, why = True, 200, ""
            else:
                ok, status, why = anim_mod.owner_authorize(isinstance(self.connection, ssl.SSLSocket), "https" if self._scheme == "https" else None, direct,
                                                           self.headers.get("X-Meowdoku-Admin"), lc.token)
            if not ok:
                if status == 401:
                    misses.append(now)
                return refuse(why, status)
            try:
                digest = str(json.loads(self.rfile.read(length) or b"{}").get("digest", ""))
                done = (sessions.anim.activate(digest) if path.endswith("/activate") else
                        sessions.anim.graduate(digest) if path.endswith("/graduate") else sessions.anim.delete(digest))
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
                if not self._require("owner_tools"):               # [AUTHZ] edit-tier equivalent; the loopback + admin token below stays the authority
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
            # Worker-only payload mailbox: credentials are process-local and
            # distinct from browser ray credentials. Never accepts browser Origin.
            if path0.startswith("/api/client_rays/engine/"):
                from meowdoku.client_ray_relay import RelayError, MAX_SCENE_BYTES, MAX_BODY
                try:
                    if self.client_address[0] not in ("127.0.0.1", "::1") or self.headers.get("Origin") is not None:
                        raise RelayError("engine mailbox is loopback-only",403)
                    job=self.headers.get("X-Meowdoku-Engine-Job", "")
                    auth=self.headers.get("Authorization", "")
                    if not auth.startswith("Bearer "):
                        raise RelayError("missing engine credential",403)
                    secret=auth[7:]
                    sessions.client_ray_bridge.authenticate(job,secret)
                    action=path0.rsplit("/",1)[1]
                    length=int(self.headers.get("Content-Length",0))
                    if not 0<length<=(MAX_SCENE_BYTES if action=="scene" else MAX_BODY):
                        raise RelayError("invalid bounded engine request",413)
                    body=json.loads(self.rfile.read(length))
                    if not isinstance(body,dict):raise RelayError("expected engine request object")
                    return self._json(sessions.client_ray_bridge.engine(job,secret,action,body))
                except (ValueError,TypeError,KeyError) as error:
                    self.close_connection=True
                    return self._json({"error":str(error)},getattr(error,"status",400))
            self._classify()                                      # [ACCESS]
            if self._tier == "shader":
                if self._shader_request(path0, "POST"):
                    return
            if path0.startswith("/api/mathlab/"):              # [MATHLAB]
                return self._mathlab_post(path0)
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
            if path0 in ("/api/tickets", "/api/tickets/status"):
                if self._is_viewer():
                    return self._refuse_viewer()
                if not self._require("owner_tools" if path0.endswith("/status") else "play"):
                    return
                if self.headers.get("X-Meowdoku-Ticket") != "1":
                    self.close_connection = True
                    return self._json({"error": "missing ticket header"}, HTTPStatus.FORBIDDEN)
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > tickets_mod.MAX_BYTES:
                    self.close_connection = True
                    return self._json({"error": "ticket is too large or empty"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                try:
                    payload = json.loads(self.rfile.read(length))
                    if path0.endswith("/status"):
                        result = sessions.tickets.update(payload.get("id"), payload.get("status"), payload.get("note", ""))
                        return self._json(result)
                    table = self._table()
                    with table.lock:
                        game = table.state()
                    server_state = {k: game.get(k) for k in ("game", "player", "board", "level", "status", "phase", "tick") if k in game}
                    log_seq = logring_mod.RING.seq
                    server_log = logring_mod.RING.fetch(max(0, log_seq - 80), 80)["entries"]
                    result = sessions.tickets.submit(payload, peer=self.client_address[0], tier=self._tier,
                                                     server_state=server_state, log_seq=log_seq, server_log=server_log)
                    print(f"[ticket] {result['id']} from {self.client_address[0]}", flush=True)
                    return self._json(result, HTTPStatus.CREATED)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except FileNotFoundError:
                    return self._json({"error": "ticket not found"}, HTTPStatus.NOT_FOUND)
            if path0 in ("/api/loading_anim", "/api/loading_candidate") and not self._is_viewer():
                return self._anim_post(int(self.headers.get("Content-Length") or 0), temporary=path0.endswith("candidate"))
            if path0 == "/api/obj_assets":
                if self._is_viewer():
                    return self._refuse_viewer()
                if not self._require("studio_write"):
                    return
                account = self._account()
                if account is None:
                    return self._json({"error": "Log in to add an OBJ asset"}, HTTPStatus.UNAUTHORIZED)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if self.headers.get("Transfer-Encoding") or not 0 < length <= obj_assets_mod.MAX_BYTES:
                        self.close_connection = True
                        raise obj_assets_mod.AssetError("OBJ upload size is invalid", 413)
                    if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                        raise obj_assets_mod.AssetError("Expected application/json")
                    result = sessions.obj_assets.put(json.loads(self.rfile.read(length)), account["id"])
                    return self._json(result, HTTPStatus.CREATED)
                except (ValueError, TypeError) as error:
                    return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
            if path0 == "/api/marble_objects" and not self._is_viewer():
                if not self._require("studio_write"):
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if self.headers.get("X-Meowdoku-Marble") != "1":
                    return self._json({"error": "missing marble object header"}, HTTPStatus.FORBIDDEN)
                if length <= 0 or length > marble_objects_mod.MAX_BYTES:
                    self.close_connection = True
                    return self._json({"error": "marble object is too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                try:
                    value = json.loads(self.rfile.read(length))
                    account = self._account()
                    result = sessions.marble_objects.for_account(account["id"] if account else None).put(value)
                    if account is not None:
                        sessions.accounts.claim(account["id"], "marble_object", result["digest"])
                    return self._json(result, HTTPStatus.CREATED)
                except (ValueError, TypeError, marble_objects_mod.ObjectError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            if path0 == "/api/marble_camera/housing":
                if self._is_viewer():
                    return self._refuse_viewer()
                if not self._require("studio_write"):
                    return
                from meowdoku.marble_camera_housing import HousingError, MAX_REQUEST_BYTES, housing_response
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_REQUEST_BYTES:
                        raise HousingError("Invalid bounded camera housing request", 413)
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise HousingError("Incomplete camera housing request")
                    result = housing_response(body, sessions.marble_photos.spectral_root)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(result)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(result)
                    return
                except (ValueError, TypeError) as error:
                    self.close_connection = True
                    return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
            if path0 in ("/api/marble_photo", "/api/marble_photo/cancel", "/api/marble_photo/live", "/api/marble_photo/retouch", "/api/marble_photo/film_preview") and not self._is_viewer():
                if not self._require("share_live" if path0 == "/api/marble_photo/live" else "photo_submit"):
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if path0 == "/api/marble_photo/live":
                    if length <= 0 or length > 4096:
                        return self._json({"error": "invalid live control request"}, HTTPStatus.BAD_REQUEST)
                    try:
                        from meowdoku.marble_live import update_live
                        body = json.loads(self.rfile.read(length))
                        if not isinstance(body, dict):
                            raise ValueError("live control must be an object")
                        digest = str(body.get("digest", ""))
                        self._job_authorize(digest)
                        result = (sessions.marble_photos.cancel(digest, live_only=True) if body.get("stop") is True
                                  else update_live(sessions.marble_photos, digest, body))
                        return self._json(result)
                    except (ValueError, TypeError, marble_photo_mod.PhotoError) as error:
                        return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
                if path0 == "/api/marble_photo/film_preview":
                    if length <= 0 or length > 4096:
                        return self._json({"error": "invalid film preview request"}, HTTPStatus.BAD_REQUEST)
                    try:
                        from meowdoku.film_science import preview_film
                        body = json.loads(self.rfile.read(length))
                        return self._json(preview_film(body, sessions.marble_photos.spectral_root))
                    except (ValueError, TypeError, marble_photo_mod.PhotoError) as error:
                        return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
                if path0 == "/api/marble_photo":
                    if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/octet-stream":
                        return self._json({"error": "expected MRP1 application/octet-stream"}, HTTPStatus.BAD_REQUEST)
                    if length < 16 or length > marble_photo_mod.MAX_UPLOAD_BYTES:
                        self.close_connection = True
                        return self._json({"error": "marble photo upload size is invalid"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    try:
                        keys, account = self._job_identity()
                        import ssl
                        scheme="https" if isinstance(self.connection,ssl.SSLSocket) else "http"
                        result = sessions.marble_photos.start(self.rfile.read(length), sessions.marble_objects.for_account(account["id"] if account else None), owners=keys,
                            client_bridge=sessions.client_ray_bridge,
                            bridge_url=f"{scheme}://127.0.0.1:{self.server.server_port}/api/client_rays/engine",
                            bridge_ca=sessions.state.tls_dir/"ca.crt")
                        if account is not None:
                            sessions.accounts.claim(account["id"], "marble_photo", result["digest"])
                        return self._json(result, HTTPStatus.ACCEPTED)
                    except marble_photo_mod.PhotoError as error:
                        return self._json({"error": str(error)}, error.status)
                if path0 == "/api/marble_photo/retouch":
                    if length <= 0 or length > 4096:
                        return self._json({"error": "invalid retouch request"}, HTTPStatus.BAD_REQUEST)
                    try:
                        body = json.loads(self.rfile.read(length))
                        digest = str(body.get("digest", ""))
                        self._job_authorize(digest)
                        return self._json(sessions.marble_photos.retouch(digest, body))
                    except (ValueError, TypeError, marble_photo_mod.PhotoError) as error:
                        return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
                if length <= 0 or length > 1024:
                    return self._json({"error": "invalid photo cancel request"}, HTTPStatus.BAD_REQUEST)
                try:
                    digest = json.loads(self.rfile.read(length)).get("digest", "")
                    self._job_authorize(digest)
                    return self._json(sessions.marble_photos.cancel(digest))
                except (ValueError, TypeError, marble_photo_mod.PhotoError) as error:
                    return self._json({"error": str(error)}, getattr(error, "status", HTTPStatus.BAD_REQUEST))
            if path0 == "/api/account/login":
                return self._login_post()
            if path0 in ("/api/account/signup", "/api/account/login"):
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 1024:
                    self._set_cookie = []
                    return self._json({"error": "expected {\"username\", \"password\"}"}, HTTPStatus.BAD_REQUEST)
                try:
                    body = json.loads(self.rfile.read(length))
                    username, password = str(body.get("username", "")), str(body.get("password", ""))
                    user = (sessions.accounts.signup(username, password) if path0.endswith("signup")
                            else sessions.accounts.login(username, password))
                    self._set_cookie = []
                    self._bind_instance(user["id"])
                    self._set_account_cookie(sessions.accounts.create_session(user["id"]))
                    # Carry forward this browser's existing passwordless profile
                    # (profiles.py) onto the new/just-logged-in account, so
                    # game stats/inventory earned before the account existed
                    # are not left behind. First claim only: an account that
                    # already has a profile, or a profile another account
                    # already claimed, is left as-is.
                    from http.cookies import SimpleCookie
                    jar = SimpleCookie(self.headers.get("Cookie", ""))
                    held = jar[PLAYER_COOKIE].value if PLAYER_COOKIE in jar else None
                    name = sessions.profiles.by_token(held) if held else None
                    if name:
                        sessions.accounts.claim(user["id"], "profile", name)
                    return self._json({"account": user}, HTTPStatus.CREATED if path0.endswith("signup") else HTTPStatus.OK)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self._set_cookie = []
                    return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except accounts_mod.AccountError as error:
                    self._set_cookie = []
                    return self._json({"error": str(error)}, error.status)
            if path0 in ("/api/account/admin/grant", "/api/account/admin/revoke"):
                if not self._require("account_manage"):
                    return
                body = self._body_json()
                if body is None:
                    return
                self._set_cookie = []
                try:
                    act = authz_mod.grant_admin if path0.endswith("grant") else authz_mod.revoke_admin
                    act(sessions.accounts_db, str(body.get("username", "")), self._principal().username)
                except authz_mod.AuthzError as error:
                    return self._json({"error": error.message}, error.status)
                return self._json({"ok": True, "admins": self._admin_list()})
            if path0 == "/api/admin/account/grant":              # bootstrap: the operator's loopback + admin token creates the first admin
                lc = sessions.lifecycle
                if lc is None:
                    return self._json({"error": "operator actions are not available on this server"}, HTTPStatus.SERVICE_UNAVAILABLE)
                forwarded = any(self.headers.get(h) for h in ("X-Forwarded-For", "Forwarded", "X-Real-IP", "Via"))
                ok, status, why = lifecycle_mod.admin_authorize(self.client_address[0], self.headers.get("Host", ""), forwarded,
                                                                self.headers.get("X-Meowdoku-Admin"), lc.token)
                if not ok:
                    self.close_connection = True
                    return self._json({"error": why}, status)
                body = self._body_json()
                if body is None:
                    return
                import hashlib
                self._set_cookie = []
                try:
                    authz_mod.grant_admin(sessions.accounts_db, str(body.get("username", "")),
                                          "operator:" + hashlib.sha256(lc.token.encode("utf-8")).hexdigest()[:12])
                except authz_mod.AuthzError as error:
                    return self._json({"error": error.message}, error.status)
                return self._json({"ok": True, "admins": self._admin_list()})
            if path0 == "/api/lowerings/checkout":
                if not self._require("studio_write"):
                    return
                body = self._body_json()
                if body is None:
                    return
                try:
                    purpose = body.get("purpose")
                    if not (isinstance(purpose, list) and len(purpose) == 2 and all(isinstance(x, str) for x in purpose)):
                        raise ValueError("purpose must be [family, name]")
                    caps, prefer = body.get("capabilities"), body.get("prefer") or []
                    if not (isinstance(caps, list) and all(isinstance(x, str) for x in caps) and isinstance(prefer, list)
                            and all(isinstance(x, str) for x in prefer)):
                        raise ValueError("capabilities and prefer are lists of backend names")
                    seconds = float(body.get("seconds", 60))
                    if not 0 < seconds <= 3600:
                        raise ValueError("seconds must be in (0, 3600]")
                except (ValueError, TypeError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                key = self._job_identity()[0][0]
                lease = lowering_mod.checkout(sessions.accounts_db, tuple(purpose), caps, key, seconds, prefer=tuple(prefer))
                if lease is None:
                    return self._json({"error": "no lowering fits that purpose and those capabilities"}, HTTPStatus.NOT_FOUND)
                import dataclasses
                out = dataclasses.asdict(lease)
                out.pop("account_key", None)
                return self._json(out)
            if path0 == "/api/lowerings/checkin":
                if not self._require("studio_write"):
                    return
                body = self._body_json()
                if body is None:
                    return
                try:
                    lease_id, generation = int(body["lease_id"]), int(body["generation"])
                except (KeyError, ValueError, TypeError):
                    return self._json({"error": "expected {\"lease_id\", \"generation\"}"}, HTTPStatus.BAD_REQUEST)
                key = self._job_identity()[0][0]
                if not lowering_mod.checkin(sessions.accounts_db, lease_id, key, generation):
                    return self._json({"error": "no such open lease for this identity"}, HTTPStatus.CONFLICT)
                return self._json({"ok": True})
            if path0 == "/api/account/logout":
                from http.cookies import SimpleCookie
                jar = SimpleCookie(self.headers.get("Cookie", ""))
                sessions.accounts.logout(jar[ACCOUNT_COOKIE].value if ACCOUNT_COOKIE in jar else None)
                self._set_cookie = []
                self._clear_account_cookie()
                return self._json({"ok": True})
            if path0 in ("/api/loading_library/activate", "/api/loading_library/delete", "/api/loading_library/graduate") and not self._is_viewer():     # [STUDIO]
                if not self._require("library_manage"):               # [AUTHZ] the capability is in addition to the owner token checked inside
                    return
                return self._library_post(path0, int(self.headers.get("Content-Length") or 0))
            if path0 in ("/api/admin/lifecycle", "/api/lifecycle/restore") and not self._is_viewer():
                return self._lifecycle_post(path0, int(self.headers.get("Content-Length") or 0))
            if self._is_viewer():
                return self._refuse_viewer()                  # every POST (new, action, profile, share, spectate) is an owner's
            length = int(self.headers.get("Content-Length") or 0)
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/client_rays/"):
                from meowdoku.client_ray_relay import RelayError, handle_relay_post
                if not self._require("share_live"):return
                keys, account=self._job_identity()
                if path=="/api/client_rays/join":
                    try:
                        import ssl
                        secure=isinstance(self.connection,ssl.SSLSocket)
                        if not secure and self.client_address[0] not in ("127.0.0.1","::1"):
                            raise RelayError("ray participation requires HTTPS",403)
                        origin=self.headers.get("Origin")
                        if origin and origin!=("https://" if secure else "http://")+self.headers.get("Host",""):
                            raise RelayError("cross-origin participation is not allowed",403)
                        if self.headers.get("X-Meowdoku-Rays")!="1" or not 0<length<=1024:
                            raise RelayError("invalid ray join request",400)
                        body=json.loads(self.rfile.read(length))
                        if not isinstance(body,dict):raise RelayError("expected ray join object")
                        with sessions.marble_photos.lock:
                            sessions.marble_photos._cleanup_client_bridge_locked()
                        connection=sessions.client_ray_bridge.join(body.get("job"),keys[0])
                        sessions.marble_photos.renew_participant_live(body.get("job"),keys[0])
                        return self._json(connection)
                    except (ValueError,TypeError) as error:
                        return self._json({"error":str(error)},getattr(error,"status",400))
                if handle_relay_post(self,sessions.client_ray_bridge.relay,keys[0],on_poll=sessions.marble_photos.renew_participant_live):return
                return self._json({"error":"unknown ray route"},404)
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
            if path == "/api/action/batch":
                from meowdoku import action_batch
                if self.headers.get("X-Meowdoku-Batch") != "1":
                    return self._json({"error": "missing batch header"}, HTTPStatus.FORBIDDEN)
                if length <= 0 or length > action_batch.MAX_BYTES:
                    self.close_connection = True
                    return self._json({"error": "batch exceeds 64 KiB"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                try:
                    payload = json.loads(self.rfile.read(length))
                    action_batch.validate(payload)
                    if any(a.get("type") in access_mod.EDIT_ACTIONS for a in payload.get("actions", [])):
                        if not self._require("owner_tools"):
                            return
                    # No payload-selected table, owner, profile, or credential.
                    return self._json(action_batch.execute(self._table(), payload))
                except (ValueError, TypeError, RecursionError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path.startswith("/api/action") and str(payload.get("type")) in access_mod.EDIT_ACTIONS and not self._require("owner_tools"):
                return                                            # [AUTHZ] process-wide controls (learner, Lichess, pilot weights)
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
    """Build the shared oil toy's SDF chamber and free hinged wheel on WebGPU."""
    from meowdoku import fountain3d_build
    try:
        started = time.perf_counter()
        requested = os.environ.get("MEOWDOKU_TOY_SIZES", "")
        sizes = tuple(tag.strip() for tag in requested.split(",") if tag.strip()) if requested else fountain3d_build.WHEEL_SIZES
        manifest = fountain3d_build.build(web_dir, sizes=sizes, wasm=False, variant="toy")
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain3d toy products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
        _settle_family(ctx, "fountain3d_toy", False)
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
    lanes = {
        "laws": [step(("predictor",), lambda c: prepare_predictor(web_dir, c)),
                 step(("policy",), lambda c: prepare_policy(web_dir, c)),
                 step(("graph",), lambda c: prepare_graph_products(web_dir, c))],
        "fountain": [step(both("fountain3d_toy"), lambda c: prepare_fountain3d_toy(web_dir, c)),
                     step(both("fountain"), lambda c: prepare_fountain(web_dir, c)),
                     step(both("fountain_wheel"), lambda c: prepare_fountain_wheel(web_dir, c)),
                     step(both("fountain_purrline"), lambda c: prepare_fountain_purrline(web_dir, c)),
                     step(both("fountain_mobroad"), lambda c: prepare_fountain_mobroad(web_dir, c)),
                     step(both("fountain3d_flat") + both("fountain3d"), lambda c: prepare_fountain3d(web_dir, c)),
                     step(both("fountain3d_wheel"), lambda c: prepare_fountain3d_wheel(web_dir, c))],
        "autopilot": [step(("autopilot",), lambda c: prepare_autopilot(args))],
    }
    retained = getattr(args, "retain_fountain_products", None)
    if retained:
        from meowdoku.retained_products import prepare_retained
        # Backend-only restarts must not rebuild changed, unreleased physics.
        # Verify the retained release itself, never manufacture a source digest.
        products = tuple(name for family in startup_mod.FAMILY_NAMES for name in both(family))
        lanes["fountain"] = [step(products, lambda c: prepare_retained(web_dir, Path(retained), c))]
    return lanes


def resolve_state(state_dir, web_dir: Path) -> StatePaths:
    """--state-dir PATH -> the new layout rooted there; absent -> the old layout beside web_dir (what web_dir.parent always meant)."""
    return StatePaths(Path(state_dir)) if state_dir else StatePaths.legacy_at(Path(web_dir).parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--web-dir", default=str(WEB_DIR))
    parser.add_argument("--state-dir", default=None,
                        help="the state root (sql/ content/ products/ canvas/ logs/ secrets/ run/; see state_paths.py); "
                             "default: the old layout beside --web-dir (build/meowdoku)")
    parser.add_argument("--profiles", default=None,
                        help="profile/score file (default <state>/content/profiles.json, or build/meowdoku/profiles.json without --state-dir); "
                             "a second (dev) server keeps its own so it never clobbers the live one")
    parser.add_argument("--tls-cert", default=None,
                        help="PEM certificate: serve https/wss (browsers expose WebGPU only to secure pages)")
    parser.add_argument("--tls-key", default=None, help="PEM private key for --tls-cert")
    parser.add_argument("--access", default=None,
                        help="access tier policy file (default <state>/sql/access.json, or <web-dir>/../access.json without --state-dir; see access.py)")
    parser.add_argument("--no-wasm", action="store_true",
                        help="build the fountains' GPU products only (skip the slow wasm fallback compiles)")
    parser.add_argument("--retain-fountain-products", metavar="RECEIPT",
                        help="serve SHA-256-pinned existing fountain products without rebuilding; validate before binding; other startup lanes are unchanged")
    parser.add_argument("--route", default="",
                        help="balancer route name: every response that sets a cookie also sets meowdoku_route=NAME (Apache stickysession)")
    parser.add_argument("--instance", default="",
                        help="service instance name (deploy/apache_supervisor.py): accounts that sign up or log in here are bound to it in account_instances")
    parser.add_argument("--federation-port", type=int, default=None,
                        help="[FEDERATE] the local Nodus host's 127.0.0.1 contract port that /ws/contract/<prefix> bridges to (with --federation-canvas)")
    parser.add_argument("--federation-canvas", default=None,
                        help="[FEDERATE] that Nodus instance's canvas file; its sha256 and web-dir api.yaml module:entry pairs are what a peer must promise")
    parser.add_argument("--no-mathlab", action="store_true",
                        help="[MATHLAB] do not start the math lab's single compile worker (jobs can still be queued, none run)")
    parser.add_argument("--no-idle-train", action="store_true",
                        help="do not run the background trainer that studies recorded play when the server is idle")
    args = parser.parse_args(argv)
    if args.route:
        try:
            proxy_trust_mod.route_cookie(args.route)
        except ValueError as error:
            parser.error(str(error))
    web_dir = Path(args.web_dir)
    if not (web_dir / "boards.json").exists():
        print(f"no export at {web_dir}; run examples/meowdoku/web_export.py first", flush=True)
        return 1
    if args.retain_fountain_products:
        from meowdoku.retained_products import validate_receipt
        try:
            validate_receipt(web_dir, Path(args.retain_fountain_products))
        except ValueError as error:
            print(f"[meowdoku] retained fountain startup refused: {error}; no build attempted", flush=True)
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
    state = resolve_state(args.state_dir, web_dir)
    state.ensure(restrict=False)        # ACLs on secrets/ are set by the operator (operators.py / ensure()), never by a server start
    logring_mod.RING = logring_mod.Ring(path=state.log_ring); logring_mod.tee_stdout()   # [LOG]
    profiles_path = Path(args.profiles) if args.profiles else (state.profiles_json if args.state_dir else PROFILES)
    sessions = Sessions(web_dir, Profiles(profiles_path), state)
    sessions.route = args.route
    sessions.instance = args.instance
    sessions.federation_port, sessions.federation_canvas = args.federation_port, args.federation_canvas      # [FEDERATE]
    if args.access:
        sessions.access = access_mod.Access(Path(args.access))
    sessions.access.start()                                          # [ACCESS] policy file + the server's own public IP (background lookup)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(sessions))
    lc = lifecycle_mod.Lifecycle(sessions, state, args.port, lambda: Table(web_dir, sessions.profiles))     # LIFECYCLE
    lc.server, sessions.lifecycle = server, lc
    sessions.readiness.predictor_stems = predictor_stems()
    lanes = prepare_lanes(web_dir, args)
    startup_mod.Lanes(sessions.readiness).declare(lanes)            # every product is visible as "pending" from the first request
    lifecycle_mod.record_launch(lc.paths, args.port, args.bind, web_dir)
    threading.Thread(target=lc.run, name="lifecycle", daemon=True).start()
    if not args.no_mathlab:                                          # [MATHLAB] ONE worker; running jobs from a previous run are requeued once
        sessions.mathlab_worker = math_jobs_mod.Worker(sessions.accounts_db, state).start()
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
    prepare_threads = startup_mod.Lanes(sessions.readiness).start(lanes)   # [LOADING] the builds begin once the socket is accepting

    def scan_lowerings():                                            # [LOWERINGS] once, after the products are built/retained; never fatal
        for thread in prepare_threads:
            thread.join()
        try:
            found = lowering_mod.scan(web_dir, sessions.accounts_db)
            print(f"[meowdoku] lowering library: {len(found)} products catalogued", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"[meowdoku] lowering library scan failed: {type(error).__name__}: {error}", flush=True)
    threading.Thread(target=scan_lowerings, name="lowering-scan", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lc.stopping = True
        if sessions.mathlab_worker is not None:
            sessions.mathlab_worker.stop()
        sessions.marble_photos.close()
        if not lc.exit_code:
            lc.save_all()                                   # LIFECYCLE: a Ctrl-C keeps the sessions too
        server.server_close()
    return lc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
