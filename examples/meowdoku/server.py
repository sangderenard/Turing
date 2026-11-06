"""The Meowdoku web server: the Python state machines behind the WebGL page.

    python examples/meowdoku/server.py --port 8766 --bind 0.0.0.0

Serves ``build/meowdoku/web`` (the exported page, shaders, assets and the
prebaked board database) and a JSON API over the games in ``games.py``:

    GET  /api/catalog                      -> games, Meowdoku levels and boards
    POST /api/new     {"game": "meowdoku", "id": "<board id>"}
                      {"game": "chess"|"checkers"} | {"game": "wordle", "seed": 3}
    POST /api/action  {"type": "cell", "r", "c", "button"} | {"type": "text", "text"}
                      | {"type": "submit"|"backspace"|"undo"|"reset"}
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
import json
import secrets
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
from meowdoku.purrline import Purrline  # noqa: E402
from meowdoku.profiles import Profiles  # noqa: E402
from meowdoku.realtime_socket import SnapshotEncoder, accept_key, frame, read_frame  # noqa: E402

WEB_DIR = ROOT / "build" / "meowdoku" / "web"
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
        self.chess_clock = None              # the time control the player last picked (chess against the network)

    def catalog(self) -> dict:
        return {
            "games": ["meowdoku", "chess", "checkers", "wordle", "chess_vs_net", "chess_arena", "lichess", "space_invaders", "bullet_hell", "platformer", "delve", "side_scroller", "tilequest", "boardwalk", "worldwalk", "fountain", "purrline"],
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
            self.game = FountainGame()       # the page runs the law; the server holds phase, seed, image
        elif kind == "purrline":
            self.game = Purrline(aspect=float(payload.get("aspect") or 0.0))
        else:
            raise ValueError(f"unknown game {kind!r}")
        return self.state()

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
        elif kind.startswith("fountain_") or kind.startswith("worldwalk_") or kind.startswith("delve_"):
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
        elif kind == "palette":
            self.game.set_palette(str(payload.get("name", "")))
        elif kind == "mode":
            if not hasattr(self.game, "set_mode"):
                raise ValueError("this game has no modes")
            self.game.set_mode(str(payload.get("name", "")), bool(payload.get("on")))
        elif kind in ("submit", "backspace", "undo", "reset"):
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
        boards = json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))
        self.board_count = len(boards["boards"])

    def get(self, token: str | None) -> tuple[str, Table, bool]:
        now = time.time()
        with self.lock:
            for key in [k for k, (_, seen) in self.tables.items() if now - seen > SESSION_IDLE_SECONDS]:
                del self.tables[key]
            fresh = token not in self.tables
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
            if "/api/" in line and "/api/action" not in line and "/api/state" not in line:
                sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _table(self) -> Table:
            from http.cookies import SimpleCookie
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            token, table, fresh = sessions.get(jar[COOKIE].value if COOKIE in jar else None)
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
                            binary, meta = encoder.encode(game, table.state())
                        if meta is not None:
                            self.wfile.write(frame(0x1, json.dumps({"meta": meta}).encode("utf-8")))
                        self.wfile.write(frame(0x2, binary))
                    except Exception as error:  # the page shows the refusal
                        self.wfile.write(frame(0x1, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode("utf-8")))
            except (ConnectionError, OSError):
                pass
            finally:
                self.close_connection = True

        def do_GET(self):  # noqa: N802
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
            return super().do_GET()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
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


def prepare_predictor(web_dir: Path) -> None:
    """The page's ship predictor: ``entity_motion.damped_axis`` lowered to a
    WebAssembly core (rebuilt only when its source changes), and the mobile
    reference app's loader for it."""
    import shutil
    from meowdoku.entity_motion import build_predictor
    shell = ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js"
    shutil.copyfile(shell, web_dir / "turing-wasm-shell.js")
    try:
        build_predictor(web_dir)
        print("[meowdoku] engine laws for the page: every entity_motion.PREDICTORS core (motion, travel, ..., shot, climb, project, heading, gate, ballistic) ready", flush=True)
    except Exception as error:  # the page plays without prediction
        print(f"[meowdoku] ship predictor NOT built ({type(error).__name__}: {error}); "
              "free-board games run unpredicted", flush=True)


def prepare_fountain(web_dir: Path) -> None:
    """The fountain's compute product (the page's generic compute runtime loads it) (``fountain_build.build``):
    the generated WebGPU compute shader and the wasm product for each size, and
    the manifest holding the compiler's deployment verdicts.  Rebuilt only when
    the law or the emitters change."""
    from meowdoku import fountain_build
    try:
        started = time.perf_counter()
        manifest = fountain_build.build(web_dir)
        routes = ", ".join(f"{tag}: {rec['n']} ids -> {rec['route']}" for tag, rec in manifest["sizes"].items())
        print(f"[meowdoku] fountain products ready ({routes}; {time.perf_counter() - started:.1f}s)", flush=True)
    except Exception as error:  # the page reports the missing manifest
        print(f"[meowdoku] fountain products NOT built ({type(error).__name__}: {error})", flush=True)


def prepare_policy(web_dir: Path) -> None:
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


def prepare_graph_products(web_dir: Path) -> None:
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
    parser.add_argument("--no-idle-train", action="store_true",
                        help="do not run the background trainer that studies recorded play when the server is idle")
    args = parser.parse_args(argv)
    web_dir = Path(args.web_dir)
    if not (web_dir / "boards.json").exists():
        print(f"no export at {web_dir}; run examples/meowdoku/web_export.py first", flush=True)
        return 1
    prepare_predictor(web_dir)
    prepare_policy(web_dir)
    prepare_graph_products(web_dir)
    prepare_fountain(web_dir)
    global COOKIE, PLAYER_COOKIE
    if args.port != 8766:
        COOKIE = f"meowdoku_session_{args.port}"
        PLAYER_COOKIE = f"meowdoku_player_{args.port}"
    from meowdoku import autopilot
    print(f"[meowdoku] autopilot weights loaded: {', '.join(autopilot.preload()) or 'none (train with autopilot.py)'}", flush=True)
    autopilot.load_learning()
    from meowdoku import idle_trainer
    if not args.no_idle_train:
        idle_trainer.start_default()                    # studies recorded play whenever the server is idle
    sessions = Sessions(web_dir, Profiles(Path(args.profiles)))
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(sessions))
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
