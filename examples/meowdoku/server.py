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
from meowdoku.platformer import Platformer  # noqa: E402
from meowdoku.sidescroller import SideScroller  # noqa: E402
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
        self.recorded = None                 # the game object whose result is on the profile
        self.lock = threading.RLock()
        database = json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))
        self.boards = {board["id"]: board for board in database["boards"]}
        self.levels = database["levels"]
        self.game = None

    def catalog(self) -> dict:
        return {
            "games": ["meowdoku", "chess", "checkers", "wordle", "chess_vs_net", "chess_arena", "lichess", "space_invaders", "bullet_hell", "platformer", "side_scroller"],
            "meowdoku": {"levels": self.levels,
                         "boards": [{"id": b["id"], "level": b["level"], "type": b["type"], "n": b["n"]} for b in self.boards.values()]},
        }

    def new(self, payload: dict) -> dict:
        kind = str(payload.get("game", "meowdoku"))
        # Games are views; the learner and the Lichess client are process-wide
        # background workers that a mode switch never stops.
        if kind == "meowdoku":
            self.game = MeowdokuGame(self.boards[str(payload["id"])], **({"palette": str(payload["palette"])} if payload.get("palette") else {}))
        elif kind == "chess":
            self.game = Chess("chess")
        elif kind == "checkers":
            self.game = Checkers("checkers")
        elif kind == "wordle":
            self.game = Wordle(int(payload.get("seed", 0)))
        elif kind == "chess_arena":
            self.game = ChessArena(parallel=int(payload.get("parallel", 4)), lock=self.lock)   # the learner is one per process; 4 searched games
        elif kind == "chess_vs_net":
            self.game = ChessVsNet()
        elif kind == "lichess":
            self.game = LichessGame()
        elif kind == "space_invaders":
            self.game = SpaceInvaders(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "bullet_hell":
            self.game = BulletScroller(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "platformer":
            self.game = Platformer(aspect=float(payload.get("aspect") or 0.0))
        elif kind == "side_scroller":
            self.game = SideScroller(aspect=float(payload.get("aspect") or 0.0))
        else:
            raise ValueError(f"unknown game {kind!r}")
        return self.state()

    def action(self, payload: dict) -> dict:
        if self.game is None:
            raise ValueError("no game: POST /api/new first")
        kind = str(payload.get("type"))
        if kind == "cell":
            self.game.cell(int(payload["r"]), int(payload["c"]), str(payload.get("button", "primary")))
        elif kind == "text":
            self.game.text(str(payload.get("text", "")))
        elif kind == "tick":
            if hasattr(self.game, "control"):
                self.game.control(payload)          # the held controls ride on the tick
            if hasattr(self.game, "tick"):
                self.game.tick()
        elif kind == "layout":
            if hasattr(self.game, "layout"):
                self.game.layout(float(payload.get("aspect") or 0.0))
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
            self.game.power(str(payload.get("kind", "")))
        elif kind == "palette":
            self.game.set_palette(str(payload.get("name", "")))
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
        return st

    def _record(self) -> None:
        """Put a finished game's result on the player's profile, once."""
        if self.profiles is None or self.player is None or self.game is None or self.recorded is self.game:
            return
        result = self.game.result()
        if result is not None:
            self.profiles.record(self.player, self.game.game, result)
            self.recorded = self.game

    def set_player(self, payload: dict) -> dict:
        self.player = self.profiles.ensure(str(payload.get("name", "")))
        return self.profile()

    def profile(self) -> dict:
        game = self.game.game if self.game is not None else None
        return {"player": self.player, "profile": self.profiles.profile(self.player),
                "game": game, "leaderboard": self.profiles.leaderboard(game) if game else []}


COOKIE = "meowdoku_session"
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
            self._set_cookie = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax" if fresh else None
            return table

        def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if getattr(self, "_set_cookie", None):
                self.send_header("Set-Cookie", self._set_cookie)
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            if not self.path.startswith("/api/"):
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
        print("[meowdoku] ship predictor: motion.wasm ready", flush=True)
    except Exception as error:  # the page plays without prediction
        print(f"[meowdoku] ship predictor NOT built ({type(error).__name__}: {error}); "
              "free-board games run unpredicted", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--web-dir", default=str(WEB_DIR))
    parser.add_argument("--profiles", default=str(PROFILES),
                        help="profile/score file; a second (dev) server keeps its own so it never clobbers the live one")
    args = parser.parse_args(argv)
    web_dir = Path(args.web_dir)
    if not (web_dir / "boards.json").exists():
        print(f"no export at {web_dir}; run examples/meowdoku/web_export.py first", flush=True)
        return 1
    prepare_predictor(web_dir)
    sessions = Sessions(web_dir, Profiles(Path(args.profiles)))
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(sessions))
    print(f"[meowdoku] serving {web_dir} at http://{args.bind}:{args.port}/  ({sessions.board_count} prebaked boards; chess, checkers, wordle)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
