"""The Meowdoku web server: the Python state machine behind the WebGL page.

    python examples/meowdoku/server.py --port 8766

Serves ``build/meowdoku/web`` (the exported page, shaders, assets and the
prebaked board database) and a small JSON API over one ``MeowdokuSession``,
which runs the game law through the dt system exactly as the pygame runner
does:

    POST /api/new    {"id": "<board id>"}           -> state
    POST /api/move   {"r": int, "c": int, "button": "primary"|"secondary"} -> state
    POST /api/undo, /api/reset                      -> state
    GET  /api/state                                 -> state

A state is ``{n, id, cat, x_mark, hearts, won, lost, moves}`` with the
columns as flat 0/1 lists.  The page animates the difference between
states; the server never renders anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.puzzle import Puzzle  # noqa: E402
from meowdoku.session import MeowdokuSession  # noqa: E402

WEB_DIR = ROOT / "build" / "meowdoku" / "web"


class Game:
    """One session at a time, built from a prebaked board."""

    def __init__(self, web_dir: Path):
        self.web_dir = web_dir
        self.lock = threading.Lock()
        self.boards = {board["id"]: board for board in json.loads((web_dir / "boards.json").read_text(encoding="utf-8"))["boards"]}
        self.session: MeowdokuSession | None = None
        self.board_id: str | None = None
        self.moves = 0

    def new(self, board_id: str) -> dict:
        board = self.boards[board_id]
        n = int(board["n"])
        cells = np.arange(n * n)
        puzzle = Puzzle(n, (cells // n).astype(np.float64), (cells % n).astype(np.float64),
                        np.asarray(board["region"], dtype=np.float64), np.asarray(board["solution"], dtype=np.float64))
        self.session = MeowdokuSession(n=n, puzzle=puzzle)
        self.board_id = board_id
        self.moves = 0
        return self.state()

    def move(self, r: int, c: int, button: str) -> dict:
        if self.session is None:
            raise ValueError("no game: POST /api/new first")
        if button == "secondary":
            self.session.toggle_x(int(r), int(c))
        else:
            self.session.place(int(r), int(c))
        self.moves += 1
        return self.state()

    def undo(self) -> dict:
        if self.session is not None:
            self.session.undo()
        return self.state()

    def reset(self) -> dict:
        if self.session is not None:
            self.session.reset()
            self.moves = 0
        return self.state()

    def state(self) -> dict:
        if self.session is None:
            return {"n": 0, "id": None}
        s = self.session
        return {
            "n": int(s.n), "id": self.board_id,
            "cat": np.asarray(s.cat, dtype=int).reshape(-1).tolist(),
            "x_mark": np.asarray(s.x_mark, dtype=int).reshape(-1).tolist(),
            "hearts": int(s.hearts), "won": bool(s.won), "lost": bool(s.lost), "moves": self.moves,
        }


def make_handler(game: Game):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(game.web_dir), **kwargs)

        def log_message(self, format, *args):  # noqa: A002 - quiet static logs
            if "/api/" in str(args[0]) if args else False:
                sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/state"):
                with game.lock:
                    return self._json(game.state())
            return super().do_GET()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            try:
                with game.lock:
                    if self.path.startswith("/api/new"):
                        return self._json(game.new(str(payload["id"])))
                    if self.path.startswith("/api/move"):
                        return self._json(game.move(payload["r"], payload["c"], str(payload.get("button", "primary"))))
                    if self.path.startswith("/api/undo"):
                        return self._json(game.undo())
                    if self.path.startswith("/api/reset"):
                        return self._json(game.reset())
            except Exception as error:  # the page shows the refusal
                return self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.BAD_REQUEST)
            return self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--web-dir", default=str(WEB_DIR))
    args = parser.parse_args(argv)
    web_dir = Path(args.web_dir)
    if not (web_dir / "boards.json").exists():
        print(f"no export at {web_dir}; run examples/meowdoku/web_export.py first", flush=True)
        return 1
    game = Game(web_dir)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(game))
    print(f"[meowdoku] serving {web_dir} at http://{args.bind}:{args.port}/  ({len(game.boards)} prebaked boards)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
