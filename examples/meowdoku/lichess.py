"""Play the chess network on Lichess, live, against other bots and people.

Lichess's Bot API is plain HTTP: an event stream tells the bot about
challenges and games, a game stream tells it every move, and a move is a
POST of a UCI string.  This client speaks it with urllib alone, from a
daemon thread the server starts when a token is configured:

    token  MEOWDOKU_LICHESS_TOKEN or build/meowdoku/lichess_token.txt
           (a personal API token with the bot:play scope)

On first start it upgrades the account to a bot if it is not one yet
(Lichess allows that only on an account that has never played; it is
irreversible).  Then it accepts standard-variant challenges, plays each
game with the network's greedy one-ply choice on ``games.Chess``, and
when idle challenges an online bot now and then.  Finished games can
feed the self-play learner.  The page's ``lichess`` game shows the live
board, opponent, clocks and result.

    python examples/meowdoku/lichess.py --status          # read-only account status
    python examples/meowdoku/lichess.py --challenge maia1 # one game against a bot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402

API = "https://lichess.org"
TOKEN_FILE = ROOT / "build" / "meowdoku" / "lichess_token.txt"
FILES = "abcdefgh"
PROMOTION = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
PROMOTION_LETTER = {v: k for k, v in PROMOTION.items()}


def token() -> str | None:
    value = os.environ.get("MEOWDOKU_LICHESS_TOKEN", "").strip()
    if not value and TOKEN_FILE.exists():
        value = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return value or None


# ---------------------------------------------------------------- UCI <-> our cells
def square_to_cell(square: str) -> int:
    return (8 - int(square[1])) * 8 + FILES.index(square[0])


def cell_to_square(cell: int) -> str:
    return f"{FILES[cell % 8]}{8 - cell // 8}"


def apply_uci(game: Chess, uci: str) -> None:
    """One UCI move (``e2e4``, ``e7e8q``) onto our board, turn included."""
    src, dst = square_to_cell(uci[:2]), square_to_cell(uci[2:4])
    promotion = PROMOTION.get(uci[4:5], "queen")
    game.apply_move(src, dst, promotion)
    game.turn = "black" if game.turn == "white" else "white"
    game.moves += 1


def board_from_moves(moves: str) -> Chess:
    game = Chess("lichess")
    for uci in moves.split():
        apply_uci(game, uci)
    return game


def move_to_uci(game: Chess, src: int, dst: int) -> str:
    kind, _ = game.pieces[src]
    uci = cell_to_square(src) + cell_to_square(dst)
    if kind == "pawn" and dst // 8 in (0, 7):
        uci += "q"
    return uci


# ---------------------------------------------------------------- HTTP
class Api:
    def __init__(self, access_token: str):
        self.token = access_token

    def _request(self, method: str, path: str, data: dict | None = None, stream: bool = False, accept: str = "application/json"):
        body = urllib.parse.urlencode(data).encode() if data else None
        request = urllib.request.Request(API + path, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Accept": accept, "User-Agent": "meowdoku-lichess/1.0"})
        if body is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        response = urllib.request.urlopen(request, timeout=None if stream else 30)
        return response if stream else response.read().decode("utf-8")

    def get(self, path: str) -> dict:
        text = self._request("GET", path)
        return json.loads(text) if text else {}

    def post(self, path: str, data: dict | None = None) -> dict:
        try:
            text = self._request("POST", path, data)
            return json.loads(text) if text else {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"{path}: HTTP {error.code} {detail}") from None

    def lines(self, path: str):
        """NDJSON stream: one parsed object per non-empty line."""
        response = self._request("GET", path, stream=True, accept="application/x-ndjson")
        for raw in response:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


# ---------------------------------------------------------------- the bot
class LichessBot:
    """The event loop and the games, each in its own thread."""

    def __init__(self, access_token: str, player=None, learn: bool = True, log=print):
        self.api = Api(access_token)
        self.player = player
        self.learn = learn
        self.log = log
        self.me: dict = {}
        self.games: dict[str, dict] = {}           # game id -> live view
        self.latest: str | None = None             # the game the page shows
        self.results: list[str] = []
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.idle_challenge_every = 90.0            # after a game ends, before seeking again
        self.last_challenge = 0.0
        self.pending: dict | None = None            # {"id", "to", "at"} of an unanswered challenge
        self.candidates: list[str] = []             # online bots still to try this round
        self.seeking = False

    # -- identity --------------------------------------------------------
    def connect(self) -> dict:
        self.me = self.api.get("/api/account")
        if self.me.get("title") != "BOT":
            self.log(f"[lichess] upgrading {self.me.get('username')} to a bot account")
            self.api.post("/api/bot/account/upgrade")
            self.me = self.api.get("/api/account")
        self.log(f"[lichess] connected as {self.me.get('username')} ({self.me.get('title', 'no title')})")
        return self.me

    # -- play ------------------------------------------------------------
    def choose_uci(self, game: Chess) -> str | None:
        if self.player is None:
            moves = game.all_legal_moves()
            return move_to_uci(game, *random.choice(moves)) if moves else None
        move = self.player.choose(game, greedy=True)
        return move_to_uci(game, *move) if move else None

    def play_game(self, game_id: str) -> None:
        view = {"id": game_id, "moves": "", "status": "started", "white": "?", "black": "?", "color": "white",
                "wtime": None, "btime": None, "result": None, "opponent": "?", "started": time.time()}
        with self.lock:
            self.games[game_id] = view
            self.latest = game_id
        boards, materials = [], []
        try:
            for event in self.api.lines(f"/api/bot/game/stream/{game_id}"):
                kind = event.get("type")
                if kind == "gameFull":
                    white, black = event.get("white", {}), event.get("black", {})
                    view["white"] = white.get("name", "?"); view["black"] = black.get("name", "?")
                    view["color"] = "white" if (white.get("id") or "").lower() == str(self.me.get("id", "")).lower() else "black"
                    view["opponent"] = view["black"] if view["color"] == "white" else view["white"]
                    view["rating"] = (black if view["color"] == "white" else white).get("rating")
                    state = event.get("state", {})
                elif kind == "gameState":
                    state = event
                else:
                    continue
                view["moves"] = state.get("moves", "")
                view["wtime"], view["btime"] = state.get("wtime"), state.get("btime")
                view["status"] = state.get("status", "started")
                game = board_from_moves(view["moves"])
                if view["status"] not in ("started", "created"):
                    winner = state.get("winner")
                    view["result"] = winner if winner else "draw"
                    break
                if game.turn == view["color"]:
                    from meowdoku.chess_net import encode, material
                    boards.append(encode(game.pieces, game.turn)); materials.append(material(game.pieces))
                    uci = self.choose_uci(game)
                    if uci:
                        try:
                            self.api.post(f"/api/bot/game/{game_id}/move/{uci}")
                        except RuntimeError as error:
                            self.log(f"[lichess] move refused: {error}")
        except Exception as error:
            self.log(f"[lichess] game {game_id} stream ended: {type(error).__name__}: {error}")
        result = view.get("result") or "draw"
        outcome = "draw" if result == "draw" else result            # "white" / "black" as the winner
        summary = f"{'won' if outcome == view['color'] else 'drew' if outcome == 'draw' else 'lost'} vs {view['opponent']} ({view.get('rating', '?')})"
        with self.lock:
            self.results.append(summary)
            view["summary"] = summary
        self.log(f"[lichess] {summary}: {view['status']}")
        if self.learn and self.player is not None and len(boards) >= 4:
            try:
                self.player.learn_from(boards, materials, outcome)
            except Exception as error:
                self.log(f"[lichess] learning from the game failed: {error}")

    # -- events ----------------------------------------------------------
    def accept_or_decline(self, challenge: dict) -> None:
        cid = challenge.get("id")
        if str(challenge.get("challenger", {}).get("id", "")).lower() == str(self.me.get("id", "")).lower():
            return                                            # our own outgoing challenge, echoed back
        standard = challenge.get("variant", {}).get("key") == "standard"
        speed_ok = challenge.get("speed") in ("bullet", "blitz", "rapid", "classical", "correspondence")
        if standard and speed_ok:
            self.api.post(f"/api/challenge/{cid}/accept")
            self.log(f"[lichess] accepted challenge from {challenge.get('challenger', {}).get('name')}")
        else:
            self.api.post(f"/api/challenge/{cid}/decline", {"reason": "variant" if not standard else "generic"})

    def _refill_candidates(self) -> None:
        try:
            bots = [line.get("username") for line in self.api.lines("/api/bot/online?nb=100")]
        except Exception as error:
            self.log(f"[lichess] bot list: {error}")
            bots = []
        bots = [b for b in bots if b and b.lower() != str(self.me.get("id", "")).lower()]
        random.shuffle(bots)
        self.candidates = bots

    def challenge_a_bot(self) -> None:
        """Work through the online bots until one accepts: a refusal (the
        daily bot-vs-bot cap, a closed account) moves straight to the
        next; a challenge nobody answers in 30 s is cancelled and the next
        is sent."""
        now = time.time()
        if self.pending and now - self.pending["at"] < 30.0:
            return
        if self.pending:
            try:
                self.api.post(f"/api/challenge/{self.pending['id']}/cancel")
            except RuntimeError:
                pass
            self.log(f"[lichess] {self.pending['to']} did not answer; next")
            self.pending = None
        for _attempt in range(12):
            if not self.candidates:
                self._refill_candidates()
                if not self.candidates:
                    return
            target = self.candidates.pop()
            try:
                reply = self.api.post(f"/api/challenge/{target}", {"rated": "false", "clock.limit": "300", "clock.increment": "3", "color": "random", "variant": "standard"})
            except RuntimeError as error:
                self.log(f"[lichess] {target} refused: {str(error)[-90:]}")
                continue
            cid = (reply or {}).get("id") or (reply or {}).get("challenge", {}).get("id")
            self.pending = {"id": cid, "to": target, "at": now}
            self.log(f"[lichess] challenged {target}; waiting for an answer")
            return

    def run(self) -> None:
        self.connect()
        self.last_challenge = 0.0                                 # seek a game at once
        while not self._stop.is_set():
            try:
                for event in self.api.lines("/api/stream/event"):
                    if self._stop.is_set():
                        break
                    kind = event.get("type")
                    if kind == "challenge":
                        self.accept_or_decline(event["challenge"])
                    elif kind == "gameStart":
                        self.pending = None
                        self.seeking = False
                        gid = event["game"]["id"]
                        thread = threading.Thread(target=self.play_game, args=(gid,), name=f"lichess-{gid}", daemon=True)
                        thread.start(); self.threads.append(thread)
                    elif kind == "gameFinish":
                        self.last_challenge = time.time()     # a breath before seeking again
                    elif kind in ("challengeDeclined", "challengeCanceled"):
                        who = event.get("challenge", {}).get("destUser", {}).get("name", "?")
                        self.log(f"[lichess] {who} declined; next")
                        self.pending = None
                        self.last_challenge = 0.0             # seek again at once
                    self._maybe_challenge()
            except Exception as error:
                self.log(f"[lichess] event stream: {type(error).__name__}: {error}; reconnecting")
                time.sleep(5)
            self._maybe_challenge()

    def _maybe_challenge(self) -> None:
        live = [g for g in self.games.values() if g.get("status") in ("started", "created")]
        if live:
            self.seeking = False
            return
        if time.time() - self.last_challenge > self.idle_challenge_every:
            self.seeking = True
            self.challenge_a_bot()

    def _idle_loop(self) -> None:
        """The event stream blocks while nothing happens, so the idle
        challenge runs on its own clock."""
        while not self._stop.is_set():
            time.sleep(5)
            if self.me:
                try:
                    self._maybe_challenge()
                except Exception as error:
                    self.log(f"[lichess] idle challenge: {error}")

    def start(self) -> None:
        thread = threading.Thread(target=self.run, name="lichess-events", daemon=True)
        thread.start(); self.threads.append(thread)
        idle = threading.Thread(target=self._idle_loop, name="lichess-idle", daemon=True)
        idle.start(); self.threads.append(idle)

    def stop(self) -> None:
        self._stop.set()

    # -- what the page shows ---------------------------------------------
    def snapshot(self) -> dict | None:
        with self.lock:
            if self.latest is None:
                return None
            return dict(self.games[self.latest])


_BOT: LichessBot | None = None


def bot(player=None, log=print) -> LichessBot | None:
    """The one bot of this process, started on first use when a token exists."""
    global _BOT
    if _BOT is None:
        access = token()
        if not access:
            return None
        _BOT = LichessBot(access, player=player, log=log)
        _BOT.start()
    return _BOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", action="store_true", help="show the account and whether it is a bot")
    parser.add_argument("--challenge", default="", help="challenge this bot once and play the game")
    parser.add_argument("--random", action="store_true", help="play random moves instead of the network")
    args = parser.parse_args(argv)
    access = token()
    if not access:
        print("no token: set MEOWDOKU_LICHESS_TOKEN or write build/meowdoku/lichess_token.txt", flush=True)
        return 1
    if args.status:
        # connect() enrolls non-bot accounts irreversibly; status only reads.
        me = Api(access).get("/api/account")
        print(json.dumps({k: me.get(k) for k in ("id", "username", "title", "perfs")}, indent=2))
        return 0
    player = None
    if not args.random:
        from meowdoku.chess_net import SelfPlay
        player = SelfPlay()
    client = LichessBot(access, player=player)
    client.connect()
    client.start()
    if args.challenge:
        client.api.post(f"/api/challenge/{args.challenge}", {"rated": "false", "clock.limit": "300", "clock.increment": "3", "variant": "standard"})
        print(f"[lichess] challenged {args.challenge}; waiting for the game", flush=True)
    try:
        while True:
            time.sleep(1)
            snap = client.snapshot()
            if snap and snap.get("result"):
                print(f"[lichess] done: {snap.get('summary')}", flush=True)
                if args.challenge:
                    return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
