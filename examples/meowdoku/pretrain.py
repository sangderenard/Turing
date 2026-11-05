"""Pre-train the chess value network on master games before self-play.

    python examples/meowdoku/pretrain.py            # fetch, replay, train, measure

Games come from The Week in Chess (theweekinchess.com), the weekly
collections of professional games, a few issues at about 2 MB each,
cached under ``build/meowdoku/pgn/``.  Each
game is replayed on ``games.Chess`` (full rules: castling, en passant,
promotion), every position is encoded as the network sees it, and the
target is the game's result for White discounted toward the end plus the
small material term self-play also uses, so the two phases agree.  The
network then trains in shuffled minibatches through AbstractTensor's
autograd, saves its weights, and is measured against a random mover before
and after.  ``ensure`` runs all of this once, in the arena's thread, when
no weights exist yet.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402
from meowdoku.chess_net import FEATURES, POLICY, SelfPlay, WEIGHTS, encode, material, move_index  # noqa: E402

PGN_DIR = ROOT / "build" / "meowdoku" / "pgn"
# The Week in Chess: weekly collections of the world's professional games
# (candidates, world championships, opens), one zip per issue, each about
# 2 MB and a few thousand games.
TWIC_ISSUES = (1560, 1559, 1558, 1557)
SOURCES = {f"twic{issue}": f"https://theweekinchess.com/zips/twic{issue}g.zip" for issue in TWIC_ISSUES}
MARKER = WEIGHTS.with_name("chess_net.pretrained.json")
PIECE = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
FILES = "abcdefgh"
SAN = re.compile(r"^([KQRBN])?([a-h])?([1-8])?x?([a-h])([1-8])(?:=([QRBN]))?[+#]?$")


def fetch(progress=print) -> list[Path]:
    """The cached PGN files, downloading the sources once."""
    PGN_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in SOURCES.items():
        target = PGN_DIR / f"{name}.pgn"
        local_zip = PGN_DIR / f"{name}.zip"                   # a zip placed here by hand is used as is
        if not target.exists():
            if local_zip.exists():
                data = local_zip.read_bytes()
            else:
                progress(f"[pretrain] downloading {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "meowdoku-pretrain/1.0",
                                                               "Accept": "application/zip, application/octet-stream, */*"})
                data = urllib.request.urlopen(request, timeout=60).read()
                local_zip.write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                member = next(n for n in archive.namelist() if n.lower().endswith(".pgn"))
                target.write_bytes(archive.read(member))
            progress(f"[pretrain] {name}: {target.stat().st_size / 1e6:.1f} MB of PGN")
        paths.append(target)
    return paths


def games_in(text: str):
    """(result, [san, ...]) per game: tags dropped, comments, variations and
    numbering stripped."""
    for block in re.split(r"\n\s*\n(?=\[Event)", "\n" + text):
        if "[Event" not in block:
            continue
        tags, _, moves = block.partition("\n\n")
        result_tag = re.search(r'\[Result "([^"]+)"\]', tags)
        result = result_tag.group(1) if result_tag else "*"
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        moves = re.sub(r"\{[^}]*\}", " ", moves)
        depth, cleaned = 0, []
        for ch in moves:                                      # drop variations
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0:
                cleaned.append(ch)
        tokens = []
        for token in "".join(cleaned).split():
            token = re.sub(r"^\d+\.+", "", token)
            if not token or token.startswith("$") or token in ("1-0", "0-1", "1/2-1/2", "*"):
                continue
            tokens.append(token)
        yield result, tokens


def san_move(game: Chess, san: str):
    """The (src, dst, promotion) a SAN token names on this board, or None."""
    side = game.turn
    home = 7 if side == "white" else 0
    if san.replace("0", "O").startswith("O-O-O"):
        return game.idx(home, 4), game.idx(home, 2), "queen"
    if san.replace("0", "O").startswith("O-O"):
        return game.idx(home, 4), game.idx(home, 6), "queen"
    m = SAN.match(san)
    if not m:
        return None
    kind = PIECE.get(m.group(1), "pawn")
    file_hint, rank_hint = m.group(2), m.group(3)
    dst = game.idx(8 - int(m.group(5)), FILES.index(m.group(4)))
    promotion = PIECE.get(m.group(6), "queen") if m.group(6) else "queen"
    candidates = []
    for src, (k, s) in game.pieces.items():
        if s != side or k != kind:
            continue
        r, c = game.rc(src)
        if file_hint and FILES[c] != file_hint:
            continue
        if rank_hint and 8 - r != int(rank_hint):
            continue
        if dst in game.pseudo_moves(src):
            candidates.append(src)
    if len(candidates) > 1:                                   # disambiguate by the king's safety
        candidates = [src for src in candidates if dst in game.legal_moves(src)]
    if len(candidates) != 1:
        return None
    return candidates[0], dst, promotion


def positions_of(result: str, tokens: list[str], max_plies: int = 200):
    """Encoded positions and material along one replayed game, and its outcome."""
    game = Chess("pgn")
    boards, materials, played = [], [], []
    for san in tokens[:max_plies]:
        move = san_move(game, san)
        if move is None:
            break
        boards.append(encode(game.pieces, game.turn))
        materials.append(material(game.pieces))
        played.append(move_index(move[0], move[1]))
        src, dst, promotion = move
        game.apply_move(src, dst, promotion)
        game.turn = "black" if game.turn == "white" else "white"
    outcome = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}[result]
    return boards, materials, outcome, played


def dataset(paths: list[Path], max_games: int, progress=print):
    xs, zs, ms, used, skipped = [], [], [], 0, 0
    player_targets = SelfPlay.targets
    started = time.perf_counter()
    for path in paths:
        for result, tokens in games_in(path.read_text(encoding="latin-1")):
            if used >= max_games:
                break
            boards, materials, outcome, played = positions_of(result, tokens)
            if len(boards) < 10:
                skipped += 1
                continue
            xs.append(np.stack(boards).astype(np.float32))
            zs.append(player_targets(None, materials, outcome).astype(np.float32))
            ms.append(np.asarray(played, dtype=np.int32))
            used += 1
            if used % 200 == 0:
                progress(f"[pretrain] {used} games, {sum(len(x) for x in xs)} positions, {time.perf_counter() - started:.0f}s")
    progress(f"[pretrain] {used} games replayed, {skipped} skipped, {sum(len(x) for x in xs)} positions")
    return np.concatenate(xs), np.concatenate(zs), np.concatenate(ms)


def one_hot(moves: np.ndarray) -> np.ndarray:
    rows = np.zeros((len(moves), POLICY), dtype=np.float64)
    rows[np.arange(len(moves)), moves] = 1.0
    return rows


def train(player: SelfPlay, x: np.ndarray, z: np.ndarray, m: np.ndarray | None = None, epochs: int = 3,
          batch: int = 256, progress=print) -> float:
    rng = np.random.default_rng(0)
    n = x.shape[0]
    loss = float("nan")
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            losses.append(player.net.train_step(x[idx], z[idx], player.lr, policy=one_hot(m[idx]) if m is not None else None))
        loss = float(np.mean(losses))
        progress(f"[pretrain] epoch {epoch + 1}/{epochs}: loss {loss:.4f}")
        player.net.save(player.weights)
    return loss


class Corpus:
    """The master-game positions as one array, stepped through in shuffled
    minibatches for as long as the arena runs.  Built once (replay of the
    PGNs) and cached as ``corpus_<games>.npz`` beside them, so the next
    start loads in a moment."""

    def __init__(self, x: np.ndarray, z: np.ndarray, m: np.ndarray | None = None, batch: int = 256, seed: int = 0):
        self.x, self.z, self.m, self.batch = x, z, m, batch
        self.policy_loss = float("nan")
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(x.shape[0])
        self.cursor = 0
        self.epochs = 0
        self.steps = 0
        self.loss = float("nan")

    @classmethod
    def load_or_build(cls, max_games: int = 2000, progress=print) -> "Corpus":
        cache = PGN_DIR / f"corpus_{max_games}_policy.npz"      # value targets and the master's move
        if cache.exists():
            data = np.load(cache)
            progress(f"[pretrain] corpus: {data['x'].shape[0]} positions from cache")
            return cls(data["x"], data["z"], data["m"])
        paths = fetch(progress)
        x, z, m = dataset(paths, max_games, progress)
        # Written beside and renamed into place, so a second process that
        # starts meanwhile never reads a half-written file (two servers did).
        partial = cache.with_suffix(".partial.npz")
        np.savez(partial, x=x.astype(np.float32), z=z.astype(np.float32), m=m.astype(np.int32))
        partial.replace(cache)
        return cls(x, z, m)

    def step(self, player: SelfPlay) -> float:
        """One minibatch of supervised training; the running loss is an
        exponential average over the last fifty or so steps."""
        if self.cursor + self.batch > self.x.shape[0]:
            self.order = self.rng.permutation(self.x.shape[0])
            self.cursor = 0
            self.epochs += 1
        idx = self.order[self.cursor:self.cursor + self.batch]
        self.cursor += self.batch
        policy = one_hot(self.m[idx]) if self.m is not None else None
        loss = player.net.train_step(self.x[idx], self.z[idx], player.lr, policy=policy)
        self.loss = loss if np.isnan(self.loss) else self.loss * 0.98 + loss * 0.02
        if policy is not None:
            pl = player.net.last_policy_loss
            self.policy_loss = pl if np.isnan(self.policy_loss) else self.policy_loss * 0.98 + pl * 0.02
        self.steps += 1
        if self.steps % 200 == 0:
            player.net.save(player.weights)
        return loss

    def summary(self) -> str:
        done = self.epochs + self.cursor / max(1, self.x.shape[0])
        return (f"pre-training {self.x.shape[0]} master positions · epoch {done:.2f} · value loss {self.loss:.3f}"
                + (f" · move loss {self.policy_loss:.2f}" if not np.isnan(self.policy_loss) else ""))


def ensure(player: SelfPlay, progress=print, max_games: int = 2000, epochs: int = 8) -> bool:
    """Pre-train once: skipped when the marker says it has been done."""
    import os
    if MARKER.exists() or os.environ.get("MEOWDOKU_NO_PRETRAIN"):
        return False
    paths = fetch(progress)
    x, z, m = dataset(paths, max_games, progress)
    loss = train(player, x, z, m, epochs, progress=progress)
    MARKER.write_text(json.dumps({"games": int(max_games), "positions": int(x.shape[0]), "loss": loss,
                                  "sources": SOURCES, "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--eval", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="start from new weights and redo the marker")
    args = parser.parse_args(argv)
    if args.fresh:
        for path in (WEIGHTS, MARKER):
            if path.exists():
                path.unlink()
    player = SelfPlay()
    before = player.versus_random(args.eval)
    print(f"[pretrain] vs random before: {before}", flush=True)
    ensure(player, max_games=args.games, epochs=args.epochs)
    after = player.versus_random(args.eval)
    print(f"[pretrain] vs random after: {after}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
