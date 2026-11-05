"""A chess value network on AbstractTensor (numpy backend) that learns by
playing itself.

The network is small and explicit: one hidden tanh layer over a 769-float
board encoding (12 piece planes of 64 squares plus the side to move) and a
tanh value head that predicts the game's outcome for White in [-1, 1].
The forward pass is AbstractTensor arithmetic and the gradient comes from
the library's own autograd (``AbstractTensor.autograd.grad`` over the
tracked parameters); the update runs under ``no_grad``.

Play is a one-ply lookahead: every legal move's successor position is
encoded, the whole batch is valued in one matmul, and the mover takes the
best value for its side (softmax-sampled at a temperature while learning,
greedy when asked).  A game ends when a king is taken (the chess here is
the page's pseudo-legal chess) or at ``MAX_PLIES``, a draw.

Training is Monte Carlo on the finished game: every position's target is
the outcome discounted by the plies left (sooner wins are worth more) plus
a small material term so that captures teach something before any game is
won.  ``SelfPlay`` keeps the network, the record, the losses and the
weights file; ``games.ChessArena`` ticks it from the page.

    python examples/meowdoku/chess_net.py --games 40      # train headless, report
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402

KINDS = ("king", "queen", "rook", "bishop", "knight", "pawn")
VALUES = {"king": 0.0, "queen": 9.0, "rook": 5.0, "bishop": 3.0, "knight": 3.0, "pawn": 1.0}
FEATURES = 12 * 64 + 1
MAX_PLIES = 160
WEIGHTS = ROOT / "build" / "meowdoku" / "chess_net.npz"


def encode(pieces: dict[int, tuple[str, str]], turn: str) -> np.ndarray:
    """769 floats: plane (kind, side) x square, then +1/-1 for the mover."""
    x = np.zeros(FEATURES, dtype=np.float64)
    for cell, (kind, side) in pieces.items():
        x[(KINDS.index(kind) + (0 if side == "white" else 6)) * 64 + cell] = 1.0
    x[-1] = 1.0 if turn == "white" else -1.0
    return x


def material(pieces: dict[int, tuple[str, str]]) -> float:
    """White minus black material, in pawns."""
    return sum(VALUES[k] * (1.0 if s == "white" else -1.0) for k, s in pieces.values())


class ValueNet:
    """x (N, 769) -> tanh(tanh(x W1 + b1) W2 + b2), the value for White."""

    def __init__(self, hidden: int = 96, seed: int = 0):
        from src.common.tensors import AbstractTensor as AT
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        with AT.use_backend("numpy"):
            self.W1 = AT.tensor(rng.normal(0.0, 1.0 / math.sqrt(FEATURES), (FEATURES, hidden)))
            self.b1 = AT.tensor(np.zeros((1, hidden)))
            self.W2 = AT.tensor(rng.normal(0.0, 1.0 / math.sqrt(hidden), (hidden, 1)))
            self.b2 = AT.tensor(np.zeros((1, 1)))

    def forward(self, x):
        h = (x @ self.W1 + self.b1).tanh()
        v = (h @ self.W2 + self.b2).tanh()
        return h, v

    def values(self, boards: np.ndarray) -> np.ndarray:
        from src.common.tensors import AbstractTensor as AT
        with AT.use_backend("numpy"), AT.autograd.no_grad():
            _, v = self.forward(AT.tensor(np.asarray(boards, dtype=np.float64).reshape(-1, FEATURES)))
            return np.asarray(v.numpy(), dtype=np.float64).reshape(-1)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def train_step(self, boards: np.ndarray, targets: np.ndarray, lr: float = 0.01) -> float:
        """One gradient step of the mean squared error through the tensor
        library's autograd: the parameters are tracked, the loss is formed,
        ``autograd.grad`` walks the tape, and the update runs under
        ``no_grad``."""
        from src.common.tensors import AbstractTensor as AT
        with AT.use_backend("numpy"):
            params = self.parameters()
            for p in params:
                p.requires_grad_()
            x = AT.tensor(np.asarray(boards, dtype=np.float64).reshape(-1, FEATURES))
            z = AT.tensor(np.asarray(targets, dtype=np.float64).reshape(-1, 1))
            _, v = self.forward(x)
            diff = v - z
            loss = (diff * diff).mean()
            grads = AT.autograd.grad(loss, params, retain_graph=False)
            with AT.autograd.no_grad():
                for p, g in zip(params, grads):
                    AT.copyto(p, p - g * lr)
            return float(np.asarray(loss.numpy()))

    def save(self, path: Path = WEIGHTS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=np.asarray(self.W1.numpy()), b1=np.asarray(self.b1.numpy()),
                 W2=np.asarray(self.W2.numpy()), b2=np.asarray(self.b2.numpy()))

    def load(self, path: Path = WEIGHTS) -> bool:
        from src.common.tensors import AbstractTensor as AT
        if not path.exists():
            return False
        data = np.load(path)
        if data["W1"].shape[1] != self.hidden:
            return False
        with AT.use_backend("numpy"):
            self.W1, self.b1, self.W2, self.b2 = (AT.tensor(data[k]) for k in ("W1", "b1", "W2", "b2"))
        return True


def legal_moves(game: Chess) -> list[tuple[int, int]]:
    return [(src, dst) for src, (kind, side) in list(game.pieces.items()) if side == game.turn
            for dst in game.legal_moves(src)]


def king_taken(game: Chess) -> str | None:
    """The side whose king is gone, or None."""
    kings = {side for kind, side in game.pieces.values() if kind == "king"}
    for side in ("white", "black"):
        if side not in kings:
            return side
    return None


class SelfPlay:
    """The network, its opponent (itself), the record and the trainer."""

    def __init__(self, hidden: int = 96, seed: int = 0, lr: float = 0.01, temperature: float = 0.6,
                 weights: Path = WEIGHTS):
        self.net = ValueNet(hidden, seed)
        self.rng = np.random.default_rng(seed)
        self.lr, self.temperature = lr, temperature
        self.weights = weights
        self.loaded = self.net.load(weights)
        self.games = 0
        self.record = {"white": 0, "black": 0, "draw": 0}
        self.losses: list[float] = []
        self.last_loss = float("nan")
        self.plies_history: list[int] = []

    # -- moving ----------------------------------------------------------
    def choose(self, game: Chess, greedy: bool = False) -> tuple[int, int] | None:
        moves = legal_moves(game)
        if not moves:
            return None
        boards = np.stack([encode(*game.peek(src, dst)) for src, dst in moves])
        values = self.net.values(boards)
        if game.turn == "black":
            values = -values                                        # black wants White's value low
        if greedy:
            return moves[int(np.argmax(values))]
        logits = (values - values.max()) / max(self.temperature, 1e-3)
        probability = np.exp(logits); probability /= probability.sum()
        return moves[int(self.rng.choice(len(moves), p=probability))]

    # -- a whole game ------------------------------------------------------
    def play_game(self, greedy: bool = False) -> tuple[list[np.ndarray], list[float], str]:
        game = Chess("selfplay")
        boards, materials = [], []
        for _ply in range(MAX_PLIES):
            boards.append(encode(game.pieces, game.turn))
            materials.append(material(game.pieces))
            move = self.choose(game, greedy)
            if move is None:
                break
            game.cell(*game.rc(move[0]), "primary")
            game.cell(*game.rc(move[1]), "primary")
            loser = king_taken(game)
            if loser:
                return boards, materials, ("black" if loser == "white" else "white")
        return boards, materials, "draw"

    def targets(self, materials: list[float], result: str) -> np.ndarray:
        """Outcome for White, discounted toward the end, plus a material term."""
        outcome = {"white": 1.0, "black": -1.0, "draw": 0.0}[result]
        n = len(materials)
        discounted = np.array([outcome * (0.985 ** (n - 1 - t)) for t in range(n)])
        shaping = np.tanh(np.asarray(materials) / 12.0) * 0.25
        return np.clip(discounted + shaping, -1.0, 1.0)

    def learn_from(self, boards, materials, result) -> float:
        x = np.stack(boards)
        z = self.targets(materials, result)
        loss = self.net.train_step(x, z, self.lr)
        self.losses.append(loss)
        self.last_loss = loss
        self.games += 1
        self.record[result] += 1
        self.plies_history.append(len(boards))
        if self.games % 10 == 0:
            self.net.save(self.weights)
        return loss

    def train(self, games: int, report_every: int = 10) -> None:
        for _ in range(games):
            started = time.perf_counter()
            boards, materials, result = self.play_game()
            loss = self.learn_from(boards, materials, result)
            if self.games % report_every == 0:
                recent = self.losses[-report_every:]
                print(f"[selfplay] game {self.games}: {result:5s} in {len(boards):3d} plies, loss {np.mean(recent):.4f}, "
                      f"record W{self.record['white']} B{self.record['black']} D{self.record['draw']}, "
                      f"{time.perf_counter() - started:.1f}s", flush=True)

    # -- measuring: the net as White against a random mover ------------------
    def versus_random(self, games: int = 10) -> dict[str, int]:
        score = {"win": 0, "loss": 0, "draw": 0}
        for g in range(games):
            game = Chess("eval")
            result = "draw"
            for _ply in range(MAX_PLIES):
                if game.turn == "white":
                    move = self.choose(game, greedy=True)
                else:
                    moves = legal_moves(game)
                    move = moves[int(self.rng.integers(len(moves)))] if moves else None
                if move is None:
                    break
                game.cell(*game.rc(move[0]), "primary"); game.cell(*game.rc(move[1]), "primary")
                loser = king_taken(game)
                if loser:
                    result = "loss" if loser == "white" else "win"; break
            score[result] += 1
        return score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fresh", action="store_true", help="ignore saved weights")
    parser.add_argument("--eval", type=int, default=10, help="games against a random mover before and after")
    args = parser.parse_args(argv)
    weights = WEIGHTS if not args.fresh else WEIGHTS.with_name("chess_net_fresh.npz")
    if args.fresh and weights.exists():
        weights.unlink()
    player = SelfPlay(args.hidden, lr=args.lr, weights=weights)
    print(f"[selfplay] weights loaded: {player.loaded}", flush=True)
    before = player.versus_random(args.eval)
    print(f"[selfplay] vs random before: {before}", flush=True)
    player.train(args.games)
    after = player.versus_random(args.eval)
    print(f"[selfplay] vs random after {args.games} games: {after}", flush=True)
    player.net.save(weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
