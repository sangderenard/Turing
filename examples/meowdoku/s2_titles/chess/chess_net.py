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
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s2_titles.classic.games import Chess  # noqa: E402

KINDS = ("king", "queen", "rook", "bishop", "knight", "pawn")
VALUES = {"king": 0.0, "queen": 9.0, "rook": 5.0, "bishop": 3.0, "knight": 3.0, "pawn": 1.0}
FEATURES = 12 * 64 + 1
MAX_PLIES = 160
WEIGHTS = ROOT / "build" / "meowdoku" / "chess_conv.npz"
POLICY_PLANES = 64                                     # 56 queen-like (8 directions x 7 steps) + 8 knight
POLICY = POLICY_PLANES * 64                            # logits: plane * 64 + source square
_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def move_index(src: int, dst: int) -> int:
    """The policy logit of a move: its source square on the plane of its
    direction and distance (knight jumps have their own eight planes)."""
    dr, dc = dst // 8 - src // 8, dst % 8 - src % 8
    if (dr, dc) in _KNIGHT:
        plane = 56 + _KNIGHT.index((dr, dc))
    else:
        distance = max(abs(dr), abs(dc))
        plane = _DIRS.index((dr // distance, dc // distance)) * 7 + distance - 1
    return plane * 64 + src


def legal_mask(moves) -> np.ndarray:
    mask = np.zeros(POLICY, dtype=np.float64)
    for src, dst in moves:
        mask[move_index(src, dst)] = 1.0
    return mask
PLANES = 13                                            # 12 piece planes + the side to move


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


class ConvValueNet:
    """The board as 13 planes of 8x8 (12 piece planes, one plane of the side
    to move) through two 3x3 convolutions, a hidden layer and a tanh value
    head: White's outcome in [-1, 1]. A third convolution is a residual
    refinement initialized to zero, so existing eight-tensor checkpoints
    produce exactly the same values before that layer learns.

    Each convolution is ``unfold2d`` (the library's im2col, on the autograd
    tape) followed by one matmul with the kernel matrix, so the whole
    forward is AbstractTensor arithmetic with no custom layer code; the
    gradient is ``AbstractTensor.autograd.grad`` over the eight parameter
    tensors and the update is ``abstract_nn.Adam``.  Same interface as the
    MLP it replaced: ``values(boards)`` and ``train_step(boards, targets)``
    over the 769-float encoding, reshaped into planes here.  Measured: fits
    64 random targets to zero in 50 steps, 46 ms a step; 900 positions
    forward in 0.10 s."""

    def __init__(self, channels: int = 32, hidden: int = 128, seed: int = 0, lr: float = 1e-3):
        from src.common.tensors import AbstractTensor as AT
        from src.common.tensors.abstract_nn import Adam
        rng = np.random.default_rng(seed)
        # AbstractTensor.autograd.no_grad is process-wide. Lichess evaluation
        # must finish its no_grad scope before the learner records a tape.
        self.lock = threading.RLock()
        self.channels, self.hidden, self.lr = channels, hidden, lr
        c1 = c2 = channels
        with AT.use_backend("numpy"):
            def param(shape, scale):
                tensor = AT.tensor(rng.normal(0.0, scale, shape) if scale else np.zeros(shape))
                tensor.requires_grad_()
                return tensor
            self.W1 = param((c1, PLANES * 9), math.sqrt(2.0 / (PLANES * 9))); self.b1 = param((c1, 1), 0.0)
            self.W2 = param((c2, c1 * 9), math.sqrt(2.0 / (c1 * 9))); self.b2 = param((c2, 1), 0.0)
            self.W3 = param((c2 * 64, hidden), math.sqrt(2.0 / (c2 * 64))); self.b3 = param((1, hidden), 0.0)
            self.W4 = param((hidden, 1), math.sqrt(1.0 / hidden)); self.b4 = param((1, 1), 0.0)
            self.W5 = param((c2, c2 * 9), 0.0); self.b5 = param((c2, 1), 0.0)
            # policy head: one logit per (move plane, square) from the square's features
            self.W6 = param((POLICY_PLANES, c2), math.sqrt(1.0 / c2)); self.b6 = param((POLICY_PLANES, 1), 0.0)
            self.optimizer = Adam(self.parameters(), lr=lr)
        self.last_policy_loss = float("nan")
        self.version = 0          # bumped by every weight change; the compiled evaluator copies weights only when it moves
        self._native = None
        self.native_note = "numpy"

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4,
                self.W5, self.b5, self.W6, self.b6]

    @staticmethod
    def planes(boards: np.ndarray) -> np.ndarray:
        x = np.asarray(boards, dtype=np.float64).reshape(-1, FEATURES)
        out = np.empty((x.shape[0], PLANES, 8, 8), dtype=np.float64)
        out[:, :12] = x[:, :768].reshape(-1, 12, 8, 8)
        out[:, 12] = x[:, 768][:, None, None]
        return out

    def forward(self, x):
        """x (N, 13, 8, 8) -> value (N, 1)."""
        return self.forward_both(x)[0]

    def forward_both(self, x):
        """x (N, 13, 8, 8) -> (value (N, 1), policy logits (N, POLICY))."""
        n, c1, c2 = int(x.shape[0]), self.channels, self.channels
        cols = x.unfold2d((3, 3), padding=1)                       # (N, 13*9, 64)
        h = self.W1 @ cols + self.b1                               # (N, c1, 64)
        h = h.maximum(h * 0.0)
        cols2 = h.reshape((n, c1, 8, 8)).unfold2d((3, 3), padding=1)
        h2 = self.W2 @ cols2 + self.b2                             # (N, c2, 64)
        h2 = h2.maximum(h2 * 0.0)
        residual = (self.W5 @ h2.reshape((n, c2, 8, 8)).unfold2d((3, 3), padding=1) + self.b5).tanh()
        h2 = h2 + residual
        logits = (self.W6 @ h2 + self.b6).reshape((n, POLICY))   # (N, planes, 64) -> plane * 64 + square
        h3 = h2.reshape((n, c2 * 64)) @ self.W3 + self.b3
        h3 = h3.maximum(h3 * 0.0)
        return (h3 @ self.W4 + self.b4).tanh(), logits

    def use_native(self, contract: str = "eigen", batches=(1, 8, 32)) -> bool:
        """Route ``evaluate`` through this forward compiled by the repository
        compiler under work contract ``contract`` (``native_forward``).  On
        any compile failure the net stays on the numpy path and
        ``native_note`` says why."""
        try:
            from meowdoku.s2_titles.chess.native_forward import NativeRouter
            self._native = NativeRouter(self, contract=contract, batches=batches)
            self.native_note = f"compiled ({contract}, batches {'/'.join(map(str, batches))})"
            return True
        except Exception as error:
            self._native = None
            self.native_note = f"numpy (compile failed: {type(error).__name__}: {str(error)[:80]})"
            return False

    def evaluate(self, boards: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Values for White and the policy over legal moves (masked softmax)."""
        if self._native is not None:
            return self._native.evaluate(boards, masks)
        return self.evaluate_numpy(boards, masks)

    def evaluate_numpy(self, boards: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from src.common.tensors import AbstractTensor as AT
        with self.lock, AT.use_backend("numpy"), AT.autograd.no_grad():
            v, logits = self.forward_both(AT.tensor(self.planes(boards)))
            masked = logits + (AT.tensor(np.asarray(masks, dtype=np.float64)) - 1.0) * 1e4
            probs = masked.softmax(dim=1)
            return (np.asarray(v.numpy(), dtype=np.float64).reshape(-1),
                    np.asarray(probs.numpy(), dtype=np.float64))

    def values(self, boards: np.ndarray) -> np.ndarray:
        with self.lock:
            return self._values(boards)

    def _values(self, boards: np.ndarray) -> np.ndarray:
        from src.common.tensors import AbstractTensor as AT
        with AT.use_backend("numpy"), AT.autograd.no_grad():
            v = self.forward(AT.tensor(self.planes(boards)))
            return np.asarray(v.numpy(), dtype=np.float64).reshape(-1)

    def train_step(self, boards: np.ndarray, targets: np.ndarray, lr: float | None = None,
                   policy: np.ndarray | None = None) -> float:
        """One Adam step on the value loss (mean squared error against
        ``targets``) and, when ``policy`` rows (N, POLICY) are given, the
        cross-entropy of the policy head against them.  Returns the value
        loss; the policy loss is kept in ``last_policy_loss``."""
        with self.lock:
            return self._train_step(boards, targets, lr, policy)

    def _train_step(self, boards: np.ndarray, targets: np.ndarray, lr: float | None = None,
                    policy: np.ndarray | None = None) -> float:
        from src.common.tensors import AbstractTensor as AT
        with AT.use_backend("numpy"):
            params = self.parameters()
            x = AT.tensor(self.planes(boards))
            z = AT.tensor(np.asarray(targets, dtype=np.float64).reshape(-1, 1))
            value, logits = self.forward_both(x)
            diff = value - z
            loss = (diff * diff).mean()
            if policy is not None:
                target = AT.tensor(np.asarray(policy, dtype=np.float64).reshape(-1, POLICY))
                # log-softmax spelled out on the tape (the library's own is a
                # data-only op); the shift is a constant, so the gradient is exact.
                shift = AT.tensor(np.max(np.asarray(logits.numpy()), axis=1, keepdims=True))
                shifted = logits - shift
                log_norm = shifted.exp().sum(dim=1).reshape((-1, 1)).log()
                cross_entropy = (target * (shifted - log_norm)).sum(dim=1).mean() * -1.0
                total = loss + cross_entropy
                self.last_policy_loss = float(np.asarray(cross_entropy.numpy()))
            else:
                total = loss + logits.sum() * 0.0          # keeps the policy head on the tape (zero gradient)
            grads = AT.autograd.grad(total, params)
            with AT.autograd.no_grad():
                for p, q in zip(params, self.optimizer.step(params, grads)):
                    AT.copyto(p, q)
            self.version += 1
            return float(np.asarray(loss.numpy()))

    def save(self, path: Path = WEIGHTS) -> None:
        with self.lock:
            self._save(path)

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npz")
        np.savez(partial, **{f"p{i}": np.asarray(p.numpy()) for i, p in enumerate(self.parameters())})
        partial.replace(path)

    def load(self, path: Path = WEIGHTS) -> bool:
        with self.lock:
            return self._load(path)

    def _load(self, path: Path) -> bool:
        """Checkpoints grow with the net: every stored tensor whose shape
        matches is restored, newer heads keep their fresh initialization.
        False when the eight core tensors are not all there."""
        from src.common.tensors import AbstractTensor as AT
        if not path.exists():
            return False
        data = np.load(path)
        params = self.parameters()
        core = 8
        if any(f"p{i}" not in data or data[f"p{i}"].shape != tuple(params[i].shape) for i in range(core)):
            return False
        with AT.use_backend("numpy"), AT.autograd.no_grad():
            for i, p in enumerate(params):
                if f"p{i}" in data and data[f"p{i}"].shape == tuple(p.shape):
                    AT.copyto(p, AT.tensor(data[f"p{i}"]))
        self.version += 1
        return True


def legal_moves(game: Chess) -> list[tuple[int, int]]:
    return game.all_legal_moves()


class SelfPlay:
    """The network, its opponent (itself), the record and the trainer."""

    def __init__(self, hidden: int = 128, seed: int = 0, lr: float = 1e-3, temperature: float = 0.6,
                 weights: Path = WEIGHTS, channels: int = 32):
        self.net = ConvValueNet(channels, hidden, seed, lr)
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
    def pick(self, moves, values, turn: str, greedy: bool):
        """A move from valued candidates: White takes the highest value,
        Black the lowest; softmax at the temperature unless greedy."""
        if turn == "black":
            values = -values
        if greedy:
            return moves[int(np.argmax(values))]
        logits = (values - values.max()) / max(self.temperature, 1e-3)
        probability = np.exp(logits); probability /= probability.sum()
        return moves[int(self.rng.choice(len(moves), p=probability))]

    def choose(self, game: Chess, greedy: bool = False) -> tuple[int, int] | None:
        moves = legal_moves(game)
        if not moves:
            return None
        boards = np.stack([encode(*game.peek(src, dst)) for src, dst in moves])
        return self.pick(moves, self.net.values(boards), game.turn, greedy)

    def batch_engine(self, display: Chess, parallel: int) -> "BatchEngine":
        return BatchEngine(self, display, parallel)

    # -- a whole game ------------------------------------------------------
    def play_game(self, greedy: bool = False) -> tuple[list[np.ndarray], list[float], str]:
        game = Chess("selfplay")
        boards, materials = [], []
        for _ply in range(MAX_PLIES):
            boards.append(encode(game.pieces, game.turn))
            materials.append(material(game.pieces))
            end = game.outcome()
            if end:
                return boards, materials, end
            move = self.choose(game, greedy)
            game.cell(*game.rc(move[0]), "primary")
            game.cell(*game.rc(move[1]), "primary")
        return boards, materials, "draw"

    def targets(self, materials: list[float], result: str) -> np.ndarray:
        """Outcome for White, discounted toward the end, plus a material term."""
        outcome = {"white": 1.0, "black": -1.0, "draw": 0.0}[result]
        n = len(materials)
        discounted = np.array([outcome * (0.985 ** (n - 1 - t)) for t in range(n)])
        shaping = np.tanh(np.asarray(materials) / 12.0) * 0.25
        return np.clip(discounted + shaping, -1.0, 1.0)

    def learn_from(self, boards, materials, result, policies=None) -> float:
        """A finished game: value targets from its outcome and, when the
        moves were searched, the search's visit distributions as policy
        targets for the positions that have them."""
        loss = self.learn_from_games([(boards, materials, result)])
        if policies:
            rows = [(b, pi) for b, pi in zip(boards, policies) if pi is not None]
            if rows:
                z = self.targets(materials, result)
                keep = [k for k, pi in enumerate(policies) if pi is not None]
                with self.net.lock:
                    self.net.train_step(np.stack([boards[k] for k in keep]), z[keep], self.lr,
                                        policy=np.stack([policies[k] for k in keep]))
        return loss

    def learn_from_games(self, finished) -> float:
        """One gradient step over every position of several finished games."""
        with self.net.lock:
            return self._learn_from_games(finished)

    def _learn_from_games(self, finished) -> float:
        x = np.concatenate([np.stack(boards) for boards, _, _ in finished])
        z = np.concatenate([self.targets(materials, result) for _, materials, result in finished])
        loss = self.net.train_step(x, z, self.lr)
        self.losses.append(loss)
        self.last_loss = loss
        for boards, _, result in finished:
            self.games += 1
            self.record[result] += 1
            self.plies_history.append(len(boards))
        if self.games % 10 < len(finished):
            self.net.save(self.weights)
        return loss

    def train(self, games: int, report_every: int = 10, parallel: int = 16) -> None:
        engine = BatchEngine(self, Chess("arena-0"), parallel)
        started, target, reported = time.perf_counter(), self.games + games, self.games
        while self.games < target:
            finished = engine.step()
            if finished and self.games - reported >= report_every:
                reported = self.games
                recent = self.losses[-3:]
                print(f"[selfplay] {self.games} games: last {finished[-1]}, loss {np.mean(recent):.4f}, "
                      f"record W{self.record['white']} B{self.record['black']} D{self.record['draw']}, "
                      f"{time.perf_counter() - started:.0f}s", flush=True)

    # -- measuring: the net as White against a random mover ------------------
    def versus_random(self, games: int = 10) -> dict:
        """The net as White against a random mover: results, and the mean
        material edge at the end (the finer signal while mates are rare)."""
        score = {"win": 0, "loss": 0, "draw": 0}
        edges = []
        for g in range(games):
            game = Chess("eval")
            result = "draw"
            for _ply in range(MAX_PLIES):
                end = game.outcome()
                if end:
                    result = {"white": "win", "black": "loss", "draw": "draw"}[end]; break
                if game.turn == "white":
                    move = self.choose(game, greedy=True)
                else:
                    moves = legal_moves(game)
                    move = moves[int(self.rng.integers(len(moves)))]
                game.cell(*game.rc(move[0]), "primary"); game.cell(*game.rc(move[1]), "primary")
            score[result] += 1
            edges.append(material(game.pieces))
        score["material_edge"] = round(float(np.mean(edges)), 2) if edges else 0.0
        return score


class BatchEngine:
    """Many self-play games stepped together, every move chosen by the
    tree search (``mcts.search``): ``playouts`` playouts a move, moves
    sampled from the visit counts for the first ``explore_plies`` plies
    (Dirichlet noise at the root) and the most visited move after that.
    Each finished game trains the value head on its outcome and the policy
    head on the search's visit distributions.  ``games[0]`` is the display
    game the page shows."""

    def __init__(self, player: SelfPlay, display: Chess, parallel: int, playouts: int | None = None,
                 explore_plies: int = 20):
        import os
        self.player = player
        self.games = [display] + [Chess(f"arena-{k}") for k in range(1, parallel)]
        self.boards = [[] for _ in self.games]
        self.materials = [[] for _ in self.games]
        self.policies = [[] for _ in self.games]
        self._plies = [0] * len(self.games)
        self.pending: list[tuple] = []
        self.train_every = 1
        self.playouts = int(playouts or os.environ.get("MEOWDOKU_ARENA_PLAYOUTS", 200))
        self.explore_plies = explore_plies
        self.last_search = ""

    def plies(self, index: int) -> int:
        return self._plies[index]

    def _restart(self, k: int, result: str) -> None:
        self.pending.append((self.boards[k], self.materials[k], result, self.policies[k]))
        game = self.games[k]
        game.pieces, game.ids, game.selected = {}, {}, None
        game.turn = game.sides[0]
        game.history, game.moves = [], 0
        game.setup()
        self.boards[k], self.materials[k], self.policies[k], self._plies[k] = [], [], [], 0

    def step(self) -> list[str]:
        """Every game one ply further, each by its own search."""
        from meowdoku.s2_titles.chess import mcts
        finished: list[str] = []
        for k, game in enumerate(self.games):
            end = game.outcome()
            if end is None and self._plies[k] >= MAX_PLIES:
                end = "draw"
            if end:
                finished.append(f"{end} in {self._plies[k]} plies")
                self._restart(k, end)
                continue
            self.boards[k].append(encode(game.pieces, game.turn))
            self.materials[k].append(material(game.pieces))
            exploring = self._plies[k] < self.explore_plies
            result = mcts.search(game, self.player.net, simulations=self.playouts, rng=self.player.rng,
                                 temperature=1.0 if exploring else 0.0, dirichlet=0.3)
            self.policies[k].append(result.policy)
            if k == 0:
                self.last_search = result.describe(game)
            move = result.move
            Chess.cell(game, *game.rc(move[0]), "primary"); Chess.cell(game, *game.rc(move[1]), "primary")
            self._plies[k] += 1
        if len(self.pending) >= self.train_every:
            for boards, materials, result, policies in self.pending:
                self.player.learn_from(boards, materials, result, policies)
            self.pending = []
        return finished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fresh", action="store_true", help="ignore saved weights")
    parser.add_argument("--eval", type=int, default=10, help="games against a random mover before and after")
    parser.add_argument("--parallel", type=int, default=16)
    args = parser.parse_args(argv)
    weights = WEIGHTS if not args.fresh else WEIGHTS.with_name("chess_net_fresh.npz")
    if args.fresh and weights.exists():
        weights.unlink()
    player = SelfPlay(args.hidden, lr=args.lr, weights=weights)
    print(f"[selfplay] weights loaded: {player.loaded}", flush=True)
    before = player.versus_random(args.eval)
    print(f"[selfplay] vs random before: {before}", flush=True)
    player.train(args.games, parallel=args.parallel)
    after = player.versus_random(args.eval)
    print(f"[selfplay] vs random after {args.games} games: {after}", flush=True)
    player.net.save(weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
