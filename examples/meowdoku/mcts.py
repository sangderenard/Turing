"""Monte Carlo tree search over the network: hundreds to thousands of
playouts a move, the way a network is meant to be searched.

PUCT selection (AlphaZero's rule): from the root, descend by the move with
the best  Q + c * P * sqrt(N_parent) / (1 + N)  until a position the tree
has not evaluated.  Leaves are gathered ``batch`` at a time (a virtual loss
keeps the selections apart) and go through ONE forward pass of the
network, which returns both the value and the policy prior for every
legal move; the value is backed up the path from the perspective of the
player to move at each node.  Checkmate and stalemate are exact: ±1 / 0.

    result = search(game, net, simulations=800)        # or seconds=3.0
    result.move                                        # most visited
    result.policy                                      # visit distribution (POLICY floats) as a training target
    result.value                                       # root value for the side to move

The move generator is the page's own ``Chess`` (full rules), cloned per
node; the arithmetic of the fold (priors, Q, UCB) is plain per-node
float work on small arrays and the network batch is AbstractTensor."""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402
from meowdoku.chess_net import POLICY, encode, legal_mask, move_index  # noqa: E402

VIRTUAL_LOSS = 1.0


def clone(game: Chess, move: tuple[int, int] | None = None) -> Chess:
    """A full copy (castling rights and en passant included), after ``move``."""
    child = Chess.__new__(Chess)
    child.board_id = getattr(game, "board_id", "mcts")
    child.pieces, child.ids = dict(game.pieces), dict(game.ids)
    child.selected, child.turn = None, game.turn
    child.rights, child.ep = set(getattr(game, "rights", set())), getattr(game, "ep", None)
    child.history, child.moves = [], getattr(game, "moves", 0)
    if move is not None:
        child.apply_move(*move)
        child.turn = "black" if child.turn == "white" else "white"
        child.moves += 1
    return child


class Node:
    __slots__ = ("game", "moves", "P", "N", "W", "V", "children", "terminal", "value")

    def __init__(self, game: Chess):
        self.game = game
        self.moves: list[tuple[int, int]] = game.all_legal_moves()
        self.children: dict[int, Node] = {}
        self.terminal = not self.moves
        self.value = 0.0                       # for White, exact at terminals
        if self.terminal:
            end = game.outcome()
            self.value = 1.0 if end == "white" else -1.0 if end == "black" else 0.0
        k = len(self.moves)
        self.P = np.full(k, 1.0 / max(k, 1))
        self.N = np.zeros(k)
        self.W = np.zeros(k)                   # value sums for the side to move here
        self.V = np.zeros(k)                   # virtual losses in flight

    def select(self, c_puct: float) -> int:
        n = self.N + self.V
        q = np.where(n > 0, (self.W - self.V * VIRTUAL_LOSS) / np.maximum(n, 1), 0.0)
        u = c_puct * self.P * math.sqrt(1.0 + n.sum()) / (1.0 + n)
        return int(np.argmax(q + u))


@dataclass
class Result:
    move: tuple[int, int] | None
    policy: np.ndarray
    value: float
    simulations: int
    seconds: float
    visits: dict

    def describe(self, game: Chess) -> str:
        top = sorted(self.visits.items(), key=lambda kv: -kv[1])[:3]
        return (f"{self.simulations} playouts in {self.seconds:.1f}s · value {self.value:+.2f} · "
                + " ".join(f"{a}{b}:{int(n)}" for (a, b), n in top))


def search(game: Chess, net, simulations: int = 800, seconds: float | None = None, batch: int = 32,
           c_puct: float = 1.5, rng: np.random.Generator | None = None, temperature: float = 0.0,
           dirichlet: float = 0.0, min_simulations: int = 64) -> Result:
    started = time.perf_counter()
    root = Node(clone(game))
    if root.terminal:
        return Result(None, np.zeros(POLICY), root.value, 0, 0.0, {})
    sign = 1.0 if game.turn == "white" else -1.0
    _expand([root], net)
    if dirichlet > 0 and rng is not None and len(root.moves) > 1:
        noise = rng.dirichlet([dirichlet] * len(root.moves))
        root.P = 0.75 * root.P + 0.25 * noise
    done = 0
    while True:
        if seconds is not None:
            if done >= min_simulations and time.perf_counter() - started >= seconds:
                break
            if done >= max(simulations, min_simulations) * 4:
                break
        elif done >= simulations:
            break
        pending: list[tuple[Node, list[tuple[Node, int]]]] = []
        for _ in range(batch):
            node, path = root, []
            while True:
                a = node.select(c_puct)
                path.append((node, a))
                node.V[a] += 1
                child = node.children.get(a)
                if child is None:
                    child = Node(clone(node.game, node.moves[a]))
                    node.children[a] = child
                    if child.terminal:
                        _backup(path, child.value)
                    else:
                        pending.append((child, path))
                    break
                if child.terminal:
                    _backup(path, child.value)
                    break
                if child.N.sum() + child.V.sum() == 0 and child not in [p[0] for p in pending]:
                    # evaluated but never descended: treat as a fresh leaf visit
                    pass
                node = child
            done += 1
            if seconds is None and done >= simulations:
                break
        if pending:
            _expand([n for n, _ in pending], net)
            for node, path in pending:
                _backup(path, node.value)
    visits = {root.moves[k]: float(root.N[k]) for k in range(len(root.moves))}
    policy = np.zeros(POLICY)
    for (src, dst), n in visits.items():
        policy[move_index(src, dst)] = n
    total = policy.sum()
    if total > 0:
        policy /= total
    counts = root.N
    if temperature > 0 and rng is not None:
        weights = counts ** (1.0 / temperature)
        weights = weights / weights.sum() if weights.sum() > 0 else root.P
        k = int(rng.choice(len(root.moves), p=weights))
    else:
        k = int(np.argmax(counts + 1e-6 * root.P))
    value = float(root.W[k] / root.N[k]) if root.N[k] > 0 else 0.0
    return Result(root.moves[k], policy, value, done, time.perf_counter() - started, visits)


def _expand(nodes: list[Node], net) -> None:
    """One network batch: value for White and priors over legal moves."""
    boards = np.stack([encode(n.game.pieces, n.game.turn) for n in nodes])
    masks = np.stack([legal_mask(n.moves) for n in nodes])
    values, probs = net.evaluate(boards, masks)
    for node, value, prob in zip(nodes, values, probs):
        node.value = float(value)
        prior = np.array([prob[move_index(s, d)] for s, d in node.moves])
        total = prior.sum()
        node.P = prior / total if total > 0 else np.full(len(node.moves), 1.0 / len(node.moves))


def _backup(path: list[tuple[Node, int]], value_white: float) -> None:
    for node, a in path:
        node.V[a] -= 1
        node.N[a] += 1
        node.W[a] += value_white if node.game.turn == "white" else -value_white


def best_move(game: Chess, net, simulations: int = 400, seconds: float | None = None, **kw):
    return search(game, net, simulations=simulations, seconds=seconds, **kw).move
