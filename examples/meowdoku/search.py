"""Batched minimax search over the value network.

A one-ply player takes the move whose successor looks best and is mated by
anything that calculates.  This search looks one reply deeper: for every
legal move the replies are generated, every leaf position of the whole
tree is encoded into ONE batch and valued in one forward pass, and the
tree is folded by tensor reductions: the opponent's best reply is a min
(or max) over each move's leaves, the move chosen is the max (or min) over
moves.  Padding fills the ragged move/reply table so the fold is two
reductions over a dense (moves, replies) matrix, the shape of thing the
compiler lanes take.

Terminal leaves need no network: a move after which the opponent has no
reply is mate (±1) or stalemate (0); a reply that captures the king is a
move we must not make (the root moves are legal, so only replies into our
own king are impossible and skipped).

    best_move(game, net)            -> (src, dst)
    search_value(game, net)         -> the root's searched value for White

All plies use legal moves. The best root candidates receive a third ply in
one additional batched network call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402
from meowdoku.chess_net import encode  # noqa: E402

MATE = 1.0
PAD = 2.0                                           # beyond any value: never chosen by a min, never by a max when negated


def _replies(game: Chess, src: int, dst: int):
    """The opponent's legal replies to ``src -> dst`` as successor positions."""
    saved = (game.pieces, game.ids, game.turn, getattr(game, "rights", set()), getattr(game, "ep", None))
    game.pieces, game.ids = dict(game.pieces), dict(game.ids)
    game.apply_move(src, dst)
    game.turn = "black" if game.turn == "white" else "white"
    mover = game.turn                                # the side to reply
    in_check = game.in_check(mover)
    replies = [game.peek(cell, to) for cell, to in game.all_legal_moves()]
    no_legal = not replies
    game.pieces, game.ids, game.turn = saved[:3]
    game.rights, game.ep = saved[3], saved[4]
    return replies, no_legal, in_check


def searched_values(game: Chess, net) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Every legal root move with its two-ply value for White."""
    from src.common.tensors import AbstractTensor as AT

    moves = game.all_legal_moves()
    if not moves:
        return [], np.zeros(0)
    white_to_move = game.turn == "white"
    leaves, spans, terminal = [], [], []
    for src, dst in moves:
        replies, no_legal, in_check = _replies(game, src, dst)
        start = len(leaves)
        if no_legal:
            value = (MATE if white_to_move else -MATE) if in_check else 0.0
            terminal.append(value)
            spans.append((start, start))
        else:
            terminal.append(None)
            leaves.extend(encode(pieces, turn) for pieces, turn in replies)
            spans.append((start, len(leaves)))
    values = net.values(np.stack(leaves)) if leaves else np.zeros(0)
    # The dense (moves, replies) table, padded so the opponent's fold ignores it.
    width = max(1, max(b - a for a, b in spans))
    table = np.full((len(moves), width), PAD if white_to_move else -PAD, dtype=np.float64)
    for k, (a, b) in enumerate(spans):
        table[k, : b - a] = values[a:b]
    with AT.use_backend("numpy"):
        folded = AT.tensor(table)
        # The opponent answers with the reply worst for us: min over
        # replies when White moves, max when Black moves.
        best_reply = folded.min(dim=1) if white_to_move else folded.max(dim=1)
        root = np.asarray(best_reply.numpy(), dtype=np.float64).reshape(-1)
    for k, value in enumerate(terminal):
        if value is not None:
            root[k] = value
    return moves, root


def _after(game: Chess, move: tuple[int, int]) -> Chess:
    """A full legal successor, including castling and en-passant state."""
    child = Chess("search")
    child.pieces, child.ids = dict(game.pieces), dict(game.ids)
    child.turn = game.turn
    child.rights, child.ep = set(game.rights), game.ep
    child.apply_move(*move)
    child.turn = "black" if child.turn == "white" else "white"
    return child


def _third_ply_values(game: Chess, net, moves: list[tuple[int, int]]) -> np.ndarray:
    """Minimax one ply further for selected root moves, with one leaf batch."""
    white_root = game.turn == "white"
    leaves, reply_groups = [], []
    for root_move in moves:
        after_root = _after(game, root_move)
        replies = after_root.all_legal_moves()
        if not replies:
            outcome = after_root.outcome()
            reply_groups.append((MATE if outcome == "white" else -MATE if outcome == "black" else 0.0, []))
            continue
        group = []
        for reply in replies:
            after_reply = _after(after_root, reply)
            ours = after_reply.all_legal_moves()
            if not ours:
                outcome = after_reply.outcome()
                value = MATE if outcome == "white" else -MATE if outcome == "black" else 0.0
                group.append((value, None))
                continue
            start = len(leaves)
            leaves.extend(encode(*after_reply.peek(*move)) for move in ours)
            group.append((None, (start, len(leaves))))
        reply_groups.append((None, group))
    values = net.values(np.stack(leaves)) if leaves else np.zeros(0)
    scores = []
    for terminal, group in reply_groups:
        if terminal is not None:
            scores.append(terminal)
            continue
        reply_scores = []
        for value, span in group:
            if span is not None:
                a, b = span
                value = float(values[a:b].max() if white_root else values[a:b].min())
            reply_scores.append(value)
        scores.append(min(reply_scores) if white_root else max(reply_scores))
    return np.asarray(scores, dtype=np.float64)


def best_move(game: Chess, net, *, rng: np.random.Generator | None = None,
              temperature: float = 0.0, depth: int = 3, beam: int = 3):
    moves, root = searched_values(game, net)
    if not moves:
        return None
    sign = 1.0 if game.turn == "white" else -1.0
    if depth >= 3 and len(moves) > 1:
        selected = np.argsort(-(root * sign), kind="stable")[:max(1, beam)]
        deeper = _third_ply_values(game, net, [moves[int(i)] for i in selected])
        # The deeper search ranks the selected moves against each other.
        moves = [moves[int(i)] for i in selected]
        root = deeper
    scores = root * sign
    if rng is not None and temperature > 0:
        logits = (scores - scores.max()) / temperature
        probability = np.exp(logits); probability /= probability.sum()
        return moves[int(rng.choice(len(moves), p=probability))]
    return moves[int(np.argmax(scores))]


def search_value(game: Chess, net) -> float:
    moves, root = searched_values(game, net)
    if not moves:
        return (-MATE if game.turn == "white" else MATE) if game.in_check(game.turn) else 0.0
    return float(root.max() if game.turn == "white" else root.min())
