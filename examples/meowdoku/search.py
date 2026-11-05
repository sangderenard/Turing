"""Two-ply search over the value network, batched.

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

Replies use the pseudo-legal generator for speed (a reply that leaves the
opponent's own king attacked is still counted, which only makes the search
cautious); root moves are fully legal.
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
    """The opponent's pseudo-legal replies to ``src -> dst`` as (pieces,
    turn) successor positions, plus whether the mover is then without any
    legal move (mate or stalemate) and whether it is in check."""
    saved = (game.pieces, game.ids, game.turn, getattr(game, "rights", set()), getattr(game, "ep", None))
    game.pieces, game.ids = dict(game.pieces), dict(game.ids)
    game.apply_move(src, dst)
    game.turn = "black" if game.turn == "white" else "white"
    mover = game.turn                                # the side to reply
    in_check = game.in_check(mover)
    replies = []
    if in_check:
        # In check the replies must be the legal ones (the others do not
        # exist), and none of them means mate.
        replies = [game.peek(cell, to) for cell, to in game.all_legal_moves()]
        no_legal = not replies
    else:
        for cell, (kind, side) in list(game.pieces.items()):
            if side != mover:
                continue
            for to in game.pseudo_moves(cell):
                target = game.pieces.get(to)
                if target and target[0] == "king":   # a reply that takes our king: our move was illegal; not here
                    continue
                replies.append(game.peek(cell, to))
        # Stalemate is only possible with few pseudo-legal replies; check then.
        no_legal = (len(replies) <= 12) and not game.all_legal_moves()
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


def best_move(game: Chess, net, *, rng: np.random.Generator | None = None, temperature: float = 0.0):
    moves, root = searched_values(game, net)
    if not moves:
        return None
    sign = 1.0 if game.turn == "white" else -1.0
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
