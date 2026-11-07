"""Legal, batched chess search used by Lichess and play-the-network."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.games import Chess  # noqa: E402
from meowdoku.search import best_move  # noqa: E402


class FlatNet:
    def __init__(self):
        self.batches = []

    def values(self, boards):
        self.batches.append(len(boards))
        return np.zeros(len(boards))


def test_three_ply_search_batches_leaves_and_preserves_board():
    game = Chess("search")
    before = (dict(game.pieces), dict(game.ids), game.turn, set(game.rights), game.ep)
    net = FlatNet()
    move = best_move(game, net)
    assert move in game.all_legal_moves()
    assert len(net.batches) == 2
    assert all(size > 0 for size in net.batches)
    assert (game.pieces, game.ids, game.turn, game.rights, game.ep) == before


def test_three_ply_search_keeps_fools_mate():
    game = Chess("fools-mate")
    for move in ((53, 45), (12, 28), (54, 38)):
        game.apply_move(*move)
        game.turn = "black" if game.turn == "white" else "white"
    assert best_move(game, FlatNet()) == (3, 39)


def test_legal_moves_never_capture_a_king():
    game = Chess("check")
    game.pieces = {4: ("king", "black"), 60: ("king", "white"), 12: ("rook", "white")}
    game.ids = {4: "black-king", 60: "white-king", 12: "white-rook"}
    game.turn = "white"
    assert 4 in game.pseudo_moves(12)
    assert 4 not in game.legal_moves(12)
