"""The games behind the web page, as one state protocol.

Every game is a Python state machine the server drives; the page draws
whatever state it is handed and sends back cell taps, text and buttons.
Nothing here renders.  The state a game returns:

    game, id, title, rows, cols
    cells     palette name per cell (rows*cols), the button colours
    marks     0/1 per cell: Meowdoku's mice (cells that cannot hold a cat)
    hints     0/1 per cell: legal destinations of the selected piece, or
              the cells Meowdoku's hint names
    selected  cell index or None
    pieces    [{id, cell, mesh, color}]: id is stable so the page can
              animate a move; cell may be "hud:k" for the hearts row
    palette   {name: [r, g, b]} linear RGB
    hearts, won, lost, status, moves
    input     "meowdoku" (the page's tool picks what a tap does: Cats or
              Mice; double tap is a cat, swipe paints mice),
              "select" (tap a piece, tap a destination), "text" (Wordle)
    text, text_max, rack   the Wordle entry and its letter rack
    palettes, palette_name, hints_used, hint   Meowdoku's colour schemes
              (names, the one in use), hints taken, the last hint's text

Actions: ``cell(r, c, button)`` with button in primary / secondary /
paint / double / cat / mouse (cat places a cat, mouse toggles a mouse),
``text(s)``, ``submit()``, ``backspace()``, ``undo()``, ``reset()``,
``hint()`` and ``set_palette(name)`` (Meowdoku; neither touches the dt
system's state).  Meowdoku wraps the dt-system session; chess and checkers
give piece movement with legal destinations shown before a move is made
(the "optional move": select, see, then commit); Wordle keeps six guesses
and a rack coloured by what the guesses taught.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from meowdoku.puzzle import Puzzle, next_deduction
from meowdoku.session import MeowdokuSession
from meowdoku.words import ALL, ANSWERS


def hsv(h: float, s: float, v: float) -> list[float]:
    i = int((h % 1.0) * 6.0)
    f = (h % 1.0) * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6]


class Game:
    """Base: a board of buttons, pieces on it, and the action verbs."""

    game = "game"
    input = "select"
    title = ""

    def __init__(self, board_id: str = ""):
        self.id = board_id
        self.moves = 0
        self.status = ""
        self.history: list[Any] = []

    # verbs ---------------------------------------------------------------
    def cell(self, r: int, c: int, button: str) -> None: ...
    def text(self, value: str) -> None: ...
    def submit(self) -> None: ...
    def backspace(self) -> None: ...
    def undo(self) -> None: ...
    def reset(self) -> None: ...

    # state -----------------------------------------------------------------
    def base_state(self, rows: int, cols: int) -> dict[str, Any]:
        return {
            "game": self.game, "id": self.id, "title": self.title or self.game,
            "rows": rows, "cols": cols,
            "cells": ["cell"] * (rows * cols), "marks": [0] * (rows * cols), "hints": [0] * (rows * cols),
            "selected": None, "pieces": [], "palette": {"cell": [0.8, 0.8, 0.8]},
            "hearts": None, "won": False, "lost": False, "status": self.status, "moves": self.moves,
            "input": self.input, "text": "", "text_max": 0, "rack": None,
        }

    def state(self) -> dict[str, Any]:
        raise NotImplementedError

    def result(self) -> dict[str, Any] | None:
        """A finished game's record for the player's profile: ``score``
        (higher is better) and ``detail``; None while it goes on."""
        return None


# ---------------------------------------------------------------------------
# Meowdoku: the dt-system session behind the protocol
# ---------------------------------------------------------------------------

#: Region colour schemes: (r, g, b) lists in hue order.  Regions take them in
#: the rank of their ring-relaxed hue, so regions the relaxation put far
#: apart in hue stay far apart in the scheme.  A board with n regions takes
#: n evenly spaced entries.  The cat and mouse colours are fixed for
#: contrast against the tiles (dark cat, pale mouse).
PALETTES: dict[str, list[tuple[float, float, float]]] = {
    "Candy": [(0.95, 0.35, 0.35), (1.00, 0.60, 0.15), (0.98, 0.85, 0.20), (0.60, 0.85, 0.25), (0.20, 0.72, 0.40),
              (0.15, 0.78, 0.85), (0.25, 0.45, 0.95), (0.58, 0.38, 0.90), (0.88, 0.35, 0.82), (0.98, 0.52, 0.68)],
    # Okabe-Ito (distinguishable under the common colour-vision deficiencies)
    # plus grey, lavender and sand, spread in lightness as well as hue.
    "Colour-blind safe": [(0.84, 0.37, 0.00), (0.90, 0.62, 0.00), (0.82, 0.72, 0.55), (0.94, 0.89, 0.26), (0.00, 0.62, 0.45),
                          (0.62, 0.62, 0.62), (0.34, 0.71, 0.91), (0.00, 0.45, 0.70), (0.75, 0.70, 0.90), (0.80, 0.47, 0.65)],
    "Pastel": [tuple(hsv(k / 10.0 + 0.02, 0.36, 0.95)) for k in range(10)],
}
DEFAULT_PALETTE = "Candy"


class MeowdokuGame(Game):
    game = "meowdoku"
    input = "meowdoku"

    def __init__(self, board: dict[str, Any], palette: str = DEFAULT_PALETTE):
        super().__init__(str(board["id"]))
        n = int(board["n"])
        cells = np.arange(n * n)
        puzzle = Puzzle(n, (cells // n).astype(np.float64), (cells % n).astype(np.float64),
                        np.asarray(board["region"], dtype=np.float64), np.asarray(board["solution"], dtype=np.float64))
        self.board = board
        self.n = n
        self.session = MeowdokuSession(n=n, puzzle=puzzle)
        self.title = f"Meowdoku {n}x{n} · {board.get('level', '')} · {board.get('type', '')}"
        self.palette_name = palette if palette in PALETTES else DEFAULT_PALETTE
        self.hints_used = 0
        self.hint_cells: list[int] = []
        self.hint_text = ""

    def cell(self, r: int, c: int, button: str) -> None:
        self.hint_cells, self.hint_text = [], ""
        if button in ("double", "cat"):
            self.session.place(r, c)
        elif button == "paint":
            if not self.session.x_mark.reshape(self.n, self.n)[r, c]:
                self.session.toggle_x(r, c)
        else:                                            # primary, secondary or mouse: toggle the mouse
            self.session.toggle_x(r, c)
        self.moves += 1

    def undo(self) -> None:
        self.hint_cells, self.hint_text = [], ""
        self.session.undo()

    def reset(self) -> None:
        self.hint_cells, self.hint_text = [], ""
        self.session.reset()
        self.moves = 0

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if not s.won:
            return None
        return {"score": 100 * int(s.hearts) + 10 * self.n,
                "detail": f"{self.n}x{self.n} solved with {int(s.hearts)} hearts in {self.moves} moves"}

    def set_palette(self, name: str) -> None:
        if name not in PALETTES:
            raise ValueError(f"unknown palette {name!r}; have {sorted(PALETTES)}")
        self.palette_name = name

    def hint(self) -> None:
        """Name the next deducible move without touching the dt system's
        state: only this game's highlight and counter change."""
        s = self.session
        if s.won or s.lost:
            return
        found = next_deduction(np.asarray(self.board["region"]).reshape(self.n, self.n),
                               np.asarray(s.cat).reshape(self.n, self.n), self.board["solution"], s.x_mark)
        self.hint_cells = found["cells"]
        self.hints_used += 1
        what = "cat here" if found["kind"] == "cat" else ("mouse here" if len(found["cells"]) == 1 else "mice here")
        self.hint_text = f"hint: {what} - {found['why']}"

    def colours(self) -> dict[str, list[float]]:
        hues = np.asarray(self.board["hues"], dtype=float)
        rank = np.argsort(np.argsort(hues, kind="stable"), kind="stable")
        scheme = PALETTES[self.palette_name]
        picks = [scheme[int(i * len(scheme) / self.n)] for i in range(self.n)]
        return {f"region{k}": list(picks[int(rank[k])]) for k in range(self.n)}

    def state(self) -> dict[str, Any]:
        n, s = self.n, self.session
        st = self.base_state(n, n)
        region = [int(k) for k in self.board["region"]]
        st["palette"] = self.colours()
        st["palette"].update({"cat": [0.10, 0.05, 0.03], "x": [0.97, 0.95, 0.93], "heart": [0.88, 0.10, 0.16]})
        st["palettes"], st["palette_name"] = list(PALETTES), self.palette_name
        st["cells"] = [f"region{k}" for k in region]
        cat = np.asarray(s.cat, dtype=int).reshape(-1)
        x_mark = np.asarray(s.x_mark, dtype=int).reshape(-1)
        st["marks"] = x_mark.tolist()
        hints = [0] * (n * n)
        for i in self.hint_cells:
            hints[i] = 1
        st["hints"] = hints
        st["hints_used"], st["hint"] = self.hints_used, self.hint_text
        st["pieces"] = [{"id": f"cat-{i}", "cell": int(i), "mesh": "cat", "color": "cat"} for i in np.flatnonzero(cat)]
        st["pieces"] += [{"id": f"x-{i}", "cell": int(i), "mesh": "x_mark", "color": "x"} for i in np.flatnonzero(x_mark)]
        st["pieces"] += [{"id": f"heart-{k}", "cell": f"hud:{k}", "mesh": "heart", "color": "heart"} for k in range(int(s.hearts))]
        st["hearts"] = int(s.hearts)
        st["won"], st["lost"] = bool(s.won), bool(s.lost)
        hint_note = f" · {self.hints_used} hint{'s' if self.hints_used != 1 else ''}" if self.hints_used else ""
        st["status"] = ("solved! all cats home" if s.won else "out of hearts" if s.lost
                        else f"{int(s.hearts)} ♥ · {self.moves} moves") + hint_note
        if self.hint_text and not (s.won or s.lost):
            st["status"] += f" · {self.hint_text}"
        return st


# ---------------------------------------------------------------------------
# Chess and checkers: select a piece, see its destinations, move
# ---------------------------------------------------------------------------

class BoardGame(Game):
    """8x8 two-tone board, pieces as {cell: (kind, side)}, select/move."""

    rows = cols = 8
    sides = ("white", "black")

    def __init__(self, board_id: str = ""):
        super().__init__(board_id)
        self.pieces: dict[int, tuple[str, str]] = {}
        self.ids: dict[int, str] = {}
        self.selected: int | None = None
        self.turn = self.sides[0]
        self.setup()
        self.history = []

    def setup(self) -> None: ...
    def legal_moves(self, cell: int) -> list[int]: ...
    def apply_move(self, src: int, dst: int) -> None: ...

    def idx(self, r: int, c: int) -> int:
        return r * self.cols + c

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.cols)

    def on_board(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cell(self, r: int, c: int, button: str) -> None:
        i = self.idx(r, c)
        if self.selected is not None and i in self.legal_moves(self.selected):
            self.history.append((dict(self.pieces), dict(self.ids), self.turn, getattr(self, "rights", None), getattr(self, "ep", None)))
            self.apply_move(self.selected, i)
            self.selected = None
            self.moves += 1
            self.turn = self.sides[1] if self.turn == self.sides[0] else self.sides[0]
            return
        if i in self.pieces and self.pieces[i][1] == self.turn:
            self.selected = None if self.selected == i else i
        else:
            self.selected = None

    def peek(self, src: int, dst: int) -> tuple[dict[int, tuple[str, str]], str]:
        """The position after ``src -> dst`` and the side then to move,
        without touching this game (a lookahead's successor)."""
        saved = (self.pieces, self.ids, self.turn, getattr(self, "rights", None), getattr(self, "ep", None))
        self.pieces, self.ids = dict(self.pieces), dict(self.ids)
        self.apply_move(src, dst)
        pieces = self.pieces
        turn = self.sides[1] if self.turn == self.sides[0] else self.sides[0]
        self.pieces, self.ids, self.turn = saved[:3]
        if saved[3] is not None:
            self.rights, self.ep = saved[3], saved[4]
        return pieces, turn

    def undo(self) -> None:
        if self.history:
            entry = self.history.pop()
            self.pieces, self.ids, self.turn = entry[:3]
            if len(entry) > 3 and entry[3] is not None:
                self.rights, self.ep = entry[3], entry[4]
            self.selected = None
            self.moves = max(0, self.moves - 1)

    def reset(self) -> None:
        self.pieces, self.ids, self.selected = {}, {}, None
        self.turn = self.sides[0]
        self.moves = 0
        self.history = []
        self.setup()

    def state(self) -> dict[str, Any]:
        st = self.base_state(self.rows, self.cols)
        st["palette"] = {"light": [0.86, 0.80, 0.68], "dark": [0.42, 0.30, 0.22],
                         "white": [0.92, 0.90, 0.84], "black": [0.12, 0.11, 0.12],
                         "red": [0.75, 0.12, 0.12], "gold": [0.95, 0.75, 0.2]}
        st["cells"] = [("light" if (r + c) % 2 == 0 else "dark") for r in range(self.rows) for c in range(self.cols)]
        hints = [0] * (self.rows * self.cols)
        if self.selected is not None:
            for d in self.legal_moves(self.selected):
                hints[d] = 1
        st["hints"] = hints
        st["selected"] = self.selected
        st["pieces"] = [{"id": self.ids[i], "cell": i, "mesh": kind, "color": side} for i, (kind, side) in sorted(self.pieces.items())]
        st["status"] = f"{self.turn} to move · {self.moves} moves"
        return st


class Chess(BoardGame):
    game = "chess"
    title = "Chess"
    BACK = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")

    def setup(self) -> None:
        # castling rights by the rook's home square, and the en passant
        # target (the square a double-stepping pawn passed over)
        self.rights = {0, 7, 56, 63}
        self.ep = None
        counter = 0
        for c, kind in enumerate(self.BACK):
            for r, side in ((0, "black"), (7, "white")):
                self.pieces[self.idx(r, c)] = (kind, side); self.ids[self.idx(r, c)] = f"{side}-{kind}-{counter}"; counter += 1
        for c in range(8):
            for r, side in ((1, "black"), (6, "white")):
                self.pieces[self.idx(r, c)] = ("pawn", side); self.ids[self.idx(r, c)] = f"{side}-pawn-{counter}"; counter += 1

    def pseudo_moves(self, cell: int) -> list[int]:
        """Piece movement and captures, pawn double step and promotion, with
        no regard for the king's safety; no castling or en passant."""
        if cell not in self.pieces:
            return []
        kind, side = self.pieces[cell]
        r, c = self.rc(cell)
        enemy = "black" if side == "white" else "white"
        out: list[int] = []

        def slide(directions):
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                while self.on_board(rr, cc):
                    j = self.idx(rr, cc)
                    if j in self.pieces:
                        if self.pieces[j][1] == enemy:
                            out.append(j)
                        break
                    out.append(j)
                    rr, cc = rr + dr, cc + dc

        def step(offsets):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if self.on_board(rr, cc):
                    j = self.idx(rr, cc)
                    if j not in self.pieces or self.pieces[j][1] == enemy:
                        out.append(j)

        if kind == "pawn":
            forward = -1 if side == "white" else 1
            start = 6 if side == "white" else 1
            rr = r + forward
            if self.on_board(rr, c) and self.idx(rr, c) not in self.pieces:
                out.append(self.idx(rr, c))
                if r == start and self.idx(rr + forward, c) not in self.pieces:
                    out.append(self.idx(rr + forward, c))
            for dc in (-1, 1):
                if self.on_board(rr, c + dc):
                    j = self.idx(rr, c + dc)
                    if (j in self.pieces and self.pieces[j][1] == enemy) or j == getattr(self, "ep", None):
                        out.append(j)
        elif kind == "knight":
            step(((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)))
        elif kind == "bishop":
            slide(((1, 1), (1, -1), (-1, 1), (-1, -1)))
        elif kind == "rook":
            slide(((1, 0), (-1, 0), (0, 1), (0, -1)))
        elif kind == "queen":
            slide(((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)))
        elif kind == "king":
            step(((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)))
            home = 60 if side == "white" else 4
            if cell == home and not getattr(self, "_probing", False) and not self.in_check(side):
                for rook, between, path in ((home + 3, (home + 1, home + 2), (home + 1, home + 2)),
                                            (home - 4, (home - 1, home - 2, home - 3), (home - 1, home - 2))):
                    if rook in getattr(self, "rights", set()) and self.pieces.get(rook) == ("rook", side) \
                            and all(b not in self.pieces for b in between) \
                            and not any(self.attacked(sq, enemy) for sq in path):
                        out.append(path[1])
        return out

    def apply_move(self, src: int, dst: int, promotion: str = "queen") -> None:
        kind, side = self.pieces.pop(src)
        piece_id = self.ids.pop(src)
        r0, c0 = self.rc(src); r, c = self.rc(dst)
        ep = getattr(self, "ep", None)
        if kind == "pawn" and dst == ep and c != c0 and dst not in self.pieces:   # en passant takes the passed pawn
            taken = self.idx(r0, c)
            self.pieces.pop(taken, None); self.ids.pop(taken, None)
        if kind == "king" and abs(c - c0) == 2:                                     # castling: the rook hops over
            rook_from, rook_to = (self.idx(r, 7), self.idx(r, 5)) if c > c0 else (self.idx(r, 0), self.idx(r, 3))
            if rook_from in self.pieces:
                self.pieces[rook_to] = self.pieces.pop(rook_from); self.ids[rook_to] = self.ids.pop(rook_from)
        if kind == "pawn" and r in (0, 7):
            kind = promotion
        self.pieces[dst] = (kind, side)
        self.ids[dst] = piece_id
        rights = set(getattr(self, "rights", set()))
        if kind == "king":
            rights -= {56, 63} if side == "white" else {0, 7}
        rights.discard(src); rights.discard(dst)
        self.rights = rights
        self.ep = self.idx((r0 + r) // 2, c) if kind == "pawn" and abs(r - r0) == 2 else None

    def attacked(self, cell: int, by: str) -> bool:
        """Whether ``by`` has a piece whose movement reaches ``cell``
        (castling is not an attack, so it is left out while probing)."""
        self._probing = True
        try:
            for src, (kind, side) in self.pieces.items():
                if side == by and cell in self.pseudo_moves(src):
                    return True
            return False
        finally:
            self._probing = False

    def king_of(self, side: str) -> int | None:
        for cell, (kind, s) in self.pieces.items():
            if kind == "king" and s == side:
                return cell
        return None

    def in_check(self, side: str) -> bool:
        king = self.king_of(side)
        enemy = "black" if side == "white" else "white"
        return king is not None and self.attacked(king, enemy)

    def legal_moves(self, cell: int) -> list[int]:
        """Pseudo-legal moves that do not leave the mover's own king
        attacked: you may not move into check, and in check you must get
        out of it."""
        if cell not in self.pieces:
            return []
        kind, side = self.pieces[cell]
        enemy = "black" if side == "white" else "white"
        out = []
        saved = (self.pieces, self.ids, getattr(self, "rights", set()), getattr(self, "ep", None))
        for dst in self.pseudo_moves(cell):
            if self.pieces.get(dst) == ("king", enemy):
                continue  # checkmate ends the game; a king is never captured
            self.pieces, self.ids = dict(saved[0]), dict(saved[1])
            self.apply_move(cell, dst)
            king = self.king_of(side)
            if king is not None and not self.attacked(king, enemy):
                out.append(dst)
            self.rights, self.ep = saved[2], saved[3]
        self.pieces, self.ids = saved[0], saved[1]
        return out

    def all_legal_moves(self) -> list[tuple[int, int]]:
        return [(src, dst) for src, (kind, side) in list(self.pieces.items()) if side == self.turn
                for dst in self.legal_moves(src)]

    def outcome(self) -> str | None:
        """``"white"`` / ``"black"`` for a checkmate's winner, ``"draw"`` for
        stalemate, else None (the game goes on)."""
        if self.all_legal_moves():
            return None
        if self.in_check(self.turn):
            return "black" if self.turn == "white" else "white"
        return "draw"


    def state(self) -> dict[str, Any]:
        st = super().state()
        end = self.outcome()
        if end == "draw":
            st["status"] = f"stalemate · {self.moves} moves"
        elif end:
            st["status"] = f"checkmate · {end} wins · {self.moves} moves"
            st["won"] = end == "white"; st["lost"] = end == "black"
        elif self.in_check(self.turn):
            st["status"] = f"{self.turn} to move · check · {self.moves} moves"
        return st


class Checkers(BoardGame):
    game = "checkers"
    title = "Checkers"
    sides = ("red", "black")

    def setup(self) -> None:
        counter = 0
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    if r < 3:
                        self.pieces[self.idx(r, c)] = ("disc", "black"); self.ids[self.idx(r, c)] = f"black-{counter}"; counter += 1
                    elif r > 4:
                        self.pieces[self.idx(r, c)] = ("disc", "red"); self.ids[self.idx(r, c)] = f"red-{counter}"; counter += 1

    def legal_moves(self, cell: int) -> list[int]:
        """Diagonal steps forward (kings both ways) and single jumps over an
        enemy piece onto an empty square."""
        if cell not in self.pieces:
            return []
        kind, side = self.pieces[cell]
        r, c = self.rc(cell)
        forward = -1 if side == "red" else 1
        directions = [(forward, -1), (forward, 1)] + ([(-forward, -1), (-forward, 1)] if kind == "king" else [])
        out = []
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if not self.on_board(rr, cc):
                continue
            j = self.idx(rr, cc)
            if j not in self.pieces:
                out.append(j)
            elif self.pieces[j][1] != side:
                r2, c2 = rr + dr, cc + dc
                if self.on_board(r2, c2) and self.idx(r2, c2) not in self.pieces:
                    out.append(self.idx(r2, c2))
        return out

    def apply_move(self, src: int, dst: int) -> None:
        kind, side = self.pieces.pop(src)
        piece_id = self.ids.pop(src)
        r0, c0 = self.rc(src); r1, c1 = self.rc(dst)
        if abs(r1 - r0) == 2:                                          # a jump: the piece between is taken
            mid = self.idx((r0 + r1) // 2, (c0 + c1) // 2)
            self.pieces.pop(mid, None); self.ids.pop(mid, None)
        if (side == "red" and r1 == 0) or (side == "black" and r1 == 7):
            kind = "king"
        self.pieces[dst] = (kind, side)
        self.ids[dst] = piece_id

    def state(self) -> dict[str, Any]:
        st = super().state()
        for piece in st["pieces"]:
            if piece["mesh"] == "king":
                piece["mesh"] = "disc"
                piece["color"] = "gold"
        st["status"] = f"{self.turn} to move · {self.moves} moves"
        return st


# ---------------------------------------------------------------------------
# Wordle: six guesses on the board, a rack of letters coloured by evidence
# ---------------------------------------------------------------------------

class Wordle(Game):
    game = "wordle"
    input = "text"
    title = "Wordle"
    RACK = ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")

    def __init__(self, seed: int = 0):
        super().__init__(f"wordle-{seed}")
        self.rng = random.Random(seed)
        self.answer = self.rng.choice(ANSWERS)
        self.guesses: list[str] = []
        self.entry = ""
        self.won = False

    def text(self, value: str) -> None:
        self.entry = "".join(ch for ch in value.lower() if ch.isalpha())[:5]

    def backspace(self) -> None:
        self.entry = self.entry[:-1]

    def submit(self) -> None:
        if self.won or len(self.guesses) >= 6:
            return
        if len(self.entry) != 5:
            self.status = "five letters"
            return
        if self.entry not in ALL:
            self.status = f"{self.entry.upper()} is not in the list"
            return
        self.guesses.append(self.entry)
        self.moves += 1
        self.won = self.entry == self.answer
        self.entry = ""
        self.status = "" if not self.won else "got it!"

    def result(self) -> dict[str, Any] | None:
        if not self.won:
            return None
        return {"score": 7 - len(self.guesses), "detail": f"{self.answer.upper()} in {len(self.guesses)}"}

    def undo(self) -> None:
        if self.guesses and not self.won:
            self.guesses.pop(); self.moves -= 1

    def reset(self) -> None:
        self.guesses, self.entry, self.won, self.moves, self.status = [], "", False, 0, ""

    @staticmethod
    def score(guess: str, answer: str) -> list[str]:
        """green / yellow / absent per letter, with multiplicity."""
        result = ["absent"] * 5
        pool = list(answer)
        for i, ch in enumerate(guess):
            if answer[i] == ch:
                result[i] = "green"; pool[i] = None
        for i, ch in enumerate(guess):
            if result[i] == "green":
                continue
            if ch in pool:
                result[i] = "yellow"; pool[pool.index(ch)] = None
        return result

    def state(self) -> dict[str, Any]:
        st = self.base_state(6, 5)
        st["palette"] = {"cell": [0.22, 0.22, 0.25], "entry": [0.32, 0.32, 0.36], "green": [0.33, 0.65, 0.30],
                         "yellow": [0.78, 0.66, 0.2], "absent": [0.30, 0.30, 0.33], "unknown": [0.62, 0.62, 0.66],
                         "letter": [0.93, 0.93, 0.95], "ink": [0.08, 0.08, 0.1]}
        cells, pieces = ["cell"] * 30, []
        best: dict[str, str] = {}
        rank = {"absent": 0, "yellow": 1, "green": 2}
        for g, guess in enumerate(self.guesses):
            colours = self.score(guess, self.answer)
            for i, ch in enumerate(guess):
                cells[g * 5 + i] = colours[i]
                pieces.append({"id": f"g{g}-{i}", "cell": g * 5 + i, "mesh": ch.upper(), "color": "letter"})
                if rank[colours[i]] >= rank.get(best.get(ch, "absent"), -1):
                    best[ch] = colours[i]
        row = len(self.guesses)
        if row < 6:
            for i, ch in enumerate(self.entry):
                cells[row * 5 + i] = "entry"
                pieces.append({"id": f"entry-{i}-{ch}", "cell": row * 5 + i, "mesh": ch.upper(), "color": "ink"})
        st["cells"], st["pieces"] = cells, pieces
        rack_cells, rack_pieces = [], []
        for r, line in enumerate(self.RACK):
            for c, ch in enumerate(line):
                rack_cells.append(best.get(ch.lower(), "unknown"))
                rack_pieces.append({"id": f"rack-{ch}", "cell": r * 10 + c, "mesh": ch, "color": "ink", "letter": ch})
            rack_cells += [""] * (10 - len(line))
        st["rack"] = {"rows": 3, "cols": 10, "cells": rack_cells, "pieces": rack_pieces,
                      "letters": ["".join(line) for line in self.RACK]}
        st["text"], st["text_max"] = self.entry.upper(), 5
        st["won"] = self.won
        st["lost"] = (not self.won) and len(self.guesses) >= 6
        st["status"] = self.status or ("got it!" if self.won else f"the word was {self.answer.upper()}" if st["lost"]
                                       else f"guess {len(self.guesses) + 1} of 6")
        return st


# ---------------------------------------------------------------------------
# The chess network: an arena that plays itself non-stop, and chess against it
# ---------------------------------------------------------------------------

class Engine:
    """The one engine thread.  Every piece of chess-network work (the
    learner's steps, a user's tree search, Lichess move choice and
    learning) runs on this single worker, in submission order.

    ``call(fn, ...)`` blocks the caller until the worker has run ``fn`` and
    returns its result or re-raises its exception.  Called from the worker
    itself it runs inline, so nested calls cannot deadlock.  Callers must
    not hold a lock the engine work needs while they wait on ``call``."""

    def __init__(self):
        import queue, threading
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, args, kwargs, box, done = self._queue.get()
            try:
                box.append((True, fn(*args, **kwargs)))
            except BaseException as error:  # delivered to the caller
                box.append((False, error))
            done.set()

    def call(self, fn, *args, **kwargs):
        import threading
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        box, done = [], threading.Event()
        self._queue.put((fn, args, kwargs, box, done))
        done.wait()
        ok, value = box[0]
        if not ok:
            raise value
        return value


_ENGINE = None
_ENGINE_GUARD = __import__("threading").Lock()


def engine() -> Engine:
    global _ENGINE
    with _ENGINE_GUARD:
        if _ENGINE is None:
            _ENGINE = Engine()
    return _ENGINE


_SELF_PLAY = None


def self_play():
    """One shared learner per server process (its weights persist on disk)."""
    global _SELF_PLAY
    if _SELF_PLAY is None:
        import os
        from meowdoku.chess_net import SelfPlay
        _SELF_PLAY = SelfPlay()
        # The search evaluates through the forward compiled by the repository
        # compiler under the eigen work contract; MEOWDOKU_NATIVE=0 keeps numpy.
        if os.environ.get("MEOWDOKU_NATIVE", "1") != "0":
            _SELF_PLAY.net.use_native("eigen")
    return _SELF_PLAY


class Learner:
    """The background learning of this process: self-play in batches and
    the master-game corpus, interleaved in one daemon thread that no game
    switch stops.  The arena shows its game 0; every other mode can read
    its numbers.  One per process (``learner()``)."""

    def __init__(self, parallel: int = 4):
        import threading
        self.player = self_play()
        self.lock = threading.RLock()
        self.parallel = max(1, int(parallel))
        self.display = Chess("arena-0")
        self.engine = self.player.batch_engine(self.display, self.parallel)
        self.corpus = None
        self.note = ""
        self.error = ""
        self.last_result = ""
        self.steps_per_second = 0.0
        self.started_at = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="learner", daemon=True)
        self._thread.start()

    def _step(self) -> None:
        with self.lock:
            finished = self.engine.step()
            if finished:
                self.last_result = finished[-1]
            if self.corpus is not None:
                self.corpus.step(self.player)

    def _run(self) -> None:
        import os, time
        self.started_at = time.time()
        if not os.environ.get("MEOWDOKU_NO_PRETRAIN"):
            try:
                from meowdoku.pretrain import Corpus
                self.corpus = Corpus.load_or_build(progress=lambda message: setattr(self, "note", message))
                self.note = ""
            except Exception as error:
                self.note = f"pre-training unavailable: {type(error).__name__}: {str(error)[:60]}"
        tick, count = time.perf_counter(), 0
        while not self._stop.is_set():
            try:
                engine().call(self._step)           # network work runs on the engine thread
            except Exception as error:
                import traceback
                self.error = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                return
            count += 1
            now = time.perf_counter()
            if now - tick >= 1.0:
                self.steps_per_second = count / (now - tick)
                tick, count = now, 0
            time.sleep(0)

    def summary(self) -> str:
        import time
        p = self.player
        rec = p.record
        if self.error:
            return f"training stopped: {self.error}"
        lead = (self.corpus.summary() if self.corpus is not None else self.note) or ""
        hours = (time.time() - self.started_at) / 3600 if self.started_at else 0.0
        return ((lead + " · ") if lead else "") + (
            f"self-play {self.parallel} games in flight · {self.engine.playouts} playouts a move · "
            f"{self.steps_per_second * self.parallel:.1f} plies/s · "
            f"{p.games} games · W{rec['white']} B{rec['black']} D{rec['draw']}"
            + (f" · loss {p.last_loss:.3f}" if p.games else "")
            + (f" · last: {self.last_result}" if self.last_result else "")
            + f" · learning {hours * 60:.0f} min"
            + f" · eval {p.net.native_note}")


_LEARNER = None


def learner(parallel: int = 4) -> "Learner":
    global _LEARNER
    if _LEARNER is None:
        _LEARNER = Learner(parallel)
    return _LEARNER


class ChessArena(Chess):
    """Network against network: the learner's game 0, shown live.  The
    learner itself runs in the background whichever mode is on screen."""

    game = "chess_arena"
    title = "Chess arena: the network against itself"
    input = "watch"

    def __init__(self, board_id: str = "arena", parallel: int = 4, lock=None):
        super().__init__(board_id)
        self.learner = learner(parallel)
        self.player = self.learner.player

    def tick(self) -> None:
        return None

    def cell(self, r: int, c: int, button: str) -> None:
        return None

    def reset(self) -> None:
        return None

    def undo(self) -> None:
        return None

    def state(self) -> dict:
        with self.learner.lock:
            shown = self.learner.display
            self.pieces, self.ids, self.turn = dict(shown.pieces), dict(shown.ids), shown.turn
            self.rights, self.ep = getattr(shown, "rights", set()), getattr(shown, "ep", None)
            plies = self.learner.engine.plies(0)
        st = Chess.state(self)
        st["selected"] = None; st["hints"] = [0] * 64
        st["status"] = self.learner.summary() + (f" · search: {self.learner.engine.last_search}" if self.learner.engine.last_search else "")
        st["moves"] = plies
        return st


class ChessVsNet(Chess):
    """You play White; the network answers for Black after each move."""

    def result(self) -> dict[str, Any] | None:
        end = self.outcome()
        if end is None:
            return None
        score = {"white": 1.0, "draw": 0.5}.get(end, 0.0)
        said = {"white": "beat the network", "draw": "drew the network"}.get(end, "lost to the network")
        return {"score": score, "detail": f"{said} in {self.moves} moves"}

    game = "chess_vs_net"
    title = "Chess against the network"

    def __init__(self, board_id: str = "vs_net"):
        super().__init__(board_id)
        self.player = self_play()

    def cell(self, r: int, c: int, button: str) -> None:
        before = self.moves
        super().cell(r, c, button)
        if self.moves > before and self.turn == "black" and self.outcome() is None:
            from meowdoku import mcts
            move = engine().call(mcts.best_move, self, self.player.net, simulations=3000)   # tree search on the engine thread
            if move is not None:
                super().cell(*self.rc(move[0]), "primary"); super().cell(*self.rc(move[1]), "primary")


class LichessGame(Chess):
    """The network's live game on Lichess, as the page sees it: the board
    replayed from the game stream's moves, the opponent, clocks and
    result in the status.  Starting this game starts the Lichess client
    (once per process) when a token is configured."""

    game = "lichess"
    title = "Lichess: the network against the world"
    input = "watch"

    def __init__(self, board_id: str = "lichess"):
        super().__init__(board_id)
        from meowdoku import lichess
        self.client = lichess.bot(player=self_play())
        self.learner = learner()                                  # background learning runs while we watch
        self.note = "" if self.client else "no Lichess token: set MEOWDOKU_LICHESS_TOKEN or build/meowdoku/lichess_token.txt"

    def tick(self) -> None:
        return None

    def cell(self, r: int, c: int, button: str) -> None:
        return None

    def reset(self) -> None:
        return None

    def undo(self) -> None:
        return None

    def state(self) -> dict:
        from meowdoku import lichess
        snap = self.client.snapshot() if self.client else None
        if snap:
            board = lichess.board_from_moves(snap.get("moves", ""))
            self.pieces, self.ids, self.turn, self.moves = board.pieces, board.ids, board.turn, board.moves
            self.rights, self.ep = board.rights, board.ep
        st = Chess.state(self)
        st["selected"] = None; st["hints"] = [0] * 64
        control = self.client.control_state() if self.client else None
        st["lichess_control"] = control
        if not self.client:
            st["status"] = self.note
        elif not snap:
            me = self.client.me.get("username", "…")
            pending = self.client.pending
            st["status"] = f"{me} online · " + (f"challenged {pending['to']}, waiting for an answer" if pending else "seeking a game among the online bots")
        else:
            clocks = ""
            if snap.get("wtime") is not None:
                clocks = f" · ⏱ {int(snap['wtime'] // 1000)}s / {int(snap['btime'] // 1000)}s"
            record = f" · {len(self.client.results)} games: " + ", ".join(r.split(' vs ')[0] for r in self.client.results[-6:]) if self.client.results else ""
            st["status"] = (f"{snap['white']} vs {snap['black']} · we are {snap['color']} · {self.turn} to move{clocks}"
                            + (f" · {snap.get('summary', snap['status'])}" if snap.get("result") else "") + record)
            st["won"] = bool(snap.get("result")) and snap.get("result") == snap.get("color")
            st["lost"] = bool(snap.get("result")) and snap.get("result") not in (None, "draw", snap.get("color"))
        if self.client and getattr(self.client, "last_search", ""):
            st["status"] += f" · search: {self.client.last_search}"
        if control and control["pause_requested"]:
            mode = ("finishing current game" if control["live"] else
                    "canceling pending challenge" if control["pending"] else "paused")
            st["status"] += f" · Lichess {mode}"
        st["status"] += "  ‖  " + self.learner.summary()
        return st


GAMES = {"meowdoku": MeowdokuGame, "chess": Chess, "checkers": Checkers, "wordle": Wordle,
         "chess_arena": ChessArena, "chess_vs_net": ChessVsNet, "lichess": LichessGame}
