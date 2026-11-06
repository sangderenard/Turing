"""The bullet-hell token pilot, inside the law.

``autopilot.py`` flies the ship from Python one decision at a time: build the
token observation in NumPy (``scroller_tokens``), run the network, map its
outputs to controls (``controls_from``).  This module is the same decision as
AbstractTensor text, so ``scroller_step`` traces it with the rest of the law
and every piloted lane decides every physics step, natively, with no Python in
the loop -- and the page's whole-world law wasm contains the pilot, so the
client predicts an autopilot craft exactly.

``token_policy_forward`` and the observation's layout (TOKENS, TOKEN_F, SHIP_G
...) are defined HERE and imported by ``autopilot.py``: training, the NumPy
and native single-decision products, and the law all run this one text.

``decide`` builds the observation for a contiguous block of lanes (``rows``)
from the law's own columns and runs the network on it, batched over the
block.  The network's weights are NOT constants: they are the law's declared
PARAMETER inputs (``LawPiece.parameter_shapes`` / ``set_parameters``), so
training changes them without a recompile.

Rows share the board's patch frame (the columns ``x, y`` are relative to the
player), which is the frame the player's own observation is read in.  A
piloted craft elsewhere on the board sees through its own patch (the
mirrored view of ``autopilot._MirroredView``): that projection is the next
step for friendly lanes, not done here.
"""
from __future__ import annotations

import numpy as np

from src.common.tensors import AbstractTensor

#: THE TOKEN NETWORK (see autopilot.py for the story): the observation is the
#: ship's own vector (SHIP_G values), then every body in reach as a token
#: (TOKENS of them, each TOKEN_F values), then the TOKENS-long mask.
TOKENS, TOKEN_F, SHIP_G = 64, 16, 12
TOKEN_H, SHIP_H, HEAD_H = 48, 32, 64
TOKEN_FEATURES = SHIP_G + TOKENS * TOKEN_F + TOKENS
TOKEN_PARAMS = ("We1", "be1", "We2", "be2", "Q1", "Q2", "Wg", "bg", "Wh1", "bh1", "Wh2", "bh2")
#: roll, throttle, fire, pitch, rudder
OUTPUTS = 5
#: Scales of the observation (board units to network units).
DX, DY, DV = 8.0, 10.0, 10.0


def token_param_shapes(outputs: int = OUTPUTS) -> dict[str, tuple[int, ...]]:
    """The network's parameter arrays by name: the law's declared parameter inputs."""
    return {"We1": (TOKEN_F, TOKEN_H), "be1": (1, TOKEN_H), "We2": (TOKEN_H, TOKEN_H), "be2": (1, TOKEN_H),
            "Q1": (TOKEN_H, 1), "Q2": (TOKEN_H, 1), "Wg": (SHIP_G, SHIP_H), "bg": (1, SHIP_H),
            "Wh1": (2 * TOKEN_H + SHIP_H, HEAD_H), "bh1": (1, HEAD_H), "Wh2": (HEAD_H, outputs), "bh2": (1, outputs)}


def token_policy_forward(x, We1, be1, We2, be2, Q1, Q2, Wg, bg, Wh1, bh1, Wh2, bh2):
    """x (N, TOKEN_FEATURES) -> (N, outputs) in [-1, 1].  AbstractTensor
    arithmetic only (traced to the NumPy product like ``policy_forward``):
    unpack the ship, the tokens and the mask; embed every token (shared
    weights); pool by two attention queries over the live tokens; join the
    ship's embedding; a two-layer head."""
    n = int(x.shape[0])
    ship = x[:, 0:SHIP_G]
    tokens = x[:, SHIP_G:SHIP_G + TOKENS * TOKEN_F].reshape((n * TOKENS, TOKEN_F))
    mask = x[:, SHIP_G + TOKENS * TOKEN_F:SHIP_G + TOKENS * TOKEN_F + TOKENS].reshape((n, TOKENS, 1))
    e = (tokens @ We1 + be1).tanh()
    e = (e @ We2 + be2).tanh()
    emb = e.reshape((n, TOKENS, TOKEN_H))
    pooled = []
    for q in (Q1, Q2):
        weight = ((e @ q).tanh() * 4.0).exp().reshape((n, TOKENS, 1)) * mask
        total = weight.sum(dim=1) + 1e-6
        pooled.append((weight * emb).sum(dim=1) / total)
    g = (ship @ Wg + bg).tanh()
    from src.common.tensors import AbstractTensor as _AT
    joined = _AT.cat([pooled[0], pooled[1], g], dim=1)
    h = (joined @ Wh1 + bh1).tanh()
    return (h @ Wh2 + bh2).tanh()


def _f(mask):
    return mask.to_dtype("float64")


def spread(values, rows: tuple[int, int], full):
    """``values`` (the rows' span) as a span the length of ``full``, zero outside ``rows``."""
    r0, r1 = rows
    n = int(full.shape[0])
    parts = [full[:r0] * 0.0] if r0 > 0 else []
    parts.append(values)
    if r1 < n:
        parts.append(full[r1:] * 0.0)
    return AbstractTensor.cat(parts, dim=0) if len(parts) > 1 else values


def observation(rows: tuple[int, int], c: dict):
    """The token observation (N, TOKEN_FEATURES) of the lanes in ``rows`` from
    the law's columns ``c`` (name -> span): the ship vector, the TOKENS nearest
    bodies in reach (nearest first, lane order among equals) and the mask --
    ``autopilot.scroller_tokens`` as tensor text.

    The k-th nearest is found by repeated extraction (see below), so the
    rows are exactly the first TOKENS of the Python path's stable sort."""
    from meowdoku import scroller as sc
    r0, r1 = rows
    R = r1 - r0
    blocks = ((sc.SHOTS[0], sc.PICKUPS[1]), tuple(sc.BASES))
    C = sum(b - a for a, b in blocks)

    def cand(name):
        return AbstractTensor.cat([c[name][a:b] for a, b in blocks], dim=0).reshape((1, C))

    def row(name):
        return c[name][r0:r1].reshape((R, 1))

    # relative to each row
    dx, dy, dz = cand("x") - row("x"), cand("y") - row("y"), cand("z") - row("z")
    dist = (dx * dx + dy * dy + 0.25 * dz * dz).sqrt()
    s, kd, owner = cand("slot"), cand("kind"), cand("owner")
    body = ((s == sc.ENEMY) | (s == sc.BULLET) | (s == sc.BASE_SLOT) | (s == sc.PICKUP) | (s == sc.SHOT)
            | (s == sc.MISSILE_SLOT) | ((s == sc.STAR) & (kd > 0.0)))
    elig = (cand("alive") > 0.0) & body & (cand("lane") != row("lane")) & (dist < sc.PATCH_R)

    # The TOKENS nearest bodies, nearest first, lane order among equals, by
    # repeated extraction: each round takes the nearest body left (the lowest
    # lane among equals) and removes it.  That is a stable sort's first
    # TOKENS rows at TOKENS rounds of C work each, not the C x C comparisons
    # of a rank.  ``pick`` (R, TOKENS, C) is the one-hot of token k's body.
    key = AbstractTensor.where(elig, dist, 1.0e9)
    rank = key * 0.0                      # the round that took each body (its token number)
    chosen = key * 0.0
    for k in range(TOKENS):
        nearest = key.min(dim=1).reshape((R, 1))
        tie = key == nearest
        first = tie & (_f(tie).cumsum(dim=1) == 1.0)
        taken = _f(first)
        rank = rank + taken * float(k)
        chosen = chosen + taken
        key = AbstractTensor.where(first, 2.0e9, key)       # beyond the out-of-reach lanes' 1e9
    # (once the bodies run out a round takes an out-of-reach lane over and over;
    # ``elig`` removes those below)
    # (a concatenation of the rounds' one-hots would be the same tensor; the
    # tensor backend's cat is an element loop, a broadcast compare is not)
    token_at = c["lane"][0:TOKENS].reshape((1, TOKENS, 1))                 # 0 .. TOKENS-1
    pick = _f(rank.reshape((R, 1, C)) == token_at) * (chosen * _f(elig)).reshape((R, 1, C))

    flak = (s == sc.BULLET) & (owner >= sc.STATIONS[0]) & (owner < sc.STATIONS[1])
    features = [
        dx / DX, -dy / DY, dz / sc.DEPTH,
        (cand("vx") - row("vx")) / DV, -(cand("vy") - row("vy")) / DV, (cand("vz") - row("vz")) / DV,
        _f((s == sc.ENEMY) & (kd != sc.ACE) & (kd != sc.BOSS_KIND)) + dist * 0.0,
        _f((s == sc.ENEMY) & ((kd == sc.ACE) | (kd == sc.BOSS_KIND))) + dist * 0.0,
        _f((s == sc.BULLET) & ~flak) + dist * 0.0,
        _f(flak) + dist * 0.0,
        _f(s == sc.BASE_SLOT) + dist * 0.0,
        _f(s == sc.PICKUP) + dist * 0.0,
        _f((s == sc.SHOT) | (s == sc.MISSILE_SLOT)) + dist * 0.0,
        _f(s == sc.STAR) + dist * 0.0,
        (cand("hp") / 10.0).clip(0.0, 1.0) + dist * 0.0,
        dist / sc.PATCH_R,
    ]
    table = AbstractTensor.cat([item.reshape((R, C, 1)) for item in features], dim=2)     # (R, C, TOKEN_F)
    # one rank-two product per row: matmul_double is the GEMM under the eigen
    # contract; the batched product is a scalar loop
    tokens = AbstractTensor.cat(
        [(pick[r:r + 1].reshape((TOKENS, C)) @ table[r:r + 1].reshape((C, TOKEN_F))).reshape((1, TOKENS * TOKEN_F))
         for r in range(R)], dim=0) if R > 1 else (pick.reshape((TOKENS, C)) @ table.reshape((C, TOKEN_F))).reshape((1, TOKENS * TOKEN_F))
    mask = pick.sum(dim=2)                                                 # (R, TOKENS)

    # the ship's own vector (game-wide lives read from lane 0)
    span = sc.SPD_MAX - sc.SPD_MIN
    weapon = ((row("weapon") + 0.5).floor()).clip(0.0, 2.0)
    zero = row("z") * 0.0
    ship = AbstractTensor.cat([
        2.0 * row("z") / sc.DEPTH - 1.0, row("vz") / DV, (row("spd") - sc.SPD_MIN) / span, row("bank") - sc.BANK_MAX,
        row("fuel"), row("energy"), c["lives"][0:1].reshape((1, 1)) / sc.HEALTH_MAX + zero,
        _f(row("timer") <= 0.0) + zero, row("beam") / sc.BEAM_RANGE,
        _f(weapon == 0.0), _f(weapon == 1.0), _f(weapon == 2.0)], dim=1)   # (R, SHIP_G)
    return AbstractTensor.cat([ship, tokens, mask], dim=1)


def decide(rows: tuple[int, int], c: dict, params: tuple):
    """The network's outputs (R, OUTPUTS) for the lanes in ``rows``: the
    observation of ``observation`` through ``token_policy_forward`` with the
    law's parameter inputs ``params`` (in ``TOKEN_PARAMS`` order)."""
    return token_policy_forward(observation(rows, c), *params)
