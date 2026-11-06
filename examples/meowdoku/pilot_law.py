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

import os

import numpy as np

from src.common.tensors import AbstractTensor

#: THE TOKEN NETWORK (see autopilot.py for the story): the observation is the
#: ship's own vector (SHIP_G values), then every body in reach as a token
#: (TOKENS of them, each TOKEN_F values), then the TOKENS-long mask.
#:
#: VARIANTS.  The default network (variant id ``token``) is 64 tokens, a
#: 48-wide token embedding, two pooling queries, a 64-wide head, tanh gates.
#: A trimmed one is named by an id of dotted fields -- ``t32.h32.q1.g32.ahard``
#: = 32 nearest tokens, embedding width 32, one pooling query, head width 32,
#: hard (clipped) gates.  The widths, the query count and the head are read
#: back from the PARAMETER SHAPES (``spec_of_shapes``), so a weight file's
#: arrays alone select them; the two things shapes cannot say (the token
#: count: the weights are shared across tokens; the gate family) ride in the
#: params dict as the ``variant`` entry, absent = ``token``.  The law is
#: compiled for ONE variant per process: MEOWDOKU_PILOT_ARCH (default ``token``).
SHIP_G, TOKEN_F = 12, 16
SHIP_H = 32
TOKEN_PARAMS = ("We1", "be1", "We2", "be2", "Q1", "Q2", "Wg", "bg", "Wh1", "bh1", "Wh2", "bh2")
#: roll, throttle, fire, pitch, rudder
OUTPUTS = 5
#: Scales of the observation (board units to network units).
DX, DY, DV = 8.0, 10.0, 10.0
DEFAULT_VARIANT = "token"
ACTS = ("tanh", "hard", "relu")
_DEFAULT = {"tokens": 64, "h": 48, "q": 2, "g": 64, "act": "tanh"}


def spec_of(variant: str = DEFAULT_VARIANT) -> dict:
    """The fields of a variant id (see VARIANTS above)."""
    spec = dict(_DEFAULT)
    if variant and variant != DEFAULT_VARIANT:
        for field in str(variant).split("."):
            if field.startswith("a"):
                spec["act"] = field[1:]
            elif field[0] in "thqg":
                spec["tokens" if field[0] == "t" else field[0]] = int(field[1:])
            else:
                raise ValueError(f"unknown pilot variant field {field!r} in {variant!r}")
    if spec["act"] not in ACTS or spec["q"] not in (1, 2):
        raise ValueError(f"bad pilot variant {variant!r}: {spec}")
    return spec


def spec_of_shapes(shapes: dict, variant: str = DEFAULT_VARIANT) -> dict:
    """The variant's fields with widths, query count and head read from the
    parameter shapes (``shapes``: name -> shape); tokens and gates from ``variant``."""
    spec = spec_of(variant)
    h = int(shapes["We1"][1])
    spec.update(h=h, g=int(shapes["Wh1"][1]), q=(int(shapes["Wh1"][0]) - int(shapes["Wg"][1])) // h)
    return spec


ACTIVE_VARIANT = os.environ.get("MEOWDOKU_PILOT_ARCH", DEFAULT_VARIANT) or DEFAULT_VARIANT
_ACTIVE = spec_of(ACTIVE_VARIANT)
TOKEN_COUNT = _ACTIVE["tokens"]
TOKENS, TOKEN_H, HEAD_H, QUERIES, ACT = _ACTIVE["tokens"], _ACTIVE["h"], _ACTIVE["g"], _ACTIVE["q"], _ACTIVE["act"]
TOKEN_FEATURES = SHIP_G + TOKENS * (TOKEN_F + 1)


def features_of(tokens: int) -> int:
    return SHIP_G + tokens * (TOKEN_F + 1)


def slice_tokens(x, tokens_from: int, tokens_to: int):
    """An observation at ``tokens_from`` tokens as the one at ``tokens_to`` <= it:
    the tokens are nearest first, so the nearer ones are a prefix."""
    x = np.asarray(x)
    if tokens_to == tokens_from:
        return x
    n = tokens_from * TOKEN_F
    return np.concatenate([x[..., :SHIP_G], x[..., SHIP_G:SHIP_G + tokens_to * TOKEN_F],
                           x[..., SHIP_G + n:SHIP_G + n + tokens_to]], axis=-1)


def token_param_shapes(outputs: int = OUTPUTS, variant: str | None = None) -> dict[str, tuple[int, ...]]:
    """The network's parameter arrays by name: the law's declared parameter
    inputs (of ``variant``, default the process's active one)."""
    spec = _ACTIVE if variant is None else spec_of(variant)
    H, G, q = spec["h"], spec["g"], spec["q"]
    return {"We1": (TOKEN_F, H), "be1": (1, H), "We2": (H, H), "be2": (1, H),
            "Q1": (H, 1), "Q2": (H, 1) if q == 2 else (1, 1), "Wg": (SHIP_G, SHIP_H), "bg": (1, SHIP_H),
            "Wh1": (q * H + SHIP_H, G), "bh1": (1, G), "Wh2": (G, outputs), "bh2": (1, outputs)}


def gate(t, act: str):
    """The hidden gate of family ``act``: tanh; ``hard`` = clip to [-1, 1];
    ``relu`` = clip to [0, 6] (relu6)."""
    if act == "tanh":
        return t.tanh()
    return t.clip(0.0, 6.0) if act == "relu" else t.clip(-1.0, 1.0)


def embed(tokens, We1, be1, We2, be2, act: str):
    """(n * T, TOKEN_F) -> (n * T, H): the shared two-layer token embedding."""
    e = gate(tokens @ We1 + be1, act)
    return gate(e @ We2 + be2, act)


def pool(e, mask, n: int, T: int, queries, act: str):
    """Attention pooling of the embedded tokens (n * T, H) by each query
    (H, 1), softmax over the live tokens only: a list of (n, H)."""
    H = int(e.shape[1])
    emb = e.reshape((n, T, H))
    out = []
    for q in queries:
        score = e @ q
        score = score.tanh() if act == "tanh" else score.clip(-1.0, 1.0)
        weight = (score * 4.0).exp().reshape((n, T, 1)) * mask
        total = weight.sum(dim=1) + 1e-6
        out.append((weight * emb).sum(dim=1) / total)
    return out


def head(pooled, ship, Wg, bg, Wh1, bh1, Wh2, bh2, act: str):
    g = gate(ship @ Wg + bg, act)
    joined = AbstractTensor.cat([*pooled, g], dim=1)
    h = gate(joined @ Wh1 + bh1, act)
    return h @ Wh2 + bh2


def token_policy_forward(x, We1, be1, We2, be2, Q1, Q2, Wg, bg, Wh1, bh1, Wh2, bh2, act: str | None = None):
    """x (N, features) -> (N, outputs) in [-1, 1].  AbstractTensor
    arithmetic only (traced to the NumPy product like ``policy_forward``):
    unpack the ship, the tokens and the mask; embed every token (shared
    weights); pool by the attention queries over the live tokens; join the
    ship's embedding; a two-layer head.  Widths, token count and query count
    are read from the shapes (see VARIANTS); ``act`` is the gate family
    (default the active variant's)."""
    act = ACT if act is None else act
    n = int(x.shape[0])
    G = int(Wg.shape[0])
    T = (int(x.shape[1]) - G) // (TOKEN_F + 1)
    ship = x[:, 0:G]
    tokens = x[:, G:G + T * TOKEN_F].reshape((n * T, TOKEN_F))
    mask = x[:, G + T * TOKEN_F:G + T * (TOKEN_F + 1)].reshape((n, T, 1))
    e = embed(tokens, We1, be1, We2, be2, act)
    queries = (Q1, Q2) if int(Q2.shape[0]) == int(We1.shape[1]) else (Q1,)
    out = head(pool(e, mask, n, T, queries, act), ship, Wg, bg, Wh1, bh1, Wh2, bh2, act)
    return out.tanh() if act == "tanh" else out.clip(-1.0, 1.0)


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


def observation(rows: tuple[int, int], c: dict, tokens: int | None = None):
    """The token observation (N, TOKEN_FEATURES) of the lanes in ``rows`` from
    the law's columns ``c`` (name -> span): the ship vector, the TOKENS nearest
    bodies in reach (nearest first, lane order among equals) and the mask --
    ``autopilot.scroller_tokens`` as tensor text.

    The k-th nearest is found by repeated extraction (see below), so the
    rows are exactly the first TOKENS of the Python path's stable sort."""
    from meowdoku import scroller as sc
    TOKENS = TOKEN_COUNT if tokens is None else int(tokens)
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
