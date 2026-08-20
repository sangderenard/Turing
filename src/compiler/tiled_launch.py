"""Cover a custom-size problem with prebaked peak-efficiency kernel tiles.

The aim (owner's, 2026-08-20): the compiler takes a CUSTOM size and exploits
a tiling algorithm that uses only the PREBAKED operators at peak
efficiency. The kernel bank holds size-specialized variants verified at
admission (``docs/KERNEL_BANK_DESIGN.md``); this module is the composition
layer that decomposes an arbitrary problem into calls on those fixed-size
cores, plus parametric-kernel edge tiles where the size does not divide.

This is deliberately HOST-side orchestration for now: the tile loop, the
packing, and the accumulation policy live here as the readable reference
for what the deployment layer will eventually schedule (across threads,
with its own packing) natively. Keeping it a plain object with an explicit
plan makes it the spec the deployment transform must match, the same way a
kernel's Python reference is the spec its compiled artifact must match.

Correctness of the decomposition (GEMM):

    C = alpha * A @ B + beta * C

tiled over (i, j, p) blocks becomes, per C-block (i, j):

    p == 0:  C_ij = alpha * A_i0 @ B_0j + beta * C_ij
    p  > 0:  C_ij = alpha * A_ip @ B_pj + 1.0  * C_ij

i.e. the authored kernel's own alpha/beta form is reused unchanged; only
beta varies by p-position. Every tile call is one of exactly two prebaked
shapes: the specialized (T, T, T) core, or the parametric kernel for edge
remainders. Packing copies each tile into a contiguous row-major buffer --
which is also the cache-locality transform real BLAS performs, so the copy
is not pure overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TilePlan:
    """The decomposition of one gemm call, stated before running it."""

    m: int
    n: int
    k: int
    tile: int
    full_tiles: tuple[int, int, int]      # counts of full tiles per axis
    remainders: tuple[int, int, int]      # trailing partial extents per axis
    specialized_calls: int
    parametric_calls: int


def plan_gemm_tiling(m: int, n: int, k: int, tile: int) -> TilePlan:
    fm, rm = divmod(int(m), int(tile))
    fn, rn = divmod(int(n), int(tile))
    fk, rk = divmod(int(k), int(tile))
    blocks_m = fm + (1 if rm else 0)
    blocks_n = fn + (1 if rn else 0)
    blocks_k = fk + (1 if rk else 0)
    specialized = fm * fn * fk
    total = blocks_m * blocks_n * blocks_k
    return TilePlan(
        m=int(m), n=int(n), k=int(k), tile=int(tile),
        full_tiles=(fm, fn, fk),
        remainders=(rm, rn, rk),
        specialized_calls=specialized,
        parametric_calls=total - specialized,
    )


class TiledGemm:
    """gemm at any size, composed from the bank's prebaked cores.

    ``bank`` is a :class:`~src.compiler.kernel_bank.KernelBank` holding the
    BLAS specs. The specialized (tile, tile, tile) variant is fetched (and
    admission-verified, compiling on first use) up front; the parametric
    variant covers edge tiles. ``contract`` selects the work-contract
    variant of both.
    """

    def __init__(self, bank: Any, *, tile: int = 64,
                 contract: str | None = None):
        self.bank = bank
        self.tile = int(tile)
        self.contract = contract
        self.core = bank.get(
            "gemm", contract=contract,
            specialized={"m": self.tile, "n": self.tile, "k": self.tile},
        )
        self.edge = bank.get("gemm", contract=contract)

    def __call__(self, a, b, c, alpha: float, beta: float,
                 m: int, n: int, k: int) -> np.ndarray:
        """Return ``alpha * A @ B + beta * C`` for row-major flat buffers.

        ``a`` is (m, k) flattened, ``b`` (k, n), ``c`` (m, n). ``c`` is not
        mutated; the result is a fresh array, matching the bank kernels'
        own convention of returning the written output.

        The decomposition is a deployment-vocabulary PLAN
        (``tiling_strategy.build_gemm_tile_plan``) executed by the
        deployment layer's serial-fallback executor -- this object supplies
        only the two prebaked variants and the caller-facing signature, so
        the worker pools thread the plan in one place when they land.
        """

        from .tiling_strategy import build_gemm_tile_plan, run_gemm_tile_plan

        plan = build_gemm_tile_plan(m, n, k, self.tile)
        return run_gemm_tile_plan(
            plan, self.core, self.edge, a, b, c, float(alpha), float(beta),
        )


__all__ = ["TiledGemm", "TilePlan", "plan_gemm_tiling"]
