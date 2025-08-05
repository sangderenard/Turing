# turing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Dict, Optional, Tuple, Protocol

class AlgebraViolation(Exception):
    """Raised when a backend violates required algebraic laws (length, shape, etc.)."""

class BackendMissing(Exception):
    """Raised when a required hook is not provided."""

class BitstringProtocol(Protocol):
    """
    Abstract 'carrier set' B ∈ 𝔅 = ⋃_{n∈ℕ} {0,1}^n .
    Backends may use any representation; the scaffold sees only an opaque value.
    """
    ...  # purely nominal; no runtime checks required


@dataclass(frozen=True)
class Hooks:
    """
    All hooks are *total* functions on the carrier set B.
    These are the *only* places where mechanics enter.
    """

    # --- Primitive, pointwise Boolean generator (functionally complete):
    nand: Callable[[BitstringProtocol, BitstringProtocol], BitstringProtocol]
    # Laws (responsibility of backend):
    #   length(nand(x,y)) == length(x) == length(y)

    # --- Motion (addressability):
    sigma_L: Callable[[BitstringProtocol, int], BitstringProtocol]  # σ_L^k
    sigma_R: Callable[[BitstringProtocol, int], BitstringProtocol]  # σ_R^k
    # Laws:
    #   length(σ_L^k(x)) == length(x) + k
    #   length(σ_R^k(x)) == max(length(x) - k, 0)

    # --- Structural glue:
    concat:  Callable[[BitstringProtocol, BitstringProtocol], BitstringProtocol]  # ⊔
    slice:   Callable[[BitstringProtocol, int, int], BitstringProtocol]           # [i:j]
    mu:      Callable[[BitstringProtocol, BitstringProtocol, BitstringProtocol], BitstringProtocol]
    # μ(b0,b1,ℓ)[i] = (¬ℓ[i])·b0[i] ⊕ ℓ[i]·b1[i]  (selector; 1 picks from b1)
    # Laws:
    #   length(concat(x,y)) == length(x)+length(y)
    #   length(slice(x,i,j)) == max(j-i,0)
    #   length(mu(x,y,ℓ)) == length(x) == length(y) == length(ℓ)

    # --- Size & constructors:
    length: Callable[[BitstringProtocol], int]           # |·|
    zeros:  Callable[[int], BitstringProtocol]           # 0^n

    # (Optional but useful)
    # ones: Optional[Callable[[int], BitstringProtocol]] = None


class Turing:
    """
    Purely abstract scaffold for a Turing-complete calculus over bitstrings.

    Carrier set: 𝔅 = ⋃_{n∈ℕ} {0,1}^n  (any backend representation)
    Primitive signature:
        ∇ : 𝔅 × 𝔅 → 𝔅           (NAND; pointwise)
        σ_L^k : 𝔅 → 𝔅           (left shift by k, append k zeros)
        σ_R^k : 𝔅 → 𝔅           (right shift by k, drop k LSBs)
        ⊔ : 𝔅 × 𝔅 → 𝔅           (concatenation)
        [i:j] : 𝔅 → 𝔅           (slice)
        μ : 𝔅 × 𝔅 × 𝔅 → 𝔅       (choice/merge; selector 1 picks from second arg)
        |·| : 𝔅 → ℕ            (length)
        0^n : ℕ → 𝔅            (all-zero constructor)

    All *derived* operators below are defined *only* in terms of the primitives.
    """

    def __init__(self, hooks: Hooks):
        self.h = hooks
        self._ensure_hooks()

    # --------- primitive accessors (guarded) ---------------------------------
    def _ensure_hooks(self) -> None:
        required = ("nand","sigma_L","sigma_R","concat","slice","mu","length","zeros")
        for name in required:
            if getattr(self.h, name, None) is None:
                raise BackendMissing(f"Hook {name} must be provided.")

    # Short aliases to keep math close to code
    def nand(self, x, y):  # NAND
        """NAND: ∇(x, y)"""
        self._eq_len2(x, y, "NAND")
        return self.h.nand(x, y)

    def sigma_L(self, x, k: int):
        """Left shift: σ_L^k(x)"""
        return self.h.sigma_L(x, int(k))

    def sigma_R(self, x, k: int):
        """Right shift: σ_R^k(x)"""
        return self.h.sigma_R(x, int(k))

    def concat(self, x, y):
        """Concatenation: x ⊔ y"""
        return self.h.concat(x, y)

    def slc(self, x, i: int, j: int):
        """Slice: x[i:j]"""
        return self.h.slice(x, int(i), int(j))

    def mu(self, x, y, sel):
        """Selector/merge: μ(x, y, sel)"""
        self._eq_len3(x, y, sel, "μ")
        return self.h.mu(x, y, sel)

    def length(self, x) -> int:
        """Length: |x|"""
        return int(self.h.length(x))

    def zeros(self, n: int):
        """All-zero constructor: 0^n"""
        return self.h.zeros(int(n))

    # --------- helpers: length checks ----------------------------------------
    def _len(self, x) -> int: return self.h.length(x)
    def _eq_len2(self, a, b, ctx="op"):
        if self._len(a) != self._len(b):
            raise AlgebraViolation(f"{ctx}: length mismatch {self._len(a)} vs {self._len(b)}")
    def _eq_len3(self, a, b, c, ctx="op"):
        la, lb, lc = self._len(a), self._len(b), self._len(c)
        if not (la == lb == lc):
            raise AlgebraViolation(f"{ctx}: length mismatch {la}, {lb}, {lc}")

    # =========================
    # Derived pointwise logic
    # =========================

    def NOT(self, x):
        """Negation: ¬x = nand(x, x)"""
        return self.nand(x, x)

    def AND(self, x, y):
        """Conjunction: x∧y = ¬nand(x, y) = nand(nand(x, y), nand(x, y))"""
        t = self.nand(x, y)
        return self.nand(t, t)

    def OR(self, x, y):
        """Disjunction: x∨y = ¬(¬x ∧ ¬y)"""
        nx, ny = self.NOT(x), self.NOT(y)
        return self.NOT(self.AND(nx, ny))

    def XOR(self, x, y):
        """Exclusive OR: x⊕y = (x∨y) ∧ ¬(x∧y)"""
        a = self.OR(x, y)
        b = self.AND(x, y)
        return self.AND(a, self.NOT(b))

    def XNOR(self, x, y):
        """Exclusive NOR: ¬(x⊕y)"""
        return self.NOT(self.XOR(x, y))

    def mux(self, sel, a, b):
        """Selector: mux(sel, a, b) = mu(a, b, sel)"""
        return self.mu(a, b, sel)

    # =========================
    # Structural combinators
    # =========================

    def rho_L(self, x, k: int):
        """
        Rotate left: rho_L^k(x) = x[k:len(x)] concat x[0:k]
        """
        k = int(k)
        n = self._len(x)
        k = 0 if n == 0 else k % n
        return self.concat(self.slc(x, k, n), self.slc(x, 0, k))

    # =========================
    # Arithmetic (bitstrings as MSB→LSB)
    # =========================

    def half_adder(self, a, b) -> Tuple[Any, Any]:
        """Return (sum, carry), pointwise."""
        return self.XOR(a,b), self.AND(a,b)

    def full_adder(self, a, b, cin) -> Tuple[Any, Any]:
        """sum = a⊕b⊕cin ; carry = (a∧b) ∨ (cin ∧ (a⊕b))."""
        s1, c1 = self.half_adder(a,b)
        s2, c2 = self.half_adder(s1, cin)
        carry  = self.OR(c1, c2)
        return s2, carry

    def ripple_add(self, x, y) -> Any:
        """
        Addition: ripple_add(x, y) using primitives {nand, sigma_L/sigma_R, concat, slice, mu}.
        Returns a bitstring of length length(x)+1 (with possible leading carry).
        """
        self._eq_len2(x, y, "ripple_add")
        n = self._len(x)
        if n == 0:
            return self.zeros(1)  # convention

        carry = self.zeros(n)    # initial carry 0
        sumv  = self.zeros(n)
        
        # FIX: Create a '1' bit to use in selector masks.
        one = self.NOT(self.zeros(1))

        for i in range(n - 1, -1, -1):
            xi = self.slc(x, i, i + 1)
            yi = self.slc(y, i, i + 1)
            ci = self.slc(carry, i, i + 1)

            si, co = self.full_adder(xi, yi, ci)  # both length 1

            # Pad si and co to length n
            si_pad = self.concat(self.zeros(i), self.concat(si, self.zeros(n - 1 - i)))
            co_pad = self.concat(self.zeros(i-1), self.concat(co, self.zeros(n - i))) if i > 0 else self.concat(co, self.zeros(n-1))


            # FIX: Generate selector with a '1' at the target bit position 'i'.
            sel = self.concat(self.zeros(i), self.concat(one, self.zeros(n - 1 - i)))
            # The double mu is convoluted but works with a correct `sel`
            sumv = self.mu(sumv, self.mu(self.zeros(n), si_pad, sel), sel)

            if i > 0:
                # FIX: Generate selector to update the *next* carry bit at 'i-1'.
                selc = self.concat(self.zeros(i - 1), self.concat(one, self.zeros(n - i)))
                carry = self.mu(carry, co_pad, selc)
            
        # This final calculation is redundant but safe; we leave it as is.
        x0, y0 = self.slc(x, 0, 1), self.slc(y, 0, 1)
        c0 = self.slc(carry, 0, 1)
        _, c_out = self.full_adder(x0, y0, c0)

        return self.concat(c_out, sumv)

    def succ(self, x) -> Any:
        """Successor: x + 1 implemented via ripple_add with a single 1."""
        one = self.concat(self.zeros(self._len(x)-1), self.slc(self.NOT(self.zeros(1)), 0, 1)) if self._len(x) > 0 else self.slc(self.NOT(self.zeros(1)), 0, 1)
        return self.ripple_add(x, one)

    # =========================
    # Minimal TM-like tape edits
    # =========================

    def write_bit(self, tape, idx: int, bit1):
        """
        Write a single bit at index idx using mu.
        `bit1` is a length-1 bitstring.
        """
        bit1 = bit1.copy() if isinstance(bit1, list) else [bit1]
        n = self._len(tape)
        if not (0 <= idx < n):
            raise AlgebraViolation("write_bit: index out of bounds")
        
        # FIX: Create a '1' bit to correctly form the selector mask.
        one = self.NOT(self.zeros(1))
        sel = self.concat(self.zeros(idx), self.concat(one, self.zeros(n - 1 - idx)))
        
        patch = self.concat(self.zeros(idx), self.concat(bit1, self.zeros(n - 1 - idx)))
        return self.mu(tape, patch, sel)

    def move_head_left(self, head_mask):
        """Move 1-bit head marker left by 1 (saturate at 0 via backend sigma_R/sigma_L contracts)."""
        return self.sigma_L(self.sigma_R(head_mask, 1), 0)  # relies on backend semantics

    def move_head_right(self, head_mask):
        """Move 1-bit head marker right by 1."""
        return self.sigma_R(self.sigma_L(head_mask, 1), 0)

    # =========================
    # Algorithm registry
    # =========================

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not hasattr(self, "_alg"):
            self._alg: Dict[str, Callable[..., Any]] = {}
        self._alg[name] = fn

    def run(self, name: str, *args, **kwargs) -> Any:
        if not hasattr(self, "_alg") or name not in self._alg:
            raise KeyError(f"Algorithm {name!r} not registered.")
        return self._alg[name](self, *args, **kwargs)
