# docstrings_math.py  ───────────────────────────────────────────────────────────
#
# Call `apply_math_docstrings(Turing)` once (e.g. right after scaffold import).
# All primitive + derived methods get rich, formal docstrings.

from textwrap import dedent

NAND_DOC = dedent(r"""
    ∇  —  NAND  (Sheffer stroke)            Domain: 𝔅 × 𝔅  → 𝔅
    
        ∇(x, y)  :=  ¬(x ∧ y)    (point-wise on bit-strings)
    
    Truth-table (per bit):
        x   y | ∇
        0   0 | 1
        0   1 | 1
        1   0 | 1
        1   1 | 0
    
    Functional completeness:
        ¬x     = ∇(x, x)
        x ∧ y  = ¬∇(x, y) = ∇(∇(x, y), ∇(x, y))
    
    TeX snippet:
        $\nabla(x,y)\;=\;\lnot(x\land y)$
    """)

SIGMA_L_DOC = dedent(r"""
    σ_L^k  —  left-shift by k                       Domain: 𝔅 × ℕ → 𝔅
    
        σ_L^k(b)  :=  b · 0^k        (concatenate k zero bits on the right)
    
    Algebraic law:
        |σ_L^k(b)| = |b| + k
    
    Interpretations:
      • On natural numbers (MSB→LSB), σ_L^1 multiplies by 2.
      • In Pigeon’s memory plane, it appends k pads without data loss.
    
    TeX snippet:
        $\sigma_L^{k}(b)=b \\mathbin{\\|} 0^{k}$
    """)

SIGMA_R_DOC = dedent(r"""
    σ_R^k  —  right-shift by k                      Domain: 𝔅 × ℕ → 𝔅
    
        σ_R^k(b)  :=   b[0 : |b|−k]     (drop k least-significant bits)
    
    Algebraic law:
        |σ_R^k(b)| = max(|b|−k, 0)
    
    Interpretations:
      • On natural numbers, σ_R^1 is floor-divide by 2.
      • Provenance is preserved because the dropped suffix is still available
        to hooks for logging or reversible variants.
    """)

MU_DOC = dedent(r"""
    μ  —  bitwise selector / merge                  Domain: 𝔅³ → 𝔅
    
        μ(a, b, ℓ)[i] = (¬ℓ[i])·a[i] ⊕ ℓ[i]·b[i]
    
    Semantics:
      ℓ[i]=0  → take bit a[i]  
      ℓ[i]=1  → take bit b[i]
    
    • Requires |a| = |b| = |ℓ|.  
    • Pure set-theoretic mux; use as write-mask, conditional copy, etc.
    """)

# -- any other primitive / derived docstrings here -----------------------------

DOCS = {
    'nand'    : NAND_DOC,
    'sigma_L' : SIGMA_L_DOC,
    'sigma_R' : SIGMA_R_DOC,
    'mu'      : MU_DOC,
}

def apply_math_docstrings(cls):
    """Attach the formal math docstrings to *cls* in-place."""
    for name, doc in DOCS.items():
        fn = getattr(cls, name, None)
        if fn is not None and (fn.__doc__ in (None, "")):
            fn.__doc__ = doc
    return cls
