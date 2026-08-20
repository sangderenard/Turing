"""Generic, parameterized BLAS kernels authored as COMPILER SOURCE.

The thesis this module tests (see ``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md``,
section 6, "Should we write a BLAS? Not now" -- written 2026-08-19, one day
before this module): **we do not hand-write performant kernels here.** We
write the algorithm plainly -- flat buffers, explicit loops, no threading, no
blocking, no SIMD hints -- and let the compiler's own pipeline
(``lower_ast_source_to_ssa`` -> a backend emitter -> the work contract's
identity/fusion/deployment policy) turn each parameterized instantiation into
a fast native kernel. This module is deliberately the missing GEMM-shaped
test case that document's own "revisit when" condition names.

Each kernel below is a pair:

* ``<NAME>_SOURCE`` -- a plain-Python **string**, the compiler's target.
  Never call it directly; it is not live code, the same way
  ``symbolic_fluid_dt.SYMBOLIC_FLUID_DT_SOURCE`` is not.
* ``<name>_reference`` -- a real, directly-callable Python function computing
  the identical result. This is the correctness oracle a compiled artifact
  must be checked against (never against a remembered number), and it is
  also what you get if you just want to run the math eagerly today.

Authoring rules, load-bearing and each one paid for by a real defect on the
eigh compilation effort (``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`` section 4)
one day before this module existed:

1. **Flat 1D buffers with computed row-major indices** (``a[i * n + j]``),
   never 2D tuple indexing (``A[i, j]``). This is the one convention already
   measured to reach a 25-basic-block, zero-shortfall compile for a real
   nested-loop numerical kernel (Jacobi eigh). 2D/ellipsis indexing is a
   live, separately-scoped blocker (section 4.5) -- do not reach for it here.
2. **Every loop bound is a passed-in parameter, never a literal constant.**
   Section 4.2 (open, unfixed as of this writing): under a *literal* loop
   bound, an array parameter the return value does not depend on is
   silently dead-stored -- dropped from the emitted signature -- even
   though the write is the function's whole observable effect. ``range(n)``
   with ``n`` a parameter does not trigger it. Since genericity over shape
   is the entire point of this module, every kernel already satisfies this
   by construction; do not "specialize" one to a hardcoded literal bound
   without re-reading section 4.2 first.
3. **Every loop variable is named distinctly, including across separate
   loops in the same function.** Section 4.1b (open): two *sequential*
   loops both spelled ``for i in range(n)`` emit phi nodes that do not
   dominate their uses -- LLVM's verifier rejects the module while
   ``artifact.shortfalls`` stays empty, a false green. Nested loops in one
   kernel here are named ``i``, ``j``, ``k`` outward-to-inward and never
   reused.
4. **No threading, no cache blocking/tiling, no SIMD intrinsics, no manual
   unrolling.** Per the owner's explicit direction: this work is paired
   with tuning the compiler's own automatic deployment calculation (the
   region/backend classification and the not-yet-wired thread-pool
   machinery, ``deployment_classification.py`` / ``turing_pool.c``), not
   with hand-parallelizing library code. A kernel here that already
   contains a manual optimization is exactly the "hand-written perfect
   kernel" this whole approach exists to stop needing, and it removes the
   compiler's own reason to grow that capability. If a kernel is ever
   tiled, the tiling must be something the compiler decided, not something
   read out of this file.
5. **dtype is not a parameter inside the source.** The kernels are written
   once, generically, over whatever numeric type the caller's bound
   arrays are; per-call-site specialization (already proven in this
   compiler -- every ``__specialized_<hash>`` function seen in the
   ``re.compile`` lane is exactly this mechanism) is what makes one generic
   source produce a monomorphic fast kernel per (dtype, shape) instantiation
   actually used, not an internal branch.

Escalation ladder, smallest first, matching the philosophy of
``tools/translation_scorecard.py``'s journeys: BLAS-1 no-reduction
(``scal``), BLAS-1 in-place update (``axpy``), BLAS-1 single-loop reduction
(``dot`` -- the proven scorecard level-4 shape with a real accumulator),
BLAS-2 first nested loop (``gemv``), BLAS-3 triple-nested loop (``gemm``),
and finally BLAS-1 ``rot``, the Givens plane rotation. ``rot`` is last
despite being a level-1 kernel because it is the only one whose defining
difficulty is not loop nesting: it writes back to elements it has already
read, which is a live compiler defect, not a ladder rung. Compile and
verify in this order; do not jump to ``gemm`` first.

To compile and run one kernel, see ``tools/compile_blas_probe.py`` --
same calling convention as
``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`` section 2 (proven on eigh):
``lower_ast_source_to_ssa`` -> ``emit_ssa_function_to_llvm`` ->
``compile_artifact`` -> ``prepare_artifact_execution``. Bind by
``artifact.buffer_order``, never by position or by a remembered id.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Level 0: scal  --  y[i] = alpha * x[i]
#
# Smallest possible member: one loop, one read, one write, no carried
# reduction, no in-place read-then-write of the SAME element. If this does
# not compile clean, nothing else in this file will either -- start here.
# ---------------------------------------------------------------------------

SCAL_SOURCE = """
def scal(x, y, alpha, n):
    for i in range(n):
        y[i] = alpha * x[i]
    return y
"""


def scal_reference(x, y, alpha, n):
    for i in range(n):
        y[i] = alpha * x[i]
    return y


# ---------------------------------------------------------------------------
# Level 1: axpy  --  y[i] = alpha * x[i] + y[i]
#
# Adds an in-place read-then-write of the output element itself -- the
# idiom every array-update loop in this compiler's own flagship (Shoal)
# already exercises, and the exact shape the eigh handoff's section 4.1
# fix (a load re-read across an in-place store) targeted.
# ---------------------------------------------------------------------------

AXPY_SOURCE = """
def axpy(x, y, alpha, n):
    for i in range(n):
        y[i] = alpha * x[i] + y[i]
    return y
"""


def axpy_reference(x, y, alpha, n):
    for i in range(n):
        y[i] = alpha * x[i] + y[i]
    return y


# ---------------------------------------------------------------------------
# Level 2: dot  --  sum_i x[i] * y[i]
#
# First real reduction: one carried scalar accumulator through a single
# loop. This is the proven scorecard level-4 shape ("loop, one carried
# value") doing genuine arithmetic instead of a toy update -- the highest-
# confidence rung before the ladder turns to nested loops.
# ---------------------------------------------------------------------------

DOT_SOURCE = """
def dot(x, y, n):
    total = 0.0
    for i in range(n):
        total = total + x[i] * y[i]
    return total
"""


def dot_reference(x, y, n):
    total = 0.0
    for i in range(n):
        total = total + x[i] * y[i]
    return total


# ---------------------------------------------------------------------------
# Level 3: gemv  --  y = alpha * A @ x + beta * y   (A is m-by-n, row-major)
#
# First NESTED loop: outer over rows (m), inner over columns (n) with a
# per-outer-iteration accumulator reset. This is deliberately the first
# GEMM-shaped test case the eigh handoff (section 6) named as not yet
# existing when it declined to write a BLAS "for now". Expect this rung to
# be where new compiler defects surface -- that is the point of it.
# ---------------------------------------------------------------------------

GEMV_SOURCE = """
def gemv(A, x, y, alpha, beta, m, n):
    for i in range(m):
        total = 0.0
        for j in range(n):
            total = total + A[i * n + j] * x[j]
        y[i] = alpha * total + beta * y[i]
    return y
"""


def gemv_reference(A, x, y, alpha, beta, m, n):
    for i in range(m):
        total = 0.0
        for j in range(n):
            total = total + A[i * n + j] * x[j]
        y[i] = alpha * total + beta * y[i]
    return y


# ---------------------------------------------------------------------------
# Level 4: gemm  --  C = alpha * A @ B + beta * C
#   A is m-by-k, B is k-by-n, C is m-by-n, all row-major flat.
#
# Triple-nested loop, the actual BLAS-3 target: the shape that unlocks
# block-Jacobi-style algorithms per the eigh handoff's own aside ("Only
# block Jacobi has GEMM-shaped inner work"). Loop variables i/j/k are
# distinct throughout, per authoring rule 3 above.
# ---------------------------------------------------------------------------

GEMM_SOURCE = """
def gemm(A, B, C, alpha, beta, m, n, k):
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total = total + A[i * k + p] * B[p * n + j]
            C[i * n + j] = alpha * total + beta * C[i * n + j]
    return C
"""


def gemm_reference(A, B, C, alpha, beta, m, n, k):
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total = total + A[i * k + p] * B[p * n + j]
            C[i * n + j] = alpha * total + beta * C[i * n + j]
    return C


# ---------------------------------------------------------------------------
# Level 5: rot  --  the Givens plane rotation, BLAS-1
#
#   x[i] <- c * x[i] + s * y[i]
#   y[i] <- c * y[i] - s * x[i]        (both from the PRE-rotation values)
#
# Why this rung exists at all, out of ladder order: it is the only kernel in
# this file that ``AbstractTensor.eigh`` actually contains. The Jacobi sweep
# in ``abstraction_methods/eigen.py`` has no GEMM and no GEMV anywhere in it
# -- its entire inner loop is four blocks of exactly the shape above (two
# row updates of S, two column updates of S, two column updates of V). A
# BLAS whose smallest member is ``scal`` cannot serve eigh; one with ``rot``
# can serve all of it.
#
# SIGN CONVENTION: standard BLAS ``drot`` as written above. The Jacobi
# convention in ``eigen.py`` is the transpose,
#     p <- c*p - s*q,  q <- s*p + c*q,
# which is this kernel called with ``s`` negated -- ``rot(p, q, c, -s, n)``.
# Do not fork the kernel to carry the other sign; negate at the call site,
# where the rotation angle is chosen anyway.
#
# WHY IT IS A REAL TEST AND NOT JUST A CONVENIENCE: rot reads a pair of
# elements, combines them, and writes both back -- the shape of the native
# codegen defect pinned in ``tests/test_llvm_inplace_store_aliasing.py``,
# where a value loaded before an in-place store was re-read after it and
# every plane rotation silently computed the wrong thing with ZERO emission
# shortfalls. That defect was fixed (commit ``3406310``, "Pin the loaded
# scalar so an in-place store cannot change it retroactively") and this
# kernel now compiles and runs bit-exact on BOTH buffers, measured. It is
# kept in the ladder as a second, independent guard on that fix: rot is the
# smallest kernel whose correctness depends on it.
#
# Note the scope of that fix as this kernel exercises it: ``x`` and ``y``
# here are DISTINCT buffers. The pinned repro aliased two computed indices
# of a SINGLE array. A future rot variant that rotates two rows of one
# matrix in place -- which is superficially what eigh wants -- is the
# same-buffer case and must be re-verified on its own, not assumed from
# this rung.
#
# The two temporaries ``xi``/``yi`` are load-before-store, not a manual
# optimization: the rotation is simultaneous by definition, and without
# them the kernel would be a different (wrong) function.
# ---------------------------------------------------------------------------

ROT_SOURCE = """
def rot(x, y, c, s, n):
    for i in range(n):
        xi = x[i]
        yi = y[i]
        x[i] = c * xi + s * yi
        y[i] = c * yi - s * xi
    return x
"""


def rot_reference(x, y, c, s, n):
    for i in range(n):
        xi = x[i]
        yi = y[i]
        x[i] = c * xi + s * yi
        y[i] = c * yi - s * xi
    return x

#: The escalation ladder, in the order to compile and verify them.
#: Each entry: (level, name, source text, reference callable, arity).
#: ``arity`` is the plain positional-argument order the reference and the
#: compiled entrypoint both share -- read ``artifact.buffer_order`` at bind
#: time regardless (section 2.1, rule 2: it is authoritative and can
#: disagree with ``parameter_names``).
KERNELS = (
    (0, "scal", SCAL_SOURCE, scal_reference, ("x", "y", "alpha", "n")),
    (1, "axpy", AXPY_SOURCE, axpy_reference, ("x", "y", "alpha", "n")),
    (2, "dot", DOT_SOURCE, dot_reference, ("x", "y", "n")),
    (3, "gemv", GEMV_SOURCE, gemv_reference,
     ("A", "x", "y", "alpha", "beta", "m", "n")),
    (4, "gemm", GEMM_SOURCE, gemm_reference,
     ("A", "B", "C", "alpha", "beta", "m", "n", "k")),
    (5, "rot", ROT_SOURCE, rot_reference, ("x", "y", "c", "s", "n")),
)

__all__ = [
    "SCAL_SOURCE", "scal_reference",
    "AXPY_SOURCE", "axpy_reference",
    "DOT_SOURCE", "dot_reference",
    "GEMV_SOURCE", "gemv_reference",
    "GEMM_SOURCE", "gemm_reference",
    "ROT_SOURCE", "rot_reference",
    "KERNELS",
]
