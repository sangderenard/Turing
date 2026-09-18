# Adding Dimensions of Dtype

## A manifesto for the encoding, algebra, and basis layers of AbstractTensor

---

## 0. Thesis

`AbstractTensor` can name three types: `float32`, `int64`, `bool`. Everything else
we want to compute with — complex numbers, spinors, quaternions, residue systems,
factorial-base permutation indices, intervals, jets — has nowhere to live.

The instinct is to add dtypes to a list. That is wrong, and it does not scale: it
is linear work for linear gain, and it multiplies against every backend.

**A dtype is not a label. It is a point in a product space of independent axes.**
Adding a *dimension of dtype* means adding an axis. Adding a *dtype* means naming
a point.

And the way to make that pay:

> **Declare the encoding once, in Python. Compile the entire (operation × encoding
> × backend) matrix to native code ahead of time. Ship the matrix, not the
> translator. Degrade only by the narrowest gap, per cell, and never silently.**

Get that right and the space of representable number systems flowers
combinatorially while the runtime cost stays at zero.

Get any one of the particulars wrong and it collapses into a boxed, tagged,
dynamically-dispatched interpreter that is slower than what we have now. §6 is the
list of ways that happens, and it is not a footnote — it is the load-bearing part
of the document.

---

## 1. Where we actually stand

Verified by reading and running the tree, not inferred.

### 1.1 There is no dtype system to extend

The entire vocabulary is three sentinels at `src/common/tensors/abstraction.py:351-354`,
commented "Default sentinel, can be replaced by backend". Each backend overrides
them as properties holding *different Python types* — numpy classes, `np.dtype`
instances, plain strings — and they do not agree on values:

- `accelerator_backends/glsl_tensor_backend.py:52-54` — `long_dtype_` is **int32**
- `accelerator_backends/ssa_backend.py:743-745` — `float_dtype_` is **float64**

There is no registry, no promotion table, no lattice. Two failure modes follow:

- `numpy_backend.py:881-899` — `to_dtype_` **silently defaults anything
  unrecognized to float32**. A typo in a dtype name is a silent numeric change.
- `numpy_backend.py:440-453` — `_numpy_dtype_to_torch` is a hardcoded 5-entry map
  that deliberately reports *torch* dtype objects from a numpy-backed tensor, and
  returns `None` outside it. `.dtype` on a complex tensor is `None`: **the
  abstraction cannot name what its own `fft` produces.**

### 1.2 Complex support is ragged, which is worse than absent

On the default (Nodus) backend, given `X = x.fft()`:

| operation | result |
| --- | --- |
| `X + X`, `X * X`, `X.abs()` | `NodusUnsupported: the arena cannot hold this tensor` |
| `X @ X` | `NodusUnsupported: nodus matmul is f32/f64 only` |
| `X.sum()`, `X.sqrt()` | works, returns `complex128` |
| `AT.real(X)`, `AT.imag(X)` | works — **and records no autograd node** |
| `conj`, `angle` | do not exist anywhere |

`AbstractTensor.real` (`abstraction.py:547-557`) builds its result without calling
`_pre_autograd`, so it never joins the tape. Measured: `fft → ifft → sum` yields
gradient `[1,1,…]`; `fft → real → sum` yields `None`. **The only usable exit from
the spectral domain silently destroys gradients.**

### 1.3 Every backend's native type table is narrow and private

| backend | what a number can be |
| --- | --- |
| Nodus arena (`nodus_backend.py:36-46`) | f32/f64, i8–i64, u8–u64, bool. No complex, no f16 |
| Nodus kernel ISA (`nodus/src/kernel_isa.h:23-28`) | `{I32, U32, F32}` |
| SPIR-V (`nodus/src/kernels/kernel_spirv.cpp:160-167`) | `t_u32/t_i32/t_f32`; **bitwise rejected on f32** |
| LLVM (`ssa_llvm_backend.py:458-469`) | effectively double + i32/i64 |
| WASM (`ssa_wasm_backend.py:186-201`) | f64-only value stack |
| Fortran (`ssa_fortran_backend.py:304-317`) | kind table, fails closed |

A *hand-written* native complex dtype would have to be implemented six times, plus
the C ABI enum and the arena. That is why hand-writing is off the table — not why
native types are.

### 1.4 What the compiler already gives us

Three facts that decide the architecture:

- **There is a JIT.** `llvm_jit_backend.py:129` and `llvm_optimizing_pipeline.py:350`
  both call `create_mcjit_compiler`.
- **The LLVM backend emits *textual* IR.** `_value_llvm_type`
  (`ssa_llvm_backend.py:450-471`) returns strings — `"i1"`, `"i32"`, `"i64"`,
  `"double"`, `"ptr"` — and already routes aggregates through `ptr` via
  `ssa_aggregate_outputs`. Emitting `{float, float}`, `<2 x float>`, or `i23` is
  *emitting a different string*.
- **LLVM IR has arbitrary integer widths.** `i7`, `i23`, `i128` are legal. Posits,
  factoradic lanes, GF(2ⁿ), and custom sign/exponent/mantissa layouts get machine
  representations with no emulation.

### 1.5 Two complete worlds exist, with no bridge

**The minimal-operator world.** `src/turing_machine/turing.py:20-54` defines eight
primitives — `nand`, `sigma_L`, `sigma_R`, `concat`, `slice`, `mu`, `length`,
`zeros` — and derives NOT/AND/OR/XOR/mux/half-adder/full-adder/ripple-add from
them under enforced length laws. `turing_provenance.py:62` records every primitive
call as a DAG; `survival_computer.py:44-58` compiles that graph to tape IR and runs
it on simulated cassette hardware. **Every result is traceable to NAND.**
`compiler/abstract_tensor_bitops.py:24-81` carries this over AbstractTensor —
bitstrings as rank-1 tensors of 0/1, `nand = 1 - left*right`. Tested.

This is not "limited" by design. It is limited by which lowering tools have been
built, and the precedent for building the rest is established.

`Hooks` is also the **operator-set-swap prototype**: a frozen dataclass of eight
implementations, bound once by `Turing(hooks)`, after which every derived operator
contains *zero* dispatch — it cannot, it only sees what it was handed. Three sets
exist today (pure-Python, `abstract_tensor_hooks`, and `instrument_hooks`, which
adds complete provenance tracing by swapping the set rather than branching inside
anything). §5 generalizes this.

**The gap.** `compiler/bitops.py:72-211` declares an extensive taxonomy —
`BitStruct(integer_pieces, depths, encoding)` with `Integer`, `Rational`,
`Float(mantissa, exponent)`, `Complex(real, imag)`, `Domain`, `Tensor`,
`BitTensor`, plus `Schema`, `Struct`, `Manifold`, `TaylorSeries`, `Integral`,
`Derivative`. **These constructors record widths and nothing else. None carry
arithmetic.** Only unsigned N-bit integers are actually built from NAND
(`bitops_translator.py:78-126`). The float/complex/rational tier (`GrayTableOps`,
`bitops.py:705-853`) is native-Python lookup tables with no rounding, no
normalization, no NaN/Inf, and empty `float_add_table` / `float_mul_table` dicts
marked "reserved for future".

**The carrier taxonomy exists. The algebra does not. That is the hole.**

---

## 2. Dtype as a product space

| axis | values | consumed by |
| --- | --- | --- |
| **storage** | f32, f64, i32, i64, bool | the backend |
| **algebra** | real, complex, dual, split-complex, quaternion, Clifford(p,q), RNS, tropical, group algebra… | the lifting layer |
| **layout** | how the carrier is realized — see §4 | code generation |
| **domain** | angle mod 2π, probability, log-domain, index, unit-norm | validity, simplification |
| **basis** | spatial, Fourier, eigen, irrep, wavelet | spectral dispatch |

Complex is `(f32, complex, native-vector, —, spatial)`. A unit quaternion rotation
is `(f32, quaternion, native-vector, unit-norm, spatial)`. A Lehmer code is
`(i32, symmetric-group, factorial-radix, index, —)`.

---

## 3. Basis and algebra: designing for the extreme cases

It is tempting to define an algebra as a rank-3 structure-constant tensor
`c[i,j,k]`, making `mul` a single contraction. For bilinear algebras over a fixed
basis this is exactly right and enormously powerful: complex, split-complex, dual
numbers, quaternions, octonions, Clifford algebras, and spinors **differ only in
`c`**. One `mul` implementation, an entire family free.

That generality is real, and it is also a trap. Structure constants presuppose a
finite-dimensional vector space over a commutative ring with a fixed basis and a
bilinear product. Many systems we want are not that.

### 3.1 Factorial number space

The factorial base (factoradic) writes an integer as `Σ dᵢ · i!` with `0 ≤ dᵢ ≤ i`.
Two properties break naive assumptions immediately:

1. **The digits are heterogeneous.** Digit `i` has radix `i+1`. There is no
   uniform component width, so any carrier assuming homogeneous lanes fails. This
   generalizes to mixed-radix broadly, and to tapered formats like posits where
   field widths vary per value.

2. **Its natural algebra is not arithmetic.** Factoradic digits are the Lehmer
   code of a permutation: the representation indexes the symmetric group `Sₙ`. The
   meaningful operation on two such values is **group composition**, not digit-wise
   addition with carries. An encoding whose product is composition in a non-abelian
   group has no structure-constant table over a positional basis.

Factorial space is the canonical stress test: **heterogeneous carrier plus
non-arithmetic algebra.** Accommodate it and the ordinary cases are trivial.

It also stresses the *basis* axis instructively. Just as the DFT diagonalizes
circulant operators on `ℤ/nℤ`, the **non-abelian Fourier transform on `Sₙ`**
decomposes functions on permutations into irreducible representations. The
"spectral domain" of a factorial-coded value is an irrep decomposition, not a
frequency spectrum. Basis and algebra are not independent: *the algebra determines
which bases diagonalize it.* Any design treating `basis` as a free-floating tag
will get this wrong.

### 3.2 The other extreme cases, and which assumption each breaks

| system | breaks |
| --- | --- |
| **Residue (RNS/CRT)** | positional value. `add`/`mul` are componentwise and *free*; **comparison and division are expensive.** Inverts the cost model |
| **Logarithmic (LNS)** | `mul` becomes `add`; `add` becomes a table lookup. Inverted the other way |
| **Tropical / max-plus** | additive inverse. A semiring — no subtraction exists |
| **Galois GF(2ⁿ)** | characteristic zero. `add` is XOR; `mul` is carry-less |
| **Interval / affine** | total order and exactness. Ordering is partial |
| **Dual numbers / jets** | nilpotency (`ε² = 0`); component count varies with jet order. This *is* forward-mode autodiff |
| **p-adic** | finite component count and archimedean ordering |
| **Octonions** | associativity |
| **Quaternions, Clifford** | commutativity |
| **Complex** | total order |
| **Posits, floats** | exactness — rounding is part of the semantics |
| **Modular ℤ/nℤ** | unbounded range; invertibility depends on the value |

This is a specification of the assumptions the design is forbidden to hardcode:

> homogeneous lanes · positional weighting · bilinearity over a fixed basis ·
> additive inverse · associativity · commutativity · total order · fixed component
> count · exactness · cheap comparison

### 3.3 What follows

**An algebra is a declared set of operator implementations, together with the
algebraic laws it claims to satisfy. Structure constants are one common
*generator* of that set, not the definition of it.**

An encoding declares its capabilities — semiring, ring, field, division algebra,
associative, commutative, ordered, normed, exact — and each operation declares what
it *requires*. `sort` requires `ordered` and refuses complex at declaration time.
Gaussian elimination requires `field` and refuses `ℤ/6ℤ`. A solver requiring
`associative` refuses octonions.

**Illegal compositions become declaration-time errors instead of silently wrong
runtime numbers.** §1.2 is the evidence for why that matters.

---

## 4. The realization ladder — resolved per cell

An encoding is *declared* once. How it is *realized* is decided at compile time,
**per cell of the matrix** — one (operation, encoding, backend) triple at a time.

| rung | realization | when | cost |
| --- | --- | --- | --- |
| **1. Delegate** | the backend already has the type (numpy/torch `complex64`) | type exists natively | zero |
| **2. Native composite** | built from native scalars the target *does* have — `<2 x float>`, `vec2`, `vec4`, `{float,float}`, `iN` | the target has the component scalars | ~zero |
| **3. Packed u32 shim** | arbitrary-bit-length algorithms over 32-bit words | the target cannot express the type in its scalars | 10–50× |
| **4. NAND reference** | the eight primitives, provenance-traced | validation only | ruinous |

### 4.1 Rung 2 is the common case and must stay transparent

**This is the load-bearing point of the whole ladder.** Most encodings are not
exotic — they are small tuples of ordinary floats, and every backend has ordinary
floats.

`complex64` on a GPU is a `vec2<f32>`. Its multiply is:

```
(a + bi)(c + di) = (ac − bd) + (ad + bc)i
```

Four float multiplies and two adds — **all native, all vectorizable, all full
speed.** SPIR-V has `f32`, so SPIR-V has `complex64`. Nothing is packed into
integers; nothing is emulated. Quaternions are `vec4`, Clifford blades are short
float arrays, dual numbers are two floats. The structure-constant contraction of
§3 expands at compile time into a handful of native float operations.

> **Not everything becomes an integer just because it needs a matrix.** The u32
> shim is the *floor*, not the default.

### 4.2 The ladder is chosen per cell, never per region

This is the caveat that decides whether §4.1 survives contact with real programs.

If the rung were chosen per *region*, a region would have to pick a realization
valid for every encoding inside it — which means the worst one wins. One `posit16`
in the neighbourhood and your `complex64` gets integer arithmetic, at 30× the cost,
for no reason.

Because each cell is compiled separately, `(mul, complex64, SPIR-V)` lands on rung
2 as native `vec2` ops while `(mul, posit16, SPIR-V)` lands on rung 3 as packed
u32 — in the same program, in the same region, with no interaction between them.
Degradation is also **per operation**, not per type: SPIR-V rejects bitwise ops on
f32, but `complex64` never asks for bitwise ops, so `complex64` never leaves rung 2.

**A cell is the unit of realization. Nothing coarser.**

### 4.3 Why u32 is the one true floor

Every backend has 32-bit integers. Not "mostly" — all of them: SPIR-V (`I32`/`U32`,
two of its three types), WASM `i32`, LLVM `i32`, Fortran integer kind, Nodus arena
`i32`/`u32`, GLSL `int`/`uint`.

A float-based fallback cannot serve this role: `kernel_spirv.cpp:330,400,430`
rejects bitwise ops on f32, so exact bit manipulation is unavailable on the GPU
path. A u32 substrate can express everything — software float, posits, GF(2ⁿ),
factoradic lanes, bignum, arbitrary precision — and it is *exact*, so the floor has
no rounding behaviour the fast path lacks.

Two implementation notes:

- **32×32 multiply needs a 16-bit split.** The product of two 32-bit values does
  not fit in 32 bits, and not every target has a widening multiply or a 64-bit type
  to catch it. Split each operand into 16-bit halves so partial products fit, then
  recombine with explicit carries — the trick GPU code has always used.
- **Division, sqrt, and transcendentals are the hard tier**, and they are solved
  problems: Goldschmidt/Newton–Raphson, compiler-rt's `__udivti3`, Berkeley
  SoftFloat. `compiler/wasm_math_tables.py` already builds table and Maclaurin-series
  approximations with *measured* error bounds (`MathFunction:50`, `build_table:213`,
  `build_series:325`) and should be the engine for the last tier.

### 4.4 Rung 4 is the oracle

The NAND path is too slow to compute with and exactly right for proving. Every cell
realized on rungs 1–3 can be checked against its NAND-lowered twin, with
`turing_provenance.py` demonstrating the derivation reduces to the primitive basis.
The survival computer stops being an isolated curiosity and becomes the
verification tier of the production stack.

The three lower rungs are *the same algorithm at three packing densities* — 1 bit
per element, 32 bits per word, machine-native. Each validates the one above it.

---

## 5. The matrix is the artifact

### 5.1 There is no runtime translator

The universal translator is not a thing that runs. **It is a generator, and its
output is native code.**

Say it several ways, because every one of them rules out a different wrong
implementation:

- **The plate and the print run.** The declaration is a printing plate. The matrix
  is the print run. The plate does not ride along with the newspapers.
- **Crystallization, not solution.** Compilation freezes the declaration into a
  fixed lattice of emitted cells. Nothing remains dissolved, and nothing is still
  deciding anything when the program runs.
- **The generic never exists.** There is no generic `complex_multiply` that gets
  specialized at runtime. There are *only* specializations. The generic is a
  fiction of the source language, discharged during lowering — the standard name
  for this is monomorphization.
- **A value does not know its type; the code knows.** A `complex64` at runtime is
  two floats in registers. It carries no tag, no header, no descriptor, because
  nothing downstream will ever ask it. Type information lives in the instruction
  stream, not the data.
- **At runtime there is nobody left to ask.** Every question the declaration could
  answer was answered at build time. The declaration object is not loaded, not
  consulted, not present.

Concretely, what this buys: no boxing, no tagged unions, no marshalling at operator
boundaries, no virtual calls in inner loops, no interpretive layer, and therefore
nothing standing between adjacent operations that would prevent them fusing.
**`complex64` on the GPU is bit-for-bit the machine code you would have hand-written
in the shader** — not "close to," the same, because that cell was compiled for
exactly that triple and there is no wrapper to pay for.

The cost model moves entirely to build time: compile duration and emitted code
size. Runtime overhead is zero by construction. That is what makes combinatorial
flowering affordable — the explosion is in emitted cells, which is a build-time
budget you can measure and bound, not a tax paid on every element forever.

### 5.2 Resolution is declared, never inferred

A region's dtypes are in exactly one of three states, and **which state must be
stated, not discovered**:

- **Resolved** — one encoding, known at compile time. One branch-free body, called
  directly. No runtime logic whatsoever.
- **Fanned** — a known *finite set*. Compile one specialized branch-free body per
  member; select among them **once, at region entry**, by index into a table of
  already-native functions. This is a selection, not a conversion. It is the
  sensible default, because the set of possible encodings is always finite and
  usually tiny.
- **Dynamic** — genuinely opaque at compile time. Rare, legitimate occasionally,
  and **must be explicitly declared and loudly reported.**

The failure this guards against is the *implicit* third state. Nobody deliberately
declares a dtype opaque; some conservative pass upstream merely fails to prove it
static, quietly demotes the region, and the whole neighbourhood lands in a generic
boxed body. **If a pass cannot prove staticness, that is a diagnostic, not a quiet
demotion.**

### 5.3 Fan width is the only budget knob

Fan-out's cost is code size, and the worst case is combinatorial: a region with `k`
operands over `d` encodings is `d^k` bodies. Two mitigations:

- **Fan over reachable tuples, not the cartesian product.** Emit the operand
  combinations the program actually produces, not the whole space.
- **Unify cells that lower identically.** Several encodings frequently collapse to
  the same emitted code — deduplicate *after* lowering, not before.

When a fan would be too wide, the answer is **never** to put tests back inside the
operators. It is to narrow the region until each piece resolves, or to accept a
declared-dynamic region that announces itself. "Whichever is more efficient at the
time" is a question about *which cells to emit* — it is never a question about what
to do at runtime.

### 5.4 Dispatch is a region contract, not an operator property

Generalize `Hooks` (§1.5): a region binds a whole coherent operator set once, and
every operation inside is branch-free because it can only see what it was bound.

The reason to hoist dispatch out of operators is not the branch itself. It is that
**a conditional inside an operator is a fusion barrier** — and this tree has an
entire fused-program IR (`fused_ir.py`, `fused_program_wasm_backend.py`,
`fused_program_python_backend.py`) that a dtype test would defeat. On the GPU path
it is worse than a lost optimization: a branch is warp divergence, paid on every
lane whether taken or not.

Regions already exist. `deployment_classification.py` and
`glsl_deployment_strategy.py` partition programs into graphics-output,
shader-compute, thread-workers, and host-linear. **Math regions should ride on that
existing partition rather than invent a parallel one** — ideally one boundary serves
three purposes at once: deployment region, operator-set binding, and fusion body.
Conversions and validity checks live at that single seam, and an operation the
region's set does not support is a declaration-time error.

### 5.5 Generation, and what "defer to LLVM" must mean

```
    Python declaration  (carrier + algebra + laws)
              │
              ▼
      compiler / SSA lowering
              │
              ▼
        LLVM  ── the lowering authority
              │
    ┌─────────┼──────────┬───────────┐
    ▼         ▼          ▼           ▼
 native   SPIR-V      WASM       Fortran/C
 (JIT)    (re-emit)  (re-emit)   (re-emit)
```

**A SPIR-V shader cannot call into an MCJIT'd function.** So "other backends defer
to LLVM" must mean *defer to the lowering strategy LLVM used* — each backend
re-emits the same algorithm in its own IR — and never *link against LLVM's compiled
output*. The shim is a **portable algorithm description**; the JIT is one consumer
of it, not its home. This is the difference between covering everything and
covering everything with a CPU.

### 5.6 The lifting taxonomy

| bucket | operations | work required |
| --- | --- | --- |
| **Free** | add, sub, neg, scalar-mul, reshape, transpose, cat, stack, index, gather/scatter, cumsum | none per encoding — apply per component |
| **Generated** | `mul`, `matmul`, `conj`, `norm2`, `inv` | one structure-constant table, where the algebra is bilinear |
| **Declared** | anything breaking §3.2 assumptions — RNS compare, LNS add, group composition | explicit implementation |
| **Analytic** | `exp`, `log`, `sqrt`, transcendentals | closed forms for normed algebras; otherwise series |

`matmul` over an encoding lifts to `k²` calls of the target's **native real
matmul** — 3 for complex with Karatsuba. We call the fastest kernel the hardware
has and never tell it what a complex number is.

### 5.7 What happened to the wrapper question

An earlier draft argued at length for a Python wrapper object holding `k` tensors,
versus adding a component axis to the shape. **That question is largely dissolved.**
It was about how Python glues components together — and if the cell is compiled,
Python holds a handle to a real type and glues nothing.

The wrapper survives only as the interpreted fallback's concern, where it remains
the right call for the reason originally given: a component axis lets every existing
shape-touching operation silently corrupt (`sum()` adding real to imaginary and
returning a plausible wrong number), whereas a wrapper makes the supported surface
explicit and fails loudly on anything unlifted.

---

## 6. How this dissolves into sludge

Every item below turns the design back into a boxed, dynamically-dispatched
interpreter — slower than what we have today, and harder to fix because it will
look like it works. **The particulars are the design.** None of these is a nicety.

**If resolution is inferred rather than declared.** A conservative pass fails to
prove staticness, silently demotes a region to dynamic, and everything inside gets
boxed. Nobody wrote that decision down; nobody can find it later.
→ *Resolution state is declared. Unprovable staticness is a diagnostic.* (§5.2)

**If the ladder is chosen per region instead of per cell.** The worst encoding in
the neighbourhood sets the realization for all of them, and `complex64` silently
gets shim arithmetic at 30× cost.
→ *A cell is the unit of realization.* (§4.2)

**If values carry runtime type tags.** Boxing means allocation, indirection, cache
misses, and no vectorization — a `complex64` array stops being two float planes and
becomes a pointer chase.
→ *Type lives in the instruction stream, not the data.* (§5.1)

**If any conversion layer survives into runtime.** Marshalling between operators
costs more than the arithmetic and makes fusion impossible.
→ *Adjacent cells share representation by construction; conversion happens only at
declared region boundaries.* (§5.4)

**If dispatch lives inside operators.** Each conditional is a fusion barrier on CPU
and warp divergence on GPU, paid per lane whether taken or not.
→ *Dispatch is hoisted to region entry, or absent entirely.* (§5.4)

**If the declaration is consulted at runtime.** The moment any part of the encoding
object is read during execution, you have built an interpreter and the entire
compile-time argument evaporates.
→ *The declaration is a build-time input. It is not loaded at runtime.* (§5.1)

**If degradation is silent.** A region drops to the u32 shim, runs 30× slower, and
nobody finds out for a month.
→ *Every drop emits a diagnostic naming the encoding, the operation, and the reason
the target refused:* `posit16.div degraded to packed-u32: SPIR-V has no i16
division`.

**If fan-out is unbounded.** Code size explodes, build times balloon, and
instruction cache thrash gives back everything the specialization won.
→ *Fan over reachable tuples; unify cells that lower identically.* (§5.3)

**If backends link LLVM's artifact instead of re-emitting its algorithm.** The GPU
path silently drops off the coverage claim, and "100%" quietly means "100% on CPU."
→ *Defer to the algorithm, not the artifact.* (§5.5)

**If capability requirements are unchecked.** `sort` on complex, Gaussian
elimination over `ℤ/6ℤ`, an associativity-assuming solver on octonions — each
produces confident wrong numbers.
→ *Operations declare what they require; encodings declare what they satisfy;
mismatches fail at declaration time.* (§3.3)

The through-line: **this tree's recurring defect is silent degradation.**
`to_dtype_` defaulting to float32, `real()` dropping gradients, `eigh` returning the
bare diagonal, complex `sum` succeeding while complex `add` raises,
`unravel_index_` discarding all but element 0, `ADDING_AN_OPERATOR.md`'s own warning
that a missing table entry fails quietly. The matrix is the highest-stakes place yet
to repeat it, because a silently-generic cell is invisible: correct, plausible, and
thirty times slow.

---

## 7. What must be fixed first

Independently correct, small, and load-bearing under every version of this design.

1. ~~**`AT.real` / `AT.imag` must record autograd nodes**~~ **Done** (`aff20f1`).
   Both now call `_pre_autograd`; `AbstractTensor.complex(re, im)` added as the
   inverse constructor so their backward rules have something to build a
   complex-valued gradient with. Non-Wirtinger convention: gradient of `real(x)`
   lands entirely in the real component, and vice versa.
2. ~~**`atan2` and `round` must exist.**~~ **Done** (`aff20f1`), plus `floor`.
   Still missing: `pow`, `scatter_add`, `fold`. Fixing `round`'s wrapper exposed
   a latent bug in the existing `round_(self, n=None)` hook — `np.round(data,
   None)` crashes, and nothing had ever called it without an explicit `n`
   before, because no public `.round()` existed to call it that way.
3. ~~**`to_dtype_` must fail loudly**~~ **Done** (`aff20f1`). Tries the alias
   table, then the existing torch/numpy normaliser (accepts torch dtype objects
   and native spellings like `"complex64"`), then raises `ValueError`.
4. ~~**`.dtype` must give one coherent answer**~~ **Done** (`aff20f1`).
   `_numpy_dtype_to_torch` now falls back to the numpy dtype itself instead of
   `None` for anything torch has no equivalent for — which is why `fft()`'s own
   output could not report its dtype at all.

Verified against numpy: `fft → real → sum` now yields a nonzero gradient
matching `fft → ifft → sum`'s cross-check; `atan2` gradients match the analytic
`x/(x²+y²)` form; `real`/`imag`/`complex` round-trip correctly through autograd.
100 passed / 1 pre-existing xfail across the fft, DEC/Laplacian, canonical-
backward, and backward-coverage suites (the one failure — a missing `sext`
backward rule — predates this work and is unrelated).

Next, in order: the dtype descriptor (§2), generalizing `Hooks` into a region-bound
operator set as a behaviour-preserving refactor (§5.4), complex as the first real
cell set at rung 2 (§4.1), and the u32 shim with its diagnostics (§4.3).

Precedent for the approach exists in-tree: `AbstractTensor.searchsorted`
(`abstraction.py:1919-1950`) is built from broadcast compares and a sum with no
backend hook, and `unravel_index_` moved from a `NotImplementedError` with seven
backend copies to a single base implementation using only `%` and `//` — after
which every backend, including one that had never implemented it, worked
immediately.

**Define once. Generate the lowerings. Never hand-write a type six times.**

---

## 8. Non-goals and open questions

**Non-goals.** Replacing backend-native types where they exist — rung 1 is a
feature. Making every encoding fast; rung 4 exists to be correct and slow. Bit-exact
float reproduction across backends. **Pushing anything to the u32 shim that a
target can express natively.**

**Open.**

- Does `domain` (angle mod 2π, unit-norm) belong on the dtype, or is it a separate
  constraint system? It is the axis least like the others.
- How do capability declarations interact with autograd? `conj` is antilinear;
  Wirtinger derivatives are the correct treatment for complex gradients, and the
  existing `fft` backward rule does not address this. Compiled cells need registered
  derivatives in `backward_registry` regardless.
- Where do heterogeneous carriers (factoradic, posit) store per-lane metadata — in
  the descriptor, or the value?
- Can `basis` be inferred rather than declared? The circulant-Laplacian case is
  detectable, and detecting it would let a periodic Laplacian dispatch its spectrum
  to `fft` in O(N log N) instead of the Jacobi eigensolver that currently takes
  3.3 s on an 8×8.
- What is the rung-2 synthesis rule for a target whose scalars *almost* fit — f16
  components on an f32-only ISA, say? Widen, or degrade?
- What is the honest fan-width budget, in emitted bytes, before specialization
  starts losing to instruction-cache pressure? This is measurable and unmeasured.

---

## 9. The claim

The tree already contains a provenance-traced minimal-operator computer, a working
operator-set-swap prototype, a broad declared taxonomy of encodings, a working
spectral operator, a metric-tensor Laplace–Beltrami solver, a metric-steered
convolution architecture, an LLVM JIT, and a textual-IR backend one string away
from emitting aggregate and arbitrary-width types.

What it does not contain is the layer where **a number system declares its carrier,
its algebra, and its laws once**, and every (operation × encoding × backend) cell is
then *generated* as native code — delegated where the type exists, composed from
native scalars where the components exist, packed over 32-bit words where nothing
else will land, and NAND-traced to prove all three.

That layer belongs in AbstractTensor, because it is the only place holding both the
base operators to delegate to and the autograd tape and SSA vocabulary the lift must
preserve.

Build the axes, not the list. Generate the cells, don't dispatch them. Ship the
matrix, not the translator. And hold the caveats in §6 — every one of them is the
difference between native code and sludge.
