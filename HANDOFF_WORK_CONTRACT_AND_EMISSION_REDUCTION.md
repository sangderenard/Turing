# Handover: the work contract exists, the slots learned to stay in registers

Continues `HANDOFF_SSA_IDENTITIES_AND_DEPLOYMENT.md` (whose P0–P5 frame still
governs; this document records what one session realized from it and what it
added to the ledger). Written 2026-08-19. Everything under "measured" was
taken this session and is reproducible with `tools/bench_native_step.py` or a
static read of an emitted artifact; nothing below quotes a projection as a
result.

---

## 1. State of the tree

Branch `codex/recursive-reduction-bridge`, four commits this session, nothing
uncommitted:

| commit | what |
|---|---|
| `13c14e3` | `ir_identities.py` — constant-exponent `Pow` reduction, wired at the `IRModule` finalization point in `fortran_c_shell`; FMA audited to its real blocker |
| `d6e09e8` | slot-keyed same-block register cache in `_emit_repository_call_module`; 8 pre-existing `test_llvm_repository_ssa.py` failures baselined |
| `4dfd575` | `work_contract.py` — the four presets (prove / develop / deploy / fast) |
| `a3be51b` | contract extended across the audited policy surface (flags, epsilon, embedded extraction, declared-refusing axes) |

### Verified green, all at the default (`develop`) contract

| check | result |
|---|---|
| `tools/translation_scorecard.py` | 10/10, max disagreement 0.0e+00 |
| `tests/test_precompile_to_ssa.py` | 34/34 |
| `tests/test_symbolic_fluid_native_runtime.py` | passed; `mass_err <= 1e-15` held under every preset measured |
| `tests/test_ssa_llvm_backend.py`, `test_llvm_signal_math.py` | passed |
| contract resolution and refusal | unit-checked (presets, overrides, pinning, unknown names refuse) |

### The performance ledger (fluid flagship, ns/cell, measured)

| state | ns/cell |
|---|---|
| baseline at `af00599` | 1683 |
| + exact `Pow` reduction (default) | ~900 |
| + inexact set (`deploy`) | ~480 |
| + slot-keyed register cache (default = `develop`) | **~280** |
| `deploy` with cache | ~150–195 |
| `fast` (adds contraction + host target; 17 `vfmadd` form) | ~140–165 |
| `prove` (slot-faithful, cache off — the equality-proving shape) | ~980, restores the 817-load artifact exactly |

Emitted loads on the kernel: 817 faithful → 222 reduced. Stores are
untouched under every preset: pooled in-place slot semantics, the ABI,
`watch`/`history` all read truthful storage. There is no heap allocation in
the hot path and never was — slots are entry allocas, public buffers bind
once.

### Corrections to the prior document's claims

* The FMA dependency chain read "P2 `noalias` → forwarding → chains →
  contraction". **Superseded:** the chains were recovered by construction
  (the emitter's own register cache), not by aliasing analysis. `noalias`
  remains valuable for VECTORIZATION, not for contraction.
* `tests/test_llvm_repository_ssa.py` carries 8 failures that are
  **pre-existing at `af00599`** (confirmed in a clean worktree), are
  Fortran-lane emission defects despite the file's name, and are **not
  ordering-dependent** (two fail identically in isolation — the
  ordering-assumption concern was checked and cleared). Now in
  `KNOWN_FAILING_AT_AF00599`.

---

## 2. What the contract is, in one paragraph

`src/compiler/work_contract.py`. Every switch that answers "how faithful must
this artifact be, and to whom?" lives in one frozen dataclass with four
internally consistent presets, selected by `TURING_WORK_CONTRACT`, pinnable
with `set_active_contract`, defaulting to `develop`. Legacy variables
(`TURING_POW_INEXACT`, `TURING_FMA_CONTRACT`) stay honored as single-field
overrides so recorded measurements keep their meaning. The doctrine
throughout is `fusion_levels`': a contract naming behavior no layer honors
**refuses at construction** rather than silently compiling something else.

Wired fields (each reaches a real consumer): `register_reuse`,
`inexact_identities`, `contract_multiply_add`, `compiler`/`compiler_flags`
(both zig cc sites), `resolver_epsilon` (`bounded_constants`), `extraction`
(the embedded `ExtractionContract`; per-call argument wins; `None` from both
preserves the historical gate-disabled behavior — hazard on record from
2026-08-17).

---

## 3. The unrealized ledger

Everything raised this session (or inherited) that is NOT done, ordered
roughly by how much the rest of the list leans on it.

### 3.1 The facts vocabulary (catalogue §5) — still the linchpin, still unbuilt

`accounting["facts"]` with the closed vocabulary `finite / positive /
nonzero / integral`, asserted only where proven, propagated through
identities, consumed by rewrites that refuse to fire without their premise.
Nothing this session needed it — the exact set needs no permission and the
inexact set shipped as opt-in policy — but it is what would make
`pow(x,0.5)→sqrt` **bit-exact** for the clamped-positive `state.height`,
collapsing the `develop`/`deploy` distinction for that case from policy to
proof. Design it before more identities accumulate that assume it cannot
exist.

### 3.2 Per-region heuristic fusion/faithfulness — the idea is recorded, nothing implements it

The contract is currently one broad policy per process. The stated better
shape: a PER-REGION level — faithful slots where a region's values are
watched, carried, or cross the ABI; reduce freely where the memory program
proves the interior private. The slot cache is this heuristic at emission
grain only. Prerequisites that are themselves unrealized:

* `fusion_levels.PRESERVE` and `NO_FUSION` are declared and RAISE; only
  REGIONS/FUSED are honored (via `precompile_only`).
* Some fusion levels are not wired to control-code isolation yet.
* The fusion level is not yet a `WorkContract` field — fold it in when the
  above stops raising, not before.

### 3.3 Deployment / threading (P3) — untouched

`deployment="serial"` is the only honored value; `turing_pool.c` exists,
unwired. The prior document's P3 plan stands verbatim: a dependence test
independent of unrolling + associative-reduction recognition, landing
together (the fluid loop's five accumulators are all associative
reductions; fixing the unrolling coupling alone re-rejects the loop on the
carried-bindings conjunct). `ControlDeploymentRegion` still has exactly one
construction site, inside `evaporate_unrolled_loops`.

### 3.4 Remaining identity-catalogue items

| item | status |
|---|---|
| trivial identities (§3: `x*1`, `x+0`, dedup, folding) | not built; half need §3.1's facts to be safe |
| CSE (§2.3) | not built; `quotient_common_subexpressions` exists in the decompilation lane, STILL UNREAD — read it before writing one |
| dead value elimination (§2.2) | not built; the aggregate-unpack dead loads persist (222 emitted loads still include them) |
| index modulo → compare-and-wrap (§2.4) | not built; NOW measurable — it was invisible under 1683 ns and is not under 150 |
| the two dormant simplifiers' disposition | still undecided: `aggressively_simplify_expression` (zero callers) and the CSE above |
| SSA-level `MulAdd` op | not built; per-backend expressibility recorded in §2.5 (LLVM/C/SPIR-V/WGSL yes; scalar WASM and Fortran no); must stay contract-gated |

### 3.5 P2 remainder — declare what the artifact knows

`-march=native` ships only under `fast`/contraction. Still absent: `target
triple` and `target datalayout` in the emitted module, `noalias` derived
from the carried-slot metadata (now motivated by vectorization rather than
FMA), and routing the AOT path through `llvm_optimizing_pipeline`. Blanket
`noalias` remains a lie; derived-only remains the rule.

### 3.6 Contract axes declared but refusing, with their unlock conditions

| field | honored today | unlocks when |
|---|---|---|
| `deployment` | `"serial"` | P3 wires the pools |
| `destination` | `"native"` | `source_language`/`shell_language` route through the contract instead of per-call parameters |
| `constant_arguments` | empty only | an argument-baking specializer exists; until then a non-empty list refuses |
| syscall routing | not a field, named horizon | native artifacts' OS interactions described and redirected through the contract rather than linked ambiently |

`symbolic_arguments` IS accepted today: the precise must-remain-symbolic
veto list (the fluid's runtime extents are the canonical members). It is
vacuously honored — nothing bakes arguments — and every future specializer
must treat it as a veto, which makes it the one contract field whose
enforcement is a promise about code not yet written. Test it the day a
specializer appears.

### 3.7 Store-evaporating emission — explicitly deferred

The cache evaporates LOADS only. A mode that also evaporates stores of
purely private, unwatched, uncarried values would shrink the artifact
further, and is REQUIRED (by the contract's docstring, deliberately) to
consult the watch set through the contract, not grow a new flag. Do not
build it before the per-region visibility classification (3.2) exists to
answer "private to whom".

### 3.8 Numerics follow-ups

* The `deploy` mass-error verdict is "within the flagship's own 1e-15
  gate". A finer instrument (bit-level delta against `prove` on the same
  states) was not taken; the gate held, the exact delta is unrecorded.
* `fast` surrenders cross-machine bit-stability by design (fma availability
  differs by CPU). Cross-backend verification under `fast` is therefore
  expected to disagree at the ulp level with every non-contracting backend
  — nobody has yet written down which comparisons that invalidates.

### 3.9 The 8 baselined Fortran-lane failures

Rank-mismatched array references and non-LOGICAL `merge` masks in
`ssa_fortran_backend` emission, pre-existing, now skipped by default. They
are a real defect ledger for the Fortran lane, not noise; nothing this
session touched their cause.

### 3.10 Inherited, still standing

P4 (Eigen/C++ lane: **read nodus' Eigen translation before designing the
lane ordering** — still unread), P5 (the backward arc), and the prior
document's three decisions, of which decision 1 (numerics policy) is now
substantially answered by the preset ladder + the (future) facts vocabulary,
decision 2 (`noalias` derivation source) is open and re-motivated by
vectorization, and decision 3 (absorption scope for unrolled-loop region
minting) is untouched.

---

## 4. Working rules, unchanged and re-earned

* `TEST_BASELINE_AND_HAZARDS.md` first; single files with an external
  `timeout`; a red run means you broke something.
* Baseline with `git worktree add`, never stash — used again this session
  to clear the `test_llvm_repository_ssa` question in one run.
* Separate measured from inferred. This session's instance: the FMA
  "blocked on noalias" conclusion was measured-but-incomplete — the blocker
  was real, the assumed remedy (P2) was not the only one, and the cheaper
  remedy (emission-time reuse) was sitting in the emitter's own naming
  scheme. When a conclusion says "X is the only path", check whether X is
  merely the only path THROUGH THE LAYER YOU MEASURED.
* Env-gated behavior belongs in the contract, not in new flags. The two
  legacy variables are grandfathered as overrides; do not add a third.
