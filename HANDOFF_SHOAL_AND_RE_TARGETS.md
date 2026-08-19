# Handover: Shoal verified against its oracle, re.compile emission down to one honest refusal

Continues `HANDOFF_WORK_CONTRACT_AND_EMISSION_REDUCTION.md` (whose ledger
still governs). Written 2026-08-19. One session, two named targets: the
fluid flagship — now named **Shoal** — and `re._compile`. Everything under
"measured" was taken this session; nothing quotes a projection as a result.

---

## 1. State of the tree

Branch `codex/recursive-reduction-bridge`, five commits this session:

| commit | what |
|---|---|
| `147d177` | contract routed through the direct lanes: `compile_sympy_equations` runs the shared identity pass; C/WASM private sqrt spellings contract-gated; 4-lane verifications pin `deploy`; pickle contract sidecar; `_copy_literal_payload` (the `_NamedIntConstant` crash) |
| `7a8a78a` | error channels bound to public write-back slots; `differential_translation.py` AGREED for the first time; the flagship named Shoal |
| `a9e69b5` | raise-boundary calls resolve as void native calls, anchored by argument producer |
| `939c23f` | `drop_dead_pure_region_calls` (catalogue §2.2, completeness-motivated); carried inout seeds for under-supplied region feeds |
| (this commit) | declaration-rank guard on the inout seed; Phi-incoming consumer counting in the sweep; this handoff |

### Verified green after every change (the ~40 s gate, run ~8 times)

| check | result |
|---|---|
| `tools/translation_scorecard.py` | 10/10, every run |
| `tests/test_precompile_to_ssa.py` | 34/34 |
| `tests/test_symbolic_fluid_direct_backends.py` | 5/5 (4 original + new exact-contract policy test) |
| `tests/test_symbolic_fluid_native_runtime.py` | passed, `mass_err <= 1e-15`, including one fresh lowering with the new sweep in the pipeline |

## 2. Shoal (the SymPy viscous shallow-water flagship)

Named in `symbolic_fluid_model.py`'s docstring, distinct from the NumPy
bath/SPH engines.

* **`tools/differential_translation.py`: AGREED on every observable — the
  first time.** The last divergence (`error_channels['tracer_bounds']`
  native 0.0 vs oracle 0.0324) was the binder reading channel accumulators
  through the soft `_read` path: their ids are internal allocas with no
  public buffer, and unobservable silently became 0.0 — the exact trap
  `_read`'s own docstring records. Channels now bind to the `last_*` state
  write-back slots (public ABI, oracle-verified) and refuse loudly when
  unobservable.
* The old `tools/HANDOFF_fluid_c_shell.md` defect list is STALE: the
  one-element copy was fixed at `5764d3c`, the neighbour mis-feed is gone
  (state fields all match the oracle).
* **The C-shell executable chain builds and runs with today's compiler**:
  `build/sfdc-shoal` (45 functions, all entry points) →
  `tools/build_fluid_c_shell.py` → `build/shoal-c-shell/
  symbolic_fluid_frame_shell.exe`, one frame in ~83 ms shell time.
* **Measured, open: the compiled dt controller's trajectory.** Exe vs
  (Python `run_superstep` + oracle-verified native advance) on identical
  32×32 state: conserved sums agree to ~5e-13 (height
  1027.0883086729941 vs …946; tracer equal to 2e-14), violations both
  zero — but `dt_next` differs (exe 0.016667 vs oracle 0.006518) and
  wave speed in the 4th digit (1.06069 vs 1.06091). The fully-compiled
  controller takes a different substep path. Frame-level differential
  (DIFFERENTIAL_PHASES) is the right instrument; do not guess.
* Open question: emitting `symbolic_fluid_frame` via the LLVM lane with
  hand-bound feeds did not advance state (outputs stayed initial,
  `dt_next` garbage). Unresolved whether the harness feeds or the lane;
  the C-shell exe with the same pickle DOES advance.

## 3. The work contract now governs the direct lanes

* `compile_sympy_equations` runs `reduce_constant_exponent_pow`, so the
  direct scalar lanes inherit the shared, contract-governed pass.
* The C and WASM emitters' private Pow tables carried the INEXACT set
  unconditionally — silently violating `prove`/`develop`. Now: exact
  spellings always; sqrt-family only under `inexact_identities`; C falls
  back to faithful `pow()`; scalar WASM refuses honestly (it has no pow
  instruction). New test:
  `test_exact_contracts_forbid_the_private_sqrt_spellings`.
* The 4-lane verifications (tests + `examples/build_symbolic_fluid_backends.py`)
  pin `deploy` — the preset whose documented meaning is exactly
  "inexact set, stable across hosts" — so all lanes receive identical
  reduced SSA from one pass.
* `control_repository_ssa.pkl` gains a `.contract.json` sidecar
  (identity policy at lowering time); `_cache_is_stale` refuses a cache
  lowered under a different policy. Mtime alone cannot see an
  environment change.

## 4. re.compile: three walls fell, one honest refusal stands

Walk of the walls, in order, all measured with `tools/compile_re_probe.py`:

1. **Lowering crash** (`deepcopy` of `re._constants._NamedIntConstant`,
   which sets `__reduce__ = None`): fixed by `_copy_literal_payload` —
   share the reference when copy is impossible; an immutable int subclass
   loses nothing. Three raw-deepcopy sites converted.
2. **Two unresolved raise-boundary calls** (`Tokenizer.error` returns an
   exception object its lowering cannot materialize — the construction is
   a recorded structural shortfall — and the callers never consume the
   bound result). Normalized to void native calls: authored execution
   (tell(), offset arithmetic) still runs; the dead binding is dropped;
   the callsite is tagged in `raise_boundary_callsites`. In-loop
   insertion anchors after the argument's own producer. The ABORT
   semantics of raise remain a declared gap — a future contract axis
   (Fortran `error stop` is the natural spelling).
3. **The materialized comprehension `range`** (`_mk_bitmap`'s
   `range(len(s), 0, -_CODEBITS)` carved into its own pure region whose
   projections nothing reads): `drop_dead_pure_region_calls` — catalogue
   §2.2's first inhabitant, completeness-motivated. Conservative: pure
   callee body, group-internal-only consumption, projections unconsumed
   and outside declared outputs, carried ports, and Phi-incoming records;
   a planner region losing its last call site leaves the function table
   (backend reachability then drops it). Swept 28 dead regions from re's
   closure (59 → 31 functions); every carried-value scorecard journey and
   the fluid mass gate held.

**Current state: `emitted.complete: False` with exactly ONE shortfall**,
and it is a refusal, not a gap:

* `_optimize_charset`'s loop_exit region call carries an inout pair whose
  ids (28/29) the caller declares as DYNAMIC-EXTENT ARRAYS
  (`real(c_double) :: t28(extent_dynamic_…_28_1)` — sequence storage)
  while the region's own record declares them float64 SCALARS. Regions
  share the caller's value space, so this is one id with two rank
  identities. The emitter now: seeds an under-supplied carried inout feed
  from the recorded `source_value_id` ONLY when declaration ranks agree
  (the typed view alone misses dynamic extents — measured: it called
  both sides scalar and gfortran caught the lie); on mismatch it refuses.
  The fix belongs at the planner/lowering level: decide which rank is
  true for 28/29 and make both views say it.
* Behind that wall, gfortran on the (briefly) complete emission showed
  the remaining defect surface: 7 rank-0-vs-rank-1 latch assignments
  (`t298/t299/t300 = t130` — Phi declarations losing rank against array
  incomings, the KNOWN baselined Fortran-lane family) and one dangling
  operand (`t330 = t153`: a Phi consumes 153; NOTHING produces it —
  pre-existing, not the sweep: the Phi consumes 153 as a plain arg, which
  the sweep counts. Discriminating check if doubted: run with the sweep
  disabled and look for a 153 producer).

**Bootstrap gate assessment** (the owner's stated condition: re.compile
without python callables, external native calls as the ground floor):
the lowered closure contains ZERO python-host calls — the single
python-flavored instruction is `builtins.range` as an intrinsic, and the
resolution histogram is all native_call/decomposed. The gate is now
blocked only by the rank seam above and the baselined Fortran rank
family, not by any Python dependency.

## 5. Ledger additions (on top of the prior document's §3)

1. The carried-inout rank seam (one id, two rank identities across the
   region boundary) — the single blocker for complete re emission.
2. The Fortran Phi-rank family: latch copies of array incomings into
   scalar-declared phi variables. Same root as the 8 baselined failures.
3. The compiled dt controller's trajectory divergence (Shoal §2).
4. The declared raise-abort gap (`raise_boundary_callsites` is the audit
   surface; a contract axis should decide trap vs fallthrough).
5. The LLVM-frame harness question (Shoal §2, last bullet).
6. Dead-projection sweep follow-ups: it currently keys on
   `result_convention == "ssa.aggregate"` only; statement-form region
   calls (res=None) are untouched by design.

## 6. Blocker isolation (second pass, all measured)

Each open blocker was driven to an isolated, instrumented statement so it
can be iterated on independently:

* **Rank seam (re, the one emission refusal).** At the SSA level BOTH
  sides declare 28/29 as `float64 ()` scalars; the array-ness comes from
  the sequence table (`SSASequenceDescriptor(sequence_id=28,
  column_value_ids=(28,), length_address_id=312, …)`) and the region
  formal's own `sequence_arena` accounting. Region_5's body is
  `Add(28, 29)` feeding `Eq(195, 256)` — it consumes sequence ARENAS as
  scalar Add operands. The seam is therefore a region-carving
  mis-lowering: a sequence-state read (plausibly a length) carved as a
  direct arena read. Iterate at the carving/planning layer, not the
  emitter.
* **Fortran phi-rank family.** The well-formedness sweep found ZERO
  phi-rank mismatches at the SSA level across all 31 functions — the
  `t298/t299/t300 = t130` gfortran errors are purely an EMISSION-layer
  declaration defect: the emitter's phi typing declares scalars while its
  own extent inference declares `t130` an array. Iterate inside
  `ssa_fortran_backend` phi/local declaration typing, in isolation from
  the planner.
* **Dangling operand.** Exactly one in the whole closure:
  `_optimize_charset` `while_header.1`'s Phi consumes 153, produced
  nowhere. Pre-existing (the Phi consumes it as a plain arg, which the
  dead-region sweep counts as a consumer — the sweep cannot have removed
  its producer).
* **Raise-boundary audit.** Exactly two tagged callsites, both to the
  same specialized error shell: `_compile@416` and `_compile_charset@52`.
* **Shoal dt trajectory.** The oracle holds `dt = dt_initial = 0.001`
  for all 34 intra-frame substeps (a final 0.000333 step lands the frame;
  growth appears only in the post-frame `dt_next = 0.006518`); the exe's
  `t14` equals `frame_duration/2` to 1e-12. The compiled controller
  therefore GROWS dt within the frame where the Python controller holds
  it — a control-flow/ordering divergence in `run_superstep`'s compiled
  lowering (dt-update vs frame-remaining check), not numeric drift.
* **LLVM whole-frame lane.** `tools/run_frame_native.py` (the repo's own
  harness, its own feeds) also shows no advance: `last_wave_speed` reads
  0.0 where uniform-height physics under gravity 9.81 writes ~3.13 in one
  substep. Real lane defect, isolated to "the compiled frame's superstep
  loop body never executes"; next instrument is `watch=` on the loop
  iteration counter at `emit_ssa_function_to_llvm`.

## 7. Working rules, re-earned this session

* The soft-read trap is real and it recurs: an unobservable id read as
  0.0 cost the tracer_bounds divergence. `required=True` unless a
  default is semantically meant.
* A green "same numbers" comparison can be vacuous: conserved sums match
  whether or not the state advanced. Compare a quantity the defect can
  actually move (first cell, wave speed, dt) before declaring agreement.
* The typed view and the declaration view of a value can disagree
  (dynamic extents); a guard on the wrong authority passes and the
  compiler downstream catches the lie. Guard on the authority the
  emitter actually declares with.
* gfortran is a measurement instrument: emitting "complete" source and
  compiling it are different claims (13 → 8 → behind-the-wall errors,
  each round naming the next defect class precisely).
