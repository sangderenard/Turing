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

## 6b. Second iteration: three of the isolated blockers FELL (same day)

* **Rank seam: FIXED at the root.** Structural specialization stamps
  re's ``out``/``tail`` as ``Constant []`` nodes (their sequence
  descriptors live on under the same value ids 28/29); the sequence
  recognizers saw two constants and ``out += tail`` fell through to
  arithmetic carving. ``_sequence_append_slice_ops`` now recognizes the
  whole-sequence extend with a sequence-like test admitting a Constant
  whose payload is itself a list/bytes/bytearray, lowered through the
  existing ``append_slice`` helper with clipped whole-source bounds.
  Emission complete WITH authored sequence semantics.
* **Phi-rank family: FIXED.** ``_propagate_phi_dynamic_ranks`` carries
  ``dynamic_array_ranks`` rank through Phi chains and aliases the SAME
  extent symbols; the latch copies became legal whole-array
  assignments. gfortran: 6 errors -> 1.
* **Dangling operands: now a loud emission shortfall** (per function and
  block), zero false positives across the fluid and journey programs.
* **The single remaining re wall, precisely:** ``_optimize_charset``'s
  ``while.1`` is the authored charmap-compress ``while True`` loop
  lowered to a HUSK — empty body, condition ``Phi(True, True)``
  (unconditionally true), and carried value 153 whose producer was never
  materialized. This is the elided-loop-body family (commits
  ``f07934d``/``a925a6e`` fought it in the fluid program); the body's
  ``charmap.find``/conditional-``break``/``runs.append`` content never
  lowered. Fix belongs at loop-body lowering, not emission.

## 6c. The symbol-coordination ladder (scorecard levels 10-18), measured

The diagnostic survey concluded three things: of ~12 diagnostic tools only
the scorecard, `trace_fortran_alias.py`, and `diagnose_translation.py`'s
five stage functions are generic (the rest are fluid-bound scripts around
generic machinery — `watch=`/`history=`, `trace_manifest`, the trace ring,
`SSAReferenceEvaluator` are all reusable); the materializer refuses
multi-block control, so the equivalence ladder was architecturally blind
to conditionals; and the coverage cliff falls exactly on SYMBOL
COORDINATION — keyword/default/variadic arguments, parameter order,
shadowing, closures — the constructs both debugging documents name as the
recurring defect class.

The scorecard corpus was extended accordingly (its own guard demands
extension, not trimming). Levels 10-18, probes anti-symmetric wherever
possible so any slot swap changes the value. Measured 2026-08-19 and
pinned in `tests/test_translation_scorecard.py`:

| L | rung | measured |
|---|---|---|
| 10 | keyword call, order swapped | **PASSED** |
| 11 | default argument, used and overridden | **PASSED** |
| 12 | parameter order against the alphabet | **PASSED** |
| 13 | same helper, arguments transposed | **PASSED** |
| 14 | a parameter name rebound three times | MATERIALIZE — does not render |
| 15 | mixed int/float arithmetic | **PASSED** |
| 16 | authored ``if`` | MATERIALIZE — the materializer CRASHES (`NoneType .id`), a defect in its refusal path |
| 17 | ``while`` loop | **EQUIVALENT — runs and is wrong**: returns its scalar twice, `(0.5, 0.5)` for `0.5` |
| 18 | ``any()`` over a generator predicate | MATERIALIZE — predicate region unrendered |

Level 17 is the headline: the first `while` rung ever scored is a live
member of the silent-failure class the scorecard exists to catch — it
lowers with no shortfall, materializes, executes, and publishes its
carried scalar twice.

Tool pairing for the four stalls: L14/L16/L18 live at the
materializer/lowering seam (`diagnose_translation` stages 1-2 generalized,
`dump_comprehension_graph.py` for L18); L17 is a runs-wrong equivalence
defect — read the materialized Python directly, then `watch=` on the loop
exit if the duplication is upstream of materialization.

## 6d. Third iteration: level 17 fell twice, and the fix reached Shoal

* **While double-publication FIXED** (`7d1fd43`'s predecessor + `3ee307b`):
  the structural recovery pass re-published a named output under its
  stale pre-loop identity because ``published`` was keyed by raw graph
  id and could not see the carried phi. The recovery still runs in full
  (its boolop chains and values-map entries feed later source-linked
  calls — the first, skip-the-recovery version of this fix broke
  Shoal's dt-control call resolution and was reworked); only the
  duplicate Ret argument is filtered.
* **While latch guard FIXED** (`7d1fd43`): the latch evaluated the
  next-iteration guard on the header phi (pre-update value), lagging one
  iteration. Carried names now rebind to their updated values around the
  latch's condition lowering, restored to the phi after. **Level 17
  PASSES — the first authored while to compile and compute correctly.**
* **The fix propagated to Shoal**: a fresh exe's `last_wave_speed`
  matches the oracle to all 16 digits (was off in the 4th) — the
  compiled dt controller's intra-frame trajectory converged. The one
  remaining divergence is the post-frame ``dt_next`` proposal
  (`t14 = frame_duration/2` exactly, vs oracle 0.006518) — a narrow,
  isolated path, plausibly another stale-identity binding on the
  frame's return.
* The LLVM whole-frame no-advance is NOT the latch guard (fresh
  post-fix lowering still static) — it stands as its own blocker.
* **The ``dt_next`` remainder, refined to a proof (owner's determinism
  check ran first: no PRNG anywhere in the dt system, the one
  ``perf_counter`` is confined to the never-called realtime mode, and
  the oracle is bit-identical across runs).** The true ``dt_next`` is
  ``min(pi_proposal, metrics.dt_limit)`` with ``dt_limit = 0.013255``,
  so it can NEVER be the exe's ``t14 = 0.0166667`` — t14 is not bound
  to ``dt_next`` at all. The imposter is exactly
  ``shrink * dt_max = 0.5 * (1/30)``: the controller's REJECTION-arm
  dt, never taken in this run (all 34 substeps accept). The frame's
  return binds the rejection branch's value instead of the accepted
  PI proposal — conditional-arm output selection, the same family as
  level 16's materializer crash. Not numerics, not non-determinism.
* Scorecard: **16/19**, remaining stalls are the three MATERIALIZE rungs
  (rebound name, authored if crash, generator predicate).

## 6e. Fourth iteration: ONE root defect owned items 1 AND 2 — and its fix
## moves the whole frontier (2026-08-19, later session)

* **Root cause, proven at the node level.** Every graph-express node is
  born with ``constant=None`` (``ProcessGraph.add_node`` stamps the key
  unconditionally), and ``_constant_value`` treated the KEY'S PRESENCE as
  proof of a literal. So the callsite structural prune in
  ``_fold_callsite_structural_values`` asked "is this if's predicate a
  constant?" about a live ``greater(x, 0)`` node, got literal ``None``,
  took ``bool(None)`` as a static proof of False, aliased the merge Phi
  to the lexically-else arm, and DELETED the then-arm and the ``ast.If``
  node. **Every dynamic conditional in every lowered program was
  silently flattened this way.** Level 16's crash and Shoal's ``t14``
  imposter were two faces of this one defect. (Old ``t14``'s exact path,
  read from ``build/sfdc-shoal3/control_repository_ssa.txt``: frame's
  dt_next <- run_superstep output 240 = ``_restore_type(130 = step_with_
  dt_control_used output 445)`` — correct wiring; the imposter was
  manufactured inside step_with_dt_control_used's flattened arms.)
* **Fixes landed** (commits ``d50ba56``, ``63f0951``): the two
  ``_constant_value`` discriminators (glsl_deployment_strategy,
  shell_reference_tables) now count a ``None`` payload only on declared
  constants; the materializer refuses statement-form calls (res=None) by
  name instead of crashing on ``NoneType .id``; and the four-block
  CondBr diamond is reconstructed as a Python if/else (arm ownership
  decided by each Phi's ``incoming_blocks``, never positionally).
  **Level 16 PASSES — the first authored if to reach EQUIVALENT; 17/19.**
  The ~40s gate is green, including a fresh fluid advance lowering.
* **The same presence-check trap survives, unmeasured, in:**
  ``loop_composer._constant`` (trip counts — downstream isinstance
  guards likely mask it), ``fortran_c_shell`` ``literal_value`` (~3870)
  and default-literal reads (~6067), ``symbolic_equation_compiler``.
  Fix on measurement, not in bulk.
* **Consequence, faced honestly: the prior "almost-there" states of BOTH
  flagships stood on silently deleted authored branches.** The fix
  trades silent wrongness for loud, earlier refusals:
  - **Shoal frame**: the pickle regenerates cleanly, but Fortran emission
    now dies on ``cannot express literal [] in Fortran`` — the
    ``unresolved = []`` seed in ``run_superstep`` (whose consuming arms
    used to be deleted) now crosses the frame->run_superstep call as
    feed argument 19 (value 2481869020677 in
    ``build/sfdc-condfix/control_repository_ssa.txt`` line 5/6). An
    empty-list seed needs SEQUENCE lowering at the feed boundary, not a
    scalar Const. Repro: ``python tools/build_fluid_c_shell.py
    build/sfdc-condfix/control_repository_ssa.pkl
    build/shoal-c-shell-condfix``. The old baseline artifacts
    (``build/sfdc-shoal3``, exe comparisons) remain but are now known to
    describe a program with fabricated straight-line control — do not
    chase dt_next numbers on that ground again.
  - **re.compile**: lowering now refuses earlier — "conditional control
    duplicated scheduled regions in '_compile'" (every region 3..53
    counted 3x after ``overlay_scheduled_control``). Measured overlay
    inputs: 29 conditional programs; the if/elif cascade produces
    strict-subset TAIL programs (each level owns its arm plus the whole
    remaining chain), and several authored ifs produce two IDENTICAL
    programs (equal region sets) that ``known_nesting`` chains into each
    other. Spy: scratchpad ``probe_re_overlay.py`` (monkeypatch of
    ``overlay_scheduled_control`` in a throwaway script; nothing patched
    in-repo).
  The elided-loop-body husk (item 3) and the L14/L18 stalls remain, now
  BEHIND these walls; L14's cause is newly isolated though — see below.
* **Level 14, isolated at the SSA level**: ``sc14__train`` calls
  ``t4 = Call(sc14__helper)(t5)`` where ``t5`` is only produced LATER by
  the region-1 aggregate load — the helper is fed the not-yet-computed
  rebound value; a scheduling/feed mis-bind at planning, and the
  materializer's used-before-produced refusal is honest.

## 6f. Fifth iteration: the consequences carried through — Shoal's frame
## compiles WITH real conditionals and computes oracle physics

Committing to the fold fix meant full rendering, not re-eliding what the
old prune deleted. Three emitter defects stood between the honest frame
program and a runnable exe; all three fell (commit ``6a7bc70``, which
also carries the c_bool fix below):

* **The empty-sequence seed**: ``unresolved = []`` crossing the frame
  call into run_superstep's arena dummy — Const [] now materializes as
  a zero-filled ARRAY local when the DECLARED view (typed view or
  array-base table; the occurrence shape lies) says array; emptiness
  rides the separate length cell. Populated list literals still refuse.
* **The inline guard consulted the wrong authority**: ``_may_inline``
  read only the occurrence shape, so the seed inlined as the bare
  literal ``0`` — INTEGER(4) scalar against a REAL(8) intent(inout)
  array dummy.
* **The numeric->logical coercion had no kind**: ``(x /= 0)`` is default
  LOGICAL(4) against a ``logical(c_bool)`` VALUE dummy; now spelled
  ``logical(x /= 0, kind=c_bool)``.

**Measured state of the new exe** (``build/shoal-c-shell-condfix``, 49
functions — up from 45; the surviving arms are real):

* It **builds clean** and, run under ``cdb`` (whose debug heap pads
  allocations), completes one frame with **oracle-exact physics**:
  ``state.height.sum = 1027.0883086729941`` — every printed digit the
  oracle's (the old exe differed in the tail), ``last_wave_speed =
  1.0609116970111032``, violations 0.
* **The dt_next imposter is DEAD**: ``t14`` no longer reads the
  rejection-arm ``shrink*dt_max``. It now reads **0.0** — an UNWRITTEN
  output slot (the soft-read rule: unobservable is not zero). The frame's
  dt_next publication has to be wired to the surviving conditional's
  merge value; a binding gap, no longer a wrong-arm selection.
* **A real runtime defect remains**: outside the debugger it dies with
  an access violation — ``vcvtsi2sd`` (int64 load feeding an int->double
  conversion) from just past a page boundary: a read past the end of a
  heap INTEGER array, sequence length/key machinery indexing beyond its
  extent. Repro: run the exe plainly (instant crash, before any output);
  instrument: ``cdb -hd -g -G -c "k 20; q" symbolic_fluid_frame_shell.exe
  1`` catches it, plain ``cdb`` (debug heap on) masks it and lets the
  frame finish — which is itself the proof the compute is otherwise
  sound.

* **Owner-suggested instrument for the OOB read, assessed by measurement:
  the reversible machine executor.** ``BinaryMachineProgram.load_pe``
  ingested the real 960 KB gfortran exe in 15 s — **805 functions raised
  through the existing decompiler** — and executed the entry
  transparently, pausing after 25 transitions at the external-target
  base: the CRT startup's first import call, awaiting an external-call
  completion. The instrument is real and would give the bad index full
  provenance with rewind; what it needs first is (a) external-call
  completions for the msvcrt/kernel32 surface the gfortran CRT touches
  at startup, and (b) the derived read-head accelerator for throughput —
  the faulting read is millions of transitions in. Probe:
  scratchpad ``probe_machine_pe.py``. The full readiness/potential report —
  the diagnostic decision tree, the bidirectional read head's tested state,
  what the PE lane can answer that the round-trip Python cannot, and the
  concrete gates before it can chase this OOB read — is
  ``docs/DIAGNOSTIC_DECISION_TREE_AND_MACHINE_READINESS.md``.

## 6g. re's wall, named exactly (measured with a LoopShaderReduction spy)

* Every big loop in re's closure is blocked from control composition by
  ONE cause: ``blockers=('Raise',)`` — ``_compile`` node 431 (51 body
  regions), ``_parse`` 290/1000, ``dis_`` 440, ``_compile_charset`` 60,
  ``getuntil`` 37, ``_parse_flags`` 65/136. The tuple destructuring
  (``for op, av in p``) passed. A blocked loop gets
  ``control_program=None``, its body regions schedule FLAT, and the
  conditional overlay then finds the cascade's markers scattered across
  THREE scopes (top level + two inner loops), inserts one cascade copy
  per scope (``embed``'s insert-once flag is per-SequenceBlock), and the
  duplication guard refuses — correctly fatal, since lowering must not
  fabricate control.
* The loop composer's validated-raise carve-out covers only ``if cond:
  raise``-with-no-orelse; ``_compile``'s raises are cascade-terminal
  ``else: raise`` arms whose effective predicate is a conjunction. The
  principled path: compose raise-carrying loops and let the (now real)
  conditional overlay own the arms, with the raise statement lowering
  into the declared abort gap (Fortran ``error stop`` is the ledger's
  named spelling, behind a contract axis). That is next-session design
  work, not a filter tweak.
* Secondary, real, and needed once loops compose: ``embed``'s
  cross-scope duplication (thread ONE insert-once state through the
  recursion, or refuse when a nested control's markers span scopes).

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
