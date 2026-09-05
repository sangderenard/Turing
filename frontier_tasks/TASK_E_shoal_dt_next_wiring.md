# Task E — Shoal's dt_next output reads 0.0: find the missing binding link

Read `README.md` in this directory first. Difficulty: medium-high, mostly
because verification is expensive. **This is a diagnosis-first task**: the
deliverable is the exact binding chain for one value, with the missing link
named. A fix is optional and budgeted.

## Hard budget

A full rebuild (`tools/build_fluid_c_shell.py`) re-lowers the whole fluid
program and runs gfortran: **~8 minutes**. You get **at most 3 rebuilds**
this session. Everything else must come from reading the pickle and the
generated Fortran, which is free.

## The symptom, and what is already ruled out

The compiled fluid frame (`symbolic_fluid_frame`) returns, among its
outputs, `dt_next` — the controller's proposed next timestep. History of
this one value:

* Old compiler (conditionals silently flattened): the exe published
  `t14 = 0.0166667`, provably the never-taken rejection-arm value.
* Current compiler (conditionals real, commits `d50ba56`..`6a7bc70`): the
  exe computes ORACLE-EXACT physics (height sum matches the oracle to
  every printed digit) but **`t14` reads 0.0**. The oracle's dt_next for
  this frame is `0.006518`.

So the wrong-arm defect is dead; what remains is that the output slot is
**never written** — the soft-read rule ("a value you can't observe is not
zero", `HANDOFF_SHOAL_AND_RE_TARGETS.md` §7) showing up as a 0-filled
slot. Also note: the exe currently crashes OUTSIDE a debugger from an
unrelated out-of-bounds read (§6f) — run it under
`cdb -g -G .\symbolic_fluid_frame_shell.exe 1` (debug heap masks the OOB)
to see the output JSON; `t14` is in it.

## Free evidence (no rebuild needed)

The current pickle: `build/sfdc-condfix/control_repository_ssa.pkl`,
readable text beside it (`control_repository_ssa.txt`). Load it:

```python
import pickle, sys
sys.path.insert(0, r"C:\dev\Powershell\turing")
with open(r"build/sfdc-condfix/control_repository_ssa.pkl", "rb") as fh:
    module, outputs, exports = pickle.load(fh)
frame = next(fn for n, fn in module.functions.items() if n.endswith("symbolic_fluid_frame"))
```

Known facts about the chain, measured on the OLD pickle
(`build/sfdc-shoal3/…txt`, still present) — the healthy-era wiring you are
comparing against:

* frame: `t14 = Load(...)` with accounting
  `ssa_call_result_from: (run_superstep_specialized, 240)` — i.e. t14 is
  the frame's read of run_superstep's value 240;
* run_superstep: `Ret(..., 240, ...)` where
  `240 = cast_like(130, 5)` (`_restore_type(last_dt_next, ref_dt)`) and
  `130` is `step_with_dt_control_used`'s output 445 (its dt_next).

Your first job: walk the SAME chain in the NEW pickle and find where it
breaks. Candidate break points, in likelihood order:

1. Inside `run_superstep`'s new lowering, `last_dt_next` is now a value
   merged through the surviving conditionals (`dt_cap`/`last_dt_next`
   updates inside `if`/`elif` arms → conditional-carried Phis). Check
   whether the value the `Ret` names is the MERGE (Phi) or a stale
   pre-conditional identity. The while-arc fixes (`git show 86fb257
   3ee307b`) fixed exactly this shape for loop phis in
   `recover_structural_source_outputs` / the Ret-filtering pass in
   `src/compiler/fortran_c_shell.py`; conditional-carried phis
   (`binding: 'conditional_carried'` attribute) may need the same
   treatment there.
2. The frame's named output: `frame.metadata["named_outputs"]` /
   `outputs` (the pickle's second element) — check which value id the
   frame's dt_next output slot names, and whether anything in the frame
   writes it.
3. The C shell feeds `t14` back as next frame's `dt_initial`
   (`state_feedback={"dt_initial": "t14"}` in
   `tools/build_fluid_c_shell.py`) — irrelevant to WHY t14 is unwritten,
   but explains why nobody noticed at one frame.

## Definition of done

* Minimum (no rebuild needed): `## FINDINGS` here with the full chain —
  frame slot ← frame instruction ← run_superstep Ret operand ←
  producer — and the first link that is absent or names a value nothing
  writes. Quote the actual instructions from the new pickle.
* Stretch (uses your rebuild budget): the fix, then
  `timeout 600 python tools/build_fluid_c_shell.py build/sfdc-condfix/control_repository_ssa.pkl build/shoal-c-shell-condfix`
  and run under cdb as above. Success = `t14` in the output JSON is a
  plausible dt (0 < t14 <= 1/30; the oracle says 0.006518) while
  `state.height.sum` stays exactly `1027.0883086729941`. Then the gate
  (README rule 3, 68 passed), then commit.
* If the chain is intact in the pickle and the zero appears only in the
  emitted Fortran, say so — that relocates the defect to the emitter and
  is a complete, valuable finding on its own.

## FINDINGS 2026-08-19

**None of the three candidate break points in this file is the actual
cause.** No rebuild was used — everything below came from the existing
pickle (`build/sfdc-condfix/control_repository_ssa.pkl`) and the
existing generated Fortran already on disk from earlier this session
(`build/shoal-c-shell-condfix/symbolic_fluid_frame_shell.f90`), both
predating today's Task A/B/C work (none of which touched the fluid/dt
lowering path).

**The chain traced link by link, all confirmed intact:**

1. Frame: `t14 = Load(...) {aggregate_index: 1, source_output_id: 14}`
   unpacking `run_superstep`'s aggregate call result at index 1 — exactly
   the healthy pattern.
2. `run_superstep`'s own `Ret(t239, t240, t378, ...)` — position 1 is
   `t240` — matches.
3. `t240`'s producer: `t240 = cast_like(t130, t5)` in block `while_exit`
   — matches `_restore_type(last_dt_next, ref_dt)` exactly.
4. `t130`'s producer: `t130 = Load(t480) {aggregate_index: 11,
   source_output_id: 130}` in block `loop_exit`, unpacking
   `step_with_dt_control_used`'s own 13-element aggregate result at
   index 11 — `step_with_dt_control_used`'s own `dt_next`, correctly
   wired.
5. The generated Fortran's call signature confirms the SAME positional
   mapping end to end: the frame passes exactly two output slots
   (`t13, t14` — the last 2 of 369 arguments) to `run_superstep`, whose
   own dummy-argument list has `t239, t240` (matching names, matching
   position) as its last two declared `intent(out)` parameters.

**Where it actually breaks — an SSA identity collision, not a stale or
missing binding.** Checked whether value 130 is *also* one of
`run_superstep`'s own formal arguments (`130 in [int(a.id) for a in
run_superstep.args]`): **True**. Counted every instruction in the whole
function that produces `t130` as a result: **exactly one** — the
`Load` in `loop_exit` above. So value 130 is simultaneously:

* a formal/carried parameter of `run_superstep` — the emitted Fortran
  declares it `real(c_double), intent(inout) :: t130`, meaning the
  backend believes this value arrives valid from the caller, and
* the SSA result of a body instruction (the Load above) that is
  supposed to freshly compute it from `step_with_dt_control_used`'s
  own return.

A value cannot legitimately be both — this is the exact malformation
`939c23f`'s own code comment names and declines to repair blind: *"The
call-site's own AST node id coincides with a value some OTHER,
already-existing instruction already produces... Adopting it here
would give two different instructions the same SSA identity, which is
exactly the class of bug the freshening pass later in this function
cannot safely repair."* This is a concrete, previously-unnamed instance
of that exact class, landing on `run_superstep`'s own dt_next.

**Confirmed in the emitted Fortran, not just inferred from the pickle.**
Searched the entire ~36 KB subroutine body for every occurrence of
`t130`: **three**, total — its `intent(inout)` declaration, its use as
the *input* to the `cast_like`-emitting call that produces `t240`, and
its appearance somewhere inside the giant `step_with_dt_control_used`
call's own argument list (fed as an INPUT to that call, consistent with
"inout seed", never received as an output there). **There is no
assignment statement to `t130` anywhere in the body.** The Fortran
backend, having decided 130 is an inbound `intent(inout)` formal,
never emits code for the Load instruction that was supposed to write
the loop's own freshly-computed value into it — so `t130` silently
retains whatever the CALLER seeded it with (plausibly zero on first
use), which is exactly the observed symptom: `t14` (fed transitively
by t130) reads `0.0`.

**What the real fix needs**: find where value 130 was chosen as
`run_superstep`'s carried-inout-seed identity — the mechanism `939c23f`
added (`ssa_fortran_backend.py`'s `_region_call`, "seeds the inout
dummy from the caller value the projection itself records
(source_value_id)") is the most likely origin, since it's exactly the
kind of pass that invents/reuses an id without checking whether the
SAME function independently produces that id elsewhere via an ordinary
instruction. The fix is either (a) freshen the Load's own result to a
new, non-colliding id wherever loop-body/region lowering assigns it, or
(b) make the inout-seed mechanism check `produced_ids` (a body
instruction's own result) before adopting an existing value id, and
choose a fresh one on collision — the same freshening discipline
`939c23f`'s OWN comment describes doing for a different collision
class, just not extended to cover this one. Not attempted here — the
inout-seed machinery is delicate (Task B's own regression this session
is a cautionary tale for touching identity-resolution code without a
very deep dive), and a wrong fix risks breaking the many OTHER
legitimately-seeded inout carries elsewhere in this same function
(`step_with_dt_control_used`'s huge argument list has many more).

No rebuild spent (all 3 remain available); no code changed. This is a
complete, precisely-evidenced root cause — a specific, previously-
unnamed instance of an already-documented defect class, not a new
mystery.
