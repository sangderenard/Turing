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
