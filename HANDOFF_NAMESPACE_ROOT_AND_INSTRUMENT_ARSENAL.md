# The namespace root, and the arsenal that found it

Continuation of `FUSION_LEVELS_AND_THE_PYTHON_ROUND_TRIP.md` and
`SSA_STRUCTURE_VOCABULARY_HANDOFF_2026-08-18.md`. Written at the end of the
session following the long weekend after 2026-08-18. This document carries
two things forward: the one named defect standing between the compiler and
its flagship program, and the debugging doctrine that the week's instruments
made possible -- because the doctrine is now the more valuable asset.

## Where the compiler stands

The translation scorecard (`tools/translation_scorecard.py`) is **10/10**:
every corpus journey -- straight-line arithmetic through the Adam-shaped
triple-carry loop and the bare `w = update(w)` round trip -- lowers,
materializes back to Python, executes, and matches the authored program
exactly. Seven structural fixes got it there, each pinned by tests that fail
loudly on regression:

| fix | commit |
|---|---|
| carried parameters materialized pre-snapshot (frozen-value miscompile) | 66cdf6c |
| regions scheduled in dependency order | 5aca5a4 |
| carried slots seeded in the preheader (aliased-IO regulated) | df09057 |
| region callees defaulted; silent region skips refused; plan terminals exported | acf9ca8 |
| the control program gained a CALL STATEMENT (`__plan_callsite_N__`) | f07934d |
| storage-formal ABI declared (LAPACK WORK-array style) and leased at entry | 362eeb5 |
| region carver temporaries seeded above the caller watermark | a1aee9b |

The `test_precompile_to_ssa` suite is 34/34 (was 25/34 for the whole prior
era). The aliased-IO design is now *regulated rather than trusted*: one slot
per carried value, seeded before the loop, read-then-written through calls in
place, declared in metadata where it crosses a signature.

## The one named defect: hierarchy ids published as local ids

The whole-program fluid test (`test_symbolic_fluid_native_runtime.py` -- the
dt system ingested whole by the AOT compiler) is **red**, and the root is
named to one boundary:

A planner region's instruction list arrives on its PlanLines with value ids
minted in the **hierarchy namespace**. The caller consumes the region's
published `output_ids` as **local-namespace** integers. No translation is
applied at the publication boundary, so one integer can mean a scalar
parameter in the caller and a GetElementPtr address inside the region. In
the fluid advance, caller ids 80/89 (`tracer_diffusivity`, `viscosity`)
collide with region-internal address temporaries; the caller reads a height
cell (~1.0) where its 1.0e-4 diffusivity belongs; the physics then
*correctly* reports a bound violation (diffusion number 1.6 is genuinely
unstable), and the controller *correctly* rejects every dt, because the
mis-plumbed value is dt-invariant.

Read that chain again before touching anything: **every layer downstream of
the collision behaves correctly.** The step's arithmetic matches the SymPy
oracle for the inputs it received. The controller's same-reason-at-every-dt
detector did exactly its job. The only defect is one untranslated integer.

**The fix belongs where `region_signatures` / `output_ids` are derived**
(`lower_control_sections_to_ssa` in `precompile_to_ssa.py`): translate
region line ids through the hierarchy-to-local correlation before
publishing. `assign_hierarchy_ids` / `reduce_hierarchy_identities`
(`hierarchical_plan.py`) own that correlation. The carver-temporary variant
of the same collision is already fixed by the watermark (a1aee9b) -- do not
mistake that for the whole fix; it was measured insufficient (region_2's
colliding ids survive it because they are PlanLine ids, not carver
temporaries).

Verification, when fixed: the exact probe sequence in the decision tree's
worked hunt, ending with `test_symbolic_fluid_native_runtime.py` green with
attempts `[(0.2, False), (0.1, True), (0.1, True)]`.

## The doctrine, distilled

The hunt that found this took one sitting. The same class of defect
historically cost days. The difference was not luck; it was a discipline the
instruments now enforce, and it is worth stating as doctrine:

1. **Read the rejection's shape before reading any IR.** dt-independent
   rejection with healthy metrics = plumbing; dt-responsive = physics. The
   attempt log answered this in one glance.
2. **Route before diagnosing.** The reference evaluator executing the same
   SSA answers "lowering or emission?" in one command. Never skip it; a
   wrong routing multiplies every later step.
3. **One theory at a time, killed by measurement.** Step returns, region
   bodies, call pairing, unpack contracts -- each was a confident suspect,
   each was exonerated by a direct read before moving on. Stacked unverified
   theories are how a hunt produces a confident wrong fix.
4. **Read magnitudes as evidence of identity.** 0.728 could not be a
   violation; it had to be a tracer. What a number *is* often names which
   value arrived before any dataflow is traced.
5. **Capture-and-replay at the oracle before blaming arithmetic.** The
   formula was correct for its inputs. Without the isolated replay the
   "fix" would have mutilated correct code.
6. **A measurement you cannot take is not a zero.** The artifact's interior
   slots are unreadable; the evaluator's values are not. Choose the
   instrument that can actually observe, and refuse conclusions from the
   one that cannot.

The instruments that enforce this, all built this week and all wired into
`tools/TRANSLATION_DEBUGGING.md`: the round-trip materializer (read the
compiled program as Python), the instrumented Python shell (value-level
traces, `first_divergence`), the SSA self-checks (formal parity, id scale,
output contracts, storage ABI), the scorecard (the frontier as a fact under
test), and the reference evaluator (independent execution, `result.values`).

## After the namespace fix

1. **The fluid test green** is the immediate deliverable; it validates the
   entire week against the flagship.
2. **The backward arc** is the finish line the direction note names:
   gradient computed inside the compiled loop, closing the stateful-network
   story. The optimizer loop already compiles exactly (scorecard level 7);
   what remains is capturing the backward pass into the same authored
   program shape the compiler now handles.
3. **Known debts, deliberately parked:** the loop-exit anchor-inserted dead
   call (extend call statements to sequence level; retire anchor insertion);
   the cffi import-time entanglement that blocks isolated baselines (the
   JIT lane's kernels build on import even when only the AOT path is used);
   the `test_fortran_fidelity` pair and the thread-safety verification of
   fftfree lanes / nodus pool noted in memory.
