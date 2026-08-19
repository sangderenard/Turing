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
| address temporaries allocated module-wide; recovery refuses addresses | (this session) |

The `test_precompile_to_ssa` suite is 34/34 (was 25/34 for the whole prior
era). The aliased-IO design is now *regulated rather than trusted*: one slot
per carried value, seeded before the loop, read-then-written through calls in
place, declared in metadata where it crosses a signature.

## The one named defect: RESOLVED, and the diagnosis it corrected

The whole-program fluid test (`test_symbolic_fluid_native_runtime.py` -- the
dt system ingested whole by the AOT compiler) is **green**, with attempts
`[(0.2, False), (0.1, True), (0.1, True)]`, the exact sequence this document
named as the finish line.

The collision was real and located exactly where the previous session said it
was -- caller ids 80/89 (`tracer_diffusivity`, `viscosity`) occupied by
region-internal `GetElementPtr` addresses, so the caller read a height cell
(~1.0) where its 1.0e-4 diffusivity belonged. **The named mechanism was
wrong, and measuring it took one probe.** The ids are not PlanLine ids and
were never in the hierarchy namespace:

* region_2's PlanLines produce `40, 41, 47, 59, 69, 79` and nothing else;
  region_1's stop at 88. Dumping every `PlanClosure` reaching
  `plan_region_to_ssa_instrs` finds no line anywhere in the program that
  outputs 80 or 89.
* Both ids are minted later, by `lower_indexing_to_ssa_addressing`
  (`ir_indexing.py`), which lowers `Indexed` to `GetElementPtr`+`Load` and
  allocated its addresses from **each function's own** maximum id: region_2's
  max is 79, so its first address is 80; region_1's max is 88, so its first
  is 89. A planner region shares its caller's value space, so "one past this
  function's max" is not free -- it is whatever the caller happens to hold.
* The binding is made by the structural-output recovery in
  `fortran_c_shell.py`, not by `region_signatures`/`output_ids` at all. For a
  `desired_id` the caller lacks, it scans callees for any instruction whose
  `res.id` equals that integer, appends it to the call's `output_ids`, and
  materializes a `GetElementPtr`+`Load` unpack. It matched on the bare
  integer, so it bound a caller scalar to an address.

Two fixes, each independently sufficient (verified by disabling the other):

| fix | what it closes |
|---|---|
| `lower_indexing_to_ssa_addressing` allocates from one module-wide watermark | the collision itself -- an address can no longer take an id used anywhere in the module |
| the recovery pass refuses a `GetElementPtr` result as a recovered output | the binding -- an address is a location in the callee's storage, never a value the caller asked for |

Both are kept. The first makes the collision unrepresentable; the second is
correct on its own terms regardless of numbering, and would have refused this
defect even with the ids as they were.

The sentence from the previous handoff still stands and is worth keeping:
**every layer downstream of the collision behaved correctly.** The step's
arithmetic matched the SymPy oracle for the inputs it received; the
controller's same-reason-at-every-dt detector did exactly its job.

*What the wrong diagnosis cost, and why it was still worth writing down:* the
previous session named the boundary (region output publication), the two
colliding integers, and the class of defect -- all correct, all load-bearing.
It named the wrong minting site because it inferred it rather than dumping
the PlanLines. The correction took one probe precisely because everything
around it had been measured. **Record which parts of a diagnosis were
measured and which were inferred**; the inferred parts are where the next
session should aim its first probe.

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

1. **The fluid test is green** -- the immediate deliverable, and it validates
   the entire week against the flagship.
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
