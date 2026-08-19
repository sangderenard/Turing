# Diagnostic decision tree, and the machine lane's readiness for it

**Date:** 2026-08-19.
**Continues:** `HANDOFF_SHOAL_AND_RE_TARGETS.md` §6f (which references this
report), `docs/REVERSIBLE_MACHINE_COMPLETION_AUDIT.md` (the machine's own
verified/not-yet ledger), `READ_HEAD_STATE_MACHINE_PLAN.md`,
`tools/DIFFERENTIAL_PHASES.md`.

This session ended with a confusing error of exactly the class this report
is about: a compiled artifact that is **oracle-exact in every physical
observable and still dies with an access violation** — an out-of-bounds
int64 read (`vcvtsi2sd` from just past a page boundary) that a padded debug
heap silently absorbs. The Python-side instruments proved the *program* is
right; the crash lives in the *artifact*. That split is the organizing idea
here: every instrument in this repository tests a different layer's claim,
and a confusing error is usually one whose symptom surfaces two layers away
from its cause. The decision tree routes by which layer's claim is broken;
the rest of the report assesses, with measured evidence, how ready the
deepest instrument — PE decompilation under the reversible machine with the
bidirectional read head — actually is, and what it can do that nothing else
can.

---

## 1. The decision tree

Start from what you can already say about the wrong value. Each branch names
the instrument, what its verdict means, and where to fall through when the
verdict is "this layer is clean."

```
A wrong value / crash / refusal appeared.
│
├─ 1. Is the LOWERING's own claim suspect?  (did the compiler even build
│     the control/dataflow the author wrote?)
│     INSTRUMENT: translation scorecard (tools/translation_scorecard.py)
│       — LOWER → MATERIALIZE → EXECUTE → EQUIVALENT on minimal probes.
│     Write the smallest authored program with the same *shape*
│     (a while, an if, a rebound name...) and score it. A rung that runs
│     and is wrong is a compiler defect you can iterate on in seconds.
│     Levels 16/17 both fell this way; level 16's probe led straight to
│     the constant=None root defect that had silently flattened every
│     dynamic conditional in every program.
│     └─ clean → 2
│
├─ 2. Is the MATHEMATICS wrong at full scale?  (the shape is fine, the
│     numbers drift)
│     INSTRUMENT: independent-oracle differential
│       (tools/differential_translation.py; phases in
│       tools/DIFFERENTIAL_PHASES.md) — sympy.lambdify over the authored
│       equations vs the artifact, bitwise-identical inputs, first
│       divergence in authored order. Compare quantities the defect can
│       MOVE (first cell, wave speed, dt) — conserved sums match
│       vacuously.
│     └─ AGREED → 3
│
├─ 3. Does the EMITTED SOURCE lie about its declarations?  (SSA is right,
│     the target text is not)
│     INSTRUMENT: the target compiler as a measurement device. gfortran
│     names declaration/ABI lies precisely: this arc alone it caught
│     scalar-vs-array rank lies, INTEGER-literal-into-REAL-dummy
│     (the inlined [] seed), and LOGICAL(4)-into-LOGICAL(1) (the
│     kind-less coercion). Emitting "complete" source and compiling it
│     are different claims; treat each error batch as the next defect
│     class named exactly.
│     └─ compiles clean → 4
│
├─ 4. Does the BINARY misbehave at runtime?  (compiles, runs, and is
│     wrong or dies)
│     4a. Cheap native triage first: cdb.
│         `cdb -hd -g -G` (debug heap OFF) reproduces release-condition
│         crashes and gives the faulting instruction; plain cdb (debug
│         heap ON) pads heap allocations — if the crash vanishes and the
│         run completes, you have BOTH a strong hint (out-of-bounds heap
│         access) AND a free full-run observation of the otherwise-sound
│         program. That exact contrast is how the Shoal frame was shown
│         to be oracle-exact while still carrying an OOB read.
│     4b. The faulting instruction is not the defect — the INDEX'S
│         PROVENANCE is. When the "where did this bad value come from"
│         question survives 4a:
│     INSTRUMENT: PE decompilation under the reversible machine
│       (BinaryMachineProgram.load_pe → tape → rewind). See §3.
│     └─ the binary is faithful → 5
│
└─ 5. The program is faithfully wrong: the defect is in the AUTHORED
      source or its contract (identity policy, extraction, feeds).
      INSTRUMENT: work-contract review + the handoff ledger. This is a
      policy question, not a debugging one — env-gated behavior belongs
      in the work contract (prove/develop/deploy/fast), never in new
      flags, and never in "dead-coding" live arms.
```

Two standing rules cut across every branch:

* **A value you cannot observe is not zero** (`required=True` on reads).
  Both t14 stories were this rule: first an imposter filled the slot, now
  an unwritten slot reads 0.0.
* **Pin the conclusion to the layer you measured.** "The SSA is right"
  says nothing about the emitted Fortran; "the physics matches" says
  nothing about heap bounds. The Shoal frame is simultaneously
  oracle-exact (layer 2) and crashing (layer 4) — both true.

---

## 2. The bidirectional read head: readiness and potential

`src/compiler/x86_tensor_read_head.py`.

**What it is.** A decoder/encoder of AMD64 instruction bytes expressed as a
pure state machine over **20 named int64 `AbstractTensor` registers, one
scalar per parallel lane** (`X86ReadHeadState.REGISTER_NAMES`: cursor,
phase, opcode, rex, modrm, sib, displacement, immediate, ...).
`transition(batch, state) -> state` is pure and **entirely branchless**
(every update is a `where`-select), and `ReadHeadDirection.FORWARD/BACKWARD`
is in the type system — bidirectionality is constitutive, not bolted on.

**The authority hierarchy, which must not be inverted.** The scalar
`InstructionSpec` vocabulary is the source of truth for what instructions
*mean*; the tensor read head is a **derived accelerator** for reading,
writing, and reversing instruction streams. This is enforced by test, not
convention: `test_bidirectional_profile_is_derived_from_authoritative_specs`
and `test_every_authoritative_token_has_reference_and_tensor_roundtrip`
(tests/test_x86_reversible_read_head.py, 15 tests) pin that the head's
bidirectional profile is generated from the authoritative specs and that
every token round-trips through both the reference decoder and the tensor
head. New ISAs enter at the spec+scalar layer; the head follows.

**Readiness, as tested today:**

| capability | evidence |
|---|---|
| forward decode == reference decode | per-token roundtrip test over the whole authoritative vocabulary |
| exact reversal of *partial* decode | `test_reverse_restores_exact_partial_decode_and_forward_can_branch` |
| write (encode) through the same head | `test_bidirectional_head_writes_exact_controlled_instruction`, incl. required legacy prefixes as both read and write policy |
| forked reversible futures | `test_fork_has_independent_reversible_future_and_acknowledgement` |
| many lanes, published registers | `test_virtual_cores_advance_concurrently_and_publish_every_register`; `register_tensor()` is the stable observation ABI |
| compiled form | tests/test_read_head_wasm_state_machine.py — the head itself lowered as a state-machine artifact (the READ_HEAD_STATE_MACHINE_PLAN.md slice) |

**Potential for diagnostics.** The head is the piece that makes the machine
lane's transparency *cheap in principle*: because decode is a pure,
branchless, batched tensor program, it can (a) decode many speculative
positions at once — the "what instruction would this byte offset be?"
question a corrupted-control investigation asks constantly; (b) run
BACKWARD over bytes, which is what lets a tape rewind re-derive code state
rather than store it; and (c) be compiled itself (Wasm/GLSL) so the
transparency layer does not pay Python-interpreter prices forever. It is
also the *write* head: reverse-compilation forms are read by the same head
(`test_same_bidirectional_head_reads_the_reverse_compilation_forms`), which
is what "decompiling" means here — not a disassembly listing, but a
round-trippable raising with the writer in the same vocabulary.

**Honest limits.** The head decodes and encodes; it does not *mean*. All
semantic execution (what ADD does to flags) lives in the scalar semantics
and the machine's effect handlers. Its GPU/native kernels are not present;
today its acceleration story is the Wasm tier below.

---

## 3. Decompiling in intense diagnostic analysis: the reversible machine on a PE

`src/compiler/binary_machine_program.py` (`BinaryMachineProgram.load_pe`),
audited in `docs/REVERSIBLE_MACHINE_COMPLETION_AUDIT.md`.

**What the lane does.** It raises a real Windows PE — the *shipped
artifact*, not a re-materialization — into a token multigraph through the
existing decompiler, links capability-supplied DLLs, and executes it as a
reversible machine: **every instruction, external completion, console
input, shared-memory synchronization, and child deployment is an exact
tape edge**, replayable cold from immutable segments, rewindable to any
prior state, with bounded hot storage (4,096 resident states) and
fail-closed behavior on anything it does not understand.

**Verified readiness (from the audit — these are proofs, not plans):**

* Real `cmd.exe` ingested, relocated by 256 MiB (372 relocations applied),
  and run to `/c "echo hello"` **exit code 0** — 16,010 guest steps, 499
  deterministic capability completions, 15,459 instructions committed
  through authenticated Wasm journals.
* Exact cold reversal: reversing once hydrated only positions
  16,384→16,509, returned to the non-halted predecessor, and one forward
  step reproduced the persisted halted state exactly.
* A 69,949-record real interactive tape continued for 7,756 further
  events and reopened at 77,705 dependency-validated records.
* 247 tests on the machine/tape/SSA/recompilation/VFS/registry slice.

**Measured this session, on OUR artifact** (scratchpad
`probe_machine_pe.py`): the 960 KB gfortran-emitted
`symbolic_fluid_frame_shell.exe` was raised into **805 functions in 15 s**,
and the machine executed its entry transparently, pausing after 25
transitions at the external-target base — the CRT startup's first import
call, correctly fail-closed because no capability completions were
configured. That is the machine doing exactly what it promises: it stopped
*at the boundary of what it can prove*, with the boundary named.

**Why this is the terminal instrument for confusing errors.** For the class
of defect where the *symptom* (a faulting read) is far from the *cause*
(whoever computed the bad index), everything else in the tree gives you a
point observation. The machine gives you provenance: run forward to the
fault under full observation, then **step backward** through exact states
to the instruction that produced the out-of-range value, then further back
to the value *that* was computed from — with the heap, registers, and every
external effect visible and byte-exact at each step. No re-run
nondeterminism (the tape is the run), no debug-heap masking (guest memory
is the machine's own paged model, page-version-coherent even under
self-modifying code), no "it only crashes outside the debugger." The
Shoal OOB read — invisible to the scorecard, the oracle differential,
gfortran, and masked by cdb's padded heap — is precisely shaped for it.

**What stands between here and using it on the Shoal exe** (in order):

1. **CRT import surface.** The deterministic port exposes 205 handler
   identities matching 153 of `cmd.exe`'s 286 imports; a gfortran/MinGW
   exe touches an overlapping but different msvcrt/kernel32 startup
   surface. The audit's own next-work list (coherent capability families)
   is the path; the first pending import of our exe is already known from
   the probe.
2. **Throughput.** The Wasm recompilation tier is real (88% of a proven
   command prefix compiled; per-instruction authenticated checkpoints;
   XMM, REP STOS, and the scalar core admitted) but native/GPU kernels are
   absent, and the fault in our frame is millions of transitions deep.
   Options that don't require completing that program: start the tape
   *near* the fault (the frame's structure is known — one substep suffices
   to reach the sequence machinery), or shrink the grid/frame so the first
   substep is small; the crash is in setup-adjacent sequence indexing, not
   in accumulated state.
3. Nothing else. Ingestion, linking, reversal, observation, and
   fail-closed diagnostics are already proven on a harder binary than ours.

---

## 4. How this differs from the round-trip Python instrument

The materializer round trip (`ssa_python_materializer` + the scorecard) and
the PE machine are both "run the compiled thing and look," and they are
opposites in almost every property that matters:

| | round-trip Python (materializer/scorecard) | PE binary under the reversible machine |
|---|---|---|
| **object under test** | the SSA program, re-spelled as Python | the shipped artifact: emitted Fortran through gfortran, its linker, its CRT, its heap |
| **what a pass proves** | the *compiler's semantics* preserved the authored meaning | the *delivered binary* behaves, instruction by instruction |
| **blind spots** | everything below SSA: declaration kinds (LOGICAL(4) vs c_bool), literal inlining, extents, heap bounds, calling conventions, CRT behavior — the entire class this session's walls lived in | authored *intent*: it faithfully executes a wrong program without complaint; equivalence still needs an oracle on top (tree branch 2) |
| **failure output** | a value comparison ("(0.5, 0.5) for 0.5") — WHAT is wrong | an exact state history — WHY it is wrong, by rewinding to the producer |
| **determinism** | re-execution, subject to host Python | the tape IS the run; cold replay is byte-exact |
| **cost today** | seconds; runs in the ~40s gate | 15 s to raise; execution currently gated on import coverage and interpreter/Wasm throughput |
| **direction** | forward only | bidirectional — reverse is a first-class operation of both the machine and the read head |
| **refusal style** | honest refusal at unspelled ops (by design: only borrowed vocabulary) | fail-closed at unproven imports/instructions, with the exact RIP named |

They compose rather than compete: the round trip is the *fast* claim about
meaning; the machine is the *deep* claim about the artifact. The decision
tree runs the cheap one first — and this session demonstrated why both
directions of that ordering are needed: the scorecard found the
conditional-flattening root defect that no amount of binary inspection
would have named, and the binary now carries an OOB read that no amount of
scorecard work can even see.

---

## 5. Recorded conclusions

1. The bidirectional read head is **ready as a component**: spec-derived,
   round-trip-tested against the authoritative vocabulary, exactly
   reversible mid-decode, compiled as its own state machine. Its diagnostic
   potential is unlocked through the machine, not standalone.
2. PE decompilation under the reversible machine is **ready as an
   instrument for provenance questions** — proven end-to-end on a real,
   harder binary — and is gated for *our* binary on CRT import families
   and throughput, both on the audit's existing next-work list. It is the
   only instrument in the tree that answers "where did this bad value come
   from" instead of "here is where it hurt."
3. The Python round trip and the PE machine test different layers'
   claims; neither subsumes the other, and the Shoal frame is the standing
   example of a program green on one and red on the other.
4. First concrete target when the lane is picked up: the Shoal sequence
   OOB read (`HANDOFF_SHOAL_AND_RE_TARGETS.md` §6f), with the shrunk-frame
   entry to keep the transition count inside interpreter reach.
