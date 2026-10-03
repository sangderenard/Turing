# CONTINUATION: llvm_dt_system per-state program binding + contiguous spans

Lane approved by the user: edit only the compiler's INPUT
(`examples/llvm_dt_system.py`, its generators, or the compiler itself);
never edit anything the compiler emits.

## 2026-10-03 -- goal 1: per-state program binding

- Cause (observed): `instantiate_state` called `bind_pieces`, which assigned
  `PieceState`/`advance_pieces` as module globals; `dt_system_over` reads the
  global `advance_pieces`, so every persistent state ran the program of the
  state instantiated last.  Repro:
  `engine_toy/tests/test_orbital_jumper.py::test_interleaved_states_each_run_their_own_program`
  (strict xfail).
- Fix: `bind_program(namespace)` -- a dict of this module's globals plus the
  state's generated namespace, with `dt_system_over` re-bound as
  `types.FunctionType(dt_system_over.__code__, program, ...)`.  Same code
  object, program text unchanged; the binding is per state, exactly the way
  the compiled lane binds it (module source + that state's generated source
  are one module).  `instantiate_state` stores it as `state.program`;
  `dt_system` calls `state.program["dt_system_over"]`.  `bind_pieces` kept
  for the compiled lane (lowered_system + compiler_probes use its bindings).
- Result: xfail marker removed; the test passes (9 s).
- Baseline profile, one station 120 s frame (scratchpad
  `prof_station_spans.py`): 4.14-4.43 s wall; `__setitem__` 74,880 calls from
  advance_pieces, 1.82 s cum; `prepare_artifact_execution` 1,920 calls from
  `LLVMPiece.__call__` (the instantiated fast path is missed), 0.29 s;
  `inspect.signature` 6,733 calls via AbstractTensor `wrapped` (abstraction.py
  :3586), 0.2 s.

## 2026-10-03 -- eager parity harness + a pre-existing piece defect

- Harness (scratchpad `spans_parity.py base|work OUT.pkl`, `compare A B`):
  dumps every ndarray/AbstractTensor field of the state, plus
  (advanced, dt_next, telemetry), per round, bitwise.  Scenarios: two LLVM
  drift laws (batch 4) x schedule {sequential, parallel} x rollback
  {False, True}, 6 rounds each; orbital craft (6 thrusters, throttled 4
  frames) + 3 batch-1 stations + 1 batch-4 station, interleaved, 12 frames.
  Baseline = the goal-1 file copied to scratchpad `spans_base/`.
- Found: two identical runs differ in ONE span -- orbital/batched
  `propellant_supply` lane 0 (1.4e-311 / 6.8e-312 / 1.2e-311...).  The batch-4
  `orbital_craft_propellant_supply_t6` piece standalone returns
  `[garbage, 1, 1, 1]` when its Piecewise condition is false (batch 1: 1.0).
  Pre-existing native piece defect, outside this lane; flagged as a separate
  task.  The harness skips exactly that (scenario, field) and says so.
- Determinism check base vs base-copy: 0 differences elsewhere.

## 2026-10-03 -- goal 2, part A: per-state piece binding, in-place outputs, one span

- Finding (observed): the 1,920 `prepare_artifact_execution` calls per
  station frame were the shared-piece rebinding.  `equation_piece` caches
  pieces in `_PIECE_MEMORY_CACHE`, so the craft and every station hold the
  SAME `LLVMPiece` objects; each `instantiate_system` re-ran
  `piece.instantiate`, the last state won, and every other state's calls
  failed the `column is span` check and prepared a fresh ABI per call.
- Fix: `own_pieces` -- each state (instantiate_state, RoundPiece, Subcycle)
  takes a shallow `copy.copy` of every `LLVMPiece` (artifact/ids/SSA shared,
  `_execution`/`_bound` the state's).  After: 0 per-call preparations in the
  station frame.
- Output in place (existing mechanism: `prepare_artifact_execution` aliases
  ANY fed value id, outputs included; `structure_native.NativeBank` already
  feeds its `out` buffer this way).  `LLVMPiece.instantiate(columns,
  outputs=None)`; `instantiate_pieces(pieces, state, schedule)` hands a
  piece the column views for its `<c>_next` outputs ONLY under `sequential`
  and only when the piece does not read `c` (the kernel runs op-by-op over
  whole buffers, so an output aliasing an input it still reads is unsafe --
  cf. memory project-llvm-inplace-store-aliasing).  The piece keeps its own
  buffer when the output id is also an input id (orbital momentum piece:
  `dt_prev_next` IS buffer 7 = `dt`), when one id fills several declared
  outputs (CSE), or when the extent differs.  `piece.in_place` records what
  landed.  Orbital craft: supply 1/1, actuation 7/7, gravity 3/3 in place;
  momentum/position/thrust_cost read their own columns -> own buffers.
  Program text unchanged: the spelled `state.c[...] = o` copies the span
  onto itself.
- One contiguous span: `state_spans(names, columns, batch)` -- columns, `dt`,
  telemetry as views into one float64 span (`state.span`); a column that is
  not `(batch,)` is refused.  Precedent: `prepare_artifact_execution`'s
  per-dtype scalar arena.  Eager-only; the compiled lane declares each field
  as its own span and is untouched.
- Parity (spans_parity.py): 9 scenarios, 7428 field-rounds, 0 differences.

## 2026-10-03 -- goal 2, part B: channel publication as one row slice

- `piece_source`: each law's 15 `pub_values[slot] = v` + 15
  `pub_present[slot] = f` became `state.pub_values[i*C:(i+1)*C] =
  AbstractTensor.tensor([...])` and the same for `pub_present` (constructs
  already in this program: `AbstractTensor.tensor([...])` builds
  `Metrics.error_channels`; slice store = `index_assign`).
- Parity: 0 differences.  Station frame (profiled): 4.14-4.44 s -> 2.1-2.65 s;
  advance_pieces 2.37 s -> 0.64 s cum; `__setitem__` 74,880 -> 21,120.
- Remaining stores: 9 per law per attempt into the (P,) participant spans
  (exchange_time + present, contract, dt_limit + present, the 4 Courant
  slots).  In the existing layout each law owns ONE element of each (P,)
  span, so there is no row to slice; options (open question to the user):
  one whole-span store per field from the laws' scalars in causal order, or
  a declared (P, K) participant span with the (P,) fields as views (needs
  the compiled lane to know the aliasing).
- NEXT: dt-managed native compile on the new source (air + pool b1,
  `lowered_system`, C backend).

## 2026-10-03 -- the durable air/pool pieces did not unpickle

- `LLVMPiece.load(artifacts/llvm_pieces/pool_step/b1/pool_step.piece)` raised
  `AttributeError: 'SSASequenceTable' object has no attribute 'sequences'`
  (then the same for `SSARecordTable.records`).  Cause (observed with a
  monkeypatched `__setstate__`): these tables gained `__reduce__` (commit
  56255f42); a table pickled before that arrives via the default protocol,
  no `__init__` runs, and its state is `{"sequences": ...}` -- but
  `__setstate__` read `self.sequences`.  Fix in `src/transmogrifier/ssa.py`:
  the three `__setstate__` read the attribute when `__reduce__` set it, else
  the state's own entry.  Both b1 pieces load (24/17 and 32/19 args/outs).

## 2026-10-03 -- targeted tests

- engine_toy: test_orbital_jumper.py 11 passed (31 s, incl. the flipped
  interleaved test); test_orbital_craft_machine.py 13 passed; test_orbital_game.py
  7 passed.
- turing tests/dt_system/test_llvm_dt_system.py: 2 passed.
- tests/dt_system/test_llvm_dt_system_python_lane.py: 12 failed / 1 passed --
  IDENTICAL against `git show HEAD:examples/llvm_dt_system.py` (pre-imported
  from scratchpad head_lds/): its plain `Piece` lacks `instantiate`, which
  `require_piece` demands (PIECE_API).  Pre-existing, not touched here.
- Compile: the 10-minute foreground cap killed the first inline attempt
  (`timeout 595`); rerun as the single background build (pid 29984,
  09:53:35), events relayed in the report.

## 2026-10-03 -- native compile: stalls in the compiler, same on the baseline

- Instrumented run (faulthandler every 900 s, pid 23772, 10:00:35):
  progress stops at +42 s on `callsite-tensor-specialization ...
  callsite-progress round=1 callsites=1 caller=dt_system_over
  callee=run_superstep`.  Dumps at 15 and 30 min: inside ONE callsite fold,
  `_propagate_callsite_tensor_specializations` (gds 18157, then 18090) ->
  `_fold_callsite_structural_values` (gds 22209) -> ~20-deep recursive
  `_tensor_descriptor`/`_tensor_descriptor_rule` (gds 19357/20110) ->
  `record_shape_transformation` -> `identity_concordance.latest` ->
  `history` (linear genexpr).  Same frames, different depths and caller
  lines: not a repeating round, but no visible progress counter for 29 min.
  Killed by Windows pid at 30 min, per the user's rule.
- A 120 s-dump rerun died in faulthandler's own dump ("Windows fatal
  exception: access violation"); `identity_concordance.py` line numbers had
  moved (3226 -> 3275) during the session -- other agents are editing the
  compiler (gds, precompile_to_ssa, symbolic_*, hierarchical_plan,
  identity_concordance; HEAD moved to 408155a7 at 10:27).
- Baseline: the SAME lowering on the goal-1-only source (scratchpad
  spans_base, no row slices, no spans) reaches the same callsite at +40 s and
  is still in it 12 min later -> killed.  The stall does not come from this
  change; it is the current compiler tree at that callsite.  The native
  compile on the new source is therefore UNVERIFIED, not failed.
- Unprofiled station frame (3 frames x 2 alternating runs): base 1.53-1.60 s,
  new 0.98-1.08 s.  (The 4.1-4.7 s figures are under cProfile.)
- OPEN: (1) the run_superstep callsite fold stall (gds -- other agents'
  file); (2) the 9 per-law (P,) participant stores -- layout question;
  (3) AbstractTensor `inspect.signature` per wrapped call (substrate);
  (4) test_llvm_dt_system_python_lane.py 12 failures pre-existing (Piece lacks
  instantiate); (5) batch-4 Piecewise lane-0 garbage (separate task).
