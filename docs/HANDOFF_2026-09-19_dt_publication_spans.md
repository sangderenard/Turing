# Handoff 2026-09-19 — seven-law compile, publication rows, and what I got wrong

Written by the agent that did the work in this session. It is deliberately
weighted toward concessions and hard numbers, because several things I asserted
during the session turned out to be wrong and a reader needs to know which
statements are measurements and which were inferences that did not survive.

**Goal being worked toward:** all seven chamber laws compiling into one native
step and rendering through the existing GLSL shaders, with no Python in the
render loop.

---

## 1. Tree state

Branch `codex/recursive-reduction-bridge`, HEAD `b74c1874`
("The dt system steps several participants as spans, each judged on its own"),
parent `b8e69910` ("Let two LLVM pieces link, and give the shifts integer
operands").

Uncommitted:

| path | what it is | verified? |
| --- | --- | --- |
| `src/compiler/ssa_extent_audit.py` (new) | finds Phi arms that disagree with their merge about element count | yes, on a probe |
| `src/compiler/precompile_to_ssa.py` (mod) | loop scheduler no longer orders read-after-read | non-regressing only; **never confirmed to fix anything** |
| `src/common/dt_system/dt_controller.py` (mod) | publication read hoisted to one site; `spans` threaded instead of `state` | tests pass; **disproven as a fix** |
| `examples/dt_negotiation_helpers.py.bak` | untracked, dated Sep 17, not from this session | not mine |

Worktrees created this session and **left behind — please clean up**:
`C:\Users\alber\AppData\Local\Temp\wtp` (at `b8e69910`). Also present from
earlier work: `wtb` (`e1fe2ef7`), `wt_head` (`d4d745b5`), and a prunable one
under a claude scratchpad.

Note `wtp` has been contaminated: I copied HEAD's `examples/llvm_dt_system.py`
and `src/common/dt_system/participants.py` into it for a bisect. It is not a
clean `b8e69910` checkout.

---

## 2. Measured facts

These were observed, not reasoned.

* **All seven laws already compile to LLVM individually.** 28 pieces in
  `artifacts/llvm_pieces` (7 laws × b1/b8/b16/b64), each with its own `.ll`,
  `.dll`, `.lib`, built 2026-09-17 07:35:52 at O2. Per-law cost 6.6–13.4 s and
  **flat across batch** (`voxel_air_step` b1 9.1 s, b64 7.6 s). Batch size is
  not a cost driver at the piece stage.
* **The combined two-law program lowers at `b8e69910`**: 167 functions in
  481.4 s. It does not lower at `b74c1874`.
* **Lowering the two-law program costs ~8–12 minutes per attempt.** This is the
  binding constraint on iteration. Budget accordingly; do not plan a loop that
  needs many attempts.
* **The seven-law program has never completed a combined lowering.**
  `build/seven_linked/`, `build/seven_inline/` and
  `artifacts/llvm_pieces_grid_link/` do not exist.
* **The C emitter can report `COMPLETE` for a program that is wrong.** A `(64,)`
  span Phi emitted `t31 = (double *)(&scalar)` and a consumer read 64 elements
  from it, with no shortfall. The guard at `ssa_c_backend.py:2496` compares
  `buffer_type` **strings**, so a pointer to one double and a pointer to sixty-four
  are the same type to it.
* **`float(tensor)` used to truncate silently** — `float(0.877)` read as `0.0`.
  `AbstractTensor.__float__` was added in `b74c1874`.
* **`tests/dt_system/ -m fast` deselects the only test that lowers**
  (`test_llvm_dt_system.py`). A green fast suite is not evidence about
  `examples/llvm_dt_system.py`. 76 passed / 69 deselected throughout.
* **`tests/test_ir_sequence_tables.py` + `tests/test_loop_carried_producers.py`:
  14 failed, 37 passed** — identical with and without the scheduler change, and
  identical at `b8e69910`. Those 14 failures are pre-existing and unrelated.

---

## 3. The refusal that blocks the combined program

`ValueError: control effect order conflicts with value dependencies`, raised in
`precompile_to_ssa._schedule_loop_callsites` → `dependency_order` → `schedule`
(around line 9370 before my edit).

The cycle, read off the traceback's frame locals (`run`, `dependencies`,
`signatures` are all live there — no patching needed to obtain this):

```
[35] __plan_callsite_142__   deps=[32]        reads(23)          writes(142)
[34] __scheduled_region_20__ deps=[35,53]     reads(142)         writes(293,294)
[39] __scheduled_region_23__ deps=[34]        reads(294)         writes(149,299)
[52] __scheduled_region_44__ deps=[39,49]     reads(480,23)      writes(481)
[30] __scheduled_region_45__ deps=[27,52]     reads(484,23)      writes(485)
[32] LoopControlBlock        deps=[3,30,31]   reads(263,381,485) writes()
```

Two rules build those edges:

1. an **arena** rule — a node depends on the last node that touched the same
   arena, with no read/write distinction;
2. a **terminal guard** — `dependencies[position].add(last_terminal)` for a
   placed callsite that follows a `LoopControlBlock`.

`35 → 32` is the terminal guard. `32 → 30` is a real value dependency. The path
between them (`35 → 52 → 30`) exists **only** because all three *read* arena 23.
`sequence_accesses` yields only sequence/table arenas, so arena 23 is a dict or
list, and the most likely candidates are `metrics.error_channels` and
`targets.error_limits`, which `step_with_dt_control_used` reads in at least five
places per attempt.

**The cycle is in `dt_controller.py`, not in `examples/llvm_dt_system.py`.**
Measured: with `b8e69910`'s `dt_controller` and HEAD's `llvm_dt_system` +
`participants`, there is no cycle.

---

## 4. The second, separate refusal

With the cycle out of the way, the same configuration reaches:

```
FortranEmissionError: full-native execution contract rejected the linked repository SSA
  undefined_operands=2, in step_1__advance_pieces, blocks if_merge.1 / if_merge.2
  callee ssa_sequence_309_store, ssa_sequence_operation: table_store, sequence_id: 309
  structural_outputs: (304, 'call', 'call-result-unavailable')
                      (308, 'call', 'call-result-unavailable')
```

Mapping the ids against the block trace: sequence `309` is
`state.publications`, holding two entries keyed `145`/`151`; sequences `301` and
`305` are the two `channels={...}` dicts, with 4 and 3 entries — which matches
`voxel_air_step` publishing all of `energy_j, power_w, div_inf, mass_err` and
`pool_step` publishing three. The values `304`/`308` stored into `309` are the
`Publication(...)` **instances**, and those are the unavailable call results.

So: the dicts lower as tables. The constructed dataclass instance is what has no
representation. (`lower_table_store` in `ir_sequence_tables.py` emits `Ret []` —
a void function reporting through a status arena — so the store itself is not the
thing missing a result.)

---

## 5. Concessions — claims I made that were wrong

Listed because a reader will otherwise find them in the session log and in code
comments and take them as established.

1. **"The publication rows caused the scheduling cycle."** Wrong. The program
   refused identically with the emitted rows removed — same cycle, same node ids.
2. **I used the Sep-18 built C as proof of that**, on the grounds that it
   contains `Publication` zero times. That only showed the tree that built lacked
   the rows. It was a correlation presented as attribution. The Sep-18 DLL was
   built from a dirty tree, so even `HEAD~1` is not necessarily what built it.
3. **"A compiled step cannot hold a Python object; there is no storage kind for
   an arbitrary Python class."** Overstated. The dict became a `table`. The
   accurate statement is narrower: a constructed dataclass instance has no
   materialized call result, and `None` has no storage.
4. **Then I over-corrected** and said the rows "get much further than I claimed."
   Also wrong in the other direction — the original point survives; it is the
   `Publication(...)` instance specifically.
5. **"The state read after a guarded terminal edge is the cause; hoisting it
   fixes the refusal."** Disproven by measurement: identical cycle with the hoist
   in place. The hoist is kept only because reading the value once where it is
   produced is better than twice; its comments in `dt_controller.py` now say
   explicitly that it was made as a fix and did not fix.
6. **I claimed the phi repro reproduced the seven-law wall.** It did not — it
   built a *value* phi and emitted `COMPLETE`; the wall is an address phi.
7. **I called the two-law lowering "seconds"** when planning the extent audit. It
   is ~8 minutes.
8. **I spent roughly 25 minutes and four git/worktree maneuvers** establishing
   "the cycle is in `dt_controller`", which the error message plus the commit diff
   already implied. The user's correction stands and is worth repeating to whoever
   picks this up: *blame tells you locality at best, and this error printed its own
   dependency graph.* Read the error, not the history.
9. **Most importantly — the RAR scheduler fix is probably off the critical path.**
   It makes the *dict* form schedule. The dicts are what the span layout replaces.
   It is a real dependence-analysis correction (RAW/WAR/WAW need ordering,
   read-after-read does not) and it regresses nothing measured, but it was me
   making the doomed form work. Do not treat it as progress toward the goal.

---

## 6. What the work actually is

One problem wearing three faces: **the compiled lane is handed keyed and
constructed values where it needs declared, id-indexed spans.**

* the scheduling cycle ← `metrics.error_channels`, `targets.error_limits` as dicts
* `call-result-unavailable` ← `Publication(...)` as a constructed instance
* the amalgamated fold being the only report ← rows that cannot exist as objects

The machinery already exists and is not being used in the lane:
`error_channels.py` is built as id-indexed spans with presence-never-a-zero, and
`participants.StepSpans` already fixes the publication layout.

Specified layout (from reading `dt_system_contract`, which declares `PieceState`
fields as spans with explicit extents):

* `pub_tau`, `pub_tau_present`, `pub_contract`, `pub_dt_limit`,
  `pub_dt_limit_present` at `span(P)`
* `pub_values`, `pub_present`, `pub_limits`, `pub_limits_present` flattened at
  `span(P*C)`, `index = participant * C + column`
* presence bits carry what `None` carried; `BIND`/`HOLD` become numeric codes
* **the program pins its own channel order.** Not
  `error_channels.declared_channels()`, which returns "in id order" — a
  per-process accumulation. A compiled artifact bakes its column index in, so two
  processes that declared in different orders would read each other's columns.
  This is the same class of fault as the two channel registries that disagreed
  about which id a name has (fixed in `b74c1874`).

The open scoping question I stopped on, and did not decide: whether to take
channels-and-limits-as-spans all the way into `Metrics`/`Targets` now — which
touches every dt caller in the tree — or to do the publication rows first and
leave the controller's own dicts for a second pass. Only the second pass removes
the table arenas from the loop, so only the second pass removes the cycle without
the scheduler change.

---

## 7. Shortest path to something on screen

The host side is already built and is **not** blocking: `build/field_shell/`
holds `demo_host.exe`, six GLSL shaders, `field_volume.comp.glsl`, `glslcheck.exe`
and SDL2.dll. It links the step statically and needs five symbols beside it:

```c
extern void STEP_ENTRY(void **buffers, long long *extents);
extern const size_t step_buffer_count, *step_element_counts, *step_element_sizes;
extern int step_field_buffer, step_telemetry_buffer;
```

`demo_host.exe` as built is `-DSTUB_STEP` — an analytic blob, no sim attached.
`NX/NY/NZ` default to 16, i.e. 4096 cells, which would need a b4096 piece set
that does not exist.

**Build it `-DNX=4 -DNY=4 -DNZ=4` and the cell count is 64 — exactly the b64
piece set already on disk.** A 4³ volume is small but it is a real 3D texture
through the real shaders with all seven laws, and it needs no new piece build.
`NativeSystem.layout()`/`write_layout()` (added in `b74c1874`) exist to fill the
five descriptors.

Also worth knowing: the emitter's Phi refusal is gated on `prod(shape) != 1`, so
it **cannot fire at b1**. A seven-law run at b1 isolates whether anything other
than extent blocks the combined program. `seven_lower_emit.py` in the session
scratchpad does lower → extent audit → emit and stops before `cl.exe`; it has
never been run.

---

## 8. Things left unfixed and known

* The C-lane gather is silently wrong at HEAD (returns `src[idx[0]]` broadcast).
* `energy_exchange_fraction`'s optional-presence bit reads as absent.
* `n_s` collides across four laws and lagged edges are undeclared — prerequisites
  for the merged symbolic reduction, not for seven independent laws.
* `CEmissionShortfall` is only `(operation, reason)` and `summarize_c_shortfalls`
  *counts* by that pair, so every Phi in a run collapses into one unlocated line.
  Adding the value id needs a **new field** — putting it in `reason` would stop
  the summary summarising. This is why a 34-minute run currently returns almost
  no information.
* Nothing in the compiled lane publishes participant rows yet, so
  `dt_controller` falls back to the blended energy/power pin.
