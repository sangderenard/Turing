# Handover: dt controller interior, and the audio-driven fluid demo

Written 2026-08-16. Everything below is **uncommitted working tree** unless a
commit hash is named.

---

## 1. The main piece of work: give the controller a pluggable interior

### The finding

`run_superstep` (`src/common/dt_system/dt_controller.py`) and
`solve_window_bisect` (`src/common/dt_system/dt_solver.py`) are two separate
bodies of text. They share no core loop -- separate rejection handling,
separate snapshot/restore, separate landing logic, separate return shapes.

They are **not** redundant, and merging them is not the task:

- the **solver** hunts for an acceptable scenario if one exists. It brackets
  dt in `[dt_min, dt_max]` and bisects until a chosen `field`/`objective`
  reaches `target` within `eps`. Its two dts are a *search bracket*, not a
  requested step size.
- the **controller** drives at a rate, substepping internally as needed, always
  staying inside `Targets` (`cfl`, `div_max`, `mass_max`, `error_limits`). It
  is somewhat solverish but its job is *stay within*, not *land on*.

Their eternal fates do not need resolving. The solver can stay where it is.

### The change

Upgrade the **controller** to have a pluggable interior with three trial rules.
Only one hook is needed: *given the remaining window, the state and the last
metrics, what dt is tried next?*

| interior | trial rule | reset on failure |
|---|---|---|
| `pinned` | dt is a constant, clamped to the remaining window | **none -- see below** |
| `steered` | the PI proposal, halved on rejection (binary division) | existing |
| `solved` | bracket and bisect toward a target (binary search) | existing |

`steered` and `solved` already share every reset mechanism that matters --
snapshot, restore, re-evaluate, accept-or-shrink. They differ *only* in how the
next candidate is chosen. So this is one hook with three implementations, not
two subsystems being fused.

`solved` uses the same bracket-and-bisect **rule** as `solve_window_bisect`.
That is a rule, not a dependency: nothing in the solver module needs to move or
change.

### Why `pinned` dropping the reset mechanics is the load-bearing part

With dt fixed there is no smaller candidate to analyse, so a failed step is not
a retry case at all. It is a genuine physics rejection and must be reported
immediately, naming the violated bound.

This is not theoretical. The current crash looks like this:

```
attempt 1: dt=0.00586147  mass_err=8.840e-16 div_inf=0.000e+00 max_vel=1.084e+00
attempt 2: dt=0.00293073  mass_err=8.836e-16 div_inf=0.000e+00 max_vel=1.084e+00
attempt 3: dt=0.00146537  mass_err=4.416e-16 div_inf=0.000e+00 max_vel=1.084e+00
attempt 4: dt=0.000732684 mass_err=1.766e-15 div_inf=0.000e+00 max_vel=1.084e+00
RuntimeError: adaptive timestep controller failed
```

dt halved four times and `max_vel` did not move a digit. The failing condition
was dt-independent, so retrying could never have fixed it. Under `pinned` that
is one attempt and an honest message.

### Related: the attempt log hides the actual cause

`dt_controller.py:131` rejects on

```python
(not ok) or hard_failure or (mass_err > mass_max)
         or (div_inf > div_max * 10.0) or channel_failure
```

but the attempt log records only `dt`, `mass_err`, `div_inf`, `max_vel`. In the
trace above none of those exceeded anything, so the cause was `not ok` (a
height or tracer violation) or `channel_failure` -- **neither of which is
printed**. Add the failing term and the offending channel to the log
regardless of which interior lands.

### Also note

`STController.update_dt_max(max_vel, dx)` recomputes `dt_max` from CFL on every
step and overwrites any externally assigned value. That is correct behaviour
*for steered*. It only became an obstacle because the demo asked a steering
controller to hold a constant. Under `pinned` it should not be consulted at
all -- do not "fix" it by clamping.

### Open questions to settle first

- Does `Metrics`-based rejection stay common to all three interiors? It
  probably must: that is physics acceptance, not search.
- Return shape. `run_superstep` returns `(advanced, dt_next, metrics)`;
  `solve_window_bisect` returns only the final `Metrics`.

---

## 2. The fluid demo's drive is still wrong

`src/compiler/symbolic_fluid_source.py`

`VoiceCoil` is correct: Thiele-Small parameters (`free_air_resonance`,
`moving_mass`, `force_factor`, `mechanical_q`), stiffness and loss **derived**
from them (`k = Mms(2*pi*Fs)^2`, `Rms = 2*pi*Fs*Mms/Qms`), force integrated
twice to reach displacement. Measured response is a real driver: peak at Fs,
~12 dB/octave rolloff above it (25 Hz -> 24.3 mm, 200 Hz -> 0.74 mm,
1000 Hz -> 0.13 mm).

`drive_surface_cone` is **wrong**. It does:

```python
height[source_row, source_column] = 1.0 + displacement
```

That is a Dirichlet assignment. It:

- does not scale with dt, which is why halving dt four times changed nothing;
- fabricates water, so it cannot conserve mass;
- lands outside the in-step mass balance, so `mass_err` reads 1e-16 while the
  drive is free to pump the pool.

The brief was **force injection**. The cone's force must enter the *fluid* as
force so the momentum equation integrates it -- then the coupling is dt-scaled
and mass-conserving by construction. Neither Dirichlet nor Neumann is the
answer. Fix this together with the `pinned` interior; they are the same bug
seen from two ends.

---

## 3. State of the demo

`examples/symbolic_fluid_live.py`. Defaults, with only `--audio FILE` given:
sample-rate substepping, voice coil, and an AVI at
`build/symbolic-fluid-live/pool.avi`.

Timing arrangement (this was got backwards once -- do not re-invert it):

- **frame** = one video frame, `1/video_fps`
- **dt** = one audio sample, `1/sample_rate`
- 1470 substeps per video frame at 44.1 kHz / 30 fps
- one capture per superstep, so `video_stride == 1`

Verified: `advanced=0.033333333  attempts=1470` on consecutive frames, valid
`RIFF`/`AVI ` header, JPEG frames present, PCM muxed. Cost is real -- about
23 s per video frame at 24^2.

Both former hacks are **gone**: the `pinned` interior holds the substep, so dt
is no longer re-imposed each frame, and `max_iters` is derived from the pinned
period rather than from whatever dt the controller last proposed.

Body forcing was **removed** from the equations entirely
(`symbolic_fluid_model.py`); the model is unforced again and no longer needs
`Sin`. Do not reintroduce a lattice forcing -- a force varying along one axis
has identically zero divergence, so the height field cannot respond and the
result is static stripes at any amplitude.

---

## 4. Trig solvers: both, in all four lanes, no math library

`grep -rn '"libm"' src/` returns nothing. `LLVMTrigSolver.LIBM` and its
passthrough were removed, and four call sites that defaulted to `"libm"` now
default to `"lut"`.

| lane | lut | continuous | selector |
|---|---|---|---|
| LLVM | yes | yes | `LLVMTrigSolver` |
| C | yes | yes | `trig_solver="lut"｜"continuous"` |
| WASM | yes | yes | `trig_solver="lut"｜"continuous"` |
| Fortran | yes | yes | `trig_solver="lut"｜"continuous"` |

All default to `lut`. One vocabulary across every backend. Coefficients, pi and
the error bound come from `bounded_constants.sin_series_terms()`; tables from
`lut_for`. C and WASM cross-checks both land on **8.297525990949572e-10**,
identical to the last digit, which is the evidence they consume one definition
rather than four lookalikes. Lookups are casefolded, so `Sin`/`sin`/`SIN` all
resolve -- a spelling mismatch is what had left Fortran's trig unreachable.

---

## 5. Outstanding compiler work

**Task: `get` has no SSA handler, and the drop is silent.** The last 4 LLVM
shortfalls in the fluid control module are one `any`, and the cause is not
`any`. The graph is complete -- `get vid=8 parents=[(5,'operand'),(1,'arg:0'),
(7,'arg:1')]`, `greater vid=9`, `GeneratorExp vid=15` -- but `Handler` has only
`GetElementPtr`/`GetAttr`, and `'get'` is in neither `ast_ssa_name_map` nor
`ssa_name_map`. So the comprehension body is dropped **with no shortfall
recorded**: the loop's loaded key and limit end with zero consumers (dead
loads), the generator yields nothing, and the linker fills `any`'s operand with
an anonymous `linked_call_frame_storage` slot that is `float64` only because
that is the generic scalar slot type.

Two fixes, and the second matters more than this program:

1. lower `get(mapping, key, default)` on a keyed mapping -- compare the key's
   token against `.keys`, select `.values` or the default. Everything needed
   exists now.
2. **an operation with no handler must become a named shortfall instead of
   vanishing.** The census read 10, then 4, while a whole expression was
   missing from it. Fix the reporting first; it is cheap and it will tell you
   whether anything else has been falling out of the census all along.

---

## 5b. The demo does not run the compiler, and that is the real cost

`symbolic_fluid_native_runtime.load_symbolic_fluid_managed_functions` does:

```python
exec(SYMBOLIC_FLUID_DT_SOURCE, namespace)
```

(line 106 at `4abc962`, so it predates the 2026-08-16 session.) Only the
*stencil* is compiled. The traversal -- the row/column loops, the `%` neighbour
wrapping, ~30 numpy scalar reads per cell, the 11-tuple unpack, the `max`
reductions, the mass accumulation -- is interpreted CPython. That is also why
there is a per-cell marshalling cost at all: an interpreted loop can only hand
scalars across the boundary one cell at a time.

Measured on a 24^2 grid, after a 30% adapter improvement:

```
kernel + ABI   10.6 us/cell  (65%)   ~4 us of it ctypes call overhead
traversal       5.8 us/cell  (35%)   the exec'd Python loop
```

`SYMBOLIC_FLUID_DT_SOURCE` is the *same text* `symbolic_fluid_direct_control`
lowers to repository SSA. There are two paths over one source -- one compiled,
one exec'd -- and the demo runs the exec'd one. Re-exposing it to the compiler
is the order-of-magnitude change; further marshalling work is polish on the
wrong side of the boundary.

### Where `get` is actually dropped

Narrowed on 2026-08-16. It is **not** the reducer:

```
graph:    get(error_channels, name, 0.0) -> greater -> GeneratorExp   present
reducer:  reduce_abstract_tensor_topology leaves `get` in the graph   present
planner:  no PlanLine is emitted for it                               DROPPED
SSA:      loop loads keys/values with zero consumers; `any` receives
          an anonymous linked_call_frame_storage slot
```

Narrowed once more, later the same day, and the earlier reading was wrong.

`compute_lines` in `glsl_deployment_strategy.py:1697` emits a `PlanLine` for
**every** node in `region_nodes`, so an op is not skipped by dispatch. `get` is
missing from `region_nodes` itself -- it never joins a region. And a census
comparing authored nodes against realised SSA values does **not** flag it,
which means `get`'s value id *does* exist as an SSA value somewhere: the value
survives, the operation does not.

So this is not a missing `Handler` entry and not an unhandled-op drop. It is
region membership. Look at how `deployment_nodes` is selected
(`glsl_deployment_strategy.py:5660`, fed from `node_ids`) and why a mapping
lookup is not considered part of the numeric compartment.

A census of "authored node with no SSA value" was written and **deliberately
removed**: it produced false positives for `items`/`loopresult`, which are
resolved away into the keyed-mapping slots on purpose, and it missed `get` for
the reason above. Do not re-add that shape of check without solving the
replaced-vs-dropped distinction first -- a diagnostic that reports false drops
and misses real ones is worse than none.

### The render is the other cost, and it is not the compiler's fault

JPEG encode per video frame, measured:

```
grid  24: rgb 0.09 ms   jpeg  380 ms
grid  64: rgb 0.15 ms   jpeg 1752 ms
grid 128: rgb 0.39 ms   jpeg 7030 ms
```

Against ~2.2 s of physics per frame at 24^2. Building the RGB is free; it is
entirely the encoder, and at 128^2 it costs several times the simulation. It
feeds nothing back into the physics, so it can be strided and moved to a worker
thread without affecting a single result. Not yet done.

## 6. Repository gotchas

- `turing/` is its **own git repo**; the parent ignores it via `/*/`.
  `git log -- turing/` from the parent shows nothing. Branch here is
  `codex/recursive-reduction-bridge`.
- **Never `git stash` a file in `turing/` to test "does this fail without my
  change?"** It reverts to that repo's HEAD and discards the large uncommitted
  campaign, so you get a false answer. Neutralize the specific edit instead.
- The live-growth path **edits source during builds**. `autogenesis.py` and
  `influence_field.py` both changed on their own this session. Check
  `git status` after a build before assuming an edit is yours.

### Known-failing before you start

- `test_ir_sequence_tables::test_compiled_retained_loop_mutates_caller_sequence_record`
- `test_precompile_to_ssa::test_whole_object_region_signature_preserves_planner_value_shapes`

Both predate this session, confirmed by neutralizing the specific edit.
`test_ssa_llvm_backend::test_native_sgd_wrapper_...` is flaky in large batches
(Windows DLL reuse) and passes in isolation -- one red result there is not
reliable on its own.

### Verifying compiler work

```bash
python -m src.compiler.symbolic_fluid_direct_control --worker --output <dir>
```

~30 s, ~0.5 GB. Then emit every function and count shortfalls. **Compile and
run an artifact** -- `compile_artifact` + `prepare_artifact_execution` -- do not
trust a shortfall count alone. Four silent miscompiles were found this session
that reported success while producing wrong code.

---

## 7. Commits

In `turing/`:

- `3555dfa` walk a keyed mapping as its own key and value vectors
- `1566b59` relax influence transport instead of enumerating paths
- `0523efe` key SymPy autogenesis SSA components by value id (machine-applied;
  the defect is real and hand-confirmed, but **its tests were not run**)
- `38213de` keep a keyed mapping's slot correlation frame-local
- `4abc962` checkpoint of the symbolic-fluid compiler campaign

In the parent:

- `ce325c0` record the LLVM lane's 256 -> 10 shortfall reduction

Everything after `3555dfa` -- the trig solvers, the libm removal, the forcing
removal, the voice coil, the AVI and audio-rate wiring -- is uncommitted.
