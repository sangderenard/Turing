# Continuation report — 2026-09-03 (session cut by usage limit)

Branch `nogodsnomasters`. Everything below is committed by the commit that
adds this file. One process was left running (see "Running processes").

## Rules stated by the user this session (binding)

- ONE program, THREE stages: authored Python -> AbstractTensor Python -> native.
  The AbstractTensor stage is REQUIRED, never an option. lambdify is a sympy
  product, proof-only. There are NO scalar cores; execution is batched only;
  the per-scalar C/LLVM/Fortran symbolic emitters are legacy comparison tools.
- Nothing "interprets" a sympy law except the compiler: it ingests sympy into
  the process graph/SSA and defers to tensor operators. A hand-written
  `PythonCodePrinter` subclass is a second interpreter and is the wrong tool.
- Never run a build/lowering without approval (this session's plan WAS
  approved: "run whatever compile processes exist to get through this plan").
- No broad test sweeps; verify with seconds-long repros.
- Fix identities in the authored source / compiler, not intermediaries.

## Done and verified (all committed)

1. dt system: recursion -> loop retry with float64-floor exhaustion; `rollback=False`
   no-save lane; `max_iters` outer cap (non-throwing); energy/power time-scale pin
   (`Targets.energy_exchange_fraction`, channels `energy_j`/`power_w`), no-exchange
   hold, pre-trial `state.dt_limit_hint()`; shadow-trajectory amplification
   (`dt_system/shadow.py`, `Targets.shadow_growth_max`); general damping release
   (`dt_system/damping_release.py`).
   ROOT CAUSE of the tire "explosion": `coerce_metrics` float()-truncated
   tensor-valued metrics (max_vel < 1 m/s read as 0 -> unbounded CFL growth).
   Fixed in `dt_scaler._scalar`. Tire now completes windows in both lanes,
   zero rejections, lanes bit-identical (`tests/dt_system/test_tire_no_restore_windows.py`).
2. Compiler: persistent source-digest-keyed symbolic program cache
   (`symbolic_equation_compiler.compile_symbolic_program` /
   `symbolic_equations_cached`); sympy construction now runs once per file
   revision (10 min once, then instant). Python-material path reads NO build
   artifact; `--rate-hz` added (derived 1024; the managed-tire bundle was built
   at 120 and its manifest is STALE: 128 vs 144 vertices, 378 vs 382 edges).
3. Authored-source identity fixes: energy diagnostics mass broadcast;
   member-material fracture gate `_positive` -> `Max` (exact);
   `abstract_ui_vehicles._hard_positive` -> `Max`; gather backward accumulates
   repeated indices (tire VJP now matches finite differences).
4. LLVM: `Tanh` as declared libm call (vehicle body now emits, 0 shortfalls).
5. Parity harness `tools/frame_parity.py` + `tests/test_frame_parity.py`:
   roller fixture bit-exact python/C/LLVM/Fortran over 64 fed-back frames;
   member material ~1e-14 after the gate fix. Vehicle-body run: python+C built,
   LLVM stopped on Tanh (now fixed, NOT re-run).
6. Jacobian pre-check verdict: the tire step map is not usefully linearisable
   at rest (autograd vs FD power iterations disagree 1e4x); a finite shadow
   perturbation decays; shadow.py is the mechanism (1 extra forward/step).

## Working-tree state at this commit (READ THIS FIRST)

- `src/compiler/vehicle_python_compilation.py`: `vehicle_python_runtime_bindings`
  now binds the sympy laws through `symbolic_abstract_tensor_source(...)`
  (cse + `_abstract_tensor_python` from abstract_ui_vehicles) — the lambdify
  runtime path is REMOVED. With the ORIGINAL printer (restored per the user's
  objection) this source contains numeric receivers (`(0).maximum(x)`,
  `(3).sqrt()`, `math.pi`) and RAISES at the first call for the member material
  and vehicle body. So `--python-material` is currently BROKEN at runtime.
  `tests/test_symbolic_abstract_tensor_stage.py` is the pinned target (it passed
  only with the now-reverted printer patch).
- The correct fix (next step 1): produce the AbstractTensor stage from the
  compiler's own SSA -> Python emission instead of any sympy printer. Candidates
  found, not yet read: `src/compiler/ssa_python_materializer.py`,
  `src/compiler/fused_program_python_backend.py` (`compile_single_region_python`,
  dialect tables; used by `fortran_fidelity.verify_fortran_module`).
- `tools/run_vehicle_native_assembly.py`: manifest reads removed, `--rate-hz`
  added, no `--reference` switch (rejected).
- Known pre-existing failing test (not ours):
  `test_symbolic_equation_ssa_is_accepted_directly_by_llvm_and_fortran`
  (Fortran literal spelling `2.0_c_double`).

## Running processes

- Native dually build, started 12:50:47, pid 16920:
  `python tools/build_vehicle_validator_native.py --output build/vehicle_validator_dually_o0
  --assembly-profile dually-axle --contract deploy --optimization O0`
  Logs: scratchpad `native_build.log` / `native_build.err` under
  `C:\Users\alber\AppData\Local\Temp\claude\C--dev-Powershell\2b95b048-...\scratchpad\`.
  Last seen: still `[1/4] emitting compiler-owned C sections` after ~45 CPU-min
  (1.4 GB). Historical stage 1 was 8-13 min; this may be the deploy contract's
  outlining. Let it finish or kill it; nothing else depends on it except step 3.

## Plan (approved) and what comes next, in order

1. AbstractTensor stage from the compiler: emit Python for the symbolic law's
   SSA via the compiler's Python backend (AbstractTensor dialect), bind that in
   `vehicle_python_runtime_bindings`, make `tests/test_symbolic_abstract_tensor_stage.py`
   pass, delete `symbolic_abstract_tensor_source`'s printer dependence.
2. Re-run vehicle-body parity: `python tools/frame_parity.py --programs vehicle_body --frames 64`
   (LLVM now emits). C compile of the 250 KB body takes minutes; run detached.
3. When the native build finishes: extend frame_parity to compare N frames of
   the Python program against the DLL through the same feedback loop.
4. Headless Python validator run to surface remaining authored defects:
   `python tools/run_vehicle_native_assembly.py build/python_validator_run
   --assembly-profile car --python-material --headless-frame build/python_validator_run/final.png
   --rate-hz 120 ...` (create the dir first; slow: hours at eager speed).
5. Refinements: shadow first-step seed (unaligned random seed gave growth 8114
   once); persist compiled cores by cache identity for "run if available".

## Measurements worth keeping

- Cold tire, managed lane: ~166 substeps per 7.8 ms (window 1/1024) and 171 per
  8.3 ms (window 1/120), median dt 2.6e-5..3e-5 either way; damping release
  factor 20 over 20 ms barely changes counts (stiffness-bound, not
  transient-bound). Native fixed shell: 48 microsteps/tick -> 2.0e-5 s at 1024 Hz,
  1.7e-4 s at 120 Hz (why "the air sim fails at 120").
- Eager tire forward 150 ms (batch 1); forward+VJP 4.3 s.
