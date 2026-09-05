# Tire fidelity ladder — action plan

Status: `reduced` law authored and independently verified. 2026-09-04.
Supersedes the sequencing in PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md:
that document's phases (empirical Green's-function extraction) are now the
**`fine` -> `spectral` -> `green`** work inside this ladder, not a
standalone track. Its non-negotiable constraints (sanctioned pipeline only,
no full build without validating small first, verify numerically, extend
never shrink) still apply everywhere in this document.

## Why this exists

The rig/pillar/roller physics and the fine deformable-mesh tire physics
share one authored program (`vehicle_graph_tick_vector` /
`balloon_tire_vector_step`), run identically by the Python host
(`tools/run_vehicle_native_assembly.py`) and, once compiled, the native
host. Today there is only one tire fidelity: the full mesh, every step,
for every purpose. That makes the full detailed simulation a mandatory
first step for *everything*, including ordinary rig/pillar work that
doesn't need it — and the full mesh is genuinely slow (eager-Python
per-step cost when run through the Python host; a long one-time compile
when run through the native host). The fine mesh is real, valuable
ground-truth work, but it belongs to *training* (building the spectral and
green models), not to ordinary runtime.

## The four modes

One shared input selects the mode; both hosts read it identically because
it's part of the authored program, not host logic.

- **`fine`** — today's real deformable membrane mesh. Ground truth.
  Reserved for the offline runs that build `spectral`/`green` (training),
  never a required gate for ordinary rig/validator work.
- **`spectral`** — the raw, not-yet-fully-fitted circumferential-mode
  capture from PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md's Phase 1/2. Used
  during training, and as a live mode once enough modes are captured to be
  useful even before the final rational fit exists.
- **`green`** — the fitted Vector-Fitting state-space Green's-function
  model from that same plan's Phase 3/4. The preferred live mode once
  fitted and validated.
- **`reduced`** — the failure-safe fallback, always available, no
  dependency on any fitted data. Real physics from the tire's own rest
  geometry (see below), not a placeholder.

Live/interactive selection preference (best available wins):
`green` > `spectral` > `reduced`. `fine` is never selected automatically;
it is only ever run deliberately, by name, for training.

## Normalization contract across modes

Switching modes at runtime (as green/spectral become available, or as a
quality setting) must not visibly pop. All four modes are required to
agree on the tire's effective footprint under a given load: contact patch
extent, deflected height, and enclosed volume, all derived from the *same*
authored rest geometry (the same ring stations `vehicle_tire_ring_model.py`
already uses for the Pappus volume law) rather than four independently
tuned models. `reduced`'s contact-patch-area integral and the Pappus ring
volume already share that same station data, which is what makes this
enforceable rather than aspirational; `spectral`/`green` must be fit
against the *same* fine-mesh runs whose static/mode-0 response is checked
against that same ring geometry (mode 0's `delta_V` channel, already
planned to feed the existing gas law in Phase 5 of the other document).

## Architecture correction: separate function, not a flag inside one

Earlier text in this document (and an earlier implementation pass) treated
`reduced` as an additive force layered on top of the still-fully-running
fine mesh, selected by a flag inside `balloon_tire_vector_step`. That is
wrong and has been undone. A flag branch inside one AbstractTensor-authored
function cannot skip the expensive membrane/bending/bead-implicit work
eagerly (`.where()`-style selection still evaluates both sides), so it
cannot deliver the actual point of the reduced modes: real speed, and
`fine` reserved for training only.

The corrected architecture, now landed:
- `balloon_tire_vector_step` (unmodified, restored to its original form) --
  `fine` mode only, used only for deliberate training-data generation
  (docs/PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md Phase 1), never a
  default live path.
- `balloon_tire_reduced_vector_step` (new, separate function, same call
  signature for drop-in host dispatch) -- modes 1/2/3, called by the host
  *instead of* the mesh step, never both. Runs no membrane, bending, or
  bead-implicit solve at all: the tire mesh is purely kinematic (rigidly
  follows the hub, exactly like `balloon_tire_vector_initialize`), and
  `rim_force`/`rim_moment` come directly from whichever contact law
  `tire_fidelity_mode` selects.
- The host (`vehicle_native_graph_program.py`'s tire recurrence, and any
  eager test harness) must call one function or the other based on mode --
  not implemented yet, see Sequencing below.

This also incidentally fixed the standing stability bug: with no
bead-implicit solve for the reduced force's `free_velocity` to perturb,
all three modes ran finite/stable at 300 steps with no special dt handling
required. The bead-implicit solver's interaction with a large forced
input was confirmed as the real root cause of the earlier instability, not
the force magnitude itself.

## Three selectable reduced-family variants, kept side by side

Per explicit instruction: keep all of them selectable for comparison/
tuning, don't collapse to one "winner" yet.

- **mode 1 -- reduced** (contact-patch integral): the original design,
  `vehicle_tire_reduced_contact_law.py`'s real ring-geometry integral,
  `force = pressure * contact_patch_area`. No moment (radial force at the
  bead ring only).
- **mode 2 -- wrench**: bare-bones spring-damper from the hub to the
  nearest ground/roller surface, one-sided (engages only on boundary
  penetration against the mean shoulder radius). No moment arm (single
  query point at the hub).
- **mode 3 -- wrench-per-vertex**: the same spring-damper evaluated
  independently at each bead vertex. Gives a real moment for off-center
  contact (the force naturally varies by azimuth), still far cheaper than
  the full membrane.

Modes 2/3 share `wrench_spring_n_per_m` / `wrench_damping_n_s_per_m`,
seeded from the already-tuned `bead_stiffness_n_per_m` /
`bead_damping_n_s_per_m` as a starting point, not a fitted value --
mode 3's first verification run gave an unrealistically large force (~38M N
for the same test case mode 1 gives ~22kN for) because the same lumped
stiffness applied per-vertex across 32 springs in parallel is far too
stiff. Real tuning (by hand, or the explicitly-planned-for-later ADAM fit
against a `fine`-mode reference) is expected to move these; they are
deliberately parameterized for exactly that, not treated as finished
values.

## `reduced` (mode 1) — done, verified

`src/compiler/vehicle_tire_reduced_contact_law.py`: at axial position `z`
the rest cross-section (from the same four ring stations already used for
`vehicle_tire_ring_model.py`'s Pappus volume law) is a solid disk of radius
`r(z)`, piecewise-linear across the three station segments. Where that disk
extends a depth `H` below flat ground, its chord width is
`2*sqrt(r(z)**2 - H**2)`; integrating that along `z` gives the real
contact-patch area implied by the tire's own rest geometry, and
`force = pressure * contact_patch_area` is the standard pneumatic-tire load
relation, not a tuned spring. The axial integral is unrolled as fixed
5-point Gauss-Legendre quadrature per segment (same pattern as the
Goertzel/ring-volume laws already in this compiler).

Verified in `tools/verify_reduced_contact_law.py` against an independent
`scipy.integrate.quad` reference: matches to ~1e-13 relative error across
the realistic compression range (0 up to ~30% of the shoulder radius).
Documented, not hidden: accuracy degrades (~1e-3 relative error in the
verification fixture) once compression depth approaches a *bead* radius —
the tire compressed nearly onto its own rim, well outside where this
fallback should ever be trusted. A caller must clamp compression depth well
inside the tire's own station radii.

Done: authored in `balloon_tire_reduced_vector_step` (see the architecture
correction above), gated by `tire_fidelity_mode==1`, using 8
`ring_*_r_m`/`ring_*_z_m` scalar parameters (appended to `parameter_names`,
not new function arguments -- extend, never shrink). First-step force
magnitude matches hand calculation (~21-22kN for the verification
fixture's geometry/pressure) and, since the architecture correction removed
the bead-implicit solve this force used to perturb, 300 steps run finite
and stable with no special dt handling -- the earlier "known real issue"
entry here (step-0 impulsive instability, suspected bead-implicit
interaction) is resolved as a side effect of the function split, not
patched around. The `--tire-dt-fraction` interim workaround that used to be
here is no longer needed for this reason and should not be recommended
going forward; `tools/run_vehicle_native_assembly.py
--tire-fidelity-mode reduced` needs no special dt tuning once host dispatch
(Sequencing, below) lands.

Still real, still open: no time-of-impact ramp-in on first contact (a
sudden large step-0 force is still physically abrupt even though it's no
longer numerically unstable), and `estimate_tire_critical_dt` still only
perturbs position, not velocity/damping-coupled modes -- both remain
correct follow-ups, just no longer urgent for basic stability.

## Damage metric and burst LUT (captured, not yet started)

Off-road/abuse conditions will legitimately drive compression past the
"realistic" range this document's `reduced` law is accurate in — and in
reality that's exactly when tire fabric/cord damage and bursts happen. That
needs a damage metric shared across all four fidelity modes (so the same
real abuse causes the same failure regardless of which mode is currently
live — the failure-mode analogue of the footprint normalization contract
above), plus an offline-built lookup table of real burst behavior so no
mode ever has to simulate fracture physics live:

- **Metric.** `fine` mode already carries real per-face material state
  (`face_material`'s lamé/damping fields in `vehicle_balloon_tire_program.
  py`); its natural damage metric is peak fabric/cord strain against a real
  failure strain. `reduced`/`spectral`/`green` have no per-face data, so
  each needs its own proxy (for `reduced`: compression depth as a fraction
  of shoulder radius is the obvious candidate) calibrated so it crosses the
  same real threshold `fine` would under equivalent load — not an
  independently tuned number per mode.
- **LUT.** Sweep deliberate `fine`-mode failure runs offline (impact speed,
  angle, load, obstacle sharpness), capture the resulting burst deformation
  sequence, key it by the damage-metric state and a few cheap descriptors
  at the moment of failure. Any live mode, on crossing its own version of
  the threshold, plays back the nearest matching captured sequence instead
  of simulating fracture.
- **Sequencing.** This depends on `reduced` actually being wired into a
  live mode-selected path first (next section) — there is no live signal
  to attach a damage metric to yet. Comes after that, before or alongside
  fitting `spectral`/`green` (the LUT-building sweep reuses the same
  deliberate `fine`-mode training-run infrastructure that Phase 1 of
  PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md needs anyway).

## Rendering: keep the full visual mesh under non-`fine` modes

Whatever fidelity mode is driving the physics, the rendered mesh can stay
the full detailed one via a cheap polar remap, not a second visual model:
each rest vertex already has a natural `(theta, r, z)` around the spin
axis, so per vertex, hold `theta`/`z` fixed and rescale only `r` by
interpolating that vertex's rest radius against how much its nearest ring
station (the same `ring_*_r_m` stations `reduced` now reads) compressed
under the live low-dimensional state. Fully vectorizable, no
re-simulation, and the same "skin a detailed mesh to a simple rig" idea
skeletal animation already uses. Depends on `reduced` (below) actually
producing a live per-station deflection to remap against; not started.

## Sequencing

1. **Done**: author and verify modes 1/2/3 standalone, as the separate
   `balloon_tire_reduced_vector_step` function (this document's previous
   sections). Eager execution only, no compiler/lowering touched.
2. **Next, not started**: host dispatch. `tools/run_vehicle_native_assembly.
   py`'s `--tire-fidelity-mode` flag currently sets the `tire_fidelity_mode`
   input but nothing reads it at the call site -- `_PythonVehicleMaterial`'s
   tick path (and `vehicle_native_graph_program.py`'s `vehicle_tire_
   recurrence`, which calls `balloon_tire_vector_step` by name) needs to
   call `balloon_tire_reduced_vector_step` instead when a non-fine mode is
   selected. This touches the same call graph the compiler harmonizer work
   (Thread B) is already fighting with, so: validate with a small direct
   eager-execution repro first (as already done for the standalone
   function), and if/when this needs to go through `lower_ast_source_to_ssa`
   for the native path, use a small targeted lowering repro
   (`inspect.getsource`-based, seconds) before touching a full build --
   never a full build without explicit approval.
3. Once host dispatch works, run `tools/run_vehicle_native_assembly.py`
   with each of modes 1/2/3 against the dually profile and confirm ordinary
   rig/pillar work no longer pays fine-mesh cost, and compare them visually/
   numerically against each other.
4. Tune `wrench_spring_n_per_m`/`wrench_damping_n_s_per_m` (currently
   seeded from the bead spring, confirmed too stiff for mode 3) -- by hand
   first, ADAM-against-a-`fine`-mode-reference once that's wanted.
5. Only then resume PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md's Phase 1
   (`fine`-mode training data collection) as an explicit, separate,
   deliberately-invoked pass — never a default gate again.
6. Fit `spectral`/`green` per that document's Phase 2-4, wire the
   preference cascade (`green` > `spectral` > `reduced`) into mode
   selection.

## Native-shim microstepping (captured, not started)

`vehicle_tire_recurrence`'s microstep loop (`for microstep in range(
microstep_count):`) is Python-level, so even the cheap reduced kernels
still pay full eager AbstractTensor dispatch overhead once per microstep,
not once per outer step. The fix is not to encode the repeat count into the
authored control flow itself, but a native shim: a small compiled wrapper
taking a repetition count and looping internally (cheap per-iteration once
compiled), called once from Python instead of N times. Fits the existing
`NativeLawStage`/native-kernel pattern (`native_law_kernels.py`) rather
than requiring new infrastructure. Not started -- real follow-on work,
separate from the fidelity-mode dispatch above.

## Do not repeat

- Do not build a second, bespoke rig/pillar/roller path outside
  `vehicle_native_graph_program.py` / `tools/run_vehicle_native_assembly.py`
  — see `CONTINUATION_DUALLY_FULL_VALIDATOR_2026-09-02.md`'s explicit "do
  not make a tire-only one-wheel validator / pretend validator /
  replacement pillar" rule. (Violated once this session; reverted.)
- Do not let `fine` mode become a mandatory step again, for any purpose
  other than deliberately training `spectral`/`green`.
- Do not tune `reduced`'s force law by hand-picked constants divorced from
  the tire's own ring-station geometry; if it needs a correction term,
  derive it from the same authored geometry or from a fit against real
  `fine`-mode runs, and verify numerically before trusting it.
