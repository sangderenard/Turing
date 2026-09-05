# Spectral Green's-function reduced-order tire model — action plan

Status: Phase 1 starting. 2026-09-04.

## Goal

Replace the full triangle-mesh tire simulation, for everything except the
contact patch itself, with a compiled, cheap, real-time reduced model: one
linear state-space operator per circumferential Fourier mode, identified
from the fine mesh as ground truth, coupled to the existing gas law at
mode 0 and to a local contact-patch reaction kernel (separate work, not
started) at whichever ring is currently in contact.

This is a Green's-function model: `G_m(omega)` (or, in the mode's own
small state space, `(M_m, C_m, K_m)`) maps a force applied at one ring, at
circumferential mode `m`, to the displacement it produces at every ring,
at that same mode, given tire state `s` (pressure, load, ...). The fine
mesh is the *experimental apparatus* that measures `G`; it is not present
in the compiled runtime model at all once this is built.

## Non-negotiable constraints (carried over from the session that produced
this plan -- do not relitigate these, they were hard-won)

- Every native kernel goes through the sanctioned pipeline: SymPy law ->
  the compiler's own AbstractTensor-stage materializer -> `lower_ast_
  source_to_ssa` with a batch `ExtractionContract`. No `compile_ast_aot`
  / AOT capture, no lane loops over the batch axis, no bespoke
  AbstractTensor edits (only translation-preserving ones, and only if
  truly necessary).
- **No full build (`build_vehicle_validator_native.py`, ~6+ minutes) until
  each new piece here is validated standalone**, at small scale, in
  seconds, against real numbers. Use `inspect.getsource` on real functions
  for repros, or author a real small SymPy law and lower it directly --
  never a fabricated stand-in function that only approximates the real
  shape. (A repro that "looks like" the real bug but isn't built from the
  real functions is not a valid test; this was learned the hard way.)
- A law is not "done" when its eager AbstractTensor stage works. It is
  done when it also lowers cleanly through the real pipeline (`lower_ast_
  source_to_ssa`) -- "come out valid on the other side, ready to be
  compiled" is the bar for every new SymPy law this plan produces.
- Extend, never shrink, when an ABI/call shape needs to change to make
  something compile. (See the harmonizer track, below.)
- Verify numerically against an independent reference before declaring
  anything correct -- "it ran without an error" is not sufficient.

## A separate, already-in-progress track: the compiler harmonizer

Unrelated to this plan's content, but relevant to its "no full build yet"
rule: the same session found and is fixing a real compiler defect (call
argument shapes silently misaligning across two callees of one record
parameter that materialize different field subsets). Two new passes exist
in `src/compiler/fortran_c_shell.py`: `_propagate_record_field_demand`
and `_harmonize_call_argument_shapes`, both called early in
`_class_surface_ssa_program`. That work is ongoing (real build attempts
have gotten progressively further, most recently past the original
blocking case to a *different* instance of the same class, in
`_no_exchange_observed -> _scalar`). A future agent should check
`tools/repro_targets_expansion.py`, `tools/repro_no_exchange_observed.py`,
and the running/most-recent full-build log before assuming that work is
finished or stalled. **Do not conflate the two tracks**: a full build
failing on the harmonizer issue is not evidence against anything in this
plan, and this plan's own testing must not depend on the harmonizer track
being finished (Phase 1-2 below use only already-compiled, already-
validated native law kernels; no new SSA lowering is needed until Phase 4).

## Phase 1 -- empirical Green's-function extraction (data generation, no
new compiled artifact)

1. Reuse the real, already-validated tire program (`balloon_tire_python_
   program`) and the native law stand-ins already built and verified
   today (`TURING_LAW_NATIVE=llvm`: gas, membrane, bead). No new
   compilation needed for this phase.
2. Settle the tire to a real equilibrium at a chosen state `s` (pressure,
   load) -- reuse the settling logic already proven in the scratch script
   `tire_alone.py` from earlier tonight (dead-still at ambient charge,
   confirmed).
3. Excite the tread ring with `f(theta) = A * cos(m * theta)` (a pure
   spatial mode, not a point impulse -- this is what lets the diagonality
   question in Phase 2 be answered directly rather than inferred).
4. Run N steps at the tire's own measured critical dt (`vehicle_balloon_
   tire_stability.estimate_tire_critical_dt`, already built), well under
   the stability limit.
5. Record radial (and, once mode 0 works, axial/tangential) displacement
   at every ring, every step.
6. FFT spatially (over the circumferential angle, at each ring, each
   step) using `AbstractTensor.rfft` (already a real primitive). FFT
   temporally over the recorded series to get `u_hat(ring, m, omega)`.
7. `G(ring', ring, m, omega) = u_hat(ring', m, omega) / f_hat(ring, m,
   omega)`. Save to disk (a real artifact, not held only in memory) keyed
   by `(m, s)`.

First concrete deliverable: `tools/spectral_greens_impulse.py`, single
mode, single state, small ring count, real numeric result compared for
sanity (does the response look like a real damped oscillator's transfer
function -- resonance peak, correct DC value, decays at high frequency).

## Phase 2 -- diagonality / mode-coupling verification

Using the same recorded data: does exciting mode `m` alone produce output
energy at `m' != m` beyond a real noise floor? Decide, with the evidence
recorded in this file (update it, do not re-derive the decision later
from memory):

- pure diagonal (scalar `G_m` per mode), or
- small dense per-mode matrix state (`M_m, C_m, K_m` with a handful of
  channels: radial, axial, circumferential, sidewall L/R, `delta_V`), or
- banded cross-mode coupling if diagonal fails outright.

## Phase 3 -- rational fitting (Vector Fitting) to `(M_m, C_m, K_m)`

Fit a stable state-space realization to the measured `G_m(omega)` samples
per mode (Vector Fitting -- the standard rational-approximation algorithm
for exactly this problem, used throughout power-systems/RF engineering;
do not hand-roll a raw least-squares fit on the matrix entries, it will
not respect the stability constraint). Sanity check: every pole has
negative real part; fit residual is small across the sampled band.

## Phase 4 -- author the per-mode SymPy law (the first NEW compiled
artifact this plan produces)

1. One small, symbolic law: given the CURRENT mode state and the ALREADY-
   FITTED, ALREADY-DISCRETIZED `(A_m, B_m)` (the continuous `(M_m, C_m,
   K_m)` turned into a discrete-time update via matrix exponential,
   computed offline in plain numpy -- the compiled law consumes `A_m,
   B_m` as ordinary runtime data, it does not re-derive them
   symbolically), the law computes `q_m[t+dt] = A_m @ q_m[t] + B_m @
   f_m[t]`.
2. Lower via `lower_ast_source_to_ssa` + a batch `ExtractionContract`
   (batch over modes and/or lanes), exactly the pattern in `src/compiler/
   native_law_kernels.py`.
3. Validate small-scale and fast (`SSAReferenceEvaluator`, or a tiny
   native compile at a batch of a handful of modes) against a plain-numpy
   reference of the same linear update, before anything else touches it.
4. Only once that passes: wire in as another stand-in via the existing
   `native_law_kernels.py` pattern.

## Phase 5 -- coupling to the gas law and the (separate, not-started)
local contact kernel

- Mode 0's `delta_V` channel feeds the gas law that already exists and is
  already compiled and validated (`compile_balloon_gas_ssa`). No new law
  here -- this is wiring, not new physics.
- The contact ring's `f_m` each step comes from the local reaction kernel
  discussed earlier in the session (not yet started); away-from-contact
  rings see `f_m = 0` (free response) each step. The Green's-function
  model is the far-field propagator; the local kernel is its boundary
  source term. Do not try to make the spectral model represent contact
  directly -- it is a linear model and contact is not linear.

## Phase 6 -- integration test, small scale first

1. Single mode, single ring, no contact: does the spectral operator
   reproduce a *known* free-vibration decay against the fine mesh, at a
   scale (seconds) that can be checked every iteration?
2. Only once solid: multi-mode, then multi-ring, then -- much later, and
   only after this whole plan's pieces are each independently validated
   -- a full build.

## Open questions, carried forward rather than decided here

- Mode truncation `M_max`: expect small (low modes should dominate) but
  this is exactly what Phase 2's data answers; do not guess a number.
- State-bucketing for `s` (pressure, load): discrete lookup table vs.
  continuous interpolation of `(M, C, K)` vs. `s`. Undecided.
- Whether "which ring is in contact right now" is computed by the
  reduced model itself or supplied by the existing contact solver.
  Undecided -- likely the latter, to keep the spectral model itself
  linear.
