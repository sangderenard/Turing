# Collision tied to the integrator: catalogue methods and a dt-graph fixed-point cycle — design

Date: 2026-09-29. Status: **design for review, nothing implemented.**
Companion to `docs/UNION_TYPE_DESIGN_2026-09-29.md` in form. Source locations
are cited by function name; line numbers drift.

## 0. Direction (user, condensed)

- Nothing outside the honorary equations for collision. Any good CS collision
  method described in pure math goes into the appropriate honorary section,
  as a sympy symbolic form plus its solver — as many as are good.
- For the Woodshop: the collision system runs **tied to the integrator**; the
  two communicate within the substep for a real resolution. Advanced
  collision is its own integrator, so this is a **fixed-point cycle in the dt
  graph**, a capability the graph may define. Modular participants, one
  control.
- Forces are never integrated outside a force-integrator step; steps stay at
  a low enough order to be continuous (the controller subdivides).
- Force transfer through the encountered material (beam / sediment / fluid by
  declared nature) is a response method among the options, chosen by the
  material, not an invented contact stiffness.

## 1. What exists (from three read-only maps)

**Catalogue.** A flat script of sympy globals `eq_<PREFIX><section>_<n>`
discovered by regex and grouped by `ENGINE_PREFIXES` (Newton, Timoshenko,
Faraday, …, Poncelet, Woodshop). Sections are added by a function-scoped
`_expand_*()` returning an `eqs` dict, installed with `globals().update`.
A law is `sp.Eq(out, rhs)` with a source comment; validity is a separate
`LawScale(regime, laws, valid_if, …)`. No entry is a *procedure*: the
nearest patterns are stage equations with explicit iterate symbols —
`eq_NS14_6` (`u^{n+1}`), `eq_MX3_1` (`S_prev`), XPBD `eq_N11_4..6` (the
per-constraint Δλ/Δx of one Gauss-Seidel sweep, without the sweep) — with the
loop owned by the consumer (`newton_world_dt_pieces` composes
`momentum + dt·derivative` from `eq_N1_2`). `equation_piece` refuses
`Derivative`/`Integral`; `compile_sympy_equations` forbids an output as its
own input ("recurrence belongs to the caller"). Lowered constructs include
`Max/Min/Abs/sign/Heaviside/Piecewise→Select`, trig, exp/log, floor/ceiling.
N11 "game-physics constraints" already hosts a CS method (XPBD) under Newton.
No numerical-methods engine exists.

**dt graph.** `RoundNode.schedule ∈ {"sequential", "parallel"}`; sequential
pieces communicate through state columns in causal order (collision already
reads the integrator's proposed positions this substep); "parallel" defers
writes. No iterate-to-convergence construct: `run_superstep`'s `max_iters` is
subdivision of the window; its retry loop halves dt on rejection, never
re-runs the same dt. A nested round lands its parent's window or refuses
(`_RoundTransaction` restore and re-raise). `METRIC_FIELDS` has no residual;
`error_channels` overflow rejects/halves rather than iterates. **The piece
lane runs `run_superstep` with `rollback=False`** — the documented "no-save,
in-place, no retry" mode — by omission; `SuperstepPlan.rollback=True` is
consulted only by `run_superstep_plan`.

**Material solvers.** `frame_solver.py`: 12×12 Timoshenko element,
`reaction = K·u − f`, modes, per-component stiffness; `eq_T7_2` reduced
stiffness; `engine_mounts` solves deflection → reads force back. No
imposed-displacement entry. `feet`: punch stiffness `eq_T11_1` + bearing cap
`eq_JA3_6` (a complete granular response). Fluid: buoyancy/drag laws,
`fluids.py` (ρ, μ); no added mass, no box submerged volume. Materials declare
E/G/Poisson (`OrthotropicWood`) but not yield/crushing nor nature;
`MachinePart.material` is an unbound string. Timescale pattern: `craft_graph`
publishes `2/√(k_secant/m)`; `frame_solver.modes()` gives first modes.
Detection today: AABB of rotated corners + `argmin` — over-approximate, no
contact point.

## 2. Catalogue: the computational-collision methods

Every method below is declared as **stage equations** in the catalogue's own
pattern — one simultaneous map per stage, iterate symbols explicit
(`λ^{k}`, `λ^{k+1}`), a source comment, a `LawScale` where validity is
conditional — and the loop is owned by the dt-graph cycle (§3). Each is a
real pure-math implementation; the "solver" is the cycle running the stage.

### 2.1 Detection
| method | stage equations (symbolic) | notes / validity |
|---|---|---|
| Separating axes, oriented boxes | for each of 15 axes `a` (3+3 face normals, 9 edge cross products): `s_a = |(c_b−c_a)·a| − Σ_i h_{a,i}|a·u_{a,i}| − Σ_j h_{b,j}|a·u_{b,j}|`; `separated = Max_a(s_a) > 0`; `δ = −Max_a(s_a)`; `n = a*` (the maximizing axis, sign from `(c_b−c_a)·n ≥ 0`) | exact for convex boxes; replaces the AABB-of-corners test |
| Contact point / patch | face-face: clip the incident face against the reference face's side planes (Sutherland–Hodgman as `Piecewise` inside/outside per edge); edge-edge: closest points of two segments (closed form) | gives where the load lands on a member (§4) |
| GJK (convex distance) | support `s_A(d) = argmax_{x∈A} x·d` (box: `c + Σ h_i sign(d·u_i) u_i`); Minkowski support `s_{A−B}(d) = s_A(d) − s_B(−d)`; simplex step: `w^{k} = s_{A−B}(d^k)`, `d^{k+1} = −closest_point(simplex^k ∪ w^k)`; terminate `w^k·d^k ≤ tol` | for non-box convex parts later |
| EPA (penetration depth) | expand the terminating simplex: face `f*` closest to origin, `w = s_{A−B}(n_{f*})`, `δ = n_{f*}·w` when `n_{f*}·w − dist(f*) ≤ tol` | pairs with GJK |
| Conservative advancement (time of impact) | `t^{k+1} = t^k + d(t^k)/v_max` with `d` the current distance, until `d ≤ tol` or `t ≥ dt` | continuous detection; prevents tunnelling |

### 2.2 Response (constraint dynamics)
| method | stage equations | notes |
|---|---|---|
| Complementarity (Signorini) | `w = A λ + b`, `w ≥ 0`, `λ ≥ 0`, `λᵀw = 0` (N5 KKT already states this) | the problem statement |
| Projected Gauss–Seidel | `λ_i^{k+1} = Max(0, λ_i^k − (b_i + Σ_j A_{ij} λ_j^{k or k+1})/A_{ii})` | one sweep = one stage; the cycle repeats it |
| Sequential impulses (velocity form) | `Δλ_i = −(J_i v + b_i)/(J_i M^{-1} J_iᵀ)`, `λ_i^{k+1} = Max(0, λ_i^k + Δλ_i)`, `v ← v + M^{-1} J_iᵀ (λ_i^{k+1} − λ_i^k)` | Catto; the same fixed point in velocity space |
| Friction cone projection | `λ_t^{k+1} = clamp(λ_t^k + Δλ_t, −μ λ_n, +μ λ_n)` (box cone; exact cone as `Min(|λ_t|, μ λ_n)·λ_t/|λ_t|`) | Coulomb; `eq_N8_2/3` for regularised sliding |
| Baumgarte stabilisation | `b_i ← b_i + (β/dt)·δ_i` | drift correction *inside* the solve, never a teleport |
| Restitution | `v_n^+ = −e v_n^−` — `eq_N5_6` exists | Newton's law of restitution as the target |
| XPBD | `eq_N11_4..6` exist | position-based alternative |
| Convergence residual | `r^k = Max_i |λ_i^{k+1} − λ_i^k|` or the complementarity residual `Max_i |λ_i w_i|` | what the cycle tests |

### 2.3 Force-transfer responses (by declared material nature)
| nature | response | exists |
|---|---|---|
| structural member | `F = K_red·δ` at the contact DOFs (`eq_T7_2` from `frame_solver` component matrices; `eq_T19_2` closed form for a cantilever); yield cap `BR14`; timescale `first_hz` | K, modes, reduced-stiffness law: yes; imposed-displacement solve and contact-point→DOF mapping: **no** |
| granular | punch stiffness `eq_T11_1` + bearing cap `eq_JA3_6` (as `feet`) | yes |
| fluid | buoyancy `eq_N4_4`/`AR1`, drag `eq_N4_5`/`N11_7`, ρ/μ from `fluids.py` | laws yes; added mass and box submerged volume: **no** |

**Home for these.** Response methods extend **N11** (already "game-physics
constraints" under Newton). Detection needs a section; the catalogue names
engines after people, so the proposal is a new engine for geometric
detection — name to be chosen by the user (e.g. Gilbert, for
Gilbert–Johnson–Keerthi) — holding SAT, contact clipping, GJK/EPA and
conservative advancement. Every entry gets its `LawScale` (convexity,
rigidity, small-δ where relevant) and enters the markdown catalogue too
(N8/N11 currently exist only in the Python file).

## 3. The dt-graph fixed-point cycle

A new `RoundNode.schedule = "fixed_point"`, with a `CyclePlan(residual_metric,
tolerance, max_iterations)` beside `SuperstepPlan`. Semantics within one
controlled substep `dt`:

1. **Checkpoint** the state at substep start (`copy_shallow`, existing).
2. **Iterate columns** — the collision's accumulated multipliers `λ` (and
   any other declared iterate) — are *not* restored between iterations; they
   are the fixed point being sought. Declared per column.
3. Each iteration: restore the non-iterate columns to the checkpoint; run the
   children in causal order at the same `dt` (integrator proposes positions
   and velocities using the current `λ`; detection reads the proposal;
   response updates `λ` by one PGS/SI sweep); read the published `residual`.
4. **Converged** (`residual ≤ tolerance`): the last iteration's state stands;
   the round publishes as today.
5. **Not converged** after `max_iterations`: the substep is rejected
   (`ok=False`) and the controller halves `dt` — which requires the piece lane
   to honour `SuperstepPlan.rollback` (today it selects the no-retry lane by
   omission). The cycle's contract is `BIND`; its exchange time is the
   response's own (`2/√(k_secant/m)` pattern, or the member's first mode).

Implementation points: `dt_graph._advance_children` (new schedule branch);
`llvm_dt_system.interpret_round` (accept the schedule) and `piece_source`
(spell the bounded loop — plain Python with a data-dependent exit, which the
source compiler lowers; iterate columns spelled as columns the restore
skips); `METRIC_FIELDS` gains `residual` (reduced by max); `dt_system_over`
/ `dt_system` pass `rollback` from the plan. The cycle is opaque to
`run_superstep`, which sees only `(ok, metrics)` — legal and modular.

Woodshop's round becomes: detection (SAT + clipping over contact lanes) →
gravity → **cycle**[ N1.2 momentum with `force + Jᵀλ`, N1.1 position,
response sweep ] → publish. `_resolve_pair`, `_resolve_floor`, `_translate`
teleports and `_run_law` go.

## 4. What has no existing mechanism (consolidated)

- A convergence predicate and a `residual` metric that iterates instead of
  rejecting; re-invoking pieces within one substep at the same `dt`; iterate
  columns exempt from restore.
- Rollback in the piece lane (`dt_system_over`/`RoundPiece` pass none).
- Two lane spaces in one state (items, contact pairs) and gather/scatter-add
  across them; additive force composition (gravity assigns `force_z`).
- Pair geometry as a piece (variable part counts; today per-step Python).
- The catalogue section(s) themselves; N8/N11 in the markdown.
- Material nature keyed by `MachinePart.material`; wood yield/crushing; an
  imposed-displacement solve; contact-point → member DOF mapping; added
  mass; box submerged volume; explicit √(E/ρ) transit-time law.

## 5. Proposed order

1. **Catalogue, minimal working set:** SAT (15 axes) + face clipping for
   detection; PGS/SI sweep + friction cone + Baumgarte + residual for
   response; `LawScale`s; markdown entries. Then GJK/EPA, conservative
   advancement.
2. **dt system:** `fixed_point` schedule + `CyclePlan` + `residual` metric +
   rollback honoured in the piece lane; lane spaces + gather/scatter-add;
   additive forces.
3. **Woodshop:** contact lanes declared; detection and response as pieces in
   the cycle; the old resolution removed.
4. **Force-transfer responses** by material nature (granular first, it is
   complete; then the member response once the imposed-displacement solve
   exists; then fluid once added mass and submerged volume are declared).

## 6. Questions for the user

1. Section homes and names: response into N11 (existing), detection into a
   new engine — which name?
2. Cycle as a `schedule` value on `RoundNode` (proposed) or a distinct node
   kind?
3. Residual: multiplier change or complementarity residual as the default?
4. Iterate columns declared per column (proposed) or per piece?
5. Confirm the order in §5, or reorder.
