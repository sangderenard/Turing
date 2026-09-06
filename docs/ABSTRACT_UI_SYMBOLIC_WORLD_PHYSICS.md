# AbstractUI symbolic world physics

The first world-physics program is eight simultaneous SymPy equations. SymPy
is the numerical authority; the repository translates the expressions through
its canonical ProcessGraph and SSA representation and emits a WebAssembly
module with a published scalar-arena ABI.

For each axis `i ∈ {x,y,z}`, one step first computes implicitly damped velocity
and a semi-implicit trial position:

```text
d       = 1 + linear_drag Δt
v*i     = (vi + Δt (gravity_i + force_i inverse_mass)) / d
qi      = pi + Δt v*i
```

An AABB boundary contributes a signed unilateral penetration correction:

```text
ci      = max(0, minimum_i + radius - qi)
        - max(0, qi + radius - maximum_i)

p̄i      = qi  + contact_softness ci
v̄i      = v*i + contact_softness ci / Δt
```

At `contact_softness = 1`, a floor or wall projects the body back to its valid
boundary and removes the violating velocity in one step. Lower values produce
compliant recovery. This is intentionally an adequate game-world contact law,
not a claim of material-accurate deformation.

Interior walls use the same compiled transition. The mesh realization emits
one collider for every actual wall segment and elevated lintel; door and gate
intervals remain absent rather than receiving invisible blocking boxes. The
host chooses the contacted face and publishes its outward violation normal
`n`, plane coordinate `h`, and semantic wall identity. The additional compact
constraint is:

```text
wall_penetration = obstacle_active max(0, n · p̄ + radius - h)
p̂                = p̄ - contact_softness n wall_penetration
v̂                = v̄ - contact_softness n wall_penetration / Δt
```

Only planar `x/z` wall normals are selected for the present capsule-like
player. Lintels participate only when the player's vertical sphere overlaps
their real height. The selected collider carries both its authored semantic
part identity and dense runtime part ID into telemetry and future contact
records.

Boundary traversal uses a source-relative planar rotation followed by target
translation. For yaw matrix `R(cos θ, sin θ)`:

```text
pT = portal_target + R (p̂ - portal_source)
vT = R v̂

p_next = (1-a) p̂ + a pT
v_next = (1-a) v̂ + a vT
```

`a = portal_active` is normally zero or one. Fractional values are defined and
can animate a transition or cutscene. The current browser host leaves it zero
until a traversal event supplies a destination representation.

The six state outputs are accompanied by contact penetration and specific
kinetic energy metrics. All coefficients, bounds, portal anchors, and portal
yaw components are ordinary WASM inputs. The living map renders identified
number inputs for the editable parameter surface and persists their values with
the other representation edits; changing one affects subsequent physics steps
without recompiling.

The compiled reference currently binds gravity and the outer player boundary
to the browser entity cycle. Static-object welding, candidate generation,
per-wall contacts, and representation replacement remain later selected
stages. Soft bodies should be a separate archetype-selected equation program—
not a tax silently imposed on every building and source-code object.
