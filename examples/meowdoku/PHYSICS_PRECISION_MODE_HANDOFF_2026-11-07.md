# Particle physics precision modes — 2026-11-07 handoff

## Current contract

`fountain3d.py` is a soft-sphere DEM law. Its state includes position, velocity, and angular velocity (`wx/wy/wz`). Every contact uses a sphere radius and the same spring, slip, Coulomb friction, and rolling-torque equations. Each particle's moment is `I = 0.4 m r²`, regardless of its rendered mesh. Particle-particle friction and rolling coefficients combine by geometric mean; the analytic SDF chamber calls the same wall-contact equation with the particle's own `mu` and `roll`. The chamber does not carry a physical surface-material ID.

`mu0..mu3`, `roll0..roll3`, restitution, drag, and gravity are runtime inputs. Setting `mu` or `roll` to zero removes that force or torque from the result, but the compiled shader still evaluates the contact arithmetic. This is a behavior switch, not a performance switch. The law's `irr` varies radius by ID at run start; it is not directional shape irregularity. PBR/Phong roughness is optical and has no physics mapping. Studio separately saves physical `sim-mu` and `sim-roll` values and calculates the final mesh's center of mass and full inertia tensor. Current particle deployment consumes the supported scalar values but still uses spherical contact and scalar moment.

The existing compiler already demonstrates a real cost boundary: `fountain3d_build.py` produces distinct flat and SDF products because `make_law(..., sdf=False)` removes SDF work at trace time. Marble Madness selects the toy SDF product and parks all but one of its 4,096 slots. The toy product also contains the body variant. A runtime checkbox cannot remove compiled per-particle work from that product.

## Honest controls now

- Expose the existing friction and rolling-resistance uniforms for Marble Madness, dice, and grain experiments. Zero means frictionless or no rolling resistance. The controls change physics at a step boundary without rebuilding, but do not promise a faster kernel.
- The new visual spin stream may rotate an authored mesh from the law's actual angular velocity. That is still sphere-contact dynamics.
- Keep material optical roughness distinct from physical `mu` and `roll`. A physical material preset can explicitly specify separate contact coefficients and mixing rules once wall materials have physical IDs; do not silently reuse the PBR slider.

## Next compiled lanes

1. Measure current toy and many-grain products on the RTX 3060 with the existing contact diagnostics (`n_drop`, energy, speed) and a rolling-slip probe. Preserve baseline manifests and timings.
2. Add a **compile-time** contact-mode argument to `make_law`, alongside the existing `sdf`/`body` variants. Trace separate fast sphere and precise products through the single sanctioned compiler entry. The fast product must actually omit unwanted torque/rolling branches in generated WGSL; measure instruction count and frame time before claiming savings. Keep its manifest and state ABI explicit. WebGPU can build promptly; Wasm remains lazy until fallback need or idle time.
3. For directional mechanical qualities, use Studio's verified center of mass and inertia tensor as per-object inputs, add orientation to the law state, and integrate `I_world^{-1}(torque - omega × I_world omega)`. Test against the existing Studio rigid-roll solver and conservation cases. An on-page quaternion alone changes appearance, not this dynamics.
4. For explicit dice and grain shape contact, define a compiled particle-shape ABI and measured contact algorithm against both particles and SDF walls. The authored closed mesh and its marking relief are the geometry authority. Add physical wall-material IDs, pair mixing, and per-contact work diagnostics before surface-specific scuff accumulation. Compile a separate shape-contact product so large grain runs keep the fast sphere path.

No explicit-shape or tensor-inertia toggle should be shown as operative until its product, manifest, and validation exist.
