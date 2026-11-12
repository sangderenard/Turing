# SPH continuous-field publication, source only

Measured before repair on 2026-11-11: DiscreteFluid.sample_at returned
186.7980965587383 kg/m^3 rather than 373.5961931174766 at the midpoint of two
equal-mass supporting particles, and 320 K rather than their 280/320 K mean
300 K. Its repeated advanced-index += operations retained only the final
neighbor. Existing uniform-scalar interpolation and pair-density tests both
passed, so they did not expose this publication defect.

Changed only sample_at accumulation to np.add.at for density, normalization,
pressure, temperature, salinity and velocity. The existing compact poly6 law,
particle state, force law, integrator and controller are unchanged. New tests
check unequal supporting masses/densities, all fields, permutation invariance,
zero outside support, empty particles/queries and no physical-state mutation.

Validation: 9 tests passed in 3.01 seconds across test_discrete_interpolation,
test_discrete_fluid_density, test_density_permutation,
test_discrete_source_conservation and test_bath_adapter. Scoped git diff
--check passed. No full-suite, GPU, live bowl, SPH/Studio coupling, deployment
or atmosphere integration claim. The hazards reader refused existing malformed
entry names; newest ten receipts and the legacy opening were read directly.

Canonical integration findings: StudioVisibleBake -> StudioMeshBake ->
feature_packet/StudioMeshEnvironment retains actual source-mesh boundaries for
the incumbent Fountain3D contact law. The wheel already reacts to contacts via
equal/opposite hinge torque. Bath SPH sample_at publishes a continuous field,
but its simulator does not currently expose set_solids; its managed adapter's
optional registry handoff alone does not connect it to Studio mesh contacts.
HybridFluid.sample_at explicitly omits its particles. The current browser
liquid style sees only the display-packed position/color/radius/motion flag,
not full mass, density, velocity, phase or optical material identity.

Moisture laws already publish LWC/IWC/RWC and aerosol extinction in 1/m, but
relative humidity is not condensed water, mie_weight is not a complete phase
function, and extinction alone does not specify scattering albedo. No optical
coefficients were invented and no legacy atmosphere coefficients were changed.
The requested glass bowl, physical liquid and immersed marble remain an
integration acceptance target, not a delivered scene.

Broader targeted gate: 16 passed, 2 failed in 7.28 seconds. Both failures are
in test_bath_run_hooks (headless and OpenGL adapter supersteps): dynamic M
metrics lacks pub_exchange_time required by the current dt controller.
Both reproduced identically in a fresh untouched c214b7fc checkout at
C:/dev/sph-baseline (2 failed, 1 passed in 0.77 seconds). This is an additional
existing integration blocker for a managed Bath liquid engine, not repaired
by the sampling change. No controller workaround was added.

Continuation, 2026-11-11 UTC: the sibling spectral-analyzer now has an explicit
ExtinctionGrid binding to its existing native forward/sensor ray transport.
from_density_sampler(fluid.sample_at, ..., mass_extinction_m2_kg=...) calls the
real SPH publisher, requiring a supplied material-valid coefficient. A
cross-repository fixture measured 0.5163623094558716 m^-1 versus the direct
kernel-sum expectation 0.516362301252964, with all physical state unchanged;
empty particles publish zeros. This is scalar extinction only, not invented
fog, scattering, liquid free-surface geometry or a replacement simulator.

The native optical seam passed 114 focused spectral CPU tests (one skipped) and a real-GPU
gate on AMD Radeon(TM) Graphics: six composed shader variants, 544 analytic
segments, absent/zero field identity, camera and source Beer-Lambert factors,
directional integrated transmittance and resource cleanup. Detailed API and
limits are in spectral-analyzer/camera_designer/STUDIO_VOLUME_EXTINCTION.md;
receipt in spectral-analyzer/docs/20261111-volume-extinction-receipt.md.
The initial no-atmosphere-integration statement above describes that earlier
sampling-only checkpoint. Static native optical input is now implemented;
live Studio publication, mesh/SPH contact coupling and the bowl remain open.
