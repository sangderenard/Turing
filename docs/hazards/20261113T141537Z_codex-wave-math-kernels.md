2026-11-13 SPECTRAL T4 WAVE LAWS (MEASURED, Windows, branch codex/wave-math-kernels):

The four mathematical sites taken from Spectral's `wave_t4.cpp` now have
SymPy-authored laws: propagating angular-spectrum transfer, evanescent decay,
absorbing aperture material transmission, and complex 2x2 Jones mixing. The
existing `compile_sympy_equations` -> AbstractTensor source ->
`piece_from_law` route produced f64 LLVM batch pieces; all four matched their
T4 formulas in a four-lane focused numerical gate. The same equations emitted
f32 deployable SPIR-V products with the existing product publisher. All four
assembled and validated. No result was inferred from source emission alone.

The compiler's SPIR-V operation table lacked Pascal-case `Sin`, `Cos`, and
`Exp` aliases emitted by its own SymPy route. Adding those names made the f32
wave products complete. The backend still refuses f64 transcendental SPIR-V
as required by GLSL.std.450, rather than silently narrowing T4's f64 transfer
table. The f32 GPU products therefore have a distinct precision contract.

`tests/test_spectral_wave_laws.py`: 8 passed in 39.23 s, including the four
native numerical gates and four assembly/validation gates. Running it beside
`test_ssa_spirv_backend.py -x` stopped at the pre-existing axis-reduction
wording assertion after 19 passes; the failure expects `no axis` while the
actual shortfall says `sum over axis 0 ...`. That failure was not changed.

Nodus row trial from the companion branch: each f32 shader ran 262,144 lanes
through the native shader job and tensor state library on devices 0, 1, 2;
all four families matched the T4-formula fixture. First-dispatch times are
recorded in the Nodus companion report and include host/Vulkan setup. This
does not install the FFT, arena continuation, or thick-lens aperture route.
