2026-10-28 PRECISION TESTS ON MAIN (MEASURED, cloud): two main-side
refusals stopped every Precision[k] program from lowering after #911.
(1) Region values: a Precision[k] parameter, an element of or store into
a Precision[k] array, and a precision operation's result now take their
dtype from the numeric descriptor (its element, binary64 by default)
instead of reaching "region value N has no settled dtype". (2) The frame
fixed-point digest pickled the retained Precision ClassDef, whose lexical
bindings hold the `math` module; it now records a module by its import
path. Focused: test_precision_pipeline, test_ssa_glsl_compute_backend,
test_microprecision_profiler, test_precision_compute_product: 58 passed,
2 failed (Exp/Log source-class concordance, failing on main too). The
profiler test now expects the C lane to pass, Fortran to stop at its
compiler, and WebGPU at its runtime. No gate or corpus run.
