2026-10-28 PRECISION EXP/LOG METHODS (MEASURED, cloud): `x.exp()` and
`x.log()` on a two-limb Precision failed in two ways. (1) The method's
return annotation `-> 'Precision'` posted width 1 on the result, which
contradicted the precision operator's width 2 (a source_value_class
disagreement). A width-carrying class's annotation now posts no width, as
the call-result path already did. (2) Once renamed to precision_exp, the
node kept the AbstractTensor `tensor_candidate: exp`, so lowering emitted a
one-double tensor exp (unary_double). The rename now drops it. Needs #969.
Focused tests (precision pipeline, GLSL lane, profiler, compute product):
pass. test_complex_precision two-real-expansions and the two
test_rational_compiler errors fail on main too. No gate or corpus run.
