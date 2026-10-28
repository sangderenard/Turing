2026-10-28 LANE D: THE `or "float64"` DEFAULTS IN src/compiler (MEASURED, cloud):
Nine were left on main 61560c15. Hooked over 58 float, division and precision
corpus programs, five fired, all on Python semantics spelled as a fallback:
a float literal beside integer operands (ir_indexing preserving ops,
hierarchical_plan promoted_numeric_dtype) and true division / sqrt / exp /
log of integers (ir_indexing, hierarchical_plan truediv, glsl
_fold_callsite_structural_values). They now read the operands' records
through `dtype_layout.python_float_result_dtype`: the widest float operand,
else Python's float (`PYTHON_FLOAT_DTYPE`, binary64) only when every
operand is integral; an unsettled operand leaves the result unsettled (the
glsl scalar fold refuses by name). The two Precision sites in
_build_shell_hierarchy_plan read the Precision type's own limb element
(`extended_precision.limb_element_name`, which refuses an element without
limb facts); a plain refusal there fired on real `Precision[k]` values.
precompile_to_ssa._numeric_join's default was unreachable and is removed.
Left: tensor_ssa_lowering's bool-buffer physical dtype (never reached here).
58/58 PASS as on main; c819 PASS; test_hierarchical_plan,
test_precision_pipeline, test_precision_surface, test_complex_precision and
test_rational_precision give main's failure names. Not the full corpus.
`int / int` beyond 2**53 already refuses loudly at run time (NativeIntOverflow).
