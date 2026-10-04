# Parked reverse-precision experiment

Parked after the user's 2026-10-03 request to reduce model intensity and judge
progress against elapsed time. This is **unverified experimental source**,
not a landed compiler correction or a numerically accepted orbital planner.

`compiler.patch` preserves exactly the session's changes to
`src/common/tensors/topological_reducer.py` and
`src/compiler/vehicle_inverse_compilation.py`. The adjacent `.py.saved` file
preserves the new source tests. Root reverse-applied only those owned hunks
after `git apply --reverse --check` succeeded; neither a stash nor a checkout
of a path was used. The unrelated symbolic-process-graph changes remain intact.
The working compiler is back to its pre-experiment contents.

The patch transfers numeric class/limb facts to tuple projections using return
slot producer references, fixes multi-loss labels, uses the established
nonconstant receiver convention for min/max, and renders graph-declared integer
index constants directly. It does not solve the complete reverse precision
path. The last source test run had four passes and one literal clamp failure
(`int.shape`); that run predates the final Constant-kind guard. The final tiny
native-source probe still refused gradient collapse before LLVM emission.
Loss collapse was resolved, but no native reverse numerical result exists.

The known blocker is the scalar `unbroadcast` return: source return producers
0 and 52 carry Precision[2], while fallback reshape producer 62 has no numeric
descriptor. The real same-shape call should return the original wide G.
Existing shape specialization is later than source numeric propagation, and
its identity expansion unconditionally appends reshape even for equal shapes.
The later result publisher transports shape, not the selected producer's
numeric class. Do not infer a return type from the helper's name, input type,
or a subset of return sites. Do not add general reshape support to Precision
to mask this problem.

Read-only Luna evidence identified the existing control-owned selection in
`glsl_deployment_strategy.py`: it selects a terminal return by its source span,
reads that return's slot producer, updates output identity history, and prunes
the other return sites. A complete selected-scalar numeric transfer remains
unestablished. Generic authored `int(alias)` projection transport, precise
constant ingress and Precision min/max comparisons also remain open.

Receipts and guarded reproductions remain in the local temporary directory:

- `orbital_fused_precision_probe.py` and `orbital_fused_precision_watch.ps1`
- `orbital_return_producer_trace.json` and `orbital_return_producer_watch.ps1`
- `orbital_numeric_projection_trace.json`
- `orbital_fused_precision_capture/`

The final guarded trace took 25.6 s with 1,775,083,520-byte peak process memory.
All experiment processes ended. No production planner bank was built. A future
resumption must recheck the patch against its then-current owners and establish
a tiny complete native gate before rebuilding orbital rows. The numerical
four-ULP and exact-zero checks are unchanged.
