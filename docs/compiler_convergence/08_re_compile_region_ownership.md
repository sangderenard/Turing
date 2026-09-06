# Task 8 — `re._compile` nested-region ownership

## Observed code state

- `tools/compile_re_probe.py` is the real entry-point reproducer: it binds
  `re._compile`, applies `program_extraction.yaml`, lowers through
  `lower_ast_source_to_ssa`, and attempts Fortran emission.
- `overlay_scheduled_control()` accepts `known_nesting` because region-set
  containment alone cannot distinguish equal-set parent/child controls.
- Both structural and ordinary deployment paths call the overlay with
  `_loop_reduction_nesting_hints()`.  The current code deliberately refuses
  cross-scope controls and missing nested regions.
- The report's specific regions 37/53 are not a stable code contract and have
  not been remeasured after deterministic relabeling.

## Work sequence

1. Run the standalone probe once from a clean checkpoint and capture its exact
   current exception, progress phase, controlled sets, and nesting hints.
2. Reduce the failure to a focused test using structural ownership—not region
   numbers copied from one build.
3. Determine whether ownership is lost while forming loop plans, nesting hints,
   projecting controls, or overlaying them.  Preserve current refusal tests.
4. Fix the earliest layer that loses real lexical ownership; do not teach the
   overlay to guess from coincident region sets.

## Acceptance

- The focused test identifies controls by stable source/structural evidence.
- Existing cross-scope refusal tests remain green.
- `compile_re_probe.py` reaches complete emission or a later, truthful
  shortfall without double ownership.
