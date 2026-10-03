# Continuation: name-carried arm through the concorded alias chain

Base: main dbc59f0a. Uncommitted.

## Fix 1 (applied): `_carried_name_arm` follows the concorded alias chain

Woodshop `_resolve_floor` failed with `carried-name-arm-missing arm=88
initial=66`. Both ids are in-place store versions of `momentum`
(`momentum[2] = ...`, then the nested `momentum[:2] -= ...`). The planner
aliases both (`control_value_alias`, PLANNING) to the call result: store
versions are versions of one storage. The builder never binds a store
version. `_carried_name_arm` looked the arm up in `external_values` only,
found nothing, saw the arm's authored `name_binding` row and refused it.
The initial id resolved because `external_value()` follows `value_aliases`.

Change (`src/compiler/precompile_to_ssa.py`):

- `_concorded_resident(value_id)`: new method; the resident under
  `concorded_value_aliases` (the PLANNING aliases as resolved at builder
  entry, not the live view loops rewrite). The local closure in the loop
  carried-update check now delegates to it (same code, one key).
- `_carried_name_arm`: after the binding lookup misses, resolve arm and
  initial through `_concorded_resident`. Same resident: the arm carries the
  entered storage, take the snapshot. Distinct resident with a binding: take
  that binding. Otherwise the old authored-write check and refusal.
- `_recorded_name_write` docstring states the in-place-store case.

Probe: `tools/compiler_probes/probe_nested_inplace_arm.py`. Before the fix it
raised `carried-name-arm-missing` on the inner `if` (store over store, the
Woodshop shape). After the fix it lowers.

Gate: all listed probes exit 0; `probe_scalar_native_correctness` 0
failures; audit first lines 0,1,0,1,5,0,0 (unchanged).

## Defect 2 (open, not fixed): IndexedStore has no lexical location

After fix 1 the probe lowers but native != Python on every branch
combination. Minimal shape, also refused at dbc59f0a before fix 1:

    def f(x, a, b):
        m = x * 1.0
        if a < 0.0:
            m[:2] -= b
        return m

Observed chain:

1. Reducer `src/common/tensors/topological_reducer.py`, the indexed
   assignment path (`new_node("IndexedStore", "indexed_store")`): the
   IndexedStore node gets no `expr_obj` and no `source_span`. Its
   siblings in the arm (`Indexed`, `Sub`) carry the Subscript/AugAssign
   with lineno.
2. Planner `glsl_deployment_strategy._branch_compartments`: membership
   comes from source-positioned `expr_obj` (or a `source_span` fallback
   that only `setattr` gets). The IndexedStore gets no `(if, body)`
   membership; `Indexed`/`Sub` do.
3. `_ordinary_conditional_control_programs`: the store's region is not a
   body region. The control program puts it at `root.sequence[2]`, after
   the conditional.
4. SSA: the store call sits in `if_merge` and reads the slice result
   defined only in `if_true`. The false path stores an undefined value
   (observed zeros in `m[0:2]`).

The fix belongs at the reducer (give the store its authored statement's
location so branch membership finds it), not at the planner or builder.
How the store should carry it (`expr_obj` vs `source_span` +
`effect_span_nodes` like SetAttr) is the open decision.

## Woodshop

Fix 1 removes the `_resolve_floor` refusal. Defect 2 means the
`momentum[...]` stores in that function will run unguarded in native code
until fixed.
