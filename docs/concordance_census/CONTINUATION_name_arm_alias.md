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

## 2026-10-03 (later): defect 2 fixed by a book cell; defect 3 found and fixed

Uncommitted, on a2020e0b.

### Defect 2: the IndexedStore's identity is a posted site cell

Rule (user): nothing is valid that does not go through the concordance with
edges. The store's identity is therefore a NOVEL book cell, not a location
attribute and not a side table.

- New page `indexed_store_site` (reduction scope, site VALUE_ID), fact
  `IndexedStoreSiteFact(span: Ref)`, transform `indexed_store_site` (arity 2)
  in `concordance_declarations.py` (appended block).
- Writer `topological_reducer.post_indexed_store_site`: when `bind_target`
  must mint the store (`m[:2] -= t`: the target Subscript node is the read),
  it posts `(scope, NEW)` NOVEL(INDEXED_STORE_SITE_MINT, (target span cell,
  effect cell)). The effect cell is the target-read node cell and the stored
  value's node cell, joined by one `cell_set` row (DERIVED). `new_node(...,
  source_cell=site)` then posts the store's `ingestion_value` row DERIVED
  from the site; the relabel's `canonical_value` row derives from that. No
  `source_span` attribute is stamped. A store that reuses the authored
  target node (`m[2] = ...`) keeps that node's own row; no site.
- Reader `topological_reducer.indexed_store_site_span`: node identity cell
  -> `canonical_value` -> `ingestion_value` -> `indexed_store_site` along
  posted edges (`edges_into`), then the site's `source_span` fact. The
  positions only place; the cell identifies.
- Membership: `glsl_deployment_strategy._branch_compartments` adds an
  IndexedStore with no `expr_obj` to `effect_span_nodes` through that reader
  (7 lines + 1 import; that file belongs to the stall lane).
- Measured on the probe: store 23 (`m[:2] -= t`) had membership `[]`, now
  `[(inner if, body), (outer if, body)]`; the "if_merge Call reads a value
  defined at if_true.1" detector entry is gone.

### Defect 3 (pre-existing at a2020e0b, straight-line too)

`m[2] = m[2] * 0.5; t = b - 1.0; m[:2] -= t` stores 0.0 into `m[2]` with no
control at all (measured on clean a2020e0b). Chain, observed by trapping
`SSAValue.shape` writes: `propagate_repository_ssa_call_metadata` fills the
shared kernel `index_assign_double`'s `values` formal with (2,) from the
second store's call, then the reverse fill `enrich(function, actual,
formal)` writes (2,) onto the first store's scalar actual (fill-only treats
`()` as absent). The kernel then reads a 2-span from a scalar slot.

Fix (`tensor_ssa_lowering.py`, reverse fill): a shared in-place kernel
(declares `ssa_output_argument`; the call carries none) back-fills only its
declared output argument. Its other formals are filled by every caller and
are no fact about this caller's actual. A blanket skip of such calls was
tried first and rejected: it dropped real (8,) shapes. The same leak was
corrupting the audit's `view` case: `peak = advance(...)` and
`float(state.memory[0:4].max())` were stamped (8,) span from `restore`'s
`self.memory[...] = snapshot`; at a2020e0b the view case's C emission was
INCOMPLETE (two `Cast` shortfalls: "array cast requires a matching static
shape"); with the fix it completes, LLVM shortfalls () both ways.

### Probe (now C and LLVM)

`probe_nested_inplace_arm.py` emits both lanes. All 8 rows equal CPython:
(1,3) and (1,0.5) -> [1.5,-2,3.25,4]; (-1,0.5) -> [1.5,-2,1.625,4];
(-1,3) -> [-0.5,-4,1.625,4]. Before: 4/4 C rows wrong.

### Gates

- Audit, 7 cases: findings 0,1,0,1,5,0,0 before and after. Unsourced facts
  with only this lane's changes applied to clean a2020e0b: identical except
  view 4083 -> 3970 (the leaked enrichment writes are gone). (The main tree's
  other numbers move with other lanes' uncommitted edits.)
- test_record_return_site_identity: 14 passed, 1 xfailed.
- test_pruned_loop_return: 1 passed.
- test_precompile_to_ssa: 13 failed / 91 passed; failing set identical to
  clean a2020e0b.
- probe_scalar_native_correctness: 0 failures.

### Woodshop `_resolve_floor`: next wall (not fixed, held)

Seconds-long slices (scratch) of the `_resolve_floor` shape:

    def f(x, a, b):
        m = x * 1.0
        v = m / 2.0
        if a < b:
            m[2] = v[1]
        return m

refuse at the full-native contract on clean a2020e0b and here alike:
"4 formals but only 3 named or ABI-accounted", the extra formal carrying
`unbound_variant_source_id` of `m`, `variant_column: row`. Cause, observed
by tracing `lower_control_sections_to_ssa` locals: the "payload left as a
direct method input" rule (`(indexed_base_ids & scalar_use_ids) -
declared_sequence_ids - statically_shaped_ids`) classifies `m` as a
heterogeneous variant payload, because `m` is an indexed base, also a
plain operand (`m / 2.0`), and `value_shapes` holds only the parameter's
shape at that point. `m` is produced in this function by a Mul of a (4,)
tensor. Straight-line (`v = m / 2.0; m[2] = b; return m + v`) lowers
because no region receives the row column. `_resolve_floor` has the shape
(`velocity = momentum / mass; if velocity[2] < 0.0: momentum[2] = ...`).
Open decision: which declared fact excludes a locally produced tensor from
the variant-payload rule (the rule's own comment scopes it to method
inputs). The whole-program check is still the user's
`build/woodshop_outer_native_probe.py` run.

## 2026-10-03 (evening): variant-payload rule reads storage identity

Decision (coordinator): the declared fact is STORAGE IDENTITY on the book.
A candidate (indexed base that is also a plain operand, not a declared
sequence) is a "payload left as a direct method input" only when its
storage root, walked along DERIVED edges, is a formal's storage. The "no
known shape yet" subtraction is deleted from that rule (it still guards the
loop-target rule above it, which is separate).

- Page `storage_root` (function scope, value), fact
  `StorageRootFact(kind FORMAL | PRODUCED | VIEW)`, stage `storage_root`,
  transforms `storage_formal` / `storage_produced` (arity 1), reasons
  `storage_producer_unrouted`, `storage_root_unknown`
  (`concordance_declarations.py`, appended).
- Writer `precompile_to_ssa._post_storage_roots`, in a scope minted per
  lowering (`storage_root:<control>`): a parameter (version 0 of
  `identity_table`) or the receiver is a FORMAL root NOVEL from its
  version-0 `name_binding` cell (else its `canonical_value` cell); an
  instruction result is a PRODUCED root NOVEL from its `canonical_value`
  cell, except Indexed / IndexedStore / GetElementPtr / Load results and
  results declaring `ssa_storage_view`, and planning aliases, which are
  VIEW rows DERIVED from the viewed value's root cell. Anything else is
  `Unresolved(STORAGE_ROOT_UNKNOWN)`.
- Reader `_storage_root_is_formal`: DERIVED walk on `storage_root`.

Results: the `_resolve_floor` slices (`v = m / 2.0; if a < b: m[2] = v[1]`,
`if v[2] < a: m[:2] -= b`, and the two-level momentum/tangent slice) lower
and match CPython in C and LLVM; `m` is PRODUCED and mints no row column.

Still refused, identically on clean a2020e0b (not a regression):

    def f(x, a, b):          # x: declared span parameter
        y = x * a
        y[0] = x[1] + b
        return y

`x` is a formal, indexed and used as an operand, so the rule (as decided)
makes it a payload and mints a row column nothing binds. A declared span
formal is not heterogeneous; which declared fact separates it is open.

Gates (this lane's changes alone on clean a2020e0b): audit findings
0,1,0,1,5,0,0; unsourced identical to HEAD except view 4083 -> 4022.
test_record_return_site_identity + test_pruned_loop_return: 15 passed,
1 xfailed. test_precompile_to_ssa: 13 failed / 91 passed, same set.
probe_scalar_native_correctness: 0 failures. Probe: 8/8 rows equal.

### Woodshop whole program (approved run, 2026-10-03)

`build/woodshop_outer_native_probe.py`, faulthandler dump at 30 min, one
build. `_resolve_floor` lowered: no `carried-name-arm-missing`, no
unbound variant row; it reached record-ABI materialization with the rest.
The build died later, in the frame linker while building source-call
records (`_linked_caller_member`):

    woodshop_outer_physics___advance_newton_dt_system__specialized_...
    callsite 42: callee ..._ensure_newton_dt_system__specialized_...
    formal ... is member ('value', 0) of declared field
    'woodshop.WorldMachine.custody' of record ..., bound to caller record
    20, which has no such field

Same class as the 2026-09-29 wall (row-handle record parameter;
`probe_row_handle_record_parameter.py`), now on `custody` in the newton
dt-system path instead of `center_xyz` in `_sync_newton_lanes`. Outside
this lane (fortran_c_shell frame linker). The traceback's source lines did
not match the code shown, because fortran_c_shell.py was being edited by
another lane during the run.

## 2026-10-03 (night): declared span formals; row-handle wall moved once

### Declared span formal is its own binding

`parameter_abi_kind` page (read scope, parameter) -> `ParameterAbiKind`
(the contract's storage vocabulary as an enum). Writer
`precompile_to_ssa.post_parameter_abi_kinds` at the control handoff, from
the contract declarations (`parameter_value_abi`, `parameter_record_abi`,
passed by the one `_class_surface_ssa_program` call site): a declared kind
is NOVEL(PARAMETER_ABI_DECLARATION, (version-0 `name_binding` cell,)); no
declaration is `Unresolved(PARAMETER_ABI_UNDECLARED)` derived from that
cell. The payload rule mints a row column for a formal-rooted candidate
only when the root formal's row is RECORD or Unresolved. `y = x * a;
y[0] = x[1] + b` (x a declared span) now lowers and matches CPython in C
and LLVM, as do the `_resolve_floor` slices.

### Row-handle wall, callsite 42 (fixed)

Observed with a read-only hook on `_linked_caller_member`: the callee
record (`woodshop.WorldMachine`, declared row columns of
`_ensure_newton_dt_system`, which never indexes `items`) was paired by
`call_record_pair_concordance` to the caller's `items[]` row record 20,
which held columns only for the leaves `_advance_newton_dt_system` reads.
Same pooled representation, missing leaf. Fix (frame linker,
`discovery_linked_member`): `grow_pooled_row_column` mints the caller's
column NOVEL(DECLARED_ROW_COLUMN, the callee column's cell) and registers
it on that row record, only for a value member of a SPAN field whose
callee formal is a row column of the same record identity.

### Next wall, callsite 84 (held, question to the user)

`_advance_newton_dt_system` -> `_set_momentum(item, momentum)`: callee
record 0 has identity `WorldMachine` (storage identities
`WorldMachine.<field>`), a flat per-field record parameter from the
`item: WorldMachine` annotation; the caller binds its `items[]` row record
(identity `woodshop.WorldMachine`, columns). The forwarding pass records
"bound record identities differ". Two identities, one class.

### Callsite 84 fixed by decision (A): annotation -> class -> contract record

Page `parameter_record_class` (module, function, parameter) -> contract
record identity, DERIVED(`parameter_annotation` cell, the
`class_declaration` cell the annotation names in its module, the
`source_record_class_concordance` cell). Writer and reader
`fortran_c_shell._annotated_parameter_record_identity`; the two record-ABI
selection sites replace an annotated parameter's inferred view by that
contract record's receipt, and the record-forwarding identity check reads
the joined identity from the book. Join detail that mattered: the class
declaration and the contract's record class derive from DIFFERENT cells of
one `source_span` row (the span was revised between stages); the row is the
construct, so the join reads every column of it. A class with a span but no
`class_declaration` row gets one, DERIVED from its span. Seconds-long repro:
scratch `rhmod/rhworld.py` (`Rules._set_momentum(item: Body)`): the
forwarding edge `(step self items[]) -> (_set_momentum item)` now exists.
Not yet verified natively: the repro's `_set_momentum` momentum formal is
still `linked_caller_member` Unresolved / `argument_binding` minted from
absence, so the write may not reach the caller's column.

### Next wall (held): `_sync_newton_lanes` -> `center_xyz`, frame link round 1

    callee woodshop_outer_physics__center_xyz formal ... is sequence member
    ('column', 0) of 'woodshop.WorldMachine.orientation_deg_xyz'; the bound
    caller record 10 names no such member

The callee holds the fixed-shape span leaf `orientation_deg_xyz` (shape
[3]) as a sequence descriptor; the caller's row record holds it as the
row-selected view plus the pooled `.column`. Representation decision.

### Write-back check (coordinator, 2026-10-03 night): a miscompile, pre-existing

Repro (scratch `rhmod/rhworld.py` + `probe_rh.py --run`, real contract base
`program_extraction.yaml` + ABI): `Rules.step` loops `self.items`, calls the
staticmethod `_set_momentum(item, item.mass * dt)`. CPython momentum
[0.5, 1.0, 1.5]; native C and LLVM [0, 0, 0], here and on clean a2020e0b.

Chain, observed:
1. Minimal form, no rows: `def outer(item, dt): item.momentum =
   item.mass * dt; return dt` with `momentum` a mutable scalar field. The
   SetAttr survives reduction; the control handoff
   (`_class_surface_ssa_program`, scalar field writes) builds a Store only
   when the field has a getter (`if not getters: continue`). Written,
   never read: no Store, write dropped (no Mul either).
   FIXED: page `record_field_incoming_slot` (function, parameter, field)
   -> minted id; `_record_field_incoming_slot` mints it at the handoff
   NOVEL(NESTED_RECORD_PART, the SetAttr's field-state cells) as the
   Store's destination, and record-ABI materialization adopts it as the
   field's formal (it minted a fresh one before). Now `outer` and a callee
   `setm(item, v)` both Store into the field formal.
2. Rows: the caller's `item = self.items[identity]` is lowered through the
   `ssa_sequence_*_lookup` call, and its row record (5) is NOT the
   declared-row-column record (mass/momentum `.column` spans). At the call
   the link grows a fresh scalar formal for `momentum` on record 5 (the
   field-demand `grow`), i.e. caller storage minted from nothing: the
   callee's Store lands there. Still 0.0. OPEN.

Gates after (1): audit 0,1,0,1,5,0,0; identity tests 15 passed 1 xfailed;
test_precompile_to_ssa 13/91 same set; test_native_record_read_order 1
failed / 3 passed, the same variant (`first_write`) failing on clean
a2020e0b; probes 0 failures.
