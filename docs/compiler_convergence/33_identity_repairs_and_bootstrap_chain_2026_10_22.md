# Identity repairs and the bootstrap core chain — 2026-10-21/22

Branch `claude/nice-cerf-7w6uk5`, commits `5d02b30..8cbc54a` on top of
`913e4c2`.  Every repair below was found by observation (a trap on the writer,
the two records printed side by side) and is recorded in a receipt comment at
the pass that now holds the rule.  Numbers are measured; nothing here claims
the bootstrap or the managed tire is complete.

## How the evidence was taken

- Bootstrap: `python -u -m tools.bootstrap_compiler_concordance --lower-only
  --jobs 3 --continue-after-failure`, each run in its own clean worktree of one
  commit.  Units are compared by `qualified_names`, never by unit index (the
  plan renumbers when the source changes).
- Single-unit replay (`scratchpad/s/unit_run.py`, not committed): runs the
  worker's own `tools.compile_project_catalogue` on a frozen plan with only
  the toolchain-fingerprint check stubbed, linking each dependency from a
  replay directory or the run's own product.  A trap is a `PRELUDE` file.
  This is how the core chain below was walked unit by unit on the live tree.
- Gate: the 49 files of the ~40-second gate list, serially, against the clean
  `913e4c2` worktree.  Tail tests of `test_loop_carried_producers.py` hang on
  base and on every branch run alike (per-test timeout 1200 s).
- Native corpus: `tests/test_native_semantics_corpus.py` lowers each
  `tests/native_semantics_corpus/c*.py` through `lower_ast_source_to_ssa`,
  requires zero concordance findings, emits C, compiles, and compares every
  input against Python.

## Repairs (program — wrong record — rule)

Identity and provenance:

- `semantic_output`: structural recovery takes output identity from the
  reducer's rows through one helper; the finish writer still compares
  independently (`_reducer_declared_output_ids`).
- Optional aggregate return (`IdentityPage._fork_of_scope`): the one semantic
  output was revised to leaf 0, so `Ret` dropped leaf 1.  The
  `semantic_output` row of an optional aggregate names the aggregate;
  `function_output` rows stay physical.  `Ret` is now `(p0, q0, p1, q1)`.
- One executable per symbol: two shells with one specialization contract were
  both lowered under one name (the second `loop_scope` post was refused).  A
  symbol is lowered once; a different contract under one symbol refuses.
- `formal_literal`: only literal callsites posted, so one `1000` callsite read
  as unanimous and the generic `decimal_integer_glyph_tokens` copy formatted
  every integer as 1000.  A runtime callsite revises the row to
  `Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT)`.
- Structural If fold retires the rejected arm's effects from
  `source_sequence_mutation_records`.
- Call-metadata fixed point snapshots the same contract keys the edge
  transports (`sequence_row_shape` was missing).
- Record-field call edges post storage/rank on the callee formal's row
  (`_reindex`'s sequence helper; four core units each blocking ~100).
- Accepted (dependency) callees are immutable sources: neither physical call
  input adaptation nor settled call-value propagation writes onto them.
- The table-store column retype is a REVISE of the column's row.
- `external_value` refines only to a resolved dtype.

Types and storage:

- Primitive class key is physical only for scalar/span carriers
  (`argument_binding_history` tuple arms; `_validate_fact`'s shared union).
- An arena row is a span in the structural view whatever its extent
  (`_reindex_in_place`).
- `builtins.range` declares its comprehension target `int`; an `Input` with a
  builtin scalar class is a rank-0 descriptor of that dtype; an int literal
  operand is int64 (weak against float tensors).
- Only a scalar resident's settled row revises a region's cut-time feed dtype.

Control and features:

- A membership query on a sequence the control tree mutates is placed at its
  authored node (c20: `root(2)` returned -1).
- A validation guard nested in a loop-body arm is placed inside that arm
  (`use-not-dominated` in `_validate_row`).
- `sum(collection[, start])` over a resident generator arena is a resident
  query (integer rows); float rows are a named shortfall (Python's empty float
  sum is the int 0).  Static-range generators take the same query instead of a
  region tensor sum over a shapeless span (`k=0` gave 10 for 416).  The C lane
  no longer reduces a shapeless span as one element.
- Storage requirements read `sequence_static_capacity_bounds` (written by the
  control builder, never read).

## Measurements

Bootstrap (by qualified name):

| run | commit | compiled-unverified | partial | failed |
|---|---|---|---|---|
| base | 913e4c2 | 37 | 47 | 270 |
| cur | 5d02b30 snapshot | 42 | 51 | 262 |
| snap2 | 4709ba1 | 40 | 53 | 263 |

`4709ba1` regressed six units (the primitive-reference union key); `7faa2df`
restored them.  Of the 262 failures in `cur`, 185 were dependency cascades;
eight core units (`IdentityPage`/`IdentityBook` methods and
`MonotonicIdSource.mint`) each blocked about 100.

Core chain on the live tree (single-unit replay, linked in order):
`IdentityBook._ref_from_key`, `IdentityPage._fork_of_scope`,
`LiveScopeProjection.excludes/.reaches`, `IdentityPage._reaches_origin`,
`identity_compression.thaw_page`, `IdentityPage._reindex`,
`identity_spill._reindex_in_place`, `_replace_cells`, `_truncate_fact`,
`_validate_fact`, `_hashable`, `RowField.admits`, `_validate_row`,
`_detached_ordinals`, `MonotonicIdSource.mint`, `id_space.compose`: all
compile; the composed products audit clean except `_validate_row` (below).

Gate: the `cur` snapshot run reported 130 failed, 1010 passed against base's
positions; 12 base failures pass and one new failure
(`test_writer_refuses_when_recovery_and_posted_row_disagree`) was repaired in
`1e8fc64`.  Later targeted comparisons (same files, branch vs base) show no
new failures and 7 + 16 + 2 additional base failures passing.

Native corpus: 27 programs plus the unbounded-list boundary and the nested
loop-guard audit test pass.

## Accounting is a view of the posted row: the writer inventory

The `ssa-physical-unposted` audit only reports a value whose accounting label
reached the final module without a row.  A static inventory (every write of
`physical_dtype`, `program_abi_storage`, `program_abi_rank` and
`sequence_arena` in `src/`, classified by whether the written value is the
posted fact, a fallback literal beside it, or an unposted literal) found the
writers the audit cannot see because they record nothing.  Every SSA
accounting writer in the compiler proper now posts on the row first and sets
its labels through `_frame_accounting_from_posted` (or the builder's
`_posted_physical_accounting`):

- A record formal, keyed-mapping handle, nested leaf, sequence member or
  `table` member that entered a frame as a bare `SSAValue` had no row; the
  REVISE that followed was a no-op and the declared literals stood in
  accounting.  `_adopt_graph_formal` posts the row (ADOPTED from the graph
  cell, or MINTED from its declaring cells, exactly as structural recovery
  adopts a value) with its declared physical fields.  Unit 81
  (`_result_layout`, keyed `function.blocks`) moved from failed to partial.
- Linked-contract formals (`linked_value_abi`): the `table` contract never
  posted because the dtype test compared an absent contract dtype with the
  row's; accounting then fell back to the dict.  Both branches are views.
- Existing formals whose dtype/shape were rewritten in place (record fields,
  reused keyed parts, keyed lookup helpers, `eps` constants, the reduced
  scalar of a full tensor reduction, returned-record call arguments) are now
  REVISEs of their rows; a settled Phi with no row refuses.
- The native storage-effects accelerator wrote pointer element types into
  accounting with no row while the Python scan posted the same inference: the
  witness scan is the one publisher, the accelerator a refusing cross-check.
- Result-type settlement copied a base's or source's accounting dtype onto
  the target; only the target's posted row is mirrored.
- Boolean tensors' double-backed ABI, the axis-dropping index view, projected
  iterable layouts and optional record pairs post before they label.

Readers keep reading accounting: emission runs with no attached book by
design (`emission_book`), and agreement of every label with its row is the
audit's to enforce.

Record schemas: `{**projection_schemas, **complete_schemas}` replaced the
book-backed view of a class (`source_record_view_schema` row) by the AST-only
constructor schema of the same fields, so no record field could be joined to
its `class_field_declaration` (unit 41, `page.row_fields`).  One record per
class now keeps the view's row and adds constructor completeness.

## Tuple merges across statements

`if f: t = (x, y) else: t = (7, 8)` followed by `a, b = t` returned 0 on base
and branch alike: the structural fold aliased `t[k]` to ledger leaf `k` only
for a literal aggregate, never for the reducer's `aggregate_phi`.  The merged
ledger is its member Phis by position, so the fold now reads it (corpus c28).
A call arm (`t = pair(x)`) has no layout at reduction; once its callsite
projections publish, the planner posts its layout and builds the same
`aggregate_phi` (corpus c29).  An OPTIONAL_AGGREGATE call arm is left out:
its presence exists only at SSA level, where the merge must compose it.

## Optional tuples: the presence test and an optional call arm

`fork = t if c else None; if fork is None or x: return; a, b = fork` refused
"no proven presence guard": the `None` comparison was itself counted as a
consumer needing a guard, and the fact a guarding `or` proves on its
fallthrough edge was kept whole.  A `None` comparison of the optional is now
its presence test (rewritten to read the optional's predicate); consumers
are guarded by the predicate fact or by that test reading false; facts close
over `or`-false, `and`-true and `not` (corpus c30).  Unguarded, wrong-arm and
changed-guard consumers still refuse.

When the present arm is itself an OPTIONAL_AGGREGATE call
(`IdentityPage._resolved`, unit 33: `self._fork_of_scope(row[0]) if c else
None`), presence is `c` and the callee's presence.  The present arm carries
`call is not None` -- the comparison optional-call lowering already rewrites
to the callee's presence -- inheriting the call's branch membership; the
absent arm carries `False`; one presence member Phi, merged with the payload
members, is the optional's predicate (corpus c31).

Open: a statement-level merge with an optional call or `None` arm
(`if c: fork = f(x) else: fork = None`) still has no presence member; the
planner's aggregate merge leaves such arms out.

## Carried indices, literal lists and the while test

Corpus programs c35-c40 index a list by a value that changes per iteration.
All failed on base, not only on this branch.

- **Carried index folded.** `index = 0; for ...: total += xs[index];
  index += 1` read `xs[0]` every iteration. Two consumers read the index
  operand's graph parent, the literal 0, as an invariant. That literal is the
  entry arm of the loop's carried Phi. The consumers are the structural fold's
  aggregate alias (`_fold_callsite_structural_values`) and the shell's row-op
  recovery (`_sequence_row_operations.integer_literal`). Both now consult one
  reader of the reducer's `loop_carried_bindings`
  (`_loop_carried_initial_ids`).
- **Literal list never written.** `xs = [5, 7, 11]; return xs[n]` returned 0.
  The all-constant list had been folded to a Constant node, and the list birth
  (`_list_literal_mutations`) keyed on the AST being `ast.List`. The birth now
  reads the list's SOURCE_AGGREGATE_LAYOUT members and takes each member from
  the graph by identity.
- **While-test region outside the loop.** A region that a while test
  evaluates lexically (`everything[index]` in `while index < count and ...`)
  was scheduled after the loop. Only the test root counted as the condition,
  so the header and latch read a value defined at `while_exit` (bootstrap
  unit 43, use-not-dominated). The condition is now the test root's dataflow
  ancestors inside the test's source span.

## Short-circuit operators

`if d != 0 and 10 // d > 1` trapped with SIGFPE at `d == 0`. The reducer
recorded the division as guarded by the BoolOp, but no control program owned
that membership. The shell also had a second evaluator: structural recovery
lowered BoolOps as eager `Select` chains. As a result `x = d and 60 // d` had
no definition, and `x = n or 7` lost its parameter.

The operator is now what Python says it is. `a and b` is `b if a else a`;
`a or b` is `a if a else b`.

- **Source normalization.** It makes every BoolOp binary
  (`a and (b and c)`), with a receipt.
- **Membership.** The guarded operand is recorded under the conditional's
  `body` (and) or `orelse` (or) arm. It is no longer coordinator work.
- **Control program.** `_short_circuit_control_program` builds the
  ConditionalBlock. Its result alias merge is the BoolOp node's one
  definition.
- **Consumers.** Control expressions read a merged value as a leaf, in
  `_attach_graph_control_expressions` and in while predicates. Overlay
  nesting reads only the guarded operand as the arm. The while-predicate
  rehome does not treat a merged node as predicate work.

Lowering made this visible in three places:

- **Latch incoming block.** When the latch test owns a conditional, the
  header Phis' backedge incoming block is the block that branches back.
- **Latch redefinition.** A second definition of an already-defined value at
  the latch is now a VERSIONED_WRITE. Before, it reused the SSAValue object,
  and region calls keep argument copies by id that the redefinition clone
  never reached.
- **Read positions.** `_graph_control_expression` leaves carry the operand
  position they are read at, so a carried name resolves through its
  `lexical_read_binding`. Before, it resolved to the pre-loop literal (c47).

Corpus c41-c49 cover `and`/`or` guards in `if` and `while`, n-ary chains,
guarded calls, guards inside loops and value contexts.

## Typed loop targets and the d10dce2 follow-ups

- **Loop-target type.** In `for x in xs`, `xs` is a local list built by
  appends (c25). The planner's `declared()` took the DomainNode's provisional
  float64 for the untyped Input `x`. The builder then imposed that dtype on
  the arena row, and the helpers disagreed with the int64 the sequence row
  concorded.
  - The descriptor rule now gives a single-column local list's target the
    descriptor its appends and literal leaves agree on.
  - `declared()` asks the descriptor rule for an Input with no declared
    tensor/specialization/node dtype before falling back to the domain.
- **Rank 0 in the accounting view.** `_frame_accounting_from_posted`
  mirrored a field only when `not in (None, False)`, so a recorded rank of 0
  never mirrored. It now uses `_frame_field_recorded`.
- **Equal-rank refinement.** The transformation ledger treats an equal-rank
  proposal that agrees on every recorded field and only records more as a
  refinement. Absence is not a claim.
- **Test stand-in.** `tests/compiler_provenance_inputs.publish_authored_ssa`
  posts the physical facts its authored accounting states, as a real
  producer does.
- **Open: `pieces` arena.** In test_forwarded_tuple_field, ensure's arena
  row says `opaque_ref` when total's lease copies it. The shell's
  canonical-sequence sweep retypes only the occurrences to float64, later.
  The descriptor column dtype has to reach the arena's row at precompile.

## Unbounded lists, unclaimed row loads and the 9cc6910 regressions

Each repair below names the program, the wrong record and the rule.

- **Unbounded private lists are DYNAMIC at birth.** `xs = []; for i in
  range(n): xs.append(i * i)` (c50).
  - Wrong record: the descriptor was FIXED and nothing proved its rows. The
    frame allocated one element. LLVM's append reported "full" into a
    status nobody reads (n=5 returned 1, not 35), and C refused the program.
  - Rule: the shell's one capacity survey
    (`_static_sequence_capacity_survey`) also returns the local lists no
    bound covers. Such a list is born with its own heap-owner cell. Its
    descriptor turns DYNAMIC once its column layouts are known
    (`_heap_sequence_contract`).
  - Growth is the existing checked reserve
    (`_lower_dynamic_sequence_mutation`).
  - The root wrappers start a frame-owned dynamic sequence empty (null
    arena cells, capacity 0, null owner) and release the owner after the
    call.
- **The C lane spells the heap vocabulary.** The C lane now spells the
  `ssa_heap_allocation` vocabulary that LLVM already spelled:
  - the owned-heap `Alloca`;
  - the `copy_elements` memmove;
  - `release_owner`;
  - the cell-indirect GEP;
  - `turing_heap_release`.
- **The row concordance at descriptor birth.** The descriptor took column
  0's dtype from the arena's row (int64), while `sequence_row_dtype` still
  said `unknown`. The next reuse concorded `unknown` back over int64. The
  descriptor's column dtypes are now posted to the row concordance at birth.
- **One entry per lowering.** The dynamic lowering listed its entry first,
  while callers took the last function, so `xs.append` called
  `__reserve`. `SSASequenceLowering.entry` is now the helper no other
  helper in the lowering calls. Every positional pick was replaced.
- **Helpers own their formals.** The dynamic entry reused its fixed
  insert's formal objects. Registration re-keys ids in place, scope by
  scope, so the fixed helper's row lost its physical fields. Each helper
  now gets its own formal occurrences.
- **Unclaimed row loads.** `xs[n - 1] if n > 0 else 0` (c52) and
  `if n > 0: last = xs[n - 1]` (c53).
  - Wrong record: a row load was scheduled only by a region that owned its
    `Indexed` instruction or consumed its value. An arm value whose only
    consumer is the merge had neither, became an unaccounted formal and
    read 0. Both fail on clean HEAD.
  - Rule: an unclaimed row load is emitted where its value is first asked
    for, as the sequence length already is. A result alias asks in its arm.
    A carried-name arm asks at the arm's exit.
- **Escaped collections are mutated.** `xs = []; fill(xs, n); return
  len(xs)`.
  - Wrong record: the fold treated the empty literal as `xs`'s contents and
    returned 0.
  - Rule: a collection passed into a call that may retain or mutate it is
    mutated. This is the one escape definition the capacity survey uses
    (`_call_escaped_argument_ids`).
  - Such a list is now an honest `dynamic-growth-unavailable` refusal
    (below).
- **Refusal covers call-result columns.** `ys = make(n)` silently returned
  0: the root view's column is the call's output slot, so neither wrapper
  published capacity. Both wrappers now refuse any root-owned capacity
  over an unproven column.
- **9cc6910 regressions found by bootstrap snap6:**
  - units 060/064: pursued definitions were not made binary. Every pursued
    tree is now normalized first.
  - units 048/049: `if serial < 0 or serial > bits: raise` placed the
    guard inside the `or`'s own arm. A guard on a conditional merge now
    follows that conditional.
  - unit 072: a comprehension authored in a short-circuit operand had no
    arm. The BoolOp's guarded operand is its arm. A region-less conditional
    is never nested past an intervening loop. An inserted loop takes over
    the region markers it owns.
  - unit 043: a latch-generation row load was minted with no dtype. It now
    takes its identity's recorded row dtype.

## Lists across call edges and f-strings

Each repair below names the program, the wrong record and the rule.

- **One capacity contract per storage.** `xs = []; fill(xs, n)` (c55) and
  `ys = make(n)` (c56). `_settle_linked_dynamic_sequences` joins the views
  of one storage over the exact `linked_value_edges`.
  - A component is DYNAMIC in every view when it grows without a bound
    anywhere: an unbounded local list, an append inside a loop, or any
    append to a parameter or call result.
  - Every view publishes `linked_dynamic_sequence_ids`, and descriptor
    birth reads it.
  - A parameter view's owner is a parameter member.
- **The owner is a storage member.** `_storage_values` emits the heap owner
  last, so every helper built from a heap descriptor takes it. The
  hand-added owners in allocate, reserve, the dynamic entry and extend are
  gone.
  - Pruners keep it through one rule, `_retain_companion_formals` (see
    the next section); the descriptor-liveness protection first added to
    the structural-recovery filter was circular and is gone.
  - Before this, f"a@{n}"'s token list lost its owner. The caller then
    leased a second owner for the same arena.
- **Binding refuses two contracts.** `_bind_sequence_storage_members` now
  raises when one bound view is DYNAMIC and the other is not. It used to
  return False, and the linker gave the callee fresh length/capacity cells:
  `fill` appended into storage the caller never read.
- **Returned sequences bind the owner, and owner ids are translated.** The
  returned-sequence binding includes the owner. The callee output's
  accounting was copied raw into the caller's result, naming the callee's
  owner id. An owner id now crosses only through the call's storage
  binding.
- **Element type is one fact.**
  - A mutable sequence argument takes the callee formal's declared row
    contract when the caller states none (`fill(dst: list[int])`).
  - A single-column sequence's settled linked contract is its declared
    column dtype.
  - A loop target over a call-result list reads the iterable's linked
    contract, not the domain's provisional float64.
- **Glyph sequences.** A JoinedStr recipe destination and the glyph
  projections hold int64 StringTable tokens, declared at birth. Within a
  function, a sequence `extend`/`replace` fills from a DYNAMIC source is
  DYNAMIC. Together these make f"a@{n}" DYNAMIC before its first step's
  helper is lowered.
- **Helper calls bind the live helper.** Each literal append re-registers
  the same helper name, so `callee_input_ids` are stamped once the
  function is complete, from the live helper's formals. This includes
  calls between helpers.
- **Extend walks int64.** The loop index and source length now match the
  int64 length cell. LLVM had refused the i32 index Phi.
- **Bare returns in procedures.** `if v == 0: dst.append(7); return`
  (c59). A value-less return produced no control, so "0" was spelled
  "00". It is now the return edge with zero slots.
- **Mutable escapes only.** The fold's escape rule counts only mutable
  collections. Builtins that read or copy a collection (`bytes`, `sorted`,
  `sum`, ...) do not escape it. Before a call's extraction identity is
  resolved, its static builtin symbol names it.

## Scope-local identities, one owner per arena, returned lists

The user asked for whole categories, not single exceptions. A static scan of
`src/` (scratch `inventory.py`) listed the writers that bypass the books:
descriptor rebuilds that drop fields, accounting copied from another value,
dtype/shape assignments, positional function picks and formal-list rewrites.
Each category below now has one rule and one helper.

- **Descriptor remaps.** Five linker sites rebuilt a descriptor field by
  field and dropped the fields they did not name, the heap owner among them.
  `map_sequence_descriptor` maps every identity and keeps every other field.
  The "all members mapped" guards read `_sequence_descriptor_identity_ids`.
- **Accounting across a call.** `_accounting_across_call` is the one
  translator for the callee result, the aggregate-unpacked output, and both
  caller-actual -> callee-formal ownership passes.
  - A scope-local id (type tag, heap owner, optional payload, iterable
    target, tensor shape/rank/count) crosses only through the call's exact
    binding.
  - A formal fed a heap cell without its owner takes no heap contract.
    `found.append(Finding(..., detail))` had copied the root's owner id
    onto the append helper's row formal.
- **The audit checks it.** New finding `accounting-foreign-value-id`: a
  scope-local id must name a value of the same function. Its first catches:
  - unit 24 (`page`'s tag formal pruned);
  - `concordance_controller` `pi_update` (a union merge's tag Phi pruned).
- **Companions are retained.** `scope_local_accounting_ids` is the single
  definition:
  - `_retain_companion_formals` applies it in every formal pruner;
  - `prune_unused_phis` counts named ids as uses;
  - `drop_dead_pure_structural_instructions` protects them, replacing its
    own tensor-key list.
- **Dynamic dtype at birth applies to the dynamic set only.** `key: str`'s
  span contract had been declared as a sequence column, and the lookup's
  `Eq` became the tensor kernel `binary_double`.
- **One owner per arena.**
  - An optional glyph payload (seq 19) is born on its f-string arena's data
    cell (seq 11). It adopts that cell's owner.
  - Optional-glyph payload copies join the dynamic closure.
  - `replace` grows a DYNAMIC destination through its checked reserve, and
    publishes only the length when source and destination share the arena.
- **Builtin collection methods do not escape.**
  `scopes.setdefault(k, {})`'s `{}` is the table operation's default, not
  an escape into unknown code.
- **Arena order is access order.** `r = []; if n < 0: return r;
  r.append(2*n); return r` (c60).
  - The append publishes its arena id as a value, and `dependency_order`'s
    later-producer rule pulled it ahead of the guard. An arena feed now
    never takes a later mutation as its producer, and a return reads the
    arenas it returns.
  - The exit merge `Phi(1, 1)` minted a phantom sequence. A merge of one
    identity is that identity.
  - A local list returned to a binding caller seeds the dynamic settlement.
- **Accepted components are immutable in optional lowering.** Unit 35
  lowered unit 33's record-covered call a second time (4 projections for 2
  receipts).

Measured: corpus 59/59, 0 findings. Gate at a8ad2b0: 125 failed, versus 128
at g11 (3 fixed); the 2 new ones are fixed at 9f1643c. Focused suites: 20
failed versus 22. Units 24, 33 and 35 compile on replay; 34 stops at the
variadic return lane.

## Loop exits, sequence effects and one identity per display member

Found by core-semantics probe batches (scratch batch1/batch2, run against the
authored Python with a type-checking harness). Each entry is program -- wrong
record -- rule; every rule is applied at the pass that lost the record.

- `for ... else` / `while ... else`: normalized into a break flag
  (`_normalize_loop_else`). The structural fold leaves a post-loop read of a
  break continuation or carried update unresolved. A `for` target read after
  the loop is carried (pre-loop value, element). A name assigned this loop's
  own target is not a descendant-loop induction. The C lane copies the
  Phis on an edge as one parallel copy (`i = -5; for i in range(10)` returned
  10 for 9).
- List display members: an append's identity is (display, ordinal)
  (`ControlSequenceMutation.display_member`), not the member's producer.
  The producer key misplaced `[5, n, 2, 9]`'s `n` after the sort that read
  the list. It also retired `k = n + 1`'s region when `k` sat in two
  displays, which minted `k` as a root formal. Placement, guards, effect
  dedupe, installed-effect matching, region retirement, the completeness
  audit and contract provenance all read the display.
- `sorted()` of a private list is heap-owned DYNAMIC. Its sort workspaces
  are the function's own storage, reserved to the row count; they were
  caller-provided root formals with capacity 0. A copy states the
  destination's heap contract once its layout is concorded.
- In-place sequence methods are retained effects.
  - Before, only `append` and `add` were retained, and `pop`, `extend`,
    `remove`, `insert`, `reverse` and `pop(i)` vanished silently.
  - New helpers: `lower_sequence_delete_at`, `lower_sequence_insert_at`
    (Python's index rule, DYNAMIC reserve-and-retry) and
    `lower_sequence_reverse`.
  - The structural fold no longer reads a method-mutated receiver's
    initializer as its contents (`_method_mutated_receiver_ids`).
  - `_mutated_sequence_node_ids` is the one definition of a collection
    whose contents change after its initializer. The fold and region
    planning both read it; the latter had folded a stored-into display
    into a region `Const`.
- A sequence effect outside a loop precedes every later read of its receiver
  (`_sequence_effect_read_edges` -> region reducer). An append's computed
  argument was fused with the post-append reads into one region (cycle).
- A reader of an arena follows the arena's display birth (extend of a
  display argument). A captured row load is a read of its index and sequence
  (`xs[len(xs) - 1]` read its index before definition).
- Types:
  - A row load revises its `ssa_value` row to the column dtype.
  - An arena with no concorded row is born `unknown`, not the planner's
    float64 capture guess.
  - A region's arena formal reads its element type, or heap layout, from
    the descriptor.
  - `_planned_scalar_dtypes` types plan-time values from the source:
    literals, annotated parameters, `range` targets, the shared region
    operator rules (`hierarchical_plan.REGION_*`), and a collection's
    element type joined from its writers.
  - A DYNAMIC sequence whose element type is proven is declared with it at
    birth.
- Reductions (`max`/`min`/`sum`/`any`/`all`) over a sequence use its live
  length (`sequence_extent_parts`) and read heap arenas through their cell.
  `max`/`min` of an empty sequence raises through the validation runtime,
  which the C lane now links (`turing_validation_error`), as do `pop` and
  `remove` on absent rows.
- A dict subscript is a key lookup. The fold had aliased `d[1]` to the
  dict's second ledger leaf, a key (silent).
- Chained comparisons are the conjunction of their adjacent comparisons
  (`_normalize_chained_comparisons`). A chain was dispatch metadata and read
  a minted input (always 0).
- Smaller fixes:
  - A minted indexed store states its site's span, so `d[1] += 5` is placed
    at its statement.
  - The builder prologue counts post-scheduled table operations as
    scheduled.
  - Appending a sequence length feed revises the region signature, so block
    geometry has its cause.
  - A loop-grown private sequence is DYNAMIC whoever minted it
    (comprehension collections).

Measured at 6300d55:
- Corpus: 107 programs, all value-correct with 0 concordance findings. Four
  return a float where Python returns an int (c144, c145, c152, c156).
- Gate (gate12 file split, 49 files): 123 failed, the same set as at g16.
  No new failure across g17 to g21; gate22 is pending at the time of
  writing.

## Differential fuzzing: one storage per value, one position per block

A differential fuzzer now generates programs from composable snippets
(lists, dicts, sets, tuples, loops, `break`/`else`, short-circuit, helpers;
an extended set adds `enumerate`, `zip`, membership, `count`/`index`, float
math, early return, concat, replication, `del`, `while`-grown lists,
index-loop stores). It compiles each through `lower_ast_source_to_ssa` and C,
compares the result with Python (value and type), and runs each native call
in a forked child so a segfault is a result, not a lost run. The harness
also applies the native-emission product's own refusal
(`_unexplained_root_argument_ids`), which `emit_ssa_to_c` alone does not
check. Five seeds, 280 programs. Every failure below was minimized to a
corpus program, traced to the record that was wrong, and fixed there.

Silent wrong answers found and fixed:
- `ys = list(xs); ys.append(n)`: the copy was a materialization alias of
  its source, so `ys` shared `xs`'s storage (`len(xs)` read 4). A recorded
  snapshot owns its storage (`_sequence_concat_ops`).
- The same copy took its source's capacity, and its own appends were not
  counted. A snapshot's bound is the source's plus its own straight-line
  appends.
- `[5, 7] + [n, 7]` took the planner's elementwise-add extent (2) as its
  arena and wrote 4 rows (a native segfault once its fill ran). A concat is
  bounded by the sum of its operands.
- `[2] * (n % 4 + 1)` took the literal's one-row extent; its fill reported
  "full" into a status nobody reads. A fixed fill that exhausts its
  capacity now raises (validation code 75). A runtime-count replication is
  unbounded (DYNAMIC). Its fill reserves its count, but C emission still
  refuses it (the heap cell keeps the planner extent `(1,)`): loud.
- The fill's count `n % 4 + 1` was consumed only by the replication, which
  left the region as a table operation; nothing published it. Values read
  by a region's recorded table operations cross the region boundary.
- A concat's reset/append_slice claimed to replace its region, so the
  region's other work (that count, and the fill after it) was skipped. An
  operation replaces its region only when the region has nothing left to
  publish.
- `xs.count(v)` / `xs.index(v)` have no lowering. Their results become root
  inputs, which the native product refuses; the harness now applies that
  refusal.

Placement and identity faults found and fixed:
- Five lexical placement passes each kept a private `block_position`; only
  one positioned a `SequenceQueryBlock`. `_control_block_position` is now
  the one rule. A `for` loop is placed where its iterable ends, and its body
  lines are its span (the `len(xs)` of `range(len(xs))` was placed inside,
  then after, the loop).
- Region placement took the first member in four passes and the last
  (`_dispatch_region_completion_positions`) in one. All five use the
  completion rule.
- A birth takes no WAR edge from an earlier reader; the birth rule already
  orders the reader after it. A birth is a display's clear and member
  appends, or a region whose recorded operations reset or fill the arena (a
  concat or replication result).
- An effect inside a loop is observed after its outermost loop. That loop
  now precedes later reads of the receiver (`_sequence_effect_read_edges`).
- Heap columns are `ptr` cells, shapeless, in the owner as in every helper.
  The repository-SSA transport posted int64/ptr alternately on one shape
  edge. Readers of the element type read the descriptor or
  `heap_element_dtype`.
- `s.add(x)` was classified numerical work because `add` is a tensor op.
  A recorded sequence mutation is a control effect.
- Loop growth makes only births DYNAMIC (aggregate, comprehension,
  `LoopResult`, generator); record fields and mapping values are views. A
  local dict with no proven key universe is unbounded and DYNAMIC.
- A loop over a comprehension iterates the result its producer loop
  records (`LoopIterationOutput`), not the `ListComp` node.
- A bare `zip(xs, ys)` iterates its argument sequences, bounded by the
  shortest.
- A slice is never a row read.
- `_planned_scalar_dtypes` reads a constant collection's literals, a linked
  value's posted ABI contract (`linked_value_abi`), `builtins.int`, and a
  `pop`'s receiver. All four int-as-float corpus results now return ints.

## A list passed to a call is one storage at both ends

The fuzzer's call-edge programs found three silent wrong answers with a
clean audit, present since before this round (686fe50):
- `setfirst(xs, 9)` on the caller's `[n, 4]` returned the caller's old
  `xs[0]`; `add_one(xs)` lost its append; an empty list grown by two
  appends and then stored into by a callee read length 1.

The chain was observed with traps on the writers:
1. Callsite specialization split the caller's display into per-element
   member formals (`_apply_callsite_aggregate_descriptors`), the value view
   used for tuples.
2. The plan bound the caller's members to them. The callee's whole `xs`
   formal, which its store or append still read, had no binding.
3. Frame linking found no source for it and minted storage from absence
   (`STORAGE_MINTED_FROM_ABSENCE`). The callee wrote that storage; the
   caller never read it.

Rule (`_aggregate_binds_by_member`): an aggregate is bound by member only
when neither side changes it and the callee reads it only through
constant-index projections. Otherwise it is one storage, bound whole. The
callee side reads the one mutation definition (`_mutated_sequence_node_ids`)
and, for a mutable collection, any other use (`for x in xs`, `len(xs)`,
`xs[i]`). The caller side reads the same definition without call escapes:
`xs.append(7); size(xs)` had bound the callee to the two display leaves
of a three-row list. The plan builder reads the decision the
specialization recorded (a child formal with no members and no leaves);
it does not recompute it on the specialized copy, whose folded literals
can disagree (`pick(xs, 2)`).

Binding whole exposed three more faults behind it:
- An escaped local list had no row bound, so its arena became entry-ABI
  storage that the exported wrapper sized at one row (`len(xs)` read 1).
  The settlement (`_settle_linked_dynamic_sequences`) now also publishes
  `linked_bounded_sequence_ids`: a local list whose every escape is an
  exact argument binding, into a component no function grows, or a
  by-member binding, is bounded by its own rows.
- A call that receives a mutable collection is an effect on it, as an
  append is. Without that edge the reducer fused `n % 3 + 1` (the call's
  argument) with the reads after the call into one region placed before
  the call (`_sequence_effect_read_edges`).
- `first(xs)` returning `xs[0]` is the caller's `n`, by the planner's
  identity forwarding. Linking published the callee formal's provisional
  `int` and its view label onto the caller's int64 formal. A call whose
  result is a caller formal now reads it without redefining its row. A
  recovered member formal is typed by the planner's scalar rule, which
  reads a parameter's settled linked-ABI class when it has no annotation
  of its own.

The new fuzz tier then found three more on the same boundary:
- `pick(xs, n % 3) * 10 + width(xs)`: both calls bind the caller list's
  length cell through the exact argument binding and the descriptor's
  member roles. The frame-slot owner check read record fields only, so
  each call leased the cell for itself and the second binding was split
  to a copy. A callee formal that is member ROLE of a sequence bound
  exactly to caller sequence S, where the slot is member ROLE of S, is
  owned by (S, ROLE) at every call.
- `ys = [0]; grow(ys, 2); width(ys)`: `width`'s callsite copy took the
  display literal `[0]` as its formal (static specialization) and folded
  `len(xs)` to 1. The caller's tensor descriptor had the same fault. A
  callsite specializes on an argument only when its initializer is its
  contents at that call (`_callsite_argument_is_initializer`): the
  caller's own effects do not change it and it escapes into no other
  call.
- A by-member binding reads only the members the callee projects, so a
  split counts as a non-growing site when any leaf is bound and the
  aggregate itself is not.

Corpus c180 to c193 hold the family. A new fuzz tier (`FUZZ_EXT=3`) adds
helpers that store into, append to, grow, swap, index and measure a
caller's list.

## Open frontier (stated, not worked around)

- **Variadic returns.** `return ()` / `return (x,)` and `return []` /
  `return [x]`: the reducer classifies VARIADIC, but the sequence-lane
  lowering is not built (units 34, 162, 212; scratch c60_vartuple and
  c61_varlist).
- **`'|'.join(tokens)` in an f-string** has no glyph recipe (unit 6).
- **Dict `update()`.** `clear()` on a dict is the resident clear (corpus
  c63/c64), and a local dict filled in a loop is now DYNAMIC (c161).
  `update()` is still not a mutation. `identity_spill._replace_cells`
  (units 45/206) has this shape.
- `test_forwarded_tuple_field_has_one_sequence_resident`: `ensure`'s arena
  row says `opaque_ref` while its descriptor says float64. `total`'s lease
  copies the row before the canonical sweep retypes the occurrences.

- Generic `str()`/`!r` of `Any`-typed facts (ten f-string units): needs a
  runtime type dispatch over every fact class.
- Planner region inputs still default to float64 when nothing proves a
  type (measured: about 3,000 fallbacks across the corpus). Later stages
  re-type every one that reaches a corpus result; all 121 corpus programs
  return the Python type.
- Strings as values (`s = "abc" if c else "hello"`, `len(s)`, `s == "abc"`,
  `s.startswith(...)`) are dispatch metadata. Nothing lowers them, so the
  value is minted as a root input; the concordance reports
  `unaccounted-formal` (c114, c217). A string-keyed dict refuses (c241).
- Loud refusals, each with a named record:
  - `d.get(k, 0)` in a loop: control order cycle (c202).
  - A standalone slice `ys = xs[1:n]`: no lowering (only `r += xs[a:b]`
    is recognized); refused at C emission (c204).
  - `xs[i % 4] += i` in a loop: a carried read with no lexical binding row
    (c211).
  - Nested list stores: no source cell for the control block (c212).
  - Early `return` from a loop over a list parameter (c216).
  - `n in xs` with `xs` a folded constant list VARIABLE: no membership is
    produced (the structural recovery only covers fields and dicts). An
    inline numeric literal (`n in [6, 3, 4]`) is now spelled as the
    disjunction of equalities (ast_control_normalization).
  - Dict `del d[k]` and `d.pop(k)` on a display dict: the deletion is
    placed by graph-id order (`value_id < effect_id`) into a neighbouring
    region (a cycle), and `pop` has no lowering. Both should be one lexical
    mapping effect.
  - Runtime-count replication `[v] * k`: see above.
- Now lowered: `xs.count(v)` / `xs.index(v)` (resident queries; a missing
  `index` raises), `del xs[i]` on a list (positional pop), bare `zip`.
- `xs = [0, 0]; for ...: t += xs[0]; xs = [i + n, i]` (a display re-bound
  inside a loop) is wrong at 4c70971 and after, with two audit findings
  (`descriptor-member-shared`, `unaccounted-formal`).
- Lists across call edges: growing (c55), returning (c56, c60), storing,
  appending, iterating, indexing and measuring (c180 to c193) pass.
  Still loud:
  a dict display passed to a callee that adds a key has no capacity
  contract (dict escapes are not settled); a `list[int]` parameter bound
  whole and iterated into another list types its loaded rows with the
  provisional `int` element spelling ("declared integer width
  disagreement").
- The exported C wrapper still sizes private root storage with no storage
  requirement at one row (`ssa_c_backend`, `element_count ... else 1`).
  No program reaches it now that escaped lists are bounded or refused, but
  the default is not a proof.
- `any(generator)` over a loop generator is now the resident query, as
  `sum` was (corpus c61). A reduction over a STRUCTURAL aggregate whose
  members were unrolled at reduction still miscompiles: `any((1 > n,
  2 > n, 3 > n))`, `any([x > n for x in xs])`, `any(x > n for x in xs)`,
  `sum(x * n for x in xs)` with `xs` a literal list. The audit flags each
  (`has no storage` / `neither a formal nor defined`). The region takes
  the aggregate as an unproduced input, and no layer assembles it from its
  members.
- `t = (n > 0, n > 2, n > 4); len(t)` returns 0 with a CLEAN audit (also at
  HEAD). The chain was observed:
  1. `_list_literal_mutations` births only lists, so the tuple's length
     cell is the frame's zeroed storage.
  2. A tuple birth (tried, then reverted) emits the appends. But the
     members' producer, region 0, is owned by the later conditional as its
     predicate region (`t[1]`), so the appends read the members as formals.
  3. The shell's structural recovery then rebuilds `n > k` from `n`, after
     parameter naming has dropped `n` as declared-only.
  A fix must place a predicate region before the effects that consume it,
  then birth tuples like lists.
- Local class construction from a source-derived schema is
  `constructor_complete=False` by design (c19).
