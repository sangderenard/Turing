2026-10-26 CONTROL-OWNED LOOP CONTINUATION IDENTITIES (MEASURED, Windows):
# Control-only carried loops and exact returned continuation

After the separate loop-class provenance fix, `last(values:list[float])`
with `previous=0.0; for value in values: isinstance guard; previous=value;
return previous`, called with `[1.5]`, audited clean but returned0.0.
The typed LLVM emitter mirrored the final SSA: its callee contained only
Ret(formal0), with no loop or Load. A source-side trap observed builder
finish Ret[] and later source-output recovery exposing raw loop Input0 as
that formal. No backend return-value patch was made.

The observed producer chain and repairs are:

* Evaporation rejected the authored If AST even after its graph predicate
  folded away. The retained realization checked a weaker live-control
  predicate and left the rejected UNROLL plan without iterations. Both
  decisions now consume the SAME existing eligibility predicate and select
  the incumbent native source realization when iterative control owns it.
* LoopResult port10 moved graph.roots0→10 and previous's binding history,
  but return_site_slot, semantic_output, and positional result_0 still
  named0. The port transaction now reads the exact ingestion Ref through
  graph.node_map to identify the authored returned Name. It updates only
  that binding's continuation slots/output, keeping the loop's own returns
  on their iteration values. No global alias0→10 is posted.
* Shader-loop analysis rejected the resulting control-only loop for having
  no shader region. Its declared carried binding is actual coordinator
  work (static iterable Load and Phi), now admitted beside the existing
  sequence-mutation/source-call work exceptions.
* `_remove_dead_metadata_nodes` retired Constant2(seed0) although the live
  loop_carried_bindings record still consumed it. Its eligibility writer
  now protects exact carried/break pairs, target initials, and break-site
  values. No generic integer-attribute scan or capacity default was added.
* A reserved unknown backedge could reach its typed producer before any
  region declared its type. The producer now resolves the same undefined
  non-argument LOOP_SEED object and SSA_VALUE row from its actual address
  cause. Concrete reservations and arguments keep their existing behavior.

Measured verification:

`python -m pytest -q tests/test_llvm_singleton_iterable_layout.py -x --tb=short`

29passed20.86s:7 raw LLVM aggregate native cases,7 public source programs
on C/LLVM/Fortran (two native activations each), and the focused reservation
writer test. Float singleton/multiple values, signed int64 values above
2**53, and bool True/False all match authored values exactly. Every source
audit is zero findings; source metadata asserts the returned exit identity
is distinct from the body's shared iteration producer and has no empty Ret.

`python -m pytest -q tests/test_loop_composer.py -k 'retained or return or carried or seed or break or static_iterable' --tb=short`

12passed1.48s.

`python -m pytest -q tests/test_structural_return_site_pruning.py tests/test_native_loop_predicate_class_abi.py tests/test_llvm_native_integer_types.py --tb=short`

31passed14.45s. Diff checks are clean for this lane.

A broader100-case LoopComposer attempt reported two failures and then
stopped advancing after about60 outcomes. Only its verified pytest PID23532
was stopped; no full-file completion is claimed. The next collected test
was test_declared_tuple_parameter_resolves_isinstance_schema_guard, but no
stack trace proved that it (rather than teardown) was the active slow code.
Both reported failures were then isolated on shared and untouched clean
C:/dev/tg-child-review at6f98a0b9 after reading its hazards:

* test_mutable_public_tensor_publishes_shape_without_freezing_value:
  formal_shape(tick,state) Derived names no source cell.
* test_declared_tuple_loop_target_resolves_exact_type_guard_for_mutation:
  the same three dangling minted value IDs.

Untouched baseline:2failed3.93s. Shared:2failed3.71s. These measured baseline
failures remain open; no audit guard or fixture expectation was weakened.
No full manager, bootstrap, or unrelated returned-arena repair was run by
this lane. This receipt supersedes the intermediate public-source blocker
status in the singleton LLVM aggregate receipt.

Root clean integration checkout C:/dev/tg-child-review at769351d9: combined native iterable layouts, mixed integer conversions, scoped extend children and helper scope/direct-copy tests completed63passed3deselected40.88s. The3deselected cases are the explicitly tracked returned-array SOURCE native failures; no full green claim includes them.
