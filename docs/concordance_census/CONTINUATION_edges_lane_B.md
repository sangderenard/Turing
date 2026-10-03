# Concordance edges, lane B (process_graph_autograd decisions)

Rule (user, absolute, 2026-10-03): "nothing is valid in any way that doesn't
go through the concordance leaving edges".  Lane B owns three edge-less
decisions in `src/compiler/process_graph_autograd.py`: argument_roles as a
node attribute (0519095d), unscoped AbstractTensor forward graphs
(de609156), the per-book rebuilt backward-rule graph (51b4cebe).

## 2026-10-03 baseline (tree as found, other lanes' WIP present)

`python tools/audit_identity_concordance.py` (all cases, 57 s), findings /
unsourced worklist:
view 493 rows 0 (unsourced-fact x57); toplevel 322/1 (onw x1, uf x46,
ui x10); energy 490/0 (uf x48, ui x10); controller 530/1 (onw x1, uf x53,
ui x1); controller_untyped 533/5 (onw x5, uf x52, ui x1); mapping 24/0
(uf x15); oscillator 331/0 (uf x29).  Matches the brief (0,1,0,1,5,0,0).
Note: turing/.venv lacks yaml; the gates run on the system Python 3.11.

Read probe (scratch, not in repo) on the graph-reverse VJP test
(`left*right`): at the fold the declaring Call is motion node 6 in scope
`ingestion:training_motion`; its ingestion row derives (fusion edge) from
`ingestion:adjoint` node 3, whose row is Unsourced(FORWARD_GRAPH_UNSCOPED)
-- the forward graph from `abstract_tensor_program_to_process_graph` has
no scope (decision 2 is upstream of decision 1's cells).

## 2026-10-03 Decision 2 (unscoped forward graphs) applied

`abstract_tensor_program_to_process_graph` mints `ingestion:abstract_tensor`
(stage INGESTION) and sets it as the graph's `ingestion_value_scope`.
- Page `abstract_tensor_program_value` (ingestion_scope, value):
  AbstractTensorValueFact(op, callee, shape, dtype); one NOVEL(INGEST_SOURCE)
  root per SSATensorProgram value the bridge reads (function args, Const,
  reshape/view, Call results).  The recorded program is the source.
- `ingestion_value` (scope, node) for every reachable node: DERIVED from
  the program value it ingests plus every program value that decided it:
  structural constants read through `const_value` (opcode, dim, indices),
  the broadcast buffers whose operands it recovers, and for a recognized
  mean the sum and its denominator.  CONCORD.
Result: adjoint rows are NOVEL(ADJOINT_OF, (forward cell,)); the verify
probe shows no `forward_graph_unscoped` row left in the reverse-VJP book.

## 2026-10-03 Decision 1 (argument_roles) applied

- Page `callsite_argument_role` (ingestion_scope, call, argument):
  ArgumentRoleFact(ArgumentRole.GRADIENT|OPERAND|METADATA) (Enum, not str).
  `_AdjointBuilder._post_argument_roles`, stage ADJOINT, CONCORD:
  GRADIENT DERIVED(gradient argument's adjoint ingestion cell);
  OPERAND DERIVED(bound forward value's cell); METADATA from an operator
  attribute DERIVED(forward operator's cell); METADATA from a signature
  default DERIVED(rule formal's cells via `_formal_identity_cells`: the
  Input identity cell + its `scalar_parameter` row).  No forward cell:
  Unsourced(FORWARD_GRAPH_UNSCOPED).
- The `argument_roles` node attribute is gone.
- `fuse_forward_loss_backward` (`_copy_argument_roles`): each copied Call's
  rows re-posted in the motion scope DERIVED from the backward row (stage
  FORWARD_LOSS_BACKWARD_FUSION).
- Reader `declared_argument_role(graph, node, position)` (read only, posts
  nothing): node identity cell (canonical, then ingestion), then DERIVED
  edges back through identity pages until a cell with role rows.
- glsl fold read site (`_propagate_callsite_planner_specializations`):
  reads the row; GRADIENT/OPERAND -> tensor_argument as before; the role
  cell is appended to the contribution's cells, so the
  `planner_specialization` row (Unresolved or fact) cites it.
Verified chain on the reverse-VJP probe: motion role row <- adjoint role
row <- (adjoint gradient cell | `ingestion:abstract_tensor` forward cells).

## 2026-10-03 Gates after decisions 1+2

- reverse VJP: 1 passed (9.9 s).
- native_scalar_loss_adjoint: 2 passed (28 s, one process).
- linear motion: 1 passed (21 s).
- orbital transfer: 12 passed (238 s).  The probe's LAW_BLOCKERS is now
  empty (9 laws), so the 8p/4xf in the brief is stale; not this lane.
- AT multi-output ingestion test: 1 passed.
- xor nested meta-learning: no fast test exists (only the example's
  `--stage` build); not run.
- audit findings 0,1,0,1,5,0,0 (unchanged).  unsourced-fact +1 in view,
  toplevel, energy, controller, controller_untyped (57/46/48/53/52 ->
  58/47/49/54/53).  Not this lane: a hooked run of the view case calls
  none of the lane-B functions and the book has neither new page; other
  lanes edited topological_reducer / tensor_ssa_lowering / ir_indexing
  between the two runs.

## 2026-10-03 Decision 3 (rebuilt rule graph): held, one question

Observed (reverse-VJP book): the rule graph's 3626 `source_span` rows are
module `<string>`, each NOVEL(INGEST_SOURCE, ()) posted by
`graph_express2.post_source_span` inside `build_from_ast`; nothing names
BACKWARD_RULES.  Also ~400 rule-graph `ingestion_value` rows are
Unsourced(SYNTHESIZED_NO_SOURCE) at stage REDUCTION (110 in
`ingestion:<string>`, the rest in `lexical_reads:<helper>` scopes).
Cross-book edges are impossible (books are separate); the link has to be a
per-book rule-definition row whose key is the rule's identity.  The design
doc gives every source construct one NOVEL root and does not say whether a
synthesized source's spans may be that root.  Question raised (see report).

## 2026-10-03 Decision 3 applied (coordinator: YES)

- Page `backward_rule_definition` (registry SCOPE, name NAME):
  BackwardRuleDefinitionFact(BackwardRuleKind.RULE|HELPER, sha256 digest).
  `_post_backward_rule_definitions` (process_graph_autograd) posts, per
  book, NOVEL(INGEST_SOURCE) roots, CONCORD: ("BACKWARD_RULES", opname) with
  the digest of the entry's `python` declaration (parameters, body), and
  (helper.__module__, helper.__qualname__) with the digest of its
  inspect.getsource text.  89 rows.
- `build_from_ast(source_cells=None)`: {top-level definition name: cell}.
  Stamps `_turing_source_cells` on every node of that definition and keeps
  the map on the tree; `_annotate_visual_source_owners` re-stamps from the
  map (nodes a rewrite made later, e.g. annotation BinOps), definitions get
  their own cell, module-level nodes all cells.
- `post_source_span(..., source_cells=None)`: cells given or stamped (node,
  else owner definition) -> DERIVED(cells); a revision DERIVED(cells,
  previous span cell).  None = authored file, NOVEL root as before.
- Result (reverse-VJP book): rule-graph span cells 3647, NOVEL roots 0
  (was all 3626 rows), 3823 edges into backward_rule_definition cells.
- The ~398 Unsourced(SYNTHESIZED_NO_SOURCE) `ingestion_value` rows at
  REDUCTION are NOT rule-specific: topological_reducer `new_node` callers
  that pass no `source`/`source_cell` (StaticReference 220 via
  static_reference_node / first_class_function_node, `str` 104 via
  `_set_operands` -> node_identity_cell fallback from build_graph.connect,
  captured Inputs 32, Phis 20, ...).  Same callers do it for every
  authored program (the plan-70 step-3 worklist).  A StaticReference is
  cached per key and shared by uses, so no single span is its source; the
  derivation is not local.  Reported, not edited.
- Stale `argument_roles` comment in concordance_declarations (~372) fixed.

Gates: VJP + scalar_loss x2 + linear + AT multi-output in one process:
5 passed (57 s; several reverse compiles, one book each).  Orbital 12
passed (280 s).  Audit findings 0,1,0,1,5,0,0; unsourced 58/47+10/49+10/
54+1/53+1/15/29 -- identical to the post-1+2 run: the audit cases are
forward compiles and never build the rule graph, so lane B reduces none of
their counts.  Reductions are in reverse-compile books: forward_graph_
unscoped rows -> 0; rule-graph span roots from nothing -> 0.
