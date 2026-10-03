# Continuation: orbital work item 3, external functions supplied at runtime

Base: main ce587b82. Uncommitted. 2026-10-03.

Goal: the orbital set's applied undefined Functions (r1..r3, F1..F3) compile
as declared externals called through an ABI slot bound at load, with the
host's r()/F() seam (engine_toy/orbital_jumper.py OrbitalJumper.r / F)
supplying them. Derivatives of externals are item 4; the declaration must
be able to carry an optional supplied derivative.

## 2026-10-03: probe written, baseline failure

`tools/compiler_probes/probe_external_function.py`: one law `y = 2*f(t) + t`
-> `compile_sympy_equations` -> `piece_from_law` -> LLVM and C, called with a
Python-supplied `f` (`host_f = cos(t) + 0.25`). About 8 s to the failure.

Baseline: `compile_sympy_equations` ok (the call is `Call callee='f'`).
`piece_from_law` fails at `fortran_c_shell.py:45685
_lower_ast_source_to_ssa_impl`:
`full-native execution contract rejected the linked repository SSA:
undefined_operands=1` (value `t2`, the `f(t)` result, feeding
`external_probe__external_probe__planned_region_0`). Same failure class as
the orbital laws `force_cost_integral`, `initial_condition_lhs`,
`terminal_condition_lhs` (CONTINUATION_orbital_probe.md, failure 3). Cause
unchanged: the materialized AbstractTensor source spells `t2 = f(t)` with
`f` unbound in `python_bindings`.

## 2026-10-03: search for an existing host-callable ABI

What exists:

1. **None in the C module lane (`ssa_c_backend`), the LLVM lane
   (`ssa_llvm_backend`) or `native_package`.** No function-pointer slot,
   callback table or bind-at-load surface. The public ABIs are
   `void entry(void **buffers, long long *extents)` (C module) and the LLVM
   buffer ABI (`prepare_artifact_execution`).
2. **The llvm_piece seam (link time, not runtime).** A callable in
   `python_bindings` DECLARES itself (`extraction_contract.llvm_piece_of`:
   `artifact` + `argument_ids` / `output_ids`); `ExtractionContract.decide`
   short-circuits to `USE_NATIVE` / `native_abi: llvm-buffer` /
   `loader: static-link`; `fortran_c_shell` links the piece's SSA as
   `linked_repository_ssa` and tags its root `metadata["llvm_piece"]`;
   `ssa_c_backend` (~2234) emits the tagged function as a shim building
   `void *piece_buffers[]` + `int32_t piece_extents[]` and calling
   `extern void <symbol>(void **, int32_t *)`; `ssa_llvm_backend`
   (`_piece_symbols`, ~3559) does the same in LLVM. The symbol is resolved
   by the linker.
3. **The shell external-reference seam (runtime, host-serviced).**
   `fortran_c_shell` (~12560-12680) turns a `use_native` call boundary whose
   parameters name `execution: shell_io.external_references` into an
   `ExternalReferenceCallBlock`; `precompile_to_ssa` (~6430) lowers it to an
   SSA `Call` carrying `external_reference`, `external_identity`,
   `external_callsite_id`, `native_abi`, `result_dtype`, ... Arguments are
   forced to `opaque_ref`; it is an ordered effect placed in the control
   tree. Only the Fortran lane emits it (`ssa_fortran_backend`
   `_external_reference_call`: one thunk
   `turing_external_reference_<digest>` per callsite). The C shell adapts
   only `native_abi: cpython-c-api` (a private embedded CPython, tagged
   record ABI in `shell_external_references.py`); any other `native_abi`
   raises "C/Fortran shell has no adapter for external native ABIs". The
   full-native gate (`_full_native_link_failures`) rejects `cpython-c-api`
   and python shell profiles, but would accept a USE_NATIVE boundary with
   another `native_abi` and `callbacks: reject`.

Neither seam is a runtime function-pointer slot in the C/LLVM lanes. Which
one an external rides is a design choice; stopped and asked (below). No
source edited besides the probe and this file.

## Open question (asked 2026-10-03)

Should a declared external lower through (A) the llvm_piece seam (declared
callable in `python_bindings` -> contract short-circuit -> tagged leaf
function -> C/LLVM shim calling the piece buffer ABI
`void (void **buffers, int32_t *extents)`, with the symbol replaced by a
function pointer read from a slot table the host binds at load), or (B) the
shell external-reference seam (`ExternalReferenceCallBlock` -> SSA `Call`
with `external_reference` attributes -> per-callsite thunk, with a new
`native_abi` adapter added beside `cpython-c-api` and implemented in the C
module and LLVM lanes)?

## 2026-10-03: decision (coordinator): (A), the llvm_piece seam extended

A declared external is the piece-shaped leaf `void (void **buffers,
int32_t *extents)` tagged `llvm_piece` with `binding: runtime-slot`; C and
LLVM call it through a slot table the program exports and the host fills at
load; unfilled slot = loud error; external NOVEL (minted), callsites DERIVED.

## 2026-10-03: built; probe green in C and LLVM, both host kinds

New `src/compiler/external_functions.py` is the single owner:
- `declared_external_functions` (sympy `AppliedUndef` class -> name, arity);
  `compile_sympy_equations` records `metadata["external_functions"]`.
- `post_external_functions` (called from `_post_symbolic_program`): page
  `external_function` (program, NEW) NOVEL(`declare_external_function`,
  first applying equation cell); `external_function_name` (program, name)
  DERIVED, holds the minted id so a re-post reuses it; `external_callsite`
  (program, output, srepr(call)) DERIVED from (external cell, output cell).
  Pages declared in `concordance_declarations.py` (new section).
- `ExternalFunction` / `ExternalSlotABI`: the leaf. `llvm_piece_of` accepts
  an `ExternalSlotABI` artifact; `_decide_llvm_piece` gives
  `loader: runtime-slot`, `callbacks: reject`, `native_abi: llvm-buffer`
  (passes `_full_native_link_failures`). `fortran_c_shell` adds `binding`
  to the `llvm_piece` record and the external's `piece_record()`.
- Signature: specialized at the callsite shape, read by running the law's
  own AbstractTensor stage once with recording stubs
  (`external_callsite_shapes`). With an `LLVMPiece` implementation the leaf
  is that piece's (deep-copied) module and ABI. Otherwise the signature
  module is lowered through `lower_ast_source_to_ssa` from a declaration def
  whose body is `a0 * nan`: never emitted under the tag; a lane that ignored
  the tag would produce NaN, not a plausible number. Buffers = args then
  result; extents = (result numel,).
- C lane (`ssa_c_backend`): `static turing_external_fn <entry>__external_slots[N]`,
  exported `<entry>__bind_external(int32 slot, void *fn)` and
  `<entry>__external_fault_take()`; the leaf shim loads the slot, and an
  empty slot records `slot+1` as the fault and returns.
- LLVM lane (`ssa_llvm_backend`): `@<entry>.external_slots` global, the same
  binder / fault reader, and per-slot `@<entry>.external_unbound.<k>` stubs;
  the call site is `select(slot == null, unbound stub, slot)` then an
  indirect call (no block split, phis keep their predecessors).
- Artifacts carry `external_slots` rows (slot, external, ABI);
  `CModuleExecution.run` / `LLVMExecution.run` call
  `check_external_faults` and raise on a fault or a host-callback exception.
- Host: `bind_external_slots(artifact, {name: impl})` refuses a missing
  external; impl = Python callable (wrapped once with CFUNCTYPE over the
  buffer ABI; an exception inside it fills NaN and is re-raised after the
  run), `LLVMPiece` (its own entry address; buffer order must match), or a
  raw native address.
- `piece_from_law(..., externals=)` binds the declared externals.

Probe (`probe_external_function.py`, ~1 min): Python-supplied f and
LLVMPiece-supplied f, LLVM and C: max abs error 0.000e+00 for all four;
emptying the slot raises "external 'f' (slot 0) was called with its slot
unfilled" in both lanes.

## 2026-10-03: a root that only calls a piece took the single-function LLVM path

`initial_condition_lhs` (root = `r1(0), r2(0), r3(0)`, no planned region)
failed in `emit_ssa_function_to_llvm` -> `_kernel_signature` ->
`extract_llvm_function`: `KeyError: LLVM SSA symbol 'external_r1__r1' has
no function definition`. Cause: `_internal_call_closure` deliberately does
not follow `llvm_piece` leaves, so the closure is 1 and the dispatcher chose
the single-function emitter, which has no piece rendering. Static pieces had
the same hole. Fix at the dispatcher: a root that calls any `llvm_piece`
function goes to `_emit_repository_call_module`.

## 2026-10-03: orbital set results

`tests/test_orbital_transfer_compile.py`: 4 passed / 7 xfailed -> 8 passed /
4 xfailed (7 + the new book test). Flipped (native C and LLVM vs sympy,
host externals bound at load from `probe_orbital_transfer.host_externals`):
- `initial_condition_lhs`: rel error 0.000e+00
- `terminal_condition_lhs`: rel error 0.000e+00
- `equation_of_motion_rhs` (was (2, 3)): rel error 1.777e-15. It carries
  the Greek names mu_1 / mu_2: item 2 is not a blocker for it.

Still xfail:
- `force_cost_integral` -> item 4 (Integral). With the externals declared,
  the lowering passes the full-native gate and then the LLVM emitter
  refuses: `planned_region_0/1: get_tensor: operation has no repository LLVM
  emission` (audit gap 10). Separately, the law's own AbstractTensor stage
  no longer runs eagerly: since aa5f1aac (sympy int constants become
  floats) the quadrature's axis constants are floats (`t2 = -1.0`,
  `t3 = 0.0`), and `t1.unsqueeze(-1.0)` raises `TypeError: integer argument
  expected, got float` in `numpy_backend.unsqueeze_`. The callsite-shape
  read (`external_callsite_shapes`) runs that stage, so this law needs its
  externals passed pre-specialized (as `ExternalFunction`s) until the axis
  constants are integers again.
- `equation_of_motion_lhs`, `total_energy_expression`, `orbital_transfer_raw`:
  Derivative of an external, item 4, at `compile_sympy_equations`
  (`process_graph_autograd.py:2203`, no adjoint rule for `call`).

Work list: item 3 removed from `WORK_ITEMS`/`LAW_BLOCKERS`; item 4 text now
names the Integral's `get_tensor` emission gap too.

Gate: `probe_emission_chain.py` failures 0; `audit_identity_concordance.py`
findings 0,1,0,1,5,0,0 (unchanged).

## Open

- Item 4 plugs in at `ExternalFunction.derivative` (carried on the leaf's
  `llvm_piece` record as `derivative`); no backward rule reads it yet.
- One specialization per external per law: an external called at two
  shapes in one law is refused (`external_callsite_shapes`).
- Elementwise signature only (every argument at the result shape); a
  zero-argument external is refused.
- Lowered call instructions carry the external only through the leaf's
  `llvm_piece` record (`external`, `external_identity`); the book's
  callsite rows are at the symbolic level.
- The Fortran lane does not read `llvm_piece`; it would emit the NaN
  declaration body.

# Next lane (coordinator, after c03ae4e6): regression 0, item 4, craft probe

## 2026-10-03: step 0, structural constants (aa5f1aac regression) FIXED

Rule (`symbolic_equation_compiler._structural_constants`): a constant node
keeps its integer (dtype int64, attribute `structural_constant`) when EVERY
consumer reads it at a parameter its operation's declared `AbstractTensor`
signature annotates as an integer (`_structural_operand`: operand position
= signature position, receiver calls pass operand 0 as `self`;
`_declares_integer` parses the annotation). Any value consumer keeps it a
float64 number (the aa5f1aac Piecewise arm). Declarations added:
`AbstractTensor.sum/prod/max` `dim: int | tuple[int, ...] | None` (was
unannotated; `unsqueeze` already declared `dim: int`). The compile cache
record now carries `tensor_operation_signatures` (observed: a stale cache
entry served `sum(0.0)` after the annotation changed). Result:
force_cost_integral materializes `t2 = -1`, `t3 = 0` again.
Tests: `tests/test_symbolic_structural_constants.py` (Piecewise arms stay
float; quadrature axis constants int and the AbstractTensor stage matches
5-point Gauss-Legendre to 1e-13) -- 2 passed.

## 2026-10-03: item 4a, derivatives of externals -> declared derivative externals

`compile_sympy_equations(..., external_derivatives={r1: v1, v1: a1, ...})`
(undefined Function -> undefined Function the host declares as its
derivative). The ingestion's Derivative branch (`symbolic_process_graph`)
runs `sympy.diff` first, then `external_functions.resolve_external_derivatives`:
`Derivative(r1(s), (s, 2))` -> `a1(s)`, the chain-rule form
`Subs(Derivative(r1(x), x), x, g)` -> `v1(g)`. A derivative of an external
with none declared raises `UndeclaredExternalDerivative` naming it (no
finite differences; graph reversal would only meet an opaque call). What is
left (a Derivative of compiled content) goes to the existing graph reversal
unchanged. Call nodes from an `AppliedUndef` now carry
`attributes["external_function"]`; `metadata["external_functions"]` is read
from the ingested graph (so it includes v_i / a_i), plus
`metadata["external_derivatives"]`; the cache record carries the map.
Book: new page `external_derivative` (program, name) DERIVED from the
external's and its derivative's cells; callsites are posted on the resolved
form. `externals_for_law` links `ExternalFunction.derivative`.
Probe: `probe_orbital_transfer.external_derivatives()` declares
`r_i -> v_i -> a_i`; `host_externals` supplies v_i / a_i as the derivatives
of the host's own r_i. Results: equation_of_motion_lhs rel err 0.0,
total_energy_expression 2.2e-16 (C and LLVM).

## 2026-10-03: item 4b, the Integral lowers natively (audit gap 10)

Repro (seconds, AST lane): `t1 = AbstractTensor.get_tensor(<5 nodes>);
t4 = t1.unsqueeze(-1); t12 = L * t4; t27 = t12.sum(0)` with L a (4,) span.
Four defects, each fixed at its owner:
1. `get_tensor(<literal sequence>)` reached the planner as an opaque op
   (rank-0 extents, "get_tensor: operation has no repository LLVM
   emission"). `OPERATOR_ALIASES["get_tensor"] = "tensor"`
   (`operator_catalog.py`): the class's own constructor has the same
   idempotent normalization `asarray -> tensor` was aliased for; the
   canonical `tensor` path gives the literal Const its (5,) shape.
   (`test_public_abstract_tensor_api_is_explicitly_classified` fails before
   and after: atan2/complex/interp/... are unclassified, unrelated.)
2. `_tensor_descriptor_rule` (reductions) read the axis only from an
   `axis`/`dim` ATTRIBUTE; positional `x.sum(0)` settled rank 0. Now reads
   the `arg:0` literal, the same reading the unsqueeze rule gives.
3. Region `value_shapes` (`glsl_deployment_strategy` `_value_shape_dtype`)
   re-derived reduction / axis-insertion extents with its own rule (sum ->
   `()`, unsqueeze -> source shape) while `value_ranks` beside it read the
   descriptor. For sum/prod/min/max/any/all/unsqueeze/squeeze it now takes
   the descriptor's settled shape (one record, not two).
4. A capture made by a shape-only op outside the region (`t1.unsqueeze`) is
   fed its source's storage; `propagate_repository_ssa_call_metadata`
   restamped the formal (5, 1) with the feed's (5,) (trapped at
   `tensor_ssa_lowering.py:1203`). `PlanClosure.value_views` (new field,
   carried through the three PlanClosure rebuilds) records (capture,
   storage, view shape, op); the planner's formal declares
   `ssa_storage_view`, which the existing enrichment rule already honours.
Reference side (probe): an Integral SymPy cannot integrate is evaluated as
its declared Gauss-Legendre rule (`declared_quadrature`), the lowering's
meaning, so the check is of the lowering, not of the rule's truncation.
force_cost_integral: rel err 1.4e-16 (C and LLVM).

## 2026-10-03: item 4c, several callsite shapes per external

orbital_transfer_raw calls r1 at s (4,), at 0 and at L. `externals_for_law`
now reads the shape per callsite (each Call respelled `name__callsite_k`
for one eager run of the AbstractTensor stage), groups by shape, and gives
each (external, shape) its own leaf and slot (`r1__s4`, `r1__s`; a single
shape keeps the bare name). The respelling is applied to a copy of the
compilation (`_respelled`), never the cached one. The leaf's
`llvm_piece` record carries `external` (the declared name the host binds
by) and `leaf`; `bind_external_slots` fills every slot of an external with
its one implementation, wrapped per slot ABI. Derivative leaves link per
shape.

## 2026-10-03: the whole set compiles

`tests/test_orbital_transfer_compile.py`: 12 passed, 0 xfailed (was 8/4).
`WORK_ITEMS` and `LAW_BLOCKERS` are empty: items 1-4 resolved; item 2
(Greek names) needed no change (mu_i laws pass as written).
orbital_transfer_raw (16 inputs, all four Equalities): rel err 1.8e-15.
Gate: emission chain failures 0; concordance audit 0,1,0,1,5,0,0
(unchanged); regression set vs a clean HEAD worktree (408155a7, short
path, removed after): identical failure sets in test_precompile_to_ssa
(13), test_symbolic_fluid_native_runtime (1), test_aggregate_call_identity
(4), test_symbolic_equation_compiler (1), test_symbolic_process_graph (3);
translation scorecard 11/19 both; test_ssa_fusion_regions,
test_region_kernel_dedup, test_abstract_tensor_indexing pass.

## 2026-10-03: step 2, the craft and the benchmark are one system

`tools/compiler_probes/probe_orbital_craft_binding.py` (~3 min): the whole
set (the three Equalities as written plus `total_energy` and `force_cost`
named) compiled once (LLVM and C), externals bound at load to a LIVE
`OrbitalJumper` (Earth + Moon, 1000 kg, raw force (0, 0, 2) N via `F()`, 6
rounds to t = 1457 s): r_i/v_i from `r()`, a_i = (the craft's compiled
N4.1 piece at that position + applied force) / m, F_i = `applied_force()` / m;
s = recorded instants only (no interpolation; F, constant over the run,
also answers the quadrature nodes). Both lanes:
- equation of motion residual a - (F_grav1 + F_grav2 + F) <= 1.1e-16 |a|
  (the set's gravity IS the craft's catalogue N4.1);
- r(0) - r_start, r(L) - r_end exactly 0;
- total energy vs numpy at the craft states 2.6e-16 (craft drift over the
  run 1.2e-7, the integrator's);
- m * force_cost vs the craft's own fuel_impulse 1.6e-15.
Found on the way: in the whole set the quadrature's axis `-1` was the same
memoized node as the law's `-1` (`-mu * ...`), so its role was mixed
(numeric) and `unsqueeze(-1.0)` returned. `_place_axis` / `_reduce_axes`
now make their own axis constants (`ingest.literal`). And the compile
cache did not digest the helpers the lowering reaches: the digest now
covers the `symbolic_process_graph` and `external_functions` modules whole.

## 2026-10-03: concordance rows for every decision (user rule)

- Constant role: `constant_role` row (ingestion scope, constant node),
  DERIVED from the constant's `ingestion_value` cell, every consumer's cell
  and, per structural use, the consuming op's `tensor_operation_parameter`
  cell (NOVEL(`declared_signature`) row per declared positional parameter:
  name, annotation, integer). `_post_constant_roles` posts and the dtype is
  read back from the row (the attribute `constant_role_row` cites it).
- Operand edges: sympy `make_node`, the envelope removal, and
  `ingest_sympy_process_model` / `symbolically_reduce_process_graph` node
  builds now write operands only through `_set_operands`
  (`INGEST_EDGE` / `REMOVE_NODE` / `REPLACE_INPUTS`). Whole set: 215
  operand edges, 238 `identity_transition` rows in the ingestion scope
  (appends plus the envelope's retires).
- Transforms: `symbolic_transform` row (ingestion scope, result node) for
  each Derivative of an external resolved to its declared derivative(s)
  (DERIVED from the authored Derivative's subexpression cell, every
  `external_derivative` declaration the chain read, the result cell) and
  for the Integral's quadrature (DERIVED from the authored Integral's cell
  and the result cell). Whole set: 6 + 1 rows.
- Pages (appended to `concordance_declarations.py`): `tensor_operation_
  parameter`, `constant_role`, `symbolic_transform`; transform
  `declared_signature`; earlier `external_function`,
  `external_function_name`, `external_derivative`, `external_callsite`.

Gate:
- `audit_identity_concordance.py`: findings 0,1,0,1,5,0,0 and unsourced
  facts/identities IDENTICAL to a clean 408155a7 worktree with every file
  of this lane overlaid (4083/2831/3800/4546/4431/197/2952). The main
  tree shows different unsourced counts (view 4022, energy 3862, ...):
  those come from other lanes' uncommitted work, not from these files.
- Whole-set compile book: `ingestion_value` unsourced 13, unchanged (the
  envelope plus 12 rows of other graphs built during the compile).
- Tests: orbital 12/12, structural constants 2/2; symbolic process graph /
  equation compiler failure sets identical to baseline (3, 1).
  One `equation_of_motion_rhs` run hit `LLVM ERROR: out of memory` while
  other lanes were building; it passes alone.

## Open

- `get_tensor` lowering is by the catalogue alias (`get_tensor -> tensor`),
  not a backend table entry; the C/LLVM backends see the canonical
  `tensor` Const path. If a `_TENSOR` entry is wanted instead, say so.
- The externals' signature modules (NaN-bodied leaves) mint ~8 ids each
  outside a Novel post (`unsourced-identity` in the lowered module's
  report: 96 for the whole set) -- the AST lane's generic minting, not
  ingestion; not addressed.
- The probe's F host answers quadrature nodes only because the commanded
  force is constant over the run; a varying force needs the craft's
  recorded F history at those nodes (refused today).
- Lowered SSA call instructions do not carry the external-call identity as
  a row; the book has it at the symbolic level (callsite rows).
