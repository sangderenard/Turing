# Continuation: orbital externals (items 3 and 4) -- summary

Condensed 2026-10-03; full working notes are in git history of this file.
Lane committed as c03ae4e6 (externals as runtime slots) and a7f57c1e (item 4).

## Status

The whole orbital transfer set compiles as written and matches the reference
in both C and LLVM. Errors: equation_of_motion_lhs 0, total_energy_expression
2.2e-16, force_cost_integral 1.4e-16, orbital_transfer_raw 1.8e-15 (also
initial/terminal_condition_lhs 0, equation_of_motion_rhs 1.8e-15).
`tests/test_orbital_transfer_compile.py`: 12 passed / 0 xfailed.
`WORK_ITEMS` and `LAW_BLOCKERS` are empty; item 2 (Greek names) needed no change.

## The external ABI as built (decision: extend the llvm_piece seam)

Owner: `src/compiler/external_functions.py`.
- Declaration: `declared_external_functions` reads sympy `AppliedUndef` classes
  (name, arity); `compile_sympy_equations` records `metadata["external_functions"]`.
- Leaf: `ExternalFunction` / `ExternalSlotABI`, piece-shaped
  `void (void **buffers, int32_t *extents)`, tagged `llvm_piece` with
  `binding: runtime-slot`. `_decide_llvm_piece` gives `loader: runtime-slot`,
  `callbacks: reject`, `native_abi: llvm-buffer` (passes the full-native gate).
  Buffers = args then result; extents = (result numel,). Elementwise signature
  only: every argument at the result shape.
- Signature: specialized at the callsite shape (`external_callsite_shapes` runs the law's AbstractTensor stage with recording stubs). With an `LLVMPiece` implementation the leaf is that piece's module; otherwise a declaration whose body is `a0 * nan`, never emitted under the tag, so an ignoring lane yields NaN.
- One slot per call shape: `externals_for_law` groups callsites by shape and
  gives each (external, shape) its own leaf and slot (`r1__s4`, `r1__s`; a
  single shape keeps the bare name). Callsites are respelled `name__callsite_k`
  on a copy (`_respelled`), never the cached compilation. The leaf record carries
  `external` (name the host binds by) and `leaf`.
- C lane (`ssa_c_backend`): `static turing_external_fn <entry>__external_slots[N]`,
  exported `<entry>__bind_external(int32 slot, void *fn)` and
  `<entry>__external_fault_take()`. The shim loads the slot; an empty slot records
  `slot+1` as the fault and returns.
- LLVM lane (`ssa_llvm_backend`): `@<entry>.external_slots` global, same binder and
  fault reader, per-slot `@<entry>.external_unbound.<k>` stubs; call site is
  `select(slot == null, unbound stub, slot)` then an indirect call (no block split).
- Loud errors: `CModuleExecution.run` / `LLVMExecution.run` call
  `check_external_faults` and raise on a fault or a host-callback exception.
  Emptying a slot gives "external 'f' (slot 0) was called with its slot unfilled".
- Host: `bind_external_slots(artifact, {name: impl})` refuses a missing external and
  fills every slot of an external with its one implementation. impl is a Python
  callable (wrapped once with CFUNCTYPE over the buffer ABI; an exception inside
  fills NaN and is re-raised after the run), an `LLVMPiece` (its own entry address;
  buffer order must match), or a raw native address.
  `piece_from_law(..., externals=)` binds the declared externals.
- Optional derivative: carried as `ExternalFunction.derivative`, on the leaf's
  `llvm_piece` record as `derivative`; derivative leaves link per shape.
- Dispatcher fix: a root that only calls a piece goes to `_emit_repository_call_module`.
- Probe: `tools/compiler_probes/probe_external_function.py` (~1 min): Python and LLVMPiece suppliers, C and LLVM, error 0.

## Item 4: derivatives and the Integral

Derivatives: `compile_sympy_equations(..., external_derivatives={r1: v1, v1: a1, ...})`
maps an undefined Function to the undefined Function the host declares as its
derivative. The ingestion Derivative branch (`symbolic_process_graph`) runs
`sympy.diff`, then `external_functions.resolve_external_derivatives`:
`Derivative(r1(s),(s,2))` -> `a1(s)`; chain form `Subs(Derivative(r1(x),x),x,g)`
-> `v1(g)`. A Derivative of an external with none declared raises
`UndeclaredExternalDerivative` naming it: no finite-difference fallback. A
Derivative of compiled content still goes to the existing graph reversal.
`metadata["external_derivatives"]` and the cache record carry the map.
Probe side: `probe_orbital_transfer.external_derivatives()` declares r_i -> v_i -> a_i;
`host_externals` supplies v_i / a_i as derivatives of the host's own r_i.

Integral lowering (audit gap 10), four defects, each fixed at its owner:
1. `get_tensor(<literal sequence>)` reached the planner as an opaque op:
   `OPERATOR_ALIASES["get_tensor"] = "tensor"` (`operator_catalog.py`).
2. `_tensor_descriptor_rule` read a reduction axis only from an `axis`/`dim`
   attribute; positional `x.sum(0)` settled rank 0. Now reads the `arg:0` literal.
3. Region `value_shapes` (`glsl_deployment_strategy` `_value_shape_dtype`)
   re-derived reduction/axis-insertion extents with its own rule; now takes the
   descriptor's settled shape for sum/prod/min/max/any/all/unsqueeze/squeeze.
4. A capture made by a shape-only op outside the region was fed its source's
   storage and restamped (5,1) with (5,): `PlanClosure.value_views` (new field,
   carried through the three rebuilds) records it; the planner's formal declares
   `ssa_storage_view`, which the enrichment rule already honours.
Reference side: an Integral SymPy cannot integrate is evaluated as its declared
Gauss-Legendre rule (`declared_quadrature`), so the check is of the lowering.

## constant_role rows: structural int vs numeric float

Rule (`symbolic_equation_compiler._structural_constants`): a constant keeps its
integer (int64, attribute `structural_constant`) only when EVERY consumer reads it
at a parameter its operation's declared AbstractTensor signature annotates as an
integer (`_structural_operand`, `_declares_integer`). Any value consumer keeps it
float64 (the Piecewise arm). Regression this fixed: aa5f1aac turned all sympy int
constants to floats, so the quadrature axis constants became `-1.0`/`0.0` and
`unsqueeze(-1.0)` raised TypeError. Declarations added:
`AbstractTensor.sum/prod/max` `dim: int | tuple[int, ...] | None`.
Book row `constant_role` (ingestion scope) is DERIVED from the constant's
`ingestion_value` cell, each consumer's cell and the consuming op's
`tensor_operation_parameter` cell (NOVEL `declared_signature` per parameter); the
dtype is read back from the row. Other pages: `external_function` (NOVEL),
`external_function_name`, `external_derivative`, `external_callsite`,
`symbolic_transform` (one per resolved external Derivative and for the quadrature),
all declared in `concordance_declarations.py`. Operand edges are written only via
`_set_operands`. Tests: `tests/test_symbolic_structural_constants.py` (2 passed).

## Craft-binding probe

`tools/compiler_probes/probe_orbital_craft_binding.py` (~3 min): the whole set
compiled once (C and LLVM), externals bound at load to a LIVE `OrbitalJumper`
(Earth + Moon, 1000 kg, force (0,0,2) N, 6 rounds to t=1457 s). Shows the set and
the craft are one system: equation-of-motion residual <= 1.1e-16 |a| (the set's
gravity is the craft's catalogue N4.1); r(0), r(L) exact; total energy vs numpy
2.6e-16; m * force_cost vs the craft's fuel_impulse 1.6e-15.

## Open items

- External identity rows crossing books: lowered SSA call instructions carry the
  external only via the leaf's `llvm_piece` record; the callsite rows are symbolic
  level. Being handled by another lane
  (`CONTINUATION_symbolic_cache_and_external_rows.md`; do not edit here).
- The Fortran lane does not read `llvm_piece`; it would emit the NaN declaration body.
- Zero-argument externals are refused (elementwise signature only).
- The probe's F host answers quadrature nodes only because the commanded force is
  constant; a varying force needs the craft's recorded F history there and is refused.
- Signature modules mint ~8 ids each outside a Novel post (96 `unsourced-identity` set-wide); `get_tensor` lowers by catalogue alias, not a backend table entry.

## Traps

- `test_public_abstract_tensor_api_is_explicitly_classified` fails before and after: unrelated.
- Compile cache digests `symbolic_process_graph` and `external_functions` whole plus `tensor_operation_signatures` (a stale entry once served `sum(0.0)`).
- `_place_axis` / `_reduce_axes` make their own axis constants (`ingest.literal`); a memoized shared `-1` mixed the role.
- Concordance audit baseline 0,1,0,1,5,0,0; main-tree unsourced counts differ from other lanes' uncommitted work.
- Probes take ~1-3 min; do not launch builds without the user's say.
