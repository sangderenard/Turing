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
