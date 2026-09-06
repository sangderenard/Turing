# Adding an operator to the Turing tensor/compiler pipeline

This is the checklist derived from actually adding the bitwise shift operators
(`<<`/`>>` → `shl`/`shr`) end to end — from a Python source expression on an
`AbstractTensor` all the way to executed WebAssembly. Bitwise `&`/`|`/`^`/`~`
(`bitand`/`bitor`/`bitxor`/`invert`) were already partly wired, so comparing
"what `shl` was missing" against "what `bitand` already had" is what produced
this list. An operator that is missing from *any one* of these tables usually
fails **silently** — it is dropped from a numeric region, or reduced to a bare
Handler name nothing recognizes — rather than raising, so all of them matter.

The tables are grouped by the pipeline stage that consumes them, in the order a
source expression flows through them.

---

## Stage 0 — First ask whether you need a new primitive at all

**Read [`../DTYPE_AND_SPECTRAL_DOMAIN_MANIFESTO.md`](../DTYPE_AND_SPECTRAL_DOMAIN_MANIFESTO.md)
before starting.** The ten-stage checklist below is the cost of adding a genuine
new *primitive* — an operation no existing operator can express. It is ten
hand-maintained tables per operator per backend, and, as noted above, omitting any
one of them fails silently. That cost is worth paying exactly when it is
unavoidable, and is pure waste otherwise.

**The standard: define once in terms of base operators; let backends override for
speed; never hand-write the same operation into six backend tables.**

If the operation is derivable from operators that already exist, define it once in
`abstraction.py` or `abstraction_methods/` and stop. Every backend then gets it
immediately — including backends that have no implementation of it at all — and it
inherits the autograd tape, the SSA vocabulary, and every lowering path for free.
Backends may still override the hook with a native call purely as an optimisation.

Two precedents in-tree:

- **`AbstractTensor.searchsorted`** (`abstraction.py:1919-1950`) — built from
  broadcast compares and a sum, with no backend hook at all.
- **`unravel_index_`** (`abstraction.py:2795-2808`) — was a `NotImplementedError`
  base with **seven** duplicated backend implementations, one of which silently
  returned the coordinates of element 0 alone and discarded the rest of an array.
  Replaced by a single base implementation using only `%` and `//`; every backend
  worked immediately, including one that had never implemented it. The seven
  native versions remain valid as overrides, but none is required.

A useful test: *could this be written using operators the tree already has?* If
yes, the ten stages below are the wrong tool. Reach for them when the answer is
genuinely no — a new hardware-level primitive, a new instruction, a new
irreducible elementwise kernel.

### Silent failure is the through-line

The warning at the top of this document — that a missing table entry fails silently
rather than raising — is the same defect the manifesto catalogues elsewhere:
`to_dtype_` silently defaulting unknown dtypes to float32, `AT.real` silently
recording no autograd node, `eigh` silently returning the bare diagonal for any
constant-diagonal matrix, complex `sum` succeeding while complex `add` raises.

**Whatever you add, make its absence loud.** A missing lowering should raise; a
degraded lowering should announce itself with the operator, the target, and the
reason. A slow or unsupported answer is acceptable; an unannounced one is not.

---

## Stage 1 — Source AST → ProcessGraph node

**`src/compiler/ast_process_graph.py` — `_BINARY` / `_UNARY` / `_COMPARE`.**
Maps a Python `ast` operator class to a canonical op-name string
(`ast.LShift → "shl"`). Also `_CALLS` if the operator is also reachable as a
named function; it pulls names straight from `fused_ir.ELEMENTWISE_*`.

## Stage 2 — Canonical elementwise vocabulary

**`src/common/tensors/fused_ir.py` — `ELEMENTWISE_UNARY` / `ELEMENTWISE_BINARY`
/ `ELEMENTWISE_ALIASES`.** The backend-neutral set of "this is one fused
elementwise tensor op". `canonical_elementwise_op(name)` consults it; the JIT
tape→FusedProgram path and several profiles derive their fusible-op set from it.

## Stage 3 — Eager execution on `AbstractTensor`

The operator must actually run when a live tensor is observed (the AOT observer
executes the program; the JIT path traces it).

- **`src/common/tensors/abstraction.py`** — the dunder method
  (`__lshift__`/`__rshift__`), the `from .abstraction_methods.elementwise
  import … as elementwise_*` aliases, and the `AbstractTensor.__op__ =
  elementwise_op` bindings at the bottom of the file. If the operator instead
  routes through `_apply_operator` (like `+`/`~`), add its name to the
  `arithmetic_ops` set there **and** implement it in every backend's
  `_apply_operator__` (numpy/torch/jax/pure/c). The shift operators use the
  *valuewise* path instead (below), which is backend-agnostic and needs no
  per-backend code.
- **`src/common/tensors/abstraction_methods/elementwise.py`** — the operator
  method (`def __lshift__`), and a `_scalar_kernel` entry giving the per-element
  Python semantics. Comparisons, logicals, and bitwise ops live here rather
  than on the backend arithmetic path because they are integer-structural /
  non-differentiable. **Dtype dispatch**: `&`/`|`/`^`/`~` are *logical* on
  boolean operands and *bitwise* on integers (see `_is_bool_like` / `_both_bool`
  — the boolean readings coincide with bitwise for `&`/`|`/`^`, but `~` differs,
  so it dispatches to `logical_not` vs `invert`).

> **Type resolution, not forced casts.** The reduce helpers (`_at_shl`) use
> AbstractTensor's own `<<`, so the result dtype is *resolved* from the
> operands, never forced. Note the current front end materialises any integer
> result as **int64** (a valuewise `int(a) << int(b)` is an unbounded Python
> int), which is why an int32 *working type* end to end is not yet achievable
> from these source expressions.

## Stage 4 — ProcessGraph reduction (AST BinOp → canonical op)

**`src/common/tensors/topological_reducer.py` — `_BITOPS_TO_EXECUTABLE`.** The
reducer resolves a `BinOp(op=LShift)` via `_qualified_handler` →
`ssa_registry` → a `Handler`, then maps the Handler's value to an executable op
name through this table. Without `"Shl": "shl"` / `"Shr": "shr"` here the op is
left as the bare Handler name `"Shl"`, which no downstream table recognizes.
(This is why `bitxor` worked and `shl` did not: `Xor→bitxor` was mapped, the
shifts were not.)

## Stage 5 — SSA registry

**`src/transmogrifier/ssa_registry.py`** — the `Handler` enum member (e.g.
`Shl`/`Shr` already existed), the **alias tuples** mapping every spelling
(`'shl'`, `'binop:lshift'`, `'__lshift__'`, `'binaryop:<<'`, …) to that Handler,
and `BITOPS_EXPANDABLE_OPS` if the op participates in bit-level expansion. The
canonical lowercase op name (`'shl'`) must appear in the alias tuple or
`_qualified_handler` cannot find it.

## Stage 6 — Operator signatures / backend adapter

**`src/transmogrifier/operator_defs.py`** — all of:
- a `_at_<op>` callable (`_at_shl = _abstract_tensor_reduce(lambda l, r: l << r)`),
- entries in `abstract_tensor_funcs` for **both** spellings the graph can carry:
  the SymPy-style name (`"LShift"`) and the canonical lowercase (`"shl"`),
- membership in `_abstract_tensor_binary_names` (or the unary set) — this drives
  `abstract_tensor_sigs`, which classifies the op as `sig_binary_elementwise`
  (input/output arity, concurrency). A missing signature means the op is not
  treated as a fusable tensor primitive.

## Stage 7 — Operator catalogue

**`src/common/tensors/operator_catalog.py` — `COMPOSITE_MATH_OPERATORS`** (or
the appropriate category set). Classifies the op as tensor math for the
dispatch/reduction machinery.

## Stage 8 — AOT dispatch-region membership (the easy one to miss)

**`src/compiler/glsl_deployment_strategy.py` — `_is_dispatch_metadata_node`.**
This predicate decides whether a node is *coordinator/structural metadata*
(excluded from numeric GPU/CPU regions) or real numeric work. It had an
explicit `coordinator_bit_shift` clause that classified **every** `<<`/`>>` as
coordinator-side, so tensor shifts never reached a numeric region. A genuinely
scalar shift (address/index arithmetic) is already caught generically by
`static_scalar_expression`, exactly as a scalar `&` is — there is no
`coordinator_bitand`. The fix was to delete the shift special-case so shifts are
treated like the other bitwise ops. **When adding an operator, make sure no
metadata/coordinator predicate here special-cases its AST node out of numeric
dispatch.**

## Stage 9 — Numeric backends (emission)

- **`src/compiler/fused_program_wasm_backend.py`** — `_INTEGER_BINARY_INSTRUCTION`
  and/or `_BINARY_INSTRUCTION`/`_UNARY_INSTRUCTION` (the WASM opcode per op), any
  composed lowering (integer `invert` is `x XOR -1`; integer `min`/`max` are
  compare+`select`), and `plan_static_data`'s constant packing must use the
  working type's representation (integer bytes for i32/i64, not float64).
- **`src/compiler/fused_program_python_backend.py`** — `_ELEMENTWISE_TEMPLATES`
  (the NumPy/PyTorch/AbstractTensor source spelling, e.g. `"{0} << {1}"`) or
  `_NAMED_FUNCTIONS`. This backend is also the fidelity **oracle**.
- Other backends (C, Fortran/SSA, GLSL/WebGL) have their own op → instruction
  tables (`c_backend.py`'s `binary_codes`, `ssa_registry`, the GLSL tensor
  backends). A pure-integer op needs a real native lowering there before its
  `[c]`/`[fortran]`/`[glsl]` fidelity variants can pass; the WASM and
  valuewise/NumPy paths are sufficient for the AOT WebAssembly target.

## Stage 10 — Test harness (to prove it)

**`src/compiler/wasm_fidelity.py`** — `_NUMPY_DTYPES` / `_INTEGER_VALUE_TYPES`
and the `_RUN_SCRIPT` typed-array view. Integer working types need an integer
oracle dtype, an `Int32Array`/`BigInt64Array` memory view, and integer feed
data (i64 rides over the JS/JSON boundary as decimal strings because BigInt has
no JSON encoding).

---

## Chasing down existing users (the two-context caveat)

`&`/`|`/`^`/`~` mean **logical** on boolean masks and **bitwise** on integers.
Existing code that wrote `~mask` on a boolean expecting logical negation must be
made explicit — the C backend stores masks as `0.0`/`1.0` doubles and loses the
bool dtype, so dtype dispatch cannot rescue it there. Convert those sites to
`mask.logical_not()` (done for `compression/bitstream.py`,
`compression/entropy_scan.py`, `compression/entropy_symbols.py`). `&`/`|`/`^` on
0/1 masks are safe (bitwise coincides with logical), only `~` genuinely differs.

## Deferred follow-ups

- **Autograd reverse/backward functions.** Routing bitwise ops through
  `_pre_autograd` during capture may require backward definitions. They are
  non-differentiable integer ops, so this is a deferrable follow-up, not a
  blocker for the forward/compile path.
- **Deterministic public-output projection.** `project_public_numerical_program`
  can non-deterministically fall back to exposing every live value (including a
  redundant baked `tensor_from_list`) as an output for some single-return
  valuewise programs. This is pre-existing (a lone `x <= y` triggers it too) and
  orthogonal to operator registration.
- **int32 working type end to end.** Blocked only by the front end materialising
  integer results as int64; the i32 emission path itself is exercised and exact.
