# Memory manager, deep copy, and nodus interop — design handoff

Written 2026-08-18. This closes a session that ended in a design decision, not
a finished feature. Read the "Where this is going" section first; the code
state at the bottom exists to be built on or discarded, not defended.

## The problem that started it

`copy.deepcopy` could not be lowered. Chasing that exposed the real gap, which
is much bigger than deep copy and is the actual subject of this document:

**This compiler has no runtime memory allocation at all.**

Evidence, not inference:

* `fortran_c_shell.py` `allocation_lines` — the generated C shell calls
  `calloc(count, sizeof(c_type))` once per ABI parameter at startup, sizes
  fixed at compile time, and passes those `slots[]` pointers into the Fortran
  entry as `intent(inout)` arguments. Fortran never allocates; it only writes
  into arrays it was handed.
* `ir_sequence_tables.py` `_unsupported_destination` — a `DYNAMIC` capacity
  policy is rejected outright with `DYNAMIC_GROWTH_UNAVAILABLE` and the
  message *"dynamic sequence growth requires an explicit allocation and
  arena-replacement ABI"*. `precompile_to_ssa.py` only ever builds
  `SSASequenceCapacityPolicy.FIXED`.

So the compiler already knows exactly what it is missing, and says so in its
own error text. Deep copy is simply the first feature that cannot be faked
without it: a copy needs storage that did not exist at compile time.

## Where this is going

### Nodus is the operating system

The decision is **not** to design a new memory manager, and **not** to
redesign anything in nodus. Nodus is nearly finished and is the authority.
The compiler adopts it.

The design precedent being followed: unify all backends by requiring a memory
manager, the same way thread management is already unified. OOP is deliberately
*not* the general style here — but a memory manager, a locker, and a thread
manager are precisely the cases where OOP is the right tool. Those can be
written as small classes and lowered to SSA. Nothing else should be.

Keep it small, simple, fast.

### Storage model

Everything lives, behind the scenes, in contiguous tensors.

Nodus already has the dtype vocabulary this requires
(`nodus/include/common/tensors/abstraction/tensor_types.h`):

* `Bytes`, `Bytes2`, `Bytes4`, `Bytes8` — genuinely untyped storage dtypes,
  deliberately distinct from `U8/U16/U32/U64`. Raw byte pools.
* `Ptr` — a real pointer/handle dtype, sized `sizeof(void*)`.
* `TensorLayout::Opaque` alongside `Dense`/`Strided`.
* `TensorSliceMeta` with `base_shape`/`start`/`step` and an `index_handle`,
  i.e. slicing one pool into per-instance views is already a modelled concept.

Per-dtype pools are the intended shape: typed pools for ordinary scalar
fields, `Bytes*` pools where storage is genuinely opaque, `Ptr` pools for
handle tables. The manager only needs a pool's dtype and extent.

### Pointers vs offsets — the rule

**Offsets inside a pool. Pointers only at the host boundary.**

If an intra-pool reference is an offset, growth can move the pool's base
without invalidating anything — which is what makes "arena replacement" cheap
instead of a rewrite-the-world problem. A `Ptr` stored *inside* a pool would
be invalidated by exactly the realloc the manager exists to perform.

`Ptr` is therefore an interop currency for host-owned memory, not intra-pool
storage. Root/stack storage may stay offset-based even if the heap itself
hands out pointers at its edge.

This matches what already exists: `opaque_ref` in the Fortran backend is
already a bit pattern carried in a slot (`transfer`-preserved), not a machine
address.

### C shell vs C++ shell

There is a concrete trigger, not a taste call. The C shell allocates once with
`calloc` at startup and never grows. A heap needs `realloc`/`free` during
execution. That single requirement is what forces graduation to the C++ shell,
which is also what moves this into nodus interop territory. Everything else
about the C shell continues to work and does not need replacing.

### The interop path — and why C++ ingestion is not the blocker

The strategic point: **we do not need full C++ ingestion**, because anything
can be expressed *in nodus*, and nodus is C++. C++ becomes one of the OOP
languages we neuter into reliability and take charge of — deciding who handles
how things execute — rather than a language we must parse in general.

What already exists and makes this viable:

* `src/compiler/nodus_canvas_kpn.py` — a **bidirectional, self-verifying**
  bridge between repository-SSA regions and nodus `CANVAS V4` saves. A
  module's `ROWS` program *is* a stack serialization of a straight-line SSA
  region. The writer proves its own serialization by running the reader's
  stack simulation over the rows it emitted and comparing against the source;
  anything it cannot spell is a named shortfall, never a wrong row program.
  Records it does not model (UUIDs, meta-groups, ropes, LEDs, overlays) are
  preserved verbatim, so load → translate → save does not destroy canvas state.
* Nodus scheduling is real: rows execute against a module's value stack every
  scheduler tick (`thread_manager.cpp`), `Input` rows feed from edge FIFOs,
  `EDGE` records are KPN FIFO channels, `NODE` records carry
  `GraphRuntime::NodeContract`.
* Tool vocabulary is shared: `abstract_tensor.<name>` ids formed by
  `register_abstract_tensor_tool_ir` from `canonical_ops.json`, the same
  append-only catalog the KernelIR membrane uses.

Known constraint to design around: the bridge handles **straight-line** SSA
regions. Control flow is where it currently stops.

### The C++ ingestion plan, for when it is needed

Not abandoned — staged.

1. Use `src/compiler/cpp_shell_desugar.py` when it suffices. It is explicitly
   *not* a C++ parser: it rewrites a narrow C++-like shell into C text for the
   trusted `pycparser` route (`machine_code_lifting.py`), handling
   `class → struct` + free functions `Foo__method(struct Foo* self, ...)`,
   constructors → `Foo__new`, and single inheritance as struct embedding. It
   hard-rejects templates, operator overloading, `virtual`, exceptions, and
   namespaces via `CppShellUnsupported` — never silently misdesugaring.
2. **Rejected material is kicked to a new, dedicated shim stage**: C++ source
   in, fully desugared C++ out, with every piece of sugar worked out into
   something interpretable like AST. This includes following library sources
   recursively to try to capture the whole graph and turn it into our own
   static code.
3. That shim will likely need source-specific file overrides, special cases,
   and schema records. The machinery for this already exists and is
   declarative: `boundary_namespaces/<language>/**/*.node.json` with actions
   `schema`, `spoof`, `exclude` (see `boundary_namespaces/README.md`).
   Resolution is language-first then lexical OOP scope; missing directories
   are skipped, not errors; only declarative JSON is ever loaded, never code.

This is an extensive job and should be started fresh, deliberately.

## Why the ingestion side is already multi-language

Worth knowing before touching it: the graph walker is *already* language
-parametric, in production, not hypothetically. `source_language` is used with
`python`, `javascript`, and `sympy`. It gates exactly one line in
`graph_express2.py`:

```python
if special is None and self.source_language.casefold() == "python":
    special = interpret_python_special_case(current)
if special is None:
    special = interpret_special_case(current)
```

Everything else dispatches on `type(node).__name__` through the shared
structural interpreter (`node_special_cases.py`), which is why the Python-AST
and SymPy front ends converge on one walker. A new language needs node objects
whose type names map into `role_schemas` — not a new traversal.

## Code state at the end of this session

### Deleted deliberately

`src/compiler/ssa_deepcopy.py` is **gone**, along with its Fortran support
(`turing_heap`, `Alloca`/`StridedMemoryCopy`/`HeapLoad`/`HeapStore` lowering,
module-scope heap declarations) and its `ssa_features.py` registration.

It was a complete, compiling, runtime-verified generic deep-copy engine —
iterative worklist, seen-table, correct for shared references and cycles. It
was deleted because it invented a **private word heap** with its own memory
model that had no relationship to this compiler's actual object system. It did
not use the repo's own object representation, so it could never fit. The
lesson is recorded here so it is not rebuilt: *do not invent a parallel memory
model; adopt nodus's.*

### Kept

* `src/transmogrifier/graph/python_identity_programs.py` — `copy.deepcopy`
  resolves to the pre-existing `Handler.Deepcopy` vocabulary member. Verified
  through the real `ProcessGraph.build_from_ast` entrypoint.
* `src/compiler/fortran_c_shell.py` — two changes, both **incomplete work in
  progress**, kept only because they encode real findings:
  * `irreducible_classes = set(constructor_symbol_by_class)` disables the
    `self_is_field_storage` collapse for any class with a constructor. That
    collapse makes a constructor argument's id *be* the resulting object's
    storage identity, which is only sound when the argument is always a fresh
    value. Verified to flip the flag correctly for nested classes.
  * A `deepcopy` branch that re-enters class construction using the source
    record's own current field values instead of fresh AST call arguments.

### Known broken

The deep copy is **not** correct. A nested `Outer`/`Inner` test compiles and
emits a genuine second `__init__` call, but both constructions still write
into the same `t3` arena — the copy is not independent. `self_is_field_storage`
was ruled out as the cause by direct test. The likely mechanism (unverified) is
that pre-binding field parameters into `remap` before the `clone_value` /
`"record_instance"` minting loop makes those ids take the "already bound, skip"
path, so fresh arena storage is never minted.

Do not build on this. It is a probe, not a foundation. The real fix is the
memory manager, not more patching of the constructor path.

### Test state

`tests/test_fortran_c_shell.py::test_whole_object_sequence_field_is_real_record_storage_not_scalar_slot`
fails. **Verified pre-existing** — it fails identically with these changes
stashed.

## Suggested first move for the next session

Do not start by writing a memory manager. Start by establishing what the
existing `nodus_canvas_kpn.py` bridge can already carry — specifically whether
nodus's own memory management can be reached from an SSA region through it,
and where the straight-line-region constraint actually bites. That answer
determines whether the manager is adopted wholesale or needs a lowering path
built for it, and it is cheap to find out.
