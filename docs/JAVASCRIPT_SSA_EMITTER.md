# Repository-SSA JavaScript emitter

Runtime services are now selected through the named, content-addressed utility
inventory described in `JAVASCRIPT_RUNTIME_UTILITIES.md`. Emitted functions
also publish performance/inline labels for AbstractUI visualization and later
backend-specific optimization policy.

`src.compiler.ssa_javascript_backend.emit_ssa_module_to_javascript` prints a
dependency-free ECMAScript module from repository SSA. JavaScript is treated
as the ordinary browser control language here. WGSL shaders and WebAssembly
products are deliberately outside this first boundary.

## Classes are the first-class surface

When `IRModule.class_table` retains `SSAClassDefinition` records, the emitter
prints real JavaScript classes before exposing a free-function entry point.
It preserves the facts the SSA class representation actually owns:

- the complete class identity;
- each field's name and physical slot;
- each method's function-table reference and linked SSA function name.

Every emitted constructor carries `ssaIdentity` and immutable `fieldLayout`
metadata. `SSA_CLASSES` maps the full SSA identity to its constructor, so a
dotted identity never has to be guessed from a JavaScript identifier. A field
addressed by its numeric SSA slot is redirected to the corresponding named
JavaScript property. Constructor and method wrappers invoke the linked SSA
function bodies. Missing bodies, duplicate layout members, and sanitized-name
collisions are explicit emission shortfalls.

The emitted public surface preserves order of appearance: class definitions,
field declarations, methods, and logical parameters remain in authored order.
Field slots and physical SSA formal positions are retained as separate ABI
coordinates; adapters use those coordinates without sorting the source-facing
surface. This rule belongs to the language-neutral class emission plan so the
same contract can govern future C++ and Java printers.

The minimal SSA table does not retain inheritance, field types, defaults,
static-method status, or complete source signatures. The emitter does not
invent them. A later entry that receives a richer `ClassSchema` can add those
facts while proving its projection agrees with the SSA table.

## Generic control template

ECMAScript has no `goto`, while repository SSA permits arbitrary control-flow
graphs. Each emitted function therefore uses a block dispatcher:

```javascript
let block = "entry";
let predecessor = null;
for (;;) {
  switch (block) {
    // one case per SSA basic block
  }
}
```

Branches update `predecessor` and `block`. Phi inputs are first evaluated into
temporaries and then committed together, preserving parallel-copy semantics
on loop back-edges. Planned regions whose return record is declared only by a
call site's `output_ids` are recovered and checked across all call sites.

The supported first slice includes calls, constants, branches, Phi nodes,
returns, one-dimensional addresses, loads/stores, attributes, casts, selects,
and every operation in the 57-member portable elementwise catalogue. The
JavaScript numeric table is keyed by canonical tensor names and resolves both
direct repository opcodes and `Call[tensor_operation=...]`; it therefore covers
all 46 elementwise operations currently accepted by the Python fidelity
backend without duplicating its private lowering table. Repository `And`,
`Or`, `Xor`, and `Not` retain their integer-bitwise meanings, distinct from
`LAnd`, `LOr`, and `LNot`.

Unknown instructions and 64-bit bitwise uses produce shortfalls instead of
disappearing from the emitted program. JavaScript `Number` remains the scalar
working representation; an exact `BigInt` policy is still required before i64
bitwise operations can be advertised.

## Public ABI

A module with a selected root function exports that function and accepts
either:

- an array in `PROGRAM_ABI.bufferOrder`; or
- an object keyed by SSA value id.

Pointer-like formals keep their typed array or object. Scalar formals may be a
number or a one-element typed array. Results are returned as an array in SSA
`Ret` order. A class-only module has no fabricated program entry; it exports
`SSA_CLASSES`, `PROGRAM_ABI`, and its class constructors.

## Current boundary

This is the JavaScript host emitter, not yet the heterogeneous container. The
next layer should attach deployment records for WGSL and Wasm/assembly regions
to this module while keeping JavaScript classes and the ordinary control graph
as the owner of program identity and resident state.
