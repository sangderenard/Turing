# SSA structure vocabulary, and the materializers

*Updated 2026-08-18, later the same day: materializer 1 is built, and the
drifted counts this document told you not to touch are now understood and
corrected. See those two sections below.*

Written 2026-08-18, continuing `MEMORY_MANAGER_AND_NODUS_INTEROP_HANDOFF_2026-08-18.md`.
Read that one first for the memory-manager decision (nodus is the authority;
`mem_backend.h` already *is* the unified allocator). This document covers the
shared vocabulary work that followed, and the pieces built on top of it.

## The goal this serves

turing and nodus already agreed on arithmetic. They did not agree on
**program structure** -- control flow, storage, and the definitions of classes
and functions. Structure travelled as out-of-band tables, so a program could be
transported but its shape could not be *stated* in the shared vocabulary.

That is what breaks bit-exact round-tripping. A method that crosses a boundary
as a name arrives without its body; a constant authored inside that body -- a
custom trig epsilon is the motivating case -- does not survive at all. A
definition has to be expressible as an operator for its contents to travel.

Nothing here is for people writing tensor code. You define a class with
`class`. These exist so a complete program, structure included, crosses
suites intact.

## What is done

### Shared catalog (nodus)

`ops/canonical_ops.json` is the **single source of truth**, not any Python
file. `ops/generate_canonical_ops.py` emits both `include/canonical_ops.h` and
`ops/canonical_ops_generated.py`; both are marked never-hand-edit. Author as
data, generate both languages, and they cannot drift.

Three new op classes were added alongside the computation classes:

    control  br condbr indirectbr ret trap phi select deploy join
    memory   alloca store getelementptr getattr setattr indexed indexedstore
             fill strided_store_fill strided_memory_copy deepcopy
    value    const static_ref class_define field_define method_define
             function_define

These are not computations: they write no equally-shaped slot, so the schema
forbids them a `ct_op`, and they compose from nothing, so they name no Tier-1
family. Only `select` is Tier-0 lowerable (`kernel_op: SELECT`), because both
its operands are already values -- unlike `condbr`, which is an edge.

`returns: "void"` was added for operations publishing no SSA result (a branch,
a store, a return). The emitted `OpDesc` gained a matching `returns_void`
field; the pre-existing `returns_bool` alone would have recorded them as
returning a value.

IDs are append-only positions. The new ops took 66..91 and nothing moved.

### SSA vocabulary (turing)

`Handler` gained `SetAttr`, `ClassDefine`, `FieldDefine`, `MethodDefine`,
`FunctionDefine` (now 117 members). `REPOSITORY_SSA_OPERATORS` derives from
`Handler`, so it stayed in sync by itself.

`SetAttr` was a genuine hole, not merely unnamed: `GetAttr` had no counterpart,
so a field *write* could only be spelled as a slot store, which drops the
field's name and therefore its meaning.

### AbstractTensor mirrors (turing)

`src/common/tensors/abstraction_methods/ssa_structure.py`, attached as
`AbstractTensor.ssa`:

    define_class  define_field  define_method  define_function
    class_table   accessor      handler

Nothing is re-declared. The constructors return the **real** repository SSA
objects (`SSAClassDefinition`, `SSAClassField`, `SSAClassMethod`,
`SSAClassTable`), verified by type identity -- so the mirrors cannot drift from
what they mirror.

Attached one attribute deeper than the tensor methods deliberately: this is
program structure, not a tensor operation, and it must not surface to someone
browsing tensor methods.

The `transmogrifier` import is **deferred to call time** on purpose: that
package pull brings in the graph/simulator stack, and `abstraction` must stay
importable without paying for it. Keep it that way. It also caught its own
bug -- a wrong relative depth surfaced at call time rather than breaking import
for every consumer. (`ssa_structure.py` sits one level deeper than
`topological_reducer.py`, so it needs four dots, not three.)

### The sigmoid mystery, solved

Both checkers reported `sigmoid` as a registration hole. The cause was a
**parser bug in `ops/verify_canonical_ops.py`**, not a catalog error.

It split the CTensorOp enum body on commas *before* stripping comments. A block
comment containing a comma is torn in half by that split, so `/* ... */` can no
longer match either piece, and the member following the comment is silently
dropped. The comment explaining why `CT_OP_SIGMOID` is appended last (rather
than grouped with the transcendentals, since `is_unary_op` classifies by enum
*range*) contains exactly such a comma.

So `CT_OP_SIGMOID` was invisible to the checker all along, and the op was
unregisterable in both directions: adding it was rejected, omitting it was
rejected. Comments are now stripped from the whole body before splitting;
the header parses at 51 members and sigmoid reads its true ordinal 49 -- the
next free `ct_value`, preserving the gapless 0..N-1 block.

`verify_canonical_ops.py` now reports **no disagreements at all**:
`OK: 93 canonical ops (50 vendored from CTensorOp, 58 lowerable) agree with turing`.

For the record on where sigmoid lives: `abstract_nn/activations.py` has
`Sigmoid(Activation)` alongside `ReLU`/`Tanh`/`GELU`, with its own
`bw_sigmoid`. That is the homed one. `AbstractTensor.sigmoid` is a *primitive*
hoisted up so the tape had a lowerable atom -- its own docstring records that
it used to be seven ops here and compiling failed with "sigmoid has no
captured basic-operator lowering". `_sigmoid_stable` is now a one-line
delegate to it.

## Commits

    nodus   4ad29f6  control/memory/value classes, 21 structural ops
    nodus   5dec226  class/function/accessor definitions
    nodus   10053ef  register sigmoid + fix the parser that hid it
    turing  a290aed  SetAttr + 4 definition opcodes
    turing  9792198  AbstractTensor.ssa mirrors
    turing  bc84ca9  sigmoid across the backend lanes, and the counts explained
    turing  5a3dc2b  SSA definitions -> Python AST (materializer 1)

## What is NOT done -- the next work

### 1. SSA objects -> Python AST -- DONE, with one seam left open

`src/compiler/ssa_python_materializer.py` (commit `5a3dc2b`). Definitions in,
real `ast.ClassDef`/`ast.FunctionDef` out, reached from the mirrors as
`AbstractTensor.ssa.to_python`.

It is the Python *destination* in `oop_schema`'s sense rather than a new path:
an `SSAClassDefinition` goes through `ClassSchema.from_ssa_class_definition`,
so the SSA form's lossiness stays stated in one place. Its scalar spellings
are audited against `ssa_llvm_backend`'s likeness tables at import, the same
contract `ssa_reference_evaluator` holds -- an opcode spelled there but absent
here fails the import.

Calibrated against the symbolic fluid step, the case the reference evaluator
is calibrated on: the 291-instruction body materializes, compiles, executes,
and reproduces the authored SymPy equations to 2.1e-17, and agrees with the
reference evaluator on a second sample.

**The seam left open is structured control flow.** A multi-block function
raises and names its blocks. Turning a CFG back into `if`/`while` is
decompilation with its own correctness argument, and approximating it on the
way past would emit Python that reads right and means something else. That is
the next piece of this direction, and it is deliberately separate.

Also declared rather than guessed: `ULt`/`ULe`/`SExt`/`ZExt`/`FpToUi` have no
exact Python spelling without a stated bit width, so they raise.

### 2. C++ dynamic class/function building (nodus side)

The counterpart: build classes and functions dynamically on the nodus side from
the same definitions.

### 3. Then the C++ shell

Per the plan: static-compile nodus' thread manager into every C++ shell, or let
shells use an abstract-tensor DLL. Expect roughly a week of iteration to make
that reliable and to confirm turing compiles it correctly. Do not start this
before 1 and 2 exist.

## The drifted counts -- understood, then corrected (commit `bc84ca9`)

`tests/test_ssa_operator_contract.py` had three failures, left alone here
because the cause of the drift was not understood. It is now. Two causes,
both real vocabulary:

* `cast_like` was added to `TYPE_AND_DEVICE_OPERATORS` by the same commit
  (`4abc962`) that wrote `== 223`, so the assertion was born stale by one;
* `sigmoid` joined `ELEMENTWISE_UNARY` in `c967b1d` afterwards.

Chasing the second surfaced a live breakage rather than a numbering question.
`validate_c_opcode_alignment()` was raising: `ctensor_ops.h` carries
`CT_OP_SIGMOID` at ordinal 49, but the handwritten LLVM lane's mirror tuple
stopped at 48, so `opcode_index[unary_codes["sigmoid"]]` was a `KeyError` on
any tape that reached it. Registering the op in the C header and the nodus
catalog had left three of the four lanes behind.

Sigmoid is now lowered everywhere it is advertised -- LLVM (the same
branch-free stable form as `ctensor_ops.c`, verified by JIT-compiling the
module and running it), Fortran (through `tanh`, which is elemental and
saturates rather than overflowing), and WASM (one entry in
`wasm_math_tables`, domain +/-16, 32 KB, measured 7.3e-7 against a 1e-6
epsilon). `shared_implemented_tensor_operations()` now contains it.

With the causes understood the counts moved to meet them: 223 -> 225 and
56 -> 57 here, and 66 -> 93 in `test_backend_capability_audit.py`, that last
being the 26 structural ops the shared catalog gained plus sigmoid. Each
carries a comment naming what it counts, so the next drift is readable rather
than mysterious.

## Known-failing, pre-existing, deliberately untouched

Verified pre-existing by restoring the tree to HEAD and running the same
files -- not by stashing, which in this nested repo discards the whole WIP:

* `test_c_backend_llvm_ssa.py` -- the translation table's
  `cast_double_to_bool_values`/`cast_double_to_double_values` drift, and four
  tape-lowering cases;
* `test_llvm_jit_backend.py` -- the torture baselines;
* `test_fortran_fidelity.py` -- a `capture_feed_origins` `KeyError`.

The first of those is the same shape as the counts above: `c967b1d`'s cast
work added symbols the table's expectation never learned about. Worth doing
the same reading before anyone edits the set.

## Standing lesson from the prior arc

Recorded because it cost a whole session: the deleted `ssa_deepcopy.py` was a
working, runtime-verified deep-copy engine that had to be thrown away because
it invented a *private word heap* unrelated to this compiler's object system.
Do not invent a parallel memory model. Adopt nodus'
(`mem_backend.h`: real `alloc`/`resize`/`free` across CPU/CUDA/Vulkan/Torch/
filesystem, with `GP_MEM_CAP_BYREF_SAFE` telling you per-backend whether raw
pointers may be stored at all).
