# Existing GLSL AST reader correction

The duplicate direct reader and separate CFG emission are rejected architecture.
No renamed reader or copied parser is used in this correction.

Changes in glsl_source_ingestion.py are shared lexer punctuation/increments only.
Changes in glsl_ast_ingestion.py extend its existing expression reader with
indexed/field references and its statement handling with ordinary Python AST
If/While, counter-for loops, break/continue, assignments and fixed local arrays.
Counter-for continue executes its increment before the next condition, without
rewriting continues of nested loops. Short circuit remains Python AST BoolOp.
Unsupported expressions remain visible shortfalls. No direct SSA writer added.

Fifteen existing/new AST, direct GLSL source and WebGL source-roundtrip tests
pass in 6.24 seconds, including sanctioned compiler lowering and nested-loop
continue behavior. The tiny while example in build/check_glsl_ast_public_entry.py compiles
via lower_ast_source_to_ssa with the existing ExtractionContract and int32
scalar formal. It produces main__main and planned regions, with normal log
artifacts/identity_logs/main.20261109T202440.ok.log.xz. Initial attempt without
an extraction contract correctly failed; no contract gate was bypassed.

This is NOT full traversal readiness. Source helper functions with out/inout,
vector and record semantics, typed local/storage memory and source-to-AST
provenance still need completion/review before the original law can pass this
path. Ordinary Python AST execution is not proof of f32 GPU arithmetic parity.
No shader rebuilt or activated. Release worker owns removing products that
came from the rejected reader, both full-traversal Wasm and WGSL.


Follow-up correction: reviewer found flattened block shadowing and untyped
zero-filled local arrays. The reader now rejects duplicate/shadowed declarations
and refuses local arrays with a typed-storage/initialization shortfall. No array
support is claimed. Scalar source helpers now become ordinary AST function
and call nodes; out/inout parameters fail visibly rather than becoming value
parameters. Shared entry selection prioritizes the declared void main over
other void helpers. Nineteen AST/source/WebGL regressions pass in 5.59 seconds.

The next blocker is preservation of exact GLSL local widths: the sanctioned
source compiler annotation propagation at fortran_c_shell.py currently maps
Python int/float locals to int64/float64. Merely parsing GLSL into Python is
not proof of int32/float32 execution. Original GLSL-to-AST source correspondence
also must reach the normal evidence path; an ast.unparse test only establishes
normal compilation of the generated Python AST. No full traversal readiness.

## Typed annotations and original-source entry

The public lower_ast_source_to_ssa entry now accepts source_language='glsl'
through the existing glsl_ast_ingestion reader. It keeps the generated AST
rather than discarding its source correspondence with unparse/reparse.
Existing SourceDefinition and SOURCE_SPAN records retain exact document/token
identities. Owner stamping now accepts a per-expression/statement
_turing_source_definition, a book-independent declaration, alongside its
existing generated-function source_cells support. Exception conversion hooks
are deliberately not reused: they have different activity semantics.

The shared dtype_layout authority now resolves explicit scalar width
annotations/forward strings while preserving Python int64/float64 defaults.
The normal parameter/local annotation lane consumes that resolver. A literal
fallback in fortran_c_shell contradicted its own 'stated dtype wins' comment;
it now uses setdefault rather than overwriting an explicit annotation.
GLSL literals are represented as typed immutable declarations in the ordinary
AST lane, so the original float32 literal does not silently become float64.

Direct original-GLSL compilation proves both operands and the Mul result are
float32; retained identity-book checks prove actual GLSLToken source rows and
outgoing edges exist. A changed token location is rejected. These two tests
pass in 5.07 seconds. This is a bounded source/type/evidence pass, not a shader
execution or full traversal pass.

Typed integer loops remain blocked: a source int32 value meets an int64
physical ABI in structural concordance. The checker remains unchanged. An
attempt to fix ABI arms from identity_table histories was removed after the
existing loop composer documentation showed those histories are scope-free;
lexical LOOP_CARRIED_BINDING/source declaration evidence must be followed.
Out/inout, records/vectors and typed local storage remain incomplete. No live
artifact was generated or published.

A broader dtype regression subset recorded 44 passes and five failures:
public C ABI rejects int128/uint128, and three synthetic scalar-merge tests
lack SSA identity roots. Those tests do not exercise the changed annotation
entrypoints; no changes were made to weaken their refusals.

### Exact local scalar receipt follow-up

The existing reducer now retains fixed-width local annotation references at
`bind_target`, on the actual bound source value rather than reconstructed name
histories. Existing linked ABI settlement reads that exact annotation, checks
its dtype, preserves its provenance, and rejects conflicting physical contracts.
The initial int32 local mismatch is resolved. The bounded loop still fails at a
generated Phi: the loop update values remain int64. Canonical update receipt
transport remains under investigation; concordance is unchanged and nothing is
published. Current regressions: 23 passed, one known failing loop deselected;
the focused loop run failed in 10.97 seconds. This is partial work, not WebGPU
or full traversal readiness.

### Folded loop seed width

The For and While control builders reconstructed folded seed widths from Python
literal values, discarding the now-correct physical ABI. Both now select an
existing scalar VALUE_TYPE_ABI arm when present and include that receipt in
CONTROL_CONST provenance. Unannotated seeds retain their existing fallback.
This removes the earlier generated-Phi physical-concordance refusal. The
focused test now stops later at physical-call input adaptation: round 2 reports
one change while repeating round 1's state. That guard remains intact.
The 23-test reader regression subset still passes (5.69 seconds); the complete
loop test is still failing and no traversal product is activated. A candidate
change to integer evidence closure was tested, proved irrelevant, and reverted.

### Call-adapter lineage investigation

Dynamic tracing established that exact-feed settlement substituted a preheader
seed for a converted loop update: the conversion lost the producer marker and
feed_ids still named the seed. The adapter now recognizes only a validated
CALL_INPUT_CONVERSION receipt for the exact caller/callee/formal/result and
follows its source producer. Existing loop_scope declarations distinguish the
exact outer/carried generations when the actual is a header Phi. No arbitrary
actual-over-feed preference or cycle-guard relaxation was added.
The focused loop still fails: exact region feed receipts disagree int32/int64.
The remaining widening writer is under independent investigation. Adapter test
subset: 15 passed, 4 failed (two numeric-conversion attribute expectations and
two physical span transport disagreements); baseline comparison is not yet
established, so this is not reported as a clean regression result.

### Accepted original-reader integer loop

The exact local physical width now reaches PlanClosure.value_shapes through
existing _dtype_producer_cells, allowing the established authored scalar
operator inference to retain it. The remaining widening was a control
aggregate-result Load: it shared identity with the correct producer but was
classified as provisional. Such a Load now retains its declared width only
when its posted physical dtype already proves that same width.

Typed source literals carry both the exact annotation and same-value scalar
ABI as direct materialization causes. The existing audit accepts this domain
only with matching scalar width, annotation provenance and integer range;
unannotated Python literals keep their original int64 rule. No audit is
skipped. Raw GLSL loop plus reader and mutation suites: 30 passed in 15.05s.
Separate unannotated Python-int regression: 1 passed in 4.99s.

Wasm emission remains blocked: Add reports a minted i64 operand where i32 is
required, followed by its dependent unavailable result. Thus source/SSA and
concordance are verified, but no executable/runtime numerical claim is made.
The new shader/full traversal remains inactive.


### Executable original-reader int32 loop

The subsequent Wasm blocker was two unconditional Python integer-literal
carrier resets in the existing hierarchical planner, including one after its
inference loop. Both now consult the exact same-value scalar VALUE_TYPE_ABI
and original ANNOTATION_DECLARATION, require agreement with the planned source
dtype, and check representable range. Without that authored carrier, Python's
existing literal domain remains unchanged. No implicit backend narrowing was
added. Earlier ineffective semantic-class override experiments were removed.

The raw GLSL while loop now emits a complete Wasm module and executes n=0,1,2,
5,12,31 with the expected sums. Post-emission full concordance passes. Focused
source, audit mutation, Python-default, reader and roundtrip suite: 32 passed
in 17.51 seconds. NumPy warns about scalar argument conversion in the existing
Wasm execution wrapper; this is not a numerical failure. Full BVH, out/inout,
and typed local storage remain incomplete; no shader is published or enabled.

The adapter subset's four failures were also reproduced with the committed
HEAD adapter substituted in the same working tree (15 pass / 4 fail in both).
This isolates them from this adapter edit, not from all concurrent tree edits;
it is not a claim of a pristine repository baseline.


### Normal WebGPU path, isolated validation pending

The registered backend identity webgpu_static_aggregate_index_elision removes
only a duplicate Const index whose every use is a proven static aggregate GEP
slot. It checks exact SSA definition/use identity, call publication mapping,
index equality, carrier types and protected outputs. It leaves numerical i64
uses unchanged and reports before/after hashes through the existing backend
identity receipt. It does not retype a value or introduce an emitter lane.

The existing WGSL Call/GEP spelling now treats a one-result callee as the scalar
it actually emits: slot zero aliases that scalar; multiple results retain
array indexing. Invalid slot indexes are explicit shortfalls.

Normal emit_module now reports a complete original-reader int32 loop shader.
Fifteen focused metadata-elision/projection tests pass, including negative i64
use, mismatched index, unknown owner and duplicate-definition cases. Combined
new typed-source/backend suite: 24 passed. Browser compile/dispatch validation
is pending on an isolated 32-lane fixture, not the Studio site. Four broader
backend identity fixture failures also reproduce with committed HEAD identity
code in the current tree (missing SSA shape cells before identity application).

Out/inout research found ParameterContract COPY and OUT/INOUT declarations but
no executable scalar COPY lowering. Existing lowering handles ALIAS/INOUT
record effects. Scalar GLSL input helpers compile through the public Wasm path;
copy-out, early-return undefined output state, and typed storage remain open.


### Actual WebGPU execution verified

Real Chrome validation first rejected the normal loop emitter's `break if`
statement in the loop body. That WGSL form belongs in a continuing block;
the existing emitter now spells the same loop-header exit as
`if (!condition) { break; }`. Its regression no longer expects invalid syntax.

The regenerated original-reader shader compiled and executed successfully on
Chrome WebGPU with an NVIDIA Ampere adapter. A single 32-lane dispatch returned
all expected integer sums for distinct n=0..31 (ending at 465), with no compiler
messages and no validation error. Shader SHA256:
29149D722C7F9BB72246E18D9695126B93D6273DFE1F3CEA47C8FEA72926518E.
The isolated test server/tab were closed; no Studio product was changed.

Focused final suite: 48 passed in 19.66 seconds. The module after registered backend
identity preparation also passes full concordance (separate regression).
The wider WebGPU suite has six failures, reproduced with both committed HEAD
backend modules substituted in the same current tree (9 passed / 6 failed);
this is an isolated-module baseline, not a pristine-tree claim.
