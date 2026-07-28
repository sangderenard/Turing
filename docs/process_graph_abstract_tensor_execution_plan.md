# ProcessGraph to AbstractTensor execution audit and plan

Date: 2026-07-28

## Intended execution contract

The intended path is:

```text
Python AST / SymPy / another graph source
    -> ProcessGraph
    -> canonical operator and structural reduction
    -> topological and control-aware planning
    -> numerical ProcessGraph regions
    -> GraphDeepCompiler-generated Python callable
    -> public AbstractTensor operations
    -> selected backend
    -> optional forward capture
    -> CapturedFusedProgram
    -> backend compilation and repeated execution
```

This is deliberately similar to executing a generated sequence of Torch
instructions. The generated callable should contain ordinary tensor
operations. AbstractTensor supplies those operations and retains the backend;
NumPy, Torch/CUDA, C, GLSL, or another backend performs the primitive work.

The compiler does not need a second implementation of JPEG, Mandelbrot, DCT,
or tensor indexing. It needs to preserve and route the existing AbstractTensor
program correctly.

## Confirmed existing pieces

### Canonical AbstractTensor operation table

`src/transmogrifier/operator_defs.py` contains `abstract_tensor_funcs` and
`abstract_tensor_sigs`. The table covers SymPy spellings, Python/AST spellings,
canonical AbstractTensor operations, aliases, creation operations, accessors,
shape operations, reductions, comparisons, linear algebra, and explicitly
composed operations.

Most handlers call the public AbstractTensor surface rather than a concrete
backend. This is the correct backend-neutral boundary.

### Deep graph compilation

`GraphDeepCompiler` emits a pure Python function from a levelled
ProcessGraph. Simple arithmetic is emitted as Python operators; other graph
operators call handlers from the supplied table. With AbstractTensor operands,
both forms naturally invoke AbstractTensor overloads.

`EphemeralProcessGraphCallable` supplies `abstract_tensor_funcs` to this
compiler. It is intentionally a transient projection of the ProcessGraph, not
a replacement for the graph.

### Backend selection and capture

The GLSL deployment coordinator enters
`AbstractTensor.use_backend("glsl", device)`, tensorizes array-shaped public
inputs, executes numerical regions, and can record each region using forward
capture. Captured numerical regions can then become `CapturedFusedProgram`
instances and GLSL programs.

This is already the correct broad strategy: generate ordinary AbstractTensor
instructions, select a backend, execute once to capture when appropriate, and
reuse the compiled backend program.

### Structural coordinator

The deployment shell separately evaluates containers, branches, loops,
function references, external calls, and routing between numerical regions.
That separation is necessary until a backend compiler is deliberately given
native control flow or a KPN representation.

## Current gaps

### 1. Value domains are implicit

ProcessGraph nodes currently do not carry a reliable, planner-visible
classification such as:

- static value;
- Python scalar;
- tensor;
- structural container or reference;
- control effect;
- external effect.

Operator-table membership is not enough to infer this. `Not` can mean tensor
logical negation or scalar control negation. `shape`, indexing, comparisons,
and reductions also produce values whose domains depend on their operands and
arguments.

The observed Boolean `logical_not` failure is a direct consequence.

### 2. Signatures describe arity better than value kind

`abstract_tensor_sigs` is broad and useful for operator availability, but many
entries use conservative generic signatures. They do not yet express which
inputs and outputs are tensors, scalars, shapes, indices, tuples, or effects.
Consequently the planner cannot use the signatures alone to partition a mixed
program correctly.

### 3. Resolved function graphs and retained Python callables overlap

Call nodes may have both a resolved `callee_ref` and a
`static_python_reference`. The current GLSL coordinator can prefer the
retained Python callable in that situation. Calling a Python function with
AbstractTensor operands still executes correct backend-neutral tensor math,
but its Python control flow is then outside the function's ProcessGraph shell.

This explains why some sophisticated algorithms already run successfully
while their full control topology is not yet compiled.

The contract needs to be explicit:

- A resolved project function should execute through its ProcessGraph shell.
- A Python callable should remain only for a declared external/runtime
  boundary or an explicit fallback mode.
- The two cases must be visible in profiling and graph metadata.

### 4. Static context is not yet a first-class compiled input

Module constants, function defaults, classes, and imported references are
retained in several places, but child-function compilation does not yet have
one uniform rule for resolving them. These values should be classified and
either folded, installed in a shell's static table, represented as an explicit
input, or declared external.

They should not be replaced by literals in a demo call.

### 5. `device` is not a deep-compiler semantic

`GraphDeepCompiler.build_function(device=...)` accepts a device, but the
generated function itself does not use that argument. Backend and device
selection currently come from the surrounding AbstractTensor context and
tensor inputs. That can be a valid design, but it should be documented as such
or the unused parameter should eventually be removed.

### 6. Batch capability is not proven by operator availability

Most tensor primitives naturally accept outer batch dimensions. That does not
prove that an entire algorithm preserves an added frame dimension:

- shape manipulations may assume a fixed rank;
- indexing may consume the outer dimension;
- entropy-event structures may flatten frames together;
- variable-length results need offsets or masks;
- AVI emission is ordered and stateful even when frame computation is batched.

Batching must be validated by stage and represented in graph metadata rather
than asserted globally.

## Implementation plan

### Phase 0 — Preserve and characterize the committed baseline

1. Keep the restored committed demo and compiler code unchanged.
2. Add small ProcessGraph execution tests independent of Mandelbrot.
3. Run each numerical test through NumPy, Torch, C, and GLSL where available.
4. Record whether each call used a resolved graph shell, a retained Python
   callable, or an external boundary.

Success means the execution route is observable before it is changed.

### Phase 1 — Add value-domain analysis

Introduce a backend-independent analysis result for every retained node:

```text
STATIC
SCALAR
TENSOR
STRUCTURAL
CONTROL
EXTERNAL
UNKNOWN
```

This is analysis metadata, not a new operator vocabulary. Infer domains from
known inputs, constants, canonical operator semantics, and function
signatures. Preserve `UNKNOWN` rather than guessing.

Initial tests must cover:

- scalar `not` versus tensor `logical_not`;
- scalar and tensor comparisons;
- shape and rank access;
- tensor indexing with scalar, slice, tuple, and tensor indices;
- scalar loop bounds around tensor bodies;
- a conditional selecting tensor computations.

### Phase 2 — Partition from domains and effects

Use the domain analysis when forming deployment regions:

- tensor-producing computation enters AbstractTensor numerical regions;
- scalar and structural computation remains in the coordinator unless folded
  or explicitly compiled into backend control flow;
- external effects are dispatch boundaries;
- control nodes own their dependent regions without making dormant branches
  execute eagerly.

No special case may mention Mandelbrot, JPEG, AVI, or a demo function.

### Phase 3 — Make resolved calls use resolved graphs

Define one callee-resolution rule and test it:

1. `callee_ref` with a graph: call the child shell.
2. declared external reference: call the external table.
3. explicit fallback mode: call retained Python and report that fallback.
4. unresolved call: fail with its graph location and reference information.

Install static values for child shells through one general static-context
table. Test ordinary defaults, module constants, imports, nested calls,
kwargs, and recursion references.

### Phase 4 — Validate AbstractTensor parity before fusion

Execute representative ProcessGraphs without capture on NumPy and Torch, then
C and GLSL. Compare values, shapes, dtypes, and control results.

The generated callable should look like ordinary AbstractTensor programming.
If it does not, fix operator reduction or graph construction rather than the
demo.

### Phase 5 — Capture numerical regions

After uncaptured execution is correct:

1. Capture each topologically closed numerical region.
2. Compile its forward tape for C and GLSL.
3. Route outputs to dependent regions through preallocated shell storage.
4. Verify repeated execution changes only declared dynamic inputs.
5. Compare captured and uncaptured results.

Control flow remains represented by ProcessGraph. A tape only represents the
executed numerical region and must not claim to preserve dormant branches.

### Phase 6 — Prove batching by subsystem

Add an outer batch dimension to:

1. Mandelbrot solve and color planes.
2. Block extraction, DCT, quantization, and zigzag ordering.
3. Coefficient-event generation with per-frame offsets or masks.
4. Entropy coding in bounded frame groups.

Measure and document the first stage that cannot remain a dense tensor batch.
Ordered AVI packet emission drains completed frames sequentially and should
not prevent earlier numerical stages from running in batches.

### Phase 7 — Build the recording shell

Only after the preceding phases:

- one AST-ingested root owns solve-through-AVI semantics;
- input controls and audio arrive through bounded FIFO lanes;
- numerical frame batches run ahead within a bounded window;
- encoded frame and audio packets drain in deterministic order;
- one writer lifetime owns OpenDML segmentation and final indexes;
- shell profiling reports control, upload, compute, entropy, packet output,
  and finalization separately.

The live window may present returned buffers. It may not encode, collect,
interleave, or finalize recording data.

### Phase 8 — Optimize generally

Optimization then belongs in reusable components:

- ProcessGraph topology reduction;
- domain-aware dispatch planning;
- backend fusion-pattern registries;
- reusable tensor workspaces;
- stateful GLSL buffers;
- batched launch planning;
- asynchronous FIFO scheduling;
- backend-native reductions and matrix operations where profitable.

The Mandelbrot program becomes a demanding integration test, not the home of
those mechanisms.

## Completion criteria

The work is ready to return to the large demo when:

- scalar and tensor operations are partitioned correctly;
- resolved project calls execute through child graph shells;
- static defaults and module constants resolve without call-site literals;
- the same small graph passes on NumPy, Torch, C, and GLSL;
- captured and uncaptured executions agree;
- profiling identifies every Python fallback and external boundary;
- a batched numerical test demonstrates an actual outer frame dimension;
- no implementation or planner branch contains a demo-specific name.
