# ProcessGraph compiler spine

The compiler path is:

```text
Python AST
  -> ProcessGraph.build_from_ast()
       (operators and canonical tensor method/function calls)
  -> optional Turing-provenance BitOps primitive expansion
       (Python-list or AbstractTensor carrier)
  -> metadata-rich SSA
  -> FusedProgram
       -> private C slot plan or fused GLSL shader
  -> Nodus AbstractTensor GraphIR tools
```

## ProcessGraph node contract

Compiler metadata lives on ordinary ProcessGraph nodes. The canonical
operation name, ordered input/output roles, scalar attributes and constants,
tensor dtype/shape/device, control metadata, source span, and BitBit accounting
are graph attributes. There is no parallel `ProcessOp` object to keep in sync.

`BitBitBuffer.quanta_metadata()` preserves mask quanta, `bitsforbits`,
PID-domain labels, and source-node provenance without copying either storage
plane or its UUID tables.

## BitOps

`expand_bitops_process_graph` invokes `BitOpsTranslator.apply_bits`, whose
hooks are instrumented by `Turing`'s existing `ProvenanceGraph`. Recorded
primitive subgraphs are imported through the ordinary
`provenance_to_process_graph` bridge and spliced into the source graph. NAND,
mux, ripple-add, multiplication, and structural recipes therefore remain in
`Turing`/`BitOpsTranslator`; the graph pass contains no arithmetic recipes.

Currently expanded operations:

- `bitand`, `bitor`, `bitxor`, `invert`;
- `add`, `sub`, `mul`.

Unexpanded operations remain present with `bitops_status=unexpanded`.

`AbstractTensorBitOpsTranslator` supplies the existing Turing calculus with an
AbstractTensor carrier. Its eight hooks are compositions of ordinary tensor
arithmetic, concatenation, slicing, shape inspection, and same-backend
construction. The derived BitOps recipes are unchanged, and their primitive
provenance can be spliced into the source ProcessGraph through
`abstract_tensor_bitops_factory(like)`.

## Backend boundary

`lower_ssa_to_fused_program` targets the established backend-neutral
`FusedProgram`. The C backend compiles that IR to a private native slot plan;
GLSL lowers the same IR directly to shader locals. There is no public,
competing `PrimitiveProgram` or `GlslProgram` schema.
The elementwise region supports the complete canonical set of 56 lowerable
unary, binary, comparison, logical, bitwise, shift, and cast primitives,
including scalar operands and broadcasting. Shape-changing operations remain
explicit FusedProgram kernel kinds rather than being misrepresented as
elementwise instructions. GLSL compiles captured creation/fill, reshape,
stack, concatenate, expand, permute, repeat, slice/index-select, matmul,
reduction, and cumulative-sum regions through the backend's established
native shader emitters.

The live [AbstractTensor Mandelbrot video demo](mandelbrot_glsl_video_demo.md)
now exposes that frontier through a stateful deployment shell. The complete
recording function enters ProcessGraph, the scheduler produces dispatch
subgraphs, and the shell owns their function-table registry, named feeds and
outputs, FIFO interface, and profiling state. The general coordinator now
resolves structural host values and explicit external-call boundaries while
executing every scheduled numerical region through the AbstractTensor operator
table on GLSL. Semantic AST nodes are excluded rather than emitted as no-op
dispatches.

The first scheduled invocation is the ephemeral specialization boundary: each
tensor-producing callable records a forward tape, lowers to the shared
FusedProgram IR, and compiles to exactly one GLSL shader. In the current
Mandelbrot-to-JPEG graph this yields thirteen shaders and two scalar/shape
coordinator regions. Imported Python functions, entropy coding, container
framing, and I/O remain explicit host boundaries; they are not disguised as
empty shaders or numerical fallbacks.

The deployment shell tree also shares a root-owned hierarchical profiler.
Region dispatches, graph-backed child-shell calls, and explicit external
boundaries retain their shell path while contributing CPU, GPU, call-count,
and launch-count measurements to one report. This makes dispatch partitioning
and optimization evidence-driven: a caller can rank actual expensive graph
sections instead of treating the complete translated program as one opaque
duration.

## Nodus

`process_graph_to_nodus_graph_ir` emits operation-tool nodes, directional
ports, named roles, connections, tensor metadata, and BitBit accounting.
Nodus validates canonical names against its generated operation catalog while
retaining inputs, constants, returns, and other structural nodes explicitly.

The execution benchmark uses the narrower sibling boundary: one captured
equal-shape `FusedProgram` is serialized as canonical names and value ids,
then lowered by Nodus to its TensorMath-backed calculator. Nodus does not
restate the benchmark expression. This transport does not replace the
ProcessGraph → GraphIR path; it is the prepared numeric-region handoff after
the graph compiler has selected such a region.

## Focused verification

```powershell
python -m pytest tests/test_ast_process_graph.py tests/test_bitops_process_graph.py tests/test_ssa_primitive_lowering.py tests/test_nodus_graph_ir.py -q
```
