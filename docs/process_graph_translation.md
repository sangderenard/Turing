# ProcessGraph compiler spine

The compiler path is:

```text
Python AST
  -> ProcessGraph.build_from_ast(
       semantic=True, profile="program" | "tensor_control")
       (one expanded entrypoint, or its compiler projection)
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

## Natural tensor source

The semantic AST importer is not intended to compile all of Python. It uses
Python syntax to recover process flow around a smaller executable language:
AbstractTensor operations. The compiler-facing `tensor_control` profile keeps
tensor-valued dataflow, the loops/branches/contexts that govern it, required
shape and scalar dependencies, and explicit host materialization boundaries.
Imports, formatting, UI code, unused helpers, and validation-only branches do
not enter backend partitioning.

The `program` profile keeps every node belonging to the transitive entrypoint:
tensor work, Python control, validation, UI, containers, byte materialization,
file I/O, and finalization. The optional `complete` profile additionally keeps
unreachable definitions from every supplied source file for coverage audits
and source archaeology. Neither profile is sent wholesale to GLSL; backend
partitioning selects legal numerical regions from it.

For the recording demo, the source-level dependency walk first constructs one
AST module containing exactly one top-level function,
`mandelbrot_recording_program`. Its body contains the original transitive
helper/class definitions followed by the original `animate_glsl` body.
ProcessGraph ingests that one function AST; it does not receive the source-file
bundle and does not discover the program from an execution tape.

- the complete single-function audit graph has 7,739 nodes;
- the recording program projection has 6,161 nodes across 84 reached
  definitions;
- all 6,161 recording nodes are forward-reachable from the one function root;
- the recording tensor/control projection has 4,003 nodes;
- the narrower encoder-only tensor/control projection has 1,899 nodes.

The importer recognizes canonical tensor operation spellings in
both method and function form. For example,
`tanh((x + y).sin())` enters the graph as `add -> sin -> tanh`, rather than as
opaque Python calls. The recognized names come from the established fused
operation vocabulary plus the structural canonical operations; this mapping
only resolves syntax and never supplies alternate numerical implementations.

Multi-file ingestion registers definitions and class methods before visiting
bodies. Calls then link to their original function/class regions, and an
entrypoint identifies one master program without copying those functions into
a generated source file. Constructed types survive branch and loop merges, so
methods on an optional writer still resolve after the writer is conditionally
created. A compiler call with a literal `entrypoint=` also has a `compiles`
edge to that source definition; numerical source is therefore not hidden
behind a Python compiler wrapper.

Function definitions own every node in their source region through explicit
`contains` edges. The one program root also owns required module/environment
dependencies. These structural edges make the complete program
forward-traversable but are excluded by one shared edge-role contract from
SSA operands and backend fusion inputs.
In the complete audit graph, loops, indexing, slices, contexts, generators,
containers, comprehensions, exceptions, attributes, mutation, functions, and
classes remain explicit ProcessGraph nodes. The generic structural fallback
reads the established `operator_defs.role_schemas`; it is not a second AST
schema. The tensor/control projection then removes nodes that do not
participate in the selected entrypoint's tensor process.

The reduction boundaries remain unchanged:

- canonical tensor arithmetic uses the established AbstractTensor/FusedProgram
  vocabulary and SSA correlation;
- operations advertise whether BitOps can implement them, but are selected as
  BitOps candidates only after dtype inference proves an integer/bit domain;
  expansion continues to obtain implementations exclusively from
  `BitOpsTranslator` and Turing provenance;
- symbolic SymPy expressions continue through
  `ProcessGraph.build_from_expression` and the SymPy/SSA registry.

The AST importer contains no numerical implementation for any of those three
paths.

## Backend boundary

`lower_ssa_to_fused_program` targets the established backend-neutral
`FusedProgram`. The C backend compiles that IR to a private native slot plan;
GLSL lowers the same IR directly to shader locals. There is no public,
competing `PrimitiveProgram` or `GlslProgram` schema.
The equal-shape region supports canonical elementwise unary/binary operations,
scalar operands, `nand`, and `select`.

Structural primitives such as `concat`, `slice`, shifts, and shape-changing
`mu` cannot fit that packet. They return structured `LoweringIssue` records and
no executable program. A future region/view packet should represent their
shapes and storage ranges explicitly.

The current complete slice is substantial but deliberately bounded: the
canonical cross-language catalog contains 66 operations, GLSL implements 56
canonical primitives, and the equal-shape whole-program fusion region accepts
40. Existing standalone GLSL matmul, reduction, and layout kernels should
become neighboring regions in a ProcessGraph partitioner rather than being
restated inside the elementwise emitter.

The [AbstractTensor Mandelbrot video demo](mandelbrot_glsl_video_demo.md) no
longer uses execution-tape capture. Its source extractor places the complete
start-to-finish recording program inside one function AST and that single
function becomes one ProcessGraph:

```text
frame/control loop
  -> compiled AbstractTensor Mandelbrot source
  -> count/Y/Cb/Cr
  -> JPEG DCT, quantization, events, Huffman scan, packed octets
  -> tensor octets to bytes
  -> AVI video/audio chunks
  -> segment and superindex finalization
  -> header patching and close
```

The display numerical region has an executable structured GLSL lowering:
resolved function regions are inlined, the canonical Python control loop
becomes one GLSL loop without source unrolling, scalar controls are packed into
one resident feed, and count/Y/Cb/Cr are produced by one multi-output dispatch.
The newly complete recording graph is an ingestion/completeness result, not a
claim that host file operations have become shader instructions. Its next
compiler stage is to partition this one program into legal GLSL numerical
regions and explicit host-I/O regions.

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
