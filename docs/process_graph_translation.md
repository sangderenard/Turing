# ProcessGraph compiler spine

The compiler path is:

```text
Python AST
  -> ProcessGraph.build_from_ast(semantic=True)
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

## Natural tensor source

The semantic AST importer recognizes canonical tensor operation spellings in
both method and function form. For example,
`tanh((x + y).sin())` enters the graph as `add -> sin -> tanh`, rather than as
opaque Python calls. The recognized names come from the established fused
operation vocabulary plus the structural canonical operations; this mapping
only resolves syntax and never supplies alternate numerical implementations.

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
