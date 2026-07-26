# ProcessGraph compiler spine

The compiler path is:

```text
Python AST
  -> ProcessGraph.build_from_ast(semantic=True)
  -> optional Turing-provenance BitOps primitive expansion
  -> metadata-rich SSA
  -> C PrimitiveProgram or fused GLSL program
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

## Backend boundary

`lower_ssa_to_primitive_program` targets the equal-shape program shared by the
one-call C executor and fused GLSL backend. It supports canonical elementwise
unary/binary operations, scalar operands, `nand`, and `select`.

Structural primitives such as `concat`, `slice`, shifts, and shape-changing
`mu` cannot fit that packet. They return structured `LoweringIssue` records and
no executable program. A future region/view packet should represent their
shapes and storage ranges explicitly.

## Nodus

`process_graph_to_nodus_graph_ir` emits operation-tool nodes, directional
ports, named roles, connections, tensor metadata, and BitBit accounting.
Nodus validates canonical names against its generated operation catalog while
retaining inputs, constants, returns, and other structural nodes explicitly.

## Focused verification

```powershell
python -m pytest tests/test_ast_process_graph.py tests/test_bitops_process_graph.py tests/test_ssa_primitive_lowering.py tests/test_nodus_graph_ir.py -q
```
