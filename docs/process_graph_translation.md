# ProcessGraph compiler spine

The compiler path is:

```text
Python AST
  -> semantic ProcessGraph / ProcessOp
  -> optional BitOps primitive expansion
  -> metadata-rich SSA
  -> C PrimitiveProgram or fused GLSL program
  -> Nodus AbstractTensor GraphIR tools
```

## Stable semantic payload

`ProcessOp` separates compilation data from visualization labels and live
Python objects. It records the canonical operation name, ordered input/output
roles, scalar attributes and constants, tensor dtype/shape/device, control
metadata, and source span.

`BitQuantaSpec` preserves the accounting contract of `BitBitBuffer`: mask
quanta, `bitsforbits`, PID-domain labels, and source-node provenance. It
describes a live buffer without copying either storage plane or its UUID
tables.

## BitOps

`expand_bitops_process_graph` invokes the existing `Turing` algebra with a
symbolic ProcessGraph carrier. The NAND, mux, ripple-add, and structural
recipes therefore remain in `turing_machine/turing.py`; the compiler does not
maintain a second arithmetic implementation.

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

