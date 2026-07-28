# Deferred AST → SymPy → ProcessGraph round trip

Turing has enough existing machinery to make this route viable:

1. Translate Python AST mathematical regions into `sympy.codegen.ast`
   (`FunctionDefinition`, `CodeBlock`, `Assignment`, `FunctionCall`, and
   `Return`).
2. Use `CodeBlock.topological_sort()`, `CodeBlock.cse()`, and
   `ProcessGraph.full_recombinatorics()` for symbolic organization and
   reduction.
3. Reuse `ProcessGraph.to_sympy()`'s expression-registry plus
   `ExpressionTensor` representation.
4. Rebuild the result through `ProcessGraph.build_from_expression()`.

This is deliberately deferred. A complete implementation must explain tensor
shape, tensor-valued assignment, AbstractTensor calls, stateful calls, repeated
assignment, and control-flow boundaries. Pure assignment blocks can use SymPy
directly; FIFO writes, entropy state, file output, and similar effects must
remain explicit ProcessGraph operations.

The immediate compiler route should instead reuse the existing shared
AST/SymPy/BitOps vocabulary for `Name`/`Load`, `Assign`/`Store`, `Call`, and
`Return`, plus BitOps' existing live-subgraph splicing pattern.
