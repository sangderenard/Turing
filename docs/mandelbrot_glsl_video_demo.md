# AbstractTensor Mandelbrot video demo

The Mandelbrot demo's former compilation route is disabled:

```text
ordinary AbstractTensor quadratic-family program
  -X-> GradTape capture
```

That route captured only the executed elementwise solve/palette/YCbCr prefix,
then projected the resulting FusedProgram into ProcessGraph. It never imported
the complete encoder through AST and never optimized the complete program as a
ProcessGraph. Keeping it enabled made partial execution capture look like the
requested compiler path.

Calling any of the demo's `capture_*` entry points now fails explicitly. The
general GradTape and FusedProgram facilities remain available elsewhere; only
their misleading use as this demo's compiler has been disabled.

## Current deployment-shell frontier

The complete saved recording function—including DCT, quantization,
coefficient events, Huffman lookup, bit packing, marker stuffing, and AVI
interleaving—is imported through AST into ProcessGraph. The topological
reducer normalizes lexical values, constants, calls, indexing, and function
references. The GLSL deployment strategizer turns the reduced graph into a
stateful shell containing scheduled numerical subgraphs, function-table
shells, named input/output bindings, and bounded input/output FIFOs.

The shell, rather than a tape captured from one convenient prefix, is now the
demo's execution contract. Its currently verified lifecycle is:

```text
plan ProcessGraph dispatches
  -> prepare each numerical subgraph as an ephemeral AbstractTensor callable
  -> capture each tensor-producing callable as one CapturedFusedProgram
  -> compile each captured program to one GLSL compute shader
  -> execute the complete saved schedule with named feeds
  -> route imported functions through the explicit external-function table
  -> return counts, color planes, coefficients, and the framed JPEG
```

The complete recording graph currently contains fifteen planned numerical
regions. Thirteen produce tensors and compile to thirteen GLSL shaders; the
other two perform scalar/shape coordination and remain in the Python
coordinator. Each clean subgraph passes through the existing deep compiler
and calls the public AbstractTensor operator table under the GLSL backend.
Function-table shells share one registry, so graph-backed calls can recurse
through the scheduled shells without turning callee names into public inputs.
Imported compression helpers remain explicit Python-call boundaries; their
AbstractTensor operations still select GLSL, but their bodies are not yet
inlined ProcessGraphs.

Semantic-only AST nodes do not become empty shaders or FIFO movements.
Attributes, slices, imports, assignment/store wrappers, context managers,
control nodes, and external calls evaporate from the GLSL dispatch set. A
value they describe becomes a boundary feed only when a real downstream
numeric region consumes it.

The non-animated `--only-glsl` image path now executes that complete saved
recording schedule and writes the JPEG produced by the graph. A 16×16
cross-check produced exactly the same quantized coefficients and JFIF bytes as
the established AbstractTensor encoder, and the focused deployment/compression
suite passes.

This is deliberately a scheduled set of fused shaders, not one monolithic
video shader. The numerical regions now cover the canonical 56 primitive GLSL
operations plus native creation, reshape/layout, indexing, reduction,
cumulative-sum, and matrix kernels. Entropy coding, byte framing, file I/O,
and other host boundaries remain explicit coordinator work. The older live
animation loop has not yet been honestly migrated to the complete
recording-shell input/output contract.

## Hierarchical shell profiling

Every deployment shell in a function-table registry shares one profiler owned
by the root shell. With `--profile`, each numerical region and explicit
external boundary reports its full shell path, operator sequence, call count,
CPU wall time, GPU elapsed time, and physical GLSL dispatch count. Child-shell
measurements therefore arrive in the same root report instead of becoming
unattributed time.

`profile_report()` returns the latest invocation, `profile_summary(window=N)`
aggregates mean and p95 values across recent invocations, and
`profile_lines(window=N)` formats the expensive sections in descending order.
The animated demo places the hottest current section in the window title and
prints the full rolling table when it exits; the headless image command prints
the complete table immediately.
