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

## Source ingestion now present

`build_mandelbrot_encoder_process_graph()` now scans the master program, the
original Mandelbrot demo source, and the original compression sources together.
The entrypoint composes the original parametric solve, palette, and
`encode_ycbcr_jfif`; it does not copy their implementations and does not execute
an AbstractTensor tape.

The default result is a tensor/control projection, not an attempted compiler
for all of Python. AbstractTensor values and operations are typed explicitly;
Python supplies loops, branches, contexts, calls, and shape/scalar
dependencies. Host byte materialization remains an explicit boundary. Use
`profile="complete"` only to inspect every source syntax node.

Current source-front-end measurements on the complete bundle are:

- complete audit graph: 11,928 nodes;
- tensor/control compiler graph: 1,916 nodes (84% removed);
- warm complete builds: roughly 1.0–1.1 seconds;
- warm tensor/control builds: roughly 1.0–1.1 seconds, including construction
  of the audit graph and projection.

Those are compiler-front-end timings, not render or shader timings.

The resulting graph is acyclic and contains no opaque AST placeholders. Its
resolved entrypoint path reaches:

```text
parametric solve
  -> palette / YCbCr planes
  -> 8x8 DCT and quantization
  -> coefficient-event collection
  -> JPEG Huffman symbol/codeword construction
  -> prefix placement and packed octets
  -> marker stuffing
  -> tensor-octet byte boundary
```

All Python syntax used by the source bundle is registered in the existing
ProcessGraph role-schema table for audit coverage. Tensor arithmetic retains
canonical operation names and `execution_domain="abstract_tensor"`.
Bitwise/integer nodes advertise BitOps capability but are not selected without
integer dtype evidence. The ingestion layer implements no alternate arithmetic.

## Remaining compiler frontier

This is complete source ingestion, not yet a claim of complete executable
fusion. Function/control regions must next be inlined or region-lowered,
optimized as a ProcessGraph, partitioned at true synchronization and I/O
boundaries, and then lowered to GLSL. Final tensor-octet transfer and RIFF/JFIF
structural bytes may remain explicit output/container boundaries.
