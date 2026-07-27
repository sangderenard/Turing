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

`build_mandelbrot_encoder_process_graph()` now imports the master program, the
original Mandelbrot demo source, and the original compression sources together.
The entrypoint composes the original parametric solve, palette, and
`encode_ycbcr_jfif`; it does not copy their implementations and does not execute
an AbstractTensor tape.

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
ProcessGraph role-schema table. Tensor arithmetic retains canonical operation
names, while bitwise/integer nodes merely advertise eligibility for the
existing Turing/BitOps expansion pass. The ingestion layer implements no
alternate arithmetic.

## Remaining compiler frontier

This is complete source ingestion, not yet a claim of complete executable
fusion. Function/control regions must next be inlined or region-lowered,
optimized as a ProcessGraph, partitioned at true synchronization and I/O
boundaries, and then lowered to GLSL. Final tensor-octet transfer and RIFF/JFIF
structural bytes may remain explicit output/container boundaries.
