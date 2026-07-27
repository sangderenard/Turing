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

## Required replacement

The replacement must import the complete tensor program—including DCT,
quantization, coefficient events, Huffman lookup, bit packing, and marker
stuffing—through AST into ProcessGraph. Optimization and GLSL lowering must
consume that graph. Final tensor-octet transfer and RIFF/JFIF structural bytes
may remain explicit output/container boundaries.
