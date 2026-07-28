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

## Complete start-to-finish source ingestion

`mandelbrot_recording_function_ast()` performs a source-level dependency walk,
then returns an AST module with exactly one top-level function:
`mandelbrot_recording_program`. That function contains the original reachable
helper and class definitions followed by the original `animate_glsl` body.
`build_mandelbrot_recording_process_graph()` gives only that one function AST
to semantic ingestion. No replacement rendering loop, source-file bundle
input, preliminary ProcessGraph discovery, or AbstractTensor execution tape is
involved.

The default recording result uses `profile="program"` and retains the complete
transitive program: Python control and UI, AbstractTensor calculation, JPEG
entropy work, tensor-to-byte materialization, AVI/audio writes, segment and
superindex updates, header patching, and final close. Use
`profile="tensor_control"` for compiler partitioning and `profile="complete"`
to inspect even the unused definitions in the source bundle.

Current source-front-end measurements on the complete bundle are:

- complete single-function audit graph: 7,739 nodes;
- recording program projection: 6,161 nodes and 84 reached definitions;
- forward reachability from `mandelbrot_recording_program`: 6,161 / 6,161
  nodes;
- recording tensor/control projection: 4,003 nodes;
- encoder-only tensor/control projection: 1,899 nodes.

Those are compiler-front-end timings, not render or shader timings.

The resulting graph is acyclic and contains no opaque AST placeholders. Its
resolved entrypoint path reaches:

```text
frame/control loop
  -> compiler-selected parametric solve
  -> palette / YCbCr planes
  -> 8x8 DCT and quantization
  -> coefficient-event collection
  -> JPEG Huffman symbol/codeword construction
  -> prefix placement and packed octets
  -> marker stuffing
  -> tensor-octet byte boundary
  -> AVI video chunks and synchronized PCM chunks
  -> RIFF/OpenDML segment indexes and superindexes
  -> size/header patching and final close
```

All Python syntax used by the source bundle is registered in the existing
ProcessGraph role-schema table for audit coverage. Definitions explicitly own
their complete regions, and the one root owns required environment nodes, so
the whole retained program is forward-traversable. A shared compiler contract
prevents those ownership edges from becoming SSA or fused-program operands.
Tensor arithmetic retains canonical operation names and
`execution_domain="abstract_tensor"`. Bitwise/integer nodes advertise BitOps
capability but are not selected without integer dtype evidence. The ingestion
layer implements no alternate arithmetic.

## Executable structured GLSL path

The live solve and color path is now compiled from that ProcessGraph rather
than from an execution tape:

```text
original Python source
  -> tensor/control ProcessGraph
  -> resolved function-region inlining
  -> one structured GLSL program
     - 92 canonical scalar primitives
     - one real GLSL `for` loop
     - four resident outputs: count, Y, Cb, Cr
  -> one compute dispatch per frame
  -> thin OpenGL presentation shader
```

The iteration loop is never source-unrolled. Compiling with 3 or 3,000
iterations produces the same 92 primitives and one loop; the iteration count
is a specialization of that loop bound. Eight changing scalar controls share
one persistent SSBO, so the complete dispatch uses three input bindings
(`unit_x`, `unit_y`, and controls) instead of ten independently managed scalar
buffers. Y/Cb/Cr stay resident and the presentation shader reads them directly.
The CPU does not reconstruct RGB for the window.

The local nested palette function also now closes over its enclosing
ProcessGraph values. This exposed and fixed a general semantic-AST issue:
nested definitions must inherit the enclosing lexical environment, not restart
from module globals. Tensor loop carries similarly remain typed through
`loop_result`.

Calling the former `capture_*` demo entry points still fails explicitly.
GradTape remains useful for execution recording elsewhere, but it is not this
source compiler.

## Measured profile

Measurements below are from an NVIDIA RTX 3060 under the PyOpenGL/pygame host.
They exclude the first five frames from stage means. AST ingestion, graph
projection, and shader-source generation currently take about 2.0 seconds
once at startup; shader compilation itself is cached by generated-source hash.

| Workload | Physical dispatches | GPU solve | Total frame |
|---|---:|---:|---:|
| 512×256, 64 iterations | 1/frame | 0.045 ms | 1.648 ms |
| 1920×1080, 128 iterations | 1/frame | 1.141 ms | 2.561 ms |
| 512×256, 4,000 iterations | 1/frame | 2.246 ms | 4.841 ms |

The `--profile` report separates control calculation, control upload, compute
submission, compute wait, GPU timer-query time, presentation, optional
encoding, and total frame time. For example:

```powershell
python -m src.common.tensors.accelerator_backends.demo_mandelbrot_fusion --animate --only-glsl --width 512 --height 256 --iterations 4000 --animation-frames 100 --no-detail-network --profile
```

## JPEG/AVI compilation frontier

The whole recording path is now honestly one entrypoint-expanded ProcessGraph,
but it is not one shader yet. Baseline JPEG contains global
boundaries—two-sided block transforms, reductions, prefix scans, scatter and
compaction, entropy bit packing, a final byte-count readback, and host
JFIF/OpenDML framing. A compute shader has no whole-dispatch global barrier, so
these must be partitioned into ordered kernels unless a later pass proves a
larger legal region.

Several general GLSL reductions now make that staged graph cheaper without
changing the compression algorithm:

- contiguous aligned first-axis slices are zero-copy SSBO range views;
- contiguous prefixes are zero-copy views even when their logical count is
  smaller than the backing allocation;
- a dispatch-batch scope retains required memory barriers but performs
  redundant binding cleanup and driver error validation once;
- compatible deferred branches at `stack`/`cat` boundaries use the existing
  multi-output fused-program path;
- a cast to the backend's already-active dtype is a zero-dispatch view.

At 512×256, 64 iterations, 4:4:4 MJPEG, these changes reduced the measured
recording path from 266.3 to 213.0 total physical GLSL launches per frame and
from roughly 455 to 371 ms/frame. One launch is the solve/color shader; the
others are encoder stages. The solve itself remains about 0.05 ms. This makes
the remaining problem unambiguous: it is dispatch-graph compilation and
entropy pipeline partitioning, not Mandelbrot arithmetic, rendering, disk
throughput, or a hidden CPU image transform.

The next compiler step is to lower the already-ingested full encoder graph into
neighboring elementwise, matmul/DCT, reduction, scan, scatter, and transfer
regions, then apply the same inspectible ProcessGraph cost model across those
regions. Final tensor-octet transfer and RIFF/JFIF structural bytes remain
explicit output/container boundaries.
