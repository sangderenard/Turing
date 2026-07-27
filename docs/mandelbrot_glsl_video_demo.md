# AbstractTensor Mandelbrot video demo

The Mandelbrot demo is an end-to-end exercise of the library rather than a
private shader with an AbstractTensor label:

```text
ordinary AbstractTensor quadratic-family program
  -> forward capture
  -> FusedProgram
  -> semantic ProcessGraph fusion plan
  -> one GLSL solve + palette + YCbCr dispatch
  -> backend-agnostic AbstractTensor JPEG mathematics
  -> 4:4:4 baseline JFIF bytes
  -> OpenDML MJPEG AVI with optional PCM audio
```

The only host representation required by compression is the intentional final
boundary from an octet tensor to bytes, plus the RIFF/JFIF structural headers.
The DCT, quantization, signed magnitude conversion, coefficient event
collection, canonical Huffman lookup, variable-length bit packing, marker
stuffing, and PCM conversion are AbstractTensor programs and therefore use the
selected backend.

## Run it

GPU animation and recording:

```powershell
python -m src.common.tensors.accelerator_backends.demo_mandelbrot_fusion --animate --width 512 --height 256 --only-glsl --iterations 24 --record-avi mandelbrot.avi
```

Audio-reactive offline recording, without real-time playback racing the
offline frame clock:

```powershell
python -m src.common.tensors.accelerator_backends.demo_mandelbrot_fusion --animate --width 512 --height 256 --only-glsl --iterations 24 --audio "music.mp3" --silent-audio --record-avi mandelbrot.avi
```

Width and height must currently be multiples of eight because the first JPEG
encoder deliberately exposes whole 8×8 blocks rather than silently padding or
cropping.

## What is fused now

ProcessGraph selects the entire fixed-iteration fractal solve, palette, and
RGB-to-YCbCr-equivalent plane construction as one multi-output compute
dispatch. The three component DCTs are expressed as one batched
AbstractTensor matmul program. Component coefficient-event extraction is also
batched, and invariant DCT, quantization, zigzag, and Huffman tensors stay
resident across frames.

The GLSL backend then retains elementwise expressions until a genuine layout,
reduction, prefix-sum, indexed update, or host-read boundary. Full-frame JPEG
scans avoid the older unpack-to-bits/repack-to-bytes carry path; that path
remains available when a large image is intentionally split into multiple MCU
batches.

The window caption includes the physical compute-launch count for the latest
frame. The final console report gives total launches per frame and shader-cache
hits/misses. This is deliberately separate from the phrase “one dispatch” in
the ProcessGraph line: that one dispatch describes the solve/palette/YCbCr
region, not the complete variable-length encoder.

On the development RTX 3060, the 256×128, 24-iteration, ten-frame recording
probe fell from roughly 1,462 physical launches per frame in the initial eager
encoder to about 260 per frame after fusion, resident resources, component
batching, and the direct final-scan path. The same probe records around three
frames per second. These are engineering measurements, not portable
performance claims.

## Honest remaining frontier

The AVI is valid and independently readable, but the complete encoder is not
yet one shader. JPEG entropy coding contains reductions, prefix sums,
gathers/scatters, component-dependent Huffman tables, a data-dependent valid
bit count, and byte stuffing. ProcessGraph currently plans the elementwise
front region; the encoder body is still ordinary AbstractTensor execution with
backend-local fusion between structural boundaries.

The next compiler step is not another application-specific shader. It is to
teach ProcessGraph planning and GLSL lowering to recognize fused
elementwise-plus-reduction/layout regions, hoist shape-only tensor creation,
and reuse their buffers across frames. A final variable-length byte count must
remain an explicit synchronization/I/O boundary unless the container writer
learns to consume a device-side bounded buffer and count directly.

Blindly replaying a forward autograd tape is not yet a correct substitute.
`to_dtype` currently does not produce a tape node, while composite indexed
updates can record both their internal mutation and their public operation.
Those capture gaps must be resolved at the common AbstractTensor/ProcessGraph
boundary before a recorded JPEG scan can be treated as a faithful executable
graph.

Production import of ProcessGraph no longer imports its demonstration test
suite, constructs the symbolic chalkboard problem, or eagerly imports plotting
and Torch. Those remain available only when their paths are actually invoked.
