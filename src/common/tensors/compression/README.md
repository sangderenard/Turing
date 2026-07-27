# AbstractTensor compression

This package builds codecs from the same backend-independent operations used by
the rest of the tensor system. It has no NumPy or SciPy implementation path.

## The first primitive: canonical Huffman coding

A Huffman table maps a symbol to two values:

- an integer code;
- the number of meaningful low bits in that integer.

The code length is essential because the integers `0`, `00`, and `000` have the
same numeric value but are different bit strings.

Canonical Huffman coding stores only code lengths and symbol order. The first
code of each width follows this recurrence:

```text
first[1] = 0
first[n] = (first[n-1] + count[n-1]) * 2
```

Symbols of the same width receive consecutive codes. This makes a complete
table reconstructible from JPEG's compact DHT representation.

`CanonicalHuffmanTable` performs that reconstruction with AbstractTensor
comparisons, reductions, broadcasting, indexing, and arithmetic. Validation
checks the Kraft bound, which states that the allocated prefixes cannot consume
more than the complete binary-code space.

`huffman_code_lengths` also builds a tree from observed integer frequencies
without a host-side priority queue. Fixed tensor slots carry node weights and
symbol-membership rows. Each merge uses tensor top-k selection, gathers two
membership rows, increments their leaf depths, and updates the surviving slot.
Stable slot priorities make equal-frequency trees deterministic across
backends.

## Parallel codeword representation

`encode_codewords` gathers every symbol's code and length, then expands all
codewords simultaneously into an MSB-first tensor:

```text
symbols → codes/lengths → parallel quotient/remainder extraction → bits + mask
```

The explicit validity mask is intentional. It retains the ownership and exact
length of every symbol until the next stage compacts codewords into a continuous
bitstream.

`compact_codewords` now performs that compaction. A cumulative sum of code
lengths supplies each symbol's starting bit offset. Tensor scatter places valid
bits into those destinations, and a final 8-wide weighted reduction produces
real byte values. The result retains the symbol offsets and lengths needed to
attach `BitBitBuffer` provenance without reverse-engineering the byte stream.

`unpack_octets` performs the exact inverse byte-to-bit expansion.
`decode_with_provenance` uses the retained symbol offsets and lengths to
reconstruct every integer code in parallel and match it against the canonical
table. Encoding, packing, unpacking, and this provenance-guided decoding path
are lossless: the decoded symbol tensor must equal the input symbol tensor
exactly on every registered backend.

`decode_huffman_octets` is the independent wire decoder. It needs only a
canonical table, an octet tensor, and the exact valid-bit count. A tensor state
machine accumulates prefixes, compares them with the complete table in
parallel, detects invalid or truncated prefixes, and compacts completed symbols
with prefix sums and scatter. Retained encoder provenance is not consulted.

Canonical tables may use either an implicit dense alphabet or an explicit
tensor of unique integer symbols. Frequency construction also handles the
one-symbol edge case with a one-bit code, preserving the number of repeated
symbols on the wire.

Frequency construction produces the unconstrained optimal tree when
`max_bits=None`. When a limit is supplied, an AbstractTensor Package-Merge
construction produces an optimal tree under that limit. Packages retain tensor
membership rows over the original symbols, so their final reduction yields
bounded code lengths without a separate numerical implementation. A limit is
rejected only when the active alphabet cannot physically fit.

## Where loss enters

Huffman coding and bit packing do not discard information. An orthonormal DCT
is mathematically reversible, although floating-point round trips are limited
by the selected backend's precision. JPEG quantization deliberately discards
coefficient precision, and optional chroma subsampling deliberately discards
spatial color detail. A genuinely lossless codec path must skip those two JPEG
steps or replace them with reversible integer transforms and prediction.

This distinction is structural: the package can support both lossless and
lossy formats without describing lossy JPEG preparation as lossless entropy
coding.

## Neutral coefficient events

The reusable compression layer does not belong to a file extension, company,
or container. `coefficient_events` accepts any ordered block coefficients and
collects the information commonly needed by entropy formats:

- differential DC values;
- compact nonzero AC events;
- zero runs before each AC event;
- reversible signed-magnitude categories and payloads;
- valid event counts and trailing-zero counts;
- the original tensor shape needed for exact reconstruction.

These fields round-trip exactly through AbstractTensor on every tested backend.
A format adapter may translate them into JPEG run/category bytes, a different
published standard, or an experimental container without duplicating the
mathematics. Names such as JPEG describe an interchange contract at the edge;
they do not define ownership of the underlying transforms, prediction,
run-length coding, or canonical entropy primitives.

## Entropy symbol streams

`entropy_symbols` performs the next neutral translation. DC differences become
magnitude-category symbols with separate payload bits. Each nonzero AC event
becomes a collision-free integer token:

```text
1 + zero_run * (max_magnitude_bits + 1) + magnitude_category
```

Token zero is the block terminator. Symbols, amplitude payloads, payload
lengths, and validity remain parallel tensors. Fixed-capacity block rows can be
compacted through prefix sums and scatter, counted against an explicit
alphabet, passed through general Huffman construction, packed into octets, and
decoded without any format-specific numerical path. The inverse translation
reconstructs the coefficient events and original coefficient tensor exactly.

## Self-contained entropy scans

`entropy_scan` joins each Huffman codeword with the raw amplitude bits belonging
to that symbol, then compacts the combined rows into one continuous octet
stream. Its independent decoder alternates between canonical-prefix state and
the payload width declared by the decoded symbol. Truncated Huffman prefixes
and truncated amplitude payloads are both rejected.

This is now a coefficient entropy encoder: the resulting bytes contain all
symbols and signed coefficient payloads needed to recover the encoded scan.
It is not yet an image or video file encoder. Color conversion, complete block
ordering, table/shape metadata, format markers, frame timing, audio, and a
container remain outer layers.

## GLSL execution and fusion

The same compression functions run under `AT.use_backend("glsl")`; there is no
compression-specific shader implementation. Forward capture records the
ordinary AbstractTensor scan. The branchless, equal-shape AC run/category
calculation lowers through the existing `FusedProgram` GLSL emitter to one
compute shader.

The whole scan is not one fused shader today. Prefix sums, reductions,
shape-changing operations, indexing, and scatter are explicit boundaries of
the current equal-shape elementwise fusion compiler. They still execute through
the selected AbstractTensor backend, and their recorded graph gives a concrete
map for later generic fusion expansion. Python control-flow alternatives would
require AST/process-graph capture; tensor masks and static unrolling remain
ordinary recorded dataflow.

## JPEG direction

JPEG is a first consumer rather than the owner of these primitives:

1. RGB to YCbCr;
2. optional 4:2:0 chroma reduction;
3. batched 8x8 DCT;
4. quantization;
5. zigzag gather;
6. DC differential and AC run/category symbols;
7. the canonical Huffman layer here;
8. byte stuffing and marker serialization;
9. Motion-JPEG frames in an AVI container with PCM audio.

The numerical stages remain AbstractTensor programs. Container bookkeeping is a
binary I/O boundary, not an alternate numerical backend.

The first block-transform path is also present. `orthonormal_dct_basis` creates
the DCT-II matrix using tensor ranges, multiplication, and cosine.
`block_view_2d` exposes non-overlapping blocks through reshape and permutation,
and `dct_2d_blocks` applies the separable transform as two batched matrix
multiplications. JPEG luma preparation then performs level shifting,
quantization, symmetric rounding, and zigzag gathering without a numerical
package escape.

## Writable images and video

The JPEG adapter now writes complete baseline grayscale and RGB JFIF images:

```text
samples
  → 8x8 DCT and quantization
  → block-interleaved DC/AC events
  → standard ordered Huffman tables
  → amplitude-interleaved entropy scan
  → one-bit byte padding and 0xFF stuffing
  → SOI/APP0/DQT/SOF0/DHT/SOS/EOI serialization
```

The numerical route remains AbstractTensor. Marker construction and file bytes
are the explicit serialization boundary. Independent Pillow decoding verifies
the resulting files; Pillow is not used to encode them.

RGB input uses a real baseline 4:4:4 path: AbstractTensor converts RGB to
full-resolution YCbCr, transforms and quantizes each component, preserves
independent DC predictors, uses the standard luminance and chrominance Huffman
tables, and emits MCU order as Y, Cb, Cr. Chroma subsampling is not silently
performed.

The same adapters are available directly from an `AbstractTensor`:

```python
image_bytes = image.jpg()
image.jpg(path="frame.jpg")
frames.avi(path="animation.avi", fps=30)
```

An image has shape `(height, width)` or `(height, width, 3)`. A recording has
shape `(frames, height, width)` or `(frames, height, width, 3)`. These methods
are deliberately thin redirects into this package.

`containers.avi.MJPEGAVIWriter` is a stateful AVI writer. It accepts complete
JPEG frames and optional PCM chunks incrementally, writes interleaved `00dc`
and `01wb` chunks, and retains only compact index entries. Ordinary AVI 1.0
output receives an `idx1` seek index. OpenDML output is split into `AVI `/`AVIX`
RIFF segments and receives per-stream `ix00`/`ix01` standard indexes plus
header-resident `indx` superindexes and a `dmlh` total-frame count. No ffmpeg,
external codec, or numerical package participates.

Audio samples are numerical tensors until the byte boundary. `PCMFormat`
supports mono or stereo signed 16-bit PCM and IEEE float32 PCM. `encode_pcm`
performs gain, finite/range validation, clipping, and s16 quantization through
the selected AbstractTensor backend; `struct` only serializes the final scalar
values. `RationalAudioScheduler` assigns the integer number of samples due
after each video frame from exact rational cumulative time, so rates such as
29.97 fps do not accumulate drift. High-level `tensor.avi(...)` calls
interleave one audio packet per video frame and can pad a short input to the
video duration.

The JPEG scan is streamed by MCU-row batches. Each batch carries the preceding
Y/Cb/Cr DC predictors and the unfinished entropy byte into the next batch.
Only the last batch applies JPEG's one-bit fill, and `0xFF` stuffing occurs as
bytes become complete. Changing `mcu_rows_per_batch` therefore changes peak
working memory without changing a single output byte.

Color is not three independent grayscale files. RGB is converted to YCbCr;
the three full-resolution 4:4:4 components have separate quantization and
Huffman table selection, independent DC predictor histories, and a single
interleaved JPEG entropy scan.

Current recording limits are deliberate and visible:

- JPEG color is full-resolution 4:4:4 rather than 4:2:0;
- width and height must be divisible by eight;
- MJPEG compresses each video frame independently and can therefore be much
  larger than an inter-frame MPEG-family encoding;
- PCM is uncompressed: s16 quantizes normalized input to signed 16-bit samples,
  while f32 preserves values at IEEE float32 precision;
- MPEG video and MP3/AAC audio are not implemented.

Examples:

```powershell
python -m src.common.tensors.compression.jpeg.demo_frame
python -m src.common.tensors.compression.containers.demo_video --output tensor-field.avi --backend glsl --width 1920 --height 1080 --frames 240 --fps 30 --tone-hz 220 --sample-rate 48000 --channels 2 --pcm-dtype s16le --opendml
```

The fused Mandelbrot renderer can record the actual resident GLSL solve. When
`--audio` is supplied, the same decoded Pluck input that controls the dream
path is written as a synchronized PCM stream. Recording uses an offline frame
clock: live speaker playback is disabled, control sampling and PCM extraction
both use `frame / record_fps`, and the displayed palette phase/drive are passed
into the AbstractTensor JPEG colorizer. Slow encoding therefore cannot make
audio run ahead or silently turn an animated display into repeated static
frames.

```powershell
python -m src.common.tensors.accelerator_backends.demo_mandelbrot_fusion --only-glsl --animate --width 3840 --height 2160 --iterations 4096 --audio music.wav --silent-audio --record-avi mandelbrot-music.avi --record-fps 30
```

## Scaling frontier

The former `block_count × 63 × 63` coefficient selection was replaced by a
fixed 63-step tensor scan and rank/scatter compaction, making temporary event
storage linear in the blocks in one MCU-row batch. JPEG files are written
incrementally and AVI media is appended frame by frame, so the implementation
no longer retains a recording or a full-frame entropy scan in memory.

An 8K render is consequently a workload question rather than a container
architecture blocker. The active peak still includes one input frame plus the
DCT/event tensors for `mcu_rows_per_batch` rows; lowering that option trades
more dispatch/serialization boundaries for less memory. MJPEG and 4:4:4 also
make the output intentionally large, so OpenDML should be enabled for long
recordings.

The container layout follows the
[Microsoft AVI RIFF reference](https://learn.microsoft.com/en-us/windows/win32/directshow/avi-riff-file-reference)
and the
[OpenDML AVI 1.02 specification](https://www.jmcgowan.com/odmlff2.pdf).
Tests walk every RIFF segment, resolve both levels of indexes to the actual
JPEG/PCM payloads, and use Pillow as an independent JPEG decoder. On Windows,
the AVIFile system reader independently opens and seeks every frame and PCM
sample in the AVI 1.0 A/V form. That legacy API intentionally sees only the
first `AVI ` RIFF section of a forced multi-`AVIX` test; the OpenDML-specific
cross-segment guarantee is therefore validated structurally against the
published two-level index rules rather than claimed from that AVI 1.0 reader.
