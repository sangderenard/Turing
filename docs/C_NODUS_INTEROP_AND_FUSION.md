# C backend, Nodus interop, and fusion boundaries

The C backend is the most direct native bridge from AbstractTensor to Nodus,
but “native bridge” must not collapse the repository's several program-like
representations into one. They encode different things at different levels.

## The canonical primitive vocabulary

`CTensorOp` in `ctensor_ops.h` is the canonical vocabulary of scalar and
elementwise primitives lowered by the C backend. Its values are read from the
compiled library by CFFI; Python does not maintain a second numeric table.
Every binary, unary, and comparison operator reaches the C backend through
AbstractTensor's one `_apply_operator__` method. Backend convenience hooks are
thin adapters to that same path.

This resolves the former structural defect in which binary operators used
`_apply_operator__`, while unary and comparison operators used separate
integer-dispatch side doors. It also gives Nodus's `KernelIR` a natural answer
to its current missing detail: `UNARY`, `BINARY`, and `CMP` are useful coarse
instruction classes, while a shared or mechanically translated `CTensorOp`
identifies the operation within each class.

The two projects should generate their enums from one schema before treating
numeric values as a stable ABI. Until then, translate by symbolic name and
version the serialized format.

## Keep the representations distinct

| Representation | Meaning | Appropriate use |
|---|---|---|
| AbstractTensor operator call | Backend-neutral mathematical intent | User algorithms and universal compositions |
| Autograd tape/process graph | Differentiation provenance and scheduling | Gradient construction and training |
| `FusedProgram` | Replayable AbstractTensor graph with feeds, state, metadata, and mode | Model-level capture and optimization input |
| Transmogrifier SSA/rewrite forms | Compiler and graph-transformation research | Normalization and lowering experiments |
| `CTensorOp` | Primitive native operation vocabulary | Shared semantic names and backend lowering |
| `FusedProgram` → private C slot plan | Equal-shape elementwise execution packet | Amortizing Python/CFFI calls without introducing a second semantic IR |
| Nodus `KernelIR` | Typed, low-level SSA compute kernel | CPU/GPU code generation and dispatch |
| Nodus path/Kpath tapes | Spatial/tool trajectories and provenance | Geometry, motion, and media scheduling |
| CTensor/Nodus tensor ABI | Buffer shape, dtype, strides, ownership, device | Zero-copy data exchange |

In particular, an autograd tape is not a wire format, a path tape is not a
compute instruction stream, and `FusedProgram` is not yet a KernelIR binary
encoding.

## First fused execution boundary

`c_primitive_program.py` and `ctensor_execute_primitive_program` implement a
deliberately narrow proof:

1. Python submits an instruction array and its feeds once.
2. C copies feeds into a contiguous slot workspace.
3. C executes any chain of the canonical equal-shape unary, binary, comparison,
   scalar, min, and max primitives.
4. C copies the selected output slot back once.

A four-operation sigmoid therefore crosses CFFI once rather than once for
negation, exponentiation, addition, and division. The current interpreter
still materializes one full slot per result. Its purpose is to prove the
program boundary and opcode fluency, not to claim finished kernel fusion.

The prepared form binds feeds and result slots once and reuses them. An
initial Windows CPU measurement of a four-operation chain showed approximately
2.2x lower dispatch time at 32 elements, 1.08x at 4,096 elements, and parity at
262,144 elements. These are dispatch diagnostics, not general performance
claims. At large sizes both paths still execute four C loops and incur the
same intermediate-memory traffic.

`autograd.forward_capture()` now records forward operations independently of
backward-rule eligibility. Every operation node carries `backward_available`
and a three-way `backward_status`: `available`, intentionally
`nondifferentiable`, or `missing`. `missing_backward_ops()` is consequently a
live development audit rather than a noisy list of predicates that should
never receive gradients.

Callers may supply temporary backward implementations either on capture with
`autograd.forward_capture(backward_overrides={...})` or for one gradient run
with `autograd.grad(..., backward_overrides={...})`. Call-time overrides take
precedence over tape defaults and the global registry. Captured nodes label a
tape-carried implementation as `override`; the registry is never mutated.
This makes finite-difference or experimental numerical derivatives easy to
inject without presenting them as canonical mathematics.

`compile_elementwise_tape` proves that a real forward trace can be normalized
into this program and replayed in C. A captured sigmoid arithmetic chain
lowers to `neg`, `exp`, scalar `add`, and reverse scalar `truediv`, then crosses
CFFI once. Smooth canonical unary paths (`exp`, `log`, `sqrt`) now preserve
ordinary autograd connectivity; their backward rules run on both NumPy and C.
Modulo also has a piecewise backward rule, and the C implementation was aligned
with NumPy/Torch floor-remainder semantics for negative inputs.

The next optimization is liveness-based slot reuse. After that, recognized
straight-line subgraphs can be emitted as C loops so intermediates remain in
registers:

```text
for i:
    output[i] = 1 / (1 + exp(-input[i]))
```

That removes both repeated CFFI transitions and repeated full-array memory
traffic. Existing Nodus microkernels such as fused add/multiply are useful
specialized targets; they should be selected from the same normalized
private C execution plan rather than exposed as unrelated AbstractTensor operators.

## Two related routes

The immediate native submission route is:

```text
AbstractTensor algorithm executed once on any backend
  -> forward autograd trace
  -> validated elementwise normalization
  -> FusedProgram -> private CTensor slot plan (portable interpreter/debug target)
  -> one CFFI call, or the same native packet hosted directly by Nodus
```

The broader compiler route remains a separate Python responsibility:

```text
AbstractTensor/FusedProgram semantics
  -> typed SSA with shapes, dtype, constants, and regions
  -> SPIR-V-compatible operations
  -> SPIR-V module
```

Nodus KernelIR is a compatibility and hosting target for that typed work, not
a reason to move Turing's lowering compiler into Nodus. Only elementwise
regions lower through the established FusedProgram. Reductions, matmul, indexing,
FFT, geometry kernels, and stateful operations remain region boundaries until
they receive explicit typed instructions. FFT continues to lower through
`fftfree`; it should not be reimplemented as an elementwise fused region.

## Native handoff milestones

1. Define a versioned schema that generates `CTensorOp` names for Python, C,
   and Nodus, plus Nodus `UNARY/BINARY/CMP` sub-operations.
2. Specify a shared tensor descriptor: dtype, rank, shape, strides, device,
   ownership, lifetime, and error reporting. The current C backend is
   double-only and contiguous; Nodus is primarily float32.
3. Capture eligible forward autograd-trace regions as typed FusedPrograms,
   with explicit constants and broadcasting rather than implicit Python
   behavior.
4. Add liveness allocation and direct output-buffer execution.
5. Let Nodus consume the packet in-process, bypassing CFFI. Preserve the C
   interpreter as the reference implementation and parity oracle. Keep the
   Python-to-SPIR-V compiler as a distinct, higher-level effort.
6. Add cost-based fusion: tiny chains benefit from fewer crossings, while
   large kernels benefit most from eliminating intermediate memory traffic.
7. Only then stabilize serialization and cache compiled KernelIR by program,
   dtype, shape signature, and target capabilities.

This route makes the C backend useful immediately while preserving clean
ownership: Turing captures mathematical intent, the primitive schema gives
the projects a shared language, and Nodus owns native scheduling and code
generation.

## Persistent Tensor Calculator

The workspace now contains a standalone `tensor-calculator` native library.
It is the shared execution substrate beneath, rather than another translation
format between, Turing and Nodus.

- Turing's ordinary eager C calls remain on the existing direct C functions:
  benchmarks showed that routing every tiny operation through a second Python
  binding added 15–40% dispatch overhead.
- Prepared Turing FusedPrograms can bind their CTensor buffers once to
  Tensor Calculator, compile handle resolution once, replay synchronously, or
  submit asynchronously. Set `TENSOR_CALCULATOR_PROGRAMS=1`; `auto` selects it
  when `TENSOR_CALCULATOR_WORKERS` is nonzero.
- Nodus maps its existing in-memory tensors and binds those exact pointers as
  external calculator tensors. No payload is copied.
- The calculator supplies both F32 and F64 versions of the canonical 28
  primitives, while Turing's historical CTensor storage remains F64-only.

The calculator follows AbstractTensor's singular-access trust model.
Allocation, binding, release, and preparation are externally sequenced and
have no registry mutex. Prepared synchronous execution takes no mutex.
Synchronization exists only for the explicitly optional asynchronous queue
and job completion. Submissions resolve and pin tensors before queueing, so
workers never access the mutable registry.

This is also why the direct and persistent forms coexist. A known-pointer
single operation should call the calculator's raw synchronous ABI. A repeated
dependent chain should use an opaque prepared program. Independent large
chains may enter the worker queue. Selection is a measured submission policy,
not a change in mathematical semantics.

## Five-way execution benchmark

`benchmark_nodus_calculator.py` runs the same float64 sigmoid chain on:

- eager AbstractTensor with NumPy;
- eager AbstractTensor with Torch, on a selectable device;
- eager AbstractTensor with the C backend;
- the developing GLSL backend as one fused compute shader, with a resident
  GPU input; and
- an already-prepared Tensor Calculator program bound directly to Nodus
  `InMemoryBackend` buffers.

Build the native runner in Nodus, then run from the Turing repository:

```powershell
cmake --build ..\nodus\build --config Release --target nodus_precomposed_calculator_benchmark
python -m src.common.tensors.benchmark_nodus_calculator --elements 262144 --warmup 10 --repeats 30
```

The command runs one backend at a time and prints that backend's median and
parity result before starting the next. No backend samples overlap. After the
final result it writes `benchmark_timing.png`, a four-panel comparison of:

- synchronized steady-state min, median, and max execution time;
- GLSL device-query time beside its host-visible completion time;
- setup, context, compilation, upload, output allocation, and readback costs;
  and
- host-visible and device-only throughput, plus an exact millisecond table.

Use `--plot <path>` to choose the PNG destination, `--show-plot` to open it
after saving, `--no-plot` for tabular-only operation, and `--quiet` to suppress
per-backend progress.

Use `--torch-device cuda` for a CUDA comparison, or
`--nodus-executable <path>` when the Nodus build is elsewhere. `--output`
writes the full result table as CSV. GLSL is included by default in the command
line benchmark and requires a real OpenGL 4.3+ compute context; use
`--without-glsl` on machines that intentionally lack one.

The comparison intentionally measures different execution modes. The three
Python rows include eager AbstractTensor dispatch and result allocation on
every repetition. The Nodus row excludes process startup and setup, then
replays a prebound, preallocated native program. Setup is reported separately,
all rows evaluate the full output against the analytical result, and CUDA is
synchronized around each timed sample. This makes the benchmark useful for
measuring the dispatch cost that precomposition removes, but it is not a claim
that the underlying Nodus arithmetic loops are intrinsically faster than
NumPy or Torch.

The GLSL row compiles all four primitives into one shader. Its input and output
SSBOs stay resident, the shader cache is warm for measured samples, and
`glFinish()` makes
the host-side `median_sec` a completed-operation measurement rather than a
command-submission measurement. Initial buffer creation/upload and final
readback are excluded from `median_sec` and reported separately.
`GL_TIME_ELAPSED` queries independently report actual device work as
`gpu_median_sec`. Context acquisition, initial upload, and shader compilation
plus first dispatch are also separate fields. This split makes both the fused
shader's strength and the current Python/OpenGL resource-management overhead
visible.

GLSL storage is float32, whereas these NumPy, C, Nodus, and default Torch rows
use float64. The benchmark therefore uses a documented float32 parity tolerance
for GLSL and records the dtype in every row. It does not silently present
reduced precision as exact float64 parity.

Small arrays should expose the precomposed path's low dispatch overhead. Large
arrays expose its present limitation: `neg`, `exp`, `add`, and `div` are still
four separate loops with full intermediate buffers. Loop fusion and liveness-
based buffer reuse are the relevant next optimizations there. GLSL already
performs that mathematical fusion and now accepts a caller-owned output chunk;
host-provided wrapped SSBOs extend the same persistent-buffer path to Pluck and
Nodus.
