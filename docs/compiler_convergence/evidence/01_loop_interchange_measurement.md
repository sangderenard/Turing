# Loop-interchange integration evidence — 2026-08-20

## Subject and toolchain

- Authored kernel: `src/common/tensors/blas.py::GEMM_SOURCE`
- Authored SHA-256: `ac7333ea79d700cb7f3590ed26e397a57908008aac0a1bdfd6ecd49a4bb3cd5a`
- Transformed SHA-256: `8254f41d1a51e2c0e8fa3e1a2296d0d7e6d470f71533cf4cf4d80b0bd17429e0`
- Compiler: ziglang-bundled clang 21.1.0, target
  `x86_64-unknown-windows-gnu`
- Fast flags reaching clang: `-O2 -O3 -ffast-math -march=native`
- Timing path: `tools.benchmark_blas_vs_numpy.run_direct`; prepare once,
  warm once, time repeated `execution.run()` calls with `perf_counter`.

The controlled comparison used the same current compiler, generated kernel,
inputs, native backend, and fast flags.  Its control contract differed only in
setting `inexact_identities=False`, which prevents loop interchange.  This is
stronger isolation than comparing two different repository revisions.

## Controlled results

| square size | repetitions | control GF/s | interchanged GF/s | speedup |
|---:|---:|---:|---:|---:|
| 64 | 200 | 1.353 | 1.226 | 0.91x |
| 128 | 80 | 1.424 | 1.307 | 0.92x |
| 256 | 20 | 1.267 | 1.187 | 0.94x |
| 512 | 4 | 0.320 | 1.259 | 3.93x |
| 1024 | 2 | 0.163 | 1.019 | 6.25x |

At 1024 cubed the control took 13.168 s/call and the interchanged kernel took
2.108 s/call.  NumPy measured 24.30 GF/s in the interchanged run.  Interchange
therefore removes the old loop order's large-matrix cache collapse, but it is
not itself the remaining backend-performance solution.

The small-size loss is real and bounds the next task: tile-choice measurements
must be rebuilt on the interchanged floor, and any eventual size-aware chooser
must preserve the crossover rather than assuming interchange wins universally.
The canonical source compiler does not currently know runtime `m`, `n`, and
`k`, so this integration does not invent a compile-time threshold it cannot
observe.

## Shape and correctness evidence

Optimizing the emitted LLVM with the same fast flags produces `<4 x double>`
loads, multiplies, adds, and stores in the inner `j` loop.  The planned-region
helper is inlined by clang.  Thus the post-transform 64-cubed result is the
intended unit-stride native shape even though call overhead and small working
sets make it slower at that size.

`tests/test_loop_interchange.py` has 13 passing checks.  They cover exact-mode
refusal, the narrow licensed grammar, adversarial accumulator/effect/alias
cases, authored-versus-transformed numerical equivalence, compiled fast-mode
equivalence to NumPy, JSON-shaped receipts, and identical SSA instruction IDs
and receipts in two fresh Python processes.

The bounded compiler regression floor also passed:

- translation scorecard: 17/19 equivalent; the two existing materialization
  stops are name rebinding and `any()` over a generator predicate;
- precompile-to-SSA: 34 passed;
- symbolic fluid native runtime: 1 passed;
- abstract tensor indexing: 2 passed;
- SSA fusion regions: 1 passed;
- region kernel deduplication: 2 passed;
- compiled linalg: 6 passed, 1 strict xfail.
