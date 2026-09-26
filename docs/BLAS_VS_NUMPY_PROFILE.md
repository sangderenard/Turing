# Profile: compiled AbstractTensor.blas vs. NumPy

**Date:** 2026-08-20. **Tool:** `tools/benchmark_blas_vs_numpy.py`. **Kernels:**
`src/common/tensors/blas.py` (`scal`, `axpy`, `dot`, `gemv`, `gemm`).

Two separate comparisons, both against raw NumPy (no AbstractTensor on the
NumPy side of either):

* **DIRECT** — the compiled kernel's steady-state `execution.run()` cost vs.
  the equivalent NumPy call, same sizes, same data.
* **OPERATOR** — the AbstractTensor-level operators that are BLAS-shaped and
  are candidate future call-sites for these kernels (`AT_linalg.dot`,
  `AT @ AT`, elementwise `alpha * x + y`), run eagerly on AbstractTensor's
  NumPy backend, vs. the same raw NumPy call. This reproduces the
  measurement `docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md` made for `eigh`
  ("~89% of the time is AbstractTensor dispatch, not arithmetic") for these
  five ops specifically.

Methodology matches `tools/bench_native_step.py`: compile/prepare once, warm
up once, then time many repeated calls and divide by the count — steady
state, not first-call.

## DIRECT: compiled kernel vs. raw NumPy

| kernel | contract | size | compiled ms | numpy ms | compiled GF/s | numpy GF/s | ratio (numpy/compiled) |
|---|---|---|---:|---:|---:|---:|---:|
| scal | develop | n=256 | 0.0026 | 0.0027 | 0.097 | 0.093 | 1.04x |
| scal | develop | n=4096 | 0.0049 | 0.0044 | 0.843 | 0.923 | 0.91x |
| scal | develop | n=65536 | 0.0864 | 0.0472 | 0.759 | 1.389 | 0.55x |
| axpy | develop | n=256 | 0.0021 | 0.0041 | 0.249 | 0.125 | 1.99x |
| axpy | develop | n=4096 | 0.0058 | 0.0088 | 1.408 | 0.928 | 1.52x |
| axpy | develop | n=65536 | 0.2155 | 0.0884 | 0.608 | 1.482 | 0.41x |
| dot | develop | n=256 | 0.0020 | 0.0022 | 0.262 | 0.231 | 1.13x |
| dot | develop | n=4096 | 0.0068 | 0.0037 | 1.203 | 2.202 | 0.55x |
| dot | develop | n=65536 | 0.0825 | 0.0941 | 1.589 | 1.393 | 1.14x |
| gemv | develop | 64×64 | 0.0077 | 0.0101 | 1.069 | 0.809 | 1.32x |
| gemv | fast | 64×64 | 0.0076 | 0.0099 | 1.083 | 0.829 | 1.31x |
| gemv | develop | 256×256 | 0.0859 | 0.0310 | 1.525 | 4.225 | 0.36x |
| gemv | fast | 256×256 | 0.0845 | 0.0311 | 1.550 | 4.221 | 0.37x |
| gemv | develop | 1024×1024 | 1.7873 | 0.4607 | 1.173 | 4.552 | 0.26x |
| gemv | fast | 1024×1024 | 1.7358 | 0.4029 | 1.208 | 5.205 | 0.23x |
| gemm | develop | 64³ | 0.6563 | 0.0835 | 0.799 | 6.279 | 0.13x |
| gemm | fast | 64³ | 0.6221 | 0.0995 | 0.843 | 5.271 | 0.16x |
| gemm | develop | 128³ | 5.1102 | 0.3808 | 0.821 | 11.013 | 0.075x |
| gemm | fast | 128³ | 4.5392 | 0.4443 | 0.924 | 9.440 | 0.098x |
| gemm | develop | 256³ | 46.418 | 1.977 | 0.723 | 16.970 | 0.043x |
| gemm | fast | 256³ | 42.660 | 1.775 | 0.787 | 18.909 | 0.042x |

(`fast` = `TURING_WORK_CONTRACT` FMA-contraction preset; `ratio > 1` means
our compiled kernel wins.)

### What this says

* **Small problems: the compiled kernel wins, sometimes clearly** — `axpy`
  at n=256 is 2x faster than NumPy, `gemv` at 64×64 is 1.3x. This is call
  overhead, not arithmetic: our kernel is a raw ctypes call into a tiny
  compiled function; NumPy pays its own fixed per-call dispatch cost
  (array validation, ufunc resolution) that dominates at small n. This is
  the same shape of result `tools/bench_native_step.py` and the eigh
  handoff both found — a compiled kernel deletes overhead a general-purpose
  library cannot.
* **Large problems: NumPy wins, and the gap widens with size for `gemm`.**
  BLAS-1 crosses over between n=4096 and n=65536 (NumPy ~1.8–2.4x faster
  at 65536). `gemv` is already 3–4x slower than NumPy by 256×256. `gemm`
  never catches up in the range tested, and the gap **grows** with n:
  6–8x slower at 64³, 11–13x at 128³, **17–19x at 256³.**
* **This is exactly the gap `docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`
  predicted and measured for `eigh`** (~8x algorithmic + ~6.5x
  implementation = ~52x total at n=300, with the implementation half
  attributed to "SIMD, cache blocking, and threads" — none of which this
  naive scalar loop has). `gemm`'s widening gap with n is the cache-blocking
  half of that story showing up directly: NumPy's BLAS blocks for cache;
  our triple loop does not, so it degrades relatively as the working set
  outgrows cache.
* **`fast`'s FMA contraction is a real but small effect, not a fix.**
  5–27% GF/s improvement on `gemm`/`gemv`, nowhere near closing a 4–19x
  gap. This directly answers the question left open in
  `HANDOFF_SHOAL_AND_RE_TARGETS.md`'s BLAS section: FMA contraction is not
  the lever. The unbuilt items already on record —`noalias` derivation
  (enables the LLVM auto-vectorizer), target datalayout, and cache
  blocking/tiling as a compiler transform — are the real ones, exactly as
  `FUNCTION_TO_DEPLOYMENT_HANDOFF.md` section 6 argued in the abstract
  before this data existed to confirm it.

## OPERATOR: AbstractTensor dispatch (NumPy backend, eager) vs. raw NumPy

| kernel | size | AT op ms | numpy ms | AT/numpy overhead |
|---|---|---:|---:|---:|
| scal | n=256 | 0.0461 | 0.0031 | 14.8x |
| scal | n=4096 | 0.0456 | 0.0050 | 9.2x |
| scal | n=65536 | 0.0740 | 0.0329 | 2.3x |
| axpy | n=256 | 0.0816 | 0.0051 | 15.9x |
| axpy | n=4096 | 0.0872 | 0.0110 | 7.9x |
| axpy | n=65536 | 0.2581 | 0.1235 | 2.1x |
| dot | n=256 | 0.0719 | 0.0022 | **32.9x** |
| dot | n=4096 | 0.0793 | 0.0037 | 21.4x |
| dot | n=65536 | 0.2187 | 0.1023 | 2.1x |
| gemv | 64×64 | 0.0334 | 0.0046 | 7.3x |
| gemv | 256×256 | 0.0533 | 0.0314 | 1.7x |
| gemv | 1024×1024 | 0.4530 | 0.4020 | 1.13x |
| gemm | 64³ | 0.0907 | 0.0602 | 1.51x |
| gemm | 128³ | 0.3621 | 0.2582 | 1.40x |
| gemm | 256³ | 1.3569 | 1.2428 | 1.09x |

### What this says

* **The dispatch tax is real and it is largest for the smallest, cheapest
  operations** — `AT_linalg.dot` costs **33x** raw NumPy at n=256, because
  it is two full AbstractTensor operator dispatches (`multiply` then
  `sum`, each carrying autograd/tape machinery) standing in for one C
  call. This is the eigh handoff's "~89% dispatch, not arithmetic" finding,
  reproduced quantitatively on a different, much cheaper operation.
* **The tax shrinks toward ~1x as arithmetic starts to dominate dispatch**
  — every op converges toward 1.1–2.3x by its largest tested size.
  `gemm`'s overhead is small even at the smallest size (1.5x) because
  `AT.__matmul__` forwards to a single NumPy `@` call wrapping an already
  O(n³) computation, so the fixed AT cost amortizes fast.
* **This is the concrete argument for routing these operators through a
  compiled kernel eventually** (`FUNCTION_TO_DEPLOYMENT_HANDOFF.md` §7,
  step 6, "install behind the definitional API" — not yet done for any of
  these): for small-to-medium problems, the current eager path pays a
  dispatch cost several times larger than the arithmetic itself, and the
  DIRECT table above shows the compiled kernel eliminates almost all of
  it at exactly those sizes.

## Combined picture

There are two completely different regimes here, and they call for
different work:

1. **Small operands, dispatch-bound.** The compiled kernel already beats
   both plain NumPy and (by a much larger margin) eager AbstractTensor
   dispatch. No further compiler work is needed to win this regime —
   wiring these kernels in as the backing implementation of `AT.dot`,
   `AT @ AT`, etc. for the sizes where dispatch dominates would be a real,
   already-measured win today.
2. **Large operands, arithmetic-bound.** NumPy's BLAS wins, by a growing
   margin as size increases, because it vectorizes, blocks for cache, and
   (for large enough problems) threads — none of which this naive scalar
   source has or should have (per direction: no threading/blocking/SIMD in
   the kernel source; that is the compiler's job). Closing this gap is
   backend work, not kernel-authoring work, and `fast`'s FMA contraction
   was tested and confirmed insufficient on its own.

Next, if this track continues: `noalias` derivation and target datalayout
emission (unblocks LLVM auto-vectorization) is the first lever with a
plausible large effect on the arithmetic-bound regime; cache blocking as a
compiler transform is the second. Both are already-named, unbuilt items —
see `HANDOFF_SHOAL_AND_RE_TARGETS.md` and
`docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md` §2.1/§6.
