# Fast GEMM tile re-baseline — 2026-08-20

The ladder was rebuilt after all stride/prebake/compiler changes in the new root
`build/convergence-final2-bank`. The executable report is
`02_gemm_fast_ladder.json` beside this document.  The command was:

```powershell
python tools/kernel_bank_probe.py --root build/convergence-final2-bank --ladder 32 64 128 256 --kernel gemm --contract fast --rebuild-policy fresh --output docs/compiler_convergence/evidence/02_gemm_fast_ladder.json
```

`fresh` refuses a nonempty root and never deletes artifacts.  `missing` reuses
matching rows and compiles absences; `reuse-only` turns absences into refused
diagnostic rows.  Refusals and unexpected errors remain in the JSON and text
chart rather than disappearing.

## Rebuilt chart

| tile | worst absolute error | first launch | relaunch median | compute median | GF/s |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.13e-14 | 4.114 ms | 25.9 us | 116.8 us | 0.561 |
| 64 | 3.91e-14 | 4.531 ms | 62.6 us | 472.3 us | 1.110 |
| 128 | 9.24e-14 | 2.105 ms | 0.0 us | 3.251 ms | 1.290 |
| 256 | 2.56e-13 | 4.248 ms | 280.1 us | 27.359 ms | 1.226 |

Each profile has five warm and three cold samples.  All rows carry the source
SHA-256 and compiler fingerprint in the JSON.  All four were admitted against
the NumPy-backed reference.

## Live chooser checks

The isolated chart ranks 128 cubed highest by
`2 * tile**3 / compute_median`. Against the original live-bank rule, repeated
calls produced the same complete decision:

- 256 cubed, `must_divide=True`: tile 128;
- 384 cubed, `must_divide=True`: tile 128;
- 300 cubed, edges allowed: tile 128;
- 300 cubed, `must_divide=True`: explained refusal because none of the fitting
  admitted cores divides every axis.

Every specialized manifest also carries the concrete ABI parameter order,
matrix shapes, C-order strides, flat offsets, deterministic name-to-ID
binding, and literal specialized extents.

The unforced 256-cubed demo used the same explicit `fast` contract for the
parametric kernel, all tile cores, and `decide_tiling`.  It selected tile 128,
executed four independent C-block lanes with worst error 1.56e-13, and measured
27.79 ms serial versus 8.68 ms with the eight-worker pool (3.20x). The pooled
composition was 3.28x faster than the 28.49 ms single interchanged call.
Its consumed launch artifact is `03_gemm_256_prebaked_launch_matrix.json`:
eight module calls, four lanes, four chunk-one spans, with source and packed
strides for every A/B/C window.

Focused checks include six chooser/prebake tests and the unforced end-to-end
demo test. A one-worker budget now refuses composition absent positive serial
calibration; arbitrary edges remain prebakable through zero-filled square-core
packing and valid-window publication.

## Composed-product correction

Native execution proved that isolated-core throughput was insufficient as a
chooser: it omitted K-step count, output-lane count, and execution-slot waves.
The corrected chooser uses measured core time to estimate that complete
critical path. It selects 64 for 256 cubed (sixteen lanes across eight slots)
and 128 for 1024 cubed (64 lanes, eight K steps). The native measurements are:

| shape | chosen tile | native serial | native pool | pool GF/s | NumPy | error |
|---:|---:|---:|---:|---:|---:|---:|
| 256³ | 64 | 29.89 ms | 4.71 ms | 7.12 | 1.27 ms | 1.56e-13 |
| 1024³ | 128 | 1582.40 ms | 221.13 ms | 9.71 | 68.90 ms | 4.69e-13 |

A forced 256 tile at 1024 cubed measured 227.02 ms, confirming the selected
128 tile's advantage. Generated assembly contains AVX `vmulpd` and
`vfmaddpd`; the remaining gap is not a failure to vectorize the unit-stride
inner loop.
