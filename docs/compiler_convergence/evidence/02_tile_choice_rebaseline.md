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

The chart ranks 128 cubed highest by `2 * tile**3 / compute_median`.  Against
the live bank, repeated calls produced the same complete decision:

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
