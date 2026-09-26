# Rational feature timing

**Date:** 2026-09-24. **Tool:** `tools/benchmark_rational_features.py`.

This is an eager-substrate diagnostic for the motivating workload: start with
one tensor, apply six nonzero tensor divisions, and observe the quotient only
at the end.  It measures the eight subsets of `{complex, rational, precision}`
at 256 elements using the complete AbstractTensor surface.  The canonical
`numpy` backend is selected through `AbstractTensor.use_backend`; the tool
never imports or instantiates a backend class and never accesses backend
storage.  Precision width is two limbs.

Before NumPy was restored as the process default, validation resolved to
Nodus and exposed that native `complex128` arithmetic raises
`NodusUnsupported` there instead of falling back.  The recorded command keeps
its backend selection explicit so historical reruns do not depend on whatever
default is configured later.

Build, division-chain, and final-collapse costs are separate.  `Observe x` is
`(chain + collapse)` relative to the ordinary tensor in the same real or
complex domain; it excludes construction.  Each entry is the median of three
runs after one warmup.  Every result is checked against direct NumPy
evaluation.

```text
Python 3.11.7 | Windows AMD64 | AbstractTensor backend NumPyTensorOperations
py -3.11 tools/benchmark_rational_features.py --backend numpy --size 256 --steps 6 --limbs 2 --warmups 1 --repeats 3
```

| Variant | Set | Stored tensors | Build ms | Chain ms | Collapse ms | Observe x | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary real | - | 1 | 0.861 | 0.563 | 0.007 | 1.00x | 0.000e+00 |
| Precision[2] | P | 2 | 1.930 | 161.131 | 0.196 | 283.08x | 4.441e-16 |
| Rational | R | 2 | 82.241 | 155.116 | 0.338 | 272.77x | 6.661e-16 |
| RationalPrecision[2] | R+P | 4 | 238.783 | 416.972 | 25.257 | 775.98x | 4.441e-16 |
| ordinary complex | C | 1 | 0.761 | 0.798 | 0.005 | 1.00x | 0.000e+00 |
| ComplexPrecision[2] | C+P | 4 | 3.384 | 672.424 | 0.482 | 837.68x | 6.667e-16 |
| ComplexRational | C+R | 4 | 166.756 | 1846.386 | 0.416 | 2299.02x | 1.333e-15 |
| ComplexRationalPrecision[2] | C+R+P | 8 | 493.631 | 5420.051 | 50.040 | 6809.52x | 6.667e-16 |

## Reading the result

- The current real Rational chain costs about the same as the two-limb
  Precision chain on this small eager workload.  Rational replaces repeated
  quotient evaluation with component products, but each eager result also
  constructs and proves new component limits.
- Combining rational and precision is cumulative rather than free: four
  stored coefficient tensors and wide final division make R+P about 2.9
  times the observed cost of R alone here.
- Complex rational division is the expensive corner.  It expands into the
  ordinary complex quotient identity over rational coefficients, so one
  apparent division performs multiple rational multiplications, additions,
  cancellations, and safety proofs.
- Collapse is cheap for Rational because it performs one ordinary division.
  It is visible for RationalPrecision because the deferred boundary performs
  one wide division.  Most current cost remains in the chain, not collapse.

These numbers describe the present Python/eager implementation, especially
its whole-tensor limit inspection.  They are not a compiled-kernel forecast.
The benchmark is intended to remain fixed while direct descriptors, compiler
lowering, and cheaper limit receipts are added; those changes should be judged
by rerunning the same task rather than by comparing different expressions.
