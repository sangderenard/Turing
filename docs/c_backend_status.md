# AbstractTensor C backend status

The CFFI backend is now selectable through the ordinary AbstractTensor
placement contract:

```python
with AbstractTensor.use_backend("c"):
    values = AbstractTensor.tensor([[1.0, 2.0], [3.0, 4.0]])
```

It is a functional experimental CPU lowering target, not a replacement for
NumPy's high-level API. Its compiled C primitives now cover creation,
elementwise and scalar arithmetic, broadcasting, elementary functions,
comparison and masks, arbitrary-rank layout transforms, reductions,
cumulative sums, index selection, repetition, two-dimensional matrix
multiplication, padding, top-k, log-softmax, stacking, and concatenation.
Focused parity tests exercise these through AbstractTensor rather than by
calling C-backend methods as an alternate public API.

## Correctness repairs

- Registered the backend as `c`, so backend selection no longer rejects it.
- Removed the unconditional skip from its test module.
- Made core hooks obey the AbstractTensor calling convention.
- Allowed nested AbstractTensor construction to reach C stack/concatenate.
- Replaced unsafe dtype buffers. The old conversion allocated `int[]` or
  `float[]` and wrapped the result as a `double[]`; conversions now preserve
  the backend's double-storage invariant.

The last point does not constitute true dtype support. CTensor still has no
dtype field and all storage remains double precision. Integer and float32
casts currently quantize values in C and return them in safe double storage.

## Primitive closure and specialization

The design goal is deliberately smaller than duplicating NumPy. Advanced
mathematics belongs in AbstractTensor and should be composed from universal
operations whose backend hooks can be lowered into SSA, SPIR-V, or native
code. Backend-specific high-level functions are optional optimized lowerings,
not the semantic definition.

All scalar, elementwise binary, comparison, and unary primitives now enter
the C backend through `_apply_operator__`. Public hooks such as `exp_`,
`equal_`, and `clamp_min_` are thin adapters because the existing
AbstractTensor surface invokes those names, but they immediately rejoin the
single dispatcher. C uses `binary_double`, `binary_scalar_double`, and
`unary_double`; the formerly separate arithmetic functions and the Python
`_unary_c`/`_comparison_c` side doors have been removed.

The canonical opcode numbering lives in `ctensor_ops.h` as `CTensorOp`.
CFFI obtains the enum values from the compiled library, so Python does not
maintain a parallel table of magic integers. This is intentionally shaped for
later mapping to Nodus `KernelIR` unary/binary/comparison sub-operations.

The specialization folders are therefore the acceptance surface. Current
tests demonstrate that the C primitive set runs:

- `AbstractTensor.linalg.eye` and `norm`;
- the abstract neural-network `Linear` layer;
- the abstract MSE loss;
- the Riemannian cotangent mesh-Laplace composition, including shaped
  first-axis gather and ordinary multidimensional slicing;
- a broadcast linear expression, mask selection, softmax, reductions, and
arbitrary-rank layout composition.

Basic tuple indexing is likewise an AbstractTensor policy rather than a C
policy. The shared indexing specialization lowers integers, slices, negative
indices, and ellipses into backend `index_select` plus metadata reshape.
The C backend implements those primitives and only adds its shaped first-axis
gather required by constructs such as `vertices[triangles]`.

YoungMan's spline additionally requires the universal linear solve. Riemannian
and convolution specializations require stronger slicing/assignment and,
where applicable, fold/unfold primitives. Those are more meaningful next
targets than mechanically mirroring every convenience method on NumPy.

## Gap to a broad mathematical base

A mechanical comparison with the canonical NumPy backend now finds 94 hooks
on the C class and 34 NumPy hooks absent from it. Several of those 34 are
conversion conveniences or optional specialized kernels. Remaining
capability gaps include:

1. General slicing and mutation, scatter assignment, nonzero/argwhere, and
   diagonal construction.
2. Batched matrix multiplication and the primitive support required by the
   universal LU/solve path.
3. True bool/integer/float32/complex storage rather than numeric values held
   in double storage.
4. Fold/unfold and interpolation primitives used by convolutional
   specializations.
5. Runtime engineering: a prebuilt-library cache, removal of deprecated
   `ffi.verify`, allocation/error checks in C, explicit ownership, and
   benchmark coverage.

## FFT specialization

FFT must use the sibling `fftfree` project rather than acquire another
handwritten transform. `fftfree` already supplies a compiled C ABI with
reusable plans, R2C/C2R/C2C modes, batching, and explicit parallel dispatch.
The current workspace has a built `fft_cffi` library.

The remaining integration prerequisite is honest complex storage in CTensor.
Returning a final-axis real/imag pair from `fft_` would silently violate the
AbstractTensor FFT contract and break universal complex arithmetic. Once the
dtype/storage milestone lands, the C backend should bind `fft_init_full`,
`fft_execute`, `fft_execute_complex_batched`, and `fft_free`, with plan
caching keyed by transform shape and configuration. Higher spectral
algorithms remain AbstractTensor compositions.

## Recommended implementation order

First give CTensor explicit dtype and ownership metadata plus contiguous
strides. Then complete slicing/mutation and batched matmul so universal
linear algebra and geometry can run. Complex storage and the `fftfree` bridge
form the next separate milestone; fold/unfold follows according to the
convolution specialization tests.

Every family should use NumPy as the behavioral oracle and test zero-sized
tensors, scalars, negative axes, non-square shapes, arbitrary rank, and
invalid inputs. Python may coordinate shapes and allocation, but numerical
loops belong in the compiled C source.

## Specialization benchmark

Run the comparable CPU workloads with:

```powershell
python -m src.common.tensors.benchmark_backend_specializations --size 32 --warmup 2 --repeats 7
```

The demo performs three end-to-end AbstractTensor tasks:

- a broadcast neural projection, sigmoid, and MSE-style loss;
- a pairwise five-dimensional metric/affinity field with normalization and
  entropy;
- the Riemannian cotangent mesh Laplacian.

It reports tensor setup separately from warm execution and certifies every
output against NumPy before presenting timings. On the initial Windows CPU
run, all nine cases passed parity. Median warm times in milliseconds were:

| Task | NumPy | Torch CPU | C |
|---|---:|---:|---:|
| Neural projection/loss | 0.461 | 0.705 | 0.441 |
| Pairwise metric field | 0.607 | 0.907 | 0.684 |
| Riemann mesh Laplace | 1.837 | 3.165 | 3.566 |

These are small-problem dispatch measurements, not throughput claims. The C
backend is already competitive on dense compositions but loses on the mesh
task because many small gathers and realized intermediate tensors cross the
CFFI boundary. Larger-size sweeps and allocation reuse are required before
drawing performance conclusions.
