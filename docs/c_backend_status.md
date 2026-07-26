# AbstractTensor C backend status

The CFFI backend is now selectable through the ordinary AbstractTensor
placement contract:

```python
with AbstractTensor.use_backend("c"):
    values = AbstractTensor.tensor([[1.0, 2.0], [3.0, 4.0]])
```

It is a functional experimental CPU backend, not yet a broad replacement for
NumPy. Its compiled C kernels currently cover elementwise arithmetic, scalar
arithmetic, square root, two-dimensional matrix multiplication, dimension
means, padding, top-k, log-softmax, paired gathering, stacking, concatenation,
and a collection of indexing helpers. Focused parity tests exercise selection
through AbstractTensor, arbitrary-rank stack/concatenate, reduction,
composition, and matrix multiplication.

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

## Gap to a broad mathematical base

A mechanical comparison with the canonical NumPy backend finds 49 hooks on
the C class and 76 NumPy hooks absent from it. Some NumPy-only hooks are
conversion conveniences, but the important missing families are:

1. Shape and layout: reshape, flatten, squeeze/unsqueeze, transpose, permute,
   swapaxes, and expand/broadcast.
2. Reductions and selection: sum, product, min/max, argmax, cumulative sum,
   any/all, nonzero, argwhere, and general gather/scatter behavior.
3. Elementary mathematics: exp, log, abs, negation, floor/ceil/round, finite
   checks, and minimum/maximum.
4. Comparisons and masks: equality and ordered comparisons returning a real
   boolean tensor, logical operations, and `where`.
5. Linear algebra: batched/broadcast matrix multiplication, diagonal
   construction, einsum or a deliberately smaller contraction primitive, and
   decomposition/solve support.
6. Spectral work: real/complex storage followed by FFT/IFFT and frequency
   helpers.
7. Runtime engineering: a prebuilt-library cache, removal of deprecated
   `ffi.verify`, allocation/error checks in C, explicit ownership, and
   benchmark coverage.

## Recommended implementation order

First give CTensor explicit dtype and ownership metadata, contiguous strides,
and C kernels for reshape/view, transpose, broadcasted binary operations, sum,
min/max, comparisons, and masks. That produces a credible dense numerical
core. Add batched matmul and solve next, then transcendental functions.
Complex storage and FFT should be a separate milestone rather than complicate
the initial dtype model.

Every family should use NumPy as the behavioral oracle and test zero-sized
tensors, scalars, negative axes, non-square shapes, arbitrary rank, and
invalid inputs. Python may coordinate shapes and allocation, but numerical
loops belong in the compiled C source.
