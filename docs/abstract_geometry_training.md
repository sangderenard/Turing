# Abstract geometry and refinement training

The black-box YoungMan demonstration now keeps its numerical reconstruction
path in `AbstractTensor`:

```text
AbstractTensor signed field and edge interpolation
    -> exported five-dimensional AbstractTensor manifold values
    -> streamed control batches
    -> AbstractTensor thin-plate vector spline
    -> AbstractTensor surface/Jacobian evaluations
    -> host topology decisions with AbstractTensor error norms
    -> AbstractTensor induced metric
    -> host triangle/edge incidence
    -> AbstractTensor cotangent weights, mass, and Laplacian
```

Host arrays remain intentionally responsible for discrete connectivity,
deduplicating control identities, convergence decisions, pandas reports, and
OpenGL upload. The source and reconstructed continuum references now share
the rank-N `AbstractTensor` Laplace--Beltrami operation in `laplace_nd`: the
former queries the continuous source manifold and the latter queries only
the published spline. These are explicit realization boundaries rather than
silent numeric backend changes.

The mesh operator now shares one immutable `CotangentTopology` assembly with
the established host DEC implementation. The host chooses triangle, edge,
boundary, and nonmanifold identities once; both numeric paths consume those
same identities. Cotangents, edge weights, lumped mass, flux divergence,
degeneracy masking, and the final division never leave `AbstractTensor`.
Irregular edge/vertex accumulation is a stable sort plus prefix-sum segment
reduction, so there is no numeric `.tolist()`/`np.add.at` break in the tensor
graph.

This cotangent operator is part of the Riemann/DEC suite, not a private demo
kernel. It is composed from canonical tensor indexing, arithmetic,
reductions, concatenation, cumulative sum, comparison, and selection. Each
selected backend therefore executes its normal primitives; no
mesh-Laplacian-specific NumPy or Torch implementation is hidden underneath.

That path is differentiated on NumPy, C, and Torch backends. Supporting it
also closed three general tape defects rather than adding demo exceptions:
`cat` and `cumsum` now record their canonical backward operations, reduction
backward restores the actual reduced axes before broadcasting, and repeated
integer-array indices accumulate in the indexing adjoint. Reverse scalar
arithmetic is recorded under the ordinary canonical operations (`add`,
`sub`, `mul`, and so on), not a second set of reverse-only tape names.

## Interpolator lineage

The spline follows the early JavaScript interpolation system's central
contract: a discrete, editable control program maps one intrinsic coordinate
set into several synchronized output channels. It retains explicit controls
and fits all embedding dimensions together. Unlike the former SciPy
`RBFInterpolator`, its thin-plate kernel, augmented polynomial system, solve,
and evaluation are expressed through `AbstractTensor`.

The FIFO boundary remains meaningful: YoungMan publishes immutable batches;
the interpolation program consumes a finite batch prefix and cannot consult
the source afterward. A later local/piecewise implementation can replace the
global thin-plate system without changing this publication contract.

## Which autograd path trains the predictor

Turing contains two related gradient layers:

- tape reverse mode in `autograd.py`, used by `AbstractTensor.backward` and
  `autograd.grad`;
- AutoAutograd's whiteboard/batched-VJP scheduler, which packages residual
  jobs and asynchronous graph execution around tensor operations.

The refinement predictor deliberately selects the first path. Its model is an
`abstract_nn.Sequential` of the established `Linear`, `Tanh`, and `Identity`
modules with `MSELoss`. One eager forward is captured into the shared
`FusedProgram`; every optimization epoch and every later alpha inference
replays that program through `ProgramRunner`. `AutogradProcess` rebuilds the
reverse-mode tape around each replay, performs the parameter update, and
exports forward/backward graphs and schedules. Training is accepted only when
loss decreases and those graphs are populated; there is no hand-written MLP
execution path or swallowed best-effort gradient call.

The AutoAutograd FluxSpring path is not claimed by this demo. During this work
its separate in-progress module could not collect because another concurrent
edit left an incomplete statement. It should receive its own acceptance run
after that work settles rather than being silently substituted for the
working tape path.

## Geometry-linked learning task

The pilot triangulation now retains a bounded sample from every refinement
generation rather than training only on its already-equalized final mesh.
Each sample derives topology-local features from triangle chart coordinates,
embedded edge lengths, area, aspect/conditioning, centroid geometry, and
polynomial position terms. Three Laplace targets are kept distinct:

- `reconstruction`: continuous spline Laplace minus continuous source
  Laplace;
- `discretization`: mesh Laplace minus continuous spline Laplace;
- `laplace`: mesh Laplace minus continuous source Laplace (their total).

This prevents a learner intended to improve triangulation from being trained
primarily against spline reconstruction error. The headless benchmark
defaults to `discretization`; deployment is always judged against the total
source-relative Laplace error. Targets use:

```text
log(1 + local_Laplace_error / robust_error_scale)
```

The captured `abstract_nn` program learns this refinement-pressure field. Its held-out
validation must beat a constant baseline and meet the configured correlation
threshold before the prediction is allowed to affect geometry. An accepted
model becomes an alpha map: the configured highest-pressure quantile requests
extra local refinement,
while the ordinary position and tangent certificates remain mandatory. A
second triangulation pass consumes that field and reports pilot and guided
triangle counts separately. The candidate is retained only if the guided
mesh itself converges and a fresh, area-weighted total Laplace objective beats
the pilot; prediction accuracy alone cannot authorize deployment.

Training uses at most 2,048 deterministically spaced history samples per solve
so epsilon-driven mesh growth does not silently change the training benchmark.
The live HUD and CSV report epochs, seed, samples, initial/final loss, loss
ratio, graph sizes, concurrency width, tensor setup, optimization, validation
inference, guided triangulation, independent certification, validation
quality, and whether alpha was deployed.

Alpha is presently a geometric-density proposal, not proof of improved
Laplace accuracy. More density can worsen a cotangent stencil when triangle
angles or boundary neighborhoods degrade. The next acceptance layer must
therefore include multiscale Laplace disagreement, minimum-angle/aspect
quality, and boundary-aware stencil evidence; guided output should only
supersede the pilot when those independent measurements improve.

## Cross-surface teacher and hinge mechanics

Laplace labels are privileged teacher data, never inference inputs. Corpus
generation evaluates several manifold families and phases, places the target
surface in a held-out group, and trains the deployed model only from chart,
embedded metric, aspect, and adjacency geometry. This prevents the alpha map
from simply receiving the answer it is meant to predict.

Training labels can be relaxed over adjacent triangles by a configurable
membrane-style diffusion step. It does not alter held-out labels and is not
part of the differentiable loss. Separately, the triangulator measures the
maximum principal angle
between neighboring tangent planes in the full embedding dimension. A
configurable hinge-angle limit requests conforming refinement while preserving
connectivity: it is a bend limit, not a fracture rule. The hinge angle is also
an inference feature so the learned density policy can distinguish smooth
regions from rapidly turning stencils.

## Backend and device benchmark

The CLI separates geometry and training placement:

```text
--tensor-backend numpy
--training-backend torch --training-device cuda --training-dtype float32
```

`c` is also a valid geometry and training backend. The complete headless
YoungMan → spline → triangulation → cotangent-Laplace → captured-neural
training round trip is exercised under C, not just the isolated dense layer.
The work required no demo-specific C branches: shared tuple indexing,
broadcasted batched matmul, persistent `copyto`, and the canonical elementary
function set were completed in the backend and its Nodus operation catalog.

The validated mixed-precision contract keeps source, spline, metric, and
Laplace validation in NumPy FP64 while placing only dense neural training on
CUDA FP32. Analytic/reference Laplace values are never reduced to FP32.

On an RTX 3060, an isolated 8,000-row, 1,000-epoch training comparison took
29.71 seconds with NumPy FP64 and 16.89 seconds with Torch/CUDA FP32 (1.76x).
The complete seven-example benchmark took 43.45 seconds on NumPy and 36.55
seconds with CUDA training (1.19x), because corpus construction remains a
host-side cost.

A historical 3,000-epoch CUDA-training run passed held-out validation
(correlation 0.529, validation loss 0.226 versus a 0.305 baseline) and reduced
independently measured Laplace RMS from 17.615 to 16.770 while increasing the
mesh from 1,238 to 1,406 triangles. That candidate exhausted its refinement
rounds, so the present stricter gate would retain the converged pilot rather
than deploy it. The result remains evidence that the learned pressure contains
useful signal, not evidence of a production-ready remesher.

## Reproducible benchmark

The headless matrix runner sweeps held-out manifolds, phases, and random seeds
and writes every result, including rejection reasons, to one CSV:

```powershell
python -m src.common.tensors.youngman.benchmark_geometry_training --output artifacts/geometry-training/matrix.csv
```

Its defaults keep geometry and certification in NumPy FP64 while training on
Torch/CUDA FP32. Use smaller phase, seed, epoch, and mesh limits for smoke
testing. A rejected candidate is a valid benchmark result; it must not become
an unreported fallback or disappear from aggregate statistics.

Running the entire geometry pipeline under Torch/CUDA remains an audit mode.
It exposed backend/device pooling, scalar-wrapper, implicit-device, and
precision assumptions. Host topology decisions and frequent realization make
that mode slower today, and small backend-dependent topology differences can
amplify through the cotangent operator.
