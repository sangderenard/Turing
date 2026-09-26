# Native Fortran affine learner

`src.compiler.native_affine_learner` turns a trusted Python benchmark file into
a native `bind(C)` Fortran learning kernel hosted by the common C shell. Python
supplies exact examples at build time; the resulting binary performs
optimization, pruning, verification, and RGB visualization without Python or
Pygame at runtime.

Build and run the included eight-value sorting problem:

```powershell
python -m src.compiler.native_affine_learner examples/learnable_sort.py
```

`learnable_sort.py` now exposes `build_process_problem()`, so this historical
command dispatches to the shared compiler process described in
`NATIVE_SORTING_PROCESS_LEARNER.md`. It captures the sorting network's forward
Graph IR, derives its reverse graph, schedules both with the renderer, lowers
the regions to SSA, and emits the native Fortran/C-shell runnable. It does not
use the legacy hand-written affine Fortran generator below.

The window runs continuously until it is closed. Its left panel is the fixed
reference/oracle stick-and-ball graph. Its right panel is the changing affine
neural graph: live coefficients are colored sticks and inputs/outputs are
balls. Bottom bands show held-out loss, exactness, and relative operation cost.

The callable form for legacy files that expose only `build_benchmark()` is:

```python
from src.compiler.native_affine_learner import compile_learning_window

learner = compile_learning_window(
    "examples/learnable_sort.py",
    "build/native-affine-learner",
)
learner.run()                 # continuous native window; close it to stop
learner.run(frames=600)       # bounded run for automation
```

The compilation contract deliberately separates parameters:

- **Locked:** exact training/validation pairs, dimensions, and reference cost.
- **Open and fed back:** matrix weights, bias, and epoch.
- **Open controls:** learning rate and pruning pressure.

Use `--console` to build the earlier standalone ANSI Fortran visualizer instead.

The Python file must define:

```python
def build_benchmark(*, seed, train_samples, validation_samples):
    return {
        "name": "problem_name",
        "train_inputs": ...,       # shape: samples, input features
        "train_targets": ...,      # exact oracle outputs
        "validation_inputs": ...,
        "validation_targets": ...,
        "reference_operations": 100,
    }
```

The executable learns `y = A*x + b` with gradient descent, soft-thresholding,
and a progressively tightening operation budget. Small coefficients are pruned
until the affine candidate crosses below the declared reference cost. It shows the evolving matrix, training and
held-out error, exact validation count, candidate operation count, and an
example target/guess pair. Its best cost-aware candidate is written to
`best-affine-model.txt`.

The included sorting oracle is intentionally beyond a fixed affine map. The
model can learn approximate order statistics but should not be mistaken for an
exact replacement merely because error decreases. The exact count and held-out
loss stay visible, preserving the distinction between learning and proof.
