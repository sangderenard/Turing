# Native Fortran affine learner

`src.compiler.native_affine_learner` turns a trusted Python benchmark file into
a standalone Fortran executable. Python supplies exact examples at build time;
the resulting binary performs optimization, pruning, verification, terminal
visualization, and model export without Python or Pygame at runtime.

Build and run the included eight-value sorting problem:

```powershell
python -m src.compiler.native_affine_learner examples/learnable_sort.py
```

The callable form is:

```python
from src.compiler.native_affine_learner import compile_learning_visualizer

learner = compile_learning_visualizer(
    "examples/learnable_sort.py",
    "build/native-affine-learner",
)
learner.run()                 # live native visualization
learner.run(epochs=10_000)    # run a longer native experiment
```

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
