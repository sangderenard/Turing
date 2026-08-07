# Native sorting process learner

The sorting learner is compiled through the same program path as other Turing
programs. There is no application-specific Fortran source generator:

1. `process_forward` is captured as a `FusedProgram` and projected to the
   forward card graph.
2. `capture_forward_reverse_cycle` derives the reverse program before output
   pruning and proposes corrected comparator-gate inputs.
3. The learning cycle and graph renderer are scheduled as compiler regions.
4. The normal IR → SSA → Fortran backend emits both regions and the normal C
   shell hosts their arenas and display port.

The renderer also goes through this path. Its RGB planes are `Fill` spans in
preallocated arenas. Sparse graph pixels and live numeric segment nodes are
written with indexed scatter operations, so image size never becomes a list of
Fortran literals.

Build a continuous native window:

```powershell
python -m src.compiler.native_sorting_process_learner examples/learnable_sort.py --output build/native-sorting-process
```

The prior command remains compatible and selects this compiler path when the
source provides `build_process_problem()`:

```powershell
python -m src.compiler.native_affine_learner examples/learnable_sort.py --output build/native-sorting-process --train-samples 9600 --frames 12000
```

Use `--compile-only` to create the executable without launching it. A frame
count of zero runs continuously until the native window closes. Comparator
gate outputs rotate into their input arena addresses through the standard
C-shell state-feedback contract; the old input arena becomes the next output
scratch. The application does not allocate, copy, or repackage them between
frames.

Python callers can retain the artifact and run it repeatedly:

```python
from src.compiler.native_sorting_process_learner import compile_sorting_process_window

window = compile_sorting_process_window(
    "examples/learnable_sort.py",
    "build/native-sorting-process",
    batch_size=64,
)
window.run()            # continuous
window.run(frames=600)  # bounded
```
