# Affine piece solving and matmul replacement

`src.compiler.affine_matmul_solver` tests whether a `FusedProgram` can be
replaced by one affine matrix multiplication.

It performs two independent analyses:

1. Every operation is solved in isolation as `y = A x + b` and checked on
   held-out probes. Certified local systems are lifted into sequential state
   matrices and composed.
2. The complete program is probed directly on zero, basis, and deterministic
   held-out inputs. This catches nonlinear pieces, and also recognizes cases
   where nonlinear terms cancel in the final observable result.

Bias is represented honestly by the homogeneous matrix

```text
[y]   [A b] [x]
[1] = [0 1] [1]
```

so “replace by matmul” means one multiplication in augmented coordinates. For
one-input/one-output programs, a certified result can also materialize a
`FusedProgram` containing reshape, matmul, bias addition, and output reshape.

Run the exact and deliberately nonlinear examples:

```powershell
python -m src.compiler.affine_matmul_solver
python -m src.compiler.affine_matmul_solver --nonlinear
```

The nonlinear command exits with status 2 and reports the blocking operation.

Feeds selected in `variable_feed_ids` are the variables being solved. Other
feeds are held as coefficients. This distinction matters: multiplying a
variable by a fixed captured coefficient is linear, while multiplying two
selected variable feeds is bilinear and cannot be represented by one fixed
matrix over those feeds.

Certification is empirical over a finite deterministic probe set. It is strong
evidence for captured numerical code, not a symbolic proof over every possible
floating-point value or control path. Tighten tolerances and increase
`probe_count` for higher assurance; retain explicit nonlinear boundaries when
the certificate fails.

A captured forward/reverse cycle exposes the same check directly:

```python
capture = solver.capture()
analysis = capture.analyze_matmul_replacement()
if analysis.fully_replaceable:
    replacement = analysis.replacement
```

The cycle's `solve_for` parameters become the matrix variables; its other feeds
remain fixed coefficients.
