# Forward/reverse parameter-solving cycle

`src.common.tensors.abstract_nn.forward_reverse_cycle` captures a real
AbstractTensor forward graph without output pruning, derives a target-driven
reverse graph from the canonical backward registry, and fuses both into one
`FusedProgram`.

The fused call has three observable boundaries:

1. current model/input parameters and incidental constants enter as feeds;
2. every retained terminal forward output enters again as a desired target;
3. current forward outputs and proposed predecessor parameters leave together.

This makes one correction cycle portable to any `FusedProgram` target. The
included native path emits Fortran through the shared SSA backend, compiles its
`bind(C)` subroutine, and invokes it from Python using the generated ABI.
`FortranCycleExecutable.cycle()` feeds each native proposed parameter back into
the next native invocation; its optional target hook can replace desired-output
feeds between calls.

## Run it

```powershell
python -m src.common.tensors.abstract_nn.forward_reverse_cycle --iterations 12
```

To emit and execute the same final cycle in Fortran:

```powershell
python -m src.common.tensors.abstract_nn.forward_reverse_cycle `
  --iterations 12 `
  --emit-fortran build/forward_reverse_cycle `
  --compile-fortran
```

The build directory is ignored. The command prints the solved parameters,
retained forward outputs, generated source/library paths, and native outputs.

## Hooks

Target strategies receive `(iteration, current_outputs)` and must return one
target for every retained output. Built-ins are:

- `FixedTargets`: fixed named goals while unmentioned terminal outputs hold
  their current value;
- `InterpolatedTargets`: scheduled movement from current values toward goals;
- any callable implementing the same protocol.

Correction strategies receive the current and proposed parameter mappings.
Built-ins are:

- `GradientCorrection`: scheduled gradient descent expressed completely inside
  the fused Python/Fortran IR;
- `ClippedCorrection`: gradient capture followed by a bounded host update;
- `CallableCorrection`: arbitrary Python postprocessing.

`emit_fortran()` rejects host-only correction hooks rather than pretending that
their omitted behavior was fused. Custom corrections become natively fusible
when expressed as `FusedProgram` operations with their state exposed as feeds
and outputs.

## What “solve” means

The reverse graph computes a local VJP of squared residuals. Repeated cycles can
solve parameters when the objective supplies a useful gradient, but this is not
a promise of a unique or exact inverse. Non-injective systems can have many
solutions; inconsistent targets can have none; nonlinear objectives can require
step schedules, constraints, regularization, or alternate correction hooks.
