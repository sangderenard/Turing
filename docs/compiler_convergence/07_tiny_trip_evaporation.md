# Task 7 — tiny-trip loop evaporation defect (resolved 2026-08-20)

## Resolution

- Loop decisions are hierarchical. A nested carried recurrence forms a
  preservation closure over itself and its lexical owners; unrolling is a
  lower-priority identity and cannot cross that closure.
- The 2³ baked-GEMM reference evaluator pin now passes instead of xfails.
- A native 4³ specialized GEMM admits through `KernelBank` and receives a
  performance-chart row.
- The blanket bank refusal at/below the unroll limit was removed.

## Remaining boundary

An unrelated dependent-inner-bound ABI issue remains pinned in
`test_literal_loop_bound_parameter_loss.py`: the mutated parameters survive,
but the outer induction capture can still appear as one unnamed formal. The
register-blocked GEMM path does not exhibit that leak.
