# Exception control retrospective — 2026-10-30

## Status

The slope is down, but exception handling is not complete. Authored `try`, `except`, `raise`, `assert`, and several `finally` exits now have control IR and SSA paths with source-owned concordance records. Bounded native C and LLVM cases have executed handled raises, assertions, nested propagation, and numeric finalizers. These cases establish those paths, not universal exception semantics.

The exact source of `tensor_octets_to_bytes` in `src/common/tensors/compression/bitstream.py` is the current real-program probe. It now passes both the first guard's exception-control failure and the loop composer's obsolete `Raise` veto. The next refusal is an unsettled dtype for `ndims` in the codec function. No native codec artifact was admitted. The older `repros/codec_roundtrip.py` driver named in the continuation note is absent from the checked Turing worktrees and canonical checkout; the probe compiled the actual function source through `lower_ast_source_to_ssa`.

## What changed

- Authored exception sites acquired exact source transfers, control owners, handler routes, and SSA checkpoints. A root `raise` or `assert` can terminate through an executable fault exit. Nested unmatched exceptions route outward through their enclosing `try` controls.
- `finally` lowering now carries source-owned input and output identities across normal, return, loop, and exceptional exits that have been represented. Empty output sets are valid when there is no value to carry.
- Zero-argument built-in exception constructors directly under `raise` no longer create a second, non-executable exception site. The real `raise` transfer remains.
- Structural folding now records an exact retirement when it rejects an authored `raise` or `assert` arm. Exception-control construction and the final concordance audit consume that same record. This fixed the first guard in the codec source.
- The loop composer no longer treats an authored `Raise` as a blanket reason to withhold the loop control program. A bounded handler raise inside a `for` loop now follows its source-owned outward route in native C and LLVM execution. This does not retire the legacy `ValidationBlock` path for direct `if` guards; that parallel path still needs reconciliation.
- The `view` audit's late method reachability and Ellipsis index facts were repaired at their writers. Its latest run reported 348 rows across 21 functions, zero findings, and a closed latch.

## Evidence and limits

The targeted exception gate (`tests/test_source_exception_admission.py`, `tests/test_metadata_copy_control_ownership.py`, and `tests/test_native_float_int_status.py`) passed 82 tests on 2026-10-30. The structural-retirement regression compiles a bounded guard whose selected terminal arm makes a following `raise` unreachable; its control-coverage and admission audits are clean. A bounded `try` inside a `for` loop compiled and executed on native C and LLVM: two normal iterations returned the expected value, and a NaN conversion reached the handler whose new `ValueError` left the loop. The exact codec source still refuses at its `ndims` dtype, so that program's `try` and `except` have not executed.

Other exception shapes still require execution evidence, including finalizers that read a name absent on an unmatched edge and any path behind the codec dtype refusal. The current result must not be described as complete native exception support.

## What the failures taught us

The recurring failure was a pass changing a value or control path without settling every record that described it. Early work sometimes fixed a downstream refusal while leaving the writer's incomplete decision intact. The effective rule is local: the pass that selects, removes, or routes a path posts its exact consequence in the concordance at that moment. Later graph traversal, SSA lowering, and audits read that record instead of reconstructing intent from source text, names, or a surviving node count.

The structural guard was a compact example. Ingestion recorded its `raise`; the fold selected the other arm and removed the graph node; retained source control still showed the authored `raise`. The fold now publishes the retirement tied to its existing arm-selection proof. The next frontier exposed an older AST veto in loop eligibility: it stopped construction before the already existing control overlay could nest the Try under the loop. Removing that veto let the source-owned exception route execute in a bounded native case.

## Next measured step

Investigate the unsettled `ndims` dtype at the exact producer/consumer boundary in the codec compile. The legacy direct-guard `ValidationBlock` path should be checked against source-owned exception control so an authored raise has one executable owner. Once the exact codec source reaches an artifact, execute its normal and exceptional paths against Python behavior before claiming the real program works.
