# Task 2 — tile-choice re-baseline after interchange

## Observed code state

- `src/compiler/tiling_strategy.py` now ranks admitted square cores by their
  projected composed critical path: measured core time multiplied by K steps
  and output-lane waves over the available execution slots. Isolated GF/s is
  retained as evidence, but no longer mistaken for whole-product latency.
- `KernelBank` fingerprints all compiler Python mtimes, so task 1 makes old
  manifests stale even if their source kernel is unchanged.
- `tools/kernel_bank_probe.py --matrix` builds parametric variants only.
- `tools/kernel_bank_probe.py --specialize` requests GEMM 8³, which
  `KernelBank.get()` deliberately refuses because it is at the loop evaporator
  threshold.
- The real 32/64/128/256 ladder is hard-coded in
  `tools/demo_gemm_tiled_deployment.py`, uses the default contract for its
  specialized cores, and is coupled to host-pool execution.
- `KernelBank` still writes a manifest note claiming parameter IDs are unstable
  across lowerings.  That statement conflicts with the deterministic-ID
  baseline and must be validated and corrected as part of the re-baseline.

## Work sequence

1. Add an explicit ladder option to a bank-focused tool.  It must accept kernel,
   contract, sizes, build root, and rebuild policy; it must not require running
   the host deployment demo.
2. Build and admit fast-contract GEMM cores at 32, 64, 128, and 256 in a new
   root.  Preserve refusals and errors as rows rather than skipping them.
3. Render a chart containing source/compiler identity, specialization,
   correctness error, first-launch, relaunch, compute median, samples, and
   derived GF/s.
4. Run `decide_tiling` for representative exact and edge-bearing task sizes.
   Assert that its selected tile equals the best live chart row under its
   documented ranking and that `must_divide` behaves correctly.
5. Re-run the 256³ demo without `--tile` so the chooser, not the test author,
   selects the core.  Keep correctness and lane-count assertions.

## Acceptance

- All successful rows were rebuilt after interchange and admitted against
  NumPy.
- The chooser ignores stale/refused variants and selects the measured best
  fitting live row.
- Repeating the choice against unchanged manifests is deterministic.
- The manifest binding note truthfully describes deterministic parameter IDs,
  or a test demonstrates why that identity is not guaranteed at this layer.

## Final composed-path re-baseline — 2026-08-20

The native-product measurements exposed the missing term in the original
ranking. For 256 cubed, isolated throughput preferred 128, but four 128 lanes
cannot occupy eight execution slots. The composed estimates were 6.707 ms
(32), 3.795 ms (64), 6.578 ms (128), and 26.993 ms (256), so the compiler now
selects 64 and all sixteen output lanes participate.

`workers` now means parked background workers for the CPU executors; the
caller drains the same atomic work queue. Thus an eight-core plan embeds seven
workers plus the caller, avoiding the prior nine-way oversubscription. Browser
workers retain their distinct no-caller accounting.

The canonical 1024-cubed run independently selects 128: projected composed
times were 429.261, 242.893, 210.490, and 215.940 ms. A forced 256 comparison
measured 227.02 ms versus the selected 128 product's 221.13 ms, validating the
ordering rather than merely restating it.
