# The kernel bank and launch coordination

**Date:** 2026-08-20. **Code:** `src/compiler/kernel_bank.py`,
`tools/kernel_bank_probe.py`. **Continues:** `docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`
(the worked single case), `docs/BLAS_VS_NUMPY_PROFILE.md` (the measurements
that make routing worth having), `src/common/tensors/blas.py` (the first
registered kernels).

**For:** whoever works on this next, and specifically the
progressive-region-replacement effort (gradually replacing program regions
with compiled versions until the whole program may be replaced) — section 5
is your interface.

## 1. What it is

One object, `KernelBank`, that takes compiling out of user hands:

* **Variants across a parameter matrix.** A variant = (kernel, work
  contract, optional size specialization). `bank.get(name,
  contract=..., specialized=...)` compiles on miss; a matrix can be
  prebuilt (`tools/kernel_bank_probe.py --matrix`).
* **Contained filesystem impact.** Everything lives under one root
  (default `build/kernel_bank/`): `<root>/<kernel>/<variant-key>/` holding
  the `.ll`, the built library, a pickled `(module, outputs)` for fast
  re-materialization, and a `manifest.json` (schema
  `turing.kernel-bank.v1`) that makes each directory self-describing.
  `bank.inventory()` is a scan of manifests, not a central index that can
  go stale.
* **Verified admission.** No artifact serves a call until its output
  matches the kernel's own Python reference on seeded probe inputs.
  Failures are recorded in the manifest (`verification.admitted: false`
  plus the reason) and the variant is never selected. Demonstrated live
  the day this was built: `gemm` specialized to literal sizes was refused
  at admission with a recorded emission shortfall (see section 4), while
  every parametric build and the `dot` specialization admitted at
  0.0e+00 error.
* **Per-call launch coordination.** `LaunchCoordinator.launch(name,
  **arguments)`: derive sizes from the call, select the matching variant
  (exact-size specialized when admitted, else parametric; optionally
  trigger a verified specialized build for next time), execute, fall back
  to the Python reference when nothing compiled serves, and append the
  decision to `routing_log.jsonl`. Observed live: the same `dot n=512`
  call routed `parametric` before a specialized build existed and
  `specialized` immediately after one was admitted — no caller change.

**Launches, not batches.** Each launch resolves and executes immediately;
there is no queueing, grouping, or deferred scheduling anywhere in this
module. Cross-call scheduling/placement is the compiler's deployment
machinery (`deployment_classification` / Deploy-Join / `turing_pool`).

## 2. Identity and staleness

A variant key hashes: kernel name, source SHA-256, contract name,
specialization dict, and a **compiler fingerprint** (newest mtime over
`src/compiler/**/*.py` — the same authority
`symbolic_fluid_native_runtime._cache_is_stale` uses). Consequences:

* Editing any compiler source invalidates every key; stale artifacts are
  simply never matched again and rebuild on next use. (Observed: edits to
  `kernel_bank.py` itself churned the bank mid-session — correct, if
  noisy. The stale directories are inert and can be deleted freely.)
* SSA value ids are NEVER trusted from a manifest — they are unstable
  across lowerings. Binding tables (`parameter_names`, `named_outputs`)
  are re-derived from the loaded module every materialization. The
  manifest's `binding` block is informational only.

## 3. Contract tie-ins

* Variants compile under `set_active_contract(contract)`; the contract
  name is part of the variant identity, so `develop` and `fast` builds of
  one kernel coexist.
* The size specializer is this tree's **first argument-baking
  specializer**, so it implements the promised first test of the work
  contract's `symbolic_arguments` veto ("every future specializer must
  treat this list as a veto"): baking a vetoed name raises `BankRefusal`
  before any compilation. The contract's own `constant_arguments` axis
  still refuses non-empty lists; when that axis is wired for real, this
  source-level specializer should migrate onto it and this document
  updated.

## 4. Defect pinned by the admission gate (open)

Specializing `gemm` to literal sizes (`m=8, n=8, k=8` via
signature-drop + prologue assignment) produces a LOUD emission shortfall:

    call argument position(s) [3] feed
    kb_gemm_<key>__gemm__planned_region_0, whose body unpacks that
    parameter ...

i.e. with sizes baked, a planner region's call feeds a parameter its body
then unpacks — the literal-bound defect family
(`FUNCTION_TO_DEPLOYMENT_HANDOFF.md` section 4.2's neighborhood), but as a
loud refusal rather than 4.2's silent dead-store. `dot` specializes and
admits cleanly, so the trigger involves the nested-loop shape. Repro is
one line: `open_blas_bank(...).get("gemm", specialized={"m": 8, "n": 8,
"k": 8})`. Until fixed, the bank behaves exactly as designed: refusal
recorded, launches route to the parametric build, nobody gets a wrong
number.

## 5. Interface for progressive region replacement

The bank was shaped so that effort plugs in rather than rebuilding:

1. **A region registers as a `KernelSpec`** — exactly what a BLAS kernel
   is: `(name, authored source string, entrypoint function name, Python
   reference callable, parameter_order, size_parameters,
   example_inputs(sizes, rng))`. The reference callable for a region is
   the region's original Python — which the auto-port/materializer arc
   can produce mechanically. Registration is just a dict passed to
   `KernelBank(root, specs)`; `blas_kernel_specs()` is the worked example.
2. **Admission is the safety property.** A region variant that fails its
   own original-code oracle is recorded and never routed to — so
   progressive replacement can be aggressive about ATTEMPTING regions
   (even ones expected to hit open compiler defects) without ever
   shipping a wrong result. The refused manifests double as a defect
   worklist for compiler sessions.
3. **`routing_log.jsonl` is the evidence stream** — which regions
   actually run compiled, at which sizes, how often the reference
   fallback fires. That is the prioritization signal for which regions to
   fix/specialize next.
4. **Stable surfaces** (treat as the interface; version-bump
   `MANIFEST_SCHEMA` on change): the manifest schema, `KernelSpec`
   fields, `BankRefusal` semantics, `bank.get/select/inventory`,
   `LaunchCoordinator.launch`.

## 6. Open questions / next steps

* The `gemm` specialization shortfall (section 4) — first compiler defect
  on this track's worklist, with the dead-store defect (4.2) behind it.
* Tensor-op-authored kernel sources: possible in principle, currently
  gated on shape propagation for tensor parameters
  (`FUNCTION_TO_DEPLOYMENT_HANDOFF.md` section 4.5, measured) — scalar
  loop sources remain the right authoring style until that lands.
* Routing policy is deliberately minimal (specialized > parametric >
  reference). The profile data (`BLAS_VS_NUMPY_PROFILE.md`) supports
  size-threshold policies later; admission-time
  `verification.probe_call_seconds` is already recorded per variant as a
  seed for that.
* Cross-process artifact reuse currently re-materializes via the pickled
  module + re-emit (subsecond for BLAS-scale kernels). If region-scale
  modules make that cost real, extend the manifest with what
  `prepare_artifact_execution` needs so the built library can be reloaded
  without re-emission.
