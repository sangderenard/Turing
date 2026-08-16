"""The default deployment-planning stage of a bundle build.

Composes the layer built underneath it -- region execution classification,
per-backend strategy selection, and persisted calibration verdicts -- into
one pass that runs on every build: for each captured region, for each
backend of interest, decide serial/pool/dispatch with the evidence
attached.  The plan is published in the bundle manifest and consulted by
the shells that already have executors.

Default-on is safe by one rule, applied twice:

- Absence of evidence changes nothing.  With no calibration verdicts the
  choices reduce to the static class preferences, which reproduce today's
  behavior exactly (thread-eligible work pools, GPU-shaped work
  dispatches, everything else stays serial).
- Only explicit measurement closes a gate.  ``browser_threading_veto``
  returns a reason iff a machine-local verdict demoted every legal wasm
  pool -- an absent, stale, or foreign verdict keeps the gate open, so a
  bundle built on a fresh machine behaves exactly as before this stage
  existed.

Calibration lookups here are read-only (probing at build time would time
the build machine's Python, not the deployed runtime); shells that execute
frames are the ones that probe, through ``calibrated_verdict``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .deployment_calibration import CalibrationStore, WorkloadSignature
from .deployment_classification import (
    RegionExecutionClassification,
    classify_region_executions,
)
from .deployment_lowering import (
    DeploymentStrategyChoice,
    select_deployment_strategy,
)

# The backends a bundle build plans for by default: the shells that read
# the manifest today plus the native tier the pool runtime serves.
DEFAULT_PLANNED_BACKENDS = ("wasm", "webgpu", "c", "llvm", "fortran")


def region_workload_signature(
    backend: str, region_program: Any,
) -> WorkloadSignature:
    """Stable calibration identity for one region on one backend.

    Identity hashes the operation stream, not the region index -- a
    renumbered but identical region keeps its verdict; a changed program
    re-probes.
    """

    program = getattr(region_program, "program", region_program)
    steps = list(program.steps)
    digest = hashlib.sha256()
    for step in steps:
        digest.update(str(step.op_name).encode("utf-8"))
        digest.update(b"\0")
    return WorkloadSignature(
        backend=str(backend),
        kind="region",
        work=len(steps),
        identity=digest.hexdigest()[:24],
    )


@dataclass(frozen=True)
class RegionDeploymentDecision:
    """One region's full decision row: class plus per-backend choices."""

    region_index: int
    classification: RegionExecutionClassification
    choices: tuple[DeploymentStrategyChoice, ...]

    def choice_for(self, backend: str) -> DeploymentStrategyChoice | None:
        for choice in self.choices:
            if choice.backend == backend:
                return choice
        return None

    def as_record(self) -> dict[str, Any]:
        return {
            "execution_class": self.classification.execution_class,
            "strategies": {
                choice.backend: {
                    "strategy": choice.strategy,
                    "workers": choice.workers,
                    "calibration_demoted": choice.calibration_demoted,
                    "reasons": list(choice.reasons),
                }
                for choice in self.choices
            },
        }


@dataclass(frozen=True)
class RegionDeploymentPlan:
    decisions: tuple[RegionDeploymentDecision, ...]

    def decision_for(self, region_index: int) -> RegionDeploymentDecision | None:
        for decision in self.decisions:
            if decision.region_index == int(region_index):
                return decision
        return None

    def classifications(self) -> dict[int, RegionExecutionClassification]:
        return {
            decision.region_index: decision.classification
            for decision in self.decisions
        }

    def as_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            str(decision.region_index): decision.as_record()
            for decision in self.decisions
        }


def plan_region_deployments(
    region_programs: Mapping[int, Any],
    *,
    deployment_regions: Sequence[Any] = (),
    backends: Sequence[str] = DEFAULT_PLANNED_BACKENDS,
    presentation_channels=None,
    calibration_store: CalibrationStore | None = None,
) -> RegionDeploymentPlan:
    """Classify every region and choose a strategy per backend.

    ``calibration_store`` is optional; when given, verdicts refine choices
    (lookups only -- see module docstring).  A store read that fails for
    any environmental reason degrades to no-verdict rather than failing
    the build.
    """

    classifications = classify_region_executions(
        region_programs,
        deployment_regions=deployment_regions,
        presentation_channels=presentation_channels,
    )
    decisions: list[RegionDeploymentDecision] = []
    for region_index, classification in sorted(classifications.items()):
        region_program = region_programs[region_index]
        choices: list[DeploymentStrategyChoice] = []
        for backend in backends:
            verdict = None
            if calibration_store is not None:
                try:
                    verdict = calibration_store.lookup(
                        region_workload_signature(backend, region_program)
                    )
                except OSError:
                    verdict = None
            choices.append(select_deployment_strategy(
                backend=backend,
                execution_class=classification.execution_class,
                join_mode="barrier",
                calibration=verdict,
            ))
        decisions.append(RegionDeploymentDecision(
            region_index=int(region_index),
            classification=classification,
            choices=tuple(choices),
        ))
    return RegionDeploymentPlan(decisions=tuple(decisions))


def browser_threading_veto(plan: RegionDeploymentPlan) -> str | None:
    """Reason to withhold the browser thread plan, or ``None`` to keep it.

    Conservative by construction: the veto fires only when measurement --
    not a capability gap, not missing evidence -- demoted every legal wasm
    pool.  If any region still chooses the pool, or no calibration verdict
    exists at all, the gate stays open and the runtime's own eligibility
    checks remain the deciders, exactly as before this stage existed.
    """

    wasm_choices = [
        choice
        for decision in plan.decisions
        for choice in decision.choices
        if choice.backend == "wasm"
    ]
    if not wasm_choices:
        return None
    if any(choice.strategy == "pool" for choice in wasm_choices):
        return None
    demoted = [
        choice for choice in wasm_choices if choice.calibration_demoted
    ]
    if not demoted:
        return None
    return (
        "calibration demoted every eligible wasm pool to serial "
        f"({len(demoted)} measured region(s)); withholding the browser "
        "thread plan"
    )


__all__ = [
    "DEFAULT_PLANNED_BACKENDS",
    "RegionDeploymentDecision",
    "RegionDeploymentPlan",
    "browser_threading_veto",
    "plan_region_deployments",
    "region_workload_signature",
]
