"""Per-region execution-class assignment: the planner the region data promised.

Two places in this compiler explicitly reserved this decision without making
it.  ``SSADeploymentRegion``'s docstring (transmogrifier/ssa.py) calls itself
"a scheduling permission, not a selected GPU/API backend -- a later deployment
pass can bind the region to GLSL, CUDA, SIMD, threads, or keep the recorded
linear schedule".  ``site_bundle._region_target_capabilities`` publishes
per-region *capability* and says outright "there is no planner here deciding
which target a region should actually run on".  This module is that pass.

It answers, for each scheduled region (the ``FusedProgram`` subgraphs produced
by ``reduce_scheduled_shader_regions`` and captured per region index), one
question: what KIND of execution site should serve this region?

- ``graphics-output``: the region produces the named channels the published
  page's shader surface subscribes to (conventionally red/green/blue -- see
  ``site_bundle`` and ``wasm_html_shell``'s ``configuration.channels``).  It
  must land where the presentation can read it; routing it anywhere else
  orphans the picture.
- ``shader-compute``: a registered compute-stage machine target (webgpu, glsl)
  can express every operation in the region, so the region may be dispatched
  as a compute shader as-is.
- ``thread-workers``: the region is one lane of a proven-independent
  barrier-join deployment (``ControlDeploymentRegion`` /
  ``SSADeploymentRegion`` lanes) and its extent effect permits element
  splitting, so it can be handed to already-running workers (the
  ``ensureTileWorkers`` pool in ``wasm_html_shell``, or any host pool) as a
  job.
- ``host-linear``: none of the above -- keep the recorded linear schedule.

Classification is evidence, not fiat: every assignment and every veto carries
a reason string, and the full eligibility set is preserved alongside the
chosen class so a downstream dispatcher with different constraints (a client
with no WebGPU, a host with no worker pool) can fall back without
re-deriving anything.

Precedence when several classes are eligible: graphics-output first (the
surface subscription is a hard placement constraint), then shader-compute
over thread-workers -- mirroring the published page's existing surface
priority (WebGPU compute before WASM threads,
``site_bundle._SHADER_SURFACE_LANGUAGES``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .wasm_class_modules import fused_program_extent_effect


GRAPHICS_OUTPUT = "graphics-output"
SHADER_COMPUTE = "shader-compute"
THREAD_WORKERS = "thread-workers"
HOST_LINEAR = "host-linear"

EXECUTION_CLASSES = (
    GRAPHICS_OUTPUT, SHADER_COMPUTE, THREAD_WORKERS, HOST_LINEAR,
)

# The channel names a published page's presentation reads by default; a page
# with an authored configuration overrides this via ``presentation_channels``.
DEFAULT_PRESENTATION_CHANNELS = frozenset({"red", "green", "blue"})

# Extent effects that permit splitting one region across workers by element
# range.  Mirrors the runtime veto in wasm_html_shell's threadingEligible():
# a collective or global-state region evaluated once per tile turns one world
# into many worlds.
_SPLITTABLE_EXTENT_EFFECTS = frozenset({"pointwise"})


@dataclass(frozen=True)
class RegionExecutionClassification:
    """One region's chosen execution class, with its complete evidence."""

    region_index: int
    execution_class: str
    eligible: tuple[str, ...]
    extent_effect: str
    compute_shader_targets: tuple[str, ...]
    deployment_region_id: int | None
    lane_count: int
    presentation_outputs: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_class not in EXECUTION_CLASSES:
            raise ValueError(
                f"unknown execution class {self.execution_class!r}; "
                f"one of {EXECUTION_CLASSES}"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "execution_class": self.execution_class,
            "eligible": list(self.eligible),
            "extent_effect": self.extent_effect,
            "compute_shader_targets": list(self.compute_shader_targets),
            "deployment_region_id": self.deployment_region_id,
            "lane_count": self.lane_count,
            "presentation_outputs": list(self.presentation_outputs),
            "reasons": list(self.reasons),
        }


def _compute_shader_targets(region_program: Any) -> tuple[str, ...]:
    """Registered compute-stage targets that can express every region op.

    Answered from declared ``TargetCapabilities`` (stage + operation
    vocabulary), never by emitting.  Deprecated targets are excluded: a
    classification must not point at a target the default lookup would
    redirect away from.
    """

    from . import machine_targets

    program = getattr(region_program, "program", region_program)
    used = {str(step.op_name) for step in program.steps}
    return tuple(
        name
        for name, target in machine_targets.targets().items()
        if target.capabilities.stage == "compute"
        and not target.capabilities.deprecated
        and target.capabilities.supports(used)
    )


def _barrier_lane_memberships(
    deployment_regions: Sequence[Any],
) -> dict[int, tuple[int | None, int]]:
    """Map scheduled region index -> (deployment region_id, lane count).

    Only multi-lane barrier-join deployments count: one lane is not
    parallelism, and a non-barrier join (indexed/product/reduce) needs a
    dedicated consumer (see llvm_simd_deployment for the REDUCE case), not a
    generic worker pool.  Accepts both ``ControlDeploymentRegion`` and
    ``SSADeploymentRegion`` -- everything used here is common to both.
    """

    memberships: dict[int, tuple[int | None, int]] = {}
    for deployment in deployment_regions:
        join = getattr(deployment, "join", None)
        mode = getattr(getattr(join, "mode", None), "value", None)
        lanes = tuple(getattr(deployment, "lanes", ()) or ())
        if mode != "barrier" or len(lanes) < 2:
            continue
        region_id = getattr(deployment, "region_id", None)
        region_id = None if region_id is None else int(region_id)
        for lane in lanes:
            for region_index in getattr(lane, "region_indices", ()) or ():
                memberships[int(region_index)] = (region_id, len(lanes))
    return memberships


def classify_region_executions(
    region_programs: Mapping[int, Any],
    *,
    deployment_regions: Sequence[Any] = (),
    region_extent_effects: Mapping[int, str] | None = None,
    presentation_channels: frozenset[str] | set[str] | None = None,
) -> dict[int, RegionExecutionClassification]:
    """Assign an execution class to every scheduled region.

    ``region_programs`` maps region index -> captured region ``FusedProgram``
    (the same mapping ``_region_target_capabilities`` reads).
    ``deployment_regions`` is the region-bound parallelism proof --
    ``ControlProgram.deployment_regions`` after
    ``bind_control_deployments_to_regions``, or ``SSADeploymentRegion``
    entries from an ``IRModule.deployment_table``.  ``region_extent_effects``
    may be supplied when already computed; otherwise each region's effect is
    derived here with ``fused_program_extent_effect``.
    """

    channels = frozenset(
        str(name) for name in (
            DEFAULT_PRESENTATION_CHANNELS
            if presentation_channels is None
            else presentation_channels
        )
    )
    lane_memberships = _barrier_lane_memberships(deployment_regions)
    classifications: dict[int, RegionExecutionClassification] = {}
    for region_index, region_program in region_programs.items():
        region_index = int(region_index)
        program = getattr(region_program, "program", region_program)
        extent_effect = str(
            (region_extent_effects or {}).get(
                region_index, fused_program_extent_effect(program)
            )
        )
        presentation_outputs = tuple(sorted(
            name for name in (program.outputs or {}) if str(name) in channels
        ))
        compute_targets = _compute_shader_targets(program)
        deployment_region_id, lane_count = lane_memberships.get(
            region_index, (None, 0)
        )

        eligible: list[str] = []
        reasons: list[str] = []
        if presentation_outputs:
            eligible.append(GRAPHICS_OUTPUT)
            reasons.append(
                "produces presentation channel(s) "
                f"{', '.join(presentation_outputs)}: the shader surface "
                "subscribes to these outputs by name"
            )
        if compute_targets:
            eligible.append(SHADER_COMPUTE)
            reasons.append(
                "every operation is expressible by compute-stage target(s) "
                f"{', '.join(compute_targets)}"
            )
        else:
            reasons.append(
                "no registered compute-stage target expresses the region's "
                "full operation vocabulary"
            )
        if lane_count >= 2:
            if extent_effect in _SPLITTABLE_EXTENT_EFFECTS:
                eligible.append(THREAD_WORKERS)
                reasons.append(
                    f"lane of barrier deployment {deployment_region_id} "
                    f"({lane_count} independent lanes) with "
                    f"{extent_effect} extent effect"
                )
            else:
                reasons.append(
                    f"barrier deployment {deployment_region_id} lane vetoed "
                    f"for workers: {extent_effect} extent effect turns one "
                    "world into many if evaluated per tile"
                )
        if not eligible:
            reasons.append(
                "no placement evidence; keeping the recorded linear schedule"
            )

        chosen = next(
            (
                execution_class
                for execution_class in EXECUTION_CLASSES
                if execution_class in eligible
            ),
            HOST_LINEAR,
        )
        classifications[region_index] = RegionExecutionClassification(
            region_index=region_index,
            execution_class=chosen,
            eligible=tuple(eligible),
            extent_effect=extent_effect,
            compute_shader_targets=compute_targets,
            deployment_region_id=deployment_region_id,
            lane_count=lane_count,
            presentation_outputs=presentation_outputs,
            reasons=tuple(reasons),
        )
    return classifications


__all__ = [
    "DEFAULT_PRESENTATION_CHANNELS",
    "EXECUTION_CLASSES",
    "GRAPHICS_OUTPUT",
    "HOST_LINEAR",
    "SHADER_COMPUTE",
    "THREAD_WORKERS",
    "RegionExecutionClassification",
    "classify_region_executions",
]
