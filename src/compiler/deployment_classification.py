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
    compute_payoffs: Sequence["ComputePayoff"] = (),
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

    ``compute_payoffs`` (see ``webgpu_compute_payoff``) adds the second
    question -- not "can a compute target express this" but "does one
    dispatch pay": a region named by a payoff that does not pay is not offered
    ``shader-compute``.  Without payoffs, eligibility is capability alone, as
    before.
    """

    channels = frozenset(
        str(name) for name in (
            DEFAULT_PRESENTATION_CHANNELS
            if presentation_channels is None
            else presentation_channels
        )
    )
    lane_memberships = _barrier_lane_memberships(deployment_regions)
    unpaid_regions = {
        int(index): payoff
        for payoff in compute_payoffs if not payoff.pays
        for index in payoff.regions
    }
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
        unpaid = unpaid_regions.get(region_index)
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
        if compute_targets and unpaid is not None:
            reasons.append(
                "compute-stage target(s) could express it, but a dispatch "
                f"does not pay: {'; '.join(unpaid.reasons)}"
            )
        elif compute_targets:
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


# ---------------------------------------------------------------------------
# WebGPU compute payoff: a measured criterion, not a name match.
#
# ``shader-compute`` eligibility above answers "CAN a compute target express
# this region".  It says nothing about whether dispatching one PAYS: a
# dispatch costs a submit plus a readback round trip, so a region too small to
# amortize that is faster left on the host.  The functions below answer the
# second question from facts the compiler has already settled in the lowered
# SSA module -- planned-region feed shapes, result shapes, reduction axes --
# and from a declared device cost model.  The same facts are published with
# every verdict so a caller can re-decide under a measured model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebGPUComputeCostModel:
    """Declared device constants.  Defaults are conservative assumptions for
    a mid-range client GPU against single-threaded scalar host code; a page
    that has measured its own device passes the measured values instead."""

    host_ops_per_s: float = 2.0e8
    gpu_thread_ops_per_s: float = 1.0e9
    gpu_resident_threads: int = 1024
    dispatch_overhead_s: float = 2.0e-4
    transfer_bytes_per_s: float = 4.0e9
    shader_element_bytes: int = 4
    min_speedup: float = 2.0
    max_binding_bytes: int = 128 * 1024 * 1024


@dataclass(frozen=True)
class RegionComputeFacts:
    """What one planned region does, read off its lowered SSA."""

    region_index: int
    callee: str
    arithmetic_passes: int
    broadcast_moves: int
    reduction_passes: int
    lanes: int
    reduction_input_elements: int

    @property
    def element_ops(self) -> int:
        """Arithmetic element operations (broadcasts are index remaps on a
        shader, reductions touch every input element once)."""

        return (
            self.arithmetic_passes * self.lanes
            + self.reduction_input_elements
        )


@dataclass(frozen=True)
class ComputePayoff:
    """A WebGPU-compute verdict for a chain of device-resident regions."""

    pays: bool
    reasons: tuple[str, ...]
    regions: tuple[int, ...]
    element_ops: int
    data_parallel_extent: int
    invocations: int
    bytes_in: int
    bytes_out: int
    host_roundtrips: int
    host_seconds: float
    gpu_seconds: float
    speedup: float

    def as_record(self) -> dict[str, Any]:
        return {
            "pays": self.pays,
            "reasons": list(self.reasons),
            "regions": list(self.regions),
            "element_ops": self.element_ops,
            "data_parallel_extent": self.data_parallel_extent,
            "invocations": self.invocations,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "host_roundtrips": self.host_roundtrips,
            "host_seconds": self.host_seconds,
            "gpu_seconds": self.gpu_seconds,
            "speedup": self.speedup,
        }


def _elements(shape: Sequence[int] | None) -> int:
    count = 1
    for extent in tuple(shape or ()):
        count *= int(extent)
    return count


def region_compute_facts(module: Any, function_name: str) -> tuple[RegionComputeFacts, ...]:
    """Read per-region compute facts from a lowered repository-SSA module.

    ``function_name`` is the lowered entry; its planned-region ``Call``
    instructions carry ``region_index`` and ``feed_shapes``, and each callee
    is the region's body.  A body ``Call`` is classified by the shapes it
    moves, never by the kernel it names: a reduction declares its axis
    (``dim``/``axis``); an operand whose shape differs from the result's is a
    broadcast (a shader index remap); everything else is an arithmetic pass
    over the result extent.  Constant operands (shape descriptors, opcodes)
    are excluded -- they are not data.
    """

    entry = module.functions[function_name]
    facts: list[RegionComputeFacts] = []
    for block in entry.blocks.values():
        for instruction in block.instrs:
            attributes = instruction.attributes
            if str(instruction.op) not in {"Call", "call"}:
                continue
            if "region_index" not in attributes or "feed_shapes" not in attributes:
                continue
            body = module.functions.get(str(attributes.get("callee", "")))
            if body is None:
                continue
            constants = {
                int(item.res.id)
                for body_block in body.blocks.values()
                for item in body_block.instrs
                if str(item.op) in {"Const", "const"} and item.res is not None
            }
            arithmetic = broadcast = reduction = lanes = reduced_in = 0
            for body_block in body.blocks.values():
                for item in body_block.instrs:
                    if str(item.op) not in {"Call", "call"} or item.res is None:
                        continue
                    result_shape = tuple(getattr(item.res, "shape", ()) or ())
                    result_elements = _elements(result_shape)
                    tensor_shapes = [
                        tuple(getattr(arg, "shape", ()) or ())
                        for arg in item.args
                        if int(arg.id) not in constants
                        and _elements(getattr(arg, "shape", ())) > 1
                    ]
                    if "dim" in item.attributes or "axis" in item.attributes:
                        reduction += 1
                        reduced_in += max(
                            (_elements(shape) for shape in tensor_shapes), default=0,
                        )
                        lanes = max(lanes, result_elements)
                    elif result_elements > 1:
                        if any(shape != result_shape for shape in tensor_shapes):
                            broadcast += 1
                        else:
                            arithmetic += 1
                        lanes = max(lanes, result_elements)
            facts.append(RegionComputeFacts(
                int(attributes["region_index"]), str(attributes["callee"]),
                arithmetic, broadcast, reduction, lanes, reduced_in,
            ))
    return tuple(facts)


def webgpu_compute_payoff(
    module: Any,
    function_name: str,
    *,
    model: WebGPUComputeCostModel = WebGPUComputeCostModel(),
) -> ComputePayoff:
    """Whether dispatching a function's tensor regions as ONE compute chain
    beats leaving them on the host.

    Criterion (every term is a declared measurable):

    - *data-parallel extent*: the largest region result, in elements;
    - *arithmetic intensity*: element operations over the bytes that must
      cross the host boundary (function formals in, returned values out --
      region-to-region intermediates stay device resident and cost nothing);
    - *no host round trip*: the chain is only offered when every region is
      a tensor region (arithmetic, broadcast or reduction); a region with no
      tensor extent between them would force a readback;
    - *buffer size*: each boundary buffer must fit one binding.

    Pays when ``host_seconds >= min_speedup * gpu_seconds`` where the GPU
    time is one dispatch overhead, the boundary transfer, and the longest
    invocation's work (one invocation per output element of the chain).
    """

    entry = module.functions[function_name]
    facts = region_compute_facts(module, function_name)
    tensor_regions = tuple(fact for fact in facts if fact.lanes > 1)
    returned = [
        arg
        for block in entry.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in {"Ret", "Return", "ret", "return"}
        for arg in instruction.args
    ]
    element = int(model.shader_element_bytes)
    bytes_in = sum(_elements(getattr(arg, "shape", ())) for arg in entry.args) * element
    bytes_out = sum(_elements(getattr(arg, "shape", ())) for arg in returned) * element
    element_ops = sum(fact.element_ops for fact in tensor_regions)
    extent = max((fact.lanes for fact in tensor_regions), default=0)
    invocations = max(1, sum(_elements(getattr(arg, "shape", ())) for arg in returned))
    reasons: list[str] = []
    vetoes: list[str] = []
    if not tensor_regions:
        vetoes.append("no region has a data-parallel extent above one element")
    scalar_regions = [fact.region_index for fact in facts if fact.lanes <= 1]
    host_roundtrips = 0
    biggest = max(
        [_elements(getattr(arg, "shape", ())) * element for arg in (*entry.args, *returned)],
        default=0,
    )
    if biggest > model.max_binding_bytes:
        vetoes.append(
            f"a boundary buffer of {biggest} bytes exceeds the "
            f"{model.max_binding_bytes}-byte binding limit"
        )
    host_seconds = element_ops / model.host_ops_per_s
    waves = -(-invocations // int(model.gpu_resident_threads))
    gpu_seconds = (
        model.dispatch_overhead_s
        + (bytes_in + bytes_out) / model.transfer_bytes_per_s
        + waves * (element_ops / invocations) / model.gpu_thread_ops_per_s
    )
    speedup = host_seconds / gpu_seconds if gpu_seconds > 0 else 0.0
    reasons.append(
        f"{len(tensor_regions)} tensor region(s), extent {extent}, "
        f"{element_ops} element ops over {invocations} invocation(s); "
        f"{bytes_in} B in, {bytes_out} B out, "
        f"{len(scalar_regions)} extent-free region(s) stay on the host"
    )
    reasons.append(
        f"host {host_seconds * 1e6:.1f} us vs device {gpu_seconds * 1e6:.1f} us "
        f"(speedup {speedup:.2f}x, needs {model.min_speedup:.2f}x)"
    )
    pays = not vetoes and speedup >= model.min_speedup
    if not pays and not vetoes:
        reasons.append("too little work to amortize one dispatch and readback")
    return ComputePayoff(
        pays, tuple((*reasons, *vetoes)),
        tuple(fact.region_index for fact in tensor_regions),
        element_ops, extent, invocations, bytes_in, bytes_out, host_roundtrips,
        host_seconds, gpu_seconds, speedup,
    )


__all__ = [
    "ComputePayoff",
    "RegionComputeFacts",
    "WebGPUComputeCostModel",
    "region_compute_facts",
    "webgpu_compute_payoff",
    "DEFAULT_PRESENTATION_CHANNELS",
    "EXECUTION_CLASSES",
    "GRAPHICS_OUTPUT",
    "HOST_LINEAR",
    "SHADER_COMPUTE",
    "THREAD_WORKERS",
    "RegionExecutionClassification",
    "classify_region_executions",
]
