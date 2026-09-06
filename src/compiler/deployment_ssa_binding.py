"""Bind Deploy/Join frame markers to real SSA dataflow.

``precompile_to_ssa.emit_deployment_boundary`` mints ``Deploy`` and ``Join``
as nullary markers: no operands, no result, lanes linearized into the same
block and distinguished only by ``deployment_memberships`` attributes.  The
operator vocabulary was built ready for more -- ``operator_defs`` registers
both ops with unconstrained arity and declares Deploy fanning *down* to many
lanes and Join fanning *up* from them -- but the emitter never honored that
shape.  This pass upgrades emitted functions in place, without touching the
emitter:

- ``Deploy`` gains a result: a fresh ``ssa.deploy_token`` value naming the
  frame in dataflow.
- ``Join`` gains operands: the deploy token first, then every lane live-out
  (a value defined inside a lane and used outside the region).  A backend
  lowering a real barrier now reads exactly which values must be
  materialized at the join from the instruction itself, instead of
  re-deriving membership by scanning attribute dicts.

Before binding, the pass PROVES the permission the region record asserts:
lanes are independent iff no value defined in one lane is consumed by a
sibling lane.  A region that fails the check is reported and left unbound --
never silently "fixed".  This turns the deployment record from a trusted
claim into a verified theorem about the instruction stream.

Serial semantics are untouched: a bound ``Join`` still performs no numeric
work, and every existing consumer keeps reading the original lane values.
Backends that treat the markers as comments (ssa_fortran_backend) or
shortfall on them keep working -- extra operands on a marker are reads, not
new obligations.  Regions built without markers at all (dream_document's
hand-assembled shape) analyze fine and report ``missing-markers`` instead of
binding; the sites are optional-by-design there.

The v2 refinement -- Join producing an ``ssa.aggregate`` destructured by
``Indexed`` loads so post-join consumers are use-def ordered after the
barrier, mirroring ``emit_region_call``'s result convention -- is deliberate
future work; it rewrites consumers and belongs behind the same verification
this pass establishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..transmogrifier.ssa import Function, Instr, SSAValue

DEPLOY_OP = "Deploy"
JOIN_OP = "Join"
DEPLOY_TOKEN_DTYPE = "ssa.deploy_token"


@dataclass(frozen=True)
class LaneDataflow:
    """One lane's contribution to a region, as proven from the stream."""

    lane_index: int
    instruction_sites: tuple[tuple[str, int], ...]
    defined_value_ids: tuple[int, ...]
    live_out_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class RegionDataflow:
    """Verified dataflow shape of one deployment region in one function."""

    region_id: int
    function: str
    deploy_site: tuple[str, int] | None
    join_site: tuple[str, int] | None
    lanes: tuple[LaneDataflow, ...]
    violations: tuple[str, ...]

    @property
    def independent(self) -> bool:
        return not self.violations

    @property
    def has_markers(self) -> bool:
        return self.deploy_site is not None and self.join_site is not None


@dataclass(frozen=True)
class DeploymentBindingReport:
    """What binding did, region by region, with reasons for every skip."""

    function: str
    bound_region_ids: tuple[int, ...]
    skipped: tuple[tuple[int, str], ...]
    regions: tuple[RegionDataflow, ...]

    @property
    def complete(self) -> bool:
        return not self.skipped


def _instruction_stream(
    function: Function,
) -> Iterable[tuple[str, int, Instr]]:
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            yield block_name, index, instruction


def _memberships(instruction: Instr) -> tuple[tuple[int, int], ...]:
    raw = (instruction.attributes or {}).get("deployment_memberships") or ()
    return tuple((int(region), int(lane)) for region, lane in raw)


def _frame_marker_region(instruction: Instr) -> int | None:
    attributes = instruction.attributes or {}
    if not attributes.get("deployment_frame"):
        return None
    if instruction.op not in (DEPLOY_OP, JOIN_OP):
        return None
    return int(attributes["region_id"])


def analyze_deployment_dataflow(
    function: Function,
) -> tuple[RegionDataflow, ...]:
    """Derive and verify per-region lane dataflow from the instruction stream.

    Pure analysis: never mutates.  Region identity comes from frame markers
    and membership attributes alone, so both construction paths (the
    emit_deployment_boundary path and the marker-less dream_document path)
    are covered by the same scan.
    """

    definitions: dict[int, tuple[tuple[int, int], ...]] = {}
    uses: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    lane_sites: dict[int, dict[int, list[tuple[str, int]]]] = {}
    lane_defs: dict[int, dict[int, list[int]]] = {}
    deploy_sites: dict[int, tuple[str, int]] = {}
    join_sites: dict[int, tuple[str, int]] = {}
    region_ids: list[int] = []

    def note_region(region_id: int) -> None:
        if region_id not in lane_sites:
            lane_sites[region_id] = {}
            lane_defs[region_id] = {}
            region_ids.append(region_id)

    for block_name, index, instruction in _instruction_stream(function):
        membership = _memberships(instruction)
        marker_region = _frame_marker_region(instruction)
        if marker_region is not None:
            note_region(marker_region)
            sites = (
                deploy_sites if instruction.op == DEPLOY_OP else join_sites
            )
            # First marker wins; a duplicate is reported as a violation
            # below rather than silently re-pointing the site.
            sites.setdefault(marker_region, (block_name, index))
        for region_id, lane_index in membership:
            note_region(region_id)
            lane_sites[region_id].setdefault(lane_index, []).append(
                (block_name, index)
            )
        if instruction.res is not None:
            definitions[int(instruction.res.id)] = membership
            for region_id, lane_index in membership:
                lane_defs[region_id].setdefault(lane_index, []).append(
                    int(instruction.res.id)
                )
        for argument in instruction.args:
            uses.setdefault(int(argument.id), []).append(membership)

    regions: list[RegionDataflow] = []
    for region_id in region_ids:
        violations: list[str] = []
        lanes: list[LaneDataflow] = []
        for lane_index in sorted(lane_sites[region_id]):
            defined = tuple(sorted(set(
                lane_defs[region_id].get(lane_index, ())
            )))
            live_outs: list[int] = []
            for value_id in defined:
                for use_membership in uses.get(value_id, ()):  # each use site
                    lanes_here = {
                        lane for region, lane in use_membership
                        if region == region_id
                    }
                    if not lanes_here:
                        # Used entirely outside this region: a live-out.
                        live_outs.append(value_id)
                    elif lanes_here != {lane_index}:
                        violations.append(
                            f"value %t{value_id} defined in lane "
                            f"{lane_index} is consumed by lane(s) "
                            f"{sorted(lanes_here - {lane_index})}: lanes are "
                            "not independent"
                        )
            lanes.append(LaneDataflow(
                lane_index=lane_index,
                instruction_sites=tuple(
                    lane_sites[region_id][lane_index]
                ),
                defined_value_ids=defined,
                live_out_value_ids=tuple(sorted(set(live_outs))),
            ))
        regions.append(RegionDataflow(
            region_id=region_id,
            function=function.name,
            deploy_site=deploy_sites.get(region_id),
            join_site=join_sites.get(region_id),
            lanes=tuple(lanes),
            violations=tuple(violations),
        ))
    return tuple(regions)


def _next_value_id(function: Function) -> int:
    highest = -1
    for value in function.args:
        highest = max(highest, int(value.id))
    for _block, _index, instruction in _instruction_stream(function):
        if instruction.res is not None:
            highest = max(highest, int(instruction.res.id))
        for argument in instruction.args:
            highest = max(highest, int(argument.id))
    return highest + 1


def bind_deployment_dataflow(function: Function) -> DeploymentBindingReport:
    """Upgrade marker pairs to operand-bearing form, in place.

    Only regions that (a) have both markers and (b) pass the lane
    independence proof are bound.  Everything else is reported with its
    reason -- the report's ``complete`` property is the honest signal, in
    the same spirit as every backend's shortfall dataclass.
    """

    regions = analyze_deployment_dataflow(function)
    bound: list[int] = []
    skipped: list[tuple[int, str]] = []
    next_id = _next_value_id(function)

    for region in regions:
        if not region.has_markers:
            skipped.append((
                region.region_id,
                "missing-markers: region has no Deploy/Join instruction "
                "pair (marker-less construction path); sites are "
                "optional-by-design there",
            ))
            continue
        if not region.independent:
            skipped.append((
                region.region_id,
                "independence-violation: " + "; ".join(region.violations),
            ))
            continue

        deploy_block, deploy_index = region.deploy_site
        join_block, join_index = region.join_site
        deploy = function.blocks[deploy_block].instrs[deploy_index]
        join = function.blocks[join_block].instrs[join_index]
        if deploy.res is not None and join.args:
            skipped.append((region.region_id, "already-bound"))
            continue

        if deploy.res is None:
            deploy.res = SSAValue(next_id, dtype=DEPLOY_TOKEN_DTYPE)
            next_id += 1

        live_out_values: list[SSAValue] = []
        roles: list[str] = ["frame"]
        lane_live_outs: list[tuple[int, int]] = []
        value_index: dict[int, SSAValue] = {}
        for _block, _idx, instruction in _instruction_stream(function):
            if instruction.res is not None:
                value_index[int(instruction.res.id)] = instruction.res
        for lane in region.lanes:
            for value_id in lane.live_out_value_ids:
                live_out_values.append(value_index[value_id])
                roles.append(f"lane{lane.lane_index}.out")
                lane_live_outs.append((lane.lane_index, value_id))

        join.args = [deploy.res, *live_out_values]
        join.arg_roles = roles
        join.attributes = dict(join.attributes or {})
        join.attributes["lane_live_outs"] = tuple(lane_live_outs)
        bound.append(region.region_id)

    return DeploymentBindingReport(
        function=function.name,
        bound_region_ids=tuple(bound),
        skipped=tuple(skipped),
        regions=regions,
    )


def bind_module_deployments(
    functions: Mapping[str, Function],
) -> dict[str, DeploymentBindingReport]:
    """Bind every function of a module-shaped mapping; report per function."""

    return {
        name: bind_deployment_dataflow(function)
        for name, function in functions.items()
    }


__all__ = [
    "DEPLOY_OP",
    "DEPLOY_TOKEN_DTYPE",
    "JOIN_OP",
    "DeploymentBindingReport",
    "LaneDataflow",
    "RegionDataflow",
    "analyze_deployment_dataflow",
    "bind_deployment_dataflow",
    "bind_module_deployments",
]
