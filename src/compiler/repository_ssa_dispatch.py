"""Plan repository-SSA deployment frames for real native executors.

The repository SSA stream is always a valid linear schedule.  This module is
the generic consumer of the parallel permission retained beside it in
``IRModule.deployment_table``.  It proves lane independence from the emitted
instructions, follows each lane's complete internal-call closure, and reports
which frames a native backend may deploy.  An internal call is a deployable
lane boundary, not a reason to serialize; a single-lane
``independent_iterations`` region is a loop template whose lane must first be
outlined (``deployment_outlining.py``) into a callable closure.

Planning is PURE and execution is NATIVE.  There is deliberately no Python
runtime dispatcher here: the executing pool is ``turing_pool.c``, linked into
the compiled artifact by ``ssa_c_backend`` when an outlined region emits a
``turing_pool_deploy_span`` (see ``CModuleArtifact.pool_required``).  The
former ``RepositorySSAFrameExecutor`` Python-pool path was removed as a
runtime lane outside the compiled product's ethos.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deployment_frame import DeploymentJoin
from .deployment_lowering import DeploymentStrategyChoice, select_deployment_strategy
from .deployment_ssa_binding import RegionDataflow, analyze_deployment_dataflow


@dataclass(frozen=True)
class SSALaneDispatch:
    lane_index: int
    instruction_sites: tuple[tuple[str, int], ...]
    roots: tuple[str, ...]
    closure: tuple[str, ...]


@dataclass(frozen=True)
class SSAFrameDispatch:
    function: str
    region_id: int
    lanes: tuple[SSALaneDispatch, ...]
    choice: DeploymentStrategyChoice
    join: DeploymentJoin
    launchable: bool
    shortfalls: tuple[str, ...] = ()

    @property
    def parallel(self) -> bool:
        return self.launchable and self.choice.parallel


@dataclass(frozen=True)
class RepositorySSADispatchPlan:
    backend: str
    frames: tuple[SSAFrameDispatch, ...]

    @property
    def parallel_frames(self) -> tuple[SSAFrameDispatch, ...]:
        return tuple(frame for frame in self.frames if frame.parallel)

    @property
    def complete(self) -> bool:
        return all(not frame.shortfalls for frame in self.frames)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "frames": [
                {
                    "function": frame.function,
                    "region_id": frame.region_id,
                    "launchable": frame.launchable,
                    "parallel": frame.parallel,
                    "strategy": frame.choice.as_record(),
                    "lanes": [
                        {
                            "lane_index": lane.lane_index,
                            "instruction_sites": [list(site) for site in lane.instruction_sites],
                            "roots": list(lane.roots),
                            "closure": list(lane.closure),
                        }
                        for lane in frame.lanes
                    ],
                    "shortfalls": list(frame.shortfalls),
                }
                for frame in self.frames
            ],
        }


@dataclass(frozen=True)
class RepositorySSADeploymentBuild:
    directory: Path
    root_artifact: Any
    plan: RepositorySSADispatchPlan
    closure_artifacts: Mapping[str, Any]
    manifest_path: Path


def compile_repository_ssa_deployment(
    module: Any,
    root_name: str,
    directory: str | Path,
    *,
    entry_name: str | None = None,
    optimization: str = "O3",
    cores: int | None = None,
) -> RepositorySSADeploymentBuild:
    """Compile an LLVM root plus every pool-selected lane call closure.

    Closure artifacts are deduplicated by root function.  A closure that the
    LLVM emitter refuses remains a manifest shortfall and is never presented
    as launchable.  This gives deployment builds one auditable product rather
    than compiling a serial root and discarding its deployment table.
    """

    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan = plan_repository_ssa_dispatch(module, backend="llvm", cores=cores)
    root_artifact = emit_ssa_function_to_llvm(
        module, str(root_name), entry_name=entry_name or str(root_name),
    )
    compile_artifact(
        root_artifact, directory=destination / "root", optimization=optimization,
    )
    closure_artifacts: dict[str, Any] = {}
    closure_records: dict[str, Any] = {}
    roots = tuple(dict.fromkeys(
        root
        for frame in plan.parallel_frames
        for lane in frame.lanes
        for root in lane.roots
    ))
    for root in roots:
        artifact = emit_ssa_function_to_llvm(
            module, root, entry_name=f"deploy_{len(closure_artifacts)}",
        )
        closure_artifacts[root] = artifact
        if artifact.complete:
            compile_artifact(
                artifact,
                directory=destination / "closures" / str(len(closure_artifacts) - 1),
                optimization=optimization,
            )
        closure_records[root] = {
            "complete": bool(artifact.complete),
            "library": (
                None if artifact.library_path is None
                else str(artifact.library_path)
            ),
            "shortfalls": [
                {
                    "function": str(item.function),
                    "operation": str(item.operation),
                    "reason": str(item.reason),
                }
                for item in artifact.shortfalls
            ],
        }
    manifest = {
        "schema": "turing.repository-ssa-deployment-build.v1",
        "root": str(root_name),
        "root_library": str(root_artifact.library_path),
        "optimization": str(optimization),
        "deployment": plan.as_manifest(),
        "closures": closure_records,
    }
    manifest_path = destination / "deployment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return RepositorySSADeploymentBuild(
        destination, root_artifact, plan, closure_artifacts, manifest_path,
    )


def _internal_closure(module: Any, roots: Sequence[str]) -> tuple[str, ...]:
    wanted: set[str] = set()
    for root in roots:
        if str(root) not in module.functions:
            continue
        wanted.update(module.reachable_functions(str(root)))
    return tuple(name for name in module.functions if name in wanted)


def _dataflow_by_region(function: Any) -> dict[int, RegionDataflow]:
    return {
        int(region.region_id): region
        for region in analyze_deployment_dataflow(function)
    }


def plan_repository_ssa_dispatch(
    module: Any,
    *,
    backend: str = "llvm",
    cores: int | None = None,
) -> RepositorySSADispatchPlan:
    """Consume every repository deployment record and choose its executor.

    A lane may contain a trivial internal-call closure of any depth.  The
    closure is followed in module order and retained on the manifest.  A lane
    with no callable root is not silently called parallel: arithmetic
    instruction outlining is a different lowering and remains a named
    shortfall.
    """

    if cores is None:
        cores = os.cpu_count() or 1
    frames: list[SSAFrameDispatch] = []
    for function_name, regions in module.deployment_table.items():
        function = module.functions.get(str(function_name))
        if function is None:
            continue
        analyzed = _dataflow_by_region(function)
        for region in regions:
            evidence = analyzed.get(int(region.region_id))
            shortfalls: list[str] = []
            if evidence is None:
                shortfalls.append("deployment record has no emitted lane memberships")
            elif not evidence.independent:
                shortfalls.extend(evidence.violations)
            iteration_region = (
                str(getattr(region, "schedule", "")) == "independent_iterations"
                and region.iteration_space is not None
            )
            outlined = (
                (module.metadata or {}).get("deployment_outlines", {})
                .get((str(function_name), int(region.region_id)))
                is not None
            )
            lanes: list[SSALaneDispatch] = []
            for lane in region.lanes:
                roots = tuple(
                    name for name in dict.fromkeys(map(str, lane.callees))
                    if name in module.functions
                )
                closure = _internal_closure(module, roots)
                if not roots:
                    shortfalls.append(
                        f"lane {int(lane.index)} has no internal-call root; "
                        "run outline_independent_iteration_lanes first"
                        if iteration_region else
                        f"lane {int(lane.index)} has no internal-call root; "
                        "SSA instruction outlining is required"
                    )
                lanes.append(SSALaneDispatch(
                    lane_index=int(lane.index),
                    instruction_sites=tuple(lane.instruction_sites),
                    roots=roots,
                    closure=closure,
                ))
            join = getattr(getattr(region, "join", None), "mode", None)
            join_mode = str(getattr(join, "value", join or "barrier"))
            if iteration_region:
                stop = str(region.iteration_space[1])
                work = int(stop) if stop.isdigit() else int(cores)
            else:
                work = len(lanes)
            choice = select_deployment_strategy(
                backend=str(backend),
                execution_class="thread-workers",
                join_mode=join_mode,
                work=max(2, work) if iteration_region else work,
                cores=int(cores),
            )
            launchable = (
                not shortfalls
                and choice.parallel
                and (
                    len(lanes) >= 2
                    or (iteration_region and outlined and bool(lanes))
                )
            )
            frames.append(SSAFrameDispatch(
                function=str(function_name),
                region_id=int(region.region_id),
                lanes=tuple(lanes),
                choice=choice,
                join=region.join,
                launchable=launchable,
                shortfalls=tuple(dict.fromkeys(shortfalls)),
            ))
    return RepositorySSADispatchPlan(str(backend), tuple(frames))


__all__ = [
    "RepositorySSADispatchPlan",
    "RepositorySSADeploymentBuild",
    "SSAFrameDispatch",
    "SSALaneDispatch",
    "plan_repository_ssa_dispatch",
    "compile_repository_ssa_deployment",
]
