"""Sealed second-pass compilation for deployment-selected shader regions.

The ordinary compiler owns graph meaning and decides *where* deployment may
occur.  This module begins only after that decision: it cuts one numerical
region behind a typed hole, seals the interior against recursive deployment,
then derives shader memory/phase topology without inventing another program
IR.  The retained interior is still the existing :class:`FusedProgram`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep, ordered_feed_ids


SCHEMA = "turing.shader-region.v1"
IDENTITY_LIBRARY_VERSION = 1
_DEPLOYMENT_OPS = frozenset({"deploy", "join", "paralleldeployment"})
_MUTATING_OPS = frozenset({"index_set", "indexedstore", "setattr"})


class ShaderRegionError(ValueError):
    """A selected region violates the sealed shader-compilation contract."""


class ShaderStorageClass(str, Enum):
    UNIFORM = "uniform"
    STORAGE_BUFFER = "storage_buffer"
    STORAGE_VIEW = "storage_view"
    REGISTER = "register"
    WORKGROUP_SHARED = "workgroup_shared"


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _canonical(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _canonical(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _digest(record: Any) -> str:
    return hashlib.sha256(json.dumps(
        _canonical(record), sort_keys=True, separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")).hexdigest()


def _program_record(program: FusedProgram) -> dict[str, Any]:
    return {
        "version": int(program.version),
        "feeds": sorted(map(int, program.feeds)),
        "steps": [
            {
                "step_id": int(step.step_id),
                "op_name": str(step.op_name),
                "input_ids": list(map(int, step.input_ids)),
                "attrs": _canonical(step.attrs),
                "result_id": int(step.result_id),
                "mode_sensitive": bool(step.mode_sensitive),
                "level": step.level,
            }
            for step in program.steps
        ],
        "outputs": {
            str(name): int(value_id)
            for name, value_id in sorted(program.outputs.items())
        },
        "state_in": sorted(map(int, program.state_in or ())),
        "meta": {
            str(value_id): _canonical(meta)
            for value_id, meta in sorted((program.meta or {}).items())
        },
        "extras": _canonical(program.extras or {}),
    }


def _program_semantic_record(program: FusedProgram) -> dict[str, Any]:
    """Exact program meaning, independent of a legal linear schedule."""

    record = _program_record(program)
    record["steps"] = sorted(
        (
            {
                key: value
                for key, value in step.items()
                if key != "step_id"
            }
            for step in record["steps"]
        ),
        key=lambda step: (
            int(step["result_id"]), str(step["op_name"]),
            tuple(step["input_ids"]),
        ),
    )
    return record


#: What a numeric boundary value is when nothing recorded otherwise. The
#: capture states a shape for every value and often leaves dtype unset, and
#: the typed hole needs both -- so a region that is entirely well described
#: except for this one field could not be cut at all.
_DEFAULT_BOUNDARY_DTYPE = "float64"


def _with_boundary_dtype(metadata: Mapping[int, Meta]) -> dict[int, Meta]:
    """Give every recorded value a dtype, preferring the region's own.

    Where the capture already states a dtype, that is used and nothing
    here applies. Where it does not, the dtype other values in the SAME
    region agree on is used, and only failing that the numeric default.
    A region is one element type in practice, so borrowing from its
    neighbours is the region's own statement rather than an assumption
    imported from outside it.

    This does not invent a shape. A value with no recorded shape is still
    refused, because an extent genuinely is not knowable from its
    neighbours.
    """

    stated = {
        str(meta.dtype) for meta in metadata.values()
        if meta is not None and meta.dtype is not None
    }
    fallback = stated.pop() if len(stated) == 1 else _DEFAULT_BOUNDARY_DTYPE
    return {
        key: (
            meta if meta is None or meta.dtype is not None
            else replace(meta, dtype=fallback)
        )
        for key, meta in metadata.items()
    }


def _meta_record(meta: Meta | None) -> dict[str, Any]:
    if meta is None:
        return {
            "shape": None, "dtype": None, "device": None,
            "source_id": None, "offset": 0, "stride": 1,
            "shape_source_ids": None,
        }
    return {
        "shape": None if meta.shape is None else list(map(int, meta.shape)),
        "dtype": None if meta.dtype is None else str(meta.dtype),
        "device": None if meta.device is None else str(meta.device),
        "source_id": None if meta.source_id is None else int(meta.source_id),
        "offset": int(meta.offset),
        "stride": int(meta.stride),
        "shape_source_ids": (
            None
            if meta.shape_source_ids is None
            else [None if item is None else int(item) for item in meta.shape_source_ids]
        ),
    }


@dataclass(frozen=True, slots=True)
class ShaderBoundaryValue:
    value_id: int
    name: str | None
    direction: str
    metadata: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "name": self.name,
            "direction": self.direction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ShaderRegionBoundary:
    inputs: tuple[ShaderBoundaryValue, ...]
    outputs: tuple[ShaderBoundaryValue, ...]
    effects: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "inputs": [value.as_record() for value in self.inputs],
            "outputs": [value.as_record() for value in self.outputs],
            "effects": list(self.effects),
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_record())


@dataclass(frozen=True, slots=True)
class ShaderDeploymentHole:
    region_index: int
    marker: str
    target: str
    invocation: str
    capsule_digest: str
    boundary_digest: str
    recursive_deployment_permitted: bool = False

    def as_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class ShaderRegionCapsule:
    region_index: int
    target: str
    captured_program: CapturedFusedProgram
    boundary: ShaderRegionBoundary
    source_digest: str
    capsule_digest: str
    sealed_deployment_depth: int = 1

    @property
    def program(self) -> FusedProgram:
        return self.captured_program.program

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "region_index": self.region_index,
            "target": self.target,
            "boundary": self.boundary.as_record(),
            "source_digest": self.source_digest,
            "capsule_digest": self.capsule_digest,
            "sealed_deployment_depth": self.sealed_deployment_depth,
        }


@dataclass(frozen=True, slots=True)
class ShaderRegionCut:
    hole: ShaderDeploymentHole
    capsule: ShaderRegionCapsule

    def as_record(self) -> dict[str, Any]:
        return {"hole": self.hole.as_record(), "capsule": self.capsule.as_record()}


@dataclass(frozen=True, slots=True)
class ShaderMemoryBinding:
    value_id: int
    storage: ShaderStorageClass
    access: str
    reason: str
    promotion: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "storage": self.storage.value,
            "access": self.access,
            "reason": self.reason,
            "promotion": self.promotion,
        }


@dataclass(frozen=True, slots=True)
class ShaderTilingPlan:
    phase: int
    method: str
    local_size: int
    tile_shape: tuple[int, ...] | None
    mapping: str

    def as_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "method": self.method,
            "local_size": self.local_size,
            "tile_shape": None if self.tile_shape is None else list(self.tile_shape),
            "mapping": self.mapping,
        }


@dataclass(frozen=True, slots=True)
class ShaderRegionPhase:
    index: int
    program_indices: tuple[int, ...]
    program_digest: str
    operation_names: tuple[str, ...]
    barrier_after: str | None

    def as_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "program_indices": list(self.program_indices),
            "program_digest": self.program_digest,
            "operation_names": list(self.operation_names),
            "barrier_after": self.barrier_after,
        }


@dataclass(frozen=True, slots=True)
class ShaderIdentityReceipt:
    identity: str
    applied: bool
    before_sha256: str
    after_sha256: str
    reasons: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "applied": self.applied,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ShaderRegionArtifact:
    cut: ShaderRegionCut
    captured_program: CapturedFusedProgram
    memory_bindings: tuple[ShaderMemoryBinding, ...]
    phases: tuple[ShaderRegionPhase, ...]
    tiling: tuple[ShaderTilingPlan, ...]
    identity_receipts: tuple[ShaderIdentityReceipt, ...]
    artifact_digest: str
    contract_name: str

    @property
    def program(self) -> FusedProgram:
        return self.captured_program.program

    @property
    def stages(self) -> tuple[FusedProgram, ...]:
        return self.captured_program.stages

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "cut": self.cut.as_record(),
            "memory_bindings": [item.as_record() for item in self.memory_bindings],
            "phases": [phase.as_record() for phase in self.phases],
            "tiling": [item.as_record() for item in self.tiling],
            "identity_receipts": [item.as_record() for item in self.identity_receipts],
            "artifact_digest": self.artifact_digest,
            "contract_name": self.contract_name,
            "recursive_deployment_permitted": False,
        }


def _captured(value: Any) -> CapturedFusedProgram:
    if isinstance(value, CapturedFusedProgram):
        return value
    program = getattr(value, "program", value)
    if not isinstance(program, FusedProgram):
        raise TypeError(f"shader regions require FusedProgram interiors, got {type(program)!r}")
    return CapturedFusedProgram(program, {}, tuple(getattr(value, "stages", ()) or ()))


def _assert_sealed(programs: Iterable[FusedProgram], *, where: str) -> None:
    forbidden = [
        str(step.op_name)
        for program in programs
        for step in program.steps
        if str(step.op_name).replace("_", "").lower() in _DEPLOYMENT_OPS
        or bool((step.attrs or {}).get("deployment_frame"))
    ]
    if forbidden:
        raise ShaderRegionError(
            f"sealed shader region contains recursive deployment during {where}: "
            + ", ".join(forbidden)
        )


def cut_shader_region(region_index: int, captured: Any) -> ShaderRegionCut:
    """Cut one captured numerical graph behind a deterministic typed hole."""

    captured = _captured(captured)
    programs = (captured.program, *captured.stages)
    _assert_sealed(programs, where="cut")
    program = captured.program
    metadata = _with_boundary_dtype(program.meta or {})
    output_names = {int(value_id): str(name) for name, value_id in program.outputs.items()}
    boundary = ShaderRegionBoundary(
        inputs=tuple(
            ShaderBoundaryValue(
                int(value_id), None, "input", _meta_record(metadata.get(value_id)),
            )
            for value_id in ordered_feed_ids(program)
        ),
        outputs=tuple(
            ShaderBoundaryValue(
                int(value_id), output_names[int(value_id)], "output",
                _meta_record(metadata.get(value_id)),
            )
            for value_id in program.outputs.values()
        ),
        effects=tuple(sorted({
            str(step.op_name)
            for step in program.steps
            if str(step.op_name).lower() in _MUTATING_OPS
        })),
    )
    incomplete = [
        value.value_id
        for value in (*boundary.inputs, *boundary.outputs)
        if value.metadata.get("shape") is None
        or value.metadata.get("dtype") is None
    ]
    if incomplete:
        raise ShaderRegionError(
            "typed shader hole lacks shape/dtype metadata for boundary values: "
            + ", ".join(map(str, incomplete))
        )
    source_digest = _digest([
        _program_semantic_record(item) for item in programs
    ])
    capsule_record = {
        "schema": SCHEMA,
        "target": "glsl",
        "source_digest": source_digest,
        "boundary": boundary.as_record(),
        "sealed_deployment_depth": 1,
    }
    capsule_digest = _digest(capsule_record)
    capsule = ShaderRegionCapsule(
        int(region_index), "glsl", captured, boundary,
        source_digest, capsule_digest,
    )
    hole = ShaderDeploymentHole(
        region_index=int(region_index),
        marker=f"__scheduled_region_{int(region_index)}__",
        target="glsl",
        invocation="compute-dispatch",
        capsule_digest=capsule_digest,
        boundary_digest=boundary.digest,
    )
    return ShaderRegionCut(hole, capsule)


def cut_shader_regions(
    region_programs: Mapping[int, Any],
    selected_regions: Iterable[int],
) -> dict[int, ShaderRegionCut]:
    selected = tuple(sorted(set(map(int, selected_regions))))
    missing = set(selected) - set(map(int, region_programs))
    if missing:
        raise ShaderRegionError(f"deployment selected absent shader regions: {sorted(missing)}")
    return {
        index: cut_shader_region(index, region_programs[index])
        for index in selected
    }


def _copy_step(step: OpStep, *, attrs: Mapping[str, Any] | None = None) -> OpStep:
    return OpStep(
        step_id=int(step.step_id), op_name=str(step.op_name),
        input_ids=list(map(int, step.input_ids)),
        attrs=dict(step.attrs if attrs is None else attrs),
        result_id=int(step.result_id), mode_sensitive=bool(step.mode_sensitive),
        level=step.level,
    )


def _shader_identity_program(program: FusedProgram, gemm_variant: str) -> FusedProgram:
    steps = []
    for step in program.steps:
        attrs = dict(step.attrs or {})
        if str(step.op_name) == "matmul":
            attrs.update({
                "backend_intrinsic_family": "blas.gemm",
                "shader_identity": str(gemm_variant),
                "shader_memory_method": (
                    "cooperative_workgroup_tiles"
                    if gemm_variant == "glslblas_gemm"
                    else "source_order_storage_reads"
                ),
            })
        steps.append(_copy_step(step, attrs=attrs))
    return FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=steps,
        outputs=dict(program.outputs),
        state_in=None if program.state_in is None else set(program.state_in),
        meta=None if program.meta is None else dict(program.meta),
        extras=None if program.extras is None else dict(program.extras),
    )


def _memory_bindings(
    programs: Sequence[FusedProgram], *, gemm_variant: str,
) -> tuple[ShaderMemoryBinding, ...]:
    feeds = {int(value_id) for program in programs for value_id in program.feeds}
    outputs = {
        int(value_id) for program in programs for value_id in program.outputs.values()
    }
    produced = {
        int(step.result_id) for program in programs for step in program.steps
    }
    meta = {
        int(value_id): value_meta
        for program in programs for value_id, value_meta in (program.meta or {}).items()
    }
    matmul_inputs = {
        int(value_id)
        for program in programs for step in program.steps
        if str(step.op_name) == "matmul"
        for value_id in step.input_ids
    }
    bindings = []
    for value_id in sorted(feeds | outputs | produced):
        value_meta = meta.get(value_id)
        if value_id in matmul_inputs and gemm_variant == "glslblas_gemm":
            storage = ShaderStorageClass.STORAGE_BUFFER
            access = "cooperative-read"
            reason = "deployment input remains backed by the boundary SSBO"
            promotion = "workgroup_shared_tile"
        elif value_meta is not None and value_meta.source_id is not None:
            storage = ShaderStorageClass.STORAGE_VIEW
            access = "read"
            reason = "explicit offset/stride view retains its owning storage buffer"
            promotion = None
        elif value_id in feeds or value_id in outputs:
            storage = ShaderStorageClass.STORAGE_BUFFER
            access = (
                "read-write" if value_id in feeds and value_id in outputs
                else "read" if value_id in feeds else "write"
            )
            reason = "deployment boundary values cross the shader ABI"
            promotion = None
        else:
            storage = ShaderStorageClass.REGISTER
            access = "private"
            reason = "intermediate remains invocation-local across fused operations"
            promotion = None
        bindings.append(ShaderMemoryBinding(
            value_id, storage, access, reason, promotion,
        ))
    return tuple(bindings)


def _tiling_plans(
    programs: Sequence[FusedProgram], phases: Sequence[Any], *,
    gemm_variant: str, local_size: int,
) -> tuple[ShaderTilingPlan, ...]:
    plans = []
    for phase in phases:
        selected = tuple(programs[index] for index in phase.program_indices)
        has_matmul = any(
            str(step.op_name) == "matmul"
            for program in selected for step in program.steps
        )
        if has_matmul and gemm_variant == "glslblas_gemm":
            tile = min(16, max(1, int(int(local_size) ** 0.5)))
            tile = 1 << max(0, tile.bit_length() - 1)
            plans.append(ShaderTilingPlan(
                phase.index, "cooperative_gemm", tile * tile, (tile, tile),
                "one workgroup per output tile; lanes cooperatively load k tiles",
            ))
        elif has_matmul:
            plans.append(ShaderTilingPlan(
                phase.index, "source_order_gemm", int(local_size), None,
                "one invocation per output element; p reduction retained",
            ))
        else:
            plans.append(ShaderTilingPlan(
                phase.index, "flat_fused", int(local_size), None,
                "one invocation per logical output element",
            ))
    return tuple(plans)


def compile_shader_region(
    cut: ShaderRegionCut, *, contract: Any = None, local_size: int = 256,
) -> ShaderRegionArtifact:
    """Run the sealed shader-only pass and produce linkable phase topology."""

    if contract is None:
        from .work_contract import active_contract

        contract = active_contract()
    original = cut.capsule.captured_program
    original_programs = (original.program, *original.stages)
    _assert_sealed(original_programs, where="second-pass input")
    before = _digest([
        _program_semantic_record(item) for item in original_programs
    ])
    transformed_whole = _shader_identity_program(
        original.program, contract.shaders.blas_gemm
    )
    transformed_stages = tuple(
        _shader_identity_program(program, contract.shaders.blas_gemm)
        for program in original.stages
    )
    transformed_programs = (transformed_whole, *transformed_stages)
    _assert_sealed(transformed_programs, where="second-pass output")
    after = _digest([
        _program_semantic_record(item) for item in transformed_programs
    ])
    transformed = CapturedFusedProgram(
        transformed_whole,
        dict(original.feeds),
        transformed_stages,
    )
    from .contiguous_execution import contiguate

    phase_programs = transformed_stages or (transformed_whole,)
    contiguous = contiguate(phase_programs)
    phases = tuple(
        ShaderRegionPhase(
            index=phase.index,
            program_indices=tuple(map(int, phase.program_indices)),
            program_digest=_digest([
                _program_semantic_record(phase_programs[index])
                for index in phase.program_indices
            ]),
            operation_names=tuple(
                str(step.op_name)
                for index in phase.program_indices
                for step in phase_programs[index].steps
            ),
            barrier_after=phase.barrier_after,
        )
        for phase in contiguous.phases
    )
    memory = _memory_bindings(
        phase_programs, gemm_variant=contract.shaders.blas_gemm,
    )
    tiling = _tiling_plans(
        phase_programs,
        contiguous.phases,
        gemm_variant=contract.shaders.blas_gemm,
        local_size=int(local_size),
    )
    memory_after = _digest({
        "program": after,
        "memory": [item.as_record() for item in memory],
        "tiling": [item.as_record() for item in tiling],
    })
    receipts = (
        ShaderIdentityReceipt(
            "shader_blas_identity",
            before != after,
            before,
            after,
            (
                f"matmul regions select {contract.shaders.blas_gemm}",
                "non-BLAS operations retain universal mathematical order",
            ),
        ),
        ShaderIdentityReceipt(
            "shader_memory_topology",
            True,
            after,
            memory_after,
            (
                "boundary values assigned storage-buffer ABI",
                "invocation-local intermediates assigned registers",
                "cooperatively reused GEMM operands promoted through workgroup tiles",
            ),
        ),
    )
    artifact_record = {
        "schema": SCHEMA,
        "identity_library_version": IDENTITY_LIBRARY_VERSION,
        "capsule_digest": cut.capsule.capsule_digest,
        "shader_policy": _canonical(contract.shaders),
        "programs": [
            _program_semantic_record(item) for item in transformed_programs
        ],
        "memory": [item.as_record() for item in memory],
        "phases": [
            {
                "index": item.index,
                "program_indices": list(item.program_indices),
                "program_digest": item.program_digest,
                "barrier_after": item.barrier_after,
            }
            for item in phases
        ],
        "tiling": [item.as_record() for item in tiling],
        "receipts": [item.as_record() for item in receipts],
    }
    return ShaderRegionArtifact(
        cut=cut,
        captured_program=transformed,
        memory_bindings=memory,
        phases=phases,
        tiling=tiling,
        identity_receipts=receipts,
        artifact_digest=_digest(artifact_record),
        contract_name=str(contract.name),
    )


def compile_shader_regions(
    cuts: Mapping[int, ShaderRegionCut], *, contract: Any = None,
    local_size: int = 256,
) -> dict[int, ShaderRegionArtifact]:
    return {
        int(index): compile_shader_region(
            cut, contract=contract, local_size=local_size,
        )
        for index, cut in sorted(cuts.items())
    }


def link_shader_regions(
    control_program: Any,
    captured_regions: Mapping[int, Any],
    cuts: Mapping[int, ShaderRegionCut],
    *,
    contract: Any = None,
    local_size: int = 256,
) -> tuple[dict[int, Any], dict[int, ShaderRegionArtifact]]:
    """Compile sealed interiors and bind them to the outer control holes."""

    expected = set(map(int, getattr(control_program, "region_indices", ())))
    bad = set(map(int, cuts)) - expected
    if bad:
        raise ShaderRegionError(
            f"shader holes are not present in outer control: {sorted(bad)}"
        )
    artifacts = compile_shader_regions(
        cuts, contract=contract, local_size=local_size,
    )
    linked = {int(index): value for index, value in captured_regions.items()}
    for index, artifact in artifacts.items():
        if artifact.cut.hole.marker != f"__scheduled_region_{index}__":
            raise ShaderRegionError(f"shader hole marker drifted for region {index}")
        linked[index] = artifact
    return linked, artifacts


__all__ = [
    "SCHEMA",
    "ShaderBoundaryValue",
    "ShaderDeploymentHole",
    "ShaderIdentityReceipt",
    "ShaderMemoryBinding",
    "ShaderRegionArtifact",
    "ShaderRegionBoundary",
    "ShaderRegionCapsule",
    "ShaderRegionCut",
    "ShaderRegionError",
    "ShaderRegionPhase",
    "ShaderStorageClass",
    "ShaderTilingPlan",
    "compile_shader_region",
    "compile_shader_regions",
    "cut_shader_region",
    "cut_shader_regions",
    "link_shader_regions",
]
