"""Shared protocol for backend identity libraries and SSA swap procedures.

Universal identities run before this layer.  A backend rule may only rewrite
the resulting repository SSA through this registry, under the active work
contract, and must publish a deterministic receipt.  GLSL and WebGPU share
rules whenever their mathematical/device constraint is the same.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import IRModule, SSAValue
from .backend_intrinsics import BackendIntrinsicTarget, resolve_backend_intrinsic


@dataclass(frozen=True)
class BackendIdentityDecision:
    identity: str
    priority: int
    backends: tuple[str, ...]
    applied: bool
    exact: bool
    reasons: tuple[str, ...]
    before_sha256: str
    after_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "priority": self.priority,
            "backends": list(self.backends),
            "applied": self.applied,
            "exact": self.exact,
            "reasons": list(self.reasons),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True)
class BackendIdentityResult:
    module: IRModule
    outputs: dict[str, tuple[SSAValue, ...]]
    decisions: tuple[BackendIdentityDecision, ...]


def _dtype_topology(module: IRModule) -> tuple[tuple[str, int, str], ...]:
    rows = set()
    for function_name, function in module.functions.items():
        for value in function.args:
            rows.add((str(function_name), int(value.id), str(value.dtype)))
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in instruction.args:
                    rows.add((str(function_name), int(value.id), str(value.dtype)))
                if instruction.res is not None:
                    rows.add((
                        str(function_name), int(instruction.res.id),
                        str(instruction.res.dtype),
                    ))
    return tuple(sorted(rows))


def _topology_sha(module: IRModule) -> str:
    instructions = []
    for function_name, function in module.functions.items():
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                intrinsic = instruction.attributes.get("backend_intrinsic")
                instructions.append((
                    str(function_name),
                    str(block_name),
                    int(index),
                    str(instruction.op),
                    tuple(int(value.id) for value in instruction.args),
                    None if instruction.res is None else int(instruction.res.id),
                    str(instruction.attributes.get("callee") or ""),
                    (
                        str(intrinsic.get("location") or "")
                        if isinstance(intrinsic, Mapping) else ""
                    ),
                ))
    payload = json.dumps(
        {
            "values": _dtype_topology(module),
            "instructions": instructions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rewrite_output_values(
    module: IRModule,
    outputs: Mapping[str, Sequence[SSAValue]],
) -> dict[str, tuple[SSAValue, ...]]:
    rewritten = {}
    for function_name, values in outputs.items():
        function = module.functions.get(function_name)
        by_id: dict[int, SSAValue] = {}
        if function is not None:
            for value in function.args:
                by_id[int(value.id)] = value
            for block in function.blocks.values():
                for instruction in block.instrs:
                    for value in instruction.args:
                        by_id[int(value.id)] = value
                    if instruction.res is not None:
                        by_id[int(instruction.res.id)] = instruction.res
        rewritten[str(function_name)] = tuple(
            by_id.get(int(value.id), copy.deepcopy(value)) for value in values
        )
    return rewritten


def _backend_intrinsic_swaps(
    module: IRModule,
    outputs: Mapping[str, Sequence[SSAValue]],
    *,
    backend: str,
    overrides: Mapping[
        str, BackendIntrinsicTarget | Mapping[str, Any]
    ] | None,
) -> tuple[
    IRModule,
    dict[str, tuple[SSAValue, ...]],
    BackendIdentityDecision | None,
]:
    """Swap flagged semantic calls to a backend-owned intrinsic identity."""

    candidates = []
    for function_name, function in module.functions.items():
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                candidate = instruction.attributes.get(
                    "backend_intrinsic_candidate"
                )
                family = instruction.attributes.get(
                    "backend_intrinsic_family"
                )
                if not isinstance(candidate, Mapping) or not family:
                    continue
                candidates.append((
                    str(function_name), str(block_name), index,
                    str(family), dict(candidate),
                ))
    if not candidates:
        return module, _rewrite_output_values(module, outputs), None

    before = _topology_sha(module)
    transformed = copy.deepcopy(module)
    applied = 0
    unavailable = []
    for function_name, block_name, index, family, candidate in candidates:
        target = resolve_backend_intrinsic(
            family,
            backend=str(backend),
            lowering_namespace=candidate.get("lowering_namespace"),
            overrides=overrides,
        )
        if target is None:
            unavailable.append(family)
            continue
        instruction = transformed.functions[function_name].blocks[
            block_name
        ].instrs[index]
        previous = {
            "op": str(instruction.op),
            "callee": instruction.attributes.get("callee"),
        }
        instruction.op = "BackendIntrinsic"
        target_record = target.as_record()
        if str(backend) == "glsl" and family == "blas.gemm":
            from .work_contract import active_contract

            target_record["shader_variant"] = (
                active_contract().shaders.blas_gemm
            )
        instruction.attributes["backend_intrinsic"] = target_record
        instruction.attributes["backend_intrinsic_original"] = previous
        instruction.attributes["callee"] = target.symbol
        instruction.attributes["lowered_from"] = previous["op"]
        applied += 1

    if not applied:
        transformed = module
    after = _topology_sha(transformed)
    reasons = []
    if applied:
        reasons.append(
            f"swapped {applied} flagged semantic call(s) to backend-owned "
            f"{backend} intrinsic locations"
        )
    if unavailable:
        reasons.append(
            "no qualified target for: " + ", ".join(sorted(set(unavailable)))
        )
    decision = BackendIdentityDecision(
        identity="backend_intrinsic_location_swap",
        priority=10,
        backends=(str(backend),),
        applied=bool(applied),
        exact=True,
        reasons=tuple(reasons),
        before_sha256=before,
        after_sha256=after,
    )
    return (
        transformed,
        _rewrite_output_values(transformed, outputs),
        decision,
    )


def _shader_float32_storage(
    module: IRModule,
    outputs: Mapping[str, Sequence[SSAValue]],
    *,
    backend: str,
    licensed_inexact: bool,
) -> tuple[IRModule, dict[str, tuple[SSAValue, ...]], BackendIdentityDecision]:
    identity = "shader_float64_storage_to_float32"
    targets = ("glsl", "webgpu")
    before = _topology_sha(module)
    found = sum(
        dtype in {"float64", "double", "f64"}
        for _function, _value, dtype in _dtype_topology(module)
    )
    reasons = []
    applied = False
    transformed = module
    if backend not in targets:
        reasons.append(f"identity does not target backend {backend!r}")
    elif not found:
        reasons.append("SSA contains no float64 values")
    elif not licensed_inexact:
        reasons.append(
            "active work contract forbids inexact identities; float64 "
            "storage cannot be narrowed"
        )
    else:
        transformed = copy.deepcopy(module)
        seen = set()
        for function in transformed.functions.values():
            values = [*function.args]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    values.extend(instruction.args)
                    if instruction.res is not None:
                        values.append(instruction.res)
            for value in values:
                if id(value) in seen:
                    continue
                seen.add(id(value))
                if str(value.dtype).lower() in {"float64", "double", "f64"}:
                    value.dtype = "float32"
        applied = True
        reasons.append(
            f"narrowed {found} float64 SSA value identities to float32; "
            "the shader storage model requires 32-bit floats and the active "
            "contract licenses this inexact swap"
        )
    rewritten_outputs = _rewrite_output_values(transformed, outputs)
    after = _topology_sha(transformed)
    return transformed, rewritten_outputs, BackendIdentityDecision(
        identity=identity,
        priority=100,
        backends=targets,
        applied=applied,
        exact=False,
        reasons=tuple(reasons),
        before_sha256=before,
        after_sha256=after,
    )


def apply_backend_identities(
    module: IRModule,
    outputs: Mapping[str, Sequence[SSAValue]],
    *,
    backend: str,
    licensed_inexact: bool | None = None,
    intrinsic_overrides: Mapping[
        str, BackendIntrinsicTarget | Mapping[str, Any]
    ] | None = None,
) -> BackendIdentityResult:
    """Apply the ordered identity library for one backend.

    Rules consume and return repository SSA.  This is deliberately one shared
    procedure for GLSL and WebGPU; a backend-specific rule belongs here with a
    narrower target tuple, not hidden in an emitter.
    """

    if licensed_inexact is None:
        from .work_contract import active_contract
        licensed_inexact = bool(active_contract().inexact_identities)
    current = module
    current_outputs = {
        str(name): tuple(values) for name, values in outputs.items()
    }
    decisions = []
    current, current_outputs, intrinsic_decision = _backend_intrinsic_swaps(
        current, current_outputs, backend=str(backend),
        overrides=intrinsic_overrides,
    )
    if intrinsic_decision is not None:
        decisions.append(intrinsic_decision)
    current, current_outputs, decision = _shader_float32_storage(
        current, current_outputs, backend=str(backend),
        licensed_inexact=bool(licensed_inexact),
    )
    decisions.append(decision)
    return BackendIdentityResult(current, current_outputs, tuple(decisions))


__all__ = [
    "BackendIdentityDecision",
    "BackendIdentityResult",
    "apply_backend_identities",
]
