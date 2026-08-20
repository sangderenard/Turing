"""Backend-qualified locations for canonical compiler intrinsics.

The semantic operation remains backend-neutral until a backend identity
selects one of these locations.  Locations are data, not imports: repository
SSA can retain and receipt the choice without importing a runtime backend.
Hosts with a larger deployment gestalt may provide an explicit override for a
semantic family while retaining the same node-swap protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BackendIntrinsicTarget:
    backend: str
    semantic_family: str
    location: str
    symbol: str
    consumption: str
    lowering_namespaces: tuple[str, ...] = ()
    operand_positions: tuple[int, ...] = ()

    @classmethod
    def from_mapping(
        cls, backend: str, semantic_family: str, raw: Mapping[str, Any],
    ) -> "BackendIntrinsicTarget":
        return cls(
            backend=str(backend),
            semantic_family=str(semantic_family),
            location=str(raw["location"]),
            symbol=str(raw.get("symbol") or str(raw["location"]).rsplit(":", 1)[-1]),
            consumption=str(raw.get("consumption") or "deployment_bypass"),
            lowering_namespaces=tuple(map(str, raw.get("lowering_namespaces") or ())),
            operand_positions=tuple(map(int, raw.get("operand_positions") or ())),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "semantic_family": self.semantic_family,
            "location": self.location,
            "symbol": self.symbol,
            "consumption": self.consumption,
            "lowering_namespaces": list(self.lowering_namespaces),
            "operand_positions": list(self.operand_positions),
        }


_BUILTIN_TARGETS = {
    ("glsl", "blas.gemm"): BackendIntrinsicTarget(
        backend="glsl",
        semantic_family="blas.gemm",
        location=(
            "src.common.tensors.accelerator_backends.glsl_backend:"
            "glslblas_gemm"
        ),
        symbol="glslblas_gemm",
        consumption="deployment_bypass",
        lowering_namespaces=("abstract_tensor",),
        operand_positions=(0, 1),
    ),
}


def resolve_backend_intrinsic(
    semantic_family: str,
    *,
    backend: str,
    lowering_namespace: str | None = None,
    overrides: Mapping[str, BackendIntrinsicTarget | Mapping[str, Any]] | None = None,
) -> BackendIntrinsicTarget | None:
    """Resolve one backend target, allowing an explicit gestalt override."""

    family = str(semantic_family)
    backend = str(backend)
    raw_override = None if overrides is None else overrides.get(family)
    if raw_override is None:
        target = _BUILTIN_TARGETS.get((backend, family))
    elif isinstance(raw_override, BackendIntrinsicTarget):
        target = raw_override
    else:
        target = BackendIntrinsicTarget.from_mapping(
            backend, family, raw_override,
        )
    if target is None or target.backend != backend:
        return None
    namespace = None if lowering_namespace is None else str(lowering_namespace)
    if target.lowering_namespaces and namespace not in target.lowering_namespaces:
        return None
    return target


__all__ = ["BackendIntrinsicTarget", "resolve_backend_intrinsic"]
