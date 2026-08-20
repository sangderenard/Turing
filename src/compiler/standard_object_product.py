"""Generic compiled products for class-like numerical object surfaces.

The product boundary deliberately joins existing compiler authorities instead
of creating a library-specific compiler: ``KernelBank`` owns verified
parametric/specialized forward artifacts, ProcessGraph owns analytical graph
inversion, and repository SSA plus LLVM own native reverse compilation.
Publication is atomic and fails closed unless every method has both its
parametric forward and its compiled parametric VJP.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .kernel_bank import CompiledVariant, KernelBank, KernelSpec
from .llvm_training_runtime import (
    NativeGraphReverse,
    compile_native_graph_reverse,
    native_artifact_record,
)


STANDARD_OBJECT_SCHEMA = "turing.standard-object-product.v1"


@dataclass(frozen=True, slots=True)
class StandardProperty:
    name: str
    kind: str
    mutable: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mutable": bool(self.mutable),
        }


@dataclass(frozen=True)
class MethodGraphCapture:
    """A fresh AbstractTensor method run and its differentiation contract."""

    output: Any
    bindings: Mapping[str, Any]
    wrt_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class StandardMethod:
    name: str
    kernel: KernelSpec
    capture_graph: Callable[[], MethodGraphCapture]
    specializations: tuple[Mapping[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.name != self.kernel.name:
            raise ValueError(
                f"method {self.name!r} must name kernel {self.kernel.name!r}"
            )


@dataclass(frozen=True)
class StandardObject:
    name: str
    identity: str
    methods: tuple[StandardMethod, ...]
    properties: tuple[StandardProperty, ...] = ()

    def __post_init__(self) -> None:
        method_names = tuple(method.name for method in self.methods)
        property_names = tuple(prop.name for prop in self.properties)
        if not self.name or not self.identity:
            raise ValueError("standard object requires a name and identity")
        if not method_names:
            raise ValueError(f"standard object {self.identity!r} has no methods")
        if len(set(method_names)) != len(method_names):
            raise ValueError(f"standard object repeats methods: {method_names!r}")
        if len(set(property_names)) != len(property_names):
            raise ValueError(f"standard object repeats properties: {property_names!r}")


@dataclass(frozen=True)
class CompiledStandardMethod:
    spec: StandardMethod
    parametric_forward: CompiledVariant
    parametric_reverse: NativeGraphReverse
    specialized_forwards: tuple[CompiledVariant, ...]


@dataclass(frozen=True)
class StandardObjectProduct:
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    methods: Mapping[str, CompiledStandardMethod]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _variant_record(variant: CompiledVariant) -> dict[str, Any]:
    manifest = json.loads(
        (variant.directory / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "key": variant.key,
        "specialized": dict(variant.specialized),
        "manifest": str((variant.directory / "manifest.json").resolve()),
        "verification": dict(manifest.get("verification") or {}),
    }


def cook_standard_object(
    spec: StandardObject,
    *,
    directory: str | Path,
    contract: str | None = None,
) -> StandardObjectProduct:
    """Compile and publish one complete standard numerical object.

    Ordering is an invariant: every parametric forward is admitted first,
    every explicit-seed graph reverse is then compiled to a native library,
    and only then are optional exact parameter rows admitted.  Consequently a
    specialization can never make an otherwise incomplete method publishable.
    """

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bank = KernelBank(root / "forward-bank", {
        method.name: method.kernel for method in spec.methods
    })
    compiled: dict[str, CompiledStandardMethod] = {}

    for method in spec.methods:
        forward = bank.get(
            method.name, contract=contract, specialized=None,
        )
        capture = method.capture_graph()
        reverse = compile_native_graph_reverse(
            capture.output,
            bindings=capture.bindings,
            wrt_value_ids=capture.wrt_value_ids,
            name=f"{spec.identity.replace('.', '_')}__{method.name}__vjp",
            directory=root / "reverse" / method.name,
            unit_output_seed=False,
        )
        specializations = tuple(
            bank.get(
                method.name,
                contract=contract,
                specialized=dict(parameters),
            )
            for parameters in method.specializations
        )
        compiled[method.name] = CompiledStandardMethod(
            method, forward, reverse, specializations,
        )

    semantic_manifest = {
        "schema": STANDARD_OBJECT_SCHEMA,
        "object": {
            "name": spec.name,
            "identity": spec.identity,
            "properties": [prop.to_mapping() for prop in spec.properties],
        },
        "contract": contract or "develop",
        "publication_invariant": {
            "parametric_forward_required": True,
            "parametric_graph_reverse_required": True,
            "reverse_must_be_backend_compiled": True,
            "specializations_are_optional_overlays": True,
        },
        "methods": [
            {
                "name": method.name,
                "source_sha256": hashlib.sha256(
                    method.kernel.source.encode("utf-8")
                ).hexdigest(),
                "parametric_forward_key": compiled[method.name].parametric_forward.key,
                "reverse_output_value_ids": list(
                    compiled[method.name].parametric_reverse.output_value_ids
                ),
                "reverse_gradient_value_ids": {
                    str(key): int(value) for key, value in
                    compiled[method.name].parametric_reverse.gradient_value_ids.items()
                },
                "reverse_seed_value_ids": {
                    str(key): int(value) for key, value in
                    compiled[method.name].parametric_reverse.seed_value_ids.items()
                },
                "specializations": [
                    dict(variant.specialized)
                    for variant in compiled[method.name].specialized_forwards
                ],
            }
            for method in spec.methods
        ],
    }
    product_id = hashlib.sha256(_canonical(semantic_manifest)).hexdigest()
    manifest = {
        **semantic_manifest,
        "product_id": product_id,
        "artifacts": {
            method.name: {
                "parametric_forward": _variant_record(
                    compiled[method.name].parametric_forward
                ),
                "parametric_reverse": native_artifact_record(
                    compiled[method.name].parametric_reverse.artifact
                ),
                "specialized_forwards": [
                    _variant_record(variant) for variant in
                    compiled[method.name].specialized_forwards
                ],
            }
            for method in spec.methods
        },
    }
    manifest_path = root / "manifest.json"
    temporary = root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(manifest_path)
    return StandardObjectProduct(root, manifest_path, manifest, compiled)


def mathematical_sublibrary_object(
    library: Any,
    *,
    kernels: Mapping[str, KernelSpec],
    graph_captures: Mapping[str, Callable[[], MethodGraphCapture]],
    specializations: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> StandardObject:
    """Adapt a canonical mathematical sublibrary without copying its catalog."""

    names = tuple(method.name for method in library.methods)
    missing_kernels = set(names) - set(kernels)
    missing_captures = set(names) - set(graph_captures)
    extras = (set(kernels) | set(graph_captures)) - set(names)
    if missing_kernels or missing_captures or extras:
        raise ValueError(
            "mathematical object implementation does not equal its catalog: "
            f"missing_kernels={sorted(missing_kernels)!r}, "
            f"missing_graph_captures={sorted(missing_captures)!r}, "
            f"extras={sorted(extras)!r}"
        )
    matrix = dict(specializations or {})
    return StandardObject(
        name=str(library.name),
        identity=str(library.identity),
        methods=tuple(
            StandardMethod(
                name,
                kernels[name],
                graph_captures[name],
                tuple(matrix.get(name, ())),
            )
            for name in names
        ),
    )
