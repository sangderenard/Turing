"""Receipt-gated installation of compiler products in isolated workers.

Compiler bootstrap products are ordinary project-compilation products.  They
are never trusted merely because a DLL exists: activation reloads the product,
re-runs the semantic verifier named by its persisted receipt, and then uses the
same authored-source-preserving installer as standard objects.  Child compiler
workers inherit the product list through the environment, so a bounded crawl
uses the same proven deployments without surrendering source recursively.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import os
from pathlib import Path
from typing import Any, Callable


COMPILER_BOOTSTRAP_PRODUCTS_ENV = "TURING_COMPILER_BOOTSTRAP_PRODUCTS"


@dataclass(frozen=True)
class CompilerBootstrapActivation:
    product: str
    qualified_name: str
    adapter: str
    status: str
    native_probe_count: int
    fallback_probe_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "qualified_name": self.qualified_name,
            "adapter": self.adapter,
            "status": self.status,
            "native_probe_count": int(self.native_probe_count),
            "fallback_probe_count": int(self.fallback_probe_count),
        }


def _compute_dispatch_is_native_safe(
    count,
    *,
    limits,
    preferred_local_size=256,
    minimum_local_size=32,
) -> bool:
    """Mirror only the native library's terminating validation boundary."""

    try:
        count = int(count)
        preferred = int(preferred_local_size)
        minimum = int(minimum_local_size)
        if count < 0 or preferred < 1 or minimum < 1:
            return False
        local_cap = min(
            preferred,
            int(limits.max_group_size[0]),
            int(limits.max_invocations),
        )
        if local_cap < 1:
            return False
        local = 1 << (local_cap.bit_length() - 1)
        if count:
            small_target = 1 << (count - 1).bit_length()
            local = min(local, max(min(minimum, local), small_target))
        if count == 0:
            return True
        needed = (count + local - 1) // local
        group_x = min(needed, int(limits.max_group_count[0]))
        if group_x < 1:
            return False
        remaining = (needed + group_x - 1) // group_x
        group_y = min(remaining, int(limits.max_group_count[1]))
        if group_y < 1:
            return False
        group_z = (remaining + group_y - 1) // group_y
        return group_z <= int(limits.max_group_count[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _activate_compute_dispatch(product, qualified_name: str):
    from . import deployment_lowering as owner

    generous = owner.ComputeDispatchLimits(
        (65535, 65535, 65535), (1024, 1024, 64), 1024,
    )
    compact = owner.ComputeDispatchLimits((4, 3, 2), (64, 64, 64), 64)
    probes = (
        {"count": 0, "limits": generous},
        {"count": 1, "limits": generous},
        {"count": 1024, "limits": generous},
        {
            "count": 1000,
            "limits": compact,
            "preferred_local_size": 64,
            "minimum_local_size": 8,
        },
        {"count": -1, "limits": generous},
    )
    deployed = product.verify_native_record_return_callable(
        qualified_name,
        owner.plan_compute_dispatch,
        probes,
        native_precondition=_compute_dispatch_is_native_safe,
        activation_adapter="compute-dispatch-record-v1",
    )
    return owner, deployed


def _activate_qualified_scalar(product, qualified_name: str):
    """Re-prove one ABI-selected scalar leaf and return its authored owner."""

    from .project_compilation_product import _resolve_product_callable

    link = dict(product.links[str(qualified_name)])
    source_module = str(link.get("source_module") or "")
    if not source_module:
        raise ValueError("scalar bootstrap link has no source module")
    owner, authored = _resolve_product_callable(
        source_module, str(qualified_name),
    )
    library = product.root / str(link.get("native_library") or "")
    receipt_path = library.parent / "native-verification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    probes = []
    for record in receipt.get("probes") or ():
        arguments = ast.literal_eval(str(record.get("arguments") or "()"))
        keywords = ast.literal_eval(str(record.get("keywords") or "{}"))
        probes.append(dict(keywords) if keywords else tuple(arguments))
    deployed = product.verify_native_scalar_callable(
        str(qualified_name),
        authored,
        probes,
        activation_adapter="qualified-scalar-call-v1",
    )
    return owner, deployed


_ACTIVATION_ADAPTERS: dict[str, Callable[..., tuple[Any, Callable[..., Any]]]] = {
    "compute-dispatch-record-v1": _activate_compute_dispatch,
    "qualified-scalar-call-v1": _activate_qualified_scalar,
}
_ACTIVE_DEPLOYMENTS: dict[tuple[str, str], Callable[..., Any]] = {}


def compiler_bootstrap_product_paths(value: str | None = None) -> tuple[Path, ...]:
    raw = os.environ.get(COMPILER_BOOTSTRAP_PRODUCTS_ENV, "") if value is None else value
    return tuple(dict.fromkeys(
        Path(item).resolve()
        for item in str(raw).split(os.pathsep)
        if item.strip()
    ))


def set_compiler_bootstrap_products(paths) -> tuple[Path, ...]:
    resolved = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
    os.environ[COMPILER_BOOTSTRAP_PRODUCTS_ENV] = os.pathsep.join(
        str(path) for path in resolved
    )
    return resolved


def activate_compiler_bootstrap_products(paths=None) -> tuple[CompilerBootstrapActivation, ...]:
    """Re-prove and install every receipt-declared compiler deployment."""

    from .project_compilation_product import open_project_compilation_product

    selected = (
        compiler_bootstrap_product_paths()
        if paths is None else tuple(Path(path).resolve() for path in paths)
    )
    activations = []
    for product_path in selected:
        product = open_project_compilation_product(product_path)
        for qualified_name, link_value in sorted(product.links.items()):
            link = dict(link_value)
            if link.get("kind") == "source-region-integral":
                continue
            library = product.root / str(link.get("native_library") or "")
            receipt_path = library.parent / "native-verification.json"
            if not library.is_file() or not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            adapter_name = str(receipt.get("activation_adapter") or "")
            if not adapter_name:
                continue
            try:
                adapter = _ACTIVATION_ADAPTERS[adapter_name]
            except KeyError as error:
                raise ValueError(
                    f"unknown compiler bootstrap activation adapter {adapter_name!r}"
                ) from error
            owner, deployed = adapter(product, str(qualified_name))
            installed = product.install_callable(
                str(qualified_name),
                owner,
                deployed,
                targeted_source_fallback=True,
            )
            _ACTIVE_DEPLOYMENTS[(
                product.root.as_posix(), str(qualified_name),
            )] = installed.__turing_deployed_callable__
            verification = dict(
                installed.__turing_deployed_callable__.__turing_native_verification__
            )
            activations.append(CompilerBootstrapActivation(
                product=product.root.as_posix(),
                qualified_name=str(qualified_name),
                adapter=adapter_name,
                status=str(verification.get("status") or ""),
                native_probe_count=int(verification.get("native_probe_count") or 0),
                fallback_probe_count=int(verification.get("fallback_probe_count") or 0),
            ))
    return tuple(activations)


def compiler_bootstrap_runtime_state() -> tuple[dict[str, Any], ...]:
    """Return live native/fallback routing evidence for installed products."""

    records = []
    for (product, qualified_name), deployed in sorted(_ACTIVE_DEPLOYMENTS.items()):
        verification = dict(getattr(
            deployed, "__turing_native_verification__", {},
        ))
        routes = dict(getattr(deployed, "__turing_native_route_counts__", {}))
        records.append({
            "product": product,
            "qualified_name": qualified_name,
            "activation_adapter": str(
                verification.get("activation_adapter") or ""
            ),
            "verification_status": str(verification.get("status") or ""),
            "verification_native_probe_count": int(
                verification.get("native_probe_count") or 0
            ),
            "verification_fallback_probe_count": int(
                verification.get("fallback_probe_count") or 0
            ),
            "runtime_native_calls": int(routes.get("native") or 0),
            "runtime_fallback_calls": int(routes.get("fallback") or 0),
            "post_activation_native_calls": max(
                0,
                int(routes.get("native") or 0)
                - int(verification.get("native_probe_count") or 0),
            ),
            "post_activation_fallback_calls": max(
                0,
                int(routes.get("fallback") or 0)
                - int(verification.get("fallback_probe_count") or 0),
            ),
        })
    return tuple(records)


__all__ = [
    "COMPILER_BOOTSTRAP_PRODUCTS_ENV",
    "CompilerBootstrapActivation",
    "activate_compiler_bootstrap_products",
    "compiler_bootstrap_runtime_state",
    "compiler_bootstrap_product_paths",
    "set_compiler_bootstrap_products",
]
