from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Local deps: keep this runner scheduler-local
from ..autoautograd.whiteboard_cache import WhiteboardCache
from ..autoautograd.whiteboard_runtime import run_batched_vjp


# ---------------- Registry (scheduler-local, no new math) ----------------

@dataclass(frozen=True)
class OpSpec:
    """
    Scheduler-local descriptor:
      - id: stable numeric identity (for logs/telemetry)
      - symbol: name looked up in AbstractTensor space (whitelisted)
    No operators are defined here; this only constrains what the runner is allowed to invoke.
    """
    id: int
    symbol: str


class OpRegistry:
    """
    Minimal in-runner registry. Not a math surface.
    Holds metadata so the runner can enforce a whitelist and pick scalarization.
    """
    def __init__(self) -> None:
        self._by_name: Dict[str, OpSpec] = {}
        self._ids_in_use: set[int] = set()

    def register(self, name: str, *, id: int, symbol: str) -> None:
        if name in self._by_name:
            raise ValueError(f"op '{name}' already registered")
        if id in self._ids_in_use:
            raise ValueError(f"op id {id} already used")
        self._by_name[name] = OpSpec(id=id, symbol=symbol)
        self._ids_in_use.add(id)

    def get(self, name: str) -> OpSpec:
        spec = self._by_name.get(name)
        if spec is None:
            raise KeyError(f"op '{name}' is not registered")
        return spec

    def known(self) -> Tuple[str, ...]:
        return tuple(self._by_name.keys())


# ---------------- Runner (used by triage) ----------------

class BulkOpRunner:
    """
    Scheduler-owned op runner with an embedded registry.
    - Enforces that only whitelisted AbstractTensor symbols are used.
    - Delegates compute to whiteboard_runtime (single-backward; cache-aware).
    - Supports backend override per bin/job (no operator invention).
    """

    def __init__(self, *, cache: Optional[WhiteboardCache] = None) -> None:
        self.cache = cache or WhiteboardCache()
        self.registry = OpRegistry()
        # sensible defaults: these are names in your codebase
        # Feel free to adjust IDs but keep them stable once published.
        self._install_defaults()

    # --------- public API expected by triage ---------

    def try_cached(
        self,
        job: Any,  # OpJob-like
        *,
        get_attr: Callable[[int], Any],
        get_attr_version: Optional[Callable[[int], Optional[int]]] = None,  # kept for signature symmetry
        backend: Any = None,
    ) -> Optional[Tuple[float, Tuple[float, ...]]]:
        """
        Return (y, grads) on cache hit; otherwise None.
        Pure probe: construct the key and use the WhiteboardCache directly.
        """
        sys_obj = getattr(get_attr, "__self__", None)
        # Build versions aligned to src_ids
        versions: List[int] = []
        for i in job.src_ids:
            if get_attr_version is not None:
                v = get_attr_version(i)
                versions.append(0 if v is None else int(v))
            else:
                node_v = 0
                try:
                    node_v = int(getattr(sys_obj.nodes[i], "version", 0)) if sys_obj is not None else 0
                except Exception:
                    node_v = 0
                versions.append(node_v)
        # Feature shape from a sample attr
        sample = get_attr(job.src_ids[0])
        feat_shape = getattr(sample, "shape", ())
        key = self.cache.make_key(
            op_name=job.op,
            src_ids=job.src_ids,
            versions=versions,
            feat_shape=feat_shape if isinstance(feat_shape, tuple) else (),
            weight=job.weight,
            scale=job.scale,
            residual=job.residual,
            backend_tag=job.backend_tag,
        )
        return self.cache.get(key)

    def run_bin(
        self,
        op: str,
        jobs: Sequence[Any],  # sequence of OpJob
        *,
        get_attr: Callable[[int], Any],
        get_attr_version: Optional[Callable[[int], Optional[int]]] = None,
        backend: Any = None,
    ) -> List[Tuple[float, Tuple[float, ...]]]:
        """
        Compute (y, grads) 1:1 for `jobs` using a single batched whiteboard VJP.
        """
        spec = self.registry.get(op)  # enforce whitelist (raises if unknown)

        def op_apply(x: Any) -> Any:
            # Prefer class-level AbstractTensor function if available (records properly)
            AT = type(x)
            fn_cls = getattr(AT, spec.symbol, None)
            if callable(fn_cls):
                try:
                    return fn_cls(x)
                except TypeError:
                    return fn_cls(x,)
            # Fallback to instance method
            fn = getattr(x, spec.symbol)
            return fn()

        sys_obj = getattr(get_attr, "__self__", None)

        batch = run_batched_vjp(
            sys=sys_obj,
            jobs=jobs,
            op_apply=op_apply,
            get_attr=get_attr,
            backend=backend,
        )

        out: List[Tuple[float, Tuple[float, ...]]] = []
        for idx, j in enumerate(jobs):
            y = batch.ys[idx]
            grads = batch.grads_per_source[idx]

            # Build and populate cache with the same key as try_cached
            versions: List[int] = []
            for i in j.src_ids:
                if get_attr_version is not None:
                    v = get_attr_version(i)
                    versions.append(0 if v is None else int(v))
                else:
                    node_v = 0
                    try:
                        node_v = int(getattr(sys_obj.nodes[i], "version", 0)) if sys_obj is not None else 0
                    except Exception:
                        node_v = 0
                    versions.append(node_v)
            sample = get_attr(j.src_ids[0])
            feat_shape = getattr(sample, "shape", ())
            key = self.cache.make_key(
                op_name=j.op,
                src_ids=j.src_ids,
                versions=versions,
                feat_shape=feat_shape if isinstance(feat_shape, tuple) else (),
                weight=j.weight,
                scale=j.scale,
                residual=j.residual,
                backend_tag=j.backend_tag,
            )
            self.cache.put(key, (y, grads))
            out.append((y, grads))
        return out

    # --------- internal helpers ---------

    def _install_defaults(self) -> None:
        """
        Install minimal, conservative set of symbols.
        These must map to existing AbstractTensor primitives-no new math here.
        """
        # Example: 'sum_k' means "we will call AT's sum over the stacked inputs".
        self.registry.register("sum_k", id=1, symbol="sum")
        self.registry.register("prod_k", id=2, symbol="prod")
        # You can add others as you expose them through AbstractTensor; keep them minimal.
