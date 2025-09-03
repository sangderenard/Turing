from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Sequence, Optional, Union, Protocol
import numpy as np

try:
    from ..abstraction import AbstractTensor
except Exception:  # pragma: no cover - AbstractTensor may be unavailable
    AbstractTensor = None  # type: ignore

IndexLike = Union[None, Sequence[int], Iterable[int]]

# ---- Policy interface ----------------------------------------------------

class BackendPolicy(Protocol):
    # Required
    def asarray(self, x: Any) -> Any: ...
    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any: ...
    def getter(self, node: Any, attr: str) -> Any: ...
    def setter(self, node: Any, attr: str, value: Any) -> None: ...
    # Optional (No-ops by default if missing)
    def scatter_row(self, node: Any, attr: str, row_value: Any) -> None: ...
    def pre_build(self, view: "NodeAttrView") -> None: ...
    def post_build(self, view: "NodeAttrView") -> None: ...
    def pre_commit(self, view: "NodeAttrView") -> None: ...
    def post_commit(self, view: "NodeAttrView") -> None: ...

# Sensible NumPy policy (works for plain arrays & array-likes)
class NumpyPolicy:
    def asarray(self, x: Any) -> Any:
        return np.asarray(x)
    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        return np.stack(xs, axis=axis)
    def getter(self, node: Any, attr: str) -> Any:
        return getattr(node, attr) if hasattr(node, attr) else node[attr]
    def setter(self, node: Any, attr: str, value: Any) -> None:
        if hasattr(node, attr): setattr(node, attr, value)
        else: node[attr] = value
    # Optional hooks: omit for defaults

# Example AbstractTensor-ish policy (adjust to your API as needed)
class AbstractTensorPolicy(NumpyPolicy):
    """
    Falls back to NumPy for stack/asarray if your AbstractTensor lacks them.
    Swap in your real constructors (e.g., AbstractTensor.stack / from_numpy).
    """
    def __init__(self, AT):
        self.AT = AT
    def asarray(self, x: Any) -> Any:
        # Prefer existing AbstractTensor values; otherwise wrap numpy
        if isinstance(x, self.AT):
            return x
        if hasattr(self.AT, "numpy"):
            return self.AT.numpy(np.asarray(x))
        return np.asarray(x)
    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        if hasattr(self.AT, "stack"):
            return self.AT.stack(xs, axis=axis)
        return np.stack([np.asarray(x) for x in xs], axis=axis)

    def scatter_row(self, node: Any, attr: str, row_value: Any) -> None:
        tensor = self.getter(node, attr)
        idx = getattr(node, "row_index", None)
        if idx is None:
            self.setter(node, attr, row_value)
        else:
            tensor.scatter_row(idx, row_value)

# ---- View with policy manager -------------------------------------------

@dataclass
class NodeAttrView:
    """
    Vectorized view over nodes[i].<attr> with backend/override policy.

    Policy precedence:
      - If policy_overrides=True (default), use policy hook when available,
        ignoring per-instance hooks.
      - If policy_overrides=False, use provided hooks when set; otherwise
        fall back to policy, then to NumPy defaults.

    Extra hook: `scatter_row` lets a backend control how rows are written back.
    """
    nodes: Any
    attr: str
    indices: IndexLike = None
    select: Optional[Callable[[Any], Any]] = None

    # Policy & override behavior
    policy: Optional[BackendPolicy] = None
    policy_overrides: bool = True  # backend trumps per-hook by default

    # Optional per-instance hooks (leave None to defer to policy/defaults)
    stack_fn: Optional[Callable[[Sequence[Any], int], Any]] = None
    asarray_fn: Optional[Callable[[Any], Any]] = None
    getter: Optional[Callable[[Any, str], Any]] = None
    setter: Optional[Callable[[Any, str, Any], None]] = None
    scatter_row: Optional[Callable[[Any, str, Any], None]] = None  # extra hook

    # internal
    _tensor: Any = None
    _order: list[int] | None = None

    # --- hook resolution ---
    def _resolve(self):
        pol = self.policy
        if pol is None:
            sample = None
            try:
                first = self.nodes[0]
            except Exception:
                try:
                    first = next(iter(self.nodes.values()))
                except Exception:
                    first = None
            if first is not None:
                try:
                    sample = getattr(first, self.attr) if hasattr(first, self.attr) else first[self.attr]
                except Exception:
                    sample = None
            if AbstractTensor is not None and isinstance(sample, AbstractTensor):
                pol = AbstractTensorPolicy(AbstractTensor)
            else:
                pol = NumpyPolicy()
        def choose(name, local):
            if self.policy_overrides:
                # Prefer backend policy if it implements the method; else local; else numpy default
                if hasattr(pol, name):
                    return getattr(pol, name)
                if local is not None:
                    return local
            else:
                # Prefer local if provided; else backend; else numpy default
                if local is not None:
                    return local
                if hasattr(pol, name):
                    return getattr(pol, name)
            # Fallbacks for core hooks
            if name == "asarray":  return np.asarray
            if name == "stack":    return lambda xs, axis=0: np.stack(xs, axis=axis)
            if name == "getter":   return lambda n, a: getattr(n, a) if hasattr(n, a) else n[a]
            if name == "setter":   return lambda n, a, v: setattr(n, a, v) if hasattr(n, a) else n.__setitem__(a, v)
            # Optional hooks default to no-op/None
            if name in ("pre_build","post_build","pre_commit","post_commit"):
                return lambda *_args, **_kw: None
            if name == "scatter_row":
                return None
            raise RuntimeError(f"No default for hook '{name}'")
        return {
            "asarray":    choose("asarray", self.asarray_fn),
            "stack":      choose("stack", self.stack_fn),
            "getter":     choose("getter", self.getter),
            "setter":     choose("setter", self.setter),
            "scatter_row":choose("scatter_row", self.scatter_row),
            "pre_build":  choose("pre_build", None),
            "post_build": choose("post_build", None),
            "pre_commit": choose("pre_commit", None),
            "post_commit":choose("post_commit", None),
        }

    # --- core ops ---
    def build(self) -> "NodeAttrView":
        H = self._resolve()
        H["pre_build"](self)

        # Resolve node ids
        if self.indices is None:
            try:
                ids = list(range(len(self.nodes)))
            except Exception:
                ids = list(self.nodes.keys())  # mapping-like
        else:
            ids = list(self.indices)

        # Gather + optional select + asarray
        cols = []
        for i in ids:
            v = H["getter"](self.nodes[i], self.attr)
            if self.select is not None:
                v = self.select(v)
            cols.append(H["asarray"](v))

        # Homogeneity check
        if not cols:
            raise ValueError("No nodes selected.")
        ref_shape = getattr(cols[0], "shape", None)
        for c in cols:
            if getattr(c, "shape", None) != ref_shape:
                raise ValueError("Node attributes are ragged; provide select() or normalize shapes.")

        self._tensor = H["stack"](cols, axis=0)  # (len(ids), *attr_shape)
        self._order = ids

        H["post_build"](self)
        return self

    @property
    def tensor(self) -> Any:
        if self._tensor is None:
            self.build()
        return self._tensor

    def commit(self) -> None:
        if self._tensor is None or not self._order:
            return
        H = self._resolve()
        H["pre_commit"](self)

        T = self._tensor
        scatter_row = H["scatter_row"]
        for row_idx, node_id in enumerate(self._order):
            if scatter_row is not None:
                scatter_row(self.nodes[node_id], self.attr, T[row_idx])
            else:
                H["setter"](self.nodes[node_id], self.attr, T[row_idx])

        H["post_commit"](self)

    @contextmanager
    def editing(self):
        T = self.tensor
        try:
            yield T
        finally:
            self.commit()
