from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Sequence, Optional, Union, Protocol

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

# Example AbstractTensor-ish policy (adjust to your API as needed)
class AbstractTensorPolicy(BackendPolicy):
    """
    Falls back to AbstractTensor for stack/asarray if your AbstractTensor lacks them.
    Swap in your real constructors (e.g., AbstractTensor.stack / from_AbstractTensor).
    """
    def __init__(self, AT):
        self.AT = AT
    def asarray(self, x: Any) -> Any:
        return AbstractTensor.get_tensor(x)
    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        return AbstractTensor.stack(xs, dim=axis)

    def scatter_row(self, node: Any, attr: str, row_value: Any) -> None:
        tensor = self.getter(node, attr)
        idx = getattr(node, "row_index", None)
        if idx is None:
            self.setter(node, attr, row_value)
        else:
            tensor.scatter_row(idx, row_value)

# ---- View with policy manager -------------------------------------------

@dataclass
class BatchView:
    """Stacked tensor with per-job slices."""

    tensor: Any
    slice_for_job: Sequence[slice]

@dataclass
class NodeAttrView:
    """
    Vectorized view over nodes[i].<attr> with backend/override policy.

    Policy precedence:
      - If policy_overrides=True (default), use policy hook when available,
        ignoring per-instance hooks.
      - If policy_overrides=False, use provided hooks when set; otherwise
        fall back to policy, then to AbstractTensor defaults.

    Extra hook: `scatter_row` lets a backend control how rows are written back.

    Resolved hooks are cached per instance.  Pass a pre-resolved mapping via
    ``hooks`` when constructing multiple views to avoid recomputing.  Set
    ``check_shapes=False`` if the caller can guarantee that all selected node
    attributes share the same shape.
    """
    nodes: Any
    attr: Union[str, Sequence[str]]
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
    hooks: dict[str, Any] | None = None
    check_shapes: bool = True
    _hooks: dict[str, Any] | None = None
    _attr_slices: dict[str, slice] | None = None
    _attr_shapes: dict[str, tuple[int, ...]] | None = None

    def __post_init__(self):
        if self.hooks is not None:
            self._hooks = self.hooks

    # --- hook resolution ---
    def resolve(self) -> dict[str, Any]:
        return self._resolve()

    def _resolve(self):
        if self._hooks is not None:
            return self._hooks
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
            if first is not None and isinstance(self.attr, str):
                try:
                    sample = getattr(first, self.attr) if hasattr(first, self.attr) else first[self.attr]
                except Exception:
                    sample = None
            pol = AbstractTensorPolicy(AbstractTensor)

        def choose(name, local):
            if self.policy_overrides:
                if hasattr(pol, name):
                    return getattr(pol, name)
                if local is not None:
                    return local
            else:
                if local is not None:
                    return local
                if hasattr(pol, name):
                    return getattr(pol, name)
            if name == "asarray":  return AbstractTensor.asarray
            if name == "stack":    return lambda xs, axis=0: AbstractTensor.stack(xs, axis=axis)
            if name == "getter":   return lambda n, a: getattr(n, a) if hasattr(n, a) else n[a]
            if name == "setter":   return lambda n, a, v: setattr(n, a, v) if hasattr(n, a) else n.__setitem__(a, v)
            if name in ("pre_build","post_build","pre_commit","post_commit"):
                return lambda *_args, **_kw: None
            if name == "scatter_row":
                return None
            raise RuntimeError(f"No default for hook '{name}'")
        self._hooks = {
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
        return self._hooks

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

        cols = []
        is_list = not isinstance(self.attr, str)
        if is_list and self._attr_slices is None:
            self._attr_slices = {}
            self._attr_shapes = {}

        for i in ids:
            if is_list:
                pieces = []
                offset = 0
                for a in self.attr:  # type: ignore[not-an-iterable]
                    v = H["getter"](self.nodes[i], a)
                    if self.select is not None:
                        v = self.select(v)
                    arr = H["asarray"](v).reshape(-1)
                    pieces.append(arr)
                    if i == ids[0]:
                        ln = int(getattr(arr, "shape", (1,))[0])
                        self._attr_slices[a] = slice(offset, offset + ln)
                        self._attr_shapes[a] = tuple(getattr(H["asarray"](v), "shape", ()))
                        offset += ln
                row = AbstractTensor.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
                cols.append(row)
            else:
                v = H["getter"](self.nodes[i], self.attr)  # type: ignore[arg-type]
                if self.select is not None:
                    v = self.select(v)
                cols.append(H["asarray"](v))

        if not cols:
            raise ValueError("No nodes selected.")
        if self.check_shapes and not is_list:
            ref_shape = getattr(cols[0], "shape", None)
            for c in cols:
                if getattr(c, "shape", None) != ref_shape:
                    raise ValueError("Node attributes are ragged; provide select() or normalize shapes.")

        self._tensor = H["stack"](cols, axis=0)
        self._order = ids

        H["post_build"](self)
        return self

    def build_batches(self, job_batches: Sequence[Sequence[int]]) -> "BatchView":
        """Stack nodes for ``job_batches`` and track slices per job.

        ``job_batches`` supplies per-job node indices.  All referenced nodes are
        gathered and stacked once; ``slice_for_job`` records the span within the
        stacked tensor for each job.
        """

        flat: list[int] = []
        slices: list[slice] = []
        start = 0
        for batch in job_batches:
            ids = list(int(i) for i in batch)
            flat.extend(ids)
            end = start + len(ids)
            slices.append(slice(start, end))
            start = end

        view = NodeAttrView(
            self.nodes,
            self.attr,
            indices=flat,
            select=self.select,
            policy=self.policy,
            policy_overrides=self.policy_overrides,
            stack_fn=self.stack_fn,
            asarray_fn=self.asarray_fn,
            getter=self.getter,
            setter=self.setter,
            scatter_row=self.scatter_row,
            hooks=self.resolve(),
            check_shapes=self.check_shapes,
        ).build()

        return BatchView(view.tensor, tuple(slices))

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
        is_list = not isinstance(self.attr, str)
        for row_idx, node_id in enumerate(self._order):
            row = T[row_idx]
            if is_list:
                assert self._attr_slices is not None and self._attr_shapes is not None
                for a in self.attr:  # type: ignore[not-an-iterable]
                    sl = self._attr_slices[a]
                    seg = row[sl]
                    shape = self._attr_shapes[a]
                    if hasattr(seg, "reshape"):
                        seg = seg.reshape(shape)
                    if scatter_row is not None:
                        scatter_row(self.nodes[node_id], a, seg)
                    else:
                        H["setter"](self.nodes[node_id], a, seg)
            else:
                if scatter_row is not None:
                    scatter_row(self.nodes[node_id], self.attr, row)
                else:
                    H["setter"](self.nodes[node_id], self.attr, row)

        H["post_commit"](self)

    @contextmanager
    def editing(self):
        T = self.tensor
        try:
            yield T
        finally:
            self.commit()
