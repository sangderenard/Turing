
from __future__ import annotations

# Internal, framework-independent gradient bookkeeping helpers.

def requires_grad_(self, requires_grad: bool = True):
    """Enable or disable gradient tracking on this tensor."""
    self._requires_grad = requires_grad
    tape = getattr(self, "_tape", None)
    if tape is not None:
        tape.create_tensor_node(self)
    return self


@property
def requires_grad(self) -> bool:
    return getattr(self, "_requires_grad", False)


@requires_grad.setter
def requires_grad(self, value: bool) -> None:  # pragma: no cover - simple setter
    self._requires_grad = value


def backward(
    self,
    grad_output=None,
    *,
    retain_graph: bool = False,
):
    """Compute gradients for parameters with respect to this scalar tensor."""
    tape = getattr(self, "_tape", None) or autograd.tape
    try:
        tape.mark_loss(self)
    except Exception:
        pass
    params = tape.parameter_tensors()
    if not params:
        return None
    autograd.grad(
        self,
        params,
        grad_outputs=grad_output,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return None


@property
def grad(self):
    return getattr(self, "_grad", None)


def detach(self):
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.data
    result._requires_grad = False
    return result


@property
def is_leaf(self) -> bool:
    return not hasattr(self, "_grad_node")


def retain_grad(self):  # pragma: no cover - placeholder for API compatibility
    return self


@property
def grad_fn(self):
    return getattr(self, "_grad_node", None)


def zero_grad(self, *, clear_cache: bool = False):  # pragma: no cover - simple helper
    self._grad = None
    if clear_cache:
        tape = getattr(self, "_tape", None)
        if tape is not None:
            tid = id(self)
            if tape.graph.has_node(tid):
                anns = tape.graph.nodes[tid].get("annotations")
                if anns and "cache" in anns:
                    del anns["cache"]
            node = tape._nodes.get(tid)
            if node is not None:
                ctx_anns = node.ctx.get("annotations")
                if ctx_anns and "cache" in ctx_anns:
                    del ctx_anns["cache"]
    return self


def register_hook(self, hook):  # pragma: no cover - hooks not supported
    raise NotImplementedError("register_hook not supported for this tensor")


"""Lightweight automatic differentiation helpers.
\nThis module implements a backend-agnostic autograd system.  All tensor
implementations in the repository use the same pure-Python tape mechanics; no
external library is required for differentiation.  Only a limited set of
primitive arithmetic operators is currently covered, matching the needs of the
educational examples built on top of the tape.
\nThe implementation below provides two main pieces:
\n* ``GradTape`` – records a compact computation graph.
* ``Autograd`` – exposes ``record`` and ``grad`` helpers and houses backward
  rules for primitive operators (``add``, ``mul``, ``truediv``, ``pow``, ...).
\nBackends call ``Autograd.record`` after executing an operator to append a node
to the tape.  ``Autograd.grad`` then walks the recorded graph in reverse to
accumulate gradients for requested inputs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
import os
from contextlib import contextmanager

import networkx as nx
import hashlib
from .nested_pack import pack_nested_to_tensor

# Integrate the lightweight backward registry so that backward rules are
# resolved dynamically rather than being baked into the tape at record time.
from .backward import BACKWARD_REGISTRY
from .backward_registry import BACKWARD_RULES

import math
import statistics
import re

try:  # NumPy is an optional dependency for the repository
    import numpy as np
except Exception:  # pragma: no cover - tested in environments without numpy
    np = None  # type: ignore


@dataclass
class GradNode:
    """Single operation in the automatic differentiation graph."""

    op: str
    parents: List[Tuple[int, int]]
    ctx: Dict[str, Any]


class GradTape:
    """Minimal tape to record operations for reverse-mode autodiff.

    Nodes are keyed by ``id(tensor)`` similar to the provenance tracking
    used in the Turing scaffold. Each node knows its parents and the
    positional slot they occupied during the forward pass. Traversal
    yields nodes in reverse topological order suitable for backprop.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, GradNode] = {}
        self.graph = nx.DiGraph()
        self._op_index = 0
        self._parameters: Dict[int, int] = {}
        self._tensor_refs: Dict[int, Any] = {}
        self._param_index = 0
        self._loss_tensor: Any | None = None
        self._loss_id: int | None = None
        # Structural tensors are allowed to require_grad but are not treated
        # as trainable parameters and are excluded from parameter lists and
        # strict connectivity diagnostics.
        self._structural: set[int] = set()
        # Optional label exclude patterns for parameter registration.
        self._param_exclude_labellist: list[re.Pattern[str]] = []
        try:
            allow_env = os.environ.get("AUTOGRAD_PARAM_EXCLUDE_LABELS")
            if allow_env:
                parts: list[str] = []
                for chunk in allow_env.split(","):
                    parts.extend([p for p in chunk.split("|") if p])
                for pat in parts:
                    self._param_exclude_labellist.append(re.compile(pat))
        except Exception:
            self._param_exclude_labellist = []

    # ------------------------------------------------------------------
    # node utilities
    # ------------------------------------------------------------------
    def create_tensor_node(self, tensor: Any) -> None:
        """Register ``tensor`` as a root node in the graph."""
        tid = id(tensor)
        self.graph.add_node(tid, kind="tensor")
        self._tensor_refs[tid] = tensor
        try:
            tensor._tape = self  # type: ignore[attr-defined]
        except Exception:
            pass
        if getattr(tensor, "requires_grad", False):
            # Exclude structural tensors or those matching exclude label patterns
            anns = self.graph.nodes[tid].get("annotations", {})
            lbl = anns.get("label") or getattr(tensor, "_label", None)
            excluded = tid in self._structural or bool(anns.get("structural"))
            if not excluded and isinstance(lbl, str) and self._param_exclude_labellist:
                try:
                    for rx in self._param_exclude_labellist:
                        if rx.search(lbl):
                            excluded = True
                            break
                except Exception:
                    pass
            if excluded:
                node = self.graph.nodes[tid]
                node.setdefault("stateful", False)
                return
            if tid not in self._parameters:
                pid = self._param_index
                self._param_index += 1
                self._parameters[tid] = pid
            pid = self._parameters[tid]
            node = self.graph.nodes[tid]
            node["param_id"] = pid
            node["stateful"] = True
        else:
            node = self.graph.nodes[tid]
            node.setdefault("stateful", False)

    # loss/parameter utilities
    # ------------------------------------------------------------------
    def mark_loss(self, tensor: Any) -> None:
        """Declare ``tensor`` as the loss node for training."""
        tid = id(tensor)
        self._loss_tensor = tensor
        self._loss_id = tid
        self.graph.add_node(tid, kind="tensor")
        self.graph.nodes[tid]["loss"] = True
        self._tensor_refs[tid] = tensor

    # Structural parameter utilities
    # ------------------------------------------------------------------
    def mark_structural(self, tensor: Any, *, label: str | None = None) -> None:
        """Mark ``tensor`` as structural (non-trainable, excluded from params)."""
        tid = id(tensor)
        self._structural.add(tid)
        self.graph.add_node(tid, kind="tensor")
        anns = self.graph.nodes[tid].setdefault("annotations", {})
        anns["structural"] = True
        if label is not None:
            anns.setdefault("label", label)
        # If it was previously registered as a parameter, drop it
        if tid in self._parameters:
            try:
                del self._parameters[tid]
            except Exception:
                pass

    def parameter_tensors(self) -> List[Any]:
        items = sorted(self._parameters.items(), key=lambda x: x[1])
        result: List[Any] = []
        for tid, _ in items:
            if tid in self._structural:
                continue
            anns = self.graph.nodes.get(tid, {}).get("annotations", {})
            if anns.get("structural"):
                continue
            ref = self._tensor_refs.get(tid)
            if ref is not None:
                result.append(ref)
        return result

    def parameters(self) -> Tuple[Any | None, Dict[int, int]]:
        params = self.parameter_tensors()
        if not params:
            return None, {}
        tensor = pack_nested_to_tensor(params, cls=type(params[0]))
        return tensor, dict(self._parameters)

    def export_training_state(self) -> Tuple[nx.DiGraph, nx.DiGraph, Any | None, Dict[int, int]]:
        if self._loss_tensor is None:
            raise ValueError("Loss tensor has not been marked")
        fwd = self.export_forward_graph()
        bwd = self.export_backward_graph(self._loss_tensor)
        params_tensor, id_map = self.parameters()
        return fwd, bwd, params_tensor, id_map

    # ------------------------------------------------------------------
    # strict validation utilities
    # ------------------------------------------------------------------
    def validate_backward_ops(self, result: Any) -> list[dict[str, Any]]:
        """Return a list of missing-backward op diagnostics reachable from ``result``.

        Walk the recorded graph from ``result`` and report any operation nodes
        whose ``op`` is not registered in BACKWARD_REGISTRY. Each entry
        includes the op name and basic context (shapes, dtypes) to aid debugging.
        """
        missing: list[dict[str, Any]] = []
        for tid, node in self.traverse(result):
            op = node.op
            fn = BACKWARD_REGISTRY._methods.get(op)
            if fn is not None:
                continue
            ctx = node.ctx
            info = {
                "tensor_id": tid,
                "op": op,
                "input_shapes": ctx.get("input_shapes"),
                "result_shape": ctx.get("result_shape"),
                "input_dtypes": ctx.get("input_dtypes"),
                "result_dtype": ctx.get("result_dtype"),
                "params": ctx.get("params", {}),
            }
            missing.append(info)
        return missing

    # ------------------------------------------------------------------
    # orphan detection utilities
    # ------------------------------------------------------------------
    def orphan_data(self) -> List[Dict[str, Any]]:
        """Collect metadata for graph nodes that have no children.

        Returns
        -------
        list of dict
            Each entry contains the ``id`` of the tensor, a ``tensor``
            reference if still alive, and any stored ``annotations``.
        """

        orphans: List[Dict[str, Any]] = []
        for nid in self.graph.nodes:
            if self.graph.out_degree(nid) == 0:
                info = {
                    "id": nid,
                    "tensor": self._tensor_refs.get(nid),
                    "annotations": self.graph.nodes[nid].get("annotations", {}),
                }
                orphans.append(info)
        return orphans

    # ------------------------------------------------------------------
    # recording utilities
    # ------------------------------------------------------------------
    def record(
        self,
        op: str,
        inputs: Iterable[Any],
        result: Any,
        *,
        start: float | None = None,
        end: float | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Any:
        """Append a new node representing ``op`` to the tape.

        Parameters
        ----------
        op:
            Name of the operator being recorded.
        inputs:
            Operands participating in the operation.
        result:
            Tensor produced by the operation.
        start, end:
            Optional timestamps captured immediately before and after the
            operator executed.  When provided, they are stored on both the
            ``GradNode`` and the corresponding graph node for later
            benchmarking.
        """

        # Ensure a sequence of inputs; a single tensor should not be iterated elementwise.
        if isinstance(inputs, (list, tuple, set)):
            inputs = list(inputs)
        else:
            inputs = [inputs]
        parent_ids = [(id(t), pos) for pos, t in enumerate(inputs)]

        def _dtype(x: Any) -> Any:
            try:
                return x.get_dtype()  # type: ignore[attr-defined]
            except Exception:
                data = getattr(x, "data", x)
                return getattr(data, "dtype", getattr(x, "dtype", None))

        def _device(x: Any) -> Any:
            try:
                return x.get_device()  # type: ignore[attr-defined]
            except Exception:
                data = getattr(x, "data", x)
                return getattr(data, "device", getattr(x, "device", None))

        def _backend(x: Any) -> Any:
            return type(x).__name__

        def _strides(x: Any) -> Any:
            data = getattr(x, "data", x)
            return getattr(data, "strides", None)

        elapsed = None
        if start is not None and end is not None:
            elapsed = end - start

        # Tape payload policy: reduce memory by avoiding raw data storage.
        # AUTOGRAD_TAPE_DATA = 'none' | 'summary' | 'full'
        _payload_mode = os.environ.get("AUTOGRAD_TAPE_DATA", "none").lower()

        def _summarize(x: Any) -> Dict[str, Any]:
            d = getattr(x, "data", x)
            return {
                "shape": getattr(d, "shape", getattr(x, "shape", None)),
                "dtype": getattr(d, "dtype", getattr(x, "dtype", None)),
                "device": getattr(d, "device", getattr(x, "device", None)),
            }

        if _payload_mode == "full":
            inputs_payload = [x.data if hasattr(x, "data") else x for x in inputs]
            result_payload = result.data if hasattr(result, "data") else result
        elif _payload_mode == "summary":
            inputs_payload = [_summarize(x) for x in inputs]
            result_payload = _summarize(result)
        else:  # 'none' (default)
            inputs_payload = None
            result_payload = None

        ctx = {
            "inputs": list(inputs),
            "result": result,
            "inputs_data": inputs_payload,
            "result_data": result_payload,
            "input_shapes": [getattr(x, "shape", None) for x in inputs],
            "result_shape": getattr(result, "shape", None),
            "input_dtypes": [_dtype(x) for x in inputs],
            "result_dtype": _dtype(result),
            "input_devices": [_device(x) for x in inputs],
            "result_device": _device(result),
            "input_backends": [_backend(x) for x in inputs],
            "result_backend": _backend(result),
            "input_strides": [_strides(x) for x in inputs],
            "result_strides": _strides(result),
            "start": start,
            "end": end,
            "elapsed": elapsed,
            "params": params or {},
        }
        node = GradNode(op=op, parents=parent_ids, ctx=ctx)
        self._nodes[id(result)] = node

        # Build the global operation graph
        op_name = f"op_{self._op_index}"
        self._op_index += 1
        self.graph.add_node(
            op_name,
            kind="op",
            op=op,
            start=start,
            end=end,
            elapsed=elapsed,
            ctx=ctx,
        )
        for t in inputs:
            tid = id(t)
            self.graph.add_node(tid, kind="tensor")
            self.graph.add_edge(tid, op_name)
        rid = id(result)
        self.graph.add_node(rid, kind="tensor")
        self.graph.add_edge(op_name, rid)

        # Attach graph references so tensors know their generating node and tape.
        try:
            result._grad_node = node  # type: ignore[attr-defined]
            result._tape = self  # type: ignore[attr-defined]
            result._grad_tape = self  # legacy alias expected by some tests
        except Exception:
            pass

        # Ensure the result tensor carries at least the operation name as a label
        anns = self.graph.nodes[rid].get("annotations", {})
        if "label" not in anns:
            self.annotate(result, label=op)
        return result

    # ------------------------------------------------------------------
    # metadata utilities
    # ------------------------------------------------------------------
    def annotate(self, tensor: Any, **metadata: Any) -> None:
        """Attach ``metadata`` to tape entries for ``tensor``.

        The global graph node for ``tensor`` is always updated (creating it if
        necessary).  When a corresponding :class:`GradNode` exists, its context
        ``annotations`` are synchronised as well.  Callers may therefore
        annotate tensors regardless of whether they have been recorded on the
        tape yet.
        """

        tid = id(tensor)

        # Always ensure the tensor exists in the global graph and update its
        # annotations.
        self.graph.add_node(tid, kind="tensor")
        g_anns = self.graph.nodes[tid].setdefault("annotations", {})
        g_anns.update(metadata)

        node = self._nodes.get(tid)
        if node is not None:
            anns = node.ctx.setdefault("annotations", {})
            anns.update(metadata)

            # Also annotate the generating op node if present.
            try:
                for pred in self.graph.predecessors(tid):
                    if self.graph.nodes[pred].get("kind") == "op":
                        p_anns = self.graph.nodes[pred].setdefault("annotations", {})
                        p_anns.update(metadata)
            except Exception:
                pass

    def auto_annotate_eval(self, tensor: Any) -> None:
        """Annotate ``tensor`` with an evaluation hash and lineage.

        The evaluation hash is a SHA256 digest of the tensor's raw data when
        available.  ``lineage`` captures all ancestor tensors contributing to
        ``tensor``'s value, preferring previously assigned ``label`` metadata
        and falling back to tensor ``id`` values.
        """

        meta: Dict[str, Any] = {}
        data = getattr(tensor, "data", tensor)
        try:
            if hasattr(data, "tobytes"):
                raw = data.tobytes()
            else:
                raw = str(data).encode()
            meta["eval_hash"] = hashlib.sha256(raw).hexdigest()
        except Exception:
            pass

        try:
            lineage: List[Any] = []
            names: List[Any] = []
            for tid, _ in self.traverse(tensor):
                anns = self.graph.nodes.get(tid, {}).get("annotations", {})
                names.append(anns.get("label", tid))
            lineage = list(reversed(names))
            if lineage:
                meta["lineage"] = lineage
        except Exception:
            pass

        if meta:
            self.annotate(tensor, **meta)

    # ------------------------------------------------------------------
    # traversal utilities
    # ------------------------------------------------------------------
    def node(self, tensor: Any) -> Optional[GradNode]:
        """Return the ``GradNode`` for ``tensor`` if present."""

        return self._nodes.get(id(tensor))

    def traverse(self, result: Any) -> Generator[Tuple[int, GradNode], None, None]:
        """Yield ``(tensor_id, GradNode)`` in reverse topological order.

        Parameters
        ----------
        result:
            The final tensor whose history should be walked.
        """

        visited: set[int] = set()
        order: List[Tuple[int, GradNode]] = []

        def dfs(tid: int) -> None:
            node = self._nodes.get(tid)
            if node is None or tid in visited:
                return
            visited.add(tid)
            for pid, _ in node.parents:
                dfs(pid)
            order.append((tid, node))

        dfs(id(result))
        for item in reversed(order):
            yield item

    def required_cache(self, result: Any) -> set[int]:
        """Return tensor IDs whose ``data`` must be kept for backward."""

        required: set[int] = set()
        for tid, node in self.traverse(result):
            rule = BACKWARD_RULES.get(node.op, {})
            py = rule.get("python", {})
            params = list(py.get("parameters", []))[1:]
            body = py.get("body", "")
            num_inputs = len(node.parents)
            for idx, param in enumerate(params):
                if not param:
                    continue
                if not re.search(rf"\b{re.escape(param)}\b(?!\.shape)", body):
                    continue
                if idx < num_inputs:
                    pid, _ = node.parents[idx]
                    required.add(pid)
                else:
                    required.add(tid)
        return required

    def export_forward_graph(self) -> nx.DiGraph:
        """Return a forward computation graph of recorded operations.

        The returned graph is a :class:`networkx.DiGraph` where nodes are
        keyed by ``id(tensor)``.  Each node carries three attributes:

        ``op``
            Name of the operator that produced the tensor or ``None`` for
            leaf inputs.
        ``cached``
            ``True`` when the tensor has been annotated with a ``cache`` flag
            via :meth:`annotate` and therefore must retain its ``data`` for
            backward computations.
        ``metadata``
            Free-form dictionary of any additional annotations associated with
            the tensor.  It is copied verbatim from the internal tape.

        Edges denote data dependencies from input tensors to the tensors they
        help produce.
        """

        g = nx.DiGraph()

        # First populate nodes using the existing global graph which already
        # stores annotations.  Operation names are resolved by examining the
        # generating op node, if present.
        for tid, data in self.graph.nodes(data=True):
            if data.get("kind") != "tensor":
                continue
            anns = data.get("annotations", {})
            op_name = None
            try:
                for pred in self.graph.predecessors(tid):
                    pdata = self.graph.nodes[pred]
                    if pdata.get("kind") == "op":
                        op_name = pdata.get("op")
                        break
            except Exception:
                pass
            pid = self._parameters.get(tid)
            g.add_node(
                tid,
                op=op_name,
                cached=bool(anns.get("cache")),
                metadata=anns,
                param_id=pid,
                stateful=bool(data.get("stateful")),
                loss=(tid == self._loss_id),
            )

        # Add edges by following the operation nodes in the global graph so
        # dependencies are captured directly from the tape rather than from
        # the chronological recording order.  Each op connects its input
        # tensors to every tensor it produces.
        for nid, data in self.graph.nodes(data=True):
            if data.get("kind") != "op":
                continue
            inputs = [
                src
                for src in self.graph.predecessors(nid)
                if self.graph.nodes[src].get("kind") == "tensor"
            ]
            outputs = [
                dst
                for dst in self.graph.successors(nid)
                if self.graph.nodes[dst].get("kind") == "tensor"
            ]
            for src in inputs:
                for dst in outputs:
                    g.add_edge(src, dst)

        return g

    def export_backward_graph(self, result: Any) -> nx.DiGraph:
        """Return a backward computation graph starting at ``result``.

        The returned graph is a :class:`networkx.DiGraph` whose nodes are
        keyed by ``id(tensor)``.  Each node describes the backward operator
        associated with the tensor that produced it during the forward pass.

        ``op``
            Name of the forward operator or ``None`` for leaf inputs.
        ``fn``
            Callable retrieved from :data:`BACKWARD_REGISTRY` implementing the
            backward rule for ``op``.  ``None`` for leaves or unregistered ops.
        ``required``
            List of tensor IDs whose ``data`` must be retained in order to
            execute this backward operation.  This list is derived from the
            per-operator requirements in :data:`BACKWARD_RULES`.

        Edges are directed from each tensor to the tensors it depends on for
        gradient propagation (i.e. reverse of the forward graph).
        """

        g = nx.DiGraph()
        required = self.required_cache(result)

        for tid, node in self.traverse(result):
            fn = BACKWARD_REGISTRY._methods.get(node.op)

            needed: List[int] = []
            for pid, _ in node.parents:
                if pid in required:
                    needed.append(pid)
            if tid in required:
                needed.append(tid)

            pid = self._parameters.get(tid)
            g.add_node(
                tid,
                op=node.op,
                fn=fn,
                required=needed,
                param_id=pid,
                stateful=tid in self._parameters,
                loss=(tid == self._loss_id),
            )

            for pid2, _ in node.parents:
                if pid2 not in g:
                    g.add_node(
                        pid2,
                        op=None,
                        fn=None,
                        required=[pid2] if pid2 in required else [],
                        param_id=self._parameters.get(pid2),
                        stateful=pid2 in self._parameters,
                        loss=(pid2 == self._loss_id),
                    )
                g.add_edge(tid, pid2)

        return g


class TapeProfiler:
    """Compute basic timing statistics from a :class:`GradTape`.

    The profiler groups recorded operations by opcode and reports summary
    statistics.  If ``normalize`` is requested the elapsed time for each
    operation is divided by the total number of elements across all operands
    participating in that operation.
    """

    def __init__(self, tape: GradTape) -> None:
        self.tape = tape

    def per_op(self, normalize: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return per-operation timing statistics.

        Parameters
        ----------
        normalize:
            When ``True`` divide each timing measurement by the total flat size
            of all input operands.
        """

        groups: Dict[str, List[float]] = {}
        for _, data in self.tape.graph.nodes(data=True):
            if data.get("kind") != "op":
                continue
            elapsed = data.get("elapsed")
            if elapsed is None:
                continue
            if normalize:
                ctx = data.get("ctx", {})
                shapes = ctx.get("input_shapes", [])
                size = 0
                for shp in shapes:
                    if isinstance(shp, tuple):
                        n = 1
                        for dim in shp:
                            n *= int(dim)
                        size += n
                    elif shp is not None:
                        size += int(shp)
                    else:
                        size += 1
                if size > 0:
                    elapsed = elapsed / size
            groups.setdefault(data.get("op"), []).append(elapsed)

        stats: Dict[str, Dict[str, Any]] = {}
        for op, times in groups.items():
            times.sort()
            mean = sum(times) / len(times)
            if len(times) >= 2:
                q1, median, q3 = statistics.quantiles(times, n=4)
            else:
                q1 = median = q3 = times[0]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = [t for t in times if t < lower or t > upper]
            stats[op] = {
                "count": len(times),
                "mean": mean,
                "q1": q1,
                "median": median,
                "q3": q3,
                "outliers": outliers,
            }
        return stats


# ----------------------------------------------------------------------------
# Autograd engine
# ----------------------------------------------------------------------------


class Autograd:
    """Reverse-mode autodiff engine."""

    def __init__(self) -> None:
        self.tape = GradTape()
        self._no_grad_depth = 0
        self.capture_all = False
        # Strict mode: when True validate that all recorded ops reachable from
        # the loss have registered backward rules before attempting backprop.
        import os
        self.strict = os.environ.get("AUTOGRAD_STRICT", "0") not in ("0", "false", "False", None)
        # Allowlisted labels for strict connectivity checks (regex patterns)
        self._strict_label_allowlist: list[re.Pattern[str]] = []
        try:
            allow_env = os.environ.get("AUTOGRAD_STRICT_ALLOW_LABELS")
            if allow_env:
                parts: list[str] = []
                for chunk in allow_env.split(","):
                    parts.extend([p for p in chunk.split("|") if p])
                for pat in parts:
                    self._strict_label_allowlist.append(re.compile(pat))
        except Exception:
            # Non-fatal if env parsing fails
            self._strict_label_allowlist = []

    @contextmanager
    def no_grad(self) -> Generator[None, None, None]:
        """Context manager to temporarily disable gradient recording."""
        self._no_grad_depth += 1
        try:
            yield
        finally:
            self._no_grad_depth -= 1

    # ------------------------------------------------------------------
    # Strict-mode allowlist helpers
    # ------------------------------------------------------------------
    def whitelist(self, *tensors: Any) -> None:  # pragma: no cover - convenience
        """Mark tensors as allowed to be unused under strict connectivity checks."""
        for t in tensors:
            tape = getattr(t, "_tape", self.tape)
            try:
                tape.annotate(t, strict_allow_unused=True)
            except Exception:
                pass

    def whitelist_labels(self, *patterns: str) -> None:  # pragma: no cover - convenience
        """Add regex patterns to allowlist labels under strict mode."""
        for pat in patterns:
            try:
                self._strict_label_allowlist.append(re.compile(pat))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Structural parameter exclusion helpers
    # ------------------------------------------------------------------
    def structural(self, *tensors: Any, label: str | None = None) -> None:  # pragma: no cover - convenience
        """Mark one or more tensors as structural (non-trainable)."""
        for t in tensors:
            tape = getattr(t, "_tape", self.tape)
            try:
                tape.mark_structural(t, label=label)
            except Exception:
                pass

    def structural_labels(self, *patterns: str) -> None:  # pragma: no cover - convenience
        """Exclude tensors with labels matching these regex patterns from params."""
        for pat in patterns:
            try:
                rx = re.compile(pat)
            except Exception:
                continue
            try:
                self.tape._param_exclude_labellist.append(rx)
            except Exception:
                pass

    def record(
        self,
        op: str,
        inputs: Iterable[Any],
        result: Any,
        *,
        start: float | None = None,
        end: float | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Any:
        """Record an operation on the appropriate tape if supported."""

        op = {"truediv": "div"}.get(op, op)
        if op not in BACKWARD_REGISTRY._methods or self._no_grad_depth > 0:
            return result
        tape = getattr(result, "_tape", None)
        if tape is None:
            for t in inputs:
                tape = getattr(t, "_tape", None)
                if tape is not None:
                    break
        if tape is None:
            tape = self.tape
        tape.record(op, inputs, result, start=start, end=end, params=params)
        try:
            required = tape.required_cache(result)
            for tid in required:
                if tape.graph.has_node(tid):
                    anns = tape.graph.nodes[tid].setdefault("annotations", {})
                    anns["cache"] = True
                node = tape._nodes.get(tid)
                if node is not None:
                    ctx_anns = node.ctx.setdefault("annotations", {})
                    ctx_anns["cache"] = True
        except Exception:
            pass
        return result

    def grad(
        self,
        output: Any,
        inputs: Iterable[Any],
        grad_outputs: Any | None = None,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> List[Any]:
        # Strict-mode preflight: fail fast on missing backward ops
        if self.strict:
            # Always validate on the tape that produced the output, not the engine tape.
            tape_v = getattr(output, "_tape", self.tape)
            try:
                missing = tape_v.validate_backward_ops(output)
            except Exception:
                missing = []
            if missing:
                lines = [
                    f"missing backward for op='{m['op']}' result_shape={m.get('result_shape')} input_shapes={m.get('input_shapes')}"
                    for m in missing
                ]
                detail = "\n".join(lines)
                raise RuntimeError("Strict autograd: missing backward implementations for reachable ops:\n" + detail)
            # Also validate parameter connectivity: any input not present in the
            # backward graph, unless unused params are explicitly allowed.
            try:
                bwd_graph = tape_v.export_backward_graph(output)
            except Exception:
                bwd_graph = None
            if (
                not allow_unused
                and bwd_graph is not None
                and inputs is not None
            ):
                broken: list[str] = []
                import networkx as nx
                # Helper: whitelist check via per-tensor annotations or label allowlist
                def _is_strict_whitelisted(t: Any) -> bool:
                    try:
                        tid = id(t)
                        anns = tape_v.graph.nodes.get(tid, {}).get("annotations", {})
                    except Exception:
                        anns = {}
                    if isinstance(anns.get("strict_allow_unused"), bool) and anns.get("strict_allow_unused"):
                        return True
                    lbl = anns.get("label") or getattr(t, "_label", None)
                    if isinstance(lbl, str):
                        for rx in self._strict_label_allowlist:
                            try:
                                if rx.search(lbl):
                                    return True
                            except Exception:
                                continue
                    return False
                # Helper: structural check: if parameter is structural, skip
                def _is_structural(t: Any) -> bool:
                    try:
                        tid = id(t)
                        if tid in getattr(tape_v, "_structural", set()):
                            return True
                        anns = tape_v.graph.nodes.get(tid, {}).get("annotations", {})
                        return bool(anns.get("structural"))
                    except Exception:
                        return False
                # Helper: describe immediate op neighbors and whether each has a backward rule and a path-to-loss
                def _describe_neighbors(pid: int) -> list[dict[str, any]]:
                    desc: list[dict[str, any]] = []
                    G = tape_v.graph
                    for op_node in G.successors(pid):
                        node_data = G.nodes.get(op_node, {})
                        if node_data.get("kind") != "op":
                            continue
                        op_name = node_data.get("op")
                        has_bw = BACKWARD_REGISTRY._methods.get(op_name) is not None
                        # find produced tensor(s)
                        results = []
                        for rid in G.successors(op_node):
                            if G.nodes.get(rid, {}).get("kind") == "tensor":
                                results.append(rid)
                        # path to loss from any result
                        to_loss = False
                        for rid in results:
                            try:
                                if tape_v._loss_id is not None and nx.has_path(G, rid, tape_v._loss_id):
                                    to_loss = True
                                    break
                            except Exception:
                                pass
                        desc.append({
                            "op": op_name,
                            "op_node": op_node,
                            "has_backward": has_bw,
                            "results": results,
                            "result_shapes": [tape_v._nodes.get(rid, {}).ctx.get("result_shape") if rid in tape_v._nodes else None for rid in results],
                            "path_to_loss_from_result": to_loss,
                        })
                    return desc
                for idx, p in enumerate(inputs):
                    try:
                        pid = id(p)
                    except Exception:
                        continue
                    if bwd_graph.has_node(pid) or _is_strict_whitelisted(p) or _is_structural(p):
                        continue
                    # Connectivity diagnostics: forward presence, consumers, path existence
                    present = tape_v.graph.has_node(pid)
                    consumers = []
                    for rid, node in tape_v._nodes.items():
                        ctx = node.ctx
                        ins = ctx.get("inputs", [])
                        in_ids = [id(t) for t in ins]
                        if pid in in_ids:
                            consumers.append({
                                "op": node.op,
                                "result_id": rid,
                                "result_shape": ctx.get("result_shape"),
                                "input_ids": in_ids,
                            })
                    try:
                        to_loss = tape_v._loss_id is not None and nx.has_path(tape_v.graph, pid, tape_v._loss_id)
                    except Exception:
                        to_loss = None
                    neighbors = _describe_neighbors(pid)
                    lbl = getattr(p, "_label", None) or getattr(p, "shape", None) or f"input[{idx}]"
                    broken.append(
                        "\n".join([
                            f"param index={idx} id={pid} label={lbl}",
                            f"  on_forward={present} path_to_loss={to_loss}",
                            f"  consumers={consumers}",
                            f"  neighbor_ops={neighbors}",
                        ])
                    )
                if broken:
                    raise RuntimeError("Strict autograd: parameter(s) not connected to loss.\n" + "\n".join(broken))
        if isinstance(inputs, (list, tuple, set)):
            inputs = list(inputs)
        else:
            inputs = [inputs]
        out_grad = grad_outputs
        if out_grad is None:
            out_grad = output.ones_like()

        tape = getattr(output, "_tape", self.tape)
        grad_map: Dict[int, Any] = {id(output): out_grad}

        with self.no_grad():
            for tid, node in tape.traverse(output):
                grad_out = grad_map.get(tid)
                if grad_out is None:
                    continue
                bw = BACKWARD_REGISTRY._methods.get(node.op)
                if bw is None:
                    continue
                go = grad_out
                params = node.ctx.get("params", {})
                if params:
                    import inspect
                    sig = inspect.signature(bw)
                    allowed = {k: v for k, v in params.items() if k in sig.parameters}
                else:
                    allowed = {}
                parent_grads = bw(go, *node.ctx["inputs"], **allowed)
                if not isinstance(parent_grads, (list, tuple)):
                    parent_grads = (parent_grads,)
                for (pid, _), g in zip(node.parents, parent_grads):
                    if g is None:
                        continue
                    if pid in grad_map:
                        grad_map[pid] = grad_map[pid] + g
                    else:
                        grad_map[pid] = g

        results: List[Any] = []
        for idx, inp in enumerate(inputs):
            g = grad_map.get(id(inp))
            if g is None:
                if allow_unused:
                    results.append(None)
                else:
                    raise ValueError(
                        f"No gradient found for input at index {idx} with id={id(inp)}"
                    )
            else:
                if hasattr(inp, "data") and isinstance(inp.data, list):
                    g = g.tolist()
                results.append(g)

        for inp, g in zip(inputs, results):  # attach gradients to tensors
            try:
                inp._grad = g
            except Exception:
                pass

        if not retain_graph:
            if tape is self.tape:
                self.tape = GradTape()
            else:
                tape._nodes.clear()
                tape.graph.clear()
                tape._op_index = 0
        return results

    def train(
        self,
        loss_fn: Callable[[], Any],
        epochs: int,
        lr: float = 1e-2,
        params: Optional[List[Any]] = None,
    ) -> None:
        """Train ``params`` by repeatedly evaluating ``loss_fn``.

        This thin wrapper delegates to :meth:`AutogradProcess.training_loop`
        so callers can quickly execute self-contained optimisation runs using
        the existing :class:`GradTape`.  The returned
        :class:`~src.common.tensors.autograd_process.AutogradProcess` carries
        the full forward/backward graphs and execution schedules for the final
        iteration which can be used for further analysis or replay.
        """

        if params is None:
            params = []

        from .autograd_process import AutogradProcess

        proc = AutogradProcess(self.tape)
        proc.training_loop(loss_fn, params, steps=epochs, lr=lr)
        return proc


autograd = Autograd()

try:  # pragma: no cover
    from .abstraction import AbstractTensor

    AbstractTensor.autograd = autograd  # type: ignore[attr-defined]
    AbstractTensor._requires_grad = False  # default internal flag
except Exception:  # pragma: no cover
    pass

