
from __future__ import annotations

# Internal, framework-independent gradient bookkeeping helpers.

def requires_grad_(self, requires_grad: bool = True):
    """Enable or disable gradient tracking on this tensor."""
    self._requires_grad = requires_grad
    return self


@property
def requires_grad(self) -> bool:
    return getattr(self, "_requires_grad", False)


@requires_grad.setter
def requires_grad(self, value: bool) -> None:  # pragma: no cover - simple setter
    self._requires_grad = value


def backward(self, *args, **kwargs):  # pragma: no cover - not yet implemented
    raise NotImplementedError(
        "backward not supported; use AbstractTensor.autograd.grad instead"
    )


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

The project purposely keeps the autograd core extremely small.  Only a tiny
subset of primitive arithmetic operators are supported which is sufficient for
educational examples and the pure/numpy backends.  Torch and JAX backends rely
on their native autograd systems and therefore bypass this module entirely.

The implementation below provides two main pieces:

* ``GradTape`` – records a very small computation graph.
* ``Autograd`` – exposes ``record`` and ``grad`` helpers and houses backward
  rules for primitive operators (``add``, ``mul``, ``truediv``, ``pow``, ...).

Backends without a native autograd can call ``Autograd.record`` after executing
an operator to append a node to the tape.  ``Autograd.grad`` then walks the
recorded graph in reverse to accumulate gradients for requested inputs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
from contextlib import contextmanager

import networkx as nx

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

    # ------------------------------------------------------------------
    # node utilities
    # ------------------------------------------------------------------
    def create_tensor_node(self, tensor: Any) -> None:
        """Register ``tensor`` as a root node in the graph."""
        tid = id(tensor)
        self.graph.add_node(tid, kind="tensor")
        try:
            tensor._tape = self  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # annotation utilities
    # ------------------------------------------------------------------
    def annotate(self, tensor: Any, **metadata: Any) -> None:
        """Attach ``metadata`` to ``tensor``'s graph node if present."""
        tid = id(tensor)
        if tid not in self.graph:
            return
        node = self.graph.nodes[tid]
        annotations = node.setdefault("annotations", {})
        annotations.update(metadata)

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

        ctx = {
            "inputs": list(inputs),
            "result": result,
            "inputs_data": [x.data if hasattr(x, "data") else x for x in inputs],
            "result_data": result.data if hasattr(result, "data") else result,
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
        return result

    # ------------------------------------------------------------------
    # metadata utilities
    # ------------------------------------------------------------------
    def annotate(self, tensor: Any, **metadata: Any) -> None:
        """Attach ``metadata`` to the :class:`GradNode` for ``tensor``.

        The elementwise helpers use this to stash bookkeeping information
        (e.g. evaluation mode and scalar lifting).  Missing nodes are ignored
        so callers may unconditionally attempt to annotate intermediates.
        """

        tid = id(tensor)
        node = self._nodes.get(tid)
        if node is None:
            return

        anns = node.ctx.setdefault("annotations", {})
        anns.update(metadata)

        if self.graph.has_node(tid):
            g_anns = self.graph.nodes[tid].setdefault("annotations", {})
            g_anns.update(metadata)

            # Also annotate the generating op node if present.
            try:
                for pred in self.graph.predecessors(tid):
                    if self.graph.nodes[pred].get("kind") == "op":
                        p_anns = self.graph.nodes[pred].setdefault("annotations", {})
                        p_anns.update(metadata)
            except Exception:
                pass

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
            g.add_node(
                tid,
                op=op_name,
                cached=bool(anns.get("cache")),
                metadata=anns,
            )

        # Now add edges between tensors based on the recorded parent links.
        for tid, node in self._nodes.items():
            for pid, _ in node.parents:
                g.add_edge(pid, tid)

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
    """Very small reverse-mode autodiff engine."""

    def __init__(self) -> None:
        self.tape = GradTape()
        self._no_grad_depth = 0

    @contextmanager
    def no_grad(self) -> Generator[None, None, None]:
        """Context manager to temporarily disable gradient recording."""
        self._no_grad_depth += 1
        try:
            yield
        finally:
            self._no_grad_depth -= 1

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
                parent_grads = bw(go, *node.ctx["inputs"])
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
        for inp in inputs:
            g = grad_map.get(id(inp))
            if g is None:
                if allow_unused:
                    results.append(None)
                else:
                    raise ValueError("No gradient found for one of the inputs")
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


autograd = Autograd()

try:  # pragma: no cover
    from .abstraction import AbstractTensor

    AbstractTensor.autograd = autograd  # type: ignore[attr-defined]
    AbstractTensor._requires_grad = False  # default internal flag
except Exception:  # pragma: no cover
    pass

