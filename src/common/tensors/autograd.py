
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


def zero_grad(self):  # pragma: no cover - simple helper
    self._grad = None
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

import networkx as nx

# Integrate the lightweight backward registry so that backward rules are
# resolved dynamically rather than being baked into the tape at record time.
from .backward import BACKWARD_REGISTRY

import math
import statistics

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
        elapsed = None
        if start is not None and end is not None:
            elapsed = end - start
        ctx = {
            "inputs": [x.data if hasattr(x, "data") else x for x in inputs],
            "result": result.data if hasattr(result, "data") else result,
            "input_shapes": [getattr(x, "shape", None) for x in inputs],
            "result_shape": getattr(result, "shape", None),
            "start": start,
            "end": end,
            "elapsed": elapsed,
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


# ---------------------------------------------------------------------------
# Primitive backward implementations
# ---------------------------------------------------------------------------


def _arr(x):
    return np.asarray(x)


def bw_add(g, a, b):
    g = _arr(g)
    return g, g


def bw_sub(g, a, b):
    g = _arr(g)
    return g, -g


def bw_mul(g, a, b):
    g, a, b = _arr(g), _arr(a), _arr(b)
    return g * b, g * a


def bw_truediv(g, a, b):
    g, a, b = _arr(g), _arr(a), _arr(b)
    return g / b, -g * a / (b * b)


def bw_pow(g, a, b):
    g, a, b = _arr(g), _arr(a), _arr(b)
    return g * b * (a ** (b - 1)), g * (a ** b) * np.log(a)


def bw_sin(g, x):
    g, x = _arr(g), _arr(x)
    return g * np.cos(x)


def bw_cos(g, x):
    g, x = _arr(g), _arr(x)
    return -g * np.sin(x)


## All primitive backward rules have been removed from this file.
## The BACKWARD_REGISTRY should be populated externally (e.g., in a backend-specific module).


# ----------------------------------------------------------------------------
# Autograd engine
# ----------------------------------------------------------------------------


class Autograd:
    """Very small reverse-mode autodiff engine."""

    def __init__(self) -> None:
        self.tape = GradTape()

    def record(
        self,
        op: str,
        inputs: Iterable[Any],
        result: Any,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> Any:
        """Record an operation on the appropriate tape if supported."""

        if op not in BACKWARD_REGISTRY._methods:
            return result
        tape = getattr(result, "_tape", None)
        if tape is None:
            for t in inputs:
                tape = getattr(t, "_tape", None)
                if tape is not None:
                    break
        if tape is None:
            tape = self.tape
        tape.record(op, inputs, result, start=start, end=end)
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

        for tid, node in tape.traverse(output):
            grad_out = grad_map.get(tid)
            if grad_out is None:
                continue
            bw = BACKWARD_REGISTRY._methods.get(node.op)
            if bw is None:
                continue
            go = grad_out.data if hasattr(grad_out, "data") else grad_out
            parent_grads = bw(go, *node.ctx["inputs"])
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

