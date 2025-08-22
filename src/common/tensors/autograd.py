
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
    result = type(self)(track_time=self.track_time)
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

# Integrate the lightweight backward registry so that backward rules are
# resolved dynamically rather than being baked into the tape at record time.
from .backward import BACKWARD_REGISTRY

import math

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

    # ------------------------------------------------------------------
    # recording utilities
    # ------------------------------------------------------------------
    def record(
        self,
        op: str,
        inputs: Iterable[Any],
        result: Any,
    ) -> Any:
        """Append a new node representing ``op`` to the tape."""

        inputs = list(inputs)
        parent_ids = [(id(t), pos) for pos, t in enumerate(inputs)]
        ctx = {
            "inputs": [x.data if hasattr(x, "data") else x for x in inputs],
            "result": result.data if hasattr(result, "data") else result,
            "input_shapes": [getattr(x, "shape", None) for x in inputs],
            "result_shape": getattr(result, "shape", None),
        }
        node = GradNode(op=op, parents=parent_ids, ctx=ctx)
        self._nodes[id(result)] = node
        # Attach graph references to the tensor itself so that each tensor knows
        # about its generating node and tape.
        try:
            result._grad_node = node  # type: ignore[attr-defined]
            result._grad_tape = self  # type: ignore[attr-defined]
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

# ----------------------------------------------------------------------------
# Primitive backward rules
# ----------------------------------------------------------------------------

def _to_numpy(x: Any) -> "np.ndarray":
    """Best effort conversion of ``x`` to a NumPy array."""

    if np is None:
        raise RuntimeError("NumPy is required for autograd operations")
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.asarray(x)


def _reduce_like(grad: "np.ndarray", like: Any) -> "np.ndarray":
    """Reduce ``grad`` so that its shape matches ``like``."""

    target = _to_numpy(like)
    g = grad
    while g.ndim > target.ndim:
        g = g.sum(axis=0)
    for axis, size in enumerate(target.shape):
        if size == 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(target.shape)


def bw_add(grad_out: Any, x: Any, y: Any) -> Tuple[Any, Any]:
    g = _to_numpy(grad_out)
    return _reduce_like(g, x), _reduce_like(g, y)


def bw_sub(grad_out: Any, x: Any, y: Any) -> Tuple[Any, Any]:
    g = _to_numpy(grad_out)
    return _reduce_like(g, x), _reduce_like(-g, y)


def bw_mul(grad_out: Any, x: Any, y: Any) -> Tuple[Any, Any]:
    g = _to_numpy(grad_out)
    gx = g * _to_numpy(y)
    gy = g * _to_numpy(x)
    return _reduce_like(gx, x), _reduce_like(gy, y)


def bw_truediv(grad_out: Any, x: Any, y: Any) -> Tuple[Any, Any]:
    g = _to_numpy(grad_out)
    a, b = _to_numpy(x), _to_numpy(y)
    gx = g / b
    gy = -g * a / (b * b)
    return _reduce_like(gx, x), _reduce_like(gy, y)


def bw_pow(grad_out: Any, x: Any, y: Any) -> Tuple[Any, Any]:
    g = _to_numpy(grad_out)
    a, b = _to_numpy(x), _to_numpy(y)
    gx = g * b * np.power(a, b - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gy = g * np.power(a, b) * np.log(np.where(a == 0, 1.0, a))
    return _reduce_like(gx, x), _reduce_like(gy, y)


# Register primitive backward rules with the shared registry.  This keeps the
# actual backward implementations in one place and allows other parts of the
# system to consult the same registry when extending the set of supported
# operators.
BACKWARD_REGISTRY.register("add", bw_add)
BACKWARD_REGISTRY.register("radd", bw_add)
BACKWARD_REGISTRY.register("sub", bw_sub)
BACKWARD_REGISTRY.register("rsub", lambda g, x, y: bw_sub(g, y, x))
BACKWARD_REGISTRY.register("mul", bw_mul)
BACKWARD_REGISTRY.register("rmul", bw_mul)
BACKWARD_REGISTRY.register("truediv", bw_truediv)
BACKWARD_REGISTRY.register("rtruediv", lambda g, x, y: bw_truediv(g, y, x))
BACKWARD_REGISTRY.register("pow", bw_pow)
BACKWARD_REGISTRY.register("rpow", lambda g, x, y: bw_pow(g, y, x))


# ----------------------------------------------------------------------------
# Autograd engine
# ----------------------------------------------------------------------------


class Autograd:
    """Very small reverse-mode autodiff engine."""

    def __init__(self) -> None:
        self.tape = GradTape()

    def record(self, op: str, inputs: Iterable[Any], result: Any) -> Any:
        """Record an operation on the tape if a backward rule exists."""

        if op not in BACKWARD_REGISTRY._methods:
            return result
        self.tape.record(op, inputs, result)
        return result

    def grad(
        self,
        output: Any,
        inputs: Iterable[Any],
        grad_outputs: Any | None = None,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> List[Any]:
        if np is None:
            raise RuntimeError("NumPy is required for autograd operations")

        inputs = list(inputs)
        out_grad = grad_outputs
        if out_grad is None:
            out_grad = np.ones_like(output.data if hasattr(output, "data") else output)

        grad_map: Dict[int, Any] = {id(output): _to_numpy(out_grad)}

        for tid, node in self.tape.traverse(output):
            grad_out = grad_map.get(tid)
            if grad_out is None:
                continue
            bw = BACKWARD_REGISTRY._methods.get(node.op)
            if bw is None:
                continue
            parent_grads = bw(grad_out, *node.ctx["inputs"])
            for (pid, _), g in zip(node.parents, parent_grads):
                if g is None:
                    continue
                g = _to_numpy(g)
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
            self.tape = GradTape()
        return results


autograd = Autograd()

try:  # pragma: no cover
    from .abstraction import AbstractTensor

    AbstractTensor.autograd = autograd  # type: ignore[attr-defined]
    AbstractTensor._requires_grad = False  # default internal flag
except Exception:  # pragma: no cover
    pass

