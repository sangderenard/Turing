
from __future__ import annotations

# Global flag to control backend autograd deferral
ALLOW_BACKEND_AUTOGRAD_DEFER = False

def set_backend_autograd_defer(allow: bool):
    """Set whether backend autograd deferral is allowed at runtime."""
    global ALLOW_BACKEND_AUTOGRAD_DEFER
    ALLOW_BACKEND_AUTOGRAD_DEFER = allow


def requires_grad_(self, requires_grad=True):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'requires_grad_'):
        self.data.requires_grad_(requires_grad)
        return self
    raise NotImplementedError("requires_grad_ not supported for this backend")


@property
def requires_grad(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'requires_grad'):
        return self.data.requires_grad
    return False


def backward(self, *args, **kwargs):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'backward'):
        return self.data.backward(*args, **kwargs)
    raise NotImplementedError("backward not supported for this backend")


@property
def grad(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'grad'):
        return self.data.grad
    return None


def detach(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'detach'):
        result = type(self)(track_time=self.track_time)
        result.data = self.data.detach()
        return result
    raise NotImplementedError("detach not supported for this backend")


@property
def is_leaf(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'is_leaf'):
        return self.data.is_leaf
    return True


def retain_grad(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'retain_grad'):
        self.data.retain_grad()
        return self
    raise NotImplementedError("retain_grad not supported for this backend")


@property
def grad_fn(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'grad_fn'):
        return self.data.grad_fn
    return None


def zero_grad(self):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'grad') and self.data.grad is not None:
        self.data.grad.zero_()
        return self
    raise NotImplementedError("zero_grad not supported for this backend")


def register_hook(self, hook):
    if ALLOW_BACKEND_AUTOGRAD_DEFER and hasattr(self.data, 'register_hook'):
        return self.data.register_hook(hook)
    raise NotImplementedError("register_hook not supported for this backend")


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

import math

try:  # NumPy is an optional dependency for the repository
    import numpy as np
except Exception:  # pragma: no cover - tested in environments without numpy
    np = None  # type: ignore


@dataclass
class GradNode:
    """Single operation in the automatic differentiation graph.

    Attributes
    ----------
    op:
        Name of the primitive operator that produced the tensor.
    parents:
        List of ``(tensor_id, arg_pos)`` pairs describing the incoming
        edges for this node. ``tensor_id`` is the ``id()`` of the input
        tensor, ``arg_pos`` is the positional index in the forward call.
    backward:
        Callable implementing the local backward rule. It receives the
        gradient w.r.t. the node's output and must return an iterable of
        gradients matching ``parents`` order.
    """

    op: str
    parents: List[Tuple[int, int]]
    backward: Callable[[Any], Iterable[Any]]


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
        backward_fn: Callable[[Any], Iterable[Any]],
    ) -> Any:
        """Append a new node representing ``op`` to the tape.

        Parameters
        ----------
        op:
            Name of the operation.
        inputs:
            Iterable of input tensors from which the result was computed.
        result:
            The output tensor produced by ``op``.
        backward_fn:
            Function implementing the local backward rule for ``op``.

        Returns
        -------
        Any
            Passes ``result`` through unchanged to ease functional style.
        """

        parent_ids = [(id(t), pos) for pos, t in enumerate(inputs)]
        node = GradNode(op=op, parents=parent_ids, backward=backward_fn)
        self._nodes[id(result)] = node
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


PRIMITIVE_BACKWARD: Dict[str, Callable[..., Tuple[Any, ...]]] = {
    "add": bw_add,
    "radd": bw_add,
    "sub": bw_sub,
    "rsub": lambda g, x, y: bw_sub(g, y, x),
    "mul": bw_mul,
    "rmul": bw_mul,
    "truediv": bw_truediv,
    "rtruediv": lambda g, x, y: bw_truediv(g, y, x),
    "pow": bw_pow,
    "rpow": lambda g, x, y: bw_pow(g, y, x),
}


# ----------------------------------------------------------------------------
# Autograd engine
# ----------------------------------------------------------------------------


class Autograd:
    """Very small reverse-mode autodiff engine."""

    def __init__(self) -> None:
        self.tape = GradTape()

    def record(self, op: str, inputs: Iterable[Any], result: Any) -> Any:
        bw = PRIMITIVE_BACKWARD.get(op)
        if bw is None:
            return result
        inputs = list(inputs)
        data_inputs = [x.data if hasattr(x, "data") else x for x in inputs]

        def backward_fn(grad_out: Any) -> Iterable[Any]:
            return bw(grad_out, *data_inputs)

        self.tape.record(op, inputs, result, backward_fn)
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
            parent_grads = node.backward(grad_out)
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
                    results.append(g.tolist())
                else:
                    results.append(g)

        if not retain_graph:
            self.tape = GradTape()
        return results


autograd = Autograd()

try:  # pragma: no cover
    from .abstraction import AbstractTensor

    AbstractTensor.autograd = autograd  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

