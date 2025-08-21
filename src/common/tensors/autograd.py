from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple


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
