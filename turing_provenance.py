"""
 turing_provenance.py ────────────────────────────────────────────────────────────
 Graph‑level instrumentation layer for the abstract `Turing` scaffold.

 Usage
 -----
     from turing import Turing, Hooks
     from turing_provenance import ProvenanceGraph, instrument_hooks

     raw_hooks   = Hooks(...your backend funcs...)
     graph       = ProvenanceGraph()
     prov_hooks  = instrument_hooks(raw_hooks, graph)

     tm = Turing(prov_hooks)
     ... run algorithms ...

 After execution, the `graph` object contains a full provenance DAG of every
 primitive call the `Turing` instance performed.

 • Each *primitive* invocation becomes a node.
 • Edges record data‑flow ("x feeds into op #7" etc.).
 • The scheme is backend‑agnostic: opaque carrier objects are accepted; only
   their `id()` is used for linkage so no heavy copies occur.

 This file is purely infrastructural: **no concrete backend, no tests**.
"""

from __future__ import annotations

import itertools
import dataclasses
from typing import Any, Callable, Dict, List, Tuple
from turing import Hooks, BitstringProtocol, BackendMissing  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
#  Provenance graph datastructures
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ProvNode:
    idx:        int
    op:         str                     # e.g. 'nand', 'sigma_L'
    args:       Tuple[int, ...]         # id(arg) per positional arg
    kwargs:     Dict[str, Any]
    out_obj_id: int                     # id(result)

@dataclasses.dataclass
class ProvEdge:
    src_idx: int        # node where the value is produced
    dst_idx: int        # node that consumes the value
    arg_pos: int        # which positional slot in dst (0,1,2,…)

class ProvenanceGraph:
    """Simple in‑memory DAG recorder."""

    def __init__(self):
        self._nodes: List[ProvNode] = []
        self._edges: List[ProvEdge] = []
        self._next_idx = itertools.count().__next__  # cheap auto‑increment
        # map object id -> last producer node idx
        self._producer: Dict[int, int] = {}

    # ----- public API ---------------------------------------------------------

    @property
    def nodes(self) -> List[ProvNode]:
        return self._nodes

    @property
    def edges(self) -> List[ProvEdge]:
        return self._edges

    def add_call(self, op: str, args: Tuple[Any, ...], kwargs: Dict[str, Any], result: Any):
        idx = self._next_idx()
        arg_ids = tuple(id(a) for a in args)
        out_id  = id(result)
        node    = ProvNode(idx, op, arg_ids, kwargs, out_id)
        self._nodes.append(node)

        # build edges arg -> this node
        for pos, a in enumerate(args):
            src = self._producer.get(id(a))
            if src is not None:
                self._edges.append(ProvEdge(src, idx, pos))
        # mark result as produced by this node
        self._producer[out_id] = idx
        return result  # pass through

# ──────────────────────────────────────────────────────────────────────────────
#  Hook wrapper utility
# ──────────────────────────────────────────────────────────────────────────────

_PRIMITIVE_NAMES = (
    'nand', 'sigma_L', 'sigma_R', 'concat', 'slice', 'mu',
    'length', 'zeros'
)

def instrument_hooks(raw: Hooks, graph: ProvenanceGraph) -> Hooks:
    """Return a *new* Hooks with every primitive instrumented to log into *graph*."""

    missing = [name for name in _PRIMITIVE_NAMES if getattr(raw, name, None) is None]
    if missing:
        raise BackendMissing(f"Raw backend missing hooks: {', '.join(missing)}")

    # helper to wrap callables
    def _wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            graph.add_call(name, args, kwargs, result)
            return result
        wrapper.__name__ = f"prov_{name}"
        return wrapper

    # Build new Hooks dataclass with wrapped functions.
    return Hooks(
        nand      = _wrap('nand',      raw.nand),
        sigma_L   = _wrap('sigma_L',   raw.sigma_L),
        sigma_R   = _wrap('sigma_R',   raw.sigma_R),
        concat    = _wrap('concat',    raw.concat),
        slice     = _wrap('slice',     raw.slice),
        mu        = _wrap('mu',        raw.mu),
        length    = _wrap('length',    raw.length),
        zeros     = _wrap('zeros',     raw.zeros),
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Convenience introspection helpers
# ──────────────────────────────────────────────────────────────────────────────

def graph_as_dot(g: ProvenanceGraph) -> str:
    """Return a Graphviz DOT representation of the provenance graph."""
    out = ["digraph provenance {"]
    for node in g.nodes:
        label = f"{node.idx}: {node.op}"
        out.append(f"  n{node.idx} [label=\"{label}\"];")
    for e in g.edges:
        out.append(f"  n{e.src_idx} -> n{e.dst_idx} [label=\"arg{e.arg_pos}\"];")
    out.append("}")
    return "\n".join(out)
