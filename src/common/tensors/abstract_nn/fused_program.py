"""FusedProgram IR and runner.

This module implements the initial scaffolding for the unified program
intermediate representation described in ``docs/FUSED_PROGRAM_IR.md``.
The IR captures a linear sequence of tensor operations detached from the
Autograd tape and can be replayed deterministically with a ``training``
flag to alter mode sensitive operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import networkx as nx

from ..abstraction import AbstractTensor as AT
from ..graph_translator import GraphTranslator
from ....transmogrifier.ilpscheduler import ILPScheduler


# ---------------------------------------------------------------------------
# Dataclasses mirroring the design document
# ---------------------------------------------------------------------------


@dataclass
class Meta:
    """Per-id snapshot of tensor metadata."""

    shape: Iterable[int] | None = None
    dtype: str | None = None
    device: str | None = None


@dataclass
class OpStep:
    """Single linearised tensor operation."""

    step_id: int
    op_name: str
    input_ids: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)
    result_id: int = -1
    mode_sensitive: bool = False


@dataclass
class FusedProgram:
    """Unified program representation for AbstractTensor graphs."""

    version: int
    feeds: Set[int]
    steps: List[OpStep]
    outputs: Dict[str, int]
    state_in: Set[int] | None = None
    meta: Dict[int, Meta] | None = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_fused_program(
    graph: nx.DiGraph,
    *,
    outputs: Dict[str, int] | None = None,
    version: int = 1,
    scheduler_cls: type[ILPScheduler] = ILPScheduler,
) -> FusedProgram:
    """Construct a :class:`FusedProgram` from ``graph``.

    The ``graph`` is expected to contain ``tensor`` and ``op`` nodes with the
    following minimal attributes:

    - tensor nodes: ``kind='tensor'``, optional ``shape``, ``dtype`` and
      ``device``.
    - op nodes: ``kind='op'``, ``op_name`` (method on ``AbstractTensor``),
      optional ``attrs`` dict and ``mode_sensitive`` bool.

    Feeds are inferred as tensor nodes with no producing op predecessors.
    Steps are linearised according to ASAP levels from ``ILPScheduler``.
    """

    outputs = outputs or {}

    translator = GraphTranslator(graph)
    order = translator.schedule(scheduler_cls)

    feeds: Set[int] = set()
    steps: List[OpStep] = []
    meta: Dict[int, Meta] = {}

    for nid, data in graph.nodes(data=True):
        if data.get("kind") == "tensor":
            pred_ops = [p for p in graph.predecessors(nid) if graph.nodes[p].get("kind") == "op"]
            if not pred_ops:
                if isinstance(nid, int):
                    feeds.add(nid)
            m = Meta(
                shape=tuple(data.get("shape", [])) or None,
                dtype=data.get("dtype"),
                device=data.get("device"),
            )
            if any(v is not None for v in (m.shape, m.dtype, m.device)):
                if isinstance(nid, int):
                    meta[nid] = m

    for nid in order:
        data = graph.nodes[nid]
        if data.get("kind") != "op":
            continue
        if not isinstance(nid, int):
            raise TypeError("Op node ids must be integers")
        input_ids = [
            int(tid)
            for tid in graph.predecessors(nid)
            if graph.nodes[tid].get("kind") == "tensor"
        ]
        result_candidates = [
            int(tid)
            for tid in graph.successors(nid)
            if graph.nodes[tid].get("kind") == "tensor"
        ]
        result_id = result_candidates[0] if result_candidates else nid
        step = OpStep(
            step_id=nid,
            op_name=str(data.get("op_name")),
            input_ids=input_ids,
            attrs=dict(data.get("attrs", {})),
            result_id=result_id,
            mode_sensitive=bool(data.get("mode_sensitive", False)),
        )
        steps.append(step)

    return FusedProgram(
        version=version,
        feeds=feeds,
        steps=steps,
        outputs=outputs,
        meta=meta or None,
    )


# ---------------------------------------------------------------------------
# Program runner
# ---------------------------------------------------------------------------


class ProgramRunner:
    """Execute a :class:`FusedProgram` using ``AbstractTensor`` ops."""

    def __init__(self, program: FusedProgram) -> None:
        self.program = program

    def __call__(
        self,
        feeds: Dict[int, AT],
        *,
        training: bool = False,
    ) -> Dict[str, AT]:
        prog = self.program
        store: Dict[int, AT] = {}

        missing = prog.feeds - set(feeds)
        if missing:
            raise KeyError(f"Missing feeds: {sorted(missing)}")
        store.update(feeds)

        for step in prog.steps:
            args = [store[i] for i in step.input_ids]
            cls = args[0].__class__ if args else AT
            fn = getattr(cls, step.op_name)
            if step.mode_sensitive:
                result = fn(*args, training=training, **step.attrs)
            else:
                result = fn(*args, **step.attrs)
            store[step.result_id] = result

        return {name: store[i] for name, i in prog.outputs.items() if i in store}


__all__ = [
    "Meta",
    "OpStep",
    "FusedProgram",
    "build_fused_program",
    "ProgramRunner",
]
