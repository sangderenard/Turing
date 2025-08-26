from __future__ import annotations

"""High level processing utilities for :mod:`autograd`.

This module provides :class:`AutogradProcess`, a companion to the lightweight
``autograd`` implementation.  It aggregates optional post processing stages
such as forward/backward graph extraction, caching requirements, execution
schedules and a tiny training loop.  The intent is to retain the most detailed
representation of an abstract tensor computation so that higher level tooling
can introspect or render it.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

import networkx as nx
import pandas as pd

from .autograd import GradTape
from .abstraction import AbstractTensor
from .graph_translator import GraphTranslator


@dataclass
class AutogradProcess:
    """Coordinate post processing for a :class:`GradTape`.

    Parameters
    ----------
    tape:
        The tape whose recorded operations will be analysed.
    """

    tape: GradTape
    forward_graph: nx.DiGraph | None = None
    backward_graph: nx.DiGraph | None = None
    forward_schedule: List[int] = field(default_factory=list)
    backward_schedule: List[int] = field(default_factory=list)
    stages: Dict[str, List[int]] = field(default_factory=dict)
    cache: set[int] = field(default_factory=set)
    training_log: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Graph and schedule construction
    # ------------------------------------------------------------------
    def build(self, result: Any) -> None:
        """Populate forward/backward graphs and schedules for ``result``."""

        self.forward_graph = self.tape.export_forward_graph()
        self.backward_graph = self.tape.export_backward_graph(result)

        # Use the project's ILP scheduler to determine execution order and
        # layer assignments for both graphs.
        f_sched = GraphTranslator(self.forward_graph)
        self.forward_schedule = f_sched.schedule()
        b_sched = GraphTranslator(self.backward_graph)
        self.backward_schedule = b_sched.schedule()

        self.cache = self.tape.required_cache(result)
        self.stages["forward"] = self.forward_schedule
        self.stages["backward"] = self.backward_schedule

    # ------------------------------------------------------------------
    # Staging utilities
    # ------------------------------------------------------------------
    def stage(self, label: str, nodes: Iterable[int]) -> None:
        """Associate ``nodes`` with the stage ``label``."""

        self.stages[label] = list(nodes)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def training_loop(
        self,
        forward_fn: Callable[[], Any],
        params: Iterable[Any],
        *,
        steps: int = 1,
        lr: float = 0.01,
    ) -> None:
        """Run a simple gradient-descent training loop.

        ``forward_fn`` is expected to build a computation using ``params`` and
        return a scalar loss tensor.  After each iteration the tape is cleared
        and rebuilt to keep the recorded graph compact.
        """

        params = list(params)
        for step in range(steps):
            self.tape._nodes.clear()
            self.tape.graph.clear()
            for p in params:
                self.tape.create_tensor_node(p)
            result = forward_fn()
            if isinstance(result, tuple):
                loss, meta_loss = result
            else:
                loss, meta_loss = result, float(result.item())
            self.tape.mark_loss(loss)
            grads = AbstractTensor.autograd.grad(loss, params, retain_graph=True)
            for p, g in zip(params, grads):
                p.data = p.data - lr * g
            self.training_log.append({"step": step, "loss": float(meta_loss)})

        # Use the final iteration to populate graphs and schedules
        self.build(loss)
        self.tape._nodes.clear()
        self.tape.graph.clear()

    # ------------------------------------------------------------------
    # Tabulation helpers
    # ------------------------------------------------------------------
    def _stage_of(self, nid: int) -> str | None:
        for label, nodes in self.stages.items():
            if nid in nodes:
                return label
        return None

    def summary_table(self) -> Dict[str, pd.DataFrame]:
        """Return tables for graph nodes and training metadata."""

        if self.forward_graph is None or self.backward_graph is None:
            raise RuntimeError("build() must be called before requesting a table")

        f_index = {tid: i for i, tid in enumerate(self.forward_schedule)}
        b_index = {tid: i for i, tid in enumerate(self.backward_schedule)}
        rows: List[Dict[str, Any]] = []
        for tid, data in self.forward_graph.nodes(data=True):
            rows.append(
                {
                    "id": tid,
                    "op": data.get("op"),
                    "forward_order": f_index.get(tid),
                    "backward_order": b_index.get(tid),
                    "cached": tid in self.cache or bool(data.get("cached")),
                    "stage": self._stage_of(tid),
                    "param_id": data.get("param_id"),
                    "loss": bool(data.get("loss")),
                }
            )
        graph_df = pd.DataFrame(rows).sort_values("forward_order").reset_index(drop=True)
        train_df = pd.DataFrame(self.training_log)
        return {"graph": graph_df, "training": train_df}

    # ------------------------------------------------------------------
    # Process tree
    # ------------------------------------------------------------------
    def process_tree(self) -> nx.DiGraph:
        """Return a tree organised by stage labels."""

        if self.forward_graph is None:
            raise RuntimeError("build() must be called before requesting a tree")

        tree = nx.DiGraph()
        tree.add_node("training")
        for label, nodes in self.stages.items():
            tree.add_node(label)
            tree.add_edge("training", label)
            for nid in nodes:
                op = self.forward_graph.nodes[nid].get("op") if nid in self.forward_graph else None
                tree.add_node(nid, op=op)
                tree.add_edge(label, nid)
        return tree
