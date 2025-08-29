"""
graph_runner.py
---------------

Simple runner that consumes a graph capsule, computes ILP schedules over the
fused graph, and executes a training loop reporting loss and parameter stats.

Execution semantics
- Scheduling: uses ILPScheduler on an adapted process graph to obtain ASAP
  levels. Levels are iterated each epoch to reflect timing; actual numeric
  compute still uses the AbstractTensor model inside the capsule.
- Training: standard forward → loss → backward → Adam step using existing
  autograd and optimizer utilities. Parameters are updated in place.
"""

from __future__ import annotations

from typing import Any, Dict
import math
import networkx as nx

from ..abstraction import AbstractTensor as AT
from ..autograd import autograd
from .optimizer import Adam
from .utils import as_list
from ....transmogrifier.ilpscheduler import ILPScheduler


def _to_process_graph(nx_graph: nx.DiGraph):
    G = nx.DiGraph()
    for nid, data in nx_graph.nodes(data=True):
        attrs = {k: v for k, v in data.items() if k not in {"parents", "children"}}
        attrs.setdefault("parents", [])
        attrs.setdefault("children", [])
        attrs.setdefault("label", str(nid))
        G.add_node(nid, **attrs)
    for u, v in nx_graph.edges():
        G.add_edge(u, v)
        G.nodes[u]["children"].append((v, "dep"))
        G.nodes[v]["parents"].append((u, "dep"))
    return type("Proc", (), {"G": G, "role_schemas": {}})()


class GraphRunner:
    def __init__(self, capsule, *, epochs: int = 100, lr: float = 1e-2, epsilon: float = 1e-6) -> None:
        if capsule.fused_graph is None:
            raise ValueError("Capsule has no fused graph; build it first.")
        self.capsule = capsule
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        # Adapt to process graph and precompute ASAP levels
        self.proc = _to_process_graph(capsule.fused_graph)
        self.scheduler = ILPScheduler(self.proc)
        self.asap = self.scheduler.compute_levels("asap", "dependency")
        # Group nodes by level for reporting
        self.levels: Dict[int, list[Any]] = {}
        for nid, lvl in self.asap.items():
            self.levels.setdefault(lvl, []).append(nid)
        # Optimizer on live parameters
        params = list(self.capsule.layer.parameters())
        self.opt = Adam(params, lr=self.lr)

    def run(self) -> None:
        cap = self.capsule
        layer = cap.layer
        X, Y = cap.X, cap.Y
        if layer is None or X is None or Y is None:
            raise ValueError("Capsule is missing model or data for runner.")

        def mse(a, b):
            return ((a - b) ** 2).mean()

        for epoch in range(1, self.epochs + 1):
            # Simulate scheduled execution bands (report only)
            for lvl in sorted(self.levels):
                nodes = self.levels[lvl]
                # Filter out control nodes for reporting clarity
                core = [n for n in nodes if not (isinstance(n, str) and n.startswith("cond_"))]
                print(f"[sched] level {lvl}: {len(core)} nodes")

            # Reset tape for a clean capture per step
            autograd.tape = autograd.__class__().tape

            # Zero grads
            for p in layer.parameters():
                if hasattr(p, "zero_grad"):
                    p.zero_grad()
                elif hasattr(p, "grad") and p.grad is not None:
                    p._grad = AT.zeros_like(p.grad)

            # Forward + loss
            pred = layer.forward(X)
            autograd.tape.annotate(pred, label="GraphRunner.pred")
            loss = mse(pred, Y)
            autograd.tape.annotate(loss, label="GraphRunner.loss")
            # Backward and step
            loss.backward()
            params = list(layer.parameters())
            grads = [p.grad for p in params]
            try:
                for i, g in enumerate(grads):
                    if g is not None:
                        autograd.tape.annotate(g, label=f"GraphRunner.grad[{i}]")
            except Exception:
                pass
            new_params = self.opt.step(params, grads)
            for p, np_ in zip(params, new_params):
                AT.copyto(p, np_)

            # Report
            loss_val = float(loss.item())
            # basic parameter summary: L2 norm of first weight matrix if exists
            wnorm = float('nan')
            W = getattr(layer, 'W', None)
            if W is None:
                W = params[0] if params else None
            if W is not None:
                wnorm_t = None
                # Prefer AbstractTensor.linalg.norm which returns a tensor-like
                try:
                    wnorm_t = AT.norm(W)
                except Exception:
                    wnorm_t = None
                # Convert to python float robustly
                if wnorm_t is not None:
                    if hasattr(wnorm_t, 'item'):
                        try:
                            wnorm = float(wnorm_t.item())
                        except Exception:
                            pass
                    else:
                        try:
                            wnorm = float(wnorm_t)
                        except Exception:
                            pass
                # Fallback: compute via Python if conversion failed
                if math.isnan(wnorm):
                    try:
                        def _sum_sq(x):
                            if isinstance(x, list):
                                return sum(_sum_sq(v) for v in x)
                            try:
                                v = float(x)
                            except Exception:
                                return 0.0
                            return v * v
                        wnorm = math.sqrt(_sum_sq(as_list(W)))
                    except Exception:
                        pass
            print(f"[epoch {epoch}] loss={loss_val:.3e}  ||W||={wnorm:.3e}")
            if loss_val <= self.epsilon and loss_val == loss_val:  # second check guards NaN
                print(f"Converged: loss <= {self.epsilon:.3e} at epoch {epoch}")
                break
