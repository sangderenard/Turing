"""
graph_capsule.py
----------------

Build a tiny linear regression network, execute one forward (and optional
backward) pass, and materialize a fused forward+backward graph augmented with
explicit conditional control nodes that a runner/compiler can consume:

- cond_backward: boolean node that indicates whether the backward subgraph
  should be executed.
- cond_tape: boolean node that indicates whether a runner should record a tape
  (tracking) for this execution (use False for pure inference).
- cond_materialize: enum node that indicates whether tensors should be kept as
  AbstractTensor wrappers ("abstract") or stripped to backend payloads ("pure").

The capsule exposes the fused graph plus a small bundle of side data
(tensor/grad stores, parameter ids, loss id) without redefining any operators
or execution rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import networkx as nx

from ..abstraction import AbstractTensor as AT
from ..autograd import autograd
from ..autograd_process import AutogradProcess

# Reuse the minimal linear-regression data builder
from .linear_layer_demo import build_synthetic_regression
from .core import Linear


@dataclass
class GraphCapsuleConfig:
    run_backward: bool = True
    keep_tape: bool = True
    materialize: str = "abstract"  # "abstract" | "pure"


class NNGraphCapsule:
    """Self-contained graph capsule for a tiny linear NN.

    - Captures forward/backward graphs from the autograd tape.
    - Fuses them and injects a control node with config flags.
    - Optionally materializes tensors to backend-native payloads.
    """

    def __init__(self, config: GraphCapsuleConfig | None = None) -> None:
        self.config = config or GraphCapsuleConfig()
        self.proc: AutogradProcess | None = None
        self.fused_graph: nx.DiGraph | None = None
        self.tensor_store: Dict[int, Any] = {}
        self.grad_store: Dict[int, Any] = {}
        self.loss_tid: int | None = None
        self.params_tids: Tuple[int, ...] | None = None
        # Expose model/data for runners
        self.layer = None   # legacy alias for single-layer demos
        self.model = None   # generic model reference
        self.X = None       # legacy alias for inputs
        self.Y = None       # legacy alias for targets
        self.inputs = None
        self.targets = None

    def _add_control_nodes(self, combined: nx.DiGraph) -> nx.DiGraph:
        g = combined.copy()

        # Backward execution condition node
        g.add_node(
            "cond_backward",
            kind="control",
            op="cond",
            name="run_backward",
            value=bool(self.config.run_backward),
            dtype="bool",
        )
        # Gate the backward subgraph by pointing to loss as its root
        if g.has_node("loss"):
            g.add_edge("cond_backward", "loss")

        # Tape tracking condition node (for runner to honor during execution)
        g.add_node(
            "cond_tape",
            kind="control",
            op="cond",
            name="record_tape",
            value=bool(self.config.keep_tape),
            dtype="bool",
        )

        # Materialization policy node (enum)
        g.add_node(
            "cond_materialize",
            kind="control",
            op="cond",
            name="materialize",
            value=str(self.config.materialize),
            dtype="enum[abstract|pure]",
        )

        return g

    def _materialize_tensor(self, t: Any) -> Any:
        if self.config.materialize != "pure":
            return t
        # Iteratively unwrap AbstractTensor -> .data until reaching a
        # backend-native payload to ensure no wrappers remain.
        obj = t
        try:
            from ..abstraction import AbstractTensor as _AT
        except Exception:
            _AT = AT  # fallback
        depth = 0
        while isinstance(obj, _AT) and hasattr(obj, "data") and depth < 8:
            nxt = getattr(obj, "data", None)
            if nxt is None:
                break
            obj = nxt
            depth += 1
        return obj

    def build_from_linear(self, *, N: int = 64, in_dim: int = 4, out_dim: int = 3) -> None:
        # Fresh tape for a clean capture
        autograd.tape = autograd.__class__().tape  # reset to an empty tape instance

        # Dataset and model
        X, Y, W_true, b_true = build_synthetic_regression(N=N, in_dim=in_dim, out_dim=out_dim)
        like = AT.get_tensor()
        layer = Linear(in_dim=in_dim, out_dim=out_dim, like=like, init="xavier")

        # Forward
        pred = layer.forward(X)
        loss = ((pred - Y) ** 2).mean()

        # Backward (optional) – compute grads for current layer params
        params = list(layer.parameters())
        # Mark the loss on the tape so the fused graph has a 'loss' sentinel
        autograd.tape.mark_loss(loss)
        if self.config.run_backward:
            autograd.grad(loss, params, retain_graph=True, allow_unused=False)

        # Build fused graphs via AutogradProcess
        proc = AutogradProcess(autograd.tape)
        proc.build(loss)
        self.proc = proc
        self.fused_graph = self._add_control_nodes(proc.combined_graph)

        # Persist tensor/grad stores according to materialization policy
        # Use the tape's tensor references (ids stable across graph nodes)
        for tid, tref in autograd.tape._tensor_refs.items():
            self.tensor_store[tid] = self._materialize_tensor(tref)

        # Keep grads for parameters if they exist
        self.params_tids = tuple(id(p) for p in params)
        for p in params:
            if getattr(p, "grad", None) is not None:
                self.grad_store[id(p)] = self._materialize_tensor(p.grad)

        # Locate and record loss tid for convenience
        self.loss_tid = getattr(proc.tape, "_loss_id", None)

        # Optionally drop the live tape metadata to simulate capture-only mode
        if not self.config.keep_tape:
            autograd.tape._nodes.clear()
            autograd.tape.graph.clear()
            autograd.tape._tensor_refs.clear()

        # Stash model and data for runner use
        self.layer = layer
        self.model = layer
        self.X = X
        self.Y = Y
        self.inputs = X
        self.targets = Y

    # ----------------------------- generic path -----------------------------
    def build_from_model(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        *,
        loss_fn: Any | None = None,
        params: list[Any] | None = None,
    ) -> None:
        """Capture graphs for an arbitrary model + dataset.

        Parameters
        ----------
        model: Any
            Object exposing ``parameters()`` and a ``forward(inputs)`` method (or is callable).
        inputs: Any
            Model inputs tensor(s) on the AbstractTensor surface.
        targets: Any
            Supervision targets used by the loss function.
        loss_fn: callable, optional
            Function taking (pred, targets) -> loss. Defaults to MSE if None.
        params: list, optional
            Explicit parameter list. Defaults to list(model.parameters()).
        """
        # Fresh tape
        autograd.tape = autograd.__class__().tape

        # Ensure inputs/targets/params are attached to the current tape so that
        # operator recording uses this tape (and not a stale one captured on the
        # tensors when they were created earlier).
        def _attach_to_tape(obj: Any) -> None:
            try:
                from ..abstraction import AbstractTensor as _AT
            except Exception:
                _AT = None  # type: ignore
            if _AT is not None and isinstance(obj, _AT):
                try:
                    obj._tape = autograd.tape  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    autograd.tape.create_tensor_node(obj)
                except Exception:
                    pass
                return
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    _attach_to_tape(it)
            elif isinstance(obj, dict):
                for it in obj.values():
                    _attach_to_tape(it)

        _attach_to_tape(inputs)
        # Targets typically do not require grad; attach for consistency only.
        _attach_to_tape(targets)

        # Resolve forward
        if hasattr(model, "forward") and callable(getattr(model, "forward")):
            pred = model.forward(inputs)
        elif callable(model):
            pred = model(inputs)
        else:
            raise TypeError("model must be callable or provide a .forward() method")

        # Loss
        if loss_fn is None:
            loss = ((pred - targets) ** 2).mean()
        else:
            loss = loss_fn(pred, targets)

        # Params
        if params is None:
            if hasattr(model, "parameters") and callable(getattr(model, "parameters")):
                params = list(model.parameters())
            else:
                params = []

        # Attach params to current tape and zero grads to avoid stale values from
        # previous runs/tapes. Ensure requires_grad is set when applicable.
        for p in params:
            _attach_to_tape(p)
            try:
                if not getattr(p, "requires_grad", False) and hasattr(p, "requires_grad_"):
                    p.requires_grad_(True)
            except Exception:
                pass
            try:
                # Prefer zero_grad with clear_cache to drop any cached internals
                p.zero_grad(clear_cache=True)
            except TypeError:
                try:
                    p.zero_grad()
                except Exception:
                    try:
                        p._grad = None  # type: ignore[attr-defined]
                    except Exception:
                        pass

        # Mark and optionally backward with diagnostics for missing grads
        autograd.tape.mark_loss(loss)
        if self.config.run_backward and params:
            # Proactively analyze graph connectivity before calling grad
            try:
                bwd_graph = autograd.tape.export_backward_graph(loss)
            except Exception:
                bwd_graph = None
            missing: list[tuple[int, str]] = []
            if bwd_graph is not None:
                for p in params:
                    pid = id(p)
                    if not bwd_graph.has_node(pid):
                        lbl = getattr(p, "_label", None) or getattr(p, "shape", None) or "param"
                        missing.append((pid, str(lbl)))
            if missing:
                # Deep-dive: was the param even recorded on the forward tape?
                fwd_nodes = set(n for n, d in autograd.tape.graph.nodes(data=True) if d.get("kind") == "tensor")
                details = []
                for pid, lbl in missing:
                    present = pid in fwd_nodes
                    # Find ops that reference this param id as an input
                    consumers = []
                    for rid, node in autograd.tape._nodes.items():
                        ctx = node.ctx
                        ins = ctx.get("inputs", [])
                        in_ids = [id(t) for t in ins]
                        if pid in in_ids:
                            op = node.op
                            consumers.append({
                                "op": op,
                                "result_id": rid,
                                "result_shape": ctx.get("result_shape"),
                                "input_ids": in_ids,
                            })
                    # Path existence from param tensor node to loss tensor node in the global graph
                    try:
                        import networkx as nx
                        has_path = autograd.tape._loss_id is not None and nx.has_path(autograd.tape.graph, pid, autograd.tape._loss_id)
                    except Exception:
                        has_path = None
                    details.append(f"param id={pid} label={lbl} on_forward={present} path_to_loss={has_path} consumers={consumers}")
                # In strict mode, let Autograd.grad perform its own deep diagnostics; otherwise raise here
                if not getattr(autograd, 'strict', False):
                    msg = (
                        "No gradient path to one or more parameters.\n" +
                        "\n".join(details)
                    )
                    raise RuntimeError(msg)
            # If all present in backward graph, compute grads
            autograd.grad(loss, params, retain_graph=True, allow_unused=False)

        # Build fused graphs
        proc = AutogradProcess(autograd.tape)
        proc.build(loss)
        self.proc = proc
        self.fused_graph = self._add_control_nodes(proc.combined_graph)

        # Persist stores
        self.tensor_store.clear(); self.grad_store.clear()
        for tid, tref in autograd.tape._tensor_refs.items():
            self.tensor_store[tid] = self._materialize_tensor(tref)

        self.params_tids = tuple(id(p) for p in params)
        for p in params:
            if getattr(p, "grad", None) is not None:
                self.grad_store[id(p)] = self._materialize_tensor(p.grad)

        self.loss_tid = getattr(proc.tape, "_loss_id", None)

        if not self.config.keep_tape:
            autograd.tape._nodes.clear()
            autograd.tape.graph.clear()
            autograd.tape._tensor_refs.clear()

        # Stash model and data
        self.model = model
        # keep legacy alias for compatibility
        self.layer = model if getattr(self, "layer", None) is None else self.layer
        self.inputs = inputs
        self.targets = targets
        self.X = inputs
        self.Y = targets

    def bundle(self) -> Dict[str, Any]:
        """Return a standardized bundle for runner/compiler consumption."""
        return {
            "graph": self.fused_graph,
            "controls": {
                "run_backward": bool(self.config.run_backward),
                "record_tape": bool(self.config.keep_tape),
                "materialize": str(self.config.materialize),
            },
            "loss_tid": self.loss_tid,
            "param_tids": self.params_tids,
            "tensor_store": self.tensor_store,
            "grad_store": self.grad_store,
        }

    # Lightweight summaries for quick inspection
    def summarize(self) -> None:
        if self.proc is None or self.fused_graph is None:
            print("Capsule is empty; call build_from_linear() first.")
            return
        fg = self.proc.forward_graph
        bg = self.proc.backward_graph
        print("Forward nodes:", 0 if fg is None else fg.number_of_nodes())
        print("Backward nodes:", 0 if bg is None else bg.number_of_nodes())
        print("Fused nodes:", self.fused_graph.number_of_nodes())
        print("Control flags:", {
            "run_backward": self.config.run_backward,
            "keep_tape": self.config.keep_tape,
            "materialize": self.config.materialize,
        })
        if self.loss_tid is not None:
            print("Loss tid:", self.loss_tid)
        if self.params_tids:
            print("Param tids (count):", len(self.params_tids))
