"""
autograd_audit.py
------------------

Container for fully wiring an arbitrary AbstractTensor-based model into the
autograd auditing chain. Handles:

- Tape lifecycle and attachment (inputs/params/targets → current tape)
- Zeroing grads after reconnections to avoid stale gradients
- Forward/loss capture with optional strict-mode diagnostics
- Fused forward+backward graph build via AutogradProcess
- Optional training loop using the current schedule (eager path)

This centralizes logic previously scattered between demos so that tests and
future codegen runners can reuse a consistent, auditable setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os

from ..abstraction import AbstractTensor as AT
from ..autograd import autograd
from ..autograd_process import AutogradProcess
from .optimizer import Adam


@dataclass
class AuditConfig:
    strict: bool = os.environ.get("AUTOGRAD_STRICT", "0") not in ("0", "false", "False", None)
    run_backward: bool = True
    materialize: str = "abstract"  # reserved for future use


class AutogradAuditSession:
    def __init__(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        *,
        loss_fn: Any | None = None,
        params: List[Any] | None = None,
        config: AuditConfig | None = None,
    ) -> None:
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.loss_fn = loss_fn
        self.params = params
        self.config = config or AuditConfig()

        self.proc: AutogradProcess | None = None
        self.fused_graph = None
        self.loss = None

    # --------------- helpers ---------------
    def _attach_to_tape(self, obj: Any) -> None:
        if isinstance(obj, AT):
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
                self._attach_to_tape(it)
        elif isinstance(obj, dict):
            for it in obj.values():
                self._attach_to_tape(it)

    def _ensure_params(self) -> List[Any]:
        if self.params is not None:
            return list(self.params)
        if hasattr(self.model, "parameters") and callable(getattr(self.model, "parameters")):
            return list(self.model.parameters())
        return []

    def _zero_param_grads(self, params: List[Any]) -> None:
        for p in params:
            try:
                p.zero_grad(clear_cache=True)
            except TypeError:
                try:
                    p.zero_grad()
                except Exception:
                    try:
                        p._grad = None  # type: ignore[attr-defined]
                    except Exception:
                        pass

    # --------------- capture ---------------
    def capture(self) -> None:
        # Fresh tape
        autograd.tape = autograd.__class__().tape

        # Attach inputs/targets to this tape
        self._attach_to_tape(self.inputs)
        self._attach_to_tape(self.targets)

        # Resolve params and attach to tape; ensure requires_grad; zero grads
        params = self._ensure_params()
        for p in params:
            self._attach_to_tape(p)
            try:
                if not getattr(p, "requires_grad", False) and hasattr(p, "requires_grad_"):
                    p.requires_grad_(True)
            except Exception:
                pass
        self._zero_param_grads(params)

        # Forward + loss
        if hasattr(self.model, "forward") and callable(getattr(self.model, "forward")):
            pred = self.model.forward(self.inputs)
        elif callable(self.model):
            pred = self.model(self.inputs)
        else:
            raise TypeError("model must be callable or provide a .forward() method")

        if self.loss_fn is None:
            loss = ((pred - self.targets) ** 2).mean()
        else:
            loss = self.loss_fn(pred, self.targets)
        self.loss = loss

        # Mark loss for diagnostics/graph export
        autograd.tape.mark_loss(loss)

        # Optional strict-mode checks prior to grad
        if self.config.strict:
            try:
                # Trigger strict-mode preflight by calling grad with zero params list
                # (we don’t want to compute grads yet; just run the checks)
                autograd.grad(loss, [], retain_graph=True, allow_unused=True)
            except RuntimeError as e:
                # Bubble up strict diagnostics to the caller
                raise
            except Exception:
                pass

        # Optional gradient pass for connectivity verification
        if self.config.run_backward and params:
            autograd.grad(loss, params, retain_graph=True, allow_unused=False)

        # Build fused graphs for the captured whole
        self.proc = AutogradProcess(autograd.tape)
        self.proc.build(loss)
        self.fused_graph = self.proc.combined_graph

    # --------------- reporting ---------------
    def schedules(self) -> Tuple[Dict[Any, int], Dict[Any, int]]:
        if self.proc is None:
            raise RuntimeError("capture() must be called before schedules().")
        # Derive ASAP/ALAP by reusing ILPScheduler via GraphTranslator already used in build()
        fwd = self.proc.forward_graph
        if fwd is None:
            return {}, {}
        asap = {nid: fwd.nodes[nid].get("level") for nid in fwd.nodes if fwd.nodes[nid].get("level") is not None}
        # ALAP is not directly stored; return forward-level map and let caller compute if needed
        return asap, {}

    # --------------- training (eager) ---------------
    def train(self, *, epochs: int = 100, lr: float = 1e-2, epsilon: float = 1e-6, print_sched: bool = False) -> None:
        params = self._ensure_params()
        if not params:
            raise RuntimeError("model.parameters() returned no trainable params.")
        opt = Adam(params, lr=lr)

        def mse(a, b):
            return ((a - b) ** 2).mean()

        for epoch in range(1, epochs + 1):
            # Fresh tape per step
            autograd.tape = autograd.__class__().tape
            # Attach inputs/targets/params to current tape and clear grads
            self._attach_to_tape(self.inputs)
            self._attach_to_tape(self.targets)
            for p in params:
                self._attach_to_tape(p)
            self._zero_param_grads(params)

            # Optional schedule band print (from prior capture) for context
            if print_sched and self.proc is not None and self.proc.forward_graph is not None:
                levels = {}
                for nid, data in self.proc.forward_graph.nodes(data=True):
                    lvl = data.get("level")
                    if lvl is not None:
                        levels.setdefault(lvl, 0)
                        levels[lvl] += 1
                for lvl in sorted(levels):
                    print(f"[sched] level {lvl}: {levels[lvl]} nodes")

            # Forward + loss
            if hasattr(self.model, "forward") and callable(getattr(self.model, "forward")):
                pred = self.model.forward(self.inputs)
            else:
                pred = self.model(self.inputs)
            loss = self.loss_fn(pred, self.targets) if self.loss_fn is not None else mse(pred, self.targets)
            autograd.tape.mark_loss(loss)

            # Strict preflight on this step’s tape (no masking)
            if self.config.strict:
                autograd.grad(loss, [], retain_graph=True, allow_unused=True)

            # Backprop + step
            loss.backward()
            grads = [p.grad for p in params]
            new_params = opt.step(params, grads)
            for p, np_ in zip(params, new_params):
                AT.copyto(p, np_)

            loss_val = float(loss.item())
            print(f"[epoch {epoch}] loss={loss_val:.3e}")
            if loss_val <= epsilon and loss_val == loss_val:
                print(f"Converged: loss <= {epsilon:.3e} at epoch {epoch}")
                break

