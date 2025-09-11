
"""
autograd_probes.py — Gradient debugging utilities for your custom autograd/tape.

Drop this file anywhere importable (e.g., alongside your demo) and use:

    from autograd_probes import (
        set_strict_mode, annotate_params, parameter_summary,
        show_missing_ops, reachability_report, grad_presence,
        probe_losses
    )

Typical workflow inside your training script (before loss.backward()):

    set_strict_mode(True)  # fail fast on missing backwards (relies on env var)

    wheels = register_param_wheels(spec)
    for w in wheels:
        w.rotate(); w.bind_slot()
    params = [w.bind_slot() for w in wheels]
    annotate_params(params)  # nice labels in reports

    losses = {
        "loss_out_only": loss_out,
        "hist_loss_only": hist_loss,
        "combined":      loss_out + hist_loss,
    }

    probe_losses(losses, params)

This prints:
- Any missing backward ops reachable from each loss (loudly).
- Whether each parameter is present on the backward graph and reachable
  from the loss.
- Whether autograd.grad returns a gradient for each parameter.

All functions accept an optional `ag` argument if you want to pass your
autograd module explicitly. If omitted, we try common imports.
"""

from __future__ import annotations
import os
from typing import Dict, Iterable, List, Optional, Tuple

def _resolve_autograd(ag=None):
    if ag is not None:
        return ag
    # Try a few likely module paths. Only import what's available.
    try:
        from autograd import autograd as _ag  # project-local style
        return _ag
    except Exception:
        pass
    try:
        # common repo layout: src.common.tensors.autograd.autograd
        from src.common.tensors.autograd import autograd as _ag
        return _ag
    except Exception:
        pass
    try:
        # fallback if the package is named exactly "autograd"
        import autograd as _ag
        return _ag
    except Exception:
        pass
    raise ImportError("Could not locate your project's autograd module. "
                      "Pass it explicitly via ag=...")

# -------------------- Convenience helpers --------------------

def set_strict_mode(enabled: bool = True):
    """Enable/disable strict mode via env var (engine should read this)."""
    os.environ["AUTOGRAD_STRICT"] = "1" if enabled else "0"

def _get_tape(ag=None):
    ag = _resolve_autograd(ag)
    # Some stacks expose tape at ag.tape; others at ag.autograd.tape
    tape = getattr(ag, "tape", None)
    if tape is None and hasattr(ag, "autograd"):
        tape = getattr(ag.autograd, "tape", None)
    if tape is None:
        raise AttributeError("Autograd module does not expose `tape`.")
    return tape, ag

def annotate_params(params: Iterable, *, ag=None, label_fmt: str = "param[{idx}]"):
    """Apply human-readable labels to params on the tape (no-op if unsupported)."""
    tape, _ = _get_tape(ag)
    annotate = getattr(tape, "annotate", None)
    if annotate is None:
        return  # no labeling facility; silently ignore
    for i, p in enumerate(params):
        try:
            annotate(p, label=label_fmt.format(idx=i))
        except Exception:
            # Keep going even if a particular object can't be annotated
            pass

def parameter_summary(*, ag=None) -> Tuple[int, List[int]]:
    """Return (#params, list of ids) as seen by the tape (trainables only)."""
    tape, _ = _get_tape(ag)
    pt = getattr(tape, "parameter_tensors", None)
    if pt is None:
        return (0, [])
    params = pt()
    return (len(params), [id(p) for p in params])

def _label_for(tape, obj, default: str):
    # Try to pull a human label from the tape's graph metadata.
    try:
        g = getattr(tape, "graph", None)
        if g is None:
            return default
        node = g.nodes.get(id(obj), {})
        ann  = node.get("annotations", {}) if isinstance(node, dict) else {}
        return ann.get("label", default)
    except Exception:
        return default

# -------------------- Core probes --------------------

def show_missing_ops(tag: str, loss_tensor, *, ag=None) -> List[dict]:
    """Validate backward coverage from `loss_tensor` and print missing ops.

    Returns the list of missing-op dicts for further inspection.
    """
    tape, _ = _get_tape(ag)
    validate = getattr(tape, "validate_backward_ops", None)
    if validate is None:
        print(f"[{tag}] validate_backward_ops not available on this tape.")
        return []
    try:
        missing = validate(loss_tensor) or []
    except Exception as e:
        print(f"[{tag}] validate_backward_ops raised: {e}")
        return []
    if missing:
        print(f"[{tag}] Missing backward implementations:")
        for m in missing:
            op  = m.get("op")
            msg = m.get("message", "")
            ish = m.get("input_shapes")
            osh = m.get("result_shape")
            print(f"  - {op} {msg} inputs={ish} -> {osh}")
    else:
        print(f"[{tag}] All required backward rules present.")
    return missing

def reachability_report(tag: str, loss_tensor, params: Iterable, *, ag=None):
    """Report whether each param is on the backward graph and reachable from loss."""
    tape, ag = _get_tape(ag)
    # Ensure the loss is marked for export if the engine requires it.
    mark_loss = getattr(tape, "mark_loss", None)
    if callable(mark_loss):
        try:
            mark_loss(loss_tensor)
        except Exception:
            pass

    export = getattr(tape, "export_training_state", None)
    if export is None:
        print(f"[{tag}] export_training_state not available on this tape.")
        return

    try:
        fwd, bwd, _, _ = export()  # we only need bwd (a graph-like object)
    except Exception as e:
        print(f"[{tag}] export_training_state raised: {e}")
        return

    loss_id = id(loss_tensor)

    # We try to use networkx if the graph is an nx.DiGraph;
    # otherwise fall back to a manual BFS on common attributes.
    def _has_path(bwd_graph, src, dst) -> bool:
        try:
            import networkx as nx  # type: ignore
            if hasattr(bwd_graph, "has_node") and isinstance(bwd_graph, (nx.DiGraph, nx.MultiDiGraph)):
                return bwd_graph.has_node(src) and bwd_graph.has_node(dst) and nx.has_path(bwd_graph, src, dst)
        except Exception:
            pass
        # Fallback BFS on .successors or adjacency
        try:
            if hasattr(bwd_graph, "successors"):
                succ = bwd_graph.successors
                seen = set([src])
                stack = [src]
                while stack:
                    u = stack.pop()
                    if u == dst:
                        return True
                    try:
                        for v in succ(u):
                            if v not in seen:
                                seen.add(v); stack.append(v)
                    except Exception:
                        pass
                return False
        except Exception:
            pass

        # Last resort: adjacency dict style
        adj = getattr(bwd_graph, "adj", None)
        if isinstance(adj, dict):
            seen = set([src])
            stack = [src]
            while stack:
                u = stack.pop()
                if u == dst:
                    return True
                for v in getattr(adj.get(u, {}), "keys", lambda: [])():
                    if v not in seen:
                        seen.add(v); stack.append(v)
            return False

        # Unknown graph type — best effort
        return False

    print(f"[{tag}] Backward reachability (loss -> params):")
    for i, p in enumerate(params):
        pid   = id(p)
        on_g  = False
        try:
            on_g = (hasattr(bwd, "has_node") and bwd.has_node(pid)) or \
                   (hasattr(bwd, "nodes") and pid in getattr(bwd, "nodes"))
        except Exception:
            pass
        reachable = _has_path(bwd, loss_id, pid)
        label = _label_for(tape, p, default=f"param[{i}]")
        print(f"  - {label}: on_graph={on_g}, reachable={reachable}")

def grad_presence(tag: str, loss_tensor, params: Iterable, *, ag=None):
    """Call autograd.grad on (loss, params) and print which params get grads."""
    _, ag = _get_tape(ag)
    grad_fn = getattr(ag, "grad", None)
    if grad_fn is None:
        print(f"[{tag}] autograd.grad not found on provided module.")
        return

    try:
        gs = grad_fn(loss_tensor, list(params), allow_unused=False)
    except Exception as e:
        print(f"[{tag}] autograd.grad(..., allow_unused=False) raised: {e}")
        gs = grad_fn(loss_tensor, list(params), allow_unused=True)

    tape, _ = _get_tape(ag)
    print(f"[{tag}] Gradient presence:")
    for i, (p, g) in enumerate(zip(params, gs)):
        label = _label_for(tape, p, default=f"param[{i}]")
        status = "OK" if g is not None else "None"
        # If g is a tensor-like, try to give a tiny shape hint
        try:
            shp = getattr(g, "shape", None) or getattr(g, "get_shape", lambda: None)()
        except Exception:
            shp = None
        hint = f" shape={tuple(shp)}" if shp is not None else ""
        print(f"  - {label}: {status}{hint}")

# -------------------- One-shot multi-loss probe --------------------

def probe_losses(losses: Dict[str, object], params: Iterable, *, ag=None):
    """Run missing-op check, reachability, and grad presence for each loss.

    Args:
        losses: A dict mapping a readable tag to loss tensors.
        params: Iterable of trainable parameter tensors.
        ag:     Optional explicit autograd module.
    """
    if not isinstance(losses, dict) or not losses:
        print("[probe] No losses provided.")
        return

    for tag, lt in losses.items():
        print("=" * 72)
        print(f"[probe] {tag}")
        show_missing_ops(tag, lt, ag=ag)
        reachability_report(tag, lt, params, ag=ag)
        grad_presence(tag, lt, params, ag=ag)
    print("=" * 72)
