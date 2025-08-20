from __future__ import annotations
from typing import Tuple, List, Optional
from ..abstraction import AbstractTensor
from .core import Model
from .losses import Loss
from .optimizer import Adam
from .utils import as_list
from .hooks import hook_panel

# -------- Gradient control ---------------------------------------------------
class GradControl:
    """
    Configure gradient-norm control.
    - per_param_max_norm: clip each param's grad to this L2 norm (optional)
    - max_global_norm: clip the *global* grad L2 norm across all params (optional)
    - target_global_norm: after clipping, rescale grads to this global norm (optional)
    """
    def __init__(
        self,
        per_param_max_norm: Optional[float] = None,
        max_global_norm: Optional[float] = None,
        target_global_norm: Optional[float] = None,
    ):
        self.per_param_max_norm = per_param_max_norm
        self.max_global_norm = max_global_norm
        self.target_global_norm = target_global_norm


def _l2(g) -> float:
    # scalar float L2 for any AbstractTensor
    return float(((g * g).sum()).sqrt().item())


def _global_l2(grads: List) -> float:
    s = 0.0
    for g in grads:
        s += float((g * g).sum().item())
    return s ** 0.5


def _clip_per_param_norm(grads: List, max_norm: float):
    """Return (scaled_grads, norms_before, scales)."""
    norms = [_l2(g) for g in grads]
    eps = 1e-9
    scales = [1.0 if n == 0.0 else min(1.0, max_norm / (n + eps)) for n in norms]
    clipped = [g * s for g, s in zip(grads, scales)]
    return clipped, norms, scales


def _clip_global_norm(grads: List, max_norm: float):
    """Return (scaled_grads, global_norm_before, scale)."""
    gnorm = _global_l2(grads)
    eps = 1e-9
    scale = 1.0 if gnorm == 0.0 else min(1.0, max_norm / (gnorm + eps))
    if scale < 1.0:
        grads = [g * scale for g in grads]
    return grads, gnorm, scale

def _to_scalar(x):
    # AbstractTensor / torch / numpy all expose .item() for 0-D
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    # Handle things that only have tolist()
    if hasattr(x, "tolist"):
        t = x.tolist()
        # collapse 1-element nests
        while isinstance(t, (list, tuple)) and len(t) == 1:
            t = t[0]
        if isinstance(t, (int, float, bool)):
            return float(t)
    # Already a Python number?
    if isinstance(x, (int, float, bool)):
        return float(x)
    raise TypeError(f"Can't convert {type(x)} to scalar float")



def train_step(
    model: Model,
    loss_fn: Loss,
    optimizer: Adam,
    x: AbstractTensor,
    y: AbstractTensor,
    debug: bool = False,
    grad_control: Optional[GradControl] = None,
) -> Tuple[float, float]:
    hook_panel.run('step_start', model=model, x=x, y=y)
    pred = model.forward(x)
    hook_panel.run('forward', model=model, x=x, pred=pred)
    loss = loss_fn.forward(pred, y)
    hook_panel.run('loss', model=model, pred=pred, y=y, loss=loss)
    grad_pred = loss_fn.backward(pred, y)
    hook_panel.run('backward', model=model, grad_pred=grad_pred)
    model.backward(grad_pred)
    if debug:
        def norms(t: AbstractTensor) -> float:
            return float(((t * t).sum()).sqrt().item())

        for i, l in enumerate(model.layers):
            b0 = float(l.b[0, 0].item()) if l.b is not None else None
            hook_panel.run('debug', layer=l, i=i, W=l.W, gW=l.gW, b0=b0)

    # Collect params/grads in a stable order
    params: List[AbstractTensor] = []
    grads: List[AbstractTensor] = []
    per_layer_norms_before = []
    for layer in model.layers:
        params.extend([p for p in layer.parameters()])
        g_list = [layer.gW] + ([layer.gb] if layer.b is not None else [])
        grads.extend(g_list)
        w_n = _l2(layer.gW)
        b_n = _l2(layer.gb) if layer.b is not None else None
        per_layer_norms_before.append({"W": w_n, "b": b_n})

    hook_panel.run(
        'grad_stats_before',
        per_layer=per_layer_norms_before,
        global_norm=_global_l2(grads),
    )

    # --- Gradient norm control ---
    if grad_control is not None:
        if grad_control.per_param_max_norm is not None:
            grads, norms_before, scales = _clip_per_param_norm(
                grads, grad_control.per_param_max_norm
            )
            hook_panel.run(
                'grad_clipped_per_param',
                norms_before=norms_before,
                scales=scales,
            )
        if grad_control.max_global_norm is not None:
            grads, g_before, g_scale = _clip_global_norm(
                grads, grad_control.max_global_norm
            )
            hook_panel.run(
                'grad_clipped_global',
                global_norm_before=g_before,
                scale=g_scale,
            )
        if grad_control.target_global_norm is not None:
            g_now = _global_l2(grads)
            if g_now > 0.0:
                s = grad_control.target_global_norm / g_now
                grads = [g * s for g in grads]
                hook_panel.run(
                    'grad_rescaled_to_target',
                    global_norm_before=g_now,
                    scale=s,
                    target=grad_control.target_global_norm,
                )

    # Per-layer/global norms after control (for visibility)
    per_layer_norms_after = []
    idx = 0
    for layer in model.layers:
        gW = grads[idx]
        idx += 1
        gB = grads[idx] if layer.b is not None else None
        if layer.b is not None:
            idx += 1
        per_layer_norms_after.append({
            "W": _l2(gW),
            "b": _l2(gB) if gB is not None else None,
        })

    hook_panel.run(
        'grad_stats_after',
        per_layer=per_layer_norms_after,
        global_norm=_global_l2(grads),
    )

    new_params = optimizer.step(params, grads)
    i = 0
    for layer in model.layers:
        layer.W = new_params[i]
        i += 1
        if layer.b is not None:
            layer.b = new_params[i]
            i += 1
    model.zero_grad()
    hook_panel.run('step_end', model=model, x=x, y=y, loss=loss)
    return _to_scalar(loss), _global_l2(grads)

def train_loop(
    model: Model,
    loss_fn: Loss,
    optimizer: Adam,
    X: AbstractTensor,
    Y: AbstractTensor,
    epochs: int = 2000,
    log_every: int = 1,
    provenance_tracker=None,
    grad_control: Optional[GradControl] = None,
):
    losses = []
    for e in range(1, epochs + 1):
        l, g = train_step(
            model,
            loss_fn,
            optimizer,
            X,
            Y,
            debug=(e == 1),
            grad_control=grad_control,
        )
        losses.append(l)
        hook_panel.run('epoch_end', epoch=e, loss=l)
        if provenance_tracker is not None:
            provenance_tracker.record(e, l, model)
        if (e % log_every) == 0:
            hook_panel.run('log', epoch=e, loss=l, grad_global=g)
            print(f"[{e}] loss={l:.6f}  || grad||={g:.6f}")
    return losses
