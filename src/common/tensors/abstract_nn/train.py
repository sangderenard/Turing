from __future__ import annotations
from typing import Tuple, List
from ..abstraction import AbstractTensor
from .core import Model
from .losses import Loss
from .optimizer import Adam
from .utils import as_list
from .hooks import hook_panel

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



def train_step(model: Model, loss_fn: Loss, optimizer: Adam, x: AbstractTensor, y: AbstractTensor, debug: bool = False) -> Tuple[float, float]:
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
    params: List[AbstractTensor] = []
    grads: List[AbstractTensor] = []
    for layer in model.layers:
        params.extend([p for p in layer.parameters()])
        grads.extend([layer.gW] + ([layer.gb] if layer.b is not None else []))
    new_params = optimizer.step(params, grads)
    i = 0
    for layer in model.layers:
        layer.W = new_params[i]; i += 1
        if layer.b is not None:
            layer.b = new_params[i]; i += 1
    model.zero_grad()
    hook_panel.run('step_end', model=model, x=x, y=y, loss=loss)
    return _to_scalar(loss), 0.0

def train_loop(model: Model, loss_fn: Loss, optimizer: Adam, X: AbstractTensor, Y: AbstractTensor, epochs: int = 2000, log_every: int = 1, provenance_tracker=None):
    losses = []
    for e in range(1, epochs + 1):
        l, _ = train_step(model, loss_fn, optimizer, X, Y, debug=(e == 1))
        losses.append(l)
        hook_panel.run('epoch_end', epoch=e, loss=l)
        if provenance_tracker is not None:
            provenance_tracker.record(e, l, model)
        if (e % log_every) == 0:
            hook_panel.run('log', epoch=e, loss=l)
            print(f"[{e}] loss={l:.6f}")
    return losses
