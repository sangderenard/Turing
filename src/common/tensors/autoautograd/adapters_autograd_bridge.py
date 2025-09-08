# adapters_autograd_bridge.py

from typing import Sequence, Tuple, Callable, Any

from ..abstraction import AbstractTensor
from ..autograd import autograd

# a few operator aliases when there's no named method on AbstractTensor
_BIN_ALIASES = {
    "add":      lambda a, b: a + b,
    "sub":      lambda a, b: a - b,
    "mul":      lambda a, b: a * b,
    "truediv":  lambda a, b: a / b,
    "pow":      lambda a, b: a ** b,
}

def _as_at(x: Any):
    # unwrap AbstractTensor -> backend buffer -> tensor
    if isinstance(x, AbstractTensor):
        x = x.data
    return AbstractTensor.asarray(x)

def run_op_and_grads(
    op_name: str,
    *at_inputs: AbstractTensor,
    loss: Callable[[AbstractTensor], AbstractTensor] | str = "sum",
) -> Tuple[AbstractTensor, Tuple[AbstractTensor, ...]]:
    """
    Execute one forward op and return (forward_output_tensor, grads_per_input_tensor).

    - op_name: name of the forward operator (e.g., "maximum", "relu", "sum", "add", "mul"...).
    - at_inputs: tensor scalars/arrays.
    - loss: "sum" (default) or a callable mapping the op output -> scalar loss tensor.
    """
    # 1) wrap tensor -> AbstractTensor on the global tape with grads enabled
    ats = [AbstractTensor.get_tensor(x, requires_grad=True) for x in at_inputs]  # attaches to tape
    # 2) find and run the op
    if hasattr(AbstractTensor, op_name):              # e.g., maximum(x,y), sigmoid(x), sum(x)...
        fn = getattr(AbstractTensor, op_name)
        y = fn(*ats)
    elif op_name in _BIN_ALIASES and len(ats) == 2:   # add/mul/etc via Python operators
        y = _BIN_ALIASES[op_name](ats[0], ats[1])
    else:
        raise ValueError(f"Unknown op '{op_name}' for {len(ats)} inputs")

    # 3) build a scalar loss so gradients are well-defined
    L = y.sum() if loss == "sum" else loss(y)

    # (optional) catch missing backward rules early
    missing = autograd.tape.validate_backward_ops(L)
    if missing:
        names = sorted({m['op'] for m in missing if m.get('op')})
        raise RuntimeError(f"Missing backward rules for: {names}")

    # 4) differentiate w.r.t. the provided inputs, then read .grad
    autograd.grad(L, ats, retain_graph=False, allow_unused=True)
    grads = tuple(_as_at(a.grad) for a in ats)

    return _as_at(y), grads


# ---- bridge to your spring edges ------------------------------------------------

def push_impulses_from_op(
    sys,
    op_name: str,
    src_ids: Sequence[int],
    out_id: int,
    *,
    residual: AbstractTensor | None = None,
    scale: float = 1.0,
) -> float:
    """
    Compute local grads for a single op on node thetas and push impulses onto edges.
    Returns the forward value so you can write it to the out node if you want.

    residual: full-vector `(y - target)` if already computed; otherwise ``None``.
    """
    # gather current control parameters from nodes
    vals = [AbstractTensor.array(sys.nodes[i].ctrl, dtype=float) for i in src_ids]

    y_at, grads_at = run_op_and_grads(op_name, *vals)  # grads in same order as src_ids

    # push impulses (classic "local jacobian^T * residual" pattern)
    if residual is not None:
        r = AbstractTensor.tensor(residual)
        for i, g in zip(src_ids, grads_at):
            g_vec = AbstractTensor.asarray(g)
            try:
                g_scalar = float((g_vec * r).sum())
            except Exception:
                g_scalar = 0.0
            sys.impulse(i, out_id, op_name, scale * (-g_scalar))

    return float(AbstractTensor.asarray(y_at))
