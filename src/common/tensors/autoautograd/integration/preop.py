from __future__ import annotations

from typing import Any, Tuple

from ...abstraction import AbstractTensor
from ...abstract_nn.activations import ACTIVATIONS


def _sigmoid(x: AbstractTensor) -> AbstractTensor:
    return 1.0 / (1.0 + (-x).exp())


def preactivate_src(sys: Any, nid: int) -> Tuple[AbstractTensor, dict]:
    """Pre-activation helper for a source node.

    Parameters
    ----------
    sys : Any
        System object exposing ``nodes`` mapping.
    nid : int
        Source node id.

    Returns
    -------
    y : AbstractTensor
        Transformed value for the node.
    meta : dict
        Metadata containing intermediates for parameter gradient calculations.
    """
    n = sys.nodes[nid]
    x = n.p
    ecc_raw, w, b = AbstractTensor.get_tensor(n.param[0]), AbstractTensor.get_tensor(n.param[1]), AbstractTensor.get_tensor(n.param[2])
    gate = _sigmoid(ecc_raw)
    z = x * w + b
    act_name = getattr(n, "activation", "tanh")
    act_cls = ACTIVATIONS.get(act_name, ACTIVATIONS["tanh"])
    z_act = act_cls()(z)
    y = (1.0 - gate) * z + gate * z_act
    meta = {
        "x": x,
        "z": z,
        "z_act": z_act,
        "gate": gate,
        "w": w,
        "b": b,
        "ecc_raw": ecc_raw,
    }
    return y, meta

