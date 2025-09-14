from types import SimpleNamespace

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import ParamWheel
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue
from src.common.tensors.autoautograd.whiteboard_runtime import _WBJob
from src.common.tensors.autoautograd.fluxspring.demo_spectral_routing import FLUX_PARAM_SCHEMA


def _sys_for_slot(wheels, slot):
    nodes = {}
    for i, w in enumerate(wheels):
        attrs = {name: AT.tensor(0.0) for name in FLUX_PARAM_SCHEMA}
        attr = w.label.rsplit(".", 1)[-1]
        attrs[attr] = w.params[slot]
        nodes[i] = SimpleNamespace(**attrs)
    return SimpleNamespace(nodes=nodes)


def test_multi_attribute_param_schema_grad():
    a = AT.tensor(1.0); a.requires_grad_(True)
    w = AT.tensor(2.0); w.requires_grad_(True)
    b = AT.tensor(3.0); b.requires_grad_(True)
    wheels = [
        ParamWheel(a, lambda t: None, slots=1, label="n.ctrl.alpha"),
        ParamWheel(w, lambda t: None, slots=1, label="n.ctrl.w"),
        ParamWheel(b, lambda t: None, slots=1, label="n.ctrl.b"),
    ]
    for wheel in wheels:
        wheel.rotate(); wheel.bind_slot()
    mgr = SlotBackpropQueue(wheels)

    def _fn(alpha, w_, b_):
        return alpha * w_ + b_

    job = _WBJob(
        job_id="train",
        op=None,
        src_ids=tuple(range(len(wheels))),
        residual=AT.tensor(1.0),
        fn=_fn,
        param_schema=("alpha", "w", "b"),
    )
    mgr.queue_job(0, job)
    res = mgr.process_slot(0, sys=_sys_for_slot(wheels, 0))
    assert res is not None
    g = AT.get_tensor(res.grads_per_source_tensor)
    assert bool(g.any())
