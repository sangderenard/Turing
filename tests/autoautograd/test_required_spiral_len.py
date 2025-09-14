from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import (
    FluxSpringSpec,
    NodeSpec,
    NodeCtrl,
    LearnCtrl,
    EdgeSpec,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    DECSpec,
    SpectralCfg,
    register_param_wheels,
    required_spiral_len,
)
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue


def _build_spec(win_len: int) -> FluxSpringSpec:
    param = AT.tensor(1.0)
    param.requires_grad_(True)
    node = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(w=param, learn=LearnCtrl(alpha=False, w=True, b=False)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=0,
        transport=EdgeTransport(learn=EdgeTransportLearn(False, False, False, False, False)),
        ctrl=EdgeCtrl(learn=LearnCtrl(False, False, False)),
    )
    spectral = SpectralCfg(enabled=True, win_len=win_len)
    return FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node],
        edges=[edge],
        faces=[],
        dec=DECSpec(D0=[[0.0]], D1=[]),
        spectral=spectral,
    )


def test_required_spiral_len_grows_with_delay():
    spec = _build_spec(5)
    base = required_spiral_len(spec)
    assert base == 5
    assert required_spiral_len(spec, extra_delay=3) == base + 3


def test_wheels_and_queue_respect_required_len():
    extra = 2
    spec = _build_spec(4)
    slots = required_spiral_len(spec, extra)
    wheels = register_param_wheels(spec, extra_delay=extra)
    assert all(len(w.versions()) == slots for w in wheels)
    q = SlotBackpropQueue(wheels, slots=slots)
    assert q.slots == slots
