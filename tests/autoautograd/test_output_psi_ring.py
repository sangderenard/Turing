from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.fs_types import (
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
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick
from src.common.tensors.autoautograd.fluxspring.fs_harness import RingHarness


def _build_spec(win_len: int) -> FluxSpringSpec:
    ctrl = NodeCtrl(alpha=AT.tensor(0.0), w=AT.tensor(1.0), b=AT.tensor(0.0), learn=LearnCtrl(True, True, True))
    node = NodeSpec(
        id=0,
        p0=AT.zeros(1),
        v0=AT.zeros(1),
        mass=AT.tensor(1.0),
        ctrl=ctrl,
        scripted_axes=[0],
    )
    ectrl = EdgeCtrl(alpha=AT.tensor(0.0), w=AT.tensor(0.0), b=AT.tensor(0.0), learn=LearnCtrl(True, True, True))
    transport = EdgeTransport(kappa=AT.tensor(0.0), learn=EdgeTransportLearn())
    edge = EdgeSpec(src=0, dst=0, transport=transport, ctrl=ectrl)
    dec = DECSpec(D0=[[0.0]], D1=[])
    spectral = SpectralCfg(enabled=True, win_len=win_len, hop_len=win_len, window="hann")
    return FluxSpringSpec(
        version="psi-out-test",
        D=1,
        nodes=[node],
        edges=[edge],
        faces=[],
        dec=dec,
        spectral=spectral,
    )


def test_out_psi_ring_eviction_and_grad():
    spec = _build_spec(3)
    param = AT.tensor(0.5)
    param.requires_grad_(True)
    psi = AT.zeros(1)
    harness = RingHarness(default_size=3)

    for _ in range(4):
        psi, stats = pump_tick(psi, spec, eta=0.0, external={0: param}, harness=harness)

    evicted = stats.get("evicted_psi", {}).get(0)
    assert evicted is not None
    expected = param.reshape(-1)
    assert AT.allclose(evicted, expected)

