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
from src.common.tensors.autoautograd.fluxspring.spectral_readout import phi_histogram_loss
from src.common.tensors.abstraction import AbstractTensor as AT


def _build_spec():
    ctrl = NodeCtrl(alpha=AT.tensor(1.0), w=AT.tensor(1.0), b=AT.tensor(0.0), learn=LearnCtrl(True, True, True))
    n0 = NodeSpec(id=0, p0=AT.zeros(3), v0=AT.zeros(3), mass=AT.tensor(1.0), ctrl=ctrl, scripted_axes=[0, 2])
    n1 = NodeSpec(id=1, p0=AT.zeros(3), v0=AT.zeros(3), mass=AT.tensor(1.0), ctrl=ctrl, scripted_axes=[0, 2])
    ectrl = EdgeCtrl(alpha=AT.tensor(1.0), w=AT.tensor(1.0), b=AT.tensor(0.0), learn=LearnCtrl(True, True, True))
    transport = EdgeTransport(kappa=AT.tensor(1.0), learn=EdgeTransportLearn())
    edge = EdgeSpec(src=0, dst=1, transport=transport, ctrl=ectrl, temperature=AT.tensor(0.0), exclusive=False)
    dec = DECSpec(D0=[[-1.0, 1.0]], D1=[])
    spectral = SpectralCfg(enabled=True, win_len=32, hop_len=32, window="hann")
    return FluxSpringSpec(
        version="phi-test",
        D=3,
        nodes=[n0, n1],
        edges=[edge],
        faces=[],
        dec=dec,
        spectral=spectral,
    )


def test_phi_ring_updates_and_hist_loss():
    spec = _build_spec()
    psi = AT.zeros(len(spec.nodes), dtype=float)
    harness = RingHarness(default_size=32)
    tick_hz = 64.0
    t = AT.arange(32, dtype=float) / tick_hz
    signal = (2 * AT.pi() * 8.0 * t).sin()
    for val in signal:
        psi, _ = pump_tick(psi, spec, eta=0.0, external={0: val}, harness=harness)
    rb = harness.get_phi_ring(1)
    assert rb is not None
    loss_lo = phi_histogram_loss(rb, band_idx=0, total_bands=2, tick_hz=tick_hz)
    loss_hi = phi_histogram_loss(rb, band_idx=1, total_bands=2, tick_hz=tick_hz)
    loss_lo_f = float(AT.get_tensor(loss_lo).item())
    loss_hi_f = float(AT.get_tensor(loss_hi).item())
    assert loss_lo_f < loss_hi_f
