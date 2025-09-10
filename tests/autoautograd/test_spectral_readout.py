import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.spectral_readout import (
    compute_metrics,
    gather_recent_windows,
    batched_bandpower_from_windows,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    SpectralCfg,
    SpectralMetrics,
    NodeSpec,
    NodeCtrl,
    FluxSpringSpec,
    DECSpec,
)
from src.common.tensors.autoautograd.fluxspring.fs_harness import RingHarness


def _sine(freq: float, N: int, tick_hz: float) -> AT:
    t = AT.arange(N, dtype=float) / tick_hz
    return (2 * AT.pi() * freq * t).sin()


def test_bandpower_centroid_flatness():
    N = 400
    tick_hz = 400.0
    freq = 40.0
    x = _sine(freq, N, tick_hz)
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="rect",
        metrics=SpectralMetrics(
            bands=[[30.0, 50.0], [100.0, 150.0]],
            centroid=True,
            flatness=True,
        ),
    )
    m = compute_metrics(x, cfg, return_tensor=False)
    assert m["bandpower"][0] > 10 * m["bandpower"][1]
    assert abs(m["centroid"] - freq) < 2.0
    assert m["flatness"] < 0.2


def test_window_function_applied():
    N = 128
    tick_hz = 128.0
    freq = 20.0
    x = _sine(freq, N, tick_hz)
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="zeros",
        metrics=SpectralMetrics(bands=[[10.0, 30.0]]),
    )
    m = compute_metrics(x, cfg, return_tensor=False)
    assert m["bandpower"][0] == pytest.approx(0.0, abs=1e-6)


def test_coherence_identical_signals():
    N = 256
    tick_hz = 256.0
    freq = 30.0
    s = _sine(freq, N, tick_hz)
    buf = AT.stack([s, s], dim=1)
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="rect",
        metrics=SpectralMetrics(coherence=True),
    )
    m = compute_metrics(buf, cfg, return_tensor=False)
    assert abs(m["coherence"] - 1.0) < 1e-6


def test_gather_recent_windows_wrap_and_pad():
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=1.0,
        win_len=3,
        hop_len=3,
        window="rect",
        metrics=SpectralMetrics(),
    )
    node0 = NodeSpec(
        id=0,
        p0=AT.zeros(1),
        v0=AT.zeros(1),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(),
        scripted_axes=[0, 0],
    )
    node1 = NodeSpec(
        id=1,
        p0=AT.zeros(1),
        v0=AT.zeros(1),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(),
        scripted_axes=[0, 0],
    )
    harness = RingHarness()
    for i in range(7):
        harness.push_node(node0.id, AT.tensor([float(i)]), size=5)
    harness.push_node(node1.id, AT.tensor([10.0]), size=2)
    harness.push_node(node1.id, AT.tensor([11.0]), size=2)
    spec = FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node0, node1],
        edges=[],
        faces=[],
        dec=DECSpec(D0=[], D1=[]),
        spectral=cfg,
    )
    W, ids = gather_recent_windows(spec, [0, 1], cfg, harness)
    assert ids == [0, 1]
    assert AT.get_tensor(W[0]).tolist() == [4.0, 5.0, 6.0]
    assert AT.get_tensor(W[1]).tolist() == [0.0, 10.0, 11.0]


def test_batched_bandpower_from_windows():
    tick_hz = 100.0
    Nw = 50
    t = AT.arange(Nw, dtype=float) / tick_hz
    win1 = (2 * AT.pi() * 20.0 * t).sin()
    win2 = (2 * AT.pi() * 40.0 * t).sin()
    W = AT.stack([win1, win2])
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=Nw,
        hop_len=Nw,
        window="rect",
        metrics=SpectralMetrics(bands=[[15.0, 25.0], [35.0, 45.0]]),
    )
    bp = batched_bandpower_from_windows(W, cfg)
    bp_np = AT.get_tensor(bp)
    assert bp_np[0, 0] > 0.9 and bp_np[0, 1] < 0.1
    assert bp_np[1, 1] > 0.9 and bp_np[1, 0] < 0.1
