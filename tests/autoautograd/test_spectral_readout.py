import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.spectral_readout import (
    compute_metrics,
    batched_bandpower_from_windows,
    quantile_band_targets,
    gather_recent_windows,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    SpectralCfg,
    SpectralMetrics,
)
from src.common.tensors.autoautograd.fluxspring.fs_harness import RingHarness
import numpy as np

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


def test_quantile_band_targets_even_split():
    tick_hz = 90.0
    N = 90
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="rect",
    )
    harness = RingHarness(default_size=N)
    freqs = [5.0, 15.0, 25.0]
    t = AT.arange(N, dtype=float) / tick_hz
    waves = [(2 * AT.pi() * f * t).sin() for f in freqs]
    for i in range(N):
        for nid, w in enumerate(waves):
            harness.push_premix(nid, w[i])
    targets = quantile_band_targets(list(range(3)), cfg, harness)
    assert set(targets.keys()) == {0, 1, 2}
    for nid, f in zip(range(3), freqs):
        lo, hi = targets[nid]
        assert lo <= f <= hi + 1e-6


def test_gather_recent_windows():
    tick_hz = 50.0
    N = 4
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="rect",
    )
    harness = RingHarness(default_size=N)
    for i in range(N):
        harness.push_premix(0, AT.tensor(float(i)))
        harness.push_premix(1, AT.tensor(float(i + 10)))
    W, kept = gather_recent_windows([0, 1], cfg, harness)
    assert kept == [0, 1]
    W_np = AT.get_tensor(W)
    expected = np.array([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
    assert W_np.shape == expected.shape
    assert np.allclose(W_np, expected)
