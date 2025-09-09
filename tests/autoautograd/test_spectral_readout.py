import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.spectral_readout import (
    compute_metrics,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    SpectralCfg,
    SpectralMetrics,
)


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
    m = compute_metrics(x, cfg)
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
    m = compute_metrics(x, cfg)
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
    m = compute_metrics(buf, cfg)
    assert abs(m["coherence"] - 1.0) < 1e-6
