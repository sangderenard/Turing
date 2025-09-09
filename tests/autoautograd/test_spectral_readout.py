import numpy as np
from src.common.tensors.autoautograd.fluxspring.spectral_readout import compute_metrics

def test_windowing_reduces_power():
    fs = 1000
    N = 256
    t = np.arange(N) / fs
    sig = np.sin(2 * np.pi * 100 * t)
    rect_cfg = {
        "tick_hz": fs,
        "win_len": N,
        "window_fn": "rect",
        "metrics": {"bands": [[90, 110]]},
    }
    hann_cfg = {**rect_cfg, "window_fn": "hann"}
    rect = compute_metrics(sig, rect_cfg)
    hann = compute_metrics(sig, hann_cfg)
    assert float(hann["bandpower"][0]) < float(rect["bandpower"][0])


def test_metrics_for_sine():
    fs = 1000
    N = 256
    t = np.arange(N) / fs
    sig = np.sin(2 * np.pi * 100 * t)
    buf = np.stack([sig, sig])
    cfg = {
        "tick_hz": fs,
        "win_len": N,
        "window_fn": "rect",
        "metrics": {
            "bands": [[90, 110]],
            "centroid": True,
            "flatness": True,
            "coherence": True,
        },
    }
    m = compute_metrics(buf, cfg)
    band = m["bandpower"]
    assert band.shape == (2, 1)
    assert band[0, 0] > 0
    assert abs(float(m["centroid"][0]) - 100) < 1.0
    assert float(m["flatness"][0]) < 1e-2
    coherence = float(m["coherence"].item())
    assert abs(coherence - 1.0) < 1e-2
