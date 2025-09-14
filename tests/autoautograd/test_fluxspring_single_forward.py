from src.common.tensors.autoautograd.fluxspring.demo_spectral_routing import (
    build_spec,
    generate_signals,
    train_routing,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import SpectralCfg, SpectralMetrics
from src.common.tensors.autoautograd import fluxspring as fs


def test_train_routing_single_forward(monkeypatch):
    bands = [[0.0, 1.0]]
    spectral_cfg = SpectralCfg(
        enabled=False,
        win_len=2,
        hop_len=1,
        tick_hz=2.0,
        metrics=SpectralMetrics(bands=bands),
    )
    spec = build_spec(spectral_cfg)
    sine_chunks, noise_frames = generate_signals(bands, 2, 2.0, frames=2)

    calls = {"count": 0}
    orig = fs.fs_dec._pump_tick

    def _counting(*args, **kwargs):
        calls["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fs.fs_dec, "_pump_tick", _counting)

    import pytest

    with pytest.raises(RuntimeError):
        train_routing(spec, spectral_cfg, sine_chunks, noise_frames)
    assert calls["count"] == 2
