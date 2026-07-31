import numpy as np

from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    _open_control_stream,
)


def test_missing_audio_uses_procedural_controls_and_silent_pcm():
    stream = _open_control_stream(None, gain=1.0, duration=0.25)
    try:
        controls = stream.sample(0.1)
        assert all(
            np.isfinite(value)
            for value in (
                controls.loudness,
                controls.bass,
                controls.low_mid,
                controls.high_mid,
                controls.treble,
            )
        )
        assert stream.sample_rate == 48_000
        assert stream.samples.shape == (12_000,)
        assert not np.any(stream.samples)
    finally:
        stream.close()
