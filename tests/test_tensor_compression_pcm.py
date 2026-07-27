import struct

import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.pcm import (
    PCMFormat,
    RationalAudioScheduler,
    encode_pcm,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_s16_pcm_quantization_is_backend_independent(backend):
    pcm_format = PCMFormat(sample_rate=48_000, channels=1)
    with AT.use_backend(backend):
        encoded = encode_pcm(
            AT.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
            pcm_format=pcm_format,
        )
    assert struct.unpack("<5h", encoded) == (
        -32768,
        -16384,
        0,
        16384,
        32767,
    )


def test_stereo_pcm_is_sample_interleaved_and_gain_clipped():
    pcm_format = PCMFormat(sample_rate=44_100, channels=2)
    with AT.use_backend("numpy"):
        encoded = encode_pcm(
            AT.tensor([[0.25, -0.25], [0.75, -0.75]]),
            pcm_format=pcm_format,
            gain=2.0,
        )
    assert struct.unpack("<4h", encoded) == (
        16384,
        -16384,
        32767,
        -32768,
    )


def test_f32_pcm_preserves_normalized_values():
    pcm_format = PCMFormat(
        sample_rate=96_000,
        channels=1,
        sample_format="f32le",
    )
    with AT.use_backend("numpy"):
        encoded = encode_pcm(
            AT.tensor([-0.5, 0.25, 1.0]),
            pcm_format=pcm_format,
        )
    assert struct.unpack("<3f", encoded) == pytest.approx((-0.5, 0.25, 1.0))
    assert pcm_format.wave_format_tag == 3
    assert len(pcm_format.wave_format_ex()) == 18


def test_rational_audio_scheduler_never_accumulates_frame_drift():
    scheduler = RationalAudioScheduler(
        sample_rate=48_000,
        fps=29.97,
    )
    counts = [scheduler.samples_for_next_frame() for _ in range(10_000)]
    expected = (
        10_000 * scheduler.sample_rate * scheduler.scale
    ) // scheduler.rate
    assert sum(counts) == expected
    assert max(counts) - min(counts) <= 1
