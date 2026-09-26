"""AbstractTensor synthesis and spectral controls for the columnar world.

The first audio source is deterministic synthesis. Microphone capture and
uploaded files can later provide the same mono sample tensor without changing
the FFT feature contract consumed by the compiled visual state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ...tensors.abstraction import AbstractTensor
from ...tensors.compression.pcm import PCMFormat, encode_pcm


@dataclass(frozen=True, slots=True)
class ColumnarAudioTrack:
    """One playable mono tensor and its frame-aligned spectral controls."""

    samples: AbstractTensor
    sample_rate: int
    feature_fps: int
    feature_feeds: dict[str, tuple[float, ...]]

    @property
    def duration(self) -> float:
        return int(self.samples.shape[0]) / float(self.sample_rate)

    def wave_bytes(self) -> bytes:
        """Serialize the exact analyzed tensor as mono signed-16 PCM WAVE."""

        pcm_format = PCMFormat(
            sample_rate=self.sample_rate,
            channels=1,
            sample_format="s16le",
        )
        payload = encode_pcm(self.samples, pcm_format=pcm_format)
        fmt = pcm_format.wave_format_ex()
        riff_size = 4 + 8 + len(fmt) + 8 + len(payload)
        return b"".join((
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVEfmt ",
            struct.pack("<I", len(fmt)),
            fmt,
            b"data",
            struct.pack("<I", len(payload)),
            payload,
        ))


def synthesize_columnar_audio(
    *,
    duration: float = 8.0,
    sample_rate: int = 24_000,
    feature_fps: int = 30,
) -> ColumnarAudioTrack:
    """Synthesize a loop and analyze all feature frames in one tensor FFT."""

    frame_count = int(round(float(duration) * int(feature_fps)))
    samples_per_frame = int(sample_rate) // int(feature_fps)
    if frame_count < 1 or samples_per_frame < 2:
        raise ValueError("audio duration and rates must produce complete frames")
    if samples_per_frame * int(feature_fps) != int(sample_rate):
        raise ValueError("sample_rate must be divisible by feature_fps")

    sample_count = frame_count * samples_per_frame
    time = AbstractTensor.arange(sample_count, dtype="float64") / float(sample_rate)
    tau = 6.283185307179586
    slow_a = (tau * 0.25 * time).sin()
    slow_b = (tau * 0.375 * time + 1.0471975511965976).sin()
    slow_c = (tau * 0.625 * time + 2.0943951023931953).sin()
    voice_a = (tau * 110.0 * time + 0.85 * slow_a).sin()
    voice_b = (tau * 165.0 * time + 0.55 * slow_b).sin()
    voice_c = (tau * 220.0 * time + 0.35 * slow_c).sin()
    breath = 0.72 + 0.12 * slow_a + 0.09 * slow_b + 0.07 * slow_c
    samples = (breath * (
        0.46 * voice_a + 0.31 * voice_b + 0.20 * voice_c
    )).clamp(-0.95, 0.95)

    blocks = samples.reshape((frame_count, samples_per_frame))
    window = AbstractTensor.hanning(samples_per_frame).reshape(
        (1, samples_per_frame)
    )
    spectrum = AbstractTensor.rfft(blocks * window, axis=1)
    power = AbstractTensor.real(spectrum) ** 2 + AbstractTensor.imag(spectrum) ** 2
    frequencies = AbstractTensor.rfftfreq(
        samples_per_frame,
        d=1.0 / float(sample_rate),
        like=samples,
    )

    def normalized_band(low: float, high: float) -> AbstractTensor:
        mask = ((frequencies >= low) & (frequencies < high)).to_dtype("float64")
        band = (power * mask.reshape((1, -1))).sum(dim=1)
        total = power.sum(dim=1) + 1.0e-12
        return (band / total).sqrt()

    low = normalized_band(70.0, 140.0)
    middle = normalized_band(140.0, 195.0)
    high = normalized_band(195.0, 280.0)
    level = (blocks * blocks).mean(dim=1).sqrt().minimum(1.0)
    feeds = {
        "audio_low": tuple(float(value) for value in low.tolist()),
        "audio_mid": tuple(float(value) for value in middle.tolist()),
        "audio_high": tuple(float(value) for value in high.tolist()),
        "audio_level": tuple(float(value) for value in level.tolist()),
    }
    return ColumnarAudioTrack(
        samples=samples,
        sample_rate=int(sample_rate),
        feature_fps=int(feature_fps),
        feature_feeds=feeds,
    )


__all__ = ["ColumnarAudioTrack", "synthesize_columnar_audio"]
