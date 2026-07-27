"""AbstractTensor PCM quantization and exact frame/audio cadence accounting."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import struct

from ..abstraction import AbstractTensor
from .bitstream import tensor_octets_to_bytes


@dataclass(frozen=True)
class PCMFormat:
    """Uncompressed sample representation carried by an AVI audio stream."""

    sample_rate: int = 48_000
    channels: int = 2
    sample_format: str = "s16le"

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("PCM sample_rate must be positive")
        if self.channels not in {1, 2}:
            raise ValueError("courtesy PCM currently supports mono or stereo")
        if self.sample_format not in {"s16le", "f32le"}:
            raise ValueError("PCM sample_format must be 's16le' or 'f32le'")

    @property
    def bits_per_sample(self) -> int:
        return 16 if self.sample_format == "s16le" else 32

    @property
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8

    @property
    def block_align(self) -> int:
        return self.channels * self.bytes_per_sample

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.block_align

    @property
    def wave_format_tag(self) -> int:
        return 1 if self.sample_format == "s16le" else 3

    def wave_format_ex(self) -> bytes:
        """Return the canonical 16-byte PCM/IEEE-float WAVEFORMAT payload."""
        payload = struct.pack(
            "<HHIIHH",
            self.wave_format_tag,
            self.channels,
            self.sample_rate,
            self.bytes_per_second,
            self.block_align,
            self.bits_per_sample,
        )
        # PCM conventionally omits cbSize. IEEE float is non-PCM WAVEFORMATEX
        # and carries an explicit zero extension length.
        return payload if self.wave_format_tag == 1 else payload + b"\x00\x00"


def _validate_audio_shape(
    samples: AbstractTensor,
    pcm_format: PCMFormat,
) -> tuple[int, int]:
    if not isinstance(samples, AbstractTensor):
        raise TypeError("PCM samples must be an AbstractTensor")
    if samples.ndims() == 1:
        frame_count, channels = samples.shape[0], 1
    elif samples.ndims() == 2:
        frame_count, channels = samples.shape
    else:
        raise ValueError("PCM samples must have shape (samples,) or (samples, channels)")
    if channels != pcm_format.channels:
        raise ValueError(
            f"PCM input has {channels} channels; format requires "
            f"{pcm_format.channels}"
        )
    if not bool(samples.isfinite().all().item()):
        raise ValueError("PCM samples must be finite")
    return int(frame_count), int(channels)


def encode_pcm(
    samples: AbstractTensor,
    *,
    pcm_format: PCMFormat,
    gain: float = 1.0,
    clip: bool = True,
) -> bytes:
    """Quantize normalized tensor samples and serialize interleaved PCM bytes."""
    _validate_audio_shape(samples, pcm_format)
    values = samples * gain
    if clip:
        values = values.clamp(-1.0, 1.0)
    elif not bool(((values >= -1.0) & (values <= 1.0)).all().item()):
        raise ValueError("PCM samples exceed [-1, 1] while clipping is disabled")

    flat = values.reshape(-1)
    if pcm_format.sample_format == "s16le":
        nonnegative = (flat >= 0).to_dtype("float")
        scale = nonnegative * 32767.0 + (1.0 - nonnegative) * 32768.0
        quantized = flat.sign() * ((flat.abs() * scale + 0.5) // 1)
        twos_complement = quantized.to_dtype("int64") % 65536
        low_octets = twos_complement % 256
        high_octets = (twos_complement // 256) % 256
        interleaved = AbstractTensor.cat(
            (
                low_octets.unsqueeze(1),
                high_octets.unsqueeze(1),
            ),
            dim=1,
        ).reshape(-1)
        return tensor_octets_to_bytes(interleaved)
    return b"".join(
        struct.pack("<f", float(value))
        for value in flat.tolist()
    )


class RationalAudioScheduler:
    """Assign integer audio sample frames to video frames without drift."""

    def __init__(
        self,
        *,
        sample_rate: int,
        fps: int | float | Fraction,
    ) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        cadence = Fraction(fps).limit_denominator(1_000_000)
        if cadence <= 0:
            raise ValueError("fps must be positive")
        self.sample_rate = int(sample_rate)
        self.rate = int(cadence.numerator)
        self.scale = int(cadence.denominator)
        self.frame_index = 0
        self.sample_index = 0

    def samples_for_next_frame(self) -> int:
        """Advance one video frame and return its exact audio-frame budget."""
        self.frame_index += 1
        next_sample = (
            self.frame_index * self.sample_rate * self.scale
        ) // self.rate
        count = next_sample - self.sample_index
        self.sample_index = next_sample
        return count


__all__ = ["PCMFormat", "RationalAudioScheduler", "encode_pcm"]
