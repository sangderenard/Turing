"""Record an animated AbstractTensor field with our JPEG and AVI writers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ...abstraction import AbstractTensor as AT
from ..pcm import PCMFormat, RationalAudioScheduler
from .avi import MJPEGAVIWriter


def record_tensor_field(
    *,
    output: Path,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    backend: str,
    device: str | None = None,
    color: bool = True,
    tone_hz: float = 0.0,
    sample_rate: int = 48_000,
    channels: int = 2,
    pcm_dtype: str = "s16le",
    opendml: bool = False,
    segment_bytes: int = 1 << 30,
) -> Path:
    if width % 8 or height % 8:
        raise ValueError("frame dimensions must be divisible by 8")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    pcm_format = (
        PCMFormat(
            sample_rate=sample_rate,
            channels=channels,
            sample_format=pcm_dtype,
        )
        if tone_hz > 0
        else None
    )
    scheduler = (
        RationalAudioScheduler(sample_rate=sample_rate, fps=fps)
        if pcm_format is not None
        else None
    )
    audio_position = 0
    with AT.use_backend(backend, device=device):
        x = AT.arange(width).unsqueeze(0) + AT.zeros((height, 1))
        y = AT.arange(height).unsqueeze(1) + AT.zeros((1, width))
        x = x / max(width, 1)
        y = y / max(height, 1)
        with MJPEGAVIWriter(
            output,
            width=width,
            height=height,
            fps=fps,
            pcm_format=pcm_format,
            opendml=opendml,
            segment_bytes=segment_bytes,
        ) as writer:
            for frame_index in range(frame_count):
                phase = 2.0 * math.pi * frame_index / frame_count
                field = (
                    127.0
                    + 54.0 * (2.0 * math.pi * x + phase).sin()
                    + 43.0 * (2.0 * math.pi * y - 1.7 * phase).cos()
                    + 31.0
                    * (
                        4.0 * math.pi * (x + y)
                        + 0.7 * phase
                    ).sin()
                )
                if color:
                    color_phase = field * (2.0 * math.pi / 255.0)
                    samples = AT.stack(
                        (
                            127.5 + 127.5 * color_phase.cos(),
                            127.5
                            + 127.5
                            * (color_phase + 2.0943951023931953).cos(),
                            127.5
                            + 127.5
                            * (color_phase + 4.1887902047863905).cos(),
                        ),
                        dim=-1,
                    )
                else:
                    samples = field
                samples = ((samples + 0.5) // 1).clamp(0, 255)
                writer.append_frame(samples.jpg())
                if scheduler is not None:
                    count = scheduler.samples_for_next_frame()
                    positions = (
                        AT.arange(count) + audio_position
                    ) / sample_rate
                    base = (
                        2.0 * math.pi * tone_hz * positions
                    ).sin() * 0.22
                    if channels == 1:
                        audio_samples = base
                    else:
                        channel_values = [base]
                        for channel in range(1, channels):
                            channel_values.append(
                                (
                                    2.0 * math.pi * tone_hz * positions
                                    + channel * 0.37
                                ).sin()
                                * 0.22
                            )
                        audio_samples = AT.stack(
                            tuple(channel_values), dim=1
                        )
                    writer.append_audio_tensor(audio_samples)
                    audio_position += count
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tensor_field.avi"))
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch", "c", "glsl"),
        default="numpy",
    )
    parser.add_argument(
        "--device",
        help="backend device, for example cuda with --backend torch",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="record one luma plane instead of the default RGB frames",
    )
    parser.add_argument(
        "--tone-hz",
        type=float,
        default=0.0,
        help="include a generated PCM tone; 0 keeps the AVI video-only",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument(
        "--pcm-dtype",
        choices=("s16le", "f32le"),
        default="s16le",
    )
    parser.add_argument(
        "--opendml",
        action="store_true",
        help="write AVI/AVIX segments with two-level OpenDML indexes",
    )
    parser.add_argument(
        "--segment-bytes",
        type=int,
        default=1 << 30,
        help="maximum OpenDML movi payload per RIFF segment",
    )
    args = parser.parse_args(argv)
    destination = record_tensor_field(
        output=args.output,
        width=args.width,
        height=args.height,
        frame_count=args.frames,
        fps=args.fps,
        backend=args.backend,
        device=args.device,
        color=not args.grayscale,
        tone_hz=args.tone_hz,
        sample_rate=args.sample_rate,
        channels=args.channels,
        pcm_dtype=args.pcm_dtype,
        opendml=args.opendml,
        segment_bytes=args.segment_bytes,
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
