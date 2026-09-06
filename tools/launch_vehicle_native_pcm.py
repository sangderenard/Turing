"""Pre-expand audio or generated excitation into a native rig PCM tape.

No external decoder process is started.  ``soundfile``/libsndfile decodes an
input file; generated profiles require only the Python standard library.  The
native vehicle C shell receives one fixed binary payload containing a stereo
graph-exciter tape plus four clean pulsed carrier lanes. The shell maps left
and right excitation to the matching suspension paths, offsetting the rear
consumption time by wheelbase and vehicle speed. The resultant world-space
motion of each ``suspension.<corner>.coilover_chassis`` attachment—not the
input tape, road, hub, or wheel center—amplitude-modulates its carrier for
stereo system-audio output.
"""

from __future__ import annotations

import argparse
from array import array
import math
from pathlib import Path
import struct
import subprocess
import tempfile

MAGIC = b"TVPCM3\0\0"
RIG_MODES = {"cage-drive": 0, "suspension-test": 1}


def _decode(path: Path) -> tuple[list[float], list[float], int]:
    import soundfile as sf

    with sf.SoundFile(path) as stream:
        channels = int(stream.channels)
        rate = int(stream.samplerate)
        held = array("f")
        held.frombytes(stream.buffer_read(stream.frames, dtype="float32"))
    if channels < 1:
        raise ValueError("audio file has no channels")
    frames = len(held) // channels
    left = [float(held[index * channels]) for index in range(frames)]
    right_index = 1 if channels > 1 else 0
    right = [float(held[index * channels + right_index]) for index in range(frames)]
    return left, right, rate


def _resample(values: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate == target_rate or not values:
        return values
    count = max(1, round(len(values) * target_rate / source_rate))
    scale = source_rate / target_rate
    result = []
    for index in range(count):
        source = min(len(values) - 1, index * scale)
        lo = int(source)
        hi = min(len(values) - 1, lo + 1)
        mix = source - lo
        result.append(values[lo] * (1.0 - mix) + values[hi] * mix)
    return result


def _raised_cosine_pulse(phase: float, duty: float) -> float:
    wrapped = phase - math.floor(phase)
    if wrapped >= duty:
        return 0.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * wrapped / duty)


def _generate(kind: str, duration: float, rate: int, speed: float,
              bump_spacing: float, whoop_wavelength: float) -> tuple[list[float], list[float]]:
    count = max(1, round(duration * rate))
    speed_magnitude = max(abs(speed), 0.05)
    left, right = [0.0] * count, [0.0] * count
    for index in range(count):
        t = index / rate
        distance = speed_magnitude * t
        bump_phase = distance / max(bump_spacing, 0.05)
        bump_index = int(math.floor(bump_phase))
        bump = 0.045 * _raised_cosine_pulse(bump_phase, 0.16)
        whoop = 0.065 * math.sin(2.0 * math.pi * distance / max(whoop_wavelength, 0.25))
        if kind in {"bumps", "interleaved"}:
            (left if bump_index % 2 == 0 else right)[index] += bump
        if kind in {"whoopdedoos", "interleaved"}:
            left[index] += whoop
            right[index] += whoop
    return left, right


def _expand(left: list[float], right: list[float], rate: int, speed: float,
            wheelbase: float, chord: tuple[float, float, float, float],
            carrier_depth: float, pulse_hz: float) -> array:
    count = min(len(left), len(right))
    expanded = array("f")
    for index in range(count):
        t = index / rate
        pulse = _raised_cosine_pulse(t * pulse_hz, 0.22)
        # The first two lanes are a graph-exciter tape, not result coordinates.
        # Clean carrier pulses ride separately; only the native shell may
        # modulate them with resultant suspension-attachment motion.
        carriers = tuple(
            carrier_depth * pulse * math.sin(2.0 * math.pi * frequency * t)
            for frequency in chord
        )
        expanded.extend((float(left[index]), float(right[index]),
                         *map(float, carriers)))
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "cage-drive excites chassis/cage attachments while unilateral "
            "roller carriages cannot pull a wheel down; suspension-test "
            "drives the bidirectionally locked roller-carriage trajectory. "
            "In both modes system audio observes resultant coilover_chassis "
            "positions only."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--generate", choices=("bumps", "whoopdedoos", "interleaved"))
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--vehicle-speed-m-s", type=float, default=8.0)
    parser.add_argument("--wheelbase-m", type=float, default=1.24)
    parser.add_argument("--rig-mode", choices=tuple(RIG_MODES), default="cage-drive")
    parser.add_argument("--bump-spacing-m", type=float, default=1.6)
    parser.add_argument("--whoop-wavelength-m", type=float, default=3.8)
    parser.add_argument("--carrier-chord-hz", default="27.5,34.375,41.25,51.5625")
    parser.add_argument("--carrier-depth", type=float, default=0.22)
    parser.add_argument("--carrier-pulse-hz", type=float, default=2.0)
    parser.add_argument("--pcm-output", type=Path)
    args = parser.parse_args()
    chord = tuple(float(value) for value in args.carrier_chord_hz.split(","))
    if len(chord) != 4 or any(value <= 0 for value in chord):
        raise ValueError("--carrier-chord-hz requires four positive frequencies")
    if args.audio:
        left, right, source_rate = _decode(args.audio)
        left = _resample(left, source_rate, args.sample_rate)
        right = _resample(right, source_rate, args.sample_rate)
    else:
        left, right = _generate(
            args.generate, args.duration, args.sample_rate,
            args.vehicle_speed_m_s, args.bump_spacing_m, args.whoop_wavelength_m,
        )
    payload = _expand(
        left, right, args.sample_rate, args.vehicle_speed_m_s,
        args.wheelbase_m, chord, args.carrier_depth, args.carrier_pulse_hz,
    )
    temporary = None
    if args.pcm_output:
        pcm_path = args.pcm_output.resolve()
        pcm_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="turing_vehicle_pcm_")
        pcm_path = Path(temporary.name) / "wheel-input.tvpcm"
    with pcm_path.open("wb") as stream:
        stream.write(struct.pack(
            "<8sIIIQff", MAGIC, args.sample_rate, 6, RIG_MODES[args.rig_mode],
            len(payload) // 6,
            args.vehicle_speed_m_s, args.wheelbase_m,
        ))
        stream.write(payload.tobytes())
    completed = subprocess.run([str(args.executable.resolve()), str(pcm_path)], check=False)
    if temporary is not None:
        temporary.cleanup()
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
