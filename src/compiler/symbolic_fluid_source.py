"""A sound file driving one cell of the pool, as a boundary condition.

The stencil is untouched. This only says what one cell's surface is doing at
the start of each frame; every wave after that -- the spreading ring, the
reflection off the wrap-around edge, the interference where two rings meet --
is computed by the equations. That is the difference between driving the pool
and drawing on it.

Rate is the whole design problem. Audio carries tens of thousands of samples a
second; the pool advances one frame in a thirtieth of one, and a ripple needs
several frames to cross a visible distance. Feeding raw samples in at frame
rate would sample a waveform far below its own frequency, which is aliasing --
the surface would jitter with noise bearing no relation to the sound. So the
drive is the signal's *envelope* over each frame: loud passages push harder,
quiet ones let the surface settle, and a beat arrives as a ring. That is the
part of the audio the pool can actually resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceDrive:
    """Per-frame drive amplitudes in [0, 1], and where they came from."""

    envelope: np.ndarray
    sample_rate: int
    source: str

    def __len__(self) -> int:
        return int(self.envelope.size)

    def at(self, frame: int) -> float:
        """The drive for one frame; the signal loops so a run never stops."""

        if self.envelope.size == 0:
            return 0.0
        return float(self.envelope[int(frame) % self.envelope.size])


def read_surface_drive(path: str | Path, frame_duration: float) -> SurfaceDrive:
    """Reduce a WAV file to one drive amplitude per frame.

    Mixes channels, takes the root-mean-square of every block of samples one
    frame long, and normalises against the loudest block so the result is
    independent of how hot the file was recorded.
    """

    source = Path(path)
    with wave.open(str(source), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        raw = stream.readframes(stream.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(
            f"{source.name}: {width * 8}-bit samples are not supported; "
            "8, 16 or 32-bit PCM is"
        )
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if dtype is np.uint8:        # 8-bit PCM is unsigned, centred on 128
        samples = samples - 128.0
    if channels > 1:
        samples = samples[: samples.size // channels * channels]
        samples = samples.reshape(-1, channels).mean(axis=1)

    block = max(1, int(round(rate * float(frame_duration))))
    usable = samples.size // block * block
    if usable == 0:
        raise ValueError(f"{source.name} is shorter than one frame")
    blocks = samples[:usable].reshape(-1, block)
    envelope = np.sqrt((blocks * blocks).mean(axis=1))
    loudest = float(envelope.max())
    if loudest > 0.0:
        envelope = envelope / loudest
    return SurfaceDrive(envelope, rate, source.name)


@dataclass
class VoiceCoil:
    """A driver stated in Thiele-Small parameters, driven by force.

    ``free_air_resonance`` (Fs), ``moving_mass`` (Mms), ``force_factor`` (Bl)
    and ``mechanical_q`` (Qms) are the figures a driver is actually specified
    by, so suspension stiffness and mechanical loss are derived from them
    rather than invented:

        stiffness = Mms * (2*pi*Fs)**2
        loss      = 2*pi*Fs*Mms / Qms

    Defaults describe a large long-throw driver: 25 Hz free-air resonance,
    100 g moving mass, 15 T.m force factor. Force is integrated twice -- to
    velocity, then to displacement -- so the cone has real mass rolloff above
    resonance instead of following the signal directly.
    """

    free_air_resonance: float = 25.0
    moving_mass: float = 0.1
    force_factor: float = 15.0
    mechanical_q: float = 4.0
    velocity: float = 0.0
    displacement: float = 0.0

    @property
    def angular_resonance(self) -> float:
        return 2.0 * np.pi * float(self.free_air_resonance)

    @property
    def stiffness(self) -> float:
        return float(self.moving_mass) * self.angular_resonance ** 2

    @property
    def loss(self) -> float:
        return (
            self.angular_resonance * float(self.moving_mass)
            / float(self.mechanical_q)
        )

    @property
    def resonance_hz(self) -> float:
        return float(self.free_air_resonance)

    def step(self, current: float, dt: float) -> float:
        """Advance the cone one sample period; returns its displacement."""

        acceleration = (
            self.force_factor * float(current)
            - self.stiffness * self.displacement
            - self.loss * self.velocity
        ) / float(self.moving_mass)
        self.velocity += acceleration * float(dt)
        self.displacement += self.velocity * float(dt)
        return self.displacement


def drive_surface_cone(
    state,
    coil: "VoiceCoil",
    sample: float,
    dt: float,
    *,
    row: int | None = None,
    column: int | None = None,
) -> float:
    """Hold one cell's surface at the cone's displacement."""

    height = state.height
    source_row = height.shape[0] // 2 if row is None else int(row)
    source_column = height.shape[1] // 2 if column is None else int(column)
    displacement = coil.step(sample, dt)
    height[source_row, source_column] = 1.0 + displacement
    return displacement


def emit_tracer(
    state,
    rate: float,
    dt: float,
    *,
    row: int | None = None,
    column: int | None = None,
    radius: int = 0,
) -> float:
    """Release dye at the cone, as a rate rather than a level.

    The initial dye is a finite blob: the flow disperses it and diffusion
    finishes it, so after a few seconds there is nothing left to show the
    motion. A source keeps the picture alive.

    Injected proportional to ``dt``, so a halved substep injects half as much
    and the amount released over a second does not depend on how the window
    happened to be subdivided. Clamped to the local water column because the
    model publishes ``tracer_bounds`` as ``max(0, -tracer, tracer - height)``:
    dye beyond the depth is a declared physical violation, not a bright colour.

    Returns the amount actually released.
    """

    tracer = state.tracer
    height = state.height
    source_row = tracer.shape[0] // 2 if row is None else int(row)
    source_column = tracer.shape[1] // 2 if column is None else int(column)
    span = max(0, int(radius))
    rows = range(source_row - span, source_row + span + 1)
    columns = range(source_column - span, source_column + span + 1)
    released = 0.0
    for r in rows:
        for c in columns:
            r %= tracer.shape[0]
            c %= tracer.shape[1]
            ceiling = float(height[r, c])
            before = float(tracer[r, c])
            after = min(before + float(rate) * float(dt), ceiling)
            tracer[r, c] = after
            released += after - before
    return released


def drive_surface_cell(
    state,
    amplitude: float,
    *,
    row: int | None = None,
    column: int | None = None,
) -> None:
    """Displace one cell's surface, leaving its momentum alone.

    Raising the surface without adding momentum is a monopole: the fluid
    spreads outward evenly and the ring is circular. Pushing the momentum
    instead would bias one direction and give a dipole, which reads as a
    shove rather than a splash.
    """

    height = state.height
    source_row = height.shape[0] // 2 if row is None else int(row)
    source_column = height.shape[1] // 2 if column is None else int(column)
    height[source_row, source_column] += float(amplitude)




@dataclass
class SurfacePlayback:
    """Plays the driving signal on the simulation's clock, not the wall's.

    The pool advances a fixed slice of *simulated* time each frame, but a frame
    costs whatever wall time it costs -- a finer grid, or a timestep the
    controller had to retry, and it costs more. Playing the file at its own
    rate would drift away from the surface within seconds, and the ring you see
    would stop belonging to the beat you hear.

    So each frame's audio is resampled to the wall time that frame actually
    took. When the simulation runs slower than real time the sound stretches
    with it; the pool and the speakers stay on one clock. The cost is that
    the resampling ratio changes between frames, which is audible as a seam if
    frame times swing wildly -- steadier frames sound cleaner.
    """

    drive: "SurfaceDrive | None"
    samples: np.ndarray
    block: int
    device_rate: int
    file_rate: int = 0
    mixer: object | None = None

    def frame_audio(self, frame: int, elapsed: float) -> np.ndarray:
        """One frame of the signal, stretched to ``elapsed`` wall seconds."""

        if self.samples.size == 0 or elapsed <= 0.0:
            return np.zeros(0, dtype=np.int16)
        start = (int(frame) * self.block) % self.samples.size
        chunk = np.take(
            self.samples, np.arange(start, start + self.block),
            mode="wrap",
        )
        wanted = max(1, int(round(self.device_rate * float(elapsed))))
        # Linear resampling: the ratio is set by measurement, so it cannot be
        # baked in ahead of time.
        position = np.linspace(0.0, chunk.size - 1, wanted)
        resampled = np.interp(position, np.arange(chunk.size), chunk)
        peak = float(np.abs(resampled).max())
        if peak > 0.0:
            resampled = resampled / peak * 24000.0
        return resampled.astype(np.int16)

    def play(self, frame: int, elapsed: float) -> None:
        if self.mixer is None:
            return
        block = self.frame_audio(frame, elapsed)
        if block.size == 0:
            return
        stereo = np.repeat(block[:, None], 2, axis=1)
        self.mixer.sndarray.make_sound(np.ascontiguousarray(stereo)).play()


def open_surface_playback(
    path: str | Path,
    frame_duration: float,
    *,
    drive: "SurfaceDrive | None" = None,
    device_rate: int = 44100,
) -> SurfacePlayback:
    """Read the PCM and prepare playback, silent if no device is available.

    ``drive`` is optional: an envelope is only needed when the pool steps at
    the frame rate and cannot resolve the waveform. Stepping at the sample rate
    it consumes the samples themselves, and reducing them to an envelope would
    throw away the very thing the cone is there to follow.
    """

    source = Path(path)
    with wave.open(str(source), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        raw = stream.readframes(stream.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if dtype is np.uint8:
        samples = samples - 128.0
    if channels > 1:
        samples = samples[: samples.size // channels * channels]
        samples = samples.reshape(-1, channels).mean(axis=1)

    mixer = None
    try:
        import pygame

        pygame.mixer.init(frequency=device_rate, size=-16, channels=2)
        mixer = pygame
    except Exception:
        # No device, or a headless host: the surface still moves, silently.
        mixer = None
    return SurfacePlayback(
        drive, samples, max(1, int(round(rate * float(frame_duration)))),
        int(device_rate), int(rate), mixer,
    )


__all__ = [
    "SurfaceDrive", "read_surface_drive", "drive_surface_cell",
    "SurfacePlayback", "open_surface_playback",
    "VoiceCoil", "drive_surface_cone", "emit_tracer",
]
