"""Live shell for the SymPy -> repository SSA -> LLVM fluid program.

The shell owns presentation only.  Fluid mathematics remains in
``symbolic_fluid_model.py`` as SymPy equations, and timestep rollback/retry
remains in the repository ``dt_system``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.compiler.symbolic_fluid_native_runtime import (
    compile_native_symbolic_fluid_step,
    load_symbolic_fluid_managed_functions,
)


def _rgb(state: SymbolicFluidGridState) -> np.ndarray:
    """Presentation-only mapping from published fields to an RGB image."""

    height = state.height
    anomaly = np.clip((height - 0.98) * 6.0, 0.0, 1.0)
    tracer = np.clip(state.tracer / np.maximum(height, 1.0e-30), 0.0, 1.0)
    speed = np.sqrt(state.momentum_x ** 2 + state.momentum_y ** 2)
    speed = np.clip(speed * 14.0, 0.0, 1.0)
    return np.stack((tracer, speed + 0.25 * anomaly, anomaly), axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--frames", type=int, default=0, help="zero runs until closed")
    parser.add_argument("--frame-duration", type=float, default=1.0 / 30.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scale", type=int, default=20)
    parser.add_argument(
        "--audio", type=Path, default=None,
        help="WAV file whose envelope drives one cell of the surface",
    )
    parser.add_argument(
        "--avi", type=Path,
        default=Path("build/symbolic-fluid-live/pool.avi"),
        help="render to an AVI with the repository MJPEG/PCM writer",
    )
    parser.add_argument(
        "--video-fps", type=int, default=30,
        help="video frame rate; the pool is sampled every Nth step",
    )
    parser.add_argument(
        "--no-audio-rate", dest="audio_rate", action="store_false",
        help="step at the frame rate with the envelope drive instead",
    )
    parser.set_defaults(audio_rate=True)
    parser.add_argument(
        "--silent", action="store_true",
        help="drive the surface from the file but do not play it",
    )
    parser.add_argument(
        "--drive", type=float, default=0.05,
        help="surface displacement at full scale, in units of still depth",
    )
    parser.add_argument(
        "--dye", type=float, default=2.0,
        help="dye released at the cone per second, in depth units",
    )
    parser.add_argument(
        "--dye-radius", type=int, default=1,
        help="cells either side of the cone that also release dye",
    )
    parser.add_argument(
        "--build-directory", type=Path,
        default=Path("build/symbolic-fluid-live/llvm"),
    )
    parser.add_argument(
        "--no-avi", dest="avi", action="store_const", const=None,
        help="run without rendering a file",
    )
    args = parser.parse_args()
    if args.audio is None:
        # Audio rate and the voice coil need a signal; without one the pool
        # runs at the frame rate as before.
        args.audio_rate = False

    native = compile_native_symbolic_fluid_step(args.build_directory)
    advance = load_symbolic_fluid_managed_functions(native)[
        "symbolic_fluid_advance"
    ]
    state = SymbolicFluidGridState.initial(args.grid, args.grid)
    targets = Targets(
        cfl=0.45,
        div_max=1.0,
        mass_max=1.0e-8,
        error_limits={"height_positivity": 0.0, "tracer_bounds": 0.0},
    )
    controller = STController(dt_min=1.0e-8, dt_max=args.frame_duration)
    dt = float(args.dt)

    pygame = None
    screen = None
    clock = None
    if not args.headless:
        import pygame as pygame_module

        pygame = pygame_module
        pygame.init()
        screen = pygame.display.set_mode(
            (args.grid * args.scale, args.grid * args.scale)
        )
        pygame.display.set_caption("SymPy fluid · repository SSA · LLVM")
        clock = pygame.time.Clock()

    drive = None
    playback = None
    sample_period = None
    # A pinned audio-rate frame is thousands of substeps and takes many
    # seconds. Presenting once per frame means the window services no paint or
    # input messages for that whole time, so the compositor shows it blank and
    # it cannot even be closed. Present from inside the substep instead,
    # throttled by wall clock so it costs the simulation almost nothing.
    presented = [time.perf_counter()]
    alive = [True]

    def present(force: bool = False) -> None:
        if pygame is None:
            return
        now = time.perf_counter()
        if not force and now - presented[0] < 0.05:
            return
        presented[0] = now
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                alive[0] = False
        pixels = np.ascontiguousarray((_rgb(state) * 255.0).astype(np.uint8))
        image = pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1))
        screen.blit(pygame.transform.scale(image, screen.get_size()), (0, 0))
        pygame.display.flip()
    frame_started = time.perf_counter()
    if args.audio is not None:
        from src.compiler.symbolic_fluid_source import open_surface_playback

        if args.audio_rate:
            from src.compiler.symbolic_fluid_source import VoiceCoil

            # No envelope here. One substep is one sample, so the coil is
            # driven by the waveform itself; an envelope would discard it.
            playback = open_surface_playback(args.audio, args.frame_duration)
            samples = playback.samples
            source_rate = playback.file_rate
            peak = float(np.abs(samples).max()) or 1.0
            samples = samples / peak
            # The frame is one video frame; the audio sample period is the
            # *sub*-step inside it. Putting the sample period on the frame made
            # the controller propose a dt larger than the window it was
            # subdividing, so every frame clamped to a single step.
            args.frame_duration = 1.0 / float(args.video_fps)
            # The substep is one sample *of the file*. Using the output
            # device rate stepped an 8 kHz recording at 44.1 kHz spacing, so
            # the pool heard it five times slower than it was played.
            sample_period = 1.0 / float(source_rate)
            dt = sample_period
            coil = VoiceCoil()
            print(
                f"{source_rate} Hz sub-steps, "
                f"{int(round(args.frame_duration / sample_period))} per video "
                f"frame; cone resonance {coil.resonance_hz:.1f} Hz"
            )
        if not args.silent:
            from src.compiler.symbolic_fluid_source import open_surface_playback

            if playback is None:
                playback = open_surface_playback(
                    args.audio, args.frame_duration, drive=drive,
                )
            print(
                "audio on the simulation clock"
                if playback.mixer is not None else
                "no audio device; the surface still moves"
            )
        if drive is not None:
            print(
                f"driving one cell from {drive.source}: {len(drive)} frames "
                f"of envelope at {drive.sample_rate} Hz"
            )

    writer = None
    video_stride = 1
    if args.avi is not None:
        from fractions import Fraction
        from src.common.tensors.compression.containers.avi import MJPEGAVIWriter
        from src.common.tensors.compression.pcm import PCMFormat

        rate = playback.device_rate if playback is not None else 44100
        pcm = (
            PCMFormat(channels=1, sample_rate=rate, sample_format="s16le")
            if playback is not None else None
        )
        writer = MJPEGAVIWriter(
            args.avi, width=args.grid, height=args.grid,
            fps=Fraction(args.video_fps), pcm_format=pcm,
        )
        # One video frame every Nth simulation frame. At audio rate that is
        # sample_rate/fps steps; at ordinary rate it is one.
        # One superstep advances exactly one video frame.
        video_stride = 1
        print(
            f"rendering {args.avi} at {args.video_fps} fps, "
            f"one video frame per {video_stride} step(s)"
        )

    sample_cursor = [0]
    frame = 0
    running = True
    try:
        while running and (args.frames == 0 or frame < args.frames):
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
            if not running:
                break
            if drive is not None and not args.audio_rate:
                # A boundary condition applied between accepted frames, never
                # inside a step: the controller may reject and retry a step,
                # and a source that fired on each attempt would inject a
                # different amount of water depending on how many retries the
                # timestep happened to need.
                from src.compiler.symbolic_fluid_source import drive_surface_cell

                drive_surface_cell(state, args.drive * drive.at(frame))
            if playback is not None:
                # The frame that just finished sets the stretch for this one,
                # so the sound tracks the simulation's actual pace instead of
                # assuming it keeps up with real time.
                now = time.perf_counter()
                playback.play(frame, now - frame_started)
                frame_started = now
            attempts: list[dict] = []
            stepper = advance
            audio_rate_active = args.audio is not None and args.audio_rate
            if audio_rate_active:
                from src.compiler.symbolic_fluid_source import (
                    drive_surface_cone, emit_tracer,
                )

                def stepper(state, step_dt, _advance=advance):
                    # One call is one audio sample: the cone integrates the
                    # force for that sample, then the fluid takes the step.
                    drive_surface_cone(
                        state, coil,
                        args.drive
                        * float(samples[sample_cursor[0] % samples.size]),
                        float(step_dt),
                    )
                    sample_cursor[0] += 1
                    if args.dye > 0.0:
                        emit_tracer(
                            state, args.dye, float(step_dt),
                            radius=args.dye_radius,
                        )
                    result = _advance(state, step_dt)
                    present()
                    return result

            advanced, dt, metrics = run_superstep(
                state,
                args.frame_duration,
                dt,
                state.dx,
                targets,
                controller,
                stepper,
                # One iteration per substep. Under a pinned interior that is
                # the pinned period, not whatever dt the controller last
                # proposed -- the proposal is not what will be taken.
                max_iters=max(256, 64 + int(
                    args.frame_duration
                    / (sample_period if audio_rate_active else max(dt, 1e-12))
                )),
                substep="pinned" if audio_rate_active else "steered",
                substep_dt=sample_period if audio_rate_active else None,
                attempt_log=attempts,
            )
            rejected = sum(not item["accepted"] for item in attempts)
            print(
                f"frame={frame:05d} advanced={float(advanced):.8g} "
                f"dt_next={float(dt):.8g} attempts={len(attempts)} "
                f"rejected={rejected} mass_error={metrics.mass_err:.3e} "
                f"wave_speed={state.last_wave_speed:.6g}"
            )
            present(force=True)
            if not alive[0]:
                running = False
            if writer is not None and frame % video_stride == 0:
                from src.common.tensors.abstraction import AbstractTensor as AT

                surface = np.clip(
                    (state.height - 1.0) * 6.0 + 0.5, 0.0, 1.0
                )
                dye = np.clip(state.tracer / np.maximum(state.height, 1e-9), 0.0, 1.0)
                speed = np.clip(
                    np.sqrt(state.momentum_x ** 2 + state.momentum_y ** 2)
                    / np.maximum(state.height, 1e-9) * 3.0, 0.0, 1.0,
                )
                rgb = np.stack((surface, dye, speed), axis=-1) * 255.0
                writer.append_frame(AT.get_tensor(rgb.astype(np.float64)).jpg())
                if playback is not None:
                    block = playback.samples
                    start = (frame * playback.block) % max(block.size, 1)
                    chunk = np.take(
                        block, np.arange(start, start + playback.block),
                        mode="wrap",
                    )
                    peak = float(np.abs(chunk).max()) or 1.0
                    writer.append_audio(
                        (chunk / peak * 24000.0).astype("<i2").tobytes()
                    )
            frame += 1
    finally:
        if writer is not None:
            print("wrote", writer.close())
        if pygame is not None:
            pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
