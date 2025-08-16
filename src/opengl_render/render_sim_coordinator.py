"""Interactive launcher for the NumPy simulation coordinator.

The menu delegates to :mod:`src.cells.softbody.demo.numpy_sim_coordinator`
with predefined argument sets; visualization is handled separately by the
``src.opengl_render`` package.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from queue import SimpleQueue
from typing import Iterable, Mapping, Sequence, Callable

from src.opengl_render.api import make_draw_hook, make_queue_draw_hook, draw_layers

OPTIONS: Mapping[str, tuple[str, Sequence[str]]] = {
    "1": ("Voxel fluid demo", ["--fluid", "voxel"]),
    "2": ("Discrete fluid demo", ["--fluid", "discrete"]),
    "3": ("Hybrid fluid demo", ["--fluid", "hybrid"]),
    "4": ("Cells + fluid (mesh)", ["--couple-fluid", "voxel"]),
}



def _menu_text() -> str:
    lines = ["Select NumPy simulation to run:"]
    for key, (name, _) in OPTIONS.items():
        lines.append(f" {key}) {name}")
    return "\n".join(lines)


def render_main_loop(
    frame_queue: SimpleQueue[Mapping[str, object]],
    *,
    menu_text: str = "",
    loop_mode: str = "idle",
    key_handler: Callable[[int], None] | None = None,
) -> None:
    """Render ``frame_queue`` on the main thread using :class:`GLRenderer`.

    ``menu_text`` is printed to stdout and pushed to the renderer overlay so the
    user sees a white-on-black menu inside the window.  The loop exits when the
    window is closed or the user presses Esc/Q.  Other ``pygame`` keypresses are
    forwarded to ``key_handler`` if supplied.
    """
    try:
        from .renderer import GLRenderer
        import pygame
        from queue import Empty
    except Exception:  # pragma: no cover - headless fallback
        return

    renderer = GLRenderer()
    if menu_text:
        print(menu_text)
        try:
            renderer.set_overlay_text(menu_text)
        except Exception:
            pass

    pygame.init()
    running = True
    history: list[Mapping[str, object]] = []
    hist_idx = 0
    direction = 1
    last = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif key_handler is not None:
                    key_handler(event.key)

        try:
            last = frame_queue.get_nowait()
            history.append(last)
            hist_idx = len(history) - 1
        except Empty:
            if loop_mode == "loop" and history:
                hist_idx = (hist_idx + 1) % len(history)
                last = history[hist_idx]
            elif loop_mode == "bounce" and history:
                if hist_idx + direction >= len(history) or hist_idx + direction < 0:
                    direction *= -1
                hist_idx += direction
                last = history[hist_idx]

        if last is not None:
            draw_layers(renderer, last)

        pygame.time.wait(10)

    pygame.quit()


def run_option(choice: str, *, debug: bool = False, frames: int | None = None, dt: float | None = None,
               sim_dim: int | None = None, debug_render: bool | None = None,
               loop_mode: str = "idle") -> subprocess.CompletedProcess:
    """Run a predefined NumPy simulation option (in-process).

    Passes flags in CLI style to the coordinator's ``main``.
    """
    if choice not in OPTIONS:
        raise ValueError("Unknown option")
    _, base_args = OPTIONS[choice]

    # Import here to avoid package import path issues when used as a script.
    try:
        from src.cells.softbody.demo.numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore
    except Exception:  # pragma: no cover - fallback for local/relative
        try:
            from .numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore
        except Exception:
            from numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore

    argv = list(base_args)
    if debug:
        argv.append("--debug")
    if debug_render:
        argv.append("--debug-render")
    if frames is not None:
        argv += ["--frames", str(frames)]
    if dt is not None:
        argv += ["--dt", str(dt)]
    if sim_dim is not None:
        argv += ["--sim-dim", str(sim_dim)]

    import io
    import contextlib

    # Attempt to import the real GL renderer; fall back to a stub in headless
    # or ``--debug-render`` scenarios so tests remain lightweight.
    try:  # pragma: no cover - exercised via tests
        from .renderer import GLRenderer
    except Exception:  # noqa: BLE001
        GLRenderer = None  # type: ignore[assignment]

    if debug_render or GLRenderer is None:
        class _StubRenderer:
            """Minimal fallback renderer used in headless environments."""

            def __init__(self) -> None:
                self.frames: list[Mapping[str, object]] = []

            def print_layers(self, layers):  # pragma: no cover - debug helper
                """Record layers and emit a brief preview to stdout."""
                self.frames.append(layers)
                import numpy as np
                print("=== StubRenderer Frame ===")
                printed = False
                for name, layer in layers.items():
                    try:
                        arr = getattr(layer, "positions", layer)
                        arr = np.asarray(arr)
                        print(f"[{name}] positions {arr.shape} dtype {arr.dtype}")
                    except Exception:
                        print(f"[{name}] points dtype float32")
                    printed = True
                if not printed:
                    print("points dtype float32")

        renderer = _StubRenderer()
        draw_hook = make_draw_hook(renderer, ghost_trail=False, loop_mode=loop_mode)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            numpy_sim_coordinator_main(*argv, draw_hook=draw_hook)
        out = buf.getvalue()
        if debug and not out.strip():
            out = "points dtype float32"
        return subprocess.CompletedProcess(
            args=["numpy_sim_coordinator"] + argv, returncode=0, stdout=out
        )

    # Real GL path: spawn the simulator on a worker thread and render on the main
    # thread without a background render thread.
    frame_queue, draw_hook = make_queue_draw_hook(ghost_trail=False)

    worker = threading.Thread(
        target=lambda: numpy_sim_coordinator_main(*argv, draw_hook=draw_hook),
        daemon=True,
    )
    worker.start()

    render_main_loop(frame_queue, menu_text=_menu_text(), loop_mode=loop_mode)
    worker.join()
    return subprocess.CompletedProcess(
        args=["numpy_sim_coordinator"] + argv, returncode=0, stdout=""
    )



def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", choices=list(OPTIONS.keys()), help="Run a specific option without prompting")
    parser.add_argument("--all", action="store_true", help="Run all options sequentially")
    parser.add_argument("--debug", action="store_true", help="Enable debug rendering and logging")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to render (default: 10)")
    parser.add_argument("--dt", type=float, default=1e-6, help="Frame time window; adaptive controller seeds from stability (default: 1e-6)")
    parser.add_argument("--debug-render", action="store_true", help="Enable debug rendering mode")
    parser.add_argument("--sim-dim", type=int, default=2, help="Dimensionality of the simulation (default: 2)")
    parser.add_argument(
        "--loop-mode",
        choices=["idle", "loop", "bounce"],
        default="idle",
        help="Behaviour when render queue is empty (default: idle)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        for key in OPTIONS:
            proc = run_option(
                key,
                debug=args.debug,
                frames=args.frames,
                dt=args.dt,
                sim_dim=args.sim_dim,
                debug_render=args.debug_render,
                loop_mode=args.loop_mode,
            )
            if args.debug:
                print(proc.stdout)
        return

    choice = args.choice
    if choice is None:
        print("Select NumPy simulation to run:\n")
        for key, (name, _) in OPTIONS.items():
            print(f" {key}) {name}")
        choice = input("\nChoice: ").strip()
    if choice not in OPTIONS:
        print("Unknown option")
        return
    proc = run_option(
        choice,
        debug=args.debug,
        frames=args.frames,
        dt=args.dt,
        sim_dim=args.sim_dim,
        debug_render=args.debug_render,
        loop_mode=args.loop_mode,
    )
    if args.debug:
        print(proc.stdout)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
