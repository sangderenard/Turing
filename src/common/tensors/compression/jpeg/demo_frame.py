"""Render an AbstractTensor Mandelbrot field into our own baseline JFIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...abstraction import AbstractTensor as AT
from ...accelerator_backends.demo_mandelbrot_fusion import mandelbrot_escape

def render_mandelbrot_frame(
    *,
    width: int,
    height: int,
    iterations: int,
    center: complex,
    span: float,
    backend: str,
    output: Path,
    device: str | None = None,
    color: bool = True,
) -> Path:
    if width % 8 or height % 8:
        raise ValueError("frame dimensions must be divisible by 8")
    if width < 8 or height < 8:
        raise ValueError("frame dimensions must be at least 8 by 8")
    with AT.use_backend(backend, device=device):
        x = AT.arange(width) / max(width - 1, 1)
        y = AT.arange(height) / max(height - 1, 1)
        unit_x = x.unsqueeze(0) + AT.zeros((height, 1))
        unit_y = y.unsqueeze(1) + AT.zeros((1, width))
        aspect = width / height
        cx = center.real + (unit_x - 0.5) * span * aspect
        cy = center.imag + (unit_y - 0.5) * span
        counts = mandelbrot_escape(cx, cy, iterations)
        scaled = counts * (255.0 / max(iterations, 1))
        if color:
            phase = 6.283185307179586 * (counts / max(iterations, 1))
            escaped = (counts < iterations).to_dtype("float")
            channels = (
                127.5 + 127.5 * phase.cos(),
                127.5 + 127.5 * (phase + 2.0943951023931953).cos(),
                127.5 + 127.5 * (phase + 4.1887902047863905).cos(),
            )
            samples = AT.stack(
                tuple(channel * escaped for channel in channels),
                dim=-1,
            )
        else:
            samples = scaled
        samples = ((samples + 0.5) // 1).clamp(0, 255)
        return samples.jpg(path=output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument(
        "--center",
        type=complex,
        default=complex(-0.743643887, 0.131825904),
    )
    parser.add_argument("--span", type=float, default=0.01)
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
        "--output",
        type=Path,
        default=Path("tensor_mandelbrot.jpg"),
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="encode one luma plane instead of the default RGB image",
    )
    args = parser.parse_args(argv)
    destination = render_mandelbrot_frame(
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        center=args.center,
        span=args.span,
        backend=args.backend,
        device=args.device,
        output=args.output,
        color=not args.grayscale,
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
