"""Publish the Python columnar state machine as a Wasm RGB-preview page."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .columnar_multifluid_kernels import columnar_multifluid_rgb_step


_PAGE = '''
TURING_PAGE = {
    "entrypoint": "columnar_multifluid_rgb_step",
    "title": "Managed Columnar Multifluid World",
    "slug": "managed-columnar-multifluid-world",
    "width": 384,
    "height": 268,
    "probe_size": 16,
    "feeds": {
        "column_x": 0.5,
        "column_y": 0.5,
        "rest_surface": 1.0,
        "displacement": 0.0,
        "displacement_velocity": 0.0,
        "managed_time": 0.0,
        "dt": 0.025
    },
    "feed_expressions": {
        "column_x": "(x + 0.5) * 10.0 / w",
        "column_y": "(y + 0.5) * 7.0 / h",
        "rest_surface": "1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))",
        "displacement": "0.0",
        "displacement_velocity": "0.0",
        "managed_time": "0.0",
        "dt": "0.025"
    },
    "state_feedback": {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "managed_time": "next_time"
    },
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
    "remove_loops": True
}
'''


SOURCE = "\n\n".join((
    _PAGE,
    inspect.getsource(columnar_multifluid_rgb_step),
))


PRESENTATION_SHADER = '''#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D turing_output_texture;
uniform vec2 turing_resolution;
layout(location = 0) out vec4 turing_output_0;

void main() {
    vec2 uv = gl_FragCoord.xy / max(turing_resolution, vec2(1.0));
    uv.y = 1.0 - uv.y;
    turing_output_0 = texture(turing_output_texture, uv);
}
'''


def build_demo(destination: Path):
    """Compile Python through AST/ProcessGraph and publish its Wasm shell."""

    from ....compiler.site_bundle import build_program_bundle

    return build_program_bundle(
        SOURCE,
        destination,
        source_filename="columnar_multifluid_web_demo.py",
        include_backends=False,
        include_mathematics=False,
        presentation_shader=PRESENTATION_SHADER,
        shader_configuration={
            "output_texture": {"channels": ["red", "green", "blue"]},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[5],
    )
    arguments = parser.parse_args(argv)
    bundle = build_demo(arguments.destination.resolve())
    print(bundle.page_path)
    return 0


__all__ = ["PRESENTATION_SHADER", "SOURCE", "build_demo", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
