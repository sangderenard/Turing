"""Publish the Python columnar state machine as a Wasm RGB-preview page."""

from __future__ import annotations

import argparse
import inspect
import json
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
        "dt": 0.025,
        "ink_red": 0.0,
        "ink_yellow": 0.0,
        "ink_green": 0.0,
        "ink_cyan": 0.0,
        "ink_blue": 0.0,
        "ink_magenta": 0.0
    },
    "feed_expressions": {
        "column_x": "(x + 0.5) * 10.0 / w",
        "column_y": "(y + 0.5) * 7.0 / h",
        "rest_surface": "1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))",
        "displacement": "0.0",
        "displacement_velocity": "0.0",
        "managed_time": "0.0",
        "dt": "0.025",
        "ink_red": "0.0",
        "ink_yellow": "0.0",
        "ink_green": "0.0",
        "ink_cyan": "0.0",
        "ink_blue": "0.0",
        "ink_magenta": "0.0"
    },
    "state_feedback": {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "managed_time": "next_time",
        "ink_red": "next_ink_red",
        "ink_yellow": "next_ink_yellow",
        "ink_green": "next_ink_green",
        "ink_cyan": "next_ink_cyan",
        "ink_blue": "next_ink_blue",
        "ink_magenta": "next_ink_magenta"
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


def build_pages(destination: Path):
    """Build the demo plus a stable top-level GitHub Pages entrypoint."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundle = build_demo(destination)
    relative_page = bundle.page_path.relative_to(destination).as_posix()
    landing = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url={relative_page}">
<title>Managed Columnar Multifluid World</title>
</head>
<body>
<p>Opening the <a href="{relative_page}">managed columnar multifluid world</a>.</p>
<script>location.replace(new URL({json.dumps(relative_page)}, document.baseURI));</script>
</body>
</html>
'''
    (destination / "index.html").write_text(
        landing, encoding="utf-8", newline="\n"
    )
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "deployment.json").write_text(
        json.dumps(
            {
                "schema": "turing-pages-deployment-v1",
                "program": "managed-columnar-multifluid-world",
                "entrypoint": relative_page,
                "version": bundle.manifest["version"]["id"],
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[5],
    )
    parser.add_argument(
        "--pages",
        action="store_true",
        help="also write a stable top-level GitHub Pages entrypoint",
    )
    arguments = parser.parse_args(argv)
    builder = build_pages if arguments.pages else build_demo
    bundle = builder(arguments.destination.resolve())
    print(bundle.page_path)
    return 0


__all__ = [
    "PRESENTATION_SHADER", "SOURCE", "build_demo", "build_pages", "main"
]


if __name__ == "__main__":
    raise SystemExit(main())
