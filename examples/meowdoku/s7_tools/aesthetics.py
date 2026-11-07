"""The aesthetics sheet: one board per size, simple to preposterous, for a
generator style, so every change to the generator can be looked at.

    python examples/meowdoku/aesthetics.py --style relaxed
    python examples/meowdoku/aesthetics.py --style free --sizes 6,9,16,32,64

Each panel is the fresh board (regions only, the hidden cats not shown)
rendered by the board compositor with flat region colours at a cell size
that keeps every panel about ``--panel-px`` wide.  Sheets land in
``build/meowdoku/aesthetics/<style>_<git-short-sha>.png`` so a sheet names
the generator revision that drew it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s5_export import tiles  # noqa: E402
from meowdoku.s2_titles.classic.puzzle import generate_batch  # noqa: E402


def git_short_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--style", default="relaxed")
    parser.add_argument("--sizes", default="6,9,16,32,64,128")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--panel-px", type=int, default=640)
    parser.add_argument("--palette", default=None, help="named palette (pastel, vivid, warm, cool, ink) with hue-contrast assignment; default: the shader's formula")
    parser.add_argument("--backend", default=None, help="AbstractTensor backend for generation (default: the module default, torch on cuda)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(ROOT / "build" / "meowdoku" / "aesthetics"))
    args = parser.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    panels = []
    for n in sizes:
        started = time.perf_counter()
        options = {}
        if args.backend:
            options = {"backend": args.backend, "device": args.device}
        puzzles = generate_batch(n, 1, seed=args.seed, style=args.style, **options)
        elapsed = time.perf_counter() - started
        puzzle = puzzles.puzzle(0)
        cell_px = max(3, args.panel_px // n)
        image = tiles.compose_board(np.zeros(n * n), np.zeros(n * n), puzzle.region, hearts=0,
                                    sprites={}, cell_px=cell_px, grid_px=1 if n >= 32 else 2, max_hearts=0,
                                    palette=args.palette)
        sizes_of = np.bincount(puzzle.region.astype(int), minlength=n)
        label = f"{n}x{n}  {elapsed:.1f}s  largest {sizes_of.max()}/{n*n}  singletons {(sizes_of == 1).sum()}"
        print(f"[aesthetics] {args.style} {label}", flush=True)
        panels.append((image, label))
    pad, strip = 24, 28
    width = max(p.shape[1] for p, _ in panels)
    height = max(p.shape[0] for p, _ in panels)
    sheet = Image.new("RGB", (len(panels) * (width + pad) + pad, height + strip + 2 * pad), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(panels):
        x = pad + index * (width + pad)
        srgb = (np.clip(image, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(srgb), (x, pad + strip))
        draw.text((x, pad + 4), label, fill=(230, 230, 230))
    sha = git_short_sha()
    draw.text((pad, height + strip + pad + 2), f"style={args.style} palette={args.palette or 'shader'} seed={args.seed} generator@{sha}", fill=(160, 160, 160))
    path = out / f"{args.style}_{args.palette or 'shader'}_{sha}.png"
    sheet.save(path)
    print(f"[aesthetics] sheet -> {path} ({sheet.size[0]}x{sheet.size[1]} px)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
