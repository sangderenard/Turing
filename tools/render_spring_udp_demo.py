"""Render recorded NFAB UDP positions from the native cloth demo as a GIF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(source: Path, target: Path) -> None:
    frames: dict[int, dict[int, tuple[float, float, float]]] = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, node = int(row["frame"]), int(row["node"])
            frames.setdefault(frame, {})[node] = tuple(float(row[k]) for k in ("x", "y", "z"))
    if len(frames) != 101 or any(len(nodes) != 81 for nodes in frames.values()):
        raise ValueError("expected 101 complete UDP frames with 81 nodes each")
    font = ImageFont.load_default()
    images = []
    for frame in range(0, 101, 2):
        image = Image.new("RGB", (720, 540), "#08111b")
        draw = ImageDraw.Draw(image)
        positions = frames[frame]

        def project(node: int) -> tuple[int, int]:
            x, y, z = positions[node]
            return (round(183 + 245 * x + 75 * z), round(94 - 120 * y + 45 * z))

        for node in range(81):
            x, y = node % 9, node // 9
            for adjacent in ((node + 1,) if x < 8 else ()) + ((node + 9,) if y < 8 else ()):
                draw.line((project(node), project(adjacent)), fill="#34c9da", width=2)
        for node in range(81):
            px, py = project(node)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#b4ffff")
        for node in (0, 8):
            px, py = project(node)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#ffc56c")
        draw.text((24, 20), "Nodus cloth / native SymPy -> AbstractTensor -> LLVM", font=font, fill="#e7faff")
        draw.text((24, 42), f"fabric 20261112  |  UDP frame {frame + 1:03d}  |  t = {frame * .01:.2f} s", font=font, fill="#8ab3c5")
        draw.text((24, 513), "9 x 9 spring sheet  /  top corners pinned  /  81 identity-vector pairs per frame", font=font, fill="#8ab3c5")
        images.append(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(target, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    render(args.source, args.target)
    print(args.target)


if __name__ == "__main__":
    main()
