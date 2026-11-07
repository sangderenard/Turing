"""A gallery of Meowdoku boards at phone-playable sizes.

    python examples/meowdoku/gallery.py --sizes 5,6,7,8,9,10 --seeds 3

Cells are never smaller than ``--cell-px`` (default 44 px, a finger-sized
target: a 10x10 board is then 440 px wide, inside a phone's portrait width).
Each board is a fresh puzzle from ``puzzle.generate`` (the random generator
to 8x8, the relaxation generator above) with a few moves played through
``MeowdokuSession`` so cats, X marks and lost hearts all appear, composed
with the ray-traced tile sprites.  Writes one image per board and one
gallery sheet under ``build/meowdoku/gallery``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s5_export import tiles  # noqa: E402
from meowdoku.s2_titles.classic.session import MeowdokuSession  # noqa: E402


def find_sprites(tile_root: Path) -> dict[str, tiles.TileSprite]:
    """The Phong sprites when rendered, else the baked path-traced ones,
    else the single renders."""
    from meowdoku.s5_export.phong_tiles import load_phong_sprites

    sprites: dict[str, tiles.TileSprite] = load_phong_sprites(tile_root / "phong") if (tile_root / "phong").is_dir() else {}
    if len(sprites) == 3:
        return sprites
    bake = tile_root / "bake"
    if bake.is_dir():
        sprites = tiles.load_sprites(bake, tiles.tile_order_package())
    for role, folder in (("x_mark", "rendered_x"), ("heart", "rendered_heart"), ("cat", "rendered_cat")):
        if role not in sprites and (tile_root / folder).is_dir():
            sprites.update(tiles.load_sprites(tile_root / folder, tiles.tile_order_package(roles=(role,))))
    return sprites


def play_a_little(session: MeowdokuSession, rng: np.random.Generator) -> None:
    """A few moves: two or three right cats, an X, sometimes a wrong cat."""
    n = session.n
    solution = session.puzzle.solution.reshape(n, n)
    columns = [int(np.argmax(solution[r])) for r in range(n)]
    rows = rng.choice(n, size=min(n, 4), replace=False)
    for r in rows[:2 + (n > 6)]:
        session.place(int(r), columns[int(r)])
    r = int(rows[-1])
    session.toggle_x(r, (columns[r] + 2) % n)
    if rng.uniform() < 0.5:
        r = int(rows[-2])
        session.place(r, (columns[r] + 3) % n)        # wrong: costs a heart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="5,6,7,8,9,10")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cell-px", type=int, default=44)
    parser.add_argument("--out", default=str(ROOT / "build" / "meowdoku" / "gallery"))
    args = parser.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sprites = find_sprites(ROOT / "build" / "meowdoku" / "tiles")
    print("[gallery] sprites:", sorted(sprites), flush=True)
    from PIL import Image

    boards: dict[tuple[int, int], np.ndarray] = {}
    for n in sizes:
        for seed in range(args.seeds):
            session = MeowdokuSession(n=n, seed=100 + seed)
            play_a_little(session, np.random.default_rng(n * 97 + seed))
            image = tiles.compose_board(np.asarray(session.cat, dtype=np.float64).reshape(-1),
                                        np.asarray(session.x_mark, dtype=np.float64).reshape(-1),
                                        session.puzzle.region, hearts=int(session.hearts),
                                        sprites=sprites, cell_px=args.cell_px)
            boards[(n, seed)] = image
            Image.fromarray((np.clip(image, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)).save(
                out / f"board_{n}x{n}_seed{100 + seed}.png")
            print(f"[gallery] {n}x{n} seed {100 + seed}: {image.shape[1]}x{image.shape[0]} px, hearts {session.hearts}", flush=True)
    # One sheet: a column per size, a row per seed, every board on its own
    # dark card of the largest board's size.
    pad = 16
    cell_h = max(image.shape[0] for image in boards.values()) + pad
    cell_w = max(image.shape[1] for image in boards.values()) + pad
    sheet = np.full((args.seeds * cell_h + pad, len(sizes) * cell_w + pad, 3), 0.08)
    for (n, seed), image in boards.items():
        y = pad + seed * cell_h
        x = pad + sizes.index(n) * cell_w
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    sheet_path = out / "gallery.png"
    Image.fromarray((np.clip(sheet, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)).save(sheet_path)
    print(f"[gallery] sheet -> {sheet_path} ({sheet.shape[1]}x{sheet.shape[0]} px)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
