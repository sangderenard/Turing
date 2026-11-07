"""Bake the Meowdoku tile sprites: many refinement passes through
spectral-analyzer's native thick-lens BDPT, resumable.

    python examples/meowdoku/bake_tiles.py --passes 24

Each pass renders every role (X mark, heart, cat) once into
``build/meowdoku/tiles/bake/pass_NN/<job>``; passes already holding a linear
frame are skipped, so the bake resumes where it stopped.  ``load_sprites``
averages every pass found under the bake root.  After each pass a 9x9
preview board is recomposed to ``build/meowdoku/tiles/bake/preview_9x9.png``
so the convergence is visible while the bake runs.

Run it from a real console: a renderer launched from a detached background
shell never starts its GPU work (measured 2026-11-05).  One pass is about a
minute for the X and four to five minutes each for the heart and the cat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s5_export import tiles  # noqa: E402

ROLE_JOBS = {"x_mark": "meowdoku_x_mark", "heart": "meowdoku_heart", "cat": "meowdoku_cat"}


def pass_done(pass_dir: Path, job: str) -> bool:
    job_dir = pass_dir / job
    return any(p.name.endswith("_linear.npy") and not p.name.endswith("_sum_linear.npy")
               for p in job_dir.glob("*.npy")) if job_dir.is_dir() else False


def preview(bake_root: Path, cell_px: int = 48) -> Path:
    """A 9x9 board with the current averaged sprites."""
    from meowdoku.s2_titles.classic.session import MeowdokuSession

    sprites = tiles.load_sprites(bake_root, tiles.tile_order_package())
    n = 9
    session = MeowdokuSession(n=n, seed=4)
    solution = session.puzzle.solution.reshape(n, n)
    columns = [int(np.argmax(solution[r])) for r in range(n)]
    session.place(0, columns[0])
    session.place(4, columns[4])
    session.toggle_x(2, (columns[2] + 3) % n)
    session.place(6, (columns[6] + 2) % n)            # wrong: costs a heart
    image = tiles.compose_board(np.asarray(session.cat, dtype=np.float64).reshape(-1),
                                np.asarray(session.x_mark, dtype=np.float64).reshape(-1),
                                session.puzzle.region, hearts=int(session.hearts),
                                sprites=sprites, cell_px=cell_px)
    from PIL import Image
    path = bake_root / "preview_9x9.png"
    Image.fromarray((np.clip(image, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)).save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--roles", default="x_mark,heart,cat")
    parser.add_argument("--out", default=str(ROOT / "build" / "meowdoku" / "tiles" / "bake"))
    args = parser.parse_args(argv)
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    bake_root = Path(args.out)
    bake_root.mkdir(parents=True, exist_ok=True)
    package = tiles.write_tile_package(bake_root / "meowdoku_tiles.json", roles=roles)
    if not tiles.native_kernels_available():
        print("native kernels are not built; nothing to bake", flush=True)
        return 1
    started = time.perf_counter()
    for index in range(int(args.passes)):
        pass_dir = bake_root / f"pass_{index:02d}"
        for role in roles:
            job = ROLE_JOBS[role]
            if pass_done(pass_dir, job):
                continue
            t = time.perf_counter()
            tiles.render_tiles(package, pass_dir, jobs=(job,))
            print(f"[bake] pass {index:02d} {role:6s} {time.perf_counter() - t:5.0f}s "
                  f"(elapsed {(time.perf_counter() - started) / 60:.1f} min)", flush=True)
        try:
            print(f"[bake] preview -> {preview(bake_root)}", flush=True)
        except Exception as error:  # the preview is a courtesy; the bake goes on
            print(f"[bake] preview skipped: {type(error).__name__}: {error}", flush=True)
    print(f"[bake] done: {args.passes} passes in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
