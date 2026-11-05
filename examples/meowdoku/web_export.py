"""Export everything the WebGL Meowdoku page needs, prebaked.

    python examples/meowdoku/web_export.py --boards 16

Writes ``build/meowdoku/web/``:

- ``shaders/base_material.webgl.{vert,frag}.glsl``: Pluck's base-material
  Phong shader as Turing's ``glsl_project_ir`` compiled it for WebGL 2
  (copied from spectral-analyzer's generated shaders);
- ``assets.json``: the three tile models (the glyph extrusions, flattened
  positions and face normals, material index per role) and the material
  storage tables (``pbr``, ``phong``, ``enamel``, ``texstack``) exactly as
  ``material_db.MaterialDatabase.build_tensors`` lays them out, which the
  compiled shader reads back through ``turing_ssbo_load_*``;
- ``boards.json``: the prebaked board database: per size and style a set
  of region maps, each with the leaf the game scores against (the sampled
  walk's for free boards, the constructed cat map for relaxed ones), its
  ring-relaxed hues, and its taxonomy record and level key.

The page pulls these; nothing is generated at play time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import tiles, tree  # noqa: E402
from meowdoku.phong_tiles import PHONG_MATERIALS, LIGHTS, _spectral, glyph_mesh  # noqa: E402
from meowdoku.puzzle import generate_batch  # noqa: E402
from meowdoku.tiles import SPECTRAL_ROOT  # noqa: E402

WEB_DIR = ROOT / "build" / "meowdoku" / "web"
SHADERS = ("base_material.webgl.vert.glsl", "base_material.webgl.frag.glsl")


def export_shadow_fragment(target: Path) -> None:
    """The same fragment shader with its own shadow-map path switched on
    (``#define PLUCK_SHADOW_MAP``, the opt-in it documents for WebGL
    adapters), compiled through Turing's GLSL project IR like the tracked
    generated shader.  The page renders a depth pre-pass from the key light
    into ``uShadowMaps`` and hands ``uShadowMatrices`` per draw."""
    from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl

    source = (SPECTRAL_ROOT / "csrc" / "shaders" / "base_material.frag.glsl").read_text(encoding="utf-8")
    source = source.replace("#version 430 core", "#version 430 core\n#define PLUCK_SHADOW_MAP 1", 1)
    translated = compile_glsl_project_to_webgl(source, source_name="base_material_fragment_shadow", stage_hint="fragment")
    if not translated.complete:
        report = "\n".join(item.format() for item in translated.diagnostics)
        raise RuntimeError(f"shadow fragment translation is incomplete:\n{report}")
    target.write_text(translated.source, encoding="utf-8", newline="\n")


def export_assets(out: Path) -> dict:
    _, _, MaterialDatabase = _spectral()
    db = MaterialDatabase()
    material_index = {role: db.register(f"meowdoku_{role}", PHONG_MATERIALS[role]) for role in PHONG_MATERIALS}
    tensors = db.build_tensors()
    tables = {
        "pbr": tensors["pbr"], "phong": tensors["phong_compat"],
        "enamel": tensors["enamel"], "texstack": tensors["texture_stack"],
    }
    meshes = {}
    for role in PHONG_MATERIALS:
        verts, normals = glyph_mesh(role)
        meshes[role] = {
            "positions": np.round(verts.reshape(-1), 6).tolist(),
            "normals": np.round(np.repeat(normals, 3, axis=0).reshape(-1), 6).tolist(),
            "material": int(material_index[role]),
            "triangles": int(verts.shape[0]),
            "extent": float(np.abs(verts).max()),
        }
    assets = {
        "meshes": meshes,
        "materials": {name: {"floats_per_record": int(table.shape[1]), "records": int(table.shape[0]),
                             "data": np.asarray(table, np.float32).reshape(-1).tolist()}
                      for name, table in tables.items()},
        "lights": [{"position": list(p), "color": list(c), "intensity": float(i)} for p, c, i in LIGHTS],
    }
    (out / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    return assets


def export_boards(out: Path, sizes: tuple[int, ...], per_style: int, seed: int) -> dict:
    boards = []
    for n in sizes:
        for style in ("free", "relaxed"):
            started = time.perf_counter()
            puzzles = generate_batch(n, per_style, seed=seed + n, style=style)
            records = tree.catalog(puzzles, seed=seed, limit_leaves=256)
            leaves, decisions, _, solved = tree.sample_solutions(puzzles.region, seed=seed + 7)
            for index, record in enumerate(records):
                solution = leaves[index] if (style == "free" and solved[index]) else puzzles.solution[index]
                grid = puzzles.region[index].reshape(n, n).astype(int)
                hues = tiles.ring_hues(grid, seed=index)
                taxonomy = tree.classify(record)
                boards.append({
                    "id": f"{style}-{n}-{index}", "n": n, "style": style,
                    "region": grid.reshape(-1).tolist(),
                    "solution": np.asarray(solution, dtype=int).reshape(-1).tolist(),
                    "hues": np.round(hues, 4).tolist(),
                    "level": taxonomy["level"], "type": record["type"],
                    "decisions": int(record["decisions"]), "leaves": record.get("leaves"),
                    "walk_decisions": int(record["walk_decisions"]),
                    "largest_share": taxonomy["largest_share"], "singletons": taxonomy["singletons"],
                })
            print(f"[export] {style:7s} {n}x{n}: {per_style} boards in {time.perf_counter() - started:.1f}s", flush=True)
    levels: dict[str, int] = {}
    for board in boards:
        levels[board["level"]] = levels.get(board["level"], 0) + 1
    database = {"boards": boards, "levels": dict(sorted(levels.items()))}
    (out / "boards.json").write_text(json.dumps(database), encoding="utf-8")
    return database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="5,6,7,8,9,10")
    parser.add_argument("--boards", type=int, default=12, help="boards per size per style")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=str(WEB_DIR))
    args = parser.parse_args(argv)
    out = Path(args.out)
    (out / "shaders").mkdir(parents=True, exist_ok=True)
    for name in SHADERS:
        shutil.copyfile(SPECTRAL_ROOT / "csrc" / "shaders" / "generated" / name, out / "shaders" / name)
    export_shadow_fragment(out / "shaders" / "base_material.webgl.shadow.frag.glsl")
    for name in ("index.html", "app.js"):
        shutil.copyfile(HERE / "web" / name, out / name)
    assets = export_assets(out)
    print("[export] assets:", {role: mesh["triangles"] for role, mesh in assets["meshes"].items()},
          {name: table["records"] for name, table in assets["materials"].items()}, flush=True)
    database = export_boards(out, tuple(int(s) for s in args.sizes.split(",")), args.boards, args.seed)
    print(f"[export] {len(database['boards'])} boards; levels: {database['levels']}", flush=True)
    print(f"[export] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
