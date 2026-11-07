"""Move examples/meowdoku modules into the pipeline layout, in stages, leaving a shim at every old path.

    python tools/meowdoku_layout_move.py --stage 1            # move + shim + import rewrite (idempotent)
    python tools/meowdoku_layout_move.py --stage 1,2,3 --dry  # print the plan only

The layout and the audit are in ``examples/meowdoku/LAYOUT_AUDIT.md``.  What this does, and ONLY this:

* ``git mv`` each module of the chosen stages to ``examples/meowdoku/<package>/<name>.py`` (history kept);
* re-point the path expressions that count directories above the file (``parents[N]``) by the extra depth;
* leave a SHIM at the old path: ``import meowdoku.<old>`` gives the moved module itself (same object), and
  ``python examples/meowdoku/<old>.py`` / ``python -m meowdoku.<old>`` still runs it as ``__main__``;
* rewrite ``from meowdoku import x`` / ``from meowdoku.x import ...`` to the new dotted names in the files whose text is
  NOT part of a product source digest and is not in flight (see HASHED and HOT) -- those keep the old names, which the
  shims serve, so their text (and so every digest) is untouched.

Stages:  1 = digest-neutral and independent of in-flight files;  2 = the three files whose text is hashed into every law
digest AND counts directories (entities, entity_motion, scroller): moving them re-keys the law products (a rebuild);
3 = the in-flight files (server, startup, loading_anim, mobroad, fountain*, web_export), after the other workers land.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "examples" / "meowdoku"
DOT = "meowdoku"
CRLF = "\r\n"          # the working tree is CRLF (core.autocrlf); text mode turns the \n written into this

# old module -> (package path under examples/meowdoku, stage)
PLAN: dict[str, tuple[str, int]] = {}


def _put(package: str, stage: int, *names: str, stage_of: dict | None = None) -> None:
    for name in names:
        PLAN[name] = (package, (stage_of or {}).get(name, stage))


# 1. generic law substrate (stage 2: entities / entity_motion count directories and are hashed)
_put("s1_engine", 1, "sphere_motion", "sweep", "walkers", "world_view", "roads")
_put("s1_engine", 2, "entities", "entity_motion")
# 2. bespoke game material, one package per title
_put("s2_titles/classic", 1, "games", "puzzle", "tree", "words", "program", "shader", "page", "session", "play")
_put("s2_titles/chess", 1, "chess_net", "mcts", "search", "native_forward", "lichess", "pretrain")
_put("s2_titles/common", 1, "characters")
_put("s2_titles/scroller", 2, "scroller")
_put("s2_titles/platformer", 1, "platformer")
_put("s2_titles/sidescroller", 1, "sidescroller")
_put("s2_titles/invaders", 1, "invaders")
_put("s2_titles/delve", 1, "delve", "delve_rules", "delve_world", "delve_plan")
_put("s2_titles/tilequest", 1, "tilequest")
_put("s2_titles/worldwalk", 1, "worldwalk")
_put("s2_titles/boardwalk", 1, "boardwalk")
_put("s2_titles/purrline", 1, "purrline")
_put("s2_titles/mobroad", 3, "mobroad")
_put("s2_titles/fountain", 1, "glyph_start", "fountain_game")
_put("s2_titles/fountain", 3, "fountain", "fountain3d")
_put("s2_titles/tipjar", 1, "tipjar")
# 3. training
_put("s3_train", 1, "autopilot", "idle_trainer", "pilot_decl", "pilot_law", "pilot_teachers", "pilot_train", "pilot_variants")
# 4. generation / compile
_put("s4_generate", 1, "graph_wasm", "policy_wasm", "split_lowering")
_put("s4_generate", 3, "fountain_build", "fountain3d_build")
# 5. export / assets
_put("s5_export", 1, "phong_tiles", "bake_tiles", "tiles")
_put("s5_export", 3, "web_export")
# 6. serving
_put("s6_serve", 1, "realtime_socket", "lifecycle", "share", "access", "profiles")
_put("s6_serve", 3, "server", "startup", "loading_anim")
# 7. probes, parity checks, one-offs
_put("s7_tools", 1, "boardwalk_probe", "delve_probe", "delve_parity", "invaders_probe", "invaders_parity", "purrline_probe",
     "tilequest_probe", "worldwalk_probe", "worldwalk_parity", "fountain_parity", "split_probe_source",
     "profile_matmul_lanes", "gallery", "aesthetics")

SIDE_FILES = {"tipjar": ["tipjar_fixture.json"]}       # data a module reads beside itself: moves with it

#: modules whose TEXT enters a product source digest (law modules, LawPiece.source_modules, the builders): no import
#: rewriting in them, so the digests only change where a path expression had to.
HASHED = {"entities", "entity_motion", "walkers", "sweep", "sphere_motion", "roads", "delve_rules", "pilot_law", "scroller",
          "fountain", "fountain3d", "fountain_build", "fountain3d_build", "platformer", "sidescroller", "invaders", "delve",
          "tilequest", "worldwalk", "boardwalk", "purrline", "mobroad"}
#: files other workers have in flight: no import rewriting either (the shims serve their old names)
HOT = {"server", "startup", "loading_anim", "web_export", "fountain_game", "mobroad", "fountain", "fountain3d"}

DESCRIPTIONS = {
    "s1_engine": "STAGE 1 - the generic law substrate: entity games, motion, sphere/world helpers (no game rules or content).",
    "s2_titles": "STAGE 2 - bespoke material: one package per title (rules, content, the title's own law). Generic code does not belong here.",
    "s3_train": "STAGE 3 - training: autopilot networks, teachers, the idle trainer. Reads laws, writes weights.",
    "s4_generate": "STAGE 4 - generation / compile: the builders that turn a law (and weights) into page products and manifests.",
    "s5_export": "STAGE 5 - export / assets: the prebaked page directory (build/meowdoku/web) and its sprites.",
    "s6_serve": "STAGE 6 - serving: the web server, sessions, sockets, lifecycle, share, access, start-up.",
    "s7_tools": "Probes, parity checks, galleries and one-offs. Nothing in the pipeline depends on these (see LAYOUT_AUDIT.md for the exceptions).",
}


def depth(package: str) -> int:
    return len(package.split("/"))


def new_dotted(name: str) -> str:
    return f"{DOT}." + PLAN[name][0].replace("/", ".") + f".{name}"


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=True)


SHIM = '''"""Compatibility shim: ``meowdoku.{name}`` moved to ``{dotted}`` (examples/meowdoku/{package}/{name}.py).

Importing this module returns the moved module itself (the same object: monkeypatching and ``inspect.getsource`` see the
real file); running it as a script or ``python -m`` runs the moved module as ``__main__``.  Generated by
tools/meowdoku_layout_move.py."""
import importlib
import sys

REAL = "{dotted}"
if __name__ == "__main__":
    import runpy
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_root), str(_root / "examples")]
    runpy.run_module(REAL, run_name="__main__", alter_sys=True)
else:
    def __getattr__(attribute):          # a reference taken while an import cycle was in flight still reaches the module
        return getattr(sys.modules[REAL], attribute)

    sys.modules[__name__] = importlib.import_module(REAL)
'''


def fix_paths(text: str, name: str, d: int) -> str:
    text = re.sub(r"(Path\(__file__\)\.resolve\(\)\.parents\[)(\d+)\]", lambda m: f"{m.group(1)}{int(m.group(2)) + d}]", text)
    text = re.sub(r"(\bHERE\.parents\[)(\d+)\]", lambda m: f"{m.group(1)}{int(m.group(2)) + d}]", text)
    text = text.replace("str(HERE.parent)", f"str(HERE.parents[{d}])")
    if name == "web_export":
        text = text.replace('HERE / "web" / name', 'HERE.parent / "web" / name')
    return text


def rewrite_imports(path: Path, stem: str, located: dict[str, str]) -> list[str]:
    """Point ``from meowdoku import x`` at the new dotted name for every x that has moved. Returns the lines left alone."""
    if stem in HASHED or stem in HOT:
        return []
    text = path.read_text(encoding="utf-8")
    left: list[str] = []

    def from_pkg(m: re.Match) -> str:
        indent, names, tail = m.group(1), m.group(2), m.group(3) or ""
        out = []
        for part in [p.strip() for p in names.split(",") if p.strip()]:
            base = part.split()[0]
            if base in located:
                package = located[base].rsplit(".", 1)[0]
                out.append(f"{indent}from {package} import {part}")
            else:
                out.append(f"{indent}from {DOT} import {part}")
        return "\n".join(out) + tail

    text = re.sub(r"^([ \t]*)from meowdoku import ([A-Za-z_][\w, ]*?)([ \t]*#.*)?$", from_pkg, text, flags=re.M)

    def from_mod(m: re.Match) -> str:
        return f"{m.group(1)}from {located[m.group(2)]} import" if m.group(2) in located else m.group(0)

    text = re.sub(r"^([ \t]*)from meowdoku\.(\w+) import", from_mod, text, flags=re.M)
    for line in text.splitlines():
        if re.match(r"\s*from meowdoku import \(", line) or re.match(r"\s*import meowdoku\.", line):
            left.append(line)
    path.write_text(text, encoding="utf-8", newline=CRLF)
    return left


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", default="1", help="comma list of stages: 1 independent, 2 digest window, 3 in-flight files")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--no-rewrite", action="store_true", help="move and shim only")
    args = parser.parse_args()
    stages = {int(s) for s in args.stage.split(",")}
    todo = [n for n, (_, st) in PLAN.items() if st in stages and (MD / f"{n}.py").exists() and "REAL = " not in (MD / f"{n}.py").read_text(encoding="utf-8")[:1500]]
    print(f"stages {sorted(stages)}: {len(todo)} modules to move")
    if args.dry:
        for n in todo:
            print(f"  {n}.py -> {PLAN[n][0]}/{n}.py (stage {PLAN[n][1]})")
        return 0
    for name in todo:
        package = PLAN[name][0]
        target = MD / package / f"{name}.py"
        for part in [package] + ([package.split("/")[0]] if "/" in package else []):
            directory = MD / part
            directory.mkdir(parents=True, exist_ok=True)
            top = part.split("/")[0]
            init = directory / "__init__.py"
            if not init.exists():
                init.write_text(f'"""{DESCRIPTIONS.get(top, "")}"""\n' if part == top else f'"""Title package: {part.split("/")[-1]}."""\n', encoding="utf-8")
        git("mv", str(MD / f"{name}.py"), str(target))
        for side in SIDE_FILES.get(name, []):
            git("mv", str(MD / side), str(target.parent / side))
        text = fix_paths(target.read_text(encoding="utf-8"), name, depth(package))
        target.write_text(text, encoding="utf-8", newline=CRLF)
        (MD / f"{name}.py").write_text(SHIM.format(name=name, dotted=new_dotted(name), package=package), encoding="utf-8", newline=CRLF)
    located = {n: new_dotted(n) for n in PLAN if (MD / PLAN[n][0] / f"{n}.py").exists()}
    if not args.no_rewrite:
        for path in sorted(MD.rglob("*.py")):
            if "REAL = " in path.read_text(encoding="utf-8")[:1500] or path.name == "__init__.py":
                continue
            leftover = rewrite_imports(path, path.stem, located)
            for line in leftover:
                print(f"  NOT REWRITTEN {path.relative_to(ROOT)}: {line.strip()}")
    git("add", "-A", str(MD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
