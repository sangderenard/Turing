"""Orbital craft machine -> one native dt system (link mode), first wall.

Run (PYTHONPATH must include C:\\dev\\Powershell\\engine_toy, and the build
directory is the first argument; default: a ``build`` folder next to turing):

    python tools/compiler_probes/probe_orbital_craft_native_dt.py <build_dir>

What it does, in order, and stops at the first wall:

1. ``orbital_craft()`` -> ``craft_machine_dt_pieces(craft, 1, green_coast=True)``
   (no game, no MachineCraft, no dt state).  Before that call every piece the
   call will ask ``equation_piece`` for is located in the piece cache and
   checked with ``piece_staleness``; a missing or stale piece STOPS the probe
   (``equation_piece`` would rebuild it, and this probe never builds a piece).
2. ``lowered_system(paths, directory=..., piece_mode="link")`` -- the
   extraction contract is the one ``lowered_system`` builds itself.  Prints the
   full concordance report, any ConcordanceRefusal payload in full, and phase
   timings.  A stack dump is armed every 900 s (faulthandler) so a run with no
   progress lines can be told from a spin.
3. If a NativeSystem comes back: the columns that ``craft_machine_columns``
   does not cover are listed (the jumper's own state columns come from
   ``OrbitalJumper._initial_columns``), because the round comparison against
   the Python lane needs every column of the lowered state.
"""

from __future__ import annotations

import faulthandler
import hashlib
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE_TOY = ROOT.parent / "engine_toy"
for entry in (ENGINE_TOY, ROOT / "examples", ROOT):
    sys.path.insert(0, str(entry))

faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)

STARTED = time.time()


def stamp(message: str) -> None:
    print(f"[probe +{time.time() - STARTED:8.1f}s] {message}", flush=True)


def cached_piece_paths(craft, center_count: int, batch: int, green_coast: bool):
    """The cache path of every piece ``craft_machine_dt_pieces`` will request,
    named the way ``equation_piece`` names it (same key function, same root)."""
    import honorary_engine_equation_catalogue as honorary
    from orbital_craft_machine import craft_machine_equations

    paths = []
    for name, equations in craft_machine_equations(
            craft, center_count, green_coast=green_coast):
        key = honorary._llvm_piece_cache_key(name, batch, tuple(equations))
        paths.append((name, honorary._LLVM_LAW_CACHE_DIR / name / f"b{batch}"
                      / key / f"{name}.piece"))
    return paths


def main() -> int:
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "orbital_craft_native_dt"
    build.mkdir(parents=True, exist_ok=True)

    stamp("importing engine_toy craft machine")
    from orbital_craft_machine import (
        craft_machine_columns, craft_machine_dt_pieces, orbital_craft)
    from orbital_jumper import ORBITAL_CHANNEL_NAMES
    from src.compiler.native_law_kernels import LLVMPiece, piece_staleness

    craft = orbital_craft()
    stamp("craft built")

    # ---- (2) cache check, never a rebuild --------------------------------
    wanted = cached_piece_paths(craft, 1, 1, True)
    stamp(f"{len(wanted)} piece(s) requested; checking cache + staleness")
    missing, stale = [], []
    for name, path in wanted:
        if not path.is_file():
            missing.append((name, path))
            continue
        changed = piece_staleness(LLVMPiece.load(path))
        if changed:
            stale.append((name, path, changed))
    if missing or stale:
        for name, path in missing:
            print(f"MISSING  {name}: {path}", flush=True)
        for name, path, changed in stale:
            print(f"STALE    {name}: {path}\n         changed: {list(changed)}",
                  flush=True)
        print("STOP: not rebuilding.", flush=True)
        return 2
    stamp("all pieces cached and current")

    pieces, labels = craft_machine_dt_pieces(craft, 1, green_coast=True)
    paths = [path for _name, path in wanted]
    for label, path in zip(labels, paths):
        print(f"  {label}: {path}", flush=True)

    # ---- (3) the lowering -------------------------------------------------
    from llvm_dt_system import lowered_system
    from src.compiler.identity_concordance import concordance_report

    stamp("lowered_system(piece_mode='link'): start")
    try:
        system = lowered_system(
            paths, directory=build, piece_mode="link",
            channel_names=ORBITAL_CHANNEL_NAMES,
            progress=lambda message: stamp(f"lowered_system: {message}"))
    except BaseException as error:
        stamp(f"lowering FAILED: {type(error).__name__}")
        traceback.print_exc(file=sys.stdout)
        for attribute in ("payload", "report", "refusal", "args", "__dict__"):
            value = getattr(error, attribute, None)
            if value:
                print(f"--- {type(error).__name__}.{attribute} ---", flush=True)
                print(value, flush=True)
        return 1
    stamp("lowering done")
    print(concordance_report(system.module), flush=True)

    # ---- (4) the round comparison needs every column ---------------------
    covered = set(craft_machine_columns(craft))
    uncovered = [name for name in system.columns if name not in covered]
    stamp(f"NativeSystem built; {len(system.columns)} state columns, "
          f"{len(uncovered)} not in craft_machine_columns (jumper columns)")
    print(uncovered, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
