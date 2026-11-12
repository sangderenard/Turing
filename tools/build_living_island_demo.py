"""Build one real island: two Living Data Map pages joined by a door.

    python tools/build_living_island_demo.py --out build/living-island-demo

Writes ``living_island_outside.html`` (a world with one lot) and ``living_island_inside.html`` (that lot's island:
its own world, its own document). The lot's entry doorway carries a door sheet whose partner is in the island;
the island's one passage carries the way back. Both pages are ordinary projections of the existing engine
(``project_world_to_div_map``); this tool only builds two small worlds and declares the door's two halves
(``abstract_ui_doorways``). Each projection takes minutes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.abstract_ui_div_map import project_world_to_div_map  # noqa: E402
from src.compiler.abstract_ui_doorways import (  # noqa: E402
    declare_island_door, declare_island_passage, door_identity,
)
from src.compiler.abstract_ui_introspection import build_introspective_world  # noqa: E402
from src.compiler.abstract_ui_islands import island_identity  # noqa: E402

OUTSIDE_PAGE = "living_island_outside.html"
INSIDE_PAGE = "living_island_inside.html"


class Lot:
    """The outside: one lot, as found."""

    plaque: str
    mailbox: int

    def knock(self, times: int) -> bool:
        return times > 0


class Hearth:
    """The lot's interior: its own place."""

    embers: float
    kettle: bool

    def rest(self, hours: float) -> float:
        return hours


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    outside = build_introspective_world(Lot, seed="living-island-demo")
    lot = outside.regions[0].buildings[0].identity
    island = replace(build_introspective_world(Hearth, seed="living-island-demo"),
                     identity=island_identity("demo/living-island", "lot"))
    passage = island.regions[0].identity

    outside = declare_island_door(outside, lot, island.identity, href=INSIDE_PAGE)
    island = declare_island_passage(island, passage, lot=lot, outside=outside.identity, href=OUTSIDE_PAGE)

    receipt = {"schema": "living-island-demo.v1", "door": door_identity(lot, island.identity),
               "outside_world": outside.identity, "island_world": island.identity, "pages": {}}
    for name, world, title in ((OUTSIDE_PAGE, outside, "Outside: a lot, as found"),
                               (INSIDE_PAGE, island, "Inside: the lot's island")):
        started = time.monotonic()
        print(f"[{name}] projecting...", flush=True)
        page = project_world_to_div_map(world, title=title)
        rendered = page.html
        if not isinstance(rendered, str) or "<html" not in rendered.lower():
            raise RuntimeError(f"{name}: the projection produced no HTML document")
        (out / name).write_text(rendered, encoding="utf-8", newline="\n")
        doors = [opening for box in page.model["document_geometry"]["boxes"]
                 for opening in box.get("openings", ()) if opening.get("transition") == "door"]
        receipt["pages"][name] = {
            "bytes": len(rendered.encode("utf-8")), "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "seconds": round(time.monotonic() - started, 1),
            "door_ends": [{key: opening.get(key) for key in ("identity", "port_role", "pair", "island", "href",
                                                             "facing", "side", "width", "height")}
                          for opening in doors],
        }
        print(f"[{name}] {receipt['pages'][name]['bytes']} bytes, {receipt['pages'][name]['seconds']} s, "
              f"{len(doors)} door ends", flush=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
