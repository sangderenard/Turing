"""Bake one map into a folder of static pages: one page per place, one link per declared edge.

    python examples/meowdoku/deploy/bake_site.py --map manual --out DIR
    python examples/meowdoku/deploy/bake_site.py --map-file some.map.json --name world --out DIR

The pages are the ones GET /site/<map>/ serves on first look (speaktome_api.site, the same renderer); baking writes
all of them ahead of time, with relative links, so the folder can be opened from disk, copied into a public web
folder, or served from any path. `--map manual` is the SpeakToMe manual; `--map-file` is any map of the
abstract-ui-div-map-v0 form. Nothing is fetched and no server is contacted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import rest_api  # noqa: E402
from meowdoku.living_data_query import GraphView  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--map", choices=("manual",))
    source.add_argument("--map-file")
    parser.add_argument("--name", default=None, help="the map's short name (default: the --map value)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.map == "manual":
        if rest_api.manual is None:
            parser.error("the SpeakToMe checkout has no speaktome_man package")
        document = rest_api.manual.as_map()
    else:
        document = json.loads(Path(args.map_file).read_text(encoding="utf-8"))
    name = args.name or args.map
    if not name:
        parser.error("--map-file needs --name")
    with tempfile.TemporaryDirectory() as scratch:      # the cache is not wanted here; only the baked folder is
        atlas = rest_api.site.Site(Path(scratch), GraphView).open(name, document)
        count = atlas.bake(Path(args.out))
    print(json.dumps({"map": name, "digest": atlas.digest, "pages": count, "out": str(Path(args.out).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
