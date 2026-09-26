"""Compile a complete desktop GLSL fragment through project IR to WebGL 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage", choices=("vertex", "fragment"), default="fragment")
    args = parser.parse_args(argv)

    translated = compile_glsl_project_to_webgl(
        args.source.read_text(encoding="utf-8"),
        source_name=args.source.stem,
        stage_hint=args.stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(translated.source, encoding="utf-8", newline="\n")
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(translated.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not translated.complete:
        for diagnostic in translated.diagnostics:
            print(diagnostic.format(), file=sys.stderr)
        return 1
    print(args.output)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
