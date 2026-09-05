"""Export a repository-SSA entry point as an executable ES module.

This is intentionally program-agnostic: the pickle supplies the IRModule and
the selected entry supplies the call closure.  No authored function is
reimplemented by the exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from src.compiler.ssa_javascript_backend import (
    emit_ssa_module_to_javascript,
)


def export_repository_ssa_javascript(
    repository_ssa: Path,
    entry: str,
    output: Path,
) -> dict:
    with repository_ssa.open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    artifact = emit_ssa_module_to_javascript(module, entry)
    if not artifact.complete:
        raise RuntimeError("\n".join(
            shortfall.format() for shortfall in artifact.shortfalls
        ))
    function = module.functions[entry]
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.mjs").write_text(artifact.source, encoding="utf-8")
    arguments = []
    for position, value in enumerate(function.args):
        accounting = dict(value.accounting or {})
        arguments.append({
            "position": position,
            "valueId": int(value.id),
            "dtype": str(value.dtype or "unknown"),
            "shape": list(value.shape or ()),
            "pointer": int(value.id) in artifact.pointer_formals,
            "programParameter": accounting.get("program_abi_parameter"),
            "programField": accounting.get("program_abi_field"),
            "programStorage": accounting.get("program_abi_storage"),
            "programRank": accounting.get("program_abi_rank"),
            "mutable": accounting.get("program_abi_mutable"),
            "accounting": accounting,
        })
    manifest = {
        "schema": "turing.repository-ssa-javascript.v1",
        "entry": entry,
        "module": "program.mjs",
        "returnConvention": "array-in-ssa-ret-order",
        "arguments": arguments,
        "functionCount": len(module.functions),
        "sourceArtifact": str(repository_ssa.resolve()),
    }
    (output / "program.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository_ssa", type=Path)
    parser.add_argument("entry")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = export_repository_ssa_javascript(
        args.repository_ssa, args.entry, args.output,
    )
    print(json.dumps({
        "entry": manifest["entry"],
        "arguments": len(manifest["arguments"]),
        "functionCount": manifest["functionCount"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
