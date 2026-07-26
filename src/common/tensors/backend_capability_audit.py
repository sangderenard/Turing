"""Audit the shared canonical operation catalog across native lowering targets.

This is deliberately an audit, not the source of truth. Nodus owns the
versioned JSON table today; Turing owns CTensorOp; GLSL owns its expression
table. The report joins those live surfaces and makes every hole explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .accelerator_backends.glsl_backend import canonical_op, GLSLUnsupportedOp


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[4] / "nodus" / "ops" / "canonical_ops.json"


def audit_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for operation in payload["ops"]:
        name = operation["name"]
        try:
            canonical_op(name)
            glsl = True
        except GLSLUnsupportedOp:
            glsl = False
        c_native = bool(operation.get("ct_op") or operation.get("c_fn"))
        nodus = bool(operation.get("lowerable"))
        rows.append(
            {
                "name": name,
                "class": operation["class"],
                "c_native": c_native,
                "glsl": glsl,
                "nodus_kernel": nodus,
                "complete": c_native and glsl and nodus,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=default_catalog_path())
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return a failing status while any catalog operation has a gap",
    )
    args = parser.parse_args(argv)

    rows = audit_catalog(args.catalog)
    headers = ("name", "class", "c_native", "glsl", "nodus_kernel", "complete")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    complete = sum(row["complete"] for row in rows)
    print(
        f"catalog={len(rows)} complete={complete} "
        f"c_native={sum(row['c_native'] for row in rows)} "
        f"glsl={sum(row['glsl'] for row in rows)} "
        f"nodus_kernel={sum(row['nodus_kernel'] for row in rows)}"
    )
    for row in rows:
        if row["complete"]:
            continue
        gaps = [
            target
            for target, present in (
                ("c", row["c_native"]),
                ("glsl", row["glsl"]),
                ("nodus", row["nodus_kernel"]),
            )
            if not present
        ]
        print(f"{row['name']}: missing {','.join(gaps)}")
    return int(args.require_complete and complete != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
