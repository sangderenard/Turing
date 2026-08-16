"""Minimal repro of the ``any(get(...) > limit for k, limit in d.items())`` seam.

Prints the lowered SSA body so the projected columns, the comparison and the
reduction can be read directly.  Seconds, not the 24 s full control build.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

CASES = {
    "iterate": (
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
    ),
    "compare": (
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        if limit > total:\n"
        "            total = limit\n"
        "    return total\n"
    ),
    "get": (
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        total = total + metrics.error_channels.get(name, 0.0)\n"
        "    return total\n"
    ),
    "anyget": (
        "def root(metrics):\n"
        "    return any(\n"
        "        float(metrics.error_channels.get(name, 0.0)) > float(limit)\n"
        "        for name, limit in metrics.error_channels.items()\n"
        "    )\n"
    ),
    "castloop": (
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        total = total + float(limit)\n"
        "    return total\n"
    ),
    "anyrange": (
        "def root(count):\n"
        "    return any(float(index) > 1.0 for index in range(count))\n"
    ),
    "subscript": (
        "def root(metrics):\n"
        "    total = 0.0\n"
        "    for name, limit in metrics.error_channels.items():\n"
        "        total = total + metrics.error_channels[name]\n"
        "    return total\n"
    ),
    "anyplain": (
        "def root(metrics):\n"
        "    return any(\n"
        "        float(limit) > 1.0\n"
        "        for name, limit in metrics.error_channels.items()\n"
        "    )\n"
    ),
}


def brief(value) -> str:
    accounting = dict(getattr(value, "accounting", None) or {})
    owner = accounting.get("program_abi_keyed_owner")
    part = accounting.get("program_abi_keyed_part")
    tag = f" [{owner}.{part}]" if owner else ""
    if accounting.get("projected_row_source_id") is not None:
        tag += (
            f" [row {accounting['projected_row_source_id']}"
            f".{accounting['projected_row_column']}]"
        )
    return f"%{value.id}:{value.dtype}{tag}"


def main() -> int:
    wanted = sys.argv[1:] or list(CASES)
    for case in wanted:
        source = CASES[case]
        print(f"===== {case} =====")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "root", name=case, extraction_contract=CONTRACT,
        )
        for name, function in module.functions.items():
            print(f"  {name}({', '.join(brief(a) for a in function.args)})")
            for block_name, block in function.blocks.items():
                print(f"    {block_name}:")
                for instruction in block.instrs:
                    operands = " ".join(
                        f"%{argument.id}" for argument in instruction.args
                    )
                    result = (
                        f"%{instruction.res.id} = "
                        if instruction.res is not None else ""
                    )
                    attributes = {
                        key: str(value)[:40]
                        for key, value in (instruction.attributes or {}).items()
                        if key in {
                            "binding", "projection", "induction", "value",
                            "region_index", "aggregate_index", "callee",
                            "feed_ids", "output_ids", "collection_value_id",
                            "source_value_id", "target_dtype",
                            "source_operator", "comparison", "source_type",
                        }
                    }
                    print(
                        f"      {result}{instruction.op} {operands}"
                        + (f"  {attributes}" if attributes else "")
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
