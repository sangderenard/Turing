"""Minimal repro: two call sites build ``error_channels`` with different
literal shapes (empty vs populated) and both feed one shared consumer.

This is the construction side the existing ``repro_keyed_get.py`` does not
cover -- that repro receives ``error_channels`` as an already-shaped ABI
input; here the dict is built as a literal inside the compiled source, the
way ``dt_controller.py``'s fallback ``Metrics(0.0, 0.0, 0.0, 0.0,
hard_failure=True)`` (an implicit empty dict) and ``dt_scaler.py``'s
``coerce_metrics`` (an explicit, populated dict) actually do.
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
    # Baseline: one construction site, populated, no ambiguity possible.
    "one_site_populated": (
        "def consume(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
        "def root(displacement, energy):\n"
        "    channels = {'displacement': displacement, 'energy': energy}\n"
        "    return consume(channels)\n"
    ),
    # Two call sites into the SAME consumer: one empty dict, one populated.
    "two_sites_empty_vs_populated": (
        "def consume(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
        "def root(displacement, energy, failed):\n"
        "    normal = {'displacement': displacement, 'energy': energy}\n"
        "    fallback = {}\n"
        "    a = consume(normal)\n"
        "    b = consume(fallback)\n"
        "    return a + b + failed\n"
    ),
    # Same, but the fallback is chosen at runtime by a branch (closer to the
    # real program: `if remaining > eps: ... else: normal path`).
    "two_sites_branch_selected": (
        "def consume(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
        "def root(displacement, energy, remaining):\n"
        "    if remaining > 0.0:\n"
        "        channels = {}\n"
        "    else:\n"
        "        channels = {'displacement': displacement, 'energy': energy}\n"
        "    return consume(channels)\n"
    ),
    # Same fix as would be applied to the real Metrics fallback: both
    # branches now produce the SAME key set (displacement, energy), just
    # with a zero default on the branch that has nothing to report.
    "two_sites_same_schema": (
        "def consume(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
        "def root(displacement, energy, remaining):\n"
        "    if remaining > 0.0:\n"
        "        channels = {'displacement': 0.0, 'energy': 0.0}\n"
        "    else:\n"
        "        channels = {'displacement': displacement, 'energy': energy}\n"
        "    return consume(channels)\n"
    ),
    # One aggregate object, constructed once; only its VALUES are
    # conditional.  If Phi-of-whole-aggregate is what breaks, this should
    # succeed where two_sites_same_schema (Phi of two separate dicts) failed.
    "one_object_conditional_values": (
        "def consume(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
        "def root(displacement, energy, remaining):\n"
        "    channels = {'displacement': 0.0, 'energy': 0.0}\n"
        "    if remaining > 0.0:\n"
        "        channels['displacement'] = displacement\n"
        "        channels['energy'] = energy\n"
        "    return consume(channels)\n"
    ),
}


def brief(value) -> str:
    accounting = dict(getattr(value, "accounting", None) or {})
    owner = accounting.get("program_abi_keyed_owner")
    part = accounting.get("program_abi_keyed_part")
    tag = f" [{owner}.{part}]" if owner else ""
    return f"%{value.id}:{value.dtype}{tag}"


def main() -> int:
    wanted = sys.argv[1:] or list(CASES)
    for case in wanted:
        source = CASES[case]
        print(f"===== {case} =====", flush=True)
        try:
            module, _outputs, _exports = lower_ast_source_to_ssa(
                source, "root", name=case, extraction_contract=CONTRACT,
            )
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {str(error)[:600]}", flush=True)
            continue
        for name, function in module.functions.items():
            print(f"  {name}({', '.join(brief(a) for a in function.args)})", flush=True)
            for block_name, block in function.blocks.items():
                print(f"    {block_name}:", flush=True)
                for instruction in block.instrs:
                    operands = " ".join(f"%{a.id}" for a in instruction.args)
                    result = (
                        f"%{instruction.res.id} = "
                        if instruction.res is not None else ""
                    )
                    print(f"      {result}{instruction.op} {operands}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
