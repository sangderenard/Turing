"""Fast repro: a record parameter of a keyed field's value record.

Reproduces, in seconds, the frame-linker failure the Woodshop whole-program
compile reaches after ~520 s (``_sync_newton_lanes`` -> ``center_xyz``):

    ValueError: ... callee ... formal N is member (...) of declared field
    'Item.orientation' of record 0, bound to caller record 3, which has no
    such field

Shape of the program.  ``World.items`` is a keyed field whose rows are the
record ``Item``.  The caller indexes the mapping, reads ONE leaf of the row
(``mass``) and hands the row to a callee whose parameter is bound to ``Item``.
The callee reads a leaf the caller never touched (``orientation``).

Observed records at the raise (``_linked_caller_member``):

* caller: ``Item`` is a pooled row -- the lookup result is the row handle and
  the row's leaves are columns ``items[].<leaf>.column`` minted lazily by
  ``materialize_nested_record`` for the leaves this function reads.  The
  caller's row record therefore holds ``mass`` and ``mass.column`` only.
* callee: ``Item`` is expanded as a flat record parameter -- one per-field
  formal per declared leaf (scalar ``mass``, span ``orientation``).
* the call's ``argument_bindings`` pair the two records correctly.

One identity, two representations; the linker cannot map the callee's
``orientation`` formal onto a caller row that never minted that column.
The drafted rule (docs/PARKED_2026-09-29.md §2): a record parameter whose
type is a keyed field's ``value_record`` is a row handle.

Run from the repository root:

    python -u tools/compiler_probes/probe_row_handle_record_parameter.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

BASE_CONTRACT = REPO / "extraction_contracts" / "program_extraction.yaml"

SOURCE = (
    "def center(item):\n"
    "    return item.orientation[0] * item.mass\n"
    "\n"
    "def sync(world, item_id):\n"
    "    item = world.items[item_id]\n"
    "    heavy = item.mass * 2.0\n"
    "    return center(item) + heavy\n"
)


def contract_path() -> pathlib.Path:
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {
        "records": {
            "World": {
                "identity": "World",
                "fields": {
                    "items": {
                        "storage": "keyed", "dtype": "int64", "rank": 1,
                        "key_encoding": "integer_identity",
                        "value_record": "Item",
                        "value_identity": "key",
                    },
                },
            },
            "Item": {
                "identity": "Item",
                "fields": {
                    "mass": {"storage": "scalar", "dtype": "float64"},
                    "orientation": {
                        "storage": "span", "dtype": "float64",
                        "rank": 1, "shape": [3],
                    },
                },
            },
        },
        "bindings": [
            {"function": "sync", "parameter": "world", "record": "World"},
            {"function": "center", "parameter": "item", "record": "Item"},
        ],
        "values": [],
    }
    build = REPO / "build"
    build.mkdir(exist_ok=True)
    path = build / "probe_row_handle_record_parameter.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            SOURCE, "sync", name="row_handle_probe",
            extraction_contract=ExtractionContract(contract_path()),
        )
    print("LOWERED functions:", sorted(module.functions))
    for name, function in module.functions.items():
        print(name, [
            (int(value.id), (value.accounting or {}).get("program_abi_field"))
            for value in function.args
        ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
