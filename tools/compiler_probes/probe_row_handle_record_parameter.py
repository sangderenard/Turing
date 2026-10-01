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


def book_assertions(module) -> list[str]:
    """Concordance step 7 (plan 90, section 3.7), read from the book only.

    * one ``record_parameter_row_handle`` row for the Indexed value in
      ``sync``, with inbound edges;
    * for every formal of ``center``, one ``argument_binding`` row whose
      fact is not ``Unresolved`` -- or whose reason is printed;
    * one ``linked_caller_member`` row per ``Item`` leaf ``center`` reads
      (``orientation``, ``mass``), with an inbound edge;
    * zero ``result_storage_binding`` rows for ``center``'s record formals;
    * no MINTED id defined in the two functions lacks a mint record.
    """

    from src.compiler.concordance_declarations import (
        ARGUMENT_BINDING, LINKED_CALLER_MEMBER, RECORD_PARAMETER_ROW_HANDLE,
        RESULT_STORAGE_BINDING,
    )
    from src.compiler.identity_concordance import (
        MINT_PAGE, Unresolved, identity_book,
    )
    from src.compiler.id_space import MINTED, has_flag

    book = identity_book(module)
    failures: list[str] = []
    functions = {
        name.rsplit("__", 1)[-1]: function
        for name, function in module.functions.items()
        if name.endswith("__sync") or name.endswith("__center")
    }
    sync, center = functions["sync"], functions["center"]

    handles = book.pages.get(RECORD_PARAMETER_ROW_HANDLE.name)
    handle_rows = [] if handles is None else [
        row for row in handles.rows()
        if isinstance(row[1], tuple) and row[1][0] == str(sync.name)
        and not isinstance(handles.latest(row), Unresolved)
    ]
    if len(handle_rows) != 1:
        failures.append(f"row handles in sync: {handle_rows!r} (want one)")
    for row in handle_rows:
        edges = book.edges_into(book.latest_ref(RECORD_PARAMETER_ROW_HANDLE, row))
        print("row handle", row[1], "->", handles.latest(row), "edges:", len(edges))
        if not edges:
            failures.append(f"row handle {row!r} has no inbound edge")

    bindings = book.pages.get(ARGUMENT_BINDING.name)
    for formal in center.args:
        rows = [] if bindings is None else [
            row for row in bindings.scope_rows(str(center.name))
            if int(row[1]) == int(formal.id)
        ]
        facts = [bindings.latest(row) for row in rows]
        for row, fact in zip(rows, facts):
            if isinstance(fact, Unresolved):
                print(f"center formal {int(formal.id)}: Unresolved({fact.reason.name}) at {row[2]!r}")
            else:
                print(f"center formal {int(formal.id)}: {fact!r} at {row[2]!r}")
        if not rows:
            failures.append(f"center formal {int(formal.id)} has no argument_binding row")

    linked = book.pages.get(LINKED_CALLER_MEMBER.name)
    linked_rows = [] if linked is None else [
        row for row in linked.scope_rows(str(sync.name))
        if row[2] == str(center.name) and isinstance(linked.latest(row), int)
    ]
    for row in linked_rows:
        edges = book.edges_into(book.latest_ref(LINKED_CALLER_MEMBER, row))
        print("linked member", row[3], "->", linked.latest(row), "edges:", len(edges))
        if not edges:
            failures.append(f"linked member {row!r} has no inbound edge")
    leaf_formals = [
        formal for formal in center.args
        if (formal.accounting or {}).get("program_abi_field")
    ]
    if len(linked_rows) < len(leaf_formals):
        failures.append(
            f"linked members {len(linked_rows)} < record formals {len(leaf_formals)}"
        )

    leases = book.pages.get(RESULT_STORAGE_BINDING.name)
    leased = [] if leases is None else [
        row for row in leases.scope_rows(str(sync.name))
        if int(row[2]) in {int(formal.id) for formal in leaf_formals}
    ]
    if leased:
        failures.append(f"record formals of center were leased: {leased!r}")

    minted_with_record = {
        row[1] for row in (
            () if book.pages.get(MINT_PAGE.name) is None
            else book.pages[MINT_PAGE.name].rows()
        )
    }
    for function in (sync, center):
        defined = {int(value.id) for value in function.args} | {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        orphans = sorted(
            value_id for value_id in defined
            if has_flag(value_id, MINTED) and value_id not in minted_with_record
        )
        print(f"{function.name}: minted ids without a mint record: {len(orphans)}")
        if orphans:
            failures.append(f"{function.name}: unsourced identities {orphans!r}")
    return failures


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
    failures = book_assertions(module)
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
