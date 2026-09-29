"""Struct intake: bound ctypes classes reach the module as struct/union rows.

Rung 1 of the laid-out type work.  The program below never touches its bound
``ctypes.Structure`` / ``ctypes.Union`` classes; the only thing under test is
that the sanctioned entry reads them from the LIVE classes at binding intake
and that the finished module's struct and union tables hold rows equal to what
ctypes itself reports, counted in the host schema, with their layout state and
member claims on the identity book.

    python -u tools/compiler_probes/probe_struct_intake.py
"""
from __future__ import annotations

import ctypes
import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.transmogrifier.ctypes_layout import host_schema, type_identity  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"


class Pair(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("n", ctypes.c_int64)]


class Words(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_int32), ("hi", ctypes.c_int32),
                ("tail", ctypes.c_float)]


class Cell(ctypes.Union):
    _fields_ = [("pair", Pair), ("words", Words)]


SOURCE = "def bump(k):\n    return k + 1\n"


BINDINGS = {"Cell": Cell, "Pair": Pair, "Words": Words}


def lower():
    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {}, "bindings": [],
            "values": [{
                "function": "bump", "parameter": "k",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            }],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            SOURCE, "bump", name="struct_intake",
            python_bindings=dict(BINDINGS), extraction_contract=contract,
            runtime_closure_only=True,
        )


failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def main() -> int:
    module, _outputs, _exports = lower()
    schema = host_schema()

    for ctype in (Pair, Words):
        row = module.struct_table.by_identity(type_identity(ctype))
        check(f"{ctype.__name__} is a struct row on the module", row is not None)
        if row is None:
            continue
        check(f"{ctype.__name__} counted in the host schema", row.schema == schema)
        check(f"{ctype.__name__} size", row.size == ctypes.sizeof(ctype))
        check(f"{ctype.__name__} alignment", row.alignment == ctypes.alignment(ctype))
        for name, _member in ctype._fields_:
            member = row.field(name)
            slot = getattr(ctype, name)
            check(f"{ctype.__name__}.{name} offset and size",
                  member is not None and member.offset == slot.offset
                  and member.size == slot.size)
        check(f"{ctype.__name__} layout state is resolved",
              module.struct_table.layout_state(row.struct_id)[0] == "resolved")

    union = module.union_table.by_identity(type_identity(Cell))
    check("Cell is a union row on the module", union is not None)
    if union is not None:
        check("Cell counted in the host schema", union.schema == schema)
        check("Cell size", union.size == ctypes.sizeof(Cell))
        check("Cell alignment", union.alignment == ctypes.alignment(Cell))
        check("Cell members are the two structs",
              {m.name for m in union.members} == {"pair", "words"})
        check("Cell layout state is resolved",
              module.union_table.layout_state(union.union_id)[0] == "resolved")
        found = module.layout_row_by_identity(type_identity(Cell))
        check("layout_row_by_identity finds the union",
              found is not None and found[0] == "union")

    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
