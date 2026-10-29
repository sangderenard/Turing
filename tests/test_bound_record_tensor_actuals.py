"""A tensor field of a bound record is a tensor actual with its own extents.

``helper(head.cfg.small, rows)`` and ``helper(head.cfg.big, rows)`` pass two
tensor fields of one bound record (held by a plain authored object).  Each
field carries its literal's shape and class into the call, so the two
callsites mint two copies of ``helper``, each reading its own
``int(table.shape[1])``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler.concordance_declarations import literal_identity


@dataclass(frozen=True)
class Tables:
    small: object
    big: object


class Head:
    def __init__(self, tables):
        self.tables = tables


SOURCE = """
def helper(table, row):
    width = int(table.shape[1])
    return row * width

def f(head, rows):
    return helper(head.tables.small, rows) + helper(head.tables.big, rows)
"""


def _head():
    return Head(Tables(
        AT.get_tensor([[0] * 8] * 2, dtype="int64"),
        AT.get_tensor([[0] * 256] * 2, dtype="int64"),
    ))


def _plan_lines(plan):
    for item in getattr(plan, "items", None) or ():
        if hasattr(item, "opcode"):
            yield item
        yield from _plan_lines(item)
        callee = getattr(item, "callee", None)
        if callee is not None:
            yield from _plan_lines(callee)


def test_a_plain_object_holding_tensors_is_one_literal_by_content():
    assert literal_identity(_head()) == literal_identity(_head())


def test_each_tensor_field_mints_its_own_copy_of_the_callee():
    compilation = compile_ast_aot(
        SOURCE,
        "f",
        {"head": _head(), "rows": AT.get_tensor([1, 2], dtype="int64")},
        python_bindings={"Tables": Tables, "Head": Head},
        precompile_only=True,
        mutable_parameters=("rows",),
    )

    widths = sorted(
        dict(line.attributes).get("value")
        for line in _plan_lines(compilation.hierarchy_plan)
        if line.opcode == "Const"
    )
    assert widths == [8, 256]
