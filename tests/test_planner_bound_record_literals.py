"""Bound record literals type through the planner as the Python values they are.

A frozen record holding tensors, a tuple field and an ``IntEnum`` member are
ordinary Python values: the structural fold compares the record by content,
a field read off it is that field's class, and an ``IntEnum`` member is an
exact Python int.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.compiler.concordance_declarations import literal_identity


class Phase(enum.IntEnum):
    START = 0
    RUN = 3


@dataclass(frozen=True)
class Config:
    rows: tuple


@dataclass(frozen=True)
class State:
    cursor: object
    phase: object


def _project(source, bindings, mutable):
    compilation = compile_ast_aot(
        source,
        "step",
        bindings,
        python_bindings={"Phase": Phase},
        precompile_only=True,
        mutable_parameters=mutable,
    )
    return project_public_numerical_program(compilation)


def _result_dtype(program):
    return program.meta[int(program.steps[-1].result_id)].dtype


def test_a_record_of_tensors_is_one_literal_by_content():
    first = State(AT.get_tensor([1, 2], dtype="int64"), (0, 0))
    same = State(AT.get_tensor([1, 2], dtype="int64"), (0, 0))
    other = State(AT.get_tensor([1, 3], dtype="int64"), (0, 0))

    assert literal_identity(first) == literal_identity(same)
    assert literal_identity(first) != literal_identity(other)
    # A record holding no tensor keeps its own equality.
    assert literal_identity(Config((5, 7))) == Config((5, 7))


def test_an_int_enum_member_multiplies_as_an_exact_int():
    program = _project(
        "def step(offsets):\n    return offsets * Phase.RUN\n",
        {"offsets": AT.get_tensor([10, 20], dtype="int64")},
        ("offsets",),
    )

    assert _result_dtype(program) == "int64"


def test_fields_of_a_bound_record_type_the_arithmetic_they_feed():
    program = _project(
        "def step(state, config, offsets):\n"
        "    advanced = state.cursor + config.rows[0]\n"
        "    return advanced + offsets * Phase.RUN\n",
        {
            "state": State(
                AT.get_tensor([1, 2], dtype="int64"),
                AT.get_tensor([0, 0], dtype="int64"),
            ),
            "config": Config((5, 7)),
            "offsets": AT.get_tensor([10, 20], dtype="int64"),
        },
        ("offsets",),
    )

    assert _result_dtype(program) == "int64"
