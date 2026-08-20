"""A parameter that declares its extents compiles as a tensor, not a scalar.

The scalar-versus-tensor decision reads EXTENTS, not rank: an operation is
spelled with its scalar opcode when the result and every operand have an empty
shape (``hierarchical_plan.plan_region_to_ssa_instrs``). An undeclared
parameter arrives shapeless, so it is indistinguishable from a rank-0 scalar
and its arithmetic is compiled as scalar arithmetic -- the wrong program for a
tensor, reported as a success.

``program_abi`` could already declare ``storage: span`` and ``rank: 2`` for a
parameter, and both were ignored on this path (recorded in
``tools/HANDOFF_fluid_c_shell.md``: the record-field ABI fact "drops rank",
and the default domain then "collapses to scalar" because a node with no
tensor record gets ``DomainNode(shape=(1,1,1))`` whose unit dimensions are
filtered away to ``()``). Rank alone could not have fixed it either -- the
gate needs extents.

So the boundary can now state ``shape``, and this pins that the statement is
honoured end to end: the formals carry it, the instruction operands carry it,
and the operation stops being scalar-spelled.
"""
from __future__ import annotations

import pathlib
import warnings

import pytest

yaml = pytest.importorskip("yaml")

from src.compiler.extraction_contract import (
    ExtractionContract,
    ExtractionContractError,
    ProgramABIField,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CONTRACT = REPO / "extraction_contracts" / "program_extraction.yaml"
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"

SOURCE = "def f(x, y):\n    return x * y\n"


# -- the contract field itself ---------------------------------------------

def test_a_field_may_state_its_extents():
    field = ProgramABIField.from_mapping(
        "probe", {"storage": "span", "dtype": "float64", "rank": 2,
                  "shape": [3, 4]},
    )
    assert field.shape == (3, 4)
    assert field.receipt()["shape"] == [3, 4]


def test_a_field_without_extents_still_parses():
    """``shape`` is optional; rank-only declarations keep working."""

    field = ProgramABIField.from_mapping(
        "probe", {"storage": "span", "dtype": "float64", "rank": 2},
    )
    assert field.shape is None
    assert "shape" not in field.receipt()


def test_extents_must_agree_with_rank():
    """Two statements about the same axes disagreeing is the diagnostic."""

    with pytest.raises(ExtractionContractError, match="rank"):
        ProgramABIField.from_mapping(
            "probe", {"storage": "span", "dtype": "float64", "rank": 2,
                      "shape": [4]},
        )


@pytest.mark.parametrize("bad", [[0], [-1], [2.5], ["4"]])
def test_an_extent_must_be_a_positive_integer(bad):
    with pytest.raises(ExtractionContractError):
        ProgramABIField.from_mapping(
            "probe", {"storage": "span", "dtype": "float64", "rank": 1,
                      "shape": bad},
        )


# -- end to end through the compiler ---------------------------------------

def _contract(tmp_path, values):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {"records": {}, "bindings": [], "values": values}
    raw["roots"] = {
        "authored": [str(REPO / "examples")], "repository": [str(REPO)]
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return ExtractionContract(path)


def _span(parameter, extents):
    return {
        "function": "f", "parameter": parameter, "storage": "span",
        "dtype": "float64", "rank": len(extents), "shape": list(extents),
        "python_type": TENSOR_TYPE,
    }


def _lowered(tmp_path, values, name):
    contract = _contract(tmp_path, values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _ = lower_ast_source_to_ssa(
            SOURCE, "f", name=name, extraction_contract=contract
        )
    formals = {}
    operands = []
    for function_name, function in module.functions.items():
        if function_name.endswith("__f"):
            formals = {
                int(a.id): tuple(a.shape) for a in function.args
            }
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op).casefold() == "mul":
                    operands.append((
                        str(instruction.op),
                        tuple(instruction.res.shape),
                        [tuple(a.shape) for a in instruction.args],
                    ))
    return formals, operands


def test_an_undeclared_parameter_is_shapeless_and_spelled_scalar(tmp_path):
    """The baseline this exists to change -- and it is silent, not a failure."""

    formals, operands = _lowered(tmp_path, [], "undecl")
    assert all(shape == () for shape in formals.values())
    assert operands, "expected a multiply"
    opcode, result_shape, argument_shapes = operands[0]
    assert opcode == "Mul"          # the SCALAR spelling
    assert result_shape == ()
    assert all(shape == () for shape in argument_shapes)


def test_declared_extents_reach_the_formals_and_the_operands(tmp_path):
    values = [_span("x", (4,)), _span("y", (4,))]
    formals, operands = _lowered(tmp_path, values, "declared")

    assert formals, "expected formals"
    assert any(shape == (4,) for shape in formals.values()), (
        f"declared extents never reached the formals: {formals}"
    )

    opcode, result_shape, argument_shapes = operands[0]
    assert result_shape == (4,)
    assert argument_shapes == [(4,), (4,)]


def test_rank_alone_is_not_enough(tmp_path):
    """Recorded so the distinction is not lost again.

    ``rank`` says how many axes there are; the gate reads how long they are.
    A rank-only declaration leaves the value shapeless, which is exactly why
    the rank that ``program_abi`` already carried never fixed this.
    """

    values = [
        {"function": "f", "parameter": "x", "storage": "span",
         "dtype": "float64", "rank": 1, "python_type": TENSOR_TYPE},
        {"function": "f", "parameter": "y", "storage": "span",
         "dtype": "float64", "rank": 1, "python_type": TENSOR_TYPE},
    ]
    formals, operands = _lowered(tmp_path, values, "rankonly")
    assert all(shape == () for shape in formals.values())
    assert operands[0][1] == ()
