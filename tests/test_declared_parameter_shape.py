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

import ast
import pathlib
import warnings

import pytest

yaml = pytest.importorskip("yaml")

from src.compiler.extraction_contract import (
    ExtractionContract,
    ExtractionContractError,
    ProgramABIContract,
    ProgramABIField,
)
from src.compiler.fortran_c_shell import (
    _normalize_top_level_guard_returns,
    lower_ast_source_to_ssa,
)

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


def test_nested_record_field_names_its_exact_schema():
    field = ProgramABIField.from_mapping(
        "probe", {"storage": "record", "record": "NodeTable"},
    )

    assert field.record == "NodeTable"
    assert field.receipt() == {
        "storage": "record",
        "dtype": None,
        "rank": 0,
        "mutable": False,
        "record": "NodeTable",
    }


def test_nested_record_field_cannot_omit_its_schema():
    with pytest.raises(ExtractionContractError, match="record is required"):
        ProgramABIField.from_mapping("probe", {"storage": "record"})


def test_nested_record_schema_must_exist_in_the_same_contract():
    with pytest.raises(ExtractionContractError, match="unknown record 'Missing'"):
        ProgramABIContract.from_mapping({
            "records": {
                "Outer": {
                    "fields": {
                        "inner": {"storage": "record", "record": "Missing"},
                    },
                },
            },
            "bindings": [],
            "values": [],
        })


def test_keyed_integer_identity_preserves_existing_deterministic_ids():
    field = ProgramABIField.from_mapping("probe", {
        "storage": "keyed", "dtype": "int64", "rank": 1,
        "key_encoding": "integer_identity",
    })

    assert field.key_encoding == "integer_identity"
    assert field.receipt()["key_encoding"] == "integer_identity"


def test_keyed_values_can_name_a_record_row_schema():
    contract = ProgramABIContract.from_mapping({
        "records": {
            "Node": {
                "fields": {"kind": {"storage": "scalar", "dtype": "int64"}},
            },
            "Graph": {
                "fields": {
                    "nodes": {
                        "storage": "keyed", "dtype": "int64", "rank": 1,
                        "key_encoding": "integer_identity",
                        "value_record": "Node",
                    },
                },
            },
        },
        "bindings": [],
        "values": [],
    })

    assert contract.records["Graph"].fields["nodes"].value_record == "Node"


def test_keyed_record_can_use_deterministic_key_as_row_identity():
    contract = ProgramABIContract.from_mapping({
        "records": {
            "Node": {"fields": {
                "kind": {"storage": "scalar", "dtype": "int64"},
            }},
            "Graph": {"fields": {"nodes": {
                "storage": "keyed", "dtype": "int64", "rank": 1,
                "key_encoding": "integer_identity",
                "value_record": "Node", "value_identity": "key",
            }}},
        },
        "bindings": [], "values": [],
    })

    field = contract.records["Graph"].fields["nodes"]
    assert field.value_identity == "key"
    assert field.receipt()["value_identity"] == "key"


def test_scalar_token_vocabulary_is_ordered_reversible_not_hashed():
    field = ProgramABIField.from_mapping("probe", {
        "storage": "scalar", "dtype": "int64",
        "token_vocabulary": ["Constant", "Input"],
    })

    assert field.token_vocabulary == ("Constant", "Input")
    assert field.receipt()["token_vocabulary"] == ["Constant", "Input"]


@pytest.mark.parametrize("bad", [[0], [-1], [2.5], ["4"]])
def test_an_extent_must_be_a_positive_integer(bad):
    with pytest.raises(ExtractionContractError):
        ProgramABIField.from_mapping(
            "probe", {"storage": "span", "dtype": "float64", "rank": 1,
                      "shape": bad},
        )


# -- end to end through the compiler ---------------------------------------

def test_single_exit_normalization_is_targeted_and_records_authored_guards():
    tree = ast.parse(
        "def selected(x):\n"
        "    if x < 0:\n"
        "        return 1\n"
        "    return 2\n\n"
        "def adjacent(x):\n"
        "    if x < 0:\n"
        "        return 3\n"
        "    return 4\n"
    )

    receipt = _normalize_top_level_guard_returns(tree, ("selected",))

    assert receipt == ({
        "function": "selected",
        "result_name": "__turing_single_exit_result",
        "guard_count": 1,
        "source_lines": (2,),
    },)
    selected, adjacent = tree.body
    assert sum(isinstance(node, ast.Return) for node in ast.walk(selected)) == 1
    assert sum(isinstance(node, ast.Return) for node in ast.walk(adjacent)) == 2


def test_nested_return_guards_keep_path_specific_phi_inputs():
    source = (
        "def guarded(x):\n"
        "    if x < 0:\n"
        "        return 1\n"
        "    if x == 0:\n"
        "        return 2\n"
        "    if x == 1:\n"
        "        return 3\n"
        "    return 4\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _ = lower_ast_source_to_ssa(
            source, "guarded", name="guarded_single_exit",
        )

    function = next(
        value for name, value in module.functions.items()
        if name.endswith("__guarded")
    )
    assert function.metadata["source_conditional_count"] == 3
    assert function.metadata["lowered_conditional_count"] == 3
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    constants = {
        instruction.attributes.get("value"): int(instruction.res.id)
        for instruction in instructions
        if instruction.op == "Const"
        and instruction.res is not None
        and (instruction.res.accounting or {}).get("authored_constant")
    }
    phis = [instruction for instruction in instructions if instruction.op == "Phi"]
    assert len(phis) == 3
    inner = next(
        phi for phi in phis
        if {int(value.id) for value in phi.args}
        == {constants[3], constants[4]}
    )
    middle = next(
        phi for phi in phis
        if [int(value.id) for value in phi.args]
        == [constants[2], int(inner.res.id)]
    )
    outer = next(
        phi for phi in phis if phi is not inner and phi is not middle
    )
    assert [int(value.id) for value in outer.args] == [
        constants[1], int(middle.res.id),
    ]
    assert sum(instruction.op == "Ret" for instruction in instructions) == 1


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


def test_nested_record_leaf_becomes_the_only_physical_input(tmp_path):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {
        "records": {
            "CompilerProcessGraph": {
                "identity": "CompilerProcessGraph",
                "fields": {
                    "G": {"storage": "record", "record": "CompilerDiGraph"},
                },
            },
            "CompilerDiGraph": {
                "identity": "CompilerDiGraph",
                "fields": {
                    "enabled": {
                        "storage": "scalar", "dtype": "bool",
                        "mutable": False,
                    },
                },
            },
        },
        "bindings": [{
            "function": "read_flag", "parameter": "graph",
            "record": "CompilerProcessGraph",
        }],
        "values": [],
    }
    path = tmp_path / "nested-record.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _ = lower_ast_source_to_ssa(
            "def read_flag(graph):\n"
            "    if isinstance(graph.G.enabled, bool):\n"
            "        return graph.G.enabled\n"
            "    return False\n",
            "read_flag",
            name="nested_record",
            extraction_contract=ExtractionContract(path),
        )

    function_name = next(
        name for name in module.functions if name.endswith("__read_flag")
    )
    function = module.functions[function_name]
    assert [
        (value.dtype, value.accounting["program_abi_field"])
        for value in function.args
    ] == [("bool", "G.enabled")]
    records = module.record_tables[function_name].records
    outer = next(
        record for record in records.values()
        if record.identity == "CompilerProcessGraph"
    )
    nested = records[outer.fields[0].record_id]
    assert nested.identity == "CompilerDiGraph"
    assert nested.fields[0].value_ids == (function.args[0].id,)


def test_keyed_record_scalar_field_loads_from_deterministic_row_column(
    tmp_path,
):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {
        "records": {
            "CompilerProcessGraph": {
                "identity": "CompilerProcessGraph",
                "fields": {
                    "G": {"storage": "record", "record": "CompilerDiGraph"},
                },
            },
            "CompilerDiGraph": {
                "identity": "CompilerDiGraph",
                "fields": {
                    "nodes": {
                        "storage": "keyed", "dtype": "int64", "rank": 1,
                        "key_encoding": "integer_identity",
                        "value_record": "CompilerNode",
                        "value_identity": "key",
                    },
                },
            },
            "CompilerNode": {
                "identity": "CompilerNode",
                "fields": {
                    "kind": {"storage": "scalar", "dtype": "int64"},
                },
            },
        },
        "bindings": [{
            "function": "read_kind", "parameter": "graph",
            "record": "CompilerProcessGraph",
        }],
        "values": [],
    }
    path = tmp_path / "keyed-record-row.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _ = lower_ast_source_to_ssa(
            "def read_kind(graph, node_id):\n"
            "    data = graph.G.nodes[node_id]\n"
            "    return data.get('kind')\n",
            "read_kind",
            name="keyed_record_row",
            extraction_contract=ExtractionContract(path),
        )

    function_name = next(
        name for name in module.functions if name.endswith("__read_kind")
    )
    function = module.functions[function_name]
    columns = [
        value for value in function.args
        if (value.accounting or {}).get("program_abi_field")
        == "G.nodes[].kind.column"
    ]
    assert len(columns) == 1
    assert columns[0].dtype == "int64"
    field_loads = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding")
        == "program_abi_record_field"
    ]
    assert len(field_loads) == 1
    assert field_loads[0].attributes["program_abi_field"] == (
        "G.nodes[].kind"
    )
    row_record = module.record_tables[function_name].records[
        int(field_loads[0].res.accounting["program_abi_row_handle"])
    ]
    assert row_record.identity == "CompilerNode"
    assert row_record.fields[0].value_ids == (field_loads[0].res.id,)
    keys = next(
        value for value in function.args
        if (value.accounting or {}).get("program_abi_field")
        == "G.nodes.keys"
    )
    lookup = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("keyed_lookup_owner") == "G.nodes"
    )
    assert int(lookup.args[0].id) == int(keys.id)
    assert int(lookup.args[1].id) == int(keys.id)


def test_keyed_record_vocabulary_lowers_isinstance_to_exact_token_tests(
    tmp_path,
):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {
        "records": {
            "Graph": {"identity": "Graph", "fields": {
                "nodes": {
                    "storage": "keyed", "dtype": "int64", "rank": 1,
                    "key_encoding": "integer_identity",
                    "value_record": "Node", "value_identity": "key",
                },
            }},
            "Node": {"identity": "Node", "fields": {
                "expr_obj": {
                    "storage": "scalar", "dtype": "int64",
                    "token_vocabulary": ["builtins.tuple", "builtins.list"],
                },
            }},
        },
        "bindings": [{
            "function": "is_aggregate", "parameter": "graph",
            "record": "Graph",
        }],
        "values": [],
    }
    path = tmp_path / "vocabulary-type-test.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _ = lower_ast_source_to_ssa(
            "def is_aggregate(graph, node_id):\n"
            "    data = graph.nodes[node_id]\n"
            "    return isinstance(data.get('expr_obj'), (tuple, list))\n",
            "is_aggregate",
            name="vocabulary_type_test",
            extraction_contract=ExtractionContract(path),
        )

    function_name = next(
        name for name in module.functions if name.endswith("__is_aggregate")
    )
    function = module.functions[function_name]
    tests = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("program_abi_vocabulary_type_test")
    ]
    assert [instruction.op for instruction in tests] == ["Eq", "Eq", "Or"]
    constants = {
        instruction.attributes.get("program_abi_vocabulary_token"):
            instruction.attributes.get("value")
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("program_abi_vocabulary_token")
    }
    assert constants == {"builtins.tuple": 1, "builtins.list": 2}
    assert not any(
        (value.accounting or {}).get("program_abi_storage") == "reference"
        for value in function.args
    )
