"""Schema identities remain source evidence outside numerical region ABIs."""

from pathlib import Path
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import begin_identity_book, end_identity_book


def test_arange_schema_class_operand_has_no_runtime_dtype_requirement():
    book, token = begin_identity_book()
    try:
        policy = ExtractionContract(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts/program_extraction.yaml"
        ).with_program_abi({
            "records": {}, "bindings": [],
            "values": [
                {"function": "root", "parameter": "lengths",
                 "storage": "span", "dtype": "int64", "rank": 1,
                 "python_type": "src.common.tensors.abstraction.AbstractTensor"},
                {"function": "root", "parameter": "count",
                 "storage": "scalar", "dtype": "int64", "rank": 0,
                 "python_type": "builtins.int"},
            ],
        })
        module, _outputs, _exports = lower_ast_source_to_ssa(
            "def root(lengths, count):\n"
            "    return AbstractTensor.arange(count, cls=type(lengths))\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="arange_schema_operand", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
        from src.compiler.concordance_declarations import (
            NUMERICAL_SCHEMA_OPERAND, PROCESS_GRAPH_OPERANDS,
        )

        book = module.metadata["identity_book"]
        rows = tuple(book.page(NUMERICAL_SCHEMA_OPERAND).rows())
        assert rows
        for row in rows:
            cell = book.latest_ref(NUMERICAL_SCHEMA_OPERAND, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            assert row[0] in sources and row[2] in sources
            assert any(
                source.page == PROCESS_GRAPH_OPERANDS
                and (row[2], row[1]) in book.page(source.page).cell(
                    source.row, source.column
                )
                for source in sources
            )
        aranges = [
            instruction
            for function in module.functions.values()
            for block in function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op).casefold() == "arange"
        ]
        assert aranges
        assert all("kw:cls" not in instruction.arg_roles for instruction in aranges)
        from src.compiler.identity_concordance import (
            CorrelationTable, Derived, Mode,
        )
        from src.compiler.concordance_declarations import PLANNER_HIERARCHY

        assert CorrelationTable._numerical_schema_operand_findings(module) == []
        row = rows[0]
        receipt = book.latest_ref(NUMERICAL_SCHEMA_OPERAND, row)
        sources = tuple(source for source, _stage in book.edges_into(receipt))
        book.post(
            NUMERICAL_SCHEMA_OPERAND,
            (row[0], "unrelated_operand", row[2]),
            book.page(NUMERICAL_SCHEMA_OPERAND).latest(row),
            stage=PLANNER_HIERARCHY, provenance=Derived(sources),
            mode=Mode.CONCORD,
        )
        findings = CorrelationTable._numerical_schema_operand_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == "numerical-schema-operand"
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("target", ["int64", "float64", "bool"])
def test_cast_dtype_literal_is_projected_into_operation_metadata(target):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{"function": "root", "parameter": "values",
                    "storage": "span", "dtype": "bool", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        f"def root(values):\n    return values.to_dtype({target!r})\n",
        "root", name=f"cast_metadata_{target}", extraction_contract=policy,
        progress=lambda _: None, progress_bars=False,
    )
    from src.compiler.concordance_declarations import NUMERICAL_DTYPE_OPERAND
    from src.compiler.identity_concordance import CorrelationTable

    book = module.metadata["identity_book"]
    rows = tuple(book.page(NUMERICAL_DTYPE_OPERAND).rows())
    assert rows
    assert all(book.page(NUMERICAL_DTYPE_OPERAND).latest(row)[1] == target
               for row in rows)
    assert CorrelationTable._numerical_dtype_operand_findings(module) == []
    instructions = [
        instruction for function in module.functions.values()
        for block in function.blocks.values() for instruction in block.instrs
    ]
    assert not any(
        str(instruction.op).casefold() == "const"
        and instruction.attributes.get("value") == target
        for instruction in instructions
    )
    casts = [instruction for instruction in instructions
             if str(instruction.op).casefold() == "to_dtype"]
    assert casts
    assert all(len(instruction.args) == 1
               and instruction.attributes["dtype"] == target
               for instruction in casts)
    from src.compiler.identity_concordance import Derived, Mode
    from src.compiler.concordance_declarations import PLANNER_HIERARCHY

    row = rows[0]
    receipt = book.latest_ref(NUMERICAL_DTYPE_OPERAND, row)
    literal_cells, _dtype = book.page(NUMERICAL_DTYPE_OPERAND).latest(row)
    sources = tuple(source for source, _stage in book.edges_into(receipt))
    book.post(
        NUMERICAL_DTYPE_OPERAND, row,
        (literal_cells, "float32"), stage=PLANNER_HIERARCHY,
        provenance=Derived((*sources, receipt)), mode=Mode.REVISE,
    )
    assert len(CorrelationTable._numerical_dtype_operand_findings(module)) == 1


def test_unrolled_cast_uses_the_cloned_literal_cell():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{"function": "root", "parameter": "values",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(values):\n"
        "    out = values * 0\n"
        "    for index in range(2):\n"
        "        out = out + (values >= index).to_dtype('int64')\n"
        "    return out\n",
        "root", name="unrolled_cast_literal", extraction_contract=policy,
        progress=lambda _: None, progress_bars=False,
    )
    from src.compiler.concordance_declarations import (
        NUMERICAL_DTYPE_OPERAND, SOURCE_LITERAL_VALUE,
    )
    from src.compiler.identity_concordance import CorrelationTable

    book = module.metadata["identity_book"]
    assert CorrelationTable._numerical_dtype_operand_findings(module) == []
    rows = tuple(book.page(NUMERICAL_DTYPE_OPERAND).rows())
    assert rows
    assert any(
        any(source.page == SOURCE_LITERAL_VALUE
            for literal_cell in book.page(NUMERICAL_DTYPE_OPERAND).latest(row)[0]
            for source, _stage in book.edges_into(literal_cell))
        for row in rows
    ), "the cast must consume a copied literal fact, not only authored literals"


def test_record_cast_target_type_does_not_require_settled_reshape_geometry():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {"Pack": {"identity": "Pack", "fields": {
            "valid": {"storage": "span", "dtype": "bool", "rank": 2},
            "max_bits": {"storage": "scalar", "dtype": "int64", "rank": 0},
        }}},
        "bindings": [{"function": "root", "parameter": "pack", "record": "Pack"}],
        "values": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "class Pack:\n"
        "    valid: AbstractTensor\n"
        "    max_bits: int\n"
        "def root(pack: Pack):\n"
        "    valid = pack.valid.reshape(-1, pack.max_bits)\n"
        "    return valid.to_dtype('int64')\n",
        "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="record_cast_target_type", extraction_contract=policy,
        progress=lambda _: None, progress_bars=False,
    )
    from src.compiler.concordance_declarations import (
        NUMERICAL_DTYPE_OPERAND, PLAN_REGION_VALUE_DTYPE,
    )
    from src.compiler.identity_concordance import CorrelationTable

    book = module.metadata["identity_book"]
    assert CorrelationTable._numerical_dtype_operand_findings(module) == []
    rows = tuple(book.page(NUMERICAL_DTYPE_OPERAND).rows())
    assert rows
    for row in rows:
        receipt = book.latest_ref(NUMERICAL_DTYPE_OPERAND, row)
        type_cells = [cell for cell, _stage in book.edges_into(receipt)
                      if cell.page == PLAN_REGION_VALUE_DTYPE]
        assert len(type_cells) == 1
        assert type_cells[0].row == row[0].row
        assert book.page(PLAN_REGION_VALUE_DTYPE).cell(
            type_cells[0].row, type_cells[0].column
        ) == "int64"


@pytest.mark.parametrize("explicit_dtype", [True, False])
def test_dynamic_constructor_region_reads_its_settled_explicit_dtype(monkeypatch, explicit_dtype):
    from src.compiler import hierarchical_plan

    class ObservedConstructorRegion(Exception):
        pass

    original = hierarchical_plan.plan_region_to_ssa_instrs

    def inspect_region(region, **kwargs):
        constructors = [item for item in region.items
                        if getattr(item, "opcode", None) == "zeros"]
        if constructors:
            result_id = constructors[0].outputs[0]
            assert (result_id, (), "float64") in region.value_shapes
            assert dict(constructors[0].attributes)["dtype"] == "float64"
            if not explicit_dtype:
                from types import SimpleNamespace
                from src.compiler.concordance_declarations import NUMERICAL_CONSTRUCTOR_DEFAULT
                from src.compiler.identity_concordance import CorrelationTable, current_identity_book

                book = current_identity_book()
                assert tuple(book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).rows())
                module = SimpleNamespace(metadata={"identity_book": book})
                assert CorrelationTable._numerical_constructor_default_findings(module) == []
            raise ObservedConstructorRegion
        return original(region, **kwargs)

    monkeypatch.setattr(hierarchical_plan, "plan_region_to_ssa_instrs", inspect_region)
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{"function": "root", "parameter": "like",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
    })
    dtype_argument = ", dtype='float64'" if explicit_dtype else ""
    with pytest.raises(ObservedConstructorRegion):
        lower_ast_source_to_ssa(
            "def root(like, count: int):\n"
            f"    return AbstractTensor.zeros((count + 1,), cls=type(like){dtype_argument})\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="explicit_constructor_type_record", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_tensor_byte_materialization_keeps_rows_and_explicit_target(monkeypatch, dtype):
    from src.compiler import hierarchical_plan
    from src.compiler.concordance_declarations import CANONICAL_VALUE, SEQUENCE_ROW_DTYPE
    from src.compiler.concorded_literals import origin_cell
    from src.compiler.identity_concordance import current_identity_book

    class ObservedTensorRegion(Exception):
        pass

    original = hierarchical_plan.plan_region_to_ssa_instrs

    def inspect_region(region, **kwargs):
        tensors = [item for item in region.items
                   if getattr(item, "opcode", None) == "tensor"]
        if tensors:
            item = tensors[0]
            wanted = dtype or "int64"
            assert (item.outputs[0], (), wanted) in region.value_shapes
            if dtype is not None:
                assert dict(item.attributes)["dtype"] == dtype
            book = current_identity_book()
            source = item.inputs[0]
            current = book.latest_ref(
                CANONICAL_VALUE, (tuple(kwargs["lexical_read_scope"]), int(source)),
            )
            page = book.page(SEQUENCE_ROW_DTYPE)
            claims = tuple(
                book.latest_ref(SEQUENCE_ROW_DTYPE, row)
                for row in page.rows() if page.latest(row)[0] == ("int64",)
            )
            assert any(
                cause.page == CANONICAL_VALUE
                and cause.row == claim.row
                and origin_cell(book, cause) == origin_cell(book, current)
                for claim in claims
                for cause, _stage in book.edges_into(claim)
            )
            raise ObservedTensorRegion
        return original(region, **kwargs)

    monkeypatch.setattr(hierarchical_plan, "plan_region_to_ssa_instrs", inspect_region)
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{"function": "root", "parameter": "data",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "builtins.bytes"}],
    })
    target = "" if dtype is None else f", dtype={dtype!r}"
    with pytest.raises(ObservedTensorRegion):
        lower_ast_source_to_ssa(
            f"def root(data: bytes):\n    return AbstractTensor.tensor(list(data){target})\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="tensor_byte_rows", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
