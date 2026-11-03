"""Schema identities remain source evidence outside numerical region ABIs."""

from pathlib import Path

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
