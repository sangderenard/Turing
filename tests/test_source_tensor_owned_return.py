"""An owned tensor returned by an authored source function reaches native C."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    CALL_RECORD, SSA_VALUE, TENSOR_ELEMENTWISE_EXTENT_BINDING,
    TENSOR_SOURCE_OWNED_RETURN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


@pytest.mark.parametrize(
    ("element_dtype", "increment"),
    (("int64", "9007199254740993"), ("float64", "0.25")),
    ids=("int64-above-double-integer-precision", "float64-quarter"),
)
def test_source_owned_tensor_return_native(tmp_path, element_dtype, increment):
    source = (
        "def make(count: int) -> AbstractTensor:\n"
        f"    return AbstractTensor.arange(count).to_dtype('{element_dtype}')\n"
        "\n"
        "def root(count: int):\n"
        f"    return (make(count) + {increment}).sum().item()\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "count", "storage": "scalar",
            "dtype": "int64", "rank": 0, "shape": [],
            "python_type": "builtins.int",
        }],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name=f"source_owned_return_{element_dtype}",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    book = identity_book(module)
    page = book.page(TENSOR_SOURCE_OWNED_RETURN)
    rows = tuple(page.rows())
    assert len(rows) == 1
    row = rows[0]
    fact = page.latest(row)
    source_function = module.functions[row[3]]
    caller_function = module.functions[row[1]]
    loaded_from = {}
    for function, carrier_id in (
        (source_function, fact[0]), (caller_function, fact[2]),
    ):
        definitions = tuple(
            instruction for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
            and int(instruction.res.id) == int(carrier_id)
        )
        assert len(definitions) == 1 and definitions[0].op == "Load"
        assert len(definitions[0].args) == 1
        loaded_from[function.name] = int(definitions[0].args[0].id)
        assert all(
            instruction.op != "Alloca" or instruction.res is None
            or int(instruction.res.id) != int(carrier_id)
            for block in function.blocks.values()
            for instruction in block.instrs
        )

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"source_owned_return_{element_dtype}")
    root = module.functions[exports[0]]
    count_port = dict(root.metadata["parameter_names"])["count"]
    output_ids = tuple(
        int(value.id) for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    assert len(output_ids) == 1
    for count in (0, 1, 7):
        expected = count * (count - 1) // 2 + count * (
            9007199254740993 if element_dtype == "int64" else 0.25
        )
        execution = artifact.prepare_execution({
            int(count_port): np.asarray(count, dtype=np.int64),
        })
        for repeat in range(2):
            result = execution.run().buffers[output_ids[0]]
            actual = np.asarray(result).item()
            assert actual == expected, (count, repeat, actual, expected)

    if element_dtype == "int64":
        key = (book.latest_ref(TENSOR_SOURCE_OWNED_RETURN, row).row,
               book.latest_ref(TENSOR_SOURCE_OWNED_RETURN, row).column)
        original = page.cells[key]
        try:
            page.cells[key] = (
                *original[:3], original[1], *original[4:],
            )
            assert CorrelationTable._tensor_source_owned_return_findings(module)
            page.cells[key] = (
                loaded_from[source_function.name], *original[1:],
            )
            assert CorrelationTable._tensor_source_owned_return_findings(module)
        finally:
            page.cells[key] = original
        assert CorrelationTable.build(module).findings(module) == []

        carrier_ref = ssa_value_identity_cell(
            caller_function, fact[2], book=book,
        )
        assert carrier_ref is not None
        carrier_page = book.page(SSA_VALUE)
        carrier_key = (carrier_ref.row, carrier_ref.column)
        carrier_fact = carrier_page.cells[carrier_key]
        try:
            carrier_page.cells[carrier_key] = replace(
                carrier_fact, physical_dtype="int64",
            )
            assert CorrelationTable._tensor_source_owned_return_findings(module)
        finally:
            carrier_page.cells[carrier_key] = carrier_fact
        assert CorrelationTable.build(module).findings(module) == []

        # The elementwise receipt cites the source Call's historical record.
        # Its later owner-tail revision cannot redeem a forged old semantic
        # result binding: both records must still name make(4) -> root(1).
        extent_page = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
        extent_refs = tuple(
            book.latest_ref(TENSOR_ELEMENTWISE_EXTENT_BINDING, extent_row)
            for extent_row in extent_page.rows()
            if extent_row[0] == row[0]
        )
        cited_records = tuple(
            parent for extent_ref in extent_refs
            for parent, _stage in book.edges_into(extent_ref)
            if parent.page == CALL_RECORD
        )
        assert len(cited_records) == 1
        record_ref = cited_records[0]
        assert record_ref.column < book.latest_ref(
            CALL_RECORD, record_ref.row,
        ).column
        record_page = book.page(CALL_RECORD)
        record_key = (record_ref.row, record_ref.column)
        old_records = record_page.cells[record_key]
        assert any(
            record.callee_symbol == source_function.name
            and (row[4], fact[4]) in record.result_bindings
            for record in old_records
        )
        try:
            record_page.cells[record_key] = tuple(
                replace(
                    record,
                    result_bindings=tuple(
                        binding for binding in record.result_bindings
                        if binding != (row[4], fact[4])
                    ),
                ) if record.callee_symbol == source_function.name else record
                for record in old_records
            )
            assert CorrelationTable._tensor_elementwise_extent_binding_findings(
                module,
            )
        finally:
            record_page.cells[record_key] = old_records
        assert CorrelationTable.build(module).findings(module) == []
