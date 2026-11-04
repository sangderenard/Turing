"""An optional scalar uses its booked payload and tag through the guard."""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import VALUE_TYPE_ABI
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("expression_form", (False, True), ids=("if", "ifexp"))
def test_optional_count_guard_uses_tag_and_scalar_payload(tmp_path, expression_form):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "octets", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0,
             "type_expectations": ["builtins.int", "builtins.NoneType"]},
        ],
        "records": {}, "bindings": [],
    })
    statement_source = """def root(octets: AbstractTensor, count: int | None = None) -> int:
    if count is None:
        limit = octets.shape[0]
    else:
        limit = int(count)
    return limit + octets.sum().item()
"""
    expression_source = """def root(octets: AbstractTensor, count: int | None = None) -> int:
    limit = octets.shape[0] if count is None else int(count)
    return limit + octets.sum().item()
"""
    source = expression_source if expression_form else statement_source
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="optional_scalar_cast_abi",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    inputs = dict(root.metadata["parameter_names"])
    count_id = int(inputs["count"])
    tag_id = int(dict(root.metadata["native_value_type_tags"])[count_id])
    book = identity_book(module)
    abis = tuple(
        book.page(VALUE_TYPE_ABI).latest(row)
        for row in book.page(VALUE_TYPE_ABI).rows()
        if row[-1] == count_id and "root" in str(row[0])
    )
    assert any(
        tuple(arm.identity for arm in abi.physical_alternatives)
        == ("builtins.NoneType", "builtins.int")
        for abi in abis
    )

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "optional_scalar_cast_abi")
    return_ids = tuple(int(value.id)
                       for block in root.blocks.values()
                       for instruction in block.instrs
                       if instruction.op == "Ret"
                       and not instruction.attributes.get("fault_exit")
                       for value in instruction.args)
    assert len(return_ids) == 1
    for payload, tag, expected in ((7, 1, 13), (0, 1, 6),
                                   (12345, 0, 9)):
        feeds = {
            int(inputs["octets"]): np.asarray([1, 2, 3], dtype=np.int64),
            count_id: np.asarray([payload], dtype=np.int64),
            tag_id: np.asarray([tag], dtype=np.int64),
        }
        result = artifact.prepare_execution(feeds).run().buffers
        assert int(result[return_ids[0]].item()) == expected
