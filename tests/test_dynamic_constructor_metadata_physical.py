"""A dynamic constructor's computed metadata has physical SSA proof."""

from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_METADATA_RESULT_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


def test_dynamic_zeros_metadata_crosses_real_region_call(monkeypatch):
    class ObservedMetadata(Exception):
        pass

    original = tensor_ssa_lowering.publish_computed_region_result_metadata

    def inspect(module):
        changed = original(module)
        book = identity_book(module)
        rows = tuple(book.page(TENSOR_METADATA_RESULT_BINDING).rows())
        if not rows:
            return changed
        assert len(rows) == 1
        row = rows[0]
        fact = book.page(TENSOR_METADATA_RESULT_BINDING).latest(row)
        callee = module.functions[row[2]]
        expected = (
            ("int32", "span", 1),
            ("int32", "scalar", 0),
            ("int32", "scalar", 0),
        )
        binding = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, row)
        causes = {source for source, _stage in book.edges_into(binding)}
        for value_id, physical in zip(fact[1], expected, strict=True):
            source = ssa_value_identity_cell(callee, value_id, book=book)
            assert source is not None and source in causes
            recorded = book.page(SSA_VALUE).cell(source.row, source.column)
            assert (recorded.physical_dtype, recorded.storage,
                    recorded.rank) == physical
        raise ObservedMetadata

    monkeypatch.setattr(
        tensor_ssa_lowering, "publish_computed_region_result_metadata", inspect,
    )
    source = """def root(like: AbstractTensor, count: int) -> AbstractTensor:
    return AbstractTensor.zeros((count,), cls=type(like))
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "like", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    with pytest.raises(ObservedMetadata):
        lower_ast_source_to_ssa(
            source, "root", name="dynamic_zeros_metadata_physical",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
