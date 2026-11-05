"""A checked runtime reshape carries its source's physical span to a cast."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_RUNTIME_RESHAPE_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import function_scope_of


def test_runtime_reshape_preserves_int64_span_before_typed_cast(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor):\n"
        "    shaped = values.reshape(-1, 8)\n"
        "    return shaped.to_dtype('float64').sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="runtime_reshape_physical_cast_native",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_RESHAPE_VALIDITY)
    rows = tuple(page.rows())
    assert len(rows) == 1
    row = rows[0]
    function = module.functions[row[1]]
    scope = function_scope_of(function)
    source_ref = book.latest_ref(SSA_VALUE, (scope, page.latest(row)[0]))
    result_ref = book.latest_ref(SSA_VALUE, (scope, row[2]))
    validity_ref = book.latest_ref(TENSOR_RUNTIME_RESHAPE_VALIDITY, row)
    assert source_ref is not None and result_ref is not None
    assert validity_ref is not None
    result_fact = book.page(SSA_VALUE).cell(result_ref.row, result_ref.column)
    assert (
        result_fact.dtype, result_fact.physical_dtype,
        result_fact.storage, result_fact.rank,
    ) == ("int64", "int64", "span", 2)
    assert {source_ref, validity_ref} <= {
        cause for cause, _ in book.edges_into(result_ref)
    }

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_reshape_physical_cast")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for values in (
        np.empty(0, dtype=np.int64),
        np.arange(8, dtype=np.int64),
        np.arange(16, dtype=np.int64),
    ):
        expected = values.reshape(-1, 8).astype(np.float64).sum()
        actual = np.asarray(
            artifact.prepare_execution({parameter: values}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("float64")
        assert actual.item() == expected.item()
