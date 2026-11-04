"""A private mixed broadcast returns an exact scalar without escaping storage."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_HEAP_CONSUMER_NONESCAPE,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import function_scope_of


@pytest.fixture(scope="module")
def private_broadcast():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in ("left", "right")
        ],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(left: AbstractTensor, right: AbstractTensor):\n"
        "    a = left.unsqueeze(1)\n"
        "    b = right.unsqueeze(0)\n"
        "    return (a + b).sum().item()\n",
        "root", name="private_mixed_broadcast",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)
    return module, exports[0]


def test_private_mixed_broadcast_native_exact_integer_result(private_broadcast, tmp_path):
    module, root = private_broadcast
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "private_broadcast")
    parameters = dict(module.functions[root].metadata["parameter_names"])
    for left, right in (([1, 2], [3, 4, 5]),
                        ([2**53 + 1, 2**53 + 2], [-7, 3, 9]),
                        ([], [1, 2]), ([1, 2], [])):
        feeds = {
            parameters["left"]: np.asarray(left, dtype=np.int64),
            parameters["right"]: np.asarray(right, dtype=np.int64),
        }
        expected = sum(a + b for a in left for b in right)
        execution = artifact.prepare_execution(feeds).run()
        outputs = [value for key, value in execution.buffers.items() if key not in feeds]
        assert len(outputs) == 1
        assert outputs[0].dtype == np.dtype("int64")
        assert outputs[0].item() == expected


def test_scalar_nonescape_requires_posted_numeric_scalar_storage(private_broadcast):
    module, _ = private_broadcast
    book = identity_book(module)
    proofs = book.page(TENSOR_HEAP_CONSUMER_NONESCAPE)
    target, output_id = next(
        (fact[0], logical)
        for row in proofs.rows()
        for fact in (proofs.latest(row),)
        for logical, data, alias in fact[3]
        if logical == data and alias is None
        and module.tensor_tables[fact[0]].by_id(logical) is None
    )
    values = book.page(SSA_VALUE)
    ref = book.latest_ref(SSA_VALUE, (function_scope_of(module.functions[target]), output_id))
    original = values.cell(ref.row, ref.column)
    for changed in (replace(original, storage="span", rank=1),
                    replace(original, physical_dtype="ptr")):
        values.cells[(ref.row, ref.column)] = changed
        try:
            assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
        finally:
            values.cells[(ref.row, ref.column)] = original
