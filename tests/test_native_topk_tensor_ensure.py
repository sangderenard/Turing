"""Top-k indices pass through authored ensure_tensor into a native gather."""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def _assert_family_audit_rejects_other_type_and_member(module):
    from src.compiler.concordance_declarations import (
        NATIVE_TENSOR_FAMILY_SOURCE, NATIVE_TENSOR_RESULT_FAMILY,
        NATIVE_TENSOR_WRAP,
    )
    from src.compiler.native_tensor_family import family_findings

    book = module.metadata["identity_book"]
    page = book.page(NATIVE_TENSOR_RESULT_FAMILY)
    cell = book.latest_ref(NATIVE_TENSOR_RESULT_FAMILY, next(iter(page.rows())))
    key = (cell.row, cell.column)
    fact = page.cells[key]
    declaration = fact[0]
    source_page = book.page(NATIVE_TENSOR_FAMILY_SOURCE)
    source_key = (declaration.row, declaration.column)
    source = source_page.cells[source_key]
    assert family_findings(module) == []
    try:
        source_page.cells[source_key] = (*source[:2], "OtherRawTensor", *source[3:])
        assert any(item.kind == "native-tensor-family" for item in family_findings(module))
        source_page.cells[source_key] = source
        page.cells[key] = (*fact[:6], 0, fact[7])
        assert any(item.kind == "native-tensor-family" for item in family_findings(module))
    finally:
        source_page.cells[source_key] = source
        page.cells[key] = fact
    assert family_findings(module) == []
    from dataclasses import replace
    from src.compiler.ssa_record_return_state import function_scope_of

    wrap_row = next(iter(book.page(NATIVE_TENSOR_WRAP).rows()))
    table = next(table for name, table in module.tensor_tables.items()
                 if function_scope_of(module.functions[name]) == wrap_row[0]
                 and (descriptor := table.by_id(wrap_row[1])) is not None
                 and descriptor.alias_of is not None)
    descriptor = table.by_id(wrap_row[1])
    try:
        table.tensors[wrap_row[1]] = replace(
            descriptor, data_value_id=descriptor.tensor_id,
            alias_of=None, owns_allocation=True,
        )
        assert any(item.kind == "native-tensor-wrap-storage"
                   for item in family_findings(module))
    finally:
        table.tensors[wrap_row[1]] = descriptor
    assert family_findings(module) == []
    try:
        table.tensors[wrap_row[1]] = replace(descriptor, storage="output")
        assert any(item.kind == "native-tensor-wrap-storage"
                   for item in family_findings(module))
    finally:
        table.tensors[wrap_row[1]] = descriptor
    wrap_page = book.page(NATIVE_TENSOR_WRAP)
    wrap_ref = book.latest_ref(NATIVE_TENSOR_WRAP, wrap_row)
    wrap_key = (wrap_ref.row, wrap_ref.column)
    wrapping = wrap_page.cells[wrap_key]
    try:
        wrap_page.cells[wrap_key] = (*wrapping[:4], None)
        assert any(item.kind == "native-tensor-wrap"
                   for item in family_findings(module))
    finally:
        wrap_page.cells[wrap_key] = wrapping
    assert family_findings(module) == []


@pytest.mark.parametrize(
    ("data_values", "cast_weights", "runtime_count"),
    (
        ((1, 2, 3, 4), False, False),
        ((1.9, 2.2, 3.8, 4.7), True, False),
        ((1, 2, 3, 4), False, True),
        ((1.9, 2.2, 3.8, 4.7), True, True),
    ),
    ids=("int64-direct", "float64-to-int64-weights", "runtime-count-negated-priority",
         "runtime-count-cast-weights-negated-priority"),
)
def test_topk_indices_ensure_tensor_gather_native(
    tmp_path, data_values, cast_weights, runtime_count,
):
    data_array = np.asarray(
        data_values, dtype=np.float64 if cast_weights else np.int64,
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "data", "storage": "span",
            "dtype": str(data_array.dtype), "rank": 1, "shape": [4],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    receiver = "weights" if cast_weights else "data"
    source = ("def root(data: AbstractTensor, count: int) -> int:\n"
              if runtime_count else "def root(data: AbstractTensor) -> int:\n")
    if cast_weights:
        source += "    weights = data.to_dtype('int64')\n"
    indent = "    "
    if runtime_count:
        source += "    if 0 < count and count <= 4:\n"
        indent = "        "
    selector = f"-{receiver}" if runtime_count else receiver
    source += (
        f"{indent}raw = AbstractTensor.topk({selector}, k={'count' if runtime_count else 2}, dim=0).indices\n"
        f"{indent}selected = {receiver}.ensure_tensor(raw).to_dtype('int64')\n"
        f"{indent}return {receiver}[selected].sum().item()\n"
    )
    if runtime_count:
        source += "    return 0\n"
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=(
            "native_topk_tensor_ensure_cast_weights" if cast_weights
            else "native_topk_tensor_ensure"
        ),
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    if not cast_weights and not runtime_count:
        _assert_family_audit_rejects_other_type_and_member(module)
    assert not [
        (function.name, block.name, instruction)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and "ensure_tensor" in str(instruction.attributes.get("callee", ""))
    ]
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native_topk_tensor_ensure")
    root = module.functions[exports[0]]
    names = dict(root.metadata["parameter_names"])
    output_ids = [argument.id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret"
                  for argument in instruction.args]
    assert len(output_ids) == 1
    for count, expected in (((2, 3), (1, 1), (4, 10), (0, 0))
                            if runtime_count else ((None, 7),)):
        feeds = {int(names["data"]): data_array}
        if runtime_count:
            feeds[int(names["count"])] = np.asarray(count, dtype=np.int64)
        execution = artifact.prepare_execution(feeds).run()
        actual = np.asarray(execution.buffers[output_ids[0]])
        if not cast_weights:
            assert actual.dtype == np.dtype(np.int64)
        assert actual.item() == expected
