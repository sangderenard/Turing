"""A callsite byte class types list(bytes) before hierarchy reads it."""

import inspect
from pathlib import Path

from src.common.tensors.compression import octet_lz
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SEQUENCE_ROW_DTYPE, SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_real_tensor_list_literal_and_bytes_have_distinct_row_proofs():
    source = inspect.getsource(octet_lz._tensor) + (
        "\ndef root(packet: bytes, dictionary: bytes):\n"
        "    a = _tensor([0])\n"
        "    b = _tensor(packet + dictionary)\n"
        "    return a, b\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"}
            for name in ("packet", "dictionary")
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="real_tensor_bytes_materializer",
        source_module=octet_lz.__name__, python_bindings=vars(octet_lz),
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    rows = {
        book.page(SEQUENCE_ROW_DTYPE).latest(row)
        for row in book.page(SEQUENCE_ROW_DTYPE).rows()
        if "_tensor" in str(row)
    }
    assert rows.issuperset({
        (("int64",), "builtin iterable materialization"),
        (("int64",), "specialized literal materialization"),
    })
    assert any(
        row[0][0] == (octet_lz.__name__, "_tensor")
        and row[0][1] == "specialization"
        and row[1] == 0
        and book.page(SOURCE_VALUE_CLASS).latest(row)[0] == "builtins.bytes"
        for row in book.page(SOURCE_VALUE_CLASS).rows()
        if isinstance(row[0], tuple) and len(row[0]) == 3
    )
    assert CorrelationTable.build(module).findings(module) == []

