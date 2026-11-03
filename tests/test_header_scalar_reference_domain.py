"""A tensor reference must preserve the integer widening in `_header`."""

import inspect
from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.compression import octet_lz
from src.compiler.concordance_declarations import SSA_VALUE
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_record_return_state import function_scope_of


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts/program_extraction.yaml"
)


def test_header_integer_cast_keeps_native_scalar_operator_with_c_reference():
    reference = c_backend_repository_ssa_reference()
    assert reference.operation("shr") is not None
    assert reference.operation("bitand") is not None
    source = (
        inspect.getsource(octet_lz._header)
        + "\ndef root(x: int) -> bytes:\n    return _header(x)\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="header_scalar_reference_domain",
        extraction_contract=CONTRACT, tensor_ssa_reference=reference,
        progress=lambda _message: None, progress_bars=False,
    )
    book = identity_book(module)
    for suffix, opcode, source_operator in (
        ("__planned_region_0", "Shr", "shr"),
        ("__planned_region_1", "BitAnd", "bitand"),
    ):
        region = next(
            function for name, function in module.functions.items()
            if "_header__specialized_" in name and name.endswith(suffix)
        )
        instructions = tuple(
            instruction for block in region.blocks.values()
            for instruction in block.instrs
        )
        conversion = next(
            instruction for instruction in instructions
            if instruction.op == "Cast"
            and instruction.attributes.get("source_operator")
            == source_operator
        )
        operation = next(
            instruction for instruction in instructions
            if instruction.op == opcode
        )
        assert operation.args[0].id == conversion.res.id
        assert operation.res.dtype == "int64"
        assert operation.res.shape == ()
        cell = book.latest_ref(
            SSA_VALUE, (function_scope_of(region), int(conversion.res.id)),
        )
        assert cell is not None
        fact = book.page(SSA_VALUE).cell(cell.row, cell.column)
        assert fact.dtype == "int64"
        assert fact.physical_dtype != "float64"
    assert CorrelationTable.build(module).findings(module) == []
