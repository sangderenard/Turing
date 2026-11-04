"""A sequence carrier's cell shape is not a tensor Phi's logical extent."""

from pathlib import Path

import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import SSA_BLOCK, SSA_VALUE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, PROVEN_SHAPE_PAGE, begin_identity_book,
    end_identity_book, identity_book, record_proven_shape, shape_scope_of,
)
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.tensor_ssa_lowering import (
    settle_shape_preserving_value_metadata,
)
from src.transmogrifier.ssa import (
    BasicBlock, Function, Instr, IRModule, SSAValue,
    SSATensorDescriptor, SSATensorTable,
)
from tests.compiler_provenance_inputs import publish_authored_ssa


CONTRACT = Path(__file__).resolve().parents[1] / (
    "extraction_contracts/program_extraction.yaml"
)


def test_bytes_return_phi_does_not_take_ptr_cell_shape_as_byte_extent():
    source = ("def root(data: bytes) -> bytes:\n"
              "    data = bytes(data)\n"
              "    header = bytearray((2,))\n"
              "    if not data:\n"
              "        return bytes(header)\n"
              "    return bytes(header) + bytes(data)\n")
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "values": [{"function": "root", "parameter": "data",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "builtins.bytes"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="bytes_return_phi_shape",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    sequence = module.sequence_tables[function.name]
    assert sequence.by_id(12) is not None
    assert not any(
        row[1] == 13
        for row in identity_book(module).page(PROVEN_SHAPE_PAGE).rows()
    )
    assert CorrelationTable.build(module).findings(module) == []


@pytest.mark.parametrize("closed", (True, False))
def test_tensor_phi_requires_both_arm_descriptors_and_control(closed):
    left = SSAValue(0, "float64", (2,))
    right = SSAValue(1, "float64", (2,))
    merged = SSAValue(2, "float64")
    function = Function("closed_tensor_phi", [left, right], {
        "entry": BasicBlock("entry", []),
        "left": BasicBlock("left", []),
        "right": BasicBlock("right", []),
        "join": BasicBlock("join", [
            Instr("Phi", [left, right], merged,
                  attributes={"incoming_blocks": ("left", "right")}),
            Instr("Ret", [merged], None),
        ]),
    })
    table = SSATensorTable()
    for value in ((left, right) if closed else (left,)):
        table.register(SSATensorDescriptor(
            tensor_id=value.id, data_value_id=value.id,
            dtype="float64", shape=(2,), storage="input",
        ))
    module = IRModule({function.name: function},
                      tensor_tables={function.name: table})
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        for value in ((left, right) if closed else (left,)):
            record_proven_shape(
                shape_scope_of(function), value.id, value.shape,
                value.dtype, sources=(book.latest_ref(
                    SSA_VALUE, (function_scope_of(function), value.id),
                ),),
            )
        assert settle_shape_preserving_value_metadata(module) is closed
        assert merged.shape == ((2,) if closed else ())
        shape_rows = tuple(
            candidate for candidate in book.page(PROVEN_SHAPE_PAGE).rows()
            if candidate[1] == 2
        )
        assert len(shape_rows) == (1 if closed else 0)
        if not closed:
            return
        ref = book.latest_ref(PROVEN_SHAPE_PAGE, shape_rows[0])
        causes = {source for source, _ in book.edges_into(ref)}
        for value_id in (0, 1):
            assert book.latest_ref(SSA_VALUE, (
                function_scope_of(function), value_id,
            )) in causes
            assert book.latest_ref(PROVEN_SHAPE_PAGE, (
                shape_scope_of(function), value_id,
            )) in causes
        for label in ("left", "right"):
            assert book.latest_ref(SSA_BLOCK, (
                function_scope_of(function), function.name, label,
            )) in causes
    finally:
        end_identity_book(token)
