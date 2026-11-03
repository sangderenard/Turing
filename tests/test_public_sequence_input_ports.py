"""Public bytes input ports exist even when no returned slice uses them."""

import inspect
from pathlib import Path

from src.common.tensors.compression import octet_lz
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SEQUENCE_CONTRACT_CONCORDANCE, SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_storage_requirements import (
    public_sequence_capacity_sources, public_sequence_input_ports,
)


def test_actual_read_header_public_bytes_input_has_exact_native_ports():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "_read_header", "parameter": "packet",
             "storage": "span", "dtype": "int64", "rank": 1,
             "python_type": "builtins.bytes"},
            {"function": "_read_header", "parameter": "count",
             "storage": "scalar", "dtype": "int64", "rank": 0,
             "shape": [], "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        inspect.getsource(octet_lz._read_header), "_read_header",
        extraction_contract=contract,
        python_bindings=dict(vars(octet_lz)),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function_name = "_read_header___read_header"
    assert public_sequence_capacity_sources(module, function_name) == ()
    ports = public_sequence_input_ports(module, function_name)
    assert len(ports) == 1
    port = ports[0]
    function = module.functions[function_name]
    args = {value.id: value for value in function.args}
    assert {port.data_id, port.length_id, port.capacity_id} <= args.keys()
    table = module.sequence_tables[function_name]
    descriptor = table.sequences[port.data_id]
    assert port.descriptor_cell.row == (table.owner, port.data_id)
    assert descriptor.column_value_ids == (port.data_id,)
    assert descriptor.length_address_id == port.length_id
    assert descriptor.capacity_value_id == port.capacity_id
    assert port.contract_cell.page == SEQUENCE_CONTRACT_CONCORDANCE
    book = table.book
    assert port.contract_cell in {
        parent for parent, _stage in book.edges_into(port.descriptor_cell)
    }
    assert port.source_identity_cell in {
        parent for parent, _stage in book.edges_into(port.source_class_cell)
    }
    assert book.page(SOURCE_VALUE_CLASS).cell(
        port.source_class_cell.row, port.source_class_cell.column,
    )[0] == "builtins.bytes"

    # A conflicting class on the same authored origin must remove the port.
    page = book.page(SOURCE_VALUE_CLASS)
    key = (port.source_class_cell.row, port.source_class_cell.column)
    original = page.cells[key]
    page.cells[key] = ("builtins.str", *original[1:])
    try:
        assert public_sequence_input_ports(module, function_name) == ()
    finally:
        page.cells[key] = original


def test_public_bytes_length_native_needs_no_host_descriptor_feeds(tmp_path):
    import numpy as np
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "packet", "storage": "span",
                    "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"}],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(packet: bytes) -> int:\n    return len(packet)\n", "root",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function = module.functions["root__root"]
    assert public_sequence_capacity_sources(module, function.name) == ()
    ports, = public_sequence_input_ports(module, function.name)
    artifact = emit_ssa_module_to_c(module, function.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native_input_length")
    returned, = next(instruction.args for block in function.blocks.values()
                     for instruction in block.instrs if instruction.op == "Ret")
    for length in (0, 4, 256):
        execution = artifact.prepare_execution({
            ports.data_id: np.arange(length, dtype=np.int64),
        }).run()
        assert execution.buffers[returned.id].item() == length
        assert execution.buffers[ports.length_id].item() == length
        assert execution.buffers[ports.capacity_id].item() == length
