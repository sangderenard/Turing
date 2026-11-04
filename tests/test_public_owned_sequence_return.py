"""A returned bytes concat keeps its exact native owner across the boundary."""

from dataclasses import replace
from copy import deepcopy
import hashlib
from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    ArtifactFact, ArtifactPart, Backend, EMISSION_ARTIFACT,
    PROCESS_GRAPH_OPERANDS,
    PUBLIC_OWNED_SEQUENCE_RETURN, SOURCE_OWNED_SEQUENCE_RETURN,
    SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import SSASequenceCapacityPolicy


def test_authored_bytes_concat_return_has_one_owned_native_descriptor(tmp_path, monkeypatch):
    # This is the exact four-operand bytes Add shape in compress_octets'
    # return, with the called fragments supplied through its public ABI.
    source = """def compress_tail(header: bytes, encoded_length: bytes,
                      coded_commands: bytes, coded_literals: bytes) -> bytes:
    return header + encoded_length + coded_commands + coded_literals
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "compress_tail", "parameter": parameter,
             "storage": "span", "dtype": "int64", "rank": 1,
             "python_type": "builtins.bytes"}
            for parameter in (
                "header", "encoded_length", "coded_commands", "coded_literals",
            )
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "compress_tail", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function_name = "compress_tail__compress_tail"
    assert CorrelationTable._source_owned_sequence_return_findings(module) == []
    assert CorrelationTable._source_bytes_concat_class_findings(module) == []
    concat_rows = [
        row for row in module.metadata["identity_book"].page(
            SOURCE_VALUE_CLASS
        ).rows()
        if module.metadata["identity_book"].page(SOURCE_VALUE_CLASS).cell(
            row, module.metadata["identity_book"].latest_ref(
                SOURCE_VALUE_CLASS, row,
            ).column,
        )[2] == "source bytes concatenation"
    ]
    assert len(concat_rows) == 6  # Three source Adds and their exact class relabels.
    assert CorrelationTable._public_owned_sequence_return_findings(module) == []
    (ports,) = public_owned_sequence_returns(module, function_name)
    assert ports.data_id == 6
    assert ports.element_dtype == "int64"
    table = module.sequence_tables[function_name]
    descriptor = table.sequences[ports.data_id]
    assert descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC
    assert descriptor.heap_owner_value_id == ports.owner_id
    assert descriptor.length_address_id == ports.length_id
    assert descriptor.capacity_value_id == ports.capacity_id

    book = table.book
    class_page = book.page(SOURCE_VALUE_CLASS)
    source_concat = next(
        book.latest_ref(SOURCE_VALUE_CLASS, row)
        for row in concat_rows
        if any(parent.page == PROCESS_GRAPH_OPERANDS
               for parent, _ in book.edges_into(
                   book.latest_ref(SOURCE_VALUE_CLASS, row)
               ))
    )
    class_key = (source_concat.row, source_concat.column)
    original_class = class_page.cells[class_key]
    class_page.cells[class_key] = (
        "builtins.int", 1, "source bytes concatenation",
    )
    try:
        assert CorrelationTable._source_bytes_concat_class_findings(module)
    finally:
        class_page.cells[class_key] = original_class

    source_page = book.page(SOURCE_OWNED_SEQUENCE_RETURN)
    source = source_page.cell(ports.source_cell.row, ports.source_cell.column)
    source_key = (ports.source_cell.row, ports.source_cell.column)
    source_page.cells[source_key] = replace(
        source, class_cells=source.class_cells[:-1],
    )
    try:
        assert CorrelationTable._source_owned_sequence_return_findings(module)
        assert public_owned_sequence_returns(module, function_name) == ()
    finally:
        source_page.cells[source_key] = source

    public_page = book.page(PUBLIC_OWNED_SEQUENCE_RETURN)
    public_cell = book.latest_ref(
        PUBLIC_OWNED_SEQUENCE_RETURN, (function_name, ports.data_id),
    )
    public = public_page.cell(public_cell.row, public_cell.column)
    public_key = (public_cell.row, public_cell.column)
    public_page.cells[public_key] = replace(
        public, owner_cell=ports.descriptor_cell,
    )
    try:
        assert CorrelationTable._public_owned_sequence_return_findings(module)
        assert public_owned_sequence_returns(module, function_name) == ()
    finally:
        public_page.cells[public_key] = public

    artifact = emit_ssa_module_to_c(module, function_name)
    assert artifact.complete, artifact.shortfalls
    assert artifact.output_heap_sequences == ((
        ports.data_id, ports.length_id, ports.capacity_id,
        ports.owner_id, ports.element_dtype,
    ),)
    assert CorrelationTable._runtime_result_abi_findings(module) == []
    runtime_page = book.page(EMISSION_ARTIFACT)
    runtime_cell = book.latest_ref(EMISSION_ARTIFACT, (
        function_name, Backend.C_MODULE, ArtifactPart.RUNTIME_ABI,
    ))
    original = runtime_page.cell(runtime_cell.row, runtime_cell.column)
    forged = deepcopy(original.location[0])
    owned = next(buffer["owned_sequence"] for buffer in forged["buffers"]
                 if "owned_sequence" in buffer)
    owned["owner_buffer"] = owned["length_buffer"]
    payload = repr(forged).encode("utf-8")
    runtime_key = (runtime_cell.row, runtime_cell.column)
    runtime_page.cells[runtime_key] = ArtifactFact(
        hashlib.sha256(payload).hexdigest(), len(payload), (forged,),
    )
    try:
        assert CorrelationTable._runtime_result_abi_findings(module)
    finally:
        runtime_page.cells[runtime_key] = original

    import numpy as np
    import pytest
    from src.compiler import runtime_abi

    assert artifact.runtime_abi()["schema"] == "turing-runtime-abi-v3"
    artifact.compile(tmp_path / "native_owned_tail")
    parameters = dict(module.functions[function_name].metadata["parameter_names"])
    for fragments in (
        (b"", b"", b"", b""),
        (b"\x00\xff", b"\x80", b"abc", b"\x7f"),
        (bytes(range(256)), b"header" * 17, bytes(range(255, -1, -1)), b"z" * 4097),
    ):
        execution = artifact.prepare_execution({
            parameters[name]: np.frombuffer(fragment, dtype=np.uint8)
            for name, fragment in zip(
                ("header", "encoded_length", "coded_commands", "coded_literals"),
                fragments, strict=True,
            )
        })
        previous = None
        for _ in range(3):
            execution.run()
            result = execution.buffers[ports.data_id]
            assert bytes(result.tolist()) == b"".join(fragments)
            assert execution.buffers[ports.length_id].item() == sum(map(len, fragments))
            assert execution._owned_result_cells[ports.owner_id].item() == 0
            assert execution._owned_result_cells[ports.data_id].item() == 0
            if previous is not None:
                assert bytes(previous.tolist()) == b"".join(fragments)
            previous = result
        # A native consumer may release an already-released owner safely.
        artifact.release_owned_results(execution._pointers)

    def refuse_copy(*_args):
        raise ValueError("deliberate materialization failure")

    with monkeypatch.context() as patch:
        patch.setattr(runtime_abi, "copy_owned_sequence_results", refuse_copy)
        with pytest.raises(ValueError, match="deliberate materialization failure"):
            execution.run()
    assert execution._owned_result_cells[ports.owner_id].item() == 0
    execution.run()
    assert bytes(execution.buffers[ports.data_id].tolist()) == b"".join(fragments)
    feeds = {
        parameters[name]: np.frombuffer(fragment, dtype=np.uint8)
        for name, fragment in zip(
            ("header", "encoded_length", "coded_commands", "coded_literals"),
            fragments, strict=True,
        )
    }
    allocated = artifact.prepare_execution(feeds).buffers
    standalone = artifact.compile_standalone(tmp_path / "owned_tail_executable", feeds)
    completed = standalone.run(frames=3, warmup=1)
    assert "frames=3" in completed.stdout
    raw = standalone.final_outputs_path.read_bytes()
    offset, materialized = 0, None
    for value_id in artifact.buffer_order:
        if value_id == ports.data_id:
            count = int(np.frombuffer(raw, dtype=np.uint64, count=1, offset=offset)[0])
            offset += 8
            materialized = np.frombuffer(raw, dtype=ports.element_dtype, count=count, offset=offset)
            offset += materialized.nbytes
        else:
            offset += allocated[value_id].nbytes
    assert offset == len(raw)
    assert bytes(materialized.tolist()) == b"".join(fragments)

    # Exercise the codec's actual packet boundary. Only its final authored
    # concatenation is native here; the encoder stages remain the Python
    # reference until whole compress_octets lowering is verified.
    from src.common.tensors.compression import octet_codec, octet_huffman, octet_lz
    for data in (b"", b"a", b"abcabcabcabcx", bytes(range(256))):
        header, commands, literals = octet_codec._split(octet_lz.lz_encode(data))
        coded_commands = octet_huffman.encode_octets(commands)
        fragments = (header, octet_lz._header(len(coded_commands)),
                     coded_commands, octet_huffman.encode_octets(literals))
        native = artifact.prepare_execution({
            parameters[name]: np.frombuffer(fragment, dtype=np.uint8)
            for name, fragment in zip(
                ("header", "encoded_length", "coded_commands", "coded_literals"),
                fragments, strict=True,
            )
        }).run()
        packet = bytes(native.buffers[ports.data_id].tolist())
        assert packet == octet_codec.compress_octets(data)
        assert octet_codec.decompress_octets(packet) == data
