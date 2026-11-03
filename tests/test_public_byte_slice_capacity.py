"""An authored bytes slice keeps a source-proved public capacity."""

from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import public_sequence_capacity_sources


SOURCE = """def root(packet: bytes, at: int, command_bytes: int) -> tuple[bytes, bytes, bytes]:
    return packet[:at], packet[at:at + command_bytes], packet[at + command_bytes:]
"""


def test_public_bytes_slice_capacity_tracks_exact_packet_and_outputs(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "packet", "storage": "span",
             "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"},
            *(
                {"function": "root", "parameter": name,
                 "storage": "scalar", "dtype": "int64", "rank": 0,
                 "shape": [], "python_type": "builtins.int"}
                for name in ("at", "command_bytes")
            ),
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, "root", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function = module.functions["root__root"]
    relations = public_sequence_capacity_sources(module, function.name)
    assert len(relations) == 3
    assert {item.input_sequence_id for item in relations} == {2}
    assert {item.output_sequence_id for item in relations} == {4, 7, 10}
    assert len({item.output_length_id for item in relations}) == 3
    assert len({item.output_capacity_id for item in relations}) == 3
    assert all(item.input_length_id != item.output_length_id for item in relations)
    assert CorrelationTable._source_sequence_capacity_bound_findings(module) == []

    arguments = {int(argument.id): argument for argument in function.args}
    for item in relations:
        for member_id in (
            item.output_data_id, item.output_length_id,
            item.output_capacity_id,
        ):
            accounting = arguments[member_id].accounting or {}
            assert accounting["caller_provided_workspace"] is True
            assert accounting["source_sequence_capacity_bound"] == item.bound_cell
        assert arguments[item.input_data_id].accounting[
            "program_abi_parameter"
        ] == "packet"
    artifact = emit_ssa_module_to_c(module, function.name)
    assert artifact.complete, artifact.shortfalls
    import numpy as np
    from src.compiler.concordance_declarations import (
        ArtifactPart, Backend, EMISSION_ARTIFACT,
    )
    from src.compiler.identity_concordance import ConcordanceRefusal
    import pytest

    abi = artifact.runtime_abi()
    assert abi["schema"] == "turing-runtime-abi-v2"
    receipt = artifact.emission.book.latest_ref(
        EMISSION_ARTIFACT, (artifact.name, Backend.C_MODULE, ArtifactPart.RUNTIME_ABI),
    )
    causes = {parent for parent, _stage in artifact.emission.book.edges_into(receipt)}
    assert all(item.bound_cell in causes for item in relations)
    artifact.compile(tmp_path / "byte_slices")
    parameters = dict(function.metadata["parameter_names"])
    namespace = {}
    exec(SOURCE, namespace)
    for packet, at, count in (
        (b"", 0, 0), (bytes((0, 127, 128, 255)), 1, 2),
        (bytes(range(32)), 3, 10), (bytes(range(9)), 99, 5),
        (bytes(range(9)), -4, 2),
    ):
        execution = artifact.prepare_execution({
            parameters["packet"]: np.frombuffer(packet, dtype=np.uint8),
            parameters["at"]: at, parameters["command_bytes"]: count,
        })
        expected = namespace["root"](packet, at, count)
        for _ in range(2):
            execution.run()
            assert tuple(bytes(execution.buffers[item.output_data_id].tolist())
                         for item in relations) == expected
    original_lengths = artifact.output_length_sources
    artifact.output_length_sources = (
        (original_lengths[0][0], original_lengths[1][1]),
        (original_lengths[1][0], original_lengths[0][1]),
        *original_lengths[2:],
    )
    try:
        with pytest.raises(ConcordanceRefusal, match="emission concordance"):
            artifact.runtime_abi()
    finally:
        artifact.output_length_sources = original_lengths

    # Metadata is a forwarding index only. A coincident numeric source ID
    # without the source/descriptor concordance join yields no capacity.
    original = function.metadata["sequence_capacity_sources"]
    function.metadata["sequence_capacity_sources"] = ((4, 1),)
    try:
        assert public_sequence_capacity_sources(module, function.name) == ()
    finally:
        function.metadata["sequence_capacity_sources"] = original
