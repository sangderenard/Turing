"""A dynamic loop receiver keeps its authored shape-read binding."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.precompile_to_ssa import _ControlSSABuilder


def test_dynamic_shape_read_keeps_loop_carried_lexical_binding(monkeypatch):
    class AtShapeRead(BaseException):
        pass

    observed = []
    original = _ControlSSABuilder._resolve_read

    def inspect(self, value_id, bindings, reader):
        if reader.startswith("shape dimension") and self.enclosing_loop_states:
            observed.append((value_id, set(bindings), tuple(self.enclosing_loop_states)))
            raise AtShapeRead
        return original(self, value_id, bindings, reader)

    monkeypatch.setattr(_ControlSSABuilder, "_resolve_read", inspect)
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "iterations", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    source = """def root(values: AbstractTensor, iterations: int) -> int:
    current_weights = values
    for _ in range(1, iterations):
        pair_count = current_weights.shape[0] // 2
        current_weights = current_weights[:pair_count]
    return current_weights.shape[0]
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="loop_shape_read_probe",
            python_bindings={"AbstractTensor": AbstractTensor},
            extraction_contract=contract,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtShapeRead:
        pass
    assert len(observed) == 1
    assert observed[0][1] == {"current_weights"}
    assert len(observed[0][2]) == 1
