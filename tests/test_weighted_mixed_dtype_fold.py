"""A runtime tensor's empty extent tuple does not make it a scalar."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


def test_weighted_runtime_multiply_fold_retains_float_dtype_and_dynamic_extents(
    monkeypatch,
):
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128,64,32,16,8,4,2,1))
    shaped = bits.reshape(-1,8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    observed = []
    original = strategy._fold_callsite_structural_values

    def traced(graph, **kwargs):
        result = original(graph, **kwargs)
        for node_id, data in graph.G.nodes(data=True):
            if not (
                str(data.get("type") or data.get("op") or "").casefold()
                in {"mul", "mult"}
                and (data.get("source_span") or {}).get("line") == 4
            ):
                continue
            descriptor = dict(data.get("tensor") or {})
            if descriptor:
                observed.append((node_id, descriptor))
        return result

    monkeypatch.setattr(strategy, "_fold_callsite_structural_values", traced)
    lower_ast_source_to_ssa(
        source, "root", name="weighted_mixed_dtype_fold",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None, progress_bars=False,
    )
    assert observed
    assert all(
        descriptor.get("dtype") == "float64"
        and descriptor.get("shape") == ()
        and descriptor.get("rank") == 2
        and descriptor.get("metadata_state") == "dynamic"
        for _node_id, descriptor in observed
    ), observed
