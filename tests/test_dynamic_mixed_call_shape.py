"""A runtime Mul's display shape cannot certify a fixed consumer extent."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
import src.compiler.tensor_ssa_lowering as tensor_ssa_lowering


def test_runtime_product_and_static_weights_keep_distinct_call_shapes(monkeypatch):
    class AtMetadataBoundary(Exception):
        pass

    observed = []

    def inspect_before_metadata(module, *, finalize_private_owners=True):
        regions = tuple(
            fn for fn in module.functions.values()
            if any(
                ins.op == "Call"
                and ins.attributes.get("callee") == "reduce_dim_double"
                for block in fn.blocks.values() for ins in block.instrs
            )
        )
        assert len(regions) == 1
        consumer = regions[0]
        source = consumer.args[0]
        table = module.tensor_tables[consumer.name]
        dynamic_input = table.by_id(int(source.id))
        assert dynamic_input is not None
        assert dynamic_input.metadata_state == "unresolved"
        assert dynamic_input.byte_size is None
        assert tuple(source.shape) == (1, 8)  # display shape, not full extent
        book = identity_book(module)
        assert not any(
            row[0] == consumer.name and row[1] == int(source.id)
            for row in book.page("ssa_call_shape").rows()
        )
        observed.append(True)
        raise AtMetadataBoundary

    monkeypatch.setattr(
        tensor_ssa_lowering,
        "publish_computed_region_result_metadata",
        inspect_before_metadata,
    )
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
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="dynamic_mixed_call_shape_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtMetadataBoundary:
        pass
    assert observed == [True]
