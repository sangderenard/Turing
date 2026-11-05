"""Runtime scatter count follows the input span and all update values."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.precompile_to_ssa import SSALoweringIncompleteError
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import posted_ssa_value_fact


def test_dynamic_int64_scatter_uses_every_update_and_exact_count(tmp_path):
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "octets", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(octets: AbstractTensor):
    counts = AbstractTensor.scatter(
        AbstractTensor.zeros((256,), cls=type(octets)),
        octets, octets + 1, dim=0,
    )
    return counts.sum().item()
"""
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="native_dynamic_scatter_count",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    root = module.functions[exports[0]]
    input_id = dict(root.metadata["parameter_names"])["octets"]
    fact = posted_ssa_value_fact(
        root, int(input_id), book=module.metadata["identity_book"],
    )
    assert (fact.physical_dtype, fact.storage, fact.rank) == (
        "int64", "span", 1,
    )
    port = artifact.buffer_order.index(int(input_id))
    assert (artifact.buffer_dtypes[port], artifact.buffer_ranks[port]) == (
        "int64", 1,
    )
    output_ids = [
        argument.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for argument in instruction.args
    ]
    assert len(output_ids) == 1
    artifact.compile(tmp_path / "native_dynamic_scatter_count")
    for input_values, expected in (
        ((), 0.0), ((7,), 8.0),
        ((3, 3, 1, 255), 266.0),
        ((0, 255, 255, 1), 515.0),
    ):
        feeds = {int(input_id): np.asarray(input_values, dtype=np.int64)}
        actual = artifact.prepare_execution(feeds).run().buffers[output_ids[0]]
        assert float(np.asarray(actual).item()) == expected


def test_independent_dynamic_updates_do_not_claim_selector_count():
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": name, "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        } for name in ("octets", "updates")],
        "records": {}, "bindings": [],
    })
    source = """def root(octets: AbstractTensor, updates: AbstractTensor):
    counts = AbstractTensor.scatter(
        AbstractTensor.zeros((256,), cls=type(octets)),
        octets, updates, dim=0,
    )
    return counts.sum().item()
"""
    with pytest.raises(
        SSALoweringIncompleteError,
        match="scatter-add needs a declared destination shape and index extent",
    ):
        lower_ast_source_to_ssa(
            source, "root", name="independent_dynamic_scatter_count",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
