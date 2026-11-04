"""A scalar ABI remains scalar beside a live tensor input."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import posted_ssa_value_fact


def test_planned_int_cast_keeps_scalar_abi_beside_tensor_input(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "octets", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    source = """def root(octets: AbstractTensor, count: int) -> int:
    integer = int(count)
    return integer + octets.sum().item()
"""
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="planned_scalar_cast_abi",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    casts = tuple(
        (function, instruction)
        for function in module.functions.values()
        if "__planned_region_" in function.name
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Cast"
        and instruction.attributes.get("lowered_from") == "int"
    )
    assert len(casts) == 1
    function, cast = casts[0]
    assert cast.attributes["scalar_cast_identity"] is True
    assert cast.res is not None
    fact = posted_ssa_value_fact(
        function, int(cast.res.id), book=identity_book(module),
    )
    assert (fact.dtype, fact.physical_dtype, fact.storage, fact.rank, fact.shape) == (
        "int64", "int64", "scalar", 0, (),
    )
    source_descriptor = module.tensor_tables[function.name].by_id(
        int(cast.args[0].id)
    )
    assert source_descriptor.metadata_state == "static"
    assert source_descriptor.shape == ()

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "planned_scalar_cast_abi")
    inputs = dict(module.functions[exports[0]].metadata["parameter_names"])
    for octets, count in (([1, 2, 3], 7), ([1, 2, 3], -2), ([], 0)):
        feeds = {
            inputs["octets"]: np.asarray(octets, dtype=np.int64),
            inputs["count"]: np.asarray(count, dtype=np.int64),
        }
        buffers = artifact.prepare_execution(feeds).run().buffers
        result = tuple(value for key, value in buffers.items() if key not in feeds)
        assert len(result) == 1
        assert result[0].dtype == np.dtype("int64")
        assert int(result[0].item()) == count + sum(octets)
