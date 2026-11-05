"""A booked runtime prefix Slice keeps its source storage and true count."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("rank, late_zero", [(2, False), (3, True)])
def test_runtime_prefix_slice_uses_authored_upper_and_view_count(
    tmp_path, rank, late_zero,
):
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {
                "function": "root", "parameter": "membership",
                "storage": "span", "dtype": "int64", "rank": rank,
                "python_type":
                "src.common.tensors.abstraction.AbstractTensor",
            },
            {
                "function": "root", "parameter": "selected_count",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
        ],
        "records": {}, "bindings": [],
    })
    source = """def root(membership: AbstractTensor, selected_count: int):
    selected = membership[:selected_count]
    return selected.sum().item()
"""
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="native_runtime_prefix_slice_view",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    parameter_ids = dict(root.metadata["parameter_names"])
    assert {argument.id for argument in root.args
            if argument.dtype != "ptr"} == set(parameter_ids.values())
    assert all(argument.dtype != "opaque_ref" for argument in root.args)
    region = next(function for function in module.functions.values()
                  if function.name.endswith("__planned_region_0"))
    result_id, = region.metadata["source_region_integral"][
        "output_value_ids"
    ]
    descriptor = module.tensor_tables[region.name].by_id(result_id)
    assert descriptor.storage == "view"
    assert descriptor.alias_of == parameter_ids["membership"]
    assert descriptor.data_value_id == parameter_ids["membership"]
    assert not any(
        instruction.op == "Call" and "index_select" in str(
            instruction.attributes.get("callee", ""))
        for block in region.blocks.values()
        for instruction in block.instrs
    )

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    output_id = next(
        instruction.args[0].id
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and instruction.args
    )
    artifact.compile(tmp_path / "native_runtime_prefix_slice_view")
    # The late zero axis carries no data, but its large middle dimension
    # catches an overflowing intermediate count before multiplication by zero.
    membership = (np.empty((4, 1 << 30, 0), dtype=np.int64)
                  if late_zero else np.arange(12, dtype=np.int64).reshape(4, 3))
    extreme_bounds = (np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    for upper in ((3, *extreme_bounds) if late_zero else
                  (0, 1, 2, -1, -3, -9, 4, 9, *extreme_bounds)):
        feeds = {
            parameter_ids["membership"]: membership,
            parameter_ids["selected_count"]:
                np.asarray(upper, dtype=np.int64),
        }
        actual = artifact.prepare_execution(feeds).run().buffers[output_id]
        assert int(np.asarray(actual).item()) == int(
            membership[:upper].sum()
        )
