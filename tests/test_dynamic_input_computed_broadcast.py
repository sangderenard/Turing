"""Original span inputs and computed arange spans share exact runtime geometry."""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.concordance_declarations import TENSOR_ELEMENTWISE_EXTENT_BINDING
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("rank", (1, 2, 3))
def test_original_input_and_computed_ordinals_broadcast_natively(tmp_path, rank):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data", "storage": "span",
                    "dtype": "int64", "rank": rank,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, multiplier: int) -> int:\n"
        f"    count = data.shape[{rank - 1}]\n"
        "    index = AbstractTensor.arange(count, cls=type(data))\n"
        "    return (data * multiplier + index).sum().item()\n",
        "root", name=f"input_computed_broadcast_r{rank}",
        extraction_contract=contract, python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"broadcast_r{rank}")
    function = module.functions[exports[0]]
    inputs = dict(function.metadata["parameter_names"])
    output, = [value.id for block in function.blocks.values() for instruction in block.instrs
               if instruction.op == "Ret" for value in instruction.args]
    shape = tuple(range(2, rank + 2))
    shapes = (shape, *(shape[:axis] + (0,) + shape[axis + 1:] for axis in range(rank)))
    if rank == 3:
        # Empty geometry must not overflow its count on preceding axes.
        shapes += ((65536, 65536, 0),)
    for actual_shape in shapes:
        data = np.arange(np.prod(actual_shape), dtype=np.int64).reshape(actual_shape)
        if data.size:
            data.flat[0] = 2**53 + 1
        expected = int((data * 2 + np.arange(actual_shape[-1], dtype=np.int64)).sum())
        execution = artifact.prepare_execution({
            int(inputs["data"]): data,
            int(inputs["multiplier"]): np.asarray(2, dtype=np.int64),
        }).run()
        actual = np.asarray(execution.buffers[output])
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == expected
    book = identity_book(module)
    bindings = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    row = next(row for row in bindings.rows()
               if bindings.latest(row)[0] == "mul")
    fact = bindings.latest(row)
    function = module.functions[row[1]]
    extent = next(instruction for block in function.blocks.values()
                  for instruction in block.instrs
                  if instruction.res is not None and int(instruction.res.id) == fact[5])
    result = next(instruction.res for block in function.blocks.values()
                  for instruction in block.instrs
                  if instruction.res is not None and int(instruction.res.id) == row[2])
    original = extent.args
    extent.args = [result]
    try:
        assert CorrelationTable._tensor_elementwise_extent_binding_findings(module)
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    finally:
        extent.args = original


@pytest.mark.parametrize("expression", ("data * multiplier", "multiplier - data"))
def test_declared_float_span_and_runtime_scalar_keep_native_geometry(tmp_path, expression):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data", "storage": "span",
                    "dtype": "float64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, multiplier: float) -> float:\n"
        f"    return ({expression}).sum().item()\n",
        "root", name="input_scalar_float_geometry",
        extraction_contract=contract, python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "scalar_float")
    function = module.functions[exports[0]]
    inputs = dict(function.metadata["parameter_names"])
    output, = [value.id for block in function.blocks.values() for instruction in block.instrs
               if instruction.op == "Ret" for value in instruction.args]
    for values in ([], [0.5, -1.25, 4.0]):
        data = np.asarray(values, dtype=np.float64)
        execution = artifact.prepare_execution({
            int(inputs["data"]): data,
            int(inputs["multiplier"]): np.asarray(2.0, dtype=np.float64),
        }).run()
        expected = (data * 2.0 if expression == "data * multiplier" else 2.0 - data).sum()
        np.testing.assert_array_equal(execution.buffers[output], expected)
