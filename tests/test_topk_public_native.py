"""The admitted named top-k pair crosses source, region, SSA, and C."""

from pathlib import Path

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
from src.compiler.ssa_storage_requirements import (
    module_storage_requirements, public_tensor_capacity_sources,
)
from src.compiler.concordance_declarations import TENSOR_OUTPUT_CAPACITY_BOUND


@pytest.mark.parametrize("shape", ((3,), (3, 4)))
@pytest.mark.parametrize("field", ("values", "indices"))
def test_guarded_dynamic_topk_has_concorded_capacity_and_computed_ports(tmp_path, shape, field):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": len(shape), "shape": list(shape),
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(values, k: int):\n"
        "    if k < 0:\n        raise ValueError('negative k')\n"
        "    if k > 3:\n        raise ValueError('large k')\n"
        f"    return AbstractTensor.topk(values, k=k, dim=0).{field}\n",
        "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="topk_dynamic_capacity", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    root_name = "topk_dynamic_capacity__root"
    ports = public_tensor_capacity_sources(module, root_name)
    assert len(ports) == 1
    assert ports[0].input_data_id == 1
    book = module.metadata["identity_book"]
    bound = book.page(TENSOR_OUTPUT_CAPACITY_BOUND).cell(
        ports[0].bound_cell.row, ports[0].bound_cell.column)
    assert bound.capacity_shape == shape
    region = module.functions[root_name + "__planned_region_2"]
    requirements = module_storage_requirements(module)[region.name]
    assert requirements[ports[0].output_data_id].element_count == np.prod(shape)
    assert module.tensor_tables[region.name].tensors[
        ports[0].output_data_id].metadata_state == "dynamic"
    artifact = emit_ssa_module_to_c(module, root_name)
    assert artifact.complete, artifact.shortfalls
    assert CorrelationTable.build(module).findings(module) == []
    artifact.compile(tmp_path / "dynamic_topk")
    assert artifact.runtime_abi()["schema"] == "turing-runtime-abi-v2"
    input_id = ports[0].input_data_id
    output_id = ports[0].output_data_id
    values = np.array([2**53 + 1, 2**53 + 3, 2**53 + 2], dtype=np.int64)
    expected = np.array([1, 2, 0], dtype=np.int64)
    if field == "values":
        expected = values[expected]
    if len(shape) == 2:
        values = np.repeat(values[:, None], shape[1], axis=1)
        expected = np.repeat(expected[:, None], shape[1], axis=1)
    k_id = next(argument.id for argument in module.functions[root_name].args
                if argument.id != input_id)
    for k in (0, 1, 2, 3, 1, 0):
        execution = artifact.prepare_execution({
            input_id: values, k_id: np.array([k], dtype=np.int64),
        }).run()
        assert execution.buffers[output_id].shape == (k, *shape[1:])
        assert np.array_equal(execution.buffers[output_id], expected[:k])
        execution.run()
        assert np.array_equal(execution.buffers[output_id], expected[:k])

    selected = next(instruction for block in region.blocks.values()
                    for instruction in block.instrs
                    if instruction.op == "Min"
                    and instruction.attributes.get("source_operator") == "topk")
    selected.args = [selected.args[1], selected.args[0]]
    assert any(finding.kind == "tensor-output-capacity-bound-disagreement"
               for finding in CorrelationTable.build(module).findings(module))


@pytest.mark.parametrize("field,expected", (
    ("values", [2**53 + 2, 2**53 + 1]),
    ("indices", [2, 1]),
))
def test_public_topk_named_leaf_has_one_native_aggregate_call(
    tmp_path, field, expected,
):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1, "shape": [3],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(values):\n"
        f"    return AbstractTensor.topk(values,k=2,dim=0).{field}\n",
        "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="topk_public", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    root = module.functions["topk_public__root"]
    region = module.functions["topk_public__root__planned_region_0"]
    kernel = module.functions["topk_int64_dim"]
    calls = tuple(
        instruction for block in region.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == kernel.name
    )
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 5
    assert call.attributes["callee_input_ids"] == tuple(
        argument.id for argument in kernel.args[:5]
    )
    named = dict(kernel.metadata["named_outputs"])
    assert call.attributes["callee_output_ids"] == (
        named["values"], named["indices"],
    )
    assert call.res.dtype == "ssa.aggregate"
    assert not any(
        instruction.op == "Alloca"
        for block in region.blocks.values()
        for instruction in block.instrs
    )
    assert CorrelationTable.build(module).findings(module) == []

    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"topk_{field}")
    abi = artifact.runtime_abi()
    assert tuple(item["shape"] for item in abi["buffers"]) == ([3], [2])
    input_id, output_id = (item["value_id"] for item in abi["buffers"])
    execution = artifact.prepare_execution({
        input_id: np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64),
    }).run()
    assert execution.buffers[output_id].tolist() == expected


@pytest.mark.parametrize("k,dim,expected_shape,expected", (
    (4, 0, [3], [2, 1, 0]),
    (4, -1, [3], [2, 1, 0]),
    (0, 0, [0], []),
))
def test_public_topk_static_selected_count_bounds_native_k(
    tmp_path, k, dim, expected_shape, expected,
):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1, "shape": [3],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(values):\n"
        f"    return AbstractTensor.topk(values,k={k},dim={dim}).indices\n",
        "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="topk_bounded", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    region = module.functions["topk_bounded__root__planned_region_0"]
    instructions = tuple(i for block in region.blocks.values()
                         for i in block.instrs)
    call = next(i for i in instructions
                if i.op == "Call"
                and i.attributes.get("callee") == "topk_int64_dim")
    bounded_ids = {i.res.id for i in instructions if i.op == "Min"}
    assert len(bounded_ids) == 1
    assert (call.args[4].id in bounded_ids or any(
        i.op == "Cast" and i.res.id == call.args[4].id
        and i.args[0].id in bounded_ids for i in instructions
    ))
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, "topk_bounded__root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"topk_{k}_{dim}")
    abi = artifact.runtime_abi()
    assert tuple(item["shape"] for item in abi["buffers"]) == (
        [3], expected_shape,
    )
    input_id, output_id = (item["value_id"] for item in abi["buffers"])
    execution = artifact.prepare_execution({
        input_id: np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64),
    }).run()
    assert execution.buffers[output_id].tolist() == expected


@pytest.mark.parametrize("k,dim,reason", (
    (-1, 0, "negative k"),
    (2, -2, "out-of-range axis"),
))
def test_public_topk_invalid_static_selection_refuses(k, dim, reason):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1, "shape": [3],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    with pytest.raises(ValueError, match=reason):
        lower_ast_source_to_ssa(
            "def root(values):\n"
            f"    return AbstractTensor.topk(values,k={k},dim={dim}).indices\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="topk_invalid", extraction_contract=policy,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
