"""The owned tensor's native pointer port is one physical output."""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_aggregate_abi import analyze_aggregate_abi
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import owned_tensor_data_bindings


def test_owned_dynamic_tensor_has_one_native_pointer_output():
    source = '''def root(lengths: AbstractTensor, bit_count: int) -> AbstractTensor:
    return AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
'''
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "lengths", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "bit_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="owned_arange_publication_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    first = analyze_aggregate_abi(module)
    second = analyze_aggregate_abi(module)
    seen_bindings = []
    for name, function in module.functions.items():
        bindings = owned_tensor_data_bindings(module, name)
        if not bindings:
            continue
        seen_bindings.extend(bindings)
        returned = next(
            instruction.args for block in function.blocks.values()
            for instruction in block.instrs if instruction.op == "Ret"
        )
        native_first = first.outputs_by_callee[name]
        native_second = second.outputs_by_callee[name]
        assert tuple(value.id for value in native_first) == tuple(
            value.id for value in native_second
        )
        for binding in bindings:
            assert any(value.id == binding.tensor_id for value in returned)
            assert all(value.id != binding.tensor_id for value in native_first)
            assert sum(
                value.id == binding.data_value_id for value in native_first
            ) == 1
            assert next(
                value for value in native_first
                if value.id == binding.data_value_id
            ).dtype == "ptr"
    assert sorted(binding.element_dtype for binding in seen_bindings) == [
        "float64", "int64",
    ]

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    assert not CorrelationTable.build(module).findings(module)
    for name in module.functions:
        for binding in owned_tensor_data_bindings(module, name):
            assert (
                f"*((void **)(out{binding.data_value_id})) = "
                f"(void *)(t{binding.data_value_id});"
            ) in artifact.source
