"""A logical int64 cast with double storage stays on the double kernel."""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, Mode, Novel, begin_identity_book,
    end_identity_book, identity_book,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import (
    function_scope_of, posted_ssa_value_fact,
)
from src.compiler.concordance_declarations import (
    INGESTION, INGEST_SOURCE, SSA_VALUE, SSAValueFact, SSAValueOrigin,
)
from src.compiler.tensor_ssa_lowering import _int64_binary_kernel
from src.transmogrifier.ssa import Function, SSAValue


@pytest.mark.parametrize("physical,expected", [
    ("float64", False), ("int64", True), (None, False),
])
def test_integer_binary_selector_reads_physical_span_fact(physical, expected):
    scalar = SSAValue(0, "int64", ())
    span = SSAValue(1, "int64", ())
    result = SSAValue(2, "int64", ())
    function = Function("physical_binary_selection", [scalar, span], {})
    book, token = begin_identity_book()
    try:
        scope = function_scope_of(function)
        for value, dtype, storage, rank in (
            (scalar, "int64", "scalar", 0),
            (span, physical, "span", 1),
        ):
            book.post(
                SSA_VALUE, (scope, int(value.id)),
                SSAValueFact(
                    value.dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                    physical_dtype=dtype, storage=storage, rank=rank,
                ),
                stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )
        assert _int64_binary_kernel(
            "sub", result, (scalar, span), owner=function, book=book,
        ) is expected
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("target_dtype", ("int64", "float64"))
def test_cast_backed_mask_inverse_uses_typed_binary_storage(tmp_path, target_dtype):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(values: AbstractTensor):
    valid = values > 0.0
    numeric_valid = valid.to_dtype('{target_dtype}')
    return (1 - numeric_valid).sum().item()
""".format(target_dtype=target_dtype)
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name=f"cast_backed_mask_inverse_{target_dtype}",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    book = identity_book(module)
    cast_calls = [
        (function, instruction)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == {
            "int64": "cast_double_to_int64_values",
            "float64": "cast_double_to_double_values",
        }[target_dtype]
    ]
    assert len(cast_calls) == 1
    cast_function, cast_call = cast_calls[0]
    cast_fact = posted_ssa_value_fact(
        cast_function, int(cast_call.res.id), book=book,
    )
    assert (cast_fact.dtype, cast_fact.physical_dtype,
            cast_fact.storage, cast_fact.rank) == (
                target_dtype, target_dtype, "span", 1,
            )
    binary_executions = [
        (function, instruction)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if (instruction.op == "Sub" or instruction.op == "Call"
            and instruction.attributes.get("callee") in {
                "binary_int64", "binary_double", "binary_scalar_int64",
                "binary_scalar_double",
            })
        and any(int(argument.id) == int(cast_call.res.id)
                for argument in instruction.args)
    ]
    assert len(binary_executions) == 1
    binary_function, binary = binary_executions[0]
    if target_dtype == "float64":
        assert binary.op == "Call"
        assert binary.attributes["callee"] == "binary_scalar_double"
    else:
        assert (binary.op == "Sub" or binary.attributes.get("callee")
                in {"binary_int64", "binary_scalar_int64"})
    assert posted_ssa_value_fact(
        binary_function, int(cast_call.res.id), book=book,
    ).physical_dtype == target_dtype

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"cast_backed_inverse_{target_dtype}")
    entry = module.functions[exports[0]]
    parameter = int(dict(entry.metadata["parameter_names"])["values"])
    returned, = [
        int(instruction.args[0].id)
        for block in entry.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    ]
    for values in ([], [1.25], [1.25, -2.5, 0.0, 3.75]):
        data = np.asarray(values, dtype=np.float64)
        expected = int(np.sum(1 - (data > 0).astype(np.int64)))
        feeds = {parameter: data}
        execution = artifact.prepare_execution(feeds).run()
        assert np.asarray(execution.buffers[returned]).item() == expected


@pytest.mark.parametrize("static_shape", [False, True])
def test_float_values_cast_to_integer_before_cancelling_sum(tmp_path, static_shape):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 1,
            **({"shape": (3,)} if static_shape else {}),
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor):\n"
        "    return values.to_dtype('int64').sum().item()\n",
        "root", name="cast_then_exact_integer_sum",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "cast_then_exact_sum")
    entry = module.functions[exports[0]]
    parameter = int(dict(entry.metadata["parameter_names"])["values"])
    returned, = [
        int(instruction.args[0].id)
        for block in entry.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    ]
    data = np.asarray([2**53, 1, -(2**53)], dtype=np.float64)
    execution = artifact.prepare_execution({parameter: data}).run()
    assert np.asarray(execution.buffers[returned]).item() == 1
