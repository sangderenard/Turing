from pathlib import Path
from dataclasses import replace

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.concordance_declarations import (
    NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, NUMERICAL_CONSTRUCTOR_DEFAULT,
    SSA_VALUE, TENSOR_SSA_LOWERING,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, identity_book,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.numerical_constructor_defaults import _ssa_canonical_origin


def test_selected_literal_constructor_default_is_native_and_audited(tmp_path):
    # The authored integer tuple is payload; the selected SSA constructor's
    # declared dtype default owns its physical double storage.
    source = """def root(bits: AbstractTensor):
    return bits.ensure_tensor((13, -7, 0, 42, 5, 9, 1, 11))
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="literal_default_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    const = next(
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Const"
        and instruction.attributes.get("constructor_default") is not None
    )
    assert const.res.accounting["physical_dtype"] == "float64"
    owner = next(function for function in module.functions.values()
                 if any(const is instruction for block in function.blocks.values()
                        for instruction in block.instrs))
    descriptor = module.tensor_tables[owner.name].by_id(const.res.id)
    assert descriptor is not None
    assert descriptor.dtype == "float64"
    assert descriptor.metadata_state == "static"
    assert descriptor.shape == (8,)
    assert CorrelationTable.build(module).findings(module) == []

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "literal_default")
    bit_id = dict(module.functions[exports[0]].metadata["parameter_names"])["bits"]
    feed = {bit_id: np.arange(8, dtype=np.int64)}
    output = [value for key, value in artifact.prepare_execution(feed).run().buffers.items()
              if key not in feed]
    assert len(output) == 1
    assert output[0].tolist() == [13.0, -7.0, 0.0, 42.0, 5.0, 9.0, 1.0, 11.0]

    # The receipt checks the live Const payload, not just its selected dtype.
    const.attributes["value"] = (13, -7, 0, 42, 5, 9, 1, 12)
    assert any(
        finding.kind == "numerical-constructor-default"
        for finding in CorrelationTable._numerical_constructor_default_findings(module)
    )
    const.attributes["value"] = (13, -7, 0, 42, 5, 9, 1, 11)

    book = identity_book(module)
    declaration = next(
        book.latest_ref(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, row)
        for row in book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE).rows()
        if row[1] == "ensure_tensor"
    )
    signature = book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE).cell(
        declaration.row, declaration.column,
    )[0]
    assert signature in {cause for cause, _ in book.edges_into(declaration)}

    result = const.res
    from src.compiler.ssa_record_return_state import function_scope_of
    row = (function_scope_of(owner), int(result.id))
    previous = book.latest_ref(SSA_VALUE, row)
    physical = book.page(SSA_VALUE).cell(previous.row, previous.column)
    book.post(
        SSA_VALUE, row, replace(physical, physical_dtype="int64"),
        stage=TENSOR_SSA_LOWERING,
        provenance=Derived((previous,)), mode=Mode.REVISE,
    )
    assert any(
        finding.kind == "numerical-constructor-default"
        for finding in CorrelationTable._numerical_constructor_default_findings(module)
    )


def test_literal_constructor_rejects_conflicting_ssa_canonical_ancestor():
    source = """def root(bits: AbstractTensor):
    return bits.ensure_tensor((13, -7, 0, 42, 5, 9, 1, 11))
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="literal_ancestor_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    row = next(row for row in book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).rows()
               if book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).latest(row)[0]
               == "ensure_tensor")
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, row)
    _op, payload_ssa, result_ssa, *_rest = book.page(
        NUMERICAL_CONSTRUCTOR_DEFAULT,
    ).cell(receipt.row, receipt.column)
    payload_origin = _ssa_canonical_origin(
        book, payload_ssa.row[0], payload_ssa.row[1],
    )[1]
    assert payload_origin is not None and payload_origin != row[0]
    assert _ssa_canonical_origin(
        book, result_ssa.row[0], result_ssa.row[1],
    ) == (result_ssa, row[0])

    latest = book.latest_ref(SSA_VALUE, result_ssa.row)
    fact = book.page(SSA_VALUE).cell(latest.row, latest.column)
    book.post(
        SSA_VALUE, result_ssa.row, fact,
        stage=TENSOR_SSA_LOWERING,
        provenance=Derived((latest, payload_origin)), mode=Mode.REVISE,
    )
    assert _ssa_canonical_origin(
        book, result_ssa.row[0], result_ssa.row[1],
    ) == (None, None)
    assert any(
        finding.kind == "numerical-constructor-default"
        for finding in CorrelationTable._numerical_constructor_default_findings(module)
    )


def test_literal_weights_cross_dynamic_broadcast_and_axis_sum(tmp_path):
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    shaped = bits.reshape(-1, 8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).sum().item()
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="literal_weights_axis_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "literal_weights_axis")
    bit_id = dict(module.functions[exports[0]].metadata["parameter_names"])["bits"]
    weights = np.asarray((128, 64, 32, 16, 8, 4, 2, 1), dtype=np.float64)
    for count in (16, 64):
        data = np.arange(count, dtype=np.int64)
        feed = {bit_id: data}
        output = [value for key, value in artifact.prepare_execution(feed).run().buffers.items()
                  if key not in feed]
        expected = (data.reshape(-1, 8).astype(np.float64)
                    * weights).sum(axis=1).sum()
        assert len(output) == 1
        assert output[0].item() == expected
