"""Narrow tensor work keeps the compilation's existing sequence descriptors."""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_scalar_sequence_view_survives_tensor_closure_slice():
    # The parent had the int64 sequence column. The old narrowed IRModule
    # omitted sequence_tables and to_dtype refused that same column's span.
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [{
        "function": "root", "parameter": "start", "storage": "scalar",
        "dtype": "int64", "rank": 0, "python_type": "builtins.int",
    }]})
    module, _, _ = lower_ast_source_to_ssa(
        "def make(values):\n"
        "    return AbstractTensor.tensor(list(values)).to_dtype('int64')\n"
        "def root(start: int):\n"
        "    return make([start]).sum().item()\n",
        "root", name="scalar_sequence_registry",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    assert any(descriptor.column_dtypes == ("int64",)
               for table in module.sequence_tables.values()
               for descriptor in table.sequences.values())
