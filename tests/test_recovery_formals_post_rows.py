"""A formal appended by structural recovery has its ``ssa_value`` row."""
from __future__ import annotations

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.frame_identity_book import _frame_value_cell
from src.compiler.identity_concordance import (
    begin_identity_book,
    end_identity_book,
)
from src.compiler.structural_concordance import function_value_structure
from tests.test_aggregate_call_identity import _contract


def test_recovered_parameter_formal_has_ssa_value_row():
    book, token = begin_identity_book()
    try:
        module, _, _ = lower_ast_source_to_ssa(
            '''
def capture(array):
    return (array.copy(),)

def recover(saved):
    return saved[0]

def tick(array):
    saved = capture(array)
    return recover(saved)
''', "tick",
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            name="recovery_formal_rows", runtime_closure_only=True,
            extraction_contract=_contract("tick", {"array": np.zeros((2, 3))}),
        )
        recover = next(
            function for name, function in module.functions.items()
            if name.startswith("recovery_formal_rows__recover")
            and "planned_region" not in name
        )
        assert len(recover.args) == 1
        formal = recover.args[0]
        resumed, resumed_token = begin_identity_book(
            module.metadata["identity_book"])
        try:
            assert _frame_value_cell(recover, int(formal.id)) is not None
            structure, cell = function_value_structure(
                recover, int(formal.id))
        finally:
            end_identity_book(resumed_token)
        assert structure is not None and cell is not None
    finally:
        end_identity_book(token)
