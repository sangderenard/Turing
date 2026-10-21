"""``id(x)`` on a structural parameter lowers and carries its parameter rows."""
from __future__ import annotations

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    DECLARED_PARAMETER, FUNCTION_PARAMETER, ParameterDeclaration,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
SOURCE = (
    "def root(x):\n"
    "    return id(x)\n"
)


def test_id_of_structural_parameter_lowers_and_carries_forward():
    book, token = begin_identity_book()
    try:
        module, _, exports = lower_ast_source_to_ssa(
            SOURCE, "root", name="id_builtin",
            extraction_contract=ExtractionContract(CONTRACT),
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            identity_book=book,
        )
        function = module.functions[exports[0]]
        assert not CorrelationTable.build(module).findings(module)
        scope = function.metadata["tensor_shape_concordance_scope"]
        page = book.pages[FUNCTION_PARAMETER.name]
        rows = {str(row[1]): page.latest(row) for row in page.scope_rows(scope)}
        assert rows.get("x") is not None
        assert dict(function.metadata["parameter_names"]).get("x") == int(rows["x"])
        declared = book.pages[DECLARED_PARAMETER.name]
        assert any(
            declared.latest(row) == ParameterDeclaration.USED
            for row in declared.scope_rows(scope)
        )
    finally:
        end_identity_book(token)
