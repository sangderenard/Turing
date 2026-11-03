"""Imported LLVM argument declarations, not resident pointee dtypes, own ABI facts."""

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.accelerator_backends.llvm_repository_ssa import (
    post_repository_kernel_identity,
)
from src.compiler.concordance_declarations import (
    REPOSITORY_KERNEL_VALUE, SSA_VALUE,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.ssa_record_return_state import function_scope_of


def _formal_facts(function, book):
    assert post_repository_kernel_identity(function, book)
    scope = function_scope_of(function)
    root_row = ("llvm_repository", function.name)
    origins = function.metadata["llvm_repository_kernel"][1]
    pairs = []
    for formal in function.args:
        position = origins[formal.id][0]
        declaration = book.page(REPOSITORY_KERNEL_VALUE).latest(
            (root_row, position)
        )
        physical = book.page(SSA_VALUE).latest((scope, formal.id))
        pairs.append((declaration, physical))
    return pairs


def test_topk_pointer_formals_do_not_become_scalars_from_pointee_dtype():
    book, token = begin_identity_book()
    try:
        kernel = c_backend_repository_ssa_reference().module.functions[
            "topk_int64_dim"
        ]
        pairs = _formal_facts(kernel, book)
        assert [fact.text.partition(" ")[0] for fact, _ in pairs] == [
            "ptr", "ptr", "i32", "i32", "i32", "ptr", "ptr",
        ]
        for index in (0, 1, 5, 6):
            _declaration, physical = pairs[index]
            assert physical.physical_dtype is None
            assert physical.storage is None
            assert physical.rank is None
        for index in (2, 3, 4):
            _declaration, physical = pairs[index]
            assert physical.physical_dtype == "int32"
            assert physical.storage == "scalar"
            assert physical.rank == 0
    finally:
        end_identity_book(token)


def test_zeros_fill_kernel_keeps_declared_double_scalar_and_pointer_output():
    book, token = begin_identity_book()
    try:
        kernel = c_backend_repository_ssa_reference().module.functions[
            "fill_double"
        ]
        pairs = _formal_facts(kernel, book)
        assert [fact.text.partition(" ")[0] for fact, _ in pairs] == [
            "ptr", "double", "i32",
        ]
        assert pairs[0][1].physical_dtype is None
        assert pairs[0][1].storage is None
        assert pairs[0][1].rank is None
        assert pairs[1][1].physical_dtype == "float64"
        assert pairs[1][1].storage == "scalar"
        assert pairs[1][1].rank == 0
        assert pairs[2][1].physical_dtype == "int32"
        assert pairs[2][1].storage == "scalar"
        assert pairs[2][1].rank == 0
    finally:
        end_identity_book(token)
