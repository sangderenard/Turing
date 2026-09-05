from src.common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from src.common.tensors.operator_catalog import (
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
)
from src.compiler.ssa_fortran_backend import supported_tensor_operations as fortran_ops
from src.compiler.ssa_llvm_backend import supported_tensor_operations as llvm_ops
from src.compiler.ssa_numeric_operators import (
    CANONICAL_TENSOR_SSA_OPERATORS,
    NUMERIC_SSA_OPERATORS,
    REPOSITORY_SSA_OPERATORS,
    TENSOR_SSA_OPERATOR_BY_NAME,
    TENSOR_SSA_OPERATORS,
    backend_operator_inventories,
    shared_implemented_tensor_operations,
)
from src.transmogrifier.ssa_registry import Handler


def test_contract_contains_every_repository_and_tensor_operator():
    assert len(REPOSITORY_SSA_OPERATORS) == len(Handler) == 119
    assert {row.handler for row in REPOSITORY_SSA_OPERATORS} == set(Handler)
    # 225, not the 223 this line was born with: the assertion was written in
    # the same commit that added ``cast_like`` and never counted it, and
    # ``sigmoid`` joined ELEMENTWISE_UNARY afterwards.  Both are real
    # vocabulary, so the number moved to meet them.
    assert len(TENSOR_SSA_OPERATORS) == len(CANONICAL_ABSTRACT_TENSOR_OPERATORS) == 225
    assert CANONICAL_TENSOR_SSA_OPERATORS == CANONICAL_ABSTRACT_TENSOR_OPERATORS


def test_nonprimitive_tensor_operations_remain_named_repository_calls():
    direct = TENSOR_SSA_OPERATOR_BY_NAME["add"]
    composite = TENSOR_SSA_OPERATOR_BY_NAME["softmax"]
    structural = TENSOR_SSA_OPERATOR_BY_NAME["unfold3d"]

    assert direct.handler is Handler.Add
    assert direct.repository_spelling == "Add"
    assert composite.handler is Handler.Call
    assert composite.repository_spelling == "Call[tensor_operation=softmax]"
    assert structural.handler is Handler.Call
    assert structural.repository_spelling == "Call[tensor_operation=unfold3d]"


def test_numeric_view_is_derived_from_the_complete_elementwise_catalogue():
    expected = ELEMENTWISE_UNARY | ELEMENTWISE_BINARY
    assert len(NUMERIC_SSA_OPERATORS) == len(expected) == 57
    assert {row.name for row in NUMERIC_SSA_OPERATORS} == expected
    assert expected <= fortran_ops()
    assert expected <= llvm_ops()


def test_each_backend_keeps_its_own_operator_inventory():
    inventories = backend_operator_inventories()
    assert set(inventories) == {"c", "llvm", "fortran", "wasm", "javascript"}
    assert all(inventories.values())
    assert len({len(operations) for operations in inventories.values()}) > 1
    # Backend-private primitives are legitimate vocabulary too; they are not
    # required to masquerade as canonical AbstractTensor operations.
    assert {"broadcast", "sum_dim", "extent"} <= inventories["llvm"]
    typed_casts = {
        "int_trunc", "zext", "sext", "fptosi", "fptoui", "sitofp", "uitofp",
    }
    # Typed casts are SSA instructions shared by the mixed-value native lanes.
    assert typed_casts <= inventories["llvm"]
    assert typed_casts <= inventories["fortran"]
    assert typed_casts <= inventories["c"]
    assert not typed_casts & inventories["wasm"]
    shared = shared_implemented_tensor_operations()
    assert ELEMENTWISE_UNARY - {
        "tan", "int_trunc", "zext", "sext", "fptosi", "fptoui",
        "sitofp", "uitofp",
    } <= shared
    assert {
        "add", "sub", "mul", "truediv", "pow", "mod", "floordiv",
        "logical_and", "logical_or", "bitand", "bitor", "bitxor",
        "shl", "shr",
    } <= shared
