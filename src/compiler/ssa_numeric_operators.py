"""Complete operator contract between tensor programs and repository SSA.

This module deliberately records both vocabularies involved in backend
lowering:

* :data:`REPOSITORY_SSA_OPERATORS` is every real repository-SSA ``Handler``;
* :data:`TENSOR_SSA_OPERATORS` is every canonical AbstractTensor operation.

The latter does not pretend that every tensor operation is a primitive SSA
opcode.  Operations with a real primitive use it.  The rest are represented
honestly as ``Call`` instructions carrying ``tensor_operation=<name>`` until a
linked repository-SSA implementation expands them.  This is the contract C,
LLVM, Fortran, WebAssembly, and future backends coordinate against.

The historical module name is retained because an unfinished backend-parity
change introduced it under that name.  ``PORTABLE_NUMERIC_OPERATORS`` remains
as a compatibility view, but it is not the complete backend contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from ..common.tensors.operator_catalog import (
    ACCESSOR_OPERATORS,
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    COMPOSITE_MATH_OPERATORS,
    CREATION_OPERATORS,
    HOST_BOUNDARY_OPERATORS,
    REDUCTION_AND_LINALG_OPERATORS,
    SHAPE_AND_INDEX_OPERATORS,
    SPECTRAL_OPERATORS,
    TYPE_AND_DEVICE_OPERATORS,
    VALUE_LIFECYCLE_OPERATORS,
)
from ..transmogrifier.ssa_registry import Handler


@dataclass(frozen=True, slots=True)
class RepositorySSAOperator:
    """One real opcode in the repository SSA instruction vocabulary."""

    name: str
    handler: Handler


@dataclass(frozen=True, slots=True)
class TensorSSAOperator:
    """The deterministic SSA representation of one canonical tensor op."""

    name: str
    category: str
    handler: Handler
    tensor_operation: str | None = None

    @property
    def is_direct(self) -> bool:
        return self.handler is not Handler.Call

    @property
    def repository_spelling(self) -> str:
        if self.tensor_operation is None:
            return self.handler.value
        return f"{Handler.Call.value}[tensor_operation={self.tensor_operation}]"


REPOSITORY_SSA_OPERATORS = tuple(
    RepositorySSAOperator(handler.value, handler) for handler in Handler
)
REPOSITORY_SSA_OPERATOR_BY_NAME = MappingProxyType(
    {row.name: row for row in REPOSITORY_SSA_OPERATORS}
)


# Canonical tensor operations with a genuine direct repository opcode.  Every
# other canonical operation remains a named tensor Call until its repository
# implementation is linked.  Do not manufacture primitive opcodes for those
# operations: Call is the repository SSA's modular extension mechanism.
_DIRECT_TENSOR_HANDLERS: Mapping[str, Handler] = MappingProxyType({
    "add": Handler.Add,
    "sub": Handler.Sub,
    "mul": Handler.Mul,
    "truediv": Handler.Div,
    "floordiv": Handler.FloorDiv,
    "mod": Handler.Mod,
    "pow": Handler.Pow,
    "matmul": Handler.MatMul,
    "neg": Handler.Neg,
    "abs": Handler.Abs,
    "bitand": Handler.And,
    "bitor": Handler.Or,
    "bitxor": Handler.Xor,
    "invert": Handler.Not,
    "shl": Handler.Shl,
    "shr": Handler.Shr,
    "logical_and": Handler.LAnd,
    "logical_or": Handler.LOr,
    "logical_not": Handler.LNot,
    "equal": Handler.Eq,
    "not_equal": Handler.Ne,
    "less": Handler.Lt,
    "less_equal": Handler.Le,
    "greater": Handler.Gt,
    "greater_equal": Handler.Ge,
    "int_trunc": Handler.Trunc,
    "zext": Handler.ZExt,
    "sext": Handler.SExt,
    "fptosi": Handler.FpToSi,
    "fptoui": Handler.FpToUi,
    "sitofp": Handler.SiToFp,
    "uitofp": Handler.UiToFp,
    "tensor_from_list": Handler.Const,
})


_CATEGORY_GROUPS = (
    ("elementwise_unary", ELEMENTWISE_UNARY),
    ("elementwise_binary", ELEMENTWISE_BINARY),
    ("creation", CREATION_OPERATORS),
    ("shape_and_index", SHAPE_AND_INDEX_OPERATORS),
    ("reduction_and_linalg", REDUCTION_AND_LINALG_OPERATORS),
    ("composite_math", COMPOSITE_MATH_OPERATORS),
    ("spectral", SPECTRAL_OPERATORS),
    ("type_and_device", TYPE_AND_DEVICE_OPERATORS),
    ("value_lifecycle", VALUE_LIFECYCLE_OPERATORS),
    ("accessor", ACCESSOR_OPERATORS),
    ("host_boundary", HOST_BOUNDARY_OPERATORS),
)


def _tensor_rows() -> tuple[TensorSSAOperator, ...]:
    categories: dict[str, str] = {}
    for category, names in _CATEGORY_GROUPS:
        for name in sorted(names):
            categories.setdefault(name, category)
    missing = CANONICAL_ABSTRACT_TENSOR_OPERATORS - categories.keys()
    extra = categories.keys() - CANONICAL_ABSTRACT_TENSOR_OPERATORS
    if missing or extra:
        raise RuntimeError(
            "canonical tensor operator categories drifted: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    rows = []
    for name in sorted(CANONICAL_ABSTRACT_TENSOR_OPERATORS):
        handler = _DIRECT_TENSOR_HANDLERS.get(name, Handler.Call)
        rows.append(TensorSSAOperator(
            name=name,
            category=categories[name],
            handler=handler,
            tensor_operation=name if handler is Handler.Call else None,
        ))
    return tuple(rows)


TENSOR_SSA_OPERATORS = _tensor_rows()
TENSOR_SSA_OPERATOR_BY_NAME = MappingProxyType(
    {row.name: row for row in TENSOR_SSA_OPERATORS}
)
CANONICAL_TENSOR_SSA_OPERATORS = frozenset(TENSOR_SSA_OPERATOR_BY_NAME)

# Compatibility names retained for the unfinished 49-operation implementation.
# These now derive from the real elementwise catalogue instead of restating it.
PORTABLE_NUMERIC_OPERATORS = frozenset(
    ELEMENTWISE_UNARY | ELEMENTWISE_BINARY
)
PORTABLE_UNARY_OPERATORS = frozenset(ELEMENTWISE_UNARY)
PORTABLE_BINARY_OPERATORS = frozenset(ELEMENTWISE_BINARY)
NUMERIC_SSA_OPERATORS = tuple(
    row for row in TENSOR_SSA_OPERATORS
    if row.name in PORTABLE_NUMERIC_OPERATORS
)
NUMERIC_OPERATOR_BY_NAME = MappingProxyType(
    {row.name: row for row in NUMERIC_SSA_OPERATORS}
)


def backend_operator_inventories() -> Mapping[str, frozenset[str]]:
    """Return each backend's own authored/lowerable operator vocabulary.

    These are intentionally *not* padded to the AbstractTensor or repository
    SSA vocabulary.  A backend owns exactly the operations its lowering model
    needs.  The dual-IR bridge is responsible for deciding whether an incoming
    operation is direct, decomposed, dispatched, or kept at a host boundary.

    Imports are local so an individual backend may consume the two IR
    vocabularies without importing all of its peers.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_dispatch_operations,
        covered_operations,
    )
    from .fused_program_wasm_backend import supported_tensor_operations as wasm
    from .ssa_fortran_backend import supported_tensor_operations as fortran
    from .ssa_llvm_backend import supported_tensor_operations as llvm

    return MappingProxyType({
        "c": frozenset(c_dispatch_operations() | covered_operations()),
        "llvm": frozenset(llvm()),
        "fortran": frozenset(fortran()),
        "wasm": frozenset(wasm()),
    })


def shared_implemented_tensor_operations() -> frozenset[str]:
    """Canonical operations implemented by all four backend lanes."""

    inventories = backend_operator_inventories()
    return frozenset.intersection(*inventories.values())


__all__ = [
    "CANONICAL_TENSOR_SSA_OPERATORS",
    "NUMERIC_OPERATOR_BY_NAME",
    "NUMERIC_SSA_OPERATORS",
    "PORTABLE_BINARY_OPERATORS",
    "PORTABLE_NUMERIC_OPERATORS",
    "PORTABLE_UNARY_OPERATORS",
    "REPOSITORY_SSA_OPERATOR_BY_NAME",
    "REPOSITORY_SSA_OPERATORS",
    "RepositorySSAOperator",
    "TENSOR_SSA_OPERATOR_BY_NAME",
    "TENSOR_SSA_OPERATORS",
    "TensorSSAOperator",
    "backend_operator_inventories",
    "shared_implemented_tensor_operations",
]
