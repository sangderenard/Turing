"""Canonical hierarchy for compiler-visible mathematical libraries.

The hierarchy is semantic, not a backend registry.  A method exists here
because authored mathematical material exists; native, Python and shader
products project this same record and separately state which realizations they
actually package.  This prevents a backend's current coverage from shrinking
the mathematical library, and prevents a public method name from masquerading
as an implemented kernel.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from typing import Any

from .blas import BLAS_ROLES, BLASRole


CATALOG_SCHEMA = "turing.mathematical-library.v1"


@dataclass(frozen=True, slots=True)
class MathematicalParameter:
    name: str
    kind: str
    access: str

    def to_mapping(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "access": self.access}


@dataclass(frozen=True, slots=True)
class MathematicalMethod:
    name: str
    identity: str
    level: int
    parameters: tuple[MathematicalParameter, ...]
    result: dict[str, Any]
    source_symbol: str
    source: str
    source_sha256: str
    abstract_operators: tuple[str, ...] = ()

    def to_mapping(self, *, include_source: bool = True) -> dict[str, Any]:
        record = {
            "name": self.name,
            "identity": self.identity,
            "level": self.level,
            "parameters": [item.to_mapping() for item in self.parameters],
            "result": dict(self.result),
            "source_symbol": self.source_symbol,
            "source_sha256": self.source_sha256,
            "abstract_operators": list(self.abstract_operators),
        }
        if include_source:
            record["source"] = self.source
        return record


@dataclass(frozen=True, slots=True)
class MathematicalSubLibrary:
    name: str
    identity: str
    methods: tuple[MathematicalMethod, ...]

    def method(self, name: str) -> MathematicalMethod:
        for method in self.methods:
            if method.name == str(name):
                return method
        raise KeyError(
            f"unknown method {name!r} in {self.identity}; expected one of "
            f"{tuple(item.name for item in self.methods)!r}"
        )

    def to_mapping(self, *, include_source: bool = True) -> dict[str, Any]:
        return {
            "name": self.name,
            "identity": self.identity,
            "methods": [
                method.to_mapping(include_source=include_source)
                for method in self.methods
            ],
        }


@dataclass(frozen=True, slots=True)
class MathematicalLibrary:
    name: str
    identity: str
    libraries: tuple[MathematicalSubLibrary, ...]

    def library(self, name: str) -> MathematicalSubLibrary:
        for library in self.libraries:
            if library.name == str(name):
                return library
        raise KeyError(
            f"unknown mathematical sublibrary {name!r}; expected one of "
            f"{tuple(item.name for item in self.libraries)!r}"
        )

    def to_mapping(self, *, include_source: bool = True) -> dict[str, Any]:
        return {
            "schema": CATALOG_SCHEMA,
            "name": self.name,
            "identity": self.identity,
            "libraries": [
                library.to_mapping(include_source=include_source)
                for library in self.libraries
            ],
        }


def _method_from_blas_role(role: BLASRole) -> MathematicalMethod:
    """Derive the method signature from its authored algorithm source."""

    tree = ast.parse(role.source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == role.name
    )
    reads: set[str] = set()
    writes: set[str] = set()
    extents: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                extents.update(
                    argument.id for argument in node.iter.args
                    if isinstance(argument, ast.Name)
                )
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            (writes if isinstance(node.ctx, ast.Store) else reads).add(
                node.value.id
            )
    parameters = []
    for name in role.parameter_order:
        if name in extents:
            kind, access = "extent", "read"
        elif name in reads or name in writes:
            kind = "buffer"
            access = (
                "read_write" if name in reads and name in writes
                else "write" if name in writes else "read"
            )
        else:
            kind, access = "scalar", "read"
        parameters.append(MathematicalParameter(name, kind, access))
    returned = next(
        (node.value for node in ast.walk(function) if isinstance(node, ast.Return)),
        None,
    )
    if isinstance(returned, ast.Name) and returned.id in role.parameter_order:
        result = {"kind": "parameter", "parameter": returned.id}
    else:
        result = {"kind": "scalar"}
    source_bytes = role.source.encode("utf-8")
    return MathematicalMethod(
        name=role.name,
        identity=role.identity,
        level=role.level,
        parameters=tuple(parameters),
        result=result,
        source_symbol=f"{role.name.upper()}_SOURCE",
        source=role.source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        abstract_operators=(
            (role.abstract_operator,) if role.abstract_operator else ()
        ),
    )


BLAS_LIBRARY = MathematicalSubLibrary(
    name="blas",
    identity="blas",
    methods=tuple(_method_from_blas_role(role) for role in BLAS_ROLES.values()),
)

TURING_MATHEMATICAL_LIBRARY = MathematicalLibrary(
    name="Turing mathematical library",
    identity="turing.math",
    libraries=(BLAS_LIBRARY,),
)


class AbstractTensorBLAS:
    """Graph-visible BLAS projection installed on an AbstractTensor class.

    These methods intentionally compose public tensor operations.  They do not
    call the generated NumPy/DLL loader, because doing so would materialize
    backend data and cut autograd/ProcessGraph.  Backend selection may later
    replace the resulting canonical roles with any packaged realization.
    """

    def __init__(self, tensor_type: type):
        self.tensor_type = tensor_type
        self.catalog = BLAS_LIBRARY

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(method.name for method in self.catalog.methods)

    def _pair(self, left: Any, right: Any):
        left = self.tensor_type.get_tensor(left)
        return left, left.ensure_tensor(right)

    def scal(self, x: Any, alpha: Any, *, y: Any = None):
        x = self.tensor_type.get_tensor(x)
        return x * alpha

    def axpy(self, x: Any, y: Any, alpha: Any):
        x, y = self._pair(x, y)
        return alpha * x + y

    def dot(self, x: Any, y: Any):
        x, y = self._pair(x, y)
        return (x * y).sum()

    def gemv(
        self, a: Any, x: Any, *, y: Any = None,
        alpha: Any = 1.0, beta: Any = 0.0,
    ):
        a, x = self._pair(a, x)
        product = alpha * (a @ x)
        return product if y is None else product + beta * a.ensure_tensor(y)

    def gemm(
        self, a: Any, b: Any, *, c: Any = None,
        alpha: Any = 1.0, beta: Any = 0.0,
    ):
        a, b = self._pair(a, b)
        product = alpha * (a @ b)
        return product if c is None else product + beta * a.ensure_tensor(c)

    def rot(self, x: Any, y: Any, c: Any, s: Any):
        x, y = self._pair(x, y)
        return c * x + s * y, c * y - s * x


class AbstractTensorMathematicalLibrary:
    """The outer namespace installed on one AbstractTensor implementation."""

    def __init__(self, tensor_type: type):
        self.catalog = TURING_MATHEMATICAL_LIBRARY
        self.blas = AbstractTensorBLAS(tensor_type)

    @property
    def libraries(self) -> tuple[str, ...]:
        return tuple(library.name for library in self.catalog.libraries)


class AbstractTensorProviderBLAS:
    """Adapt an installed array-level BLAS provider back into tensors."""

    def __init__(self, tensor_type: type, provider: Any):
        self.tensor_type = tensor_type
        self.provider = provider
        self.catalog = BLAS_LIBRARY

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.provider.methods)

    def _tensor(self, value: Any):
        return self.tensor_type.get_tensor(value)

    @staticmethod
    def _wrap(template: Any, value: Any):
        return template.ensure_tensor(value)

    def scal(self, x: Any, alpha: Any, *, y: Any = None):
        x = self._tensor(x)
        value = self.provider.scal(
            x.data, alpha, y=None if y is None else self._tensor(y).data,
        )
        return self._wrap(x, value)

    def axpy(self, x: Any, y: Any, alpha: Any):
        x, y = self._tensor(x), self._tensor(y)
        return self._wrap(x, self.provider.axpy(x.data, y.data, alpha))

    def dot(self, x: Any, y: Any):
        x, y = self._tensor(x), self._tensor(y)
        return self._wrap(x, self.provider.dot(x.data, y.data))

    def gemv(
        self, a: Any, x: Any, *, y: Any = None,
        alpha: Any = 1.0, beta: Any = 0.0,
    ):
        a, x = self._tensor(a), self._tensor(x)
        value = self.provider.gemv(
            a.data, x.data,
            y=None if y is None else self._tensor(y).data,
            alpha=alpha, beta=beta,
        )
        return self._wrap(a, value)

    def gemm(
        self, a: Any, b: Any, *, c: Any = None,
        alpha: Any = 1.0, beta: Any = 0.0,
    ):
        a, b = self._tensor(a), self._tensor(b)
        value = self.provider.gemm(
            a.data, b.data,
            c=None if c is None else self._tensor(c).data,
            alpha=alpha, beta=beta,
        )
        return self._wrap(a, value)

    def rot(self, x: Any, y: Any, c: Any, s: Any):
        x, y = self._tensor(x), self._tensor(y)
        left, right = self.provider.rot(x.data, y.data, c, s)
        return self._wrap(x, left), self._wrap(x, right)


class AbstractTensorProviderMathematicalLibrary:
    """Installed product projection retaining the outer library hierarchy."""

    def __init__(self, tensor_type: type, product: Any):
        self.catalog = TURING_MATHEMATICAL_LIBRARY
        self.product = product
        self.blas = AbstractTensorProviderBLAS(tensor_type, product.blas)

    @property
    def libraries(self) -> tuple[str, ...]:
        return tuple(self.product.libraries)


def install_abstract_tensor_mathematical_library(
    tensor_type: type,
) -> AbstractTensorMathematicalLibrary:
    """Install ``math.blas`` plus the convenient ``blas`` alias on a tensor."""

    namespace = AbstractTensorMathematicalLibrary(tensor_type)
    setattr(tensor_type, "_semantic_math", namespace)
    setattr(tensor_type, "math", namespace)
    setattr(tensor_type, "blas", namespace.blas)

    def install_compiled(cls, product):
        installed = AbstractTensorProviderMathematicalLibrary(cls, product)
        setattr(cls, "compiled_math", product)
        setattr(cls, "math", installed)
        setattr(cls, "blas", installed.blas)
        return product

    def use_semantic(cls):
        semantic = cls._semantic_math
        setattr(cls, "math", semantic)
        setattr(cls, "blas", semantic.blas)
        return semantic

    setattr(
        tensor_type,
        "install_mathematical_library",
        classmethod(install_compiled),
    )
    setattr(
        tensor_type,
        "use_semantic_mathematical_library",
        classmethod(use_semantic),
    )
    return namespace


__all__ = [
    "BLAS_LIBRARY",
    "AbstractTensorBLAS",
    "AbstractTensorMathematicalLibrary",
    "AbstractTensorProviderBLAS",
    "AbstractTensorProviderMathematicalLibrary",
    "CATALOG_SCHEMA",
    "MathematicalLibrary",
    "MathematicalMethod",
    "MathematicalParameter",
    "MathematicalSubLibrary",
    "TURING_MATHEMATICAL_LIBRARY",
    "install_abstract_tensor_mathematical_library",
]
