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


__all__ = [
    "BLAS_LIBRARY",
    "CATALOG_SCHEMA",
    "MathematicalLibrary",
    "MathematicalMethod",
    "MathematicalParameter",
    "MathematicalSubLibrary",
    "TURING_MATHEMATICAL_LIBRARY",
]
