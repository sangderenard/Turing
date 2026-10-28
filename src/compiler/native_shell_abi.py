"""Shared leaf helpers of the native shell ABI: identifiers, extents, storage indices (relocated from fortran_c_shell)."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not result or result[0].isdigit():
        result = "turing_" + result
    return result


def _entrypoint(module: Any, name: str | None = None) -> Any:
    selected = name or module.api.entry
    if selected is None:
        raise ValueError("Fortran module has no selected entry point")
    return module.api.entry_point(str(selected))


def _extent_values(
    entry: Any,
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    values: dict[str, int] = {}
    unresolved: set[str] = set()
    for parameter in entry.parameters:
        if parameter.role != "extent":
            continue
        name = str(parameter.name)
        fixed = re.fullmatch(r"extent_([1-9][0-9]*)", name)
        if fixed is None:
            unresolved.add(name)
        else:
            values[name] = int(fixed.group(1))
    for name, value in dict(overrides or {}).items():
        if name not in values and name not in unresolved:
            raise ValueError(f"unknown Fortran extent override {name!r}")
        if int(value) < 1:
            raise ValueError(f"Fortran extent {name!r} must be positive")
        values[name] = int(value)
        unresolved.discard(name)
    if unresolved:
        names = ", ".join(sorted(unresolved))
        raise ValueError(
            "shape-dynamic Fortran extents require explicit positive "
            f"extent_overrides: {names}"
        )
    return values


def _element_count(parameter: Any, extents: Mapping[str, int]) -> int:
    dynamic_dimensions = tuple(getattr(parameter, "extents", ()) or ())
    if dynamic_dimensions:
        count = 1
        for dimension in dynamic_dimensions:
            count *= int(extents[str(dimension)])
        return max(count, 1)
    if not tuple(parameter.shape or ()) and parameter.extent is not None:
        return max(int(extents[str(parameter.extent)]), 1)
    count = 1
    for extent in tuple(parameter.shape or ()):
        count *= int(extents.get(f"extent_{int(extent)}", extent))
    return max(count, 1)


def _source_name(parameter: Any) -> str:
    return str(parameter.source_name or parameter.name)


def _fortran_storage_index(
    parameter: Any,
    extents: Mapping[str, int | str],
    linear_index: str,
) -> str:
    """Map one C-row-major logical index to Fortran array storage.

    The API shape is semantic and remains in Python/NumPy dimension order.
    A ``bind(C)`` Fortran dummy with that shape stores its first dimension
    fastest, so the outer shell must perform this boundary permutation once.
    Resident feedback arenas stay in Fortran order and require no copies.
    """

    if str(getattr(parameter, "layout", "") or "") == "dense-row-major":
        return linear_index

    dynamic_dimensions = tuple(getattr(parameter, "extents", ()) or ())
    shape = (
        tuple(extents[str(name)] for name in dynamic_dimensions)
        if dynamic_dimensions
        else tuple(
            extents.get(f"extent_{int(size)}", int(size))
            for size in tuple(parameter.shape or ())
        )
    )
    if len(shape) <= 1:
        return linear_index

    def stride(sizes: tuple[int | str, ...]) -> int | str:
        if all(isinstance(size, int) for size in sizes):
            count = 1
            for size in sizes:
                count *= size
            return count
        return " * ".join(f"({size})" for size in sizes)

    terms = []
    for dimension, size in enumerate(shape):
        c_stride = stride(shape[dimension + 1:])
        fortran_stride = stride(shape[:dimension])
        divisor = str(c_stride) if isinstance(c_stride, int) else f"({c_stride})"
        multiplier = (
            str(fortran_stride)
            if isinstance(fortran_stride, int) else f"({fortran_stride})"
        )
        modulus = str(size) if isinstance(size, int) else f"({size})"
        coordinate = (
            f"(({linear_index}) / {divisor}) % {modulus}"
            if c_stride != 1
            else f"({linear_index}) % {modulus}"
        )
        terms.append(
            coordinate
            if fortran_stride == 1
            else f"({coordinate}) * {multiplier}"
        )
    return " + ".join(terms)


def _c_string(value: str) -> str:
    return json.dumps(str(value))
