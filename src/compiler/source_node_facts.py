"""Source-node facts: location, signature and constant/static value readers over authored AST nodes."""

from __future__ import annotations

import ast
import sys
from typing import Any


def _ast_source_location(expression: ast.AST) -> tuple[Any, ...]:
    return (
        type(expression),
        getattr(expression, "lineno", None),
        getattr(expression, "col_offset", None),
        getattr(expression, "end_lineno", None),
        getattr(expression, "end_col_offset", None),
    )


def _ast_source_signature(expression: ast.AST) -> tuple[Any, ...]:
    """Stable authored-occurrence key for reduced/copied AST nodes."""

    return (*_ast_source_location(expression), ast.dump(expression, include_attributes=False))


def _constant_value(data: dict[str, Any]) -> Any:
    """Read one normalized literal without confusing a real ``None``."""

    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return expression.value
    if "constant" in data:
        payload = data["constant"]
        # Every graph-express node is born with ``constant=None`` (see
        # ProcessGraph.add_node), so the key's presence alone proves
        # nothing: reading it unconditionally resolved arbitrary
        # computations -- an ``if``'s live predicate included -- to the
        # literal ``None``, and bool(None) then "proved" the branch
        # statically false. A ``None`` payload only counts as a literal
        # when the node itself is declared constant.
        if payload is not None or str(
            data.get("type")
        ) in {"Constant", "Const", "const"} or str(
            data.get("op") or ""
        ).casefold() == "const":
            return payload
    attributes = data.get("attributes") or {}
    if "value" in attributes:
        return attributes["value"]
    raise KeyError("constant ProcessGraph node has no literal payload")


def _static_python_value(bindings: dict[str, Any], path: str) -> Any:
    """Resolve one reducer-retained Python reference for coordination."""

    parts = str(path).split(".")
    try:
        value = bindings[parts[0]]
    except KeyError as exc:
        candidates = []
        for module in tuple(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if isinstance(namespace, dict) and parts[0] in namespace:
                candidates.append(namespace[parts[0]])
        identities = {id(candidate): candidate for candidate in candidates}
        if len(identities) != 1:
            raise KeyError(
                f"static Python reference {path!r} has no retained binding"
            ) from exc
        value = next(iter(identities.values()))
        bindings[parts[0]] = value
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def _is_runtime_value_id(value_id: Any) -> bool:
    """Whether ``value_id`` names a runtime value node (an integer id or its
    digit spelling). A compile-time reference (a resolved type/module left as a
    _StaticPythonReference) is not a runtime value id -- it is a compile-time
    constant, and must be excluded from runtime value-id sets rather than forced
    through ``int()``."""

    if isinstance(value_id, bool):
        return False
    if isinstance(value_id, int):
        return True
    if isinstance(value_id, str):
        return value_id.lstrip("-").isdigit()
    return False
