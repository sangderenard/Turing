"""Python-only semantic overlays for structural ProcessGraph ingestion.

Source extraction and graph interpretation are deliberately separate. The
extraction contract resolves a live callable while that identity is still
available and attaches its receipt to the corresponding ``ast.Call``. This
module consumes that receipt at the graph boundary without admitting source.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

from .node_special_cases import SpecialCase
from .python_identity_programs import resolve_python_identity


_EXTRACTION_ACTIONS = frozenset({
    "ingest_python",
    "intrinsic",
    "python_host_call",
    "use_native",
    "decompile_machine",
    "reject",
})


def extraction_receipt(node: Any) -> dict[str, Any] | None:
    """Return a detached, minimally validated call receipt if one exists."""

    value = getattr(node, "_extraction_contract", None)
    if not isinstance(value, Mapping):
        return None
    receipt = dict(value)
    action = str(receipt.get("action") or "")
    if action not in _EXTRACTION_ACTIONS:
        return None
    receipt["action"] = action
    receipt["parameters"] = dict(receipt.get("parameters") or {})
    return receipt


def _receipt_attributes(receipt: Mapping[str, Any]) -> dict[str, Any]:
    attributes = {
        "extraction_contract": dict(receipt),
        "extraction_action": receipt["action"],
        "extraction_rule": receipt.get("rule_id"),
        "extraction_identity": receipt.get("identity"),
        "extraction_classification": receipt.get("classification"),
    }
    if receipt["action"] == "intrinsic":
        parameters = dict(receipt.get("parameters") or {})
        attributes["backend_intrinsic_candidate"] = {
            "semantic_identity": receipt.get("identity"),
            "lowering_namespace": parameters.get("lowering_namespace"),
            "ingested_fallback": bool(
                parameters.get("ingest_fallback_source", False)
            ),
        }
    return attributes


def _call_spelling(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _named_integer_origin(value: Any, path: str) -> dict[str, Any] | None:
    """Describe the CPython named-int wrapper category, if ``value`` is one."""

    value_type = type(value)
    if (
        value_type is int
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value_type.__dict__.get("__reduce__", object()) is not None
    ):
        return None
    symbolic_name = getattr(value, "name", None)
    if not isinstance(symbolic_name, str) or not symbolic_name:
        return None
    return {
        "schema": "turing.python-named-integer.v1",
        "path": str(path),
        "module": str(value_type.__module__),
        "type": str(value_type.__qualname__),
        "name": symbolic_name,
        "integer_value": int(value),
    }


def canonicalize_python_static_data(
    value: Any,
    *,
    path: str,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Canonicalize named integers inside one static Python value tree."""

    origin = _named_integer_origin(value, path)
    if origin is not None:
        return int(value), (origin,)

    if isinstance(value, tuple):
        values = []
        origins = []
        for index, item in enumerate(value):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        return (tuple(values), tuple(origins)) if origins else (value, ())
    if isinstance(value, list):
        values = []
        origins = []
        for index, item in enumerate(value):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        return (values, tuple(origins)) if origins else (value, ())
    if isinstance(value, (set, frozenset)):
        values = []
        origins = []
        ordered_items = sorted(
            value,
            key=lambda item: (
                str(type(item).__module__),
                str(type(item).__qualname__),
                repr(item),
            ),
        )
        for index, item in enumerate(ordered_items):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        if not origins:
            return value, ()
        container = frozenset if isinstance(value, frozenset) else set
        return container(values), tuple(origins)
    if isinstance(value, dict):
        values = {}
        origins = []
        for index, (key, item) in enumerate(value.items()):
            canonical_key, key_origins = canonicalize_python_static_data(
                key,
                path=f"{path}.key[{index}]",
            )
            canonical_item, item_origins = canonicalize_python_static_data(
                item,
                path=f"{path}[{key!r}]",
            )
            values[canonical_key] = canonical_item
            origins.extend(key_origins)
            origins.extend(item_origins)
        return (values, tuple(origins)) if origins else (value, ())
    return value, ()


def canonicalize_python_static_bindings(
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a persistence-safe copy of a Python static environment."""

    return {
        name: canonicalize_python_static_data(
            value,
            path=str(name),
        )[0]
        for name, value in bindings.items()
    }


def interpret_python_static_value(
    value: Any,
    *,
    path: str,
) -> SpecialCase | None:
    """Reduce a Python-only named integer wrapper to a plain graph constant.

    CPython uses private ``int`` subclasses for symbolic constants in a few
    source modules.  Some of them deliberately set ``__reduce__ = None``;
    retaining such a live wrapper in a resolved ProcessGraph makes ordinary
    graph serialization try to call that non-callable reducer.  The wrapper
    contributes no runtime behavior: its integer value is the program value
    and its name/type are source provenance.

    Keep this recognition at the Python ingestion boundary.  The reducer uses
    the returned special case to create the ordinary ``Constant`` leaf at the
    exact ``Name``/``Attribute`` occurrence and redirects that occurrence to
    it.  No native boundary is introduced and source pursuit is unchanged.
    """

    canonical, origins = canonicalize_python_static_data(value, path=path)
    if not origins:
        return None
    return SpecialCase(
        "Constant",
        {
            "value": canonical,
            "python_static_origins": origins,
        },
        canonical,
    )


def interpret_python_special_case(node: Any) -> SpecialCase | None:
    """Classify Python syntax without performing callable source discovery.

    A non-terminal ``Call`` overlay retains the ordinary call role schema and
    authored argument edges. Its receipt states whether callee source was
    admitted, retained at a boundary, decompiled, or rejected.
    """

    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return SpecialCase("GetAttr", {"attribute": node.attr}, None)

    if not isinstance(node, ast.Call):
        return None

    receipt = extraction_receipt(node)
    attributes = _receipt_attributes(receipt) if receipt is not None else {}
    spelling = _call_spelling(node)
    identity = receipt.get("identity") if receipt is not None else None
    program = resolve_python_identity(identity)
    if program is not None:
        attributes.update({
            "python_identity_program": program.mapping(),
            "python_replacement_kind": program.kind,
        })
        operator = program.direct_operator
        if operator is not None:
            attributes.update(program.direct_attributes)
            attributes["argument_count"] = len(node.args)
            return SpecialCase(operator, attributes, None, terminal=False)

    # Preserve the pre-contract convenience behavior for isolated structural
    # ingestion. Governed compilation always selects by resolved identity.
    if receipt is None and spelling in {"float", "int", "bool"}:
        return SpecialCase(
            spelling, {"cast": spelling}, None, terminal=False,
        )
    if receipt is None and spelling == "print":
        return SpecialCase(
            "stream_publish",
            {"stream": "text", "argument_count": len(node.args)},
            None,
            terminal=False,
        )

    if receipt is None:
        return None

    # Terminal with respect to source pursuit, but not argument dataflow.
    return SpecialCase("Call", attributes, None, terminal=False)


__all__ = [
    "canonicalize_python_static_bindings",
    "canonicalize_python_static_data",
    "extraction_receipt",
    "interpret_python_special_case",
    "interpret_python_static_value",
]
