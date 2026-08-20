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


__all__ = ["extraction_receipt", "interpret_python_special_case"]
