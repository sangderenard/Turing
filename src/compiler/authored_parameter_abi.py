"""Authored parameter annotation ABI contracts (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
from typing import Any, Mapping


def _current_authored_parameter_annotations(
    graph_obj: Any,
) -> dict[str, str]:
    """Return the exact annotation spellings for this function graph."""

    catalogue = dict(
        graph_obj.graph.get("function_parameter_annotations") or {}
    )
    if catalogue and all(isinstance(value, str) for value in catalogue.values()):
        return {str(name): str(value) for name, value in catalogue.items()}
    function_name = str(graph_obj.graph.get("function_name") or "")
    owner = graph_obj.graph.get("method_owner")
    candidates = (
        f"{owner}.{function_name}" if owner else None,
        str(graph_obj.graph.get("qualified_name") or "") or None,
        function_name or None,
    )
    for identity in candidates:
        annotations = catalogue.get(identity) if identity is not None else None
        if isinstance(annotations, Mapping):
            return {
                str(name): str(value)
                for name, value in annotations.items()
            }
    return {}


def _authored_tensor_parameter_value_abi(
    graph_obj: Any,
) -> dict[str, dict[str, Any]]:
    """Return the native value ABI stated by tensor parameter annotations.

    ``torch.Tensor``, ``numpy`` tensor spellings, and ``AbstractTensor`` all
    state the same thing at source ingestion: this parameter is tensor span
    storage.  They do not state extents, so the physical ABI remains a flat
    span while tensor shape SSA carries its logical rank and dimensions.

    This is the parameter counterpart of
    :func:`_authored_annotation_field_receipt`.  Publishing it before
    deployment planning lets exact call-frame propagation carry tensor
    identity into compositional AbstractTensor sources such as ``solve``;
    otherwise those formals are indistinguishable from scalar values until
    after the operations that need their runtime shape have already planned.
    """

    from ..transmogrifier.graph.node_special_cases import (
        tensor_annotation_identity,
    )

    receipts: dict[str, dict[str, Any]] = {}
    for parameter, spelling in _current_authored_parameter_annotations(
        graph_obj
    ).items():
        try:
            annotation = ast.parse(str(spelling), mode="eval").body
        except SyntaxError:
            continue
        identity = tensor_annotation_identity(
            annotation,
            getattr(graph_obj, "python_bindings", {}) or {},
        )
        if identity is None:
            continue
        receipts[str(parameter)] = {
            "storage": "span",
            # Native storage is flat. Logical rank is a runtime tensor-shape
            # value and is never guessed from the frontend annotation. The
            # annotation also does not state a dtype; omitting it preserves
            # the exact dtype carried by the call edge instead of allowing an
            # ``unknown`` placeholder to become a physical integer ABI.
            "rank": 1,
            "mutable": False,
            "python_type": str(identity),
        }
    return receipts


def _authored_sequence_annotation_contract(
    annotation: str,
) -> tuple[str, int, bool, tuple[str, ...]] | None:
    """Interpret only explicit homogeneous collection parameter annotations."""

    try:
        expression = ast.parse(str(annotation), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Subscript):
        return None
    container = (
        expression.value.id
        if isinstance(expression.value, ast.Name)
        else expression.value.attr
        if isinstance(expression.value, ast.Attribute)
        else ""
    )
    element_nodes = (
        tuple(expression.slice.elts)
        if isinstance(expression.slice, ast.Tuple)
        else (expression.slice,)
    )
    scalar_dtypes = {
        "bool": "bool",
        "int": "int64",
        "float": "float64",
        # Repository SSA represents authored text by its deterministic token.
        "str": "int64",
    }

    def scalar_dtype(node: ast.AST) -> str | None:
        spelling = (
            node.id if isinstance(node, ast.Name)
            else node.attr if isinstance(node, ast.Attribute)
            else ""
        )
        return scalar_dtypes.get(str(spelling))

    if container in {"Mapping", "MutableMapping", "Dict", "dict"}:
        if len(element_nodes) != 2:
            return None
        dtypes = tuple(scalar_dtype(node) for node in element_nodes)
        if any(dtype is None for dtype in dtypes):
            return None
        return (
            "unique", 2,
            container in {"MutableMapping", "Dict", "dict"},
            tuple(str(dtype) for dtype in dtypes),
        )
    if container not in {
        "Sequence", "Iterable", "Collection", "List", "list",
        "Tuple", "tuple", "Set", "set", "FrozenSet", "frozenset",
    }:
        return None
    # tuple[T, ...] is one homogeneous sequence. A fixed heterogeneous tuple
    # is a record and must not be flattened through this path.
    if len(element_nodes) == 2 and isinstance(
        element_nodes[1], ast.Constant
    ) and element_nodes[1].value is Ellipsis:
        element_nodes = element_nodes[:1]
    if len(element_nodes) != 1:
        return None
    dtype = scalar_dtype(element_nodes[0])
    if dtype is None:
        return None
    return (
        "unique" if container in {"Set", "set", "FrozenSet", "frozenset"}
        else "duplicates",
        1,
        container in {"List", "list", "Set", "set"},
        (dtype,),
    )


def _authored_text_parameter_transforms(
    graph_obj: Any,
) -> tuple[tuple[int, int, str, str], ...]:
    """Represent each runtime ``str`` formal by its exact UTF-8 sequence."""

    identity = graph_obj.graph.get("identity_table") or {}
    transforms = []
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        try:
            expression = ast.parse(str(annotation), mode="eval").body
        except SyntaxError:
            continue
        spelling = (
            expression.id
            if isinstance(expression, ast.Name)
            else expression.attr
            if isinstance(expression, ast.Attribute)
            else ""
        )
        history = tuple(map(int, identity.get(str(parameter_name), ())))
        if spelling != "str" or not history:
            continue
        source_id = int(history[0])
        transforms.append((
            source_id, source_id, str(parameter_name), "utf8",
        ))
    return tuple(transforms)
