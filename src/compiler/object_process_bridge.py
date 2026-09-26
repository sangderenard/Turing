"""Raise one source object method through the ordinary ProcessGraph parser.

This module only selects and identifies a method.  Its body is still ingested
by :class:`ProcessGraph`; no parallel Python-expression compiler is introduced.
The retained object/method record is the upper endpoint for recursive
provenance into BitOps and lower machine representations.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from ..transmogrifier.graph.graph_express2 import ProcessGraph


@dataclass(frozen=True, slots=True)
class ObjectMethodIdentity:
    class_name: str
    method_name: str
    graph_identity: str
    class_source_span: tuple[int, int, int, int]
    method_source_span: tuple[int, int, int, int]
    decorators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RaisedObjectMethod:
    identity: ObjectMethodIdentity
    process_graph: ProcessGraph
    source_filename: str


def _source_span(node: ast.AST) -> tuple[int, int, int, int]:
    return (
        int(getattr(node, "lineno", 0)),
        int(getattr(node, "col_offset", 0)),
        int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
        int(getattr(node, "end_col_offset", getattr(node, "col_offset", 0))),
    )


def raise_object_method_to_process_graph(
    source: str | ast.AST,
    *,
    class_name: str,
    method_name: str,
    source_filename: str = "<object-source>",
    materialize_memory: bool = False,
) -> RaisedObjectMethod:
    """Select one class method and ingest its body as an ordinary process graph."""

    if not class_name or not method_name:
        raise ValueError("class_name and method_name are required")
    tree = ast.parse(source, filename=source_filename) if isinstance(source, str) else source
    if not isinstance(tree, ast.Module):
        tree = ast.Module(body=[tree], type_ignores=[])
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(
            f"expected exactly one class {class_name!r}; found {len(classes)}"
        )
    class_node = classes[0]
    methods = [
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(
            f"expected exactly one method {class_name}.{method_name}; "
            f"found {len(methods)}"
        )
    method = methods[0]
    selected = copy.deepcopy(method)
    # Class decorators apply to method binding, not to the selected method
    # body's dataflow.  Method decorators are retained as provenance metadata
    # but are not executed by graph ingestion.
    selected.decorator_list = []
    selected.name = f"{class_name}__{method_name}"
    selected_module = ast.fix_missing_locations(ast.Module(
        body=[selected],
        type_ignores=[],
    ))
    process = ProcessGraph(materialize_memory=materialize_memory)
    process.build_from_ast(
        selected_module,
        filename=source_filename,
        semantic=True,
    )
    identity = ObjectMethodIdentity(
        class_name,
        method_name,
        f"{class_name}.{method_name}",
        _source_span(class_node),
        _source_span(method),
        tuple(ast.unparse(item) for item in method.decorator_list),
    )
    process.G.graph["object_origin"] = {
        "class_name": identity.class_name,
        "method_name": identity.method_name,
        "graph_identity": identity.graph_identity,
        "class_source_span": identity.class_source_span,
        "method_source_span": identity.method_source_span,
        "decorators": identity.decorators,
        "source_filename": source_filename,
    }
    return RaisedObjectMethod(identity, process, source_filename)


__all__ = [
    "ObjectMethodIdentity",
    "RaisedObjectMethod",
    "raise_object_method_to_process_graph",
]
