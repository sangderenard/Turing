"""Shallow interpretation of one method into nodus GraphIR.

"Shallow interpretation" is the examination phase with a special recursion
limit: composites are examined, recursion stops at fundamental AbstractTensor
operators and external references, and NOTHING lowers -- no SSA, no SPIR-V,
no DLL. The product is a graph already in the language nodus speaks, handed
over for the destination's own tools to execute.

Scope delivered here: one method, with pursuit INTO module-local helper
functions (their dataflow continues in place, arguments bound to parameters)
and resolution of ``self.NAME`` class-level constants. Recursion stops at
canonical operator spellings and at anything unresolvable in the module --
which surfaces as an explicit ``call`` node, never silently. Cross-module
pursuit (helpers imported from elsewhere) is not yet wired.

Every stage is existing machinery, configured:

  * ``ast_process_graph.build_semantic_ast`` -- the compiler's semantic
    source lowering, which spells Python as the canonical operator
    vocabulary, links def-use through its environment, and records the
    return in ``graph.roots``. Its per-function scope IS the carve-out.
  * the return's ancestor cone -- keeps exactly the dataflow that produces
    the result and drops guard machinery (hook calls, recorders, ``self``).
  * ``process_graph_to_nodus_graph_ir`` -- the emitter nodus already parses
    into registered ``abstract_tensor.<op>`` tool nodes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodGraphIR:
    """One translated method: the wire text and its calling convention."""

    qualname: str
    graph_ir: str
    # Labels of the surviving input nodes, in emission order -- the order a
    # destination must supply argument values in.
    inputs: tuple[str, ...]
    # Operation vocabulary the slice uses (diagnostic ledger). An entry named
    # "call" means something did NOT resolve -- visible, not hidden.
    operations: tuple[str, ...]


def _find_class_and_function(
    tree: ast.Module, qualname: str,
) -> tuple[ast.ClassDef | None, ast.AST]:
    class_name, _, method_name = qualname.rpartition(".")
    if class_name:
        for definition in tree.body:
            if isinstance(definition, ast.ClassDef) and definition.name == class_name:
                for member in definition.body:
                    if (isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and member.name == method_name):
                        return definition, member
        raise LookupError(f"no method {qualname!r} in source")
    for definition in tree.body:
        if (isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
                and definition.name == method_name):
            return None, definition
    raise LookupError(f"no function {qualname!r} in source")


def _module_helpers(tree: ast.Module) -> dict:
    """Module-local functions a method body may be pursued into."""

    return {
        definition.name: definition
        for definition in tree.body
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_constants(owner: ast.ClassDef | None) -> dict:
    """Class-level literal constants, readable as ``self.NAME``."""

    if owner is None:
        return {}
    constants: dict = {}
    for member in owner.body:
        targets = ()
        value = None
        if isinstance(member, ast.Assign):
            targets, value = member.targets, member.value
        elif isinstance(member, ast.AnnAssign) and member.value is not None:
            targets, value = (member.target,), member.value
        for target in targets:
            if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
                constants[target.id] = value.value
    return constants


def method_to_graph_ir(
    source: str, qualname: str, *, filename: str = "<source>",
) -> MethodGraphIR:
    """Shallow-interpret ``qualname`` from ``source`` and emit nodus GraphIR."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .ast_process_graph import build_semantic_ast
    from .nodus_graph_ir import process_graph_to_nodus_graph_ir
    import networkx as nx

    tree = ast.parse(source)
    owner, function = _find_class_and_function(tree, qualname)
    graph = ProcessGraph(0, False, materialize_memory=False)
    build_semantic_ast(
        graph, function, filename=filename,
        helpers=_module_helpers(tree),
        self_constants=_class_constants(owner),
    )

    roots = list(getattr(graph, "roots", ()) or ())
    if not roots:
        raise ValueError(f"{qualname}: semantic lowering recorded no result root")
    root = roots[-1]
    keep = nx.ancestors(graph.G, root) | {root}

    class _Slice:
        pass
    view = _Slice()
    view.G = graph.G.subgraph(keep)

    graph_ir = process_graph_to_nodus_graph_ir(view)

    inputs = tuple(
        str(graph.G.nodes[nid].get("label"))
        for nid in view.G.nodes
        if graph.G.nodes[nid].get("op") == "input"
    )
    operations = tuple(sorted({
        str(graph.G.nodes[nid].get("op"))
        for nid in view.G.nodes
        if graph.G.nodes[nid].get("op") not in ("input", "const", "return")
    }))
    return MethodGraphIR(
        qualname=qualname, graph_ir=graph_ir,
        inputs=inputs, operations=operations,
    )
