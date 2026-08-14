import sympy
import networkx as nx
import numpy as np
import ast
import builtins
import copy
import importlib
import inspect
import math
import os
import pickle
import textwrap
import types
from typing import Any
from sympy import Sum, IndexedBase, Idx, symbols, Function
from ...compiler.bitops import BitTensorMemoryGraph
from colorama import Fore, Style, init
from ..solver_types import Operation, NodeSet, Node, READWRITE, DomainNode, Edge
from ..operator_defs import default_funcs, operator_signatures, role_schemas
from ..ilpscheduler import ILPScheduler
from ..function_table import ExternalFunctionTable, FunctionTable
from .node_special_cases import (
    annotate_types,
    expand_ellipsis_subscripts,
    fold_constant_getattr,
    hoist_walrus_assignments,
    interpret_special_case,
    dissolve_spans,
    tensor_operation_name,
)
from .python_special_cases import (
    extraction_receipt,
    interpret_python_special_case,
)
import colorsys
import random
import time
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessGraphSnapshot:
    """Immutable-by-convention topology copy taken at one graph revision."""

    revision: int
    graph: Any


class ProcessGraphAccessor:
    """Lock-aware read access for live compiler/visualization observers."""

    def __init__(self, owner: "ProcessGraph"):
        self._owner = owner

    def snapshot(self) -> ProcessGraphSnapshot:
        owner = self._owner
        with owner._graph_condition:
            copied = owner.G.copy(as_view=False)
            for node_id, level in owner.levels.items():
                if node_id in copied:
                    copied.nodes[node_id].setdefault("level", level)
            return ProcessGraphSnapshot(owner._graph_revision, copied)

    def wait_for_change(
        self,
        after_revision: int,
        timeout: float | None = None,
    ) -> ProcessGraphSnapshot:
        owner = self._owner
        with owner._graph_condition:
            owner._graph_condition.wait_for(
                lambda: owner._graph_revision > after_revision,
                timeout=timeout,
            )
        return self.snapshot()

    def subscribe(self, callback, *, replay: bool = True):
        """Receive an exact snapshot after each serialized graph mutation."""

        owner = self._owner
        with owner._graph_condition:
            owner._graph_subscribers.append(callback)
        if replay:
            callback(self.snapshot())

        def unsubscribe():
            with owner._graph_condition:
                if callback in owner._graph_subscribers:
                    owner._graph_subscribers.remove(callback)

        return unsubscribe

class _RandomFloatQueue(deque):
    """
    Drop-in stand-in for any queue class used elsewhere in ProcessGraph.
    • .get()   → random float   (so a consumer can keep pulling values)
    • .put(_)  → absorbed       (writer calls succeed but are ignored)
    • __call__ → random float   (if the code treats the consumer as a func)
    """
    __slots__ = ()                      # no per-instance dict → cheap
    def get(self):
        return random.random()
    def put(self, _):
        # silently accept anything; we’re a sink
        pass
    __call__ = get                      # allow direct call pattern

# initialise one global instance once; re-use it everywhere
_DUMMY_QUEUE = _RandomFloatQueue()
SIMD_DEFAULT_CONCURRENCY = 4  # default concurrency for SIMD operations
from collections.abc import Callable


def _annotate_visual_source_owners(tree: ast.AST) -> None:
    """Attach lexical provenance used only by observers and diagnostics.

    The AST remains the compiler authority.  These annotations let optional
    evolution observers attribute a later graph expansion to the source class
    or function that owned the node without reconstructing lexical scope from
    lossy graph edges.
    """

    class OwnerVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []
            self.classes: list[str] = []

        def visit(self, node):
            if isinstance(node, ast.AST):
                node._turing_source_scope = tuple(self.scope)
                node._turing_source_class = (
                    self.classes[-1] if self.classes else None
                )
            return super().visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            node._turing_source_scope = tuple((*self.scope, node.name))
            node._turing_source_class = node.name
            self.scope.append(node.name)
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()
            self.scope.pop()

        def _visit_function(self, node):
            node._turing_source_scope = tuple((*self.scope, node.name))
            node._turing_source_class = (
                self.classes[-1] if self.classes else None
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

    OwnerVisitor().visit(tree)


@dataclass(frozen=True)
class _ASTReferenceAlternatives:
    """Compile-time-only alternatives for conservative source lookup."""

    values: tuple


def _same_ast_reference(left, right):
    if left is right:
        return True
    left_callable = left.__func__ if inspect.ismethod(left) else left
    right_callable = right.__func__ if inspect.ismethod(right) else right
    if left_callable is right_callable:
        return True
    if isinstance(left, _ASTReferenceAlternatives) and isinstance(
        right,
        _ASTReferenceAlternatives,
    ):
        return len(left.values) == len(right.values) and all(
            _same_ast_reference(left_value, right_value)
            for left_value, right_value in zip(left.values, right.values)
        )
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        return type(left) is type(right) and left == right
    return False


def _merge_ast_reference(left, right):
    values = []
    for value in (
        *(left.values if isinstance(left, _ASTReferenceAlternatives) else (left,)),
        *(right.values if isinstance(right, _ASTReferenceAlternatives) else (right,)),
    ):
        if not any(_same_ast_reference(value, existing) for existing in values):
            values.append(value)
    return values[0] if len(values) == 1 else _ASTReferenceAlternatives(tuple(values))


def _merge_ast_bindings(bindings, additions):
    merged = dict(bindings)
    changed = False
    for name, value in additions.items():
        if value is None:
            continue
        if name not in merged:
            merged[name] = value
            changed = True
            continue
        combined = _merge_ast_reference(merged[name], value)
        if not _same_ast_reference(combined, merged[name]):
            merged[name] = combined
            changed = True
    return merged, changed


def _ast_aggregate_kind(value):
    """Return a proven aggregate storage kind from source-binding identity.

    Local literal recovery deliberately records the constructor class itself
    (``items = []`` -> ``items: list``), because no ingestion-time container
    object is created.  Call binding then forwards that exact identity onto a
    callee formal.  Treat the class and a rare already-materialized structural
    value equivalently, and accept alternatives only when every call site
    proves the same kind.
    """

    aggregate_types = {list, set, dict, tuple, bytes, bytearray}
    if isinstance(value, _ASTReferenceAlternatives):
        kinds = {_ast_aggregate_kind(option) for option in value.values}
        kinds.discard(None)
        return kinds.pop() if len(kinds) == 1 and all(
            _ast_aggregate_kind(option) is not None for option in value.values
        ) else None
    if value in aggregate_types if isinstance(value, type) else False:
        return value.__name__
    value_type = type(value)
    return value_type.__name__ if value_type in aggregate_types else None


def _class_field_reference(owner, attribute, seen):
    """Infer a field's constructor provenance without constructing an object."""

    key = (owner, attribute)
    if key in seen:
        return None
    definition = _source_ast_definition(owner)
    if not isinstance(definition, ast.ClassDef):
        return None
    field_bindings = _import_ast_bindings(
        definition,
        _ast_definition_bindings(owner),
        package=str(getattr(inspect.getmodule(owner), "__package__", "") or ""),
    )
    field_bindings.setdefault("self", owner)
    field_bindings.setdefault("cls", owner)
    values = []
    unresolved = False
    next_seen = {*seen, key}
    for statement in ast.walk(definition):
        if isinstance(statement, ast.Assign):
            value = statement.value
            matched = any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
                and target.attr == attribute
                for target in statement.targets
            )
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            target = statement.target
            matched = (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
                and target.attr == attribute
            )
        else:
            continue
        if value is None or not matched:
            continue
        resolved = _resolve_ast_value_reference(value, field_bindings, next_seen)
        if resolved is None:
            unresolved = True
        else:
            values.append(resolved)
    if unresolved or not values:
        return None
    result = values[0]
    for value in values[1:]:
        result = _merge_ast_reference(result, value)
    return result


def _resolve_ast_value_reference(expression, bindings, seen=frozenset()):
    """Resolve source identity only; never evaluate application behavior."""

    if isinstance(expression, ast.Constant) and expression.value is None:
        # ``None`` is a known union alternative. Represent its structural type
        # so it is distinct from failure to resolve source provenance.
        return type(None)
    if isinstance(expression, ast.Call):
        target = _resolve_ast_parent_reference(expression.func, bindings, seen)
        if target is None and isinstance(expression.func, ast.Name):
            target = {
                "bytearray": bytearray,
                "bytes": bytes,
                "dict": dict,
                "list": list,
                "set": set,
                "tuple": tuple,
            }.get(expression.func.id)
        return target if inspect.isclass(target) else None
    if isinstance(expression, ast.IfExp):
        left = _resolve_ast_value_reference(expression.body, bindings, seen)
        right = _resolve_ast_value_reference(expression.orelse, bindings, seen)
        if left is None or right is None:
            return None
        return _merge_ast_reference(left, right)
    return _resolve_ast_parent_reference(expression, bindings, seen)


def _resolve_ast_parent_reference(expression, bindings, seen=frozenset()):
    """Resolve only enough Python context to obtain a callee's source AST."""

    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    if not isinstance(expression, ast.Attribute):
        return None
    owner = _resolve_ast_parent_reference(expression.value, bindings, seen)
    if owner is None:
        return None
    if isinstance(owner, _ASTReferenceAlternatives):
        resolved = tuple(
            value
            for value in (
            _resolve_ast_parent_reference(
                ast.Attribute(value=ast.Name(id="owner"), attr=expression.attr),
                {"owner": candidate},
                seen,
            )
            for candidate in owner.values
            )
            if value is not None
        )
        # Attribute lookup itself narrows a union to alternatives that carry
        # that field/method. An alternative lacking it would raise before this
        # source dependency could run; it is not a competing method target.
        if not resolved:
            return None
        result = resolved[0]
        for value in resolved[1:]:
            if not _same_ast_reference(result, value):
                return None
        return result
    try:
        return getattr(owner, expression.attr)
    except AttributeError:
        if inspect.isclass(owner):
            return _class_field_reference(owner, expression.attr, seen)
        return None


_SOURCE_AST_TEMPLATE_CACHE = {}


def _source_ast_definition(value):
    """Return a fresh AST definition without retaining the Python callable."""

    if inspect.ismethod(value):
        value = value.__func__
    if not (inspect.isfunction(value) or inspect.isclass(value)):
        return None
    code = getattr(value, "__code__", None)
    try:
        source_file = str(inspect.getsourcefile(value) or "")
    except (OSError, TypeError):
        return None
    cache_key = (
        str(getattr(value, "__module__", "")),
        str(getattr(value, "__qualname__", getattr(value, "__name__", ""))),
        source_file,
        int(getattr(code, "co_firstlineno", -1)),
        id(code) if code is not None else id(value),
    )
    cached = _SOURCE_AST_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    try:
        source = textwrap.dedent(inspect.getsource(value))
    except (OSError, TypeError):
        return None
    parsed = ast.parse(
        source,
        filename=str(inspect.getsourcefile(value) or "<resolved-parent>"),
    )
    definition = next(
        (
            statement
            for statement in parsed.body
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        ),
        None,
    )
    if definition is not None:
        _SOURCE_AST_TEMPLATE_CACHE[cache_key] = definition
        return copy.deepcopy(definition)
    return None


def _attach_external_methods(retained_class, definition):
    """Materialise methods bound onto a class outside its own body.

    ``AbstractTensor.reshape = _reshape_methods.reshape`` in the defining
    module makes ``reshape`` a real method whose ``def`` lives in another
    file, so ingesting the class's own source text alone never sees it and
    the whole aliased family is missing from every downstream method table.
    Pull each such function's own definition in under the attribute name it
    was bound to, so it becomes an ordinary method of the retained class.
    Only a plain name or dotted reference is treated as a bound method; a
    literal or call result is an ordinary class attribute, not an alias.
    """

    if not isinstance(definition, ast.ClassDef):
        return definition
    module = inspect.getmodule(retained_class)
    if module is None:
        return definition
    try:
        module_tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return definition
    present = {
        member.name
        for member in definition.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for statement in module_tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not isinstance(statement.value, (ast.Name, ast.Attribute)):
            continue
        for target in statement.targets:
            if (
                not isinstance(target, ast.Attribute)
                or not isinstance(target.value, ast.Name)
                or target.value.id != retained_class.__name__
                or target.attr in present
            ):
                continue
            method = _source_ast_definition(
                getattr(retained_class, target.attr, None)
            )
            if not isinstance(
                method, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            # The bound attribute name is the method's name here; the source
            # function may have been defined under a different one.
            method.name = target.attr
            definition.body.append(method)
            present.add(target.attr)
    return definition


def _filter_discovered_definition(definition, module):
    """Discard unreferenced class surface before recursive discovery."""

    if not isinstance(definition, ast.ClassDef):
        return definition
    requested_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    methods = {
        member.name: member
        for member in definition.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Keep the two mandatory constructor spellings explicit in the authored
    # membership data.  A set-union expression here was faithfully ingested as
    # ``bitor``; later numerical lowering then had no right to project this
    # structural set operation into scalar arithmetic.
    retained_methods = set(requested_attributes)
    retained_methods.add("__new__")
    retained_methods.add("__init__")
    pending = list(retained_methods)
    while pending:
        method_name = pending.pop()
        method = methods.get(method_name)
        if method is None:
            continue
        dependencies = {
            node.func.attr
            for node in ast.walk(method)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
            )
        }
        for dependency in dependencies - retained_methods:
            retained_methods.add(dependency)
            pending.append(dependency)
    retained_body = [
        member
        for member in definition.body
        if isinstance(member, (ast.Assign, ast.AnnAssign))
        or (
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name
            in retained_methods
        )
    ]
    if not retained_body:
        retained_body = [ast.copy_location(ast.Pass(), definition)]
    filtered = ast.ClassDef(
        name=definition.name,
        bases=[],
        keywords=[],
        body=retained_body,
        decorator_list=[],
        type_params=list(getattr(definition, "type_params", ())),
    )
    return ast.copy_location(filtered, definition)


def instance_attribute_slot(attributes, attribute_name):
    """The deterministic instance-storage slot for one class field, or ``None``.

    ``attributes`` is a ``class_schema["attributes"]``-shaped sequence (see
    ``_class_schema_from_ast``): each item has ``name``/``storage``, in
    source declaration order.  Every ``storage == "instance"`` attribute
    gets the next integer, in that order -- the single authoritative
    computation both ``build_class_navigation_table``
    (``shell_reference_tables.py``) and ingestion-time attribute-operator
    construction (``topological_reducer.py``'s ``bind_target``/
    ``resolve_expression``) must share, so a field's real position in its
    class's layout is computed exactly once, not reimplemented twice with
    room to drift.  Returns ``None`` for a class-level or method member --
    those are not instance storage and have no slot.
    """

    slot = 0
    for attribute in attributes:
        if str(attribute["storage"]) != "instance":
            continue
        if str(attribute["name"]) == attribute_name:
            return slot
        slot += 1
    return None


def _class_schema_from_ast(definition):
    """Record only class syntax already being ingested by ``ProcessGraph``.

    This is descriptive AST metadata: attributes and direct method definitions
    with source-derived program identifiers.  It deliberately does not infer
    execution, data flow, process edges, or a runtime object model.
    """

    attributes = []
    seen_attributes = set()

    def add_attribute(name, annotation, storage):
        if name in seen_attributes:
            return
        seen_attributes.add(name)
        attributes.append({
            "name": name,
            "identity": f"{definition.name}.{name}",
            "storage": storage,
            "annotation": None if annotation is None else ast.unparse(annotation),
            "permissions": (),
        })

    methods_by_name = {}
    for member in definition.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods_by_name[member.name] = member
    methods = list(methods_by_name.values())
    for member in definition.body:
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            add_attribute(member.target.id, member.annotation, "class")
        elif isinstance(member, ast.Assign):
            for target in member.targets:
                if isinstance(target, ast.Name):
                    add_attribute(target.id, None, "class")
    for method in methods:
        for statement in ast.walk(method):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        add_attribute(target.attr, None, "instance")
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    add_attribute(
                        target.attr,
                        statement.annotation,
                        "instance",
                    )
    source_identity = getattr(definition, "_python_source_identity", None)
    class_identity = (
        ".".join(part for part in source_identity if part)
        if source_identity is not None
        else definition.name
    )
    return {
        "class_name": definition.name,
        "class_identity": class_identity,
        "class_node_id": id(definition),
        "permissions": (),
        "attributes": tuple(attributes),
        "methods": tuple({
            "name": method.name,
            "graph_identity": f"{definition.name}.{method.name}",
            "ast_node_id": id(method),
            "parameters": tuple(
                argument.arg for argument in method.args.args
            ),
            "permissions": (),
        } for method in methods),
    }


def _ast_qualified_name(expression):
    """Return a dotted source name without importing or executing it."""

    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        owner = _ast_qualified_name(expression.value)
        return f"{owner}.{expression.attr}" if owner else expression.attr
    return None


def _state_machine_schema_from_ast(definition):
    """Describe an explicitly marked AbstractTensor state-machine class."""

    bases = tuple(
        name
        for base in definition.bases
        if (name := _ast_qualified_name(base)) is not None
    )
    marker = next(
        (
            name for name in bases
            if name.rsplit(".", 1)[-1] == "AbstractTensorStateMachine"
        ),
        None,
    )
    if marker is None:
        return None
    methods = tuple(
        member.name
        for member in definition.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return {
        "class_name": definition.name,
        "identity": definition.name,
        "marker": marker,
        "bases": bases,
        "transition_identity": (
            f"{definition.name}.transition"
            if "transition" in methods
            else None
        ),
        "ast_node_id": id(definition),
    }


def _map_ir_from_ast(tree):
    """Build the map member that accompanies numeric and control IR.

    Empty permission tuples are explicit: AST ingestion has found identities
    but no permission declarations. Policy is not guessed from Python naming
    conventions or method bodies.
    """

    objects = tuple(
        _class_schema_from_ast(definition)
        for definition in ast.walk(tree)
        if isinstance(definition, ast.ClassDef)
    )
    state_machines = tuple(
        state_machine
        for definition in ast.walk(tree)
        if isinstance(definition, ast.ClassDef)
        if (state_machine := _state_machine_schema_from_ast(definition))
        is not None
    )
    module_annotations = tuple(
        {
            "name": statement.target.id,
            "identity": statement.target.id,
            "annotation": ast.unparse(statement.annotation),
            "value": (
                None if statement.value is None
                else ast.unparse(statement.value)
            ),
            "ast_node_id": id(statement),
        }
        for statement in getattr(tree, "body", ())
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    )
    class_annotations = tuple(
        {
            "class_name": definition.name,
            "members": tuple(
                {
                    "name": statement.target.id,
                    "identity": f"{definition.name}.{statement.target.id}",
                    "annotation": ast.unparse(statement.annotation),
                    "value": (
                        None if statement.value is None
                        else ast.unparse(statement.value)
                    ),
                    "ast_node_id": id(statement),
                }
                for statement in definition.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            ),
        }
        for definition in getattr(tree, "body", ())
        if isinstance(definition, ast.ClassDef)
    )
    function_annotations = tuple(
        {
            "identity": (
                f"{definition.name}.{function.name}"
                if isinstance(definition, ast.ClassDef)
                else function.name
            ),
            "locals": tuple(
                {
                    "name": statement.target.id,
                    "annotation": ast.unparse(statement.annotation),
                    "value": (
                        None if statement.value is None
                        else ast.unparse(statement.value)
                    ),
                    "ast_node_id": id(statement),
                }
                for statement in ast.walk(function)
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            ),
        }
        for definition in getattr(tree, "body", ())
        for function in (
            (definition,)
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
            else tuple(
                member for member in definition.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            if isinstance(definition, ast.ClassDef)
            else ()
        )
    )
    schema_roots = tuple(
        item["ast_node_id"]
        for item in module_annotations
    ) + tuple(
        member["ast_node_id"]
        for record in class_annotations
        for member in record["members"]
    )
    schema_statements = tuple(
        statement
        for statement in getattr(tree, "body", ())
        if isinstance(statement, ast.AnnAssign)
    ) + tuple(
        statement
        for definition in getattr(tree, "body", ())
        if isinstance(definition, ast.ClassDef)
        for statement in definition.body
        if isinstance(statement, ast.AnnAssign)
    )
    schema_node_ids = tuple(dict.fromkeys(
        id(node)
        for statement in schema_statements
        for node in ast.walk(statement)
    ))
    return {
        "schema": {
            "module": {"annotations": module_annotations},
            "classes": class_annotations,
            "functions": function_annotations,
        },
        "schema_roots": schema_roots,
        "schema_node_ids": schema_node_ids,
        "objects": objects,
        "state_machines": state_machines,
        "graphs": tuple(
            {
                "identity": method["graph_identity"],
                "ast_node_id": method["ast_node_id"],
                "permissions": method["permissions"],
            }
            for object_schema in objects
            for method in object_schema["methods"]
        ),
        "permissions": (),
    }


def _ast_definition_bindings(value):
    """Obtain names needed only while recursively discovering source AST."""

    if inspect.ismethod(value):
        value = value.__func__
    module = inspect.getmodule(value)
    bindings = {}
    if inspect.isfunction(value):
        # getclosurevars returns only names actually referenced by this
        # function.  Copying vars(module) here makes unrelated module members
        # look eligible for discovery and destroys lexical ownership.
        closure = inspect.getclosurevars(value)
        bindings.update(closure.builtins)
        bindings.update(closure.globals)
        bindings.update(closure.nonlocals)
        owner = module
        for component in str(value.__qualname__).split(".")[:-1]:
            if component == "<locals>":
                owner = None
                break
            owner = getattr(owner, component, None)
            if owner is None:
                break
        if inspect.isclass(owner):
            definition = _source_ast_definition(value)
            if isinstance(
                definition,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                parameters = [
                    *definition.args.posonlyargs,
                    *definition.args.args,
                ]
                if parameters:
                    bindings[parameters[0].arg] = owner
    elif inspect.isclass(value) and module is not None:
        definition = _source_ast_definition(value)
        referenced_names = (
            {
                member.id
                for member in ast.walk(definition)
                if isinstance(member, ast.Name)
                and isinstance(member.ctx, ast.Load)
            }
            if definition is not None
            else set()
        )
        module_bindings = vars(module)
        bindings.update(
            {
                name: module_bindings[name]
                for name in referenced_names
                if name in module_bindings
            }
        )
    return bindings


def _ast_call_argument_bindings(call, target, call_bindings):
    """Map resolvable actual source identities onto a callee's formal names."""

    definition = _source_ast_definition(target)
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    parameters = [*definition.args.posonlyargs, *definition.args.args]
    actuals = []
    if isinstance(call.func, ast.Attribute):
        receiver = _resolve_ast_value_reference(call.func.value, call_bindings)
        actuals.append(receiver)
    actuals.extend(
        _resolve_ast_value_reference(argument, call_bindings)
        for argument in call.args
    )
    def source_identity(actual):
        if actual is None or isinstance(actual, _ASTReferenceAlternatives):
            return actual
        if inspect.isclass(actual) or inspect.ismodule(actual):
            return actual
        owner = type(actual)
        return owner if _source_ast_definition(owner) is not None else actual

    resolved = {
        parameter.arg: source_identity(actual)
        for parameter, actual in zip(parameters, actuals)
        if actual is not None
    }
    parameter_names = {parameter.arg for parameter in parameters}
    resolved.update(
        {
            keyword.arg: source_identity(value)
            for keyword in call.keywords
            if keyword.arg in parameter_names
            and (
                value := _resolve_ast_value_reference(
                    keyword.value,
                    call_bindings,
                )
            )
            is not None
        }
    )
    return resolved


def _ast_local_constructor_bindings(definition, bindings):
    """Recover local storage kinds from explicit constructors and literals."""

    def lexical_nodes():
        pending = list(reversed(getattr(definition, "body", ())))
        while pending:
            node = pending.pop()
            yield node
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            pending.extend(reversed(tuple(ast.iter_child_nodes(node))))

    resolved = dict(bindings)
    changed = True
    while changed:
        changed = False
        for statement in lexical_nodes():
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            reference = None
            if isinstance(value, ast.List):
                reference = list
            elif isinstance(value, ast.Set):
                reference = set
            elif isinstance(value, ast.Dict):
                reference = dict
            elif isinstance(value, ast.Tuple):
                reference = tuple
            elif value is not None:
                reference = _resolve_ast_value_reference(value, resolved)
            if reference is None:
                continue
            if isinstance(statement, ast.Assign):
                for target_node in statement.targets:
                    if (
                        isinstance(target_node, ast.Name)
                        and target_node.id not in resolved
                    ):
                        resolved[target_node.id] = reference
                        changed = True
            elif (
                isinstance(statement.target, ast.Name)
                and statement.target.id not in resolved
            ):
                resolved[statement.target.id] = reference
                changed = True
    return resolved


def _import_ast_bindings(tree, bindings, package=None):
    """Make imports and literal module constants visible to source discovery."""

    resolved = dict(bindings)
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            for imported in statement.names:
                try:
                    value = importlib.import_module(imported.name)
                except ImportError:
                    continue
                resolved[imported.asname or imported.name.split(".")[0]] = value
        elif isinstance(statement, ast.ImportFrom):
            module_name = (
                "." * int(statement.level)
                + str(statement.module or "")
            )
            try:
                module = importlib.import_module(module_name, package=package)
            except (ImportError, TypeError, ValueError):
                continue
            for imported in statement.names:
                try:
                    resolved[imported.asname or imported.name] = getattr(
                        module,
                        imported.name,
                    )
                except AttributeError:
                    continue
    # A literal module assignment is already a fully reduced compiler fact.
    # Retain it beside imported bindings so function subgraphs can resolve
    # globals such as chunk sizes without executing module code or fabricating
    # a runtime input.
    # Only module assignments are lexical globals.  This helper also receives
    # discovered FunctionDef and ClassDef nodes; treating their local or class
    # assignments as globals lets a class field default (for example
    # ``symbols = None``) shadow a method parameter with the same spelling.
    module_body = tree.body if isinstance(tree, ast.Module) else ()
    for statement in module_body:
        if isinstance(statement, ast.Assign):
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value_node = statement.value
        else:
            continue
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (TypeError, ValueError, SyntaxError):
            continue
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    resolved[target.id] = value
        elif isinstance(statement.target, ast.Name):
            resolved[statement.target.id] = value
    return resolved


def _heat_escape(level: int) -> str:
    """An ANSI truecolor escape that gets warmer and brighter every level
    the upward search climbs -- cool blue at the first pass, through
    yellow, to hot white the deeper it has to go looking for a definition
    this ``while True:`` loop was never given a constant bound to stop at."""

    t = max(0.0, min(level / 8.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        r, g, b = int(70 + u * 185), int(120 + u * 135), int(220 - u * 170)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 255, int(255 - u * 55) if u < 1 else 255, int(50 + u * 205)
    return f"\x1b[1m\x1b[38;2;{r};{g};{b}m"


_HEAT_RESET = "\x1b[0m"


def _depth_escape(depth: int) -> str:
    """The downward counterpart to ``_heat_escape``: green through cyan to
    violet, deepening (not brightening) the further this descends into a
    just-discovered definition's own nested functions/classes. Cool/violet
    for descent, warm/bright for the upward search -- the two are never
    mistakable for each other even color-blind, because they also use
    different marker glyphs (``v`` vs ``!``)."""

    t = max(0.0, min(depth / 6.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        r, g, b = int(60 + u * 20), int(190 - u * 10), int(110 + u * 110)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = int(80 + u * 120), int(180 - u * 130), int(220 + u * 15)
    return f"\x1b[1m\x1b[38;2;{r};{g};{b}m"


def _safe_repr(value, *, max_length=180):
    if value is None:
        return "None"
    try:
        text = repr(value)
    except Exception as exc:
        return f"<repr-error {exc!r}>"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _walk_definitions_with_depth(node, depth=1):
    """Yield ``(member, depth)`` for every nested FunctionDef/AsyncFunctionDef
    /ClassDef inside ``node``, depth-first, unlike ``ast.walk`` which gives
    no depth at all. Depth 1 is a direct child definition of ``node``."""

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield child, depth
            yield from _walk_definitions_with_depth(child, depth + 1)
        else:
            yield from _walk_definitions_with_depth(child, depth)


class _ConsumedGeneratorLoopLowerer(ast.NodeTransformer):
    """Turn a directly consumed one-clause generator into ordinary control.

    ``for result in (expr for item in source if predicate): body`` has no
    need for a runtime generator object: its producer and consumer are one
    lexical loop.  Preserve the filter as an ordinary ``if`` and the yielded
    expression as an ordinary assignment, leaving all scheduling to the
    existing retained-loop planner. Multi-clause generators remain explicit
    until nested-loop break/else ownership is represented.
    """

    def visit_For(self, node):
        node = self.generic_visit(node)
        iterator = node.iter
        if not (
            isinstance(iterator, ast.GeneratorExp)
            and len(iterator.generators) == 1
            and not iterator.generators[0].is_async
        ):
            return node
        generator = iterator.generators[0]
        assignment = ast.copy_location(
            ast.Assign(targets=[node.target], value=iterator.elt),
            iterator.elt,
        )
        body = [assignment, *node.body]
        for predicate in reversed(generator.ifs):
            body = [ast.copy_location(
                ast.If(test=predicate, body=body, orelse=[]),
                predicate,
            )]
        lowered = ast.For(
            target=generator.target,
            iter=generator.iter,
            body=body,
            orelse=node.orelse,
            type_comment=node.type_comment,
        )
        return ast.copy_location(lowered, node)


def _lower_consumed_generator_loops(tree):
    lowered = _ConsumedGeneratorLoopLowerer().visit(tree)
    ast.fix_missing_locations(lowered)
    return lowered


def _expand_unresolved_ast_parents(
    tree,
    bindings,
    *,
    package=None,
    include=None,
    pursuit_roots=None,
    tensor_code_references=None,
    profile_verbose=False,
    progress=None,
):
    """Discover missing source definitions and return AST parent links.

    The returned definitions are ordinary AST objects.  They are subsequently
    ingested by ``ProcessGraph.build_graph``; no callable is installed in a
    function table or deferred to runtime.

    ``progress``, if given, is called once per node discovered by the
    upward search with one already-colored, already-``!``-decorated line.
    The color and exclamation count both scale with ``pass_index`` -- this
    loop is a genuine ``while True:``, not bounded by any constant depth,
    and each additional pass means the search climbed one level further
    looking for a source definition that was not where the previous pass
    expected it.

    Also returns ``root_bindings`` -- ``bindings`` plus every name this
    call actually resolved via real imports (``_import_ast_bindings``,
    ``importlib.import_module`` against ``package``).  The caller must
    store it back onto ``self.python_bindings`` for later stages (name
    resolution during reduction, ``static_bindings`` in
    ``topological_reducer.py``) to see it -- this function's own use of
    it (discovering additional source definitions) is real but internal,
    and previously never escaped this call at all.
    """

    if isinstance(tree, ast.Module):
        module = tree
    else:
        module = ast.Module(body=[tree], type_ignores=[])
        ast.fix_missing_locations(module)

    def emit(message):
        if profile_verbose:
            print(message, flush=True)
        if progress is not None:
            progress(message)

    root_bindings = _import_ast_bindings(module, bindings, package=package)
    # Built-ins are ordinary lexical fallback bindings in Python.  Put them
    # through the same source/host-code implementation resolver as imports and
    # globals so a source-less builtin can be decompiled rather than becoming
    # an unexplained callee token later in reduction.
    for builtin_name, builtin_value in vars(builtins).items():
        root_bindings.setdefault(str(builtin_name), builtin_value)
    tensor_code_references = {
        str(name): reference
        for name, reference in dict(tensor_code_references or {}).items()
    }

    def call_target(call, call_bindings):
        """Resolve lexical storage/method identity before tensor references.

        The mapping contains source-code references, not runtime handlers.  Its
        callable is used only long enough to retrieve the referenced AST and
        its definition bindings; the ProcessGraph/FunctionTable owns the body
        after ingestion.
        """

        lexical_target = _resolve_ast_parent_reference(call.func, call_bindings)
        if lexical_target is not None:
            return lexical_target
        tensor_name = tensor_operation_name(call)
        if tensor_name is not None:
            referenced = tensor_code_references.get(str(tensor_name))
            if referenced is not None:
                call._tensor_code_reference = str(tensor_name)
                return referenced
        return None
    # Resolution context belongs to the definition containing a call.  Never
    # merge module globals from one discovered function into another
    # function's namespace: globals are lookup material, not dependency edges.
    node_bindings = {id(member): root_bindings for member in ast.walk(module)}
    definitions = [
        node
        for node in ast.walk(module)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]
    definitions_by_name = {}
    for definition in definitions:
        definition_bindings = _ast_local_constructor_bindings(
            definition,
            root_bindings,
        )
        definition._python_bindings = definition_bindings
        for member in ast.walk(definition):
            node_bindings[id(member)] = definition_bindings
        definitions_by_name.setdefault(definition.name, []).append(definition)

    unavailable_identities = set()
    target_definitions = {}
    target_bindings = {}
    started = time.perf_counter()
    def lexical_calls(definition):
        """Calls executed by one definition, excluding nested definitions.

        A nested function/class body is source available to its enclosing
        definition, but it is not executed merely because the enclosing
        definition runs.  Its calls join the worklist only when a reachable
        call admits that nested definition itself.
        """

        calls = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                calls.append(node)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                if node is definition:
                    for statement in node.body:
                        self.visit(statement)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                if node is definition:
                    for statement in node.body:
                        if not isinstance(statement, (
                            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                        )):
                            self.visit(statement)

            def visit_Lambda(self, node):
                return

        Visitor().visit(definition)
        return tuple(calls)

    def root_definition(identity):
        parts = tuple(part for part in str(identity).split(".") if part)
        if not parts:
            return None
        candidates = [
            definition for definition in module.body
            if isinstance(definition, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ))
            and definition.name == parts[0]
        ]
        for part in parts[1:]:
            candidates = [
                member
                for candidate in candidates
                for member in getattr(candidate, "body", ())
                if isinstance(member, (
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ))
                and member.name == part
            ]
        return candidates[0] if len(candidates) == 1 else None

    roots = tuple(dict.fromkeys(map(str, pursuit_roots or ())))
    if roots:
        root_definitions = tuple(root_definition(root) for root in roots)
        missing_roots = tuple(
            root for root, definition in zip(roots, root_definitions)
            if definition is None
        )
        if missing_roots:
            raise ValueError(
                "source-pursuit roots are absent or ambiguous: "
                f"{missing_roots!r}"
            )
        pending_calls = deque(
            call
            for definition in root_definitions
            for call in lexical_calls(definition)
        )
    else:
        pending_calls = deque(
            node for node in ast.walk(module) if isinstance(node, ast.Call)
        )
    all_calls = list(pending_calls)
    call_owners = {}
    for definition in definitions:
        for call in (
            member for member in ast.walk(definition)
            if isinstance(member, ast.Call)
        ):
            call_owners[id(call)] = definition
    binding_revisions = {}
    processed_revisions = {}
    work_items = 0
    contract_source_limits = (
        dict(getattr(include, "limits", {}).get("python_source") or {})
        if include is not None else {}
    )
    max_work_items = int(contract_source_limits.get("max_work_items", 0))
    max_dependency_depth = int(
        contract_source_limits.get("max_dependency_depth", 0)
    )
    call_depths = {id(call): 0 for call in pending_calls}
    definition_depths = {id(definition): 0 for definition in definitions}

    def definition_calls(definition):
        return lexical_calls(definition)

    def install_definition_bindings(definition, definition_bindings):
        definition._python_bindings = definition_bindings
        aggregate_kinds = dict(
            getattr(definition, "_python_aggregate_binding_kinds", {}) or {}
        )
        aggregate_kinds.update({
            str(name): kind
            for name, value in definition_bindings.items()
            if (kind := _ast_aggregate_kind(value)) is not None
        })
        definition._python_aggregate_binding_kinds = aggregate_kinds
        for member in ast.walk(definition):
            node_bindings[id(member)] = definition_bindings

    def requeue_definition(definition):
        definition_id = id(definition)
        binding_revisions[definition_id] = (
            binding_revisions.get(definition_id, 0) + 1
        )
        calls = definition_calls(definition)
        definition_depth = definition_depths.get(id(definition), 0)
        for call in calls:
            call_owners[id(call)] = definition
            call_depths[id(call)] = definition_depth
        all_calls.extend(calls)
        pending_calls.extend(calls)

    while pending_calls:
        if max_work_items and work_items >= max_work_items:
            raise RuntimeError(
                "extraction contract python max_work_items exceeded: "
                f"{max_work_items}"
            )
        node = pending_calls.popleft()
        work_items += 1
        call_depth = call_depths.get(id(node), 0)
        call_bindings = node_bindings.get(id(node), root_bindings)
        owner_definition = call_owners.get(id(node))
        revision = binding_revisions.get(
            id(owner_definition) if owner_definition is not None else 0,
            0,
        )
        process_key = (id(node), revision)
        if process_key in processed_revisions:
            continue
        processed_revisions[process_key] = True
        call_bindings = node_bindings.get(id(node), root_bindings)
        target = call_target(node, call_bindings)
        identity_target = target.__func__ if inspect.ismethod(target) else target
        if not callable(identity_target):
            continue
        extraction_decision = (
            include.decide(identity_target)
            if include is not None and hasattr(include, "decide")
            else None
        )
        if extraction_decision is not None:
            node._extraction_contract = extraction_decision.receipt()
        if include is not None and not (
            extraction_decision.ingest_parent
            if extraction_decision is not None
            else include(identity_target)
        ):
            # A rich extraction contract is authoritative. In particular,
            # intrinsic/use-native/host-call choices must never fall through
            # to the historical "perhaps decompile it" host-code resolver.
            if extraction_decision is not None:
                continue
            from ...compiler.host_code_modules import (
                resolve_host_code_identity,
            )
            if resolve_host_code_identity(identity_target) is None:
                continue
        identity = (
            str(getattr(identity_target, "__module__", "")),
            str(getattr(
                identity_target,
                "__qualname__",
                getattr(identity_target, "__name__", ""),
            )),
        )
        argument_bindings = _ast_call_argument_bindings(
            node, identity_target, call_bindings,
        )
        if identity in target_definitions:
            definition = target_definitions[identity]
            combined, bindings_changed = _merge_ast_bindings(
                target_bindings.get(identity, {}), argument_bindings,
            )
            if bindings_changed:
                target_bindings[identity] = combined
                install_definition_bindings(definition, combined)
                requeue_definition(definition)
            continue
        if identity in unavailable_identities:
            continue

        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        else:
            call_name = None
        existing = definitions_by_name.get(call_name, ())
        if len(existing) == 1:
            definition = existing[0]
            target_definitions[identity] = definition
            definition_bindings, changed = _merge_ast_bindings(
                _ast_local_constructor_bindings(
                    definition, _ast_definition_bindings(identity_target),
                ),
                argument_bindings,
            )
            target_bindings[identity] = definition_bindings
            install_definition_bindings(definition, definition_bindings)
            # The old module-wide initial queue hid this admission edge: a
            # lexical callee's calls had already been queued even when its
            # argument bindings did not change.  Reachable pursuit must enqueue
            # the body exactly when the call first admits the definition.
            requeue_definition(definition)
            continue

        source_target = identity_target
        emit(
            f"[ast-parent] call line={getattr(node, 'lineno', '?')} col={getattr(node, 'col_offset', '?')} "
            f"resolving {call_name or '<unknown>'} -> {identity[1]} "
            f"source={_safe_repr(getattr(node.func, 'id', None) or getattr(node.func, 'attr', None) or getattr(node.func, 'name', None))}"
        )
        source_definition = _source_ast_definition(source_target)
        if source_definition is None:
            # Readable source is only the first implementation source.  A
            # source-less host callable may still have a complete binary body
            # which the repository machine-code pipeline can raise to SSA.
            # Attach that immutable module to this exact call occurrence; the
            # reducer gives it an ordinary FunctionTable reference later.
            from ...compiler.host_code_modules import (
                extract_host_code_library,
                materialize_host_code_library,
            )

            decompile_parameters = (
                dict(extraction_decision.parameters)
                if extraction_decision is not None else {}
            )
            host_library = extract_host_code_library(
                source_target,
                **({
                    "max_functions": int(decompile_parameters["max_functions"]),
                    "max_total_bytes": int(decompile_parameters["max_total_bytes"]),
                    "max_dependency_depth": int(
                        decompile_parameters["max_dependency_depth"]
                    ),
                } if extraction_decision is not None else {}),
            )
            if host_library is not None:
                host_root = host_library.root
                node._host_ssa_module = materialize_host_code_library(
                    host_library
                )
                node._host_ssa_root = host_library.materialized_root_function
                node._host_ssa_raw_blockers = host_library.blockers
                node._host_ssa_blockers = host_library.effective_blockers
                node._host_ssa_hard_blockers = host_library.hard_blockers
                node._host_ssa_legalization_shortfalls = (
                    host_library.legalization_shortfalls
                )
                node._host_machine_state_complete = (
                    host_library.machine_state_complete
                )
                node._host_machine_bodies_complete = (
                    host_library.machine_bodies_complete
                )
                node._host_dependency_context_complete = (
                    host_library.dependency_context_complete
                )
                node._host_repository_ssa_complete = (
                    host_library.repository_ssa_complete
                )
                node._host_uses_machine_state_dialect = (
                    any(
                        unit.result.uses_machine_state_dialect
                        for unit in host_library.units
                    )
                )
                node._host_native_module = getattr(
                    host_root.result, "retained_native_module", None
                )
                node._host_ssa_cache_key = host_library.root_cache_key
                node._host_ssa_cache_path = str(host_root.cache_path)
                node._host_ssa_cache_hit = all(
                    unit.cache_hit for unit in host_library.units
                )
                node._host_ssa_library_cache_keys = tuple(
                    unit.cache_key for unit in host_library.units
                )
                node._host_ssa_library_cache_paths = tuple(
                    str(unit.cache_path) for unit in host_library.units
                )
                node._host_ssa_dependency_edges = host_library.dependencies
                node._host_ssa_unresolved_dependencies = (
                    host_library.unresolved_dependencies
                )
                emit(
                    f"[ast-parent] decompiled host module for {identity[1]} "
                    f"root={host_library.materialized_root_function!r} "
                    f"units={len(host_library.units)} "
                    f"functions={len(node._host_ssa_module.functions)} "
                    f"blockers={len(host_library.effective_blockers)} "
                    f"unresolved_dependencies="
                    f"{len(host_library.unresolved_dependencies)} "
                    f"cache_hit={node._host_ssa_cache_hit}"
                )
                continue
            unavailable_identities.add(identity)
            emit(
                f"[ast-parent] unresolved source for {identity[1]} "
                f"reason=source_unavailable"
            )
            continue
        process_graph_boundary = getattr(
                source_target,
                "__process_graph_boundary__",
                None,
            )
        if process_graph_boundary is not None:
                # Preserve an explicitly declared terminal boundary while the
                # callable is still available for source discovery.  Ordinary
                # discovered functions retain AST only; this callable marker
                # is intentionally limited to the declared host crossing.
                source_definition._process_graph_boundary = (
                    process_graph_boundary
                )
                source_definition._process_graph_boundary_callable = (
                    source_target
                )
        source_definition = _filter_discovered_definition(source_definition, module)
        source_depth = call_depth + 1
        if max_dependency_depth and source_depth > max_dependency_depth:
            raise RuntimeError(
                "extraction contract python max_dependency_depth exceeded: "
                f"{source_depth}/{max_dependency_depth} at {identity[0]}.{identity[1]}"
            )
            # The AST node's short ``name`` is insufficient once an unbounded
            # closure contains same-named definitions from different modules.
            # Preserve the exact live Python identity used to discover it;
            # later map/function tables can qualify without retaining the
            # callable itself.
        source_definition._python_source_identity = identity

        module.body.append(source_definition)
        emit(
                f"[ast-parent] discovered definition {getattr(source_definition, 'name', identity)!r} "
                f"from {identity[1]} work_item={work_items} "
                f"kind={type(source_definition).__name__} line={getattr(source_definition, 'lineno', '?')}"
            )
        if progress is not None:
                bang = "!" * min(work_items, 12)
                progress(
                    f"{_heat_escape(work_items)}[upward search W{work_items}]{bang} "
                    f"found {getattr(source_definition, 'name', identity)!r} "
                    + (
                        f"(contract depth {source_depth}/{max_dependency_depth})"
                        if max_dependency_depth else
                        "(unbounded -- no constant depth limit)"
                    )
                    + f"{bang}{_HEAT_RESET}"
                )
            # ``source_definition`` itself is included first, matching what
            # ``ast.walk(source_definition)`` used to yield before this loop
            # tracked depth -- it is the node the upward search just found,
            # already logged above, not a downward descent in its own right.
        new_definitions = [source_definition]
        for member, member_depth in _walk_definitions_with_depth(source_definition):
                new_definitions.append(member)
                emit(
                    f"[ast-parent] nested {type(member).__name__} {getattr(member, 'name', '?')!r} "
                    f"depth={member_depth} inside {getattr(source_definition, 'name', identity)!r} "
                    f"line={getattr(member, 'lineno', '?')}"
                )
                if progress is not None:
                    arrows = "v" * min(member_depth, 12)
                    progress(
                        f"{_depth_escape(member_depth)}[downward descent D{member_depth}]{arrows} "
                        f"found {getattr(member, 'name', '?')!r} nested inside "
                        f"{getattr(source_definition, 'name', identity)!r}{arrows}{_HEAT_RESET}"
                    )
        definitions.extend(new_definitions)
        for new_definition in new_definitions:
            definition_depths[id(new_definition)] = source_depth
        for new_definition in new_definitions:
                definitions_by_name.setdefault(
                    new_definition.name,
                    [],
                ).append(new_definition)
            # Attribute selection is a dependency edge to the selected
            # attribute, not to every definition on its owning object.  Keep
            # the owner available to _ast_definition_bindings() as lexical
            # context for self/cls, but ingest only the requested method AST.
            # Appending the whole class here makes unrelated methods look like
            # executed dependencies and recursively pulls their callees into
            # the submitted program.
        definition = source_definition
        target_definitions[identity] = definition
        definition_bindings = _import_ast_bindings(
                source_definition,
                _ast_definition_bindings(source_target),
                package=str(
                    getattr(
                        inspect.getmodule(source_target),
                        "__package__",
                        package,
                    )
                    or ""
                ),
            )
        if isinstance(source_definition, ast.ClassDef):
            # Every method in this discovered class resolves ``self``/``cls``
            # against the class whose source was selected.  The old global
            # rescanner happened to regain that owner on later full passes;
            # the worklist must install it explicitly when admitting the
            # class.  This is source identity only: no instance is created
            # and no runtime attribute lookup is performed.
            definition_bindings.setdefault("self", source_target)
            definition_bindings.setdefault("cls", source_target)
        definition_bindings = _ast_local_constructor_bindings(
                source_definition,
                definition_bindings,
            )
        definition_bindings, _changed = _merge_ast_bindings(
                definition_bindings,
                argument_bindings,
            )
        target_bindings[identity] = definition_bindings
        install_definition_bindings(source_definition, definition_bindings)
        for new_definition in new_definitions:
            new_definition._python_bindings = definition_bindings
            new_definition._python_aggregate_binding_kinds = dict(
                getattr(source_definition, "_python_aggregate_binding_kinds", {})
                or {}
            )
        requeue_definition(source_definition)
        if profile_verbose:
                print(
                    "[ast-parent-profile] "
                    f"resolved={identity[0]}.{identity[1]} "
                    f"new_definitions={len(new_definitions)} "
                    f"total_definitions={len(definitions)} "
                    f"work_items={work_items} elapsed={time.perf_counter() - started:.3f}s",
                    flush=True,
                )

    parent_links = []
    unresolved_calls = []
    definitions_by_source_identity = {
        tuple(map(str, source_identity)): definition
        for definition in definitions
        if (
            isinstance(
                (source_identity := getattr(
                    definition, "_python_source_identity", None
                )),
                tuple,
            )
            and len(source_identity) == 2
        )
    }
    for call in tuple(dict.fromkeys(all_calls)):
        call_bindings = node_bindings.get(id(call), root_bindings)
        target = call_target(call, call_bindings)
        if inspect.ismethod(target):
            target = target.__func__
        definition = None
        if callable(target):
            identity = (
                str(getattr(target, "__module__", "")),
                str(
                    getattr(
                        target,
                        "__qualname__",
                        getattr(target, "__name__", ""),
                    )
                ),
            )
            definition = target_definitions.get(identity)
            if definition is None:
                # Binding revisions can refine a call's receiver after its
                # source body was admitted under an earlier worklist cache
                # key.  The submitted AST carries the exact module/qualname
                # identity on every discovered definition; use that durable
                # source record before considering any spelling fallback.
                definition = definitions_by_source_identity.get(identity)
        if definition is None and isinstance(call.func, ast.Name):
            # A lexical function name can be matched to its unique in-scope
            # definition.  An attribute spelling cannot: ``items.append`` is
            # not a reference to an unrelated discovered class's ``append``
            # method.  Owner resolution above is the only sound way to link
            # an attribute call.  Falling back by basename aliases unrelated
            # methods and routes runtime receivers into the wrong shell.
            candidates = definitions_by_name.get(call.func.id, ())
            if len(candidates) == 1:
                definition = candidates[0]
        if definition is not None and definition is not call:
            emit(
                f"[ast-parent] linked parent {getattr(definition, 'name', '?')!r} -> call line={getattr(call, 'lineno', '?')}"
            )
            parent_links.append((definition, call))
            continue
        if isinstance(call.func, ast.Name):
            unresolved_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            unresolved_name = call.func.attr
        else:
            unresolved_name = type(call.func).__name__
        target = call_target(call, call_bindings)
        identity_target = (
            target.__func__ if inspect.ismethod(target) else target
        )
        if not callable(identity_target):
            reason = "dynamic_or_primitive"
        elif include is not None and not include(identity_target):
            reason = "declared_boundary"
        elif _source_ast_definition(identity_target) is None:
            reason = "source_unavailable"
        else:
            reason = "missing_source_parent"
        emit(
            f"[ast-parent] unresolved call {unresolved_name!r} line={getattr(call, 'lineno', '?')} reason={reason}"
        )
        unresolved_calls.append(
            {
                "name": unresolved_name,
                "line": getattr(call, "lineno", None),
                "column": getattr(call, "col_offset", None),
                "reason": reason,
                "target_module": (
                    str(getattr(identity_target, "__module__", ""))
                    if callable(identity_target) else None
                ),
                "target_qualname": (
                    str(getattr(
                        identity_target,
                        "__qualname__",
                        getattr(identity_target, "__name__", ""),
                    ))
                    if callable(identity_target) else None
                ),
                "owner_name": (
                    str(getattr(call_owners.get(id(call)), "name", "")) or None
                ),
                "owner_source_identity": getattr(
                    call_owners.get(id(call)), "_python_source_identity", None
                ),
                "extraction_contract": extraction_receipt(call),
            }
        )

    ast.fix_missing_locations(module)
    if profile_verbose:
        print(
            "[ast-parent-profile] "
            f"complete work_items={work_items} definitions={len(definitions)} "
            f"parent_links={len(parent_links)} unresolved={len(unresolved_calls)} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    return module, tuple(parent_links), tuple(unresolved_calls), root_bindings


def _resolve(val):
    """
    Make sure anything coming out of a domain-queue is *numeric*:

    • _RandomFloatQueue → draw a float
    • other callables   → call them once
    • list/tuple        → promote to NumPy array (avoids list*float errors)
    • everything else   → leave unchanged
    """
    if isinstance(val, _RandomFloatQueue):
        return val()                    # our queue is callable → random float
    if isinstance(val, Callable):
        try:
            return val()                # user-supplied lambda, etc.
        except TypeError:
            pass                        # not a no-arg callable – ignore
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=float)
    return val

init(autoreset=True)

MAX_HUES = 12  # maximum distinct hues before wrapping
def multi_sort(collection, key_funcs):
    compound_keys = [
        tuple(func(item) for func in key_funcs)
        for item in collection
    ]
    items_with_keys = list(zip(collection, compound_keys))
    items_with_keys.sort(key=lambda x: x[1])
    return [item for item, _ in items_with_keys]

_torch_module = None
_torch_checked = False


def _optional_torch():
    """Import Torch only for ProcessGraph paths that explicitly request it."""

    global _torch_module, _torch_checked
    if not _torch_checked:
        try:
            import torch as module  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            module = None
        _torch_module = module
        _torch_checked = True
    return _torch_module

class ExpressionTensor:
    def __init__(self, data, contexts=None, sequence_length=1, domain_shape=None, function_index=None):
        self.data = data
        self.contexts = contexts or [0]
        self.sequence_length = sequence_length
        self.domain_shape = domain_shape or self._infer_shape(data)
        self.function_index = function_index

    def _infer_shape(self, data):
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            data = data[0] if data else []
        return tuple(shape)

    @property
    def shape(self):
        return self.domain_shape

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def to_numpy(self):

        # Helper to walk the nested structure and collect types
        def collect_types(x, depth=0):
            #print(f"Collecting types from: {x} at depth {depth}")
            types = []
            if isinstance(x, list):
                for item in x:
                    types.extend(collect_types(item, depth + 1))
            else:
                types.append(type(x))
            return types

        try:
            arr = np.array(self.data)
            # if object dtype, fallback to advanced conversion
            if arr.dtype == object:
                raise ValueError("object-dtype array, falling back to advanced conversion")
            return arr
        except Exception as e:
            # Fallback: recursively stack nested lists of arrays/scalars into one ndarray
            import numpy as _np
            def _recurse_stack(d):
                if isinstance(d, list):
                    subs = [_recurse_stack(x) for x in d]
                    if not subs:
                        raise ValueError("Empty list cannot be stacked")
                    shapes = [s.shape for s in subs]
                    dtypes = [s.dtype for s in subs]
                    if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                        return _np.stack(subs, axis=0)
                    else:
                        raise ValueError("Inconsistent shapes or dtypes in nested data")
                # leaf: convert scalar or array
                arr = _np.array(d)
                return arr
            try:
                return _recurse_stack(self.data)
            except Exception:
                pass

            # On failure, report types in the nested structure
            print("⚠️ Failed to convert ExpressionTensor to numpy array.")
            print(f"Error: {e}")
            all_types = collect_types(self.data)
            from collections import Counter
            type_counts = Counter(all_types)
            print("Types found in tensor data:")
            for t, count in type_counts.items():
                print(f"  {t}: {count}")
            raise ValueError("Mixed types detected in ExpressionTensor data, cannot safely convert to numpy.") from e


    def __array__(self):
        return self.to_numpy()

    def __repr__(self):
        return f"ExpressionTensor(shape={self.shape}, data={self.data})"


class ProcessGraph:
    def __init__(
        self,
        recombinatorics_level=0,
        expand_complex=False,
        materialize_memory=True,
        function_table=None,
        external_function_table=None,
        boundary_namespace=None,
        source_language="python",
    ):
        # Translation and scheduling do not require a physical memory graph.
        # Keep materialization opt-in at the call site so compiler front-ends
        # can operate even when the experimental allocator is unavailable.
        self.materialize_memory = materialize_memory
        self.MG = BitTensorMemoryGraph(size=0) if materialize_memory else None
        self.G = self.MG.G if self.MG is not None else nx.DiGraph()
        self._graph_lock = threading.RLock()
        self._graph_condition = threading.Condition(self._graph_lock)
        self._graph_revision = 0
        self._graph_subscribers = []
        self._graph_accessor = ProcessGraphAccessor(self)
        from ...compiler.evolution_metagraph import active_evolution_metagraph
        self._evolution_metagraph = active_evolution_metagraph()
        self._evolution_graph = (
            None
            if self._evolution_metagraph is None
            else self._evolution_metagraph.open_graph(
                "process-graph",
                "source ingestion",
            )
        )
        self.levels = {}
        self.node_map = {}
        self._graph_profile_verbose = False
        self._graph_build_counter = 0
        self._graph_progress = None
        # integer level for recombinatorics aggressiveness: 0=no, higher unlock more transforms
        self.recombinatorics_level = recombinatorics_level
        self.expand_complex = expand_complex
        self.domain_shape = ()
        self.roots = []
        self.role_schemas = role_schemas
        if isinstance(boundary_namespace, (str, os.PathLike)):
            from .boundary_namespace import BoundaryNamespace
            boundary_namespace = BoundaryNamespace(
                boundary_namespace, language=source_language,
            )
        elif boundary_namespace is not None and not hasattr(
            boundary_namespace, "rules_for_scope"
        ):
            raise TypeError("boundary_namespace must be a path or BoundaryNamespace")
        self.boundary_namespace = boundary_namespace
        self.source_language = str(source_language)
        
        self.scheduler = ILPScheduler(self)
        self.consumer_queues = {}
        self.function_table = (
            FunctionTable() if function_table is None else function_table
        )
        self.external_function_table = (
            ExternalFunctionTable()
            if external_function_table is None
            else external_function_table
        )

    def __getstate__(self):
        """Serialize compiler state without live synchronization observers."""

        state = dict(self.__dict__)
        serialized_bindings = {}
        for name, value in (state.get("python_bindings") or {}).items():
            if isinstance(value, types.ModuleType):
                serialized_bindings[name] = ("module", value.__name__)
                continue
            module = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", None)
            if module and qualname and "<locals>" not in str(qualname):
                serialized_bindings[name] = (
                    "qualified",
                    str(module),
                    str(qualname),
                )
                continue
            try:
                serialized_bindings[name] = (
                    "pickle",
                    pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                )
            except Exception:
                # Live interpreter state (ContextVar, locks, active contexts)
                # and local closures are not compiler state and are resolved
                # again from imported source bindings when needed.  CPython's
                # pickle reports local functions as AttributeError rather
                # than PicklingError, so every ordinary serialization failure
                # means this binding is not checkpointable.
                continue
        state["python_bindings"] = serialized_bindings
        for name in (
            "_graph_lock",
            "_graph_condition",
            "_graph_accessor",
            "_graph_subscribers",
            "_evolution_metagraph",
            "_evolution_graph",
            "_graph_progress",
        ):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        """Restore a checkpoint as an independent live ProcessGraph."""

        self.__dict__.update(state)
        restored_bindings = {}
        for name, descriptor in (
            self.__dict__.get("python_bindings") or {}
        ).items():
            try:
                if descriptor[0] == "module":
                    value = importlib.import_module(descriptor[1])
                elif descriptor[0] == "qualified":
                    value = importlib.import_module(descriptor[1])
                    for part in descriptor[2].split("."):
                        value = getattr(value, part)
                elif descriptor[0] == "pickle":
                    value = pickle.loads(descriptor[1])
                else:
                    continue
            except (ImportError, AttributeError, TypeError, ValueError):
                continue
            restored_bindings[name] = value
        self.python_bindings = restored_bindings
        self.boundary_namespace = getattr(self, "boundary_namespace", None)
        self.source_language = getattr(self, "source_language", "python")
        self._graph_lock = threading.RLock()
        self._graph_condition = threading.Condition(self._graph_lock)
        self._graph_subscribers = []
        self._graph_accessor = ProcessGraphAccessor(self)
        self._evolution_metagraph = None
        self._evolution_graph = None
        self._graph_progress = None

    def _safe_repr(self, value, *, max_length=180):
        return _safe_repr(value, max_length=max_length)

    def _node_debug_summary(self, node):
        if isinstance(node, ast.AST):
            parts = []
            if hasattr(node, "lineno") and getattr(node, "lineno", None) is not None:
                parts.append(f"line={node.lineno}")
            if hasattr(node, "col_offset") and getattr(node, "col_offset", None) is not None:
                parts.append(f"col={node.col_offset}")
            if isinstance(node, ast.Constant):
                parts.append(f"value={self._safe_repr(node.value)}")
            elif isinstance(node, ast.Name):
                parts.append(f"id={node.id}")
            elif isinstance(node, ast.Attribute):
                parts.append(f"attr={node.attr}")
            elif isinstance(node, ast.Call):
                func_kind = type(node.func).__name__
                parts.append(f"func={func_kind}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(f"body={len(node.body)}")
            if hasattr(node, "name") and getattr(node, "name", None) not in {None, ""}:
                parts.append(f"name={node.name}")
            if hasattr(node, "value") and not isinstance(node.value, ast.AST):
                parts.append(f"value={self._safe_repr(node.value)}")
            if hasattr(node, "arg") and getattr(node, "arg", None) not in {None, ""}:
                parts.append(f"arg={node.arg}")
            if hasattr(node, "id") and getattr(node, "id", None) not in {None, ""}:
                parts.append(f"id={node.id}")
            return " | ".join(parts) if parts else type(node).__name__

        parts = []
        for attr in ("name", "id", "arg", "value", "label"):
            candidate = getattr(node, attr, None)
            if candidate is None:
                continue
            if isinstance(candidate, (str, int, float, bool)) or candidate is None:
                parts.append(f"{attr}={self._safe_repr(candidate)}")
            elif not isinstance(candidate, (list, tuple, dict, set)):
                parts.append(f"{attr}={self._safe_repr(candidate)}")
        if not parts:
            parts.append(self._safe_repr(node))
        return " | ".join(parts)

    def _emit_graph_build_log(self, message: str, *, progress=None) -> None:
        """Emit verbose compiler progress for each ingested AST item."""

        if self._graph_profile_verbose:
            print(message, flush=True)
        sink = progress if progress is not None else self._graph_progress
        if sink is not None:
            sink(message)

    def graph_accessor(self) -> ProcessGraphAccessor:
        """Return the stable accessor used by live compilation observers."""

        return self._graph_accessor

    def observe_evolution_node(self, node_id, data) -> None:
        if self._evolution_graph is None:
            return
        source_span = data.get("source_span")
        if not source_span:
            expression = data.get("expr_obj")
            if expression is not None and hasattr(expression, "lineno"):
                source_span = {
                    "line": getattr(expression, "lineno", None),
                    "column": getattr(expression, "col_offset", None),
                    "end_line": getattr(expression, "end_lineno", None),
                    "end_column": getattr(expression, "end_col_offset", None),
                }
        self._evolution_metagraph.component(
            self._evolution_graph,
            node_id,
            label=str(data.get("label") or data.get("op") or node_id),
            kind=str(data.get("type") or data.get("op") or "operation"),
            attributes={
                "source_span": source_span,
                "schema_version": data.get("schema_version"),
                "source_scope": tuple(data.get("source_scope") or ()),
                "source_class": data.get("source_class"),
                "source_language": self.source_language,
                "boundary_rule": data.get("boundary_rule"),
                "boundary_action": data.get("boundary_action"),
                "extraction_action": (
                    data.get("attributes") or {}
                ).get("extraction_action"),
                "extraction_rule": (
                    data.get("attributes") or {}
                ).get("extraction_rule"),
                "extraction_identity": (
                    data.get("attributes") or {}
                ).get("extraction_identity"),
            },
        )

    def _apply_boundary_resolution(self, node_id, resolution) -> None:
        receipt = getattr(resolution, "receipt", None)
        excluded = tuple(getattr(resolution, "excluded_rule_ids", ()) or ())
        if receipt is None and not excluded:
            return
        data = self.G.nodes[node_id]
        if receipt is not None:
            mapping = receipt.mapping()
            data["boundary_rule"] = mapping["rule_id"]
            data["boundary_action"] = mapping["action"]
            data["boundary_receipt"] = mapping
            receipts = tuple(self.G.graph.get("boundary_namespace_receipts") or ())
            if mapping not in receipts:
                self.G.graph["boundary_namespace_receipts"] = (*receipts, mapping)
        if excluded:
            data["boundary_excluded_rule_ids"] = excluded
        self.observe_evolution_node(node_id, data)

    def observe_evolution_edge(self, source, target, role="data") -> None:
        if self._evolution_graph is None:
            return
        from ...compiler.evolution_metagraph import EvolutionComponentRef
        self._evolution_metagraph.relationship(
            self._evolution_graph,
            EvolutionComponentRef(self._evolution_graph.id, str(source)),
            EvolutionComponentRef(self._evolution_graph.id, str(target)),
            role=str(role or "data"),
        )

    @contextmanager
    def graph_mutation(self):
        """Serialize a topology mutation and publish one new revision."""

        snapshot = None
        callbacks = ()
        with self._graph_condition:
            yield self.G
            self._graph_revision += 1
            self._graph_condition.notify_all()
            callbacks = tuple(self._graph_subscribers)
            if callbacks:
                copied = self.G.copy(as_view=False)
                for node_id, level in self.levels.items():
                    if node_id in copied:
                        copied.nodes[node_id].setdefault("level", level)
                snapshot = ProcessGraphSnapshot(self._graph_revision, copied)
        if snapshot is not None:
            for callback in callbacks:
                callback(snapshot)

    def full_recombinatorics(self, expr, level=1):
        """
        Apply symbolic transforms with increasing aggressiveness based on level:
        level>=1: doit
        level>=2: expand
        level>=3: expand_mul, expand_power_exp
        level>=4: expand_log, trigsimp
        level>=5: cancel, apart
        level>=6: factor, simplify
        """
        if level >= 1:
            expr = expr.doit()
        if level >= 2:
            expr = sympy.expand(expr, power_exp=True, log=True,
                                 multinomial=True, complex=self.expand_complex, trig=True)
        if level >= 3:
            expr = sympy.expand_mul(expr)
            expr = sympy.expand_power_exp(expr)
        if level >= 4:
            expr = sympy.expand_log(expr)
            expr = sympy.trigsimp(expr)
        if level >= 5:
            expr = sympy.cancel(expr)
            try:
                expr = sympy.apart(expr)
            except Exception:
                pass
        if level >= 6:
            expr = sympy.factor(expr)
            expr = sympy.simplify(expr)
        return expr
    
    def deduplicate_node(self, G, nid):
        """
        Deduplicate a node in the graph by checking if it has the same label and type.
        If found, return the existing node's ID; otherwise, return the original ID.
        """
        node_data = G.nodes[nid]
        label = node_data['label']
        node_type = node_data['type']
        
        for other_nid, other_data in G.nodes(data=True):
            if other_nid != nid and other_data['label'] == label and other_data['type'] == node_type:
                G.remove_node(nid)
                return other_nid
        return nid



    def ensure_node(
        self, node, store_id=None, deduplicate=True, special_case=None,
    ):

        nid = id(node)

            

        with self.graph_mutation():
            if nid in self.G:
                return nid, True
            node_type = type(node).__name__
            #print(f"Building graph node: type={type(node).__name__}, repr={repr(node)}")

            semantic_type = special_case.type if special_case is not None else node_type
            sig = operator_signatures.get(
                semantic_type, operator_signatures['Default']
            )
            extra_args = {}
            for param in sig.get('parameters', []):
                value = getattr(node, param, None)
                if value is not None:
                    extra_args[param] = value
            source_span = None
            if isinstance(node, ast.AST) and getattr(node, "lineno", None) is not None:
                source_span = {
                    "line": getattr(node, "lineno", None),
                    "column": getattr(node, "col_offset", None),
                    "end_line": getattr(node, "end_lineno", None),
                    "end_column": getattr(node, "end_col_offset", None),
                }
            source_scope = tuple(getattr(node, "_turing_source_scope", ()))
            source_class = getattr(node, "_turing_source_class", None)
            receipt = extraction_receipt(node)
            semantic_attributes = (
                dict(special_case.attributes) if special_case is not None else {}
            )
            if receipt is not None:
                semantic_attributes.setdefault("extraction_contract", receipt)
                semantic_attributes.setdefault("extraction_action", receipt["action"])
                semantic_attributes.setdefault("extraction_rule", receipt.get("rule_id"))
                semantic_attributes.setdefault(
                    "extraction_identity", receipt.get("identity")
                )
                semantic_attributes.setdefault(
                    "extraction_classification", receipt.get("classification")
                )
            label = str(node)
            if receipt is not None and isinstance(node, ast.Call):
                label = (
                    f"{receipt['action']}: "
                    f"{receipt.get('identity') or label}"
                )
            elif special_case is not None:
                label = special_case.type
            self.G.add_node(nid,
                label=label,
                type=semantic_type,
                op=semantic_type if special_case is not None else None,
                expr_obj=node,
                source_span=source_span,
                source_scope=source_scope,
                source_class=source_class,
                extra_args={**extra_args, **semantic_attributes},
                attributes=semantic_attributes,
                constant=(
                    special_case.constant if special_case is not None else None
                ),
                extraction_contract=receipt,
                domain_node=DomainNode(
                    shape=(1,1,1), #default will be function pointer
                    unit_size=1,  # default unit size for function pointers
                ),
                store_id=store_id,
                parents=[],
                children=[])
            self.node_map[nid] = node
            self.observe_evolution_node(nid, self.G.nodes[nid])

            new_nid = self.deduplicate_node(self.G, nid)
            if new_nid != nid:
                del self.node_map[nid]
                return new_nid, True
            return nid, False

    def connect(self, src_id, tgt_id, producer_role, consumer_role, store_id=None):
        with self.graph_mutation():
            edge = Edge(
                id=(src_id, tgt_id, producer_role, consumer_role),
                operation=None,
                source=src_id,
                target=tgt_id,
                store_id=store_id
            )
            if not self.G.has_edge(src_id, tgt_id):
                self.G.add_edge(src_id, tgt_id, extra=set())
            if 'extra' not in self.G[src_id][tgt_id]:
                self.G[src_id][tgt_id]['extra'] = set()
            self.G[src_id][tgt_id]['extra'].add(edge)

            if 'children' not in self.G.nodes[src_id]:
                self.G.nodes[src_id]['children'] = []
            if 'parents' not in self.G.nodes[tgt_id]:
                self.G.nodes[tgt_id]['parents'] = []
            if tgt_id not in [p for p, _ in self.G.nodes[src_id]['children']]:
                self.G.nodes[src_id]['children'].append((tgt_id, producer_role))
            if src_id not in [p for p, _ in self.G.nodes[tgt_id]['parents']]:
                self.G.nodes[tgt_id]['parents'].append((src_id, consumer_role))
            self.observe_evolution_edge(src_id, tgt_id, consumer_role)

    def _spec_build_tasks(
        self, nid, args, spec, direction, store_id, schema_repeats,
        role_indices,
    ):
        """Return schema-directed child visits without invoking traversal."""

        if os.environ.get("TURING_GRAPH_BUILD_VERBOSE", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            print(spec.items())
        tasks = []
        for role, param in spec.items():
            indices = role_indices[role]
            if param == 1:
                offset = schema_repeats.get(role, 0)
                # Preserve the recursive builder's strict schema contract:
                # a missing required role is malformed input, not an empty
                # child list that may be silently omitted.
                selected = (indices[offset],)
                schema_repeats[role] = offset + 1
            elif param == "many":
                selected = indices[schema_repeats.get(role, 0):]
                schema_repeats[role] = len(indices)
            elif isinstance(param, tuple):
                count = param[1] if len(param) == 2 else param[0]
                selected = []
                for _ in range(count):
                    offset = schema_repeats.get(role, 0)
                    selected.append(indices[offset])
                    schema_repeats[role] = offset + 1
            else:
                selected = ()
            for idx in selected:
                if direction == "down":
                    tasks.append((
                        args[idx], nid, None, role, f"arg{idx}", store_id,
                    ))
                else:
                    tasks.append((
                        args[idx], None, nid, "output", role, store_id,
                    ))
        return tasks

    def build_graph(
        self, node, producer_id=None, consumer_id=None, producer_role=None,
        consumer_role=None, store_id=None, progress=None,
    ):
        """Build a complete source graph using an explicit work stack.

        A work item is either a node visit or a postorder connection.  This
        preserves the former recursive depth-first order while making deep
        authored expressions and cyclic object identities independent of the
        Python call-stack limit.
        """

        if not self.domain_shape:
            self.domain_shape = (1,)
        initial = (
            node, producer_id, consumer_id, producer_role, consumer_role,
            store_id,
        )
        pending = [("visit", initial)]
        root_nid = None
        graph_build_verbose = os.environ.get(
            "TURING_GRAPH_BUILD_VERBOSE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        while pending:
            action, payload = pending.pop()
            if action == "finish":
                (
                    nid, frame_producer, frame_consumer,
                    frame_producer_role, frame_consumer_role, frame_store,
                ) = payload
                if frame_producer is not None:
                    self.connect(
                        frame_producer, nid, frame_producer_role,
                        frame_consumer_role, frame_store,
                    )
                if frame_consumer is not None:
                    self.connect(
                        nid, frame_consumer, frame_producer_role,
                        frame_consumer_role, frame_store,
                    )
                if frame_producer is None and frame_consumer is None:
                    self.roots.append(nid)
                continue

            (
                current, frame_producer, frame_consumer,
                frame_producer_role, frame_consumer_role, frame_store,
            ) = payload
            self._graph_build_counter += 1
            boundary_resolution = None
            if self.boundary_namespace is not None:
                boundary_resolution = self.boundary_namespace.resolve(current, self)
            special = (
                None if boundary_resolution is None
                else boundary_resolution.special_case
            )
            if special is None:
                special = getattr(current, "_special_case", None)
            if special is None and self.source_language.casefold() == "python":
                special = interpret_python_special_case(current)
            if special is None:
                special = interpret_special_case(current)
            nid, already_defined = self.ensure_node(
                current, frame_store, special_case=special,
            )
            if root_nid is None:
                root_nid = nid
            node_type = type(current).__name__
            location = ""
            if isinstance(current, ast.AST):
                location = (
                    f" line={getattr(current, 'lineno', '?')} "
                    f"col={getattr(current, 'col_offset', '?')}"
                )
            if (
                hasattr(current, "name")
                and getattr(current, "name", None) not in {None, ""}
            ):
                location += f" name={current.name}"
            self._emit_graph_build_log(
                f"[graph-build #{self._graph_build_counter}] "
                f"{node_type}{location} nid={nid} "
                f"already_defined={already_defined} "
                f"producer={frame_producer_role or '-'} "
                f"consumer={frame_consumer_role or '-'} "
                f"details={self._node_debug_summary(current)}",
                progress=progress,
            )
            if already_defined:
                if frame_producer is not None:
                    self.connect(
                        frame_producer, nid, frame_producer_role,
                        frame_consumer_role, frame_store,
                    )
                if frame_consumer is not None:
                    self.connect(
                        nid, frame_consumer, frame_producer_role,
                        frame_consumer_role, frame_store,
                    )
                continue

            finish = (
                nid, frame_producer, frame_consumer,
                frame_producer_role, frame_consumer_role, frame_store,
            )
            if boundary_resolution is not None:
                self._apply_boundary_resolution(nid, boundary_resolution)
            if special is not None:
                data = self.G.nodes[nid]
                data["type"] = special.type
                data["op"] = special.type
                data["attributes"] = special.attributes
                data["extra_args"] = special.attributes
                data["constant"] = special.constant
                if special.terminal:
                    pending.append(("finish", finish))
                if special.type == "GetAttr" and isinstance(
                    current, ast.Attribute
                ):
                    pending.append(("visit", (
                        current.value, None, nid, "output", "value",
                        frame_store,
                    )))
                if special.terminal:
                    continue
                node_type = special.type

            tensor_op = tensor_operation_name(current)
            if tensor_op is not None:
                data = self.G.nodes[nid]
                attributes = data.get("attributes")
                if attributes is None:
                    attributes = {}
                    data["attributes"] = attributes
                attributes["tensor_candidate"] = tensor_op
                if (
                    isinstance(getattr(current, "func", None), ast.Name)
                    or getattr(current, "_tensor_code_reference", None)
                    == tensor_op
                ):
                    # Free tensor functions and explicitly supplied backend
                    # source references are authoritative. Method spelling
                    # alone is only a candidate until the receiver is proven
                    # tensor-valued after lexical reduction/specialization.
                    attributes["tensor"] = tensor_op

            schema = (
                None if boundary_resolution is None
                else boundary_resolution.role_schema
            )
            if schema is None:
                schema = self.role_schemas.get(node_type)
            args = getattr(current, "args", [])
            args = list(args) if isinstance(args, list) else [args]
            if graph_build_verbose:
                print(
                    f"[build_graph] Node {nid} ({node_type}) with schema: "
                    f"{schema}"
                )
                print(f"[build_graph] Node {nid} args: {args}")
            child_tasks = []
            if schema:
                role_indices = {}
                all_keys = (
                    list(schema.get("up", {}).keys())
                    + list(schema.get("down", {}).keys())
                )
                for key in all_keys:
                    value = getattr(current, key, None)
                    if value not in args:
                        if isinstance(value, list):
                            start = len(args)
                            args.extend(value)
                            role_indices[key] = list(range(start, len(args)))
                        else:
                            args.append(value)
                            role_indices[key] = [len(args) - 1]
                    else:
                        role_indices[key] = [args.index(value)]
                repeats = {role: 0 for role in role_indices}
                child_tasks.extend(self._spec_build_tasks(
                    nid, args, schema.get("up", {}), "up", frame_store,
                    repeats, role_indices,
                ))
                child_tasks.extend(self._spec_build_tasks(
                    nid, args, schema.get("down", {}), "down", frame_store,
                    repeats, role_indices,
                ))
            else:
                child_tasks.extend(
                    (arg, None, nid, "output", f"arg{idx}", frame_store)
                    for idx, arg in enumerate(args)
                )

            pending.append(("finish", finish))
            pending.extend(
                ("visit", task) for task in reversed(child_tasks)
            )

        return root_nid

    def _walk_all_fields(self, node, consumer_id=None, producer_role=None, consumer_role=None, store_id=None, verbose=True):
        """
        Bold mode: For objects without schema or args, traverse all public fields and attributes.
        """
        nid, already_defined = self.ensure_node(node, store_id)
        if already_defined:
            return nid

        # Use __dict__ if available, else dir()
        fields = {}
        if hasattr(node, '__dict__'):
            fields = node.__dict__
        else:
            # fallback: get all attributes that aren't private/magic
            for attr in dir(node):
                if attr.startswith('_'):  # skip magic/private by default
                    continue
                try:
                    value = getattr(node, attr)
                except Exception:
                    continue
                fields[attr] = value

        for field, value in fields.items():
            if value is not None:
                # For containers, descend recursively
                if isinstance(value, (list, tuple, set)):
                    for i, elem in enumerate(value):
                        self.build_graph(elem, consumer_id=nid, producer_role="output", consumer_role=f"{field}[{i}]", store_id=store_id, verbose=verbose)
                else:
                    self.build_graph(value, consumer_id=nid, producer_role="output", consumer_role=field, store_id=store_id, verbose=verbose)

        # Connect if needed (mirrors your usual logic)
        if consumer_id is not None:
            self.connect(nid, consumer_id, producer_role, consumer_role, store_id)
        if producer_role is None and consumer_role is None:
            self.roots.append(nid)

        #inspect if we have recovered at the end of all processing
        # a nid with no connections, implying we ran the graph
        # in the wrong direction for our process, then we can
        # run the edges through a quick swap and run this function again recursively
        # with a single depth flag to avoid infinite recursion

        if verbose:
            print(f"[walk_all_fields] Processed node {nid} with fields: {list(fields.keys())}")
        if not self.G.nodes[nid]['children'] and not self.G.nodes[nid]['parents']:
            if verbose:
                print(f"[walk_all_fields] Node {nid} has no connections, checking for recovery...")

        return nid


    def build_from_ast(
        self,
        node_or_path,
        *args,
        semantic=None,
        filename=None,
        resolve_unresolved_parents=False,
        parent_bindings=None,
        parent_include=None,
        pursuit_roots=None,
        tensor_code_references=None,
        retain=(),
        profile_verbose=False,
        progress=None,
        boundary_namespace=None,
        source_language=None,
        **kwargs,
    ):
        """Import Python source as a structural AST ProcessGraph.

        NOT the compiler entrypoint. This is one frontend step -- it builds
        the raw structural graph and nothing past it: no global/free-name
        binding, no control/region splitting (if/for/raise/comprehensions
        stay unresolved AST-node placeholders), no scheduling. Calling this
        directly and then handing the result to
        transmogrifier.ssa_builder.process_graph_to_ssa_instrs will "succeed"
        (no exception) while silently emitting garbage ops for anything
        beyond straight-line scalar expressions -- verified by trying it on
        a real function from amd64_machine_semantics.py, which produced
        Instr(op='<ast.Name object at 0x...>', args=[]) for most nodes.

        For real compilation of a Python function -- including control flow
        -- use
        src.common.tensors.accelerator_backends.aot_compile.compile_ast_aot,
        which calls this method internally as one step of a real pipeline
        (global/free-name binding via ``python_bindings``, control/region
        splitting through control_source.ControlProgram, scheduling). Pass
        ``precompile_only=True`` to plan without requiring an installed
        runtime.
        """
        import os

        if boundary_namespace is not None:
            if isinstance(boundary_namespace, (str, os.PathLike)):
                from .boundary_namespace import BoundaryNamespace
                boundary_namespace = BoundaryNamespace(
                    boundary_namespace,
                    language=source_language or self.source_language,
                )
            elif not hasattr(boundary_namespace, "rules_for_scope"):
                raise TypeError(
                    "boundary_namespace must be a path or BoundaryNamespace"
                )
            self.boundary_namespace = boundary_namespace
        if source_language is not None:
            self.source_language = str(source_language)

        supplied_ast = isinstance(node_or_path, ast.AST)
        if supplied_ast:
            tree = node_or_path
        elif isinstance(node_or_path, str) and os.path.exists(node_or_path):
            filename = filename or node_or_path
            with open(node_or_path, "r", encoding="utf-8") as stream:
                tree = ast.parse(stream.read(), filename=filename)
        elif isinstance(node_or_path, str):
            try:
                tree = ast.parse(node_or_path, filename=filename or "<string>")
            except Exception as exc:
                raise ValueError(f"Could not parse string as source code: {exc}") from exc
        else:
            raise TypeError(
                "build_from_ast expects an AST node, a filename, or a source string"
            )

        retained = () if retain is None else (
            (retain,) if inspect.isclass(retain) else tuple(retain)
        )
        retained_identities = []
        existing_classes = {
            definition.name
            for definition in getattr(tree, "body", ())
            if isinstance(definition, ast.ClassDef)
        }
        for retained_class in retained:
            if not inspect.isclass(retained_class):
                raise TypeError(
                    "retain expects a class object or an iterable of class objects"
                )
            identity = retained_class.__name__
            retained_identities.append(identity)
            if identity in existing_classes:
                continue
            definition = _source_ast_definition(retained_class)
            if not isinstance(definition, ast.ClassDef):
                raise ValueError(
                    f"cannot ingest retained class {retained_class!r}: "
                    "source is unavailable"
                )
            definition = _attach_external_methods(retained_class, definition)
            tree.body.append(definition)
            existing_classes.add(identity)

        # Dissolve recognised spans at the seam -- before parent-expansion, IR
        # mapping, state-machine planning, and the normalizer each walk the
        # tree -- so a repr-expanded feed array is collapsed exactly once and
        # never seen expanded again by any downstream pass.
        tree = _lower_consumed_generator_loops(tree)
        tree = dissolve_spans(tree)

        parent_links = ()
        unresolved_calls = ()
        self._graph_profile_verbose = bool(profile_verbose) or bool(
            os.environ.get("TURING_GRAPH_BUILD_VERBOSE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._graph_progress = progress
        self._graph_build_counter = 0
        if resolve_unresolved_parents:
            bindings = dict(getattr(self, "python_bindings", {}) or {})
            bindings.update(parent_bindings or {})
            tree, parent_links, unresolved_calls, root_bindings = (
                _expand_unresolved_ast_parents(
                tree,
                bindings,
                    package=getattr(self, "python_package", None),
                    include=parent_include,
                    pursuit_roots=pursuit_roots,
                    tensor_code_references=tensor_code_references,
                    profile_verbose=profile_verbose,
                    progress=progress,
                )
            )
            # The real imports this just resolved (importlib, against
            # ``python_package``) previously never left this call --
            # ``static_bindings`` (topological_reducer.py), the actual
            # lookup an ordinary Name resolves against, only ever saw
            # whatever ``python_bindings`` the caller supplied up front,
            # never anything discovered here.  A name imported by the
            # source itself (``from .machine_path_forest import
            # MachinePathHeadStatus``) was real and resolved, just
            # discarded before reduction could ever see it.
            self.python_bindings = root_bindings
            if parent_include is not None and hasattr(parent_include, "receipts"):
                self.G.graph["extraction_contract_receipts"] = (
                    parent_include.receipts()
                )
                self.G.graph["extraction_contract_fingerprint"] = getattr(
                    parent_include, "fingerprint", None
                )

        # Preserve class declarations as schema metadata beside the exact AST
        # nodes ProcessGraph is about to ingest; do not create a second AST
        # ingestion path or infer process topology from them.
        self.G.graph["map_ir"] = _map_ir_from_ast(tree)
        if retained_identities:
            self.G.graph["map_ir"]["selected_class_identities"] = tuple(
                dict.fromkeys(retained_identities)
            )
        # A name being a locally-defined class is a fact about the source,
        # known the moment ``map_ir`` sees its ``ClassDef`` -- not something
        # later passes should rediscover (or, absent that, fall through to
        # treating the name as an unresolved external). Publish it here,
        # once, so every later stage that creates or resolves a call to
        # this name reads the same authoritative answer.
        self.G.graph["class_definitions"] = frozenset(
            str(item["class_name"])
            for item in self.G.graph["map_ir"].get("objects", ())
        )
        from ...compiler.state_machine_ast import plan_marked_state_machines
        state_machine_plans, state_machine_shortfalls = (
            plan_marked_state_machines(tree)
        )
        self.G.graph["state_machine_controls"] = state_machine_plans
        self.G.graph["state_machine_control_shortfalls"] = (
            state_machine_shortfalls
        )
        if semantic is None:
            semantic = not supplied_ast and not resolve_unresolved_parents
        if semantic:
            from ...compiler.ast_process_graph import build_semantic_ast

            return build_semantic_ast(self, tree, filename=filename)
        # Special-cased statement constructs, handled in the ingestion
        # special-cases area (node_special_cases): a walrus in a once-evaluated
        # position is hoisted to a plain assignment so no raw NamedExpr leaks to
        # the deep compiler, and an annotated assignment becomes an ordinary
        # assignment whose declared type is captured as metadata (the real type
        # annotator) rather than discarded.
        tree = hoist_walrus_assignments(tree)
        # A constant-name ``getattr(obj, "field", default)`` is a static
        # attribute access spelled defensively; fold it to ``obj.field`` (for
        # names the source declares) so it resolves structurally instead of
        # surviving as a string constant a numeric backend cannot express.
        tree = fold_constant_getattr(tree)
        # ``obj[...]`` carries an inexpressible Ellipsis; expand it to an
        # ndim-driven full-slice index so the subscript grows an explicit
        # dependency on ``obj.ndim`` (known deterministically before it fires)
        # instead of an opaque literal.
        tree = expand_ellipsis_subscripts(tree)
        self.G.graph["type_annotations"] = {
            **(self.G.graph.get("type_annotations") or {}),
            **annotate_types(tree),
        }
        tree = ast.fix_missing_locations(tree)
        _annotate_visual_source_owners(tree)

        if profile_verbose:
            print(
                "[ast-build-profile] begin build_graph "
                f"definitions={sum(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for item in ast.walk(tree))} "
                f"nodes={sum(1 for _ in ast.walk(tree))}",
                flush=True,
            )
            build_started = time.perf_counter()
        root = self.build_graph(tree, *args, progress=progress, **kwargs)
        if profile_verbose:
            print(
                "[ast-build-profile] complete build_graph "
                f"graph_nodes={self.G.number_of_nodes()} "
                f"graph_edges={self.G.number_of_edges()} "
                f"elapsed={time.perf_counter() - build_started:.3f}s",
                flush=True,
            )
        for definition, call in parent_links:
            definition_id = id(definition)
            call_id = id(call)
            if definition_id not in self.G or call_id not in self.G:
                continue
            self.connect(
                definition_id,
                call_id,
                "definition",
                "callee",
            )
            self.G.nodes[call_id].setdefault("attributes", {})[
                "resolved_ast_parent"
            ] = definition_id
        if resolve_unresolved_parents:
            self.G.graph["resolved_ast_parent_count"] = len(parent_links)
            self.G.graph["unresolved_ast_calls"] = unresolved_calls
            self.G.graph["extraction_boundary_calls"] = tuple(
                call for call in unresolved_calls
                if call.get("extraction_contract") is not None
            )
            self.G.graph["rejected_extraction_calls"] = tuple(
                call for call in unresolved_calls
                if (call.get("extraction_contract") or {}).get("action")
                == "reject"
            )
            missing_parent_calls = tuple(
                call
                for call in unresolved_calls
                if call["reason"] == "missing_source_parent"
            )
            self.G.graph["missing_ast_parent_calls"] = missing_parent_calls
            self.G.graph["ast_parent_closure_complete"] = (
                not missing_parent_calls
            )
            if missing_parent_calls:
                raise RuntimeError(
                    "AST parent ingestion did not reach a source-definition "
                    f"fixed point: {missing_parent_calls!r}"
                )
        return root
    
    def finalize_graph_with_outputs(self):
        """
        Ensure every node satisfies its min_outputs.
        If missing, generate explicit Store nodes to fulfill output slots.
        """
        for nid in list(self.G.nodes):
            node_data = self.G.nodes[nid]
            op_type = node_data['type']
            sig = operator_signatures.get(op_type, operator_signatures['Default'])
            min_outputs = sig.get('min_outputs', 1)
            current_outputs = len(node_data['children'])
            store_id = node_data.get('store_id', None)
            while current_outputs < min_outputs:
                store_label = f"Store[{nid}:{current_outputs}]"
                store_node_id = id(store_label)
                #print(f"store_node_id: {store_node_id}, store_label: {store_label}, store_id: {store_id}")
                domain_node = DomainNode(
                    shape=(1, 1, 1),  # default shape for store nodes
                    unit_size=1,  # default unit size for store nodes
                )
                dom_id = id(domain_node)
                domain_node.id = dom_id  # ensure domain node has a unique ID
                with self.graph_mutation():
                    self.G.add_node(
                        store_node_id,
                        label=store_label,
                        type="Store",
                        domain_node=domain_node,
                        store_id=store_id,
                        expr_obj=store_label,
                        parents=[(nid, 'result')],
                        children=[]
                    )

                    node_data['children'].append((store_node_id, 'result'))

                    edge = Edge(
                        id = (nid, store_node_id, 'output', 'result'),
                        operation = None,
                        source = nid,
                        store_id = store_id,
                        target = store_node_id,
                    )
                    self.G.add_edge(nid, store_node_id, extra=[edge])

                current_outputs += 1

            # Handle tracking all indices going into index base nodes
            # As they define the allocation domain
            if op_type == 'IndexedBase':
                dynamic = False
                # Gather all incoming edges
                incoming_edges = self.G.in_edges(nid, data=True)
                index_symbols = []
                numeric_indices = []
                for src, tgt, data in incoming_edges:
                    if src in self.G.nodes:
                        src_node = self.G.nodes[src]
                        src_type = src_node['type']
                        if src_type in ('Symbol', 'Input', 'Var'):
                            dynamic = True
                        label = src_node['label']
                        index_symbols.append((label, src_type))
                        candidate = src_node.get('constant')
                        if candidate is None:
                            candidate = (src_node.get('attributes') or {}).get(
                                'value'
                            )
                        if candidate is None and isinstance(
                            src_node.get('expr_obj'), ast.Constant
                        ):
                            candidate = src_node['expr_obj'].value
                        if candidate is None:
                            candidate = label
                        try:
                            numeric = float(candidate)
                        except (TypeError, ValueError, OverflowError):
                            dynamic = True
                        else:
                            if not math.isfinite(numeric) or not numeric.is_integer():
                                dynamic = True
                            numeric_indices.append(numeric)
                self.G.nodes[nid]['index_symbols'] = index_symbols
                if not index_symbols:
                    self.G.nodes[nid]['domain_shape'] = ()
                elif not dynamic and len(numeric_indices) == len(index_symbols):
                    # If all indices are static, we can set a fixed domain shape
                    # from zero through each integral coordinate.  Every
                    # incoming index is one axis, not one row in a two-column
                    # ``(label, type)`` table.
                    extents = [
                        (min(value, 0.0), max(value, 0.0))
                        for value in numeric_indices
                    ]
                    domain_shape = tuple(
                        int(extent[1] - extent[0] + 1) for extent in extents
                    )
                    self.G.nodes[nid]['domain_shape'] = domain_shape
                else:
                    # If dynamic, we cannot set a fixed shape, but we can track the symbols
                    self.G.nodes[nid]['domain_shape'] = "dynamic"
                # Set the domain shape based on index symbols

    def group_edges_by_dataset(self, dataG):
        """
        Returns a nested dict grouping each edge by the (role, level, type) tuples found in its 'extras'.
        Structure: { level: { type: { role: [ (src, tgt), ... ] } } }
        """
        grouped = {}
        if dataG is None or not isinstance(dataG, BitTensorMemoryGraph):
            raise ValueError("dataG must be a valid BitTensorMemoryGraph instance")
        
        for src, tgt, attrs in dataG.edges(data=True):
            for ds in attrs.get('extras', []):
                level, typ, role = ds
                # Initialize nested dicts if needed
                grouped.setdefault(level, {}).setdefault(typ, {}).setdefault('input', [])
                grouped[level][typ].setdefault('intermediate', [])
                grouped[level][typ].setdefault('output', [])
                grouped[level][typ].setdefault(role, []).append((src, tgt))
        return grouped

    def check_set_involvement(self, node, nodeset):
        """
        Check if a node is involved in a nodeset.
        Returns True if the node is part of the nodeset, False otherwise.
        """
        for (lvl, typ, role), candidate_node in nodeset:
            if candidate_node == node:
                return (lvl, typ, role)  # return the role, level, type if involved
        return None  # not involved in this nodeset

    def create_data_flow_dag(self, nodesets, uG):
        # use BitTensorMemoryGraph for data flow DAG
        dataG = BitTensorMemoryGraph(size=0)
        datasets = {}   # will map dataset_id -> set of DomainNode.id
        for dataset_id, ns in nodesets.items():         # unpack the dict item
            datasets[dataset_id] = set()
            # for each process node in this nodeset
            for member in ns.member_nodes:              # Node objects
                proc_nid = member.id                     # matches uG’s node IDs
                if proc_nid not in uG:
                    continue
                dom_node = uG.nodes[proc_nid]['domain_node']
                datasets[dataset_id].add(dom_node.id)
                #print(f"Adding domain node {dom_node.id} for dataset {dataset_id} from process node {proc_nid}")
                # add the domain node as a vertex in the new DAG
                dataG.add_node(
                    dom_node.id,
                    proc_node=proc_nid,
                    label=uG.nodes[proc_nid]['label'],
                    type=uG.nodes[proc_nid]['type'],
                    original_node=proc_nid,
                    domain_node=dom_node,
                    dataset_id=dataset_id,
                )

            import itertools

            # … after you’ve added all nodes …

            # now add every uG edge (in or out) for each member
            for member in ns.member_nodes:
                n = member.id
                for src, tgt in itertools.chain(uG.in_edges(n), uG.out_edges(n)):
                    dom_src = uG.nodes[src]['domain_node'].id
                    dom_tgt = uG.nodes[tgt]['domain_node'].id

                    if dataG.has_edge(dom_src, dom_tgt):
                        dataG.edges[dom_src, dom_tgt].setdefault('extras', []).append(dataset_id)
                    else:
                        dataG.add_edge(dom_src, dom_tgt, extras=[dataset_id])

        return dataG


    def compute_levels(self, method='asap', order='processing', interference_mode='asap-maxslack'):
        """
        Compute levels using ILPScheduler.
        - method='asap' for earliest
        - method='alap' for latest
        """
        self.finalize_graph_with_outputs()  # ensure min_outputs satisfied
        self.levels = self.scheduler.compute_levels(method, order)
        if not self.materialize_memory:
            return self.levels
        
        
        self.proc_interference_graph, self.proc_lifespans = self.compute_asap_maxslack_interference(interference_mode)
        self.produce_proc_and_mem_bins(self.proc_lifespans)
        self.universal_graph_interference_bins = self.merge_proc_and_mem_graphs(self.G, self.mG, self.process_bins, self.memory_bins, self.proc_interference_graph)
        self.nodesets = self.condense_to_nodesets()
        self.dataG = self.create_data_flow_dag(self.nodesets, self.uG)
        #print exauhstive summary of items produced
        verbose = False
        if verbose:
            print(f"Levels computed: {len(self.levels)} nodes")
            print(f"Process interference graph: {len(self.proc_interference_graph.nodes)} nodes, {len(self.proc_interference_graph.edges)} edges")
            print(f"Memory interference graph: {len(self.mG.nodes)} nodes, {len(self.mG.edges)} edges")
            print(f"Process bins: {len(self.process_bins)} bins")
            print(f"Memory bins: {len(self.memory_bins)} bins")
            print(f"Nodesets: {len(self.nodesets)} sets")
            print(f"Recombinatorics level: {self.recombinatorics_level}")
            print(f"Domain shape: {self.domain_shape}")
            print(f"Universal graph: {len(self.uG.nodes)} nodes, {len(self.uG.edges)} edges")
            print(f"Universal interference bins: {len(self.uGI.nodes)} nodes, {len(self.uGI.edges)} edges")
            print(f"Universal interference graph: {len(self.uGI.nodes)} nodes, {len(self.uGI.edges)} edges")

    def extract_full_process_graph(self):
        nodes = {}
        for nid, data in self.G.nodes(data=True):
            nodes[nid] = {
                'type': data['type'],
                'label': data['label'],
                'expr_obj': data['expr_obj'],
                'parents': list(data['parents']),
                'children': list(data['children']),
                'level': self.levels.get(nid),
            }
        levels_map = {}
        for nid, lvl in self.levels.items():
            levels_map.setdefault(lvl, []).append(nid)
        # include roots list so consumer knows final outputs
        return {'nodes': nodes, 'levels': levels_map, 'roots': list(self.roots)}

    def build_from_expression(self, expr_or_tensor, *domain_dims):
        # bypass SymPy path for a recorded ProvenanceGraph
        from src.turing_machine.turing_provenance import ProvenanceGraph
        if isinstance(expr_or_tensor, sympy.Basic):
            from ...compiler.symbolic_process_graph import (
                ingest_sympy_expression,
            )

            ingest_sympy_expression(self, expr_or_tensor)
            return self
        if isinstance(expr_or_tensor, ProvenanceGraph):
            # Provenance is already a graph.  Import it directly instead of
            # asking the symbolic/AST introspector to interpret the recorder
            # object itself.  Keep the provenance node ids stable so edges,
            # schedules, and downstream SSA values all share one identity.
            self.domain_shape = (1,)
            self.roots = []
            by_idx = {node.idx: node for node in expr_or_tensor.nodes}
            incoming = {idx: [] for idx in by_idx}
            outgoing = {idx: [] for idx in by_idx}
            for edge in expr_or_tensor.edges:
                role = f"arg{edge.arg_pos}"
                incoming[edge.dst_idx].append((edge.src_idx, role))
                outgoing[edge.src_idx].append((edge.dst_idx, role))

            for idx, node in by_idx.items():
                domain_node = DomainNode(shape=(1, 1, 1), unit_size=1)
                domain_node.id = id(domain_node)
                parents = sorted(
                    incoming[idx], key=lambda item: int(item[1][3:])
                )
                children = sorted(
                    outgoing[idx], key=lambda item: int(item[1][3:])
                )
                result_length = node.metadata.get("result_length")
                tensor = (
                    {"dtype": "bit", "shape": (int(result_length),)}
                    if result_length is not None
                    else {}
                )
                with self.graph_mutation():
                    self.G.add_node(
                        idx,
                        label=node.op,
                        type=node.op,
                        op=node.op,
                        expr_obj=node,
                        extra_args={
                            "kwargs": dict(node.kwargs),
                            "arg_ids": tuple(node.args),
                            "out_obj_id": node.out_obj_id,
                            "metadata": dict(node.metadata),
                        },
                        attributes=dict(node.kwargs),
                        metadata=dict(node.metadata),
                        constant=None,
                        tensor=tensor,
                        bit_quanta=(
                            {
                                "quanta": int(result_length),
                                "bits_per_quantum": 1,
                                "pid_domains": (),
                                "source_nodes": tuple(parent for parent, _ in parents),
                            }
                            if result_length is not None
                            else None
                        ),
                        control={"recorded_by": "turing_provenance"},
                        source_span=None,
                        input_roles=tuple(role for _, role in parents),
                        output_roles=("result",),
                        schema_version=1,
                        domain_node=domain_node,
                        store_id=None,
                        parents=parents,
                        children=children,
                    )
                    self.node_map[idx] = node

            for edge in expr_or_tensor.edges:
                role = f"arg{edge.arg_pos}"
                graph_edge = Edge(
                    id=(edge.src_idx, edge.dst_idx, "result", role),
                    operation=None,
                    source=edge.src_idx,
                    target=edge.dst_idx,
                    store_id=None,
                )
                with self.graph_mutation():
                    self.G.add_edge(edge.src_idx, edge.dst_idx, extra=[graph_edge])

            self.roots = [idx for idx in by_idx if not outgoing[idx]]
            return self

        if isinstance(expr_or_tensor, tuple) and isinstance(expr_or_tensor[1], ExpressionTensor):
            registry, et = expr_or_tensor
            #print(registry)
            
            self.domain_shape = et.domain_shape
            self.roots = []
            def expr_fn(*indices):
                idx = et.data[0, -1][indices].item()
                return registry[idx]
            self.build_lateral_graph_across_domain(*self.domain_shape, expr_fn)
        elif callable(expr_or_tensor):
            self.build_lateral_graph_across_domain(*domain_dims, expr_or_tensor)
        else:
            # treat as single scalar SymPy expression (or trivial 1D shape)
            self.domain_shape = (1,)
            self.roots = []
            self.build_graph(expr_or_tensor)


    def to_sympy(self):
        from ...compiler.symbolic_process_graph import (
            process_graph_to_sympy_package,
        )

        return process_graph_to_sympy_package(self)

        # Historical source-schema-specific implementation retained below for
        # archaeology.  The canonical projector above is now authoritative.
        meta = self.extract_full_process_graph()
        nodes_meta = meta['nodes']
        cache = {}

        def emit(nid):
            if nid in cache:
                return cache[nid]
            m = nodes_meta[nid]
            typ = m['type']

            role_map = {}
            for p, role in m['parents']:
                value = emit(p)
                role_map.setdefault(role, []).append(value)

            if typ in ('Store', 'Output'):
                out = emit(m['parents'][0][0])
            elif typ == 'ImaginaryUnit':
                out = sympy.I
            elif typ == 'Symbol':
                out = sympy.Symbol(m['label'])
            elif typ == 'Integer':
                out = sympy.Integer(int(m['label']))
            elif typ in ('One','Zero','NegativeOne'):
                val = {'One':1,'Zero':0,'NegativeOne':-1}[typ]
                out = sympy.Integer(val)
            elif typ == 'IndexedBase':
                out = sympy.IndexedBase(m['label'])
            elif typ == 'Indexed':
                base = role_map.get("base", [])[0]
                indices = role_map.get("index", [])
                out = sympy.Indexed(base, *indices)
            elif typ == 'Idx':
                if "limit" in role_map and len(role_map["limit"]) == 2:
                    out = sympy.Idx(m['label'], (role_map["limit"][0], role_map["limit"][1]))
                elif "limit" in role_map and len(role_map["limit"]) == 1:
                    out = sympy.Idx(m['label'], role_map["limit"][0])
                else:
                    out = sympy.Idx(m['label'])
            elif typ in ('Mul','Add','Pow','Tuple'):
                cls = {'Mul': sympy.Mul, 'Add': sympy.Add, 'Pow': sympy.Pow, 'Tuple': sympy.Tuple}[typ]
                all_args = sum(role_map.values(), [])
                out = cls(*all_args, evaluate=False)
            elif typ == 'Sum':
                expr_obj = m['expr_obj']
                out = sympy.Sum(expr_obj.args[0], expr_obj.limits)
            else:
                expr_obj = m['expr_obj']
                all_args = sum(role_map.values(), [])
                if isinstance(expr_obj, sympy.Function):
                    out = expr_obj.func(*all_args)
                else:
                    raise ValueError(f"Unhandled type: {typ}")
            
            cache[nid] = out
            return out

        # --- Build nested list of expressions from roots ---
        roots_copy = self.roots.copy()

        def build_nested_list(emit_fn, roots, shape):
            if len(shape) == 1:
                return [emit_fn(roots.pop(0)) for _ in range(shape[0])]
            return [build_nested_list(emit_fn, roots, shape[1:]) for _ in range(shape[0])]

        nested_list_exprs = build_nested_list(emit, roots_copy, self.domain_shape)

        # --- Flatten for simplification ---
        def flatten_nested_list(nested):
            flat = []
            for item in nested:
                if isinstance(item, list):
                    flat.extend(flatten_nested_list(item))
                else:
                    flat.append(item)
            return flat

        flat_exprs = flatten_nested_list(nested_list_exprs)

        # --- Simplify / CSE ---
        simplified = [self.full_recombinatorics(e, self.recombinatorics_level) for e in flat_exprs] \
                    if self.recombinatorics_level > 0 else flat_exprs

        replacements, reduced_exprs = sympy.cse(simplified)

        # --- Build registry (defs first, then main) ---
        expression_registry = []
        registry_defs_count = 0

        for sym, defn in replacements:
            expression_registry.append(sympy.Tuple(sym, defn))
            registry_defs_count += 1

        main_start = registry_defs_count
        for expr in reduced_exprs:
            expression_registry.append(expr)

        # --- Build nested list of indices matching domain shape ---
        flat_indices = list(range(main_start, main_start + len(reduced_exprs)))

        def rebuild_nested_list(shape, flat):
            if len(shape) == 1:
                return [flat.pop(0) for _ in range(shape[0])]
            return [rebuild_nested_list(shape[1:], flat) for _ in range(shape[0])]

        nested_list_indices = rebuild_nested_list(self.domain_shape, flat_indices.copy())

        # --- Convert indices to tensor/array ---
        torch = _optional_torch()
        if torch is not None:
            indices_tensor = torch.tensor(nested_list_indices, dtype=torch.long)
            expr_tensor_data = indices_tensor.unsqueeze(0).unsqueeze(0)  # add context and sequence dims
        else:
            indices_tensor = np.array(nested_list_indices, dtype=int)
            expr_tensor_data = np.expand_dims(np.expand_dims(indices_tensor, 0), 0)

        # --- Build ExpressionTensor ---
        et = ExpressionTensor(
            contexts=[0],
            sequence_length=1,
            domain_shape=self.domain_shape,
            function_index=None
        )
        et.data = expr_tensor_data

        return expression_registry, et

    def run(self, data_sources, operator_funcs=None):
        import numpy as np
        if operator_funcs is None:
            operator_funcs = {}

        results = {}

        # Compose final lookup
        op_dispatch = {**default_funcs, **operator_funcs}

        # Traverse levels in order
        for lvl in sorted(set(self.levels.values())):
            for nid, node_level in self.levels.items():
                if node_level != lvl:
                    continue

                node_data = self.G.nodes[nid]
                typ = node_data['type']
                parents = node_data['parents']

                if not parents:
                    results[nid] = data_sources.get(node_data['label'], node_data['expr_obj'])
                else:
                    role_map = {}
                    for parent_id, role in parents:
                        val = results[parent_id]
                        role_map.setdefault(role, []).append(val)
                    func = op_dispatch.get(typ)
                    if not func:
                        raise TypeError(f"No handler for node type '{typ}'")
                    results[nid] = func(role_map)



        # Build the nested structure according to store_id
        tensor_data = self._create_nested_data_container(self.domain_shape)

        for nid, data in self.G.nodes(data=True):
            break #diagnostic avoidance
            if data['type'] == 'Store':

                if isinstance(results[nid], np.ndarray):
                    # diagnostic dump
                    print(f"Node {nid} ({node_data['label']}): Result is numpy array with shape {results[nid].shape}")
                    
                

                store_idx = data.get('store_id')
                value = results.get(nid)
                if store_idx is not None:
                    self._insert_into_nested(tensor_data, store_idx, value)

        return ExpressionTensor(data=tensor_data, domain_shape=self.domain_shape)

    def consumer_at(self, src):
        """
        Return the real consumer queue if it exists.
        Otherwise, **soft-fail** by returning the dummy queue that
        always yields random floats, so upstream logic keeps running.
        """
        q = self.consumer_queues.get(src)   # whatever container you use
        if q is not None:
            return q
        # soft-fail: keep the pipeline alive
        return _DUMMY_QUEUE
        
    ###############################################################################
    #  Improved single-step executor
    ###############################################################################
    def run_process_node(self, proc_id: int, incoming_value=None):
        """
        Execute a single *process* node (identified by ``proc_id``) once all of its
        mandatory inputs have arrived.

        Parameters
        ----------
        proc_id : int
            The node-id used in ``self.G`` for the process node we are about to run.
        incoming_value : Any, optional
            Fresh data that has just landed in one of this node’s DomainNode
            buffers.  We **cache** it but do *not* rely on the caller to tell us
            which role it belongs to – that information is on the edge metadata.

        Returns
        -------
        result : Any
            • The computed value for ``proc_id`` **once the node is ready**.  
            • *None* if the node is still waiting on other inputs.
        """

        # ------------------------------------------------------------------ setup
        if not hasattr(self, "_value_cache"):
            self._value_cache: dict[int, Any] = {}        # finalised results
        if not hasattr(self, "_pending_inputs"):
            # role-map under construction for each node:  role -> [values …]
            self._pending_inputs: dict[int, dict[str, list]] = {}

        if proc_id in self._value_cache:          # already evaluated
            return self._value_cache[proc_id]

        node_meta   = self.G.nodes[proc_id]
        parents     = node_meta.get("parents", [])
        node_type   = node_meta["type"]

        # ----------------------------------------------------------------- stash the *incoming* value (if any)
        #   We know *which* parent delivered it by consulting the process-graph
        #   edges looking at producer_role / consumer_role pairs stored in Edge.extra.
        if incoming_value is not None:
            for p_id, _ in parents:
                if not self.G.has_edge(p_id, proc_id):
                    continue
                for e in self.G[p_id][proc_id].get("extra", []):
                    if getattr(e, "target", None) == proc_id:
                        role = e.id[3]                     # consumer_role
                        self._pending_inputs \
                            .setdefault(proc_id, {}) \
                            .setdefault(role,   []) \
                            .append(incoming_value)
                        break

        # --------------------------------------------------------- are we “ready”?
        sig          = operator_signatures.get(node_type, operator_signatures["Default"])
        min_inputs   = sig.get("min_inputs", 0)
        pending_roles= self._pending_inputs.get(proc_id, {})
        have_inputs  = sum(len(v) for v in pending_roles.values())

        # If the operator needs more data, just return – we will be called again
        if have_inputs < min_inputs:
            return None

        # ------------------------------------------------------------- build role_map
        role_map: dict[str, list] = pending_roles.copy()
        role_map = {role: [_resolve(v) for v in vals]
                    for role, vals in role_map.items()}
        # Fill literals / zero-parent nodes on demand
        if not parents and not role_map:
            # constants, symbols, etc.
            lit = node_meta.get("expr_obj")
            if isinstance(lit, (int, float, complex, sympy.Basic)):
                role_map.setdefault("value", []).append(lit)

        # ------------------------------------------------------------- dispatch
        op_dispatch   = {**default_funcs}        # you can merge user overrides here
        handler = op_dispatch.get(node_type)

        # ----------  ✨ stop-gap for bare symbols / unknown nodes  ----------
        if handler is None:
            # treat the node as a literal SymPy object and just return it
            lit = node_meta.get("expr_obj")
            if isinstance(lit, sympy.Basic):
                self._value_cache[proc_id] = lit        # memoise
                return lit                              # hand it downstream
            # fall back to original failure for truly unsupported types
            raise TypeError(
                f"No handler registered for node-type '{node_type}' "
                f"with id {proc_id} (parents: {parents})"
            )
        # -------------------------------------------------------------------


        # Convert ExpressionTensor → ndarray so user handlers don’t have to care
        for k, lst in role_map.items():
            for i, item in enumerate(lst):
                if isinstance(item, ExpressionTensor):
                    lst[i] = item.to_numpy()

        try:
            result = handler(role_map)
        except Exception as err:
            raise RuntimeError(f"While executing node {proc_id} ({node_type}): {err}") from err

        # -------------------------------------------------------- commit + cleanup
        self._value_cache[proc_id]  = result
        self._pending_inputs.pop(proc_id, None)           # free memory

        # Also make the result available to this node’s DomainNode so that
        # downstream consumers can `get()` it without recomputing.
        if proc_id in self.dataG.nodes:
            dn = self.dataG.nodes[proc_id]["domain_node"]
            dn.put(("value", result))                      # simple convention

        return result


    def sort_roles(self, grouped):
        """
        Sort roles in the order: input, intermediate, output, followed by any remaining roles.
        """
        basics = ['input', 'intermediate', 'output']
        ordered_keys = []
        for lvl in sorted(grouped):
            for typ in sorted(grouped[lvl]):
                roles_present = list(grouped[lvl][typ].keys())
                for role in basics:
                    if role in grouped[lvl][typ]:
                        ordered_keys.append((lvl, typ, role))
                for role in sorted(roles_present):
                    if role not in basics:
                        ordered_keys.append((lvl, typ, role))
        return ordered_keys


    # ------------------------------------------------------------------ helpers
    def _ensure_domain(self, nid):
        """Return a DomainNode for nid, creating one if necessary."""
        dn = self.dataG.nodes[nid].get('domain_node')
        if dn is None:
            dn       = DomainNode(shape=(1, 1, 1), unit_size=1)
            dn.id    = id(dn)
            self.dataG.nodes[nid]['domain_node'] = dn
        return dn


    # ---------------------------------------------------------------- run_at
    def run_at(self, level=None, type=None, role_=None):
        """
        Execute the data-flow slice identified by (level, type, role_).
        Returns list of results produced at that slice.
        """
        results  = []
        grouped  = self.group_edges_by_dataset(self.dataG)
        for lvl, typ, role in self.sort_roles(grouped):

            # fast-path filters -------------------------------------------------
            if level is not None and lvl != level:      continue
            if type  is not None and typ != type:       continue
            if role_ is not None and role != role_:     continue

            edges = grouped[lvl][typ][role]

            if role == "input":
                for src, tgt in edges:
                    self._ensure_domain(src).put(tgt, self.consumer_at(src))

            elif role == "intermediate":
                for src, tgt in edges:
                    src_dn = self._ensure_domain(src)
                    tgt_dn = self._ensure_domain(tgt)

                    if tgt in self.mG.nodes:            # writing to memory node
                        for next_tgt in self.dataG.edges[src, tgt].get('extras', []):
                            val = src_dn.get(tgt)
                            new_val = self.run_process_node(
                                self.dataG.nodes[src].get('proc_node'), val)
                            tgt_dn.put(next_tgt, new_val)
                    else:                               # plain forward
                        tgt_dn.put(tgt, src_dn.get(tgt))

            elif role == "output":
                print("Running output role...")
                for src, _ in edges:
                    results.extend(self._ensure_domain(src).get_all())
                    print(f"Output from {src}: {results}")

        return results

    def merge_proc_and_mem_graphs(self, proc_graph, mem_graph, proc_bins, mem_bins, proc_interference_graph):
        """
        Merge process and memory graphs into a single graph.
        """
        # replace networkx DiGraph with BitTensorMemoryGraph
        self.uG = BitTensorMemoryGraph(size=0)
        self.uGI = BitTensorMemoryGraph(size=0)  # interference graph
        self.uG.add_nodes_from(proc_graph.nodes(data=True))
        self.uG.add_nodes_from(mem_graph.nodes(data=True))
        self.uGI.add_nodes_from(self.uG.nodes(data=True))
        self.uGI.add_edges_from(mem_graph.edges(data=False))
        self.uGI.add_edges_from(proc_interference_graph.edges(data=False))

        universal_graph_interference_bins = []
        #print(mem_bins, proc_bins)
        for idx, (stage1, stage2) in enumerate(zip(mem_bins, proc_bins)):
            while len(universal_graph_interference_bins) <= idx:
                universal_graph_interference_bins.append([])

            for node in stage1:
                if node in mem_graph:
                    for src, dst, data in proc_graph.edges(data=True):
                        for extra_item in data.get('extra', []):
                            #print(f"Processing edge {src} -> {dst} with extra item {extra_item}")
                            # our memory node ids come from the Edge subedge that defined them
                            if id(extra_item) == mem_graph.nodes[node].get('edge_id') and (src in stage2 or dst in stage2):
                                self.uG.add_edge(src, node, label=f"{self.G.nodes[src]['label']} -> {self.mG.nodes[node]['label']}")
                                self.uG.add_edge(node, dst, label=f"{self.mG.nodes[node]['label']} -> {self.G.nodes[dst]['label']}")
                                if not universal_graph_interference_bins[idx]:
                                    universal_graph_interference_bins[idx] = []
                                #for all permutations of src, node, and dst, add edges
                                for perm in self.tuple_perms((src, node, dst), 2):
                                    self.uGI.add_edge(*perm)
                                universal_graph_interference_bins[idx].append(node)
                                universal_graph_interference_bins[idx].append(src)
                                universal_graph_interference_bins[idx].append(dst)
        return universal_graph_interference_bins
    def tuple_perms(self, tup, r):
        """Generate all r-length permutations of the input tuple."""
        from itertools import permutations
        return list(permutations(tup, r))
    def _create_nested_data_container(self, shape):
        """Create an empty nested list structure of given shape."""
        if not shape:
            return None
        if len(shape) == 1:
            return [None] * shape[0]
        return [self._create_nested_data_container(shape[1:]) for _ in range(shape[0])]

    def _insert_into_nested(self, container, index_tuple, value):
        """Insert value into nested list structure at index_tuple."""
        sub = container
        for idx in index_tuple[:-1]:
            sub = sub[idx]
        sub[index_tuple[-1]] = value
    def produce_proc_and_mem_bins(self, lifespans):
        """
        Produce process and memory bins from lifespans.
        Returns process_bins, memory_bins, min_time, max_time.
        """
        process_bins, memory_bins, min_time, max_time = self.bin_lifespans_to_bins(lifespans)
        self.process_bins = process_bins
        self.memory_bins = memory_bins
        self.min_time = min_time
        self.max_time = max_time
        return process_bins, memory_bins, min_time, max_time
    def print_lifespans_ascii(self, width=50, sort_keys=None):
        """
        Prints an ASCII visualization of lifespans.

        :param width: width of timeline
        :param sort_keys: optional list of key functions for multi-level sort
                        Defaults to ascending start, then descending end.
        """


        for label, bins in [("process", self.process_bins),
                            ("memory", self.memory_bins),
                            ("universal", self.universal_graph_interference_bins)]:
            if not bins:
                print(f"No lifespans to visualize for {label}.")
                continue
            scale = width // (self.max_time - self.min_time + 1)

            # Build node lifespans
            node_lifespans = {}
            for idx, bin_nodes in enumerate(bins):
                for node in bin_nodes:
                    if node not in node_lifespans:
                        node_lifespans[node] = [idx, idx]
                    else:
                        node_lifespans[node][1] = idx

            # Convert to summary records
            node_summaries = [
                {'id': node, 'start': start, 'end': end, 'duration': end - start}
                for node, (start, end) in node_lifespans.items()
            ]

            # Determine sort
            if sort_keys is None:
                sort_keys = [
                    lambda x: x['start'],      # ascending start
                    lambda x: x['end']        # ascending end
                ]

            sorted_nodes = multi_sort(node_summaries, sort_keys)

            # Print
            print(f"\n=== Lifespan Timeline ({label}) ===")
            print(f"Time range: [{self.min_time}, {self.max_time}]")

            for node_info in sorted_nodes:
                node, start, end, duration = (node_info['id'], node_info['start'],
                                            node_info['end'], node_info['duration'])
                line = [' '] * width
                scaled_start = start * scale
                scaled_end = (end+1) * scale
                for i in range(scaled_start, min(scaled_end, width)):
                    line[i] = '#'
                print(f"Node {node}: |{''.join(line)}| start={start} end={end} duration={duration}")

    def bin_lifespans_to_bins(self, lifespans):
        """
        Converts lifespans into bins where each bin contains a list of node IDs.
        """
        # Determine global min/max time
        min_time = min(start for start, end in lifespans.values())
        max_time = max(end for start, end in lifespans.values())
        total_span = max_time - min_time
        start_time = min_time
        offset = 0

        if start_time < 0:
            # If start time is negative, adjust min_time to 0
            min_time = 0
            max_time += -start_time
            offset = -start_time
            total_span += offset

        
        bins = [[] for _ in range(total_span + 1)]
        memory_bins = [[] for _ in range(total_span + 1)]
        
        # replace networkx DiGraph with BitTensorMemoryGraph
        self.mG = BitTensorMemoryGraph(size=0)  # memory graph for edges
        for node, (start, end) in lifespans.items():
            start += offset
            end += offset

            start_idx = (start - min_time)
            end_idx = (end - min_time)

            

            for i in range(start_idx, end_idx + 1):
                bins[i].append(node)

        for idx, bin in enumerate(bins):
            # for each bin,  establish output and input edges as concurrent memory need nodes in the memory bins to make a concurrency graph of storage demands
            if bin:
                for node in bin:
                    for (src, dst, extra) in self.G.edges(node, data='extra'):
                        if extra:
                            for edge in extra:
                                if isinstance(edge, Edge):
                                    # check the schema of the edge for domain node shape hints
                                    # in the event of "many" count, we need to obtain the true shape
                                    # at the present moment
                                    source_node = self.G.nodes[src]
                                    target_node = self.G.nodes[dst]
                                    target_type = target_node['type']
                                    shape = (1,)  # default shape for domain nodes
                                    if target_type in self.role_schemas:
                                        if 'base' in self.role_schemas[target_type]['up']:
                                            # all items get domain nodes but base items
                                            # will have a size associated with them

                                            shape = self.role_schemas[target_type]['up']['base']
                                            
                                            if shape == 'many':
                                                symbolic_engine_object = source_node.get('expr_obj', None)
                                                if symbolic_engine_object is not None:
                                                    print(f"Symbolic engine object for source node: {symbolic_engine_object}")
                                                shape = symbolic_engine_object.shape if hasattr(symbolic_engine_object, 'shape') else (1,)
                                    
                                    domain_node = DomainNode(
                                        shape if isinstance(shape, (list, tuple)) else (shape,),
                                    
                                    )
                                    domain_node.id = id(domain_node)
                                    self.mG.add_node(
                                        id(domain_node),
                                        edge_id=id(edge),
                                        label=f"Memory for: {source_node['label']} -> {target_node['label']}",
                                        domain_node=domain_node,
                                        type='Memory',
                                        store_id=source_node.get('store_id', None),
                                    )
                                    memory_bins[idx].append(id(domain_node))
                                    # we don't extend the domain node over an additional idx
                                    # because it's tracking the process nodes that already
                                    # extend their lifespan over the same idx
        for bin in memory_bins:
            if bin:
                nodes_in_bin = set(bin)
                # Create edges between all nodes in the bin
                for i, src in enumerate(nodes_in_bin):
                    for j, dst in enumerate(nodes_in_bin):
                        if src != dst:
                            self.mG.add_edge(src, dst)

        return bins, memory_bins, min_time, max_time

    

    def compute_asap_maxslack_interference(self, mode='asap-maxslack'):
        interference_graph, lifespans = self.scheduler.compute_asap_maxslack_interference(mode)

        
        return interference_graph, lifespans

    def lateral_graph_merge(self, graphs_meta):
        for G_loc, lvl_loc, nm_loc in graphs_meta:
            for n in G_loc.nodes:
                if n in self.G:            # duplicate only if you reused the same expr object
                    continue               # (rare with id(obj) – safe to ignore)
                meta = G_loc.nodes[n]
                with self.graph_mutation():
                    self.G.add_node(
                        n,
                        label    = meta.get('label', ''),
                        type     = meta.get('type',  ''),
                        expr_obj = meta.get('expr_obj'),
                        parents  = set(),
                        children = set(),
                    )
                    self.node_map[n] = nm_loc.get(n)
                    self.levels[n]   = lvl_loc.get(n, 0)

            for u, v in G_loc.edges:
                if not self.G.has_edge(u, v):
                    with self.graph_mutation():
                        self.G.add_edge(u, v)
                        self.G.nodes[u]['children'].add(v)
                        self.G.nodes[v]['parents'].add(u)

    def group_by_level_and_type(self):
        grouping={}
        for nid in self.G.nodes:
            lvl=self.levels[nid]; tp=self.G.nodes[nid]['type']
            grouping.setdefault(lvl,{}).setdefault(tp,[]).append(nid)
        return grouping

    def build_lateral_graph_across_domain(self, *dims_and_expr):
        *dims, expr_fn = dims_and_expr

        self.domain_shape = dims
        self.roots = []

        def recurse_build(index_prefix, remaining_dims):
            if not remaining_dims:
                try:
                    base_expr = expr_fn(*index_prefix)
                except TypeError:
                    base_expr = expr_fn()
                expr = self.full_recombinatorics(base_expr, self.recombinatorics_level) if self.recombinatorics_level > 0 else base_expr
                self.build_graph(expr, store_id=index_prefix)
                
            else:
                for i in range(remaining_dims[0]):
                    recurse_build(index_prefix + (i,), remaining_dims[1:])

        recurse_build((), dims)
        

    def parse_requirements(self, proc_graph):
        nodes = proc_graph['nodes']
        levels_map = proc_graph['levels']
        # map node id to its level
        id2lvl = {nid: lvl for lvl, ids in levels_map.items() for nid in ids}
        # classified node collections
        input_nodes = {}
        intermediate_nodes = {}
        output_nodes = {}
        operations = {}
        # build Operation objects and classify parent/child roles
        for nid, data in nodes.items():
            lvl = id2lvl[nid]
            sig = operator_signatures.get(data['type'], operator_signatures['Default'])
            if sig == "Store":
                print("parse requirements operatore signature scan, found a Store, confirmed presence")
                exit()
            op = Operation(
                id=nid,
                inputs=data['parents'],
                max_inputs=sig['max_inputs'],
                outputs=data['children'],
                max_outputs=sig['max_outputs'],
                string=data['label'],
                type=data['type'],
                sequence_order=lvl,
                time_penalty=0.0
            )
            operations[nid] = op
            # classify inputs vs intermediates by examining grandparents
            for parent_id, _ in data['parents']:
                grandparents = nodes[parent_id]['parents']
                if grandparents:
                    intermediate_nodes.setdefault(lvl, {}).setdefault(data['type'], []).append(parent_id)
                else:
                    input_nodes.setdefault(lvl, {}).setdefault(data['type'], []).append(parent_id)
            # classify outputs vs intermediates by examining grandchildren
            for child_id, _ in data['children']:
                grandchildren = nodes[child_id]['children']
                if grandchildren:
                    intermediate_nodes.setdefault(lvl, {}).setdefault(data['type'], []).append(child_id)
                else:
                    output_nodes.setdefault(lvl, {}).setdefault(data['type'], []).append(child_id)

        return input_nodes, intermediate_nodes, output_nodes, operations
    
    def condense_to_nodesets(self, proc_graph=None):
        """
        After building graph, optionally condense inputs, intermediates and outputs into NodeSets,
        grouped by (type, level). Returns a dict of NodeSets keyed by (role, level, type).
        """
        if proc_graph is None:
            proc_graph = self.extract_full_process_graph()
        nodes = proc_graph['nodes']
        levels_map = proc_graph['levels']

        # classify by role (input/inter/output)
        inputs, intermediates, outputs, operations = self.parse_requirements(proc_graph)

        nodesets = {}
            

        def create_nodesets(node_group, role):

            for lvl, type_dict in node_group.items():
                for typ, nid_dict in type_dict.items():
                    ids = list(nid_dict)
                    # Determine trivial shape
                    shape = (len(ids), 1, 1)
                    ns = NodeSet(*shape)
                    ns.member_nodes = [Node(id=nid,
                                            location_in_set=ns.nd_from_flat(i),
                                            location_in_memory=self.uG.nodes[nid].get('domain_node', None),
                                            readwrite=READWRITE)
                                       for i, nid in enumerate(ids)]
                    nodesets[(lvl, typ, role)] = ns

        create_nodesets(inputs, "input")
        create_nodesets(intermediates, "intermediate")
        create_nodesets(outputs, "output")
        

        return nodesets
    
    def serialize_bands(self):
        bands={}
        for nid,lvl in self.levels.items():
            tp=self.G.nodes[nid]['type']; lbl=self.G.nodes[nid]['label']
            bands.setdefault(lvl,{}).setdefault(tp,[]).append(lbl)
        return bands



    def setup_consumer_queues(self, src, random_data=False):
        """
        Set up the consumer queue for a given source node.
        If random_data is True, use a generator to supply random float data to the queue.
        """
        if src not in self.consumer_queues:
            self.consumer_queues[src] = deque()

        def random_float_generator():
            while True:
                yield random.uniform(0.0, 1.0)

        if random_data:
            generator = random_float_generator()
            for _ in range(10):  # Populate the queue with 10 initial values
                self.consumer_queues[src].append(next(generator))
        else:
            self.consumer_queues[src].append(None)  # Default behavior

# Example usage:
# animate_data_flow(pg.dataG)

# ----------------------------
# Demo execution (compartmentalized to main)
# ----------------------------
# ----------------------------
def main():
    # Demonstration fixtures are deliberately lazy: importing ProcessGraph is
    # a compiler/library operation and must not execute the expensive symbolic
    # chalkboard construction owned by its demo suite.
    from .graph_express2_tests import test_suite

    # ----------------------------
    # Unified runner
    # ----------------------------
    def run(process_graph, data_sources, expected_fn):
        torch = _optional_torch()
        try:
            result = process_graph.run(data_sources, default_funcs)
            expected = expected_fn(data_sources)
            if isinstance(result, sympy.Basic):
                # If symbolic, turn into numeric function
                symbols = sorted(result.free_symbols, key=lambda s: s.name)
                func = sympy.lambdify(symbols, result, modules='numpy')
                values = [data_sources[str(s)] for s in symbols]
                numeric_result = func(*values)
                assert np.allclose(numeric_result, expected), \
                    f"Graph symbolic did not match expected: {numeric_result} vs {expected}"
            elif torch is not None and isinstance(result, torch.Tensor):
                # If tensor, convert to numpy and compare
                result_np = result.numpy()
                expected_np = np.array(expected)
                assert np.allclose(result_np, expected_np), \
                    f"Graph tensor did not match expected: {result_np} vs {expected_np}"
            else:
                assert np.allclose(result, expected), \
                    f"Graph numeric did not match expected: {result} vs {expected}"
            print("✅ Test passed. Graph matches expected value.")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            
        return result



        


    # ----------------------------
    # Execute all tests
    # ----------------------------
    for idx, test in enumerate(test_suite):
        print(f"\n=== Running test {idx+1}: {test['name']} ===")
        pg = ProcessGraph(5, False)
        pg.build_from_expression(test['expr_fn'], *test['dims'])
        
        
        
        #print("\n--- ASAP schedule ---")
        #pg.compute_levels(method='asap')
        #pg.print_parallel_bands()
        
        #print("\n--- ALAP schedule ---")
        #pg.compute_levels(method='alap')
        #pg.print_parallel_bands()

        #print("\n--- Maxmimum Slack Schedule ---")
        #pg.compute_levels(method='max_slack') 
        #pg.print_parallel_bands()

        # run the original data correctness
        pg.compute_levels(method='alap')  # use ASAP for correct run to match tests
        #pg.print_lifespans_ascii()
        
        data_sources = test['data_sources']()
        #pg.animate_data_flow(pg.dataG, duration=5, fps=2)
        pg.plot_simple_graph(pg.dataG, layout='shell')
        #print(run(pg, data_sources, test['expected_fn']))




if __name__ == "__main__":
    main()
