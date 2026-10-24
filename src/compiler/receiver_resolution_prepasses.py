"""Receiver-resolution prepasses: scalar-intrinsic lowering, specialized precision refresh and bound/grounded method and tensor-operation reference resolution."""

from __future__ import annotations

import ast
import os
import sys
from typing import Any
from types import SimpleNamespace

from ..common.tensors.topological_reducer import _set_operands
from .concordance_declarations import (
    BOUND_RECEIVER as _BOUND_RECEIVER,
    SCALAR_INTRINSIC_RECEIVER as _SCALAR_INTRINSIC_RECEIVER,
)
from ..transmogrifier.function_table import FunctionReference


def _syntax_copy(node: Any) -> Any:
    """Copy an expression's syntax: AST fields deep, annotations shared.

    Program: ``re.compile("ab+")`` pursued into ``re._compiler``.  Wrong
    record: ``copy.deepcopy`` of the authored call also copied the
    compiler's private annotations on its nodes (bindings holding module
    objects), and crashed with "cannot pickle 'mappingproxy' object".  Rule:
    a rewritten expression is new syntax; the annotations the compiler hung
    on the authored nodes are facts about those occurrences and are shared,
    never cloned.
    """

    if isinstance(node, list):
        return [_syntax_copy(item) for item in node]
    if not isinstance(node, ast.AST):
        return node
    copied = type(node)(**{
        field: _syntax_copy(getattr(node, field))
        for field in node._fields if hasattr(node, field)
    })
    for key, value in vars(node).items():
        if key not in node._fields:
            setattr(copied, key, value)
    return copied


def _lower_python_scalar_intrinsics(graph: Any) -> None:
    """Turn primitive method syntax with exact semantics into graph operators."""

    G = graph.G
    for node_id, data in tuple(G.nodes(data=True)):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "bit_length"
            and not expression.args
            and not expression.keywords
        ):
            continue
        attribute_id = next((
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in {"func", "callee", "function"}
            and parent in G
            and isinstance(G.nodes[parent].get("expr_obj"), ast.Attribute)
        ), None)
        receiver_id = (
            None if attribute_id is None else next((
                int(parent)
                for parent, role in G.nodes[attribute_id].get("parents") or ()
                if str(role) in {"value", "object", "base", "operand"}
            ), None)
        )
        if receiver_id is None:
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) == "operand"
            ), None)
        if os.environ.get("TURING_DEBUG_SCALAR_INTRINSIC"):
            print(
                "DEBUGSCALARINTRINSIC "
                f"fn={G.graph.get('function_name')} node={int(node_id)} "
                f"parents={data.get('parents')} attribute={attribute_id} "
                f"receiver={receiver_id}",
                file=sys.stderr,
            )
        if receiver_id is None or receiver_id not in G:
            continue
        for parent, _role in tuple(data.get("parents") or ()):
            if G.has_edge(int(parent), int(node_id)):
                G.remove_edge(int(parent), int(node_id))
            G.nodes[int(parent)]["children"] = [
                (child, role)
                for child, role in G.nodes[int(parent)].get("children", ())
                if int(child) != int(node_id)
            ]
        # The receiver was read by the attribute load; the rewritten call
        # takes that read as its ``operand`` (a fork of the read position).
        _set_operands(
            SimpleNamespace(G=G), int(node_id), [(int(receiver_id), "operand")],
            cause=_SCALAR_INTRINSIC_RECEIVER,
            fork_from=(
                {("operand", 0): (int(attribute_id), "value", 0)}
                if attribute_id is not None else None
            ),
        )
        attributes = dict(data.get("attributes") or {})
        attributes.update({
            "source_operator": "builtins.int.bit_length",
            "python_scalar_intrinsic": True,
            "extraction_identity": "builtins.int.bit_length",
        })
        data.update({
            "type": "BitLength",
            "op": "BitLength",
            "attributes": attributes,
        })


def _refresh_specialized_python_precision(graph: Any) -> None:
    """Apply exact numeric identity after planner specialization is known."""

    from ..common.tensors.topological_reducer import (
        specialize_python_precision_widths,
    )

    function_table = getattr(graph, "function_table", None)
    if function_table is not None:
        # Shared catalogue facts are installed only when every callsite
        # agrees.  Settle those callees first so the caller can read their
        # exact specialized return identity in this same planning phase.
        for entry in function_table:
            callee = getattr(entry, "graph", None)
            if (
                callee is not None
                and callee is not graph
                and (
                    callee.G.graph.get("planner_specializations")
                    or callee.G.graph.get("planner_parameter_classes")
                )
            ):
                specialize_python_precision_widths(callee)
    specialize_python_precision_widths(graph)


def _resolve_bound_function_references(graph: Any) -> None:
    """Turn a specialized callable parameter into an ordinary call edge.

    A first-class source function is represented by its opaque function-table
    address.  When that address crosses a function parameter (for example the
    dt system's ``advance`` callback), the call remains parametric, but this
    specialized card invocation can still link it to the selected callee.
    No Python callable is retained or executed to establish the edge.
    """

    specializations = graph.G.graph.get("planner_specializations") or {}
    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (
            attributes.get("python_precision_boundary")
            or
            attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        ):
            continue
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            continue
        bound = specializations.get(expression.func.id)
        if not isinstance(bound, FunctionReference):
            continue
        attributes = dict(attributes)
        attributes["callee_ref"] = int(bound.address)
        attributes["callee_resolution"] = "bound-function-parameter"
        data["attributes"] = attributes


def _resolve_grounded_method_references(graph: Any) -> None:
    """Attach internal methods only through a proven receiver class.

    A method name being unique in the ingested class catalogue says nothing
    about an unrelated receiver.  In particular, ``os.environ.get`` must not
    become ``_RandomFloatQueue.get`` merely because that is the sole authored
    class method named ``get``.  Follow explicit value producers to a
    ``class_ref`` instead; ambiguity or absence leaves the call external/static
    and fully visible rather than fabricating an internal topology edge.
    """

    class_table = graph.G.graph.get("class_table") or {}
    specializations = graph.G.graph.get("planner_specializations") or {}
    parameter_records = dict(
        graph.G.graph.get("parameter_record_abi") or {}
    )

    # A class call already carries the reducer's exact ``class_ref``.  Resolve
    # its initializer from that class-table row before compilation-unit
    # planning, at the same source-receiver seam as ordinary grounded method
    # calls.  Doing this only in the later hierarchy planner made the call
    # dynamic while the project was divided: the constructor body was absent
    # from the unit and the conditional arm therefore had no producer for the
    # constructed record.  ``class_ref`` and the class-table method edge are
    # the two existing identities; no spelling or host object participates.
    for _node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        if (
            not isinstance(expression, ast.Call)
            or attributes.get("class_ref") is None
            or any(attributes.get(key) is not None for key in (
                "callee_ref", "method_ref", "constructor_ref",
            ))
        ):
            continue
        class_identity = str(attributes["class_ref"])
        class_record = class_table.get(class_identity)
        if not isinstance(class_record, dict):
            continue
        methods = class_record.get("methods") or {}
        reference = methods.get("__new__", methods.get("__init__"))
        if reference is None:
            continue
        data["attributes"] = {
            **attributes,
            "constructor_ref": int(reference),
            "constructor_resolution": "source-class-ref",
        }
        # Persist the early unit-division dependency from the exact class
        # declaration edge.  Activation-root selection is independently
        # anchored to the authored compile target; a constructor binding is a
        # dependency, never a replacement program root.
        from .concordance_declarations import (
            CLASS_DECLARATION, CLASS_METHOD_DECLARATION,
        )
        from .identity_concordance import current_identity_book
        from .planner_concordance_posts import _post_call_binding

        method_name = "__new__" if methods.get("__new__") is not None else "__init__"
        book = current_identity_book()
        method_page = book.page(CLASS_METHOD_DECLARATION)
        call_span_row = getattr(
            expression, "_turing_source_span_row", None,
        )
        source_module = (
            call_span_row[0]
            if isinstance(call_span_row, tuple) and len(call_span_row) == 3
            else None
        )
        matching_rows = tuple(
            row for row in method_page.rows()
            if len(row) == 3
            and (
                source_module is None
                or str(row[0]) == str(source_module)
            )
            and str(row[1]) == class_identity
            and str(row[2]) == method_name
        )
        if (
            graph.G.graph.get("lexical_read_scope") is not None
            and len(matching_rows) != 1
        ):
            # A production graph may expose the dependency only after its
            # exact declaration cell is available.  Leave the call dynamic
            # rather than retaining a graph-only constructor assertion.
            data["attributes"] = attributes
            continue
        if len(matching_rows) == 1:
            method_row = matching_rows[0]
            method_cell = book.latest_ref(
                CLASS_METHOD_DECLARATION, method_row,
            )
            class_cell = book.latest_ref(
                CLASS_DECLARATION, method_row[:2],
            )
            binding_cell = _post_call_binding(
                graph, int(_node_id), int(reference), "constructor",
                source_cells=tuple(filter(None, (class_cell, method_cell))),
            )
            if binding_cell is not None:
                data["attributes"]["constructor_binding_row"] = binding_cell.row

    def source_expression_identity(expression: ast.AST) -> tuple[Any, ...]:
        """Identify one authored expression across copied AST instances."""

        return (
            type(expression).__name__,
            getattr(expression, "lineno", None),
            getattr(expression, "col_offset", None),
            getattr(expression, "end_lineno", None),
            getattr(expression, "end_col_offset", None),
            ast.dump(expression, include_attributes=False),
        )

    precision_boundary_selectors = {
        source_expression_identity(expression.func)
        for _node_id, data in graph.G.nodes(data=True)
        for expression in (data.get("expr_obj"),)
        if (
            (data.get("attributes") or {}).get("python_precision_boundary")
            or (data.get("attributes") or {}).get(
                "python_precision_operator"
            )
        )
        and isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
    }

    def declared_parameter_class(binding_name: str | None) -> str | None:
        """Resolve a receiver from its explicit program-ABI record identity."""

        if binding_name is None:
            return None
        record = parameter_records.get(str(binding_name))
        if not isinstance(record, dict):
            return None
        identity = record.get("identity")
        if identity is None:
            return None
        text = str(identity)
        matches = {
            str(candidate)
            for candidate in class_table
            if str(candidate) == text
            or str(candidate).rsplit(".", 1)[-1]
            == text.rsplit(".", 1)[-1]
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def specialized_class(binding_name: str | None) -> str | None:
        if binding_name is None or str(binding_name) not in specializations:
            return None
        receiver_type = type(specializations[str(binding_name)])
        identities = {
            receiver_type.__name__,
            receiver_type.__qualname__,
            f"{receiver_type.__module__}.{receiver_type.__qualname__}",
        }
        matches = {
            str(identity)
            for identity in class_table
            if str(identity) in identities
            or str(identity).rsplit(".", 1)[-1] == receiver_type.__name__
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def specialized_method_reference(
        binding_name: str | None,
        method_name: str,
    ) -> int | None:
        if binding_name is None or str(binding_name) not in specializations:
            return None
        receiver_type = type(specializations[str(binding_name)])
        owner_names = {
            receiver_type.__name__,
            receiver_type.__qualname__,
            f"{receiver_type.__module__}.{receiver_type.__qualname__}",
        }
        table = getattr(graph, "function_table", None)
        if table is None:
            return None
        matches = {
            int(entry.reference.address)
            for entry in table
            if str(entry.name) == str(method_name)
            and entry.graph is not None
            and str(entry.graph.G.graph.get("method_owner")) in owner_names
        }
        return next(iter(matches)) if len(matches) == 1 else None

    submitted_classes = {
        str(name) for name in graph.G.graph.get("class_definitions") or ()
    }

    def class_object_binding(attributes: dict) -> str | None:
        """The submitted class a free name IS (``Ops`` in ``Ops.m(...)``)."""

        name = attributes.get("binding_name")
        if (
            attributes.get("binding_kind") == "external"
            and name is not None
            and str(name) in submitted_classes
            and str(name) in class_table
        ):
            return str(name)
        return None

    method_bindings = {
        int(entry.reference.address): entry.graph.G.graph.get("method_binding")
        for entry in (getattr(graph, "function_table", None) or ())
        if entry.graph is not None
    }

    def class_object_call_admits(receiver_id: int, reference: Any) -> bool:
        """A call through the class object itself binds a static method.

        Program: ``Ops.min(n, 1)`` with ``Ops.min`` a static method (silent
        lane; the AbstractTensor init thread's repro).  Wrong record: the
        free name ``Ops`` resolved to no class, so the call kept its
        spelling and lowered as the canonical ``min`` reduction over
        ``(Ops, n, 1)``: 0.0 for n=0, where Python returns -1.  Rule: a free
        name that IS a submitted class is that class as a receiver, for a
        static method only.  ``Ops.m(obj, x)`` on an instance method passes
        ``obj`` as ``self``, which a receiver edge cannot state, so it stays
        unlinked.
        """

        node = graph.G.nodes.get(int(receiver_id), {})
        if class_object_binding(node.get("attributes") or {}) is None:
            return True
        return method_bindings.get(int(reference)) == "static"

    def receiver_class(value_id: int) -> str | None:
        pending = [int(value_id)]
        visited = set()
        candidates = set()
        while pending:
            current = pending.pop()
            if current in visited or current not in graph.G:
                continue
            visited.add(current)
            node = graph.G.nodes[current]
            attributes = node.get("attributes") or {}
            record_identity = attributes.get(
                "program_abi_record_identity"
            )
            if record_identity is not None:
                record_matches = {
                    str(candidate)
                    for candidate in class_table
                    if str(candidate) == str(record_identity)
                    or str(candidate).rsplit(".", 1)[-1]
                    == str(record_identity).rsplit(".", 1)[-1]
                }
                if len(record_matches) == 1:
                    candidates.update(record_matches)
                    continue
            # ``class_ref`` marks a construction; ``result_class_ref`` a
            # value that IS an instance of the class -- a call returning
            # one, or a field read of a field that holds one. The reducer
            # resolves receivers through both; so does this.
            class_ref = attributes.get("class_ref")
            if class_ref is None:
                class_ref = attributes.get("result_class_ref")
            if class_ref is not None:
                candidates.add(str(class_ref))
                continue
            if str(node.get("type")) in {"Input", "input"}:
                class_ref = declared_parameter_class(
                    attributes.get("binding_name")
                ) or specialized_class(attributes.get("binding_name"))
                if class_ref is not None:
                    candidates.add(class_ref)
                    continue
                if class_object_binding(attributes) is not None:
                    candidates.add(class_object_binding(attributes))
                    continue
            # Only identity-routing nodes may transmit a receiver class. An
            # arbitrary operation consuming an object does not return it.
            if str(node.get("type")) not in {
                "Input", "Phi", "LoopResult", "LoopExit", "Identity",
            }:
                continue
            pending.extend(
                int(parent)
                for parent, role in (node.get("parents") or ())
                if str(role) in {
                    "value", "body", "orelse", "initial", "updated",
                    "result", "operand",
                }
            )
        return next(iter(candidates)) if len(candidates) == 1 else None

    bound_selectors = {}
    for node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        expression = data.get("expr_obj")
        if (
            attributes.get("python_precision_boundary")
            or attributes.get("python_precision_operator")
            or isinstance(expression, ast.Attribute)
            and source_expression_identity(expression)
            in precision_boundary_selectors
        ):
            for key in ("bound_method_ref", "method_ref", "callee_ref"):
                attributes.pop(key, None)
            continue
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        receiver_id, method_name, receiver_expression = None, None, None
        if isinstance(expression, ast.Attribute):
            receiver_id = roles.get("value", roles.get("operand"))
            method_name, receiver_expression = expression.attr, expression.value
        elif (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
              and expression.func.id == "getattr" and len(expression.args) >= 2
              and isinstance(expression.args[1], ast.Constant)
              and isinstance(expression.args[1].value, str)
              and (data.get("attributes") or {}).get("extraction_identity") == "builtins.getattr"):
            receiver_id = roles.get("arg:0")
            method_name, receiver_expression = expression.args[1].value, expression.args[0]
        if receiver_id is None or method_name is None:
            continue
        class_identity = receiver_class(receiver_id)
        reference = (class_table.get(class_identity, {}).get("methods", {}).get(method_name)
                     if class_identity is not None else None)
        binding_name = (graph.G.nodes[receiver_id].get("attributes") or {}).get("binding_name")
        if method_name in (parameter_records.get(str(binding_name), {}).get("fields") or {}):
            continue  # An instance field can shadow a method.
        if reference is None or not class_object_call_admits(receiver_id, reference):
            continue
        data.setdefault("attributes", {}).update({
            "bound_method_ref": int(reference), "bound_receiver_id": int(receiver_id),
            "receiver_class_ref": class_identity,
        })
        bound_selectors[int(node_id)] = (int(reference), int(receiver_id), method_name, receiver_expression)

    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)):
            continue
        selectors = {int(parent) for parent, role in data.get("parents") or ()
                     if str(role) in {"callee", "func", "function"} and int(parent) in bound_selectors}
        if not selectors:
            # Source linking may already have consumed the callee edge. A
            # single retained lexical binding still proves the saved selector.
            history = tuple((graph.G.graph.get("identity_table") or {}).get(expression.func.id, ()))
            if len(history) == 1 and int(history[0]) in bound_selectors:
                selectors.add(int(history[0]))
        if len(selectors) != 1:
            continue
        selector = next(iter(selectors))
        reference, receiver_id, method_name, receiver_expression = bound_selectors[selector]
        rewritten = _syntax_copy(expression)
        rewritten.func = ast.copy_location(ast.Attribute(
            value=_syntax_copy(receiver_expression), attr=method_name, ctx=ast.Load()), expression.func)
        data.setdefault("authored_expr_obj", expression)
        data["expr_obj"] = rewritten
        data["type"], data["op"] = "Call", "call"
        attrs = dict(data.get("attributes") or {})
        for key in ("callee_ref", "external_ref", "extraction_identity", "intrinsic_identity",
                    "extraction_contract", "extraction_action", "extraction_rule", "extraction_classification",
                    "backend_intrinsic_candidate", "static_python_reference"):
            attrs.pop(key, None)
        attrs.update({"method_ref": reference, "bound_selector_id": selector,
                      "method_resolution": "bound-receiver-class-ref"})
        data["attributes"] = attrs
        parents = [(int(parent), str(role)) for parent, role in data.get("parents") or ()
                   if str(role) not in {"callee", "func", "function", "operand", "receiver"}]
        parents.append((receiver_id, "operand"))
        _set_operands(graph, int(node_id), parents, cause=_BOUND_RECEIVER)

    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (
            attributes.get("python_precision_boundary")
            or attributes.get("python_precision_operator")
            or attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        ):
            continue
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ):
            continue
        receiver_id = next((
            int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) in {"operand", "receiver"}
        ), None)
        if receiver_id is None:
            continue
        class_identity = receiver_class(receiver_id)
        reference = None
        if class_identity is not None:
            reference = (
                class_table.get(class_identity, {}).get("methods", {})
                .get(str(expression.func.attr))
            )
        if (
            reference is None
            and isinstance(expression.func.value, ast.Name)
        ):
            reference = specialized_method_reference(
                expression.func.value.id,
                expression.func.attr,
            )
        if reference is not None and not class_object_call_admits(
            receiver_id, reference,
        ):
            # Left unlinked, the call keeps its spelling, and a method named
            # like a canonical op (``min``, ``sum``) lowers as that op.
            raise ValueError(
                "class-qualified call to a non-static method has no "
                f"receiver binding: {ast.unparse(expression.func)} "
                f"({method_bindings.get(int(reference))} method)"
            )
        if reference is None:
            continue
        attributes = dict(attributes)
        attributes["method_ref"] = int(reference)
        attributes["method_resolution"] = "receiver-class-ref"
        if class_identity is not None:
            attributes["receiver_class_ref"] = class_identity
        data["attributes"] = attributes

    # Lexical loop effects are recorded before receiver classes necessarily
    # resolve. Apply the reducer's source-linked-call rule again now: the
    # resolved callee owns its effects, rather than a second opaque mutation
    # at the callsite. Unresolved external calls keep their original effects.
    linked_calls = {
        int(node_id) for node_id, data in graph.G.nodes(data=True)
        if any((data.get("attributes") or {}).get(key) is not None
               for key in (
                   "method_ref", "callee_ref", "resolved_ast_parent",
                   "indirect_callable_id",
               ))
    }
    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        effects = attributes.get("loop_state_effects")
        if not effects:
            continue
        retained = tuple(
            effect for effect in effects
            if not (
                effect.get("effect_mode", "opaque") == "opaque"
                and int(effect["effect_node_id"]) in linked_calls
            )
        )
        if len(retained) != len(effects):
            data["attributes"] = {**attributes, "loop_state_effects": retained}


def _resolve_grounded_tensor_operations(graph: Any) -> None:
    """Promote method-name candidates only along tensor-valued SSA edges."""

    specializations = graph.G.graph.get("planner_specializations") or {}

    def is_tensor_value(value: Any) -> bool:
        return (
            not isinstance(value, (str, bytes, bytearray, list, tuple, dict, set))
            and hasattr(value, "shape")
            and hasattr(value, "dtype")
        )

    tensor_values = {
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("tensor") is not None
        or (data.get("attributes") or {}).get("tensor") is not None
        or (
            data.get("type") == "Input"
            and is_tensor_value(specializations.get(str(
                (data.get("attributes") or {}).get("binding_name", "")
            )))
        )
    }
    # These ordinary ProcessGraph operators preserve tensor-valuedness when
    # at least one data operand is a tensor.  Keep this as provenance only:
    # the later SSA shape pass and tensor repository lowering still decide
    # the concrete result layout and implementation.
    tensor_value_operators = {
        "Add", "Sub", "Mult", "Mul", "Div", "FloorDiv", "Mod", "Pow",
    }
    changed = True
    while changed:
        changed = False
        for node_id, data in graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            candidate = attributes.get("tensor_candidate")
            if (
                int(node_id) in tensor_values
                and (
                    candidate is None
                    or attributes.get("tensor") is not None
                )
            ):
                continue
            if (
                str(data.get("type") or data.get("op"))
                in tensor_value_operators
                and any(
                    int(parent) in tensor_values
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {"callee", "func"}
                )
            ):
                tensor_values.add(int(node_id))
                changed = True
                continue
            if candidate is None:
                continue
            expression = data.get("expr_obj")
            property_access = (
                isinstance(expression, ast.Attribute)
                or str(data.get("op") or data.get("type") or "").casefold()
                == "getattr"
            )
            method_call = (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
            )
            if not (property_access or method_call):
                continue
            receiver = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {
                    "operand", "receiver", "value", "base", "object",
                }
            ), None)
            if receiver not in tensor_values:
                continue
            attributes = dict(attributes)
            attributes["tensor"] = str(candidate)
            attributes["tensor_resolution"] = "receiver-tensor-value"
            data["attributes"] = attributes
            tensor_values.add(int(node_id))
            changed = True
