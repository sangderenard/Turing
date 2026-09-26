"""Topological reduction stage for AbstractTensor ProcessGraphs."""

from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import MISSING, dataclass, fields, is_dataclass
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import symtable
import textwrap
import types
from typing import Any, Callable, Mapping

import networkx as nx

from ...compiler.native_compiler_accelerators import (
    lexicographical_topological_order,
)
from ...compiler.identity_concordance import current_identity_book
from .abstract_nn.token_encoder import encode_identity_tokens
from .abstract_nn.token_lexicon import structural_context_tokens

from ...compiler.shell_reference_tables import build_class_navigation_table
from ...compiler.loop_ir import (
    MAPPING_MUTATION_OPERATORS as _MAPPING_MUTATION_OPERATORS,
    SEQUENCE_MUTATION_OPERATORS as _SEQUENCE_MUTATION_OPERATORS,
)
from ...transmogrifier.function_table import (
    ExternalFunctionTable,
    FunctionTable,
    ParameterAccess,
    ParameterContract,
    ParameterScope,
    ParameterStorage,
    ParameterTransfer,
)
from ...transmogrifier.ssa_registry import ast_ssa_name_map, c_ssa_name_map
from ...transmogrifier.graph.python_special_cases import (
    canonicalize_python_static_bindings,
    interpret_python_special_case,
    interpret_python_static_value,
)
from ...transmogrifier.graph.python_identity_programs import (
    resolve_python_identity,
)
from ...transmogrifier.graph.node_special_cases import tensor_annotation_identity


logger = logging.getLogger(__name__)


_NUMERIC_FEATURE_TYPES = {
    frozenset({"precision"}): "Precision",
    frozenset({"complex", "precision"}): "ComplexPrecision",
    frozenset({"rational"}): "Rational",
    frozenset({"rational", "precision"}): "RationalPrecision",
    frozenset({"complex", "rational"}): "ComplexRational",
    frozenset({"complex", "rational", "precision"}): (
        "ComplexRationalPrecision"
    ),
}
_NUMERIC_TYPE_FEATURES = {
    name: features for features, name in _NUMERIC_FEATURE_TYPES.items()
}


@dataclass(frozen=True)
class NumericFeatureDescriptor:
    """One authored numeric wrapper contract, before repository SSA.

    This is the source-facing description shared by boundary expansion,
    operator selection, and method selection.  A composite annotation is not
    a scalar with an interesting name: its feature set determines its owned
    component paths, and Precision determines how many scalar limbs each leaf
    owns.
    """

    type_name: str
    features: frozenset[str]
    limbs: int = 1
    element_type: str | None = None

    @property
    def coefficient_paths(self) -> tuple[tuple[str, ...], ...]:
        paths: tuple[tuple[str, ...], ...] = ((),)
        if "complex" in self.features:
            paths = (("real",), ("imag",))
        if "rational" in self.features:
            paths = tuple(
                (*path, component)
                for path in paths
                for component in ("numerator", "denominator")
            )
        return paths

    @property
    def scalar_components(self) -> int:
        width = self.limbs if "precision" in self.features else 1
        return len(self.coefficient_paths) * width

    @property
    def operators(self) -> tuple[str, ...]:
        operations = ["add", "sub", "mul", "truediv", "neg"]
        if "rational" in self.features:
            operations.append("pow_integer")
        return tuple(operations)

    @property
    def methods(self) -> tuple[str, ...]:
        methods = ["collapse"]
        if self.features == frozenset({"precision"}):
            methods.extend(("sqrt", "exp", "log"))
        if "rational" in self.features:
            methods.extend(("components", "reciprocal"))
            if "complex" not in self.features:
                methods.append("quotient")
        if "complex" in self.features:
            methods.extend(("collapse_components", "conjugate"))
        return tuple(methods)

    def receipt(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "features": tuple(sorted(self.features)),
            "limbs": int(self.limbs),
            "element_type": self.element_type,
            "coefficient_paths": self.coefficient_paths,
            "scalar_components": self.scalar_components,
            "operators": self.operators,
            "methods": self.methods,
        }


def numeric_feature_descriptor(
    annotation: str | ast.AST | None,
) -> NumericFeatureDescriptor | None:
    """Normalize one supported numeric annotation into its complete shape."""

    if annotation is None:
        return None
    try:
        parsed = (
            annotation
            if isinstance(annotation, ast.AST)
            else ast.parse(str(annotation), mode="eval").body
        )
    except SyntaxError:
        return None
    target = parsed.value if isinstance(parsed, ast.Subscript) else parsed
    type_name = (
        target.id if isinstance(target, ast.Name)
        else target.attr if isinstance(target, ast.Attribute)
        else None
    )
    features = _NUMERIC_TYPE_FEATURES.get(str(type_name))
    if features is None:
        return None

    limbs, element_type = 1, None
    if isinstance(parsed, ast.Subscript):
        index = parsed.slice
        parts = index.elts if isinstance(index, ast.Tuple) else (index,)
        for part in parts:
            if isinstance(part, ast.Constant) and isinstance(part.value, int):
                limbs = max(int(part.value), 1)
            elif isinstance(part, ast.Name):
                element_type = part.id
            elif isinstance(part, ast.Constant) and isinstance(part.value, str):
                element_type = part.value
    if "precision" not in features:
        limbs = 1
    return NumericFeatureDescriptor(
        str(type_name), features, limbs, element_type,
    )


def join_numeric_feature_descriptors(
    *descriptors: NumericFeatureDescriptor | None,
) -> NumericFeatureDescriptor | None:
    """Canonical feature union and widest Precision contract."""

    present = tuple(descriptor for descriptor in descriptors if descriptor)
    if not present:
        return None
    features = frozenset().union(*(item.features for item in present))
    type_name = _NUMERIC_FEATURE_TYPES.get(features)
    if type_name is None:
        return None
    limbs = max(item.limbs for item in present)
    element_type = _widest_element(*(
        item.element_type for item in present
    ))
    return NumericFeatureDescriptor(
        type_name, features, limbs, element_type,
    )


def _numeric_feature_descriptor_from_receipt(
    receipt: Mapping[str, Any] | None,
) -> NumericFeatureDescriptor | None:
    """Recover the normalized descriptor carried by a graph value."""

    if not receipt:
        return None
    features = frozenset(map(str, receipt.get("features") or ()))
    type_name = str(receipt.get("type_name") or "")
    if not type_name or _NUMERIC_FEATURE_TYPES.get(features) != type_name:
        return None
    return NumericFeatureDescriptor(
        type_name=type_name,
        features=features,
        limbs=max(int(receipt.get("limbs") or 1), 1),
        element_type=(
            None if receipt.get("element_type") is None
            else str(receipt["element_type"])
        ),
    )


def numeric_feature_field_projection(
    descriptor: NumericFeatureDescriptor,
    attribute: str,
) -> NumericFeatureDescriptor | None:
    """Return the exact wrapper held by one structural coefficient field.

    Composite numeric wrappers are nested algebra, not packed anonymous
    scalars. ``real``/``imag`` remove the complex layer and
    ``numerator``/``denominator`` remove the rational layer. Precision width
    is orthogonal to both and therefore survives every projection unchanged.
    An empty feature set denotes an ordinary tensor coefficient and returns
    ``None``.
    """

    features = set(descriptor.features)
    attribute = str(attribute)
    if attribute in {"real", "imag"} and "complex" in features:
        features.remove("complex")
    elif attribute in {"numerator", "denominator"} and "rational" in features:
        features.remove("rational")
    else:
        return None
    projected = frozenset(features)
    if not projected:
        return None
    type_name = _NUMERIC_FEATURE_TYPES.get(projected)
    if type_name is None:
        return None
    return NumericFeatureDescriptor(
        type_name=type_name,
        features=projected,
        limbs=(descriptor.limbs if "precision" in projected else 1),
        element_type=descriptor.element_type,
    )


def numeric_feature_method_components(
    descriptor: NumericFeatureDescriptor,
    method: str,
) -> tuple[tuple[tuple[str, ...], NumericFeatureDescriptor | None], ...]:
    """Describe the ordered structural results of a numeric intrinsic."""

    if str(method) != "components":
        return ()
    if "complex" in descriptor.features:
        fields = ("real", "imag")
    elif "rational" in descriptor.features:
        fields = ("numerator", "denominator")
    else:
        return ()
    return tuple(
        ((field,), numeric_feature_field_projection(descriptor, field))
        for field in fields
    )


def _record_numeric_annotation_descriptors(graph: Any) -> dict[str, Any]:
    """Publish normalized numeric contracts at topology-reducer entry.

    Raw annotation spellings remain available for source archaeology. This
    adjacent table is the executable meaning consumed by operators, methods,
    calls, and ABI expansion, so none of those stages has to reinterpret an
    annotation independently.
    """

    target = getattr(graph, "G", graph)
    metadata = getattr(target, "graph", None)
    if not isinstance(metadata, dict):
        return {}
    by_function = metadata.get("function_parameter_annotations") or {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for function_name, parameters in by_function.items():
        described = {}
        for parameter, annotation in (parameters or {}).items():
            descriptor = numeric_feature_descriptor(annotation)
            if descriptor is not None:
                described[str(parameter)] = descriptor.receipt()
        if described:
            normalized[str(function_name)] = described
    metadata["function_parameter_numeric_descriptors"] = normalized

    local = {}
    for name, annotation in (metadata.get("type_annotations") or {}).items():
        descriptor = numeric_feature_descriptor(annotation)
        if descriptor is not None:
            local[str(name)] = descriptor.receipt()
    metadata["local_numeric_descriptors"] = local
    return {
        "parameters": normalized,
        "locals": local,
    }


def _concord_source_value_class(
    scope: str,
    value_id: int,
    class_identity: str,
    *,
    precision_limbs: int = 1,
    source: str,
) -> tuple[str, int]:
    """Commit the class and precision width carried by a source value."""

    page = current_identity_book().page("source_value_class_concordance")
    row = (str(scope), int(value_id))
    proposed = (str(class_identity), max(int(precision_limbs or 1), 1))
    incumbent = page.latest(row)
    if incumbent is not None:
        recorded = (str(incumbent[0]), int(incumbent[1]))
        if recorded != proposed:
            raise ValueError(
                "source value class concordance disagreement for "
                f"{scope!r} value {value_id}: recorded={recorded!r}, "
                f"recorded_source={tuple(incumbent[2:])!r}, "
                f"{source} says {proposed!r}"
            )
        return recorded
    page.set(row, 0, (*proposed, str(source)))
    return proposed


def _source_numeric_scope(graph: Any, fallback: str = "<module>") -> str:
    """Return the class-qualified identity of one reduced source function."""

    target = getattr(graph, "G", graph)
    metadata = getattr(target, "graph", {}) or {}
    specialized = metadata.get("source_numeric_scope")
    if specialized is not None:
        return str(specialized)
    name = str(metadata.get("function_name") or fallback)
    owner = metadata.get("method_owner")
    return name if owner is None else f"{owner}.{name}"


def specialize_python_precision_widths(graph: Any) -> bool:
    """Refresh Precision identity after exact callsite specialization.

    Source reduction necessarily sees a generic function before callsite
    planning proves structural parameters such as ``limbs``.  A width chosen
    there is therefore provisional.  This pass runs on the clean specialized
    graph, gives that specialization its own concordance scope, and repeats
    only the existing Precision identity decisions: call result width,
    arithmetic spelling, and boundary width.  It does not merge, inline, or
    execute specializations.
    """

    target_wrapper = graph
    target = getattr(graph, "G", graph)
    metadata = getattr(target, "graph", {})
    specializations = dict(metadata.get("planner_specializations") or {})

    def stable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(sorted(
                (str(key), stable(item)) for key, item in value.items()
            ))
        if isinstance(value, (tuple, list)):
            return tuple(stable(item) for item in value)
        if isinstance(value, (str, bytes, bool, int, float, type(None))):
            return value
        return repr(value)

    authored_scope = str(
        metadata.get("method_owner") or ""
    )
    function_name = str(metadata.get("function_name") or "<module>")
    authored_scope = (
        f"{authored_scope}.{function_name}"
        if authored_scope else function_name
    )
    parameter_classes = dict(metadata.get("planner_parameter_classes") or {})
    receipt = tuple(sorted(
        (str(name), stable(value))
        for name, value in specializations.items()
    )) + tuple(
        (f"class:{name}", stable(value))
        for name, value in sorted(parameter_classes.items())
    )
    digest = hashlib.sha256(repr(receipt).encode("utf-8")).hexdigest()[:16]
    scope = f"{authored_scope}#specialization:{digest}"
    identity_page = current_identity_book().page(
        "source_numeric_specialization_concordance"
    )
    identity_row = (authored_scope, digest)
    identity_fact = (scope, receipt)
    incumbent = identity_page.latest(identity_row)
    if incumbent is not None and tuple(incumbent) != identity_fact:
        raise ValueError(
            "source numeric specialization concordance disagreement for "
            f"{identity_row!r}: recorded={incumbent!r}, "
            f"proposed={identity_fact!r}"
        )
    if incumbent is None:
        identity_page.set(identity_row, 0, identity_fact)
    metadata["source_numeric_scope"] = scope
    metadata["source_numeric_specialization_receipt"] = receipt
    # A specialization is its authored function with some inputs known: the
    # authored function's committed facts hold in it (never a sibling
    # specialization's, which were decided for other inputs).
    class_page = current_identity_book().page(
        "source_value_class_concordance"
    )
    if str(authored_scope) != scope:
        for row in tuple(class_page.rows()):
            if (
                isinstance(row, tuple) and len(row) == 2
                and row[0] == authored_scope
                and row[1] in target
                and class_page.latest((scope, row[1])) is None
            ):
                for column, fact in class_page.history(row):
                    class_page.set((scope, row[1]), column, fact)
    # The parameters are what the call passed: concord them in this scope.
    for node_id, data in target.nodes(data=True):
        if data.get("type") != "Input":
            continue
        name = (data.get("attributes") or {}).get("binding_name")
        if name is None or str(name) not in parameter_classes:
            continue
        class_identity, limbs = parameter_classes[str(name)]
        _concord_source_value_class(
            scope, int(node_id), str(class_identity),
            precision_limbs=int(limbs),
            source=f"specialization call argument {name}",
        )
    # A Precision parameter is where this scope's Precision interval begins:
    # the value enters here already assembled at its concorded width (its
    # limbs are supplied by the caller), so it opens the interval as a pack.
    for node_id, data in target.nodes(data=True):
        if data.get("type") != "Input":
            continue
        fact = class_page.latest((scope, int(node_id)))
        if (
            isinstance(fact, tuple) and len(fact) >= 2
            and str(fact[0]).rsplit(".", 1)[-1] == "Precision"
        ):
            _concord_source_precision_boundary(
                scope, int(node_id), "pack", int(node_id),
                precision_limbs=max(int(fact[1]), 1),
            )

    def static_int(expression: ast.AST) -> int | None:
        try:
            value = ast.literal_eval(expression)
        except (TypeError, ValueError, SyntaxError):
            if not isinstance(expression, ast.Name):
                return None
            value = specializations.get(expression.id)
        if isinstance(value, bool):
            return None
        try:
            return max(int(value), 1)
        except (TypeError, ValueError, OverflowError):
            return None

    def is_precision(identity: Any) -> bool:
        return (
            identity is not None
            and str(identity).rsplit(".", 1)[-1] == "Precision"
        )

    function_table = getattr(target_wrapper, "function_table", None)

    def tensor_descriptor(value_id: int) -> dict[str, Any] | None:
        if int(value_id) not in target:
            return None
        descriptor = target.nodes[int(value_id)].get("tensor")
        if isinstance(descriptor, Mapping) and tuple(
            descriptor.get("shape") or ()
        ):
            return copy.deepcopy(dict(descriptor))

        # Exact callsite planning publishes return extents to the concordance
        # because specialized graphs are disposable copies.  Numeric
        # specialization runs on another such copy, so a private ``tensor``
        # field is not an adequate hand-off.  Read the same authored value
        # row used by ``publish_call_result_shape`` before concluding that a
        # Precision promotion is scalar.
        from ...compiler.identity_concordance import proven_shape_of

        data = target.nodes[int(value_id)]
        source_value_id = int(data.get("value_id", value_id))
        for function in (
            metadata.get("source_numeric_scope"),
            metadata.get("function_name"),
        ):
            if function is None:
                continue
            extents = proven_shape_of(str(function), source_value_id)
            if extents:
                materialized = (
                    copy.deepcopy(dict(descriptor))
                    if isinstance(descriptor, Mapping) else {}
                )
                materialized["shape"] = tuple(extents)
                materialized["rank"] = len(extents)
                materialized.setdefault("dtype", "float64")
                return materialized
        return (
            copy.deepcopy(dict(descriptor))
            if isinstance(descriptor, Mapping) else None
        )

    def publish_tensor(value_id: int, descriptor: Any) -> None:
        if not isinstance(descriptor, Mapping):
            return
        extents = tuple(descriptor.get("shape") or ())
        if not extents:
            return
        materialized = copy.deepcopy(dict(descriptor))
        materialized["shape"] = extents
        materialized.setdefault("rank", len(extents))
        target.nodes[int(value_id)]["tensor"] = materialized
        from ...compiler.identity_concordance import record_proven_shape

        record_proven_shape(
            scope, int(value_id), extents,
            materialized.get("dtype"), level=None,
        )

    def call_return_identity(
        attributes: Mapping[str, Any], expression: ast.Call,
    ) -> tuple[str | None, int | None, dict[str, Any] | None]:
        """Read the authored return class and exact specialized limbs."""

        class_identity = attributes.get(
            "result_class_ref", attributes.get("class_ref")
        )
        reference = attributes.get(
            "callee_ref", attributes.get("method_ref")
        )
        callee = None
        return_tensor = None
        if reference is not None and function_table is not None:
            try:
                callee = function_table.entry(int(reference)).graph
            except (KeyError, TypeError, ValueError):
                callee = None
        if class_identity is None and callee is not None:
            outputs = tuple(callee.G.graph.get("function_outputs") or ())
            identities = callee.G.graph.get("identity_table") or {}
            candidates = tuple(
                int(value_id)
                for output in outputs
                for value_id in identities.get(str(output), ())
                if int(value_id) in callee.G
            )
            for value_id in reversed(candidates):
                candidate, _width = _resolved_source_value_class(
                    callee.G, value_id
                )
                if candidate is not None:
                    class_identity = candidate
                    descriptor = callee.G.nodes[value_id].get("tensor")
                    if isinstance(descriptor, Mapping):
                        return_tensor = copy.deepcopy(dict(descriptor))
                    break
            if os.environ.get("TURING_DEBUG_PRECISION_SPECIALIZATION"):
                logger.warning(
                    "PRECISION-RETURN-IDENTITY caller=%s reference=%s "
                    "outputs=%r candidates=%r descriptors=%r",
                    authored_scope, reference, outputs, candidates,
                    tuple(
                        (value_id, _resolved_source_value_class(
                            callee.G, value_id
                        ))
                        for value_id in candidates
                    ),
                )
        width = None
        if callee is not None:
            positional = tuple(map(str,
                callee.G.graph.get("positional_parameters")
                or callee.G.graph.get("function_parameters")
                or ()
            ))
            if "limbs" in positional:
                position = positional.index("limbs")
                if position < len(expression.args):
                    width = static_int(expression.args[position])
        return (
            None if class_identity is None else str(class_identity),
            width,
            return_tensor,
        )

    changed = False
    precision_flow_widths: dict[int, int] = {}
    precision_flow_tensors: dict[int, dict[str, Any]] = {}
    # The width argument is an authored part of both Precision constructors
    # and Precision.of.  Calls returning Precision through a helper retain
    # the same proof when that call passes this specialization's exact limbs
    # parameter; the class identity was already established by source
    # reduction and is not guessed here.
    for node_id, data in tuple(target.nodes(data=True)):
        expression = data.get("expr_obj")
        attributes = data.setdefault("attributes", {})
        if not isinstance(expression, ast.Call):
            continue
        class_identity, call_width, return_tensor = call_return_identity(
            attributes, expression
        )
        if not is_precision(class_identity):
            continue
        width = None
        if len(expression.args) >= 2 and (
            isinstance(expression.func, ast.Name)
            and expression.func.id == "Precision"
            or isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "of"
        ):
            width = static_int(expression.args[1])
        if width is None:
            width = call_width
        if width is None:
            for argument in expression.args:
                if (
                    isinstance(argument, ast.Name)
                    and argument.id in specializations
                    and argument.id == "limbs"
                ):
                    width = static_int(argument)
                    break
        if width is None:
            continue
        prior = int(attributes.get("precision_limbs") or 1)
        attributes["precision_limbs"] = int(width)
        attributes["result_class_ref"] = str(class_identity)
        if return_tensor is not None:
            publish_tensor(int(node_id), return_tensor)
        _concord_source_value_class(
            scope, int(node_id), str(class_identity),
            precision_limbs=int(width),
            source="callsite-specialized Precision result",
        )
        value_id = next((
            int(parent) for parent, role in data.get("parents", ())
            if str(role) in {"arg:0", "arg0"}
        ), None)
        is_promotion = (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "of"
        )
        is_pack = (
            isinstance(expression.func, ast.Name)
            and expression.func.id == "Precision"
        )
        if value_id is not None and is_promotion:
            attributes["python_precision_boundary"] = "promote"
            for field in (
                "callee_ref", "method_ref", "constructor_ref", "class_ref",
            ):
                attributes.pop(field, None)
            accessor_id = id(expression.func)
            if accessor_id in target:
                accessor_attributes = target.nodes[accessor_id].setdefault(
                    "attributes", {}
                )
                for field in (
                    "callee_ref", "method_ref", "bound_method_ref",
                ):
                    accessor_attributes.pop(field, None)
                accessor_attributes["python_precision_boundary"] = (
                    "promote-selector"
                )
            data["type"] = "Cast"
            data["op"] = "Cast"
            _replace_inputs(
                target_wrapper, int(node_id), ((int(value_id), "operand"),)
            )
            _concord_source_precision_boundary(
                scope, int(node_id), "promote", int(value_id),
                precision_limbs=int(width),
            )
            source_tensor = tensor_descriptor(int(value_id))
            if source_tensor is not None:
                publish_tensor(int(node_id), source_tensor)
        elif value_id is not None and is_pack and value_id in target:
            value_attributes = target.nodes[value_id].get("attributes") or {}
            leaves = tuple(map(int,
                value_attributes.get("aggregate_leaf_value_ids")
                or value_attributes.get("materialized_value_ids")
                or value_attributes.get("materialized_source_value_ids")
                or ()
            ))
            if len(leaves) >= int(width) and all(
                leaf in target for leaf in leaves[:int(width)]
            ):
                leaves = leaves[:int(width)]
                attributes["python_precision_boundary"] = "pack"
                attributes["precision_pack_source_id"] = int(value_id)
                attributes["precision_pack_limb_ids"] = leaves
                for field in (
                    "callee_ref", "method_ref", "constructor_ref", "class_ref",
                ):
                    attributes.pop(field, None)
                data["type"] = PRECISION_PACK_NAME
                data["op"] = PRECISION_PACK_NAME
                _replace_inputs(
                    target_wrapper, int(node_id), tuple(
                        (leaf, f"limb:{index}")
                        for index, leaf in enumerate(leaves)
                    )
                )
                pack_page = current_identity_book().page(
                    "source_precision_pack_concordance"
                )
                pack_row = (scope, int(node_id))
                pack_fact = (leaves, int(width), int(value_id))
                incumbent_pack = pack_page.latest(pack_row)
                if (
                    incumbent_pack is not None
                    and tuple(incumbent_pack) != pack_fact
                ):
                    raise ValueError(
                        "source precision pack concordance disagreement for "
                        f"{pack_row!r}: recorded={incumbent_pack!r}, "
                        f"proposed={pack_fact!r}"
                    )
                if incumbent_pack is None:
                    pack_page.set(pack_row, 0, pack_fact)
                _concord_source_precision_boundary(
                    scope, int(node_id), "pack", int(value_id),
                    precision_limbs=int(width),
                )
                source_tensor = tensor_descriptor(int(leaves[0]))
                if source_tensor is not None:
                    publish_tensor(int(node_id), source_tensor)
        precision_flow_widths[int(node_id)] = int(width)
        seeded_tensor = tensor_descriptor(int(node_id))
        if seeded_tensor is not None:
            precision_flow_tensors[int(node_id)] = seeded_tensor
        changed = changed or prior != int(width)

    # Calls inside comprehensions return through LoopResult/tuple projection
    # nodes before the named value reaches arithmetic.  Those nodes are
    # structural carriers, not new numeric decisions.  Follow their exact
    # producer edges and refresh only values source reduction already proved
    # to be Precision; no unrelated value acquires a class here.
    for node_id, data in tuple(target.nodes(data=True)):
        identity, width = _resolved_source_value_class(target, int(node_id))
        if is_precision(identity) and int(width) > 1:
            precision_flow_widths[int(node_id)] = int(width)
            seeded_tensor = tensor_descriptor(int(node_id))
            if seeded_tensor is not None:
                precision_flow_tensors[int(node_id)] = seeded_tensor
    if os.environ.get("TURING_DEBUG_PRECISION_SPECIALIZATION"):
        for node_id, data in tuple(target.nodes(data=True)):
            if authored_scope not in {
                "compile_interpolant", "_wide_from_columns", "_pick_columns",
                "selected", "_locate_nearer_columns",
            }:
                break
            attributes = data.get("attributes") or {}
            expression = data.get("expr_obj")
            if (
                int(node_id) in precision_flow_widths
                or is_precision(attributes.get(
                    "result_class_ref", attributes.get("class_ref")
                ))
                or isinstance(expression, (ast.BinOp, ast.Call))
                or authored_scope == "selected"
                or authored_scope == "_locate_nearer_columns"
                or authored_scope == "_wide_from_columns"
                and int(node_id) in {51, 52, 58, 59, 60, 62, 66, 75}
            ):
                logger.warning(
                    "PRECISION-SPECIALIZATION scope=%s node=%s type=%s "
                    "class=%s width=%s flow=%s tensor=%r parents=%r "
                    "aggregate=%r binding=%r",
                    scope, node_id, data.get("type"),
                    attributes.get(
                        "result_class_ref", attributes.get("class_ref")
                    ), attributes.get("precision_limbs"),
                    precision_flow_widths.get(int(node_id)),
                    data.get("tensor"),
                    data.get("parents"),
                    attributes.get("aggregate_leaf_value_ids"),
                    attributes.get("binding_name"),
                )
                if (
                    authored_scope == "_wide_from_columns"
                    and int(node_id) in {
                        50, 51, 52, 59, 60, 62, 66, 75, 76, 77, 78, 79,
                        80, 81,
                    }
                    or authored_scope == "compile_interpolant"
                    or authored_scope == "_locate_nearer_columns"
                ):
                    logger.warning(
                        "PRECISION-SPECIALIZATION-DETAIL scope=%s node=%s "
                        "attributes=%r parents=%r children=%r expression=%s",
                        scope, node_id, attributes, data.get("parents"),
                        data.get("children"),
                        ast.dump(expression, include_attributes=False)
                        if isinstance(expression, ast.AST) else None,
                    )
    propagated = True
    while propagated:
        propagated = False
        for node_id, data in tuple(target.nodes(data=True)):
            parent_widths = tuple(
                precision_flow_widths[int(parent)]
                for parent, _role in data.get("parents", ())
                if int(parent) in precision_flow_widths
            )
            if not parent_widths:
                continue
            parent_tensors = tuple(
                precision_flow_tensors[int(parent)]
                for parent, _role in data.get("parents", ())
                if int(parent) in precision_flow_tensors
                and tuple(
                    precision_flow_tensors[int(parent)].get("shape") or ()
                )
            )
            width = max(parent_widths)
            if precision_flow_widths.get(int(node_id), 1) < width:
                precision_flow_widths[int(node_id)] = int(width)
                propagated = True
            if parent_tensors and int(node_id) not in precision_flow_tensors:
                selected_tensor = max(
                    parent_tensors,
                    key=lambda item: len(tuple(item.get("shape") or ())),
                )
                precision_flow_tensors[int(node_id)] = copy.deepcopy(
                    selected_tensor
                )
                propagated = True
            attributes = data.setdefault("attributes", {})
            class_identity = attributes.get(
                "result_class_ref", attributes.get("class_ref")
            )
            if not is_precision(class_identity):
                continue
            prior = int(attributes.get("precision_limbs") or 1)
            if prior == width:
                continue
            attributes["precision_limbs"] = int(width)
            attributes["result_class_ref"] = str(class_identity)
            if int(node_id) in precision_flow_tensors:
                publish_tensor(
                    int(node_id), precision_flow_tensors[int(node_id)]
                )
            _concord_source_value_class(
                scope, int(node_id), str(class_identity),
                precision_limbs=int(width),
                source="callsite-specialized Precision producer flow",
            )
            changed = True

    operator_page = current_identity_book().page(
        "source_precision_operator_concordance"
    )
    settled = True
    while settled:
        settled = False
        for node_id, data in tuple(target.nodes(data=True)):
            expression = data.get("expr_obj")
            if not isinstance(expression, (ast.BinOp, ast.UnaryOp)):
                continue
            parents = {
                str(role): int(parent)
                for parent, role in data.get("parents", ())
                if int(parent) in target
            }
            operand_ids = (
                (parents.get("lhs"), parents.get("rhs"))
                if isinstance(expression, ast.BinOp)
                else (parents.get("operand"),)
            )
            if any(value_id is None for value_id in operand_ids):
                continue
            descriptors = tuple(
                (
                    ("Precision", precision_flow_widths[int(value_id)])
                    if int(value_id) in precision_flow_widths
                    else _resolved_source_value_class(
                        target, int(value_id)
                    )
                )
                for value_id in operand_ids
            )
            precision = tuple(
                descriptor for descriptor in descriptors
                if is_precision(descriptor[0])
            )
            if not precision:
                continue
            width = max(descriptor[1] for descriptor in precision)
            operation = _qualified_handler(
                "binop" if isinstance(expression, ast.BinOp) else "unaryop",
                expression.op,
                limbs=width,
            )
            class_identity = str(precision[0][0])
            attributes = data.setdefault("attributes", {})
            prior = (
                str(data.get("op") or data.get("type")),
                int(attributes.get("precision_limbs") or 1),
            )
            data["type"] = operation
            data["op"] = operation
            attributes.update({
                "python_precision_operator": True,
                "precision_limbs": int(width),
                "result_class_ref": class_identity,
            })
            operand_tensors = tuple(
                precision_flow_tensors.get(int(value_id))
                or tensor_descriptor(int(value_id))
                for value_id in operand_ids
            )
            shaped_operands = tuple(
                descriptor for descriptor in operand_tensors
                if isinstance(descriptor, Mapping)
                and tuple(descriptor.get("shape") or ())
            )
            if shaped_operands:
                result_tensor = max(
                    shaped_operands,
                    key=lambda item: len(tuple(item.get("shape") or ())),
                )
                publish_tensor(int(node_id), result_tensor)
                precision_flow_tensors[int(node_id)] = copy.deepcopy(
                    result_tensor
                )
            _concord_source_value_class(
                scope, int(node_id), class_identity,
                precision_limbs=int(width),
                source="callsite-specialized Precision operator",
            )
            precision_flow_widths[int(node_id)] = int(width)
            row = (scope, int(node_id))
            fact = (
                str(operation), int(operand_ids[0]), class_identity,
                int(width),
            )
            incumbent = operator_page.latest(row)
            if incumbent is not None and tuple(incumbent) != fact:
                raise ValueError(
                    "source precision operator concordance disagreement for "
                    f"{row!r}: recorded={incumbent!r}, proposed={fact!r}"
                )
            if incumbent is None:
                operator_page.set(row, 0, fact)
            now = (str(operation), int(width))
            if prior != now:
                settled = True
                changed = True

    for node_id, data in tuple(target.nodes(data=True)):
        expression = data.get("expr_obj")
        attributes = data.setdefault("attributes", {})
        boundary = attributes.get("python_precision_boundary")
        if (
            boundary is None
            and isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "collapse"
        ):
            boundary = "collapse"
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and boundary in {"promote", "collapse"}
        ):
            continue
        operand_id = next((
            int(parent) for parent, role in data.get("parents", ())
            if str(role) == "operand"
        ), None)
        if operand_id is None:
            continue
        if boundary == "promote":
            width = static_int(expression.args[1]) if (
                len(expression.args) >= 2
            ) else None
        else:
            _identity, width = _resolved_source_value_class(
                target, operand_id
            )
            if not is_precision(_identity):
                continue
        if width is None:
            continue
        prior = int(attributes.get("precision_limbs") or 1)
        attributes.update({
            "python_precision_boundary": str(boundary),
            "precision_limbs": int(width),
        })
        if boundary == "collapse":
            for field in (
                "callee_ref", "method_ref", "constructor_ref", "class_ref",
            ):
                attributes.pop(field, None)
            accessor_id = id(expression.func)
            if accessor_id in target:
                accessor_attributes = target.nodes[accessor_id].setdefault(
                    "attributes", {}
                )
                for field in (
                    "callee_ref", "method_ref", "bound_method_ref",
                ):
                    accessor_attributes.pop(field, None)
                accessor_attributes["python_precision_boundary"] = (
                    "collapse-selector"
                )
            data["type"] = "Cast"
            data["op"] = "Cast"
            _replace_inputs(
                target_wrapper, int(node_id), ((int(operand_id), "operand"),)
            )
        _concord_source_precision_boundary(
            scope, int(node_id), str(boundary), int(operand_id),
            precision_limbs=int(width),
        )
        changed = changed or prior != int(width)
    return changed


def _resolved_source_value_class(
    graph: Any,
    value_id: Any,
) -> tuple[str | None, int]:
    """What this value is: its concorded class and width, or unknown."""

    target = getattr(graph, "G", graph)
    if not isinstance(value_id, int) or value_id not in target:
        return None, 1
    fact = current_identity_book().page(
        "source_value_class_concordance"
    ).latest((_source_numeric_scope(target), int(value_id)))
    if not isinstance(fact, tuple) or len(fact) < 2:
        return None, 1
    return str(fact[0]), max(int(fact[1]), 1)


def _concorded_numeric_descriptor(
    graph: Any, value_id: Any,
) -> NumericFeatureDescriptor | None:
    """What this numeral value is, from the concordance (or unknown)."""

    class_identity, limbs = _resolved_source_value_class(graph, value_id)
    if class_identity is None:
        return None
    return numeric_feature_descriptor(
        class_identity.rsplit(".", 1)[-1]
        + (f"[{int(limbs)}]" if int(limbs) > 1 else "")
    )


def _concord_numeric_feature_projection(
    scope: str,
    result_id: int,
    receiver_id: int,
    attribute: str,
    component_path: tuple[str, ...],
    descriptor: NumericFeatureDescriptor | None,
) -> tuple[Any, ...]:
    """Commit one structural numeric field projection to the identity book."""

    page = current_identity_book().page(
        "source_numeric_component_concordance"
    )
    row = (str(scope), int(result_id))
    fact = (
        int(receiver_id), str(attribute), tuple(map(str, component_path)),
        None if descriptor is None else descriptor.receipt(),
    )
    incumbent = page.latest(row)
    if incumbent is not None and tuple(incumbent) != fact:
        raise ValueError(
            "source numeric component concordance disagreement for "
            f"{row!r}: recorded={incumbent!r}, proposed={fact!r}"
        )
    if incumbent is None:
        page.set(row, 0, fact)
    return fact


def _concord_source_precision_boundary(
    scope: str,
    value_id: int,
    boundary: str,
    operand_id: int,
    *,
    precision_limbs: int,
) -> tuple[str, int, int]:
    """Commit one authored Precision boundary independently of graph copies."""

    page = current_identity_book().page(
        "source_precision_boundary_concordance"
    )
    row = (str(scope), int(value_id))
    proposed = (
        str(boundary),
        int(operand_id),
        max(int(precision_limbs or 1), 1),
    )
    incumbent = page.latest(row)
    if incumbent is not None and tuple(incumbent) != proposed:
        raise ValueError(
            "source precision boundary concordance disagreement for "
            f"{row!r}: recorded={incumbent!r}, proposed={proposed!r}"
        )
    if incumbent is None:
        page.set(row, 0, proposed)
    return proposed


def _concorded_static_parameter_bindings(
    definition: Any,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    """Return parameters whose discovered identity is compile-time static.

    Source pursuit has already bound a classmethod receiver to the class that
    Python supplies for that call.  That binding is the identity fact.  The
    reducer used to ignore it because the spelling also appeared in the
    authored parameter list, then separately interpreted the same class as a
    runtime RECORD contract.  Commit the discovered identity once so lexical
    resolution and storage classification consume the same decision.
    """

    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    if not any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in definition.decorator_list
    ):
        return {}
    positional = (*definition.args.posonlyargs, *definition.args.args)
    if not positional:
        return {}
    receiver_name = str(positional[0].arg)
    if receiver_name not in parameter_names:
        return {}
    value = dict(getattr(definition, "_python_bindings", {}) or {}).get(
        receiver_name
    )
    if not inspect.isclass(value):
        return {}
    source_identity = getattr(definition, "_python_source_identity", None)
    scope = (
        tuple(map(str, source_identity))
        if isinstance(source_identity, tuple)
        else (str(getattr(definition, "name", "<function>")),)
    )
    descriptor = (
        "static_class",
        str(getattr(value, "__module__", "")),
        str(getattr(value, "__qualname__", getattr(value, "__name__", ""))),
    )
    page = current_identity_book().page(
        "source_parameter_identity_concordance"
    )
    row = (scope, receiver_name)
    incumbent = page.latest(row)
    if incumbent is not None and incumbent != descriptor:
        raise ValueError(
            "source parameter identity concordance disagreement for "
            f"{scope!r}.{receiver_name}: recorded={incumbent!r}, "
            f"resolved={descriptor!r}"
        )
    if incumbent is None:
        page.set(row, 0, descriptor)
    return {receiver_name: value}


def _known_parameter_memory_contracts(
    definition: Any,
    parameter_names: tuple[str, ...],
    *,
    method_owner: str | None,
    static_parameter_names: frozenset[str] = frozenset(),
) -> tuple[ParameterContract, ...]:
    """Describe parameter storage only when source pursuit proved its kind.

    The records direct raw SSA data; they do not create wrapper objects or
    runtime dispatch.  Once one parameter has a proven aggregate/record kind,
    the complete ordered signature is recorded so argument positions remain
    unambiguous.  Unproved parameters retain the scalar/value defaults.
    """

    bindings = dict(getattr(definition, "_python_bindings", {}) or {})
    storage_by_name: dict[str, ParameterStorage] = {}
    for index, name in enumerate(parameter_names):
        if name in static_parameter_names:
            continue
        value = bindings.get(name)
        if any(value is kind for kind in (list, set, dict, tuple)):
            storage_by_name[name] = ParameterStorage.TABLE
        elif inspect.isclass(value):
            storage_by_name[name] = ParameterStorage.RECORD
        elif index == 0 and method_owner is not None:
            storage_by_name[name] = ParameterStorage.RECORD
    if not storage_by_name:
        return ()
    return tuple(
        ParameterContract(
            name,
            transfer=(
                ParameterTransfer.ALIAS
                if name in storage_by_name
                else ParameterTransfer.VALUE
            ),
            access=(
                ParameterAccess.INOUT
                if name in storage_by_name
                else ParameterAccess.IN
            ),
            storage=storage_by_name.get(name, ParameterStorage.SCALAR),
            scope=(
                ParameterScope.CALLER
                if name in storage_by_name
                else ParameterScope.LOCAL
            ),
        )
        for name in parameter_names
    )


@dataclass(frozen=True)
class _StaticPythonReference:
    """Ephemeral resolved Python object; never emitted as a graph value."""

    value: Any
    path: str


def _reference_has_pursuable_source(value: Any) -> bool:
    """Whether a resolved Python object still has source to pursue into.

    This is the same test the frontend's pursuit
    (``graph_express2._source_ast_definition``) uses to decide the irreducible
    boundary: a function or class whose ``inspect.getsource`` succeeds is
    pursuable; a module, builtin, C-extension, or anything with no readable
    source is irreducible -- terminal. Kept here (not imported) so the reducer
    carries no dependency on the frontend module.
    """

    import inspect

    if inspect.ismethod(value):
        value = value.__func__
    if not (inspect.isfunction(value) or inspect.isclass(value)):
        return False
    try:
        inspect.getsource(value)
    except (OSError, TypeError):
        return False
    return True


# The export table: which graph node types are runnable definitions
# (functions/methods), and how to read each one's own name off it. Python's
# three cases are the built-in defaults; a foreign language's ingestion
# (oop_language_translations.py's install_c_role_schemas, for C's
# pycparser.c_ast.FuncDef, say) extends this the same way role_schemas
# itself is extended -- registering its own node-type name and a callable
# to read that type's own name field, rather than this reducer growing an
# isinstance branch per language. This is the frontend-side half of
# FunctionTable already being "shared function references for ProcessGraph
# and SSA compilation" (function_table.py's own docstring) -- the table was
# already meant to be language-neutral; only *populating* it was Python-AST-
# specific until this registry existed.
_RUNNABLE_DEFINITION_NAME_EXTRACTORS: dict[str, Callable[[Any], str]] = {
    "FunctionDef": lambda node: str(node.name),
    "AsyncFunctionDef": lambda node: str(node.name),
    "Lambda": lambda node: (
        f"<lambda:{getattr(node, 'lineno', 0)}:"
        f"{getattr(node, 'col_offset', 0)}>"
    ),
}


def register_runnable_definition_type(
    type_name: str, name_extractor: Callable[[Any], str],
) -> None:
    """Register a foreign language's function/method-definition node type.

    ``type_name`` is ``type(node).__name__`` for that language's own
    function-definition node class (``"FuncDef"`` for ``pycparser``, say).
    ``name_extractor`` reads that node's own name (``pycparser.c_ast.FuncDef``
    keeps it at ``node.decl.name``, not ``node.name``, so this cannot be one
    generic ``.name`` access across languages). Idempotent: re-registering
    the same type name overwrites, it does not duplicate or error.
    """

    _RUNNABLE_DEFINITION_NAME_EXTRACTORS[str(type_name)] = name_extractor


def is_runnable_definition(node: Any) -> bool:
    """Is ``node`` a registered function/method-definition node, in any
    registered language -- the export-table membership test."""

    return type(node).__name__ in _RUNNABLE_DEFINITION_NAME_EXTRACTORS


def runnable_definition_name(node: Any) -> str:
    """The name of a registered function/method-definition node."""

    return _RUNNABLE_DEFINITION_NAME_EXTRACTORS[type(node).__name__](node)


# Same export-table pattern, for the two other node shapes call/return
# ownership walking needs to recognize: a call, and a return-shaped node.
# "Return" is deliberately one shared key for every registered language
# rather than a per-language key -- pycparser's own Return node is *also*
# literally named "Return" (not a naming collision to route around, an
# actual shared vocabulary word both languages use for the same construct).
#
# The returned value itself is read from the graph's own dependency
# structure (a node's "value"/"expr" parent-edge role -- role_schemas'
# Return: {"up": {"value": 1}} for Python, {"up": {"expr": 1}} for C), never
# from the raw source node's own attributes. This is deliberate, not
# incidental: everywhere else this session resolved a value this way
# (ShellMemoryReference.base_node_id, _resolve_reference_node in
# glsl_deployment_strategy.py) it went through the graph's own parents/role
# edges, not getattr on expr_obj -- the graph is the abstract, language-
# neutral representation; the raw node is not, and reaching back into it
# here would just reintroduce a second, Python-shaped assumption
# (node.value vs node.expr) into code meant to be language-neutral.
_CALL_SHAPED_TYPE_NAMES: set[str] = {"Call"}
_RETURN_VALUE_ROLES: set[str] = {"value", "expr"}


def register_call_shaped_type(type_name: str) -> None:
    """Register a foreign language's call-expression node type name."""

    _CALL_SHAPED_TYPE_NAMES.add(str(type_name))


def register_return_value_role(role: str) -> None:
    """Register a foreign language's own parent-edge role name for "this is
    the value a return-shaped node returns" (role_schemas' own "up" key for
    that language's Return-equivalent node)."""

    _RETURN_VALUE_ROLES.add(str(role))


def source_child_nodes(node: Any) -> tuple[Any, ...]:
    """Every direct child of a source node, in any registered language.

    The one traversal primitive a language must supply. Python spells it
    ``ast.iter_child_nodes``; pycparser spells it ``node.children()``
    (yielding ``(role_name, child)`` pairs). Neither understands the other:
    ``ast.NodeVisitor.generic_visit`` requires ``node._fields``, which
    pycparser nodes do not have, which is why walking a C body with an
    ``ast`` visitor raises ``AttributeError: 'Decl' object has no attribute
    '_fields'`` rather than simply finding nothing.

    Dispatch is on the protocol the node actually implements, not on a
    registered type name, because that is what genuinely varies -- and it
    means a third frontend whose nodes are ``ast``-shaped or
    ``children()``-shaped needs no registration here at all.
    """

    if isinstance(node, ast.AST):
        return tuple(ast.iter_child_nodes(node))
    children = getattr(node, "children", None)
    if callable(children):
        return tuple(child for _role, child in children())
    return ()


def source_body_statements(definition: Any) -> tuple[Any, ...]:
    """The ordered statements making up a function definition's body.

    Python keeps them directly on ``FunctionDef.body``; C wraps them in a
    ``Compound`` whose ``block_items`` holds the list (and which may be
    ``None`` for an empty body).
    """

    body = getattr(definition, "body", None)
    if body is None:
        return ()
    if isinstance(body, list):
        return tuple(body)
    block_items = getattr(body, "block_items", None)
    if block_items is not None:
        return tuple(block_items)
    # A lambda-style single-expression body is itself the only statement.
    return (body,)


def source_walk(node: Any) -> tuple[Any, ...]:
    """``ast.walk`` for any registered language, via ``source_child_nodes``.

    Callers filter the result with ``isinstance`` against the constructs
    they care about. That stays correct for a foreign language rather than
    merely not crashing: C genuinely has no ``ast.ExceptHandler`` and no
    Python-style ``for``, so a scan for those legitimately finds nothing,
    which is the right answer and not a silent gap.
    """

    collected: list[Any] = []
    stack = [node]
    while stack:
        current = stack.pop()
        collected.append(current)
        stack.extend(source_child_nodes(current))
    return tuple(collected)


def function_parameter_names(definition: Any) -> tuple[str, ...]:
    """A function definition's declared parameter names, in order.

    Genuinely per-language grammar rather than shared vocabulary, so this
    is one of the few places a language really must be taught its own
    shape: Python hangs them off ``FunctionDef.args`` as ``arg`` nodes
    across three lists (positional-only, ordinary, keyword-only), while C
    reaches ``FuncDef -> decl -> type -> args.params`` to a list of
    ``Decl`` nodes whose ``.name`` is a plain string.
    """

    arguments = getattr(definition, "args", None)
    if arguments is not None and hasattr(arguments, "posonlyargs"):
        return tuple(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *((arguments.vararg,) if arguments.vararg is not None else ()),
                *((arguments.kwarg,) if arguments.kwarg is not None else ()),
            )
        )
    declaration = getattr(definition, "decl", None)
    if declaration is not None:
        parameter_list = getattr(getattr(declaration, "type", None), "args", None)
        return tuple(
            str(parameter.name)
            for parameter in (getattr(parameter_list, "params", None) or ())
            if getattr(parameter, "name", None) is not None
        )
    return ()


def _record_owned_calls_and_returns(
    graph: Any,
    owner_reference: Any,
    function_node_id: int,
    call_owners: dict[int, Any],
    function_return_values: dict[int, list[int]],
) -> None:
    """Walk ``graph.G`` (not the raw source tree) to find every call/return
    belonging to the function at ``function_node_id``, stopping at any
    nested function/method-definition boundary -- the graph-based,
    language-neutral replacement for a source-tree ``ast.NodeVisitor`` walk,
    which cannot walk a foreign language's own node types at all (a
    ``pycparser`` node has no ``_fields`` attribute Python's
    ``ast.NodeVisitor.generic_visit`` requires). Bounded, not
    ``nx.ancestors``' full transitive closure: a nested definition's own
    calls/returns belong to *it*, not this enclosing function, the same
    rule the original visitor's ``visit_FunctionDef`` returning ``None``
    (not recursing) enforced.
    """

    visited: set[int] = set()
    stack = list(graph.G.predecessors(function_node_id))
    while stack:
        node_id = stack.pop()
        if node_id in visited or node_id not in graph.G:
            continue
        visited.add(node_id)
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        if node_id != function_node_id and is_runnable_definition(expression):
            # A nested definition owns its own calls/returns; do not
            # descend into it from here.
            continue
        node_type = str(data.get("type"))
        if node_type in _CALL_SHAPED_TYPE_NAMES:
            call_owners[node_id] = owner_reference
        if node_type == "Return":
            # The returned value's own node id, read off the Return node's
            # graph-native parent edges (role "value" for Python, "expr" for
            # C) -- never off the raw source node's attributes. Already a
            # graph node id (an "up" edge's producer), not a raw object,
            # so it needs no id()/wrapping before use.
            value_node_id = next(
                (
                    parent_id
                    for parent_id, role in (data.get("parents") or ())
                    if str(role) in _RETURN_VALUE_ROLES
                ),
                None,
            )
            if value_node_id is not None:
                function_return_values.setdefault(
                    function_node_id, [],
                ).append(value_node_id)
        stack.extend(graph.G.predecessors(node_id))


# Source-language node type name -> the canonical SSA vocabulary spelling it
# universalizes to. The point of routing through the registry rather than
# hardcoding strings here is that a second language's own spelling for the
# same construct lands on the *same* canonical type: C's ``FuncCall`` and
# Python's ``Call`` both become "Call", so downstream passes see one
# vocabulary instead of one per frontend. Extend by registering the foreign
# spelling in ssa_registry.py (c_ssa_equivalents, say) and naming its node
# type here -- not by adding a branch downstream.
_AST_PROCESS_GRAPH_ALIASES = {
    "Name": ast_ssa_name_map["name"].value,
    "Assign": ast_ssa_name_map["assign"].value,
    "Call": ast_ssa_name_map["call"].value,
    # C (pycparser c_ast), joining the same canonical spellings above.
    # StructRef is C's field read (``self->value``), the same reading
    # Python's ``attribute:load`` has -- both are Load.
    "ID": c_ssa_name_map["id"].value,
    "Assignment": c_ssa_name_map["assignment"].value,
    "FuncCall": c_ssa_name_map["funccall"].value,
    "StructRef": c_ssa_name_map["structref"].value,
}

# C node types that exist only to spell out a *type*, carrying no runtime
# value of their own. Python's grammar has no equivalent (it is untyped at
# the syntax level), which is why this set has no Python counterpart --
# these are stripped for the same reason ast.Nonlocal/ast.Global are:
# compile-time syntax, not operations. ``Decl`` is deliberately NOT here --
# it carries both the declared name and its initializer, so it is a real
# binding event, not type machinery.
_C_COMPILE_TIME_SYNTAX = frozenset({
    "TypeDecl",
    "FuncDecl",
    "PtrDecl",
    "ArrayDecl",
    "IdentifierType",
    "ParamList",
    "Typedef",
    "Struct",
    "Union",
    "Enum",
})

_BITOPS_TO_EXECUTABLE = {
    # Bitwise operators resolve to their lowercase canonical executable names,
    # the ones every numeric table registers (``fused_ir.ELEMENTWISE_BINARY``,
    # ``canonical_elementwise_op``, the WASM/C/GLSL backends). Left as the bare
    # Handler names (``And``/``Or``/``Not``/``Shl``/``Shr``) they are recognized
    # only by the SymPy-spelling path in ``abstract_tensor_funcs`` and silently
    # drop out of any consumer keyed on the canonical set -- e.g. the structural
    # region builder ``dispatch_region_to_fused_program``. ``Xor`` was already
    # mapped; the rest were the asymmetry that made ``^`` reachable but ``&``,
    # ``|``, ``~``, ``<<``, ``>>`` not.
    "And": "bitand",
    "Or": "bitor",
    "Xor": "bitxor",
    "Not": "invert",
    "Shl": "shl",
    "Shr": "shr",
    "Neg": "neg",
    "Eq": "equal",
    "Ne": "not_equal",
    "Lt": "less",
    "Le": "less_equal",
    "Gt": "greater",
    "Ge": "greater_equal",
    "MatMul": "matmul",
    "LAnd": "logical_and",
    "LOr": "logical_or",
    "LNot": "logical_not",
}


#: Element types a precision operation can be emitted at, widest first. The
#: ORDER is the dispatch rule: where two operands disagree, the wider type
#: wins, because narrowing is a loss and a loss should be asked for rather
#: than inherited from whichever operand happened to be on the left.
PRECISION_ELEMENT_TYPES: tuple[tuple[str, str], ...] = (
    ("float64", "f64"),
    ("float32", "f32"),
    ("float16", "f16"),
)

#: Spellings that mean the same element type.
PRECISION_TYPE_ALIASES = {"double": "float64", "float": "float32",
                          "f64": "float64", "f32": "float32", "f16": "float16"}

#: Operations closed over the limb representation. A sum, difference,
#: product, quotient, square root, exponential, logarithm or negation of limbed
#: value, so each has a wider counterpart meaning the same thing. Nothing else
#: belongs: there
#: is no wider form of a reduction or a reshape that would be right, so those
#: keep their ordinary name and a destination refuses the operand rather than
#: reading its limbs as channels.
#: Spelled exactly as ``ast_ssa_name_map`` resolves them, which is not
#: uniform -- four are capitalised and ``neg`` is not, and there is no
#: ``truediv`` at this layer, only ``Div``. Keying on a tidied-up spelling
#: silently misses and falls through to the ordinary operation.
PRECISION_CLOSED_OPERATIONS = (
    "Add", "Sub", "Mul", "Div", "Sqrt", "Exp", "Log", "neg",
)

#: The greatest width represented in the legacy inspectable qualified-name
#: catalogue. Repository lowering consumes the singular operation plus its
#: explicit ``precision_limbs`` attribute and is not bounded by this table.
PRECISION_LIMB_LIMIT = 8


def _build_precision_operator_names() -> dict:
    """Every precision operator name, generated once at import.

    Built here rather than formatted at each call so the set of names a
    destination may receive is a fixed, inspectable table -- a backend can be
    checked against it, and a name that is not in it is a name nothing agreed
    to emit.
    """

    names = {}
    for operation in PRECISION_CLOSED_OPERATIONS:
        for element, tag in PRECISION_ELEMENT_TYPES:
            for limbs in range(2, PRECISION_LIMB_LIMIT + 1):
                names[(operation, element, limbs)] = (
                    f"{operation.lower()}_{tag}_p{limbs}_r{limbs}"
                )
    return names


#: (operation, element type, limbs) -> the name a destination implements.
#: NOT what ingestion plants. A specific name fixes an operating width and a
#: return width, and neither is known at the point an operand is read: fusion
#: may run a batch at a width no single operation had, and whether a result
#: stays limbed is a property of its CONSUMER, which does not exist yet. These
#: are computed from the surviving operations once planning has run.
PRECISION_OPERATOR_NAMES = _build_precision_operator_names()

#: What ingestion plants: one name per operation, carrying no width.
#: Enough for an identity to know which operation this is, and -- because
#: nothing downstream recognises it either -- enough to keep the node out of
#: the fusion planner's induced subgraph, which is what carries it intact.
PRECISION_SINGULAR_NAMES = {
    operation: f"precision_{operation.lower()}"
    for operation in PRECISION_CLOSED_OPERATIONS
}
PRECISION_PACK_NAME = "precision_pack"


def _widest_element(*declared: Optional[str]) -> Optional[str]:
    """The widest element type among the operands, or ``None`` if none say.

    Highest precision decides: an operation between a wide operand and a
    narrow one is emitted at the wide type, so meeting a narrower value never
    silently costs digits the caller already paid for.
    """

    order = {element: rank for rank, (element, _tag)
             in enumerate(PRECISION_ELEMENT_TYPES)}
    best, best_rank = None, len(order)
    for value in declared:
        if not value:
            continue
        element = PRECISION_TYPE_ALIASES.get(str(value), str(value))
        rank = order.get(element)
        if rank is not None and rank < best_rank:
            best, best_rank = element, rank
    return best


def _parameter_declaration(graph: Any, name: str) -> Optional[str]:
    """A parameter's declared type, when every function declaring it agrees.

    ``annotate_types`` captures annotated ASSIGNMENTS only -- its
    ``_TypeAnnotator`` has no visitor for ``ast.arg`` -- so a declaration
    written on a parameter never reaches ``type_annotations``. It is not
    lost, though: ``build_from_ast`` already unparses every parameter
    annotation into ``function_parameter_annotations``, keyed by function
    identity. This reads that rather than teaching the annotator a second
    way to learn the same fact.

    The walk does not carry which function it is inside, so a name is only
    honoured when every function that declares it declares it identically.
    Where two disagree the answer is None and the operand reads as ordinary
    -- a miss, never a wrong width, which is the right way round for
    something whose failure is silent.
    """

    try:
        by_function = graph.G.graph.get("function_parameter_annotations") or {}
    except AttributeError:
        return None
    declared = {
        str((parameters or {}).get(name))
        for parameters in by_function.values()
        if (parameters or {}).get(name)
    }
    return declared.pop() if len(declared) == 1 else None


def _operand_numeric_descriptor(
    graph: Any, node: ast.AST,
) -> NumericFeatureDescriptor | None:
    """An operand's complete declared numeric feature contract.

    The declaration is what the source says the value IS -- captured by
    ``annotate_types`` at ingestion and kept as ``type_annotations``. Parameter
    annotations use the exact spelling retained by ``build_from_ast``. The
    result describes every wrapper axis before an operator is chosen.

    An INDEXED operand is read through to the name it indexes: ``a[i]`` where
    ``a`` is declared carries ``a``'s width. Limbs are a channel in the last
    dimension, so subscripting an array of precision values selects one of
    them and does not consume that axis -- the element is as limbed as the
    array. Without this the declaration survives only on operands spelled as
    bare names, which excludes every kernel that walks an array, and the loss
    is silent: the operation simply lowers as ordinary arithmetic.
    """

    while isinstance(node, ast.Subscript):
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    try:
        annotations = graph.G.graph.get("type_annotations") or {}
    except AttributeError:
        return None
    declared = annotations.get(node.id) or _parameter_declaration(
        graph, node.id
    )
    if not declared:
        return None
    descriptor = numeric_feature_descriptor(str(declared))
    if descriptor is not None:
        graph.G.graph.setdefault(
            "numeric_annotation_descriptors", {}
        )[str(declared)] = descriptor.receipt()
    return descriptor


def _operand_precision(graph: Any, node: ast.AST) -> tuple[int, Optional[str]]:
    """Compatibility view consumed by the established Precision pipeline."""

    descriptor = _operand_numeric_descriptor(graph, node)
    if descriptor is None or descriptor.features != frozenset({"precision"}):
        return 1, None
    return descriptor.limbs, descriptor.element_type


def _qualified_handler(prefix: str, operator: ast.AST, *, limbs: int = 1,
                       dtype: Optional[str] = None) -> str:
    """The operator's identity, singular-precision when an operand is limbed.

    Most limbs and highest precision decide what the operation IS; the caller
    passes what the operands declared and the widest of each governs. But the
    name planted here stays unspecific about width, because the width it is
    finally emitted at is not yet knowable -- fusion may batch it at a width
    no operand had, and whether it returns a limbed result depends on a
    consumer that does not exist at this point in the walk. The measured
    limbs and element type ride along as attributes for the pass that does
    know, once planning has settled which operations survived unfused.
    """

    spelling = f"{prefix}:{type(operator).__name__.lower()}"
    handler = ast_ssa_name_map.get(spelling)
    if handler is None:
        raise KeyError(f"no existing operator alias for {spelling!r}")
    canonical = _BITOPS_TO_EXECUTABLE.get(handler.value, handler.value)
    width = max(int(limbs or 1), 1)
    if width <= 1:
        return canonical
    return PRECISION_SINGULAR_NAMES.get(str(canonical), canonical)


def _c_qualified_handler(prefix: str, operator: str) -> str:
    """``_qualified_handler`` for C, whose operators are strings not classes.

    Python spells an operator as a child *node class* (``ast.Add``, hence
    ``binop:add``); pycparser spells it as a plain string attribute on the
    parent (``BinaryOp.op == '+'``, hence ``binaryop:+``). Two surface
    spellings, one canonical Handler -- which is the whole point of the
    registry, and why this returns the identical vocabulary
    ``_qualified_handler`` does rather than a parallel C-flavored one.
    """

    spelling = f"{prefix}:{str(operator).lower()}"
    handler = c_ssa_name_map.get(spelling)
    if handler is None:
        raise KeyError(f"no existing C operator alias for {spelling!r}")
    return _BITOPS_TO_EXECUTABLE.get(handler.value, handler.value)


def _c_constant_value(expression: Any) -> Any:
    """A ``pycparser`` Constant's real Python value.

    pycparser keeps every literal as the *source text* plus a type name
    (``Constant(type='int', value='10')`` -- the value is the string "10",
    not the integer 10). Downstream compiler stages treat ``data["constant"]``
    as a real value, so the conversion has to happen here, at the one place
    C literals are canonicalized, rather than being re-derived (or silently
    left as a string) at each consumer.
    """

    raw = str(getattr(expression, "value", ""))
    kind = str(getattr(expression, "type", ""))
    if "char" in kind:
        return raw.strip("'")
    if "string" in kind:
        return raw.strip('"')
    if "float" in kind or "double" in kind:
        return float(raw.rstrip("fFlL"))
    if "int" in kind:
        # C integer literals carry base prefixes and width/sign suffixes that
        # Python's int() will not accept directly (0x1FUL, 10u, 07).
        text = raw.rstrip("uUlL")
        return int(text, 0) if text else 0
    return raw


# The type given to a node standing in for something ingestion could not
# translate. Mangled data is mangled data: it needs a real, named, *non*
# operation rather than either a silent hole or a hard stop.
#
# This type name MUST NEVER be added to any operator table -- not
# ``operator_defs.operator_signatures``' executable entries, not
# ``graph_deep_compiler``'s ``op_table``, not ``ssa_webgpu_backend``'s
# ``_BINARY``/``_UNARY``, not ``fused_ir``'s ``ELEMENTWISE_*``. Its safety is
# structural, not a matter of anyone remembering to check: because no table
# maps it, every backend that meets it already reports it through that
# backend's own existing "no instruction for this" path. Nothing can quietly
# treat it as zero, which is the one line this repository does not cross --
# never silently produce a plausible wrong number.
#
# It is deliberately *one* generic type carrying free-text provenance, not a
# taxonomy of failure kinds. A closed enum of defect categories would need a
# central owner to admit every new construct a contributor or agent adds --
# exactly the bottleneck ``role_schemas``,
# ``_RUNNABLE_DEFINITION_NAME_EXTRACTORS`` and the ssa_registry spelling
# tables exist to remove. ``operator_signatures`` falls back to 'Default'
# for an unrecognized type, so a graph carrying these still finalizes and
# stays walkable; the translation simply reports how far it got.
UNTRANSLATED_NODE_TYPE = "Untranslated"


# How completely the affected construct survived translation. This is an
# ordered *scale*, not a taxonomy -- which is why it is a closed set where
# ``reason`` is deliberately free text. A scale has rungs that mean
# something relative to each other and can be compared and sorted; a
# taxonomy of failure kinds would need a central owner to admit every new
# construct a contributor or agent introduces, which is the bottleneck this
# module keeps removing. Add a rung only if it is genuinely between or
# beyond these, never to describe a new *kind* of failure.
#
# Nothing here is binary. A translation is not "worked" or "failed": it
# reached some depth and stopped, and how far it reached is the useful
# fact. ``complete`` on the backend artifacts is the cheapest summary of
# this scale, not a replacement for it.
TRANSLATION_ABSENT = "absent"      # nothing of the construct translated
TRANSLATION_PARTIAL = "partial"    # some constituents translated, some not
TRANSLATION_DEGRADED = "degraded"  # fully translated, reduced fidelity
TRANSLATION_UNKNOWN = "unknown"    # not determinable at this pass

TRANSLATION_GRADES = (
    TRANSLATION_ABSENT,
    TRANSLATION_PARTIAL,
    TRANSLATION_DEGRADED,
    TRANSLATION_UNKNOWN,
)


@dataclass(frozen=True)
class GraphTranslationShortfall:
    """One construct graph ingestion could not fully translate.

    Same contract the backend shortfall records already use
    (``WasmShortfall``, ``WGSLShortfall``): structured fields for the parts
    worth querying across many programs, and one free-text ``reason`` so
    reporting something new never requires extending a type. Recorded on
    ``graph.G.graph["translation_shortfalls"]``, alongside the existing
    ``state_machine_control_shortfalls``.

    ``grade`` carries how far the translation got, and ``observed``/
    ``expected`` the countable evidence behind that grade (operands
    resolved out of operands required, say). A reader deciding whether a
    specimen is solid enough to reason about needs the degree, not just the
    fact -- and a caller aggregating across many programs needs the counts,
    not a prose summary of them.
    """

    pass_name: str
    node_id: int
    operation: str
    role: str
    source_span: str
    reason: str
    grade: str = TRANSLATION_UNKNOWN
    observed: int = 0
    expected: int = 0

    @property
    def coverage(self) -> float | None:
        """Fraction of the construct that translated, when countable."""

        if self.expected <= 0:
            return None
        return self.observed / self.expected

    def format(self) -> str:
        where = f" at {self.source_span}" if self.source_span else ""
        role = f" ({self.role})" if self.role else ""
        ratio = (
            f" [{self.observed}/{self.expected}]" if self.expected else ""
        )
        return (
            f"{self.pass_name}: {self.operation}{role}{where} "
            f"<{self.grade}{ratio}>: {self.reason}"
        )


# Cap on a folded sequence, so `[0] * 10**9` is refused rather than
# materialized. A refused fold is not a failure: it falls through to the
# ordinary path and reports itself there like anything else.
_MAX_FOLDED_SEQUENCE_ELEMENTS = 1 << 20


def _static_sequence_literal(expression: Any) -> Any:
    """Fold literal sequence replication (``[0] * 256``) to its value.

    Returns the folded list, or ``None`` when this is not that construct.

    ``[0] * 256`` is **Python list replication, not arithmetic** -- but the
    AST spells it ``BinOp(op=Mult)``, so canonicalization would otherwise
    rewrite it into a ``Mul`` dataflow operation that was never arithmetic.
    Its operands are not in the graph either (a literal list inside a
    comprehension is never descended into), so the ``Mul`` ends up with two
    absent operands.

    This is compile-time table allocation -- the shape a decoder's lookup
    tables are declared in -- and its value is knowable now, so it becomes
    constant data. Only the both-operands-literal case folds; anything
    depending on a runtime value is left alone.
    """

    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.Mult
    ):
        return None
    for sequence_node, count_node in (
        (expression.left, expression.right),
        (expression.right, expression.left),
    ):
        if not isinstance(sequence_node, (ast.List, ast.Tuple)):
            continue
        try:
            sequence = ast.literal_eval(sequence_node)
            count = ast.literal_eval(count_node)
        except (ValueError, TypeError, SyntaxError, MemoryError):
            return None
        if isinstance(count, bool) or not isinstance(count, int):
            return None
        if count < 0:
            return None
        if len(sequence) * count > _MAX_FOLDED_SEQUENCE_ELEMENTS:
            return None
        return list(sequence) * count
    return None


def _source_span(expression: Any) -> str:
    line = getattr(expression, "lineno", None)
    if line is None:
        return ""
    column = getattr(expression, "col_offset", None)
    return f"line {line}" + (f":{column}" if column is not None else "")


def record_translation_shortfall(
    graph: Any,
    *,
    pass_name: str,
    node_id: int,
    operation: str,
    reason: str,
    role: str = "",
    source_span: str = "",
    grade: str = TRANSLATION_UNKNOWN,
    observed: int = 0,
    expected: int = 0,
) -> None:
    """Accumulate one graph-level shortfall instead of raising.

    Ingestion reports the way the backends already do -- a partial
    translation is a real, inspectable result, not a failure. Raising here
    would surface exactly one defect per run of a multi-minute build, when
    the same run could have named every one of them.
    """

    existing = graph.G.graph.get("translation_shortfalls") or ()
    graph.G.graph["translation_shortfalls"] = (
        *existing,
        GraphTranslationShortfall(
            pass_name=str(pass_name),
            node_id=int(node_id),
            operation=str(operation),
            role=str(role),
            source_span=str(source_span),
            reason=str(reason),
            grade=str(grade),
            observed=int(observed),
            expected=int(expected),
        ),
    )


def _replace_inputs(
    graph: Any,
    node_id: int,
    inputs: tuple[tuple[int, str], ...],
) -> None:
    """Replace one wrapper's incoming topology with executable operands.

    An operand that is not in the graph is replaced by an explicit
    ``UNTRANSLATED_NODE_TYPE`` node rather than passed to ``add_edge``.
    NetworkX *creates* an absent endpoint rather than raising, which turned
    a missing operand into a node with no ``type``, no ``expr_obj`` and no
    ``label`` -- invisible here and fatal thousands of nodes later, as a
    bare ``KeyError('type')`` naming nothing. The placeholder keeps the
    consuming operation's arity intact (an operation silently short an
    operand is a *wrong* program, not a smaller one) while making it
    honestly inexecutable.
    """

    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.remove_edge(predecessor, node_id)
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    consumer = graph.G.nodes[node_id] if node_id in graph.G else {}
    expected = len(inputs)
    present = sum(1 for operand, _role in inputs if operand in graph.G)
    # The grade describes the *construct*, not the individual operand: an
    # operation with one of two operands resolved is partially translated,
    # even though each absent operand is individually absent.
    grade = TRANSLATION_ABSENT if present == 0 else TRANSLATION_PARTIAL
    resolved: list[tuple[int, str]] = []
    for predecessor, role in inputs:
        if predecessor not in graph.G:
            predecessor = _untranslated_operand(
                graph,
                consumer_id=node_id,
                consumer=consumer,
                role=role,
                absent_id=predecessor,
                grade=grade,
                observed=present,
                expected=expected,
            )
        resolved.append((predecessor, role))
    graph.G.nodes[node_id]["parents"] = list(resolved)
    for predecessor, role in resolved:
        graph.G.add_edge(predecessor, node_id, role=role)
        children = graph.G.nodes[predecessor].setdefault("children", [])
        if node_id not in {child_id for child_id, _role in children}:
            children.append((node_id, role))


def _untranslated_operand(
    graph: Any,
    *,
    consumer_id: int,
    consumer: Any,
    role: str,
    absent_id: int,
    grade: str = TRANSLATION_UNKNOWN,
    observed: int = 0,
    expected: int = 0,
) -> int:
    """Materialize one explicit stand-in for an operand that never arrived."""

    expression = consumer.get("expr_obj") if hasattr(consumer, "get") else None
    operation = str(consumer.get("type") or type(expression).__name__)
    span = _source_span(expression)
    label = f"untranslated[{operation}.{role}]"
    placeholder_id = id(label)
    while placeholder_id in graph.G:
        # ``id`` of a fresh string is unique among live objects, but this
        # label is not retained, so a later identical label could reuse the
        # address. Step off any collision rather than silently aliasing two
        # unrelated stand-ins onto one node.
        placeholder_id += 1
    graph.G.add_node(
        placeholder_id,
        label=label,
        type=UNTRANSLATED_NODE_TYPE,
        op=UNTRANSLATED_NODE_TYPE.lower(),
        expr_obj=None,
        extra_args={},
        domain_node=None,
        store_id=None,
        parents=[],
        children=[],
        attributes={
            "untranslated": True,
            "consumer_node": int(consumer_id),
            "consumer_operation": operation,
            "operand_role": str(role),
            "absent_node_id": int(absent_id),
            "source_span": span,
            "translation_grade": str(grade),
            "translated_operands": int(observed),
            "expected_operands": int(expected),
        },
    )
    record_translation_shortfall(
        graph,
        pass_name="process-graph-operands",
        node_id=consumer_id,
        operation=operation,
        role=role,
        source_span=span,
        grade=grade,
        observed=observed,
        expected=expected,
        reason=(
            "operand was never ingested as a graph node; a non-operation "
            "stands in its place so the translation reports rather than "
            "computes"
        ),
    )
    return placeholder_id


def _remove_node(graph: Any, node_id: int) -> None:
    """Remove one reduced-away node and its cached adjacency metadata."""

    if node_id not in graph.G:
        return
    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    for successor in tuple(graph.G.successors(node_id)):
        graph.G.nodes[successor]["parents"] = [
            (parent_id, role)
            for parent_id, role in graph.G.nodes[successor].get(
                "parents", ()
            )
            if parent_id != node_id
        ]
    graph.roots = [root for root in graph.roots if root != node_id]
    graph.G.remove_node(node_id)


def _concord_lexical_reads(
    graph: Any, scope: str, occurrence_id: int, binding: str,
) -> None:
    """Commit, per consumer edge, the binding one ``Name`` occurrence read.

    Page ``lexical_read_binding`` row ``(scope, consumer id, role,
    ordinal)`` holds the authored binding that operand read; ``ordinal``
    counts earlier parents with the same role (a tuple's ``elts``).  The value the edge points at
    may later be rebound (a loop header, a loop result port); the binding
    it read never changes, which is what lets those rebindings choose per
    read instead of per value.
    """

    if occurrence_id not in graph.G:
        return
    page = current_identity_book().page("lexical_read_binding")
    for consumer in tuple(graph.G.successors(occurrence_id)):
        for role, ordinal, parent_id in _operand_positions(
            graph.G.nodes[consumer].get("parents", ())
        ):
            if parent_id == occurrence_id:
                page.concord((scope, consumer, role, ordinal), str(binding))


def _operand_positions(parents: Any):
    """Each ``(role, ordinal, parent)``: the operand's stable position."""

    seen: dict[Any, int] = {}
    for parent_id, role in parents:
        ordinal = seen.get(role, 0)
        seen[role] = ordinal + 1
        yield role, ordinal, parent_id


def lexical_read_binding(
    graph: Any, consumer: int, role: Any, ordinal: int = 0,
) -> str | None:
    """The authored binding one operand of ``consumer`` read, if any."""

    scope = (getattr(graph, "graph", None) or {}).get("lexical_read_scope")
    if scope is None:
        return None
    consumer = consumer if isinstance(consumer, str) else int(consumer)
    return current_identity_book().page("lexical_read_binding").latest(
        (tuple(scope), consumer, role, int(ordinal))
    )


def _redirect_value(
    graph: Any,
    old_id: int,
    producer_id: int,
) -> None:
    """Fan every use of one lexical occurrence out from its value producer."""

    if old_id == producer_id or old_id not in graph.G:
        return
    for successor in tuple(graph.G.successors(old_id)):
        successor_data = graph.G.nodes[successor]
        replacement = []
        for parent_id, role in successor_data.get("parents", ()):
            replacement.append(
                (producer_id if parent_id == old_id else parent_id, role)
            )
        successor_data["parents"] = replacement
        graph.G.add_edge(producer_id, successor, role=graph.G.edges[
            old_id,
            successor,
        ].get("role"))
        children = graph.G.nodes[producer_id].setdefault("children", [])
        for _parent_id, role in replacement:
            if _parent_id == producer_id and (
                successor,
                role,
            ) not in {
                (child_id, child_role)
                for child_id, child_role in children
            }:
                children.append((successor, role))
    graph.roots = [
        producer_id if root == old_id else root for root in graph.roots
    ]
    _remove_node(graph, old_id)


def _normalize_lexical_values(
    function_graph: Any,
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    static_bindings: dict[str, Any],
    function_table: FunctionTable,
    lexical_function_bindings: dict[str, Any] | None = None,
    closure_aggregate_kinds: dict[str, str] | None = None,
    method_owner: str | None = None,
    class_field_aggregate_kinds: Mapping[tuple[str, str], str] | None = None,
    class_field_mapping_contracts: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
    class_field_sequence_dtypes: Mapping[
        tuple[str, str], tuple[str, ...]
    ] | None = None,
    class_field_classes: Mapping[tuple[str, str], str] | None = None,
    static_parameter_bindings: Mapping[str, Any] | None = None,
) -> None:
    """Resolve unique lexical occurrences into a monotonic value DAG.

    AST ingestion intentionally gives every ``Name`` occurrence its own node.
    At topological reduction, loads are consolidated against the definition
    visible at that program point.  Consumers then fan out directly from the
    defining value; source-language ``Load`` and ``Store`` wrappers disappear.
    """

    graph = function_graph
    environment: dict[str, int] = {}
    static_environment: dict[str, _StaticPythonReference] = {}
    deleted_names: set[str] = set()
    identity_bindings: dict[str, list[int]] = {}
    ingestion_definitions: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    loop_target_bindings_by_ast: dict[int, int] = {}
    static_reference_nodes: dict[tuple[int, str], int] = {}
    first_class_function_nodes: dict[int, int] = {}
    materialized_attribute_nodes: dict[int, int] = {}
    static_constant_nodes: dict[str, int] = {}
    static_attribute_values: dict[tuple[int, str], int] = {}
    # The most recent node that mutated ``(receiver_node_id, attr)`` --
    # a whole-attribute ``SetAttr`` or an element-wise ``obj.field[i, j] =``
    # ``IndexedStore``.  A later bare read of the same field consults this so
    # the read is ordered after the write instead of depending only on the
    # unchanged receiver node (see the read side in ``resolve_expression``).
    attribute_effect_nodes: dict[tuple[int, str], int] = {}
    # Current field value, separate from the ordered write event. Explicit
    # GetAttr nodes remain in the graph for record-ABI projection; this ledger
    # only supplies exact incoming values when source control joins a write.
    attribute_value_nodes: dict[tuple[int, str], int] = {}
    parameter_names = set(function_parameter_names(statement))
    static_parameter_bindings = dict(static_parameter_bindings or {})
    value_class_scope = _source_numeric_scope(
        graph, getattr(statement, "name", "<function>")
    )
    # One reduction is one scope of read facts, minted by the book: rows at
    # ingestion ids (object addresses, unique only within this reduction)
    # and, after canonical renumbering, at canonical ids.  The graph names
    # its reduction scope, so every copy of it (a specialization) reads the
    # facts this reduction committed.
    read_scope = current_identity_book().mint_scope(
        f"lexical_reads:{value_class_scope}"
    )
    ingestion_read_scope = (read_scope, "ingestion")
    #: A bare ``return name`` has no consumer edge: its value becomes a root.
    #: The bindings each returned root value was returned under.
    return_root_bindings: dict[int, set[str]] = {}
    # A parameter annotated with an ingested class name gives a
    # receiver a real, known class identity at ingestion -- enough to
    # resolve ``receiver.attr`` through the class's own navigation table
    # (below) instead of inventing a name for it.  Only a bare ``Name``
    # annotation naming a class this source itself defines counts; anything
    # else leaves the receiver's class unknown here, and attribute access on
    # it is not slot-resolvable. Numeric wrapper annotations are allowed to be
    # generic (``ComplexRationalPrecision[2]``): source admission has already
    # made their class identity part of this same navigation table, and the
    # normalized descriptor supplies the bare authoritative identity.
    #
    # ``build_class_navigation_table`` needs only ``map_ir`` (ingestion-time,
    # AST-derived) and ``function_table`` for method references; both are
    # already populated on ``graph`` before reduction runs -- but this
    # function is called once per *function* being reduced, and
    # ``build_class_navigation_table`` is a real, previously whole-
    # compilation-scoped call (``aot_compile.py``, after all reduction
    # finishes).  Memoized on the graph itself so it still runs exactly
    # once per graph, not once per method of a retained class with many
    # methods -- calling it repeatedly, interleaved with other functions'
    # own active reduction, is not a cost concern here so much as an
    # unreviewed-reentrancy risk against shared reduction state.
    navigation_table = graph.G.graph.get("_class_navigation_table")
    if navigation_table is None:
        navigation_table = build_class_navigation_table(graph)
        graph.G.graph["_class_navigation_table"] = navigation_table
    parameter_class_names: dict[str, str] = {}
    parameter_numeric_descriptors: dict[str, NumericFeatureDescriptor] = {}
    parameter_aggregate_kinds: dict[str, str] = {}
    parameter_sequence_record_widths: dict[str, int] = {}
    known_class_identities = {
        record.identity for record in navigation_table.classes
    }

    def annotation_aggregate_kind(annotation: ast.AST | None) -> str | None:
        if isinstance(annotation, ast.Name) and annotation.id in {
            "list", "set", "dict", "tuple", "bytes", "bytearray",
        }:
            return annotation.id
        if isinstance(annotation, ast.Subscript):
            return annotation_aggregate_kind(annotation.value)
        if isinstance(annotation, ast.BinOp) and isinstance(
            annotation.op, ast.BitOr,
        ):
            kinds = tuple(filter(None, (
                annotation_aggregate_kind(annotation.left),
                annotation_aggregate_kind(annotation.right),
            )))
            return kinds[0] if len(set(kinds)) == 1 else None
        return None

    for argument in (
        *statement.args.posonlyargs,
        *statement.args.args,
        *statement.args.kwonlyargs,
    ):
        annotation = argument.annotation
        numeric_descriptor = numeric_feature_descriptor(annotation)
        declared_class_identity = (
            numeric_descriptor.type_name
            if numeric_descriptor is not None
            else annotation.id if isinstance(annotation, ast.Name) else None
        )
        if (
            declared_class_identity is not None
            and declared_class_identity in known_class_identities
        ):
            parameter_class_names[argument.arg] = declared_class_identity
            if numeric_descriptor is not None:
                parameter_numeric_descriptors[argument.arg] = (
                    numeric_descriptor
                )
        aggregate_kind = annotation_aggregate_kind(annotation)
        if aggregate_kind is not None:
            parameter_aggregate_kinds[argument.arg] = aggregate_kind
    for parameter_name, record in dict(
        graph.G.graph.get("parameter_sequence_record_abi") or {}
    ).items():
        # NOT named ``fields``: this function also closes over
        # ``static_reference_node``, which calls ``dataclasses.fields``. A
        # local of that name here makes ``fields`` a local of the WHOLE
        # enclosing scope, so the nested call resolves to this cell instead
        # of the import and raises "cannot access free variable 'fields'"
        # on any path that reaches the nested function before this loop.
        record_fields = tuple(dict(record.get("fields") or {}))
        if not record_fields:
            continue
        parameter_aggregate_kinds[str(parameter_name)] = str(
            record.get("aggregate_kind") or "tuple"
        )
        parameter_sequence_record_widths[str(parameter_name)] = len(record_fields)
    # Source pursuit records an explicitly constructed local aggregate by
    # its class (``opt_subs = {}`` -> ``opt_subs: dict``).  For a nested
    # function that name is captured runtime storage, not a compile-time
    # reference to the ``dict`` constructor.  Preserve that distinction at
    # the function boundary so indexed mutation receives a real SSA input.
    parameter_aggregate_kinds.update(closure_aggregate_kinds or {})
    parameter_aggregate_kinds.update(
        dict(
            getattr(statement, "_python_aggregate_binding_kinds", {}) or {}
        )
    )

    def _resolve_instance_attribute_slot(
        class_identity: str, attribute_name: str
    ) -> int | None:
        """A field's real position in ``class_identity``'s layout, or ``None``.

        Permission evaluation here is deliberately permissive: this call
        resolves *structural* identity for SSA construction (which slot),
        the same question ``resolve_dot`` answers for real elsewhere in the
        compiler -- it does not enforce access policy.  A denied/ambiguous
        resolution is a fact about the source (an unknown or non-attribute
        member), not a security decision, so it is treated as "not slot-
        resolvable" rather than raised.
        """

        try:
            member = navigation_table.resolve_dot(
                class_identity,
                attribute_name,
                lambda _identity, _permissions: True,
                receiver_kind="instance",
            )
        except (KeyError, PermissionError):
            return None
        return member.slot if member.kind == "attribute" else None
    exception_local_names = {
        target.id
        for handler in (
            node
            for node in source_walk(statement)
            if isinstance(node, ast.ExceptHandler)
        )
        for body_node in handler.body
        for assignment in source_walk(body_node)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            (*assignment.targets,)
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
    }
    scalar_loop_target_ast_ids: set[int] = set()

    def target_name_nodes(target: ast.AST) -> tuple[ast.Name, ...]:
        if isinstance(target, ast.Name):
            return (target,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for element in target.elts
                for name in target_name_nodes(element)
            )
        return ()

    for loop in source_walk(statement):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
        ):
            continue
        if loop.iter.func.id == "range":
            scalar_loop_target_ast_ids.update(
                id(name) for name in target_name_nodes(loop.target)
            )
        elif loop.iter.func.id == "enumerate":
            if isinstance(loop.target, (ast.Tuple, ast.List)) and loop.target.elts:
                scalar_loop_target_ast_ids.update(
                    id(name)
                    for name in target_name_nodes(loop.target.elts[0])
                )
    scalar_loop_binding_ids: set[int] = set()
    # Every ``return`` of THIS function names its output slots -- not only a
    # return that happens to sit at the top level of the body. Scanning
    # ``source_body_statements`` alone left a function whose returns are all
    # inside a loop or conditional (``while True: ... return a, b, c``)
    # with no ``function_outputs`` at all, so the per-Return identity
    # recording below never ran, the hierarchy plan built its PlanCall with
    # ``result_bindings=()``, and every caller bound zero of the callee's
    # outputs ("unmaterialized_result", 15 real outputs unbound -- diagnosed
    # against the real managed-tire compile, ``step_with_dt_control_used``).
    # Nested definitions own their own returns and are not walked into.
    def function_return_statements(definition: Any) -> tuple[Any, ...]:
        returns: list[Any] = []
        stack = list(reversed(source_body_statements(definition)))
        while stack:
            current = stack.pop()
            if isinstance(current, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
            )):
                continue
            if isinstance(current, ast.Return):
                returns.append(current)
                continue
            stack.extend(reversed(source_child_nodes(current)))
        return tuple(returns)

    return_expression_lists = tuple(
        (
            tuple(returned.elts)
            if isinstance(returned, (ast.Tuple, ast.List))
            else (returned,)
        )
        for return_statement in function_return_statements(statement)
        for returned in (return_statement.value,)
        if returned is not None
    )
    if return_expression_lists:
        arities = {len(expressions) for expressions in return_expression_lists}
        if len(arities) != 1:
            # Returns that disagree on arity have no single output surface;
            # that is a real shortfall to report, never a silent ``()``.
            graph.G.graph["function_output_shortfall"] = (
                "return arity disagrees across return statements: "
                f"{sorted(arities)!r}"
            )
        else:
            arity = next(iter(arities))
            # A slot keeps its authored name only when EVERY return spells
            # that slot as the same name; otherwise it is positional.
            slot_names = []
            positional_names = []
            for index in range(arity):
                names = {
                    expressions[index].id
                    if isinstance(expressions[index], ast.Name)
                    else None
                    for expressions in return_expression_lists
                }
                if len(names) == 1 and None not in names:
                    slot_names.append(names.pop())
                else:
                    slot_names.append(f"result_{index}")
                    positional_names.append(slot_names[-1])
            graph.G.graph["function_outputs"] = tuple(slot_names)
            # A positional slot's identity history is a list of ALTERNATIVE
            # return values (one per return site), not the version chain of
            # a rebound variable. Consumers that merge variable versions
            # across conditionals (`_ordinary_conditional_control_programs`)
            # must not treat it as one, or every earlier return's value is
            # aliased onto the next return's (measured as degenerate
            # `Phi [329, 329]` merges and "last return wins" outputs).
            graph.G.graph["positional_output_names"] = tuple(positional_names)
    def new_node(
        node_type: str,
        label: str,
        *,
        attributes: dict[str, Any] | None = None,
        parents: tuple[tuple[int, str], ...] = (),
        source: Any = None,
    ) -> int:
        # Value ids are identities: never hand out an id freed by an earlier
        # removal.  The per-graph watermark only moves forward (see
        # src.compiler.process_graph_value_ids for the shared rule).
        metadata = graph.G.graph
        node_id = max(
            int(metadata.get("value_id_watermark", -1)),
            max((int(existing) for existing in graph.G.nodes), default=-1),
        ) + 1
        metadata["value_id_watermark"] = node_id
        # A synthesized node has no `expr_obj`, so nothing downstream can
        # place it lexically unless the authored statement it stands for is
        # recorded here. `source` is that statement/expression: its span is
        # stamped as the node-level `source_span` every lexical-ownership
        # test reads first (e.g. loop_composer's `is_lexical_body_node`),
        # while `expr_obj` stays None so the node is never mistaken for the
        # authored expression itself.
        node_source_span = (
            {
                "line": getattr(source, "lineno", None),
                "column": getattr(source, "col_offset", None),
                "end_line": getattr(source, "end_lineno", None),
                "end_column": getattr(source, "end_col_offset", None),
            }
            if source is not None and getattr(source, "lineno", None) is not None
            else None
        )
        graph.G.add_node(
            node_id,
            label=label,
            type=node_type,
            op=node_type.lower(),
            expr_obj=None,
            extra_args={},
            domain_node=None,
            store_id=None,
            parents=list(parents),
            children=[],
            attributes=dict(attributes or {}),
            **({"source_span": node_source_span} if node_source_span else {}),
        )
        if node_type in {"Const", "Constant"}:
            graph.G.nodes[node_id]["constant"] = (
                attributes or {}
            ).get("value")
        for parent_id, role in parents:
            if parent_id not in graph.G:
                continue
            graph.G.add_edge(parent_id, node_id, role=role)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    def input_value(name: str, *, binding_kind: str) -> int:
        value = environment.get(name)
        if value is not None:
            return value
        aggregate_kind = parameter_aggregate_kinds.get(name)
        attributes = {
            "binding_name": name,
            "binding_kind": binding_kind,
        }
        class_identity = parameter_class_names.get(name)
        numeric_descriptor = parameter_numeric_descriptors.get(name)
        if class_identity is not None:
            # A typed parameter already IS an instance of its declared class.
            # ``class_ref`` has the narrower, executable meaning "this node
            # constructs an instance" throughout deployment planning.  Giving
            # it to an Input caused every numeric wrapper parameter to acquire
            # a synthetic ``__init__`` call, after which the linker exported
            # the constructor's private frame as thousands of root formals.
            # The source-value class concordance below is authoritative for
            # receiver identity; ``result_class_ref`` is its graph view.
            attributes["result_class_ref"] = class_identity
        if numeric_descriptor is not None:
            attributes.update({
                "numeric_feature_descriptor": numeric_descriptor.receipt(),
                "precision_limbs": numeric_descriptor.limbs,
            })
        if aggregate_kind is not None:
            attributes.update({
                "producer_kind": "aggregate_parameter",
                "aggregate_kind": aggregate_kind,
                "sequence_key_columns": (
                    (0,) if aggregate_kind in {"set", "dict"} else ()
                ),
                "sequence_column_count": (
                    2 if aggregate_kind == "dict" else
                    parameter_sequence_record_widths.get(name, 1)
                ),
                "sequence_writable": aggregate_kind not in {"tuple", "bytes"},
            })
        value = new_node(
            "Input",
            name,
            attributes=attributes,
        )
        environment[name] = value
        identity_bindings.setdefault(name, []).append(value)
        ingestion_definitions.setdefault(name, []).append((value, {}))
        if class_identity is not None:
            _concord_source_value_class(
                value_class_scope,
                value,
                class_identity,
                precision_limbs=(
                    numeric_descriptor.limbs
                    if numeric_descriptor is not None else 1
                ),
                source=f"parameter annotation {name}",
            )
        return value

    lexical_free_names_by_reference: dict[int, tuple[str, ...]] = {}

    def captured_authored_parameters() -> tuple[str, ...]:
        """Authored parameters read by a nested lexical scope.

        Python's symbol table is the authoritative closure record. Without
        materializing these outer identities, ``root(..., gain)`` where only
        ``blend`` reads ``gain`` loses the outer Input and grows one anonymous
        frame slot per callsite. The same loss removes ``static_bindings``
        from ``_normalize_lexical_values`` even though nested
        ``resolve_expression`` consumes that mapping.
        """

        try:
            source = ast.unparse(ast.Module(body=[statement], type_ignores=[]))
            module_table = symtable.symtable(
                source, "<turing-closure-parameters>", "exec",
            )
            owner = next(
                child for child in module_table.get_children()
                if child.get_name() == statement.name
            )
        except (StopIteration, SyntaxError, TypeError, ValueError):
            return ()
        free_names: set[str] = set()
        pending = list(owner.get_children())
        while pending:
            child = pending.pop()
            child_free_names = tuple(
                symbol.get_name()
                for symbol in child.get_symbols()
                if symbol.is_free()
            )
            free_names.update(child_free_names)
            reference = (lexical_function_bindings or {}).get(
                child.get_name()
            )
            if reference is not None:
                lexical_free_names_by_reference[int(reference.address)] = (
                    child_free_names
                )
            pending.extend(child.get_children())
        # A nested function can capture a sibling function whose own closure
        # owns the data cell.  The compiler flattens that Python function-cell
        # indirection into call boundaries, so carry the sibling's data
        # requirements transitively.  For example ``reduce_statement``
        # captures ``resolve_expression`` and the latter captures the outer
        # ``static_bindings`` parameter; the outer parameter must remain live
        # at calls to ``reduce_statement`` even though that spelling is not a
        # direct free symbol in its Python symbol table.
        changed = True
        while changed:
            changed = False
            for reference_address, names in tuple(
                lexical_free_names_by_reference.items()
            ):
                expanded = list(names)
                for name in names:
                    dependency = (lexical_function_bindings or {}).get(name)
                    if dependency is None:
                        continue
                    expanded.extend(lexical_free_names_by_reference.get(
                        int(dependency.address), (),
                    ))
                normalized = tuple(dict.fromkeys(expanded))
                if normalized != names:
                    lexical_free_names_by_reference[reference_address] = (
                        normalized
                    )
                    changed = True
        return tuple(
            name for name in function_parameter_names(statement)
            if name in free_names
        )

    # These are existing authored parameter identities, not new ABI values.
    # Establish them before nested-call reconstruction so every closure edge
    # carries the same object from its enclosing frame.
    for captured_parameter in captured_authored_parameters():
        input_value(captured_parameter, binding_kind="parameter")

    def record_ingestion_definition(
        name: str, value: int, target: ast.AST,
    ) -> None:
        """Record an authored binding, never a read or SSA-only artifact."""

        value_data = graph.G.nodes.get(int(value), {})
        dependency_shape = tuple(
            (str(role), str(graph.G.nodes[parent].get("op", "unknown")))
            for parent, role in value_data.get("parents", ())
            if parent in graph.G
        )
        source_span = {
            "line": getattr(target, "lineno", None),
            "column": getattr(target, "col_offset", None),
            "end_line": getattr(target, "end_lineno", None),
            "end_column": getattr(target, "end_col_offset", None),
        }
        context = {
            "spelling": str(name),
            "node_kind": type(target).__name__,
            "target": ast.dump(target, include_attributes=False),
            "producer_op": str(value_data.get("op", "unknown")),
            "dependency_shape": dependency_shape,
            "source_span": source_span,
        }
        context_tokens = structural_context_tokens(context)
        ingestion_definitions.setdefault(str(name), []).append((
            int(value), {
                **source_span,
                "context": context,
                "context_tokens": context_tokens,
                "context_token_ids": tuple(
                    encode_identity_tokens({"token": token})
                    for token in context_tokens
                ),
                "context_sha256": hashlib.sha256(
                    json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        ))

    def static_constant(name: str, value: Any) -> int:
        existing = static_constant_nodes.get(name)
        if existing is not None:
            return existing
        python_special_case = interpret_python_static_value(
            value,
            path=name,
        )
        if python_special_case is not None:
            value = python_special_case.constant
            attributes = {
                **python_special_case.attributes,
                "binding_name": name,
            }
        else:
            attributes = {"value": value, "binding_name": name}
        if isinstance(value, dict) and all(
            isinstance(key, (bool, int, float, str))
            and isinstance(item, (bool, int, float, str))
            for key, item in value.items()
        ):
            # Keep an immutable module mapping as structural source, not an
            # opaque scalar constant.  The repository-SSA lowerer materializes
            # these exact rows into a local keyed table, so dynamic source keys
            # use the same lookup path as ordinary authored dictionaries.
            attributes.update({
                "producer_kind": "compile_time_mapping",
                "aggregate_kind": "dict",
                "sequence_key_columns": (0,),
                "sequence_column_count": 2,
                "sequence_writable": False,
                "compile_time_mapping_items": tuple(value.items()),
            })
        node_id = new_node(
            "Constant",
            name,
            attributes=attributes,
        )
        static_constant_nodes[name] = node_id
        return node_id

    def is_static_literal(value: Any) -> bool:
        if value is None or value is Ellipsis or isinstance(
            value,
            (bool, bytes, complex, float, int, str),
        ):
            return True
        if isinstance(value, (tuple, list)):
            return all(is_static_literal(item) for item in value)
        if isinstance(value, dict):
            return all(
                is_static_literal(key) and is_static_literal(item)
                for key, item in value.items()
            )
        return False

    def static_reference_node(reference: _StaticPythonReference) -> int:
        """Materialize one compiler reference without exposing its Python value."""

        key = (id(reference.value), reference.path)
        existing = static_reference_nodes.get(key)
        if existing is not None and existing in graph.G:
            cached = graph.G.nodes[existing]
            if (cached.get("type") == "StaticReference" and
                    (cached.get("attributes") or {}).get("static_python_reference") == reference.path):
                return existing
        # Lexical normalization can remove an already-resolved reference
        # projection. Recreate its compiler symbol from the real reference;
        # the cache must never publish a removed or repurposed graph node.
        target = reference.value
        target_name = str(getattr(target, "__name__", ""))
        class_descriptor = graph.G.graph.get("class_table", {}).get(
            target_name
        )
        if class_descriptor is not None and is_dataclass(target):
            # Imported dataclasses enter the function table as structural
            # class references, but their defining ClassDef belongs to a
            # different source unit.  Recover the constructor schema here so
            # omitted literal defaults (for example SuperstepPlan.eps) remain
            # compiler facts instead of disappearing into Python reflection
            # at execution time.
            dataclass_fields = tuple(fields(target))
            class_descriptor["fields"] = tuple(
                field.name for field in dataclass_fields
            )
            defaults = dict(class_descriptor.get("field_defaults") or {})
            for field in dataclass_fields:
                if field.default is not MISSING and is_static_literal(
                    field.default
                ):
                    defaults[field.name] = copy.deepcopy(field.default)
            class_descriptor["field_defaults"] = defaults
        function_reference = (
            function_table.reference(target_name) if target_name else None
        )
        if getattr(target, "__self__", None) is not None:
            # A bound runtime/builtin method is identified by both callable
            # and receiver.  Linking it to an unrelated source method by the
            # final spelling alone (for example ContextVar.get versus a
            # user class's get) discards the receiver and is never valid.
            function_reference = None
        attributes = {
            "static_python_reference": reference.path,
            "reference_kind": (
                "function_subgraph"
                if function_reference is not None
                else (
                    "class_subgraphs"
                    if class_descriptor is not None
                    else "static_symbol"
                )
            ),
        }
        if function_reference is not None:
            attributes["function_ref"] = function_reference.address
        if class_descriptor is not None:
            attributes["class_ref"] = target_name
        node_id = new_node(
            "StaticReference",
            reference.path,
            attributes=attributes,
        )
        static_reference_nodes[key] = node_id
        return node_id

    def first_class_function_node(name: str, reference: Any) -> int:
        """Represent a source function used as data by its table address."""

        address = int(reference.address)
        existing = first_class_function_nodes.get(address)
        if existing is not None:
            return existing
        node_id = new_node(
            "StaticReference",
            name,
            attributes={
                "function_ref": address,
                "first_class_function_ref": address,
                "reference_kind": "function_subgraph",
            },
        )
        current_identity_book().page("callable_identity_concordance").set(
            (
                _source_numeric_scope(graph),
                int(node_id),
            ),
            0,
            int(address),
        )
        first_class_function_nodes[address] = node_id
        return node_id

    def bind_loop_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            target_identity = id(target)
            value = loop_target_bindings_by_ast.get(target_identity)
            if value is None:
                value = new_node(
                    "Input",
                    target.id,
                    attributes={
                        "binding_name": target.id,
                        "binding_kind": "loop",
                    },
                )
                loop_target_bindings_by_ast[target_identity] = value
                identity_bindings.setdefault(target.id, []).append(value)
            if target_identity in scalar_loop_target_ast_ids:
                scalar_loop_binding_ids.add(value)
            environment[target.id] = value
            _remove_node(graph, id(target))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind_loop_target(element)

    def loop_target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for element in target.elts
                for name in loop_target_names(element)
            )
        return ()

    restored_literal_nodes: dict[int, int] = {}

    def resolve_expression(expression: ast.AST | None) -> int | None:
        if expression is None:
            return None
        if isinstance(expression, ast.Constant) and id(expression) not in graph.G:
            # Structural reduction can remove a keyword wrapper and its leaf
            # before lexical call binding. A live use still has the exact
            # authored literal; materialize it, rather than drop the operand.
            if id(expression) not in restored_literal_nodes:
                restored_literal_nodes[id(expression)] = new_node(
                    "Constant", repr(expression.value),
                    attributes={"value": expression.value}, source=expression,
                )
            return restored_literal_nodes[id(expression)]
        if isinstance(expression, ast.NamedExpr):
            value = resolve_expression(expression.value)
            bind_target(expression.target, value)
            node_id = id(expression)
            if isinstance(value, int):
                _redirect_value(graph, node_id, value)
            else:
                _remove_node(graph, node_id)
            return value
        if isinstance(
            expression,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            # Comprehensions and generator expressions have owned their own
            # scope since Python 3.0: a `for x in ...` target inside one is
            # never visible outside it, even when an outer variable of the
            # same name already exists or gets bound later. (The one
            # exception, a `:=` walrus target, deliberately leaks to the
            # enclosing scope per PEP 572 -- that goes through bind_target,
            # not bind_loop_target, and is untouched here.) Without this,
            # a comprehension's `for` target silently overwrote (or was
            # later mistaken for) an unrelated same-named binding elsewhere
            # in the function -- for example `tuple(f(x) for x in seq)`
            # followed later by an ordinary `for x in other:` picking up
            # the comprehension's already-evaporated node as its own
            # "before this loop" value, and crashing much later with
            # "missing ProcessGraph input" once that node was removed.
            comprehension_target_names = {
                name
                for generator in expression.generators
                for name in loop_target_names(generator.target)
            }
            shadowed_bindings = {
                name: environment[name]
                for name in comprehension_target_names
                if name in environment
            }
            for generator in expression.generators:
                resolve_expression(generator)
            if isinstance(expression, ast.DictComp):
                resolve_expression(expression.key)
                resolve_expression(expression.value)
            else:
                resolve_expression(expression.elt)
            node_id = id(expression)
            if (
                not isinstance(expression, ast.GeneratorExp)
                and node_id in graph.G
            ):
                aggregate_kind = {
                    ast.ListComp: "list",
                    ast.SetComp: "set",
                    ast.DictComp: "dict",
                }[type(expression)]
                graph.G.nodes[node_id].setdefault(
                    "attributes", {}
                ).update({
                    # A comprehension is a dynamic resident sequence built
                    # by its producer loop, not a closure aggregate whose
                    # complete member identities are statically available.
                    "producer_kind": "sequence_materialization",
                    "aggregate_kind": aggregate_kind,
                    "sequence_key_columns": (
                        (0,) if aggregate_kind in {"set", "dict"} else ()
                    ),
                    "sequence_column_count": (
                        2 if aggregate_kind == "dict" else 1
                    ),
                    "sequence_writable": True,
                })
            if isinstance(expression, (ast.ListComp, ast.SetComp, ast.DictComp)):
                if isinstance(expression, ast.DictComp):
                    key_id = resolve_expression(expression.key)
                    item_id = resolve_expression(expression.value)
                    attributes = graph.G.nodes[node_id].setdefault("attributes", {})
                    value_id = attributes.get("sequence_row_value_id")
                    if value_id is None and isinstance(key_id, int) and isinstance(item_id, int):
                        value_id = new_node(
                            "Tuple", "dictionary comprehension row",
                            parents=((key_id, "elts"), (item_id, "elts")),
                            attributes={"producer_kind": "aggregate", "aggregate_kind": "tuple",
                                "aggregate_leaf_value_ids": (key_id, item_id),
                                "sequence_column_count": 2, "sequence_writable": False},
                            source=expression,
                        )
                        attributes["sequence_row_value_id"] = value_id
                    if isinstance(value_id, int):
                        _replace_inputs(graph, node_id, tuple(
                            (int(parent), role)
                            for parent, role in graph.G.nodes[node_id].get("parents", ())
                            if role not in {"key", "value", "elt"}
                        ) + ((value_id, "elt"),))
                else:
                    value_id = resolve_expression(expression.elt)
                if (
                    isinstance(value_id, int)
                    and value_id in graph.G
                    and node_id in graph.G
                ):
                    value_attributes = (
                        graph.G.nodes[value_id].get("attributes") or {}
                    )
                    row_leaves = tuple(map(
                        int,
                        value_attributes.get(
                            "aggregate_leaf_value_ids", ()
                        ),
                    ))
                    if (
                        value_attributes.get("aggregate_kind") == "tuple"
                        and len(row_leaves) > 1
                    ):
                        # A comprehension over a tuple expression authors a
                        # fixed-width resident row sequence.  Publish that
                        # width at normalization time, where the element and
                        # its aggregate leaves are both known; leaving the
                        # default width of one makes every downstream backend
                        # see an impossible two-value insertion into a scalar
                        # arena.
                        graph.G.nodes[node_id].setdefault(
                            "attributes", {}
                        )["sequence_column_count"] = len(row_leaves)
                for generator in expression.generators:
                    generator_id = id(generator)
                    if (
                        generator_id in graph.G
                        and isinstance(value_id, int)
                        and value_id in graph.G
                        and node_id in graph.G
                    ):
                        attributes = graph.G.nodes[
                            generator_id
                        ].setdefault("attributes", {})
                        outputs = list(
                            attributes.get("loop_iteration_outputs", ())
                        )
                        outputs.append({
                            "value_id": value_id,
                            "result_value_id": node_id,
                            "materializer_node_id": node_id,
                        })
                        attributes["loop_iteration_outputs"] = tuple(outputs)
            for name in comprehension_target_names:
                if name in shadowed_bindings:
                    environment[name] = shadowed_bindings[name]
                else:
                    environment.pop(name, None)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.IfExp):
            test_value = resolve_expression(expression.test)
            body_value = resolve_expression(expression.body)
            else_value = resolve_expression(expression.orelse)
            node_id = id(expression)
            if node_id in graph.G:
                # ``x if isinstance(x, T) else normalize_T(x)`` is an
                # authored type-normalization idiom, not runtime branching.
                # A constructor/factory opts in declaratively through its
                # Python identity program's ``ensures_schema_type`` attribute.
                # Keep that ordinary operator as the sole SSA producer; the
                # type guard and merge then disappear without specializing on
                # a Python object or inventing backend-specific machinery.
                def dotted_name(value: ast.AST) -> str | None:
                    parts = []
                    while isinstance(value, ast.Attribute):
                        parts.append(str(value.attr))
                        value = value.value
                    if isinstance(value, ast.Name):
                        parts.append(str(value.id))
                        return ".".join(reversed(parts))
                    return None

                guard = expression.test
                if (
                    isinstance(guard, ast.Call)
                    and isinstance(guard.func, ast.Name)
                    and guard.func.id == "isinstance"
                    and len(guard.args) == 2
                    and isinstance(test_value, int)
                    and test_value in graph.G
                ):
                    test_attributes = graph.G.nodes[test_value].get(
                        "attributes"
                    ) or {}
                    test_program = test_attributes.get(
                        "python_identity_program"
                    ) or {}
                    subject_id = next((
                        int(parent)
                        for parent, role in graph.G.nodes[test_value].get(
                            "parents", ()
                        )
                        if str(role) == "arg:0"
                    ), None)
                    normalized_id = (
                        int(else_value)
                        if isinstance(else_value, int) else None
                    )
                    normalized_data = (
                        graph.G.nodes[normalized_id]
                        if normalized_id is not None
                        and normalized_id in graph.G else {}
                    )
                    normalized_attributes = normalized_data.get(
                        "attributes"
                    ) or {}
                    ensured_type = normalized_attributes.get(
                        "ensures_schema_type"
                    )
                    normalized_subject = next((
                        int(parent)
                        for parent, role in normalized_data.get("parents", ())
                        if str(role) == "arg:0"
                    ), None)
                    guarded_type = dotted_name(guard.args[1])
                    if os.environ.get("TURING_DEBUG_TYPE_NORMALIZATION"):
                        logger.warning(
                            "type-normalization candidate line=%s "
                            "test_program=%r subject=%r body=%r "
                            "normalized=%r normalized_subject=%r "
                            "ensured_type=%r guarded_type=%r "
                            "guard_receipt=%r normalizer_receipt=%r "
                            "test_attributes=%r normalized_attributes=%r",
                            getattr(expression, "lineno", None),
                            test_program,
                            subject_id,
                            body_value,
                            normalized_id,
                            normalized_subject,
                            ensured_type,
                            guarded_type,
                            getattr(guard, "_extraction_contract", None),
                            getattr(expression.orelse, "_extraction_contract", None),
                            test_attributes,
                            normalized_attributes,
                        )
                    if (
                        test_program.get("object_type")
                        == "schema_type_guard"
                        and subject_id is not None
                        and body_value == subject_id
                        and normalized_subject == subject_id
                        and ensured_type
                        and guarded_type
                        and (
                            str(ensured_type) == guarded_type
                            or str(ensured_type).endswith(
                                "." + guarded_type
                            )
                        )
                    ):
                        normalization_descriptor = (
                            int(subject_id),
                            int(normalized_id),
                            str(ensured_type),
                            str(guarded_type),
                        )
                        normalization_page = current_identity_book().page(
                            "source_type_normalization_concordance"
                        )
                        normalization_row = (
                            value_class_scope, int(node_id)
                        )
                        normalization_incumbent = normalization_page.latest(
                            normalization_row
                        )
                        if (
                            normalization_incumbent is not None
                            and tuple(normalization_incumbent)
                            != normalization_descriptor
                        ):
                            raise ValueError(
                                "source type normalization concordance "
                                f"disagreement for {normalization_row!r}: "
                                f"recorded={normalization_incumbent!r}, "
                                f"resolved={normalization_descriptor!r}"
                            )
                        if normalization_incumbent is None:
                            normalization_page.set(
                                normalization_row,
                                0,
                                normalization_descriptor,
                            )
                        graph.G.nodes[normalized_id].setdefault(
                            "attributes", {}
                        )["source_type_normalization"] = {
                            "guard": "schema_type_guard",
                            "schema_type": str(ensured_type),
                            "source_ifexp": int(node_id),
                        }
                        _redirect_value(graph, node_id, normalized_id)
                        return normalized_id
                parents = tuple(
                    (value, role)
                    for value, role in (
                        (test_value, "test"),
                        (body_value, "body"),
                        (else_value, "orelse"),
                    )
                    if isinstance(value, int)
                )
                _replace_inputs(graph, node_id, parents)
                contracts = []
                for value in (body_value, else_value):
                    if not isinstance(value, int) or value not in graph.G:
                        contracts.append(None)
                        continue
                    attributes = graph.G.nodes[value].get("attributes") or {}
                    kind = attributes.get("aggregate_kind")
                    contracts.append(
                        None if kind is None else (
                            str(kind),
                            tuple(attributes.get("sequence_key_columns", ())),
                            int(attributes.get("sequence_column_count", 1)),
                            bool(attributes.get("sequence_writable", True)),
                        )
                    )
                if contracts[0] is not None and contracts[0] == contracts[1]:
                    kind, key_columns, column_count, writable = contracts[0]
                    graph.G.nodes[node_id].setdefault(
                        "attributes", {}
                    ).update({
                        "producer_kind": "aggregate_merge",
                        "aggregate_kind": kind,
                        "sequence_key_columns": key_columns,
                        "sequence_column_count": column_count,
                        "sequence_writable": writable,
                    })
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.comprehension):
            resolve_expression(expression.iter)
            bind_loop_target(expression.target)
            node_id = id(expression)
            if node_id in graph.G:
                graph.G.nodes[node_id].setdefault("attributes", {})[
                    "loop_target_bindings"
                ] = {
                    name: environment[name]
                    for name in loop_target_names(expression.target)
                }
            for condition in expression.ifs:
                resolve_expression(condition)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.Name):
            node_id = id(expression)
            if isinstance(expression.ctx, ast.Load):
                if expression.id in static_parameter_bindings:
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        static_parameter_bindings[expression.id],
                        expression.id,
                    )
                static_reference = static_environment.get(expression.id)
                if static_reference is not None:
                    _remove_node(graph, node_id)
                    return static_reference
                producer_id = environment.get(expression.id)
                if (
                    producer_id is None
                    and expression.id in (closure_aggregate_kinds or {})
                ):
                    producer_id = input_value(
                        expression.id,
                        binding_kind="closure",
                    )
                static_value = static_bindings.get(expression.id)
                function_reference = (
                    (lexical_function_bindings or {}).get(expression.id)
                    or function_table.reference(expression.id)
                )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and function_reference is not None
                ):
                    _remove_node(graph, node_id)
                    return first_class_function_node(
                        expression.id,
                        function_reference,
                    )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and static_value is not None
                    and isinstance(
                        static_value,
                        (
                            types.ModuleType,
                            type,
                            types.FunctionType,
                            types.BuiltinFunctionType,
                            types.MethodType,
                        ),
                    )
                ):
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        static_value,
                        expression.id,
                    )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and expression.id in static_bindings
                    and is_static_literal(static_value)
                ):
                    constant_id = static_constant(
                        expression.id,
                        static_value,
                    )
                    _redirect_value(graph, node_id, constant_id)
                    return constant_id
                if producer_id is None:
                    producer_id = input_value(
                        expression.id,
                        binding_kind=(
                            "parameter"
                            if expression.id in parameter_names
                            else "exception"
                            if expression.id in exception_local_names
                            else "external"
                        ),
                    )
                # Which binding each consumer read is a fact of this read,
                # not of the value: two names can hold one value (``second
                # = value``) and a loop carries one of them while the other
                # stays put.  Commit it before the occurrence dissolves into
                # a plain value edge.
                _concord_lexical_reads(
                    graph, ingestion_read_scope, node_id, expression.id,
                )
                _redirect_value(graph, node_id, producer_id)
                return producer_id
            return environment.get(expression.id)
        if isinstance(expression, ast.Attribute):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id not in environment
                and expression.value.id not in parameter_names
                and expression.value.id in static_bindings
            ):
                _remove_node(graph, id(expression.value))
                receiver = _StaticPythonReference(
                    static_bindings[expression.value.id],
                    expression.value.id,
                )
            else:
                receiver = resolve_expression(expression.value)
            if isinstance(receiver, _StaticPythonReference):
                attribute_key = (id(receiver.value), expression.attr)
                assigned_value = static_attribute_values.get(attribute_key)
                if assigned_value is not None:
                    _redirect_value(graph, id(expression), assigned_value)
                    return assigned_value
                try:
                    value = getattr(receiver.value, expression.attr)
                except AttributeError:
                    pass
                else:
                    node_id = id(expression)
                    if is_static_literal(value):
                        constant_id = static_constant(
                            f"{receiver.path}.{expression.attr}",
                            value,
                        )
                        _redirect_value(graph, node_id, constant_id)
                        return constant_id
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        value,
                        f"{receiver.path}.{expression.attr}",
                    )
            elif isinstance(receiver, int):
                expression_id = id(expression)
                attribute_id = (
                    expression_id
                    if expression_id in graph.G
                    else materialized_attribute_nodes.get(expression_id)
                )
                if attribute_id is None:
                    attribute_id = new_node(
                        "GetAttr",
                        f"getattr[{expression.attr}]",
                        attributes={
                            "attribute": expression.attr,
                            "source_type": "Attribute",
                        },
                    )
                    materialized_attribute_nodes[expression_id] = attribute_id
                # A field read immediately from an authored record
                # construction is the exact constructor argument.  Preserve
                # that identity directly, especially when the argument is a
                # first-class function reference: manufacturing a GetAttr
                # value here loses the function-table address before record
                # materialization and leaves a later call with no callee.
                receiver_data = graph.G.nodes.get(int(receiver), {})
                receiver_expression = receiver_data.get("expr_obj")
                receiver_class = (
                    receiver_data.get("attributes") or {}
                ).get("class_ref")
                records = dict(
                    (graph.G.graph.get("program_abi") or {}).get("records")
                    or {}
                )
                matching_records = tuple(
                    record
                    for name, record in records.items()
                    if receiver_class is not None
                    and (
                        str(name) == str(receiver_class)
                        or str(record.get("identity") or name)
                        == str(receiver_class)
                        or str(record.get("identity") or name).endswith(
                            "." + str(receiver_class)
                        )
                    )
                )
                if (
                    isinstance(receiver_expression, ast.Call)
                    and len(matching_records) == 1
                ):
                    field_names = tuple(
                        map(
                            str,
                            dict(matching_records[0].get("fields") or {}),
                        )
                    )
                    field_expression = next((
                        keyword.value
                        for keyword in receiver_expression.keywords
                        if keyword.arg == expression.attr
                    ), None)
                    if field_expression is None and expression.attr in field_names:
                        position = field_names.index(expression.attr)
                        if position < len(receiver_expression.args):
                            field_expression = receiver_expression.args[position]
                    if field_expression is not None:
                        field_value = resolve_expression(field_expression)
                        if isinstance(field_value, int):
                            field_attributes = (
                                graph.G.nodes.get(field_value, {}).get(
                                    "attributes"
                                ) or {}
                            )
                            function_ref = field_attributes.get(
                                "first_class_function_ref"
                            )
                            if function_ref is not None:
                                page = current_identity_book().page(
                                    "callable_identity_concordance"
                                )
                                row = (
                                    str(graph.G.graph.get("function_name")
                                        or "<module>"),
                                    int(field_value),
                                )
                                recorded = page.latest(row)
                                if recorded is None:
                                    page.set(row, 0, int(function_ref))
                                elif int(recorded) != int(function_ref):
                                    raise ValueError(
                                        "callable identity concordance "
                                        f"disagrees for {row!r}: "
                                        f"{recorded!r} != {function_ref!r}"
                                    )
                                page.set(row, 1, int(function_ref))
                            _redirect_value(
                                graph, int(attribute_id), int(field_value)
                            )
                            return int(field_value)
                read_inputs: list[tuple[int, str]] = [(receiver, "value")]
                last_write = attribute_effect_nodes.get(
                    (receiver, expression.attr)
                )
                # Order this read after the most recent recorded write to the
                # same field -- a whole-attribute ``SetAttr`` or an
                # element-wise ``obj.field[i, j] = ...`` ``IndexedStore`` --
                # so the scheduler cannot place a stale read before the
                # write that produced the field's current contents (see
                # ``attribute_effect_nodes``/``bind_target``). This is a pure
                # ordering edge, not a value producer: the read still reads
                # through ``receiver``, an in-place mutation.
                if (
                    last_write is not None
                    and last_write in graph.G
                    and last_write != attribute_id
                ):
                    read_inputs.append((last_write, "after_write"))
                _replace_inputs(
                    graph,
                    attribute_id,
                    tuple(read_inputs),
                )
                if isinstance(expression.value, ast.Name):
                    # Mirror ``SetAttr``'s ``attribute_slot`` (see
                    # ``bind_target``) on the read side: the field's real
                    # position in its class's declared instance storage, not
                    # a name invented at this call site.
                    class_identity = parameter_class_names.get(
                        expression.value.id
                    )
                    if class_identity is not None:
                        slot = _resolve_instance_attribute_slot(
                            class_identity, expression.attr
                        )
                        if slot is not None:
                            graph.G.nodes[attribute_id].setdefault(
                                "attributes", {}
                            )["attribute_slot"] = (class_identity, slot)
                receiver_attributes = (
                    graph.G.nodes[receiver].get("attributes") or {}
                )
                receiver_fact = current_identity_book().page(
                    "source_value_class_concordance"
                ).latest((value_class_scope, int(receiver)))
                receiver_class = (
                    str(method_owner)
                    if (
                        method_owner is not None
                        and isinstance(expression.value, ast.Name)
                        and expression.value.id in {"self", "cls"}
                    )
                    else str(receiver_fact[0])
                    if isinstance(receiver_fact, tuple) else None
                )
                receiver_numeric = (
                    None if not isinstance(receiver_fact, tuple)
                    else numeric_feature_descriptor(
                        str(receiver_fact[0]).rsplit(".", 1)[-1]
                        + (
                            f"[{int(receiver_fact[1])}]"
                            if int(receiver_fact[1]) > 1 else ""
                        )
                    )
                )
                projected_numeric = (
                    None if receiver_numeric is None
                    else numeric_feature_field_projection(
                        receiver_numeric, expression.attr
                    )
                )
                if (
                    receiver_numeric is not None
                    and (
                        (expression.attr in {"real", "imag"}
                         and "complex" in receiver_numeric.features)
                        or (expression.attr in {"numerator", "denominator"}
                            and "rational" in receiver_numeric.features)
                    )
                ):
                    projected_attributes = graph.G.nodes[
                        attribute_id
                    ].setdefault("attributes", {})
                    component_path = (
                        *tuple(receiver_attributes.get(
                            "numeric_component_path", ()
                        )),
                        str(expression.attr),
                    )
                    projected_attributes.update({
                        "numeric_component_path": component_path,
                        "numeric_component_receiver_id": int(receiver),
                        "numeric_component_attribute": str(expression.attr),
                        "precision_limbs": (
                            int(projected_numeric.limbs)
                            if projected_numeric is not None else 1
                        ),
                    })
                    if projected_numeric is not None:
                        projected_attributes.update({
                            "numeric_feature_descriptor": (
                                projected_numeric.receipt()
                            ),
                            "result_class_ref": projected_numeric.type_name,
                        })
                        _concord_source_value_class(
                            value_class_scope,
                            int(attribute_id),
                            projected_numeric.type_name,
                            precision_limbs=projected_numeric.limbs,
                            source="numeric-component-projection",
                        )
                    _concord_numeric_feature_projection(
                        value_class_scope,
                        int(attribute_id),
                        int(receiver),
                        str(expression.attr),
                        component_path,
                        projected_numeric,
                    )
                declared_callable_field = None
                declared_receiver_field = None
                if receiver_class is not None:
                    repository_records = dict(
                        (graph.G.graph.get("program_abi") or {}).get(
                            "records"
                        ) or {}
                    )
                    matching_receiver_records = tuple(
                        record
                        for name, record in repository_records.items()
                        if (
                            str(name) == str(receiver_class)
                            or str(record.get("identity") or name)
                            == str(receiver_class)
                            or str(record.get("identity") or name).endswith(
                                "." + str(receiver_class)
                            )
                        )
                    )
                    receiver_binding_name = receiver_attributes.get(
                        "binding_name"
                    )
                    parameter_receiver_record = dict(
                        graph.G.graph.get("parameter_record_abi") or {}
                    ).get(str(receiver_binding_name))
                    if parameter_receiver_record is not None:
                        matching_receiver_records = (
                            *matching_receiver_records,
                            parameter_receiver_record,
                        )
                    matching_receiver_records = tuple({
                        repr(record): record
                        for record in matching_receiver_records
                    }.values())
                    if len(matching_receiver_records) == 1:
                        candidate_field = dict(
                            matching_receiver_records[0].get("fields") or {}
                        ).get(str(expression.attr))
                        if isinstance(candidate_field, Mapping):
                            declared_receiver_field = dict(candidate_field)
                        if (
                            isinstance(candidate_field, Mapping)
                            and str(candidate_field.get("storage") or "")
                            == "reference"
                            and candidate_field.get("callable_signature")
                            is not None
                        ):
                            declared_callable_field = dict(candidate_field)
                if (
                    declared_receiver_field is not None
                    and str(declared_receiver_field.get("storage") or "")
                    == "keyed"
                ):
                    graph.G.nodes[attribute_id].setdefault(
                        "attributes", {}
                    ).update({
                        "producer_kind": "record_field",
                        "aggregate_kind": "dict",
                        "sequence_key_columns": (0,),
                        "sequence_column_count": 2,
                        "sequence_writable": bool(
                            declared_receiver_field.get("mutable", False)
                        ),
                        "record_field": (
                            str(receiver_class), str(expression.attr)
                        ),
                        "mapping_key_dtype": "int64",
                        "mapping_value_dtype": str(
                            declared_receiver_field.get("dtype") or "unknown"
                        ),
                        **({
                            "mapping_value_tensor": True,
                            "mapping_value_python_type": "AbstractTensor",
                        } if declared_receiver_field.get("value_tensor") else {}),
                        **({
                            "mapping_value_record": str(
                                declared_receiver_field["value_record"]
                            ),
                        } if declared_receiver_field.get("value_record")
                           is not None else {}),
                    })
                if declared_callable_field is not None:
                    graph.G.nodes[attribute_id].setdefault(
                        "attributes", {}
                    ).update({
                        "producer_kind": "record_field",
                        "record_field": (
                            str(receiver_class), str(expression.attr)
                        ),
                        "callable_reference_field": True,
                        "callable_signature": str(
                            declared_callable_field["callable_signature"]
                        ),
                        "callable_optional": bool(
                            declared_callable_field.get("optional", False)
                        ),
                    })
                field_kind = (
                    (class_field_aggregate_kinds or {}).get((
                        str(receiver_class), str(expression.attr)
                    ))
                    if receiver_class is not None else None
                )
                # A field that holds a source-local object: this read
                # yields an instance of that class (``result_class_ref``,
                # the same fact a call returning a class instance carries),
                # so a method call on it resolves through the class table.
                # Not ``class_ref``: that marks a construction, and the
                # deployment side reads its presence as one.
                field_class = (
                    (class_field_classes or {}).get((
                        str(receiver_class), str(expression.attr)
                    ))
                    if receiver_class is not None else None
                )
                if field_class is not None:
                    graph.G.nodes[attribute_id].setdefault(
                        "attributes", {}
                    )["result_class_ref"] = field_class
                if field_kind is not None:
                    mapping_contract = dict(
                        (class_field_mapping_contracts or {}).get((
                            str(receiver_class), str(expression.attr)
                        )) or {}
                    )
                    graph.G.nodes[attribute_id].setdefault(
                        "attributes", {}
                    ).update({
                        "producer_kind": "record_field",
                        "aggregate_kind": field_kind,
                        "sequence_key_columns": (
                            (0,) if field_kind in {"set", "dict"} else ()
                        ),
                        "sequence_column_count": (
                            2 if field_kind == "dict" else 1
                        ),
                        "sequence_writable": field_kind != "tuple",
                        "record_field": (
                            str(receiver_class), str(expression.attr)
                        ),
                        **mapping_contract,
                    })
                # An ordinary attribute access on a resolved receiver is a
                # real reference-operator node -- ``_replace_inputs`` above
                # already gave it the receiver as a dependency, and
                # ``attribute_slot`` its class-grounded identity where
                # known.  Without returning that identity here, every
                # caller of ``resolve_expression`` -- including ``ast.Call``
                # resolving its own ``func`` for a method call -- silently
                # receives ``None`` instead: the operator resolved
                # correctly but never reported back that it did, severing
                # the receiver as a dependency for anything built from this
                # expression (a method call, a chained attribute, ...).
                # Observation does not assign a new field version. Preserve
                # the last RHS/merge in the ledger while keeping this read's
                # own node and effect dependency for lexical scheduling.
                attribute_value_nodes.setdefault(
                    (receiver, expression.attr), int(attribute_id)
                )
                return attribute_id

        if isinstance(expression, ast.Call):
            node_id = id(expression)
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id
                in (graph.G.graph.get("class_definitions") or ())
                and node_id in graph.G
            ):
                # A source-local class construction, already known from
                # ``map_ir`` at ingestion -- no python_bindings/static
                # reference resolution needed or wanted for this: the
                # class isn't external, it's right here in the source.
                graph.G.nodes[node_id].setdefault(
                    "attributes", {},
                )["class_ref"] = expression.func.id
            if (
                isinstance(expression.func, ast.Attribute)
                and node_id in graph.G
            ):
                # A method call's receiver is wired here, through the same
                # SSA lookup (resolve_expression) every other expression
                # already goes through -- not a raw AST-node id, which is
                # only valid by coincidence (it assumes ingestion happened
                # to walk this exact receiver independently, which is not
                # guaranteed for one reached only through this call
                # expression). The receiver is an ordinary expression --
                # commonly a Name already bound by bind_target, whose real,
                # resolved value resolve_expression looks up directly.
                receiver_value = resolve_expression(expression.func.value)
                receiver_inputs = (
                    ((receiver_value, "operand"),)
                    if isinstance(receiver_value, int)
                    else ()
                )
                argument_inputs = tuple(
                    (resolved, f"arg{index}")
                    for index, argument in enumerate(expression.args)
                    if isinstance(
                        (resolved := resolve_expression(argument)), int
                    )
                )
                keyword_inputs = tuple(
                    (resolved, f"kw:{keyword.arg}" if keyword.arg else "kwargs")
                    for keyword in expression.keywords
                    if isinstance(
                        (resolved := resolve_expression(keyword.value)), int
                    )
                )
                if receiver_inputs:
                    _replace_inputs(
                        graph,
                        node_id,
                        (*receiver_inputs, *argument_inputs, *keyword_inputs),
                    )
            callee = resolve_expression(expression.func)
            if isinstance(callee, _StaticPythonReference) and node_id in graph.G:
                identity_target = (
                    callee.value.__func__
                    if inspect.ismethod(callee.value)
                    else callee.value
                )
                identity = ".".join(filter(None, (
                    str(getattr(identity_target, "__module__", "")),
                    str(getattr(
                        identity_target,
                        "__qualname__",
                        getattr(identity_target, "__name__", ""),
                    )),
                )))
                identity_program = resolve_python_identity(identity)
                if identity_program is not None:
                    call_attributes = graph.G.nodes[node_id].setdefault(
                        "attributes", {}
                    )
                    call_attributes.update({
                        "python_identity_program": identity_program.mapping(),
                        "python_replacement_kind": identity_program.kind,
                        **identity_program.direct_attributes,
                    })
                    descriptor = (
                        str(identity),
                        str(identity_program.kind),
                        identity_program.direct_operator,
                        tuple(sorted(
                            identity_program.direct_attributes.items()
                        )),
                    )
                    identity_page = current_identity_book().page(
                        "source_python_identity_concordance"
                    )
                    identity_row = (value_class_scope, int(node_id))
                    identity_incumbent = identity_page.latest(identity_row)
                    if (
                        identity_incumbent is not None
                        and tuple(identity_incumbent) != descriptor
                    ):
                        raise ValueError(
                            "source Python identity concordance disagreement "
                            f"for {identity_row!r}: "
                            f"recorded={identity_incumbent!r}, "
                            f"resolved={descriptor!r}"
                        )
                    if identity_incumbent is None:
                        identity_page.set(identity_row, 0, descriptor)
            if (
                isinstance(callee, _StaticPythonReference)
                and any(
                    callee.value is kind
                    for kind in (tuple, list, set, dict, bytes, bytearray)
                )
                and node_id in graph.G
            ):
                constructor = graph.G.nodes[node_id]
                constructor_attributes = constructor.setdefault(
                    "attributes", {}
                )
                aggregate_kind = callee.value.__name__
                constructor_attributes.setdefault(
                    "producer_kind",
                    "aggregate" if not expression.args else (
                        "aggregate_materialization"
                    ),
                )
                constructor_attributes.update({
                    "aggregate_kind": aggregate_kind,
                    "sequence_key_columns": (
                        (0,) if aggregate_kind in {"set", "dict"} else ()
                    ),
                    "sequence_column_count": (
                        2 if aggregate_kind == "dict" else 1
                    ),
                    "sequence_writable": aggregate_kind not in {
                        "tuple", "bytes"
                    },
                })
                constructor_attributes.setdefault(
                    "aggregate_leaf_value_ids",
                    tuple(
                        int(parent)
                        for parent, role in constructor.get("parents") or ()
                        if str(role).startswith(("arg", "kw:"))
                    ),
                )
            if isinstance(callee, int) and callee in graph.G:
                callee_attributes = (
                    graph.G.nodes[callee].get("attributes") or {}
                )
                callee_reference = callee_attributes.get("function_ref")
                if callee_reference is not None and node_id in graph.G:
                    graph.G.nodes[node_id].setdefault(
                        "attributes",
                        {},
                    )["callee_ref"] = callee_reference
                method_reference = callee_attributes.get("method_ref")
                if method_reference is not None and node_id in graph.G:
                    graph.G.nodes[node_id].setdefault(
                        "attributes", {}
                    )["method_ref"] = int(method_reference)
                    call_inputs = tuple(
                        graph.G.nodes[node_id].get("parents") or ()
                    )
                    if not any(
                        parent == callee and str(role) == "callee"
                        for parent, role in call_inputs
                    ):
                        _replace_inputs(
                            graph,
                            node_id,
                            (*call_inputs, (callee, "callee")),
                        )
                if (
                    callee_attributes.get("callable_reference_field")
                    and node_id in graph.G
                ):
                    call_attributes = graph.G.nodes[node_id].setdefault(
                        "attributes", {}
                    )
                    call_attributes.update({
                        "indirect_callable_id": int(callee),
                        "callable_signature": str(
                            callee_attributes["callable_signature"]
                        ),
                        "callable_optional": bool(
                            callee_attributes.get("callable_optional", False)
                        ),
                    })
                    call_inputs = tuple(
                        graph.G.nodes[node_id].get("parents") or ()
                    )
                    if not any(
                        int(parent) == int(callee)
                        and str(role) in {"callee", "function", "func"}
                        for parent, role in call_inputs
                    ):
                        _replace_inputs(
                            graph,
                            node_id,
                            (*call_inputs, (int(callee), "callee")),
                        )
            if isinstance(callee, _StaticPythonReference) and node_id in graph.G:
                reference_node_id = static_reference_node(callee)
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes",
                    {},
                )
                attributes["static_python_reference"] = callee.path
                reference_attributes = graph.G.nodes[
                    reference_node_id
                ].get("attributes") or {}
                if "function_ref" in reference_attributes:
                    attributes["callee_ref"] = reference_attributes[
                        "function_ref"
                    ]
                if "class_ref" in reference_attributes:
                    attributes["class_ref"] = reference_attributes[
                        "class_ref"
                    ]
                if str(graph.G.nodes[node_id].get("type")) == "Call":
                    if not graph.G.has_edge(reference_node_id, node_id):
                        graph.G.add_edge(
                            reference_node_id,
                            node_id,
                            role="callee",
                        )
                        graph.G.nodes[reference_node_id].setdefault(
                            "children",
                            [],
                        ).append((node_id, "callee"))
                    parents = graph.G.nodes[node_id].setdefault(
                        "parents",
                        [],
                    )
                    if (reference_node_id, "callee") not in parents:
                        parents.append((reference_node_id, "callee"))
                else:
                    # A canonical numerical operation already identifies its
                    # implementation by node type.  Preserve the wrapper as
                    # compiler metadata, but never feed it to the operation as
                    # tensor data.
                    attributes["operator_reference_node"] = reference_node_id
            # Every call's arguments need ``resolve_expression`` -- the same
            # reduction/redirection every other expression gets -- regardless
            # of what kind of callee this call has.  This used to run only
            # inside the ``_StaticPythonReference`` branch above, so an
            # ordinary method call (``pending.pop(0)``, a call through a
            # resolved-but-not-static receiver) never had its arguments
            # resolved at all during reduction: a literal like ``0`` was
            # left exactly as ingestion produced it, unreachable by anything
            # that expects reduction to have run -- the same "not translated"
            # shape as any other missing operand.
            if node_id in graph.G:
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes", {}
                )
                static_arguments = dict(
                    attributes.get("static_call_arguments") or {}
                )
                for index, argument in enumerate(expression.args):
                    resolved = resolve_expression(argument)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"arg:{index}"] = resolved.path
                        attributes.setdefault(
                            "static_call_values", {}
                        )[f"arg:{index}"] = resolved.value
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[resolved.path] = resolved.value
                    elif (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        role = f"arg:{index}"
                        if not graph.G.has_edge(resolved, node_id):
                            graph.G.add_edge(resolved, node_id, role=role)
                        children = graph.G.nodes[resolved].setdefault(
                            "children", []
                        )
                        if (node_id, role) not in children:
                            children.append((node_id, role))
                        parents = graph.G.nodes[node_id].setdefault(
                            "parents", []
                        )
                        if (resolved, role) not in parents:
                            parents.append((resolved, role))
                for keyword in expression.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = resolve_expression(keyword.value)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"kw:{keyword.arg}"] = resolved.path
                        attributes.setdefault(
                            "static_call_values", {}
                        )[f"kw:{keyword.arg}"] = resolved.value
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[resolved.path] = resolved.value
                    elif (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        role = f"kw:{keyword.arg}"
                        if not graph.G.has_edge(resolved, node_id):
                            graph.G.add_edge(resolved, node_id, role=role)
                        children = graph.G.nodes[resolved].setdefault(
                            "children", []
                        )
                        if (node_id, role) not in children:
                            children.append((node_id, role))
                        parents = graph.G.nodes[node_id].setdefault(
                            "parents", []
                        )
                        if (resolved, role) not in parents:
                            parents.append((resolved, role))
                if static_arguments:
                    attributes["static_call_arguments"] = static_arguments

            # A source-defined function may itself be a positional/keyword
            # value (callbacks, policies, hooks).  Its original Name node was
            # intentionally removed, so reconnect the compiler-owned static
            # reference to the call with the argument's real role.
            if node_id in graph.G:
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes", {}
                )
                static_arguments = dict(
                    attributes.get("static_call_arguments") or {}
                )
                for role, argument in (
                    *tuple(
                        (f"arg:{index}", argument)
                        for index, argument in enumerate(expression.args)
                    ),
                    *tuple(
                        (f"kw:{keyword.arg}", keyword.value)
                        for keyword in expression.keywords
                        if keyword.arg is not None
                    ),
                ):
                    if (
                        isinstance(argument, ast.Name)
                        and isinstance(
                            getattr(builtins, argument.id, None), type
                        )
                    ):
                        static_arguments[role] = argument.id
                        attributes.setdefault(
                            "static_call_values", {}
                        )[role] = getattr(builtins, argument.id)
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[argument.id] = getattr(builtins, argument.id)
                if static_arguments:
                    attributes["static_call_arguments"] = static_arguments
                for role, argument in (
                    *tuple(
                        (f"arg:{index}", argument)
                        for index, argument in enumerate(expression.args)
                    ),
                    *tuple(
                        (f"kw:{keyword.arg}", keyword.value)
                        for keyword in expression.keywords
                        if keyword.arg is not None
                    ),
                ):
                    resolved = resolve_expression(argument)
                    if not (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        continue
                    if not graph.G.has_edge(resolved, node_id):
                        graph.G.add_edge(resolved, node_id, role=role)
                    children = graph.G.nodes[resolved].setdefault(
                        "children", []
                    )
                    if (node_id, role) not in children:
                        children.append((node_id, role))
                    parents = graph.G.nodes[node_id].setdefault(
                        "parents", []
                    )
                    if (resolved, role) not in parents:
                        parents.append((resolved, role))

            # Rebuild every call from the values resolved at this lexical
            # program point.  Ingestion may omit a repeated Attribute/Name
            # occurrence even though the enclosing Call is retained; keeping
            # its earlier raw-AST-id inputs then leaves a real argument behind
            # an Untranslated placeholder.  The lexical reducer already owns
            # the authoritative environment, so use those resolved values for
            # ordinary arguments as well as first-class-function arguments.
            if node_id in graph.G:
                call_data = graph.G.nodes[node_id]
                call_attributes = call_data.get("attributes") or {}
                resolved_inputs: list[tuple[int, str]] = []
                if isinstance(expression.func, ast.Attribute):
                    receiver = resolve_expression(expression.func.value)
                    if isinstance(receiver, int):
                        resolved_inputs.append((receiver, "operand"))
                    if (
                        isinstance(callee, int)
                        and callee in graph.G
                        and (graph.G.nodes[callee].get("attributes") or {}).get(
                            "method_ref"
                        ) is not None
                    ):
                        resolved_inputs.append((callee, "callee"))
                elif not (
                    "callee_ref" in call_attributes
                    or "external_callee_ref" in call_attributes
                ):
                    if isinstance(callee, int):
                        resolved_inputs.append((callee, "callee"))
                    elif isinstance(callee, _StaticPythonReference):
                        resolved_inputs.append((
                            static_reference_node(callee),
                            "callee",
                        ))
                for index, argument in enumerate(expression.args):
                    resolved = resolve_expression(argument)
                    if isinstance(resolved, int):
                        resolved_inputs.append((resolved, f"arg:{index}"))
                for keyword in expression.keywords:
                    resolved = resolve_expression(keyword.value)
                    if isinstance(resolved, int):
                        resolved_inputs.append((
                            resolved,
                            f"kw:{keyword.arg}" if keyword.arg else "kwargs",
                        ))
                callee_reference = call_attributes.get(
                    "callee_ref", call_attributes.get("method_ref")
                )
                for closure_name in lexical_free_names_by_reference.get(
                    int(callee_reference)
                    if callee_reference is not None
                    else -1,
                    (),
                ):
                    closure_value = environment.get(closure_name)
                    if (
                        closure_value is None
                        and closure_name in parameter_names
                    ):
                        closure_value = input_value(
                            closure_name, binding_kind="parameter",
                        )
                    if isinstance(closure_value, int):
                        resolved_inputs.append((
                            closure_value, f"closure:{closure_name}",
                        ))
                _replace_inputs(graph, node_id, tuple(resolved_inputs))
                result_class = call_attributes.get(
                    "result_class_ref", call_attributes.get("class_ref")
                )
                if result_class is not None:
                    # A width is recorded only where the source states it;
                    # a Precision-bearing result whose width is computed is
                    # stated by its callsite specialization, not guessed here.
                    carries_width = "precision" in (_NUMERIC_TYPE_FEATURES.get(
                        str(result_class).rsplit(".", 1)[-1]
                    ) or ())
                    precision_limbs: int | None = (
                        int(call_attributes["precision_limbs"])
                        if call_attributes.get("precision_limbs")
                        else None if carries_width else 1
                    )
                    if (
                        str(result_class).rsplit(".", 1)[-1]
                        in {"Precision", "ComplexPrecision"}
                        and isinstance(expression.func, ast.Attribute)
                        and expression.func.attr == "of"
                    ):
                        precision_limbs = 2
                        if len(expression.args) >= 2:
                            try:
                                precision_limbs = int(ast.literal_eval(
                                    expression.args[1]
                                ))
                            except (ValueError, TypeError):
                                precision_limbs = None
                        if precision_limbs is not None:
                            call_attributes["precision_limbs"] = max(
                                precision_limbs, 1
                            )
                if result_class is not None and precision_limbs is not None:
                    _concord_source_value_class(
                        value_class_scope,
                        node_id,
                        str(result_class),
                        precision_limbs=precision_limbs,
                        source="resolved call result",
                    )
                if (
                    isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == "setdefault"
                    and len(expression.args) >= 2
                ):
                    default_id = resolve_expression(expression.args[1])
                    default_attributes = (
                        graph.G.nodes[default_id].get("attributes") or {}
                        if isinstance(default_id, int) and default_id in graph.G
                        else {}
                    )
                    default_kind = default_attributes.get("aggregate_kind")
                    if default_kind is not None:
                        call_attributes.update({
                            "producer_kind": "mapping_default_result",
                            "aggregate_kind": default_kind,
                            "sequence_key_columns": tuple(
                                default_attributes.get(
                                    "sequence_key_columns", ()
                                )
                            ),
                            "sequence_column_count": int(
                                default_attributes.get(
                                    "sequence_column_count", 1
                                )
                            ),
                            "sequence_writable": bool(
                                default_attributes.get(
                                    "sequence_writable", True
                                )
                            ),
                        })
                        receiver_id = resolve_expression(
                            expression.func.value
                        )
                        if isinstance(receiver_id, int) and receiver_id in graph.G:
                            receiver_contract = graph.G.nodes[
                                receiver_id
                            ].setdefault("attributes", {})
                            receiver_contract[
                                "mapping_value_aggregate_kind"
                            ] = default_kind
                            key_id = resolve_expression(expression.args[0])
                            key_attributes = (
                                graph.G.nodes[key_id].get("attributes") or {}
                                if isinstance(key_id, int) and key_id in graph.G
                                else {}
                            )
                            key_leaves = tuple(
                                key_attributes.get(
                                    "aggregate_leaf_value_ids", ()
                                )
                            )
                            receiver_contract[
                                "mapping_key_column_count"
                            ] = max(1, len(key_leaves))

        # A named callee already represented by a function-table reference is
        # not a runtime value.  Its arguments still are.
        children = tuple(source_child_nodes(expression))
        if isinstance(expression, ast.Call):
            call_data = (
                graph.G.nodes[id(expression)]
                if id(expression) in graph.G
                else {}
            )
            call_attributes = call_data.get("attributes") or {}
            if (
                isinstance(expression.func, ast.Name)
                and (
                    "callee_ref" in call_attributes
                    or "external_callee_ref" in call_attributes
                )
            ):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )
            elif isinstance(callee, _StaticPythonReference):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )
            elif (
                isinstance(callee, int)
                and callee in graph.G
                and (
                    graph.G.nodes[callee].get("attributes") or {}
                ).get("function_ref") is not None
            ):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )

        # Resolve children first so the existing executable node receives
        # producer IDs rather than lexical occurrence IDs.
        for child in children:
            if isinstance(child, ast.expr_context):
                continue
            resolve_expression(child)
        node_id = id(expression)
        if (
            isinstance(expression, (ast.Tuple, ast.List, ast.Set, ast.Dict))
            and node_id in graph.G
        ):
            aggregate = graph.G.nodes[node_id]
            aggregate_attributes = aggregate.setdefault("attributes", {})
            aggregate_attributes["producer_kind"] = "aggregate"
            aggregate_kind = {
                ast.Tuple: "tuple",
                ast.List: "list",
                ast.Set: "set",
                ast.Dict: "dict",
            }[type(expression)]
            aggregate_attributes.update({
                "aggregate_kind": aggregate_kind,
                "sequence_key_columns": (
                    (0,) if aggregate_kind in {"set", "dict"} else ()
                ),
                "sequence_column_count": (
                    2 if aggregate_kind == "dict" else 1
                ),
                "sequence_writable": aggregate_kind != "tuple",
            })
            aggregate_attributes["aggregate_leaf_value_ids"] = tuple(
                int(parent)
                for parent, role in aggregate.get("parents") or ()
                if str(role) in {
                    "elts",
                    "elt",
                    "element",
                    "item",
                    "keys",
                    "values",
                    "key",
                    "value",
                }
            )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"tuple", "list"}
            and len(expression.args) == 1
            and node_id in graph.G
        ):
            materializer = graph.G.nodes[node_id]
            materializer_attributes = materializer.setdefault(
                "attributes",
                {},
            )
            materializer_attributes["producer_kind"] = (
                "loop_materialization"
                if isinstance(expression.args[0], ast.GeneratorExp)
                else "aggregate_materialization"
            )
            materializer_attributes["materialization_axis"] = 0
            materializer_attributes["materialized_source_value_ids"] = tuple(
                int(parent)
                for parent, role in materializer.get("parents") or ()
                if str(role).startswith("arg:")
            )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"tuple", "list"}
            and len(expression.args) == 1
            and isinstance(expression.args[0], ast.GeneratorExp)
            and node_id in graph.G
        ):
            # This re-resolves generator_expression.elt, which the
            # (ListComp, SetComp, DictComp, GeneratorExp) branch above
            # already resolved once, inside its own scoped push/pop of the
            # generator's `for` target names. resolve_expression() is not
            # memoized for Name lookups -- it re-reads `environment` on
            # every call -- so this second, unscoped pass through the same
            # `elt` subtree re-looks-up the target name(s) (e.g. `address`
            # in `tuple(f(address) for address in xs)`) after that first
            # pass already popped them, silently creating a fresh
            # "external input" binding for what should be, and already
            # was, a properly loop-scoped local. That stray rebinding then
            # leaked into whatever unrelated code came later in the
            # function and happened to reuse the same name. Same
            # push/pop discipline as above, applied to this second walk.
            generator_expression = expression.args[0]
            comprehension_target_names = {
                name
                for generator in generator_expression.generators
                for name in loop_target_names(generator.target)
            }
            shadowed_bindings = {
                name: environment[name]
                for name in comprehension_target_names
                if name in environment
            }
            for generator in generator_expression.generators:
                bind_loop_target(generator.target)
            value_id = resolve_expression(generator_expression.elt)
            for generator in generator_expression.generators:
                generator_id = id(generator)
                if (
                    generator_id not in graph.G
                    or not isinstance(value_id, int)
                    or value_id not in graph.G
                ):
                    continue
                attributes = graph.G.nodes[generator_id].setdefault(
                    "attributes",
                    {},
                )
                outputs = list(
                    attributes.get("loop_iteration_outputs", ())
                )
                outputs.append({
                    "value_id": value_id,
                    "result_value_id": node_id,
                    "materializer_node_id": node_id,
                })
                attributes["loop_iteration_outputs"] = tuple(outputs)
            for name in comprehension_target_names:
                if name in shadowed_bindings:
                    environment[name] = shadowed_bindings[name]
                else:
                    environment.pop(name, None)
        return node_id if node_id in graph.G else None

    def bind_target(
        target: ast.AST,
        value: int | _StaticPythonReference | None,
    ) -> None:
        if value is None:
            return
        if isinstance(target, ast.Name):
            deleted_names.discard(target.id)
            if isinstance(value, _StaticPythonReference):
                environment.pop(target.id, None)
                static_environment[target.id] = value
                _remove_node(graph, id(target))
                return
            static_environment.pop(target.id, None)
            environment[target.id] = value
            identity_bindings.setdefault(target.id, []).append(value)
            record_ingestion_definition(target.id, value, target)
            _remove_node(graph, id(target))
            return
        if isinstance(target, ast.Attribute):
            static_value_path = None
            if isinstance(value, _StaticPythonReference):
                # Retain the Python/runtime seam explicitly. Autograd tape and
                # similar bookkeeping references are not numerical values, but
                # assigning one must not truncate the surrounding mathematical
                # program. StaticReference is the planner's established host
                # boundary node and keeps the effect visible in full geometry.
                static_value_path = value.path
                value = static_reference_node(value)
            receiver = resolve_expression(target.value)
            static_receiver = (
                receiver if isinstance(receiver, _StaticPythonReference) else None
            )
            if static_receiver is not None:
                receiver = static_reference_node(static_receiver)
            if not isinstance(receiver, int) or not isinstance(value, int):
                raise TypeError(
                    "attribute assignment requires resolved object and value "
                    f"nodes in {statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}, "
                    f"receiver={receiver!r}, value={value!r}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                # Resolving a compile-time receiver's literal attribute read
                # redirects and removes the source Attribute wrapper.  The
                # subsequent write is still a distinct runtime effect.
                node_id = new_node(
                    "SetAttr",
                    f"setattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                    # The write's own statement anchors it lexically: an
                    # unanchored SetAttr inside a loop body was not owned by
                    # the loop (`is_lexical_body_node`), so port
                    # materialization rewired the loop's own effect chain
                    # onto its own result port and closed a cycle.
                    source=target,
                )
            # In ``object.field += value`` the Attribute node is the read
            # feeding the AugAssign result.  Reusing that same node as the
            # write would create ``Attribute -> AugAssign -> SetAttr`` while
            # SetAttr is still the Attribute node: an artificial dataflow
            # cycle.  Keep the read and write as distinct program events.
            if node_id == value or nx.has_path(graph.G, node_id, value):
                node_id = new_node(
                    "SetAttr",
                    f"setattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                    # The write's own statement anchors it lexically: an
                    # unanchored SetAttr inside a loop body was not owned by
                    # the loop (`is_lexical_body_node`), so port
                    # materialization rewired the loop's own effect chain
                    # onto its own result port and closed a cycle.
                    source=target,
                )
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "SetAttr"
            node_data["op"] = "setattr"
            node_data.setdefault("attributes", {})["attribute"] = target.attr
            if static_value_path is not None:
                node_data["attributes"]["static_value_boundary"] = (
                    static_value_path
                )
            _replace_inputs(
                graph,
                node_id,
                (
                    (receiver, "object"),
                    (value, "value"),
                ),
            )
            if static_receiver is not None:
                static_attribute_values[
                    (id(static_receiver.value), target.attr)
                ] = value
            attribute_effect_nodes[(receiver, target.attr)] = node_id
            attribute_value_nodes[(receiver, target.attr)] = int(value)
            # A plain-named receiver (``counter.value = ...``) gets its own
            # identity binding the same way a bare ``ast.Name`` target does
            # above -- the field write is already a real, correctly wired
            # SetAttr node; it was just never named, so it never reached
            # ``identity_table`` for anything downstream to select by name.
            if isinstance(target.value, ast.Name):
                field_identity = f"{target.value.id}.{target.attr}"
                identity_bindings.setdefault(
                    field_identity, []
                ).append(node_id)
                record_ingestion_definition(field_identity, node_id, target)
                # ``callee_ref``/``method_ref``/``class_ref`` each let a
                # dependency walk recurse into the thing a node depends on --
                # a real node reference grounded in the class's own layout,
                # not a label invented at this call site.  ``attribute_slot``
                # is the field's actual, deterministic position in its
                # class's declared instance storage, resolved through
                # ``ClassNavigationTable.resolve_dot`` (see
                # ``navigation_table``/``_resolve_instance_attribute_slot``
                # above), known only when the receiver's static class
                # identity is known -- an annotated parameter naming a
                # locally-defined class.  Nothing is invented when it is not
                # known; the write remains correctly wired via
                # ``object``/``value`` either way.
                class_identity = parameter_class_names.get(target.value.id)
                if class_identity is not None:
                    slot = _resolve_instance_attribute_slot(
                        class_identity, target.attr
                    )
                    if slot is not None:
                        node_data.setdefault("attributes", {})[
                            "attribute_slot"
                        ] = (class_identity, slot)
            return
        if isinstance(target, ast.Subscript):
            if isinstance(value, _StaticPythonReference):
                try:
                    target_source = ast.unparse(target)
                except Exception:  # noqa: BLE001 -- diagnostic only
                    target_source = repr(target)
                raise TypeError(
                    "a static Python reference cannot be assigned through "
                    "a runtime tensor index; "
                    f"target={target_source!r}; reference={value.path!r}"
                )
            base = resolve_expression(target.value)
            index_expressions = (
                tuple(target.slice.elts)
                if isinstance(target.slice, ast.Tuple)
                else (target.slice,)
            )
            indices = tuple(
                resolve_expression(index) for index in index_expressions
            )
            if not isinstance(base, int) or not isinstance(value, int) or not all(
                isinstance(index, int) for index in indices
            ):
                try:
                    assignment_source = ast.unparse(
                        ast.Assign(targets=[target], value=ast.Constant(None))
                    ).removesuffix(" = None")
                except Exception:  # noqa: BLE001 -- diagnostic only
                    assignment_source = ast.dump(
                        target, include_attributes=False
                    )
                def component_identity(component: Any) -> str:
                    if isinstance(component, _StaticPythonReference):
                        return f"static:{component.path}"
                    if not isinstance(component, int):
                        return repr(component)
                    if component not in graph.G:
                        return f"node:{component}:absent"
                    data = graph.G.nodes[component]
                    attributes = data.get("attributes") or {}
                    binding = attributes.get("binding_name")
                    suffix = f":binding={binding}" if binding else ""
                    return (
                        f"node:{component}:{data.get('type')}:"
                        f"{data.get('label')}{suffix}"
                    )
                raise TypeError(
                    "indexed assignment requires resolved tensor, index, "
                    f"and value nodes in {statement.name}: "
                    f"target={assignment_source!r}; "
                    f"base={component_identity(base)}; "
                    "indices=("
                    + ", ".join(component_identity(index) for index in indices)
                    + "); "
                    f"value={component_identity(value)}"
                )
            node_id = id(target)
            if (
                node_id not in graph.G
                or node_id == base
                or node_id == value
                or nx.has_path(graph.G, node_id, value)
            ):
                node_id = new_node("IndexedStore", "indexed_store")
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "IndexedStore"
            node_data["op"] = "IndexedStore"
            node_data.setdefault("attributes", {})[
                "source_type"
            ] = "SubscriptStore"
            _replace_inputs(
                graph,
                node_id,
                (
                    (base, "base"),
                    *((index, "index") for index in indices),
                    (value, "value"),
                ),
            )
            if isinstance(target.value, ast.Name):
                bind_target(target.value, node_id)
            elif isinstance(target.value, ast.Attribute) and isinstance(
                target.value.value, ast.Name
            ):
                # ``obj.field[i, j] = ...`` mutates ``obj.field`` in place
                # without rebinding ``obj`` itself, so the ``ast.Name`` path
                # above never fires here. Record the mutation the same way a
                # whole-attribute ``SetAttr`` does, keyed by the resolved
                # receiver of ``obj`` (not ``base``, which is the GetAttr
                # node for ``obj.field`` and does not change), so a later
                # bare read of ``obj.field`` can depend on this write.
                outer_receiver = resolve_expression(target.value.value)
                if isinstance(outer_receiver, int):
                    attribute_effect_nodes[
                        (outer_receiver, target.value.attr)
                    ] = node_id
                    attribute_value_nodes[
                        (outer_receiver, target.value.attr)
                    ] = node_id
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, _StaticPythonReference):
                raise TypeError(
                    "a static Python reference cannot be destructured as "
                    "a runtime graph value"
                )
            starred = tuple(
                index
                for index, element in enumerate(target.elts)
                if isinstance(element, ast.Starred)
            )
            if len(starred) > 1:
                raise ValueError("destructuring permits at most one starred target")
            starred_index = starred[0] if starred else None
            trailing = (
                len(target.elts) - starred_index - 1
                if starred_index is not None else 0
            )
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    projection = slice(
                        index,
                        -trailing if trailing else None,
                    )
                    binding_target = element.value
                else:
                    projection = (
                        index - len(target.elts)
                        if starred_index is not None and index > starred_index
                        else index
                    )
                    binding_target = element
                index_id = new_node(
                    "Constant",
                    repr(projection),
                    attributes={"value": projection},
                )
                projected_id = new_node(
                    "Indexed",
                    f"unpack[{projection!r}]",
                    parents=(
                        (value, "base"),
                        (index_id, "index"),
                    ),
                )
                bind_target(binding_target, projected_id)

    def delete_target(target: ast.AST) -> int | None:
        """Lower one Python deletion target to its explicit state effect."""

        if isinstance(target, ast.Name):
            environment.pop(target.id, None)
            static_environment.pop(target.id, None)
            deleted_names.add(target.id)
            _remove_node(graph, id(target))
            return None
        if isinstance(target, ast.Attribute):
            receiver = resolve_expression(target.value)
            if isinstance(receiver, _StaticPythonReference):
                receiver = static_reference_node(receiver)
            if not isinstance(receiver, int):
                raise TypeError(
                    "attribute deletion requires a resolved object node in "
                    f"{statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}, "
                    f"receiver={receiver!r}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                node_id = new_node(
                    "DelAttr",
                    f"delattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                )
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "DelAttr"
            node_data["op"] = "delattr"
            node_data.setdefault("attributes", {})["attribute"] = target.attr
            _replace_inputs(graph, node_id, ((receiver, "object"),))
            return node_id
        if isinstance(target, ast.Subscript):
            base = resolve_expression(target.value)
            indices = (
                tuple(target.slice.elts)
                if isinstance(target.slice, ast.Tuple)
                else (target.slice,)
            )
            resolved_indices = tuple(resolve_expression(index) for index in indices)
            if not isinstance(base, int) or not all(
                isinstance(index, int) for index in resolved_indices
            ):
                raise TypeError(
                    "indexed deletion requires resolved container and index "
                    f"nodes in {statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                node_id = new_node("DelItem", "delitem")
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "DelItem"
            node_data["op"] = "delitem"
            node_data.setdefault("attributes", {})["source_type"] = "Subscript"
            _replace_inputs(
                graph,
                node_id,
                (
                    (base, "base"),
                    *((index, "index") for index in resolved_indices),
                ),
            )
            return node_id
        if isinstance(target, (ast.Tuple, ast.List)):
            effects = [delete_target(element) for element in target.elts]
            return next((effect for effect in reversed(effects) if effect is not None), None)
        raise TypeError(
            "unsupported Python deletion target: "
            f"{ast.dump(target, include_attributes=False)}"
        )

    def reduce_statement(body_statement: ast.stmt) -> int | None:
        if isinstance(body_statement, (ast.Nonlocal, ast.Global)):
            # Scope declarations affect name binding while parsing; they are
            # not runtime operations and must never reach an execution
            # backend as operator nodes.  The reducer's lexical environments
            # and explicit Input/SetAttr effects carry the resulting values.
            _remove_node(graph, id(body_statement))
            return None
        if isinstance(body_statement, ast.Delete):
            effects = [delete_target(target) for target in body_statement.targets]
            _remove_node(graph, id(body_statement))
            return next(
                (effect for effect in reversed(effects) if effect is not None),
                None,
            )
        if isinstance(body_statement, (ast.Assign, ast.AnnAssign)):
            value = resolve_expression(body_statement.value)
            targets = (
                tuple(body_statement.targets)
                if isinstance(body_statement, ast.Assign)
                else (body_statement.target,)
            )
            for target in targets:
                bind_target(target, value)
            _remove_node(graph, id(body_statement))
            return value
        if isinstance(body_statement, ast.AugAssign):
            arena_mask = None
            if isinstance(body_statement.target, ast.Name):
                current = environment.get(body_statement.target.id)
                if current is None:
                    current = input_value(
                        body_statement.target.id,
                        binding_kind=(
                            "parameter"
                            if body_statement.target.id in parameter_names
                            else "exception"
                            if body_statement.target.id in exception_local_names
                            else "external"
                        ),
                    )
            elif (
                isinstance(body_statement.target, ast.Subscript)
                and not all(
                    isinstance(item, ast.Slice)
                    or (
                        isinstance(item, ast.Constant)
                        and (
                            isinstance(item.value, int)
                            or item.value is Ellipsis
                        )
                    )
                    or (
                        isinstance(item, ast.Name)
                        and environment.get(item.id)
                        in scalar_loop_binding_ids
                    )
                    for item in (
                        tuple(body_statement.target.slice.elts)
                        if isinstance(body_statement.target.slice, ast.Tuple)
                        else (body_statement.target.slice,)
                    )
                )
            ):
                # Keep aligned boolean-mask updates arena-shaped.  Evaluating
                # ``field[mask] += rhs[mask]`` literally creates two compact,
                # data-dependent vectors and then requires a scatter.  The
                # equivalent functional form
                # ``field = where(mask, field + rhs, field)`` retains the
                # preallocated grid shape through every compiler stage.
                current = resolve_expression(body_statement.target.value)
                arena_mask = resolve_expression(
                    body_statement.target.slice
                )
                target_slice = ast.dump(
                    body_statement.target.slice,
                    include_attributes=False,
                )
                for candidate in source_walk(body_statement.value):
                    if not isinstance(candidate, ast.Subscript):
                        continue
                    if ast.dump(
                        candidate.slice,
                        include_attributes=False,
                    ) != target_slice:
                        continue
                    aligned = resolve_expression(candidate.value)
                    if isinstance(aligned, int):
                        _redirect_value(graph, id(candidate), aligned)
            else:
                current = resolve_expression(body_statement.target)
            if current is not None:
                if isinstance(body_statement.target, ast.Name):
                    # ``name += x`` reads ``name`` without a Load occurrence.
                    _concord_lexical_reads(
                        graph, ingestion_read_scope,
                        id(body_statement.target), body_statement.target.id,
                    )
                _redirect_value(
                    graph,
                    id(body_statement.target),
                    current,
                )
            resolve_expression(body_statement.value)
            node_id = id(body_statement)
            if current is None or node_id not in graph.G:
                return None
            if isinstance(arena_mask, int):
                where_id = new_node(
                    "where",
                    "where",
                    attributes={
                        "source_type": "MaskedAugAssign",
                        "arena_shaped": True,
                    },
                    parents=(
                        (arena_mask, "condition"),
                        (node_id, "true"),
                        (current, "false"),
                    ),
                )
                bind_target(body_statement.target.value, where_id)
                return where_id
            bind_target(body_statement.target, node_id)
            return node_id
        if isinstance(body_statement, ast.Return):
            returned = body_statement.value
            expressions = (
                tuple(returned.elts)
                if isinstance(returned, (ast.Tuple, ast.List))
                else (returned,)
            )
            output_names = tuple(
                graph.G.graph.get("function_outputs", ())
            )
            positional_names = set(
                graph.G.graph.get("positional_output_names", ())
            )
            resolved = []
            slot_values: list[int | None] = []
            for index, expression in enumerate(expressions):
                value = resolve_expression(expression)
                slot_values.append(value if isinstance(value, int) else None)
                if value is None:
                    continue
                resolved.append(value)
                if index < len(output_names) and (
                    str(output_names[index]) in positional_names
                ):
                    # A positional slot has no binding of its own; its
                    # history is the list of return-site values (each site
                    # is also recorded exactly in `return_slot_values`). A
                    # name-spelled slot IS a variable whose own binding
                    # already holds this value -- appending it again would
                    # only duplicate the entry and make the return site look
                    # like a rebinding to every history consumer.
                    identity_bindings.setdefault(
                        str(output_names[index]), []
                    ).append(value)
            # Every return statement names its OWN value per output slot.
            # The identity history above is a single chain per slot, so any
            # consumer reading ``history[-1]`` sees only the last return;
            # a return nested in a loop or conditional must instead branch to
            # the function exit carrying these exact values, where they are
            # merged per slot (control-aware result merging, see
            # docs/PLAN_CONTROL_AWARE_RESULT_MERGING.md). Key by source span:
            # AST object ids do not survive reduction, spans do.
            if returned is not None and getattr(returned, "lineno", None) is not None:
                return_span = (
                    int(returned.lineno),
                    int(getattr(returned, "col_offset", -1)),
                    int(getattr(returned, "end_lineno", -1)),
                    int(getattr(returned, "end_col_offset", -1)),
                )
                graph.G.graph.setdefault("return_slot_values", {})[
                    return_span
                ] = tuple(slot_values)
                # Record correlation alone does not describe the field state
                # at this return. Keep the exact receiver/value identities;
                # the linker may use only values physically available on the
                # corresponding return edge, never the final global ledger.
                graph.G.graph.setdefault("return_record_field_states", {})[
                    return_span
                ] = tuple(
                    (int(receiver), str(field), int(value))
                    for (receiver, field), value in attribute_value_nodes.items()
                    if receiver in slot_values
                )
                # Physical output slots alone cannot distinguish `return x`
                # from `return (x,)`. Preserve authored container semantics
                # beside the same exact return-site receipt before flattening.
                graph.G.graph.setdefault("return_container_kinds", {})[
                    return_span
                ] = (
                    "tuple" if isinstance(returned, ast.Tuple)
                    else "list" if isinstance(returned, ast.List)
                    else "value"
                )
            if len(expressions) == 1:
                value = resolved[0] if resolved else None
                if value is not None:
                    graph.G.graph.setdefault(
                        "return_value_nodes",
                        {},
                    )[id(body_statement)] = value
                    if isinstance(expressions[0], ast.Name):
                        return_root_bindings.setdefault(
                            int(value), set()
                        ).add(str(expressions[0].id))
                return value
            # Preserve the structural tuple/list node for callers that consume
            # it as one Python-shaped value while the output identities above
            # expose each semantic result directly to compiled call binding.
            value = resolve_expression(returned)
            if value is not None:
                graph.G.graph.setdefault(
                    "return_value_nodes",
                    {},
                )[id(body_statement)] = value
            return value
        if isinstance(body_statement, (ast.With, ast.AsyncWith)):
            static_contexts = []
            for item in body_statement.items:
                context_value = resolve_expression(item.context_expr)
                if context_value is not None and context_value in graph.G:
                    context_data = graph.G.nodes[context_value]
                    reference = (
                        context_data.get("attributes") or {}
                    ).get("static_python_reference")
                    if reference == "autograd.no_grad":
                        static_contexts.append(
                            {
                                "reference": reference,
                                "effect": "disable_backward_recording",
                                "lineno": getattr(
                                    item.context_expr,
                                    "lineno",
                                    None,
                                ),
                                "end_lineno": getattr(
                                    item.context_expr,
                                    "end_lineno",
                                    None,
                                ),
                            }
                        )
                        _remove_node(graph, context_value)
                resolve_expression(item.optional_vars)
            result = None
            for nested in body_statement.body:
                result = reduce_statement(nested)
            if static_contexts:
                recorded = list(
                    graph.G.graph.get("static_contexts", ())
                )
                recorded.extend(static_contexts)
                graph.G.graph["static_contexts"] = tuple(recorded)
                for item in body_statement.items:
                    _remove_node(graph, id(item))
                _remove_node(graph, id(body_statement))
            return result
        if isinstance(body_statement, ast.If):
            # Python parameters exist from function entry.  A parameter first
            # read inside one arm is minted there and dropped with that arm's
            # environment, so the next read mints a SECOND Input for the same
            # parameter: two formals for one authored argument, and a returned
            # in/out parameter whose storage the caller can no longer bind.
            # Mint every parameter this statement reads BEFORE the snapshot,
            # the rule loops already apply; ``input_value`` is idempotent.
            for read_name in {
                member.id
                for member in source_walk(body_statement)
                if isinstance(member, ast.Name)
                and isinstance(member.ctx, ast.Load)
            }:
                if (
                    read_name in parameter_names
                    and read_name not in environment
                    and read_name not in static_environment
                ):
                    input_value(read_name, binding_kind="parameter")
            test_value = resolve_expression(body_statement.test)
            # Control-flow value merging remains a planner responsibility.
            # Reduce lexical occurrences within each arm without pretending
            # that either arm executed unconditionally.
            before = dict(environment)
            before_attribute_effects = dict(attribute_effect_nodes)
            before_attribute_values = dict(attribute_value_nodes)
            body_environment = dict(before)
            environment.clear()
            environment.update(body_environment)
            attribute_effect_nodes.clear()
            attribute_effect_nodes.update(before_attribute_effects)
            attribute_value_nodes.clear()
            attribute_value_nodes.update(before_attribute_values)
            body_result = None
            for nested in body_statement.body:
                body_result = reduce_statement(nested)
            body_environment = dict(environment)
            body_attribute_effects = dict(attribute_effect_nodes)
            body_attribute_values = dict(attribute_value_nodes)
            environment.clear()
            environment.update(before)
            attribute_effect_nodes.clear()
            attribute_effect_nodes.update(before_attribute_effects)
            attribute_value_nodes.clear()
            attribute_value_nodes.update(before_attribute_values)
            else_result = None
            for nested in body_statement.orelse:
                else_result = reduce_statement(nested)
            else_environment = dict(environment)
            else_attribute_effects = dict(attribute_effect_nodes)
            else_attribute_values = dict(attribute_value_nodes)
            environment.clear()
            environment.update(before)
            attribute_effect_nodes.clear()
            attribute_effect_nodes.update(before_attribute_effects)
            attribute_value_nodes.clear()
            attribute_value_nodes.update(before_attribute_values)

            def terminal_branch(statements: list[ast.stmt]) -> bool:
                if not statements:
                    return False
                terminal = statements[-1]
                if isinstance(
                    terminal,
                    (ast.Return, ast.Raise, ast.Continue, ast.Break),
                ):
                    return True
                if not isinstance(terminal, ast.If):
                    return False
                return (
                    bool(terminal.orelse)
                    and terminal_branch(terminal.body)
                    and terminal_branch(terminal.orelse)
                ) or (
                    id(terminal) in graph.G
                    and bool(
                        (graph.G.nodes[id(terminal)].get("attributes") or {}).get(
                            "terminal_return_merge"
                        )
                    )
                )

            body_terminal = terminal_branch(body_statement.body)
            else_terminal = terminal_branch(body_statement.orelse)
            # A guard clause -- `if cond: body else: raise` or its mirror --
            # has only one arm that can ever reach the statement after the
            # if.  The other arm's bindings (an old value the raising arm
            # never rebinds, or a value the raising arm never gets to use)
            # are not an alternative the merge point can observe; inventing
            # a Phi between them and the live arm's value merges a value
            # that no execution ever actually produces on that edge, which
            # a downstream backend then has no real producer for.  Skip the
            # merge and let the single reachable arm's environment stand.
            if body_terminal and not else_terminal:
                environment.update(else_environment)
                attribute_effect_nodes.clear()
                attribute_effect_nodes.update(else_attribute_effects)
                attribute_value_nodes.clear()
                attribute_value_nodes.update(else_attribute_values)
            elif else_terminal and not body_terminal:
                environment.update(body_environment)
                attribute_effect_nodes.clear()
                attribute_effect_nodes.update(body_attribute_effects)
                attribute_value_nodes.clear()
                attribute_value_nodes.update(body_attribute_values)
            else:
                for name in set(before) | set(body_environment) | set(
                    else_environment
                ):
                    body_value = body_environment.get(name, before.get(name))
                    else_value = else_environment.get(name, before.get(name))
                    if body_value == else_value:
                        if body_value is not None:
                            environment[name] = body_value
                        continue
                    if (
                        isinstance(test_value, int)
                        and isinstance(body_value, int)
                        and isinstance(else_value, int)
                    ):
                        merged_attributes = {
                            "binding_name": name,
                            "source_conditional_id": id(body_statement),
                            **({
                                "initial_value_id": int(before[name]),
                            } if isinstance(before.get(name), int) else {}),
                        }
                        body_attributes = (
                            graph.G.nodes[body_value].get("attributes") or {}
                        )
                        else_attributes = (
                            graph.G.nodes[else_value].get("attributes") or {}
                        )
                        body_kind = body_attributes.get("aggregate_kind")
                        if (
                            body_kind is not None
                            and body_kind == else_attributes.get(
                                "aggregate_kind"
                            )
                        ):
                            merged_attributes.update({
                                key: body_attributes[key]
                                for key in (
                                    "aggregate_kind",
                                    "sequence_key_columns",
                                    "sequence_column_count",
                                    "sequence_writable",
                                )
                                if key in body_attributes
                            })
                            merged_attributes["producer_kind"] = (
                                "aggregate_phi"
                            )
                        merged_value = new_node(
                            "Phi",
                            name,
                            attributes=merged_attributes,
                            parents=(
                                (test_value, "test"),
                                (body_value, "body"),
                                (else_value, "orelse"),
                            ),
                        )
                        environment[name] = merged_value
                        # The identity history is the authoritative lexical
                        # version chain used by conditional-control lowering.
                        # Omitting the merge made a nested branch's continuation
                        # visible to later expressions but invisible to the
                        # control overlay, so no executable Phi was emitted for
                        # either the inner or enclosing conditional.
                        identity_bindings.setdefault(name, []).append(
                            merged_value
                        )

                # A record field is state just as a local name is.  Its
                # explicit GetAttr projection is the incoming SSA value, while
                # SetAttr contributes its RHS rather than its effect node.
                # First observation in an arm still names the receiver's
                # physical field. Seed the other edge from that projection,
                # never from an invented scalar Input or a read-history Phi.
                for field_key in (body_attribute_values.keys() | else_attribute_values.keys()):
                    if field_key in before_attribute_values:
                        continue
                    receiver_id, attribute_name = field_key
                    receiver_fact = current_identity_book().page(
                        "source_value_class_concordance"
                    ).latest((value_class_scope, int(receiver_id)))
                    receiver_class = (
                        str(receiver_fact[0])
                        if isinstance(receiver_fact, tuple) else None
                    )
                    if receiver_class is None:
                        effect_classes = {
                            str(slot[0])
                            for effect_id in (
                                body_attribute_effects.get(field_key),
                                else_attribute_effects.get(field_key),
                            )
                            if effect_id is not None
                            and int(effect_id) in graph.G
                            for slot in ((
                                graph.G.nodes[int(effect_id)].get(
                                    "attributes"
                                ) or {}
                            ).get("attribute_slot"),)
                            if (
                                isinstance(slot, tuple)
                                and len(slot) == 2
                            )
                        }
                        if len(effect_classes) == 1:
                            receiver_class = next(iter(effect_classes))
                    field_kind = (
                        (class_field_aggregate_kinds or {}).get((
                            str(receiver_class), str(attribute_name)
                        ))
                        if receiver_class is not None else None
                    )
                    initial_attributes = {
                        "attribute": attribute_name,
                        "initial_record_field_state": True,
                    }
                    if field_kind is not None:
                        mapping_contract = dict(
                            (class_field_mapping_contracts or {}).get((
                                str(receiver_class), str(attribute_name)
                            )) or {}
                        )
                        sequence_dtypes = tuple(
                            (class_field_sequence_dtypes or {}).get((
                                str(receiver_class), str(attribute_name)
                            )) or ()
                        )
                        initial_attributes.update({
                            "producer_kind": "record_field",
                            "aggregate_kind": field_kind,
                            "sequence_key_columns": (
                                (0,) if field_kind in {"set", "dict"} else ()
                            ),
                            "sequence_column_count": (
                                2 if field_kind == "dict" else 1
                            ),
                            "sequence_writable": field_kind != "tuple",
                            "record_field": (
                                str(receiver_class), str(attribute_name)
                            ),
                            **({
                                "sequence_column_dtypes": sequence_dtypes,
                            } if sequence_dtypes else {}),
                            **mapping_contract,
                        })
                    initial = new_node(
                        "GetAttr", f"getattr[{attribute_name}]",
                        attributes=initial_attributes,
                        parents=((int(receiver_id), "value"),),
                        source=body_statement,
                    )
                    before_attribute_values[field_key] = initial
                    for branch_values, branch_effects in (
                        (body_attribute_values, body_attribute_effects),
                        (else_attribute_values, else_attribute_effects),
                    ):
                        if field_key not in branch_effects:
                            branch_values[field_key] = initial
                for field_key, initial_value in (
                    before_attribute_values.items()
                ):
                    body_value = body_attribute_values.get(
                        field_key, initial_value
                    )
                    else_value = else_attribute_values.get(
                        field_key, initial_value
                    )
                    if body_value == else_value:
                        attribute_value_nodes[field_key] = int(body_value)
                        continue
                    if not isinstance(test_value, int):
                        continue
                    receiver_id, attribute_name = field_key
                    branch_effect_ids = tuple(
                        int(effect_id)
                        for effect_id in (
                            body_attribute_effects.get(field_key),
                            else_attribute_effects.get(field_key),
                        )
                        if effect_id is not None and int(effect_id) in graph.G
                    )
                    authored_field_names = {
                        f"{effect.value.id}.{effect.attr}"
                        for effect_id in branch_effect_ids
                        for effect in (graph.G.nodes[effect_id].get("expr_obj"),)
                        if isinstance(effect, ast.Attribute)
                        and isinstance(effect.value, ast.Name)
                    }
                    binding_name = (
                        next(iter(authored_field_names))
                        if len(authored_field_names) == 1 else
                        f"field:{receiver_id}.{attribute_name}"
                    )
                    body_attributes = (
                        graph.G.nodes[int(body_value)].get("attributes") or {}
                    )
                    else_attributes = (
                        graph.G.nodes[int(else_value)].get("attributes") or {}
                    )
                    merged_attributes = {
                        "binding_name": binding_name,
                        "source_conditional_id": id(body_statement),
                        "initial_value_id": int(initial_value),
                        "record_field_state": (
                            int(receiver_id), str(attribute_name)
                        ),
                    }
                    body_kind = body_attributes.get("aggregate_kind")
                    if (
                        body_kind is not None
                        and body_kind == else_attributes.get("aggregate_kind")
                    ):
                        merged_attributes.update({
                            key: body_attributes[key]
                            for key in (
                                "aggregate_kind",
                                "sequence_key_columns",
                                "sequence_column_count",
                                "sequence_writable",
                                "record_field",
                            )
                            if key in body_attributes
                        })
                        merged_attributes["producer_kind"] = "aggregate_phi"
                    merged_value = new_node(
                        "Phi",
                        binding_name,
                        attributes=merged_attributes,
                        parents=(
                            (int(test_value), "test"),
                            (int(body_value), "body"),
                            (int(else_value), "orelse"),
                        ),
                        source=body_statement,
                    )
                    identity_bindings.setdefault(binding_name, []).append(
                        merged_value
                    )
                    attribute_value_nodes[field_key] = merged_value
                    attribute_effect_nodes[field_key] = merged_value

            if (
                isinstance(test_value, int)
                and isinstance(body_result, int)
                and isinstance(else_result, int)
                and body_terminal
                and else_terminal
                and id(body_statement) in graph.G
            ):
                _replace_inputs(
                    graph,
                    id(body_statement),
                    (
                        (test_value, "test"),
                        (body_result, "body"),
                        (else_result, "orelse"),
                    ),
                )
                graph.G.nodes[id(body_statement)].setdefault(
                    "attributes", {}
                ).update({
                    "terminal_return_merge": True,
                    "terminal_return_values": (
                        int(body_result),
                        int(else_result),
                    ),
                })
            return id(body_statement)
        if isinstance(body_statement, ast.Try):
            # Python parameters exist from function entry.  A parameter first
            # read inside one arm is minted there and dropped with that arm's
            # environment, so the next read mints a SECOND Input for the same
            # parameter: two formals for one authored argument, and a returned
            # in/out parameter whose storage the caller can no longer bind.
            # Mint every parameter this statement reads BEFORE the snapshot,
            # the rule loops already apply; ``input_value`` is idempotent.
            for read_name in {
                member.id
                for member in source_walk(body_statement)
                if isinstance(member, ast.Name)
                and isinstance(member.ctx, ast.Load)
            }:
                if (
                    read_name in parameter_names
                    and read_name not in environment
                    and read_name not in static_environment
                ):
                    input_value(read_name, binding_kind="parameter")
            before = dict(environment)
            for nested in body_statement.body:
                reduce_statement(nested)
            body_environment = dict(environment)

            handler_environments = []
            for handler in body_statement.handlers:
                environment.clear()
                environment.update(before)
                if handler.name:
                    input_value(
                        handler.name,
                        binding_kind="exception",
                    )
                for nested in handler.body:
                    reduce_statement(nested)
                handler_environments.append(dict(environment))

            continuing_handler_environments = [
                candidate
                for handler, candidate in zip(
                    body_statement.handlers,
                    handler_environments,
                )
                if not (
                    handler.body
                    and isinstance(
                        handler.body[-1],
                        (ast.Raise, ast.Return),
                    )
                )
            ]
            # ``else`` belongs exclusively to the successful body path.  It
            # must therefore see that body's lexical definitions *before* the
            # successful and handler environments merge for code after the
            # whole try.  Merging first can bind a body value to the enclosing
            # Try node; an else call consuming it then points back to the Try
            # which structurally owns that call, fabricating a cycle.
            environment.clear()
            environment.update(body_environment)
            for nested in body_statement.orelse:
                reduce_statement(nested)
            successful_environment = dict(environment)

            # Only control-flow paths that can reach the statement following
            # the try participate in its lexical-value merge.  A handler
            # ending in raise/return has no continuation edge and therefore
            # cannot turn values assigned by the successful body into
            # invented external inputs.
            environments = [
                successful_environment,
                *continuing_handler_environments,
            ]
            environment.clear()
            if environments:
                common_names = set.intersection(
                    *(set(candidate) for candidate in environments)
                )
                for name in common_names:
                    values = {
                        candidate[name] for candidate in environments
                    }
                    if len(values) == 1:
                        environment[name] = values.pop()
                    elif id(body_statement) in graph.G:
                        # The branches genuinely disagree (a body assignment
                        # vs. a handler's fallback) -- exactly the case
                        # ``ast.If`` resolves with a ``Phi``.  A Try node has
                        # no single test expression to point a Phi at, but it
                        # doesn't need one: this Try's own graph node already
                        # evaluates to "whichever arm actually ran" (see
                        # ``evaluate_node``'s ``ast.Try`` handling, which
                        # re-runs body/handlers and keeps the last value) --
                        # so binding the name straight to the Try node's own
                        # id gives every later reference the same on-demand
                        # resolution a Phi would, without inventing a second
                        # node or a synthetic "did-raise" boolean.
                        environment[name] = id(body_statement)
            for nested in body_statement.finalbody:
                reduce_statement(nested)
            return id(body_statement)
        if isinstance(body_statement, (ast.For, ast.While)):
            # A discarded bound-method result is an effectful body statement
            # unless a later lowering proves otherwise.  Record the actual
            # statement calls here; loop composition must not rediscover
            # state transitions from method-name lists or output reachability.
            def current_loop_walk(node: ast.AST):
                """Walk one loop body without stealing descendant-loop effects."""

                yield node
                if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                    return
                for child in source_child_nodes(node):
                    yield from current_loop_walk(child)

            discarded_effect_calls = tuple(
                expression_statement.value
                for nested_statement in body_statement.body
                for expression_statement in current_loop_walk(
                    nested_statement
                )
                if (
                    isinstance(expression_statement, ast.Expr)
                    and isinstance(expression_statement.value, ast.Call)
                    and isinstance(
                        expression_statement.value.func,
                        ast.Attribute,
                    )
                    and isinstance(
                        expression_statement.value.func.value,
                        (ast.Name, ast.Attribute),
                    )
                )
            )
            # Mapping operations can both mutate and return a value.  Unlike
            # append/update used as bare statements, ``pop`` and
            # ``setdefault`` commonly occur inside an assignment or test.
            # Keep the original call identity so lowering must satisfy both
            # the state transition and its result.
            result_effect_calls = tuple(
                member
                for nested_statement in body_statement.body
                for member in current_loop_walk(nested_statement)
                if (
                    isinstance(member, ast.Call)
                    and isinstance(member.func, ast.Attribute)
                    and member.func.attr in {"pop", "setdefault"}
                    and isinstance(member.func.value, (ast.Name, ast.Attribute))
                )
            )
            state_effect_calls = tuple(dict.fromkeys((
                *discarded_effect_calls,
                *result_effect_calls,
            )))
            # Parameter bindings are materialized lazily at their first
            # lexical read, so a parameter first touched INSIDE the loop is
            # absent from the pre-loop snapshot -- and a name missing from
            # ``before_loop`` can never be discovered as loop-carried state.
            # That silently dropped the second carried value of an Adam-shaped
            # loop (``w`` was carried because a pre-loop statement had read
            # it; ``m`` was frozen at its entry value with no shortfall).  In
            # Python a parameter exists from function entry, so materializing
            # every parameter the loop reads BEFORE the snapshot is strictly
            # more faithful, and ``input_value`` is idempotent, so this mints
            # exactly the Input node the body's own first read would have.
            for read_name in {
                member.id
                for member in source_walk(body_statement)
                if isinstance(member, ast.Name)
                and isinstance(member.ctx, ast.Load)
            }:
                if (
                    read_name in parameter_names
                    and read_name not in environment
                    and read_name not in static_environment
                ):
                    input_value(read_name, binding_kind="parameter")
            if isinstance(body_statement, ast.For):
                iterator_value = resolve_expression(body_statement.iter)
                # Parameter/external bindings are materialized lazily at
                # their first lexical read.  Resolve the loop domain before
                # taking the initial-state snapshot so a value first read by
                # the iterator can still become explicit loop-carried state
                # when the body writes it.
                before_loop = dict(environment)
                bind_loop_target(body_statement.target)
                # ``for key, row in mapping.items()`` carries the mapping's
                # declared value-record identity onto the exact row binding.
                # The mapping field annotation is the authority; neither the
                # loop variable's spelling nor a runtime sample is used.
                mapping_value_record = None
                pending_iterator_values = (
                    [int(iterator_value)]
                    if isinstance(iterator_value, int) else []
                )
                visited_iterator_values: set[int] = set()
                while pending_iterator_values and mapping_value_record is None:
                    candidate = pending_iterator_values.pop()
                    if (
                        candidate in visited_iterator_values
                        or candidate not in graph.G
                    ):
                        continue
                    visited_iterator_values.add(candidate)
                    candidate_data = graph.G.nodes[candidate]
                    mapping_value_record = (
                        candidate_data.get("attributes") or {}
                    ).get("mapping_value_record")
                    if mapping_value_record is None:
                        pending_iterator_values.extend(
                            int(parent)
                            for parent, role in candidate_data.get(
                                "parents", ()
                            )
                            if str(role) in {
                                "operand", "value", "base", "object",
                                "receiver",
                            }
                        )
                if (
                    mapping_value_record is not None
                    and isinstance(body_statement.target, (ast.Tuple, ast.List))
                    and len(body_statement.target.elts) >= 2
                ):
                    row_target = body_statement.target.elts[1]
                    row_value = loop_target_bindings_by_ast.get(id(row_target))
                    if row_value is not None and row_value in graph.G:
                        graph.G.nodes[row_value].setdefault(
                            "attributes", {}
                        )["result_class_ref"] = str(mapping_value_record)
            else:
                resolve_expression(body_statement.test)
                # The same ordering is essential for while loops: a parameter
                # first read by the predicate is the initial version consumed
                # by the header, not a name that appeared only after entry.
                # Snapshotting before this resolution silently lost genuine
                # body updates such as ``data = data[0] if data else []``.
                before_loop = dict(environment)
            for nested in body_statement.body:
                reduce_statement(nested)
            for nested in body_statement.orelse:
                reduce_statement(nested)
            loop_id = id(body_statement)
            if loop_id in graph.G:
                body_member_ids = {
                    id(member)
                    for nested in body_statement.body
                    for member in source_walk(nested)
                }
                direct_loop_target_names = (
                    set(loop_target_names(body_statement.target))
                    if isinstance(body_statement, ast.For)
                    else set()
                )
                # Read iteration inputs from the target AST identities that
                # ``bind_loop_target`` recorded, not from the post-body lexical
                # environment.  A body may legally reassign one tuple member
                # (``for op, av in rows: ...; op = replacement``); by this
                # point ``environment['op']`` denotes that later body value,
                # but the loop's row contract still has both ``op`` and ``av``.
                current_loop_bindings = {
                    target_name.id: loop_target_bindings_by_ast[id(target_name)]
                    for target_name in target_name_nodes(body_statement.target)
                    if (
                        id(target_name) in loop_target_bindings_by_ast
                        and before_loop.get(target_name.id)
                        != loop_target_bindings_by_ast[id(target_name)]
                    )
                } if isinstance(body_statement, ast.For) else {}
                # AST ingestion initially wires name occurrences before
                # lexical rebinding is known.  A reused spelling (for example
                # a comprehension's ``width`` followed by a ``for width``)
                # must not leave the second loop body attached to the first
                # binding.  Rewrite only nodes lexically owned by this body.
                for member_id in body_member_ids:
                    if member_id not in graph.G:
                        continue
                    parents = list(
                        graph.G.nodes[member_id].get("parents") or ()
                    )
                    replacements = {
                        old: current_loop_bindings[name]
                        for name in current_loop_bindings
                        for old in identity_bindings.get(name, ())
                        if old != current_loop_bindings[name]
                    }
                    rewritten = [
                        (replacements.get(parent, parent), role)
                        for parent, role in parents
                    ]
                    if rewritten != parents:
                        _replace_inputs(
                            graph,
                            member_id,
                            tuple(rewritten),
                        )
                loop_attributes = graph.G.nodes[loop_id].setdefault(
                    "attributes",
                    {},
                )
                loop_target_bindings = current_loop_bindings
                # A local assigned unconditionally from the current iteration
                # before its first read is not recurrence.  For example
                # ``for p in rows: op, av = p[0]`` gives ``op`` a new value on
                # every iteration; exporting it as a Phi asks the latch for a
                # producer even though the next iteration overwrites it before
                # use.  Preserve genuine state (including conditional writes)
                # by accepting only a direct body assignment reached before
                # any lexical load of the same name.
                overwritten_before_read: set[str] = set()
                candidate_names = before_loop.keys() & environment.keys()
                for name in candidate_names:
                    for nested in body_statement.body:
                        loaded = any(
                            isinstance(member, ast.Name)
                            and isinstance(member.ctx, ast.Load)
                            and member.id == name
                            for member in source_walk(nested)
                        )
                        direct_targets: tuple[ast.AST, ...] = ()
                        if isinstance(nested, ast.Assign):
                            direct_targets = tuple(nested.targets)
                        elif isinstance(nested, ast.AnnAssign):
                            direct_targets = (nested.target,)
                        assigned = any(
                            target_name.id == name
                            for target in direct_targets
                            for target_name in target_name_nodes(target)
                        )
                        if assigned and not loaded:
                            overwritten_before_read.add(name)
                            break
                        if loaded:
                            break
                # Overwrite-before-read proves that a value is not a recurrent
                # body input; it does *not* prove that the pre-loop value is
                # dead. A later lexical load observes the initial value when
                # the loop executes zero times and the final body value
                # otherwise, so that name still requires a loop-result phi.
                # Restrict the scan to this function scope: a nested definition
                # captures a name under a separate invocation and is not a
                # continuation of this loop.
                def same_scope_walk(node: ast.AST):
                    yield node
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (
                            ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.Lambda,
                        )):
                            continue
                        yield from same_scope_walk(child)

                loop_end = int(getattr(
                    body_statement, "end_lineno",
                    getattr(body_statement, "lineno", -1),
                ))
                live_after_loop = {
                    member.id
                    for member in same_scope_walk(statement)
                    if isinstance(member, ast.Name)
                    and isinstance(member.ctx, ast.Load)
                    and int(getattr(member, "lineno", -1)) > loop_end
                }
                loop_carried_bindings = {
                    name: (before_loop[name], environment[name])
                    for name in before_loop.keys() & environment.keys()
                    if (
                        before_loop[name] != environment[name]
                        and name not in loop_target_bindings
                        and (
                            name not in overwritten_before_read
                            or name in live_after_loop
                        )
                        # A descendant loop's induction binding remains in
                        # Python's lexical environment after that loop, but it
                        # is not recurrent state of this enclosing loop.  Its
                        # value is produced by the descendant loop domain, not
                        # by an enclosing-loop latch.  Treating it as carried
                        # creates a Phi whose backedge can never have a body
                        # producer (for example pad_cat_'s inner ``i, size``
                        # enumerate target leaking into the outer ``arr``
                        # loop).
                        and not (
                            environment[name] in graph.G
                            and (
                                graph.G.nodes[environment[name]].get(
                                    "attributes"
                                ) or {}
                            ).get("binding_kind") == "loop"
                        )
                    )
                }
                # A terminal ``continue`` arm does not reach the conditional's
                # ordinary fallthrough environment, by design.  Its bindings
                # nevertheless reach the loop latch and are therefore genuine
                # recurrence.  The control-site ledger captured that exact
                # environment when the Continue statement was reduced.  Use it
                # when every continue edge agrees on one updated identity; if
                # different edges carry different values, a latch Phi must be
                # constructed explicitly and this source-level pass must not
                # guess which value wins.
                loop_start = int(getattr(body_statement, "lineno", -1))
                continue_sites = tuple(
                    site
                    for site_span, site in (
                        graph.G.graph.get("loop_control_site_bindings") or {}
                    ).items()
                    if (
                        site.get("loop_id") is None
                        and site.get("action") == "continue"
                        and loop_start <= int(site_span[0]) <= loop_end
                    )
                )
                for name, initial in before_loop.items():
                    if name in loop_carried_bindings or not continue_sites:
                        continue
                    continuation_values = {
                        int(site.get("bindings", {}).get(name, initial))
                        for site in continue_sites
                    }
                    if (
                        len(continuation_values) == 1
                        and next(iter(continuation_values)) != int(initial)
                    ):
                        loop_carried_bindings[name] = (
                            int(initial), next(iter(continuation_values)),
                        )
                loop_attributes["loop_carried_bindings"] = (
                    loop_carried_bindings
                )
                # Claim this loop's break/continue sites (a nested loop has
                # already claimed its own).  For each ``break`` site, every
                # name whose value at the site differs from its pre-loop
                # value reaches the loop exit through that edge, keyed by the
                # pre-loop identity so the exit merge can pair it with the
                # carried (or zero-trip) value.  A name bound ONLY on break
                # paths is not loop-carried (the fall-through path never
                # rebinds it) yet still needs a post-loop identity: its
                # continuation is the last break site's value, which the loop
                # port materialization rewires onto a LoopResult port.
                loop_break_sites: dict[tuple[int, int, int, int], dict[int, int]] = {}
                loop_break_bindings: dict[str, tuple[int, int]] = {}
                for site_span, site in (
                    graph.G.graph.get("loop_control_site_bindings") or {}
                ).items():
                    if (
                        site.get("loop_id") is not None
                        or not (loop_start <= int(site_span[0]) <= loop_end)
                    ):
                        continue
                    site["loop_id"] = loop_id
                    if site["action"] != "break":
                        continue
                    site_values: dict[int, int] = {}
                    for name, value in site["bindings"].items():
                        if name in loop_target_bindings:
                            continue
                        initial = before_loop.get(name)
                        if (
                            initial is None
                            and name in parameter_names
                            and name not in static_environment
                        ):
                            initial = input_value(name, binding_kind="parameter")
                        if (
                            initial is None
                            or int(initial) == int(value)
                            or int(initial) not in graph.G
                            or int(value) not in graph.G
                        ):
                            continue
                        site_values[int(initial)] = int(value)
                        if name not in loop_carried_bindings:
                            loop_break_bindings[name] = (int(initial), int(value))
                    loop_break_sites[site_span] = site_values
                    # The break edge CONSUMES these values (the loop-exit
                    # merge reads them), so the Break node records them as
                    # its inputs.  Without that edge a pre-loop parameter
                    # whose only later reader is the exit merge has no
                    # consumer yet and the planner prunes it as an unused
                    # parameter before the merge exists.
                    site_node_id = site.get("node_id")
                    if site_values and site_node_id in graph.G:
                        _replace_inputs(
                            graph,
                            site_node_id,
                            (
                                *(
                                    (int(initial), "site_initial")
                                    for initial in site_values
                                ),
                                *(
                                    (int(value), "site_value")
                                    for value in site_values.values()
                                ),
                            ),
                        )
                loop_attributes["loop_break_sites"] = loop_break_sites
                loop_attributes["loop_break_bindings"] = loop_break_bindings
                for name, (_initial, continuation) in loop_break_bindings.items():
                    environment[name] = continuation
                loop_attributes["loop_target_bindings"] = (
                    loop_target_bindings
                )
                loop_attributes["loop_target_initials"] = {
                    name: before_loop[name]
                    for name in loop_target_bindings
                    if name in before_loop
                }
                # This pass resolves source/value identities only.  It records
                # the body value selected by the lexical continuation, but it
                # must not manufacture a loop latch, exit, collection owner,
                # or backend schedule.  The post-canonical loop reducer will
                # either thread these values through straight-line unrolled
                # SSA or create retained-loop result ports.
                for name, (_initial, updated) in (
                    loop_carried_bindings.items()
                ):
                    environment[name] = updated
                    identity_bindings.setdefault(name, []).append(updated)
                # A discarded bound-method call is retained here only as a
                # source effect fact.  Its Python return value is not the
                # mutated state, and no synthetic state transition belongs in
                # the critical value graph before loop realization is known.
                state_effects = []
                for call in state_effect_calls:
                    receiver_expression = call.func.value
                    name = ast.unparse(receiver_expression)
                    # A parameter first referenced inside this loop is resolved
                    # while reducing the body, after ``before_loop`` was
                    # snapped.  Keep that real input as an opaque effect unless
                    # its storage provenance proves a sequence policy; never
                    # let the mutation disappear merely because its first use
                    # is nested here.
                    call_id = id(call)
                    call_parents = tuple(
                        graph.G.nodes[call_id].get("parents") or ()
                    ) if call_id in graph.G else ()
                    operand = next((
                        int(parent)
                        for parent, role in call_parents
                        if str(role) == "operand"
                    ), None)
                    initial = (
                        before_loop.get(name, environment.get(name))
                        if isinstance(receiver_expression, ast.Name)
                        else operand
                    )
                    if (
                        initial is None
                        or initial not in graph.G
                        or call_id not in graph.G
                        or name in loop_carried_bindings
                    ):
                        continue
                    call_attributes = (
                        graph.G.nodes[call_id].get("attributes") or {}
                    )
                    # A source-linked method is already an ordinary SSA call:
                    # its method_ref selects the complete function-table body,
                    # and its receiver is the explicit ``operand`` input.  It
                    # is not a second, opaque mutation of that receiver.  The
                    # callee's own GetAttr/SetAttr and calls carry its effects.
                    if any(
                        call_attributes.get(key) is not None
                        for key in (
                            "method_ref", "callee_ref", "resolved_ast_parent",
                        )
                    ):
                        continue
                    argument_ids = tuple(
                        int(parent)
                        for parent, role in call_parents
                        if str(role).startswith("arg")
                    )
                    if isinstance(receiver_expression, ast.Name):
                        environment[name] = initial
                    argument_expression = (
                        call.args[0] if len(call.args) == 1 else None
                    )
                    state_attributes = (
                        graph.G.nodes[initial].get("attributes") or {}
                    )
                    aggregate_kind = state_attributes.get("aggregate_kind")
                    sequence_policy = (
                        "unique"
                        if aggregate_kind in {"set", "dict"}
                        else (
                            "duplicates"
                            if aggregate_kind in {
                                "list", "tuple", "bytes", "bytearray"
                            }
                            else None
                        )
                    )
                    sequence_mutation = (
                        sequence_policy is not None
                        and call.func.attr in _SEQUENCE_MUTATION_OPERATORS
                        and not (
                            aggregate_kind == "dict"
                            and call.func.attr == "pop"
                        )
                        and not (
                            aggregate_kind in {"tuple", "bytes"}
                            or not state_attributes.get(
                                "sequence_writable", True
                            )
                        )
                    )
                    mapping_mutation = (
                        aggregate_kind == "dict"
                        and call.func.attr in _MAPPING_MUTATION_OPERATORS
                        and state_attributes.get("sequence_writable", True)
                    )
                    if sequence_mutation or mapping_mutation:
                        mutation_page = current_identity_book().page(
                            "source_sequence_mutation_concordance"
                        )
                        mutation_row = (value_class_scope, int(call_id))
                        mutation_fact = (
                            int(initial), str(call.func.attr),
                            str(sequence_policy), tuple(argument_ids),
                            "mapping" if mapping_mutation else "sequence",
                        )
                        incumbent = mutation_page.latest(mutation_row)
                        if (
                            incumbent is not None
                            and tuple(incumbent) != mutation_fact
                        ):
                            raise ValueError(
                                "source sequence mutation concordance "
                                f"disagreement for {mutation_row!r}: "
                                f"recorded={incumbent!r}, "
                                f"proposed={mutation_fact!r}"
                            )
                        if incumbent is None:
                            mutation_page.set(mutation_row, 0, mutation_fact)
                    state_effects.append({
                        "state_name": name,
                        "operator": call.func.attr,
                        "effect_mode": (
                            "sequence_mutation"
                            if sequence_mutation
                            else (
                                "mapping_mutation"
                                if mapping_mutation else "opaque"
                            )
                        ),
                        "sequence_policy": sequence_policy,
                        "argument_kind": (
                            "generator"
                            if isinstance(argument_expression, ast.GeneratorExp)
                            else (
                                (
                                    "filtered_sequence"
                                    if any(
                                        generator.ifs
                                        for generator in argument_expression.generators
                                    )
                                    else "sequence"
                                )
                                if isinstance(argument_expression, ast.ListComp)
                                else "value"
                            )
                        ),
                        "state_input_id": int(initial),
                        "effect_node_id": int(call_id),
                        "argument_value_ids": argument_ids,
                    })
                if state_effects:
                    loop_attributes["loop_state_effects"] = tuple(
                        state_effects
                    )
            return id(body_statement)
        if isinstance(body_statement, (ast.Break, ast.Continue)):
            # A loop-control site leaves its arm with the arm's OWN bindings
            # (``if c: dt_try = ...; break`` reaches the loop exit with that
            # dt_try).  The ``if`` merge below deliberately drops a terminal
            # arm's bindings from the fall-through environment, so the exit
            # edge must carry them explicitly.  Record the environment at the
            # site, keyed by source span (AST ids do not survive reduction);
            # the enclosing loop claims the site when it finishes reducing.
            graph.G.graph.setdefault("loop_control_site_bindings", {})[(
                int(getattr(body_statement, "lineno", -1)),
                int(getattr(body_statement, "col_offset", -1)),
                int(getattr(body_statement, "end_lineno", -1)),
                int(getattr(body_statement, "end_col_offset", -1)),
            )] = {
                "action": (
                    "break" if isinstance(body_statement, ast.Break)
                    else "continue"
                ),
                "bindings": {
                    str(name): int(value)
                    for name, value in environment.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                },
                "loop_id": None,
                "node_id": id(body_statement),
            }
            return id(body_statement) if id(body_statement) in graph.G else None
        if isinstance(body_statement, ast.Expr):
            return resolve_expression(body_statement.value)
        for child in source_child_nodes(body_statement):
            if isinstance(child, ast.expr):
                resolve_expression(child)
        return id(body_statement) if id(body_statement) in graph.G else None

    returned_values = []
    for body_statement in statement.body:
        value = reduce_statement(body_statement)
        if isinstance(body_statement, ast.Return) and value is not None:
            returned_values.append(value)
    if returned_values:
        graph.roots = list(dict.fromkeys(returned_values))
        # The function's return is a consumer of its roots: row
        # ``(scope, "return", "root", position)`` names the binding returned
        # there, when every return of that value names one binding.
        return_page = current_identity_book().page("lexical_read_binding")
        for position, root in enumerate(graph.roots):
            names = return_root_bindings.get(int(root)) if isinstance(root, int) else None
            if names and len(names) == 1:
                return_page.concord(
                    (ingestion_read_scope, "return", "root", position),
                    next(iter(names)),
                )

    # A source-linked unbound method can arrive with its receiver absent from
    # the owned subgraph even though the Attribute load itself is owned.
    # Restore the ordinary edge before lexical Name cleanup; an Attribute
    # without its receiver is not a valid structural operation. The receiver
    # can be a parameter (``self`` bound directly at the call boundary) or an
    # ordinary local (``machine = load_pe(...)``) -- both are recorded under
    # the same name in ``environment`` by ``bind_target``, so there is no
    # reason to restore only the parameter case and leave a local's receiver
    # permanently unresolved. A chained access (``machine.runner.tick``) puts
    # an Attribute (``machine.runner``), not a Name, in the receiver position
    # of the outer Attribute (``.tick``); that inner Attribute is itself just
    # another node this same loop repairs on its own matching iteration, so
    # linking straight to its node id (when it is still a live node) is
    # enough -- the two repairs do not need to happen in any particular
    # order, since neither removes a node, only reattaches an edge.
    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Attribute)
            and not data.get("parents")
            and isinstance(expression.value, (ast.Name, ast.Attribute))
        ):
            continue
        if isinstance(expression.value, ast.Name):
            receiver = environment.get(expression.value.id)
            if receiver is None:
                receiver = input_value(
                    expression.value.id,
                    binding_kind=(
                        "parameter"
                        if expression.value.id in parameter_names
                        else "local"
                    ),
                )
        else:
            inner_id = id(expression.value)
            if inner_id not in graph.G:
                continue
            receiver = inner_id
        _replace_inputs(graph, node_id, ((receiver, "value"),))

    # Any surviving lexical occurrence is either unused syntax or an unresolved
    # source label.  It is not executable work in the reduced value graph --
    # but only when nothing still structurally depends on it. A Name node
    # that still has a real successor (an Attribute load whose receiver is
    # this exact node, say) is not unused syntax; removing it anyway does
    # not erase that dependency, it just strips the successor's own record
    # of it, leaving that successor orphaned for whatever later pass expects
    # to find its receiver -- the exact "receiver Name absent" case the
    # repair above this loop exists to patch after the fact. Checking
    # liveness first means there is nothing left to repair.
    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Name):
            if graph.G.out_degree(node_id) > 0:
                continue
            _remove_node(graph, node_id)
            continue
        if (
            data.get("type") == "Input"
            and graph.G.out_degree(node_id) == 0
            and node_id not in graph.roots
            and (data.get("attributes") or {}).get("binding_kind")
            == "external"
        ):
            # A named Call target initially looks like a lexical external.
            # Once the Call owns a function-table reference, that temporary
            # value has no consumers and must not leak into the public input
            # contract.
            _remove_node(graph, node_id)
            continue
        if (
            data.get("type") == "StaticReference"
            and graph.G.out_degree(node_id) == 0
            and node_id not in graph.roots
            and (data.get("attributes") or {}).get("reference_kind")
            == "function_subgraph"
        ):
            # Resolving a direct named call first encounters the callee as a
            # possible first-class value.  Once the Call has recorded its
            # exact ``callee_ref``, that temporary token has no runtime role:
            # the referenced source body lives in FunctionTable and the Call
            # invokes its compiled slot.  Keep a function reference whenever
            # it is returned or consumed as data; remove only this genuinely
            # disconnected compiler token so it cannot be scheduled as an
            # operator of its own.
            _remove_node(graph, node_id)

    # A compile-time reference left as a root is one nothing consumes at
    # runtime. Whether that is correct depends entirely on whether it is
    # IRREDUCIBLE: pursue into whatever is demanded, and terminate only at
    # content that cannot be reduced further -- never drop something that still
    # has source to pursue.
    #
    #  * Irreducible (a module/builtin/C-extension used only at compile time,
    #    e.g. ``torch`` in ``isinstance(x, torch.Tensor)``): no readable source,
    #    no runtime value. It is a terminal external, not a runtime output, so
    #    drop it from the roots (and remove its node if present).
    #  * Pursuable (a function/class whose source we CAN read): reaching
    #    reduction as an unconsumed root means the pursuit
    #    (``_expand_unresolved_ast_parents``) did not follow it. Dropping that
    #    would silently delete real content, so SURFACE it instead of hiding it.
    #
    # Done before the invariant scan so the scan sees the cleaned graph. A
    # compile-time reference is never a runtime PARENT (that would be a genuine
    # bug), so those remain hard invariants below.
    compile_time_roots = [root for root in graph.roots if type(root) is not int]
    if compile_time_roots:
        pursuable_roots = [
            root for root in compile_time_roots
            if _reference_has_pursuable_source(getattr(root, "value", root))
        ]
        if pursuable_roots:
            raise ValueError(
                "demanded reference(s) reached reduction unresolved but still "
                "have source to pursue -- the frontend pursuit did not follow "
                f"them into their content: {pursuable_roots!r}"
            )
        graph.roots = [root for root in graph.roots if type(root) is int]
        for root in compile_time_roots:
            if root in graph.G:
                _remove_node(graph, root)

    # Stable topological relabeling turns structural token chains into compact
    # dense value IDs without allowing opaque Python object identities to
    # participate in the ordering.
    invalid_node_ids = [
        node_id for node_id in graph.G if type(node_id) is not int
    ]
    invalid_parent_ids = [
        parent_id
        for _node_id, data in graph.G.nodes(data=True)
        for parent_id, _role in data.get("parents", ())
        if type(parent_id) is not int
    ]
    assert not invalid_node_ids, (
        "compile-time references must be represented by integer-keyed "
        f"StaticReference nodes, not graph keys: {invalid_node_ids!r}"
    )
    assert not invalid_parent_ids, (
        "compile-time references must not appear as runtime parent IDs: "
        f"{invalid_parent_ids!r}"
    )
    def stable_token_atom(value: Any) -> str:
        if isinstance(value, str):
            address_repr = re.fullmatch(
                r"<([A-Za-z_][A-Za-z_0-9.]*) object at 0x[0-9A-Fa-f]+>",
                value,
            )
            return (
                address_repr.group(1)
                if address_repr is not None else value
            )
        if value is None or isinstance(value, (bool, float, int)):
            return str(value)
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (bool, float, int, str)):
            return str(enum_value)
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    def node_description(node_id: int) -> dict[str, Any]:
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        return {
            "source_span": {
                "line": getattr(expression, "lineno", -1),
                "column": getattr(expression, "col_offset", -1),
                "end_line": getattr(expression, "end_lineno", -1),
                "end_column": getattr(expression, "end_col_offset", -1),
            },
            "ast": (
                ast.dump(expression, include_attributes=False)
                if isinstance(expression, ast.AST) else type(expression).__name__
            ),
            "type": stable_token_atom(data.get("type", "")),
            "op": stable_token_atom(data.get("op", "")),
            "label": stable_token_atom(data.get("label", "")),
        }

    base_token_chains: dict[int, tuple[str, ...]] = {}
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        line = getattr(expression, "lineno", -1)
        column = getattr(expression, "col_offset", -1)
        end_line = getattr(expression, "end_lineno", -1)
        end_column = getattr(expression, "end_col_offset", -1)
        missing_position = 10**12
        ordering_prefix = (
            "priority:0" if str(data.get("type", "")) == "Input" else "priority:1",
            f"line:{line if line >= 0 else missing_position:012d}",
            f"column:{column if column >= 0 else missing_position:012d}",
            f"end_line:{end_line if end_line >= 0 else missing_position:012d}",
            f"end_column:{end_column if end_column >= 0 else missing_position:012d}",
        )
        parent_context = tuple(
            {
                "role": str(role),
                "node": node_description(parent_id),
            }
            for parent_id, role in data.get("parents", ())
            if parent_id in graph.G
        )
        base_token_chains[node_id] = (
            *ordering_prefix,
            *structural_context_tokens({
                "node": node_description(node_id),
                "parents": parent_context,
            }),
        )
    chain_versions: dict[tuple[str, ...], int] = {}
    node_token_chains: dict[int, tuple[str, ...]] = {}
    for node_id in graph.G.nodes:
        chain = base_token_chains[node_id]
        version = chain_versions.get(chain, 0)
        chain_versions[chain] = version + 1
        node_token_chains[node_id] = (*chain, f"version:{version}")
    ordered = list(
        lexicographical_topological_order(
            graph.G, key=lambda node_id: node_token_chains[node_id]
        )
    )
    mapping = {
        node_id: value_id for value_id, node_id in enumerate(ordered)
    }
    relabeled = nx.relabel_nodes(graph.G, mapping, copy=True)
    ordered_graph = nx.DiGraph()
    ordered_graph.graph.update(relabeled.graph)
    ordered_graph.graph["canonical_value_ids"] = True
    # Renumbering changes the id space; a watermark from the AST-id
    # ingestion graph must not leak into canonical allocation.
    ordered_graph.graph.pop("value_id_watermark", None)
    ordered_graph.graph["ssa_identity_tokens"] = {
        mapping[node_id]: node_token_chains[node_id]
        for node_id in ordered
    }
    map_ir = dict(ordered_graph.graph.get("map_ir") or {})
    map_ir["schema_node_ids"] = tuple(
        mapping[node_id]
        for node_id in map_ir.get("schema_node_ids", ())
        if node_id in mapping
    )
    map_ir["schema_roots"] = tuple(
        mapping[node_id]
        for node_id in map_ir.get("schema_roots", ())
        if node_id in mapping
    )
    ordered_graph.graph["map_ir"] = map_ir
    for value_id in range(len(mapping)):
        ordered_graph.add_node(value_id, **relabeled.nodes[value_id])
    ordered_graph.add_edges_from(relabeled.edges(data=True))
    graph.G = ordered_graph
    graph.roots = [
        mapping[root] for root in graph.roots if root in mapping
    ]
    graph.levels = {
        mapping[node_id]: level
        for node_id, level in graph.levels.items()
        if node_id in mapping
    }
    graph.G.graph["identity_table"] = {
        name: tuple(
            mapping[value_id]
            for value_id in value_ids
            if value_id in mapping
        )
        for name, value_ids in identity_bindings.items()
    }
    # Class facts this function committed while its values still had
    # ingestion ids follow those values into the canonical id space, exactly
    # as the identity table does; the concordance answers by canonical id.
    class_page = current_identity_book().page(
        "source_value_class_concordance"
    )
    for row in tuple(class_page.rows()):
        if (
            isinstance(row, tuple) and len(row) == 2
            and row[0] == value_class_scope and row[1] in mapping
            and mapping[row[1]] != row[1]
        ):
            for column, fact in class_page.history(row):
                class_page.set(
                    (value_class_scope, mapping[row[1]]), column, fact,
                )
    # The per-read bindings follow their consumers into canonical ids.  A
    # scope reduced again revises its canonical rows rather than keeping a
    # previous reduction's facts.
    read_page = current_identity_book().page("lexical_read_binding")
    for row in read_page.scope_rows(ingestion_read_scope):
        if len(row) == 4 and row[1] == "return":
            read_page.concord((read_scope, *row[1:]), read_page.latest(row))
        elif len(row) == 4 and row[1] in mapping:
            read_page.concord(
                (read_scope, mapping[row[1]], row[2], row[3]),
                read_page.latest(row),
            )
    graph.G.graph["lexical_read_scope"] = read_scope
    # Per-return slot values are ids in the pre-canonical space too.
    graph.G.graph["return_slot_values"] = {
        span: tuple(
            None if value_id is None or value_id not in mapping
            else mapping[value_id]
            for value_id in slot_ids
        )
        for span, slot_ids in (
            graph.G.graph.get("return_slot_values") or {}
        ).items()
    }
    # Ingestion spelling and SSA definition numbering are separate domains.
    graph.G.graph["return_record_field_states"] = {
        span: tuple(
            (mapping[receiver], field, mapping[value])
            for receiver, field, value in states
            if receiver in mapping and value in mapping
        )
        for span, states in (
            graph.G.graph.get("return_record_field_states") or {}
        ).items()
    }
    # ``identity_table`` remains the compact compatibility map used by older
    # lowering code.  This ledger preserves the original common spelling and
    # its authored rebinding version before any SSA-only phi/capture/temp
    # values are introduced.
    graph.G.graph["ingestion_identity_table"] = {
        str(name): tuple(
            {
                "spelling": str(name),
                "version": version,
                "value_id": mapping[value_id],
                "source_span": {
                    key: source.get(key)
                    for key in ("line", "column", "end_line", "end_column")
                },
                **({
                    "context": source["context"],
                    "context_tokens": source["context_tokens"],
                    "context_token_ids": source["context_token_ids"],
                    "context_sha256": source["context_sha256"],
                } if "context" in source else {}),
            }
            for version, (value_id, source) in enumerate(definitions)
            if value_id in mapping
        )
        for name, definitions in ingestion_definitions.items()
    }
    for value_id, data in graph.G.nodes(data=True):
        data["value_id"] = value_id
        attributes = data.get("attributes") or {}
        if "loop_carried_bindings" in attributes:
            attributes["loop_carried_bindings"] = {
                name: (mapping[initial], mapping[updated])
                for name, (initial, updated) in attributes[
                    "loop_carried_bindings"
                ].items()
                if initial in mapping and updated in mapping
            }
        if "loop_break_bindings" in attributes:
            attributes["loop_break_bindings"] = {
                name: (mapping[initial], mapping[continuation])
                for name, (initial, continuation) in attributes[
                    "loop_break_bindings"
                ].items()
                if initial in mapping and continuation in mapping
            }
        if "loop_break_sites" in attributes:
            attributes["loop_break_sites"] = {
                span: {
                    mapping[initial]: mapping[value]
                    for initial, value in site_values.items()
                    if initial in mapping and value in mapping
                }
                for span, site_values in attributes["loop_break_sites"].items()
            }
        if "loop_target_bindings" in attributes:
            attributes["loop_target_bindings"] = {
                name: mapping[target]
                for name, target in attributes[
                    "loop_target_bindings"
                ].items()
                if target in mapping
            }
        if "loop_target_initials" in attributes:
            attributes["loop_target_initials"] = {
                name: mapping[initial]
                for name, initial in attributes[
                    "loop_target_initials"
                ].items()
                if initial in mapping
            }
        if "source_conditional_id" in attributes:
            source_conditional_id = attributes["source_conditional_id"]
            if source_conditional_id in mapping:
                attributes["source_conditional_id"] = mapping[
                    source_conditional_id
                ]
        if "initial_value_id" in attributes:
            initial_value_id = attributes["initial_value_id"]
            if initial_value_id in mapping:
                attributes["initial_value_id"] = mapping[initial_value_id]
        if "record_field_state" in attributes:
            receiver_id, attribute_name = attributes["record_field_state"]
            if receiver_id in mapping:
                attributes["record_field_state"] = (
                    mapping[receiver_id], attribute_name,
                )
        if "terminal_return_values" in attributes:
            attributes["terminal_return_values"] = tuple(
                mapping[value_id]
                for value_id in attributes["terminal_return_values"]
                if value_id in mapping
            )
        if "loop_state_effects" in attributes:
            attributes["loop_state_effects"] = tuple(
                {
                    **effect,
                    "state_input_id": mapping[effect["state_input_id"]],
                    "effect_node_id": mapping[effect["effect_node_id"]],
                    "argument_value_ids": tuple(
                        mapping[value_id]
                        for value_id in effect["argument_value_ids"]
                        if value_id in mapping
                    ),
                }
                for effect in attributes["loop_state_effects"]
                if all(
                    effect[key] in mapping
                    for key in (
                        "state_input_id",
                        "effect_node_id",
                    )
                )
            )
        if "loop_iteration_outputs" in attributes:
            attributes["loop_iteration_outputs"] = tuple(
                {
                    key: mapping[output[key]]
                    for key in (
                        "value_id",
                        "result_value_id",
                        "materializer_node_id",
                    )
                }
                for output in attributes["loop_iteration_outputs"]
                if all(
                    output[key] in mapping
                    for key in (
                        "value_id",
                        "result_value_id",
                        "materializer_node_id",
                    )
                )
            )
        for key in (
            "aggregate_leaf_value_ids",
            "materialized_source_value_ids",
        ):
            if key in attributes:
                attributes[key] = tuple(
                    mapping[value_id]
                    for value_id in attributes[key]
                    if value_id in mapping
                )
        data["parents"] = [
            (mapping[parent_id], role)
            for parent_id, role in data.get("parents", ())
            if parent_id in mapping
        ]
        data["children"] = [
            (mapping[child_id], role)
            for child_id, role in data.get("children", ())
            if child_id in mapping
        ]

    terminal_merges = [
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("terminal_return_merge")
    ]
    for merge_id in terminal_merges:
        branch_values = set(
            (graph.G.nodes[merge_id].get("attributes") or {}).get(
                "terminal_return_values", ()
            )
        )
        if branch_values and branch_values.issubset(graph.roots):
            graph.roots = [
                root for root in graph.roots if root not in branch_values
            ]
            graph.roots.append(merge_id)


def normalize_python_attribute_special_cases(graph: Any) -> None:
    """Canonicalize Python attribute loads on a graph already in the table.

    Dependency linking can attach a previously extracted Python function
    after the root reduction pass. Apply the same AST special-case overlay to
    those graphs before deployment so linked functions cannot retain raw
    ``ast.Attribute`` nodes that backends would mistake for external values.
    """

    if str(getattr(graph, "source_language", "")).casefold() != "python":
        return
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        augmented_target_read = bool(
            isinstance(expression, ast.Attribute)
            and isinstance(expression.ctx, ast.Store)
            and any(
                str(role) == "lhs"
                and child_id in graph.G
                and isinstance(
                    graph.G.nodes[child_id].get("expr_obj"), ast.AugAssign,
                )
                for child_id, role in data.get("children") or ()
            )
        )
        if not (
            isinstance(expression, ast.Attribute)
            and (
                isinstance(expression.ctx, ast.Load)
                or augmented_target_read
            )
        ):
            continue
        special_expression = expression
        if augmented_target_read:
            # Python gives an AugAssign target Store context, but the operation
            # is read-modify-write. This graph node supplies the read operand;
            # the reducer's separate SetAttr node owns the write effect.
            special_expression = copy.deepcopy(expression)
            special_expression.ctx = ast.Load()
        special = interpret_python_special_case(special_expression)
        if special is None:
            continue
        data["type"] = special.type
        data["op"] = special.type
        data["attributes"] = {
            **dict(data.get("attributes") or {}),
            **dict(special.attributes),
        }
        data["extra_args"] = {
            **dict(data.get("extra_args") or {}),
            **dict(special.attributes),
        }


def reduce_abstract_tensor_topology(graph: Any) -> Any:
    """Apply existing ProcessGraph names to the three structural AST nodes."""

    _record_numeric_annotation_descriptors(graph)
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        function_table = FunctionTable()
        graph.function_table = function_table
    external_function_table = getattr(
        graph,
        "external_function_table",
        None,
    )
    if external_function_table is None:
        external_function_table = ExternalFunctionTable()
        graph.external_function_table = external_function_table

    function_nodes: dict[int, Any] = {}
    function_return_values: dict[int, list[int]] = {}
    call_owners: dict[int, Any] = {}
    method_owners: dict[int, str] = {}
    class_definitions: dict[str, ast.ClassDef] = {}
    external_owner_classes: dict[str, type] = {}
    lexical_parent_by_function: dict[int, int] = {}
    function_definitions = {
        int(node_id): node_data.get("expr_obj")
        for node_id, node_data in graph.G.nodes(data=True)
        if is_runnable_definition(node_data.get("expr_obj"))
    }

    class _DirectNestedFunctionVisitor(ast.NodeVisitor):
        def __init__(self, owner_id: int):
            self.owner_id = int(owner_id)

        def visit_FunctionDef(self, node):
            lexical_parent_by_function[id(node)] = self.owner_id

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            lexical_parent_by_function[id(node)] = self.owner_id

        def visit_ClassDef(self, node):
            # A class body is not itself a closure scope, but methods of a
            # class defined inside a function can close directly over that
            # function. ``interchange_reduction_loops`` does exactly this:
            # ``_RegisterBlockRewriter.visit_For`` writes the outer
            # ``state: dict``. Skipping the class body left the method with no
            # lexical parent, so closure aggregate discovery rebuilt ``state``
            # from its live Python value as ``static:state``. The indexed
            # store then disagreed with the authored dict identity and failed
            # during lexical reduction. Record methods against the enclosing
            # function; their own nested definitions are visited later from
            # the method's function record.
            for member in node.body:
                if isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
                ):
                    lexical_parent_by_function[id(member)] = self.owner_id
                elif isinstance(member, ast.ClassDef):
                    self.visit_ClassDef(member)
            return None

    for owner_id, definition in function_definitions.items():
        if not isinstance(
            definition, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        visitor = _DirectNestedFunctionVisitor(owner_id)
        for body_member in definition.body:
            visitor.visit(body_member)

    def local_aggregate_kinds(definition: Any) -> dict[str, str]:
        """Return aggregate storage owned by one lexical scope.

        Parameters are part of that scope's storage contract too. A nested
        function captures the outer parameter object, not a new untyped host
        value. ``_bind_sequence_storage_members.bind`` captures the annotated
        ``storage_bindings: dict[int, int]`` parameter and mutates it. Omitting
        annotated parameters here made that exact mapping reappear as a static
        Python closure value at the nested boundary.
        """

        if not isinstance(
            definition, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return {}
        def annotated_kind(annotation: ast.AST | None) -> str | None:
            if isinstance(annotation, ast.Name) and annotation.id in {
                "list", "set", "dict", "tuple", "bytes", "bytearray",
            }:
                return annotation.id
            if isinstance(annotation, ast.Subscript):
                return annotated_kind(annotation.value)
            if isinstance(annotation, ast.BinOp) and isinstance(
                annotation.op, ast.BitOr,
            ):
                alternatives = tuple(filter(None, (
                    annotated_kind(annotation.left),
                    annotated_kind(annotation.right),
                )))
                return (
                    alternatives[0]
                    if alternatives and len(set(alternatives)) == 1
                    else None
                )
            return None

        kinds: dict[str, str] = {}
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        ):
            kind = annotated_kind(argument.annotation)
            if kind is not None:
                kinds[argument.arg] = kind
        pending = list(reversed(definition.body))
        while pending:
            member = pending.pop()
            if isinstance(
                member,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            if isinstance(member, (ast.Assign, ast.AnnAssign)):
                value = member.value
                kind = None
                if isinstance(value, (ast.List, ast.ListComp)):
                    kind = "list"
                elif isinstance(value, (ast.Set, ast.SetComp)):
                    kind = "set"
                elif isinstance(value, (ast.Dict, ast.DictComp)):
                    kind = "dict"
                elif isinstance(value, ast.Tuple):
                    kind = "tuple"
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in {
                        "list", "set", "dict", "tuple", "bytes", "bytearray"
                    }
                ):
                    kind = value.func.id
                targets = (
                    tuple(member.targets)
                    if isinstance(member, ast.Assign)
                    else (member.target,)
                )
                if kind is not None:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            kinds[target.id] = kind
            pending.extend(reversed(tuple(source_child_nodes(member))))
        return kinds

    local_aggregate_kinds_by_function = {
        definition_id: local_aggregate_kinds(definition)
        for definition_id, definition in function_definitions.items()
    }

    def closure_aggregate_kinds(definition_id: int) -> dict[str, str]:
        """Aggregate identities inherited from lexical parent scopes."""

        inherited: dict[str, str] = {}
        parent_id = lexical_parent_by_function.get(definition_id)
        ancestry = []
        while parent_id is not None:
            ancestry.append(parent_id)
            parent_id = lexical_parent_by_function.get(parent_id)
        for ancestor_id in reversed(ancestry):
            inherited.update(
                local_aggregate_kinds_by_function.get(ancestor_id, {})
            )
        return inherited
    for _node_id, node_data in graph.G.nodes(data=True):
        class_definition = node_data.get("expr_obj")
        if isinstance(
            class_definition,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            source_identity = getattr(
                class_definition,
                "_python_source_identity",
                None,
            )
            qualified = (
                str(source_identity[1])
                if (
                    isinstance(source_identity, tuple)
                    and len(source_identity) == 2
                )
                else ""
            )
            parts = tuple(
                part for part in qualified.split(".") if part != "<locals>"
            )
            if "<locals>" not in qualified and len(parts) >= 2:
                method_owners[id(class_definition)] = parts[-2]
                module_name = str(source_identity[0])
                try:
                    module = importlib.import_module(module_name)
                    owner_class = getattr(module, parts[-2])
                except (ImportError, AttributeError, TypeError):
                    owner_class = None
                if inspect.isclass(owner_class):
                    external_owner_classes[parts[-2]] = owner_class
        if not isinstance(class_definition, ast.ClassDef):
            continue
        class_definitions[class_definition.name] = class_definition
        class_source_identity = getattr(
            class_definition, "_python_source_identity", None
        )
        if (
            isinstance(class_source_identity, tuple)
            and len(class_source_identity) == 2
        ):
            module_name, qualified_name = map(str, class_source_identity)
            target: Any = None
            try:
                target = importlib.import_module(module_name)
                for part in qualified_name.split("."):
                    if part == "<locals>":
                        target = None
                        break
                    target = getattr(target, part)
            except (ImportError, AttributeError, TypeError):
                target = None
            if inspect.isclass(target):
                external_owner_classes[class_definition.name] = target
        for member in class_definition.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_owners[id(member)] = class_definition.name
    class_field_aggregate_kinds: dict[tuple[str, str], str] = {
        (str(owner), str(field)): str(kind)
        for (owner, field), kind in dict(
            graph.G.graph.get("external_class_field_aggregate_kinds") or {}
        ).items()
    }
    # (owner, field) -> the source-local class an instance field HOLDS.
    # ``self.ordnance = OrdnanceField()`` in a method of the owner, or
    # ``ordnance: OrdnanceField`` declared on the class, states it. A field
    # read ``self.ordnance`` then yields an object of that class, and the
    # method call ``self.ordnance.step(...)`` resolves its ``method_ref``
    # through the class table exactly as ``self.step(...)`` already does --
    # which is what makes it a source-linked call the callee owns instead
    # of an opaque state effect no model can be formed for. Recorded only
    # when every assignment names ONE class; two classes for one field is
    # not a fact this table may state.
    class_field_classes: dict[tuple[str, str], str] = {}
    contested_field_classes: set[tuple[str, str]] = set()

    def note_field_class(owner: str, field: str, class_name: str) -> None:
        key = (str(owner), str(field))
        if key in contested_field_classes:
            return
        known = class_field_classes.get(key)
        if known is None:
            class_field_classes[key] = str(class_name)
        elif known != str(class_name):
            del class_field_classes[key]
            contested_field_classes.add(key)

    for owner_name, class_definition in class_definitions.items():
        for member in class_definition.body:
            if (
                isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
                and isinstance(member.annotation, ast.Name)
                and member.annotation.id in class_definitions
            ):
                note_field_class(
                    owner_name, member.target.id, member.annotation.id
                )
                continue
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in source_walk(member):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id in {"self", "cls"}
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in class_definitions
                ):
                    note_field_class(
                        owner_name, node.targets[0].attr, node.value.func.id
                    )
    class_field_mapping_contracts: dict[
        tuple[str, str], dict[str, Any]
    ] = {
        (str(owner), str(field)): {
            "mapping_key_dtype": "int64",
            "mapping_value_dtype": str(
                receipt.get("dtype") or "unknown"
            ),
            **({
                "mapping_value_tensor": True,
                "mapping_value_python_type": "AbstractTensor",
            } if receipt.get("value_tensor") else {}),
            **({
                "mapping_value_record": str(receipt["value_record"]),
            } if receipt.get("value_record") is not None else {}),
            "mapping_value_optional": bool(
                receipt.get("value_optional", False)
            ),
        }
        for (owner, field), receipt in dict(
            graph.G.graph.get("declared_class_field_contracts") or {}
        ).items()
        if str(receipt.get("storage") or "") == "keyed"
    }
    class_field_sequence_dtypes: dict[
        tuple[str, str], tuple[str, ...]
    ] = {}

    def _annotation_storage(annotation: ast.AST) -> dict[str, Any]:
        """Return the exact scalar/record handle stated by one annotation."""

        if isinstance(annotation, ast.BinOp) and isinstance(
            annotation.op, ast.BitOr
        ):
            concrete = tuple(
                member for member in (annotation.left, annotation.right)
                if not (
                    isinstance(member, ast.Constant)
                    and member.value is None
                )
                and not (
                    isinstance(member, ast.Name) and member.id == "None"
                )
            )
            if len(concrete) == 1:
                return {
                    **_annotation_storage(concrete[0]),
                    "optional": True,
                }
        name = (
            annotation.id if isinstance(annotation, ast.Name)
            else annotation.attr if isinstance(annotation, ast.Attribute)
            else ""
        )
        dtype = {
            "bool": "bool", "int": "int64", "float": "float64",
        }.get(str(name))
        if dtype is not None:
            return {"dtype": dtype}
        if tensor_annotation_identity(
            annotation, getattr(graph, "python_bindings", {}) or {},
        ) is not None:
            return {
                "dtype": "unknown",
                "tensor": True,
                "python_type": "AbstractTensor",
            }
        if name:
            # Repository records cross table boundaries by a deterministic
            # integer row handle; their physical fields remain described by
            # the record table rather than being collapsed into this scalar.
            return {"dtype": "int64", "record": str(name)}
        return {}

    def _mapping_annotation(annotation: ast.AST | None) -> dict[str, Any]:
        if not isinstance(annotation, ast.Subscript):
            return {}
        container = (
            annotation.value.id
            if isinstance(annotation.value, ast.Name)
            else annotation.value.attr
            if isinstance(annotation.value, ast.Attribute)
            else ""
        )
        if str(container).casefold() not in {"dict", "mapping", "mutablemapping"}:
            return {}
        columns = (
            tuple(annotation.slice.elts)
            if isinstance(annotation.slice, ast.Tuple)
            else (annotation.slice,)
        )
        if len(columns) != 2:
            return {}
        key = _annotation_storage(columns[0])
        value = _annotation_storage(columns[1])
        if key.get("dtype") is None or value.get("dtype") is None:
            return {}
        return {
            "mapping_key_dtype": str(key["dtype"]),
            "mapping_value_dtype": str(value["dtype"]),
            **({
                "mapping_value_tensor": True,
                "mapping_value_python_type": "AbstractTensor",
            } if value.get("tensor") else {}),
            **({
                "mapping_value_record": str(value["record"]),
            } if value.get("record") is not None else {}),
            "mapping_value_optional": bool(value.get("optional", False)),
        }

    def _sequence_annotation_dtypes(
        annotation: ast.AST | str | Any,
    ) -> tuple[str, ...]:
        if not isinstance(annotation, ast.AST):
            try:
                annotation = ast.parse(str(annotation), mode="eval").body
            except SyntaxError:
                return ()
        if not isinstance(annotation, ast.Subscript):
            return ()
        container = (
            annotation.value.id
            if isinstance(annotation.value, ast.Name)
            else annotation.value.attr
            if isinstance(annotation.value, ast.Attribute)
            else ""
        )
        if container not in {
            "Sequence", "Iterable", "Collection", "List", "list",
            "Tuple", "tuple", "Set", "set", "FrozenSet", "frozenset",
        }:
            return ()
        columns = (
            tuple(annotation.slice.elts)
            if isinstance(annotation.slice, ast.Tuple)
            else (annotation.slice,)
        )
        if (
            len(columns) == 2
            and isinstance(columns[1], ast.Constant)
            and columns[1].value is Ellipsis
        ):
            columns = columns[:1]
        if len(columns) != 1:
            return ()
        storage = _annotation_storage(columns[0])
        dtype = storage.get("dtype")
        return (str(dtype),) if dtype is not None else ()

    # AST ingestion normalizes an annotated assignment into its executable
    # assignment form, but MapIR deliberately retains the authored annotation
    # as schema data.  Read that durable copy; it is the exact cross-function
    # source of truth for fields declared in ``__init__``.
    for class_record in tuple(
        (graph.G.graph.get("map_ir") or {}).get("objects") or ()
    ):
        class_name = str(class_record.get("class_identity") or "")
        for field_record in tuple(class_record.get("attributes") or ()):
            annotation = field_record.get("annotation")
            if not class_name or not annotation:
                continue
            try:
                annotation_node = ast.parse(
                    str(annotation), mode="eval"
                ).body
            except SyntaxError:
                continue
            contract = _mapping_annotation(annotation_node)
            if contract:
                class_field_mapping_contracts[(
                    class_name, str(field_record.get("name") or ""),
                )] = contract
            dtypes = _sequence_annotation_dtypes(annotation_node)
            if dtypes:
                class_field_sequence_dtypes[(
                    class_name, str(field_record.get("name") or ""),
                )] = dtypes

    for class_name, definition in class_definitions.items():
        for member in definition.body:
            candidates: list[ast.AnnAssign] = []
            if isinstance(member, ast.AnnAssign):
                candidates.append(member)
            elif isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates.extend(
                    node for node in ast.walk(member)
                    if isinstance(node, ast.AnnAssign)
                )
            for declaration in candidates:
                field_name = (
                    declaration.target.id
                    if isinstance(declaration.target, ast.Name)
                    else declaration.target.attr
                    if (
                        isinstance(declaration.target, ast.Attribute)
                        and isinstance(declaration.target.value, ast.Name)
                        and declaration.target.value.id in {"self", "cls"}
                    )
                    else None
                )
                if field_name is None:
                    continue
                contract = _mapping_annotation(declaration.annotation)
                if contract:
                    class_field_mapping_contracts[
                        (str(class_name), str(field_name))
                    ] = contract
                dtypes = _sequence_annotation_dtypes(declaration.annotation)
                if dtypes:
                    class_field_sequence_dtypes[
                        (str(class_name), str(field_name))
                    ] = dtypes
    # The FunctionTable's retained method definitions are the authoritative
    # source bodies.  Some ingestion paths retain a skeletal ClassDef while
    # storing the complete method AST separately, so survey those definitions
    # as well instead of making the class wrapper a hidden prerequisite.
    for definition_id, class_name in method_owners.items():
        definition = function_definitions.get(int(definition_id))
        if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for declaration in (
            node for node in ast.walk(definition)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id in {"self", "cls"}
        ):
            contract = _mapping_annotation(declaration.annotation)
            if contract:
                class_field_mapping_contracts[
                    (str(class_name), str(declaration.target.attr))
                ] = contract
            dtypes = _sequence_annotation_dtypes(declaration.annotation)
            if dtypes:
                class_field_sequence_dtypes[
                    (str(class_name), str(declaration.target.attr))
                ] = dtypes

    # Imported dataclasses are legitimate authored record contracts even when
    # their class bodies are intentionally outside this compilation unit.
    # Read only declared dataclass metadata; do not instantiate the class or
    # invoke its factories.  The factory identity is enough to establish the
    # resident aggregate kind deterministically.
    for binding_name, bound_value in dict(
        getattr(graph, "python_bindings", {}) or {}
    ).items():
        if not isinstance(bound_value, type) or not is_dataclass(bound_value):
            continue
        for declared_field in fields(bound_value):
            factory = declared_field.default_factory
            aggregate_kind = next((
                kind.__name__
                for kind in (list, set, dict, tuple)
                if factory is kind
            ), None)
            if aggregate_kind is None:
                aggregate_kind = next((
                    kind.__name__
                    for kind in (list, set, dict, tuple)
                    if type(declared_field.default) is kind
                ), None)
            if aggregate_kind is not None:
                class_field_aggregate_kinds.setdefault(
                    (str(binding_name), str(declared_field.name)),
                    aggregate_kind,
                )
            dtypes = _sequence_annotation_dtypes(declared_field.type)
            if dtypes:
                class_field_sequence_dtypes.setdefault(
                    (str(binding_name), str(declared_field.name)), dtypes,
                )

    def aggregate_expression_kind(expression: ast.AST | None) -> str | None:
        if isinstance(expression, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(expression, (ast.Set, ast.SetComp)):
            return "set"
        if isinstance(expression, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(expression, ast.Tuple):
            return "tuple"
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"list", "set", "dict", "tuple"}
        ):
            return expression.func.id
        return None

    class_factory_aggregate_kinds: dict[tuple[str, str], str] = {}
    class_declared_field_aliases: set[tuple[str, str, str]] = set()
    class_field_value_aggregate_kinds: dict[tuple[str, str], str] = {}
    field_value_kind_conflicts: set[tuple[str, str]] = set()
    class_bases: dict[str, tuple[str, ...]] = {}
    for class_name, definition in class_definitions.items():
        class_bases[str(class_name)] = tuple(
            base.id for base in definition.bases if isinstance(base, ast.Name)
        )
        for member in definition.body:
            if not isinstance(member, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                tuple(member.targets)
                if isinstance(member, ast.Assign)
                else (member.target,)
            )
            value = member.value
            if isinstance(value, ast.Name):
                for target in targets:
                    if isinstance(target, ast.Name) and target.id != value.id:
                        class_declared_field_aliases.add((
                            str(class_name), str(target.id), str(value.id)
                        ))
            kind = (
                value.id
                if isinstance(value, ast.Name)
                and value.id in {"list", "set", "dict", "tuple"}
                else None
            )
            if kind is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    class_factory_aggregate_kinds[(
                        str(class_name), str(target.id)
                    )] = kind
    for owner_name, owner_class in external_owner_classes.items():
        # Pursued external methods may arrive without their containing
        # ClassDef. Read the exact class and base-class source declarations so
        # class-level factories remain compiler facts. No class is
        # instantiated and no factory is called.
        for source_class in reversed(owner_class.__mro__):
            try:
                source = textwrap.dedent(inspect.getsource(source_class))
                parsed = ast.parse(source)
            except (OSError, TypeError, SyntaxError, IndentationError):
                continue
            definition = next(
                (
                    member for member in parsed.body
                    if isinstance(member, ast.ClassDef)
                ),
                None,
            )
            if definition is None:
                continue
            for member in definition.body:
                if not isinstance(member, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    tuple(member.targets)
                    if isinstance(member, ast.Assign)
                    else (member.target,)
                )
                value = member.value
                if isinstance(value, ast.Name):
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id != value.id:
                            class_declared_field_aliases.add((
                                str(owner_name), str(target.id), str(value.id)
                            ))
                kind = (
                    value.id
                    if isinstance(value, ast.Name)
                    and value.id in {"list", "set", "dict", "tuple"}
                    else None
                )
                if kind is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        class_factory_aggregate_kinds[(
                            str(owner_name), str(target.id)
                        )] = kind
            for source_member in ast.walk(definition):
                if isinstance(source_member, ast.Assign):
                    targets = tuple(source_member.targets)
                    value = source_member.value
                elif isinstance(source_member, ast.AnnAssign):
                    targets = (source_member.target,)
                    value = source_member.value
                else:
                    continue
                outer_kind = aggregate_expression_kind(value)
                if (
                    outer_kind is None
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id in {"self", "cls"}
                ):
                    outer_kind = class_factory_aggregate_kinds.get((
                        str(owner_name), str(value.func.attr)
                    ))
                alias_source = (
                    str(value.attr)
                    if (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id in {"self", "cls"}
                    )
                    else None
                )
                for target in targets:
                    if not (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"self", "cls"}
                    ):
                        continue
                    if alias_source is not None:
                        pending = (
                            str(owner_name), str(target.attr), alias_source
                        )
                        if pending not in class_declared_field_aliases:
                            class_declared_field_aliases.add(pending)
                    if outer_kind is not None:
                        class_field_aggregate_kinds.setdefault(
                            (str(owner_name), str(target.attr)), outer_kind
                        )
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id in {"self", "cls"}
                ):
                    continue
                value_kind = class_factory_aggregate_kinds.get((
                    str(owner_name), str(value.func.attr)
                ))
                if value_kind is None:
                    continue
                for target in targets:
                    if not (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id in {"self", "cls"}
                    ):
                        continue
                    value_key = (
                        str(owner_name), str(target.value.attr)
                    )
                    previous = class_field_value_aggregate_kinds.setdefault(
                        value_key, value_kind
                    )
                    if previous != value_kind:
                        class_field_value_aggregate_kinds.pop(value_key, None)
                        field_value_kind_conflicts.add(value_key)
    changed = True
    while changed:
        changed = False
        for class_name, bases in class_bases.items():
            for base in bases:
                for (owner, factory), kind in tuple(
                    class_factory_aggregate_kinds.items()
                ):
                    if owner != base:
                        continue
                    key = (class_name, factory)
                    if key not in class_factory_aggregate_kinds:
                        class_factory_aggregate_kinds[key] = kind
                        changed = True

    field_kind_conflicts: set[tuple[str, str]] = set()
    class_field_aliases: dict[tuple[str, str], str] = {}
    pending_field_aliases: list[tuple[str, str, str]] = list(
        sorted(class_declared_field_aliases)
    )
    for class_name, definition in class_definitions.items():
        for member in definition.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for statement_member in source_walk(member):
                if isinstance(statement_member, ast.Assign):
                    targets = tuple(statement_member.targets)
                    value = statement_member.value
                elif isinstance(statement_member, ast.AnnAssign):
                    targets = (statement_member.target,)
                    value = statement_member.value
                else:
                    continue
                kind = aggregate_expression_kind(value)
                if kind is None and isinstance(value, ast.Name):
                    kind = local_aggregate_kinds_by_function.get(
                        id(member), {}
                    ).get(value.id)
                if (
                    kind is None
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id in {"self", "cls"}
                ):
                    kind = class_factory_aggregate_kinds.get((
                        str(class_name), str(value.func.attr)
                    ))
                alias_source = (
                    str(value.attr)
                    if (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id in {"self", "cls"}
                    )
                    else None
                )
                if alias_source is not None:
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id in {"self", "cls"}
                        ):
                            pending_field_aliases.append((
                                str(class_name),
                                str(target.attr),
                                alias_source,
                            ))
                if kind is None and alias_source is not None:
                    kind = class_field_aggregate_kinds.get((
                        str(class_name), alias_source
                    ))
                for target in targets:
                    if not (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id in {"self", "cls"}
                    ):
                        continue
                    value_kind = aggregate_expression_kind(value)
                    if (
                        value_kind is None
                        and isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id in {"self", "cls"}
                    ):
                        value_kind = class_factory_aggregate_kinds.get((
                            str(class_name), str(value.func.attr)
                        ))
                    if value_kind is None:
                        continue
                    value_key = (
                        str(class_name), str(target.value.attr)
                    )
                    previous_value_kind = (
                        class_field_value_aggregate_kinds.setdefault(
                            value_key, value_kind
                        )
                    )
                    if previous_value_kind != value_kind:
                        class_field_value_aggregate_kinds.pop(value_key, None)
                        field_value_kind_conflicts.add(value_key)
                if kind is None:
                    continue
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"self", "cls"}
                    ):
                        key = (str(class_name), str(target.attr))
                        if key in field_kind_conflicts:
                            continue
                        previous = class_field_aggregate_kinds.setdefault(
                            key, kind
                        )
                        if previous != kind:
                            # Conflicting assignments deliberately erase the
                            # storage proof; no method spelling may restore it.
                            class_field_aggregate_kinds.pop(key, None)
                            field_kind_conflicts.add(key)
                        elif alias_source is not None:
                            class_field_aliases[key] = alias_source
    changed = True
    while changed:
        changed = False
        for class_name, alias, target in pending_field_aliases:
            alias_key = (class_name, alias)
            target_kind = class_field_aggregate_kinds.get((
                class_name, target
            ))
            if target_kind is None or alias_key in field_kind_conflicts:
                continue
            previous = class_field_aggregate_kinds.setdefault(
                alias_key, target_kind
            )
            if previous != target_kind:
                class_field_aggregate_kinds.pop(alias_key, None)
                class_field_aliases.pop(alias_key, None)
                field_kind_conflicts.add(alias_key)
                changed = True
            elif class_field_aliases.get(alias_key) != target:
                class_field_aliases[alias_key] = target
                changed = True
            target_value_kind = class_field_value_aggregate_kinds.get((
                class_name, target
            ))
            if (
                target_value_kind is not None
                and alias_key not in field_value_kind_conflicts
            ):
                value_kind_was_present = (
                    alias_key in class_field_value_aggregate_kinds
                )
                previous_value_kind = class_field_value_aggregate_kinds.setdefault(
                    alias_key, target_value_kind
                )
                if previous_value_kind != target_value_kind:
                    class_field_value_aggregate_kinds.pop(alias_key, None)
                    field_value_kind_conflicts.add(alias_key)
                elif not value_kind_was_present:
                    changed = True
    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not is_runnable_definition(statement):
            continue
        function_name = runnable_definition_name(statement)
        owner_name = method_owners.get(node_id)

        def lexical_qualified_name(definition_id: int) -> str:
            definition = function_definitions[definition_id]
            identity = getattr(definition, "_python_source_identity", None)
            if isinstance(identity, tuple) and len(identity) == 2:
                module_name, python_qualified = map(str, identity)
                return ".".join(
                    part
                    for part in (module_name, python_qualified)
                    if part
                )
            parent_id = lexical_parent_by_function.get(definition_id)
            if parent_id is not None:
                return (
                    f"{lexical_qualified_name(parent_id)}.<locals>."
                    f"{getattr(definition, 'name', function_name)}"
                )
            method_owner = method_owners.get(definition_id)
            return (
                f"{method_owner}.{getattr(definition, 'name', function_name)}"
                if method_owner is not None
                else str(getattr(definition, "name", function_name))
            )

        qualified_name = lexical_qualified_name(node_id)
        reference = function_table.declare(
            function_name,
            qualified_name=qualified_name,
            metadata={
                "source_type": type(statement).__name__,
                "source_node": node_id,
                **(
                    {
                        "process_graph_boundary": getattr(
                            statement,
                            "_process_graph_boundary",
                        )
                    }
                    if hasattr(statement, "_process_graph_boundary")
                    else {}
                ),
                **(
                    {
                        "host_ssa_module": getattr(
                            statement, "_linked_repository_ssa_module"
                        ),
                        "host_ssa_root": str(getattr(
                            statement, "_linked_repository_ssa_root"
                        )),
                        "host_ssa_outputs": dict(getattr(
                            statement, "_linked_repository_ssa_outputs", {}
                        )),
                        "host_repository_ssa_complete": True,
                        "host_machine_state_complete": False,
                        "host_ssa_blockers": (),
                        "host_ssa_hard_blockers": (),
                        "host_ssa_legalization_shortfalls": (),
                        "host_ssa_unresolved_dependencies": (),
                        "implementation_kind": "linked-repository-ssa",
                        "implementation_variants": ("repository-ssa",),
                    }
                    if hasattr(statement, "_linked_repository_ssa_module")
                    else {}
                ),
            },
        )
        boundary_callable = getattr(
            statement,
            "_process_graph_boundary_callable",
            None,
        )
        if boundary_callable is not None:
            function_table.resolve_callable(reference, boundary_callable)
        data.setdefault("attributes", {})[
            "function_ref"
        ] = reference.address
        function_nodes[node_id] = reference
        if isinstance(statement, ast.Lambda):
            # A lambda's body is one expression, not a list of statements,
            # and is itself the implicit return value -- no explicit
            # Return-shaped node exists to walk to. Python-only construct;
            # nothing here needs generalizing for another language.
            function_return_values[node_id] = [id(statement.body)]
        else:
            _record_owned_calls_and_returns(
                graph, reference, node_id, call_owners, function_return_values,
            )

    # Source-less host callables are not external runtime handlers.  Source
    # pursuit may attach a decompiled repository-SSA module to the exact Call;
    # give its root an ordinary FunctionTable reference so all later call and
    # module machinery sees the implementation as compiled interior code.
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call):
            continue
        host_module = getattr(expression, "_host_ssa_module", None)
        host_root = getattr(expression, "_host_ssa_root", None)
        if host_module is None or host_root is None:
            continue
        if isinstance(expression.func, ast.Name):
            local_name = expression.func.id
        elif isinstance(expression.func, ast.Attribute):
            local_name = expression.func.attr
        else:
            local_name = str(host_root)
        reference = function_table.declare(
            local_name,
            qualified_name=f"host-ssa.{host_root}",
            metadata={
                "host_ssa_module": host_module,
                "host_ssa_root": str(host_root),
                "host_ssa_blockers": tuple(getattr(
                    expression, "_host_ssa_blockers", ()
                )),
                "host_ssa_raw_blockers": tuple(getattr(
                    expression, "_host_ssa_raw_blockers", ()
                )),
                "host_ssa_hard_blockers": tuple(getattr(
                    expression, "_host_ssa_hard_blockers", ()
                )),
                "host_ssa_legalization_shortfalls": tuple(getattr(
                    expression, "_host_ssa_legalization_shortfalls", ()
                )),
                "host_machine_state_complete": bool(getattr(
                    expression, "_host_machine_state_complete", False
                )),
                "host_machine_bodies_complete": bool(getattr(
                    expression, "_host_machine_bodies_complete", False
                )),
                "host_dependency_context_complete": bool(getattr(
                    expression, "_host_dependency_context_complete", False
                )),
                "host_repository_ssa_complete": bool(getattr(
                    expression, "_host_repository_ssa_complete", False
                )),
                "host_uses_machine_state_dialect": bool(getattr(
                    expression, "_host_uses_machine_state_dialect", False
                )),
                "host_native_module": getattr(
                    expression, "_host_native_module", None
                ),
                "host_ssa_cache_key": getattr(
                    expression, "_host_ssa_cache_key", None
                ),
                "host_ssa_cache_path": getattr(
                    expression, "_host_ssa_cache_path", None
                ),
                "host_ssa_cache_hit": bool(getattr(
                    expression, "_host_ssa_cache_hit", False
                )),
                "host_ssa_library_cache_keys": tuple(getattr(
                    expression, "_host_ssa_library_cache_keys", ()
                )),
                "host_ssa_library_cache_paths": tuple(getattr(
                    expression, "_host_ssa_library_cache_paths", ()
                )),
                "host_ssa_dependency_edges": tuple(getattr(
                    expression, "_host_ssa_dependency_edges", ()
                )),
                "host_ssa_unresolved_dependencies": tuple(getattr(
                    expression, "_host_ssa_unresolved_dependencies", ()
                )),
                "implementation_kind": "decompiled-host-ssa",
                "implementation_variants": tuple(
                    variant for variant, available in (
                        ("repository-ssa", bool(getattr(
                            expression, "_host_repository_ssa_complete", False
                        ))),
                        ("machine-state-ssa", bool(getattr(
                            expression, "_host_machine_state_complete", False
                        )) and bool(getattr(
                            expression, "_host_uses_machine_state_dialect", False
                        ))),
                        ("retained-native-module", getattr(
                            expression, "_host_native_module", None
                        ) is not None),
                    ) if available
                ),
            },
        )
        data.setdefault("attributes", {}).update({
            "callee_ref": int(reference.address),
            "host_ssa_root": str(host_root),
            "host_ssa_cache_key": getattr(
                expression, "_host_ssa_cache_key", None
            ),
            "host_ssa_blocker_count": len(tuple(getattr(
                expression, "_host_ssa_blockers", ()
            ))),
            "host_repository_ssa_complete": bool(getattr(
                expression, "_host_repository_ssa_complete", False
            )),
            "host_ssa_dependency_count": len(tuple(getattr(
                expression, "_host_ssa_dependency_edges", ()
            ))),
            "host_ssa_unresolved_dependency_count": len(tuple(getattr(
                expression, "_host_ssa_unresolved_dependencies", ()
            ))),
        })

    def class_field_defaults(definition: ast.ClassDef) -> dict[str, Any]:
        """Retain literal class-field defaults as structural compiler facts."""

        def is_literal(value: Any) -> bool:
            if value is None or isinstance(
                value,
                (bool, bytes, complex, float, int, str),
            ):
                return True
            if isinstance(value, (tuple, list)):
                return all(is_literal(item) for item in value)
            if isinstance(value, dict):
                return all(
                    is_literal(key) and is_literal(item)
                    for key, item in value.items()
                )
            return False

        defaults: dict[str, Any] = {}
        for member in definition.body:
            if not (
                isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
                and member.value is not None
            ):
                continue
            try:
                value = ast.literal_eval(member.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if is_literal(value):
                defaults[member.target.id] = value
        return defaults

    graph.G.graph["class_table"] = {
        class_name: {
            "methods": {
                member.name: function_nodes[id(member)].address
                for member in definition.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(member) in function_nodes
            },
            "fields": tuple(
                dict.fromkeys((
                    *(
                        member.target.id
                        for member in definition.body
                        if isinstance(member, ast.AnnAssign)
                        and isinstance(member.target, ast.Name)
                    ),
                    *(
                        target.attr
                        for member in definition.body
                        if isinstance(
                            member,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        )
                        for target in source_walk(member)
                        if isinstance(target, ast.Attribute)
                        and isinstance(target.ctx, ast.Store)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"self", "cls"}
                    ),
                    *(
                        field_name
                        for (owner, field_name), _kind
                        in class_field_aggregate_kinds.items()
                        if owner == class_name
                    ),
                ))
            ),
            # An omitted dataclass/class field is not an unknown shell input.
            # Its source-level literal default is part of the class schema,
            # just like the field order.  Preserve only literal data here:
            # arbitrary Python objects and executable default factories remain
            # unresolved and must be represented by ordinary graph structure.
            "field_defaults": class_field_defaults(definition),
            "field_classes": {
                field_name: held_class
                for (owner, field_name), held_class
                in class_field_classes.items()
                if owner == class_name
            },
        }
        for class_name, definition in class_definitions.items()
    }
    # Class navigation owns Python's inherited method surface as well as the
    # methods written directly in each body. Retained numerical wrappers are
    # intentionally thin (for example ComplexRationalPrecision inherits its
    # arithmetic from _ComplexRationalBase), so omitting this relation loses
    # the exact dunder dependency even though both class definitions and the
    # base edge are already present in the graph. Copy only absent members;
    # an override on the derived class remains authoritative.
    inheritance_changed = True
    while inheritance_changed:
        inheritance_changed = False
        for class_name, bases in class_bases.items():
            descriptor = graph.G.graph["class_table"].get(class_name)
            if descriptor is None:
                continue
            for base_name in bases:
                base_descriptor = graph.G.graph["class_table"].get(base_name)
                if base_descriptor is None:
                    continue
                for method_name, method_reference in dict(
                    base_descriptor.get("methods") or {}
                ).items():
                    if method_name not in descriptor["methods"]:
                        descriptor["methods"][method_name] = method_reference
                        inheritance_changed = True
                inherited_fields = tuple(dict.fromkeys((
                    *tuple(base_descriptor.get("fields") or ()),
                    *tuple(descriptor.get("fields") or ()),
                )))
                if inherited_fields != tuple(descriptor.get("fields") or ()):
                    descriptor["fields"] = inherited_fields
                    inheritance_changed = True
                for metadata_name in ("field_defaults", "field_classes"):
                    inherited = {
                        **dict(base_descriptor.get(metadata_name) or {}),
                        **dict(descriptor.get(metadata_name) or {}),
                    }
                    if inherited != dict(descriptor.get(metadata_name) or {}):
                        descriptor[metadata_name] = inherited
                        inheritance_changed = True
    for function_node_id, owner_name in method_owners.items():
        reference = function_nodes.get(function_node_id)
        if reference is None:
            continue
        descriptor = graph.G.graph["class_table"].setdefault(
            owner_name,
            {"methods": {}, "fields": (), "field_defaults": {}},
        )
        descriptor["fields"] = tuple(dict.fromkeys((
            *tuple(descriptor.get("fields") or ()),
            *(
                field_name
                for (owner, field_name), _kind
                in class_field_aggregate_kinds.items()
                if owner == owner_name
            ),
        )))
        function_definition = graph.G.nodes[function_node_id].get(
            "expr_obj"
        )
        if isinstance(
            function_definition,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            descriptor["methods"][function_definition.name] = int(
                reference.address
            )
    method_owner_by_reference = {
        int(reference.address): method_owners.get(function_node_id)
        for function_node_id, reference in function_nodes.items()
        if method_owners.get(function_node_id) is not None
    }
    returned_class_by_reference: dict[int, str] = {}
    def returned_class(definition: Any) -> str | None:
        if isinstance(definition, int) and definition in graph.G:
            definition = graph.G.nodes[definition].get("expr_obj")
        if not isinstance(
            definition, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return None
        known_record_classes = {
            str(owner) for owner, _field in class_field_aggregate_kinds
        }
        declared_spelling = (
            definition.returns.id
            if isinstance(definition.returns, ast.Name)
            else definition.returns.value
            if isinstance(definition.returns, ast.Constant)
            and isinstance(definition.returns.value, str)
            else None
        )
        declared_class = (
            str(declared_spelling)
            if declared_spelling is not None
            and (
                str(declared_spelling) in graph.G.graph["class_table"]
                or str(declared_spelling) in known_record_classes
            )
            else None
        )
        returned_classes = {
            returned.func.id
            for returned in (
                node.value for node in ast.walk(definition)
                if isinstance(node, ast.Return) and node.value is not None
            )
            if isinstance(returned, ast.Call)
            and isinstance(returned.func, ast.Name)
            and returned.func.id in graph.G.graph["class_table"]
        }
        inferred_class = declared_class or (
            next(iter(returned_classes))
            if len(returned_classes) == 1 else None
        )
        return None if inferred_class is None else str(inferred_class)

    for function_node_id, reference in function_nodes.items():
        definition = graph.G.nodes.get(function_node_id, {}).get("expr_obj")
        inferred_class = returned_class(definition)
        if inferred_class is not None:
            returned_class_by_reference[int(reference.address)] = inferred_class
    for entry in function_table:
        inferred_class = returned_class(
            entry.metadata.get("source_node")
        )
        if inferred_class is not None:
            returned_class_by_reference[int(entry.reference.address)] = (
                inferred_class
            )

    def propagate_returned_receiver_types(target_graph: Any) -> None:
        target_class_table = graph.G.graph.get("class_table", {})
        calls = tuple(target_graph.nodes(data=True))
        for _node_id, call_data in calls:
            expression = call_data.get("expr_obj")
            if not isinstance(expression, ast.Call):
                continue
            attributes = call_data.setdefault("attributes", {})
            if attributes.get("python_precision_boundary"):
                continue
            # Method calls can already have their exact source reference even
            # when the later generic call-correlation pass has not mirrored
            # it into ``callee_ref`` yet.  Both fields name the same function
            # table entry; refusing ``method_ref`` here loses the returned
            # object's class and therefore every field-storage contract on a
            # value such as ``self.fresh_value().accounting``.
            callee_reference = attributes.get(
                "callee_ref", attributes.get("method_ref")
            )
            returned_class = (
                None if callee_reference is None
                else returned_class_by_reference.get(int(callee_reference))
            )
            if returned_class is not None:
                attributes["result_class_ref"] = returned_class
        for _node_id, call_data in calls:
            expression = call_data.get("expr_obj")
            attributes = call_data.setdefault("attributes", {})
            if (
                not isinstance(expression, ast.Call)
                or not isinstance(expression.func, ast.Attribute)
                or attributes.get("python_precision_boundary")
                or attributes.get("method_ref") is not None
            ):
                continue
            receiver_id = next((
                int(parent)
                for parent, role in call_data.get("parents", ())
                if str(role) in {"operand", "value", "base", "object"}
            ), None)
            receiver_class = (
                None if receiver_id is None
                else _resolved_source_value_class(target_graph, receiver_id)[0]
            )
            method_reference = (
                target_class_table.get(str(receiver_class), {})
                .get("methods", {})
                .get(expression.func.attr)
            )
            if method_reference is None:
                continue
            method_reference = int(method_reference)
            attributes.update({
                "method_ref": method_reference,
                "callee_ref": method_reference,
            })
            accessor_id = id(expression.func)
            if accessor_id in target_graph:
                target_graph.nodes[accessor_id].setdefault(
                    "attributes", {}
                    ).update({
                        "accessor_kind": "method",
                        "method_ref": method_reference,
                    })

    binary_special_methods = {
        ast.Add: ("__add__", "__radd__"),
        ast.Sub: ("__sub__", "__rsub__"),
        ast.Mult: ("__mul__", "__rmul__"),
        ast.Div: ("__truediv__", "__rtruediv__"),
        ast.FloorDiv: ("__floordiv__", "__rfloordiv__"),
        ast.Mod: ("__mod__", "__rmod__"),
        ast.Pow: ("__pow__", "__rpow__"),
        ast.MatMult: ("__matmul__", "__rmatmul__"),
        ast.BitAnd: ("__and__", "__rand__"),
        ast.BitOr: ("__or__", "__ror__"),
        ast.BitXor: ("__xor__", "__rxor__"),
        ast.LShift: ("__lshift__", "__rlshift__"),
        ast.RShift: ("__rshift__", "__rrshift__"),
    }
    unary_special_methods = {
        ast.USub: "__neg__",
        ast.UAdd: "__pos__",
        ast.Invert: "__invert__",
    }

    def specialize_concorded_same_type_numeric_operator(
        target_wrapper: Any,
    ) -> bool:
        """Bind a generic wrapper dunder to its authored same-type body.

        A retained dunder is normalized before call edges have specialized its
        ``self`` and ``other`` formals.  The source-AST specialization can
        therefore cover an annotated root expression but not an operator
        reached later through another wrapper's component algebra.  Once both
        formals carry the same concorded numeric descriptor, the generic
        ``_composed_binary(op, left, right)`` promotion is provably redundant:
        its own eager contract dispatches that exact case to ``_binary_same``.
        Rebind the already-existing call and constant operation argument; do
        not synthesize algebra or infer a type from the dunder spelling.
        """

        target_graph = target_wrapper.G
        scope = _source_numeric_scope(target_graph)
        method_name = str(target_graph.graph.get("function_name") or "")
        if method_name not in {
            "__add__", "__radd__", "__sub__", "__rsub__",
            "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        }:
            return False
        inputs = {
            str(attributes.get("binding_name")): int(node_id)
            for node_id, data in target_graph.nodes(data=True)
            for attributes in (data.get("attributes") or {},)
            if data.get("type") == "Input"
            and attributes.get("binding_name") is not None
        }
        if not {"self", "other"} <= set(inputs):
            return False
        self_descriptor = _concorded_numeric_descriptor(
            target_graph, inputs["self"]
        )
        other_descriptor = _concorded_numeric_descriptor(
            target_graph, inputs["other"]
        )
        if self_descriptor is None or self_descriptor != other_descriptor:
            return False
        class_table = graph.G.graph.get("class_table", {})
        method_reference = (
            class_table.get(self_descriptor.type_name, {})
            .get("methods", {})
            .get("_binary_same")
        )
        if method_reference is None:
            return False
        changed = False
        page = current_identity_book().page(
            "source_numeric_operator_specialization_concordance"
        )
        for node_id, data in tuple(target_graph.nodes(data=True)):
            attributes = data.setdefault("attributes", {})
            if attributes.get("static_python_reference") != "_composed_binary":
                continue
            actuals = {
                str(role): int(parent)
                for parent, role in data.get("parents", ())
                if int(parent) in target_graph
            }
            operation_id = actuals.get("arg:0")
            receiver_id = actuals.get("arg:1")
            other_id = actuals.get("arg:2")
            if None in (operation_id, receiver_id, other_id):
                continue
            receiver_descriptor = _concorded_numeric_descriptor(
                target_graph, receiver_id
            )
            argument_descriptor = _concorded_numeric_descriptor(
                target_graph, other_id
            )
            if receiver_descriptor != self_descriptor or (
                argument_descriptor != self_descriptor
            ):
                continue
            fact = (
                int(receiver_id), int(other_id), int(operation_id),
                self_descriptor.receipt(), int(method_reference),
            )
            row = (scope, int(node_id))
            incumbent = page.latest(row)
            if incumbent is not None and tuple(incumbent) != fact:
                raise ValueError(
                    "source numeric operator specialization concordance "
                    f"disagreement for {row!r}: recorded={incumbent!r}, "
                    f"proposed={fact!r}"
                )
            if incumbent is None:
                page.set(row, 0, fact)
            for field in ("static_python_reference", "reference_kind"):
                attributes.pop(field, None)
            attributes.update({
                "method_ref": int(method_reference),
                "callee_ref": int(method_reference),
                "numeric_same_type_specialization": True,
                "numeric_feature_descriptor": self_descriptor.receipt(),
                "result_class_ref": self_descriptor.type_name,
                "precision_limbs": int(self_descriptor.limbs),
            })
            data["type"] = "Call"
            data["op"] = "call"
            _replace_inputs(target_wrapper, int(node_id), (
                (int(receiver_id), "operand"),
                (int(other_id), "arg:0"),
                (int(operation_id), "arg:1"),
            ))
            _concord_source_value_class(
                scope, int(node_id), self_descriptor.type_name,
                precision_limbs=self_descriptor.limbs,
                source="concorded same-type numeric operator specialization",
            )
            changed = True
        return changed

    def resolve_concorded_receiver_constructor(target_wrapper: Any) -> bool:
        """Bind ``type(receiver)(...)`` to the receiver's concorded class.

        ``type(self)`` names no class lexically; the receiver's class is the
        source-value class concordance fact the call edges already committed
        for ``self``.  Once that fact exists the call is an ordinary
        constructor of that class: stamp ``class_ref`` (which call planning
        resolves through the class table's ``__new__``/``__init__``), feed it
        only its authored arguments, and commit the constructed value's class
        at the receiver's width.  Nothing is inferred from a spelling.
        """

        target_graph = target_wrapper.G
        scope = _source_numeric_scope(target_graph)
        class_table = graph.G.graph.get("class_table", {})
        class_page = current_identity_book().page(
            "source_value_class_concordance"
        )
        page = current_identity_book().page(
            "source_receiver_constructor_concordance"
        )
        changed = False
        for node_id, data in tuple(target_graph.nodes(data=True)):
            if str(data.get("type") or "") != "Call":
                continue
            attributes = data.setdefault("attributes", {})
            if attributes.get("class_ref") is not None:
                continue
            roles = {
                str(role): int(parent)
                for parent, role in data.get("parents", ())
                if int(parent) in target_graph
            }
            callee_id = roles.get("callee")
            if callee_id is None:
                continue
            callee = target_graph.nodes[callee_id]
            callee_roles = {
                str(role): int(parent)
                for parent, role in callee.get("parents", ())
                if int(parent) in target_graph
            }
            if (
                (callee.get("attributes") or {}).get(
                    "static_python_reference"
                ) != "type"
                or set(callee_roles) != {"callee", "arg:0"}
            ):
                continue
            receiver_id = callee_roles["arg:0"]
            concorded = class_page.latest((scope, int(receiver_id)))
            if not isinstance(concorded, tuple) or len(concorded) < 2:
                continue
            class_identity, limbs = str(concorded[0]), int(concorded[1])
            if class_identity not in class_table:
                continue
            fact = (int(receiver_id), class_identity, limbs)
            row = (scope, int(node_id))
            incumbent = page.latest(row)
            if incumbent is not None and tuple(incumbent) != fact:
                raise ValueError(
                    "source receiver constructor concordance disagreement "
                    f"for {row!r}: recorded={incumbent!r}, proposed={fact!r}"
                )
            if incumbent is None:
                page.set(row, 0, fact)
            attributes.update({
                "class_ref": class_identity,
                "result_class_ref": class_identity,
                "precision_limbs": limbs,
            })
            descriptor = numeric_feature_descriptor(
                class_identity + (f"[{limbs}]" if limbs > 1 else "")
            )
            if descriptor is not None:
                attributes["numeric_feature_descriptor"] = descriptor.receipt()
            _replace_inputs(target_wrapper, int(node_id), tuple(
                (parent, role)
                for parent, role in data.get("parents", ())
                if str(role) != "callee"
            ))
            _concord_source_value_class(
                scope, int(node_id), class_identity,
                precision_limbs=limbs,
                source="concorded receiver constructor",
            )
            changed = True
        return changed

    def propagate_numeric_field_projections(target_wrapper: Any) -> bool:
        """Carry exact composite coefficient types after call specialization.

        Lexical normalization can run before an unannotated method formal has
        received its caller's numeric descriptor. Revisit existing GetAttr
        edges after that transfer so nested fields inherit the same arbitrary
        Precision width. This is a monotone publication of the receiver-edge
        fact, not a reconstruction from the field spelling alone.
        """

        target_graph = target_wrapper.G
        scope = _source_numeric_scope(target_graph)
        changed = False
        settling = True
        while settling:
            settling = False
            for node_id, data in tuple(target_graph.nodes(data=True)):
                expression = data.get("expr_obj")
                if not (
                    isinstance(expression, ast.Call)
                    and isinstance(expression.func, ast.Attribute)
                ):
                    continue
                method = str(expression.func.attr)
                receiver_id = next((
                    int(parent)
                    for parent, role in data.get("parents", ())
                    if str(role) in {"operand", "value", "object", "receiver"}
                    and int(parent) in target_graph
                ), None)
                if receiver_id is None:
                    continue
                receiver_attributes = (
                    target_graph.nodes[receiver_id].get("attributes") or {}
                )
                receiver_descriptor = _concorded_numeric_descriptor(
                    target_graph, receiver_id
                )
                if receiver_descriptor is None:
                    continue
                components = numeric_feature_method_components(
                    receiver_descriptor, method
                )
                if not components:
                    continue
                results = tuple(
                    {
                        "path": (
                            *tuple(receiver_attributes.get(
                                "numeric_component_path", ()
                            )),
                            *path,
                        ),
                        "descriptor": (
                            None if descriptor is None
                            else descriptor.receipt()
                        ),
                    }
                    for path, descriptor in components
                )
                attributes = data.setdefault("attributes", {})
                page = current_identity_book().page(
                    "source_numeric_intrinsic_concordance"
                )
                row = (scope, int(node_id))
                fact = (int(receiver_id), method, results)
                incumbent = page.latest(row)
                if incumbent is not None and tuple(incumbent) != fact:
                    raise ValueError(
                        "source numeric intrinsic concordance disagreement "
                        f"for {row!r}: recorded={incumbent!r}, "
                        f"proposed={fact!r}"
                    )
                if incumbent is None:
                    page.set(row, 0, fact)
                if attributes.get("numeric_component_results") != results:
                    attributes["numeric_component_results"] = results
                    changed = settling = True

            for node_id, data in tuple(target_graph.nodes(data=True)):
                if str(data.get("type") or "") != "Indexed":
                    continue
                parents = {
                    str(role): int(parent)
                    for parent, role in data.get("parents", ())
                    if int(parent) in target_graph
                }
                base_id, index_id = parents.get("base"), parents.get("index")
                if base_id is None or index_id is None:
                    continue
                results = tuple(
                    (target_graph.nodes[base_id].get("attributes") or {}).get(
                        "numeric_component_results", ()
                    )
                )
                index = (target_graph.nodes[index_id].get("attributes") or {}).get(
                    "value"
                )
                if not isinstance(index, int) or not (0 <= index < len(results)):
                    continue
                selected = dict(results[index])
                descriptor = _numeric_feature_descriptor_from_receipt(
                    selected.get("descriptor")
                )
                path = tuple(map(str, selected.get("path") or ()))
                attributes = data.setdefault("attributes", {})
                proposed = {
                    "numeric_component_path": path,
                    "numeric_component_receiver_id": int(base_id),
                    "numeric_component_attribute": f"result[{index}]",
                    "precision_limbs": (
                        int(descriptor.limbs) if descriptor is not None else 1
                    ),
                }
                if descriptor is not None:
                    proposed.update({
                        "numeric_feature_descriptor": descriptor.receipt(),
                        "result_class_ref": descriptor.type_name,
                    })
                    _concord_source_value_class(
                        scope, int(node_id), descriptor.type_name,
                        precision_limbs=descriptor.limbs,
                        source="numeric-intrinsic-result-projection",
                    )
                _concord_numeric_feature_projection(
                    scope, int(node_id), int(base_id), f"result[{index}]",
                    path, descriptor,
                )
                if any(attributes.get(key) != value
                       for key, value in proposed.items()):
                    attributes.update(proposed)
                    changed = settling = True

            for node_id, data in tuple(target_graph.nodes(data=True)):
                expression = data.get("expr_obj")
                attributes = data.setdefault("attributes", {})
                attribute = (
                    expression.attr
                    if isinstance(expression, ast.Attribute)
                    else attributes.get("attribute")
                    if str(data.get("type") or "") == "GetAttr"
                    else None
                )
                if attribute not in {
                    "real", "imag", "numerator", "denominator",
                }:
                    continue
                receiver_id = next((
                    int(parent)
                    for parent, role in data.get("parents", ())
                    if str(role) in {"value", "object", "base", "operand"}
                    and int(parent) in target_graph
                ), None)
                if receiver_id is None:
                    continue
                receiver_attributes = (
                    target_graph.nodes[receiver_id].get("attributes") or {}
                )
                receiver_descriptor = _concorded_numeric_descriptor(
                    target_graph, receiver_id
                )
                if receiver_descriptor is None:
                    continue
                structural = (
                    (attribute in {"real", "imag"}
                     and "complex" in receiver_descriptor.features)
                    or (attribute in {"numerator", "denominator"}
                        and "rational" in receiver_descriptor.features)
                )
                if not structural:
                    continue
                projected = numeric_feature_field_projection(
                    receiver_descriptor, str(attribute)
                )
                path = (
                    *tuple(receiver_attributes.get(
                        "numeric_component_path", ()
                    )),
                    str(attribute),
                )
                proposed = {
                    "numeric_component_path": path,
                    "numeric_component_receiver_id": int(receiver_id),
                    "numeric_component_attribute": str(attribute),
                    "precision_limbs": (
                        int(projected.limbs) if projected is not None else 1
                    ),
                }
                if projected is not None:
                    proposed.update({
                        "numeric_feature_descriptor": projected.receipt(),
                        "result_class_ref": projected.type_name,
                    })
                    _concord_source_value_class(
                        scope, int(node_id), projected.type_name,
                        precision_limbs=projected.limbs,
                        source="post-call numeric-component-projection",
                    )
                _concord_numeric_feature_projection(
                    scope, int(node_id), int(receiver_id), str(attribute),
                    path, projected,
                )
                if any(attributes.get(key) != value
                       for key, value in proposed.items()):
                    attributes.update(proposed)
                    changed = settling = True
        return changed

    def lower_class_operator_calls(
        target_wrapper: Any, *, settled: bool = False,
    ) -> None:
        """Bind authored operators through the receiver's class table.

        Lexical normalization has already replaced every Name occurrence by
        its exact producer at this seam.  Consequently the receiver edge, not
        a source spelling or an operand position guessed later, determines
        whether Python's class dispatch applies.  The concordance row commits
        that decision once for all later call planning and storage binding.
        """

        target_graph = target_wrapper.G
        scope = _source_numeric_scope(target_graph)
        class_table = graph.G.graph.get("class_table", {})
        page = current_identity_book().page(
            "source_operator_dispatch_concordance"
        )

        def descriptor(value_id: Any) -> tuple[str | None, int]:
            return _resolved_source_value_class(target_graph, value_id)

        def commit(
            node_id: int,
            receiver_id: int,
            class_identity: str,
            method_name: str,
            method_reference: int,
            reflected: bool,
        ) -> None:
            fact = (
                int(receiver_id), str(class_identity), str(method_name),
                int(method_reference), bool(reflected),
            )
            row = (scope, int(node_id))
            incumbent = page.latest(row)
            if incumbent is not None and tuple(incumbent) != fact:
                raise ValueError(
                    "source operator dispatch concordance disagreement for "
                    f"{row!r}: recorded={incumbent!r}, proposed={fact!r}"
                )
            if incumbent is None:
                page.set(row, 0, fact)

        changed = True
        while changed:
            changed = False
            for node_id, data in tuple(target_graph.nodes(data=True)):
                expression = data.get("expr_obj")
                attributes = data.setdefault("attributes", {})
                if (
                    attributes.get("class_operator_dispatch")
                    or attributes.get("python_precision_operator")
                ):
                    continue
                parents = {
                    str(role): int(parent)
                    for parent, role in data.get("parents", ())
                    if int(parent) in target_graph
                }
                receiver_id = other_id = None
                class_identity = method_name = None
                reflected = False
                if isinstance(expression, ast.BinOp):
                    methods = binary_special_methods.get(type(expression.op))
                    left_id = parents.get("lhs")
                    right_id = parents.get("rhs")
                    if methods is None or left_id is None or right_id is None:
                        continue
                    left_class, _left_limbs = descriptor(left_id)
                    right_class, _right_limbs = descriptor(right_id)
                    if left_class is None and not settled:
                        # The left operand decides first; while its type is
                        # still being settled, the reflected method is not
                        # yet the answer.
                        continue
                    candidates = (
                        (left_id, right_id, left_class, methods[0], False),
                        (right_id, left_id, right_class, methods[1], True),
                    )
                    selected = next((
                        candidate for candidate in candidates
                        if candidate[2] is not None
                        and class_table.get(str(candidate[2]), {})
                        .get("methods", {}).get(candidate[3]) is not None
                    ), None)
                    if selected is None:
                        continue
                    (
                        receiver_id, other_id, class_identity,
                        method_name, reflected,
                    ) = selected
                elif isinstance(expression, ast.UnaryOp):
                    method_name = unary_special_methods.get(type(expression.op))
                    receiver_id = parents.get("operand")
                    if method_name is None or receiver_id is None:
                        continue
                    class_identity, _limbs = descriptor(receiver_id)
                    if (
                        class_identity is None
                        or class_table.get(str(class_identity), {})
                        .get("methods", {}).get(method_name) is None
                    ):
                        continue
                else:
                    continue
                method_reference = int(
                    class_table[str(class_identity)]["methods"][method_name]
                )
                returned_class = returned_class_by_reference.get(
                    method_reference
                )
                numeric_descriptor = _concorded_numeric_descriptor(
                    target_graph, node_id
                )
                if returned_class is None and numeric_descriptor is not None:
                    returned_class = numeric_descriptor.type_name
                attributes.update({
                    "source_type": type(expression).__name__,
                    "class_operator_dispatch": True,
                    "operator_receiver_class": str(class_identity),
                    "operator_method": str(method_name),
                    "operator_reflected": bool(reflected),
                    "method_ref": method_reference,
                    "callee_ref": method_reference,
                })
                attributes.pop("numeric_composite_pending", None)
                if returned_class is not None:
                    attributes["result_class_ref"] = str(returned_class)
                    _concord_source_value_class(
                        scope,
                        int(node_id),
                        str(returned_class),
                        precision_limbs=max(
                            descriptor(receiver_id)[1],
                            descriptor(other_id)[1]
                            if other_id is not None else 1,
                        ),
                        source=f"operator method {method_name}",
                    )
                data["type"] = "Call"
                data["op"] = "call"
                call_inputs = [(int(receiver_id), "operand")]
                if other_id is not None:
                    call_inputs.append((int(other_id), "arg:0"))
                _replace_inputs(
                    target_wrapper, int(node_id), tuple(call_inputs)
                )
                commit(
                    int(node_id), int(receiver_id), str(class_identity),
                    str(method_name), method_reference, bool(reflected),
                )
                changed = True

    def lower_python_precision(target_wrapper: Any) -> None:
        """Lower the authored Precision surface to its repository operators.

        ``Precision.of`` and ``collapse`` are boundary declarations, not
        runtime calls into the eager Python implementation.  The former
        commits the class/width descriptor carried by that source occurrence;
        the latter consumes it.  Arithmetic between those boundaries is the
        existing ``precision_*`` vocabulary consumed by
        ``apply_precision_pipeline``.  This pass runs only after lexical
        normalization, so every descriptor follows the exact producer edge.
        """

        target_graph = target_wrapper.G
        scope = _source_numeric_scope(target_graph)

        def is_real_precision(identity: Any) -> bool:
            return (
                identity is not None
                and str(identity).rsplit(".", 1)[-1] == "Precision"
            )

        def concord_operator(
            node_id: int,
            operation: str,
            receiver_id: int,
            class_identity: str,
            limbs: int,
        ) -> None:
            """Commit the exact source operation carried by a wide value."""

            page = current_identity_book().page(
                "source_precision_operator_concordance"
            )
            row = (scope, int(node_id))
            proposed = (
                str(operation), int(receiver_id), str(class_identity),
                max(int(limbs or 1), 1),
            )
            incumbent = page.latest(row)
            if incumbent is not None and tuple(incumbent) != proposed:
                raise ValueError(
                    "source precision operator concordance disagreement for "
                    f"{row!r}: recorded={incumbent!r}, proposed={proposed!r}"
                )
            if incumbent is None:
                page.set(row, 0, proposed)

        for node_id, data in tuple(target_graph.nodes(data=True)):
            expression = data.get("expr_obj")
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
            ):
                continue
            attributes = data.setdefault("attributes", {})
            parents = tuple(data.get("parents", ()))
            if (
                expression.func.attr == "of"
                and is_real_precision(attributes.get(
                    "result_class_ref", attributes.get("class_ref")
                ))
            ):
                value_id = next((
                    int(parent) for parent, role in parents
                    if str(role) in {"arg:0", "arg0"}
                ), None)
                if value_id is None:
                    continue
                limbs = 2
                if len(expression.args) >= 2:
                    try:
                        limbs = max(int(ast.literal_eval(expression.args[1])), 1)
                    except (TypeError, ValueError, SyntaxError):
                        continue
                class_identity = str(attributes.get(
                    "result_class_ref", attributes.get("class_ref")
                ))
                attributes.update({
                    "precision_limbs": limbs,
                    "result_class_ref": class_identity,
                    "python_precision_boundary": "promote",
                })
                for field in (
                    "callee_ref", "method_ref", "constructor_ref", "class_ref",
                ):
                    attributes.pop(field, None)
                accessor_id = id(expression.func)
                if accessor_id in target_graph:
                    accessor_attributes = target_graph.nodes[
                        accessor_id
                    ].setdefault("attributes", {})
                    for field in (
                        "callee_ref", "method_ref", "bound_method_ref",
                    ):
                        accessor_attributes.pop(field, None)
                    accessor_attributes[
                        "python_precision_boundary"
                    ] = "promote-selector"
                data["type"] = "Cast"
                data["op"] = "Cast"
                _replace_inputs(
                    target_wrapper, int(node_id), ((value_id, "operand"),)
                )
                _concord_source_value_class(
                    scope, int(node_id), class_identity,
                    precision_limbs=limbs,
                    source="Precision.of boundary",
                )
                _concord_source_precision_boundary(
                    scope,
                    int(node_id),
                    "promote",
                    int(value_id),
                    precision_limbs=limbs,
                )

        precision_methods = {
            "sqrt": "Sqrt",
            "exp": "Exp",
            "log": "Log",
        }
        # A method can feed an operator and an operator can feed a method.
        # Settle both from the same producer-edge identities to a fixed point;
        # ordering these as two one-shot loops lost `(wide.exp() - 1)` between
        # the two and left the following collapse without a class descriptor.
        changed = True
        while changed:
            changed = False
            for node_id, data in tuple(target_graph.nodes(data=True)):
                expression = data.get("expr_obj")
                if not isinstance(expression, (ast.BinOp, ast.UnaryOp)):
                    continue
                attributes = data.setdefault("attributes", {})
                if attributes.get("python_precision_operator"):
                    continue
                parents = {
                    str(role): int(parent)
                    for parent, role in data.get("parents", ())
                    if int(parent) in target_graph
                }
                if isinstance(expression, ast.BinOp):
                    operand_ids = (
                        parents.get("lhs"), parents.get("rhs")
                    )
                    if None in operand_ids:
                        continue
                    descriptors = tuple(
                        _resolved_source_value_class(target_graph, value_id)
                        for value_id in operand_ids
                    )
                    precision = tuple(
                        item for item in descriptors
                        if is_real_precision(item[0])
                    )
                    if not precision:
                        continue
                    limbs = max(item[1] for item in precision)
                    operation = _qualified_handler(
                        "binop", expression.op, limbs=limbs
                    )
                else:
                    operand_id = parents.get("operand")
                    if operand_id is None:
                        continue
                    class_identity, limbs = _resolved_source_value_class(
                        target_graph, operand_id
                    )
                    if not is_real_precision(class_identity):
                        continue
                    operation = _qualified_handler(
                        "unaryop", expression.op, limbs=limbs
                    )
                class_identity = str(precision[0][0]) if isinstance(
                    expression, ast.BinOp
                ) else str(class_identity)
                data["type"] = operation
                data["op"] = operation
                attributes.update({
                    "python_precision_operator": True,
                    "precision_limbs": int(limbs),
                    "result_class_ref": class_identity,
                })
                _concord_source_value_class(
                    scope, int(node_id), class_identity,
                    precision_limbs=int(limbs),
                    source=f"Precision {type(expression).__name__}",
                )
                receiver_id = (
                    int(operand_ids[0]) if isinstance(expression, ast.BinOp)
                    else int(operand_id)
                )
                concord_operator(
                    int(node_id), str(operation), receiver_id,
                    class_identity, int(limbs),
                )
                changed = True
            for node_id, data in tuple(target_graph.nodes(data=True)):
                expression = data.get("expr_obj")
                attributes = data.setdefault("attributes", {})
                if (
                    attributes.get("python_precision_operator")
                    or not isinstance(expression, ast.Call)
                    or not isinstance(expression.func, ast.Attribute)
                    or expression.func.attr not in precision_methods
                ):
                    continue
                receiver_id = next((
                    int(parent) for parent, role in data.get("parents", ())
                    if str(role) in {"operand", "receiver", "value"}
                ), None)
                if receiver_id is None:
                    continue
                receiver_class, receiver_limbs = (
                    _resolved_source_value_class(target_graph, receiver_id)
                )
                if not is_real_precision(receiver_class):
                    continue
                source_operation = precision_methods[expression.func.attr]
                operation = PRECISION_SINGULAR_NAMES[source_operation]
                attributes.update({
                    "python_precision_operator": True,
                    "precision_limbs": int(receiver_limbs),
                    "result_class_ref": str(receiver_class),
                    "precision_source_operation": source_operation,
                })
                for field in (
                    "callee_ref", "method_ref", "constructor_ref",
                    "class_ref",
                ):
                    attributes.pop(field, None)
                accessor_id = id(expression.func)
                if accessor_id in target_graph:
                    accessor_attributes = target_graph.nodes[
                        accessor_id
                    ].setdefault("attributes", {})
                    for field in (
                        "callee_ref", "method_ref", "bound_method_ref",
                    ):
                        accessor_attributes.pop(field, None)
                    accessor_attributes["python_precision_operator"] = (
                        f"{expression.func.attr}-selector"
                    )
                data["type"] = operation
                data["op"] = operation
                _replace_inputs(
                    target_wrapper,
                    int(node_id),
                    ((int(receiver_id), "operand"),),
                )
                _concord_source_value_class(
                    scope, int(node_id), str(receiver_class),
                    precision_limbs=int(receiver_limbs),
                    source=f"Precision {expression.func.attr}",
                )
                concord_operator(
                    int(node_id), source_operation, int(receiver_id),
                    str(receiver_class), int(receiver_limbs),
                )
                changed = True

        for node_id, data in tuple(target_graph.nodes(data=True)):
            expression = data.get("expr_obj")
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "collapse"
            ):
                continue
            receiver_id = next((
                int(parent) for parent, role in data.get("parents", ())
                if str(role) == "operand"
            ), None)
            if receiver_id is None:
                continue
            receiver_class, receiver_limbs = (
                _resolved_source_value_class(target_graph, receiver_id)
            )
            if not is_real_precision(receiver_class):
                continue
            data.setdefault("attributes", {}).update({
                "python_precision_boundary": "collapse",
                "precision_limbs": int(receiver_limbs),
            })
            attributes = data["attributes"]
            for field in (
                "callee_ref", "method_ref", "constructor_ref", "class_ref",
            ):
                attributes.pop(field, None)
            accessor_id = id(expression.func)
            if accessor_id in target_graph:
                accessor_attributes = target_graph.nodes[
                    accessor_id
                ].setdefault("attributes", {})
                for field in (
                    "callee_ref", "method_ref", "bound_method_ref",
                ):
                    accessor_attributes.pop(field, None)
                accessor_attributes[
                    "python_precision_boundary"
                ] = "collapse-selector"
            data["type"] = "Cast"
            data["op"] = "Cast"
            _replace_inputs(
                target_wrapper,
                int(node_id),
                ((int(receiver_id), "operand"),),
            )
            _concord_source_precision_boundary(
                scope,
                int(node_id),
                "collapse",
                int(receiver_id),
                precision_limbs=int(receiver_limbs),
            )

    propagated_graphs = {id(graph.G)}
    propagate_returned_receiver_types(graph.G)
    for entry in function_table:
        entry_graph = getattr(getattr(entry, "graph", None), "G", None)
        if entry_graph is None or id(entry_graph) in propagated_graphs:
            continue
        propagated_graphs.add(id(entry_graph))
        propagate_returned_receiver_types(entry_graph)
    # Source pursuit has already made an exact definition->call dependency
    # edge. Preserve that identity directly in SSA call metadata instead of
    # attempting receiver-name inference again. This is especially important
    # for a method reached through an argument (``G.remove_node``): its owner
    # is proven by actual-argument provenance, not by the local spelling ``G``.
    for call_id, call_data in graph.G.nodes(data=True):
        expression = call_data.get("expr_obj")
        if not isinstance(expression, ast.Call):
            continue
        resolved_parent = (call_data.get("attributes") or {}).get(
            "resolved_ast_parent"
        )
        reference = function_nodes.get(int(resolved_parent)) if (
            resolved_parent is not None
        ) else None
        if reference is None:
            continue
        attributes = call_data.setdefault("attributes", {})
        attributes["callee_ref"] = int(reference.address)
        returned_class = returned_class_by_reference.get(
            int(reference.address)
        )
        if returned_class is not None:
            attributes["result_class_ref"] = returned_class
        if isinstance(expression.func, ast.Attribute):
            attributes["method_ref"] = int(reference.address)
            accessor_id = id(expression.func)
            if accessor_id in graph.G:
                graph.G.nodes[accessor_id].setdefault("attributes", {}).update({
                    "accessor_kind": "method",
                    "method_ref": int(reference.address),
                })

    # The exact callee references above are attached after lexical child
    # graphs have been normalized.  Re-run the returned-object propagation on
    # this graph now that those identities are available.
    propagate_returned_receiver_types(graph.G)

    contextual_requirements = list(
        graph.G.graph.get("contextual_requirements", ())
    )
    import_bindings: dict[
        str,
        tuple[str, str, dict[str, Any]],
    ] = {}
    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        requirement = {
            "kind": (
                "import_from"
                if isinstance(statement, ast.ImportFrom)
                else "import"
            ),
            "module": (
                statement.module
                if isinstance(statement, ast.ImportFrom)
                else None
            ),
            "level": (
                int(statement.level)
                if isinstance(statement, ast.ImportFrom)
                else 0
            ),
            "names": tuple(
                (imported.name, imported.asname)
                for imported in statement.names
            ),
        }
        if requirement not in contextual_requirements:
            contextual_requirements.append(requirement)
        data.setdefault("attributes", {})[
            "contextual_requirement"
        ] = requirement
        for imported in statement.names:
            local_name = imported.asname or imported.name.split(".")[0]
            if isinstance(statement, ast.ImportFrom):
                module = "." * int(statement.level) + (
                    statement.module or ""
                )
                qualified_name = (
                    f"{module}.{imported.name}"
                    if module
                    else imported.name
                )
            else:
                qualified_name = imported.name
            import_bindings[local_name] = (
                qualified_name,
                imported.name,
                requirement,
            )
            imported_id = id(imported)
            if imported_id in graph.G:
                graph.G.nodes[imported_id].setdefault("attributes", {})[
                    "contextual_requirement"
                ] = requirement
        logger.info(
            "retaining ProcessGraph import as a contextual requirement: %s",
            requirement,
        )
    graph.G.graph["contextual_requirements"] = tuple(
        contextual_requirements
    )
    static_bindings = dict(getattr(graph, "python_bindings", {}) or {})
    # Parent-source expansion is optional, but literal module constants are
    # required static bindings in either ingestion mode.  Recover only
    # assignments outside function/class bodies; locals remain SSA values.
    scoped_member_ids = {
        id(member)
        for _owner_id, owner_data in graph.G.nodes(data=True)
        for owner in (owner_data.get("expr_obj"),)
        if isinstance(
            owner,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        )
        for member in source_walk(owner)
        if member is not owner
    }
    for _node_id, node_data in graph.G.nodes(data=True):
        expression = node_data.get("expr_obj")
        if id(expression) in scoped_member_ids:
            continue
        if isinstance(expression, ast.Assign):
            targets = expression.targets
            value_node = expression.value
        elif isinstance(expression, ast.AnnAssign):
            targets = (expression.target,)
            value_node = expression.value
        else:
            continue
        if value_node is None:
            continue
        try:
            literal = ast.literal_eval(value_node)
        except (TypeError, ValueError, SyntaxError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                static_bindings[target.id] = literal
    python_package = getattr(graph, "python_package", None)
    for local_name, (
        qualified_name,
        imported_name,
        requirement,
    ) in import_bindings.items():
        try:
            if requirement["kind"] == "import_from":
                module_name = (
                    "." * int(requirement.get("level", 0))
                    + str(requirement.get("module") or "")
                )
                module = importlib.import_module(
                    module_name,
                    package=python_package,
                )
                static_bindings[local_name] = getattr(
                    module,
                    imported_name,
                )
            elif requirement["kind"] == "import":
                static_bindings[local_name] = importlib.import_module(
                    qualified_name,
                    package=python_package,
                )
        except (ImportError, AttributeError, TypeError, ValueError):
            # The contextual requirement remains available to deployment even
            # when it cannot be resolved in the compiler's current process.
            continue
    for builtin_name, value in vars(builtins).items():
        static_bindings.setdefault(builtin_name, value)
    graph.python_bindings = static_bindings
    function_parameters_by_address = {}
    for function_node_id, reference in function_nodes.items():
        definition = graph.G.nodes[function_node_id].get("expr_obj")
        if not isinstance(
            definition,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        function_parameters_by_address[reference.address] = {
            argument.arg
            for argument in (
                *definition.args.posonlyargs,
                *definition.args.args,
                *definition.args.kwonlyargs,
                *(
                    (definition.args.vararg,)
                    if definition.args.vararg is not None
                    else ()
                ),
                *(
                    (definition.args.kwarg,)
                    if definition.args.kwarg is not None
                    else ()
                ),
            )
        }
    lexical_functions_by_owner: dict[int, dict[str, Any]] = {}
    for child_id, parent_id in lexical_parent_by_function.items():
        child_reference = function_nodes.get(child_id)
        parent_reference = function_nodes.get(parent_id)
        child_definition = function_definitions.get(child_id)
        if (
            child_reference is None
            or parent_reference is None
            or not isinstance(
                child_definition,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ):
            continue
        lexical_functions_by_owner.setdefault(
            int(parent_reference.address), {}
        )[child_definition.name] = child_reference

    # Function extraction and structural rewrites can copy an authored Python
    # Attribute after its original ingestion overlay was applied. Reassert the
    # language special case at this AST-to-shared-graph seam.
    normalize_python_attribute_special_cases(graph)

    for _node_id, data in graph.G.nodes(data=True):
        source_type = data.get("type")
        process_type = _AST_PROCESS_GRAPH_ALIASES.get(source_type)
        if process_type is None:
            continue
        data["type"] = process_type
        data["op"] = process_type
        data.setdefault("attributes", {})["source_type"] = source_type

    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        # --- C (pycparser c_ast) canonicalization -------------------------
        # Dispatched on the type *name*, never isinstance, so this module
        # never imports pycparser (it is a frontend-optional dependency; a
        # pure-Python compile must not require it to be installed).
        #
        # This is the same universalization the Python branches below
        # perform, and it belongs here for the same reason: once a syntactic
        # element is canonicalized into the shared operator vocabulary
        # (Add/Load/Store/Call/Constant), no downstream pass needs to know
        # which language it came from. Skipping it is precisely what let raw
        # C node types survive all the way to the lexical-value pass.
        c_type_name = type(expression).__name__
        if expression is not None and c_type_name in _C_COMPILE_TIME_SYNTAX:
            # Pure type machinery, with no runtime value of its own: the
            # C counterpart of Python's Nonlocal/Global handling below.
            _remove_node(graph, node_id)
            continue
        if c_type_name == "Decl" and not isinstance(expression, ast.AST):
            # `int total = expr;` -- a declaration is not itself an
            # operation. Its initializer is the value the declared name
            # denotes, so the Decl collapses onto that value exactly the way
            # Python's own Name-store nodes collapse onto their producer.
            # A declaration with no initializer (`int total;`) names no
            # value at all yet and carries no runtime effect of its own.
            initializer = getattr(expression, "init", None)
            if initializer is not None and id(initializer) in graph.G:
                _redirect_value(graph, node_id, id(initializer))
            else:
                _remove_node(graph, node_id)
            continue
        if c_type_name == "Constant" and not isinstance(expression, ast.AST):
            value = _c_constant_value(expression)
            data["type"] = "Constant"
            data["op"] = "const"
            data["constant"] = value
            data.setdefault("attributes", {})["value"] = value
            continue
        if c_type_name == "BinaryOp":
            operation = _c_qualified_handler("binaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "BinaryOp"
            # Absent operands are handled uniformly by _replace_inputs, which
            # substitutes an explicit non-operation and records a shortfall.
            # Filtering them out here instead would drop the operand
            # silently, leaving an operation short an argument -- a wrong
            # program rather than an honestly incomplete one.
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.right), "rhs"),
                ),
            )
            continue
        if c_type_name == "UnaryOp" and not isinstance(expression, ast.AST):
            if str(expression.op) in {"+", "&", "*"}:
                # Unary plus is a no-op conversion, the same reading Python's
                # ast.UAdd gets below.
                #
                # Address-of and dereference are identity *in this value
                # graph specifically*: a node here denotes a value, and the
                # graph has no separate address space for `&x` to point
                # into that `x` does not already name. The cpp shell emits
                # `&obj` for every method receiver, and `obj`'s own node is
                # already exactly the receiver the callee needs. These stay
                # registered as GetElementPtr/Load in ssa_registry (their
                # true meanings, which a real memory model would need) --
                # collapsing them is a property of this representation, not
                # a claim that C's & and * are no-ops. Revisit when pointer
                # arithmetic or aliasing enters the shell's scope; today it
                # is excluded by CPP_LIKE_SHELL_FOR_C_INTENT.md.
                _redirect_value(graph, node_id, id(expression.expr))
                continue
            operation = _c_qualified_handler("unaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "UnaryOp"
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.expr), "operand"),),
            )
            continue
        if isinstance(expression, (ast.Nonlocal, ast.Global)):
            # Root whole-graph deployment regions are formed from this graph,
            # not only from the normalized per-function copies below.  Scope
            # declarations are compile-time syntax in both representations.
            _remove_node(graph, node_id)
            continue
        if isinstance(expression, ast.Delete):
            # The target nodes carry the executable deletion effects.  The
            # statement wrapper is ordering syntax, not an operator.
            _remove_node(graph, node_id)
            continue
        if (
            isinstance(expression, ast.Name)
            and isinstance(expression.ctx, ast.Del)
        ):
            # Deleting a lexical binding has no object-level runtime effect.
            _remove_node(graph, node_id)
            continue
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.ctx, ast.Del)
        ):
            data["type"] = "DelAttr"
            data["op"] = "delattr"
            data.setdefault("attributes", {})["attribute"] = expression.attr
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.value), "object"),),
            )
            continue
        if isinstance(expression, ast.Constant):
            data["type"] = "Constant"
            data["op"] = "const"
            data["constant"] = expression.value
            data.setdefault("attributes", {})["value"] = expression.value
        elif isinstance(expression, ast.BinOp):
            folded = _static_sequence_literal(expression)
            if folded is not None:
                # Sequence replication is table allocation, not arithmetic.
                data["type"] = "Constant"
                data["op"] = "const"
                data["constant"] = folded
                attributes = data.setdefault("attributes", {})
                attributes["value"] = folded
                attributes["source_type"] = "BinOp"
                attributes["constant_folded"] = "sequence-replication"
                continue
            left_descriptor = _operand_numeric_descriptor(
                graph, expression.left
            )
            right_descriptor = _operand_numeric_descriptor(
                graph, expression.right
            )
            numeric_descriptor = join_numeric_feature_descriptors(
                left_descriptor, right_descriptor,
            )
            attributes = data.setdefault("attributes", {})
            if numeric_descriptor is not None:
                attributes["numeric_feature_descriptor"] = (
                    numeric_descriptor.receipt()
                )
                if numeric_descriptor.features != frozenset({"precision"}):
                    attributes["numeric_composite_pending"] = True
            left_limbs, left_type = (
                (left_descriptor.limbs, left_descriptor.element_type)
                if left_descriptor is not None
                and left_descriptor.features == frozenset({"precision"})
                else (1, None)
            )
            right_limbs, right_type = (
                (right_descriptor.limbs, right_descriptor.element_type)
                if right_descriptor is not None
                and right_descriptor.features == frozenset({"precision"})
                else (1, None)
            )
            operand_limbs = max(left_limbs, right_limbs)
            operand_element = _widest_element(left_type, right_type)
            operation = _qualified_handler(
                "binop", expression.op,
                limbs=operand_limbs, dtype=operand_element,
            )
            data["type"] = operation
            data["op"] = operation
            if operand_limbs > 1:
                attributes["precision_limbs"] = operand_limbs
                attributes["precision_element"] = operand_element
            attributes["source_type"] = "BinOp"
            sequence_operand = (
                expression.left
                if isinstance(expression.left, (ast.List, ast.Tuple))
                else expression.right
                if isinstance(expression.right, (ast.List, ast.Tuple))
                else None
            )
            if isinstance(expression.op, ast.Mult) and sequence_operand is not None:
                attributes.update({
                    "producer_kind": "sequence_replication",
                    "aggregate_kind": (
                        "list" if isinstance(sequence_operand, ast.List)
                        else "tuple"
                    ),
                    "sequence_key_columns": (),
                    "sequence_column_count": 1,
                    "sequence_writable": isinstance(sequence_operand, ast.List),
                    "sequence_fill_value_ids": tuple(
                        id(element) for element in sequence_operand.elts
                    ),
                })
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.right), "rhs"),
                ),
            )
        elif isinstance(expression, ast.AugAssign):
            operation = _qualified_handler("binop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "AugAssign"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.target), "lhs"),
                    (id(expression.value), "rhs"),
                ),
            )
        elif isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.UAdd):
                _redirect_value(
                    graph,
                    node_id,
                    id(expression.operand),
                )
                continue
            numeric_descriptor = _operand_numeric_descriptor(
                graph, expression.operand
            )
            attributes = data.setdefault("attributes", {})
            if numeric_descriptor is not None:
                attributes["numeric_feature_descriptor"] = (
                    numeric_descriptor.receipt()
                )
                if numeric_descriptor.features != frozenset({"precision"}):
                    attributes["numeric_composite_pending"] = True
            operand_limbs, operand_type = (
                (numeric_descriptor.limbs, numeric_descriptor.element_type)
                if numeric_descriptor is not None
                and numeric_descriptor.features == frozenset({"precision"})
                else (1, None)
            )
            operation = _qualified_handler(
                "unaryop", expression.op, limbs=operand_limbs,
                dtype=operand_type,
            )
            data["type"] = operation
            data["op"] = operation
            if operand_limbs > 1:
                attributes["precision_limbs"] = operand_limbs
                attributes["precision_element"] = _widest_element(operand_type)
            attributes["source_type"] = "UnaryOp"
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.operand), "operand"),),
            )
        elif isinstance(expression, ast.BoolOp):
            data.setdefault("attributes", {})["source_type"] = "BoolOp"
            for value in expression.values[1:]:
                value_id = id(value)
                if value_id in graph.G:
                    graph.G.nodes[value_id].setdefault(
                        "attributes",
                        {},
                    )["coordinator_short_circuit"] = True
            _replace_inputs(
                graph,
                node_id,
                tuple(
                    (id(value), f"value:{index}")
                    for index, value in enumerate(expression.values)
                ),
            )
        elif isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            operation = _qualified_handler("compare", expression.ops[0])
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Compare"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.comparators[0]), "rhs"),
                ),
            )
        elif isinstance(expression, ast.Subscript):
            # Indexing is one operation over the parent tensor and the complete
            # index tuple.  In particular, Ellipsis is not an independent
            # scalar operation: AbstractTensor.__getitem__ and each specialized
            # backend resolve it against the parent tensor's rank.
            index_expressions = (
                tuple(expression.slice.elts)
                if isinstance(expression.slice, ast.Tuple)
                else (expression.slice,)
            )
            deleting = isinstance(expression.ctx, ast.Del)
            data["type"] = "DelItem" if deleting else "Indexed"
            data["op"] = "delitem" if deleting else "Indexed"
            data.setdefault("attributes", {})["source_type"] = "Subscript"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.value), "base"),
                    *(
                        (id(index_expression), "index")
                        for index_expression in index_expressions
                    ),
                ),
            )

            # The AST Tuple only grouped the index components.  Once those
            # components feed Indexed directly, retaining the wrapper would
            # incorrectly schedule a second tuple-producing computation.
            slice_id = id(expression.slice)
            if (
                isinstance(expression.slice, ast.Tuple)
                and slice_id in graph.G
                and graph.G.out_degree(slice_id) == 0
            ):
                for predecessor in tuple(graph.G.predecessors(slice_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != slice_id
                    ]
                graph.G.remove_node(slice_id)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ):
            operation = expression.func.attr
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Call"
            receiver_name = (
                expression.func.value.id
                if isinstance(expression.func.value, ast.Name)
                else None
            )
            call_owner = call_owners.get(node_id)
            lexical_class = (
                method_owner_by_reference.get(int(call_owner.address))
                if call_owner is not None
                else None
            )
            if receiver_name in {"self", "cls"} and lexical_class is not None:
                method_reference = (
                    graph.G.graph.get("class_table", {})
                    .get(lexical_class, {})
                    .get("methods", {})
                    .get(expression.func.attr)
                )
                if method_reference is not None:
                    method_reference = int(method_reference)
                    data.setdefault("attributes", {})["method_ref"] = (
                        method_reference
                    )
                    accessor_id = id(expression.func)
                    if accessor_id in graph.G:
                        accessor = graph.G.nodes[accessor_id]
                        accessor.setdefault("attributes", {}).update({
                            "accessor_kind": "method",
                            "method_ref": method_reference,
                        })
            if data.get("attributes", {}).get("method_ref") is None:
                receiver_id = next((
                    int(parent)
                    for parent, role in data.get("parents", ())
                    if str(role) in {"operand", "value", "base", "object"}
                ), None)
                receiver_attributes = (
                    {}
                    if receiver_id is None or receiver_id not in graph.G
                    else graph.G.nodes[receiver_id].get("attributes") or {}
                )
                receiver_class = receiver_attributes.get(
                    "result_class_ref",
                    receiver_attributes.get("class_ref"),
                )
                method_reference = (
                    graph.G.graph.get("class_table", {})
                    .get(str(receiver_class), {})
                    .get("methods", {})
                    .get(expression.func.attr)
                )
                if method_reference is not None:
                    method_reference = int(method_reference)
                    data.setdefault("attributes", {})["method_ref"] = (
                        method_reference
                    )
                    data["attributes"]["callee_ref"] = method_reference
                    accessor_id = id(expression.func)
                    if accessor_id in graph.G:
                        graph.G.nodes[accessor_id].setdefault(
                            "attributes", {}
                        ).update({
                            "accessor_kind": "method",
                            "method_ref": method_reference,
                        })
            # Wiring the receiver/arguments as this node's own ``parents``
            # used to happen right here, keyed by ``id(expression.func.
            # value)`` -- the raw Python id() of the AST node, not a real
            # graph reference.  That is only ever valid by coincidence: it
            # assumes ingestion already walked this exact receiver
            # independently, which is not guaranteed (a receiver reached
            # only through this call expression, inside a nested try/loop,
            # was never separately ingested).  ``orchestrator`` here is an
            # ordinary local variable; it already has a real, resolved SSA
            # value the moment its own assignment is processed
            # (``bind_target`` -> ``environment``) -- fabricating a new
            # node for it, or guessing at a raw id, both bypass that
            # existing value instead of looking it up.  The actual lookup
            # operator for "what did this expression resolve to" is
            # ``resolve_expression`` itself, which only becomes available
            # once lexical normalization runs (this pass runs earlier, over
            # whatever ingestion already produced). Wiring is done instead
            # in ``resolve_expression``'s own ``ast.Call`` handling
            # (``_normalize_lexical_values``, run later during reduction),
            # where ``resolve_expression`` can be called directly on the
            # receiver and arguments -- the same SSA lookup every other
            # expression in the program already goes through.
        elif isinstance(expression, ast.Call):
            call_inputs: list[tuple[int, str]] = [
                (
                    id(argument),
                    f"arg:{index}",
                )
                for index, argument in enumerate(expression.args)
            ]
            call_inputs.extend(
                (
                    id(keyword.value),
                    (
                        f"kw:{keyword.arg}"
                        if keyword.arg is not None
                        else "kwargs"
                    ),
                )
                for keyword in expression.keywords
            )
            if isinstance(expression.func, ast.Name):
                callee_name = expression.func.id
                owner = call_owners.get(node_id)
                callee_is_parameter = (
                    owner is not None
                    and callee_name in function_parameters_by_address.get(
                        owner.address,
                        (),
                    )
                )
                if callee_is_parameter:
                    attributes = data.setdefault("attributes", {})
                    attributes.pop("callee_ref", None)
                    attributes.pop("external_callee_ref", None)
                    call_inputs.insert(
                        0,
                        (id(expression.func), "callee"),
                    )
                    reference = None
                else:
                    reference = function_table.reference(callee_name)
                if (
                    not callee_is_parameter
                    and reference is None
                    and callee_name in import_bindings
                ):
                    (
                        qualified_name,
                        imported_name,
                        requirement,
                    ) = import_bindings[
                        callee_name
                    ]
                    external_reference = external_function_table.declare(
                        callee_name,
                        qualified_name=qualified_name,
                        external=True,
                        metadata={
                            "contextual_requirement": requirement,
                            "imported_name": imported_name,
                        },
                    )
                    data.setdefault("attributes", {})[
                        "external_callee_ref"
                    ] = external_reference.address
                elif (
                    not callee_is_parameter
                    and reference is not None
                ):
                    data.setdefault("attributes", {})[
                        "callee_ref"
                    ] = reference.address
                elif not callee_is_parameter:
                    call_inputs.insert(
                        0,
                        (id(expression.func), "callee"),
                    )
            else:
                call_inputs.insert(0, (id(expression.func), "callee"))
            _replace_inputs(graph, node_id, tuple(call_inputs))

            # Keyword nodes carry only source spelling once their value and
            # name have been transferred to the Call edge role.
            for keyword in expression.keywords:
                keyword_id = id(keyword)
                if (
                    keyword_id not in graph.G
                    or graph.G.out_degree(keyword_id) != 0
                ):
                    continue
                for predecessor in tuple(graph.G.predecessors(keyword_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != keyword_id
                    ]
                graph.G.remove_node(keyword_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        statement = data.get("expr_obj")
        if not isinstance(statement, ast.For):
            continue
        attributes = data.setdefault("attributes", {})
        attributes["source_type"] = "For"
        attributes["target"] = (
            statement.target.id
            if isinstance(statement.target, ast.Name)
            else ast.dump(statement.target, include_attributes=False)
        )
        loop_inputs: list[tuple[int, str]] = []
        iterator = statement.iter
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
        ):
            attributes["iterator_kind"] = "arithmetic_sequence"
            range_arguments = tuple(iterator.args)
            if not 1 <= len(range_arguments) <= 3:
                raise ValueError("range loop requires one to three arguments")
            roles = (
                ("stop",)
                if len(range_arguments) == 1
                else ("start", "stop")
                if len(range_arguments) == 2
                else ("start", "stop", "step")
            )
            if len(range_arguments) == 1:
                attributes["start"] = 0
                attributes["step"] = 1
            elif len(range_arguments) == 2:
                attributes["step"] = 1
            loop_inputs.extend(
                (id(argument), role)
                for argument, role in zip(range_arguments, roles)
            )
        else:
            attributes["iterator_kind"] = "iterable"
            loop_inputs.append((id(iterator), "iterable"))
        loop_inputs.extend(
            (id(body_statement), "body")
            for body_statement in statement.body
        )
        loop_inputs.extend(
            (id(else_statement), "else")
            for else_statement in statement.orelse
        )
        _replace_inputs(graph, node_id, tuple(loop_inputs))

        if isinstance(iterator, ast.Call) and id(iterator) in graph.G:
            iterator_id = id(iterator)
            if graph.G.out_degree(iterator_id) == 0:
                for predecessor in tuple(graph.G.predecessors(iterator_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != iterator_id
                    ]
                graph.G.remove_node(iterator_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Expr":
            continue
        # An Expr statement wraps its value(s) and is dissolved: every consumer
        # of the wrapper is reconnected directly to the wrapped value(s). There
        # is no artificial cap of one value -- a wrapper standing in for several
        # values (e.g. a call whose arguments the builder attached directly)
        # fans its slot out to all of them, preserving their order.
        predecessors = tuple(graph.G.predecessors(node_id))
        successors = tuple(graph.G.successors(node_id))
        for successor in successors:
            successor_data = graph.G.nodes[successor]
            replacement_parents = []
            for parent_id, role in successor_data.get("parents", ()):
                if parent_id != node_id:
                    replacement_parents.append((parent_id, role))
                else:
                    for predecessor in predecessors:
                        replacement_parents.append((predecessor, role))
            successor_data["parents"] = replacement_parents
            for predecessor in predecessors:
                graph.G.add_edge(predecessor, successor)
                predecessor_children = graph.G.nodes[predecessor].setdefault(
                    "children", []
                )
                if successor not in {
                    child_id for child_id, _role in predecessor_children
                }:
                    predecessor_children.append((successor, "output"))
        for predecessor in predecessors:
            graph.G.nodes[predecessor]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[predecessor].get(
                    "children", ()
                )
                if child_id != node_id
            ]
        # Roots holding the wrapper are replaced by every value it wrapped
        # (deduplicated, order preserved); a wrapper with no value is dropped.
        new_roots: list = []
        for root_id in graph.roots:
            if root_id != node_id:
                new_roots.append(root_id)
            else:
                new_roots.extend(predecessors)
        seen: set = set()
        graph.roots = [
            root for root in new_roots if not (root in seen or seen.add(root))
        ]
        graph.G.remove_node(node_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Return":
            continue
        # A Return wrapper carries its returned value(s) and is dissolved into
        # them. There is no artificial cap of one value -- a multi-value return
        # (a tuple of results) publishes every one of its values as a root,
        # order preserved.
        predecessors = tuple(graph.G.predecessors(node_id))
        for successor in tuple(graph.G.successors(node_id)):
            graph.G.nodes[successor]["parents"] = [
                (parent_id, role)
                for parent_id, role in graph.G.nodes[successor].get(
                    "parents", ()
                )
                if parent_id != node_id
            ]
        for returned in predecessors:
            graph.G.nodes[returned]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[returned].get(
                    "children", ()
                )
                if child_id != node_id
            ]
            if returned not in graph.roots:
                graph.roots.append(returned)
        graph.roots = [root for root in graph.roots if root != node_id]
        graph.G.remove_node(node_id)

    call_graph = nx.DiGraph()
    call_graph.add_nodes_from(
        reference.address for reference in function_nodes.values()
    )
    referenced_calls: list[tuple[int, Any, Any]] = []
    for call_id, owner in call_owners.items():
        if call_id not in graph.G:
            continue
        callee_address = (
            graph.G.nodes[call_id].get("attributes") or {}
        ).get("callee_ref")
        if callee_address is None:
            continue
        referenced_calls.append((call_id, owner, callee_address))
        try:
            callee_entry = function_table.entry(callee_address)
        except KeyError:
            continue
        if callee_entry.graph is None and callee_address not in {
            reference.address for reference in function_nodes.values()
        }:
            continue
        call_graph.add_edge(owner.address, callee_address)

    recursive_edges: set[tuple[int, int]] = set()
    for component in nx.strongly_connected_components(call_graph):
        if len(component) > 1:
            recursive_edges.update(
                (source, target)
                for source, target in call_graph.edges(component)
                if target in component
            )
        else:
            address = next(iter(component))
            if call_graph.has_edge(address, address):
                recursive_edges.add((address, address))
    for call_id, owner, callee_address in referenced_calls:
        if (owner.address, callee_address) not in recursive_edges:
            continue
        graph.G.nodes[call_id].setdefault("attributes", {})[
            "recursive_backedge"
        ] = True
        function_table.entry(owner).recursive = True

    for node_id, reference in function_nodes.items():
        if node_id not in graph.G:
            continue
        return_values = [
            value
            for value in function_return_values.get(node_id, ())
            if value in graph.G
        ]
        terminal_merges = [
            int(member)
            for member in graph.G
            if (
                (graph.G.nodes[member].get("attributes") or {}).get(
                    "terminal_return_merge"
                )
                and set(
                    (graph.G.nodes[member].get("attributes") or {}).get(
                        "terminal_return_values", ()
                    )
                ).issubset(return_values)
            )
        ]
        if terminal_merges:
            merged_values = {
                int(value)
                for member in terminal_merges
                for value in (
                    graph.G.nodes[member].get("attributes") or {}
                ).get("terminal_return_values", ())
            }
            return_values = [
                value for value in return_values if value not in merged_values
            ]
            return_values.extend(terminal_merges)
        statement = graph.G.nodes[node_id].get("expr_obj")
        definition_static_bindings = dict(
            getattr(statement, "_python_bindings", static_bindings)
        )
        for builtin_name, builtin_value in vars(builtins).items():
            definition_static_bindings.setdefault(
                builtin_name,
                builtin_value,
            )
        # Function ownership is already exact in the saved Python AST.  Use
        # that ownership directly: graph ancestry from only the return value
        # silently discarded assignments, calls, loops, and side effects.
        #
        # This deliberately still walks the raw ``ast`` tree, not
        # ``graph.G`` -- unlike ``_record_owned_calls_and_returns`` above,
        # which runs right after ingestion while the graph still mirrors the
        # source tree 1:1. By the time this loop runs, earlier reduction
        # passes have already restructured ``graph.G`` (edges no longer
        # necessarily match the original AST parent/child shape), so a
        # graph-native walk here would silently miss nodes a raw-tree walk
        # still finds (confirmed empirically: a graph-based version of this
        # walk lost the Return node itself for a plain `return (...)`
        # function body). Generalizing *this* walk to a foreign language
        # needs its own graph-native traversal designed against the
        # post-reduction graph shape, not a drop-in swap -- tracked
        # separately, not attempted here.
        owned_members: set[int] = set()

        def record_owned_member(member: Any) -> None:
            if isinstance(member, ast.ClassDef):
                # A class's body belongs to its own methods' entries.
                return
            if is_runnable_definition(member):
                # A nested definition's body belongs to another
                # function-table entry, not to this enclosing shell. A
                # lambda is still itself a value the enclosing scope
                # references, so it is owned without being descended into.
                if isinstance(member, ast.Lambda) and id(member) in graph.G:
                    owned_members.add(id(member))
                return
            if id(member) in graph.G and not isinstance(
                member,
                (
                    ast.arguments,
                    ast.arg,
                    ast.expr_context,
                    ast.operator,
                    ast.unaryop,
                    ast.boolop,
                    ast.cmpop,
                    ast.keyword,
                    ast.alias,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                owned_members.add(id(member))
            for child in source_child_nodes(member):
                record_owned_member(child)

        if isinstance(statement, ast.Lambda):
            # A lambda's body is one expression, and is the body itself
            # rather than a list of statements -- descend into it directly
            # instead of treating it as a nested-definition boundary.
            for child in source_child_nodes(statement.body):
                record_owned_member(child)
            if id(statement.body) in graph.G:
                owned_members.add(id(statement.body))
        else:
            for body_member in source_body_statements(statement):
                record_owned_member(body_member)
        included = owned_members
        included = {
            member
            for member in included
            if (
                member in return_values
                # An ordered side effect remains part of its lexical body
                # even when reduction detached its value operands.
                or graph.G.nodes[member].get("attributes", {}).get("ordered_effect", False)
                or any(
                    neighbor in included
                    for neighbor in graph.G.predecessors(member)
                )
                or any(
                    neighbor in included
                    for neighbor in graph.G.successors(member)
                )
            )
        }
        function_graph = copy.copy(graph)
        function_graph.G = graph.G.subgraph(included).copy()
        # Extraction decisions are occurrence contracts, not merely a root-
        # graph audit log.  Several reducer rewrites rebuild a Call node's
        # attributes while preserving its source AST identity; the global
        # boundary ledger survives that rewrite, but the per-function graph
        # previously lost the receipt needed by shell/native-call planning.
        # Restore it by exact source occurrence before detaching the function
        # graph.  This does not re-resolve or pursue the callable.
        statement_name = str(getattr(statement, "name", ""))
        statement_identity = getattr(statement, "_python_source_identity", None)
        boundary_by_location = {
            (int(boundary["line"]), int(boundary["column"])): dict(
                boundary["extraction_contract"]
            )
            for boundary in graph.G.graph.get(
                "extraction_boundary_calls", ()
            )
            if boundary.get("line") is not None
            and boundary.get("column") is not None
            and isinstance(boundary.get("extraction_contract"), Mapping)
            and (
                (
                    statement_identity is None
                    and str(boundary.get("owner_name") or "")
                    == statement_name
                )
                or (
                    statement_identity is not None
                    and tuple(boundary.get("owner_source_identity") or ())
                    == tuple(statement_identity)
                )
            )
        }
        for _member_id, member_data in function_graph.G.nodes(data=True):
            expression = member_data.get("expr_obj")
            if not isinstance(expression, ast.Call):
                continue
            # AST special-case expansion owns the generated operation's
            # receipt. Several generated operations can share one original
            # source location; the source ledger cannot distinguish them.
            receipt = getattr(expression, "_extraction_contract", None) if getattr(
                expression, "_turing_dispatch_operation", None
            ) else boundary_by_location.get((
                int(getattr(expression, "lineno", -1)),
                int(getattr(expression, "col_offset", -1)),
            ))
            if receipt is None:
                continue
            expression._extraction_contract = receipt
            special_case = interpret_python_special_case(expression)
            if special_case is not None:
                member_data.setdefault("attributes", {}).update(
                    special_case.attributes
                )
        # The definition may carry literal module constants and source-local
        # imports beyond the root graph's generic binding set.  Preserve that
        # exact static environment on the function graph consumed by compiled
        # shells; otherwise those globals reappear as missing runtime inputs.
        function_graph.python_bindings = definition_static_bindings
        function_graph.levels = {
            member: level
            for member, level in graph.levels.items()
            if member in included
        }
        function_graph.roots = return_values or [node_id]
        function_graph.function_table = function_table
        positional_parameters = ()
        keyword_only_parameters = ()
        variadic_parameters = ()
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            positional_parameters = tuple(
                argument.arg
                for argument in (
                    *statement.args.posonlyargs,
                    *statement.args.args,
                )
            )
            keyword_only_parameters = tuple(
                argument.arg for argument in statement.args.kwonlyargs
            )
            variadic_parameters = tuple(
                argument.arg
                for argument in (statement.args.vararg, statement.args.kwarg)
                if argument is not None
            )
        parameter_names = (
            *positional_parameters,
            *keyword_only_parameters,
            *variadic_parameters,
        )
        static_parameter_bindings = _concorded_static_parameter_bindings(
            statement,
            parameter_names,
        )
        static_parameter_receipts = tuple(
            (
                name,
                "static_class",
                str(getattr(value, "__module__", "")),
                str(getattr(
                    value,
                    "__qualname__",
                    getattr(value, "__name__", ""),
                )),
            )
            for name, value in static_parameter_bindings.items()
        )
        parameter_defaults = {}
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            if statement.args.vararg is not None:
                parameter_defaults[statement.args.vararg.arg] = ()
            if statement.args.kwarg is not None:
                parameter_defaults[statement.args.kwarg.arg] = {}
        scalar_parameter_names = set()
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            for argument in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            ):
                if (
                    isinstance(argument.annotation, ast.Name)
                    and argument.annotation.id
                    in {"bool", "bytes", "complex", "float", "int", "str"}
                ):
                    scalar_parameter_names.add(argument.arg)
            positional_defaults = zip(
                positional_parameters[
                    len(positional_parameters) - len(statement.args.defaults):
                ],
                statement.args.defaults,
            )
            keyword_defaults = zip(
                keyword_only_parameters,
                statement.args.kw_defaults,
            )
            for name, default_expression in (
                *positional_defaults,
                *keyword_defaults,
            ):
                if default_expression is None:
                    continue
                try:
                    value = ast.literal_eval(default_expression)
                except (ValueError, TypeError):
                    if isinstance(default_expression, ast.Name):
                        value = static_bindings.get(
                            default_expression.id,
                            default_expression,
                        )
                    else:
                        continue
                parameter_defaults[name] = value
                if isinstance(
                    value,
                    (bool, bytes, complex, float, int, str),
                ):
                    scalar_parameter_names.add(name)
        function_graph.G.graph.update(
            function_ref=reference.address,
            function_name=function_table.entry(reference).name,
            source_pursuit_active=bool(
                getattr(statement, "_source_pursuit_active", False)
            ),
            method_owner=method_owners.get(node_id),
            method_binding=(
                "class"
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and any(
                    isinstance(decorator, ast.Name)
                    and decorator.id == "classmethod"
                    for decorator in statement.decorator_list
                )
                else (
                    "static"
                    if isinstance(
                        statement,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and any(
                        isinstance(decorator, ast.Name)
                        and decorator.id == "staticmethod"
                        for decorator in statement.decorator_list
                    )
                    else (
                        "instance"
                        if method_owners.get(node_id) is not None
                        else None
                    )
                )
            ),
            function_parameters=parameter_names,
            static_parameter_bindings=static_parameter_receipts,
            positional_parameters=positional_parameters,
            keyword_only_parameters=keyword_only_parameters,
            parameter_defaults=parameter_defaults,
            scalar_parameters=tuple(sorted(scalar_parameter_names)),
            function_body=(
                tuple(statement.body)
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                else ()
            ),
            class_field_aggregate_kinds={
                field: kind
                for (owner, field), kind in class_field_aggregate_kinds.items()
                if owner == method_owners.get(node_id)
            },
            class_field_aliases={
                field: target
                for (owner, field), target in class_field_aliases.items()
                if owner == method_owners.get(node_id)
            },
            class_field_value_aggregate_kinds={
                field: kind
                for (owner, field), kind
                in class_field_value_aggregate_kinds.items()
                if owner == method_owners.get(node_id)
            },
        )
        parameter_contracts = _known_parameter_memory_contracts(
            statement,
            parameter_names,
            method_owner=method_owners.get(node_id),
            static_parameter_names=frozenset(static_parameter_bindings),
        )
        if parameter_contracts:
            function_table.set_parameter_contracts(
                reference,
                parameter_contracts,
            )
        for member in function_graph.G:
            member_data = function_graph.G.nodes[member]
            expression = member_data.get("expr_obj")
            if (
                isinstance(expression, ast.Name)
                and isinstance(expression.ctx, ast.Load)
                and expression.id in parameter_names
            ):
                member_data["type"] = "Input"
                member_data["op"] = "input"
                member_data["label"] = expression.id
                member_data["parents"] = []
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
            member_data["parents"] = [
                (parent, role)
                for parent, role in member_data.get("parents", ())
                if parent in included
            ]
            member_data["children"] = [
                (child, role)
                for child, role in member_data.get("children", ())
                if child in included
            ]
        normalization_statement = statement
        if isinstance(statement, ast.Lambda):
            normalization_statement = ast.copy_location(
                ast.FunctionDef(
                    name=function_table.entry(reference).name,
                    args=statement.args,
                    body=[
                        ast.copy_location(
                            ast.Return(value=statement.body),
                            statement.body,
                        )
                    ],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                ),
                statement,
            )
        # Normalize attributes with returned receiver classes already
        # attached.  This lets a field on an authored method result inherit
        # the class's aggregate/storage contract just like ``self.field``.
        propagate_returned_receiver_types(function_graph.G)
        local_mapping_contracts = dict(class_field_mapping_contracts)
        local_owner = method_owners.get(node_id)
        if local_owner is not None:
            for declaration in (
                candidate for candidate in ast.walk(normalization_statement)
                if isinstance(candidate, ast.AnnAssign)
                and isinstance(candidate.target, ast.Attribute)
                and isinstance(candidate.target.value, ast.Name)
                and candidate.target.value.id in {"self", "cls"}
            ):
                contract = _mapping_annotation(declaration.annotation)
                if contract:
                    local_mapping_contracts[
                        (str(local_owner), str(declaration.target.attr))
                    ] = contract
        _normalize_lexical_values(
            function_graph,
            normalization_statement,
            definition_static_bindings,
            function_table,
            lexical_functions_by_owner.get(int(reference.address), {}),
            closure_aggregate_kinds(node_id),
            method_owner=method_owners.get(node_id),
            class_field_aggregate_kinds=class_field_aggregate_kinds,
            class_field_mapping_contracts=local_mapping_contracts,
            class_field_sequence_dtypes=class_field_sequence_dtypes,
            class_field_classes=class_field_classes,
            static_parameter_bindings=static_parameter_bindings,
        )
        generator_yields = tuple(
            node_id
            for node_id, node_data in function_graph.G.nodes(data=True)
            if isinstance(
                node_data.get("expr_obj"),
                (ast.Yield, ast.YieldFrom),
            )
        )
        if generator_yields:
            function_graph.G.graph["generator_stream"] = {
                "yield_nodes": generator_yields,
                "flow_control": "downstream_capacity",
                "execution_owner": "planner_shell",
            }
        for _member, member_data in function_graph.G.nodes(data=True):
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
        if hasattr(statement, "_python_bindings"):
            delattr(statement, "_python_bindings")
        if hasattr(statement, "_python_aggregate_binding_kinds"):
            delattr(statement, "_python_aggregate_binding_kinds")
        function_graph.python_bindings = canonicalize_python_static_bindings(
            function_graph.python_bindings
        )
        function_table.resolve_graph(reference, function_graph)
    # Function graphs are attached only after the earlier root-graph pass.
    # Reuse that exact provenance propagation now that every callee body is
    # available, so a call in a function body retains its returned class.
    def propagate_call_formal_numeric_types() -> bool:
        """Carry exact numeric class facts over authored call edges.

        A retained base method deliberately does not hard-code the concrete
        wrapper used at a callsite.  The call edge is the proof: its actual
        has an already-concorded class/width, and the callee formal is the
        same value.  Commit that identity through the concordance book before
        publishing it on the formal node; copied AST annotations are not an
        independent typing channel.
        """

        entries = {
            int(entry.reference.address): entry
            for entry in function_table
            if getattr(entry, "graph", None) is not None
        }

        def ordered_actuals(call_data: Mapping[str, Any]) -> tuple[int, ...]:
            ranked = []
            for actual_id, role in call_data.get("parents", ()):
                spelling = str(role)
                if spelling in {"operand", "receiver"}:
                    rank = -1
                elif spelling.startswith("arg:"):
                    try:
                        rank = int(spelling.split(":", 1)[1])
                    except ValueError:
                        continue
                elif spelling.startswith("arg"):
                    try:
                        rank = int(spelling[3:])
                    except ValueError:
                        continue
                else:
                    continue
                ranked.append((rank, int(actual_id)))
            return tuple(actual for _rank, actual in sorted(ranked))

        changed = False
        for caller_entry in entries.values():
            caller = caller_entry.graph.G
            if not caller.graph.get("source_pursuit_active"):
                continue
            caller_scope = _source_numeric_scope(caller, caller_entry.name)
            for call_id, call_data in tuple(caller.nodes(data=True)):
                call_attributes = call_data.get("attributes") or {}
                callee_ref = call_attributes.get(
                    "callee_ref", call_attributes.get("method_ref")
                )
                if callee_ref is None or int(callee_ref) not in entries:
                    continue
                callee_entry = entries[int(callee_ref)]
                callee = callee_entry.graph.G
                callee_scope = _source_numeric_scope(
                    callee, callee_entry.name
                )
                reachability_page = current_identity_book().page(
                    "source_function_reachability_concordance"
                )
                reachability_row = (
                    caller_scope, int(call_id), callee_scope,
                )
                incumbent_reachability = reachability_page.latest(
                    reachability_row
                )
                if incumbent_reachability is not None and (
                    incumbent_reachability is not True
                ):
                    raise ValueError(
                        "source function reachability concordance "
                        f"disagreement for {reachability_row!r}: "
                        f"recorded={incumbent_reachability!r}, proposed=True"
                    )
                if incumbent_reachability is None:
                    reachability_page.set(reachability_row, 0, True)
                if not callee.graph.get("source_pursuit_active"):
                    callee.graph["source_pursuit_active"] = True
                    changed = True
                output_histories = dict(
                    callee.graph.get("identity_table") or {}
                )
                returned_value_ids = tuple(
                    int(history[-1])
                    for output_name in tuple(
                        callee.graph.get("function_outputs") or ()
                    )
                    for history in (tuple(
                        output_histories.get(str(output_name)) or ()
                    ),)
                    if history and int(history[-1]) in callee
                )
                returned_facts = tuple(dict.fromkeys(
                    _resolved_source_value_class(callee, value_id)
                    for value_id in returned_value_ids
                    if _resolved_source_value_class(
                        callee, value_id
                    )[0] is not None
                ))
                if len(returned_facts) == 1:
                    returned_class, returned_limbs = returned_facts[0]
                    result_page = current_identity_book().page(
                        "source_call_result_identity_concordance"
                    )
                    result_row = (caller_scope, int(call_id))
                    result_fact = (
                        int(callee_ref), str(returned_class),
                        int(returned_limbs),
                    )
                    incumbent_result = result_page.latest(result_row)
                    if incumbent_result is not None and tuple(
                        incumbent_result
                    ) != result_fact:
                        raise ValueError(
                            "source call result identity concordance "
                            f"disagreement for {result_row!r}: "
                            f"recorded={incumbent_result!r}, "
                            f"proposed={result_fact!r}"
                        )
                    if incumbent_result is None:
                        result_page.set(result_row, 0, result_fact)
                    call_result_attributes = caller.nodes[
                        int(call_id)
                    ].setdefault("attributes", {})
                    prior_result = (
                        call_result_attributes.get("result_class_ref"),
                        int(call_result_attributes.get(
                            "precision_limbs"
                        ) or 1),
                    )
                    _concord_source_value_class(
                        caller_scope,
                        int(call_id),
                        str(returned_class),
                        precision_limbs=int(returned_limbs),
                        source=f"call result {callee_scope}",
                    )
                    call_result_attributes.update({
                        "result_class_ref": str(returned_class),
                        "precision_limbs": int(returned_limbs),
                        "numeric_identity_source": (
                            f"call result {callee_scope}"
                        ),
                    })
                    returned_descriptor = numeric_feature_descriptor(
                        str(returned_class) + (
                            f"[{int(returned_limbs)}]"
                            if int(returned_limbs) > 1 else ""
                        )
                    )
                    if returned_descriptor is not None:
                        call_result_attributes[
                            "numeric_feature_descriptor"
                        ] = returned_descriptor.receipt()
                    changed = changed or prior_result != (
                        str(returned_class), int(returned_limbs),
                    )
                parameter_names = tuple(
                    map(str, callee.graph.get("function_parameters") or ())
                )
                actuals = ordered_actuals(call_data)
                if not parameter_names or not actuals:
                    continue
                formal_nodes = {
                    str(attributes.get("binding_name")): int(node_id)
                    for node_id, data in callee.nodes(data=True)
                    for attributes in (data.get("attributes") or {},)
                    if data.get("type") == "Input"
                    and attributes.get("binding_name") is not None
                }
                for position, actual_id in enumerate(actuals):
                    if position >= len(parameter_names):
                        break
                    class_page = current_identity_book().page(
                        "source_value_class_concordance"
                    )
                    concorded = class_page.latest((
                        caller_scope, int(actual_id),
                    ))
                    class_identity, limbs = (
                        (str(concorded[0]), int(concorded[1]))
                        if isinstance(concorded, tuple)
                        and len(concorded) >= 2
                        else _resolved_source_value_class(
                            caller, int(actual_id)
                        )
                    )
                    if class_identity is None:
                        continue
                    formal_id = formal_nodes.get(parameter_names[position])
                    if formal_id is None:
                        continue
                    formal_attributes = callee.nodes[formal_id].setdefault(
                        "attributes", {}
                    )
                    actual_attributes = (
                        caller.nodes[int(actual_id)].get("attributes") or {}
                    )
                    transfer_source = (
                        f"call actual {caller_scope}:{actual_id}"
                        + (
                            " via "
                            + str(actual_attributes["numeric_identity_source"])
                            if actual_attributes.get("numeric_identity_source")
                            else ""
                        )
                    )
                    prior = (
                        formal_attributes.get("result_class_ref"),
                        int(formal_attributes.get("precision_limbs") or 1),
                    )
                    _concord_source_value_class(
                        callee_scope,
                        formal_id,
                        str(class_identity),
                        precision_limbs=int(limbs),
                        source=transfer_source,
                    )
                    # The actual-to-formal edge transfers the instance's
                    # class identity; it does not turn the formal Input into
                    # a constructor.  ``class_ref`` is consumed by deployment
                    # as a request to call ``__init__``.  Stamping it here
                    # recursively constructed every specialized ``self`` and
                    # ``other`` parameter and exported the resulting private
                    # frames all the way to the root ABI.
                    formal_attributes.pop("class_ref", None)
                    formal_attributes.update({
                        "result_class_ref": str(class_identity),
                        "precision_limbs": int(limbs),
                        "numeric_identity_source": transfer_source,
                    })
                    descriptor = numeric_feature_descriptor(
                        str(class_identity) + (
                            f"[{int(limbs)}]" if int(limbs) > 1 else ""
                        )
                    )
                    if descriptor is not None:
                        formal_attributes["numeric_feature_descriptor"] = (
                            descriptor.receipt()
                        )
                    changed = changed or prior != (
                        str(class_identity), int(limbs),
                    )
        return changed

    # Operators reveal new exact callees; those call edges reveal the
    # concrete type of an unannotated ``self``/``other`` formal.  Settle the
    # two facts together rather than relying on traversal order.
    for _round in range(max(len(tuple(function_table)), 1) + 1):
        for entry in function_table:
            entry_wrapper = getattr(entry, "graph", None)
            if entry_wrapper is not None:
                propagate_returned_receiver_types(entry_wrapper.G)
                propagate_numeric_field_projections(entry_wrapper)
                lower_python_precision(entry_wrapper)
                specialize_concorded_same_type_numeric_operator(entry_wrapper)
                resolve_concorded_receiver_constructor(entry_wrapper)
                lower_class_operator_calls(entry_wrapper)
        if not propagate_call_formal_numeric_types():
            break

    for entry in function_table:
        entry_wrapper = getattr(entry, "graph", None)
        entry_graph = getattr(entry_wrapper, "G", None)
        if entry_graph is not None:
            # AugAssign read-modify-write structure and child roles are final
            # only after lexical normalization. Canonicalize its attribute
            # read at this post-extraction seam as well as on the root graph.
            normalize_python_attribute_special_cases(entry_wrapper)
            propagate_returned_receiver_types(entry_graph)
            propagate_numeric_field_projections(entry_wrapper)
            lower_python_precision(entry_wrapper)
            specialize_concorded_same_type_numeric_operator(entry_wrapper)
            lower_class_operator_calls(entry_wrapper, settled=True)
            # Operator calls can themselves return class instances.  Resolve
            # a following method (for example ``wide_product.collapse()``)
            # from the just-concorded result identity before call planning.
            propagate_returned_receiver_types(entry_graph)
    # External call references and static Python bindings are two views of the
    # same compile-time environment.  Join them once after every call has been
    # declared so compiled shells can invoke imported constructors/functions
    # without a second lookup mechanism.
    for entry in external_function_table:
        target = static_bindings.get(entry.name)
        if callable(target):
            external_function_table.resolve_callable(entry.reference, target)
    graph.python_bindings = canonicalize_python_static_bindings(
        graph.python_bindings
    )
    # Source discovery annotates definitions with their exact lexical Python
    # environment.  Most annotations are removed when their function graph is
    # finalized, but ProcessGraph.node_map deliberately retains every source
    # occurrence.  Canonicalize any surviving environment through the same
    # Python special case so a dead frontend annotation cannot reintroduce the
    # live wrapper after all actual uses have become Constant leaves.
    for expression in graph.node_map.values():
        bindings = getattr(expression, "_python_bindings", None)
        if isinstance(bindings, Mapping):
            expression._python_bindings = canonicalize_python_static_bindings(
                bindings
            )
    return graph


__all__ = [
    "normalize_python_attribute_special_cases",
    "reduce_abstract_tensor_topology",
    "specialize_python_precision_widths",
]
