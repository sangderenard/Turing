"""Package an emitted ``bind(C)`` Fortran module in a native C shell.

The C translation unit contains only the generic profiled launch boundary,
buffer ownership, declared state feedback, and diagnostics.  Program logic
remains in the :class:`~src.compiler.ssa_fortran_backend.FortranModule` that
the ordinary AST/Control/SSA pipeline emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import copy
from fnmatch import fnmatchcase
import json
import os
import sys
from pathlib import Path
import re
import subprocess
import ast
import inspect
import importlib
import hashlib
import textwrap
from typing import Any, Iterable, Mapping
from typing import Callable

import numpy as np

from ..common.tensors.accelerator_backends.profiled_c_shell import (
    _C_SOURCE, _C_TRACE_SOURCE,
)
from ..transmogrifier.graph.edge_roles import (
    keyword_argument_name,
    ordered_arguments,
    positional_argument_index,
)
from .fortran_toolchain import (
    aggressive_c_flags,
    aggressive_fortran_flags,
    standalone_fortran_link_flags,
    standalone_runtime_shim_sources,
)
from .ssa_fortran_backend import FortranEmissionError, fortran_compiler


_UNCOPYABLE_LITERAL_TYPES: set[str] = set()


def _copy_literal_payload(payload: Any) -> Any:
    """Deep-copy a literal payload captured off a graph node, if it can be.

    Ingested stdlib source (``re._compiler`` and friends) puts objects behind
    ``Constant`` nodes whose payloads are not always deep-copyable. Sharing
    the original reference is the honest fallback: the payload already had
    that identity everywhere upstream, so aliasing here adds no hazard a copy
    would have removed.
    """

    try:
        return copy.deepcopy(payload)
    except Exception:
        type_name = type(payload).__qualname__
        if type_name not in _UNCOPYABLE_LITERAL_TYPES:
            _UNCOPYABLE_LITERAL_TYPES.add(type_name)
            import warnings

            warnings.warn(
                f"literal payload of type {type_name} is not deep-copyable; "
                "sharing the original reference",
                RuntimeWarning,
                stacklevel=3,
            )
        return payload


_NUMPY_DTYPES = {
    "uint8": np.dtype("uint8"),
    "u8": np.dtype("uint8"),
    "bool": np.dtype("bool"),
    "logical": np.dtype("bool"),
    "float": np.dtype("float32"),
    "float32": np.dtype("float32"),
    "f32": np.dtype("float32"),
    "double": np.dtype("float64"),
    "float64": np.dtype("float64"),
    "f64": np.dtype("float64"),
    "int": np.dtype("int32"),
    "int32": np.dtype("int32"),
    "i32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "i64": np.dtype("int64"),
}


@dataclass(frozen=True)
class FortranCShellExecutable:
    directory: Path
    executable_path: Path
    fortran_source_path: Path
    c_source_path: Path
    api_path: Path
    initial_state_path: Path
    final_outputs_path: Path
    entrypoint: str

    def run(
        self,
        *,
        frames: int = 1,
        files: Mapping[str, str | Path] | None = None,
        stream_frames: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if frames < 0:
            raise ValueError("native C shell frame count cannot be negative")
        arguments = [str(self.executable_path), str(frames)]
        if stream_frames:
            arguments.append("--stream-frames")
        for name, path in sorted(dict(files or {}).items()):
            arguments.extend(("--file-" + _identifier(name).replace("_", "-"), str(Path(path).resolve())))
        return subprocess.run(
            arguments,
            cwd=str(self.directory),
            env=dict(os.environ),
            capture_output=capture_output,
            text=True,
            check=True,
        )


def _loop_carried_storage_aliases(graph_obj) -> dict[int, int]:
    """Graph-derived storage aliases for loop-carried in-place mutation.

    A loop that mutates an array through ``IndexedStore`` leaves a
    LOOPRESULT node as the array's post-loop identity. That identity is
    the SAME STORAGE as the array it mutated -- in-place is the point --
    but the planner schedules later consumers against the loopresult id,
    and a lowering that cannot link it materializes a fresh, unconnected
    FORMAL: the second of two sequential loops over one array then reads
    the original buffer and the first loop's stores silently vanish (the
    sequential same-array store defect, pinned in
    ``test_compiled_linalg.py``).

    The chase is deliberately narrow, one level per node, composed by the
    builder's existing alias-chain walk (``external_value``):

    * a ``loopresult``/``loopexit`` node aliases its preferred-role parent
      ONLY when that parent is itself an ``IndexedStore`` version or
      another loop identity -- scalar carried results (a ``total``) are
      genuine values with carried-port machinery of their own and are
      never touched;
    * an ``IndexedStore`` node aliases its ``base`` -- the store versions
      resident memory, it does not mint storage (the same rule
      ``ir_indexing`` applies inside a function, extended to the graph
      so it holds ACROSS loops).
    """

    aliases: dict[int, int] = {}
    chainable = {"indexedstore", "loopresult", "loopexit"}

    def node_kind(node_id) -> str:
        data = graph_obj.nodes.get(node_id) or {}
        return str(data.get("op") or data.get("type") or "").casefold()

    def value_id_of(node_id) -> int:
        data = graph_obj.nodes.get(node_id) or {}
        return int(data.get("value_id", node_id))

    for node_id, data in graph_obj.nodes(data=True):
        kind = str(data.get("op") or data.get("type") or "").casefold()
        parents = tuple(data.get("parents") or ())
        if kind == "indexedstore":
            base = next(
                (parent for parent, role in parents
                 if str(role) == "base" and parent in graph_obj),
                None,
            )
            if base is not None:
                aliases[value_id_of(node_id)] = value_id_of(base)
            continue
        if kind in {"loopresult", "loopexit"}:
            for preferred_role in (
                "updated", "value", "body", "initial", "orelse"
            ):
                parent = next(
                    (parent for parent, role in parents
                     if str(role) == preferred_role
                     and parent in graph_obj),
                    None,
                )
                if parent is None:
                    continue
                if node_kind(parent) in chainable:
                    aliases[value_id_of(node_id)] = value_id_of(parent)
                break
    # Never allow a cycle to reach the builder's chase (it guards with a
    # seen-set, but a self-alias is meaningless regardless).
    return {
        source: target for source, target in aliases.items()
        if source != target
    }


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not result or result[0].isdigit():
        result = "turing_" + result
    return result


def _record_receipts_for_function(
    program_abi: Mapping[str, Any],
    function_name: str,
    parameters: Iterable[str],
    *,
    method_owner: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Select explicit record bindings plus an exact method receiver schema.

    The receipt form is used after the extraction contract object has been
    reduced to serializable graph metadata.  Keep its selection semantics in
    step with :meth:`ProgramABIContract.records_for_function`: an unannotated
    ``self`` is safely identifiable from ``method_owner`` only when exactly
    one declared record has that class identity.
    """

    parameter_names = set(map(str, parameters))
    records = dict(program_abi.get("records") or {})
    selected: dict[str, Mapping[str, Any]] = {}
    for binding in tuple(program_abi.get("bindings") or ()):
        parameter = str(binding.get("parameter") or "")
        record_name = str(binding.get("record") or "")
        if (
            parameter in parameter_names
            and fnmatchcase(str(function_name), str(binding.get("function") or ""))
            and record_name in records
        ):
            selected[parameter] = records[record_name]
    if (
        method_owner is not None
        and "self" in parameter_names
        and "self" not in selected
    ):
        owner = str(method_owner)
        candidates = tuple(
            record
            for name, record in records.items()
            if (
                str(name) == owner
                or str(record.get("identity") or "") == owner
                or str(record.get("identity") or "").rsplit(".", 1)[-1]
                == owner
            )
        )
        if len(candidates) == 1:
            selected["self"] = candidates[0]
    return selected


def _authored_annotation_field_receipt(
    annotation: ast.AST,
) -> Mapping[str, Any] | None:
    """Return the exact scalar/fixed-span ABI stated by an annotation."""

    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        members = (annotation.left, annotation.right)
        concrete = tuple(
            member for member in members
            if not (isinstance(member, ast.Constant) and member.value is None)
            and not (isinstance(member, ast.Name) and member.id == "None")
        )
        if len(concrete) == 1 and len(concrete) != len(members):
            receipt = _authored_annotation_field_receipt(concrete[0])
            if receipt is not None:
                return {**dict(receipt), "optional": True}

    if isinstance(annotation, ast.Name):
        scalar = {
            "int": "int64", "float": "float64", "bool": "bool",
        }.get(annotation.id)
        if scalar is not None:
            return {
                "storage": "scalar", "dtype": scalar, "rank": 0,
                "mutable": False,
            }
        if annotation.id in {"bytes", "bytearray", "str"}:
            return {
                "storage": "span", "dtype": "int64", "rank": 1,
                "mutable": False,
                "aggregate_kind": annotation.id,
            }
        return None
    if not isinstance(annotation, ast.Subscript):
        return None
    container = (
        annotation.value.id
        if isinstance(annotation.value, ast.Name)
        else annotation.value.attr
        if isinstance(annotation.value, ast.Attribute)
        else ""
    )
    if container == "Literal":
        members = (
            tuple(annotation.slice.elts)
            if isinstance(annotation.slice, ast.Tuple)
            else (annotation.slice,)
        )
        vocabulary = tuple(
            str(member.value)
            for member in members
            if isinstance(member, ast.Constant)
            and isinstance(member.value, str)
        )
        if len(vocabulary) == len(members) and vocabulary:
            return {
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "mutable": False, "token_vocabulary": vocabulary,
            }
        return None
    if container == "Optional":
        receipt = _authored_annotation_field_receipt(annotation.slice)
        return (
            None if receipt is None
            else {**dict(receipt), "optional": True}
        )
    if container not in {"list", "tuple", "Sequence", "Iterable"}:
        return None
    element = annotation.slice
    fixed_length = None
    if isinstance(element, ast.Tuple) and element.elts:
        if container == "tuple" and not any(
            isinstance(item, ast.Constant) and item.value is Ellipsis
            for item in element.elts
        ):
            element_dtypes = tuple(
                item.id if isinstance(item, ast.Name) else None
                for item in element.elts
            )
            if len(set(element_dtypes)) != 1:
                return None
            fixed_length = len(element.elts)
        element = element.elts[0]
    if not isinstance(element, ast.Name):
        return None
    dtype = {
        "int": "int64", "float": "float64", "bool": "bool",
        "bytes": "int64", "bytearray": "int64", "str": "int64",
    }.get(element.id)
    if dtype is None:
        return None
    receipt = {
        "storage": "span", "dtype": dtype, "rank": 1,
        "mutable": False,
        "aggregate_kind": container.casefold(),
    }
    if fixed_length is not None:
        receipt["fixed_length"] = int(fixed_length)
    return receipt


def _authored_complete_record_schemas(
    tree: ast.Module,
) -> dict[str, Mapping[str, Any]]:
    """Publish dataclass constructors only when every field has a physical ABI.

    Per-function record views may safely expose only fields a function uses.
    Construction is stricter: a partial class layout would silently discard
    authored state, so one unsupported declared field refuses the whole schema.
    """

    schemas: dict[str, Mapping[str, Any]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        decorators = {
            target.id
            if isinstance(target, ast.Name)
            else target.attr
            if isinstance(target, ast.Attribute)
            else ""
            for decorator in statement.decorator_list
            for target in (
                decorator.func if isinstance(decorator, ast.Call) else decorator,
            )
        }
        if "dataclass" not in decorators:
            continue
        declared = [
            field for field in statement.body
            if isinstance(field, ast.AnnAssign)
            and isinstance(field.target, ast.Name)
        ]
        if not declared:
            continue
        fields: dict[str, Mapping[str, Any]] = {}
        complete = True
        for field in declared:
            receipt = _authored_annotation_field_receipt(field.annotation)
            if receipt is None:
                complete = False
                break
            receipt = dict(receipt)
            if field.value is not None:
                try:
                    receipt["default"] = ast.literal_eval(field.value)
                except (TypeError, ValueError):
                    pass
            fields[field.target.id] = receipt
        if complete:
            schemas[statement.name] = {
                "identity": statement.name,
                "fields": fields,
                "source_derived": True,
            }
    return schemas


def _authored_sequence_record_views(
    tree: ast.Module,
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Derive exact ``Sequence[Record]`` parameter row contracts.

    The sequence is caller-owned columnar storage; its loop target is a row
    correlation, never a Python object crossing the native ABI. Only complete
    authored record schemas are admitted, so every projected field has a
    declared physical representation before lowering begins.
    """

    views: dict[str, dict[str, Mapping[str, Any]]] = {}

    def element_record(annotation: ast.AST | None) -> str | None:
        if not isinstance(annotation, ast.Subscript):
            return None
        container = (
            annotation.value.id
            if isinstance(annotation.value, ast.Name)
            else annotation.value.attr
            if isinstance(annotation.value, ast.Attribute)
            else ""
        )
        if container not in {"list", "tuple", "Sequence", "Iterable"}:
            return None
        element = annotation.slice
        if isinstance(element, ast.Tuple) and len(element.elts) == 1:
            element = element.elts[0]
        identity = (
            element.id if isinstance(element, ast.Name)
            else element.attr if isinstance(element, ast.Attribute)
            else None
        )
        return str(identity) if identity in schemas else None

    def visit(
        qualified_name: str,
        definition: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        ):
            identity = element_record(argument.annotation)
            if identity is None:
                continue
            schema = dict(schemas[identity])
            views.setdefault(qualified_name, {})[argument.arg] = {
                "identity": str(schema.get("identity") or identity),
                "fields": copy.deepcopy(dict(schema.get("fields") or {})),
                "aggregate_kind": "tuple",
                "mutable": False,
                "source_derived": True,
            }

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(statement.name, statement)
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(f"{statement.name}.{member.name}", member)
    return views


def _authored_dataclass_record_views(
    tree: ast.Module,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Derive exact per-method record views from authored class fields.

    This is deliberately conservative.  Dataclass annotations and ordinary
    ``__init__`` assignments are admitted only when they have an unambiguous
    physical scalar/span representation.  A method that touches an unsupported
    field receives no inferred receiver view.  Per-method views include only
    fields that method reads or writes (plus fields required by same-receiver
    method calls), so compiling a small method does not manufacture ABI
    obligations for unrelated state.  Mutation is recorded in that view; the
    initializer's construction writes do not make every field mutable forever.

    The historical function name is retained because it is imported by tests
    and downstream tooling, but ordinary authored classes are intentionally
    part of the same source-derived ABI faculty.
    """

    classes = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
    }

    def field_receipt(annotation: ast.AST) -> Mapping[str, Any] | None:
        return _authored_annotation_field_receipt(annotation)

    def initialized_field_receipt(value: ast.AST) -> Mapping[str, Any] | None:
        """Prove a scalar field from its authored initializer, without guessing."""

        scalar_type = None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"int", "float", "bool"}
        ):
            scalar_type = value.func.id
        elif isinstance(value, ast.Constant):
            # bool is an int subclass, so order is significant here.
            if isinstance(value.value, bool):
                scalar_type = "bool"
            elif isinstance(value.value, int):
                scalar_type = "int"
            elif isinstance(value.value, float):
                scalar_type = "float"
        dtype = {
            "int": "int64", "float": "float64", "bool": "bool",
        }.get(scalar_type or "")
        if dtype is None:
            return None
        return {
            "storage": "scalar", "dtype": dtype, "rank": 0,
            "mutable": False,
        }

    class_fields: dict[str, dict[str, Mapping[str, Any]]] = {}
    class_methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for class_name, definition in classes.items():
        fields = {}
        methods = {}
        for statement in definition.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                receipt = field_receipt(statement.annotation)
                if receipt is not None:
                    fields[statement.target.id] = receipt
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[statement.name] = statement
        initializer = methods.get("__init__")
        if initializer is not None:
            for node in ast.walk(initializer):
                target = None
                receipt = None
                if isinstance(node, ast.AnnAssign):
                    target = node.target
                    receipt = field_receipt(node.annotation)
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    receipt = initialized_field_receipt(node.value)
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and receipt is not None
                ):
                    continue
                previous = fields.get(target.attr)
                if previous is None:
                    fields[target.attr] = receipt
                elif dict(previous) != dict(receipt):
                    # Conflicting physical declarations are not a proven ABI.
                    fields.pop(target.attr, None)
        class_fields[class_name] = fields
        class_methods[class_name] = methods

    required_cache: dict[tuple[str, str], dict[str, bool] | None] = {}

    def method_fields(
        class_name: str, method_name: str, active=(),
    ) -> dict[str, bool] | None:
        key = (class_name, method_name)
        if key in required_cache:
            return required_cache[key]
        if key in active:
            return {}
        method = class_methods.get(class_name, {}).get(method_name)
        if method is None:
            return None
        direct: dict[str, bool] = {}
        nested_calls: set[str] = set()
        for node in ast.walk(method):
            if not isinstance(node, ast.Attribute):
                continue
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
            ):
                continue
            if node.attr in class_methods.get(class_name, {}):
                if isinstance(getattr(node, "ctx", None), ast.Load):
                    nested_calls.add(node.attr)
            else:
                direct[node.attr] = bool(
                    direct.get(node.attr, False)
                    or isinstance(getattr(node, "ctx", None), ast.Store)
                )
        if not set(direct) <= set(class_fields.get(class_name, {})):
            required_cache[key] = None
            return None
        for callee in sorted(nested_calls):
            inherited = method_fields(class_name, callee, (*active, key))
            if inherited is None:
                required_cache[key] = None
                return None
            for field_name, mutable in inherited.items():
                direct[field_name] = bool(
                    direct.get(field_name, False) or mutable
                )
        required_cache[key] = direct
        return direct

    views: dict[str, dict[str, Mapping[str, Any]]] = {}
    for class_name, methods in class_methods.items():
        for method_name in methods:
            fields = method_fields(class_name, method_name)
            if fields is None or not fields:
                continue
            views[f"{class_name}.{method_name}"] = {
                "self": {
                    "identity": class_name,
                    "fields": {
                        name: dict(class_fields[class_name][name])
                        for name in sorted(fields)
                    },
                }
            }
            for name, mutable in fields.items():
                views[f"{class_name}.{method_name}"]["self"]["fields"][name][
                    "mutable"
                ] = mutable

    def annotation_class_name(annotation: ast.AST | None) -> str | None:
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Attribute):
            return annotation.attr
        return None

    def add_annotated_parameter_views(
        qualified_name: str,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        arguments = (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
        for argument in arguments:
            if argument.arg in {"self", "cls"}:
                continue
            record_name = annotation_class_name(argument.annotation)
            if record_name not in class_fields:
                continue
            used: dict[str, bool] = {}
            nested_calls: set[str] = set()
            for node in ast.walk(method):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == argument.arg
                ):
                    continue
                if node.attr in class_methods.get(record_name, {}):
                    if isinstance(getattr(node, "ctx", None), ast.Load):
                        nested_calls.add(node.attr)
                else:
                    used[node.attr] = bool(
                        used.get(node.attr, False)
                        or isinstance(getattr(node, "ctx", None), ast.Store)
                    )
            for callee in sorted(nested_calls):
                inherited = method_fields(record_name, callee)
                if inherited is None:
                    used = {}
                    break
                for field_name, mutable in inherited.items():
                    used[field_name] = bool(
                        used.get(field_name, False) or mutable
                    )
            if not used:
                continue
            if not set(used) <= set(class_fields[record_name]):
                # The annotation names the class, but this function needs a
                # field whose physical representation was not proven. Do not
                # publish a partial parameter ABI.
                continue
            fields = {
                name: {
                    **dict(class_fields[record_name][name]),
                    "mutable": bool(mutable),
                }
                for name, mutable in sorted(used.items())
            }
            views.setdefault(qualified_name, {})[argument.arg] = {
                "identity": record_name,
                "fields": fields,
            }

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_annotated_parameter_views(statement.name, statement)
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_annotated_parameter_views(
                        f"{statement.name}.{member.name}", member,
                    )
    return views


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
    extents: Mapping[str, int],
    linear_index: str,
) -> str:
    """Map one C-row-major logical index to Fortran array storage.

    The API shape is semantic and remains in Python/NumPy dimension order.
    A ``bind(C)`` Fortran dummy with that shape stores its first dimension
    fastest, so the outer shell must perform this boundary permutation once.
    Resident feedback arenas stay in Fortran order and require no copies.
    """

    dynamic_dimensions = tuple(getattr(parameter, "extents", ()) or ())
    shape = (
        tuple(int(extents[str(name)]) for name in dynamic_dimensions)
        if dynamic_dimensions
        else tuple(
            int(extents.get(f"extent_{int(size)}", size))
            for size in tuple(parameter.shape or ())
        )
    )
    if len(shape) <= 1:
        return linear_index
    terms = []
    for dimension, size in enumerate(shape):
        c_stride = 1
        for following in shape[dimension + 1:]:
            c_stride *= int(following)
        fortran_stride = 1
        for preceding in shape[:dimension]:
            fortran_stride *= int(preceding)
        coordinate = (
            f"(({linear_index}) / {c_stride}) % {size}"
            if c_stride != 1
            else f"({linear_index}) % {size}"
        )
        terms.append(
            coordinate
            if fortran_stride == 1
            else f"({coordinate}) * {fortran_stride}"
        )
    return " + ".join(terms)


def _c_string(value: str) -> str:
    return json.dumps(str(value))


# Pixel formats the native C shell's presenter understands, and whether each
# carries a per-pixel alpha channel.  ``rgba_f64_planar_layered`` is the only
# multi-layer format; its layer count comes from the request's own
# ``layer_count`` attribute rather than the format name.
_DISPLAY_PIXEL_FORMATS = {
    "rgb_f64_planar": False,
    "rgba_f64_planar": True,
    "rgba_f64_planar_layered": True,
}


_C_FILE_BROKER_SOURCE = r'''
#define TURING_FILE_HANDLE_CAPACITY 256
static FILE *turing_file_handles[TURING_FILE_HANDLE_CAPACITY] = {0};

static char *turing_file_span_string(const uint8_t *data, int64_t length) {
    char *text;
    if (data == NULL || length < 0 || (uint64_t)length >= SIZE_MAX) return NULL;
    text = (char *)malloc((size_t)length + 1);
    if (text == NULL) return NULL;
    memcpy(text, data, (size_t)length);
    text[length] = '\0';
    return text;
}

static FILE *turing_file_handle(int64_t handle) {
    if (handle < 1 || handle >= TURING_FILE_HANDLE_CAPACITY) return NULL;
    return turing_file_handles[handle];
}

int64_t turing_shell_file_open(
    const uint8_t *path_data, int64_t path_length,
    const uint8_t *mode_data, int64_t mode_length
) {
    char *path = turing_file_span_string(path_data, path_length);
    char *mode = turing_file_span_string(mode_data, mode_length);
    FILE *stream = NULL;
    int64_t handle;
    if (path != NULL && mode != NULL && mode_length > 0 && mode_length < 8) {
        stream = fopen(path, mode);
    }
    free(path);
    free(mode);
    if (stream == NULL) return 0;
    for (handle = 1; handle < TURING_FILE_HANDLE_CAPACITY; ++handle) {
        if (turing_file_handles[handle] == NULL) {
            turing_file_handles[handle] = stream;
            return handle;
        }
    }
    fclose(stream);
    return 0;
}

int64_t turing_shell_file_read(
    int64_t handle, uint8_t *destination, int64_t capacity
) {
    FILE *stream = turing_file_handle(handle);
    size_t count;
    if (stream == NULL || destination == NULL || capacity < 0) return -1;
    count = fread(destination, 1, (size_t)capacity, stream);
    return ferror(stream) ? -1 : (int64_t)count;
}

int64_t turing_shell_file_write(
    int64_t handle, const uint8_t *source, int64_t length
) {
    FILE *stream = turing_file_handle(handle);
    size_t count;
    if (stream == NULL || source == NULL || length < 0) return -1;
    count = fwrite(source, 1, (size_t)length, stream);
    return count == (size_t)length ? (int64_t)count : -1;
}

int32_t turing_shell_file_seek(int64_t handle, int64_t offset, int32_t origin) {
    FILE *stream = turing_file_handle(handle);
    if (stream == NULL || origin < SEEK_SET || origin > SEEK_END) return 0;
#if defined(_WIN32)
    return _fseeki64(stream, offset, origin) == 0;
#else
    return fseeko(stream, (off_t)offset, origin) == 0;
#endif
}

int64_t turing_shell_file_tell(int64_t handle) {
    FILE *stream = turing_file_handle(handle);
    if (stream == NULL) return -1;
#if defined(_WIN32)
    return (int64_t)_ftelli64(stream);
#else
    return (int64_t)ftello(stream);
#endif
}

int32_t turing_shell_file_flush(int64_t handle) {
    FILE *stream = turing_file_handle(handle);
    return stream != NULL && fflush(stream) == 0;
}

int32_t turing_shell_file_close(int64_t handle) {
    FILE *stream = turing_file_handle(handle);
    int status;
    if (stream == NULL) return 0;
    turing_file_handles[handle] = NULL;
    status = fclose(stream);
    return status == 0;
}

int64_t turing_shell_file_stat_size(
    const uint8_t *path_data, int64_t path_length
) {
    char *path = turing_file_span_string(path_data, path_length);
    FILE *stream;
    int64_t size = -1;
    if (path == NULL) return -1;
    stream = fopen(path, "rb");
    free(path);
    if (stream == NULL) return -1;
#if defined(_WIN32)
    if (_fseeki64(stream, 0, SEEK_END) == 0) size = (int64_t)_ftelli64(stream);
#else
    if (fseeko(stream, 0, SEEK_END) == 0) size = (int64_t)ftello(stream);
#endif
    fclose(stream);
    return size;
}

static void turing_shell_file_close_all(void) {
    int64_t handle;
    for (handle = 1; handle < TURING_FILE_HANDLE_CAPACITY; ++handle) {
        if (turing_file_handles[handle] != NULL) {
            fclose(turing_file_handles[handle]);
            turing_file_handles[handle] = NULL;
        }
    }
}

static int turing_shell_file_fail(int status) {
    turing_shell_file_close_all();
    return status;
}
'''


def _requires_file_broker(module: Any) -> bool:
    metadata = dict(getattr(module.api, "metadata", {}) or {})
    shell_io = dict(metadata.get("shell_io") or {})
    requirements = dict(shell_io.get("requirements") or {})
    if shell_io.get("boundary_plans"):
        return True
    return any(
        request.get("capability") == "files"
        for request in requirements.get("requests", ())
    )


def _native_shell_boundary_lines(
    module: Any,
    entry: Any,
    values: tuple[Any, ...],
    slot_by_parameter: Mapping[str, int],
    extents: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Render file-boundary plans as native-shell C statements.

    The target entry remains an ordinary numerical/control ABI. These lines
    run in the enclosing C shell after that entry and use its already allocated
    parameter slots. No Python file implementation and no filesystem operator
    is admitted to Fortran/LLVM/GLSL emission.
    """

    shell_io = dict((getattr(module.api, "metadata", {}) or {}).get(
        "shell_io"
    ) or {})
    plans = tuple(shell_io.get("boundary_plans") or ())
    if not plans:
        return (), (), ()
    if shell_io.get("boundary_plan_schema") != "turing.shell-boundary-plan.v1":
        raise ValueError("native shell received an unknown boundary-plan schema")

    parameters_by_source: dict[str, list[Any]] = {}
    for parameter in values:
        parameters_by_source.setdefault(
            str(parameter.source_name or parameter.name), []
        ).append(parameter)
        parameters_by_source.setdefault(str(parameter.name), []).append(parameter)

    globals_: list[str] = []
    declarations: list[str] = []
    actions: list[str] = []
    locals_by_name: dict[str, str] = {}
    public_by_name = {
        str(parameter.source_name or parameter.name): parameter
        for parameter in values
        if parameter.role in {"output", "inout"}
    }
    literal_index = 0

    def local(name: str) -> str:
        key = str(name)
        existing = locals_by_name.get(key)
        if existing is not None:
            return existing
        identifier = "turing_shell_value_" + _identifier(key)
        if identifier in locals_by_name.values():
            identifier += "_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        locals_by_name[key] = identifier
        declarations.append(f"    int64_t {identifier} = 0;")
        return identifier

    def publish(name: str, expression: str) -> list[str]:
        target = public_by_name.get(str(name))
        if target is None:
            return []
        slot = slot_by_parameter[target.name]
        return [
            f"        *(({target.c_type} *)slots[{slot}]) = "
            f"({target.c_type})({expression});"
        ]

    def scalar(operand: Mapping[str, Any]) -> str:
        kind = str(operand.get("kind") or "")
        if kind == "name":
            return local(str(operand["name"]))
        if kind == "literal" and isinstance(operand.get("value"), (int, bool)):
            return str(int(operand["value"]))
        raise ValueError(f"native shell cannot resolve scalar operand {operand!r}")

    def span(operand: Mapping[str, Any]) -> tuple[str, str]:
        nonlocal literal_index
        kind = str(operand.get("kind") or "")
        if kind in {"literal", "bytes"}:
            if kind == "bytes":
                payload = bytes.fromhex(str(operand.get("hex") or ""))
            else:
                value = operand.get("value")
                if not isinstance(value, str):
                    raise ValueError(
                        f"native shell span literal must be text/bytes: {operand!r}"
                    )
                payload = value.encode("utf-8")
            literal_index += 1
            identifier = f"turing_shell_literal_{literal_index}"
            elements = ", ".join(str(byte) for byte in payload) or "0"
            globals_.append(
                f"static const uint8_t {identifier}[] = {{{elements}}};"
            )
            return identifier, str(len(payload))
        if kind != "name":
            raise ValueError(f"native shell cannot resolve span operand {operand!r}")
        source_name = str(operand["name"])
        candidates = list(dict.fromkeys(parameters_by_source.get(source_name, ())))
        data = next((
            parameter for parameter in candidates
            if str(parameter.c_type) == "uint8_t"
            and parameter.passing == "reference"
        ), None)
        if data is None:
            raise ValueError(
                f"shell boundary name {source_name!r} has no uint8 span parameter"
            )
        data_slot = slot_by_parameter[data.name]
        length_candidates = tuple(dict.fromkeys((
            *candidates,
            *parameters_by_source.get(source_name + "_length", ()),
        )))
        length = next((
            parameter for parameter in length_candidates
            if parameter is not data
            and str(parameter.c_type) in {"int32_t", "int64_t"}
            and (
                str(parameter.source_transform or "").endswith("length")
                or str(parameter.name).casefold().endswith("length")
            )
        ), None)
        if length is not None:
            length_slot = slot_by_parameter[length.name]
            length_expression = (
                f"*(({length.c_type} *)slots[{length_slot}])"
            )
        else:
            length_expression = str(_element_count(data, extents))
        return f"(const uint8_t *)slots[{data_slot}]", length_expression

    for plan in plans:
        function_name = plan.get("function")
        if function_name not in {None, "", entry.name, entry.symbol}:
            continue
        operations = tuple(sorted(
            (dict(item) for item in plan.get("operations", ())),
            key=lambda item: int(item.get("sequence", -1)),
        ))
        if [int(item.get("sequence", -1)) for item in operations] != list(
            range(len(operations))
        ):
            raise ValueError("shell boundary operations are not contiguous and ordered")
        if any(str(item.get("operation")) == "read" for item in operations):
            raise ValueError(
                "native read boundary requires a pre-entry span publication plan"
            )
        for operation in operations:
            name = str(operation.get("operation") or "")
            arguments = tuple(operation.get("arguments") or ())
            result_name = operation.get("result")
            if name == "open" and len(arguments) == 2:
                path_data, path_length = span(arguments[0])
                mode_data, mode_length = span(arguments[1])
                if not result_name:
                    raise ValueError("shell file open must publish its handle")
                target = local(str(result_name))
                actions.append(
                    f"        {target} = turing_shell_file_open("
                    f"{path_data}, {path_length}, {mode_data}, {mode_length});"
                )
                actions.append(
                    f"        if ({target} == 0) "
                    "return turing_shell_file_fail(10);"
                )
            elif name == "write" and len(arguments) == 2:
                handle = scalar(arguments[0])
                data, length = span(arguments[1])
                target = local(str(result_name or (
                    f"discard_write_{operation.get('sequence')}"
                )))
                actions.append(
                    f"        {target} = turing_shell_file_write("
                    f"{handle}, {data}, {length});"
                )
                actions.append(
                    f"        if ({target} < 0) "
                    "return turing_shell_file_fail(11);"
                )
                if result_name:
                    actions.extend(publish(str(result_name), target))
            elif name in {"flush", "close"} and len(arguments) == 1:
                handle = scalar(arguments[0])
                actions.append(
                    f"        if (!turing_shell_file_{name}({handle})) "
                    "return turing_shell_file_fail(12);"
                )
            elif name == "tell" and len(arguments) == 1:
                if not result_name:
                    raise ValueError("shell file tell must publish its result")
                target = local(str(result_name))
                actions.append(
                    f"        {target} = turing_shell_file_tell("
                    f"{scalar(arguments[0])});"
                )
                actions.append(
                    f"        if ({target} < 0) "
                    "return turing_shell_file_fail(13);"
                )
                actions.extend(publish(str(result_name), target))
            elif name == "seek" and len(arguments) in {2, 3}:
                origin = scalar(arguments[2]) if len(arguments) == 3 else "0"
                actions.append(
                    "        if (!turing_shell_file_seek("
                    f"{scalar(arguments[0])}, {scalar(arguments[1])}, "
                    f"(int32_t)({origin}))) "
                    "return turing_shell_file_fail(14);"
                )
            else:
                raise ValueError(
                    f"native shell does not implement boundary operation {name!r}"
                )
    return tuple(globals_), tuple(declarations), tuple(actions)


def _display_configuration(module: Any, entry: Any) -> dict[str, Any] | None:
    """Resolve an optional declarative display request from the shared IO ABI.

    Three presenter shapes share one physical contract (a caller-owned
    per-layer red/green/blue[/alpha] output array, resolved by name):
    a plain opaque blit (``rgb_f64_planar``), an alpha-blended single-layer
    blit (``rgba_f64_planar``), and an alpha-composited stack of layers
    (``rgba_f64_planar_layered``, back-to-front "over" compositing, one
    request attribute ``layer_count``).  ``prefers_compute``/
    ``prefers_accelerator`` are declared deployment hints, not consumed by
    this presenter (the only backend registered today is host-native GDI,
    which has no compute/accelerator distinction to honor) -- they are
    validated and carried through so a future GL/compute-capable wrapper can
    read them without a second display-request format.
    """

    metadata = dict(getattr(module.api, "metadata", {}) or {})
    shell_io = metadata.get("shell_io") or {}
    requirements = shell_io.get("requirements") or {}
    requests = [
        request for request in requirements.get("requests", ())
        if request.get("capability") == "display_double_buffer"
    ]
    if not requests:
        return None
    if len(requests) != 1:
        raise ValueError("C shell requires one display_double_buffer request")
    attributes = dict(requests[0].get("attributes") or {})
    pixel_format = str(attributes.get("pixel_format", "rgb_f64_planar"))
    has_alpha = _DISPLAY_PIXEL_FORMATS.get(pixel_format)
    if has_alpha is None:
        raise ValueError(
            "native C shell currently supports display pixel formats "
            + ", ".join(map(repr, sorted(_DISPLAY_PIXEL_FORMATS)))
            + f"; got {pixel_format!r}"
        )
    if pixel_format == "rgba_f64_planar_layered":
        layer_count = int(attributes.get("layer_count", 0))
        if layer_count < 1:
            raise ValueError(
                "rgba_f64_planar_layered display requires a positive "
                "layer_count attribute"
            )
    else:
        layer_count = 1
    width = int(attributes.get("width", 0))
    height = int(attributes.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError("native display request needs positive width and height")
    bindings = {
        str(binding.get("resource")): str(binding.get("parameter"))
        for binding in requirements.get("bindings", ())
        if str(binding.get("entry_point")) == str(entry.name)
        and str(binding.get("resource", "")).startswith("display.")
    }
    layered = pixel_format == "rgba_f64_planar_layered"
    # Every layer needs red/green/blue.  Alpha is mandatory for the
    # single-layer alpha format (that is the whole point of requesting it)
    # but optional per layer within a layered stack -- a layer with no
    # alpha binding is simply opaque, matching the presenter's own
    # ``alpha_layers[layer] ? ... : 1.0`` fallback.
    required_channels = ("red", "green", "blue", "alpha") if (
        has_alpha and not layered
    ) else ("red", "green", "blue")
    optional_channels = ("alpha",) if layered else ()

    def _resource(layer: int, channel: str) -> str:
        return f"display.layer{layer}.{channel}" if layered else f"display.{channel}"

    missing = {
        _resource(layer, channel)
        for layer in range(layer_count)
        for channel in required_channels
    } - set(bindings)
    if missing:
        raise ValueError(
            f"{pixel_format} display lacks bindings: " + ", ".join(sorted(missing))
        )
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    layers = []
    for layer in range(layer_count):
        channels = {}
        for channel in (*required_channels, *optional_channels):
            resource = _resource(layer, channel)
            if resource not in bindings:
                continue
            parameter = parameters.get(bindings[resource])
            if parameter is None or parameter.role != "output":
                raise ValueError(f"{resource} must bind an output ABI parameter")
            if str(parameter.c_type) != "double":
                raise ValueError(f"{resource} must bind a float64 output")
            channels[channel] = parameter.name
        layers.append(channels)
    return {
        "width": width,
        "height": height,
        "title": str(attributes.get("title", "Turing native display")),
        "pixel_format": pixel_format,
        "has_alpha": has_alpha,
        # One dict per layer, {"red": param_name, "green": ..., ...};
        # opaque formats simply omit the "alpha" key per layer.
        "layers": tuple(layers),
        # Back-compat single-layer accessor: every existing caller reads
        # ``display["channels"]`` for the (always exactly rgb) case.
        "channels": (
            tuple(layers[0][channel] for channel in ("red", "green", "blue"))
            if layer_count == 1 else ()
        ),
        "frame_delay_ms": max(0, int(attributes.get("frame_delay_ms", 0))),
        "prefers_compute": bool(attributes.get("prefers_compute", False)),
        "prefers_accelerator": bool(attributes.get("prefers_accelerator", False)),
    }


def _system_file_configurations(module: Any, entry: Any) -> tuple[dict[str, Any], ...]:
    metadata = dict(getattr(module.api, "metadata", {}) or {})
    requirements = dict((metadata.get("shell_io") or {}).get("requirements") or {})
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    configurations = []
    for port in requirements.get("system_ports", ()):
        if port.get("kind") != "file" or port.get("direction") not in {
            "input", "bidirectional",
        }:
            continue
        if str(port.get("entry_point")) != str(entry.name):
            continue
        fields = {
            str(field.get("name")): str(field.get("parameter"))
            for field in port.get("fields", ())
        }
        if set(fields) < {"data", "length"}:
            raise ValueError(f"native file port {port.get('name')!r} lacks data/length fields")
        data = parameters.get(fields["data"])
        length = parameters.get(fields["length"])
        if data is None or length is None:
            raise ValueError(f"native file port {port.get('name')!r} has unknown parameters")
        if str(data.c_type) != "uint8_t" or data.passing != "reference":
            raise ValueError("native file data parameter must be a uint8 reference")
        if str(length.c_type) not in {"int32_t", "int64_t"}:
            raise ValueError("native file length parameter must be int32 or int64")
        attributes = dict(port.get("attributes") or {})
        capacity = int(attributes.get("maximum_bytes", _element_count(data, _extent_values(entry, None))))
        if capacity < 1:
            raise ValueError("native input file capacity must be positive")
        configurations.append({
            "name": str(port["name"]),
            "flag": "--file-" + _identifier(str(port["name"])).replace("_", "-"),
            "data": data,
            "length": length,
            "capacity": capacity,
            "optional": bool(port.get("optional")),
        })
    return tuple(configurations)


def emit_fortran_c_shell_source(
    module: Any,
    *,
    entrypoint: str | None = None,
    state_feedback: Mapping[str, str] | None = None,
    extent_overrides: Mapping[str, int] | None = None,
    initial_state_filename: str = "initial-state.bin",
    final_outputs_filename: str = "final-outputs.bin",
    trace: bool = False,
    trace_capacity: int = 4096,
) -> str:
    """Emit a standalone C main around one described Fortran entry point.

    ``trace`` compiles the launch digest IN. It is a compile-time decision,
    not a runtime flag: with it off the ring, its logger and the hook that
    would call it are absent from the binary entirely, so a launch pays
    nothing for a facility it was not built with. With it on, every launch
    writes one fixed-size record -- sequence, shell ns, device ns, region,
    status -- into a ring the executable owns, and main drains it at the
    end. Nothing crosses a language boundary while the program runs.
    """

    entry = _entrypoint(module, entrypoint)
    parameters = tuple(entry.parameters)
    extents = _extent_values(entry, extent_overrides)
    values = tuple(item for item in parameters if item.role != "extent")
    inputs = tuple(item for item in values if item.role in {"input", "inout"})
    outputs = tuple(item for item in values if item.role in {"output", "inout"})
    unsupported = tuple(
        item
        for item in values
        if item.role not in {"input", "inout", "workspace", "output"}
    )
    if unsupported:
        raise ValueError(
            "C shell cannot allocate parameter roles "
            + ", ".join(sorted({item.role for item in unsupported}))
        )
    slot_by_name = {
        _source_name(parameter): index
        for index, parameter in enumerate(values)
    }
    slot_by_parameter = {
        parameter.name: index for index, parameter in enumerate(values)
    }
    (
        shell_boundary_globals,
        shell_boundary_declarations,
        shell_boundary_actions,
    ) = _native_shell_boundary_lines(
        module, entry, values, slot_by_parameter, extents,
    )
    display = _display_configuration(module, entry)
    file_ports = _system_file_configurations(module, entry)
    system_parameters = {
        parameter.name
        for port in file_ports
        for parameter in (port["data"], port["length"])
    }
    if display is not None:
        expected_pixels = int(display["width"]) * int(display["height"])
        for layer in display["layers"]:
            for parameter_name in layer.values():
                parameter = next(
                    item for item in values if item.name == parameter_name
                )
                if _element_count(parameter, extents) != expected_pixels:
                    raise ValueError(
                        f"display channel {parameter_name!r} has "
                        f"{_element_count(parameter, extents)} elements; expected "
                        f"{expected_pixels}"
                    )
    feedback = dict(state_feedback or {})
    missing_feedback = {
        name
        for pair in feedback.items()
        for name in pair
        if name not in slot_by_name
    }
    if missing_feedback:
        raise ValueError(
            "state feedback references absent Fortran ABI names: "
            + ", ".join(sorted(missing_feedback))
        )

    prototype_arguments = []
    call_arguments = []
    value_index = 0
    for parameter in parameters:
        c_type = str(parameter.c_type)
        if parameter.role == "extent":
            prototype_arguments.append(c_type)
            call_arguments.append(str(extents[parameter.name]))
            continue
        pointer = parameter.passing == "reference"
        prototype_arguments.append(c_type + (" *" if pointer else ""))
        slot = f"slots[{value_index}]"
        call_arguments.append(
            f"({c_type} *){slot}" if pointer
            else f"*(({c_type} *){slot})"
        )
        value_index += 1

    allocation_lines = []
    input_read_lines = []
    for index, parameter in enumerate(values):
        c_type = str(parameter.c_type)
        file_port = next((port for port in file_ports if port["data"].name == parameter.name), None)
        count = int(file_port["capacity"]) if file_port else _element_count(parameter, extents)
        allocation_lines.extend((
            f"    slots[{index}] = calloc({count}, sizeof({c_type}));",
            f"    if (!slots[{index}]) return 3;",
        ))
        if parameter.role in {"input", "inout"} and parameter.name not in system_parameters:
            if len(
                tuple(getattr(parameter, "extents", ()) or parameter.shape or ())
            ) <= 1:
                input_read_lines.extend((
                    f"    if (fread(slots[{index}], sizeof({c_type}), {count}, state) "
                    f"!= {count}) {{",
                    f"        fprintf(stderr, \"short initial state at {_c_string(_source_name(parameter))[1:-1]}\\n\");",
                    "        return 4;",
                    "    }",
                ))
            else:
                storage_index = _fortran_storage_index(
                    parameter, extents, "logical_index"
                )
                input_read_lines.extend((
                    "    { size_t logical_index;",
                    f"      for (logical_index = 0; logical_index < {count}; ++logical_index) {{",
                    f"        {c_type} element;",
                    f"        if (fread(&element, sizeof({c_type}), 1, state) != 1) {{",
                    f"          fprintf(stderr, \"short initial state at {_c_string(_source_name(parameter))[1:-1]}\\n\");",
                    "          return 4;",
                    "        }",
                    f"        (({c_type} *)slots[{index}])[{storage_index}] = element;",
                    "      }",
                    "    }",
                ))

    file_load_lines = []
    for port in file_ports:
        data_slot = slot_by_parameter[port["data"].name]
        length_slot = slot_by_parameter[port["length"].name]
        variable = _identifier("file_" + port["name"])
        file_load_lines.extend((
            f"    const char *{variable} = turing_argument_value(argc, argv, {_c_string(port['flag'])});",
            *(
                (f"    if ({variable} == NULL) {{ fprintf(stderr, \"missing {port['flag']}\\n\"); return 8; }}",)
                if not port["optional"] else ()
            ),
            f"    if ({variable} != NULL) {{",
            "        size_t loaded_bytes = 0;",
            f"        if (!turing_read_file({variable}, (uint8_t *)slots[{data_slot}], {port['capacity']}, &loaded_bytes)) return 9;",
            f"        *(({port['length'].c_type} *)slots[{length_slot}]) = ({port['length'].c_type})loaded_bytes;",
            "    }",
        ))

    feedback_lines = []
    feedback_finalize_lines = []
    for input_name, output_name in feedback.items():
        input_slot = slot_by_name[input_name]
        output_slot = slot_by_name[output_name]
        input_parameter = values[input_slot]
        output_parameter = values[output_slot]
        if (
            input_parameter.c_type != output_parameter.c_type
            or _element_count(input_parameter, extents)
            != _element_count(output_parameter, extents)
        ):
            raise ValueError(
                f"state feedback {input_name!r}->{output_name!r} has "
                "incompatible storage"
            )
        swap = (
            f"{{ void *feedback_arena = slots[{input_slot}]; "
            f"slots[{input_slot}] = slots[{output_slot}]; "
            f"slots[{output_slot}] = feedback_arena; }}"
        )
        feedback_lines.append(f"        {swap}")
        # After the last frame the latest value is in the input address. Swap
        # once more so the public output name still denotes the final result
        # for serialization and caller inspection.
        feedback_finalize_lines.append(f"    {swap}")

    output_lines = []
    frame_output_lines = []
    output_write_lines = []
    for output_index, parameter in enumerate(outputs):
        # Several native parameters may retain the same authored source_name
        # after whole-program linking.  Publication must address this exact
        # output parameter, not whichever same-source argument happened to be
        # last in the signature.
        slot = slot_by_parameter[parameter.name]
        count = _element_count(parameter, extents)
        separator = "" if output_index == 0 else ","
        output_lines.extend((
            f"    {{ double sum = 0.0; size_t i;",
            f"      for (i = 0; i < {count}; ++i) sum += (({parameter.c_type} *)slots[{slot}])[i];",
            f"      printf(\"{separator}\\\"{_source_name(parameter)}\\\":{{\\\"first\\\":%.17g,\\\"sum\\\":%.17g}}\",",
            f"             (double)(({parameter.c_type} *)slots[{slot}])[0], sum); }}",
        ))
        frame_output_lines.extend((
            f"        {{ double sum = 0.0; size_t i;",
            f"          for (i = 0; i < {count}; ++i) sum += (({parameter.c_type} *)slots[{slot}])[i];",
            f"          printf(\"{separator}\\\"{_source_name(parameter)}\\\":{{\\\"first\\\":%.17g,\\\"sum\\\":%.17g}}\",",
            f"                 (double)(({parameter.c_type} *)slots[{slot}])[0], sum); }}",
        ))
        if len(
            tuple(getattr(parameter, "extents", ()) or parameter.shape or ())
        ) <= 1:
            output_write_lines.append(
                f"    fwrite(slots[{slot}], sizeof({parameter.c_type}), {count}, outputs_file);"
            )
        else:
            storage_index = _fortran_storage_index(
                parameter, extents, "logical_index"
            )
            output_write_lines.extend((
                "    { size_t logical_index;",
                f"      for (logical_index = 0; logical_index < {count}; ++logical_index) {{",
                f"        const {parameter.c_type} *element = &(({parameter.c_type} *)slots[{slot}])[{storage_index}];",
                f"        fwrite(element, sizeof({parameter.c_type}), 1, outputs_file);",
                "      }",
                "    }",
            ))

    display_source = ""
    display_open_lines: list[str] = []
    display_loop_condition = "frame < frames"
    display_message_lines: list[str] = []
    display_present_lines: list[str] = []
    display_close_lines: list[str] = []
    default_frames = "1"
    if display is not None:
        layer_slots = [
            {
                channel: slot_by_parameter[name]
                for channel, name in layer.items()
            }
            for layer in display["layers"]
        ]
        width = int(display["width"])
        height = int(display["height"])
        title = _c_string(display["title"])
        default_frames = "0"
        display_loop_condition = "turing_display_running && (frames == 0 || frame < frames)"
        display_source = r'''
#if !defined(_WIN32)
#error "The dependency-free native display adapter currently requires Win32"
#else
static HWND turing_display_window = NULL;
static int turing_display_running = 1;
static uint32_t *turing_display_pixels = NULL;
static int turing_display_width = 0;
static int turing_display_height = 0;

static LRESULT CALLBACK turing_display_proc(
    HWND window, UINT message, WPARAM wparam, LPARAM lparam
) {
    (void)wparam;
    (void)lparam;
    if (message == WM_CLOSE) {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY) {
        turing_display_running = 0;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(window, message, wparam, lparam);
}

static void turing_display_set_utf8_title(HWND window, const char *title) {
    int length = MultiByteToWideChar(CP_UTF8, 0, title, -1, NULL, 0);
    wchar_t *wide;
    if (length < 1) return;
    wide = (wchar_t *)calloc((size_t)length, sizeof(wchar_t));
    if (wide == NULL) return;
    if (MultiByteToWideChar(CP_UTF8, 0, title, -1, wide, length)) {
        SetWindowTextW(window, wide);
    }
    free(wide);
}

static int turing_display_open(int width, int height, const char *title) {
    WNDCLASSA window_class = {0};
    RECT rectangle = {0, 0, width, height};
    HINSTANCE instance = GetModuleHandleA(NULL);
    window_class.lpfnWndProc = turing_display_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = "TuringNativeDisplay";
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    if (!RegisterClassA(&window_class) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        return 0;
    }
    AdjustWindowRect(&rectangle, WS_OVERLAPPEDWINDOW, FALSE);
    turing_display_window = CreateWindowExA(
        0, window_class.lpszClassName, "", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rectangle.right - rectangle.left, rectangle.bottom - rectangle.top,
        NULL, NULL, instance, NULL
    );
    if (turing_display_window == NULL) return 0;
    turing_display_set_utf8_title(turing_display_window, title);
    turing_display_pixels = (uint32_t *)calloc(
        (size_t)width * (size_t)height, sizeof(uint32_t)
    );
    if (turing_display_pixels == NULL) return 0;
    turing_display_width = width;
    turing_display_height = height;
    return 1;
}

static void turing_display_messages(void) {
    MSG message;
    while (PeekMessageA(&message, NULL, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) turing_display_running = 0;
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
}

static unsigned int turing_display_channel(double value) {
    if (value <= 0.0) return 0;
    if (value >= 255.0) return 255;
    return (unsigned int)(value + 0.5);
}

static void turing_display_present_layered(
    int layer_count,
    const double *const *red_layers,
    const double *const *green_layers,
    const double *const *blue_layers,
    const double *const *alpha_layers
) {
    BITMAPINFO information = {0};
    RECT client;
    HDC device;
    size_t index;
    int layer;
    size_t count = (size_t)turing_display_width * (size_t)turing_display_height;
    // Back-to-front "over" compositing. A layer with no alpha binding
    // (alpha_layers[layer] == NULL) is opaque -- the single-layer
    // rgb_f64_planar/rgba_f64_planar cases are this loop with
    // layer_count == 1, not a separate code path.
    for (index = 0; index < count; ++index) {
        double out_r = 0.0, out_g = 0.0, out_b = 0.0;
        for (layer = 0; layer < layer_count; ++layer) {
            double a = alpha_layers[layer]
                ? alpha_layers[layer][index] / 255.0 : 1.0;
            if (a <= 0.0) continue;
            if (a > 1.0) a = 1.0;
            out_r = red_layers[layer][index]   * a + out_r * (1.0 - a);
            out_g = green_layers[layer][index] * a + out_g * (1.0 - a);
            out_b = blue_layers[layer][index]  * a + out_b * (1.0 - a);
        }
        turing_display_pixels[index] =
            turing_display_channel(out_b)
            | (turing_display_channel(out_g) << 8)
            | (turing_display_channel(out_r) << 16);
    }
    information.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    information.bmiHeader.biWidth = turing_display_width;
    information.bmiHeader.biHeight = -turing_display_height;
    information.bmiHeader.biPlanes = 1;
    information.bmiHeader.biBitCount = 32;
    information.bmiHeader.biCompression = BI_RGB;
    GetClientRect(turing_display_window, &client);
    device = GetDC(turing_display_window);
    StretchDIBits(
        device, 0, 0, client.right, client.bottom,
        0, 0, turing_display_width, turing_display_height,
        turing_display_pixels, &information, DIB_RGB_COLORS, SRCCOPY
    );
    ReleaseDC(turing_display_window, device);
}

static void turing_display_close(void) {
    free(turing_display_pixels);
    turing_display_pixels = NULL;
    if (turing_display_window != NULL && IsWindow(turing_display_window)) {
        DestroyWindow(turing_display_window);
    }
    turing_display_window = NULL;
}
#endif
'''
        display_open_lines = [
            f"    if (!turing_display_open({width}, {height}, {title})) return 7;",
        ]
        display_message_lines = [
            "        turing_display_messages();",
            "        if (!turing_display_running) break;",
        ]
        layer_count = len(layer_slots)

        def _layer_array(channel: str, c_name: str) -> list[str]:
            entries = ", ".join(
                f"(const double *)slots[{slots[channel]}]" if channel in slots
                else "NULL"
                for slots in layer_slots
            )
            return [
                f"        const double *{c_name}[{layer_count}] = {{ {entries} }};"
            ]

        display_present_lines = [
            "        {",
            *_layer_array("red", "turing_display_red_layers"),
            *_layer_array("green", "turing_display_green_layers"),
            *_layer_array("blue", "turing_display_blue_layers"),
            *_layer_array("alpha", "turing_display_alpha_layers"),
            "        turing_display_present_layered(",
            f"            {layer_count},",
            "            turing_display_red_layers,",
            "            turing_display_green_layers,",
            "            turing_display_blue_layers,",
            "            turing_display_alpha_layers);",
            "        }",
            "        turing_display_messages();",
        ]
        if int(display["frame_delay_ms"]) > 0:
            display_present_lines.append(
                f"        Sleep({int(display['frame_delay_ms'])});"
            )
        display_close_lines = ["    turing_display_close();"]

    source = "\n".join((
        # The macro has to precede the base source: the launch hook inside
        # it is guarded by `#if TURING_TRACE`, so defining it afterwards
        # would compile the ring in while leaving the hook that feeds it
        # compiled out -- a digest that is present and permanently empty.
        f"#define TURING_TRACE {1 if trace else 0}",
        _C_SOURCE,
        # `_C_TRACE_SOURCE` defines the ring types itself. The companion
        # `_C_TRACE_DECLARATIONS` exists for cffi's cdef, where the types
        # must be announced without bodies; pasting it into a real
        # translation unit redefines every struct and forward-references
        # TuringLaunchProfile before the base source declares it.
        _C_TRACE_SOURCE if trace else "",
        "",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#if !defined(_WIN32)",
        "#include <sys/types.h>",
        "#endif",
        "",
        r'''#if defined(_WIN32)
/* GCC 16's MinGW static libgfortran uses the POSIX strndup entry point, while
 * an older Windows CRT does not export it. Keep the standalone runtime archive
 * resolvable without introducing another redistributable DLL. The definition is
 * WEAK: a newer mingw CRT (or the toolchain's own CRT shim) that supplies a
 * strong strndup overrides this one, so linking both never multiply-defines the
 * symbol; when nothing else provides it, this fills the reference. */
__attribute__((weak)) char *strndup(const char *source, size_t maximum) {
    size_t length = 0;
    char *copy;
    while (length < maximum && source[length] != '\0') ++length;
    copy = (char *)malloc(length + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, source, length);
    copy[length] = '\0';
    return copy;
}
#endif
''',
        r'''static FILE *turing_open_artifact(
    const char *executable, const char *filename, const char *mode
) {
    char path[4096];
    const char *slash = strrchr(executable, '/');
    const char *backslash = strrchr(executable, '\\');
    const char *separator = slash;
    size_t directory_length;
    if (backslash != NULL && (separator == NULL || backslash > separator)) {
        separator = backslash;
    }
    if (separator == NULL) return fopen(filename, mode);
    directory_length = (size_t)(separator - executable + 1);
    if (directory_length + strlen(filename) + 1 > sizeof(path)) return NULL;
    memcpy(path, executable, directory_length);
    strcpy(path + directory_length, filename);
    return fopen(path, mode);
}
''',

        *(r'''static const char *turing_argument_value(int argc, char **argv, const char *flag) {
    int index;
    for (index = 2; index + 1 < argc; ++index) {
        if (strcmp(argv[index], flag) == 0) return argv[index + 1];
    }
    return NULL;
}

static int turing_read_file(
    const char *path, uint8_t *destination, size_t capacity, size_t *length
) {
    FILE *file = fopen(path, "rb");
    long size;
    if (file == NULL) { perror(path); return 0; }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file); return 0;
    }
    if ((unsigned long long)size > (unsigned long long)capacity) {
        fprintf(stderr, "input file exceeds compiled port capacity: %s\n", path);
        fclose(file); return 0;
    }
    if (fread(destination, 1, (size_t)size, file) != (size_t)size) {
        fclose(file); return 0;
    }
    fclose(file);
    *length = (size_t)size;
    return 1;
}
''' if file_ports else "",),
        _C_FILE_BROKER_SOURCE if _requires_file_broker(module) else "",
        *shell_boundary_globals,
        display_source,
        f"extern void {entry.symbol}({', '.join(prototype_arguments)});",
        "",
        "static int turing_fortran_compute(void *context, unsigned long long *device_ns) {",
        "    void **slots = (void **)context;",
        "    *device_ns = 0;",
        f"    {entry.symbol}({', '.join(call_arguments)});",
        "    return 1;",
        "}",
        "",
        "int main(int argc, char **argv) {",
        f"    int frames = argc > 1 ? atoi(argv[1]) : {default_frames};",
        "    int stream_frames = 0;",
        f"    void *slots[{len(values)}] = {{0}};",
        "    TuringLaunchProfile profile = {0};",
        "    TuringLaunchStats stats = {0};",
        *((
            "    TuringTraceRing trace_ring = {0};",
            f"    TuringTraceRecord trace_storage[{trace_capacity}];",
            "    TuringTraceSite trace_site = {0};",
        ) if trace else ()),
        *shell_boundary_declarations,
        "    int frame;",
        "    { int argument_index;",
        "      for (argument_index = 2; argument_index < argc; ++argument_index)",
        "        if (strcmp(argv[argument_index], \"--stream-frames\") == 0) stream_frames = 1; }",
        f"    FILE *state = turing_open_artifact(argv[0], {_c_string(initial_state_filename)}, \"rb\");",
        "    if (frames < 0) return 2;",
        "    if (!state) { perror(\"initial state\"); return 2; }",
        *allocation_lines,
        *file_load_lines,
        *input_read_lines,
        "    fclose(state);",
        *display_open_lines,
        "    turing_launch_stats_reset(&stats);",
        *((
            f"    turing_trace_ring_reset(&trace_ring, trace_storage, {trace_capacity});",
            "    trace_site.ring = &trace_ring;",
            "    trace_site.region = 0;",
        ) if trace else ()),
        f"    for (frame = 0; {display_loop_condition}; ++frame) {{",
        *display_message_lines,
        "        if (turing_profiled_launch_ex(turing_fortran_compute, slots,",
        (
            "                &profile, &stats, turing_trace_logger_address(),"
            " &trace_site, 3) != 1) return 5;"
            if trace else
            "                &profile, &stats, NULL, NULL, 3) != 1) return 5;"
        ),
        *shell_boundary_actions,
        *display_present_lines,
        *feedback_lines,
        "        if (stream_frames) {",
        "            printf(\"{\\\"event\\\":\\\"frame\\\",\\\"frame\\\":%d,\\\"outputs\\\":{\", frame + 1);",
        *frame_output_lines,
        "            printf(\"}}\\n\");",
        "            fflush(stdout);",
        "        }",
        "    }",
        *display_close_lines,
        *feedback_finalize_lines,
        *((
            "    { unsigned long long available ="
            " turing_trace_available(&trace_ring);",
            "      unsigned long long lost = turing_trace_lost(&trace_ring);",
            "      unsigned long long index;",
            "      fprintf(stderr,"
            " \"{\\\"trace\\\":{\\\"records\\\":%llu,"
            "\\\"lost\\\":%llu,\\\"launches\\\":[\", available, lost);",
            "      for (index = 0; index < available; ++index) {",
            "        const TuringTraceRecord *record ="
            " &trace_ring.records[index % trace_ring.capacity];",
            "        fprintf(stderr, \"%s{\\\"seq\\\":%llu,"
            "\\\"shell_ns\\\":%llu,\\\"device_ns\\\":%llu,"
            "\\\"region\\\":%d,\\\"status\\\":%d}\","
            " index ? \",\" : \"\", record->sequence, record->shell_ns,"
            " record->device_ns, record->region, record->status);",
            "      }",
            "      fprintf(stderr, \"]}}\"); fputc(10, stderr); }",
        ) if trace else ()),
        "    printf(\"{\\\"status\\\":%d,\\\"frames\\\":%d,\\\"shell_ns_total\\\":%llu,\\\"outputs\\\":{\",",
        "           profile.status, frame, stats.shell_ns_total);",
        *output_lines,
        "    printf(\"}}\\n\");",
        f"    {{ FILE *outputs_file = turing_open_artifact(argv[0], {_c_string(final_outputs_filename)}, \"wb\");",
        "      if (!outputs_file) { perror(\"final outputs\"); return 6; }",
        *output_write_lines,
        "      fclose(outputs_file); }",
        *(('    turing_shell_file_close_all();',) if _requires_file_broker(module) else ()),
        f"    for (frame = 0; frame < {len(values)}; ++frame) free(slots[frame]);",
        "    return 0;",
        "}",
        "",
    ))
    return source


def emit_fortran_packed_library_source(entry: Any) -> tuple[str, str]:
    """Emit a bounded-arity C entry over an arbitrary Fortran frame.

    Every published ABI parameter owns one pointer-array slot. Reference
    parameters use the slot directly; value parameters are dereferenced using
    their published C type. This preserves the complete typed ABI while
    avoiding host FFI and platform call surfaces with fixed argument-count
    limits.
    """

    symbol = f"{entry.symbol}__packed"
    prototype_arguments = []
    call_arguments = []
    for index, parameter in enumerate(entry.parameters):
        c_type = str(parameter.c_type)
        pointer = parameter.passing == "reference"
        prototype_arguments.append(c_type + (" *" if pointer else ""))
        call_arguments.append(
            f"({c_type} *)arguments[{index}]" if pointer
            else f"*(({c_type} *)arguments[{index}])"
        )
    prototype = ", ".join(prototype_arguments) or "void"
    call = ", ".join(call_arguments)
    source = "\n".join((
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#if defined(_WIN32)",
        "#  define TURING_EXPORT __declspec(dllexport)",
        "#else",
        "#  define TURING_EXPORT __attribute__((visibility(\"default\")))",
        "#endif",
        f"extern void {entry.symbol}({prototype});",
        "",
        f"TURING_EXPORT int {symbol}(void **arguments, size_t argument_count) {{",
        f"    if (argument_count != {len(entry.parameters)}) return 0;",
        "    if (argument_count != 0 && arguments == NULL) return 0;",
        f"    {entry.symbol}({call});",
        "    return 1;",
        "}",
        "",
    ))
    return source, symbol


def compile_fortran_module_c_shell(
    module: Any,
    inputs: Mapping[str, Any],
    directory: str | Path,
    *,
    entrypoint: str | None = None,
    state_feedback: Mapping[str, str] | None = None,
    extent_overrides: Mapping[str, int] | None = None,
    name: str = "turing_fortran_c_shell",
    standalone: bool = True,
    library: bool = False,
    trace: bool = False,
) -> FortranCShellExecutable:
    """Compile generated Fortran plus the generic profiled C main.

    ``library=True`` instead builds a SHARED LIBRARY (.dll/.so) from just the
    Fortran module -- the compiled section exported for other programs to link
    against, "recognize without lowering". It skips the C-shell main and all of
    the runtime input/state machinery (a DLL of a section has no run harness and
    no initial state), so a parameterful section compiles without feeds.
    """

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError("no Fortran compiler found")
    compiler = str(Path(compiler).resolve())
    gcc = str(Path(compiler).with_name("gcc.exe" if os.name == "nt" else "gcc"))
    if not Path(gcc).is_file():
        raise FortranEmissionError(f"C compiler beside gfortran is missing: {gcc}")
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry = _entrypoint(module, entrypoint)
    # A shared library exports the emitted ABI directly: dynamic extents are
    # ordinary runtime arguments supplied by its eventual caller. Requiring
    # concrete values here applies the standalone shell's allocation concern
    # to a product that has no shell and makes otherwise valid shape-dynamic
    # libraries impossible to build. Standalone products still need every
    # extent resolved because their generated C harness owns the allocations.
    extents = {} if library else _extent_values(entry, extent_overrides)
    values = tuple(item for item in entry.parameters if item.role != "extent")
    input_parameters = tuple(
        item for item in values if item.role in {"input", "inout"}
    )
    file_ports = () if library else _system_file_configurations(module, entry)
    system_parameters = {
        parameter.name
        for port in file_ports
        for parameter in (port["data"], port["length"])
    }
    state_bytes = bytearray()
    if not library:
        missing = {
            _source_name(parameter)
            for parameter in input_parameters
            if parameter.name not in system_parameters
            if _source_name(parameter) not in inputs
        }
        if missing:
            raise ValueError(
                "missing C-shell inputs: " + ", ".join(sorted(missing))
            )

        for parameter in input_parameters:
            if parameter.name in system_parameters:
                continue
            source_name = _source_name(parameter)
            dtype = _NUMPY_DTYPES.get(str(parameter.dtype).casefold())
            if dtype is None:
                raise ValueError(f"unsupported C-shell dtype {parameter.dtype!r}")
            value = np.asarray(inputs[source_name], dtype=dtype)
            expected = _element_count(parameter, extents)
            if value.size != expected:
                raise ValueError(
                    f"input {source_name!r} has {value.size} elements; "
                    f"compiled ABI requires {expected}"
                )
            state_bytes.extend(np.ascontiguousarray(value).tobytes())

    fortran_path = output / f"{name}.f90"
    c_path = output / f"{name}.c"
    api_path = output / f"{name}.api.yaml"
    state_path = output / "initial-state.bin"
    final_outputs_path = output / "final-outputs.bin"
    fortran_path.write_text(module.source, encoding="utf-8")
    packed_symbol = None
    if library:
        c_source, packed_symbol = emit_fortran_packed_library_source(entry)
    else:
        c_source = emit_fortran_c_shell_source(
            module,
            trace=trace,
            entrypoint=entry.name,
            state_feedback=state_feedback,
            extent_overrides=extents,
            initial_state_filename=state_path.name,
            final_outputs_filename=final_outputs_path.name,
        )
    c_path.write_text(c_source, encoding="utf-8")
    # Pack runtime dependencies into the contract at compile time. The
    # producer knows them here -- the compiler's own bin directory is where a
    # gfortran-built library's support DLLs live -- and a consumer must never
    # rediscover them by loader archaeology (nodus boundary error register,
    # E15: a missing libgfortran presented as a silent LoadLibrary failure
    # that mimicked an ABI bug).
    # The compiled product's public contract must name the entry point that
    # was actually selected above.  A repository module can contain linked
    # dependency controls before its authored root, so preserving the
    # emitter's incidental first-control default here can advertise a helper
    # (for example ``uleb``) as the callable product instead of the requested
    # root.  Keep every exported entry, but make the shell/library selection
    # authoritative for this artifact.
    api = replace(module.api, entry=entry.name)
    runtime_dependencies = []
    if os.name == "nt":
        toolchain_bin = Path(compiler).parent
        for dll_name in (
            "libgfortran-5.dll",
            "libquadmath-0.dll",
            "libgcc_s_seh-1.dll",
            "libwinpthread-1.dll",
        ):
            candidate = toolchain_bin / dll_name
            if candidate.exists():
                runtime_dependencies.append(
                    {"name": dll_name, "path": candidate.as_posix()}
                )
    if runtime_dependencies:
        metadata = dict(api.metadata)
        metadata["runtime_dependencies"] = runtime_dependencies
        api = replace(api, metadata=metadata)
    if packed_symbol is not None:
        metadata = dict(api.metadata)
        packed_entrypoints = dict(metadata.get("packed_entrypoints") or {})
        packed_entrypoints[str(entry.name)] = {
            "schema": "turing.packed-pointer-array.v1",
            "symbol": str(packed_symbol),
            "parameter_count": len(entry.parameters),
        }
        metadata["packed_entrypoints"] = packed_entrypoints
        api = replace(api, metadata=metadata)
    api.write(api_path)
    state_path.write_bytes(bytes(state_bytes))

    if library:
        suffix = ".dll" if os.name == "nt" else ".so"
    else:
        suffix = ".exe" if os.name == "nt" else ""
    executable = output / f"{name}{suffix}"
    fortran_object = output / f"{name}.fortran.o"
    c_object = output / f"{name}.shell.o"
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    )
    fortran_flags = aggressive_fortran_flags(
        compiler,
        # Set by emit_module when the SSA carried precision sections; the
        # C shell itself only marshals, so its flags stay unconditional.
        precision_sections=bool(getattr(module, "precision_sections", False)),
    )
    c_flags = aggressive_c_flags(compiler)
    try:
        link_flags = (
            standalone_fortran_link_flags(compiler)
            if standalone else ("-flto",)
        )
    except ValueError as error:
        raise FortranEmissionError(str(error)) from error
    if library:
        # A shared library of the section: compile the Fortran module and link
        # it with the packed C ABI adapter. There is no main or runtime input.
        commands = (
            [compiler, *fortran_flags, "-c", str(fortran_path), "-o", str(fortran_object)],
            [gcc, *c_flags, "-std=c11", "-c", str(c_path), "-o", str(c_object)],
            [
                compiler, "-shared", "-o", str(executable),
                str(fortran_object), str(c_object),
                *standalone_runtime_shim_sources(compiler, output, standalone),
                # MinGW does not export a Fortran ``bind(C)`` procedure from
                # a DLL merely because the procedure has a stable C symbol.
                # The API contract publishes that direct symbol and the
                # native verifiers call it, while the C pointer-array adapter
                # is an additional generic entry rather than a replacement.
                # Export the bind(C) surface on Windows so the DLL and its
                # descriptor cannot disagree about what is callable.
                *(("-Wl,--export-all-symbols",) if os.name == "nt" else ()),
            ],
        )
    else:
        commands = (
            [compiler, *fortran_flags, "-c", str(fortran_path), "-o", str(fortran_object)],
            [gcc, *c_flags, "-std=c11", "-c", str(c_path), "-o", str(c_object)],
            [
                compiler, str(c_object), str(fortran_object),
                *standalone_runtime_shim_sources(compiler, output, standalone),
                "-o", str(executable),
                *link_flags,
                *(
                    ["-mwindows", "-lgdi32", "-luser32"]
                    if _display_configuration(module, entry) else []
                ),
            ],
        )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=str(output),
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise FortranEmissionError(
                "native Fortran/C-shell compilation failed:\n"
                + (completed.stderr or completed.stdout)
            )
    return FortranCShellExecutable(
        directory=output,
        executable_path=executable,
        fortran_source_path=fortran_path,
        c_source_path=c_path,
        api_path=api_path,
        initial_state_path=state_path,
        final_outputs_path=final_outputs_path,
        entrypoint=entry.name,
    )


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


def _field_slot_ops(
    graph_obj: Any,
    *,
    retained_storage_identities: frozenset[str] = frozenset(),
    keyed_table_fields: frozenset[str] = frozenset(),
):
    """Recover a method's instance-field accesses as slot loads and stores.

    A class's field layout is declared once (``class_table[class]['fields']``),
    giving each field a fixed slot. ``self`` is that field arena. This reads the
    process graph's field-op nodes and returns, for one method:

    * ``self_value_id`` -- the value id of the ``self`` arena, or ``None``;
    * ``field_ops`` -- ``(kind, value_id, slot)`` for every field access in the
      graph's own schedule order, ``kind`` being ``"read"`` (a ``GetAttr``, whose
      ``value_id`` is the result the method already consumes) or ``"write"`` (a
      ``setattr``, whose ``value_id`` is the stored source). Keeping reads and
      writes in one ordered list preserves their interleaving, so a store and a
      later read of one slot stay in the order the source wrote them;
    * ``field_count`` -- the arena length, so ``self`` is a sized array.
    """

    class_table = dict(graph_obj.graph.get("class_table") or {})
    owner = graph_obj.graph.get("method_owner")
    record = (
        class_table.get(owner)
        if owner in class_table
        else (next(iter(class_table.values())) if len(class_table) == 1 else None)
    )
    fields = tuple((record or {}).get("fields") or ())
    slot_of = {name: index for index, name in enumerate(fields)}

    identity = dict(graph_obj.graph.get("identity_table") or {})
    self_history = identity.get("self") or ()
    self_value_id = int(self_history[-1]) if self_history else None

    # Order field ops by SOURCE order (node id), not the data-dependency
    # schedule. Memory ordering between a write and a later read of the same
    # field is a real dependency the AST wrote but the graph does not carry as a
    # data edge, so a topological sort is free to float the read ahead of the
    # write. Nodes are created in source order, so their ids preserve the order
    # the programmer wrote -- which is the order the memory operations must run.
    field_ops: list[tuple[str, int, int]] = []
    const_sources: dict[int, Any] = {}
    sequence_initializations: list[tuple[int, str, int]] = []
    sequence_declarations: list[tuple[int, str, int, bool]] = []
    sequence_memberships: list[tuple[int, int, int, bool]] = []
    table_lookups: list[tuple[int, int | tuple[int, ...], int]] = []
    table_stores: list[tuple[int, int | tuple[int, ...], int, int]] = []
    table_deletions: list[
        tuple[int, int | tuple[int, ...], int | None, str]
    ] = []
    tombstone_sequence_ids: set[int] = set()
    retained_sequence_ids: set[int] = set()
    nested_sequence_ids: set[int] = set()
    nested_record_fields: dict[int, tuple[str, int]] = {}
    field_aliases = dict(graph_obj.graph.get("class_field_aliases") or {})
    field_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_aggregate_kinds") or {}
    )
    # A dataclass annotation is also an exact physical record contract.  Keep
    # its span kind available to the class-field coordinator so a read such as
    # ``self.locals`` becomes the declared sequence arena, never a scalar load
    # from the receiver slot vector.  Explicit frontend aggregate evidence
    # wins when both are present.
    declared_self_record = dict(
        (graph_obj.graph.get("parameter_record_abi") or {}).get("self") or {}
    )
    declared_span_kinds = {
        str(field_name): str(field.get("aggregate_kind"))
        for field_name, field in dict(
            declared_self_record.get("fields") or {}
        ).items()
        if str(field.get("storage") or "") == "span"
        and field.get("aggregate_kind") is not None
    }
    field_aggregate_kinds = {
        **declared_span_kinds,
        **field_aggregate_kinds,
    }
    field_value_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_value_aggregate_kinds") or {}
    )

    def node_operation(data: Mapping[str, Any]) -> str:
        """Canonical ProcessGraph operation spelling for field analysis.

        Authored graph nodes carry the semantic class in ``type`` and often a
        lower-case executable spelling in ``op``.  Preferring ``op`` without
        normalizing it made ``type=GetAttr, op=getattr`` invisible to the
        whole-object resolver, exactly where object fields should become
        native record storage.
        """

        return str(data.get("op") or data.get("type") or "").casefold()

    def canonical_field(name: str) -> str:
        seen: set[str] = set()
        current = str(name)
        while current in field_aliases and current not in seen:
            seen.add(current)
            current = str(field_aliases[current])
        return current

    # A method may mention only an alias (NetworkX ``_succ``) while its record
    # storage is authored under the canonical field (``_adj``). Correlate all
    # such GetAttr occurrences to one resident sequence ID; this is storage
    # aliasing, not value copying or object dispatch.
    field_sequence_ids: dict[str, int] = {}
    lexical_sequence_ids: dict[tuple[str, str], int] = {}
    inferred_nested_table_bases: dict[int, int] = {}
    aggregate_reads: list[tuple[str, str, int]] = []

    def fixed_row_width(annotation: Any) -> int | None:
        """Read a fixed tuple row width from an authored container annotation."""

        if not isinstance(annotation, str) or not annotation.strip():
            return None
        try:
            outer = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(outer, ast.Subscript):
            return None
        element = outer.slice
        if not isinstance(element, ast.Subscript):
            return None
        tuple_name = element.value
        if not (
            isinstance(tuple_name, ast.Name) and tuple_name.id == "tuple"
            or isinstance(tuple_name, ast.Attribute)
            and tuple_name.attr in {"tuple", "Tuple"}
        ):
            return None
        columns = (
            tuple(element.slice.elts)
            if isinstance(element.slice, ast.Tuple)
            else (element.slice,)
        )
        if any(isinstance(column, ast.Constant) and column.value is Ellipsis
               for column in columns):
            return None
        return len(columns) if len(columns) > 1 else None

    annotated_row_widths: dict[int, int] = {}
    for binding_name, annotation in dict(
        graph_obj.graph.get("type_annotations") or {}
    ).items():
        width = fixed_row_width(annotation)
        if width is None:
            continue
        for value_id in identity.get(str(binding_name), ()):
            annotated_row_widths[int(value_id)] = int(width)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "getattr":
            continue
        attribute = (data.get("attributes") or {}).get("attribute")
        if field_aggregate_kinds.get(str(attribute)) not in {
            "list", "set", "dict", "tuple", "bytes", "bytearray"
        }:
            continue
        aggregate_reads.append((
            str(attribute), canonical_field(str(attribute)),
            int(data.get("value_id", node_id)),
        ))
    for attribute, canonical, result_id in aggregate_reads:
        if attribute == canonical:
            field_sequence_ids[canonical] = result_id
    for _attribute, canonical, result_id in aggregate_reads:
        field_sequence_ids.setdefault(canonical, result_id)
    # A contract-declared keyed field is a lookup table too, but it is a
    # program-ABI record field, not a class-field aggregate, so it must not
    # enter ``field_sequence_ids`` (that registry engages the object-field
    # arena machinery).  Same identity convention: the GetAttr's own value id
    # names the table, one canonical id per field.
    keyed_field_sequence_ids: dict[str, int] = {}
    if keyed_table_fields:
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            data = graph_obj.nodes[node_id]
            if node_operation(data) != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in keyed_table_fields:
                continue
            keyed_field_sequence_ids.setdefault(
                attribute, int(data.get("value_id", node_id))
            )
    # Runtime aggregates captured from a lexical/module binding are resident
    # storage exactly like aggregate fields, but they have no record slot.
    # Correlate all normalized occurrences by their authored binding identity;
    # never inspect or serialize the bound Python collection's contents.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        if (
            aggregate_kind not in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }
            or binding_name is None
            or binding_kind not in {"parameter", "closure", "external"}
        ):
            continue
        identity_key = (str(binding_kind), str(binding_name))
        sequence_id = lexical_sequence_ids.setdefault(
            identity_key, int(data.get("value_id", node_id))
        )
        sequence_declarations.append((
            sequence_id,
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
            2 if aggregate_kind == "dict" else 1,
            aggregate_kind not in {"tuple", "bytes"},
        ))
        storage_identity = f"{binding_kind}.{binding_name}"
        if storage_identity in retained_storage_identities:
            retained_sequence_ids.add(sequence_id)

    # An authored homogeneous collection annotation is sufficient physical
    # ABI evidence even when the Input node itself has no Python aggregate
    # value attached. This matters for detached compilation: Sequence[str]
    # is a token arena, not an untyped scalar that happens to be indexed.
    declared_sequence_ids = {
        int(sequence_id)
        for sequence_id, _policy, _columns, _writable
        in sequence_declarations
    }
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        contract = _authored_sequence_annotation_contract(annotation)
        history = tuple(map(int, identity.get(str(parameter_name), ())))
        if contract is None or not history:
            continue
        sequence_id = int(history[0])
        if sequence_id in declared_sequence_ids:
            continue
        policy, column_count, writable, _dtypes = contract
        lexical_sequence_ids.setdefault(
            ("parameter", str(parameter_name)), sequence_id
        )
        sequence_declarations.append((
            sequence_id, policy, int(column_count), bool(writable),
        ))
        declared_sequence_ids.add(sequence_id)
        retained_sequence_ids.add(sequence_id)

    # Source-derived ``Sequence[Record]`` metadata survives graph extraction
    # independently of any one Input wrapper. Declare its caller-owned
    # columnar arena explicitly so later field projections share one physical
    # descriptor instead of minting anonymous fallback columns.
    for parameter_name, record in dict(
        graph_obj.graph.get("parameter_sequence_record_abi") or {}
    ).items():
        history = tuple(map(int, identity.get(str(parameter_name), ())))
        fields = tuple(dict(record.get("fields") or {}))
        if not history or not fields:
            continue
        sequence_id = int(history[0])
        lexical_sequence_ids.setdefault(
            ("parameter", str(parameter_name)), sequence_id
        )
        sequence_declarations.append((
            sequence_id,
            "duplicates",
            sum(
                2 if bool(receipt.get("optional")) else 1
                for receipt in dict(record.get("fields") or {}).values()
            ),
            bool(record.get("mutable", False)),
        ))

    # An unannotated parameter can still state a complete aggregate contract
    # through authored operations.  ``key in p`` plus ``p[key]`` whose result
    # is iterated proves a keyed table whose values are child sequences.  This
    # derives storage from graph structure; it does not inspect parameter
    # contents or infer from a method's spelling.
    iterated_value_ids = {
        int(graph_obj.nodes[parent].get("value_id", parent))
        for node_id in graph_obj.nodes()
        for data in (graph_obj.nodes[node_id],)
        if (data.get("op") or data.get("type")) == "For"
        for parent, role in (data.get("parents") or ())
        if str(role) == "iterable" and parent in graph_obj
    }
    inferred_by_identity: dict[tuple[str, str], list[int]] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexed":
            continue
        result_id = int(data.get("value_id", node_id))
        if result_id not in iterated_value_ids:
            continue
        base_nodes = tuple(
            int(parent) for parent, role in (data.get("parents") or ())
            if str(role) == "base" and parent in graph_obj
        )
        if len(base_nodes) != 1:
            continue
        base_data = graph_obj.nodes[base_nodes[0]]
        base_attributes = base_data.get("attributes") or {}
        if base_attributes.get("binding_kind") != "parameter":
            continue
        binding_name = base_attributes.get("binding_name")
        if binding_name is None:
            continue
        inferred_by_identity.setdefault(
            ("parameter", str(binding_name)), []
        ).append(int(base_data.get("value_id", base_nodes[0])))
    for identity_key, observed_ids in inferred_by_identity.items():
        history = tuple(map(int, identity.get(identity_key[1], ())))
        sequence_id = int(history[0] if history else observed_ids[0])
        lexical_sequence_ids.setdefault(identity_key, sequence_id)
        sequence_declarations.append((sequence_id, "unique", 2, False))
        nested_sequence_ids.add(sequence_id)
        for value_id in (*history, *observed_ids):
            inferred_nested_table_bases[int(value_id)] = sequence_id

    # Locally constructed aggregates are resident storage too.  Previously the
    # record extractor declared only parameters, captures and object fields;
    # a local ``out = []`` or ``charmap = bytearray(256)`` therefore reached a
    # planned region as a shapeless scalar unless append/add happened to create
    # a descriptor as a side effect.  The source graph already owns the exact
    # producer identity, storage policy and writability, so publish that same
    # identity in the method sequence table.  Tuple values remain record rows
    # and are handled by projected-row/record lowering rather than pretending
    # the heterogeneous record itself is one homogeneous arena.
    singleton_structural_scalar_ids = {
        int(leaf_ids[0])
        for _candidate_id, candidate in graph_obj.nodes(data=True)
        for leaf_ids in (tuple(map(
            int,
            (candidate.get("attributes") or {}).get(
                "aggregate_leaf_value_ids", ()
            ),
        )),)
        if len(leaf_ids) == 1
    }
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        if aggregate_kind not in {
            "list", "set", "dict", "bytes", "bytearray"
        }:
            continue
        if (
            aggregate_kind == "bytes"
            and attributes.get("producer_kind")
            == "aggregate_materialization"
        ):
            # ``bytes(local_sequence)`` is a result view over the local arena,
            # not a second independently allocated sequence.  Root authority
            # below maps it to its sole source only after that relation is
            # proven; declaring another arena here fabricates an ABI input.
            continue
        sequence_id = int(data.get("value_id", node_id))
        leaf_ids = tuple(map(int, attributes.get(
            "aggregate_leaf_value_ids", ()
        )))
        if (
            len(leaf_ids) == 1 and sequence_id == leaf_ids[0]
            or sequence_id in singleton_structural_scalar_ids
        ):
            # A singleton literal/wrapper may deliberately reuse its sole
            # scalar leaf as a reducible structural identity.  It is consumed
            # by concat lowering as ``append_scalar`` and owns no arena of its
            # own. Declaring that scalar id as a sequence turns ordinary
            # parameters such as ``section_id`` into rank-1 arrays.
            continue
        sequence_declarations.append((
            sequence_id,
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
            (
                (
                    int(attributes.get("mapping_key_column_count", 1)) + 1
                ) if aggregate_kind == "dict"
                else max(
                    int(attributes.get("sequence_column_count", 1)),
                    annotated_row_widths.get(sequence_id, 1),
                )
            ),
            bool(attributes.get(
                "sequence_writable", aggregate_kind != "bytes"
            )),
        ))
        mapping_items = attributes.get("compile_time_mapping_items")
        if aggregate_kind == "dict" and mapping_items is not None:
            from .string_table import string_token

            encoded_rows = tuple(
                (
                    string_token(key) if isinstance(key, str) else key,
                    string_token(item) if isinstance(item, str) else item,
                )
                for key, item in tuple(mapping_items)
            )
            sequence_initializations.append((
                sequence_id,
                f"literal_table={encoded_rows!r}",
                2,
            ))
        if attributes.get("mapping_value_aggregate_kind") in {
            "list", "set", "dict", "bytearray"
        }:
            nested_sequence_ids.add(sequence_id)
    def table_sequence(base_id: int) -> tuple[int | None, str | None]:
        if base_id not in graph_obj:
            return None, None
        base_data = graph_obj.nodes[base_id]
        inferred_sequence_id = inferred_nested_table_bases.get(
            int(base_data.get("value_id", base_id))
        )
        if inferred_sequence_id is not None:
            return inferred_sequence_id, "parameter.inferred_nested_table"
        attributes = base_data.get("attributes") or {}
        if attributes.get("compile_time_mapping_items") is not None:
            return (
                int(base_data.get("value_id", base_id)),
                f"constant.{attributes.get('binding_name', base_id)}",
            )
        field_name = attributes.get("attribute")
        if field_aggregate_kinds.get(str(field_name)) == "dict":
            canonical = canonical_field(str(field_name))
            return (
                field_sequence_ids[canonical],
                f"{owner}.{canonical}",
            )
        if str(field_name) in keyed_field_sequence_ids:
            return (
                keyed_field_sequence_ids[str(field_name)],
                f"keyed.{field_name}",
            )
        if attributes.get("aggregate_kind") != "dict":
            return None, None
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        identity_key = (str(binding_kind), str(binding_name))
        if binding_name is None or identity_key not in lexical_sequence_ids:
            return None, None
        return (
            lexical_sequence_ids[identity_key],
            f"{binding_kind}.{binding_name}",
        )

    def authored_index_values(data: Any) -> int | tuple[int, ...] | None:
        values = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "index" and parent in graph_obj
        )
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def deletes_first_live_key(data: Any) -> bool:
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Subscript):
            return False
        key = expression.slice
        return (
            isinstance(key, ast.Call)
            and isinstance(key.func, ast.Name)
            and key.func.id == "next"
            and len(key.args) == 1
            and isinstance(key.args[0], ast.Call)
            and isinstance(key.args[0].func, ast.Name)
            and key.args[0].func.id == "iter"
            and len(key.args[0].args) == 1
            and ast.dump(key.args[0].args[0], include_attributes=False)
            == ast.dump(expression.value, include_attributes=False)
        )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        node_type = node_operation(data)
        attribute = (data.get("attributes") or {}).get("attribute")
        canonical_attribute = (
            canonical_field(str(attribute)) if attribute is not None else None
        )
        if canonical_attribute is None or canonical_attribute not in slot_of:
            continue
        if node_type == "getattr":
            result_id = data.get("value_id", node_id)
            field_ops.append((
                "read", int(result_id), slot_of[canonical_attribute]
            ))
            aggregate_kind = field_aggregate_kinds.get(str(attribute))
            if aggregate_kind in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }:
                sequence_id = field_sequence_ids[canonical_field(str(attribute))]
                if int(result_id) == sequence_id:
                    sequence_declarations.append((
                        sequence_id,
                        "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                        2 if aggregate_kind == "dict" else 1,
                        aggregate_kind not in {"tuple", "bytes"},
                    ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(sequence_id)
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(sequence_id)
        elif node_type == "setattr":
            source_parent = next(
                (
                    parent
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                ),
                None,
            )
            if source_parent is None:
                continue
            source_data = graph_obj.nodes[source_parent]
            source_id = source_data.get("value_id", source_parent)
            field_ops.append((
                "write", int(source_id), slot_of[canonical_attribute]
            ))
            source_attributes = source_data.get("attributes") or {}
            if node_operation(source_data) == "staticreference":
                reference_identity = source_attributes.get(
                    "static_python_reference"
                )
                if reference_identity is not None:
                    from .string_table import string_token

                    identity = str(reference_identity)
                    const_sources[int(source_id)] = {
                        "ssa_reference_identity": identity,
                        "reference_kind": "static-python",
                        "reference_handle": string_token(
                            "\x00turing.reference.static-python\x00" + identity
                        ),
                        "host_resident": True,
                    }
            nested_class_identity = source_attributes.get("class_ref")
            if nested_class_identity is not None:
                nested_record_fields[slot_of[canonical_attribute]] = (
                    str(nested_class_identity), int(source_id)
                )
            aggregate_kind = (
                source_attributes.get("aggregate_kind")
                or field_aggregate_kinds.get(str(attribute))
            )
            if (
                aggregate_kind in {"list", "set", "dict", "bytearray"}
                and attribute not in field_aliases
            ):
                sequence_initializations.append((
                    int(source_id),
                    "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                    2 if aggregate_kind == "dict" else 1,
                ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(int(source_id))
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(int(source_id))
            # A constant field write (``self.x = None`` / ``5`` / ``"s"``) has
            # no producer in the control body, so carry the constant value; the
            # injection materialises it before the store (None becomes the
            # absence sentinel via the tokenizer).
            if node_operation(source_data) in {"const", "constant"}:
                attrs = source_data.get("attributes") or {}
                const_sources[int(source_id)] = attrs.get(
                    "value", source_data.get("constant")
                )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], (ast.In, ast.NotIn))
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        query_id = by_role.get("lhs")
        field_id = by_role.get("rhs")
        if query_id is None or field_id is None or field_id not in graph_obj:
            continue
        field_data = graph_obj.nodes[field_id]
        field_attributes = field_data.get("attributes") or {}
        field_name = field_attributes.get("attribute")
        sequence_id = None
        if field_aggregate_kinds.get(str(field_name)) in {"dict", "set"}:
            sequence_id = field_sequence_ids[canonical_field(str(field_name))]
        elif field_attributes.get("aggregate_kind") in {"dict", "set"}:
            identity_key = (
                str(field_attributes.get("binding_kind")),
                str(field_attributes.get("binding_name")),
            )
            sequence_id = lexical_sequence_ids.get(identity_key)
        elif int(field_data.get("value_id", field_id)) in (
            inferred_nested_table_bases
        ):
            sequence_id = inferred_nested_table_bases[
                int(field_data.get("value_id", field_id))
            ]
        if sequence_id is None:
            continue
        sequence_memberships.append((
            int(data.get("value_id", node_id)),
            int(graph_obj.nodes[query_id].get("value_id", query_id)),
            int(sequence_id),
            isinstance(expression.ops[0], ast.NotIn),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexed":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_lookups.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(sequence_id),
        ))
    # ``d.get(key, default)`` is the same lookup ``d[key]`` is -- the key's
    # token walked against the table -- differing only in what the absent
    # branch yields.  Recognising only ``indexed`` left ``get`` unclaimed, so
    # its result crossed every backend as a producerless argument.  The
    # authored default rides beside the lookup by result id.
    table_lookup_defaults: dict[int, Any] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "get":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("operand") or by_role.get("value")
        key_id = by_role.get("arg:0")
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        result_id = int(data.get("value_id", node_id))
        key_id = int(graph_obj.nodes[key_id].get("value_id", key_id))
        table_lookups.append((result_id, key_id, int(sequence_id)))
        default_node = by_role.get("arg:1")
        if default_node is not None and default_node in graph_obj:
            default_data = graph_obj.nodes[default_node]
            literal = default_data.get("constant")
            if literal is None:
                literal = (
                    default_data.get("attributes") or {}
                ).get("value")
            if isinstance(literal, (int, float)) and not isinstance(
                literal, bool
            ):
                table_lookup_defaults[result_id] = float(literal)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexedstore":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        value_id = by_role.get("value")
        if None in (base_id, key_id, value_id) or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_stores.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(graph_obj.nodes[value_id].get("value_id", value_id)),
            int(sequence_id),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type")).lower() != "delitem":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = (() if deletes_first_live_key(data)
                  else authored_index_values(data))
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_value_id, storage_identity = table_sequence(base_id)
        if storage_identity is None:
            storage_identity = f"nested-table-value:{base_id}"
        table_deletions.append((
            int(data.get("value_id", node_id)),
            key_id,
            sequence_value_id,
            storage_identity,
        ))
    # A retained mapping ``pop`` is lowered by the control mutation identity,
    # not by this field-op table.  It nevertheless changes the table's
    # physical ABI: lookup-and-delete requires the same live-flag/tombstone
    # storage as authored ``del table[key]``.  Survey that structural fact
    # separately so it cannot accidentally emit a second deletion.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "pop":
            continue
        receiver_nodes = tuple(
            int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) in {"operand", "value", "object", "base", "receiver"}
            and parent in graph_obj
        )
        if not receiver_nodes:
            func_nodes = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) == "func" and parent in graph_obj
            )
            if len(func_nodes) != 1:
                continue
            func_data = graph_obj.nodes[func_nodes[0]]
            func_attributes = func_data.get("attributes") or {}
            if (
                node_operation(func_data) != "getattr"
                or str(func_attributes.get("attribute") or "") != "pop"
            ):
                continue
            receiver_nodes = tuple(
                int(parent)
                for parent, role in (func_data.get("parents") or ())
                if str(role) in {"value", "object", "base", "receiver"}
                and parent in graph_obj
            )
        if len(receiver_nodes) != 1:
            continue
        receiver_data = graph_obj.nodes[receiver_nodes[0]]
        receiver_attributes = receiver_data.get("attributes") or {}
        if receiver_attributes.get("aggregate_kind") == "dict":
            tombstone_sequence_ids.add(int(
                receiver_data.get("value_id", receiver_nodes[0])
            ))
        sequence_id, _storage_identity = table_sequence(receiver_nodes[0])
        if sequence_id is not None:
            tombstone_sequence_ids.add(int(sequence_id))
    # Runtime sequence replication (``[x] * count``) is a fill of resident
    # caller storage, not numerical multiplication and not a Python literal.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        if attributes.get("producer_kind") != "sequence_replication":
            continue
        parents = tuple(data.get("parents") or ())
        sequence_parents = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and parent in graph_obj
            and (graph_obj.nodes[parent].get("attributes") or {}).get(
                "aggregate_kind"
            ) in {"list", "tuple"}
        )
        fill_ids = tuple(
            int(leaf_id)
            for parent in sequence_parents
            for leaf_id in (
                graph_obj.nodes[parent].get("attributes") or {}
            ).get("aggregate_leaf_value_ids", ())
        )
        count_ids = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and int(parent) not in sequence_parents
        )
        if len(fill_ids) != 1 or len(count_ids) != 1:
            continue
        fill_data = graph_obj.nodes.get(fill_ids[0], {})
        literal = (fill_data.get("attributes") or {}).get(
            "value", fill_data.get("constant")
        )
        if literal is not None and not isinstance(literal, (bool, int, float)):
            continue
        result_id = int(data.get("value_id", node_id))
        sequence_initializations.append((
            result_id,
            f"fill={literal!r};count={count_ids[0]}",
            1,
        ))
    # An unannotated local can still have a fixed-width row contract when the
    # authored mutation supplies a tuple record.  The mutation extractor has
    # already expanded that tuple into its exact leaf identities.  Reconcile
    # those leaves with the resident declaration before control lowering so a
    # list initially spelled ``[]`` is not frozen as a one-column sequence and
    # then rejected when ``append((left, right))`` reaches it.  Conflicting row
    # widths remain unresolved and are refused later; choosing one would invent
    # an ABI that the source does not state.
    declared_sequence_ids = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    resident_by_value = {
        int(sequence_id): int(sequence_id)
        for sequence_id in declared_sequence_ids
    }
    for history in identity.values():
        history_ids = tuple(map(int, history))
        residents = tuple(
            value_id for value_id in history_ids
            if value_id in declared_sequence_ids
        )
        if len(set(residents)) != 1:
            continue
        resident = residents[0]
        resident_by_value.update({
            value_id: resident for value_id in history_ids
        })
    observed_row_widths: dict[int, set[int]] = {}
    for mutation in _sequence_append_call_mutations(graph_obj):
        if mutation.argument_kind != "row":
            continue
        resident = resident_by_value.get(int(mutation.sequence_value_id))
        width = len(tuple(mutation.argument_value_ids))
        if resident is not None and width > 1:
            observed_row_widths.setdefault(int(resident), set()).add(width)
    inferred_row_widths = {
        sequence_id: next(iter(widths))
        for sequence_id, widths in observed_row_widths.items()
        if len(widths) == 1
    }
    sequence_declarations = [
        (
            sequence_id,
            policy,
            max(column_count, inferred_row_widths.get(sequence_id, 1)),
            writable,
        )
        for sequence_id, policy, column_count, writable
        in sequence_declarations
    ]

    key_width_by_sequence: dict[int, int] = {}
    for _effect_or_result, key_ids, sequence_id, *_rest in (
        *table_lookups, *table_stores, *table_deletions
    ):
        if sequence_id is None:
            continue
        key_width_by_sequence[int(sequence_id)] = max(
            key_width_by_sequence.get(int(sequence_id), 1),
            len(key_ids) if isinstance(key_ids, tuple) else 1,
        )
    sequence_declarations = [
        (
            sequence_id,
            policy,
            (
                max(
                    column_count,
                    key_width_by_sequence.get(sequence_id, 1) + 1,
                )
                if policy == "unique" and column_count > 1
                else column_count
            ),
            writable,
        )
        for sequence_id, policy, column_count, writable
        in sequence_declarations
    ]
    # A record-field dict is a table exactly as a local one is, but declare it
    # only where a table operation actually addresses it: a declaration
    # materializes anonymous descriptor storage into the frame, and doing that
    # in functions that merely ITERATE the mapping displaced their public-span
    # correlation for every unrelated rank-2 field.
    field_table_ids = {
        int(sequence_id)
        for sequence_id in (
            *field_sequence_ids.values(),
            *keyed_field_sequence_ids.values(),
        )
    }
    referenced_table_ids = {
        int(sequence_id)
        for _result, _query, sequence_id in table_lookups
    } | {
        int(sequence_id)
        for _effect, _key, _value, sequence_id in table_stores
    }
    declared_ids = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    sequence_declarations.extend(
        (int(sequence_id), "unique", 2, False)
        for sequence_id in sorted(field_table_ids & referenced_table_ids)
        if int(sequence_id) not in declared_ids
    )
    return (
        self_value_id,
        tuple(field_ops),
        const_sources,
        len(fields),
        fields,
        owner,
        tuple(dict.fromkeys(sequence_initializations)),
        tuple(
            (slot_of[alias], slot_of[target])
            for alias, target in field_aliases.items()
            if alias in slot_of and target in slot_of
        ),
        tuple(dict.fromkeys(sequence_declarations)),
        tuple(dict.fromkeys(sequence_memberships)),
        tuple(dict.fromkeys(table_lookups)),
        dict(table_lookup_defaults),
        tuple(dict.fromkeys(table_stores)),
        tuple(dict.fromkeys(table_deletions)),
        tuple(sorted(retained_sequence_ids)),
        tuple(sorted(nested_sequence_ids)),
        tuple(sorted(
            (slot, identity, value_id)
            for slot, (identity, value_id) in nested_record_fields.items()
        )),
        tuple(sorted(tombstone_sequence_ids)),
    )


def _sequence_augassign_ops(graph_obj: Any) -> tuple[tuple[int, int, int], ...]:
    """Return proven ``sequence += sequence`` storage mutations.

    The graph's identity history correlates each lexical spelling with its
    successive SSA versions.  Resolve the AugAssign destination back to the
    resident aggregate producer bearing that spelling, and do the same for the
    source.  No operation-name inference is involved: both endpoints must be
    graph nodes already carrying an aggregate storage contract.
    """

    sequence_kinds = {"list", "set", "dict", "bytes", "bytearray"}
    identity = dict(graph_obj.graph.get("identity_table") or {})
    resident_by_value: dict[int, int] = {}
    for _name, history in identity.items():
        residents = tuple(
            int(value_id)
            for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident

    # Conditional/loop joins preserve the authored storage identity when every
    # incoming value already denotes that same resident arena.  Carry that
    # alias through Phi/LoopResult nodes to later AugAssign occurrences; never
    # choose an arbitrary branch when the residents differ or are unresolved.
    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            if int(node_id) in resident_by_value:
                continue
            data = graph_obj.nodes[node_id]
            if str(data.get("type") or data.get("op")).lower() not in {
                "phi", "loopresult"
            }:
                continue
            incoming = {
                resident_by_value[int(parent)]
                for parent, role in (data.get("parents") or ())
                if str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) in resident_by_value
            }
            unresolved = any(
                str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) not in resident_by_value
                for parent, role in (data.get("parents") or ())
            )
            if len(incoming) == 1 and not unresolved:
                resident_by_value[int(node_id)] = incoming.pop()
                changed = True

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
            and isinstance(expression.target, ast.Name)
        ):
            continue
        parents = {
            str(role): int(parent) for parent, role in (data.get("parents") or ())
        }
        result_id = int(data.get("value_id", node_id))
        destination_id = resident_by_value.get(parents.get("lhs", -1))
        source_id = resident_by_value.get(parents.get("rhs", -1))
        if destination_id is None or source_id is None:
            continue
        operations.append((result_id, destination_id, source_id))
    return tuple(operations)


def _sequence_concat_ops(
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str] | None = None,
    structural_aliases: Mapping[int, tuple[int, str]] | None = None,
) -> tuple[
    tuple[tuple[int, int, int, str, int | None, int | None], ...],
    tuple[tuple[int, int], ...],
    dict[int, int],
]:
    """Recognize value-producing sequence ``+`` without numericizing it.

    Python's ``bytes + bytes`` and ``list + list`` allocate a new logical
    sequence; they are not elementwise arithmetic.  ProcessGraph deliberately
    keeps the authored ``Add`` spelling, so this identity proves sequence
    semantics from the operands' aggregate contracts (including pursued-call
    result contracts) and returns both concat operations and immutable
    materialization aliases such as ``bytes([value]) -> [value]``.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    kind_by_value: dict[int, str] = {}
    singleton_by_resident: dict[int, int] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        if attributes.get("producer_kind") == "aggregate_materialization":
            source_nodes = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("arg:") and parent in graph_obj
            )
            sources = tuple(
                int(graph_obj.nodes[parent].get("value_id", parent))
                for parent in source_nodes
            )
            if len(sources) == 1:
                resident_id = resident_by_value.get(
                    sources[0], sources[0]
                )
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                # ``bytes([opcode])`` is an immutable view of a singleton
                # authored aggregate.  The materializer may carry the leaf
                # ledger even when its transient list source does not; retain
                # that exact scalar identity on the physical resident so
                # concat lowering emits append_scalar and does not demand a
                # sequence descriptor for a one-byte temporary.
                source_expression = (
                    graph_obj.nodes[source_nodes[0]].get("expr_obj")
                    if len(source_nodes) == 1 else None
                )
                source_attributes = (
                    graph_obj.nodes[source_nodes[0]].get("attributes") or {}
                    if len(source_nodes) == 1 else {}
                )
                literal_singleton = (
                    isinstance(source_expression, (ast.List, ast.Tuple, ast.Set))
                    and len(source_expression.elts) == 1
                ) or (
                    source_attributes.get("aggregate_kind") in {
                        "list", "tuple", "set"
                    }
                    and len(tuple(source_attributes.get(
                        "aggregate_leaf_value_ids", ()
                    ))) == 1
                )
                if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                    print(
                        "DEBUGMATERIALIZE "
                        f"value={value_id} kind={kind!r} sources={sources!r} "
                        f"source_expr={ast.dump(source_expression, include_attributes=False) if isinstance(source_expression, ast.AST) else None!r} "
                        f"source_kind={source_attributes.get('aggregate_kind')!r} "
                        f"source_type={graph_obj.nodes[source_nodes[0]].get('type')!r} "
                        f"source_constant={graph_obj.nodes[source_nodes[0]].get('constant')!r} "
                        f"source_attributes={source_attributes!r} "
                        f"source_leaves={tuple(source_attributes.get('aggregate_leaf_value_ids', ()))!r} "
                        f"materializer_leaves={tuple(attributes.get('aggregate_leaf_value_ids', ()))!r}",
                        file=sys.stderr,
                    )
                # Prefer the authored singleton aggregate's leaf ledger.  A
                # bytes materializer records its direct argument (the list)
                # as its leaf, while the list records the actual scalar.  The
                # latter is what append_scalar must consume.  Fall back to the
                # materializer ledger only for frontends which omit source
                # aggregate metadata.
                leaves = (
                    tuple(map(int, source_attributes.get(
                        "aggregate_leaf_value_ids", ()
                    )))
                    if literal_singleton else ()
                )
                if not leaves and literal_singleton:
                    leaves = tuple(map(int, attributes.get(
                        "aggregate_leaf_value_ids", ()
                    )))
                if len(leaves) == 1:
                    singleton_by_resident[int(resident_id)] = leaves[0]
                continue
        resident_by_value[value_id] = value_id
        kind_by_value[value_id] = kind
        leaves = tuple(map(int, attributes.get(
            "aggregate_leaf_value_ids", ()
        )))
        if len(leaves) == 1:
            singleton_by_resident[value_id] = leaves[0]

    for value_id, kind in dict(call_result_kinds or {}).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(value_id)
        kind_by_value[int(value_id)] = str(kind)
    for value_id, (resident_id, kind) in dict(
        structural_aliases or {}
    ).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(resident_id)
        resident_by_value.setdefault(int(resident_id), int(resident_id))
        kind_by_value[int(value_id)] = str(kind)
        kind_by_value.setdefault(int(resident_id), str(kind))
    # IndexedStore and its loop-result versions are deterministic names for
    # the same resident arena.  Retain that correlation while recognizing
    # structural concatenations across control-versioned sequence names.
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    changed = True
    while changed:
        changed = False
        for alias_id, source_id in storage_aliases.items():
            resident_id = resident_by_value.get(int(source_id))
            kind = kind_by_value.get(int(source_id))
            if resident_id is None or kind not in sequence_kinds:
                continue
            if resident_by_value.get(int(alias_id)) != int(resident_id):
                resident_by_value[int(alias_id)] = int(resident_id)
                kind_by_value[int(alias_id)] = str(kind)
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            value_id = int(data.get("value_id", node_id))
            resolved_parents = {
                (
                    int(resident_by_value[parent_value]),
                    str(kind_by_value[parent_value]),
                )
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                for parent_value in (
                    int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                if parent_value in resident_by_value
                and parent_value in kind_by_value
            }
            if len(resolved_parents) != 1:
                continue
            resident_id, kind = next(iter(resolved_parents))
            if resident_by_value.get(value_id) != resident_id:
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                changed = True

    operations: list[
        tuple[int, int, int, str, int | None, int | None]
    ] = []
    # Deterministic ids are source ordered, so inner concatenations precede
    # the outer expression that consumes them.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("type") or data.get("op") or "") != "Add":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_node = parents.get("lhs")
        rhs_node = parents.get("rhs")
        if lhs_node is None or rhs_node is None:
            continue
        lhs_value = int(graph_obj.nodes[lhs_node].get("value_id", lhs_node))
        rhs_value = int(graph_obj.nodes[rhs_node].get("value_id", rhs_node))
        lhs = resident_by_value.get(lhs_value)
        rhs = resident_by_value.get(rhs_value)
        if lhs is None or rhs is None:
            if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                print(
                    "DEBUGSEQMISS "
                    f"fn={graph_obj.graph.get('function_name')} "
                    f"result={int(data.get('value_id', node_id))} "
                    f"lhs_value={lhs_value} lhs_resident={lhs} "
                    f"rhs_value={rhs_value} rhs_resident={rhs} "
                    f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                    file=sys.stderr,
                )
            continue
        lhs_kind = kind_by_value.get(lhs_value, kind_by_value.get(lhs))
        rhs_kind = kind_by_value.get(rhs_value, kind_by_value.get(rhs))
        if lhs_kind != rhs_kind:
            if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                print(
                    "DEBUGSEQMISS "
                    f"fn={graph_obj.graph.get('function_name')} "
                    f"result={int(data.get('value_id', node_id))} "
                    f"lhs_kind={lhs_kind!r} rhs_kind={rhs_kind!r} "
                    f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                    file=sys.stderr,
                )
            continue
        result_id = int(data.get("value_id", node_id))
        result_kind = str(lhs_kind)
        operations.append((
            result_id, int(lhs), int(rhs), result_kind,
            singleton_by_resident.get(int(lhs)),
            singleton_by_resident.get(int(rhs)),
        ))
        if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
            print(
                "DEBUGSEQ "
                f"fn={graph_obj.graph.get('function_name')} "
                f"result={result_id} lhs={int(lhs)} rhs={int(rhs)} "
                f"lhs_scalar={singleton_by_resident.get(int(lhs))} "
                f"rhs_scalar={singleton_by_resident.get(int(rhs))} "
                f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                file=sys.stderr,
            )
        resident_by_value[result_id] = result_id
        kind_by_value[result_id] = result_kind

    aliases = tuple(sorted(
        (int(value_id), int(resident_id))
        for value_id, resident_id in resident_by_value.items()
        if int(value_id) != int(resident_id)
    ))
    singleton_materializations = {
        int(value_id): int(singleton_by_resident[int(resident_id)])
        for value_id, resident_id in aliases
        if int(resident_id) in singleton_by_resident
    }
    return tuple(operations), aliases, singleton_materializations


def _sequence_value_kinds(
    graph_obj: Any,
    *,
    return_sequence_kind_by_reference: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Resolve sequence semantics through calls and structural expressions.

    A pursued callee's result is represented in its caller by the callee's
    return value identity, not by another aggregate-construction node. Merely
    scanning nodes carrying ``aggregate_kind`` therefore loses a sequence at
    exactly the boundary where an ordinary helper returns it. Resolve direct
    aggregate facts, known call returns, materialization aliases, and sequence
    concatenations as one backend-independent structural closure.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}
    kinds: dict[int, str] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        kinds[value_id] = kind
        if attributes.get("producer_kind") != "aggregate_materialization":
            continue
        sources = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role).startswith("arg:") and parent in graph_obj
        )
        if len(sources) == 1:
            kinds[sources[0]] = kind

    call_result_kinds: dict[int, str] = {}
    return_kinds = dict(return_sequence_kind_by_reference or {})
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref", attributes.get("method_ref")
        )
        if reference is None:
            continue
        kind = return_kinds.get(int(reference))
        if kind in sequence_kinds:
            value_id = int(data.get("value_id", node_id))
            call_result_kinds[value_id] = str(kind)
            kinds[value_id] = str(kind)

    structural_aliases = {
        int(result_id): (int(source_id), "bytes")
        for result_id, source_id, _source_name, _transform
        in (
            *_utf8_encode_aliases(graph_obj),
            *(
                record for record in _bytes_join_source_transforms(graph_obj)
                if str(record[3]) == "join_bytes"
            ),
        )
    }
    concatenations, aliases, _singletons = _sequence_concat_ops(
        graph_obj,
        call_result_kinds=call_result_kinds,
        structural_aliases=structural_aliases,
    )
    for result_id, _lhs, _rhs, kind, _lhs_scalar, _rhs_scalar in concatenations:
        kinds[int(result_id)] = str(kind)
    for alias_id, resident_id in aliases:
        resident_kind = kinds.get(int(resident_id))
        if resident_kind is not None:
            kinds[int(alias_id)] = resident_kind
    return kinds


def _promote_conditional_sequence_aliases(
    control: Any,
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str] | None = None,
    structural_aliases: Mapping[int, tuple[int, str]] | None = None,
):
    """Turn branch-carried aggregate versions into resident arena carries.

    Ordinary conditional planning records every authored assignment in the
    scalar-shaped ``carried_aliases`` tuple.  Once source/call analysis proves
    those values are sequences, retain one initial arena, copy the selected
    branch value into it, and correlate the post-branch SSA spelling with that
    arena.  This is an IR correction, not a receipt-only relabeling.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock, WhileBlock,
    )

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    kind_by_value: dict[int, str] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        resident_by_value[value_id] = value_id
        kind_by_value[value_id] = kind
    for value_id, kind in dict(call_result_kinds or {}).items():
        if str(kind) in sequence_kinds:
            resident_by_value[int(value_id)] = int(value_id)
            kind_by_value[int(value_id)] = str(kind)
    for value_id, (resident_id, kind) in dict(
        structural_aliases or {}
    ).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(resident_id)
        kind_by_value[int(value_id)] = str(kind)
        kind_by_value.setdefault(int(resident_id), str(kind))

    # In-place stores and their Phi/loop-result spellings are storage
    # versions, not scalar values.  Resolve them to their original arena before
    # inspecting conditional carries, otherwise the branch becomes an opaque
    # scalar Phi and every later row projection leaks into the public ABI.
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    changed = True
    while changed:
        changed = False
        for alias_id, source_id in storage_aliases.items():
            resident_id = resident_by_value.get(int(source_id))
            kind = kind_by_value.get(int(source_id))
            if resident_id is None or kind not in sequence_kinds:
                continue
            if resident_by_value.get(int(alias_id)) != int(resident_id):
                resident_by_value[int(alias_id)] = int(resident_id)
                kind_by_value[int(alias_id)] = str(kind)
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            value_id = int(data.get("value_id", node_id))
            resolved_parents = {
                (
                    int(resident_by_value[parent_value]),
                    str(kind_by_value[parent_value]),
                )
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                for parent_value in (
                    int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                if parent_value in resident_by_value
                and parent_value in kind_by_value
            }
            if len(resolved_parents) != 1:
                continue
            resident_id, kind = next(iter(resolved_parents))
            if resident_by_value.get(value_id) != resident_id:
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                changed = True

    promoted_aliases: dict[int, tuple[int, str]] = {}
    destination_ids: set[int] = set()

    def promote(block):
        if isinstance(block, ConditionalBlock):
            scalar = []
            sequences = list(block.carried_sequence_aliases)
            for carried in block.carried_aliases:
                true_id, false_id, initial_id, merged_id = map(int, carried)
                initial_resident = resident_by_value.get(initial_id)
                initial_kind = kind_by_value.get(initial_id)
                true_resident = resident_by_value.get(true_id)
                false_resident = resident_by_value.get(false_id)
                true_kind = kind_by_value.get(true_id)
                false_kind = kind_by_value.get(false_id)
                if (
                    initial_resident is None
                    or initial_kind not in sequence_kinds
                    or true_resident is None
                    or false_resident is None
                    or true_kind != initial_kind
                    or false_kind != initial_kind
                ):
                    scalar.append(carried)
                    continue
                destination = int(initial_resident)
                sequences.append((
                    int(true_resident), int(false_resident),
                    destination, merged_id,
                ))
                destination_ids.add(destination)
                resident_by_value[merged_id] = destination
                kind_by_value[merged_id] = initial_kind
                promoted_aliases[merged_id] = (destination, initial_kind)
            return replace(
                block,
                body=promote(block.body),
                orelse=(
                    None if block.orelse is None else promote(block.orelse)
                ),
                carried_aliases=tuple(scalar),
                carried_sequence_aliases=tuple(sequences),
            )
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(promote(x) for x in block.blocks))
        if isinstance(block, LoopBlock):
            promoted_body = promote(block.body)
            scalar_carries = []
            sequence_updates: set[tuple[int, int]] = set()
            for updated_id, initial_id in block.carried_aliases:
                updated_id = int(updated_id)
                initial_id = int(initial_id)
                updated_resident = resident_by_value.get(updated_id)
                initial_resident = resident_by_value.get(initial_id)
                if (
                    updated_resident is None
                    or initial_resident is None
                    or updated_resident != initial_resident
                    or kind_by_value.get(updated_id) not in sequence_kinds
                    or kind_by_value.get(initial_id) not in sequence_kinds
                ):
                    scalar_carries.append((updated_id, initial_id))
                    continue
                sequence_updates.add((updated_id, initial_id))
                kind = str(kind_by_value[initial_id])
                promoted_aliases[updated_id] = (
                    int(initial_resident), kind,
                )
                destination_ids.add(int(initial_resident))
            scalar_ports = []
            for port_id, initial_id, updated_id in block.result_ports:
                key = (int(updated_id), int(initial_id))
                if key not in sequence_updates:
                    scalar_ports.append((
                        int(port_id), int(initial_id), int(updated_id)
                    ))
                    continue
                resident_id = int(resident_by_value[int(initial_id)])
                kind = str(kind_by_value[int(initial_id)])
                resident_by_value[int(port_id)] = resident_id
                kind_by_value[int(port_id)] = kind
                promoted_aliases[int(port_id)] = (resident_id, kind)
            return replace(
                block,
                body=promoted_body,
                carried_aliases=tuple(scalar_carries),
                result_ports=tuple(scalar_ports),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=promote(block.condition),
                body=promote(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=promote(block.callee))
        return block

    promoted = replace(control, root=promote(control.root))
    return promoted, promoted_aliases, tuple(sorted(destination_ids))


def _constant_struct_pack_materializations(graph_obj: Any):
    """Fold fully source-static ``struct.pack`` calls to byte-sequence facts."""

    import struct

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        if attributes.get("extraction_identity") != "_struct.pack":
            continue
        arguments = tuple(
            int(parent)
            for parent, role in sorted(
                data.get("parents") or (),
                key=lambda item: int(str(item[1]).split(":", 1)[1])
                if str(item[1]).startswith("arg:") else 1 << 30,
            )
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        values = []
        for argument in arguments:
            argument_data = graph_obj.nodes[argument]
            argument_attributes = argument_data.get("attributes") or {}
            if "value" in argument_attributes:
                values.append(argument_attributes["value"])
            elif argument_data.get("constant") is not None:
                values.append(argument_data["constant"])
            else:
                break
        if len(values) != len(arguments) or not values:
            continue
        try:
            payload = struct.pack(*values)
        except (struct.error, TypeError, ValueError):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(payload),
        })
        data["attributes"] = updated
        materializations.append((
            value_id,
            bytes(payload),
            int(node_id),
            "_struct.pack",
        ))
    return tuple(materializations)


def _constant_byte_literal_materializations(graph_obj: Any):
    """Publish authored byte literals through the resident sequence ABI."""

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        literal = attributes.get("value", data.get("constant"))
        if not isinstance(literal, bytes):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(literal),
        })
        data["attributes"] = updated
        materializations.append((
            value_id, bytes(literal), int(node_id), None,
        ))
    return tuple(materializations)


def _sequence_append_call_mutations(graph_obj: Any):
    """Recover authored resident ``append/add`` calls as lexical effects."""

    from .control_source import ControlSequenceMutation

    sequence_kinds = {"list", "set", "bytes", "bytearray"}
    mutations = [
        ControlSequenceMutation(
            sequence_value_id=int(record["sequence_value_id"]),
            operator=str(record["operator"]),
            argument_value_ids=tuple(map(
                int, record["argument_value_ids"]
            )),
            effect_node_id=int(node_id),
            policy=record.get("policy"),
        )
        for node_id, record in sorted(
            (
                graph_obj.graph.get("source_sequence_mutation_records")
                or {}
            ).items(),
            key=lambda item: int(item[0]),
        )
    ]
    recorded_effect_ids = {
        int(mutation.effect_node_id) for mutation in mutations
    }
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"append", "add"}
        ):
            continue
        if int(node_id) in recorded_effect_ids:
            continue
        parents = tuple(data.get("parents") or ())
        destination_node = next((
            int(parent) for parent, role in parents
            if str(role) == "operand" and int(parent) in graph_obj
        ), None)
        arguments = tuple(
            int(parent) for parent, role in parents
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        if destination_node is None or len(arguments) != 1:
            continue
        destination = graph_obj.nodes[destination_node]
        attributes = destination.get("attributes") or {}
        if attributes.get("aggregate_kind") not in sequence_kinds:
            continue
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(destination.get(
                "value_id", destination_node
            )),
            operator=str(expression.func.attr),
            argument_value_ids=(int(graph_obj.nodes[arguments[0]].get(
                "value_id", arguments[0]
            )),),
            effect_node_id=int(data.get("value_id", node_id)),
            policy=(
                "unique" if expression.func.attr == "add"
                else "duplicates"
            ),
        ))
    nodes_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    expanded_mutations = []
    for mutation in mutations:
        arguments: list[int] = []
        for value_id in mutation.argument_value_ids:
            attributes = (
                nodes_by_value.get(int(value_id), {}).get("attributes") or {}
            )
            leaves = tuple(map(
                int, attributes.get("aggregate_leaf_value_ids", ())
            ))
            if attributes.get("aggregate_kind") == "tuple" and leaves:
                arguments.extend(leaves)
            else:
                arguments.append(int(value_id))
        expanded_mutations.append(replace(
            mutation,
            argument_value_ids=tuple(arguments),
            argument_kind=(
                "row" if len(arguments) > 1 else mutation.argument_kind
            ),
        ))
    return tuple(expanded_mutations)


def _joined_byte_sequence_ids(
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str],
    declared_sequence_ids: Iterable[int],
    structural_byte_sequence_ids: Iterable[int] = (),
    transformed_call_source_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Find resident ``list[bytes]`` identities needing dual ABI views.

    The authored outer list owns an element count, while its eventual empty
    byte join owns a flattened byte stream. Parameters already express these
    as ``row_count``/``join_bytes`` source transforms; this recovers the same
    representation for locals without inspecting Python values.
    """

    declared = set(map(int, declared_sequence_ids))
    node_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    byte_values = {
        int(value_id)
        for value_id, data in node_by_value.items()
        if str((data.get("attributes") or {}).get("aggregate_kind"))
        in {"bytes", "bytearray"}
    }
    byte_values.update(
        int(value_id)
        for value_id, kind in call_result_kinds.items()
        if str(kind) in {"bytes", "bytearray"}
    )
    byte_values.update(map(int, structural_byte_sequence_ids))
    seeds: set[int] = set(map(int, transformed_call_source_ids))
    for value_id, data in node_by_value.items():
        attributes = data.get("attributes") or {}
        if attributes.get("aggregate_kind") != "list":
            continue
        leaves = tuple(map(
            int, attributes.get("aggregate_leaf_value_ids", ())
        ))
        if not leaves:
            leaves = tuple(
                int(graph_obj.nodes[parent].get("value_id", parent))
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            )
        if leaves and all(int(leaf) in byte_values for leaf in leaves):
            seeds.add(int(value_id))
    append_mutations = _sequence_append_call_mutations(graph_obj)
    for mutation in append_mutations:
        if (
            mutation.operator == "append"
            and len(mutation.argument_value_ids) == 1
            and int(mutation.argument_value_ids[0]) in byte_values
        ):
            seeds.add(int(mutation.sequence_value_id))

    if os.environ.get("TURING_DEBUG_JOINED_SEQUENCE"):
        print(
            "DEBUGJOINED "
            f"fn={graph_obj.graph.get('function_name')} "
            f"bytes={tuple(sorted(byte_values))!r} "
            f"seeds={tuple(sorted(seeds))!r} "
            f"mutations={tuple((int(item.sequence_value_id), tuple(map(int, item.argument_value_ids))) for item in append_mutations)!r} "
            f"transformed={tuple((int(value_id), str((node_by_value.get(int(value_id), {}).get('attributes') or {}).get('binding_name')), ast.dump(node_by_value.get(int(value_id), {}).get('expr_obj'), include_attributes=False) if isinstance(node_by_value.get(int(value_id), {}).get('expr_obj'), ast.AST) else None) for value_id in sorted(map(int, transformed_call_source_ids)))!r} "
            f"identities={tuple((str(name), tuple(map(int, history))) for name, history in (graph_obj.graph.get('identity_table') or {}).items() if str(name) in {'types', 'entries'})!r} "
            f"declared={tuple(sorted(declared))!r}",
            file=sys.stderr,
        )

    mutation_destinations = {
        int(item.sequence_value_id) for item in append_mutations
    }
    joined = {
        int(value_id)
        for value_id in seeds
        if (
            int(value_id) in declared
            or int(value_id) in mutation_destinations
            or (
                node_by_value.get(int(value_id), {}).get("attributes") or {}
            ).get("aggregate_kind") == "list"
            or isinstance(
                (
                    (graph_obj.nodes.get(int(value_id), {}).get("attributes")
                     or {}).get(
                        "value",
                        graph_obj.nodes.get(int(value_id), {}).get("constant"),
                    )
                ),
                list,
            )
        )
    }
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        identities = set(map(int, history))
        if identities & seeds:
            joined.update(identities & declared)
    return tuple(sorted(joined))


def _joined_list_literal_mutations(
    graph_obj: Any, joined_sequence_ids: Iterable[int]
) -> tuple[Any, ...]:
    """Materialize each authored list element into its resident dual view."""

    from .control_source import ControlSequenceMutation

    joined = set(map(int, joined_sequence_ids))
    mutations = []
    emitted: set[tuple[int, int, int]] = set()

    def append_mutation(
        sequence_id: int, effect_node_id: int, value_id: int,
    ) -> None:
        key = (int(sequence_id), int(effect_node_id), int(value_id))
        if key in emitted:
            return
        emitted.add(key)
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(sequence_id),
            operator="append",
            argument_value_ids=(int(value_id),),
            effect_node_id=int(effect_node_id),
            policy="duplicates",
            argument_kind="joined_literal_element",
        ))

    nodes = tuple(graph_obj.nodes(data=True))

    def expression_node(expression: ast.AST) -> tuple[int, int] | None:
        exact = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            if data.get("expr_obj") is expression
        ]
        if len(exact) == 1:
            return exact[0]
        position = (
            int(getattr(expression, "lineno", -1) or -1),
            int(getattr(expression, "col_offset", -1) or -1),
            int(getattr(expression, "end_lineno", -1) or -1),
            int(getattr(expression, "end_col_offset", -1) or -1),
            type(expression),
        )
        positioned = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            for candidate in (data.get("expr_obj"),)
            if isinstance(candidate, ast.AST)
            and (
                int(getattr(candidate, "lineno", -2) or -2),
                int(getattr(candidate, "col_offset", -2) or -2),
                int(getattr(candidate, "end_lineno", -2) or -2),
                int(getattr(candidate, "end_col_offset", -2) or -2),
                type(candidate),
            ) == position
        ]
        return positioned[0] if len(positioned) == 1 else None

    for node_id, data in sorted(
        nodes, key=lambda item: int(item[0])
    ):
        sequence_id = int(data.get("value_id", node_id))
        if sequence_id not in joined:
            continue
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.List):
            continue
        elements = sorted(
            [
                (
                int(parent), str(role),
                int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            ],
            key=lambda item: (
                int(item[1].split(":", 1)[1])
                if ":" in item[1] and item[1].split(":", 1)[1].isdigit()
                else item[0]
            ),
        )
        for parent, _role, value_id in elements:
            append_mutation(sequence_id, parent, value_id)

    # The source realizer may specialize an inline dynamic list into an empty
    # structural resident while retaining the authored list only on the Call
    # that consumes it (``_vector([uleb(index)])``). Its elements are still
    # ordinary ProcessGraph nodes. Correlate those exact AST objects/source
    # spans back to their deterministic value identities and initialize the
    # resident before the consuming call.
    for _call_node_id, data in sorted(nodes, key=lambda item: int(item[0])):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call) or not expression.args:
            continue
        literal = expression.args[0]
        if not isinstance(literal, ast.List):
            continue
        sequence_ids = tuple(dict.fromkeys(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "arg:0" and parent in graph_obj
        ))
        if len(sequence_ids) != 1 or sequence_ids[0] not in joined:
            continue
        for element in literal.elts:
            correlated = expression_node(element)
            if correlated is None:
                continue
            effect_node_id, value_id = correlated
            append_mutation(sequence_ids[0], effect_node_id, value_id)
    return tuple(mutations)


def _graph_control_expression(
    graph_obj: Any, node_id: int, visiting: frozenset[int] = frozenset(),
):
    """Translate a retained scalar predicate from ProcessGraph structure."""

    from .control_source import ControlExpression

    node_id = int(node_id)
    if node_id in visiting or node_id not in graph_obj:
        return ControlExpression("value", value_id=node_id)
    data = graph_obj.nodes[node_id]
    expression = data.get("expr_obj")
    attributes = data.get("attributes") or {}
    literal = attributes.get("value", data.get("constant"))
    if isinstance(expression, ast.Constant):
        literal = expression.value
        if literal is None:
            return ControlExpression("const", value_id=node_id, literal=None)
    if isinstance(literal, (bool, int, float)) and str(
        data.get("type") or ""
    ).casefold() in {"constant", "const"}:
        return ControlExpression("const", value_id=node_id, literal=literal)
    if attributes.get("aggregate_kind") in {
        "list", "set", "dict", "tuple", "bytes", "bytearray",
    }:
        return ControlExpression(
            "sequence_nonempty",
            (ControlExpression("value", value_id=node_id),),
            value_id=node_id,
            literal=attributes.get("aggregate_kind") in {"set", "dict"},
        )
    parents = tuple(data.get("parents") or ())
    ordered = tuple(
        int(parent)
        for parent, role in sorted(
            parents,
            key=lambda item: (
                int(str(item[1]).split(":")[-1])
                if str(item[1]).split(":")[-1].isdigit() else 0
            ),
        )
        if str(role) not in {"callee", "ops", "operator"}
    )
    if isinstance(expression, ast.BoolOp):
        operator = "and" if isinstance(expression.op, ast.And) else "or"
        operands = tuple(
            _graph_control_expression(
                graph_obj, parent, visiting | {node_id}
            )
            for parent in ordered
        )
        if operands:
            result = operands[0]
            for operand in operands[1:]:
                result = ControlExpression(
                    operator, (result, operand), value_id=node_id
                )
            return result
    operation = {
        "add": "add", "sub": "sub", "mul": "mul",
        "div": "div", "truediv": "div",
        "less": "lt", "lt": "lt",
        "lessequal": "le", "less_equal": "le", "le": "le",
        "greater": "gt", "gt": "gt",
        "greaterequal": "ge", "greater_equal": "ge", "ge": "ge",
        "equal": "eq", "eq": "eq",
        "notequal": "ne", "not_equal": "ne", "ne": "ne",
        "logical_and": "and", "land": "and",
        "logical_or": "or", "lor": "or",
        "logical_not": "not", "lnot": "not",
        "neg": "neg", "usub": "neg",
        "bitand": "bitand", "bitor": "bitor",
        "bitxor": "bitxor", "shl": "shl", "shr": "shr",
        "invert": "invert",
    }.get(str(data.get("op") or data.get("type") or "").casefold())
    if operation is not None:
        arity = 1 if operation in {"not", "neg", "invert"} else 2
        operands = tuple(
            _graph_control_expression(
                graph_obj, parent, visiting | {node_id}
            )
            for parent in ordered[:arity]
        )
        if len(operands) == arity:
            return ControlExpression(operation, operands, value_id=node_id)
    return ControlExpression("value", value_id=node_id)


def _attach_graph_control_expressions(control: Any, graph_obj: Any):
    """Recover structured predicates for every retained conditional."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock, WhileBlock,
    )

    def attach(block):
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=attach(block.body),
                orelse=(None if block.orelse is None else attach(block.orelse)),
                predicate_expression=_graph_control_expression(
                    graph_obj, int(block.predicate_value_id)
                ),
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block, blocks=tuple(attach(item) for item in block.blocks)
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=attach(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=attach(block.condition),
                body=attach(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=attach(block.callee))
        return block

    return replace(control, root=attach(control.root))


def _install_lexical_sequence_mutations(
    control: Any,
    graph: Any,
    dispatch_subgraphs: Iterable[Any],
    *,
    extra_mutations: Iterable[Any] = (),
):
    """Place non-loop sequence effects into their authored control scope."""

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        ControlExpression,
        LoopBlock,
        SequenceBlock,
        SequenceMutationBlock,
        StatementBlock,
        WhileBlock,
    )
    from .glsl_deployment_strategy import (
        _branch_compartments,
        _retained_control_value_id,
        _source_control_records,
    )

    mutations = tuple((
        *_sequence_append_call_mutations(graph.G),
        *tuple(extra_mutations),
    ))
    if not mutations:
        return control, ()

    existing_effect_ids: set[int] = set()

    def gather(block):
        if isinstance(block, SequenceMutationBlock):
            existing_effect_ids.add(int(block.mutation.effect_node_id))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather(child)
        elif isinstance(block, ConditionalBlock):
            gather(block.body)
            if block.orelse is not None:
                gather(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            existing_effect_ids.update(
                int(item.effect_node_id) for item in block.sequence_mutations
            )
            if isinstance(block, WhileBlock):
                gather(block.condition)
            gather(block.body)
        elif isinstance(block, CallBlock):
            gather(block.callee)

    gather(control.root)
    mutations = tuple(
        item for item in mutations
        if int(item.effect_node_id) not in existing_effect_ids
    )
    if not mutations:
        return control, ()

    def node_position(node_id: int) -> tuple[int, int, int]:
        data = graph.G.nodes.get(int(node_id), {})
        expression = data.get("expr_obj")
        span = data.get("source_span") or {}
        return (
            int(getattr(expression, "lineno", span.get("line", 1 << 30)) or (1 << 30)),
            int(getattr(expression, "col_offset", span.get("column", 0)) or 0),
            int(node_id),
        )

    region_positions = {
        int(index): min(
            (
                node_position(int(node_id))
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            ),
            default=(1 << 30, 0, int(index)),
        )
        for index, subgraph in enumerate(dispatch_subgraphs)
    }

    def block_position(block) -> tuple[int, int, int]:
        if isinstance(block, SequenceMutationBlock):
            return node_position(int(block.mutation.effect_node_id))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            line = block.lines[0]
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                return region_positions.get(
                    int(line[len("__scheduled_region_"):-2]),
                    (1 << 30, 0, 0),
                )
        if isinstance(block, ConditionalBlock) and block.source_node_id is not None:
            return node_position(int(block.source_node_id))
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                return node_position(int(source_id))
        if isinstance(block, CallBlock):
            return node_position(int(block.callsite_id))
        if isinstance(block, SequenceBlock) and block.blocks:
            return min(map(block_position, block.blocks))
        return (1 << 30, 0, 0)

    def insert_ordered(block, mutation_block):
        sequence = block if isinstance(block, SequenceBlock) else SequenceBlock((block,))
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(sequence.blocks)
        ]
        decorated.append((
            block_position(mutation_block), len(decorated), mutation_block,
        ))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                decorated, key=lambda item: (item[0], item[1])
            )
        ))

    memberships = _branch_compartments(graph)
    retained_mutation_records = (
        graph.G.graph.get("source_sequence_mutation_records") or {}
    )

    def insert_in_conditional(block, owner_id, arm, mutation_block):
        if isinstance(block, ConditionalBlock):
            if int(block.source_node_id or -1) == int(owner_id):
                if str(arm) == "body":
                    return replace(
                        block,
                        body=insert_ordered(block.body, mutation_block),
                    ), True
                return replace(
                    block,
                    orelse=insert_ordered(
                        block.orelse or SequenceBlock(()), mutation_block,
                    ),
                ), True
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_in_conditional(
                    block.orelse, owner_id, arm, mutation_block
                )
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = insert_in_conditional(
                    child, owner_id, arm, mutation_block
                )
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, LoopBlock):
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            return replace(block, body=body), inserted
        if isinstance(block, WhileBlock):
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_in_conditional(
                block.callee, owner_id, arm, mutation_block
            )
            return replace(block, callee=callee), inserted
        return block, False

    root = control.root
    unplaced = []
    for mutation in mutations:
        mutation_block = SequenceMutationBlock(mutation)
        guarded = tuple(dict.fromkeys((
            *tuple(
            (int(owner), str(arm))
            for owner, arm in memberships.get(int(mutation.effect_node_id), ())
            if str(arm) in {"body", "orelse"}
            ),
            *tuple(
                (int(owner), str(arm))
                for owner, arm in (
                    retained_mutation_records.get(
                        int(mutation.effect_node_id), {}
                    ).get("branch_memberships", ())
                )
                if str(arm) in {"body", "orelse"}
            ),
        )))
        if os.environ.get("TURING_DEBUG_SEQUENCE_MUTATION"):
            print(
                "DEBUGMUTATION "
                f"fn={graph.G.graph.get('function_name')} "
                f"effect={int(mutation.effect_node_id)} "
                f"sequence={int(mutation.sequence_value_id)} "
                f"guarded={guarded}",
                file=sys.stderr,
            )
        inserted = False
        # Innermost authored condition has the latest source position.
        for owner_id, arm in sorted(
            guarded, key=lambda item: node_position(item[0]), reverse=True,
        ):
            root, inserted = insert_in_conditional(
                root, owner_id, arm, mutation_block
            )
            if inserted:
                break
        if not inserted:
            # A branch containing only structural storage effects owns no
            # numerical region, so ordinary region overlay has no marker from
            # which to materialize its ConditionalBlock.  Reconstruct that
            # authored branch from the retained pre-fold control record, then
            # place the complete wrapper inside its nearest surviving outer
            # conditional (or at the lexical root).
            synthesized = None
            remaining_guards = guarded
            if guarded:
                owner_id, arm = sorted(
                    guarded,
                    key=lambda item: node_position(item[0]),
                    reverse=True,
                )[0]
                record = _source_control_records(graph.G).get(owner_id)
                owner_node_id = (
                    int(owner_id) if int(owner_id) in graph.G else next((
                        int(node_id)
                        for node_id, data in graph.G.nodes(data=True)
                        if int(data.get("value_id", node_id)) == int(owner_id)
                    ), None)
                )
                owner_data = (
                    {} if owner_node_id is None
                    else graph.G.nodes[int(owner_node_id)]
                )
                expression = (
                    record.get("expression")
                    if record is not None
                    else owner_data.get("expr_obj")
                )
                recorded_predicate_id = (
                    None if record is None else record.get("predicate_id")
                )
                if recorded_predicate_id is None:
                    recorded_predicate_id = next((
                        int(parent)
                        for parent, role in owner_data.get("parents", ())
                        if str(role) == "test"
                    ), None)
                predicate_id = (
                    None if not isinstance(expression, ast.If)
                    else _retained_control_value_id(
                        graph.G,
                        recorded_predicate_id,
                        expression.test,
                    )
                )
                if predicate_id is None and isinstance(expression, ast.If):
                    direct_test = next((
                        int(parent)
                        for parent, role in owner_data.get("parents", ())
                        if str(role) == "test" and int(parent) in graph.G
                    ), None)
                    if direct_test is not None:
                        predicate_id = int(
                            graph.G.nodes[direct_test].get(
                                "value_id", direct_test
                            )
                        )
                if (
                    predicate_id is None
                    and recorded_predicate_id is not None
                    and isinstance(expression, ast.If)
                ):
                    # The conditional node may have been folded out after its
                    # immutable source-control record was published.  That
                    # record already speaks in the canonical value identity
                    # consumed by region/control SSA; retain it directly.
                    predicate_id = int(recorded_predicate_id)
                if os.environ.get("TURING_DEBUG_SEQUENCE_MUTATION"):
                    print(
                        "DEBUGMUTATION-GUARD "
                        f"effect={int(mutation.effect_node_id)} "
                        f"owner={owner_id} owner_node={owner_node_id} "
                        f"recorded_predicate={recorded_predicate_id} "
                        f"predicate={predicate_id} "
                        f"expression={type(expression).__name__}",
                        file=sys.stderr,
                    )
                if predicate_id is not None:
                    empty = SequenceBlock(())
                    synthesized = ConditionalBlock(
                        int(predicate_id),
                        mutation_block if arm == "body" else empty,
                        mutation_block if arm == "orelse" else None,
                        predicate_expression=_graph_control_expression(
                            graph.G, int(predicate_id)
                        ),
                        source_node_id=int(owner_id),
                    )
                    remaining_guards = tuple(
                        item for item in guarded if item != (owner_id, arm)
                    )
            if synthesized is not None:
                for owner_id, arm in sorted(
                    remaining_guards,
                    key=lambda item: node_position(item[0]),
                    reverse=True,
                ):
                    root, inserted = insert_in_conditional(
                        root, owner_id, arm, synthesized
                    )
                    if inserted:
                        break
                if not inserted:
                    root = insert_ordered(root, synthesized)
            elif guarded:
                unplaced.append(int(mutation.effect_node_id))
            else:
                root = insert_ordered(root, mutation_block)
    return replace(control, root=root), tuple(unplaced)


def _control_expression_value_ids(expression: Any) -> frozenset[int]:
    if expression is None:
        return frozenset()
    return frozenset((
        *((int(expression.value_id),) if expression.value_id is not None else ()),
        *(
            value_id
            for operand in expression.operands
            for value_id in _control_expression_value_ids(operand)
        ),
    ))


def _control_block_consumes_values(block: Any, value_ids: Iterable[int]) -> bool:
    """Whether a control subtree observes any exact SSA identity."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceMutationBlock, SequenceQueryBlock, ValidationBlock,
        WhileBlock,
    )

    wanted = set(map(int, value_ids))
    if not wanted:
        return False
    if isinstance(block, ConditionalBlock):
        if (
            int(block.predicate_value_id) in wanted
            or _control_expression_value_ids(
                block.predicate_expression
            ).intersection(wanted)
        ):
            return True
        return (
            _control_block_consumes_values(block.body, wanted)
            or block.orelse is not None
            and _control_block_consumes_values(block.orelse, wanted)
        )
    if isinstance(block, ValidationBlock):
        return (
            int(block.predicate_value_id) in wanted
            or bool(_control_expression_value_ids(
                block.predicate_expression
            ).intersection(wanted))
        )
    if isinstance(block, SequenceMutationBlock):
        mutation = block.mutation
        return bool(
            set(map(int, mutation.argument_value_ids)).intersection(wanted)
            or _control_expression_value_ids(
                mutation.predicate_expression
            ).intersection(wanted)
        )
    if isinstance(block, SequenceQueryBlock):
        consumed = {int(block.sequence_value_id)}
        if block.default_value_id is not None:
            consumed.add(int(block.default_value_id))
        return bool(consumed.intersection(wanted))
    if isinstance(block, CallBlock):
        return bool(
            {int(caller) for caller, _callee in block.argument_bindings}
            .intersection(wanted)
            or _control_block_consumes_values(block.callee, wanted)
        )
    if isinstance(block, SequenceBlock):
        return any(
            _control_block_consumes_values(child, wanted)
            for child in block.blocks
        )
    if isinstance(block, LoopBlock):
        return _control_block_consumes_values(block.body, wanted)
    if isinstance(block, WhileBlock):
        return (
            _control_block_consumes_values(block.condition, wanted)
            or _control_block_consumes_values(block.body, wanted)
        )
    return False


def _schedule_sequence_query_dependencies(root: Any) -> Any:
    """Place a producer-loop/query unit before its first lexical consumer."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceQueryBlock, WhileBlock,
    )

    if isinstance(root, SequenceBlock):
        scheduled_children = [
            _schedule_sequence_query_dependencies(child)
            for child in root.blocks
        ]
        # Sequential composition is associative: nested SequenceBlocks carry
        # no scope or closure.  Flatten them before dependency scheduling so
        # a generator loop appended in one planner group can move ahead of a
        # consumer retained in an earlier group.
        children = [
            grandchild
            for child in scheduled_children
            for grandchild in (
                child.blocks if isinstance(child, SequenceBlock) else (child,)
            )
        ]
        changed = True
        while changed:
            changed = False
            for loop_index, loop in enumerate(children):
                if (
                    not isinstance(loop, LoopBlock)
                    or loop.source_loop_node_id is None
                ):
                    continue
                query_indexes = [
                    index
                    for index, candidate in enumerate(children)
                    if isinstance(candidate, SequenceQueryBlock)
                    and candidate.producer_loop_node_id is not None
                    and int(candidate.producer_loop_node_id)
                    == int(loop.source_loop_node_id)
                ]
                queries = [children[index] for index in query_indexes]
                if not queries:
                    continue
                produced = {
                    int(value_id)
                    for query in queries
                    for value_id in (
                        query.result_value_id, *query.result_alias_ids,
                    )
                }
                unit_indexes = {loop_index, *query_indexes}
                consumer_index = next((
                    index
                    for index, candidate in enumerate(children)
                    if index not in unit_indexes
                    and _control_block_consumes_values(candidate, produced)
                ), None)
                # Source-position insertion and later structural recovery may
                # separate a generator loop from its query, or even leave the
                # query before the loop.  The pair is one semantic producer
                # unit.  Normalize it at its earliest existing position, and
                # move it farther forward when an earlier consumer requires
                # that dominance edge.
                anchor = min(unit_indexes)
                if consumer_index is not None:
                    anchor = min(anchor, consumer_index)
                rebuilt = [
                    candidate
                    for index, candidate in enumerate(children)
                    if index not in unit_indexes
                ]
                adjusted_anchor = sum(
                    1
                    for index in range(anchor)
                    if index not in unit_indexes
                )
                rebuilt[adjusted_anchor:adjusted_anchor] = [loop, *queries]
                if rebuilt != children:
                    children = rebuilt
                    changed = True
                    break
        return replace(root, blocks=tuple(children))
    if isinstance(root, ConditionalBlock):
        return replace(
            root,
            body=_schedule_sequence_query_dependencies(root.body),
            orelse=(
                None if root.orelse is None
                else _schedule_sequence_query_dependencies(root.orelse)
            ),
        )
    if isinstance(root, LoopBlock):
        return replace(
            root, body=_schedule_sequence_query_dependencies(root.body)
        )
    if isinstance(root, WhileBlock):
        return replace(
            root,
            condition=_schedule_sequence_query_dependencies(root.condition),
            body=_schedule_sequence_query_dependencies(root.body),
        )
    if isinstance(root, CallBlock):
        return replace(
            root, callee=_schedule_sequence_query_dependencies(root.callee)
        )
    return root


def _install_lexical_sequence_queries(
    control: Any,
    graph: Any,
    dispatch_subgraphs: Iterable[Any],
):
    """Replace supported generator consumers with resident sequence queries."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceQueryBlock, WhileBlock,
    )

    graph_obj = graph.G
    queries = []
    unsupported = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"len", "next", "sum"}
        ):
            continue
        arguments = tuple(
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        if not arguments:
            continue
        sequence_id = arguments[0]
        sequence_data = graph_obj.nodes[sequence_id]
        sequence_attributes = sequence_data.get("attributes") or {}
        producer_loop_node_id = None
        if (
            sequence_data.get("type") == "LoopResult"
            and sequence_attributes.get("result_kind") == "collection"
        ):
            producer_loop_node_id = int(sequence_attributes["loop_id"])
        elif (
            expression.func.id == "len"
            and sequence_attributes.get("aggregate_kind")
            in {"list", "bytes", "bytearray", "tuple"}
            and sequence_attributes.get("producer_kind")
            in {"sequence_materialization", "aggregate_materialization"}
        ):
            producer_loop_node_id = next((
                int(parent)
                for parent, role in sequence_data.get("parents") or ()
                if str(role) == "generators"
            ), None)
        if producer_loop_node_id is None:
            continue
        identity = (data.get("attributes") or {}).get("extraction_identity")
        result_id = int(data.get("value_id", node_id))
        result_alias_ids = tuple(dict.fromkeys(
            int(graph_obj.nodes[child].get("value_id", child))
            for child, role in data.get("children") or ()
            if str(role) == "lhs"
            and int(child) in graph_obj
            and isinstance(
                graph_obj.nodes[int(child)].get("expr_obj"), ast.Name
            )
        ))
        if os.environ.get("TURING_DEBUG_BUILTIN_SELECTION"):
            print(
                "DEBUGQUERY "
                f"fn={graph_obj.graph.get('function_name')} "
                f"call={int(node_id)} result={result_id} "
                f"aliases={result_alias_ids} sequence={sequence_id}",
                file=sys.stderr,
            )
        if expression.func.id == "len":
            queries.append(SequenceQueryBlock(
                result_value_id=result_id,
                sequence_value_id=sequence_id,
                operation="length",
                source_call_node_id=int(node_id),
                extraction_identity=identity,
                result_alias_ids=result_alias_ids,
                producer_loop_node_id=producer_loop_node_id,
            ))
            continue
        if expression.func.id == "next":
            if len(arguments) != 2:
                unsupported.append(int(node_id))
                continue
            queries.append(SequenceQueryBlock(
                result_value_id=result_id,
                sequence_value_id=sequence_id,
                operation="first_or_default",
                default_value_id=int(arguments[1]),
                source_call_node_id=int(node_id),
                extraction_identity=identity,
                result_alias_ids=result_alias_ids,
                producer_loop_node_id=producer_loop_node_id,
            ))
            continue

        materializer_id = sequence_attributes.get("materializer_node_id")
        materializer = (
            {} if materializer_id not in graph_obj
            else graph_obj.nodes[int(materializer_id)]
        )
        element_id = next((
            int(parent)
            for parent, role in materializer.get("parents") or ()
            if str(role) == "elt" and int(parent) in graph_obj
        ), None)
        element_data = (
            {} if element_id is None else graph_obj.nodes[element_id]
        )
        element_literal = (element_data.get("attributes") or {}).get(
            "value", element_data.get("constant")
        )
        explicit_start = (
            None if len(arguments) == 1
            else (graph_obj.nodes[arguments[1]].get("attributes") or {}).get(
                "value", graph_obj.nodes[arguments[1]].get("constant")
            )
        )
        if element_literal != 1 or explicit_start not in {None, 0}:
            unsupported.append(int(node_id))
            continue
        queries.append(SequenceQueryBlock(
            result_value_id=result_id,
            sequence_value_id=sequence_id,
            operation="length",
            source_call_node_id=int(node_id),
            extraction_identity=identity,
            result_alias_ids=result_alias_ids,
            producer_loop_node_id=producer_loop_node_id,
        ))

    if not queries:
        return control, tuple(unsupported)

    def insert_after_producer(block, query):
        if (
            os.environ.get("TURING_DEBUG_BUILTIN_SELECTION")
            and isinstance(block, LoopBlock)
        ):
            print(
                "DEBUGQUERYLOOP "
                f"call={query.source_call_node_id} "
                f"candidate={block.source_loop_node_id}",
                file=sys.stderr,
            )
        if (
            isinstance(block, LoopBlock)
            and block.source_loop_node_id is not None
            and int(block.source_loop_node_id)
            == int(query.producer_loop_node_id)
        ):
            return SequenceBlock((block, query)), True
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = insert_after_producer(child, query)
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, ConditionalBlock):
            body, inserted = insert_after_producer(block.body, query)
            orelse = block.orelse
            if not inserted and orelse is not None:
                orelse, inserted = insert_after_producer(orelse, query)
            return replace(block, body=body, orelse=orelse), inserted
        if isinstance(block, LoopBlock):
            body, inserted = insert_after_producer(block.body, query)
            return replace(block, body=body), inserted
        if isinstance(block, WhileBlock):
            body, inserted = insert_after_producer(block.body, query)
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_after_producer(block.callee, query)
            return replace(block, callee=callee), inserted
        return block, False

    root = control.root
    unplaced = list(unsupported)
    for query in queries:
        if query.producer_loop_node_id is None:
            unplaced.append(int(query.source_call_node_id))
            continue
        root, inserted = insert_after_producer(root, query)
        if os.environ.get("TURING_DEBUG_BUILTIN_SELECTION"):
            print(
                "DEBUGQUERYPLACE "
                f"fn={graph_obj.graph.get('function_name')} "
                f"call={query.source_call_node_id} "
                f"producer={query.producer_loop_node_id} inserted={inserted}",
                file=sys.stderr,
            )
        if not inserted:
            unplaced.append(int(query.source_call_node_id))
    root = _schedule_sequence_query_dependencies(root)
    return replace(control, root=root), tuple(unplaced)


def _utf8_encode_aliases(
    graph_obj: Any,
) -> tuple[tuple[int, int, str, str], ...]:
    """Return exact ``str.encode('utf-8')`` source-boundary aliases.

    Encoding is an ABI transform: the authored caller still accepts ``str``;
    the native function receives its deterministic UTF-8 byte sequence.  Only
    the explicit UTF-8 spelling is admitted, so locale/default-codec behavior
    is never inferred.
    """

    aliases = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type") or "").casefold() != "encode":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        operand_node = parents.get("operand")
        encoding_node = parents.get("arg:0")
        if operand_node is None or encoding_node is None:
            continue
        operand = graph_obj.nodes.get(operand_node, {})
        encoding = graph_obj.nodes.get(encoding_node, {})
        operand_attributes = operand.get("attributes") or {}
        encoding_value = (encoding.get("attributes") or {}).get(
            "value", encoding.get("constant")
        )
        source_name = operand_attributes.get("binding_name")
        if (
            operand_attributes.get("binding_kind") != "parameter"
            or source_name is None
            or str(encoding_value).casefold().replace("_", "-") != "utf-8"
        ):
            continue
        aliases.append((
            int(data.get("value_id", node_id)),
            int(operand.get("value_id", operand_node)),
            str(source_name),
            "utf8",
        ))
    return tuple(aliases)


def _bytes_join_source_transforms(
    graph_obj: Any,
) -> tuple[tuple[int, int, str, str], ...]:
    """Describe ``b''.join(list(parameter))`` as native byte views.

    The list view carries authored row count while the join view carries the
    flattened bytes.  Both name the same authored iterable so a wrapper can
    materialize it exactly once and publish the two deterministic ABI views.
    """

    transforms = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type") or "").casefold() != "join":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        separator_id = parents.get("operand")
        materialized_id = parents.get("arg:0")
        if separator_id is None or materialized_id is None:
            continue
        separator = graph_obj.nodes.get(separator_id, {})
        separator_value = (separator.get("attributes") or {}).get(
            "value", separator.get("constant")
        )
        materialized = graph_obj.nodes.get(materialized_id, {})
        materialized_attributes = materialized.get("attributes") or {}
        if (
            separator_value != b""
            or materialized_attributes.get("producer_kind")
            != "aggregate_materialization"
            or materialized_attributes.get("aggregate_kind") != "list"
            or materialized_attributes.get("static_python_reference") != "list"
        ):
            continue
        sources = tuple(map(
            int, materialized_attributes.get("materialized_source_value_ids", ())
        ))
        if len(sources) != 1:
            continue
        source = graph_obj.nodes.get(sources[0], {})
        source_name = (source.get("attributes") or {}).get("binding_name")
        if source_name is None:
            continue
        transforms.extend((
            (
                int(materialized.get("value_id", materialized_id)),
                int(materialized.get("value_id", materialized_id)),
                str(source_name),
                "row_count",
            ),
            (
                int(data.get("value_id", node_id)),
                int(data.get("value_id", node_id)),
                str(source_name),
                "join_bytes",
            ),
        ))
    return tuple(transforms)


def _scalar_source_transforms(
    graph_obj: Any,
    sequence_transforms: Iterable[tuple[int, int, str, str]],
) -> tuple[tuple[int, str, str], ...]:
    """Describe scalar ABI projections such as source-backed ``len``.

    A linked call can legitimately consume a scalar whose producer is a
    structural sequence operation omitted from the numerical SSA region.
    Retaining only that scalar as an anonymous function argument leaves a
    native wrapper no exact way to construct it.  Record the authored source
    and projection at the compilation boundary instead.
    """

    transformed_values = {
        int(result_id): (
            str(source_name),
            (
                "utf8_length" if str(transform) == "utf8"
                else "materialized_length" if str(transform) == "row_count"
                else "sequence_length"
            ),
        )
        for result_id, _source_id, source_name, transform
        in sequence_transforms
    }
    records: list[tuple[int, str, str]] = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "len"
        ):
            continue
        arguments = tuple(
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        if len(arguments) != 1:
            continue
        argument = graph_obj.nodes[arguments[0]]
        argument_id = int(argument.get("value_id", arguments[0]))
        source = transformed_values.get(argument_id)
        if source is None:
            attributes = argument.get("attributes") or {}
            source_name = attributes.get("binding_name")
            if (
                attributes.get("binding_kind") == "parameter"
                and source_name is not None
            ):
                source = (str(source_name), "sequence_length")
            elif (
                attributes.get("producer_kind")
                == "aggregate_materialization"
            ):
                materialized_sources = tuple(map(
                    int, attributes.get("materialized_source_value_ids", ())
                ))
                if len(materialized_sources) == 1:
                    source_data = next((
                        candidate
                        for candidate_id, candidate in graph_obj.nodes(data=True)
                        if int(candidate.get("value_id", candidate_id))
                        == materialized_sources[0]
                    ), None)
                    source_attributes = (
                        {} if source_data is None
                        else source_data.get("attributes") or {}
                    )
                    source_name = source_attributes.get("binding_name")
                    if source_name is not None:
                        source = (
                            str(source_name), "materialized_length"
                        )
        if source is not None:
            records.append((
                int(data.get("value_id", node_id)), source[0], source[1]
            ))
    return tuple(records)


def _sequence_append_fill_ops(
    graph_obj: Any,
) -> tuple[
    tuple[int, int, int | float | bool | None, int, int], ...
]:
    """Return exact ``resident += literal_sequence * runtime_count`` rows."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(rhs.get("type") or rhs.get("op")) not in {"Mul", "Mult"}:
            continue
        operands = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        literal_id = operands.get("lhs")
        count_id = operands.get("rhs")
        if literal_id is None or count_id is None:
            continue
        literal_data = graph_obj.nodes.get(literal_id, {})
        literal = (literal_data.get("attributes") or {}).get(
            "value", literal_data.get("constant")
        )
        if isinstance(literal, (bytes, bytearray)) and len(literal) == 1:
            literal = int(literal[0])
        elif not isinstance(literal, (bool, int, float)) and literal is not None:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            literal,
            int(graph_obj.nodes[count_id].get("value_id", count_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_append_slice_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return exact ``resident += resident[lower:upper]`` mutations.

    Only an authored unit-stride slice with explicit lower and upper values is
    admitted here.  The SSA helper performs Python-compatible bound clipping;
    other slice forms remain in the ordinary lowering ledger.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}

    def _sequence_like(node: Mapping[str, Any]) -> bool:
        attributes = node.get("attributes") or {}
        if attributes.get("aggregate_kind") in sequence_kinds:
            return True
        # A structurally specialized sequence presents as its captured
        # initial value: a Constant whose payload is itself a sequence
        # object (re's ``out``/``tail`` arrive as Constant [] with
        # structural_specialization=True while their sequence descriptors
        # live on under the same value ids).
        return (
            str(node.get("type") or node.get("op")) == "Constant"
            and isinstance(
                attributes.get("value"), (list, bytes, bytearray)
            )
        )

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if str(rhs.get("type") or rhs.get("op")) != "Indexed":
            # Whole-sequence extend: ``resident += other_resident`` (re's
            # ``out += tail``). Same helper, bounds spanning the whole
            # source (the lowering clips into [0, source_length], so the
            # emitter passes 0 and a beyond-length constant). No extra
            # expression node exists to suppress, hence the None slice id.
            if _sequence_like(lhs) and _sequence_like(rhs):
                operations.append((
                    int(data.get("value_id", node_id)),
                    int(lhs.get("value_id", lhs_id)),
                    int(rhs.get("value_id", rhs_id)),
                    None,
                    None,
                    None,
                ))
            continue
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        indexed = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        source_id = indexed.get("base")
        slice_id = indexed.get("index")
        if source_id is None or slice_id is None:
            continue
        source = graph_obj.nodes.get(source_id, {})
        slice_data = graph_obj.nodes.get(slice_id, {})
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(slice_data.get("type") or slice_data.get("op")) != "Slice":
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (slice_data.get("parents") or ())
        }
        lower_id = bounds.get("lower")
        upper_id = bounds.get("upper")
        if lower_id is None or upper_id is None or "step" in bounds:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            int(source.get("value_id", source_id)),
            int(graph_obj.nodes[lower_id].get("value_id", lower_id)),
            int(graph_obj.nodes[upper_id].get("value_id", upper_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_bit_pack_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    """Recognize complete translate/reverse/fixed-bit-word materialization.

    This is the structural meaning of ``_mk_bitmap``: a 0/1 byte arena is
    packed little-endian into fixed-width words.  Recognition follows the AST
    dataflow (translate -> reverse slice -> fixed slice -> int(base=2) ->
    list comprehension), never the function name.
    """

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        result = graph_obj.nodes[node_id]
        if (result.get("attributes") or {}).get(
            "aggregate_kind"
        ) != "list":
            continue
        expression = result.get("expr_obj")
        if not isinstance(expression, ast.ListComp) or len(expression.generators) != 1:
            continue
        element_parents = tuple(
            int(parent) for parent, role in (result.get("parents") or ())
            if str(role) == "elt" and parent in graph_obj
        )
        if len(element_parents) != 1:
            continue
        conversion = graph_obj.nodes[element_parents[0]]
        conversion_expression = conversion.get("expr_obj")
        if not (
            isinstance(conversion_expression, ast.Call)
            and len(conversion_expression.args) == 2
            and isinstance(conversion_expression.args[1], ast.Constant)
            and conversion_expression.args[1].value == 2
        ):
            continue
        conversion_roles = {
            str(role): int(parent)
            for parent, role in (conversion.get("parents") or ())
        }
        slice_value_id = conversion_roles.get("arg:0")
        if slice_value_id is None or slice_value_id not in graph_obj:
            continue
        sliced = graph_obj.nodes[slice_value_id]
        sliced_roles = {
            str(role): int(parent)
            for parent, role in (sliced.get("parents") or ())
        }
        reversed_id = sliced_roles.get("base")
        inner_slice_id = sliced_roles.get("index")
        if reversed_id is None or inner_slice_id is None:
            continue
        reversed_value = graph_obj.nodes[reversed_id]
        reversed_roles = {
            str(role): int(parent)
            for parent, role in (reversed_value.get("parents") or ())
        }
        translated_id = reversed_roles.get("base")
        reverse_slice_id = reversed_roles.get("index")
        if translated_id is None or reverse_slice_id is None:
            continue
        translated = graph_obj.nodes[translated_id]
        translated_roles = {
            str(role): int(parent)
            for parent, role in (translated.get("parents") or ())
        }
        source_id = translated_roles.get("operand")
        if source_id is None:
            continue
        source = graph_obj.nodes[source_id]
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in {"bytes", "bytearray", "list"}:
            continue
        reverse_slice = graph_obj.nodes[reverse_slice_id]
        reverse_step = tuple(
            int(parent) for parent, role in (reverse_slice.get("parents") or ())
            if str(role) == "step"
        )
        if len(reverse_step) != 1:
            continue
        step_expression = graph_obj.nodes[reverse_step[0]].get("expr_obj")
        if not (
            isinstance(step_expression, ast.UnaryOp)
            and isinstance(step_expression.op, ast.USub)
            and isinstance(step_expression.operand, ast.Constant)
            and step_expression.operand.value == 1
        ):
            continue
        inner_slice = graph_obj.nodes[inner_slice_id]
        lower_ids = tuple(
            int(parent) for parent, role in (inner_slice.get("parents") or ())
            if str(role) == "lower"
        )
        if len(lower_ids) != 1:
            continue
        lower = graph_obj.nodes[lower_ids[0]]
        lower_roles = {
            str(role): int(parent)
            for parent, role in (lower.get("parents") or ())
        }
        width_node_id = lower_roles.get("rhs")
        if width_node_id is None:
            continue
        width_id = int(graph_obj.nodes[width_node_id].get(
            "value_id", width_node_id
        ))
        consumed = tuple(dict.fromkeys((
            int(translated.get("value_id", translated_id)),
            int(reversed_value.get("value_id", reversed_id)),
            int(lower.get("value_id", lower_ids[0])),
            int(sliced.get("value_id", slice_value_id)),
            int(result.get("value_id", node_id)),
        )))
        operations.append((
            int(result.get("value_id", node_id)),
            int(source.get("value_id", source_id)),
            width_id,
            consumed,
        ))
    return tuple(operations)


def _sequence_prepend_concat_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Recognize ``[scalar] + sequence`` consumed by ``base[0:0] = ...``."""

    sequence_kinds = {"list", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        store = graph_obj.nodes[node_id]
        if str(store.get("type") or store.get("op")) != "IndexedStore":
            continue
        roles = {
            str(role): int(parent)
            for parent, role in (store.get("parents") or ())
        }
        base_id = roles.get("base")
        index_id = roles.get("index")
        value_id = roles.get("value")
        if None in (base_id, index_id, value_id):
            continue
        base = graph_obj.nodes[base_id]
        index = graph_obj.nodes[index_id]
        concatenation = graph_obj.nodes[value_id]
        resident_id = resident_by_value.get(
            int(base.get("value_id", base_id))
        )
        if resident_id is None:
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (index.get("parents") or ())
        }
        if set(bounds) != {"lower", "upper"}:
            continue
        bound_values = []
        for bound_id in bounds.values():
            bound_data = graph_obj.nodes[bound_id]
            bound_values.append((bound_data.get("attributes") or {}).get(
                "value", bound_data.get("constant")
            ))
        if bound_values != [0, 0]:
            continue
        if str(concatenation.get("type") or concatenation.get("op")) != "Add":
            continue
        concat_roles = {
            str(role): int(parent)
            for parent, role in (concatenation.get("parents") or ())
        }
        singleton_id = concat_roles.get("lhs")
        tail_id = concat_roles.get("rhs")
        if singleton_id is None or tail_id is None:
            continue
        singleton = graph_obj.nodes[singleton_id]
        leaves = tuple((singleton.get("attributes") or {}).get(
            "aggregate_leaf_value_ids", ()
        ))
        if (
            (singleton.get("attributes") or {}).get("aggregate_kind") != "list"
            or len(leaves) != 1
        ):
            continue
        operations.append((
            int(store.get("value_id", node_id)),
            int(resident_id),
            int(leaves[0]),
            int(concatenation.get("value_id", value_id)),
            int(graph_obj.nodes[tail_id].get("value_id", tail_id)),
        ))
    return tuple(operations)


def _sequence_prepend_packed_call_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    """Correlate prefix splice with the pursued byte-packing call edge."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for (
        store_result_id, destination_id, prefix_id, concat_result_id,
        tail_result_id,
    ) in _sequence_prepend_concat_ops(graph_obj):
        tail_node = next((
            (node_id, data)
            for node_id, data in graph_obj.nodes(data=True)
            if int(data.get("value_id", node_id)) == int(tail_result_id)
        ), None)
        if tail_node is None:
            continue
        _tail_node_id, tail = tail_node
        attributes = tail.get("attributes") or {}
        if attributes.get("callee_ref") is None:
            continue
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (tail.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        source_id = resident_by_value.get(int(arguments[0]))
        if source_id is None:
            continue
        operations.append((
            int(store_result_id), int(destination_id), int(prefix_id),
            int(source_id), int(concat_result_id), int(tail_result_id),
            int(_tail_node_id),
        ))
    return tuple(dict.fromkeys(operations))


def _sequence_inplace_bit_pack_call_ops(
    graph: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return pursued calls whose callee is a structural bit-pack program."""

    graph_obj = graph.G
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return ()
    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        call = graph_obj.nodes[node_id]
        attributes = call.get("attributes") or {}
        reference = attributes.get("callee_ref")
        if reference is None:
            continue
        try:
            callee = function_table.entry(int(reference)).graph
        except (KeyError, TypeError, ValueError):
            continue
        if callee is None:
            continue
        contracts = _sequence_bit_pack_ops(callee.G)
        if len(contracts) != 1:
            continue
        _callee_destination, _callee_source, _width_id, _consumed = contracts[0]
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (call.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        resident_id = resident_by_value.get(int(arguments[0]))
        if resident_id is None:
            continue
        entry = function_table.entry(int(reference))
        callable_object = getattr(entry, "python_callable", None)
        if callable_object is None and "." in str(entry.qualified_name):
            parts = str(entry.qualified_name).split(".")
            for split in range(len(parts) - 1, 0, -1):
                try:
                    candidate = importlib.import_module(".".join(parts[:split]))
                except ImportError:
                    continue
                try:
                    for attribute in parts[split:]:
                        candidate = getattr(candidate, attribute)
                except AttributeError:
                    continue
                callable_object = candidate
                break
        if callable_object is None:
            continue
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            continue
        parameters = tuple(signature.parameters.values())
        if len(parameters) < 2:
            continue
        width_default = parameters[1].default
        if not isinstance(width_default, int) or width_default <= 0:
            continue
        operations.append((
            int(call.get("value_id", node_id)),
            int(resident_id),
            int(width_default),
            int(reference),
            int(node_id),
        ))
    return tuple(dict.fromkeys(operations))


def _nested_row_projection_ops(
    graph_obj: Any, control: Any,
) -> tuple[tuple[int, int, int, str], ...]:
    """Find fixed-column reads from a projected/destructured loop row."""

    aliases_by_value: dict[int, frozenset[int]] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        values = frozenset(map(int, history))
        for value_id in values:
            aliases_by_value[int(value_id)] = values
    operations = []
    for _iterable, target_id, induction, _projection in (
        getattr(control, "projected_iterable_bindings", ())
    ):
        aliases = aliases_by_value.get(
            int(target_id), frozenset((int(target_id),))
        )
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("type")) not in {"Indexed", "indexed"}:
                continue
            roles = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            base = roles.get("base")
            index = roles.get("index")
            if base not in aliases or index is None or index not in graph_obj:
                continue
            index_data = graph_obj.nodes[index]
            expression = index_data.get("expr_obj")
            literal = (
                expression.value
                if isinstance(expression, ast.Constant)
                else (index_data.get("attributes") or {}).get(
                    "value", index_data.get("value")
                )
            )
            if not isinstance(literal, int) or isinstance(literal, bool):
                continue
            operations.append((
                int(target_id), int(literal),
                int(data.get("value_id", node_id)), str(induction),
            ))
    return tuple(dict.fromkeys(operations))


def _rewrite_optional_row_handle_none_predicate(
    expression: Any, handle_ids: Iterable[int],
) -> Any:
    """Compare optional record row handles with their physical -1 sentinel."""

    if expression is None:
        return None
    handles = set(map(int, handle_ids))
    operands = tuple(
        _rewrite_optional_row_handle_none_predicate(operand, handles)
        for operand in expression.operands
    )
    rewritten = replace(expression, operands=operands)
    if rewritten.op not in {"eq", "ne"} or len(operands) != 2:
        return rewritten
    for handle_index, none_index in ((0, 1), (1, 0)):
        handle = operands[handle_index]
        none = operands[none_index]
        if (
            handle.value_id is None
            or int(handle.value_id) not in handles
            or none.op != "const"
            or none.literal is not None
        ):
            continue
        replaced = list(operands)
        replaced[none_index] = replace(none, literal=-1)
        return replace(rewritten, operands=tuple(replaced))
    return rewritten


def _record_sequence_projection_bindings(
    graph_obj: Any, control: Any,
) -> tuple[
    Any,
    tuple[tuple[int, int, str, object], ...],
    tuple[tuple[int, str, str], ...],
]:
    """Project fields of ``Sequence[Record]`` rows at their loop binding.

    The authored parameter is a columnar resident sequence. A lexical loop
    target names the current row, while each ``target.field`` names one exact
    column load at the same induction. This preserves the ordinary intuitive
    source and gives downstream regions/calls real in-loop producers instead
    of promoting field values to anonymous public inputs.
    """

    records = dict(
        graph_obj.graph.get("parameter_sequence_record_abi") or {}
    )
    identities = graph_obj.graph.get("identity_table") or {}
    aliases_by_value: dict[int, frozenset[int]] = {}
    for history in identities.values():
        aliases = frozenset(map(int, history))
        for value_id in aliases:
            aliases_by_value[int(value_id)] = aliases
    parameter_by_sequence: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for parameter_name, record in records.items():
        for value_id in identities.get(str(parameter_name), ()):
            parameter_by_sequence[int(value_id)] = (
                str(parameter_name), record,
            )
    row_loop_bindings = tuple(dict.fromkeys((
        *tuple(getattr(control, "iterable_bindings", ())),
        *tuple(
            (int(iterable_id), int(target_id), str(induction))
            for iterable_id, target_id, induction, projection
            in getattr(control, "projected_iterable_bindings", ())
            if projection is None
        ),
    )))
    bindings: list[tuple[int, int, str, object]] = []
    fields: list[tuple[int, str, str]] = []
    direct_rows: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for iterable_id, target_id, _induction in row_loop_bindings:
        selected = parameter_by_sequence.get(int(iterable_id))
        if selected is None:
            iterable_aliases = aliases_by_value.get(
                int(iterable_id), frozenset((int(iterable_id),))
            )
            selected = next((
                record for candidate, record in parameter_by_sequence.items()
                if candidate in iterable_aliases
            ), None)
        if selected is not None:
            _parameter_name, record = selected
            direct_rows[int(target_id)] = (int(iterable_id), record)

    # A filtered comprehension over a record sequence stores integer row
    # handles into its derived resident sequence.  Recover that relationship
    # from the exact mutation argument identity; no Python row object is
    # reconstructed and no name spelling participates in the correlation.
    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceMutationBlock, SequenceQueryBlock, WhileBlock,
    )

    mutations = []
    queries = []

    def gather_mutations(block):
        if isinstance(block, SequenceMutationBlock):
            mutations.append(block.mutation)
        elif isinstance(block, SequenceQueryBlock):
            queries.append(block)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather_mutations(child)
        elif isinstance(block, ConditionalBlock):
            gather_mutations(block.body)
            if block.orelse is not None:
                gather_mutations(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            mutations.extend(block.sequence_mutations)
            if isinstance(block, WhileBlock):
                gather_mutations(block.condition)
            gather_mutations(block.body)
        elif isinstance(block, CallBlock):
            gather_mutations(block.callee)

    gather_mutations(control.root)
    derived_sequences: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for mutation in mutations:
        if str(mutation.operator) not in {"append", "add"}:
            continue
        arguments = set(map(int, mutation.argument_value_ids))
        for target_id, origin in direct_rows.items():
            target_aliases = aliases_by_value.get(
                int(target_id), frozenset((int(target_id),))
            )
            if arguments.intersection(target_aliases):
                derived_sequences[int(mutation.sequence_value_id)] = origin
                effect_id = int(mutation.effect_node_id)
                if effect_id in graph_obj:
                    # Hierarchical composition namespaces resident storage,
                    # while a later authored loop still names the source
                    # materializer result.  The mutation's retained effect
                    # node is their exact correlation receipt.
                    derived_sequences[int(graph_obj.nodes[effect_id].get(
                        "value_id", effect_id
                    ))] = origin

    optional_projections: dict[
        int, tuple[tuple[int, int, int, int, str], ...]
    ] = {}
    # ``mapping.pop(key, None)`` has the same physical optional-handle ABI as
    # ``first_or_default``: its lookup result is -1 when absent.  Register the
    # exact effect identity even when no parameter-record projection exists so
    # an authored ``result is None`` predicate is rewritten consistently.
    for mutation in mutations:
        if (
            str(mutation.operator) == "pop"
            and mutation.argument_kind == "mapping_pop_default_none"
        ):
            optional_projections[int(mutation.effect_node_id)] = ()
    optional_query_calls: set[int] = set()
    for query in queries:
        if str(query.operation) != "first_or_default":
            continue
        origin = derived_sequences.get(int(query.sequence_value_id))
        if origin is None:
            continue
        origin_sequence_id, record = origin
        receivers = {
            int(query.result_value_id), *map(int, query.result_alias_ids)
        }
        physical_columns = {}
        physical_column = 0
        for field_name, receipt in dict(record.get("fields") or {}).items():
            if bool(receipt.get("optional")):
                physical_column += 1
            physical_columns[str(field_name)] = (
                physical_column, str(receipt.get("dtype") or "unknown")
            )
            physical_column += 1
        projected = []
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in physical_columns or not any(
                int(graph_obj.nodes[parent].get("value_id", parent)) in receivers
                and str(role) in {"value", "object", "base", "receiver"}
                for parent, role in (data.get("parents") or ())
                if parent in graph_obj
            ):
                continue
            column, dtype = physical_columns[attribute]
            result_id = int(data.get("value_id", node_id))
            projected.append((
                int(origin_sequence_id), int(query.result_value_id),
                result_id, int(column), str(dtype),
            ))
            fields.append((result_id, attribute, str(dtype)))
        if projected:
            optional_projections[int(query.result_value_id)] = tuple(projected)
            if query.source_call_node_id is not None:
                optional_query_calls.add(int(query.source_call_node_id))

    for iterable_id, target_id, induction in row_loop_bindings:
        selected = parameter_by_sequence.get(int(iterable_id))
        if selected is None:
            iterable_aliases = aliases_by_value.get(
                int(iterable_id), frozenset((int(iterable_id),))
            )
            selected = next((
                record for candidate, record in parameter_by_sequence.items()
                if candidate in iterable_aliases
            ), None)
        derived = False
        if selected is None:
            selected = derived_sequences.get(int(iterable_id))
            derived = selected is not None
        if selected is None:
            continue
        if derived:
            origin_sequence_id, record = selected
        else:
            _parameter_name, record = selected
            origin_sequence_id = int(iterable_id)
        field_contracts = tuple(dict(record.get("fields") or {}).items())
        field_columns = {}
        physical_column = 0
        for field_name, field_receipt in field_contracts:
            if bool(field_receipt.get("optional")):
                # Presence precedes value.  Attribute access selects the value
                # column; an explicit ``is None`` field test can consume the
                # adjacent presence column without overloading its payload.
                physical_column += 1
            field_columns[str(field_name)] = physical_column
            physical_column += 1
        # Loop targets with the same authored spelling are distinct SSA
        # identities (two separate ``for row in ...`` bindings are the common
        # case).  The graph's GetAttr receiver already carries the exact target
        # value; consulting the scope-free name history here conflates those
        # loops and places a later field load in the earlier coordinator.
        target_aliases = frozenset((int(target_id),))
        if not derived:
            # The conceptual row itself is a stable integer handle.  Field
            # values are loaded from the source columns below; retaining a
            # second flat-element load for the row would confuse column zero
            # with the record identity when a comprehension republishes it.
            bindings.append((
                int(origin_sequence_id), int(target_id), str(induction),
                "induction",
            ))
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in field_columns or not any(
                int(graph_obj.nodes[parent].get("value_id", parent))
                in target_aliases
                and str(role) in {"value", "object", "base", "receiver"}
                for parent, role in (data.get("parents") or ())
                if parent in graph_obj
            ):
                continue
            result_id = int(data.get("value_id", node_id))
            receipt = dict(next(
                field_receipt
                for field_name, field_receipt in field_contracts
                if str(field_name) == attribute
            ))
            bindings.append((
                int(origin_sequence_id), result_id, str(induction),
                (
                    (
                        "column_at_value", int(field_columns[attribute]),
                        int(target_id),
                    )
                    if derived else int(field_columns[attribute])
                ),
            ))
            fields.append((
                result_id, attribute, str(receipt.get("dtype") or "unknown")
            ))
    def expression_values(expression):
        if expression is None:
            return frozenset()
        return frozenset((
            *((int(expression.value_id),) if expression.value_id is not None else ()),
            *(value for operand in expression.operands for value in expression_values(operand)),
        ))

    def attach_optional(block):
        if isinstance(block, SequenceQueryBlock):
            return replace(
                block,
                row_handle=(
                    block.source_call_node_id is not None
                    and int(block.source_call_node_id) in optional_query_calls
                ),
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block,
                blocks=tuple(attach_optional(child) for child in block.blocks),
            )
        if isinstance(block, ConditionalBlock):
            body = attach_optional(block.body)
            orelse = (
                None if block.orelse is None
                else attach_optional(block.orelse)
            )
            predicate_expression = (
                _rewrite_optional_row_handle_none_predicate(
                    block.predicate_expression, optional_projections
                )
            )
            values = expression_values(predicate_expression)
            entry = tuple(
                projection
                for handle_id, projections in optional_projections.items()
                if int(handle_id) in values
                and predicate_expression is not None
                and predicate_expression.op == "ne"
                for projection in projections
            )
            return replace(
                block, body=body, orelse=orelse,
                predicate_expression=predicate_expression,
                entry_record_projections=tuple(dict.fromkeys((
                    *block.entry_record_projections, *entry,
                ))),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=attach_optional(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=attach_optional(block.condition),
                body=attach_optional(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=attach_optional(block.callee))
        return block

    control = replace(control, root=attach_optional(control.root))
    return (
        control,
        tuple(dict.fromkeys(bindings)),
        tuple(dict.fromkeys(fields)),
    )


def _sequence_row_operations(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> tuple[tuple[Any, ...], ...]:
    """Recover positional reads/writes of fixed-width sequence rows."""

    widths = {
        int(sequence_id): int(column_count)
        for sequence_id, _policy, column_count, _writable
        in sequence_declarations
        if int(column_count) > 1
    }
    if not widths:
        return ()
    identities = graph_obj.graph.get("identity_table") or {}
    resident_by_value = {
        int(value_id): int(sequence_id)
        for sequence_id in widths
        for history in identities.values()
        if int(sequence_id) in set(map(int, history))
        for value_id in history
    }
    resident_by_value.update({sequence_id: sequence_id for sequence_id in widths})
    changed = True
    aliases = _loop_carried_storage_aliases(graph_obj)
    while changed:
        changed = False
        for alias_id, source_id in aliases.items():
            resident = resident_by_value.get(int(source_id))
            if resident is not None and resident_by_value.get(int(alias_id)) != resident:
                resident_by_value[int(alias_id)] = resident
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            residents = {
                resident_by_value[int(graph_obj.nodes[parent].get(
                    "value_id", parent
                ))]
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                and int(graph_obj.nodes[parent].get(
                    "value_id", parent
                )) in resident_by_value
            }
            if len(residents) == 1:
                value_id = int(data.get("value_id", node_id))
                resident = next(iter(residents))
                if resident_by_value.get(value_id) != resident:
                    resident_by_value[value_id] = resident
                    changed = True

    def roles(data: Mapping[str, Any]) -> dict[str, int]:
        return {
            str(role): int(parent)
            for parent, role in data.get("parents") or ()
            if parent in graph_obj
        }

    def value_id(node_id: int) -> int:
        return int(graph_obj.nodes[node_id].get("value_id", node_id))

    def integer_literal(node_id: int) -> int | None:
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        literal = (
            expression.value if isinstance(expression, ast.Constant)
            else (data.get("attributes") or {}).get(
                "value", data.get("constant")
            )
        )
        return (
            int(literal)
            if isinstance(literal, int) and not isinstance(literal, bool)
            else None
        )

    operations: list[tuple[Any, ...]] = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        kind = str(data.get("op") or data.get("type") or "").casefold()
        node_roles = roles(data)
        if kind == "indexed":
            outer_id = node_roles.get("base")
            column_node = node_roles.get("index")
            if outer_id is None or column_node is None:
                continue
            outer = graph_obj.nodes[outer_id]
            if str(outer.get("op") or outer.get("type") or "").casefold() != "indexed":
                continue
            outer_roles = roles(outer)
            base_node = outer_roles.get("base")
            row_index_node = outer_roles.get("index")
            if base_node is None or row_index_node is None:
                continue
            sequence_id = resident_by_value.get(value_id(base_node))
            column = integer_literal(column_node)
            if (
                sequence_id is None or column is None or column < 0
                or column >= widths[sequence_id]
            ):
                continue
            operations.append((
                "load", value_id(node_id), int(sequence_id),
                value_id(row_index_node), integer_literal(row_index_node),
                int(column), value_id(outer_id),
            ))
        elif kind == "indexedstore":
            base_node = node_roles.get("base")
            row_index_node = node_roles.get("index")
            row_value_node = node_roles.get("value")
            if base_node is None or row_index_node is None or row_value_node is None:
                continue
            sequence_id = resident_by_value.get(value_id(base_node))
            row_attributes = graph_obj.nodes[row_value_node].get("attributes") or {}
            leaves = tuple(map(
                int, row_attributes.get("aggregate_leaf_value_ids", ())
            ))
            if (
                sequence_id is None
                or row_attributes.get("aggregate_kind") != "tuple"
                or len(leaves) != widths[sequence_id]
            ):
                continue
            operations.append((
                "store", value_id(node_id), int(sequence_id),
                value_id(row_index_node), integer_literal(row_index_node),
                leaves, value_id(row_value_node),
            ))
    return tuple(operations)


def _sequence_column_dtype_contracts(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> dict[int, tuple[str, ...]]:
    """Recover fixed-row column dtypes from authored annotations.

    A declaration such as ``runs: list[tuple[int, int]]`` is a physical
    contract for both resident columns, not merely evidence that the row has
    width two.  Keep that contract beside the compile artifact so row loads,
    stores, and append helpers share one type before target inference runs.
    """

    declared = {
        int(sequence_id): int(column_count)
        for sequence_id, _policy, column_count, _writable
        in sequence_declarations
    }
    if not declared:
        return {}
    identities = graph_obj.graph.get("identity_table") or {}
    scalar_dtypes = {
        "bool": "bool",
        "int": "int64",
        "float": "float64",
    }
    contracts: dict[int, tuple[str, ...]] = {}
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        contract = _authored_sequence_annotation_contract(annotation)
        if contract is None:
            continue
        _policy, _column_count, _writable, dtypes = contract
        sequence_id = next((
            int(value_id)
            for value_id in identities.get(str(parameter_name), ())
            if int(value_id) in declared
        ), None)
        if (
            sequence_id is not None
            and len(dtypes) == declared[int(sequence_id)]
        ):
            contracts[int(sequence_id)] = tuple(map(str, dtypes))
    for parameter_name, record in dict(
        graph_obj.graph.get("parameter_sequence_record_abi") or {}
    ).items():
        dtypes = tuple(
            dtype
            for field in dict(record.get("fields") or {}).values()
            for dtype in (
                ("bool", str(field.get("dtype") or "unknown"))
                if bool(field.get("optional"))
                else (str(field.get("dtype") or "unknown"),)
            )
        )
        if not dtypes:
            continue
        sequence_id = next((
            int(value_id)
            for value_id in identities.get(str(parameter_name), ())
            if int(value_id) in declared
        ), None)
        if (
            sequence_id is not None
            and len(dtypes) == declared[int(sequence_id)]
        ):
            contracts[int(sequence_id)] = dtypes
    for binding_name, annotation in dict(
        graph_obj.graph.get("type_annotations") or {}
    ).items():
        if not isinstance(annotation, str) or not annotation.strip():
            continue
        try:
            outer = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(outer, ast.Subscript):
            continue
        row = outer.slice
        if not isinstance(row, ast.Subscript):
            continue
        tuple_name = row.value
        if not (
            isinstance(tuple_name, ast.Name) and tuple_name.id in {"tuple", "Tuple"}
            or isinstance(tuple_name, ast.Attribute)
            and tuple_name.attr in {"tuple", "Tuple"}
        ):
            continue
        columns = (
            tuple(row.slice.elts)
            if isinstance(row.slice, ast.Tuple)
            else (row.slice,)
        )
        dtypes: list[str] = []
        for column in columns:
            spelling = (
                column.id if isinstance(column, ast.Name)
                else column.attr if isinstance(column, ast.Attribute)
                else ""
            )
            dtype = scalar_dtypes.get(str(spelling))
            if dtype is None:
                dtypes = []
                break
            dtypes.append(dtype)
        if not dtypes:
            continue
        history = tuple(map(int, identities.get(str(binding_name), ())))
        sequence_id = next(
            (value_id for value_id in history if value_id in declared), None
        )
        if sequence_id is None or len(dtypes) != declared[sequence_id]:
            continue
        contracts[int(sequence_id)] = tuple(dtypes)
    def literal_dtype(value: Any) -> str | None:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int64"
        if isinstance(value, float):
            return "float64"
        if isinstance(value, str):
            return "int64"
        return None

    for node_id, data in graph_obj.nodes(data=True):
        value_id = int(data.get("value_id", node_id))
        if value_id not in declared:
            continue
        items = (data.get("attributes") or {}).get(
            "compile_time_mapping_items"
        )
        if items is None:
            continue
        rows = tuple(items)
        if not rows:
            continue
        key_dtypes = {literal_dtype(key) for key, _item in rows}
        value_dtypes = {literal_dtype(item) for _key, item in rows}
        if None not in key_dtypes | value_dtypes and (
            len(key_dtypes) == len(value_dtypes) == 1
        ):
            contracts[value_id] = (
                next(iter(key_dtypes)), next(iter(value_dtypes))
            )
    return contracts


def _authored_source_sequence_ids(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> tuple[int, ...]:
    """Identify resident arenas whose initial contents come from the caller."""

    declared = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    if not declared:
        return ()
    identity = graph_obj.graph.get("identity_table") or {}
    source_ids = {
        int(value_id)
        for parameter_name in graph_obj.graph.get("function_parameters") or ()
        for value_id in identity.get(str(parameter_name), ())
        if int(value_id) in declared
    }
    self_fields = {
        str(field_name)
        for field_name, receipt in dict(
            dict(
                (graph_obj.graph.get("parameter_record_abi") or {}).get(
                    "self"
                ) or {}
            ).get("fields") or {}
        ).items()
        if str(receipt.get("storage") or "") == "span"
    }
    for node_id, data in graph_obj.nodes(data=True):
        value_id = int(data.get("value_id", node_id))
        if value_id not in declared:
            continue
        attributes = data.get("attributes") or {}
        if str(attributes.get("binding_kind") or "") in {
            "parameter", "closure", "external",
        }:
            source_ids.add(value_id)
            continue
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if (
            operation == "getattr"
            and str(attributes.get("attribute") or "") in self_fields
        ):
            source_ids.add(value_id)
    return tuple(sorted(source_ids))


def _linked_authored_parameter_aliases(
    caller: Any,
    callee: Any,
    caller_graph: Any,
    callee_graph: Any,
    argument_bindings: Any,
    caller_record_table: Any = None,
    callee_record_table: Any = None,
) -> dict[str, str]:
    """Map a linked callee formal onto an outer authored formal exactly.

    Method record fields retain their local spelling (usually ``self``).
    When the exact PlanCall binding says an authored caller parameter such as
    ``body`` supplies that receiver, the public ABI must expose
    ``body.field`` rather than ``self.field``.  Only deterministic formal
    identities participate; later same-spelling SSA versions are not aliases.
    """

    def identities(
        function: Any, graph: Any, record_table: Any,
    ) -> dict[int, str]:
        metadata = dict(getattr(function, "metadata", {}) or {})
        found = {
            int(value_id): str(name)
            for name, value_id in metadata.get("parameter_names", ())
        }
        graph_metadata = (
            dict(getattr(graph, "graph", {}) or {})
            if graph is not None else {}
        )
        identity_table = dict(graph_metadata.get("identity_table") or {})
        parameter_roots = {
            *map(str, dict(metadata.get("parameter_record_abi") or {})),
            *map(str, dict(metadata.get("parameter_value_abi") or {})),
            *map(str, graph_metadata.get("function_parameters", ()) or ()),
        }
        for name in sorted(parameter_roots):
            history = tuple(identity_table.get(name, ()))
            if history:
                found.setdefault(int(history[0]), name)
        # Method shells can remove the shapeless aggregate formal after all
        # fields are projected. The record descriptor still owns its exact
        # deterministic aggregate identity even when the final shell no
        # longer carries the ProcessGraph identity catalogue.
        records = dict(getattr(record_table, "records", {}) or {})
        for name, receipt in dict(
            metadata.get("parameter_record_abi") or {}
        ).items():
            identity = str(dict(receipt or {}).get("identity") or "")
            candidates = [
                int(record_id)
                for record_id, descriptor in records.items()
                if str(getattr(descriptor, "identity", "")) == identity
            ]
            if len(candidates) == 1:
                found.setdefault(candidates[0], str(name))
        return found

    caller_names = identities(caller, caller_graph, caller_record_table)
    callee_names = identities(callee, callee_graph, callee_record_table)
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for caller_id, callee_id in argument_bindings:
        caller_name = caller_names.get(int(caller_id))
        callee_name = callee_names.get(int(callee_id))
        if caller_name is None or callee_name is None:
            continue
        previous = aliases.setdefault(callee_name, caller_name)
        if previous != caller_name:
            ambiguous.add(callee_name)
    for name in ambiguous:
        aliases.pop(name, None)
    return aliases


def _bind_sequence_storage_members(
    storage_bindings: dict[int, int],
    callee_sequence: Any,
    caller_sequence: Any,
) -> bool:
    """Bind every physical member of one exact sequence argument."""

    if (
        callee_sequence is None
        or caller_sequence is None
        or len(callee_sequence.column_value_ids)
        != len(caller_sequence.column_value_ids)
    ):
        return False
    storage_bindings.update(zip(
        map(int, callee_sequence.column_value_ids),
        map(int, caller_sequence.column_value_ids),
    ))
    storage_bindings[int(callee_sequence.length_address_id)] = int(
        caller_sequence.length_address_id
    )
    storage_bindings[int(callee_sequence.capacity_value_id)] = int(
        caller_sequence.capacity_value_id
    )
    for attribute in ("status_address_id", "live_flags_value_id"):
        callee_member = getattr(callee_sequence, attribute, None)
        caller_member = getattr(caller_sequence, attribute, None)
        if callee_member is not None and caller_member is not None:
            storage_bindings[int(callee_member)] = int(caller_member)
    return True


def _sequence_length_values(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
    aliases: Mapping[int, int] | Iterable[tuple[int, int]] = (),
) -> dict[int, int]:
    """Map authored ``len(sequence)`` results to resident descriptors."""

    declared = {
        int(sequence_id)
        for sequence_id, _policy, _columns, _writable
        in sequence_declarations
    }
    resident = {value_id: value_id for value_id in declared}
    resident.update({
        int(alias): int(source)
        for alias, source in dict(aliases).items()
    })
    changed = True
    while changed:
        changed = False
        for alias, source in tuple(resident.items()):
            target = resident.get(int(source))
            if target is not None and resident.get(int(alias)) != target:
                resident[int(alias)] = int(target)
                changed = True
    values = {}
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "len"
        ):
            continue
        arguments = tuple(
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role).startswith("arg:") and parent in graph_obj
        )
        if len(arguments) != 1:
            continue
        argument_id = int(graph_obj.nodes[arguments[0]].get(
            "value_id", arguments[0]
        ))
        sequence_id = resident.get(argument_id)
        if sequence_id in declared:
            values[int(data.get("value_id", node_id))] = int(sequence_id)
    return values


def _class_surface_ssa_program(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Lower every planned method of a whole object to one reusable SSA unit.

    This is the whole-object emission path and it performs NO numeric
    projection.  Each method lowers its own control program plus the operator
    regions the planner already carved out -- straight through
    ``lower_control_sections_to_ssa`` -- so a method with no numeric region (a
    void constructor) and a method with one (a ``mul``) lower the same way, and
    neither builds or validates a ``FusedProgram``.  Every method becomes its
    own linkable export; nothing is folded into a single entry and nothing is
    pruned.

    This boundary deliberately precedes Fortran emission so the fully lowered
    whole-object SSA can be inspected and verified before target emission.
    """

    from ..transmogrifier.ssa import (
        IRModule, SSAMachineControlTable, SSAMachineIndirectTable,
    )
    from .glsl_deployment_strategy import _walk_planned_shells
    from .precompile_to_ssa import (
        SSALoweringShortfall,
        lower_control_sections_to_ssa,
        resolve_sequence_schemas,
    )
    from .string_table import StringTable

    # One table for the whole object: every method's string constants tokenize
    # into it, and it persists token -> word for reverse lookup.
    string_table = StringTable()

    all_functions: dict[str, Any] = {}
    all_tensor_tables: dict[str, Any] = {}
    all_sequence_tables: dict[str, Any] = {}
    all_record_tables: dict[str, Any] = {}
    all_reference_tables: dict[str, Any] = {}
    module_metadata: dict[str, Any] = {}
    machine_control_links: list[Any] = []
    machine_indirect_links: list[Any] = []
    pending_call_records: list[tuple[str, Any, Any, Any, Any]] = []

    def dependency_closure(graph: Any, seeds: Any) -> set[int]:
        """Follow the ProcessGraph's already-authored value dependencies."""

        retained = set(map(int, seeds))
        stack = list(retained)
        while stack:
            value_id = stack.pop()
            data = graph.nodes.get(int(value_id), {})
            for parent, role in data.get("parents") or ():
                parent = int(parent)
                if str(role) == "callee" or parent in retained:
                    continue
                retained.add(parent)
                stack.append(parent)
        return retained
    class_table = None
    source_function_table = getattr(
        getattr(compilation.deployment, "process_graph", None),
        "function_table",
        None,
    )
    deployment_graph = getattr(
        getattr(compilation.deployment, "process_graph", None), "G", None
    )
    program_abi = (
        {}
        if deployment_graph is None
        else dict(deployment_graph.graph.get("program_abi") or {})
    )
    function_symbols: dict[int, str] = {}
    shell_symbols: dict[int, str] = {}
    section_outputs: dict[str, tuple[Any, ...]] = {}
    export_symbols: list[str] = []
    lowering_failures: list[tuple[str, Any]] = []
    discovered_planned_shells = tuple(_walk_planned_shells(
        compilation.deployment,
        include_function_registry=not bool(getattr(
            compilation.deployment, "runtime_closure_only", False
        )),
    ))
    planned_shells = tuple(
        shell
        for shell in discovered_planned_shells
        for planned_graph in (
            getattr(getattr(shell, "process_graph", None), "G", None),
        )
        for function_reference in (
            None if planned_graph is None
            else planned_graph.graph.get("function_ref"),
        )
        if not (
            function_reference is not None
            and source_function_table is not None
            and source_function_table.entry(
                int(function_reference)
            ).metadata.get("host_ssa_module") is not None
            and bool(source_function_table.entry(
                int(function_reference)
            ).metadata.get("host_repository_ssa_complete", False))
        )
    )
    source_name_references: dict[str, set[int]] = {}
    for planned_shell in planned_shells:
        planned_graph = getattr(
            getattr(planned_shell, "process_graph", None), "G", None
        )
        if planned_graph is None:
            continue
        planned_name = planned_graph.graph.get("function_name")
        planned_reference = planned_graph.graph.get("function_ref")
        if planned_name is not None and planned_reference is not None:
            source_name_references.setdefault(
                str(planned_name), set()
            ).add(int(planned_reference))
    # Decompiled host modules share repository IR containers and FunctionTable
    # ownership. Merge them before source methods so calls link to exact roots;
    # the explicit completeness fact below distinguishes legalized repository
    # SSA from retained machine-state dialect inside those containers.
    if source_function_table is not None:
        for entry in source_function_table:
            host_module = entry.metadata.get("host_ssa_module")
            host_root = entry.metadata.get("host_ssa_root")
            if host_module is None or host_root is None:
                continue
            all_functions.update(host_module.functions)
            section_outputs.update(dict(
                entry.metadata.get("host_ssa_outputs") or {}
            ))
            function_symbols[int(entry.reference.address)] = str(host_root)
            all_tensor_tables.update(getattr(host_module, "tensor_tables", {}))
            all_sequence_tables.update(getattr(host_module, "sequence_tables", {}))
            all_record_tables.update(getattr(host_module, "record_tables", {}))
            all_reference_tables.update(
                getattr(host_module, "reference_tables", {})
            )
            machine_control_links.extend(
                getattr(
                    getattr(host_module, "machine_control_table", None),
                    "links", (),
                )
            )
            machine_indirect_links.extend(
                getattr(
                    getattr(host_module, "machine_indirect_table", None),
                    "links", (),
                )
            )
            host_blockers = tuple(entry.metadata.get("host_ssa_blockers", ()))
            host_hard_blockers = tuple(entry.metadata.get(
                "host_ssa_hard_blockers", host_blockers,
            ))
            host_legalization_shortfalls = tuple(entry.metadata.get(
                "host_ssa_legalization_shortfalls", (),
            ))
            host_unresolved_dependencies = tuple(entry.metadata.get(
                "host_ssa_unresolved_dependencies", (),
            ))
            host_repository_ssa_complete = bool(entry.metadata.get(
                "host_repository_ssa_complete", False,
            ))
            host_machine_state_complete = bool(entry.metadata.get(
                "host_machine_state_complete", False,
            ))
            host_native_module = entry.metadata.get("host_native_module")
            host_root_function = all_functions.get(str(host_root))
            if host_root_function is not None:
                from .native_code_retention import (
                    WINDOWS_AMD64_NATIVE_LINKER,
                    select_host_implementation,
                )
                implementation_decision = select_host_implementation(
                    repository_ssa_complete=host_repository_ssa_complete,
                    machine_state_ssa_complete=host_machine_state_complete,
                    retained_native_module=host_native_module,
                    target=WINDOWS_AMD64_NATIVE_LINKER,
                )
                host_root_function.metadata.update({
                    "host_ssa_complete": (
                        host_repository_ssa_complete
                    ),
                    "host_machine_state_complete": (
                        host_machine_state_complete
                    ),
                    "host_ssa_blockers": host_blockers,
                    "host_ssa_hard_blockers": host_hard_blockers,
                    "host_ssa_legalization_shortfalls": (
                        host_legalization_shortfalls
                    ),
                    "host_ssa_unresolved_dependencies": (
                        host_unresolved_dependencies
                    ),
                    "host_ssa_cache_key": entry.metadata.get(
                        "host_ssa_cache_key"
                    ),
                    "host_native_module": host_native_module,
                    "implementation_variants": entry.metadata.get(
                        "implementation_variants", ("repository-ssa",)
                    ),
                    "implementation_decision": implementation_decision,
                    "selected_implementation": implementation_decision.implementation.value,
                    "implementation_deployable": implementation_decision.deployable,
                })
    retained_storage_identities: set[str] = set()
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        if graph_obj is None:
            continue
        field_contract = _field_slot_ops(graph_obj)
        for _effect, _key, sequence_id, storage_identity in field_contract[12]:
            if sequence_id is not None:
                retained_storage_identities.add(str(storage_identity))
    # A sequence's numeric id is a global identity across the whole program
    # (it traces back to one shared ProcessGraph node's value_id), but each
    # shell (method) below lowers with its own local, independently-inferred
    # view of any sequence it touches. Two shells touching the same sequence
    # can therefore disagree not just on element dtype but on the sequence's
    # actual shape (how many storage cells it has) -- a real memory-layout
    # bug, not a cosmetic one. Survey every shell's raw declarations here,
    # before any shell's lowering commits to a shape, and resolve one
    # structural schema per sequence_id that every shell's lowering below
    # will be handed and required to agree with. See
    # ResolvedSequenceSchema in precompile_to_ssa.py for the full rationale.
    keyed_table_fields = frozenset(
        str(_field_name)
        for _record in dict(program_abi.get("records") or {}).values()
        for _field_name, _field in dict(_record.get("fields") or {}).items()
        if str(_field.get("storage") or "") == "keyed"
    )
    shell_sequence_evidence: list[dict[str, Any]] = []
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        if graph_obj is None:
            continue
        (
            _self_id, _field_ops, _const_sources, _field_count, _field_names,
            _record_identity, sequence_initializations, _field_aliases,
            sequence_declarations, _sequence_memberships, _table_lookups,
            _table_lookup_defaults, _table_stores, table_deletions,
            retained_sequence_ids, nested_sequence_ids, _nested_record_fields,
            tombstone_sequence_ids,
        ) = _field_slot_ops(
            graph_obj,
            retained_storage_identities=frozenset(retained_storage_identities),
            keyed_table_fields=keyed_table_fields,
        )
        shell_sequence_evidence.append({
            "sequence_declarations": sequence_declarations,
            "sequence_initializations": sequence_initializations,
            "table_deletions": table_deletions,
            "deletion_sequence_ids": tombstone_sequence_ids,
            "retained_sequence_ids": retained_sequence_ids,
            "nested_sequence_ids": nested_sequence_ids,
        })
    resolved_sequence_schemas, sequence_schema_shortfalls = (
        resolve_sequence_schemas(
            shell_sequence_evidence, location="sequence-schema-survey",
        )
    )
    if sequence_schema_shortfalls:
        lowering_failures.extend(
            ("<sequence-schema-survey>", item)
            for item in sequence_schema_shortfalls
        )
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        function_name = (
            graph_obj.graph.get("function_name") if graph_obj is not None else None
        )
        if function_name is None:
            continue
        if os.environ.get("TURING_DEBUG_BUILTIN_SELECTION"):
            for _node_id, _node_data in graph_obj.nodes(data=True):
                _expression = _node_data.get("expr_obj")
                _is_selection_call = (
                    isinstance(_expression, ast.Call)
                    and (
                        isinstance(_expression.func, ast.Name)
                        and _expression.func.id in {"next", "sum", "bytes"}
                        or isinstance(_expression.func, ast.Attribute)
                        and _expression.func.attr == "pack"
                    )
                )
                if not (
                    _is_selection_call
                    or isinstance(_expression, (ast.GeneratorExp, ast.comprehension))
                ):
                    continue
                print(
                    "DEBUGBUILTIN "
                    f"fn={function_name} node={int(_node_id)} "
                    f"value={_node_data.get('value_id', _node_id)} "
                    f"expr={ast.unparse(_expression)!r} "
                    f"parents={tuple(_node_data.get('parents') or ())!r} "
                    f"children={tuple(_node_data.get('children') or ())!r} "
                    f"attrs={dict(_node_data.get('attributes') or {})!r}",
                    file=sys.stderr,
                )
                if _is_selection_call:
                    _frontier = [
                        int(parent)
                        for parent, _role in _node_data.get("parents") or ()
                    ]
                    _seen = set()
                    for _depth in range(4):
                        _next_frontier = []
                        for _related_id in _frontier:
                            if (
                                _related_id in _seen
                                or _related_id not in graph_obj
                            ):
                                continue
                            _seen.add(_related_id)
                            _related = graph_obj.nodes[_related_id]
                            _related_expression = _related.get("expr_obj")
                            print(
                                "DEBUGBUILTINANCESTOR "
                                f"fn={function_name} depth={_depth} "
                                f"node={_related_id} "
                                f"kind={_related.get('type')!r} "
                                f"expr={None if _related_expression is None else ast.unparse(_related_expression)!r} "
                                f"parents={tuple(_related.get('parents') or ())!r} "
                                f"attrs={dict(_related.get('attributes') or {})!r}",
                                file=sys.stderr,
                            )
                            _next_frontier.extend(
                                int(parent)
                                for parent, _role in _related.get("parents") or ()
                            )
                        _frontier = _next_frontier
        if program_abi and not graph_obj.graph.get("parameter_record_abi"):
            selected = _record_receipts_for_function(
                program_abi,
                str(function_name),
                graph_obj.graph.get("function_parameters") or (),
                method_owner=graph_obj.graph.get("method_owner"),
            )
            if selected:
                graph_obj.graph["parameter_record_abi"] = selected
        if program_abi and not graph_obj.graph.get("parameter_value_abi"):
            parameters = set(map(
                str, graph_obj.graph.get("function_parameters") or ()
            ))
            selected_values = {}
            for binding in tuple(program_abi.get("values") or ()):
                parameter = str(binding.get("parameter") or "")
                if (
                    parameter in parameters
                    and fnmatchcase(
                        str(function_name), str(binding.get("function") or "")
                    )
                ):
                    selected_values[parameter] = dict(binding)
            if selected_values:
                graph_obj.graph["parameter_value_abi"] = selected_values
        control = getattr(shell, "shell_control_program", None)
        import os as _os, sys as _sys
        if _os.environ.get("TURING_DEBUG_REGION_ORDER"):
            def _walk_markers(block, acc):
                cls = type(block).__name__
                if cls == "StatementBlock":
                    for line in block.lines:
                        if line.startswith("__scheduled_region_"):
                            acc.append(line)
                elif cls == "SequenceBlock":
                    for child in block.blocks:
                        _walk_markers(child, acc)
                elif cls == "LoopBlock":
                    acc.append(f"LOOP[{block.induction}](")
                    _walk_markers(block.body, acc)
                    acc.append(")")
                elif cls == "WhileBlock":
                    acc.append("WHILE(")
                    _walk_markers(block.body, acc)
                    acc.append(")")
                elif cls == "ConditionalBlock":
                    acc.append("IF(")
                    _walk_markers(block.body, acc)
                    if block.orelse is not None:
                        acc.append("ELSE(")
                        _walk_markers(block.orelse, acc)
                        acc.append(")")
                    acc.append(")")
                elif cls == "CallBlock":
                    acc.append("CALL(")
                    _walk_markers(block.callee, acc)
                    acc.append(")")
                return acc
            _markers = _walk_markers(control.root, []) if control is not None else None
            print(
                f"DEBUGCLASSSHELL fn={function_name} "
                f"control_is_none={control is None} "
                f"region_indices={getattr(control, 'region_indices', None)} "
                f"root_markers={_markers}",
                file=_sys.stderr,
            )
        if control is None:
            continue
        # Some precompile-only shells retain the flat region schedule even
        # though branch compartments were already proven during partitioning.
        # Reapply the backend-neutral ordinary-conditional overlay here before
        # repository SSA sees the program.  Existing loop/while control is the
        # outer control and strict region containment nests each conditional
        # at its real lexical marker.
        from .control_source import overlay_scheduled_control
        from .glsl_deployment_strategy import (
            _ast_source_signature,
            _ordinary_conditional_control_programs,
            _source_control_expression,
        )
        def _source_conditional_ids(block):
            from .control_source import (
                CallBlock, ConditionalBlock, LoopBlock, ParallelDeployment,
                SequenceBlock, StateMachineTick, WhileBlock,
            )
            found = set()
            if isinstance(block, ConditionalBlock):
                if block.source_node_id is not None:
                    found.add(int(block.source_node_id))
                found.update(_source_conditional_ids(block.body))
                if block.orelse is not None:
                    found.update(_source_conditional_ids(block.orelse))
            elif isinstance(block, SequenceBlock):
                for child in block.blocks:
                    found.update(_source_conditional_ids(child))
            elif isinstance(block, LoopBlock):
                found.update(_source_conditional_ids(block.body))
            elif isinstance(block, WhileBlock):
                found.update(_source_conditional_ids(block.condition))
                found.update(_source_conditional_ids(block.body))
            elif isinstance(block, CallBlock):
                found.update(_source_conditional_ids(block.callee))
            elif isinstance(block, ParallelDeployment):
                for lane in block.lanes:
                    found.update(_source_conditional_ids(lane))
            elif isinstance(block, StateMachineTick):
                for _case, body in block.cases:
                    found.update(_source_conditional_ids(body))
                if block.default is not None:
                    found.update(_source_conditional_ids(block.default))
            return found

        def _lowered_source_control_ids(block):
            from .control_source import (
                CallBlock, ConditionalBlock, LoopBlock, ParallelDeployment,
                SequenceBlock, StateMachineTick, WhileBlock,
            )
            found = set()
            if isinstance(block, ConditionalBlock):
                if block.source_node_id is not None:
                    found.add(int(block.source_node_id))
                found.update(_lowered_source_control_ids(block.body))
                if block.orelse is not None:
                    found.update(_lowered_source_control_ids(block.orelse))
            elif isinstance(block, SequenceBlock):
                for child in block.blocks:
                    found.update(_lowered_source_control_ids(child))
            elif isinstance(block, LoopBlock):
                if block.source_loop_node_id is not None:
                    found.add(int(block.source_loop_node_id))
                found.update(_lowered_source_control_ids(block.body))
            elif isinstance(block, WhileBlock):
                if block.source_loop_node_id is not None:
                    found.add(int(block.source_loop_node_id))
                found.update(_lowered_source_control_ids(block.condition))
                found.update(_lowered_source_control_ids(block.body))
            elif isinstance(block, CallBlock):
                found.update(_lowered_source_control_ids(block.callee))
            elif isinstance(block, ParallelDeployment):
                for lane in block.lanes:
                    found.update(_lowered_source_control_ids(lane))
            elif isinstance(block, StateMachineTick):
                for _case, body in block.cases:
                    found.update(_lowered_source_control_ids(body))
                if block.default is not None:
                    found.update(_lowered_source_control_ids(block.default))
            return found

        represented_conditionals = _source_conditional_ids(control.root)
        conditional_controls = _ordinary_conditional_control_programs(
            graph,
            control.region_indices,
            getattr(shell, "dispatch_subgraphs", ()),
        )
        if represented_conditionals:
            from .control_source import ConditionalBlock, SequenceBlock

            def represented(program):
                return any(
                    isinstance(block, ConditionalBlock)
                    and block.source_node_id is not None
                    and int(block.source_node_id) in represented_conditionals
                    for block in program.root.blocks
                )

            conditional_controls = tuple(
                program
                for program in conditional_controls
                if not represented(program)
            )
        specialized_conditional_node_ids = tuple(dict.fromkeys(
            int(node_id) for node_id in (
                *control.specialized_conditional_node_ids,
                *tuple(graph_obj.graph.get(
                    "structurally_specialized_conditional_node_ids", ()
                )),
            )
        ))
        from .control_source import SequenceBlock, StatementBlock
        lowered_conditional_count = 0
        if conditional_controls:
            from .control_source import ConditionalBlock

            # Several conditions in one ``if/elif`` cascade can share a
            # scheduled predicate region.  Each isolated conditional view
            # names that prefix, but nesting all views would execute it once
            # per level.  Hoist only repeated top-level predicate prefixes
            # back to the flat schedule; branch regions remain owned by their
            # lexical conditional and are embedded below as usual.
            prefix_counts: dict[int, int] = {}
            prefixes_by_control: list[tuple[int, ...]] = []
            for conditional_control in conditional_controls:
                prefixes = []
                for block in conditional_control.root.blocks:
                    if isinstance(block, ConditionalBlock):
                        break
                    if (
                        isinstance(block, StatementBlock)
                        and len(block.lines) == 1
                        and block.lines[0].startswith("__scheduled_region_")
                    ):
                        region = int(
                            block.lines[0][len("__scheduled_region_"):-2]
                        )
                        prefixes.append(region)
                        prefix_counts[region] = prefix_counts.get(region, 0) + 1
                prefixes_by_control.append(tuple(prefixes))
            shared_prefixes = {
                region for region, count in prefix_counts.items() if count > 1
            }
            if shared_prefixes:
                conditional_controls = tuple(
                    replace(
                        conditional_control,
                        root=SequenceBlock(tuple(
                            block
                            for block in conditional_control.root.blocks
                            if not (
                                isinstance(block, StatementBlock)
                                and len(block.lines) == 1
                                and block.lines[0].startswith(
                                    "__scheduled_region_"
                                )
                                and int(block.lines[0][
                                    len("__scheduled_region_"):-2
                                ]) in shared_prefixes
                            )
                        )),
                        region_indices=tuple(
                            region
                            for region in conditional_control.region_indices
                            if int(region) not in shared_prefixes
                        ),
                    )
                    for conditional_control in conditional_controls
                )

            def conditional_of(program):
                return next((
                    block for block in program.root.blocks
                    if isinstance(block, ConditionalBlock)
                ), None)

            conditional_blocks = tuple(map(
                conditional_of, conditional_controls
            ))
            source_expressions = {
                int(block.source_node_id): _source_control_expression(
                    graph_obj, int(block.source_node_id)
                )
                for block in conditional_blocks
                if block is not None and block.source_node_id is not None
            }
            parent_by_child: dict[int, int] = {}
            for child_index, child in enumerate(conditional_blocks, start=1):
                if child is None or child.source_node_id is None:
                    continue
                candidates = []
                for parent_index, parent in enumerate(
                    conditional_blocks, start=1
                ):
                    if (
                        parent is None
                        or parent_index == child_index
                        or parent.source_node_id is None
                    ):
                        continue
                    expression = source_expressions.get(
                        int(parent.source_node_id)
                    )
                    if expression is None:
                        continue
                    descendants = {
                        _ast_source_signature(member) for statement in (
                            *expression.body, *expression.orelse
                        ) for member in ast.walk(statement)
                    }
                    child_expression = source_expressions.get(
                        int(child.source_node_id)
                    )
                    if (
                        child_expression is not None
                        and _ast_source_signature(child_expression) in descendants
                    ):
                        span = int(getattr(
                            expression, "end_lineno", expression.lineno
                        )) - int(expression.lineno)
                        candidates.append((span, parent_index))
                if candidates:
                    parent_by_child[child_index] = min(candidates)[1]
            # Index zero is the already-scheduled outer control program;
            # conditional programs begin at one.  The AST supplies exact
            # lexical containment, including equal-region nesting such as an
            # ``if`` whose entire arm is another ``if``.  Only maximal source
            # conditionals attach directly to the schedule root; every nested
            # conditional attaches to its nearest lexical conditional.
            direct_children: dict[int, list[int]] = {}
            for child_index in range(1, len(conditional_controls) + 1):
                direct_children.setdefault(
                    parent_by_child.get(child_index, 0), []
                ).append(child_index)
            control = overlay_scheduled_control(
                control.region_indices,
                (control, *conditional_controls),
                known_nesting={
                    parent: tuple(children)
                    for parent, children in direct_children.items()
                },
            )
            def marker_counts(block, counts):
                from .control_source import (
                    CallBlock, ConditionalBlock, LoopBlock,
                    ParallelDeployment, SequenceBlock, StateMachineTick,
                    StatementBlock, WhileBlock,
                )
                if isinstance(block, StatementBlock):
                    if (
                        len(block.lines) == 1
                        and block.lines[0].startswith("__scheduled_region_")
                    ):
                        index = int(
                            block.lines[0][len("__scheduled_region_"):-2]
                        )
                        counts[index] = counts.get(index, 0) + 1
                elif isinstance(block, SequenceBlock):
                    for child in block.blocks:
                        marker_counts(child, counts)
                elif isinstance(block, ConditionalBlock):
                    marker_counts(block.body, counts)
                    if block.orelse is not None:
                        marker_counts(block.orelse, counts)
                elif isinstance(block, (LoopBlock,)):
                    marker_counts(block.body, counts)
                elif isinstance(block, WhileBlock):
                    marker_counts(block.condition, counts)
                    marker_counts(block.body, counts)
                elif isinstance(block, StateMachineTick):
                    for _case, body in block.cases:
                        marker_counts(body, counts)
                    if block.default is not None:
                        marker_counts(block.default, counts)
                elif isinstance(block, ParallelDeployment):
                    for lane in block.lanes:
                        marker_counts(lane, counts)
                elif isinstance(block, CallBlock):
                    marker_counts(block.callee, counts)
            counts = {}
            marker_counts(control.root, counts)
            duplicates = {
                region: count for region, count in counts.items()
                if count != 1
            }
            if duplicates:
                raise FortranEmissionError(
                    "conditional control duplicated scheduled regions in "
                    f"{function_name!r}: {duplicates!r}"
                )
            lowered_conditional_count = len(conditional_controls)
        control, lexical_sequence_shortfalls = _install_lexical_sequence_mutations(
            control,
            graph,
            getattr(shell, "dispatch_subgraphs", ()),
        )
        control, lexical_query_shortfalls = _install_lexical_sequence_queries(
            control,
            graph,
            getattr(shell, "dispatch_subgraphs", ()),
        )
        control = _attach_graph_control_expressions(control, graph_obj)
        # Query placement initially sees only predicate result ids.  Once the
        # structured expression is attached, reschedule so a conditional such
        # as ``optional_row is None`` exposes its dependency on the row handle
        # produced by ``next(generator, None)``.
        control = replace(
            control,
            root=_schedule_sequence_query_dependencies(control.root),
        )
        (
            control,
            record_sequence_bindings,
            record_sequence_projection_fields,
        ) = _record_sequence_projection_bindings(graph_obj, control)
        if record_sequence_bindings:
            control = replace(
                control,
                projected_iterable_bindings=tuple(dict.fromkeys((
                    *control.projected_iterable_bindings,
                    *record_sequence_bindings,
                ))),
            )
        if os.environ.get("TURING_DEBUG_CALL_PLACEMENT"):
            from .control_source import (
                CallBlock as _DebugCallBlock,
                ConditionalBlock as _DebugConditionalBlock,
                LoopBlock as _DebugLoopBlock,
                SequenceBlock as _DebugSequenceBlock,
                WhileBlock as _DebugWhileBlock,
            )

            def _debug_calls(block, scope=()):
                if isinstance(block, _DebugCallBlock):
                    print(
                        "DEBUGCALLPLACE "
                        f"fn={function_name} callsite={block.callsite_id} "
                        f"scope={scope!r}",
                        file=sys.stderr,
                    )
                    _debug_calls(block.callee, (*scope, ("call", block.callsite_id)))
                elif isinstance(block, _DebugSequenceBlock):
                    for child in block.blocks:
                        _debug_calls(child, scope)
                elif isinstance(block, _DebugConditionalBlock):
                    _debug_calls(block.body, (*scope, ("if", block.source_node_id, "body")))
                    if block.orelse is not None:
                        _debug_calls(block.orelse, (*scope, ("if", block.source_node_id, "orelse")))
                elif isinstance(block, _DebugLoopBlock):
                    _debug_calls(block.body, (*scope, ("loop", block.source_loop_node_id)))
                elif isinstance(block, _DebugWhileBlock):
                    _debug_calls(block.condition, scope)
                    _debug_calls(block.body, (*scope, ("while", block.source_loop_node_id)))

            _debug_calls(control.root)
        lowered_conditional_count = len(_source_conditional_ids(control.root))
        function_reference = graph_obj.graph.get("function_ref")
        qualified_name = None
        if function_reference is not None and source_function_table is not None:
            try:
                qualified_name = source_function_table.entry(
                    int(function_reference)
                ).qualified_name
            except (KeyError, TypeError, ValueError):
                qualified_name = None
        symbol_source = (
            qualified_name
            if len(source_name_references.get(str(function_name), ())) > 1
            else function_name
        )
        symbol_suffix = str(symbol_source).replace(".", "__")
        specialization_contract = (
            graph_obj.graph.get("planner_specializations") or {},
            graph_obj.graph.get("planner_tensor_descriptors") or {},
        )
        if any(specialization_contract):
            specialization_digest = hashlib.sha256(
                repr(specialization_contract).encode("utf-8")
            ).hexdigest()[:12]
            symbol_suffix = (
                f"{symbol_suffix}__specialized_{specialization_digest}"
            )
        symbol = f"{artifact_name}__{symbol_suffix}"
        shell_symbols[id(shell)] = symbol
        if function_reference is not None:
            function_symbols[int(function_reference)] = symbol
        # Instance fields flow through the object's field arena: ``self`` is a
        # slot array, a field read is a load from its slot, a field write a
        # store. In whole-program precompile mode the field-op region is never
        # built (gated behind ``not precompile_only``), so recover the field ops
        # from the process graph and hand them to the lowerer as slot access.
        constant_struct_packs = _constant_struct_pack_materializations(
            graph_obj
        )
        constant_byte_literals = _constant_byte_literal_materializations(
            graph_obj
        )
        constant_byte_sequences = tuple(dict.fromkeys((
            *constant_byte_literals,
            *constant_struct_packs,
        )))
        self_id, field_ops, const_sources, field_count, field_names, record_identity, sequence_initializations, field_aliases, sequence_declarations, sequence_memberships, table_lookups, table_lookup_defaults, table_stores, table_deletions, retained_sequence_ids, nested_sequence_ids, nested_record_fields, _tombstone_sequence_ids = _field_slot_ops(
            graph_obj,
            retained_storage_identities=frozenset(retained_storage_identities),
            # A contract-declared keyed field is a lookup table, but it is a
            # program-ABI record field, NOT a class-field aggregate: seeding
            # it into class_field_aggregate_kinds engaged the object-field
            # arena machinery in every frame and displaced public-span
            # correlation for unrelated fields.  This channel reaches only
            # table recognition.
            keyed_table_fields=frozenset(
                str(_field_name)
                for _record in dict(program_abi.get("records") or {}).values()
                for _field_name, _field in dict(
                    _record.get("fields") or {}
                ).items()
                if str(_field.get("storage") or "") == "keyed"
            ),
        )
        sequence_initializations = tuple(dict.fromkeys((
            *sequence_initializations,
            *(
                (int(value_id), f"literal_bytes={payload.hex()}", 1)
                for value_id, payload, _node_id, _identity
                in constant_byte_sequences
            ),
        )))
        from .hierarchical_plan import PlanCall

        local_plan_calls = tuple(
            item
            for item in getattr(shell, "hierarchy_plan", ()).items
            if isinstance(item, PlanCall)
        )
        sequence_call_result_kinds: dict[int, str] = {}
        return_sequence_kind_by_reference: dict[int, str] = {}
        if source_function_table is not None:
            for entry in source_function_table:
                host_module = entry.metadata.get("host_ssa_module")
                host_root = entry.metadata.get("host_ssa_root")
                host_function = (
                    None if host_module is None or host_root is None
                    else host_module.functions.get(str(host_root))
                )
                if host_function is not None:
                    returned_ids = {
                        int(argument.id)
                        for block in host_function.blocks.values()
                        for instruction in block.instrs
                        if str(instruction.op).casefold() in {"ret", "return"}
                        for argument in instruction.args
                    }
                    returned_materializations = {
                        str(record.get("extraction_identity"))
                        for record in host_function.metadata.get(
                            "extraction_materializations", ()
                        )
                        if int(record.get("source_sequence_id", -1))
                        in returned_ids
                    }
                    returned_kind = next((
                        kind
                        for identity, kind in (
                            ("builtins.bytes", "bytes"),
                            ("builtins.bytearray", "bytearray"),
                            ("builtins.list", "list"),
                        )
                        if identity in returned_materializations
                    ), None)
                    if returned_kind is not None:
                        return_sequence_kind_by_reference[
                            int(entry.reference.address)
                        ] = returned_kind
                entry_graph = getattr(entry, "graph", None)
                entry_graph_obj = getattr(entry_graph, "G", None)
                if entry_graph_obj is None:
                    continue
                candidates = []
                for root_id in getattr(entry_graph, "roots", ()):
                    if int(root_id) not in entry_graph_obj:
                        continue
                    root_attributes = (
                        entry_graph_obj.nodes[int(root_id)].get("attributes")
                        or {}
                    )
                    kind = root_attributes.get("aggregate_kind")
                    if kind in {"list", "bytes", "bytearray"}:
                        candidates.append(str(kind))
                if len(set(candidates)) == 1:
                    return_sequence_kind_by_reference[
                        int(entry.reference.address)
                    ] = candidates[0]
        callsite_shells = getattr(shell, "callsite_function_shells", {})
        transformed_call_source_ids: set[int] = set()
        transformed_parameters_by_reference: dict[int, set[int]] = {}
        for planned_call in local_plan_calls:
            child_shell = callsite_shells.get(int(planned_call.callsite_id))
            child_graph = getattr(
                getattr(child_shell, "process_graph", None), "G", None
            )
            if child_graph is None:
                continue
            function_reference = child_graph.graph.get("function_ref")
            if function_reference is None:
                continue
            transformed_source_names = {
                str(source_name)
                for _result_id, _source_id, source_name, transform
                in _bytes_join_source_transforms(child_graph)
                if str(transform) in {"row_count", "join_bytes"}
            }
            transformed_parameters_by_reference.setdefault(
                int(function_reference), set()
            ).update(
                int(value_id)
                for source_name in transformed_source_names
                for value_id in (
                    child_graph.graph.get("identity_table") or {}
                ).get(source_name, ())
            )
        for planned_call in local_plan_calls:
            child_shell = callsite_shells.get(int(planned_call.callsite_id))
            child_graph = getattr(
                getattr(child_shell, "process_graph", None), "G", None
            )
            if child_graph is None:
                continue
            transformed_callee_ids = transformed_parameters_by_reference.get(
                int(child_graph.graph.get("function_ref", -1)), set()
            )
            transformed_call_source_ids.update(
                int(caller_id)
                for caller_id, callee_id
                in planned_call.argument_bindings
                if int(callee_id) in transformed_callee_ids
            )
            child_kind_by_value = _sequence_value_kinds(
                child_graph,
                return_sequence_kind_by_reference=(
                    return_sequence_kind_by_reference
                ),
            )
            for callee_id, caller_id in planned_call.result_bindings:
                child_kind = child_kind_by_value.get(int(callee_id))
                if child_kind is not None:
                    sequence_call_result_kinds[int(caller_id)] = child_kind
        for call_node_id, call_data in graph_obj.nodes(data=True):
            call_attributes = call_data.get("attributes") or {}
            reference = call_attributes.get(
                "callee_ref", call_attributes.get("method_ref")
            )
            if reference is None:
                continue
            return_kind = return_sequence_kind_by_reference.get(int(reference))
            if return_kind is not None:
                sequence_call_result_kinds.setdefault(
                    int(call_data.get("value_id", call_node_id)),
                    return_kind,
                )
        utf8_encode_aliases = (
            *_authored_text_parameter_transforms(graph_obj),
            *_utf8_encode_aliases(graph_obj),
        )
        bytes_join_transforms = _bytes_join_source_transforms(graph_obj)
        scalar_source_transforms = _scalar_source_transforms(
            graph_obj, (*utf8_encode_aliases, *bytes_join_transforms),
        )
        sequence_source_transforms = {
            int(source_id): {
                "source_name": str(source_name),
                "transform": str(transform),
            }
            for _result_id, source_id, source_name, transform
            in (*utf8_encode_aliases, *bytes_join_transforms)
        }
        source_transform_aliases = {
            int(result_id): (int(source_id), "bytes")
            for result_id, source_id, _source_name, _transform
            in (
                *utf8_encode_aliases,
                *(
                    record for record in bytes_join_transforms
                    if str(record[3]) == "join_bytes"
                ),
            )
        }
        (
            control,
            conditional_sequence_aliases,
            conditional_sequence_destinations,
        ) = _promote_conditional_sequence_aliases(
            control,
            graph_obj,
            call_result_kinds=sequence_call_result_kinds,
            structural_aliases=source_transform_aliases,
        )
        (
            sequence_concats,
            sequence_concat_aliases,
            sequence_singleton_values,
        ) = _sequence_concat_ops(
            graph_obj,
            call_result_kinds=sequence_call_result_kinds,
            structural_aliases={
                **source_transform_aliases,
                **conditional_sequence_aliases,
            },
        )
        structural_byte_sequence_ids = tuple(
            int(value_id)
            for (
                result_id, lhs_id, rhs_id, _kind,
                lhs_scalar, rhs_scalar,
            ) in sequence_concats
            for value_id in (
                int(result_id),
                *(() if lhs_scalar is not None else (int(lhs_id),)),
                *(() if rhs_scalar is not None else (int(rhs_id),)),
            )
        )
        singleton_scalar_ids = {
            int(value_id)
            for (
                _result_id, _lhs_id, _rhs_id, _kind,
                lhs_scalar, rhs_scalar,
            ) in sequence_concats
            for value_id in (lhs_scalar, rhs_scalar)
            if value_id is not None
        }
        node_by_value = {
            int(data.get("value_id", node_id)): int(node_id)
            for node_id, data in graph_obj.nodes(data=True)
        }
        if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
            for value_id in sorted(singleton_scalar_ids):
                node_id = node_by_value.get(value_id)
                data = graph_obj.nodes.get(node_id, {})
                print(
                    "DEBUGSINGLETON "
                    f"fn={graph_obj.graph.get('function_name')} "
                    f"value={value_id} node={node_id} "
                    f"type={data.get('type')!r} op={data.get('op')!r} "
                    f"expr={ast.dump(data.get('expr_obj'), include_attributes=False) if isinstance(data.get('expr_obj'), ast.AST) else None!r} "
                    f"parents={tuple(data.get('parents') or ())!r} "
                    f"histories={tuple((name, tuple(history)) for name, history in (graph_obj.graph.get('identity_table') or {}).items() if value_id in tuple(map(int, history)))!r}",
                    file=sys.stderr,
                )
        singleton_name_aliases: dict[int, int] = {}
        for value_id in singleton_scalar_ids:
            data = graph_obj.nodes.get(node_by_value.get(value_id), {})
            expression = data.get("expr_obj")
            if not (
                isinstance(expression, (ast.List, ast.Tuple, ast.Set))
                and len(expression.elts) == 1
            ):
                continue
            leaf_nodes = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and int(parent) in graph_obj
            )
            if len(leaf_nodes) == 1:
                leaf_node = leaf_nodes[0]
                singleton_name_aliases[int(value_id)] = int(
                    graph_obj.nodes[leaf_node].get("value_id", leaf_node)
                )
        for history in (graph_obj.graph.get("identity_table") or {}).values():
            ordered = tuple(map(int, history))
            inputs = tuple(
                value_id for value_id in ordered
                if value_id in node_by_value
                and str(graph_obj.nodes[node_by_value[value_id]].get(
                    "type", ""
                )).casefold() == "input"
            )
            if len(inputs) != 1:
                continue
            source_id = int(inputs[0])
            for value_id in ordered:
                if value_id not in singleton_scalar_ids or value_id == source_id:
                    continue
                data = graph_obj.nodes.get(node_by_value.get(value_id), {})
                expression = data.get("expr_obj")
                if isinstance(expression, ast.Name) and isinstance(
                    expression.ctx, ast.Load
                ):
                    singleton_name_aliases[int(value_id)] = source_id
        if conditional_sequence_destinations:
            sequence_declarations = tuple(
                (
                    int(sequence_id), str(policy), int(columns),
                    True if int(sequence_id) in set(
                        conditional_sequence_destinations
                    ) else bool(writable),
                )
                for sequence_id, policy, columns, writable
                in sequence_declarations
            )
        declared_sequence_ids = {
            int(sequence_id)
            for sequence_id, _policy, _columns, _writable
            in sequence_declarations
        }
        concat_result_ids = {
            int(result_id)
            for result_id, _lhs, _rhs, _kind, _lhs_scalar, _rhs_scalar
            in sequence_concats
        }
        for sequence_id in sorted(
            set(structural_byte_sequence_ids) - concat_result_ids
        ):
            if sequence_id not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(sequence_id), "duplicates", 1, False,
                ))
                declared_sequence_ids.add(int(sequence_id))
        from .control_source import (
            CallBlock as _ControlCallBlock,
            ConditionalBlock as _ControlConditionalBlock,
            LoopBlock as _ControlLoopBlock,
            SequenceBlock as _ControlSequenceBlock,
            SequenceMutationBlock as _ControlSequenceMutationBlock,
            SequenceQueryBlock as _ControlSequenceQueryBlock,
            WhileBlock as _ControlWhileBlock,
        )

        def _generated_sequence_ids(block):
            found = set()
            mutations = ()
            if isinstance(block, _ControlSequenceMutationBlock):
                mutations = (block.mutation,)
            elif isinstance(block, (_ControlLoopBlock, _ControlWhileBlock)):
                mutations = block.sequence_mutations
                if isinstance(block, _ControlWhileBlock):
                    found.update(_generated_sequence_ids(block.condition))
                found.update(_generated_sequence_ids(block.body))
            elif isinstance(block, _ControlSequenceQueryBlock):
                found.add(int(block.sequence_value_id))
            elif isinstance(block, _ControlSequenceBlock):
                for child in block.blocks:
                    found.update(_generated_sequence_ids(child))
            elif isinstance(block, _ControlConditionalBlock):
                found.update(_generated_sequence_ids(block.body))
                if block.orelse is not None:
                    found.update(_generated_sequence_ids(block.orelse))
            elif isinstance(block, _ControlCallBlock):
                found.update(_generated_sequence_ids(block.callee))
            for mutation in mutations:
                sequence_id = int(mutation.sequence_value_id)
                sequence_data = graph_obj.nodes.get(sequence_id, {})
                sequence_attributes = sequence_data.get("attributes") or {}
                if (
                    sequence_data.get("type") == "LoopResult"
                    and sequence_attributes.get("result_kind") == "collection"
                ):
                    found.add(sequence_id)
            return found

        generated_sequence_ids = _generated_sequence_ids(control.root)
        materialized_sequence_columns: dict[int, int] = {}
        for _node_id, data in graph_obj.nodes(data=True):
            attributes = data.get("attributes") or {}
            column_count = int(attributes.get("sequence_column_count", 1))
            if column_count <= 1:
                continue
            for source_value_id in attributes.get(
                "materialized_source_value_ids", ()
            ):
                materialized_sequence_columns[int(source_value_id)] = max(
                    column_count,
                    materialized_sequence_columns.get(
                        int(source_value_id), 1
                    ),
                )
        for sequence_id in sorted(generated_sequence_ids):
            if sequence_id not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(sequence_id), "duplicates",
                    materialized_sequence_columns.get(int(sequence_id), 1),
                    True,
                ))
                declared_sequence_ids.add(int(sequence_id))
        for call_result_id in sorted(sequence_call_result_kinds):
            if call_result_id not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(call_result_id), "duplicates", 1, True,
                ))
                declared_sequence_ids.add(int(call_result_id))
        for source_id in sorted(sequence_source_transforms):
            if source_id not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(source_id), "duplicates", 1, False,
                ))
                declared_sequence_ids.add(int(source_id))
        initialized_sequence_ids = {
            int(sequence_id)
            for sequence_id, _policy, _columns in sequence_initializations
        }
        for sequence_id in sorted(generated_sequence_ids):
            if sequence_id not in initialized_sequence_ids:
                sequence_initializations = (*sequence_initializations, (
                    int(sequence_id), "duplicates", 1,
                ))
                initialized_sequence_ids.add(int(sequence_id))
        for (
            result_id, _lhs_id, _rhs_id, _kind, _lhs_scalar, _rhs_scalar,
        ) in sequence_concats:
            if int(result_id) not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(result_id), "duplicates", 1, True,
                ))
                declared_sequence_ids.add(int(result_id))
            if int(result_id) not in initialized_sequence_ids:
                sequence_initializations = (*sequence_initializations, (
                    int(result_id), "duplicates", 1,
                ))
                initialized_sequence_ids.add(int(result_id))
        declared_column_counts = {
            int(sequence_id): int(column_count)
            for sequence_id, _policy, column_count, _writable
            in sequence_declarations
        }
        sequence_initializations = tuple(
            (
                int(sequence_id), str(policy),
                declared_column_counts.get(int(sequence_id), int(column_count)),
            )
            for sequence_id, policy, column_count in sequence_initializations
        )
        joined_sequence_ids = _joined_byte_sequence_ids(
            graph_obj,
            call_result_kinds=sequence_call_result_kinds,
            declared_sequence_ids=declared_sequence_ids,
            structural_byte_sequence_ids=structural_byte_sequence_ids,
            transformed_call_source_ids=transformed_call_source_ids,
        )

        def _control_sequence_mutations(block):
            """Yield every resident mutation retained by the coordinator."""

            if isinstance(block, _ControlSequenceMutationBlock):
                yield block.mutation
            elif isinstance(block, (_ControlLoopBlock, _ControlWhileBlock)):
                yield from block.sequence_mutations
                if isinstance(block, _ControlWhileBlock):
                    yield from _control_sequence_mutations(block.condition)
                yield from _control_sequence_mutations(block.body)
            elif isinstance(block, _ControlSequenceBlock):
                for child in block.blocks:
                    yield from _control_sequence_mutations(child)
            elif isinstance(block, _ControlConditionalBlock):
                yield from _control_sequence_mutations(block.body)
                if block.orelse is not None:
                    yield from _control_sequence_mutations(block.orelse)
            elif isinstance(block, _ControlCallBlock):
                yield from _control_sequence_mutations(block.callee)

        # A source collection expression and the storage used by its retained
        # loop are deliberately distinct SSA identities.  The mutation's
        # effect identity is their authored correlation: propagate the joined
        # list[bytes] representation to the generated storage rather than
        # recovering it from a name or allocating an unrelated slot.
        joined_set = set(map(int, joined_sequence_ids))
        control_mutations = tuple(_control_sequence_mutations(control.root))
        changed = True
        while changed:
            changed = False
            for mutation in control_mutations:
                if int(mutation.effect_node_id) not in joined_set:
                    continue
                sequence_id = int(mutation.sequence_value_id)
                if sequence_id not in joined_set:
                    joined_set.add(sequence_id)
                    changed = True
        joined_sequence_ids = tuple(sorted(joined_set))
        joined_literal_mutations = _joined_list_literal_mutations(
            graph_obj, joined_sequence_ids
        )
        if joined_literal_mutations:
            control, joined_literal_shortfalls = (
                _install_lexical_sequence_mutations(
                    control,
                    graph,
                    getattr(shell, "dispatch_subgraphs", ()),
                    extra_mutations=joined_literal_mutations,
                )
            )
            lexical_sequence_shortfalls = (
                *lexical_sequence_shortfalls,
                *joined_literal_shortfalls,
            )
        for sequence_id in joined_sequence_ids:
            if int(sequence_id) not in declared_sequence_ids:
                sequence_declarations = (*sequence_declarations, (
                    int(sequence_id), "duplicates", 1, True,
                ))
                declared_sequence_ids.add(int(sequence_id))
            if int(sequence_id) not in initialized_sequence_ids:
                sequence_initializations = (*sequence_initializations, (
                    int(sequence_id), "duplicates", 1,
                ))
                initialized_sequence_ids.add(int(sequence_id))
        identities = graph_obj.graph.get("identity_table") or {}
        region_output_value_ids = {
            int(region_index): tuple(map(
                int, subgraph.G.graph.get("deployment_outputs", ()),
            ))
            for region_index, subgraph in enumerate(
                getattr(shell, "dispatch_subgraphs", ())
            )
        }
        parameter_facts = {
            **dict(graph_obj.graph.get("parameter_defaults") or {}),
            **dict(graph_obj.graph.get("planner_specializations") or {}),
        }
        parameter_value_dtypes = {}
        constant_values = {}
        singleton_concat_value_ids = {
            int(value_id)
            for (
                _result_id, _lhs_id, _rhs_id, _kind,
                lhs_scalar, rhs_scalar,
            ) in sequence_concats
            for value_id in (lhs_scalar, rhs_scalar)
            if value_id is not None
        }
        required_root_output_value_ids = []
        root_sequence_materializations = []
        for root_id in tuple(getattr(shell.process_graph, "roots", ())):
            if int(root_id) not in graph_obj:
                continue
            root_data = graph_obj.nodes[int(root_id)]
            root_attributes = root_data.get("attributes") or {}
            # ``bytes(local_sequence)`` is an immutable materialization of
            # storage owned by this call.  At the native boundary the caller
            # receives that same arena plus its length; no observable alias to
            # the dead local bytearray escapes, so a second copy is unnecessary.
            # Keep the conversion's authored root as authority while returning
            # the exact resident sequence that the loop populated.
            if (
                root_attributes.get("producer_kind")
                == "aggregate_materialization"
                and root_attributes.get("aggregate_kind") == "bytes"
            ):
                sources = tuple(
                    int(parent)
                    for parent, role in root_data.get("parents") or ()
                    if str(role).startswith("arg:") and int(parent) in graph_obj
                )
                if len(sources) == 1:
                    source_sequence_id = int(
                        graph_obj.nodes[sources[0]].get(
                            "value_id", sources[0]
                        )
                    )
                    required_root_output_value_ids.append(source_sequence_id)
                    root_sequence_materializations.append({
                        "source_node_id": int(root_id),
                        "source_sequence_id": source_sequence_id,
                        "extraction_identity": root_attributes.get(
                            "extraction_identity"
                        ),
                        "lowering": "immutable-local-sequence-view",
                    })
                    continue
            required_root_output_value_ids.append(int(
                root_data.get("value_id", root_id)
            ))
            root_value_id = int(root_data.get("value_id", root_id))
            concat_kind = next((
                kind
                for (
                    result_id, _lhs_id, _rhs_id, kind,
                    _lhs_scalar, _rhs_scalar,
                ) in sequence_concats
                if int(result_id) == root_value_id
            ), None)
            if concat_kind is not None:
                root_sequence_materializations.append({
                    "source_node_id": int(root_id),
                    "source_sequence_id": root_value_id,
                    "extraction_identity": f"builtins.{concat_kind}",
                    "lowering": "immutable-sequence-concatenation",
                })
        def scalar_fact_dtype(fact):
            if isinstance(fact, bool):
                return "bool"
            if isinstance(fact, int):
                # An observed Python integer refines type, not machine width.
                # Keep it aligned with the authored ``int`` annotation and
                # the repository's fixed-width Python-integer ABI.
                return "int64"
            if isinstance(fact, float):
                return "float64"
            return None

        annotation_dtypes: dict[str, str] = {}
        function_identity = str(
            graph_obj.graph.get("function_name") or function_name
        )
        authored_annotations = dict(
            (
                graph_obj.graph.get("function_parameter_annotations") or {}
            ).get(function_identity, {})
        )
        for parameter_name, annotation in authored_annotations.items():
            spelling = str(annotation).strip()
            dtype = {
                "bool": "bool",
                "int": "int64",
                "float": "float64",
            }.get(spelling)
            if dtype is not None:
                annotation_dtypes[str(parameter_name)] = dtype
        if function_reference is not None and source_function_table is not None:
            try:
                source_entry = source_function_table.entry(
                    int(function_reference)
                )
            except (KeyError, TypeError, ValueError):
                source_entry = None
            callable_object = (
                None if source_entry is None
                else getattr(source_entry, "python_callable", None)
            )
            if callable_object is not None:
                try:
                    source_signature = inspect.signature(callable_object)
                except (TypeError, ValueError):
                    source_signature = None
                if source_signature is not None:
                    for parameter in source_signature.parameters.values():
                        annotation = parameter.annotation
                        spelling = (
                            annotation if isinstance(annotation, str)
                            else getattr(annotation, "__name__", "")
                        )
                        dtype = {
                            "bool": "bool",
                            "int": "int64",
                            "float": "float64",
                        }.get(str(spelling))
                        if dtype is not None:
                            annotation_dtypes[str(parameter.name)] = dtype

        for parameter_name in tuple(
            graph_obj.graph.get("function_parameters") or ()
        ):
            history = tuple(identities.get(str(parameter_name), ()))
            fact_dtype = scalar_fact_dtype(
                parameter_facts.get(str(parameter_name))
            )
            if history and fact_dtype is not None:
                parameter_value_dtypes[int(history[0])] = fact_dtype
            if history and str(parameter_name) in annotation_dtypes:
                parameter_value_dtypes.setdefault(
                    int(history[0]), annotation_dtypes[str(parameter_name)]
                )
        for value_id, data in graph_obj.nodes(data=True):
            if str(data.get("type") or "").casefold() in {
                "constant", "const",
            } or str(data.get("op") or "").casefold() == "const":
                attributes = data.get("attributes") or {}
                literal = (
                    attributes["value"]
                    if "value" in attributes
                    else data.get("constant")
                )
                materialized_value_id = int(data.get("value_id", value_id))
                if (
                    materialized_value_id in singleton_concat_value_ids
                    and isinstance(literal, (list, tuple, bytes, bytearray))
                    and len(literal) == 1
                ):
                    literal = literal[0]
                constant_values[materialized_value_id] = (
                    _copy_literal_payload(literal)
                )
            if str(data.get("type") or "").casefold() != "input":
                continue
            parameter_name = str(
                (data.get("attributes") or {}).get("binding_name") or ""
            )
            fact = parameter_facts.get(parameter_name)
            fact_dtype = scalar_fact_dtype(fact)
            if fact_dtype is not None:
                parameter_value_dtypes[int(value_id)] = fact_dtype
        # A receiver field read is a physical input just as surely as a named
        # scalar parameter.  Carry its declared record dtype into region
        # planning before the field-slot load is injected; otherwise the
        # numerical region defaults to float64 even when the enclosing record
        # correctly emits an int64/bool slot.
        self_record_fields = dict(
            (
                (graph_obj.graph.get("parameter_record_abi") or {})
                .get("self") or {}
            ).get("fields") or {}
        )
        for _kind, value_id, slot in field_ops:
            if not (0 <= int(slot) < len(field_names)):
                continue
            field = dict(self_record_fields.get(field_names[int(slot)]) or {})
            if field.get("dtype") is not None:
                parameter_value_dtypes[int(value_id)] = str(field["dtype"])
        # A free-function record assignment already has an exact dependency
        # edge: SetAttr(object=<parameter>, value=<producer>).  Preserve that
        # producer identity across the region/control call boundary so the
        # target can bind it directly to the caller's field storage.  Method
        # ``self`` writes use the class field-slot arena handled by
        # ``field_ops`` above and must not be duplicated here.
        parameter_record_write_value_ids: list[int] = []
        declared_record_parameters = {
            str(parameter_name)
            for parameter_name in dict(
                graph_obj.graph.get("parameter_record_abi") or {}
            )
            if str(parameter_name) != "self"
        }
        record_parameter_value_ids = {
            int(value_id)
            for parameter_name in declared_record_parameters
            for value_id in identities.get(parameter_name, ())
        }
        if record_parameter_value_ids:
            for _node_id, data in graph_obj.nodes(data=True):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() != "setattr":
                    continue
                parents = tuple(data.get("parents") or ())
                object_value_ids = {
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent, role in parents
                    if str(role) in {"object", "base", "receiver"}
                    and parent in graph_obj
                }
                if not object_value_ids.intersection(
                    record_parameter_value_ids
                ):
                    continue
                parameter_record_write_value_ids.extend(
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent, role in parents
                    if str(role) == "value" and parent in graph_obj
                )
        # All structural/materialization passes have now had their say about
        # lexical placement.  Reassert query-producer dominance at this final
        # control boundary so a source-position insertion cannot separate a
        # generator from the query that materializes its scalar result.
        control = replace(
            control,
            root=_schedule_sequence_query_dependencies(control.root),
        )
        module_ir, shortfalls, shell_section_outputs = (
            lower_control_sections_to_ssa(
                control,
                hierarchy_plan=getattr(shell, "hierarchy_plan", None),
                preloaded_value_aliases={
                    **_loop_carried_storage_aliases(graph_obj),
                    **{
                        int(value_id): int(resident_id)
                        for value_id, (resident_id, _kind)
                        in conditional_sequence_aliases.items()
                    },
                    **dict(sequence_concat_aliases),
                    **singleton_name_aliases,
                },
                control_name=symbol,
                identity_table=dict(graph_obj.graph.get("identity_table") or {}),
                function_outputs=tuple(
                    graph_obj.graph.get("function_outputs") or ()
                ),
                function_parameters=tuple(
                    graph_obj.graph.get("function_parameters") or ()
                ),
                value_dtypes=parameter_value_dtypes,
                constant_values=constant_values,
                required_output_value_ids=tuple(dict.fromkeys(
                    required_root_output_value_ids
                )),
                region_output_value_ids=region_output_value_ids,
                record_field_write_value_ids=tuple(dict.fromkeys(
                    parameter_record_write_value_ids
                )),
                self_value_id=self_id,
                field_ops=field_ops,
                field_const_sources=const_sources,
                field_count=field_count,
                field_names=field_names,
                record_identity=record_identity,
                record_field_dtypes={
                    str(field_name): str(field["dtype"])
                    for field_name, field in dict(
                        (
                            (graph_obj.graph.get("parameter_record_abi") or {})
                            .get("self") or {}
                        ).get("fields") or {}
                    ).items()
                    if field.get("dtype") is not None
                },
                record_field_mutability={
                    str(field_name): bool(field.get("mutable", False))
                    for field_name, field in dict(
                        (
                            (graph_obj.graph.get("parameter_record_abi") or {})
                            .get("self") or {}
                        ).get("fields") or {}
                    ).items()
                },
                sequence_initializations=sequence_initializations,
                field_aliases=field_aliases,
                sequence_declarations=sequence_declarations,
                sequence_column_dtypes=_sequence_column_dtype_contracts(
                    graph_obj, sequence_declarations
                ),
                source_sequence_ids=_authored_source_sequence_ids(
                    graph_obj, sequence_declarations
                ),
                sequence_memberships=sequence_memberships,
                table_lookups=table_lookups,
                table_lookup_defaults=table_lookup_defaults,
                table_stores=table_stores,
                table_deletions=table_deletions,
                retained_sequence_ids=retained_sequence_ids,
                nested_sequence_ids=nested_sequence_ids,
                joined_sequence_ids=joined_sequence_ids,
                joined_singleton_values=sequence_singleton_values,
                nested_record_fields=nested_record_fields,
                sequence_augassigns=_sequence_augassign_ops(graph_obj),
                sequence_concats=sequence_concats,
                sequence_append_fills=_sequence_append_fill_ops(graph_obj),
                sequence_append_slices=_sequence_append_slice_ops(graph_obj),
                sequence_bit_packs=_sequence_bit_pack_ops(graph_obj),
                sequence_prepends=_sequence_prepend_concat_ops(graph_obj),
                sequence_prepend_packed_calls=(
                    _sequence_prepend_packed_call_ops(graph_obj)
                ),
                sequence_inplace_bit_pack_calls=(
                    _sequence_inplace_bit_pack_call_ops(graph)
                ),
                sequence_row_operations=_sequence_row_operations(
                    graph_obj, sequence_declarations
                ),
                nested_row_projections=_nested_row_projection_ops(
                    graph_obj, control
                ),
                sequence_length_values=_sequence_length_values(
                    graph_obj,
                    sequence_declarations,
                    {
                        **_loop_carried_storage_aliases(graph_obj),
                        **{
                            int(value_id): int(resident_id)
                            for value_id, (resident_id, _kind)
                            in conditional_sequence_aliases.items()
                        },
                        **dict(sequence_concat_aliases),
                    },
                ),
                string_table=string_table,
                tensor_ssa_reference=tensor_ssa_reference,
                resolved_sequence_schemas=resolved_sequence_schemas,
            )
        )
        if shortfalls:
            lowering_failures.extend((symbol, item) for item in shortfalls)
        lowering_failures.extend(
            (
                symbol,
                SSALoweringShortfall(
                    "control",
                    "sequence-mutation-guard",
                    symbol,
                    "authored sequence mutation has no retained predicate "
                    f"identity: effect_node_id={effect_id}",
                ),
            )
            for effect_id in lexical_sequence_shortfalls
        )
        lowering_failures.extend(
            (
                symbol,
                SSALoweringShortfall(
                    "control",
                    "sequence-query",
                    symbol,
                    "generator consumer has no safe resident query lowering: "
                    f"call_node_id={call_id}",
                ),
            )
            for call_id in lexical_query_shortfalls
        )
        all_functions.update(module_ir.functions)
        lowered_control = module_ir.functions.get(symbol)
        if lowered_control is not None:
            lowered_control.metadata["source_function_reference"] = (
                None if function_reference is None
                else int(function_reference)
            )
            lowered_control.metadata["source_qualified_name"] = (
                str(qualified_name or function_name)
            )
            lowered_control.metadata["sequence_source_transforms"] = tuple(
                (
                    int(sequence_id),
                    str(record["source_name"]),
                    str(record["transform"]),
                )
                for sequence_id, record in sorted(
                    sequence_source_transforms.items()
                )
            )
            lowered_control.metadata["scalar_source_transforms"] = tuple(
                scalar_source_transforms
            )
            joined_set = set(map(int, joined_sequence_ids))
            joined_identity_aliases: dict[int, int] = {}
            for _name, history in (
                graph_obj.graph.get("identity_table") or {}
            ).items():
                ordered = tuple(map(int, history))
                residents = tuple(
                    value_id for value_id in ordered
                    if value_id in joined_set
                )
                if not residents:
                    continue
                resident = int(residents[-1])
                joined_identity_aliases.update(
                    (int(value_id), resident) for value_id in ordered
                )
            lowered_control.metadata[
                "joined_sequence_identity_aliases"
            ] = tuple(sorted(joined_identity_aliases.items()))
            sequence_table = lowered_control.metadata.get("sequence_table")
            sequence_descriptors = (
                {} if sequence_table is None else sequence_table.sequences
            )
            ret_value_ids = {
                int(argument.id)
                for block in lowered_control.blocks.values()
                for instruction in block.instrs
                if str(instruction.op).casefold() in {"ret", "return"}
                for argument in instruction.args
            }
            extraction_materializations = []
            sequence_materialization_aliases = dict(
                map(lambda pair: (int(pair[0]), int(pair[1])), sequence_concat_aliases)
            )
            for node_id, node_data in graph_obj.nodes(data=True):
                attributes = node_data.get("attributes") or {}
                sequence_id = int(node_data.get("value_id", node_id))
                descriptor = sequence_descriptors.get(sequence_id)
                if (
                    attributes.get("extraction_identity") == "builtins.bytes"
                    and attributes.get("producer_kind")
                    == "aggregate_materialization"
                    and sequence_id in sequence_materialization_aliases
                ):
                    extraction_materializations.append({
                        "source_node_id": int(node_id),
                        "source_sequence_id": sequence_id,
                        "resident_sequence_id": int(
                            sequence_materialization_aliases[sequence_id]
                        ),
                        "extraction_identity": "builtins.bytes",
                        "lowering": "immutable-sequence-view",
                    })
                if (
                    attributes.get("producer_kind") == "aggregate"
                    and attributes.get("aggregate_kind") == "bytearray"
                    and descriptor is not None
                    and descriptor.writable
                    and attributes.get("extraction_identity") is not None
                ):
                    extraction_materializations.append({
                        "source_node_id": int(node_id),
                        "source_sequence_id": sequence_id,
                        "extraction_identity": str(
                            attributes["extraction_identity"]
                        ),
                        "lowering": "writable-sequence-arena",
                    })
            extraction_materializations.extend(
                dict(record)
                for record in root_sequence_materializations
                if record.get("extraction_identity") is not None
                and int(record["source_sequence_id"])
                in sequence_descriptors
                and int(record["source_sequence_id"]) in ret_value_ids
            )
            extraction_materializations.extend({
                "source_node_id": int(node_id),
                "source_sequence_id": int(value_id),
                "extraction_identity": str(identity),
                "lowering": "compile-time-constant-sequence",
                "byte_count": len(payload),
            } for value_id, payload, node_id, identity in constant_struct_packs
              if int(value_id) in sequence_descriptors)
            source_output_value_ids = tuple(dict.fromkeys(
                int(history[-1])
                for name in tuple(
                    graph_obj.graph.get("function_outputs") or ()
                )
                for history in (tuple(
                    (graph_obj.graph.get("identity_table") or {}).get(
                        str(name), ()
                    )
                ),)
                if history
            ))
            lowered_source_value_ids = set(
                _lowered_source_control_ids(control.root)
            )
            # Loop targets are definitions owned by the control coordinator,
            # not public inputs.  ``ControlSSABuilder`` realizes each of
            # these identities from its resident iterable (or an exact
            # projected/static/closure-backed spelling) at the top of every
            # iteration.  Account for those authored definitions alongside
            # the loop node itself; otherwise a downstream call that consumes
            # the target makes the completeness audit report a phantom input
            # even though the target already has a concrete SSA producer.
            lowered_source_value_ids.update(
                int(target_id)
                for bindings in (
                    control.iterable_bindings,
                    control.static_iterable_bindings,
                    control.closure_iterable_bindings,
                    control.projected_iterable_bindings,
                )
                for _iterable_id, target_id, *_rest in bindings
            )
            # Collection bindings name the per-iteration source first; the
            # coordinator writes that exact value into resident collection
            # storage at the indexed destination.
            lowered_source_value_ids.update(
                int(source_value_id)
                for source_value_id, _resident_id, _induction, _start
                in control.collection_bindings
            )
            for mutation in control_mutations:
                effect_id = int(mutation.effect_node_id)
                lowered_source_value_ids.add(effect_id)
                effect_data = graph_obj.nodes.get(effect_id, {})
                lowered_source_value_ids.update(
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent, role in effect_data.get("parents") or ()
                    if str(role).startswith("arg:") and parent in graph_obj
                )
            for operation in _sequence_row_operations(
                graph_obj, sequence_declarations
            ):
                lowered_source_value_ids.add(int(operation[1]))
                lowered_source_value_ids.add(int(operation[-1]))
            for materialization in extraction_materializations:
                lowered_source_value_ids.update(
                    int(materialization[key])
                    for key in (
                        "source_node_id", "source_sequence_id",
                        "resident_sequence_id",
                    )
                    if materialization.get(key) is not None
                )
            # A declared record parameter is represented by the physical
            # scalar/span fields materialized above.  Its original object
            # identity is deliberately absent from the machine ABI, but it is
            # no longer an unlowered source value.  This used to special-case
            # ``self`` and consequently rejected an equally complete annotated
            # parameter such as ``body: CodeBuilder``.
            identities = graph_obj.graph.get("identity_table") or {}
            for parameter_name in dict(
                graph_obj.graph.get("parameter_record_abi") or {}
            ):
                lowered_source_value_ids.update(map(
                    int, identities.get(str(parameter_name), ()),
                ))
                lowered_source_value_ids.update(
                    int(data.get("value_id", node_id))
                    for node_id, data in graph_obj.nodes(data=True)
                    if str(data.get("type") or "").casefold() == "input"
                    and str((data.get("attributes") or {}).get(
                        "binding_name", ""
                    )) == str(parameter_name)
                )
            # The structural shell is lowered through the assigned hierarchy,
            # so its machine values use the hierarchy's sole global SSA
            # identity while the source graph ledger above is still local to
            # this closure.  Preserve both forms in this diagnostic/accounting
            # set.  This is an exact correlation from the deterministic value
            # table, not an ID remint or a name-based alias.
            hierarchy_values = getattr(
                shell, "hierarchy_value_table", None,
            )
            hierarchy_plan = getattr(shell, "hierarchy_plan", None)
            if hierarchy_values is not None and hierarchy_plan is not None:
                closure_id = int(hierarchy_plan.closure_id)
                globalized_lowered_ids = set()
                for value_id in lowered_source_value_ids:
                    try:
                        globalized_lowered_ids.add(
                            hierarchy_values.global_id(closure_id, value_id)
                        )
                    except KeyError:
                        pass
                lowered_source_value_ids.update(globalized_lowered_ids)
            lowered_control.metadata.update({
                "source_conditional_count": (
                    lowered_conditional_count
                    + len(specialized_conditional_node_ids)
                ),
                "lowered_conditional_count": (
                    lowered_conditional_count
                    + len(specialized_conditional_node_ids)
                ),
                "specialized_conditional_node_ids": (
                    specialized_conditional_node_ids
                ),
                "source_output_value_ids": source_output_value_ids,
                "parameter_record_abi": copy.deepcopy(
                    graph_obj.graph.get("parameter_record_abi") or {}
                ),
                "parameter_value_abi": copy.deepcopy(
                    graph_obj.graph.get("parameter_value_abi") or {}
                ),
                "parameter_sequence_record_abi": copy.deepcopy(
                    graph_obj.graph.get("parameter_sequence_record_abi") or {}
                ),
                "record_sequence_projection_fields": (
                    record_sequence_projection_fields
                ),
                "record_sequence_projection_bindings": tuple(
                    record_sequence_bindings
                ),
                "extraction_materializations": tuple(
                    extraction_materializations
                ),
                "lowered_source_value_ids": tuple(sorted(
                    lowered_source_value_ids
                )),
            })
        all_tensor_tables.update(
            getattr(module_ir, "tensor_tables", {})
        )
        all_sequence_tables.update(
            getattr(module_ir, "sequence_tables", {})
        )
        all_record_tables.update(
            getattr(module_ir, "record_tables", {})
        )
        all_reference_tables.update(
            getattr(module_ir, "reference_tables", {})
        )
        pending_call_records.extend(
            (symbol, item, graph_obj, module_ir, shell)
            for item in local_plan_calls
        )
        # The whole-object module must retain the same class/member records
        # used to resolve the method closure.  They are an ABI description of
        # field slots and function references, not runtime object dispatch.
        if class_table is None and compilation.class_navigation is not None:
            from .precompile_to_ssa import lower_class_navigation_to_ssa

            class_table = lower_class_navigation_to_ssa(
                compilation.class_navigation
            ).class_table
        section_outputs.update(shell_section_outputs)
        export_symbols.append(symbol)
    if lowering_failures:
        raise FortranEmissionError(
            "whole-object methods have operators without an SSA handler: "
            + "; ".join(
                f"{symbol}::{item.location}::{item.name} ({item.reason})"
                for symbol, item in lowering_failures
            )
        )
    if not export_symbols:
        return None, {}, ()
    if class_table is not None and function_symbols:
        class_table = replace(
            class_table,
            classes=tuple(
                replace(
                    record,
                    methods=tuple(
                        replace(
                            method,
                            function_name=function_symbols.get(
                                int(method.function_reference)
                            ),
                        )
                        for method in record.methods
                    ),
                )
                for record in class_table.classes
            ),
        )
    # Persist token -> word so the emitted object's words are reversible.
    try:
        string_table.save()
    except Exception:  # noqa: BLE001 -- reverse-lookup cache, never fatal
        pass

    def emit_outputs(name: str, function: Any) -> tuple[Any, ...]:
        # A flat operator region has no explicit return: its outputs come from
        # the lowerer as ``intent(out)`` dummies the target appends. A control
        # function names its outputs with a return instruction.
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        if returns:
            return tuple(returns[-1])
        if name in section_outputs and section_outputs[name]:
            return section_outputs[name]
        return ()

    from ..transmogrifier.ssa import (
        Instr,
        SSAChildTablePoolDescriptor,
        SSACallRecord,
        SSARecordDescriptor,
        SSARecordFieldDescriptor,
        SSARecordFieldStorage,
        SSARecordInstancePoolDescriptor,
        SSARecordInstancePoolField,
        SSARecordTable,
        SSASequenceDescriptor,
        SSASequenceTable,
        SSAValue,
    )

    source_graphs_by_symbol = {
        shell_symbols.get(
            id(shell),
            f"{artifact_name}__{graph.graph.get('function_name')}",
        ): graph
        for shell in planned_shells
        for graph in (
            getattr(getattr(shell, "process_graph", None), "G", None),
        )
        if graph is not None and graph.graph.get("function_name") is not None
    }

    abi_records = dict(program_abi.get("records") or {})

    def abi_record_for_call(data: Mapping[str, Any]):
        if str(data.get("type") or data.get("op") or "").casefold() != "call":
            return None
        attributes = dict(data.get("attributes") or {})
        candidates = tuple(filter(None, (
            attributes.get("class_ref"),
            attributes.get("static_python_reference"),
        )))
        for record_name, record in abi_records.items():
            identity = str(record.get("identity") or record_name)
            if any(
                str(candidate) in {str(record_name), identity}
                or identity.endswith("." + str(candidate))
                for candidate in candidates
            ):
                return str(record_name), record
        return None

    # A value retained for a later source-linked call can be produced inside a
    # numerical region even though it is not a public function result.  The
    # control lowerer cannot see pending PlanCall consumers yet, so extend the
    # region aggregate from both the function's explicit source-output ledger
    # and every pending PlanCall feed before call linking.  A call is an
    # authored consumer just as surely as Ret is; omitting those feeds lets the
    # region lowerer prune values which the later call-frame linker then cannot
    # bind.  These projections are ordinary local SSA values; callers may
    # remove the hidden name from their final Ret afterward.
    pending_call_feed_ids: dict[str, set[int]] = {}
    for caller_symbol, planned_call, _graph, _module, _shell in (
        pending_call_records
    ):
        pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
            int(caller_id)
            for caller_id, _callee_id in planned_call.argument_bindings
        )
    # Schema-declared record literals are structural consumers even when the
    # external/dataclass constructor has no pursued method shell. Preserve
    # their authored positional/keyword feeds through numerical partitioning.
    for caller_symbol, graph in source_graphs_by_symbol.items():
        for _node_id, data in graph.nodes(data=True):
            if abi_record_for_call(data) is None:
                continue
            pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) != "callee"
            )
    for caller_symbol, caller in all_functions.items():
        pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
            map(int, caller.metadata.get("source_output_value_ids", ()))
        )
    # Keep an exact liveness receipt before expanding these seeds to their
    # dependency closure.  A numerical region can publish one of these values
    # before the source-linked Call is installed in its final control block.
    # The late pure-region DCE must therefore not decide that projection is
    # dead merely from the temporarily incomplete instruction use-list.  This
    # is semantic accounting, not a cache: the ids are the deterministic SSA
    # identities already present in the authored graph and are rebuilt on
    # every compilation.
    for caller_symbol, required_ids in pending_call_feed_ids.items():
        caller = all_functions.get(str(caller_symbol))
        if caller is not None and required_ids:
            caller.metadata["required_source_value_ids"] = tuple(
                sorted(map(int, required_ids))
            )
    # Public structural expressions (BoolOp, tuple/record construction, field
    # publication) are not themselves numerical regions. Retain their exact
    # operand closure so every numerical ancestor survives until structural
    # SSA reconstruction; stopping at the public node alone can prune the
    # second comparison in ``a and b`` or a constructor keyword constant.
    for caller_symbol, seeds in tuple(pending_call_feed_ids.items()):
        graph = source_graphs_by_symbol.get(str(caller_symbol))
        if graph is None:
            continue
        pending_call_feed_ids[str(caller_symbol)] = dependency_closure(
            graph, seeds
        )
    for caller_name, caller in all_functions.items():
        desired_ids = tuple(dict.fromkeys((
            *map(int, caller.metadata.get("source_output_value_ids", ())),
            *sorted(pending_call_feed_ids.get(str(caller_name), ())),
        )))
        if not desired_ids:
            continue
        # A region-call's OUT-params are pointers passed as ARGS, not as
        # instruction.res -- the call writes through them directly, the
        # same in-place mechanism a loop-carried phi's own update uses.
        # Checking only `caller.args` and `instr.res` treats an id already
        # satisfied this way as still "desired", so the aggregate-unpack
        # materialization below built a SECOND, competing producer for it
        # (a GetElementPtr+Load pair reading a bogus address) -- and
        # whichever one rendered last in the backend's id-keyed pointer
        # cache silently won, clobbering the call's real, correct write
        # with garbage. Any id already referenced as an operand anywhere in
        # this function already has a valid SSAValue for it in scope and
        # must not be re-materialized.
        available = {
            int(value.id) for value in caller.args
        } | {
            int(instruction.res.id)
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        } | {
            int(argument.id)
            for block in caller.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        next_projection_id = 1 + max(available, default=0)
        for desired_id in desired_ids:
            if desired_id in available:
                continue
            producer = None
            for block in caller.blocks.values():
                for index, instruction in enumerate(block.instrs):
                    if (
                        instruction.op not in {"Call", "call"}
                        or instruction.res is None
                        or instruction.attributes.get("region_index") is None
                        or instruction.attributes.get("source_linked")
                        or instruction.attributes.get("result_convention")
                        != "ssa.aggregate"
                    ):
                        continue
                    callee_symbol = str(
                        instruction.attributes.get("callee") or ""
                    )
                    callee = all_functions.get(callee_symbol)
                    if callee is None:
                        continue
                    # A callee's ADDRESS temporaries are not values it can
                    # publish: ``GetElementPtr`` names a location inside the
                    # callee's own storage, and its id is minted by the
                    # subscript lowering, not by the graph the caller shares.
                    # Accepting one here binds a caller scalar to a pointer
                    # into an array -- the defect that made the fluid advance
                    # read a height cell as ``tracer_diffusivity``. The id
                    # collision itself is now prevented upstream; this refuses
                    # the binding on its own terms as well.
                    produced_value = next((
                        candidate.res
                        for callee_block in callee.blocks.values()
                        for candidate in callee_block.instrs
                        if candidate.res is not None
                        and int(candidate.res.id) == desired_id
                        and str(candidate.op) != "GetElementPtr"
                    ), None)
                    if produced_value is not None:
                        producer = (
                            block, index, instruction,
                            callee_symbol, produced_value,
                        )
                        break
                if producer is not None:
                    break
            if producer is None:
                source = (
                    source_graphs_by_symbol.get(str(caller_name)).nodes.get(
                        int(desired_id), {}
                    )
                    if source_graphs_by_symbol.get(str(caller_name)) is not None
                    else {}
                )
                unresolved = list(caller.metadata.get(
                    "unresolved_required_source_values", ()
                ))
                unresolved.append((
                    int(desired_id),
                    str(source.get("op") or source.get("type") or ""),
                    tuple(
                        (int(parent), str(role))
                        for parent, role in source.get("parents") or ()
                    ),
                ))
                caller.metadata["unresolved_required_source_values"] = tuple(
                    dict.fromkeys(unresolved)
                )
                continue
            block, call_index, call, callee_symbol, produced_value = producer
            declared = list(map(
                int, call.attributes.get("output_ids", ())
            ))
            if desired_id in declared:
                continue
            output_index = len(declared)
            declared.append(desired_id)
            call.attributes["output_ids"] = tuple(declared)
            index_value = SSAValue(next_projection_id, dtype="int")
            next_projection_id += 1
            address = SSAValue(next_projection_id, dtype="ptr")
            next_projection_id += 1
            result = SSAValue(
                desired_id,
                dtype=produced_value.dtype,
                shape=tuple(produced_value.shape or ()),
                device=produced_value.device,
                accounting=dict(produced_value.accounting),
            )
            block.instrs[call_index + 1:call_index + 1] = [
                Instr("Const", [], index_value, attributes={"value": output_index}),
                Instr(
                    "GetElementPtr", [call.res, index_value], address,
                    attributes={
                        "aggregate_index": output_index,
                        "source_output_id": desired_id,
                    },
                ),
                Instr(
                    "Load", [address], result,
                    attributes={
                        "aggregate_index": output_index,
                        "source_output_id": desired_id,
                    },
                ),
            ]
            existing_outputs = tuple(section_outputs.get(callee_symbol, ()))
            section_outputs[callee_symbol] = (
                existing_outputs
                if desired_id in {int(value.id) for value in existing_outputs}
                else (*existing_outputs, produced_value)
            )
            available.add(desired_id)

    # Class construction is raw caller-owned storage plus an authored
    # constructor call.  The frontend already preserves every ``Class(...)``
    # node with its exact ``class_ref`` and every method call binds its receiver
    # value through PlanCall.  Materialize that missing correlation here, at
    # the same whole-program call-frame boundary that links ordinary calls.
    #
    # Each constructor occurrence receives a distinct set of arena ids.  The
    # stable field ``storage_identity`` (for example ``Store.table``) tells us
    # which callee field it implements; the receiver value tells us *which
    # instance*.  Consequently two Store() calls never become global/shared
    # storage, and no Python object or runtime dispatcher is introduced.
    constructor_calls: list[SSACallRecord] = []
    constructor_anchors: dict[tuple[str, int], int | None] = {}
    constructor_instance_pools: dict[
        tuple[str, int], dict[str, Any]
    ] = {}
    def function_values(function: Any) -> dict[int, Any]:
        values = {int(value.id): value for value in function.args}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    # Arguments are the canonical object for an authored ABI
                    # identity. A provisional synthetic result can briefly
                    # share its integer before the collision-freshening pass;
                    # it must not shadow the argument in record/call wiring.
                    values.setdefault(
                        int(instruction.res.id), instruction.res
                    )
        return values

    def recover_structural_source_outputs(
        symbol: str, graph: Any
    ) -> None:
        """Publish source results whose producer is structural, not numeric."""

        function = all_functions.get(symbol)
        if function is None:
            return
        returns = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ]
        if not returns:
            return
        terminator = returns[-1]
        published = {int(value.id) for value in terminator.args}
        values = function_values(function)
        insertions = []
        structural_shortfalls = []
        # Selection chains need their own intermediate results; a graph node id
        # only exists for the authored expression itself. These stay inside the
        # function's own SSA numbering -- graph node keys are Python object
        # identities and are not part of that space.
        next_structural_id = 1 + max((
            *values,
            *(
                int(data["value_id"])
                for _node_id, data in graph.nodes(data=True)
                if isinstance(data.get("value_id"), int)
            ),
        ), default=0)

        def structural_boolop_value(value_id: int, data, canonical: str):
            """Lower ``and``/``or`` as the operand selection Python defines.

            ``a or b`` evaluates to ``a`` when ``a`` is truthy and otherwise to
            ``b``; it is not the boolean ``a | b``. The two agree only when
            both operands are already boolean, which is the ordinary condition
            case, so that keeps its cheap logical opcode. Any other operand --
            a dict, a reference, a number -- must keep its own value and type,
            so it lowers to ``Select``, whose mask every backend already
            resolves through the same truthiness rule it uses elsewhere.
            Returns ``None`` to let the caller fall through to the boolean
            opcode.
            """

            ordered = []
            for parent, role in data.get("parents") or ():
                role = str(role)
                if role == "callee":
                    continue
                # BoolOp operands are ordered edges; `or` is not commutative
                # over values, so the authored order is the exact order.
                index = (
                    int(role.split(":", 1)[1])
                    if role.startswith("value:") and role.split(":", 1)[1].isdigit()
                    else len(ordered)
                )
                ordered.append((index, int(parent)))
            ordered.sort()
            if len(ordered) < 2:
                return None
            operands = []
            for _index, parent in ordered:
                operand = ensure_structural_value(parent)
                if operand is None:
                    return None
                operands.append(operand)

            def destroys_value(operand) -> bool:
                # A declared container or reference has no boolean form at all:
                # combining it yields a truth value and the dict, span, or
                # record it named is gone. A declared non-boolean scalar keeps
                # its own value too. An operand whose type is still unknown is
                # left to the boolean opcode rather than guessed at here.
                storage = str(
                    (operand.accounting or {}).get("program_abi_storage") or ""
                )
                if storage in {"reference", "span", "record", "keyed"}:
                    return True
                dtype = str(getattr(operand, "dtype", "") or "").casefold()
                return bool(dtype) and dtype not in {"bool", "unknown"}

            if not any(destroys_value(operand) for operand in operands):
                return None

            nonlocal next_structural_id
            current = operands[0]
            for position, operand in enumerate(operands[1:], start=1):
                last = position == len(operands) - 1
                if last:
                    result_id = value_id
                else:
                    result_id = next_structural_id
                    next_structural_id += 1
                dtype = current.dtype or operand.dtype
                result = SSAValue(int(result_id), dtype=dtype)
                # Select(mask, when_true, when_false). `or` keeps the left
                # operand when it is truthy; `and` keeps the right one.
                arguments = (
                    [current, current, operand]
                    if canonical == "logical_or"
                    else [current, operand, current]
                )
                insertions.append(Instr(
                    "Select", arguments, result,
                    attributes={
                        "structural_operation": "boolop",
                        "semantic_family": canonical,
                        "short_circuit_selection": True,
                    },
                ))
                values[int(result_id)] = result
                current = result
            return current

        def ensure_structural_value(value_id: int):
            """Lower a missing direct expression from its exact graph edges."""

            value_id = int(value_id)
            if value_id in values:
                return values[value_id]
            data = graph.nodes.get(value_id, {})
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            attributes = dict(data.get("attributes") or {})
            if operation in {"constant", "const"}:
                expression = data.get("expr_obj")
                if (
                    "value" not in attributes
                    and "constant" not in data
                    and not isinstance(expression, ast.Constant)
                ):
                    structural_shortfalls.append((value_id, operation, "constant-value"))
                    return None
                literal = attributes.get(
                    "value",
                    data.get(
                        "constant",
                        expression.value
                        if isinstance(expression, ast.Constant) else None,
                    ),
                )
                dtype = (
                    "bool" if isinstance(literal, bool)
                    else "int64" if isinstance(literal, int)
                    else "float64" if isinstance(literal, float)
                    else None
                )
                result = SSAValue(value_id, dtype=dtype)
                insertions.append(Instr(
                    "Const", [], result, attributes={"value": literal},
                ))
                values[value_id] = result
                return result
            if operation in {"loopresult", "loopexit", "identity"}:
                parents = tuple(data.get("parents") or ())
                for preferred_role in (
                    "updated", "value", "body", "initial", "orelse"
                ):
                    for parent, role in parents:
                        if str(role) != preferred_role:
                            continue
                        result = ensure_structural_value(int(parent))
                        if result is not None:
                            values[value_id] = result
                            return result
                structural_shortfalls.append((
                    value_id, operation, "carried-value"
                ))
                return None
            if operation in {"int", "float", "bool"}:
                operands = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role).startswith("arg:")
                )
                if len(operands) != 1:
                    structural_shortfalls.append((
                        value_id, operation, "cast-arity"
                    ))
                    return None
                operand = ensure_structural_value(operands[0])
                if operand is None:
                    structural_shortfalls.append((
                        value_id, operation, f"operand:{operands[0]}"
                    ))
                    return None
                target_dtype = {
                    "int": "int64", "float": "float64", "bool": "bool",
                }[operation]
                result = SSAValue(value_id, dtype=target_dtype)
                insertions.append(Instr(
                    "Cast", [operand], result,
                    attributes={
                        "structural_operation": operation,
                        "target_dtype": target_dtype,
                    },
                ))
                values[value_id] = result
                return result
            canonical = {
                "add": "add", "sub": "sub", "mul": "mul",
                "div": "truediv", "truediv": "truediv",
                "greater": "greater", "gt": "greater",
                "less": "less", "lt": "less",
                "greaterequal": "greater_equal",
                "greater_equal": "greater_equal",
                "lessequal": "less_equal", "less_equal": "less_equal",
                "equal": "equal", "eq": "equal",
                "notequal": "not_equal", "not_equal": "not_equal",
            }.get(operation)
            expression = data.get("expr_obj")
            if operation == "boolop":
                canonical = (
                    "logical_and"
                    if isinstance(getattr(expression, "op", None), ast.And)
                    else "logical_or"
                    if isinstance(getattr(expression, "op", None), ast.Or)
                    else None
                )
                if canonical is not None:
                    selected = structural_boolop_value(
                        value_id, data, canonical,
                    )
                    if selected is not None:
                        return selected
            if canonical is None:
                structural_shortfalls.append((value_id, operation, "operator"))
                return None
            from .ssa_numeric_operators import TENSOR_SSA_OPERATOR_BY_NAME

            row = TENSOR_SSA_OPERATOR_BY_NAME.get(canonical)
            if row is None or not row.is_direct:
                structural_shortfalls.append((value_id, operation, "direct-handler"))
                return None
            arguments = []
            for parent, role in data.get("parents") or ():
                if str(role) == "callee":
                    continue
                argument = ensure_structural_value(int(parent))
                if argument is None:
                    structural_shortfalls.append((
                        value_id, operation, f"operand:{int(parent)}"
                    ))
                    return None
                arguments.append(argument)
            if len(arguments) < 1:
                structural_shortfalls.append((value_id, operation, "arity"))
                return None
            dtype = (
                "bool" if canonical in {
                    "logical_and", "logical_or", "equal", "not_equal",
                    "less", "less_equal", "greater", "greater_equal",
                } else arguments[0].dtype
            )
            result = SSAValue(value_id, dtype=dtype)
            insertions.append(Instr(
                row.handler.value,
                arguments,
                result,
                attributes={
                    "structural_operation": operation,
                    "semantic_family": canonical,
                },
            ))
            values[value_id] = result
            return result

        def literal_value(node_id: int) -> Any:
            data = graph.nodes.get(int(node_id), {})
            attributes = data.get("attributes") or {}
            if "value" in attributes:
                return _copy_literal_payload(attributes["value"])
            if "constant" in data:
                payload = data["constant"]
                # Every graph-express node is born with constant=None (see
                # ProcessGraph.add_node), so the key's presence alone proves
                # nothing. Reading it unconditionally short-circuited BEFORE
                # the ast.literal_eval / list-tuple recursion fallbacks
                # below ever ran, for every node that is not itself typed
                # Const/Constant -- a List/Tuple node whose own elements are
                # all resolvable constants (the exact case those fallbacks
                # exist for) was silently declared unresolvable instead of
                # being recursed into. A None payload only counts as a
                # literal on a node that declares itself constant.
                if payload is not None or str(
                    data.get("type")
                ) in {"Constant", "Const", "const"} or str(
                    data.get("op") or ""
                ).casefold() == "const":
                    return _copy_literal_payload(payload)
            expression = data.get("expr_obj")
            if expression is not None:
                try:
                    return ast.literal_eval(expression)
                except (TypeError, ValueError):
                    pass
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            if operation in {"list", "tuple"}:
                return [
                    literal_value(int(parent))
                    for parent, role in data.get("parents") or ()
                    if str(role) == "elts"
                ]
            raise ValueError(f"node {node_id} is not a literal")

        def authored_output_literal(output_name: str) -> Any:
            if source_function_table is None:
                raise ValueError(output_name)
            function_reference = graph.graph.get("function_ref")
            try:
                entry = source_function_table.entry(int(function_reference))
            except (KeyError, TypeError, ValueError):
                raise ValueError(output_name) from None
            callable_object = getattr(entry, "python_callable", None)
            if callable_object is None:
                raise ValueError(output_name)
            try:
                tree = ast.parse(textwrap.dedent(
                    inspect.getsource(callable_object)
                ))
            except (OSError, TypeError, IndentationError, SyntaxError):
                raise ValueError(output_name) from None
            for node in ast.walk(tree):
                target = None
                value = None
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        tuple(node.targets)
                        if isinstance(node, ast.Assign)
                        else (node.target,)
                    )
                    if any(
                        isinstance(candidate, ast.Name)
                        and candidate.id == output_name
                        for candidate in targets
                    ):
                        target = output_name
                        value = node.value
                if target is None or value is None:
                    continue
                candidate = (
                    value.args[0]
                    if isinstance(value, ast.Call) and value.args
                    else value
                )
                try:
                    return ast.literal_eval(candidate)
                except (TypeError, ValueError):
                    continue
            raise ValueError(output_name)

        identities = graph.graph.get("identity_table") or {}
        record_table = all_record_tables.get(symbol)
        returned_record_layouts = []
        named_output_ids = {}
        for name, value_id in function.metadata.get("named_outputs") or ():
            name = str(name)
            value_id = int(value_id)
            history = tuple(map(int, identities.get(name, ())))
            # A named-output hint sometimes still points at the authored
            # parameter Input. The identity ledger is the source of truth:
            # when the hint is one member of that deterministic SSA chain,
            # its final member is the current spelling. A carried/region phi
            # can legitimately sit outside the lexical history, so preserve
            # such a resolved value unchanged.
            named_output_ids[name] = (
                history[-1] if history and value_id in history else value_id
            )
        for output_name in tuple(graph.graph.get("function_outputs") or ()):
            history = tuple(identities.get(str(output_name), ()))
            if not history:
                continue
            output_id = int(history[-1])
            if output_id in published:
                continue
            data = graph.nodes.get(output_id, {})
            record = (
                None if record_table is None
                else record_table.records.get(output_id)
            )
            if record is not None:
                matched = abi_record_for_call(data)
                if matched is not None:
                    _record_name, contract_record = matched
                    existing_fields = {field.name for field in record.fields}
                    keyword_values = {
                        str(role).split(":", 1)[1]: int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role).startswith("kw:")
                    }
                    appended_fields = []
                    for field_name, field in dict(
                        contract_record.get("fields") or {}
                    ).items():
                        if field_name in existing_fields:
                            continue
                        source_id = keyword_values.get(str(field_name))
                        source = (
                            None if source_id is None
                            else ensure_structural_value(source_id)
                        )
                        if source is None:
                            continue
                        if str(field["storage"]) == "keyed":
                            # Three physical slots, correlated from the mapping
                            # literal's own key/value edges -- see the
                            # constructor-literal path.
                            continue
                        storage = {
                            "scalar": SSARecordFieldStorage.SCALAR,
                            "span": SSARecordFieldStorage.SPAN,
                            "reference": SSARecordFieldStorage.REFERENCE,
                            "record": SSARecordFieldStorage.RECORD,
                        }[str(field["storage"])]
                        if storage is SSARecordFieldStorage.RECORD:
                            continue
                        appended_fields.append(SSARecordFieldDescriptor(
                            str(field_name), storage,
                            storage_identity=(
                                f"{contract_record['identity']}.{field_name}"
                            ),
                            value_ids=(int(source.id),),
                            dtype=field.get("dtype"),
                            writable=bool(field.get("mutable", False)),
                        ))
                    if appended_fields:
                        record = replace(
                            record, fields=(*record.fields, *appended_fields)
                        )
                        record_table.records[output_id] = record
                layout = tuple(
                    int(value_id)
                    for field in record.fields
                    for value_id in field.value_ids
                    if int(value_id) in values
                )
                for value_id in layout:
                    if value_id not in published:
                        terminator.args.append(values[value_id])
                        published.add(value_id)
                returned_record_layouts.append((output_id, layout))
                continue
            reconstructed = ensure_structural_value(output_id)
            if reconstructed is not None:
                terminator.args.append(reconstructed)
                published.add(output_id)
                continue
            if output_id in values:
                terminator.args.append(values[output_id])
                published.add(output_id)
                continue
            attributes = data.get("attributes") or {}
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            result = None
            instruction = None
            if (
                operation == "input"
                and str(output_name) not in set(map(
                    str, graph.graph.get("function_parameters") or ()
                ))
            ):
                try:
                    literal = authored_output_literal(str(output_name))
                except ValueError:
                    literal = None
                if literal is not None:
                    array = np.asarray(literal)
                    result = SSAValue(
                        output_id,
                        dtype=str(array.dtype),
                        shape=tuple(map(int, array.shape)),
                        accounting={
                            "authored_output_literal": str(output_name)
                        },
                    )
                    instruction = Instr(
                        "Const", [], result,
                        attributes={
                            "value": literal,
                            "values": literal,
                            "tensor_operation": "tensor_from_list",
                        },
                    )
                else:
                    tensor = data.get("tensor") or {}
                    result = SSAValue(
                        output_id,
                        dtype=tensor.get("dtype"),
                        shape=tuple(tensor.get("shape") or ()),
                        accounting={
                            "externalized_source_output": str(output_name)
                        },
                    )
                    function.args.append(result)
                    values[output_id] = result
                    terminator.args.append(result)
                    published.add(output_id)
                    continue
            elif operation == "_tensor_from_list":
                data_parent = next((
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "arg:0"
                ), None)
                if data_parent is not None:
                    try:
                        literal = literal_value(data_parent)
                    except ValueError:
                        literal = None
                    if literal is not None:
                        array = np.asarray(literal)
                        result = SSAValue(
                            output_id,
                            dtype=str(array.dtype),
                            shape=tuple(map(int, array.shape)),
                            accounting={
                                "tensor_constructor": "tensor_from_list",
                                "requires_grad": bool(attributes.get(
                                    "requires_grad", False
                                )),
                            },
                        )
                        instruction = Instr(
                            "Const", [], result,
                            attributes={
                                "value": literal,
                                "values": literal,
                                "tensor_operation": "tensor_from_list",
                            },
                        )
            elif (
                operation == "call"
                and attributes.get("static_python_reference") == "id"
            ):
                # `id(x)` takes one argument: argument ZERO, not whichever
                # positional edge the parent set yields first.
                arguments = ordered_arguments(data.get("parents") or ())
                source_id = int(arguments[0]) if arguments else None
                if source_id is not None and source_id in values:
                    result = SSAValue(
                        output_id,
                        dtype="int64",
                        accounting={"tensor_identity": "stable-handle"},
                    )
                    instruction = Instr(
                        "Cast", [values[source_id]], result,
                        attributes={
                            "tensor_operation": "tensor_identity",
                            "reference_cast": True,
                        },
                    )
            elif operation == "boolop":
                operands = [
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role).startswith("value:")
                    and int(parent) in values
                ]
                expression = data.get("expr_obj")
                opcode = (
                    "And" if isinstance(getattr(expression, "op", None), ast.And)
                    else "Or" if isinstance(
                        getattr(expression, "op", None), ast.Or
                    ) else None
                )
                if opcode is not None and len(operands) >= 2:
                    current = values[operands[0]]
                    for operand_id in operands[1:]:
                        is_last = operand_id == operands[-1]
                        combined = SSAValue(
                            output_id if is_last else max(values) + 1,
                            dtype="bool",
                        )
                        insertions.append(Instr(
                            opcode, [current, values[operand_id]], combined,
                            attributes={"structural_operation": "boolop"},
                        ))
                        values[int(combined.id)] = combined
                        current = combined
                    result = current
                    # The instruction sequence was already appended above.
                    instruction = None
                    terminator.args.append(result)
                    published.add(output_id)
            elif operation == "ifexp":
                parameter_id = next((
                    int(value_id)
                    for name in tuple(
                        graph.graph.get("function_parameters") or ()
                    )
                    for value_id in tuple(identities.get(str(name), ()))[:1]
                ), None)
                parameter = values.get(parameter_id)
                shape = tuple(getattr(parameter, "shape", ()) or ())
                if shape and all(int(extent) >= 0 for extent in shape):
                    result = SSAValue(output_id, dtype="int64")
                    instruction = Instr(
                        "Const", [], result,
                        attributes={
                            "value": int(np.prod(shape, dtype=np.int64)),
                            "structural_operation": "nested_count",
                        },
                    )
            if output_id in published:
                continue
            if instruction is None or result is None:
                continue
            insertions.append(instruction)
            values[output_id] = result
            terminator.args.append(result)
            published.add(output_id)
        # Structural expressions can also be private feeds to a later
        # source-linked call.  They need the same exact reconstruction as a
        # public Ret value, but must not be added to Ret merely because a call
        # consumes them.  This is common for authored boolean combinations
        # passed as keyword arguments.
        authored_parameter_ids = {
            int(value_id)
            for _name, value_id in function.metadata.get(
                "parameter_names", ()
            )
        }
        for required_id in sorted(pending_call_feed_ids.get(symbol, ())):
            required_id = int(required_id)
            if required_id in values:
                existing = values[required_id]
                # Region projection can provisionally externalize a value
                # which is actually a structural source expression.  If the
                # projection is later folded away, retaining that placeholder
                # turns an internal expression into a public ABI argument
                # (observed for ``sleb(int(data_offset))``).  Claim only an
                # exact planned-region result, never an authored parameter or
                # physical program-ABI slot.  Prefer a real same-identity
                # instruction result when one already exists; otherwise let
                # the structural graph reconstruction below emit its producer.
                accounting = dict(existing.accounting or {})
                claimable = (
                    existing in function.args
                    and required_id not in authored_parameter_ids
                    and bool(accounting.get("ssa_call_result_from"))
                    and not accounting.get("program_abi_storage")
                    and not accounting.get("compiler_frame_storage")
                    and not accounting.get("linked_call_frame_storage")
                )
                if not claimable:
                    continue
                function.args.remove(existing)
                produced = next((
                    instruction.res
                    for block in function.blocks.values()
                    for instruction in block.instrs
                    if instruction.res is not None
                    and int(instruction.res.id) == required_id
                ), None)
                if produced is not None:
                    values[required_id] = produced
                    continue
                values.pop(required_id, None)
            operation = str(
                graph.nodes.get(required_id, {}).get("op")
                or graph.nodes.get(required_id, {}).get("type")
                or ""
            ).casefold()
            if operation in {
                "boolop", "constant", "const", "loopresult", "loopexit",
                "identity", "int", "float", "bool",
                "add", "sub", "mul", "div", "truediv",
                "greater", "gt", "less", "lt", "greaterequal",
                "greater_equal", "lessequal", "less_equal", "equal", "eq",
                "notequal", "not_equal",
            }:
                ensure_structural_value(required_id)
        # A named output the builder already resolved through its name
        # history stands in the Ret under the carried phi's id, which the
        # graph's identity table cannot know -- so the recovery above may
        # have re-published the same authored output under a STALE earlier
        # identity (scorecard level 17: (0.5, 0.5) for 0.5). The recovery
        # itself must run (its structural insertions and ``values`` entries
        # feed later source-linked calls); only the duplicate Ret argument
        # is dropped: any arg whose id is a non-final member of a named
        # output's identity history while that output's resolved value is
        # already present.
        argument_ids = {int(argument.id) for argument in terminator.args}
        stale_identities: set[int] = set()
        for output_name, named_id in named_output_ids.items():
            if named_id not in argument_ids:
                continue
            history = tuple(map(int, identities.get(str(output_name), ())))
            stale_identities.update(
                identity for identity in history if identity != named_id
            )
        stale_identities -= set(named_output_ids.values())
        if stale_identities:
            terminator.args = [
                argument for argument in terminator.args
                if int(argument.id) not in stale_identities
            ]
        if insertions:
            for block in function.blocks.values():
                if terminator in block.instrs:
                    index = block.instrs.index(terminator)
                    block.instrs[index:index] = insertions
                    break
            function.metadata["recovered_structural_outputs"] = tuple(
                int(instruction.res.id) for instruction in insertions
            )
        if returned_record_layouts:
            function.metadata["record_return_layouts"] = tuple(
                returned_record_layouts
            )
        if structural_shortfalls:
            function.metadata["structural_output_shortfalls"] = tuple(
                dict.fromkeys(structural_shortfalls)
            )

    # Determine the least record surface required by the whole source-linked
    # call graph.  A function that only forwards ``state`` has no local
    # GetAttr node, but it must still carry exactly the fields read by its
    # descendants.  Propagating these names backward through PlanCall's exact
    # argument bindings avoids both Python object handles and the rejected
    # alternative of expanding every schema field at every call boundary.
    record_parameter_specs: dict[tuple[str, str], Mapping[str, Any]] = {}
    record_parameter_by_value: dict[str, dict[int, tuple[str, str]]] = {}
    record_field_demands: dict[tuple[str, str], set[str]] = {}
    # Which fields a function's OWN body writes locally (a direct SetAttr).
    # ``record_field_demands`` (reads) is forwarded transitively below via
    # ``record_forwarding_edges`` so a deep callee's need reaches every
    # caller; writes need the identical treatment, or a caller several
    # calls above the actual mutation sees its own post-call read of a
    # mutable field as an ordinary, never-written value -- materializing a
    # second, disconnected "input" copy that is permanently stuck at the
    # field's pre-call snapshot while the real, correctly-threaded mutation
    # flows through a separate value the caller's own read never resolves
    # to (observed: ``last_wave_speed``/``last_height_violation``/
    # ``last_tracer_violation``, each written only inside a deeply nested
    # callee, stayed at their initial 0.0 in the compiled output).
    record_field_writes: dict[tuple[str, str], set[str]] = {}
    for source_symbol, source_graph in source_graphs_by_symbol.items():
        identities = source_graph.graph.get("identity_table") or {}
        declared = dict(
            source_graph.graph.get("parameter_record_abi") or {}
        )
        by_value = record_parameter_by_value.setdefault(source_symbol, {})
        for parameter_name, record in declared.items():
            key = (str(source_symbol), str(parameter_name))
            record_parameter_specs[key] = record
            record_field_demands.setdefault(key, set())
            record_field_writes.setdefault(key, set())
            parameter_ids = set(map(
                int, identities.get(str(parameter_name), ())
            ))
            for value_id in parameter_ids:
                by_value[int(value_id)] = key
            declared_fields = set(map(
                str, dict(record.get("fields") or {})
            ))
            for node_id, data in source_graph.nodes(data=True):
                operation = str(
                    data.get("type") or data.get("op") or ""
                ).casefold()
                if operation not in {"getattr", "setattr"}:
                    continue
                attribute = str(
                    (data.get("attributes") or {}).get("attribute") or ""
                )
                if attribute not in declared_fields:
                    continue
                roles = (
                    {"value", "object", "base"} if operation == "getattr"
                    else {"value", "object", "base", "receiver"}
                )
                if not any(
                    int(parent) in parameter_ids and str(role) in roles
                    for parent, role in data.get("parents") or ()
                ):
                    continue
                if operation == "getattr":
                    record_field_demands[key].add(attribute)
                else:
                    record_field_writes[key].add(attribute)

    record_forwarding_edges = []
    for caller_symbol, planned_call, caller_graph, _module, caller_shell in (
        pending_call_records
    ):
        call_data = caller_graph.nodes.get(
            int(planned_call.callsite_id), {}
        )
        attributes = call_data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref",
            attributes.get("method_ref", attributes.get("constructor_ref")),
        )
        child_shell = getattr(
            caller_shell, "callsite_function_shells", {}
        ).get(int(planned_call.callsite_id))
        callee_symbol = (
            shell_symbols.get(id(child_shell))
            if child_shell is not None else None
        ) or (
            None if reference is None
            else function_symbols.get(int(reference))
        )
        if callee_symbol is None:
            continue
        caller_by_value = record_parameter_by_value.get(
            str(caller_symbol), {}
        )
        callee_by_value = record_parameter_by_value.get(
            str(callee_symbol), {}
        )
        for caller_id, callee_id in planned_call.argument_bindings:
            caller_key = caller_by_value.get(int(caller_id))
            callee_key = callee_by_value.get(int(callee_id))
            if caller_key is None or callee_key is None:
                continue
            caller_record = record_parameter_specs[caller_key]
            callee_record = record_parameter_specs[callee_key]
            if str(caller_record.get("identity")) != str(
                callee_record.get("identity")
            ):
                continue
            record_forwarding_edges.append((caller_key, callee_key))

    changed = True
    while changed:
        changed = False
        for caller_key, callee_key in record_forwarding_edges:
            missing = (
                record_field_demands[callee_key]
                - record_field_demands[caller_key]
            )
            if missing:
                record_field_demands[caller_key].update(missing)
                changed = True
            missing_writes = (
                record_field_writes.get(callee_key, set())
                - record_field_writes.get(caller_key, set())
            )
            if missing_writes:
                record_field_writes.setdefault(caller_key, set()).update(
                    missing_writes
                )
                changed = True

    def materialize_parameter_record_abi(symbol: str, graph: Any) -> None:
        """Make contract-declared record fields ordinary physical SSA inputs.

        Read-only scalar views are passed by value and spans are passed as
        arenas. A scalar field actually written by the function remains
        unresolved until its reference/slot ABI is available; passing that
        field by value would silently destroy state updates across a call.
        """

        function = all_functions.get(symbol)
        if function is None:
            return
        declared_records = dict(graph.graph.get("parameter_record_abi") or {})
        if not declared_records:
            return
        identities = graph.graph.get("identity_table") or {}
        values = function_values(function)
        next_physical_id = 1 + max((
            *values,
            *(int(data.get("value_id", node_id))
              for node_id, data in graph.nodes(data=True)),
        ), default=0)
        table = all_record_tables.setdefault(symbol, SSARecordTable())
        pooled_scalar_columns: dict[tuple[str, str], SSAValue] = {}

        def record_field_candidates(
            owner_ids: set[int], field_name: str,
        ) -> tuple[int, ...]:
            """Find attribute or constant-key ``record.get`` field reads."""

            candidates = []
            for node_id, data in graph.nodes(data=True):
                operation = str(
                    data.get("type") or data.get("op") or ""
                ).casefold()
                parents = tuple(data.get("parents") or ())
                if operation == "getattr" and str(
                    (data.get("attributes") or {}).get("attribute")
                ) == str(field_name) and any(
                    int(parent) in owner_ids
                    and str(role) in {"value", "object", "base"}
                    for parent, role in parents
                ):
                    candidates.append(int(data.get("value_id", node_id)))
                    continue
                if operation != "get" or not any(
                    int(parent) in owner_ids
                    and str(role) in {"operand", "value", "object", "base"}
                    for parent, role in parents
                ):
                    continue
                expression = data.get("expr_obj")
                key = (
                    expression.args[0].value
                    if isinstance(expression, ast.Call)
                    and expression.args
                    and isinstance(expression.args[0], ast.Constant)
                    else None
                )
                if str(key) == str(field_name):
                    candidates.append(int(data.get("value_id", node_id)))
            return tuple(dict.fromkeys(candidates))

        def indexed_value_candidates(owner_ids: set[int]) -> tuple[int, ...]:
            """Find row values selected from one declared keyed field."""

            candidates = []
            for node_id, data in graph.nodes(data=True):
                operation = str(
                    data.get("type") or data.get("op") or ""
                ).casefold()
                if operation != "indexed":
                    continue
                if not any(
                    int(graph.nodes[parent].get("value_id", parent))
                    in owner_ids
                    and str(role) in {
                        "value", "base", "operand", "object",
                    }
                    for parent, role in (data.get("parents") or ())
                    if parent in graph
                ):
                    continue
                candidates.append(int(data.get("value_id", node_id)))
            return tuple(dict.fromkeys(candidates))

        def accepted_isinstance_tokens(expression: Any) -> tuple[str, ...]:
            """Return exact lexical type identities from isinstance arg 2."""

            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "isinstance"
                and len(expression.args) >= 2
            ):
                return ()

            def identity(node: ast.AST) -> str | None:
                if isinstance(node, ast.Name):
                    return {
                        "tuple": "builtins.tuple",
                        "list": "builtins.list",
                        "set": "builtins.set",
                        "dict": "builtins.dict",
                    }.get(node.id, node.id)
                if isinstance(node, ast.Attribute):
                    owner = identity(node.value)
                    return None if owner is None else f"{owner}.{node.attr}"
                return None

            accepted = expression.args[1]
            members = (
                tuple(accepted.elts)
                if isinstance(accepted, (ast.Tuple, ast.List, ast.Set))
                else (accepted,)
            )
            return tuple(filter(None, (identity(member) for member in members)))

        def materialize_nested_record(
            schema_name: str,
            owner_ids: set[int],
            descriptor_id: int,
            parameter_name: str,
            field_path: str,
            active: tuple[str, ...] = (),
            row_handle_id: int | None = None,
        ) -> None:
            """Expand one schema-known nested record into physical leaves.

            The IDs are the deterministic source/SSA identities already on
            the GetAttr chain.  A record container never receives a transient
            slot number and never crosses the native ABI; only its scalar,
            span, or reference leaves do.
            """

            nonlocal next_physical_id

            if schema_name in active:
                raise ValueError(
                    "cyclic nested program ABI record "
                    f"{' -> '.join((*active, schema_name))}"
                )
            try:
                schema = dict(abi_records[str(schema_name)])
            except KeyError as error:
                raise ValueError(
                    f"unknown nested program ABI record {schema_name!r}"
                ) from error
            schema_identity = str(schema.get("identity") or schema_name)
            nested_fields = []
            for nested_name, nested_field in dict(
                schema.get("fields") or {}
            ).items():
                candidates = record_field_candidates(
                    owner_ids, str(nested_name)
                )
                if not candidates:
                    continue
                nested_storage = str(nested_field.get("storage") or "")
                nested_path = f"{field_path}.{nested_name}"
                if nested_storage == "record":
                    child_schema = str(nested_field.get("record") or "")
                    child_id = min(candidates)
                    materialize_nested_record(
                        child_schema,
                        set(map(int, candidates)),
                        child_id,
                        parameter_name,
                        nested_path,
                        (*active, schema_name),
                        row_handle_id,
                    )
                    child_receipt = dict(abi_records[child_schema])
                    nested_fields.append(SSARecordFieldDescriptor(
                        str(nested_name),
                        SSARecordFieldStorage.RECORD,
                        storage_identity=f"{schema_identity}.{nested_name}",
                        value_ids=(),
                        record_id=child_id,
                        dtype=str(
                            child_receipt.get("identity") or child_schema
                        ),
                        writable=False,
                    ))
                    continue
                if nested_storage == "keyed":
                    key_encoding = str(
                        nested_field.get("key_encoding") or "string_token"
                    )
                    value_record = nested_field.get("value_record")
                    value_identity = nested_field.get("value_identity")
                    parts = {
                        "length": ("scalar", "int64", 0),
                        "keys": ("span", "int64", 1),
                        **({} if value_identity == "key" else {
                            "values": (
                                "span",
                                str(nested_field.get("dtype") or "float64"),
                                1,
                            ),
                        }),
                    }
                    part_ids: dict[str, int] = {}
                    for part_name, (
                        part_storage, part_dtype, part_rank,
                    ) in parts.items():
                        part_id = next_physical_id
                        next_physical_id += 1
                        part_ids[part_name] = part_id
                        part_value = SSAValue(
                            part_id,
                            dtype=part_dtype,
                            accounting={
                                "program_abi_record": schema_identity,
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": (
                                    f"{nested_path}.{part_name}"
                                ),
                                "program_abi_storage": part_storage,
                                "program_abi_rank": part_rank,
                                "program_abi_mutable": bool(
                                    nested_field.get("mutable", False)
                                ),
                                "program_abi_field_written": False,
                                "program_abi_keyed_owner": nested_path,
                                "program_abi_keyed_part": part_name,
                                "program_abi_key_encoding": key_encoding,
                                "program_abi_value_record": value_record,
                            },
                        )
                        function.args.append(part_value)
                        values[part_id] = part_value
                        nested_fields.append(SSARecordFieldDescriptor(
                            f"{nested_name}.{part_name}",
                            SSARecordFieldStorage.SCALAR
                            if part_storage == "scalar"
                            else SSARecordFieldStorage.SPAN,
                            storage_identity=(
                                f"{schema_identity}.{nested_name}."
                                f"{part_name}"
                            ),
                            value_ids=(part_id,),
                            dtype=part_dtype,
                            writable=bool(nested_field.get("mutable", False)),
                        ))
                    if value_identity == "key":
                        part_ids["values"] = part_ids["keys"]
                    for value_id in candidates:
                        mapping = values.get(int(value_id))
                        if mapping is None:
                            mapping = SSAValue(int(value_id))
                            function.args.append(mapping)
                            values[int(value_id)] = mapping
                        mapping.accounting = {
                            **dict(mapping.accounting or {}),
                            "program_abi_record": schema_identity,
                            "program_abi_parameter": str(parameter_name),
                            "program_abi_field": nested_path,
                            "program_abi_storage": "keyed",
                            "program_abi_keyed_length": part_ids["length"],
                            "program_abi_keyed_keys": part_ids["keys"],
                            "program_abi_keyed_values": part_ids["values"],
                            "program_abi_key_encoding": key_encoding,
                            "program_abi_value_record": value_record,
                            "program_abi_value_identity": value_identity,
                        }
                    if value_record is not None:
                        for row_id in indexed_value_candidates(set(candidates)):
                            materialize_nested_record(
                                str(value_record), {int(row_id)}, int(row_id),
                                parameter_name, f"{nested_path}[]",
                                (*active, schema_name),
                                int(row_id),
                            )
                    continue
                nested_dtype = nested_field.get("dtype")
                nested_rank = int(nested_field.get("rank", 0))
                mutable = bool(nested_field.get("mutable", False))
                token_vocabulary = tuple(map(
                    str, nested_field.get("token_vocabulary") or (),
                ))
                physical_ids = []
                if nested_storage == "table" and row_handle_id is not None:
                    columns = tuple(
                        dict(column)
                        for column in nested_field.get("columns") or ()
                    )
                    reachable = set(map(int, candidates))
                    changed = True
                    while changed:
                        changed = False
                        for candidate_node, candidate_data in graph.nodes(
                            data=True
                        ):
                            operation = str(
                                candidate_data.get("type")
                                or candidate_data.get("op") or ""
                            ).casefold()
                            if operation not in {"boolop", "phi"}:
                                continue
                            if not any(
                                int(graph.nodes[parent].get(
                                    "value_id", parent
                                )) in reachable
                                for parent, _role in (
                                    candidate_data.get("parents") or ()
                                )
                                if parent in graph
                            ):
                                continue
                            value_id = int(candidate_data.get(
                                "value_id", candidate_node
                            ))
                            if value_id not in reachable:
                                reachable.add(value_id)
                                changed = True
                    projected = {
                        int((value.accounting or {})[
                            "projected_row_source_id"
                        ]): []
                        for value in function.args
                        if (value.accounting or {}).get(
                            "projected_row_source_id"
                        ) is not None
                    }
                    for value in function.args:
                        accounting = value.accounting or {}
                        source_id = accounting.get("projected_row_source_id")
                        if source_id is None:
                            continue
                        projected.setdefault(int(source_id), []).append((
                            int(accounting.get("projected_row_column") or 0),
                            value,
                        ))
                    sequence_id = next((
                        source_id for source_id in projected
                        if source_id in reachable
                    ), None)
                    if sequence_id is not None and columns:
                        column_arenas = []
                        for column in columns:
                            arena = SSAValue(
                                next_physical_id,
                                dtype=str(column["dtype"]),
                                accounting={
                                    "program_abi_record": schema_identity,
                                    "program_abi_parameter": str(
                                        parameter_name
                                    ),
                                    "program_abi_field": (
                                        f"{nested_path}.columns."
                                        f"{column['name']}"
                                    ),
                                    "program_abi_storage": "span",
                                    "program_abi_rank": 1,
                                    "program_abi_mutable": mutable,
                                    "program_abi_field_written": False,
                                    "program_abi_row_identity": (
                                        "deterministic_graph_node_id"
                                    ),
                                    "program_abi_token_vocabulary": tuple(
                                        map(str, column.get(
                                            "token_vocabulary"
                                        ) or ())
                                    ),
                                },
                            )
                            next_physical_id += 1
                            function.args.append(arena)
                            values[int(arena.id)] = arena
                            column_arenas.append(arena)
                        lengths = SSAValue(
                            next_physical_id, dtype="int64",
                            accounting={
                                "program_abi_record": schema_identity,
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": f"{nested_path}.lengths",
                                "program_abi_storage": "span",
                                "program_abi_rank": 1,
                                "program_abi_mutable": mutable,
                                "program_abi_field_written": False,
                                "program_abi_row_identity": (
                                    "deterministic_graph_node_id"
                                ),
                            },
                        )
                        next_physical_id += 1
                        stride = SSAValue(
                            next_physical_id, dtype="int64",
                            accounting={
                                "program_abi_record": schema_identity,
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": (
                                    f"{nested_path}.row_stride"
                                ),
                                "program_abi_storage": "scalar",
                                "program_abi_rank": 0,
                                "program_abi_mutable": False,
                                "program_abi_field_written": False,
                            },
                        )
                        next_physical_id += 1
                        function.args.extend((lengths, stride))
                        values[int(lengths.id)] = lengths
                        values[int(stride.id)] = stride
                        row_handle = values[int(row_handle_id)]
                        row_offset = SSAValue(next_physical_id, dtype="int64")
                        next_physical_id += 1
                        setup = [Instr(
                            "Mul", [row_handle, stride], row_offset,
                            attributes={
                                "binding": "program_abi_child_table_row",
                                "program_abi_field": nested_path,
                            },
                        )]
                        pointers = []
                        for column, arena in zip(columns, column_arenas):
                            pointer = SSAValue(
                                next_physical_id, dtype=str(column["dtype"])
                            )
                            next_physical_id += 1
                            setup.append(Instr(
                                "GetElementPtr", [arena, row_offset], pointer,
                                attributes={
                                    "binding": (
                                        "program_abi_child_table_column"
                                    ),
                                    "program_abi_field": nested_path,
                                    "program_abi_column": str(column["name"]),
                                },
                            ))
                            pointers.append(pointer)
                        length_pointer = SSAValue(
                            next_physical_id, dtype="int64"
                        )
                        next_physical_id += 1
                        length = SSAValue(next_physical_id, dtype="int64")
                        next_physical_id += 1
                        setup.extend((
                            Instr(
                                "GetElementPtr", [lengths, row_handle],
                                length_pointer,
                                attributes={
                                    "binding": (
                                        "program_abi_child_table_length"
                                    ),
                                    "program_abi_field": nested_path,
                                },
                            ),
                            Instr(
                                "Load", [length_pointer], length,
                                attributes={
                                    "binding": (
                                        "program_abi_child_table_extent"
                                    ),
                                    "program_abi_field": nested_path,
                                },
                            ),
                        ))
                        inserted = False
                        for block in function.blocks.values():
                            for index, instruction in enumerate(
                                tuple(block.instrs)
                            ):
                                if (
                                    instruction.res is not None
                                    and int(instruction.res.id)
                                    == int(row_handle_id)
                                ):
                                    block.instrs[index + 1:index + 1] = setup
                                    inserted = True
                                    break
                            if inserted:
                                break
                        if inserted:
                            aliases = set(reachable)
                            projected_ids = {
                                int(value.id)
                                for _column, value in projected[sequence_id]
                            }
                            extent_results: dict[int, SSAValue] = {}
                            for block in function.blocks.values():
                                for instruction in block.instrs:
                                    if (
                                        instruction.attributes.get("binding")
                                        == "iterable_extent"
                                        and instruction.args
                                        and int(instruction.args[0].id)
                                        == int(sequence_id)
                                        and instruction.res is not None
                                    ):
                                        extent_results[int(
                                            instruction.res.id
                                        )] = length
                            for block in function.blocks.values():
                                kept = []
                                for instruction in block.instrs:
                                    if (
                                        instruction.attributes.get("binding")
                                        == "iterable_extent"
                                        and instruction.res is not None
                                        and int(instruction.res.id)
                                        in extent_results
                                    ):
                                        continue
                                    refreshed = []
                                    for argument in instruction.args:
                                        argument_id = int(argument.id)
                                        replacement = extent_results.get(
                                            argument_id
                                        )
                                        if replacement is None and (
                                            argument_id in aliases
                                        ):
                                            replacement = pointers[0]
                                        if replacement is None:
                                            projected_column = next((
                                                column
                                                for column, value in projected[
                                                    sequence_id
                                                ]
                                                if int(value.id) == argument_id
                                            ), None)
                                            if projected_column is not None:
                                                replacement = pointers[
                                                    projected_column
                                                ]
                                        refreshed.append(
                                            replacement or argument
                                        )
                                    instruction.args = refreshed
                                    kept.append(instruction)
                                block.instrs = kept
                            dropped = aliases | projected_ids | set(
                                extent_results
                            )
                            function.args = [
                                argument for argument in function.args
                                if int(argument.id) not in dropped
                            ]
                            sequence_table = all_sequence_tables.setdefault(
                                symbol, SSASequenceTable()
                            )
                            sequence_table.register(SSASequenceDescriptor(
                                int(sequence_id),
                                tuple(int(pointer.id) for pointer in pointers),
                                int(length_pointer.id), int(stride.id),
                                column_dtypes=tuple(
                                    str(column["dtype"])
                                    for column in columns
                                ),
                                key_columns=(), writable=mutable,
                            ))
                            nested_fields.append(SSARecordFieldDescriptor(
                                str(nested_name),
                                SSARecordFieldStorage.SEQUENCE,
                                storage_identity=(
                                    f"{schema_identity}.{nested_name}"
                                ),
                                sequence_id=int(sequence_id),
                                dtype="row",
                                writable=mutable,
                            ))
                            continue
                    # A declared table that cannot yet be correlated to the
                    # authored iterable stays an explicit reference frontier;
                    # it must never disappear merely because its schema was
                    # recognized.
                    nested_storage = "reference"
                    nested_dtype = None
                    nested_rank = 0
                if nested_storage == "scalar" and row_handle_id is not None:
                    # A field of a keyed value-record is a column selected by
                    # the row handle returned from the keyed lookup.  The row
                    # handle is the existing deterministic node identity; no
                    # frame-local slot or renumbering is introduced here.
                    candidate_ids = tuple(map(int, candidates))
                    producers = {
                        int(instruction.res.id)
                        for block in function.blocks.values()
                        for instruction in block.instrs
                        if instruction.res is not None
                    }
                    if not any(
                        candidate_id in producers
                        for candidate_id in candidate_ids
                    ):
                        column_key = (schema_identity, str(nested_name))
                        column = pooled_scalar_columns.get(column_key)
                        if column is None:
                            column = SSAValue(
                                next_physical_id,
                                dtype=(
                                    None if nested_dtype is None
                                    else str(nested_dtype)
                                ),
                                accounting={
                                    "program_abi_record": schema_identity,
                                    "program_abi_parameter": str(
                                        parameter_name
                                    ),
                                    "program_abi_field": (
                                        f"{nested_path}.column"
                                    ),
                                    "program_abi_storage": "span",
                                    "program_abi_rank": 1,
                                    "program_abi_mutable": mutable,
                                    "program_abi_field_written": False,
                                    "program_abi_row_identity": (
                                        "deterministic_graph_node_id"
                                    ),
                                    "program_abi_token_vocabulary": (
                                        token_vocabulary
                                    ),
                                },
                            )
                            next_physical_id += 1
                            pooled_scalar_columns[column_key] = column
                            function.args.append(column)
                            values[int(column.id)] = column
                        row_handle = values.get(int(row_handle_id))
                        if row_handle is not None:
                            result_id = candidate_ids[0]
                            pointer = SSAValue(
                                next_physical_id,
                                dtype=(
                                    None if nested_dtype is None
                                    else str(nested_dtype)
                                ),
                                accounting={
                                    "program_abi_record_column_pointer": True,
                                },
                            )
                            next_physical_id += 1
                            result = SSAValue(
                                result_id,
                                dtype=(
                                    None if nested_dtype is None
                                    else str(nested_dtype)
                                ),
                                accounting={
                                    "program_abi_record": schema_identity,
                                    "program_abi_parameter": str(
                                        parameter_name
                                    ),
                                    "program_abi_field": nested_path,
                                    "program_abi_storage": "scalar",
                                    "program_abi_rank": 0,
                                    "program_abi_mutable": mutable,
                                    "program_abi_field_written": False,
                                    "program_abi_row_handle": int(
                                        row_handle_id
                                    ),
                                    "program_abi_token_vocabulary": (
                                        token_vocabulary
                                    ),
                                },
                            )
                            setup = [
                                Instr(
                                    "GetElementPtr", [column, row_handle],
                                    pointer,
                                    attributes={
                                        "binding": (
                                            "program_abi_record_column"
                                        ),
                                        "program_abi_field": nested_path,
                                    },
                                ),
                                Instr(
                                    "Load", [pointer], result,
                                    attributes={
                                        "binding": (
                                            "program_abi_record_field"
                                        ),
                                        "program_abi_field": nested_path,
                                    },
                                ),
                            ]
                            predicate_ids: set[int] = set()
                            if token_vocabulary:
                                aliases = set(candidate_ids)
                                for test_node_id, test_data in graph.nodes(
                                    data=True
                                ):
                                    expression = test_data.get("expr_obj")
                                    accepted = accepted_isinstance_tokens(
                                        expression
                                    )
                                    if not accepted or not any(
                                        int(graph.nodes[parent].get(
                                            "value_id", parent
                                        )) in aliases
                                        and str(role).startswith("arg:0")
                                        for parent, role in (
                                            test_data.get("parents") or ()
                                        )
                                        if parent in graph
                                    ):
                                        continue
                                    if not all(
                                        token in token_vocabulary
                                        for token in accepted
                                    ):
                                        continue
                                    predicate_id = int(test_data.get(
                                        "value_id", test_node_id
                                    ))
                                    if predicate_id in producers:
                                        continue
                                    comparisons = []
                                    for token in accepted:
                                        encoded = (
                                            token_vocabulary.index(token) + 1
                                        )
                                        constant = SSAValue(
                                            next_physical_id, dtype="int64"
                                        )
                                        next_physical_id += 1
                                        compared = SSAValue(
                                            next_physical_id, dtype="bool"
                                        )
                                        next_physical_id += 1
                                        setup.extend((
                                            Instr(
                                                "Const", [], constant,
                                                attributes={
                                                    "value": encoded,
                                                    "program_abi_vocabulary_token": token,
                                                },
                                            ),
                                            Instr(
                                                "Eq", [result, constant],
                                                compared,
                                                attributes={
                                                    "program_abi_vocabulary_type_test": True,
                                                    "program_abi_field": nested_path,
                                                },
                                            ),
                                        ))
                                        comparisons.append(compared)
                                    combined = comparisons[0]
                                    for position, compared in enumerate(
                                        comparisons[1:], start=1
                                    ):
                                        merged = SSAValue(
                                            predicate_id
                                            if position == len(comparisons) - 1
                                            else next_physical_id,
                                            dtype="bool",
                                        )
                                        if int(merged.id) == next_physical_id:
                                            next_physical_id += 1
                                        setup.append(Instr(
                                            "Or", [combined, compared], merged,
                                            attributes={
                                                "program_abi_vocabulary_type_test": True,
                                                "program_abi_field": nested_path,
                                            },
                                        ))
                                        combined = merged
                                    if len(comparisons) == 1:
                                        # Preserve the authored predicate ID
                                        # while keeping the comparison result
                                        # itself an ordinary SSA boolean.
                                        final = SSAValue(
                                            predicate_id, dtype="bool"
                                        )
                                        setup.append(Instr(
                                            "Copy", [combined], final,
                                            attributes={
                                                "program_abi_vocabulary_type_test": True,
                                                "program_abi_field": nested_path,
                                            },
                                        ))
                                        combined = final
                                    predicate_ids.add(predicate_id)
                                    values[predicate_id] = combined
                            inserted = False
                            for block in function.blocks.values():
                                for index, instruction in enumerate(
                                    tuple(block.instrs)
                                ):
                                    if (
                                        instruction.res is not None
                                        and int(instruction.res.id)
                                        == int(row_handle_id)
                                    ):
                                        block.instrs[index + 1:index + 1] = setup
                                        inserted = True
                                        break
                                if inserted:
                                    break
                            if not inserted:
                                entry = function.blocks.get("entry")
                                if entry is not None:
                                    entry.instrs[0:0] = setup
                                    inserted = True
                            if inserted:
                                aliases = set(candidate_ids)
                                for block in function.blocks.values():
                                    for instruction in block.instrs:
                                        if instruction in setup:
                                            continue
                                        instruction.args = [
                                            result
                                            if int(argument.id) in aliases
                                            else argument
                                            for argument in instruction.args
                                        ]
                                function.args = [
                                    argument for argument in function.args
                                    if int(argument.id) not in (
                                        aliases | predicate_ids
                                    )
                                ]
                                for candidate_id in candidate_ids:
                                    values.pop(candidate_id, None)
                                values[result_id] = result
                                physical_ids.append(result_id)
                for value_id in candidates:
                    if physical_ids:
                        break
                    value = values.get(int(value_id))
                    if value is None:
                        value = SSAValue(
                            int(value_id),
                            dtype=(
                                None if nested_dtype is None
                                else str(nested_dtype)
                            ),
                        )
                        function.args.append(value)
                        values[int(value_id)] = value
                    elif (
                        value.dtype in {None, "unknown"}
                        and nested_dtype is not None
                    ):
                        value.dtype = str(nested_dtype)
                    value.accounting = {
                        **dict(value.accounting or {}),
                        "program_abi_record": schema_identity,
                        "program_abi_parameter": str(parameter_name),
                        "program_abi_field": nested_path,
                        "program_abi_storage": nested_storage,
                        "program_abi_rank": nested_rank,
                        "program_abi_mutable": mutable,
                        "program_abi_field_written": False,
                        "program_abi_token_vocabulary": token_vocabulary,
                    }
                    physical_ids.append(int(value_id))
                descriptor_storage = {
                    "scalar": SSARecordFieldStorage.SCALAR,
                    "span": SSARecordFieldStorage.SPAN,
                    "reference": SSARecordFieldStorage.REFERENCE,
                }.get(nested_storage)
                if descriptor_storage is None:
                    continue
                nested_fields.append(SSARecordFieldDescriptor(
                    str(nested_name),
                    descriptor_storage,
                    storage_identity=f"{schema_identity}.{nested_name}",
                    value_ids=tuple(physical_ids),
                    dtype=(
                        None if nested_dtype is None else str(nested_dtype)
                    ),
                    writable=mutable,
                ))
            if nested_fields and descriptor_id not in table.records:
                table.register(SSARecordDescriptor(
                    int(descriptor_id), schema_identity, tuple(nested_fields),
                ))

        for parameter_name, record in declared_records.items():
            demanded_fields = record_field_demands.get(
                (str(symbol), str(parameter_name)), set()
            )
            parameter_ids = set(map(
                int, identities.get(str(parameter_name), ())
            ))
            if not parameter_ids:
                continue
            record_id = next((
                int(value.id) for value in function.args
                if int(value.id) in parameter_ids
            ), min(parameter_ids))
            written_fields = {
                str((data.get("attributes") or {}).get("attribute"))
                for _node_id, data in graph.nodes(data=True)
                if str(data.get("type") or data.get("op") or "").casefold()
                == "setattr"
                and (data.get("attributes") or {}).get("attribute")
                is not None
                and any(
                    int(parent) in parameter_ids
                    and str(role) in {"value", "object", "base", "receiver"}
                    for parent, role in data.get("parents") or ()
                )
            } | record_field_writes.get(
                (str(symbol), str(parameter_name)), set()
            )
            write_source_ids_by_field: dict[str, tuple[int, ...]] = {}
            for _node_id, data in graph.nodes(data=True):
                if (
                    str(data.get("type") or data.get("op") or "").casefold()
                    != "setattr"
                ):
                    continue
                attributes = data.get("attributes") or {}
                field_name = attributes.get("attribute")
                if field_name is None or not any(
                    int(parent) in parameter_ids
                    and str(role) in {"value", "object", "base", "receiver"}
                    for parent, role in data.get("parents") or ()
                ):
                    continue
                sources = tuple(
                    int(graph.nodes[parent].get("value_id", parent))
                    for parent, role in data.get("parents") or ()
                    if str(role) == "value" and parent in graph
                )
                if sources:
                    write_source_ids_by_field[str(field_name)] = tuple(
                        dict.fromkeys((
                            *write_source_ids_by_field.get(
                                str(field_name), ()
                            ),
                            *sources,
                        ))
                    )
            fields = []
            for field_name, field in dict(record.get("fields") or {}).items():
                storage = str(field.get("storage") or "")
                mutable = bool(field.get("mutable", False))
                candidate_ids = tuple(dict.fromkeys((
                    *(
                        int(data.get("value_id", node_id))
                        for node_id, data in graph.nodes(data=True)
                        if str(
                            data.get("type") or data.get("op") or ""
                        ).casefold() == "getattr"
                        and str((
                            data.get("attributes") or {}
                        ).get("attribute")) == str(field_name)
                        and any(
                            int(parent) in parameter_ids
                            and str(role) in {"value", "object", "base"}
                            for parent, role in data.get("parents") or ()
                        )
                    ),
                    *write_source_ids_by_field.get(str(field_name), ()),
                )))
                if not candidate_ids:
                    if str(field_name) not in demanded_fields:
                        continue
                    candidate_ids = (next_physical_id,)
                    next_physical_id += 1
                if storage == "keyed":
                    # Materialize once per function. A second pass over the
                    # same symbol sees different attribute occurrences, so
                    # re-running would append a second set of slots and leave
                    # the mapping naming the first -- ids that no longer
                    # correspond to anything in this frame.
                    already = next((
                        int(existing)
                        for value_id in candidate_ids
                        if (existing := (
                            (values.get(int(value_id)) or SSAValue(-1)
                             ).accounting or {}
                        ).get("program_abi_keyed_length")) is not None
                        and int(existing) in values
                    ), None)
                    if already is not None:
                        continue
                    # A mapping keyed by words is not one opaque handle. It is
                    # a length plus two parallel vectors: the keys as the
                    # repository's universal string tokens, and the values.
                    # Because the token is content-addressed, a constant key
                    # and a name hashed at run time select the same slot, so
                    # this shape serves a fixed key set and a dynamic one
                    # identically. The mapping's own value keeps its identity
                    # and names the three slots, so the consumers that still
                    # read it can be resolved against them.
                    key_encoding = str(
                        field.get("key_encoding") or "string_token"
                    )
                    value_record = field.get("value_record")
                    value_identity = field.get("value_identity")
                    parts = {
                        "length": ("scalar", "int64", 0),
                        "keys": ("span", "int64", 1),
                        **({} if value_identity == "key" else {
                            "values": (
                                "span", str(field.get("dtype") or "float64"), 1,
                            ),
                        }),
                    }
                    part_ids: dict[str, int] = {}
                    for part_name, (
                        part_storage, part_dtype, part_rank
                    ) in parts.items():
                        part_id = next_physical_id
                        next_physical_id += 1
                        part_ids[part_name] = part_id
                        part_value = SSAValue(
                            part_id,
                            dtype=part_dtype,
                            accounting={
                                "program_abi_record": str(record["identity"]),
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": f"{field_name}.{part_name}",
                                "program_abi_storage": part_storage,
                                "program_abi_rank": part_rank,
                                "program_abi_mutable": mutable,
                                "program_abi_field_written": False,
                                "program_abi_keyed_owner": str(field_name),
                                "program_abi_keyed_part": part_name,
                                "program_abi_key_encoding": key_encoding,
                                "program_abi_value_record": value_record,
                            },
                        )
                        function.args.append(part_value)
                        values[part_id] = part_value
                        fields.append(SSARecordFieldDescriptor(
                            f"{field_name}.{part_name}",
                            SSARecordFieldStorage.SCALAR
                            if part_storage == "scalar"
                            else SSARecordFieldStorage.SPAN,
                            storage_identity=(
                                f"{record['identity']}.{field_name}.{part_name}"
                            ),
                            value_ids=(part_id,),
                            dtype=part_dtype,
                            writable=bool(mutable),
                        ))
                    if value_identity == "key":
                        part_ids["values"] = part_ids["keys"]
                    for value_id in candidate_ids:
                        mapping = values.get(int(value_id))
                        if mapping is None:
                            # The mapping's own occurrence is still a physical
                            # input: consumers that have not yet been resolved
                            # against the three slots continue to name it, and
                            # it is what carries the slot correlation.
                            mapping = SSAValue(int(value_id))
                            function.args.append(mapping)
                            values[int(value_id)] = mapping
                        mapping.accounting = {
                            **dict(mapping.accounting or {}),
                            "program_abi_record": str(record["identity"]),
                            "program_abi_parameter": str(parameter_name),
                            "program_abi_field": str(field_name),
                            "program_abi_storage": "keyed",
                            "program_abi_keyed_length": part_ids["length"],
                            "program_abi_keyed_keys": part_ids["keys"],
                            "program_abi_keyed_values": part_ids["values"],
                            "program_abi_key_encoding": key_encoding,
                            "program_abi_value_record": value_record,
                            "program_abi_value_identity": value_identity,
                        }
                    if value_record is not None:
                        for row_id in indexed_value_candidates(set(candidate_ids)):
                            materialize_nested_record(
                                str(value_record), {int(row_id)}, int(row_id),
                                str(parameter_name), f"{field_name}[]",
                                row_handle_id=int(row_id),
                            )
                    continue
                if storage == "record":
                    nested_schema = str(field.get("record") or "")
                    nested_record_id = min(candidate_ids)
                    materialize_nested_record(
                        nested_schema,
                        set(map(int, candidate_ids)),
                        nested_record_id,
                        str(parameter_name),
                        str(field_name),
                    )
                    nested_receipt = dict(abi_records[nested_schema])
                    fields.append(SSARecordFieldDescriptor(
                        str(field_name),
                        SSARecordFieldStorage.RECORD,
                        storage_identity=f"{record['identity']}.{field_name}",
                        value_ids=(),
                        record_id=nested_record_id,
                        dtype=str(
                            nested_receipt.get("identity") or nested_schema
                        ),
                        writable=False,
                    ))
                    continue
                field_written = str(field_name) in written_fields
                dtype = field.get("dtype")
                rank = int(field.get("rank", 0))
                fixed_length = field.get("fixed_length")
                physical_ids = []
                for value_id in candidate_ids:
                    value = values.get(value_id)
                    if value is None:
                        value = SSAValue(
                            value_id,
                            dtype=None if dtype is None else str(dtype),
                            shape=(
                                (int(fixed_length),)
                                if fixed_length is not None else ()
                            ),
                            accounting={
                                "program_abi_record": str(record["identity"]),
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": str(field_name),
                                "program_abi_storage": storage,
                                "program_abi_rank": rank,
                                "program_abi_mutable": mutable,
                                "program_abi_field_written": field_written,
                                "program_abi_fixed_length": fixed_length,
                            },
                        )
                        function.args.append(value)
                        values[value_id] = value
                    else:
                        if dtype is not None:
                            # The explicit program ABI is authoritative over
                            # the graph domain's numerical default.
                            value.dtype = str(dtype)
                        if fixed_length is not None:
                            value.shape = (int(fixed_length),)
                        value.accounting = {
                            **dict(value.accounting or {}),
                            "program_abi_record": str(record["identity"]),
                            "program_abi_parameter": str(parameter_name),
                            "program_abi_field": str(field_name),
                            "program_abi_storage": storage,
                            "program_abi_rank": rank,
                            "program_abi_mutable": mutable,
                            "program_abi_field_written": field_written,
                            "program_abi_fixed_length": fixed_length,
                        }
                    physical_ids.append(value_id)
                descriptor_storage = {
                    "scalar": SSARecordFieldStorage.SCALAR,
                    "span": SSARecordFieldStorage.SPAN,
                    "reference": SSARecordFieldStorage.REFERENCE,
                }[storage]
                fields.append(SSARecordFieldDescriptor(
                    str(field_name),
                    descriptor_storage,
                    storage_identity=f"{record['identity']}.{field_name}",
                    value_ids=tuple(physical_ids),
                    dtype=None if dtype is None else str(dtype),
                    writable=bool(mutable and field_written),
                ))
            if fields and record_id not in table.records:
                table.register(SSARecordDescriptor(
                    record_id, str(record["identity"]), tuple(fields),
                ))
        if not table.records:
            all_record_tables.pop(symbol, None)

    def materialize_program_abi_record_literals(symbol: str, graph: Any) -> None:
        """Lower schema-known constructor calls to field correlations.

        A dataclass-shaped boundary does not require executing its Python
        constructor. The authored argument edges already contain every field
        value; the program ABI supplies field order, defaults, storage and
        dtype. The result is an SSARecordDescriptor plus ordinary field SSA,
        with no opaque object id and no invented constructor operator.
        """

        function = all_functions.get(symbol)
        if function is None or not abi_records:
            return
        values = function_values(function)
        next_value_id = 1 + max((
            *values,
            *(int(node_id) for node_id in graph.nodes),
        ), default=0)
        constants = []
        table = all_record_tables.setdefault(symbol, SSARecordTable())
        layouts = []
        for node_id, data in graph.nodes(data=True):
            matched = abi_record_for_call(data)
            if matched is None:
                continue
            _record_name, record = matched
            record_id = int(data.get("value_id", node_id))
            if record_id in table.records:
                continue
            field_contracts = tuple(
                dict(record.get("fields") or {}).items()
            )
            keyword_values = {
                str(role).split(":", 1)[1]: int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).startswith("kw:")
            }
            # Ordered by the role's declared index, not by the order the
            # parent set happens to yield, and matching both ProcessGraph
            # spellings. This list is indexed positionally just below, so
            # taking the set's order would bind field N to whichever
            # argument iteration happened to reach first.
            positional_values = [
                int(parent)
                for parent in ordered_arguments(data.get("parents") or ())
            ]
            fields = []
            physical_layout = []
            for index, (field_name, field) in enumerate(field_contracts):
                value_id = keyword_values.get(str(field_name))
                if value_id is None and index < len(positional_values):
                    value_id = positional_values[index]
                if value_id is None and "default" in field:
                    default = field.get("default")
                    # Optional None needs a tagged optional ABI, which this
                    # scalar/span contract does not yet claim. Leave it absent
                    # rather than encoding a false floating-point zero.
                    if default is not None:
                        value_id = next_value_id
                        next_value_id += 1
                        value = SSAValue(
                            value_id, dtype=field.get("dtype"),
                            accounting={
                                "program_abi_default": str(field_name),
                                "program_abi_record": str(record["identity"]),
                            },
                        )
                        constants.append(Instr(
                            "Const", [], value, attributes={"value": default},
                        ))
                        values[value_id] = value
                source = (
                    {} if value_id is None
                    else graph.nodes.get(int(value_id), {})
                )
                source_attributes = dict(source.get("attributes") or {})
                physical_value_ids = (
                    () if value_id is None else (int(value_id),)
                )
                if (
                    value_id is not None
                    and str(field.get("storage")) == "span"
                    and field.get("fixed_length") is not None
                    and source_attributes.get("aggregate_kind") == "tuple"
                ):
                    leaves = tuple(map(
                        int,
                        source_attributes.get("aggregate_leaf_value_ids") or (),
                    ))
                    if len(leaves) == int(field["fixed_length"]):
                        # A fixed tuple is already an ordered vector of authored
                        # scalar identities. Keep those identities as the
                        # field's physical slots; there is no container object,
                        # allocation, or transient aggregate id in repository
                        # SSA.
                        physical_value_ids = leaves
                for physical_value_id in physical_value_ids:
                    if int(physical_value_id) in values:
                        continue
                    source = graph.nodes.get(int(physical_value_id), {})
                    source_attributes = dict(source.get("attributes") or {})
                    if str(
                        source.get("type") or source.get("op") or ""
                    ).casefold() in {"constant", "const"} and (
                        "value" in source_attributes or "constant" in source
                    ):
                        literal = source_attributes.get(
                            "value", source.get("constant")
                        )
                        value = SSAValue(
                            int(physical_value_id), dtype=field.get("dtype"),
                            accounting={
                                "program_abi_constructor_literal": str(
                                    field_name
                                ),
                                "program_abi_record": str(record["identity"]),
                            },
                        )
                        constants.append(Instr(
                            "Const", [], value, attributes={"value": literal},
                        ))
                        values[int(physical_value_id)] = value
                if not physical_value_ids or any(
                    int(physical_value_id) not in values
                    for physical_value_id in physical_value_ids
                ):
                    continue
                dtype = field.get("dtype")
                for physical_value_id in physical_value_ids:
                    value = values[int(physical_value_id)]
                    if value.dtype in {None, "unknown"} and dtype is not None:
                        value.dtype = str(dtype)
                if str(field["storage"]) == "keyed":
                    # The constructor argument here is a mapping literal, which
                    # is three physical slots (length, key tokens, values), not
                    # one. Correlating it needs the literal's own key/value
                    # edges, exactly as a nested record needs its own descriptor
                    # id; manufacturing a single slot from this occurrence would
                    # state a layout the record does not have.
                    continue
                storage = {
                    "scalar": SSARecordFieldStorage.SCALAR,
                    "span": SSARecordFieldStorage.SPAN,
                    "reference": SSARecordFieldStorage.REFERENCE,
                    "record": SSARecordFieldStorage.RECORD,
                }[str(field["storage"])]
                if storage is SSARecordFieldStorage.RECORD:
                    continue
                fields.append(SSARecordFieldDescriptor(
                    str(field_name), storage,
                    storage_identity=f"{record['identity']}.{field_name}",
                    value_ids=tuple(map(int, physical_value_ids)),
                    dtype=None if dtype is None else str(dtype),
                    writable=bool(field.get("mutable", False)),
                ))
                physical_layout.extend(map(int, physical_value_ids))
            if fields:
                table.register(SSARecordDescriptor(
                    record_id, str(record["identity"]), tuple(fields),
                ))
                layouts.append((record_id, tuple(physical_layout)))
        if constants:
            for block in function.blocks.values():
                if block.instrs and block.instrs[-1].op in {
                    "Ret", "ret", "Return", "return"
                }:
                    block.instrs[-1:-1] = constants
                    break
        if layouts:
            function.metadata["record_return_layouts"] = tuple(layouts)
        if not table.records:
            all_record_tables.pop(symbol, None)

    def materialize_record_phis(symbol: str) -> None:
        """Lower a control merge of like records to phis of physical fields.

        A record id is compile-time correlation, not a runtime object handle.
        Therefore an SSA Phi cannot select the conceptual record ids. It must
        select each corresponding physical field slot under the same incoming
        predecessor labels and publish a new record correlation for the merge
        result.
        """

        function = all_functions.get(symbol)
        table = all_record_tables.get(symbol)
        if function is None or table is None:
            return
        values = function_values(function)
        next_value_id = 1 + max(values, default=0)
        layouts = dict(function.metadata.get("record_return_layouts", ()))
        changed = True
        while changed:
            changed = False
            for block in function.blocks.values():
                rebuilt = []
                for instruction in block.instrs:
                    if instruction.op != "Phi" or instruction.res is None:
                        rebuilt.append(instruction)
                        continue
                    result_id = int(instruction.res.id)
                    if result_id in table.records or not instruction.args:
                        rebuilt.append(instruction)
                        continue
                    incoming = tuple(
                        table.records.get(int(argument.id))
                        for argument in instruction.args
                    )
                    if any(record is None for record in incoming):
                        rebuilt.append(instruction)
                        continue
                    first = incoming[0]
                    signatures = tuple(
                        (
                            field.name, field.storage, field.storage_identity,
                            len(field.value_ids), field.dtype,
                        )
                        for field in first.fields
                    )
                    if any(
                        record.identity != first.identity
                        or tuple(
                            (
                                field.name, field.storage,
                                field.storage_identity,
                                len(field.value_ids), field.dtype,
                            )
                            for field in record.fields
                        ) != signatures
                        for record in incoming[1:]
                    ):
                        rebuilt.append(instruction)
                        continue
                    if any(
                        int(value_id) not in values
                        for record in incoming
                        for field in record.fields
                        for value_id in field.value_ids
                    ):
                        rebuilt.append(instruction)
                        continue
                    merged_fields = []
                    merged_layout = []
                    for field_index, source_field in enumerate(first.fields):
                        merged_ids = []
                        for slot_index in range(len(source_field.value_ids)):
                            arguments = [
                                values[int(record.fields[field_index].value_ids[
                                    slot_index
                                ])]
                                for record in incoming
                            ]
                            result = SSAValue(
                                next_value_id,
                                dtype=source_field.dtype or arguments[0].dtype,
                                shape=arguments[0].shape,
                                accounting={
                                    "record_phi": result_id,
                                    "record_field": source_field.name,
                                    "record_field_slot": slot_index,
                                },
                            )
                            next_value_id += 1
                            attributes = dict(instruction.attributes or {})
                            attributes.update({
                                "record_phi": result_id,
                                "record_field": source_field.name,
                                "record_field_slot": slot_index,
                                "initial_value_id": int(arguments[0].id),
                            })
                            rebuilt.append(Instr(
                                "Phi", arguments, result,
                                arg_roles=list(instruction.arg_roles),
                                attributes=attributes,
                                source_span=instruction.source_span,
                            ))
                            values[int(result.id)] = result
                            merged_ids.append(int(result.id))
                            merged_layout.append(int(result.id))
                        merged_fields.append(replace(
                            source_field, value_ids=tuple(merged_ids)
                        ))
                    table.register(SSARecordDescriptor(
                        result_id, first.identity, tuple(merged_fields),
                    ))
                    layouts[result_id] = tuple(merged_layout)
                    changed = True
                block.instrs = rebuilt
        if layouts:
            function.metadata["record_return_layouts"] = tuple(layouts.items())

    def resolve_keyed_mapping_iterables(symbol: str, graph: Any) -> None:
        """Bind ``d.items()``/``.keys()``/``.values()`` to the mapping's slots.

        A keyed mapping is already a length and two parallel vectors, and the
        loop lowering already walks an iterable as parallel columns: column 0
        is the iterable itself and each further column is an appended source
        carrying ``projected_row_source_id``/``projected_row_column``. Those
        columns *are* the mapping's key and value vectors, so nothing new is
        built here -- only recognised. Left unrecognised they stay anonymous
        storage with no length to iterate and no slot to read, which is what
        made every consumer of a mapping unresolvable at every backend.

        Both ends of the association are exact. The reducer states the method
        as the node's own operation with the mapping as its operand, and the
        column index is carried on the appended source, so neither the mapping
        nor the column is inferred from a name or a position.
        """

        function = all_functions.get(symbol)
        if function is None:
            return
        slots_by_mapping: dict[int, dict[str, int]] = {}
        for value in function.args:
            accounting = value.accounting or {}
            if accounting.get("program_abi_storage") != "keyed":
                continue
            parts = {
                part: accounting.get(f"program_abi_keyed_{part}")
                for part in ("length", "keys", "values")
            }
            if any(slot is None for slot in parts.values()):
                continue        # unresolved in this frame; leave it alone
            slots_by_mapping[int(value.id)] = {
                part: int(slot) for part, slot in parts.items()
            }
        # The mapping identity's slot correlation is frame-local and may have
        # been dropped, while the parts themselves still name their owner.
        # Lookups rebind through the parts, so their presence alone keeps
        # this pass alive.
        has_keyed_parts = any(
            (value.accounting or {}).get("program_abi_keyed_owner")
            is not None
            for value in function.args
        )
        if not slots_by_mapping and not has_keyed_parts:
            return

        # method -> the slot each successive destructured column selects
        columns_by_method = {
            "items": ("keys", "values"),
            "keys": ("keys",),
            "values": ("values",),
        }
        replacements: dict[int, int] = {}
        for node_id, data in graph.nodes(data=True):
            method = str(
                data.get("type") or data.get("op") or ""
            ).casefold()
            columns = columns_by_method.get(method)
            if columns is None:
                continue
            owner = next((
                int(graph.nodes[parent].get("value_id", parent))
                for parent, role in data.get("parents") or ()
                if str(role) in {"operand", "value", "object", "base"}
                and parent in graph
            ), None)
            slots = (
                None if owner is None else slots_by_mapping.get(int(owner))
            )
            if slots is None:
                continue
            iterable_id = int(data.get("value_id", node_id))
            replacements[iterable_id] = slots[columns[0]]
            for value in function.args:
                accounting = value.accounting or {}
                source = accounting.get("projected_row_source_id")
                if source is None or int(source) != iterable_id:
                    continue
                column = int(accounting.get("projected_row_column") or 0)
                if column < len(columns):
                    replacements[int(value.id)] = slots[columns[column]]
            # The iterable's extent is the mapping's own declared length.
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op == "Call"
                        and instruction.attributes.get("tensor_operation")
                        == "extent"
                        and instruction.args
                        and int(instruction.args[0].id) == iterable_id
                        and instruction.res is not None
                    ):
                        replacements[int(instruction.res.id)] = slots["length"]
        # A table lookup on a keyed mapping walks the same declared vectors.
        # The descriptor was built during lowering from anonymous storage --
        # (keys, values, length, capacity) fresh arguments -- because the
        # slots only exist after record materialization.  Bind them here.
        # The mapping identity's own accounting may be frame-local-dropped,
        # so the parts are found by their owner/part markers and the lookup's
        # field by its GetAttr node in the source graph.  A caller-supplied
        # mapping is always exactly full, so capacity IS the length; the
        # status cell stays an ordinary frame-allocated scalar.
        parts_by_owner: dict[str, dict[str, int]] = {}
        owner_by_mapping: dict[int, str] = {}
        for value in function.args:
            accounting = value.accounting or {}
            if accounting.get("program_abi_storage") == "keyed" and (
                accounting.get("program_abi_field") is not None
            ):
                owner_by_mapping[int(value.id)] = str(
                    accounting["program_abi_field"]
                )
            owner = accounting.get("program_abi_keyed_owner")
            part = accounting.get("program_abi_keyed_part")
            if owner is None or part is None:
                continue
            parts_by_owner.setdefault(str(owner), {})[str(part)] = int(
                value.id
            )
        field_of_sequence: dict[int, str] = {}
        for node_id, data in graph.nodes(data=True):
            attribute = (data.get("attributes") or {}).get("attribute")
            if attribute is None:
                continue
            field_of_sequence[int(data.get("value_id", node_id))] = str(
                attribute
            )
        helper_argument_dtypes = (
            ("int64", None), ("float64", None), ("int", (1,)),
            ("int", None), ("int", (1,)), ("int64", None),
        )
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op != "Call"
                    or instruction.attributes.get("ssa_sequence_operation")
                    != "lookup"
                    or len(instruction.args) < 6
                ):
                    continue
                sequence_id = int(
                    instruction.attributes.get("sequence_id", -1)
                )
                # The mapping value's deterministic SSA identity is the
                # authoritative correlation.  Attribute spelling alone loses
                # the containing record path (``nodes`` versus ``G.nodes``)
                # and can bind a same-spelled field from another record.
                owner = owner_by_mapping.get(
                    sequence_id, field_of_sequence.get(sequence_id)
                )
                if owner is None:
                    continue
                # The slots this lookup must walk may not exist in this frame
                # yet -- for a mapping produced by a call, the linker imports
                # them later.  Stamp the owner now, while the source graph is
                # at hand; the storage is bound after call-frame linking.
                instruction.attributes["keyed_lookup_owner"] = str(owner)
        if not replacements:
            return

        values = function_values(function)
        resolved = {
            source: values[target]
            for source, target in replacements.items()
            if target in values
        }
        for block in function.blocks.values():
            kept = []
            for instruction in block.instrs:
                if (
                    instruction.res is not None
                    and int(instruction.res.id) in resolved
                ):
                    continue        # its value is the slot now
                instruction.args = [
                    resolved.get(int(argument.id), argument)
                    for argument in instruction.args
                ]
                kept.append(instruction)
            block.instrs = kept
        function.args = [
            value for value in function.args
            if int(value.id) not in resolved
        ]

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        materialize_parameter_record_abi(source_symbol, source_graph)

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        resolve_keyed_mapping_iterables(source_symbol, source_graph)

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        materialize_program_abi_record_literals(source_symbol, source_graph)

    for source_symbol in source_graphs_by_symbol:
        materialize_record_phis(source_symbol)

    # Literals are source-independent SSA producers. Record materialization
    # can introduce them after control lowering (notably tuple field leaves
    # used as Phi inputs); leaving those Const instructions beside the final
    # Ret makes earlier branch uses read uninitialized storage. Hoist every
    # function-local literal to its entry block, where it dominates every CFG
    # path. This changes placement only, never identity or value.
    for function in all_functions.values():
        if not function.blocks:
            continue
        constants = []
        seen_constant_ids = set()
        for block in function.blocks.values():
            retained = []
            for instruction in block.instrs:
                if (
                    str(instruction.op).casefold() in {"const", "constant"}
                    and instruction.res is not None
                ):
                    value_id = int(instruction.res.id)
                    if value_id in seen_constant_ids:
                        raise ValueError(
                            "one SSA constant identity is produced more than "
                            f"once in {function.name!r}: value={value_id}"
                        )
                    seen_constant_ids.add(value_id)
                    constants.append(instruction)
                else:
                    retained.append(instruction)
            block.instrs[:] = retained
        entry = next(iter(function.blocks.values()))
        entry.instrs[0:0] = constants

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        recover_structural_source_outputs(source_symbol, source_graph)

    for function in all_functions.values():
        source_output_ids = tuple(map(
            int, function.metadata.get("source_output_value_ids", ())
        ))
        if not source_output_ids:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                by_id = {
                    int(argument.id): argument
                    for argument in instruction.args
                }
                # A carried reduction returns through its LoopResult port,
                # but the value standing at that port is the carried PHI,
                # which has its own id.  The builder exports port -> phi in
                # metadata; selection by raw layout id must follow it, or
                # the component resolves to the port's unwritten field cell
                # and every carried maximum publishes zero.
                for argument in instruction.args:
                    for port_id in (
                        (argument.accounting or {}).get("carried_port_ids")
                        or ()
                    ):
                        by_id.setdefault(int(port_id), argument)
                for port_id, port_value in dict(
                    function.metadata.get("carried_port_values") or {}
                ).items():
                    # The port map is the AUTHORITY: a stale component
                    # object carrying the port's id may already sit in the
                    # Ret from earlier expansion, and it names the unwritten
                    # field cell.
                    by_id[int(port_id)] = port_value
                record_layouts = dict(
                    function.metadata.get("record_return_layouts", ())
                )
                expanded_source_ids = tuple(
                    expanded
                    for value_id in source_output_ids
                    for expanded in record_layouts.get(value_id, (value_id,))
                )
                selected = [
                    by_id[value_id]
                    for value_id in expanded_source_ids
                    if value_id in by_id
                ]
                # Some branch histories intentionally publish a predecessor
                # id (the zmap fallback is one); preserve those when the final
                # identity has no materialized spelling. Otherwise discard
                # incidental control/region outputs from the public ABI.
                if selected:
                    instruction.args = selected

    # Parameter-record ABI expansion replaces a conceptual Python receiver
    # with its physical fields. Remove an unconsumed shapeless receiver before
    # call-frame linking; waiting until final cleanup leaves an otherwise
    # complete record-to-record call asking for a nonexistent object handle.
    for function_name, function in all_functions.items():
        record_table = all_record_tables.get(function_name)
        record_ids = set(
            () if record_table is None else map(int, record_table.records)
        )
        source_graph = source_graphs_by_symbol.get(function_name)
        if source_graph is not None:
            identities = source_graph.graph.get("identity_table") or {}
            for parameter_name in (
                source_graph.graph.get("parameter_record_abi") or {}
            ):
                record_ids.update(map(
                    int, identities.get(str(parameter_name), ())
                ))
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        function.args = [
            argument for argument in function.args
            if not (
                int(argument.id) in record_ids
                and int(argument.id) not in consumed_ids
                and argument.dtype is None
                and not argument.shape
                and not argument.accounting
            )
        ]

    def clone_value(source: Any, value_id: int, *, accounting=None):
        return SSAValue(
            int(value_id),
            dtype=getattr(source, "dtype", None),
            shape=tuple(getattr(source, "shape", ()) or ()),
            device=getattr(source, "device", None),
            accounting={
                **dict(getattr(source, "accounting", {}) or {}),
                **dict(accounting or {}),
            },
        )

    def map_child_pool(pool: Any, remap: Mapping[int, int]):
        if pool is None:
            return None
        return SSAChildTablePoolDescriptor(
            handle_column=int(pool.handle_column),
            column_value_ids=tuple(
                remap[int(value_id)] for value_id in pool.column_value_ids
            ),
            length_value_id=remap[int(pool.length_value_id)],
            capacity_value_id=remap[int(pool.capacity_value_id)],
            row_stride_value_id=remap[int(pool.row_stride_value_id)],
            status_value_id=(
                None if pool.status_value_id is None
                else remap[int(pool.status_value_id)]
            ),
            live_flags_value_id=(
                None if pool.live_flags_value_id is None
                else remap[int(pool.live_flags_value_id)]
            ),
            column_dtypes=tuple(pool.column_dtypes),
            key_columns=tuple(pool.key_columns),
            writable=bool(pool.writable),
        )

    def loop_constructor_requires_instance_pool(
        graph: Any, receiver_id: int, enclosing_loop_ids: tuple[int, ...]
    ) -> bool:
        if not enclosing_loop_ids:
            return False
        output_ids = {
            int(value_id)
            for name in tuple(graph.graph.get("function_outputs") or ())
            for value_id in tuple(
                (graph.graph.get("identity_table") or {}).get(str(name), ())
            )
        }
        if int(receiver_id) in output_ids:
            return True
        # Passing the record as data stores/returns/aliases its identity beyond
        # the current iteration. Attribute lookup and a receiver-bound method
        # invocation are address/use operations only; neither extends the
        # instance lifetime. Only an actual data escape requires a row pool.
        for successor in graph.successors(int(receiver_id)):
            successor_data = graph.nodes[successor]
            successor_attributes = successor_data.get("attributes") or {}
            roles = {
                str(role) for parent, role in (
                    successor_data.get("parents") or ()
                ) if int(parent) == int(receiver_id)
            }
            successor_operation = str(
                successor_data.get("op")
                or successor_data.get("type")
                or ""
            ).casefold()
            if (
                successor_operation == "getattr"
                and roles <= {"value", "base", "object", "operand"}
            ):
                continue
            if (
                successor_attributes.get("method_ref") is not None
                and roles <= {"value", "base", "object", "operand"}
            ):
                continue
            if roles:
                return True
        return False

    def lexical_loop_ids(
        graph: Any, node_id: int, candidate_ids: Iterable[int]
    ) -> tuple[int, ...]:
        """Discard dependency-closure loop ownership outside lexical spans."""

        node_expression = graph.nodes.get(int(node_id), {}).get("expr_obj")
        node_line = getattr(node_expression, "lineno", None)
        if node_line is None:
            return tuple(map(int, candidate_ids))
        retained = []
        for loop_id in candidate_ids:
            loop_expression = graph.nodes.get(int(loop_id), {}).get("expr_obj")
            loop_line = getattr(loop_expression, "lineno", None)
            loop_end = getattr(loop_expression, "end_lineno", None)
            if loop_line is None or loop_end is None:
                retained.append(int(loop_id))
                continue
            if int(loop_line) <= int(node_line) <= int(loop_end):
                retained.append(int(loop_id))
        return tuple(retained)

    if class_table is not None:
        class_definitions = {
            str(record.identity): record for record in class_table.classes
        }
        class_alias_candidates: dict[str, list[Any]] = {}
        for definition in class_definitions.values():
            class_alias_candidates.setdefault(
                str(definition.identity).rsplit(".", 1)[-1], []
            ).append(definition)
        class_aliases = {
            alias: candidates[0]
            for alias, candidates in class_alias_candidates.items()
            if len(candidates) == 1
        }

        def resolve_class_definition(identity: Any) -> Any:
            """Resolve frontend short class refs without guessing collisions."""

            text = str(identity)
            return class_definitions.get(text) or class_aliases.get(text)

        caller_contexts = {}
        for caller_symbol, _item, caller_graph, _module, caller_shell in (
            pending_call_records
        ):
            caller_contexts.setdefault(
                str(caller_symbol), (caller_graph, caller_shell)
            )
        # A function with construction but no ordinary method/function call has
        # no pending PlanCall record.  Include every planned shell as well.
        for shell in planned_shells:
            graph = getattr(getattr(shell, "process_graph", None), "G", None)
            function_name = (
                None if graph is None else graph.graph.get("function_name")
            )
            if function_name is not None:
                caller_contexts.setdefault(
                    f"{artifact_name}__{function_name}", (graph, shell)
                )

        constructor_symbol_by_class = {}
        for definition in class_definitions.values():
            method = next((
                method for name in ("__new__", "__init__")
                for method in definition.methods
                if method.name == name and method.function_name is not None
            ), None)
            if method is None:
                continue
            constructor_symbol_by_class[str(definition.identity)] = str(
                method.function_name
            )
            short_identity = str(definition.identity).rsplit(".", 1)[-1]
            if class_aliases.get(short_identity) is definition:
                constructor_symbol_by_class[short_identity] = str(
                    method.function_name
                )

        # Materialize constructor-owned frames dependency-first.  If
        # ``Outer.__init__`` constructs ``Inner``, the Inner frame must already
        # have been copied into Outer before a caller copies Outer.  This is a
        # local topological order over the already-pursued constructor calls,
        # not another ingestion/lowering pipeline.
        ordered_contexts = []
        visiting = set()
        visited = set()

        def visit_constructor_context(symbol: str) -> None:
            if symbol in visited or symbol in visiting:
                return
            visiting.add(symbol)
            context = caller_contexts.get(symbol)
            if context is not None:
                graph, _shell = context
                dependencies = tuple(dict.fromkeys(
                    constructor_symbol_by_class[str(class_ref)]
                    for _node_id, data in graph.nodes(data=True)
                    for class_ref in ((data.get("attributes") or {}).get(
                        "class_ref"
                    ),)
                    if class_ref is not None
                    and str(class_ref) in constructor_symbol_by_class
                ))
                for dependency in dependencies:
                    visit_constructor_context(str(dependency))
                ordered_contexts.append((symbol, context))
            visiting.remove(symbol)
            visited.add(symbol)

        for caller_symbol in caller_contexts:
            visit_constructor_context(str(caller_symbol))

        # A class with a constructor is irreducible. The self-is-field-storage
        # collapse below makes a constructor argument's id BE the resulting
        # object's own storage identity, which is only sound when nothing
        # else can ever construct that class again from an existing value --
        # copy.deepcopy(n) does exactly that, re-running __init__ with n's
        # own current field values, and the collapse then aliases the copy
        # onto n's storage instead of giving it its own. Any class the
        # program constructs therefore keeps a genuine, separate receiver
        # identity rather than being flattened into a single shared scalar.
        irreducible_classes = set(constructor_symbol_by_class)

        for caller_symbol, (caller_graph, caller_shell) in ordered_contexts:
            caller = all_functions.get(caller_symbol)
            if caller is None:
                continue
            available = function_values(caller)
            graph_ids = {
                int(data.get("value_id", node_id))
                for node_id, data in caller_graph.nodes(data=True)
            }
            next_value_id = 1 + max((*available, *graph_ids), default=0)
            caller_records = all_record_tables.setdefault(
                caller_symbol, SSARecordTable()
            )
            caller_sequences = all_sequence_tables.setdefault(
                caller_symbol, SSASequenceTable()
            )
            for node_id, node_data in sorted(
                caller_graph.nodes(data=True), key=lambda item: int(item[0])
            ):
                attributes = node_data.get("attributes") or {}
                class_identity = attributes.get("class_ref")
                # StaticReference nodes deliberately carry the same class_ref
                # so member navigation can resolve them, but only the Call is
                # an instance allocation site with caller-owned storage.
                node_operation = str(
                    node_data.get("op") or node_data.get("type") or ""
                ).casefold()
                # copy.deepcopy(n) resolves (python_identity_programs.py) to
                # a Handler.Deepcopy node, not a constructor Call -- it has
                # no class_ref of its own. Its one argument (n) does: n was
                # itself constructed earlier in this same caller, so its
                # class identity and current field values are already
                # registered in caller_records by this same loop (nodes are
                # walked in source order). Re-running class construction
                # with n's OWN current field values as the "constructor
                # arguments" -- instead of fresh AST call arguments -- is
                # the deep copy: it reuses the exact caller-owned-storage
                # frame mechanism below, unchanged, rather than inventing a
                # parallel copy mechanism this compiler's objects don't have
                # a runtime address for.
                deepcopy_source_record = None
                if node_operation == "deepcopy":
                    deepcopy_source_id = next((
                        int(parent)
                        for parent, role in (node_data.get("parents") or ())
                        if str(role) not in {"callee", "func", "definition"}
                    ), None)
                    if deepcopy_source_id is None:
                        continue
                    deepcopy_source_record = caller_records.records.get(
                        int(caller_graph.nodes[deepcopy_source_id].get(
                            "value_id", deepcopy_source_id
                        ))
                    )
                    # Not locally resolvable (the source instance was not
                    # constructed in this same caller frame): do not
                    # fabricate a correlation, matching every other
                    # nested-record bail-out in this file.
                    if deepcopy_source_record is None:
                        continue
                    class_identity = deepcopy_source_record.identity
                    node_operation = "call"
                if class_identity is None or node_operation != "call":
                    continue
                class_definition = resolve_class_definition(class_identity)
                if class_definition is None:
                    continue
                constructor_method = next((
                    method for name in ("__new__", "__init__")
                    for method in class_definition.methods
                    if method.name == name and method.function_name is not None
                ), None)
                if constructor_method is None:
                    continue
                constructor_symbol = str(constructor_method.function_name)
                constructor = all_functions.get(constructor_symbol)
                constructor_table = all_record_tables.get(constructor_symbol)
                constructor_sequences = all_sequence_tables.get(
                    constructor_symbol
                )
                if (
                    constructor is None
                    or constructor_table is None
                    or not constructor_table.records
                ):
                    continue
                class_storage_identities = {
                    str(class_identity),
                    str(class_definition.identity),
                    str(class_definition.identity).rsplit(".", 1)[-1],
                }
                templates = tuple(
                    record for record in constructor_table.records.values()
                    if record.identity in class_storage_identities
                )
                if len(templates) != 1:
                    continue
                template = templates[0]
                constructor_records_by_id = {
                    int(record.record_id): record
                    for record in constructor_table.records.values()
                }

                def nested_record_closure(root: Any) -> tuple[Any, ...]:
                    """Return the authored record/storage tree rooted at ``root``.

                    Nested records are still ordinary repository-SSA record
                    descriptors.  Following their ids here makes construction
                    copy the complete caller-owned storage frame rather than
                    reducing a nested record field to one opaque handle.
                    """

                    ordered = []
                    pending = [root]
                    seen = set()
                    while pending:
                        record = pending.pop()
                        record_id = int(record.record_id)
                        if record_id in seen:
                            continue
                        seen.add(record_id)
                        ordered.append(record)
                        for field in reversed(record.fields):
                            if field.record_id is None:
                                continue
                            nested = constructor_records_by_id.get(
                                int(field.record_id)
                            )
                            if nested is not None:
                                pending.append(nested)
                    return tuple(ordered)

                template_records = nested_record_closure(template)
                receiver_id = int(node_data.get("value_id", node_id))
                if receiver_id in caller_records.records:
                    continue
                constructor_values = function_values(constructor)
                constructor_self_id = int(template.record_id)
                self_is_field_storage = (
                    str(class_identity) not in irreducible_classes
                    and any(
                        constructor_self_id in tuple(map(int, field.value_ids))
                        for field in template.fields
                    )
                )
                remap: dict[int, int] = {}
                if not self_is_field_storage:
                    remap[constructor_self_id] = receiver_id
                constructor_graph = source_graphs_by_symbol.get(
                    constructor_symbol
                )
                constructor_parameter_ids: set[int] = set()
                planned_constructor_call = max(
                    (
                        item
                        for pending_caller, item, _graph, _module, _shell
                        in pending_call_records
                        if str(pending_caller) == str(caller_symbol)
                        and int(item.callsite_id) == int(node_id)
                    ),
                    key=lambda item: len(item.argument_bindings),
                    default=None,
                )
                if planned_constructor_call is not None:
                    for caller_id, callee_id in (
                        planned_constructor_call.argument_bindings
                    ):
                        remap[int(callee_id)] = int(caller_id)
                        constructor_parameter_ids.add(int(callee_id))
                if deepcopy_source_record is not None:
                    # No AST call arguments exist for copy.deepcopy(n) --
                    # bind each of the constructor's own scalar field
                    # parameters directly to n's OWN current value for that
                    # same-named field, instead of to a fresh call argument.
                    # A field the source record does not have a value for
                    # (e.g. a nested record -- its own construction is
                    # handled when this same loop reaches ITS constructor
                    # call, not here) is left unbound; unresolved bindings
                    # are reported honestly below, not silently dropped.
                    source_fields_by_name = {
                        str(field.name): field
                        for field in deepcopy_source_record.fields
                    }
                    for field in template.fields:
                        if (
                            field.storage is not SSARecordFieldStorage.SCALAR
                            or not field.value_ids
                        ):
                            continue
                        source_field = source_fields_by_name.get(str(field.name))
                        if source_field is None or not source_field.value_ids:
                            continue
                        parameter_id = int(field.value_ids[0])
                        remap[parameter_id] = int(source_field.value_ids[0])
                        constructor_parameter_ids.add(parameter_id)
                elif constructor_graph is not None:
                    identities = constructor_graph.graph.get(
                        "identity_table"
                    ) or {}
                    parameter_names = tuple(
                        constructor_graph.graph.get("function_parameters") or ()
                    )
                    positional_names = tuple(
                        name for name in parameter_names if name != "self"
                    )
                    for parent, role in node_data.get("parents") or ():
                        role = str(role)
                        if role in {"callee", "func", "definition"}:
                            continue
                        index = positional_argument_index(role)
                        keyword = keyword_argument_name(role)
                        if index is not None:
                            name = (
                                positional_names[index]
                                if index < len(positional_names) else None
                            )
                        elif keyword is not None:
                            name = keyword
                        else:
                            name = None
                        history = tuple(identities.get(name, ()))
                        if not history:
                            continue
                        parameter_id = int(history[0])
                        remap[parameter_id] = int(
                            caller_graph.nodes[parent].get("value_id", parent)
                        )
                        constructor_parameter_ids.add(parameter_id)
                referenced_ids = tuple(dict.fromkeys(
                    int(value_id)
                    for record in template_records
                    for field in record.fields
                    for value_id in (
                        *field.value_ids,
                        *((field.record_id,) if field.record_id is not None else ()),
                    )
                ))
                for old_id in referenced_ids:
                    if old_id in remap:
                        continue
                    remap[old_id] = next_value_id
                    next_value_id += 1
                # A constructor's repository-SSA signature is its complete
                # physical frame, not merely its authored parameters and
                # record fields. Region scratch and descriptor slots are also
                # caller-owned storage, so allocate every remaining frame id
                # instead of leaving the object intrinsically host-bound.
                for value in constructor.args:
                    old_id = int(value.id)
                    if old_id in remap:
                        continue
                    remap[old_id] = next_value_id
                    next_value_id += 1

                # Constructor field writes and subsequent reads can carry
                # separate local sequence descriptors for one record slot.
                # Correlate all such authored field-op views to the canonical
                # record field before building the call frame.
                constructor_field_sequence_ids = set()
                constructor_field_sequence_ids_by_name = {}
                if constructor_graph is not None:
                    constructor_field_contract = _field_slot_ops(
                        constructor_graph
                    )
                    constructor_field_names = tuple(
                        constructor_field_contract[4]
                    )
                    for _kind, value_id, slot in constructor_field_contract[1]:
                        constructor_field_sequence_ids.add(int(value_id))
                        if 0 <= int(slot) < len(constructor_field_names):
                            constructor_field_sequence_ids_by_name.setdefault(
                                str(constructor_field_names[int(slot)]), set()
                            ).add(int(value_id))
                if constructor_sequences is not None:
                    for field in template.fields:
                        if field.sequence_id is None:
                            continue
                        canonical = constructor_sequences.by_id(
                            field.sequence_id
                        )
                        if canonical is None:
                            continue
                        canonical_ids = (
                            *canonical.column_value_ids,
                            canonical.length_address_id,
                            canonical.capacity_value_id,
                            *((canonical.status_address_id,)
                              if canonical.status_address_id is not None else ()),
                            *((canonical.live_flags_value_id,)
                              if canonical.live_flags_value_id is not None else ()),
                        )
                        resident_ids = tuple(remap[int(value_id)]
                                             for value_id in canonical_ids)
                        for local in constructor_sequences.sequences.values():
                            if (
                                int(local.sequence_id)
                                not in constructor_field_sequence_ids_by_name.get(
                                    str(field.name), set()
                                )
                                or len(local.column_value_ids)
                                != len(canonical.column_value_ids)
                                or tuple(local.key_columns)
                                != tuple(canonical.key_columns)
                            ):
                                continue
                            local_ids = (
                                *local.column_value_ids,
                                local.length_address_id,
                                local.capacity_value_id,
                                *((local.status_address_id,)
                                  if local.status_address_id is not None else ()),
                                *((local.live_flags_value_id,)
                                  if local.live_flags_value_id is not None else ()),
                            )
                            if len(local_ids) == len(resident_ids):
                                remap.update(zip(
                                    map(int, local_ids), resident_ids
                                ))

                new_arguments = []
                for old_id, new_id in remap.items():
                    if (
                        old_id == constructor_self_id
                        and old_id not in referenced_ids
                    ):
                        continue
                    if new_id in available:
                        continue
                    source = constructor_values.get(old_id, SSAValue(old_id))
                    value = clone_value(source, new_id, accounting={
                        "record_instance": str(class_identity),
                        "constructor_callsite_id": int(node_id),
                    })
                    caller.args.append(value)
                    available[new_id] = value
                    new_arguments.append(value)

                mapped_records = {}
                for record_template in reversed(template_records):
                    mapped_fields = []
                    for field in record_template.fields:
                        mapped_sequence_id = None
                        if field.sequence_id is not None:
                            source_sequence = (
                                None if constructor_sequences is None
                                else constructor_sequences.by_id(field.sequence_id)
                            )
                            if source_sequence is None:
                                continue
                            pool_ids = ()
                            if source_sequence.child_table_pool is not None:
                                pool = source_sequence.child_table_pool
                                pool_ids = (
                                    *pool.column_value_ids,
                                    pool.length_value_id,
                                    pool.capacity_value_id,
                                    pool.row_stride_value_id,
                                    *((pool.status_value_id,)
                                      if pool.status_value_id is not None else ()),
                                    *((pool.live_flags_value_id,)
                                      if pool.live_flags_value_id is not None else ()),
                                )
                            for old_id in map(int, pool_ids):
                                if old_id not in remap:
                                    remap[old_id] = next_value_id
                                    source = constructor_values.get(
                                        old_id, SSAValue(old_id)
                                    )
                                    value = clone_value(
                                        source,
                                        next_value_id,
                                        accounting={
                                            "record_instance": str(class_identity),
                                            "constructor_callsite_id": int(node_id),
                                        },
                                    )
                                    caller.args.append(value)
                                    available[next_value_id] = value
                                    next_value_id += 1
                            mapped_sequence_id = remap[
                                int(source_sequence.sequence_id)
                            ]
                            caller_sequences.register(SSASequenceDescriptor(
                                sequence_id=mapped_sequence_id,
                                column_value_ids=tuple(
                                    remap[int(value_id)]
                                    for value_id in source_sequence.column_value_ids
                                ),
                                length_address_id=remap[
                                    int(source_sequence.length_address_id)
                                ],
                                capacity_value_id=remap[
                                    int(source_sequence.capacity_value_id)
                                ],
                                status_address_id=(
                                    None
                                    if source_sequence.status_address_id is None
                                    else remap[int(source_sequence.status_address_id)]
                                ),
                                column_dtypes=tuple(source_sequence.column_dtypes),
                                key_columns=tuple(source_sequence.key_columns),
                                live_flags_value_id=(
                                    None
                                    if source_sequence.live_flags_value_id is None
                                    else remap[int(source_sequence.live_flags_value_id)]
                                ),
                                capacity_policy=source_sequence.capacity_policy,
                                writable=bool(source_sequence.writable),
                                child_table_pool=map_child_pool(
                                    source_sequence.child_table_pool, remap
                                ),
                            ))
                        mapped_fields.append(SSARecordFieldDescriptor(
                            name=field.name,
                            storage=field.storage,
                            storage_identity=field.storage_identity,
                            value_ids=tuple(
                                remap[int(value_id)]
                                for value_id in field.value_ids
                            ),
                            sequence_id=mapped_sequence_id,
                            record_id=(
                                None
                                if field.record_id is None
                                else remap[int(field.record_id)]
                            ),
                            offset=field.offset,
                            dtype=field.dtype,
                            writable=field.writable,
                        ))
                    mapped_record_id = (
                        receiver_id
                        if int(record_template.record_id) == constructor_self_id
                        else remap[int(record_template.record_id)]
                    )
                    mapped_record = SSARecordDescriptor(
                        mapped_record_id,
                        str(record_template.identity),
                        tuple(mapped_fields),
                    )
                    caller_records.register(mapped_record)
                    mapped_records[int(record_template.record_id)] = mapped_record
                mapped_fields = list(
                    mapped_records[int(template.record_id)].fields
                )
                constructor_bindings = tuple(
                    (
                        old_id,
                        (
                            "caller_value"
                            if old_id in constructor_parameter_ids
                            else "caller_storage"
                        ),
                        new_id,
                    )
                    for old_id, new_id in remap.items()
                    if old_id in {int(value.id) for value in constructor.args}
                )
                unresolved = tuple(
                    int(value.id) for value in constructor.args
                    if int(value.id) not in remap
                )
                enclosing_loop_ids = tuple(
                    int(plan.loop.node_id)
                    for plan in sorted(
                        (
                            plan for plan in caller_shell.loop_plans
                            if int(node_id) in plan.loop.body_nodes
                        ),
                        key=lambda plan: -len(plan.loop.body_nodes),
                    )
                )
                enclosing_loop_ids = lexical_loop_ids(
                    caller_graph, int(node_id), enclosing_loop_ids
                )
                requires_instance_pool = (
                    loop_constructor_requires_instance_pool(
                        caller_graph, receiver_id, enclosing_loop_ids
                    )
                )
                if requires_instance_pool:
                    destination_ids = tuple(dict.fromkeys(
                        int(caller_graph.nodes[parent].get("value_id", parent))
                        for successor in caller_graph.successors(receiver_id)
                        for successor_data in (caller_graph.nodes[successor],)
                        if str(successor_data.get("op") or "").lower()
                        in {"append", "add"}
                        for parent, role in (
                            successor_data.get("parents") or ()
                        )
                        if str(role) == "operand"
                    ))
                    mapped_leaf_fields = tuple(
                        field
                        for record in mapped_records.values()
                        for field in record.fields
                        if field.storage is not SSARecordFieldStorage.RECORD
                    )
                    pooled_fields = tuple(
                        field for field in mapped_leaf_fields
                        if field.sequence_id is not None
                    )
                    scalar_fields = tuple(
                        field for field in mapped_leaf_fields
                        if field.storage is SSARecordFieldStorage.SCALAR
                    )
                    poolable_fields = (*pooled_fields, *scalar_fields)
                    if (
                        len(destination_ids) == 1
                        and len(poolable_fields) == len(mapped_leaf_fields)
                        and poolable_fields
                    ):
                        destination = caller_sequences.by_id(
                            destination_ids[0]
                        )
                        field_pools = []
                        pool_specs = []
                        for pooled_field in pooled_fields:
                            field_sequence = caller_sequences.by_id(
                                pooled_field.sequence_id
                            )
                            template_field = next(
                                field
                                for record in template_records
                                for field in record.fields
                                if field.storage_identity
                                == pooled_field.storage_identity
                            )
                            callee_sequence = constructor_sequences.by_id(
                                template_field.sequence_id
                            )
                            if field_sequence is None or callee_sequence is None:
                                pool_specs = []
                                break
                            row_stride_id = next_value_id
                            next_value_id += 1
                            row_stride = SSAValue(
                                row_stride_id,
                                dtype="int",
                                accounting={
                                    "record_instance_pool_stride": (
                                        str(pooled_field.storage_identity)
                                    ),
                                    "constructor_callsite_id": int(node_id),
                                },
                            )
                            caller.args.append(row_stride)
                            available[row_stride_id] = row_stride
                            pool = SSAChildTablePoolDescriptor(
                                handle_column=0,
                                column_value_ids=tuple(
                                    field_sequence.column_value_ids
                                ),
                                length_value_id=int(
                                    field_sequence.length_address_id
                                ),
                                capacity_value_id=int(
                                    field_sequence.capacity_value_id
                                ),
                                row_stride_value_id=row_stride_id,
                                status_value_id=(
                                    None
                                    if field_sequence.status_address_id is None
                                    else int(field_sequence.status_address_id)
                                ),
                                live_flags_value_id=(
                                    None
                                    if field_sequence.live_flags_value_id is None
                                    else int(field_sequence.live_flags_value_id)
                                ),
                                column_dtypes=tuple(
                                    field_sequence.column_dtypes
                                ),
                                key_columns=tuple(
                                    field_sequence.key_columns
                                ),
                                writable=bool(field_sequence.writable),
                            )
                            field_pools.append(SSARecordInstancePoolField(
                                storage_identity=str(
                                    pooled_field.storage_identity
                                ),
                                storage=SSARecordFieldStorage.SEQUENCE,
                                sequence_pool=pool,
                            ))
                            pool_specs.append({
                                "pool": pool,
                                "callee_field": template_field,
                                "callee_sequence": callee_sequence,
                            })
                        scalar_specs = []
                        scalar_source_ids = tuple(dict.fromkeys(
                            int(value_id)
                            for field in scalar_fields
                            for value_id in field.value_ids
                        ))
                        if len(scalar_source_ids) > 1:
                            pool_specs = []
                        elif scalar_fields:
                            scalar_source_id = scalar_source_ids[0]
                            scalar_stride_id = next_value_id
                            next_value_id += 1
                            scalar_stride = SSAValue(
                                scalar_stride_id,
                                dtype="int",
                                accounting={
                                    "record_instance_pool_scalar_stride": (
                                        str(class_identity)
                                    ),
                                    "constructor_callsite_id": int(node_id),
                                },
                            )
                            caller.args.append(scalar_stride)
                            available[scalar_stride_id] = scalar_stride
                            for scalar_field in scalar_fields:
                                template_field = next(
                                    field
                                    for record in template_records
                                    for field in record.fields
                                    if field.storage_identity
                                    == scalar_field.storage_identity
                                )
                                field_pools.append(SSARecordInstancePoolField(
                                    storage_identity=str(
                                        scalar_field.storage_identity
                                    ),
                                    storage=SSARecordFieldStorage.SCALAR,
                                    scalar_value_id=scalar_source_id,
                                    scalar_stride_value_id=scalar_stride_id,
                                    scalar_offset=int(
                                        scalar_field.offset or 0
                                    ),
                                ))
                                scalar_specs.append({
                                    "arena_value_id": scalar_source_id,
                                    "stride_value_id": scalar_stride_id,
                                    "offset": int(scalar_field.offset or 0),
                                    "callee_value_ids": tuple(map(
                                        int, template_field.value_ids
                                    )),
                                })
                        if destination is not None and pool_specs:
                            # Preserve the historical one-field projection for
                            # existing nested-table consumers. Multi-field
                            # records use the record-level grouping below.
                            caller_sequences.sequences[
                                int(destination.sequence_id)
                            ] = replace(
                                destination,
                                child_table_pool=(
                                    pool_specs[0]["pool"]
                                    if len(pool_specs) == 1 else None
                                ),
                            )
                            record_pool = SSARecordInstancePoolDescriptor(
                                int(destination.sequence_id),
                                tuple(field_pools),
                            )
                            caller_records.records[receiver_id] = replace(
                                caller_records.records[receiver_id],
                                instance_pool=record_pool,
                            )
                            constructor_instance_pools[(
                                caller_symbol, int(node_id)
                            )] = {
                                "receiver_id": receiver_id,
                                "destination_sequence_id": int(
                                    destination.sequence_id
                                ),
                                "fields": tuple(pool_specs),
                                "scalar_fields": tuple(scalar_specs),
                            }
                            requires_instance_pool = False
                constructor_calls.append(SSACallRecord(
                    caller=caller_symbol,
                    callsite_id=int(node_id),
                    callee_reference=int(constructor_method.function_reference),
                    callee_name=str(constructor_method.name),
                    callee_symbol=constructor_symbol,
                    argument_bindings=((receiver_id, constructor_self_id),),
                    enclosing_loop_ids=enclosing_loop_ids,
                    callee_storage_value_ids=tuple(
                        int(value.id) for value in constructor.args
                    ),
                    frame_bindings=constructor_bindings,
                    unresolved_frame_value_ids=unresolved,
                    decomposition=(
                        "requires_loop_instance_pool"
                        if requires_instance_pool else None
                    ),
                ))
                later_values = sorted(
                    int(data.get("value_id", other_id))
                    for other_id, data in caller_graph.nodes(data=True)
                    if int(other_id) > int(node_id)
                )
                constructor_anchors[(caller_symbol, int(node_id))] = (
                    later_values[0] if later_values else None
                )

    call_records: dict[str, list[SSACallRecord]] = {}
    result_storage_bindings_by_call: dict[
        tuple[str, int], dict[int, int]
    ] = {}
    call_anchor_value_ids: dict[tuple[str, int], int | None] = {}
    seen_calls: set[tuple[str, int, int | None]] = set()
    for caller_symbol, planned_call, caller_graph, caller_module, caller_shell in (
        pending_call_records
    ):
        call_data = caller_graph.nodes.get(int(planned_call.callsite_id), {})
        call_operation = str(
            call_data.get("op") or call_data.get("type") or ""
        ).casefold()
        if isinstance(call_data.get("expr_obj"), ast.Attribute):
            # A bound-method selector may carry the same method_ref as the
            # authored Call that consumes it.  It is a navigation value, not
            # a second invocation with an empty argument frame.
            continue
        if call_operation == "staticreference":
            # A class StaticReference is a navigable definition handle, not a
            # runtime constructor execution.  The real Call node carries the
            # same constructor_ref and is materialized above.
            continue
        if (str(caller_symbol), int(planned_call.callsite_id)) in (
            constructor_anchors
        ):
            # The record-ABI constructor occurrence is authoritative.  A
            # PlanCall for a specialized view of the same source Call would
            # otherwise become a second execution with a partial frame.
            continue
        attributes = call_data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref",
            attributes.get("method_ref", attributes.get("constructor_ref")),
        )
        call_key = (
            str(caller_symbol), int(planned_call.callsite_id),
            None if reference is None else int(reference),
        )
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)
        child_shell = getattr(caller_shell, "callsite_function_shells", {}).get(
            int(planned_call.callsite_id)
        )
        callee_symbol = (
            shell_symbols.get(id(child_shell))
            if child_shell is not None else None
        ) or (
            None if reference is None
            else function_symbols.get(int(reference))
        )
        callee_function = (
            None if callee_symbol is None
            else all_functions.get(callee_symbol)
        )
        child_graph = getattr(
            getattr(child_shell, "process_graph", None), "G", None
        )
        structural_caller_aliases: dict[int, int] = {}
        if planned_call.enclosing_loop_ids:
            carried_by_source: dict[int, list[int]] = {}
            for carried in (
                all_functions[caller_symbol].metadata.get(
                    "carried_port_values", {}
                ) or {}
            ).values():
                accounting = dict(getattr(carried, "accounting", None) or {})
                source_id = accounting.get("source_value_id")
                if source_id is not None:
                    carried_by_source.setdefault(int(source_id), []).append(
                        int(carried.id)
                    )
            structural_caller_aliases = {
                source_id: candidates[0]
                for source_id, candidates in carried_by_source.items()
                if len(set(candidates)) == 1
            }
        exact_bindings = {
            int(callee): structural_caller_aliases.get(
                int(caller), int(caller)
            )
            for caller, callee in planned_call.argument_bindings
        }
        # An identity-table history is a sequence of distinct SSA versions of
        # one authored spelling, not an alias class. Planned call bindings
        # already carry the exact deterministic value ids at the callsite;
        # mapping every earlier version to whichever later result happens to
        # be materialized changes program order (for example ``a = f(a)``
        # becomes ``a = f(a_after)``). Never infer frame aliases by spelling.
        identity_aliases: dict[int, int] = {}
        default_literals: dict[int, Any] = {}
        if child_graph is not None and callee_function is not None:
            for value in callee_function.args:
                value_id = int(value.id)
                node = child_graph.nodes.get(value_id)
                if node is None or str(node.get("type")) not in {
                    "Constant", "Const", "const",
                }:
                    continue
                node_attributes = node.get("attributes") or {}
                if "value" in node_attributes:
                    default_literals[value_id] = _copy_literal_payload(
                        node_attributes["value"]
                    )
                elif "constant" in node:
                    default_literals[value_id] = _copy_literal_payload(
                        node["constant"]
                    )
        if child_graph is not None and source_function_table is not None:
            child_reference = child_graph.graph.get("function_ref")
            try:
                child_entry = source_function_table.entry(int(child_reference))
            except (KeyError, TypeError, ValueError):
                child_entry = None
            callable_object = (
                None if child_entry is None
                else getattr(child_entry, "python_callable", None)
            )
            if (
                callable_object is None
                and child_entry is not None
                and "." in str(child_entry.qualified_name)
            ):
                parts = str(child_entry.qualified_name).split(".")
                for split in range(len(parts) - 1, 0, -1):
                    try:
                        candidate = importlib.import_module(
                            ".".join(parts[:split])
                        )
                    except ImportError:
                        continue
                    try:
                        for attribute in parts[split:]:
                            candidate = getattr(candidate, attribute)
                    except AttributeError:
                        continue
                    callable_object = candidate
                    break
            if callable_object is not None:
                try:
                    signature = inspect.signature(callable_object)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    identities = child_graph.graph.get("identity_table") or {}
                    for parameter in signature.parameters.values():
                        if parameter.default is inspect.Parameter.empty:
                            continue
                        history = tuple(identities.get(parameter.name, ()))
                        for value_id in history:
                            node = child_graph.nodes.get(int(value_id), {})
                            # Name history also contains every later SSA
                            # assignment to the parameter's spelling.  A
                            # Python default belongs only to the authored
                            # parameter Input; marking a local reassignment as
                            # the default turns real dataflow into ``None`` at
                            # every caller.
                            if (
                                str(node.get("type") or node.get("op") or "")
                                .casefold() != "input"
                            ):
                                continue
                            default_literals[int(value_id)] = parameter.default
        frame_bindings = []
        unresolved_frame = []
        receiver_record = None
        callee_record = None
        result_storage_bindings: dict[int, int] = {}
        result_storage_bindings_by_call[(
            str(caller_symbol), int(planned_call.callsite_id)
        )] = result_storage_bindings
        result_record_bindings: dict[int, int] = {}
        if callee_symbol is not None and callee_function is not None:
            callee_result_records = all_record_tables.get(callee_symbol)
            callee_result_sequences = all_sequence_tables.get(callee_symbol)
            caller_result_records = all_record_tables.setdefault(
                caller_symbol, SSARecordTable()
            )
            caller_result_sequences = all_sequence_tables.setdefault(
                caller_symbol, SSASequenceTable()
            )
            parameter_aliases = _linked_authored_parameter_aliases(
                all_functions[caller_symbol],
                callee_function,
                caller_graph,
                child_graph,
                planned_call.argument_bindings,
                caller_result_records,
                callee_result_records,
            )
            caller_values = function_values(all_functions[caller_symbol])
            # Only ids a node EXPLICITLY declares. The old fallback to
            # ``node_id`` reached ProcessGraph's node keys, which are ``id()``
            # values -- so ``max()`` was poisoned to a memory address and every
            # storage slot allocated below it was id()-scale. Real value ids
            # are monotonic (this file already relies on that to call
            # id()-carrying arguments dead code); an id()-scale FORMAL is the
            # same defect on the signature, where it displaces the positional
            # correlation and hands the emitted function a parameter no caller
            # could name or fill.
            caller_graph_ids = {
                int(data["value_id"])
                for _node_id, data in caller_graph.nodes(data=True)
                if "value_id" in data
            }
            next_result_storage_id = 1 + max(
                (*caller_values, *caller_graph_ids), default=0
            )

            def allocate_result_storage(old_id: int) -> int:
                nonlocal next_result_storage_id
                old_id = int(old_id)
                if old_id in result_storage_bindings:
                    return result_storage_bindings[old_id]
                new_id = next_result_storage_id
                next_result_storage_id += 1
                source = function_values(callee_function).get(
                    old_id, SSAValue(old_id)
                )
                source_parameter = dict(
                    getattr(source, "accounting", {}) or {}
                ).get("program_abi_parameter")
                value = clone_value(source, new_id, accounting={
                    **({
                        "program_abi_parameter": parameter_aliases[
                            str(source_parameter)
                        ],
                    } if str(source_parameter) in parameter_aliases else {}),
                    "returned_record_storage": str(callee_symbol),
                    "callsite_id": int(planned_call.callsite_id),
                })
                all_functions[caller_symbol].args.append(value)
                caller_values[new_id] = value
                result_storage_bindings[old_id] = new_id
                return new_id

            # A returned sequence can be a frame-owned result without being
            # nested in a record.  When both sides already publish exact
            # descriptors, correlate every physical member now so the callee
            # writes directly into the caller's result arena.  This is the
            # sequence analogue of the record-field mapping below and avoids
            # inventing an unrelated workspace plus a fictitious return
            # value for ``return bytes(out)``.
            for callee_result_id, caller_result_id in (
                planned_call.result_bindings
            ):
                callee_sequence = (
                    None if callee_result_sequences is None
                    else callee_result_sequences.by_id(int(callee_result_id))
                )
                caller_sequence = caller_result_sequences.by_id(
                    int(caller_result_id)
                )
                if callee_sequence is None or caller_sequence is None:
                    continue
                if (
                    len(callee_sequence.column_value_ids)
                    != len(caller_sequence.column_value_ids)
                    or tuple(callee_sequence.key_columns)
                    != tuple(caller_sequence.key_columns)
                ):
                    continue
                callee_members = (
                    *callee_sequence.column_value_ids,
                    callee_sequence.length_address_id,
                    callee_sequence.capacity_value_id,
                    *((callee_sequence.status_address_id,)
                      if callee_sequence.status_address_id is not None else ()),
                    *((callee_sequence.live_flags_value_id,)
                      if callee_sequence.live_flags_value_id is not None else ()),
                )
                caller_members = (
                    *caller_sequence.column_value_ids,
                    caller_sequence.length_address_id,
                    caller_sequence.capacity_value_id,
                    *((caller_sequence.status_address_id,)
                      if caller_sequence.status_address_id is not None else ()),
                    *((caller_sequence.live_flags_value_id,)
                      if caller_sequence.live_flags_value_id is not None else ()),
                )
                if len(callee_members) != len(caller_members):
                    continue
                result_storage_bindings.update(zip(
                    map(int, callee_members), map(int, caller_members),
                ))

            for callee_result_id, caller_result_id in (
                planned_call.result_bindings
            ):
                if callee_result_records is None:
                    continue
                root = callee_result_records.records.get(
                    int(callee_result_id)
                )
                if root is None or int(caller_result_id) in (
                    caller_result_records.records
                ):
                    continue
                source_records = {
                    int(record.record_id): record
                    for record in callee_result_records.records.values()
                }
                pending_records = [root]
                record_order = []
                seen_record_ids = set()
                while pending_records:
                    record = pending_records.pop()
                    record_id = int(record.record_id)
                    if record_id in seen_record_ids:
                        continue
                    seen_record_ids.add(record_id)
                    record_order.append(record)
                    for field in record.fields:
                        nested = (
                            None if field.record_id is None
                            else source_records.get(int(field.record_id))
                        )
                        if nested is not None:
                            pending_records.append(nested)
                result_record_bindings[int(root.record_id)] = int(
                    caller_result_id
                )
                for record in record_order:
                    if int(record.record_id) == int(root.record_id):
                        continue
                    result_record_bindings[int(record.record_id)] = (
                        next_result_storage_id
                    )
                    next_result_storage_id += 1
                for record in reversed(record_order):
                    mapped_fields = []
                    for field in record.fields:
                        mapped_sequence_id = None
                        if field.sequence_id is not None:
                            sequence = (
                                None if callee_result_sequences is None
                                else callee_result_sequences.by_id(
                                    int(field.sequence_id)
                                )
                            )
                            if sequence is not None:
                                sequence_ids = (
                                    *sequence.column_value_ids,
                                    sequence.length_address_id,
                                    sequence.capacity_value_id,
                                    *((sequence.status_address_id,)
                                      if sequence.status_address_id is not None
                                      else ()),
                                    *((sequence.live_flags_value_id,)
                                      if sequence.live_flags_value_id is not None
                                      else ()),
                                )
                                pool = sequence.child_table_pool
                                if pool is not None:
                                    sequence_ids = (
                                        *sequence_ids,
                                        *pool.column_value_ids,
                                        pool.length_value_id,
                                        pool.capacity_value_id,
                                        pool.row_stride_value_id,
                                        *((pool.status_value_id,)
                                          if pool.status_value_id is not None
                                          else ()),
                                        *((pool.live_flags_value_id,)
                                          if pool.live_flags_value_id is not None
                                          else ()),
                                    )
                                for value_id in sequence_ids:
                                    allocate_result_storage(int(value_id))
                                mapped_sequence_id = allocate_result_storage(
                                    int(sequence.sequence_id)
                                )
                                caller_result_sequences.register(
                                    SSASequenceDescriptor(
                                        sequence_id=mapped_sequence_id,
                                        column_value_ids=tuple(
                                            result_storage_bindings[int(value_id)]
                                            for value_id in sequence.column_value_ids
                                        ),
                                        length_address_id=result_storage_bindings[
                                            int(sequence.length_address_id)
                                        ],
                                        capacity_value_id=result_storage_bindings[
                                            int(sequence.capacity_value_id)
                                        ],
                                        status_address_id=(
                                            None
                                            if sequence.status_address_id is None
                                            else result_storage_bindings[int(
                                                sequence.status_address_id
                                            )]
                                        ),
                                        column_dtypes=tuple(
                                            sequence.column_dtypes
                                        ),
                                        key_columns=tuple(sequence.key_columns),
                                        live_flags_value_id=(
                                            None
                                            if sequence.live_flags_value_id is None
                                            else result_storage_bindings[int(
                                                sequence.live_flags_value_id
                                            )]
                                        ),
                                        capacity_policy=sequence.capacity_policy,
                                        writable=bool(sequence.writable),
                                        child_table_pool=map_child_pool(
                                            sequence.child_table_pool,
                                            result_storage_bindings,
                                        ),
                                    )
                                )
                        for value_id in field.value_ids:
                            allocate_result_storage(int(value_id))
                        mapped_fields.append(SSARecordFieldDescriptor(
                            name=field.name,
                            storage=field.storage,
                            storage_identity=field.storage_identity,
                            value_ids=tuple(
                                result_storage_bindings[int(value_id)]
                                for value_id in field.value_ids
                            ),
                            sequence_id=mapped_sequence_id,
                            record_id=(
                                None if field.record_id is None
                                else result_record_bindings[int(field.record_id)]
                            ),
                            offset=field.offset,
                            dtype=field.dtype,
                            writable=field.writable,
                        ))
                    mapped_record_id = result_record_bindings[
                        int(record.record_id)
                    ]
                    caller_result_records.register(SSARecordDescriptor(
                        mapped_record_id,
                        str(record.identity),
                        tuple(mapped_fields),
                    ))
        bound_record_pairs = []
        if callee_symbol is not None:
            callee_records = all_record_tables.get(callee_symbol)
            candidates = (
                () if callee_records is None
                else tuple(callee_records.records.values())
            )
            caller_records = all_record_tables.get(caller_symbol)
            if caller_records is not None:
                for candidate in candidates:
                    bound_receiver = exact_bindings.get(
                        int(candidate.record_id)
                    )
                    if bound_receiver is None:
                        continue
                    bound_record = caller_records.records.get(
                        int(bound_receiver)
                    )
                    if bound_record is not None:
                        bound_record_pairs.append((bound_record, candidate))
            if bound_record_pairs:
                receiver_record, callee_record = bound_record_pairs[0]
        storage_bindings = dict(result_storage_bindings)
        # A source parameter transformed through ``list(source)`` and
        # ``b"".join`` has two physical sequence views in the callee. When
        # the source is itself a compiler-resident local ``list[bytes]``, bind
        # those views to its logical-count arena and deterministic flattened
        # companion instead of allocating empty, unrelated call-frame slots.
        if callee_function is not None:
            caller_function = all_functions[caller_symbol]
            caller_joined_views = dict(
                caller_function.metadata.get("joined_sequence_views", ())
            )
            caller_joined_aliases = dict(
                caller_function.metadata.get(
                    "joined_sequence_identity_aliases", ()
                )
            )
            caller_sequences = all_sequence_tables.get(caller_symbol)
            callee_sequences = all_sequence_tables.get(callee_symbol)
            callee_parameter_ids = {
                str(name): int(value_id)
                for name, value_id in callee_function.metadata.get(
                    "parameter_names", ()
                )
            }

            def bind_sequence_members(callee_sequence, caller_sequence):
                _bind_sequence_storage_members(
                    storage_bindings, callee_sequence, caller_sequence
                )

            if caller_sequences is not None and callee_sequences is not None:
                # A planned argument binding names the sequence's data value,
                # but the callable ABI owns the complete descriptor.  Carry
                # its length/capacity/status storage across the same exact
                # binding or the callee receives a valid pointer paired with
                # freshly zeroed extents (for example ``_section(payload)``
                # after ``payload = _vector(types)``).
                for caller_id, callee_id in planned_call.argument_bindings:
                    bind_sequence_members(
                        callee_sequences.by_id(int(callee_id)),
                        caller_sequences.by_id(
                            int(structural_caller_aliases.get(
                                int(caller_id), int(caller_id)
                            ))
                        ),
                    )
                for transform_sequence_id, source_name, transform in (
                    callee_function.metadata.get(
                        "sequence_source_transforms", ()
                    )
                ):
                    callee_parameter_id = callee_parameter_ids.get(
                        str(source_name)
                    )
                    caller_source_id = (
                        None if callee_parameter_id is None
                        else exact_bindings.get(int(callee_parameter_id))
                    )
                    if caller_source_id is None:
                        continue
                    outer_id = int(caller_joined_aliases.get(
                        int(caller_source_id), int(caller_source_id)
                    ))
                    flat_id = caller_joined_views.get(outer_id)
                    if flat_id is None:
                        continue
                    caller_view_id = (
                        outer_id if str(transform) == "row_count"
                        else int(flat_id) if str(transform) == "join_bytes"
                        else None
                    )
                    if caller_view_id is None:
                        continue
                    bind_sequence_members(
                        callee_sequences.by_id(int(transform_sequence_id)),
                        caller_sequences.by_id(int(caller_view_id)),
                    )
        for bound_record, candidate in bound_record_pairs:
            caller_fields = {
                field.storage_identity: field
                for field in bound_record.fields
            }
            for field in candidate.fields:
                caller_field = caller_fields.get(field.storage_identity)
                if (
                    caller_field is None
                    or not caller_field.value_ids
                    or not field.value_ids
                ):
                    continue
                # Every descriptor member has this exact physical storage
                # identity. Multiple GetAttr occurrences are views, not
                # independent ABI arenas, so bind them all to the caller's
                # canonical field slot.
                caller_storage = int(caller_field.value_ids[0])
                storage_bindings.update(
                    (int(value_id), caller_storage)
                    for value_id in field.value_ids
                )
        if receiver_record is not None and callee_record is not None:
            caller_fields = {
                field.storage_identity: field
                for field in receiver_record.fields
            }
            # Repeated GetAttr/SetAttr occurrences may have distinct local
            # sequence ids even though the record table correctly identifies
            # one physical field.  Bind every descriptor proven to be another
            # view of that field.  The proof is structural: its sequence id is
            # one of the authored field-op value ids for the same slot and its
            # row contract matches the canonical descriptor.  This preserves
            # every occurrence while giving them one caller-owned arena.
            callee_sequence_table = all_sequence_tables.get(callee_symbol)
            caller_sequence_table = all_sequence_tables.get(caller_symbol)
            callee_shell = getattr(
                caller_shell, "callsite_function_shells", {}
            ).get(int(planned_call.callsite_id))
            callee_graph = getattr(
                getattr(callee_shell, "process_graph", None), "G", None
            )
            field_value_ids_by_identity = {}
            if callee_graph is not None:
                for node_id, data in callee_graph.nodes(data=True):
                    record_field = (data.get("attributes") or {}).get(
                        "record_field"
                    )
                    if not record_field or len(record_field) != 2:
                        continue
                    field_value_ids_by_identity.setdefault(
                        f"{record_field[0]}.{record_field[1]}", set()
                    ).add(int(data.get("value_id", node_id)))
            if (
                callee_sequence_table is not None
                and caller_sequence_table is not None
            ):
                canonical_pairs = []
                for field in callee_record.fields:
                    if field.sequence_id is None:
                        continue
                    caller_field = caller_fields.get(field.storage_identity)
                    if caller_field is None or caller_field.sequence_id is None:
                        continue
                    canonical_pairs.append((
                        field.storage_identity,
                        callee_sequence_table.by_id(field.sequence_id),
                        caller_sequence_table.by_id(caller_field.sequence_id),
                    ))
                for local in callee_sequence_table.sequences.values():
                    for storage_identity, canonical, resident in canonical_pairs:
                        if int(local.sequence_id) not in (
                            field_value_ids_by_identity.get(
                                str(storage_identity), set()
                            )
                        ):
                            continue
                        if canonical is None or resident is None:
                            continue
                        if (
                            len(local.column_value_ids)
                            != len(canonical.column_value_ids)
                            or tuple(local.key_columns)
                            != tuple(canonical.key_columns)
                            or bool(local.writable) != bool(canonical.writable)
                        ):
                            continue
                        local_ids = (
                            *local.column_value_ids,
                            local.length_address_id,
                            local.capacity_value_id,
                            *((local.status_address_id,)
                              if local.status_address_id is not None else ()),
                            *((local.live_flags_value_id,)
                              if local.live_flags_value_id is not None else ()),
                        )
                        resident_ids = (
                            *resident.column_value_ids,
                            resident.length_address_id,
                            resident.capacity_value_id,
                            *((resident.status_address_id,)
                              if resident.status_address_id is not None else ()),
                            *((resident.live_flags_value_id,)
                              if resident.live_flags_value_id is not None else ()),
                        )
                        if len(local_ids) == len(resident_ids):
                            storage_bindings.update(zip(
                                map(int, local_ids), map(int, resident_ids)
                            ))
                        break
        # Snapshot the physical frame. For a recursive call caller and callee
        # are the same Function object, and propagating a missing storage slot
        # appends to caller.args. Iterating the live list would therefore grow
        # the sequence forever.
        for value in (
            () if callee_function is None else tuple(callee_function.args)
        ):
            value_id = int(value.id)
            if value_id in storage_bindings:
                frame_bindings.append((
                    value_id, "caller_storage", storage_bindings[value_id]
                ))
            elif value_id in exact_bindings:
                caller_value_id = int(exact_bindings[value_id])
                caller_node = caller_graph.nodes.get(caller_value_id)
                caller_attributes = (
                    {} if caller_node is None
                    else caller_node.get("attributes") or {}
                )
                if (
                    caller_node is not None
                    and str(caller_node.get("type")) in {
                        "Constant", "Const", "const",
                    }
                    and (
                        "value" in caller_attributes
                        or "constant" in caller_node
                    )
                ):
                    frame_bindings.append((
                        value_id,
                        "caller_literal",
                        _copy_literal_payload(
                            caller_attributes.get(
                                "value", caller_node.get("constant")
                            )
                        ),
                    ))
                elif (
                    caller_node is not None
                    and str(
                        caller_node.get("type")
                        or caller_node.get("op")
                        or ""
                    ).casefold() == "staticreference"
                    and caller_attributes.get("function_ref") is not None
                ):
                    from ..transmogrifier.function_table import (
                        FunctionReference,
                    )

                    frame_bindings.append((
                        value_id,
                        "caller_literal",
                        FunctionReference(int(
                            caller_attributes["function_ref"]
                        )),
                    ))
                else:
                    frame_bindings.append((
                        value_id, "caller_value", caller_value_id
                    ))
            elif value_id in identity_aliases:
                frame_bindings.append((
                    value_id, "caller_alias", identity_aliases[value_id]
                ))
            elif value_id in default_literals:
                frame_bindings.append((
                    value_id, "default_literal", default_literals[value_id]
                ))
            elif "record_instance" in dict(value.accounting or {}):
                # Storage introduced while constructing an object remains
                # part of that function's physical ABI even when it is not a
                # published field of the returned record. Propagate those
                # descriptor/scratch slots through an ordinary wrapper call
                # instead of reintroducing a host-object boundary.
                frame_bindings.append((
                    value_id,
                    "caller_storage",
                    allocate_result_storage(value_id),
                ))
            else:
                # Every remaining callee argument is still a concrete member
                # of the repository-SSA physical frame.  It is neither an
                # authored argument, a default, nor correlated record storage,
                # so give the caller a distinct storage slot and propagate it
                # outward.  Leaving these as "unresolved" made list/tensor
                # descriptors, loop scratch, hook tables, and tape mechanics
                # look like opaque Python dependencies even though their full
                # contents were already present in the callee signature.
                frame_bindings.append((
                    value_id,
                    "caller_storage",
                    allocate_result_storage(value_id),
                ))
        decompositions = tuple(
            instruction
            for block in all_functions[caller_symbol].blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("decomposed_plan_call")
            and instruction.attributes.get("plan_callsite_id") is not None
            and int(instruction.attributes["plan_callsite_id"])
            == int(planned_call.callsite_id)
            and (
                reference is None
                or instruction.attributes.get("callee_reference") is None
                or reference is not None
                and int(instruction.attributes.get("callee_reference"))
                == int(reference)
            )
        )
        normalized_loop_ids = lexical_loop_ids(
            caller_graph,
            int(planned_call.callsite_id),
            tuple(planned_call.enclosing_loop_ids),
        )
        recursive_region = (
            str(caller_symbol) == str(callee_symbol)
            and bool(normalized_loop_ids)
            and bool(all_functions[caller_symbol].metadata.get(
                "recursion_table"
            ))
        )
        resolution = (
            "decomposed"
            if decompositions or recursive_region
            else "unresolved"
        )
        call_records.setdefault(caller_symbol, []).append(SSACallRecord(
            caller=caller_symbol,
            callsite_id=int(planned_call.callsite_id),
            callee_reference=(None if reference is None else int(reference)),
            callee_name=str(planned_call.callee.name),
            callee_symbol=callee_symbol,
            argument_bindings=tuple(planned_call.argument_bindings),
            result_bindings=tuple(planned_call.result_bindings),
            enclosing_loop_ids=normalized_loop_ids,
            callee_storage_value_ids=(
                () if callee_function is None
                else tuple(int(value.id) for value in callee_function.args)
            ),
            frame_bindings=tuple(frame_bindings),
            unresolved_frame_value_ids=tuple(unresolved_frame),
            resolution=resolution,
            decomposition=(
                "recursion_region"
                if recursive_region
                else None if not decompositions
                else str(decompositions[0].attributes.get(
                    "ssa_sequence_operation"
                ))
            ),
        ))
        call_expression = call_data.get("expr_obj")
        call_position = (
            int(getattr(call_expression, "end_lineno", 0) or 0),
            int(getattr(call_expression, "end_col_offset", 0) or 0),
        )
        caller_result_ids = {
            int(instruction.res.id)
            for block in all_functions[caller_symbol].blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        anchors = []
        for node_id, node_data in caller_graph.nodes(data=True):
            expression = node_data.get("expr_obj")
            value_id = int(node_data.get("value_id", node_id))
            if expression is None or value_id not in caller_result_ids:
                continue
            position = (
                int(getattr(expression, "lineno", 0) or 0),
                int(getattr(expression, "col_offset", 0) or 0),
            )
            if position > call_position:
                anchors.append((*position, value_id))
        call_anchor_value_ids[(
            str(caller_symbol), int(planned_call.callsite_id)
        )] = min(anchors)[2] if anchors else None

    for record in constructor_calls:
        existing = call_records.setdefault(record.caller, [])
        duplicate_index = next((
            index for index, candidate in enumerate(existing)
            if int(candidate.callsite_id) == int(record.callsite_id)
            and candidate.callee_symbol == record.callee_symbol
        ), None)
        if duplicate_index is None:
            existing.append(record)
        else:
            candidate = existing[duplicate_index]
            # The hierarchy-owned constructor occurrence supplies lexical loop
            # ownership and explicit self/argument bindings; the record-ABI
            # occurrence supplies the complete caller-storage frame.  Merge
            # those two views into one execution record.
            existing[duplicate_index] = replace(
                record,
                argument_bindings=(
                    candidate.argument_bindings or record.argument_bindings
                ),
                result_bindings=(
                    candidate.result_bindings or record.result_bindings
                ),
                enclosing_loop_ids=(
                    candidate.enclosing_loop_ids or record.enclosing_loop_ids
                ),
            )
        call_anchor_value_ids[(record.caller, record.callsite_id)] = (
            constructor_anchors.get((record.caller, record.callsite_id))
        )

    # A method call can appear twice in the hierarchy catalogue: once as the
    # callable/attribute shell (no frame bindings) and once as the execution
    # occurrence (complete PlanCall bindings).  They are not two executions.
    # Prefer the complete frame for an identical caller/callee binding shape;
    # keep genuinely distinct complete occurrences and keep an incomplete one
    # only when no complete record can supersede it.
    for caller_symbol, records in tuple(call_records.items()):
        # Node/value ids retain authored expression order in a ProcessGraph.
        # Constructors were recovered after ordinary PlanCalls, so restore the
        # shared source order before fixed-point materialization.  Inserting
        # each call ahead of the same Ret then preserves ``construct; method``
        # rather than accidentally reversing them.
        records.sort(key=lambda record: int(record.callsite_id))
        complete_keys = {
            (record.callee_reference, record.callee_symbol)
            for record in records
            if not record.unresolved_frame_value_ids
        }
        call_records[caller_symbol] = [
            record for record in records
            if not (
                record.unresolved_frame_value_ids
                and not record.argument_bindings
                and not record.result_bindings
                and (record.callee_reference, record.callee_symbol)
                in complete_keys
            )
        ]

    # Specialize the two authored recursive fallbacks to the repository
    # operations they define. Native tensor targets implement zero-fill and
    # element count directly; retaining their Python list recursion as another
    # runtime call would duplicate that mechanism and, historically, left an
    # empty control shell with an unresolved self-call.
    from ..common.tensors.backward_registry import eps as backward_epsilon

    for caller_symbol, records in tuple(call_records.items()):
        caller = all_functions[caller_symbol]
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_values = function_values(caller)
        rebuilt = []
        for record in records:
            if (
                record.resolution == "unresolved"
                and str(record.caller) == str(record.callee_symbol)
                and record.callee_name in {"zmap", "_count"}
            ):
                rebuilt.append(replace(
                    record,
                    resolution="decomposed",
                    decomposition=(
                        "fill_zero" if record.callee_name == "zmap"
                        else "tensor_numel"
                    ),
                ))
                continue
            if (
                record.resolution == "unresolved"
                and record.callee_name == "eps"
                and len(record.result_bindings) == 1
            ):
                _callee_result_id, caller_result_id = record.result_bindings[0]
                result = caller_values.get(
                    int(caller_result_id),
                    SSAValue(int(caller_result_id), dtype="float64"),
                )
                # ``backward_registry.eps`` is an authored scalar helper.
                # The source-call placeholder may have inherited a caller's
                # tensor descriptor before decomposition; none of that shape
                # belongs to the constant which replaces the call.
                result.dtype = "float64"
                result.shape = ()
                result.accounting = {
                    **dict(result.accounting or {}),
                    "physical_dtype": "float64",
                }
                intrinsic = Instr(
                    "Const", [], result,
                    attributes={
                        "value": float(backward_epsilon()),
                        "structural_operation": "backward_epsilon",
                    },
                )
                inserted = False
                for block in caller.blocks.values():
                    for index, instruction in enumerate(block.instrs):
                        if any(
                            int(argument.id) == int(caller_result_id)
                            for argument in instruction.args
                        ):
                            block.instrs[index:index] = [intrinsic]
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    for block in caller.blocks.values():
                        if block.instrs and block.instrs[-1].op in {
                            "Ret", "ret", "Return", "return"
                        }:
                            block.instrs[-1:-1] = [intrinsic]
                            inserted = True
                            break
                if inserted:
                    caller.args = [
                        value for value in caller.args
                        if int(value.id) != int(caller_result_id)
                    ]
                    caller_values[int(caller_result_id)] = result
                    rebuilt.append(replace(
                        record,
                        resolution="decomposed",
                        decomposition="backward_epsilon",
                    ))
                    continue
            if (
                record.resolution == "unresolved"
                and record.callee_name == "_count"
                and len(record.result_bindings) == 1
            ):
                source_id = next((
                    int(caller_id)
                    for caller_id, callee_id in record.argument_bindings
                    if int(callee_id) == 0
                ), None)
                _callee_result_id, caller_result_id = (
                    record.result_bindings[0]
                )
                source = caller_values.get(source_id)
                shape = tuple(getattr(source, "shape", ()) or ())
                if not shape and caller_graph is not None and source_id is not None:
                    def inherited_shape(
                        value_id: int, seen: frozenset[int] = frozenset()
                    ) -> tuple[Any, ...]:
                        value_id = int(value_id)
                        if value_id in seen:
                            return ()
                        resident = caller_values.get(value_id)
                        resident_shape = tuple(
                            getattr(resident, "shape", ()) or ()
                        )
                        if resident_shape:
                            return resident_shape
                        source_data = caller_graph.nodes.get(value_id, {})
                        tensor = source_data.get("tensor") or {}
                        tensor_shape = tuple(tensor.get("shape") or ())
                        if tensor_shape:
                            return tensor_shape
                        for parent, role in source_data.get("parents") or ():
                            if str(role) in {
                                "operand", "value", "base", "arg:0", "lhs"
                            }:
                                parent_shape = inherited_shape(
                                    int(parent), seen | {value_id}
                                )
                                if parent_shape:
                                    return parent_shape
                        return ()

                    shape = inherited_shape(int(source_id))
                if source is None and source_id is not None:
                    source = SSAValue(
                        int(source_id),
                        shape=tuple(shape),
                        accounting={
                            "externalized_intrinsic_source": "tensor_numel"
                        },
                    )
                    caller.args.append(source)
                    caller_values[int(source_id)] = source
                if source is not None:
                    result = SSAValue(int(caller_result_id), dtype="int64")
                    intrinsic = (
                        Instr(
                            "Const", [], result,
                            attributes={
                                "value": int(np.prod(shape, dtype=np.int64)),
                                "structural_operation": "tensor_numel",
                            },
                        )
                        if shape and all(int(extent) >= 0 for extent in shape)
                        else Instr(
                            "extent", [source], result,
                            attributes={
                                "tensor_operation": "extent",
                                "extent_kind": "numel",
                                "dim": -1,
                                "structural_operation": "tensor_numel",
                            },
                        )
                    )
                    inserted = False
                    for block in caller.blocks.values():
                        for index, instruction in enumerate(block.instrs):
                            if any(
                                int(argument.id) == int(caller_result_id)
                                for argument in instruction.args
                            ):
                                block.instrs[index:index] = [intrinsic]
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        for block in caller.blocks.values():
                            if block.instrs and block.instrs[-1].op in {
                                "Ret", "ret", "Return", "return"
                            }:
                                block.instrs[-1:-1] = [intrinsic]
                                inserted = True
                                break
                    if inserted:
                        caller.args = [
                            value for value in caller.args
                            if int(value.id) != int(caller_result_id)
                        ]
                        caller_values[int(caller_result_id)] = result
                        rebuilt.append(replace(
                            record,
                            resolution="decomposed",
                            decomposition="tensor_numel",
                        ))
                        continue
            rebuilt.append(record)
        call_records[caller_symbol] = rebuilt

    # A parameter specialized to a compile-time literal is not a runtime
    # scalar once the specialized body no longer consumes it.  This includes
    # higher-order FunctionTable references and ordinary immutable/default
    # literals such as ``None``. Once every incoming call proves the same
    # category, erase the dead argument from the physical frame and from those
    # bindings. No sentinel value is introduced; genuinely dynamic optionals
    # remain arguments and still require a tagged ABI.
    from ..transmogrifier.function_table import FunctionReference

    incoming_by_callee: dict[str, list[tuple[str, int]]] = {}
    for caller_symbol, records in call_records.items():
        for index, record in enumerate(records):
            incoming_by_callee.setdefault(
                str(record.callee_symbol), []
            ).append((str(caller_symbol), index))
    for callee_symbol, incoming in incoming_by_callee.items():
        callee = all_functions.get(callee_symbol)
        if callee is None or not incoming:
            continue
        consumed = {
            int(argument.id)
            for block in callee.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        removable = set()
        for argument in callee.args:
            argument_id = int(argument.id)
            if argument_id in consumed:
                continue
            bindings = []
            complete = True
            for caller_symbol, record_index in incoming:
                record = call_records[caller_symbol][record_index]
                binding = next((
                    (kind, source)
                    for callee_id, kind, source in record.frame_bindings
                    if int(callee_id) == argument_id
                ), None)
                if binding is None:
                    complete = False
                    break
                bindings.append(binding)
            if complete and bindings:
                function_references = all(
                    kind == "caller_literal"
                    and isinstance(source, FunctionReference)
                    for kind, source in bindings
                )
                literal_values = all(
                    kind in {"caller_literal", "default_literal"}
                    and not isinstance(source, FunctionReference)
                    for kind, source in bindings
                )
                if literal_values:
                    first = bindings[0][1]
                    try:
                        literal_values = all(
                            source == first for _kind, source in bindings[1:]
                        )
                    except (TypeError, ValueError):
                        literal_values = False
                if function_references or literal_values:
                    removable.add(argument_id)
        if not removable:
            continue
        callee.args = [
            argument for argument in callee.args
            if int(argument.id) not in removable
        ]
        for caller_symbol, record_index in incoming:
            record = call_records[caller_symbol][record_index]
            call_records[caller_symbol][record_index] = replace(
                record,
                frame_bindings=tuple(
                    binding for binding in record.frame_bindings
                    if int(binding[0]) not in removable
                ),
                callee_storage_value_ids=tuple(
                    value_id for value_id in record.callee_storage_value_ids
                    if int(value_id) not in removable
                ),
            )

    # Materialize the first ordinary repository-SSA call frames.  Eligibility
    # is contract based: every callee argument is explained, exactly one
    # planner result is bound, the callee's authored conditional catalogue is
    # fully lowered, and the callee itself has no unresolved source calls.
    # Anything outside that proof remains an unresolved call record.
    from ..transmogrifier.ssa import Instr, SSAValue

    # Calls form a dependency graph, not a declaration-order list.  Resolve it
    # to a fixed point: once every authored call in a leaf is materialized, its
    # callers become eligible in the next round, continuing outward through an
    # arbitrarily deep source closure.  A single pass strands whichever caller
    # happened to be visited before its callee and falsely reports complete
    # source as unresolved at emission.
    # Structural-result discovery may conservatively allocate caller-owned
    # storage before numerical aggregate lowering proves that the aggregate is
    # returned through SSA instead.  Such storage is not part of the authored
    # ABI.  Remove only allocations with this exact provenance when no
    # instruction consumes them; live record arenas remain ordinary arguments.
    for function in all_functions.values():
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        function.args = [
            argument for argument in function.args
            if not (
                (argument.accounting or {}).get("returned_record_storage")
                and int(argument.id) not in consumed_ids
            )
        ]
    changed = True
    while changed:
        changed = False
        callee_callers = {
            caller: tuple(records) for caller, records in call_records.items()
        }
        for caller_symbol, records in tuple(call_records.items()):
            caller = all_functions[caller_symbol]
            caller_graph = source_graphs_by_symbol.get(caller_symbol)
            values = {int(value.id): value for value in caller.args}
            # Ids genuinely produced by an existing instruction, as opposed to
            # a shapeless placeholder some other record's processing may have
            # `setdefault`-ed into `values` this same round. A record's own
            # authored callsite id (below, `returns_aggregate`) is not
            # guaranteed disjoint from some unrelated value's id drawn from a
            # different numbering source (e.g. a required-source-value
            # resolved earlier via aggregate unpacking) -- reusing an
            # already-produced id for a new, unrelated result would give two
            # different instructions the same SSA identity.
            produced_ids = {
                int(instruction.res.id)
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }
            values.update({
                int(instruction.res.id): instruction.res
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            })
            for pending_record in records:
                if len(pending_record.result_bindings) == 1:
                    _callee_id, caller_id = pending_record.result_bindings[0]
                    caller_tensor = (
                        caller_graph.nodes.get(int(caller_id), {}).get("tensor")
                        or {}
                    )
                    values.setdefault(int(caller_id), SSAValue(
                        int(caller_id),
                        dtype=caller_tensor.get("dtype"),
                        shape=tuple(caller_tensor.get("shape") or ()),
                    ))
                elif len(pending_record.result_bindings) > 1:
                    values.setdefault(
                        int(pending_record.callsite_id),
                        SSAValue(
                            int(pending_record.callsite_id),
                            accounting={
                                "ssa_aggregate_outputs": tuple(
                                    int(caller_id)
                                    for _callee_id, caller_id
                                    in pending_record.result_bindings
                                )
                            },
                        ),
                    )
            next_value_id = 1 + max(values, default=0)
            rebuilt_records = []

            def resolve_call_feed(
                source_id: int, prelude: list[Instr]
            ) -> SSAValue | None:
                """Resolve structural call feeds at their invocation site."""

                nonlocal next_value_id
                source_id = int(source_id)
                if source_id in values:
                    return values[source_id]
                if caller_graph is None:
                    return None
                data = caller_graph.nodes.get(source_id, {})
                operation = str(
                    data.get("op") or data.get("type") or ""
                ).casefold()
                attributes = data.get("attributes") or {}
                if (
                    operation == "call"
                    and str(attributes.get("static_python_reference") or "")
                    == "len"
                ):
                    sequence_id = next((
                        int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role) in {"arg:0", "operand", "value"}
                    ), None)
                    sequence_table = all_sequence_tables.get(caller_symbol)
                    descriptor = (
                        None if sequence_id is None or sequence_table is None
                        else sequence_table.by_id(sequence_id)
                    )
                    sequence_aliases = {
                        int(alias): int(resident)
                        for alias, resident in dict(
                            caller.metadata.get("value_aliases", {})
                        ).items()
                    }
                    seen_aliases: set[int] = set()
                    while (
                        descriptor is None
                        and
                        sequence_id is not None
                        and int(sequence_id) in sequence_aliases
                        and int(sequence_id) not in seen_aliases
                    ):
                        seen_aliases.add(int(sequence_id))
                        sequence_id = sequence_aliases[int(sequence_id)]
                        descriptor = (
                            None if sequence_table is None
                            else sequence_table.by_id(sequence_id)
                        )
                    descriptor = (
                        None if sequence_id is None or sequence_table is None
                        else descriptor
                    )
                    length_address = (
                        None if descriptor is None
                        else values.get(int(descriptor.length_address_id))
                    )
                    if length_address is None:
                        return None
                    result = SSAValue(source_id, dtype="int64")
                    prelude.append(Instr(
                        "Load", [length_address], result,
                        attributes={
                            "binding": "sequence_len_call_feed",
                            "sequence_id": int(sequence_id),
                        },
                    ))
                    values[source_id] = result
                    return result
                if operation == "getattr":
                    attribute = str(attributes.get(
                        "attribute", ""
                    ))
                    receiver_id = next((
                        int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role) in {
                            "value", "object", "base", "operand"
                        }
                    ), None)
                    table = all_record_tables.get(caller_symbol)
                    record = (
                        None if table is None or receiver_id is None
                        else table.records.get(receiver_id)
                    )
                    field = (
                        None if record is None else next((
                            field for field in record.fields
                            if str(field.name) == attribute
                        ), None)
                    )
                    if field is not None and len(field.value_ids) == 1:
                        return values.get(int(field.value_ids[0]))
                    return None
                if operation == "boolop":
                    operands = []
                    for parent, role in data.get("parents") or ():
                        if not str(role).startswith("value:"):
                            continue
                        operand = resolve_call_feed(int(parent), prelude)
                        if operand is None:
                            return None
                        operands.append(operand)
                    expression = data.get("expr_obj")
                    opcode = (
                        "LAnd"
                        if isinstance(getattr(expression, "op", None), ast.And)
                        else "LOr"
                        if isinstance(getattr(expression, "op", None), ast.Or)
                        else None
                    )
                    if opcode is None or len(operands) < 2:
                        return None
                    current = operands[0]
                    for index, operand in enumerate(operands[1:], start=1):
                        is_last = index == len(operands) - 1
                        result_id = source_id if is_last else next_value_id
                        if not is_last:
                            next_value_id += 1
                        result = SSAValue(
                            result_id,
                            dtype="bool",
                        )
                        prelude.append(Instr(
                            opcode, [current, operand], result,
                            attributes={
                                "structural_operation": "boolop",
                                "call_feed": True,
                            },
                        ))
                        current = result
                    values[source_id] = current
                    return current
                return None

            def pending_result_id(pending):
                if len(pending.result_bindings) == 1:
                    return int(pending.result_bindings[0][1])
                if len(pending.result_bindings) > 1:
                    return int(pending.callsite_id)
                return None

            def downstream_anchor(value_id, seen=frozenset()):
                value_id = int(value_id)
                if value_id in seen:
                    return None
                for candidate in records:
                    if not any(
                        str(kind) in {
                            "caller_value", "caller_alias", "caller_storage"
                        }
                        and int(source) == value_id
                        for _callee_id, kind, source in candidate.frame_bindings
                    ):
                        continue
                    candidate_result = pending_result_id(candidate)
                    if candidate_result is None:
                        continue
                    if any(
                        int(argument.id) == candidate_result
                        for block in caller.blocks.values()
                        for instruction in block.instrs
                        for argument in instruction.args
                    ):
                        return candidate_result
                    nested = downstream_anchor(
                        candidate_result, seen | {value_id}
                    )
                    if nested is not None:
                        return nested
                return None

            def source_loop_blocks(loop_id: int) -> frozenset[str]:
                """Return the CFG compartment owned by one authored loop.

                ``SSACallRecord.enclosing_loop_ids`` and the loop-header Phi
                both carry the reducer's source ProcessGraph identity.  Walk
                only the true/body side and stop at the header and false/exit
                edge; this preserves nested-loop blocks without inferring
                lexical ownership from block names or dictionary order.
                """

                loop_id = int(loop_id)
                header = next((
                    block
                    for block in caller.blocks.values()
                    if any(
                        instruction.op == "Phi"
                        and (
                            instruction.attributes.get(
                                "source_loop_node_id"
                            ) == loop_id
                            or instruction.attributes.get("source_name")
                            == f"iteration_{loop_id}"
                        )
                        for instruction in block.instrs
                    )
                ), None)
                branch = (
                    None if header is None else next((
                        instruction
                        for instruction in header.instrs
                        if instruction.op == "CondBr"
                    ), None)
                )
                if branch is None:
                    return frozenset()
                body_name = str(branch.attributes.get("true_target"))
                exit_name = str(branch.attributes.get("false_target"))
                owned = set()
                pending = [body_name]
                while pending:
                    block_name = pending.pop()
                    if (
                        block_name in owned
                        or block_name in {header.name, exit_name}
                        or block_name not in caller.blocks
                    ):
                        continue
                    owned.add(block_name)
                    pending.extend(caller.blocks[block_name].successors)
                return frozenset(owned)

            def replace_at_callsite_marker(
                record: SSACallRecord,
                sequence: list[Instr],
            ) -> bool:
                """Fill a scheduled call STATEMENT in place.

                When the control program scheduled this callsite as a
                statement (the ``__plan_callsite_N__`` marker), position is
                the plan's decision and linking only supplies the callee
                symbol and frame bindings. The marker's result objects are
                preserved by IDENTITY: downstream consumers -- a carried
                phi's latch operand above all -- already hold those exact
                SSAValue objects, so the spliced sequence's producers are
                rebound onto them rather than minting replacements.
                """

                for block in caller.blocks.values():
                    for index, instruction in enumerate(block.instrs):
                        attributes = instruction.attributes or {}
                        if not attributes.get("plan_callsite_marker"):
                            continue
                        if int(attributes.get("plan_callsite_id", -1)) != int(
                            record.callsite_id
                        ):
                            continue
                        marker_result = instruction.res
                        marker_output_ids = tuple(
                            int(v) for v in attributes.get("output_ids") or ()
                        )
                        block.instrs[index:index + 1] = sequence
                        if marker_result is not None:
                            rebound = False
                            for spliced in reversed(sequence):
                                if spliced.res is None:
                                    continue
                                if (
                                    int(spliced.res.id) == int(marker_result.id)
                                    or int(spliced.res.id) in marker_output_ids
                                ):
                                    spliced.res = marker_result
                                    values[int(marker_result.id)] = marker_result
                                    rebound = True
                                    break
                            if not rebound and sequence:
                                # A scalar native call publishes through its
                                # own result value; rebind the call itself.
                                for spliced in reversed(sequence):
                                    if spliced.op == "Call":
                                        spliced.res = marker_result
                                        values[int(marker_result.id)] = (
                                            marker_result
                                        )
                                        break
                        return True
                return False

            def insert_at_loop_anchor(
                record: SSACallRecord,
                sequence: list[Instr],
            ) -> bool:
                if not record.enclosing_loop_ids:
                    return False
                owned = source_loop_blocks(record.enclosing_loop_ids[-1])
                if not owned:
                    return False
                anchor_value_id = call_anchor_value_ids.get((
                    str(caller_symbol), int(record.callsite_id)
                ))
                if anchor_value_id is None:
                    return False
                produced_ids = {
                    int(instruction.res.id)
                    for instruction in sequence
                    if instruction.res is not None
                }
                for instruction in sequence:
                    produced_ids.update(
                        int(output_id) for output_id in (
                            (instruction.attributes or {}).get(
                                "output_ids"
                            ) or ()
                        )
                    )
                for block_name, block in caller.blocks.items():
                    if block_name not in owned:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if (
                            instruction.res is not None
                            and int(instruction.res.id)
                            == int(anchor_value_id)
                        ):
                            # The anchor names WHERE the authored program
                            # placed this call; region reordering may have
                            # legally moved a consumer of the call's outputs
                            # ahead of that anchor.  A call may never follow
                            # a consumer of its own results, so clamp to the
                            # earliest such consumer.
                            index = min(index, next((
                                consumer_index
                                for consumer_index, candidate in enumerate(
                                    block.instrs[:index]
                                )
                                if any(
                                    int(argument.id) in produced_ids
                                    for argument in candidate.args
                                )
                            ), index))
                            block.instrs[index:index] = sequence
                            return True
                return False

            for record in records:
                result_storage_bindings = (
                    result_storage_bindings_by_call.setdefault(
                        (str(record.caller), int(record.callsite_id)), {}
                    )
                )
                callee = (
                    None if record.callee_symbol is None
                    else all_functions.get(record.callee_symbol)
                )
                if callee is not None:
                    parameter_aliases = _linked_authored_parameter_aliases(
                        caller,
                        callee,
                        caller_graph,
                        source_graphs_by_symbol.get(str(record.callee_symbol)),
                        record.argument_bindings,
                        all_record_tables.get(str(caller_symbol)),
                        all_record_tables.get(str(record.callee_symbol)),
                    )
                    current_frame_ids = {
                        int(argument.id) for argument in callee.args
                    }
                    refreshed_frame_bindings = [
                        binding for binding in record.frame_bindings
                        if int(binding[0]) in current_frame_ids
                    ]
                    bound_frame_ids = {
                        int(binding[0]) for binding in refreshed_frame_bindings
                    }
                    # Linking a callee can grow its physical frame: storage
                    # required by a newly materialized nested call becomes an
                    # ordinary callee argument.  Call records are discovered
                    # before that fixed point, so extend the caller frame here
                    # instead of permanently stranding an otherwise complete
                    # call behind a stale argument snapshot.
                    if (
                        str(record.caller) != str(record.callee_symbol)
                    ):
                        for argument in callee.args:
                            argument_id = int(argument.id)
                            if argument_id in bound_frame_ids:
                                continue
                            # A callee argument that is a record field
                            # (``program_abi_parameter``/``program_abi_field``
                            # accounting) may already have its OWN physical
                            # value in the caller -- e.g. the caller reads the
                            # same mutable field directly elsewhere in its own
                            # body (materialize_parameter_record_abi minted it
                            # independently, before this call's frame was
                            # known to need it too). Cloning a fresh value
                            # here instead of reusing that one splits one
                            # logical field into two disconnected physical
                            # slots: only the freshly-cloned one is actually
                            # threaded through the call and mutated, while
                            # the caller's own pre-existing value (which may
                            # be what the caller itself returns/reports)
                            # stays frozen at its initial snapshot forever.
                            # Observed exactly this way for
                            # ``last_wave_speed``/``last_height_violation``/
                            # ``last_tracer_violation``: written only inside a
                            # deeply nested callee, each reported 0 in the
                            # compiled output because the caller's own return
                            # expression referenced the orphaned clone, not
                            # the one the call chain actually mutated.
                            field_key = None
                            argument_accounting = dict(
                                argument.accounting or {}
                            )
                            parameter_name = argument_accounting.get(
                                "program_abi_parameter"
                            )
                            field_name = argument_accounting.get(
                                "program_abi_field"
                            )
                            if parameter_name is not None:
                                parameter_name = parameter_aliases.get(
                                    str(parameter_name), str(parameter_name)
                                )
                            if parameter_name is not None and field_name is not None:
                                field_key = (str(parameter_name), str(field_name))
                            existing_storage = None
                            if field_key is not None:
                                existing_storage = next((
                                    caller_argument
                                    for caller_argument in caller.args
                                    if (
                                        str(
                                            (caller_argument.accounting or {})
                                            .get("program_abi_parameter")
                                        ),
                                        str(
                                            (caller_argument.accounting or {})
                                            .get("program_abi_field")
                                        ),
                                    ) == field_key
                                    and int(caller_argument.id) != argument_id
                                ), None)
                            if existing_storage is not None:
                                caller_storage = existing_storage
                            else:
                                caller_storage = clone_value(
                                    argument,
                                    next_value_id,
                                    accounting={
                                        **({
                                            "program_abi_parameter": str(
                                                parameter_name
                                            ),
                                        } if parameter_name is not None else {}),
                                        "linked_call_frame_storage": str(
                                            record.callee_symbol
                                        ),
                                        "callsite_id": int(record.callsite_id),
                                    },
                                )
                                next_value_id += 1
                                caller.args.append(caller_storage)
                                values[int(caller_storage.id)] = caller_storage
                            refreshed_frame_bindings.append((
                                argument_id,
                                "caller_storage",
                                int(caller_storage.id),
                            ))
                            bound_frame_ids.add(argument_id)
                            changed = True
                    record = replace(
                        record,
                        callee_storage_value_ids=tuple(
                            int(argument.id) for argument in callee.args
                        ),
                        frame_bindings=tuple(refreshed_frame_bindings),
                        unresolved_frame_value_ids=tuple(
                            int(value_id)
                            for value_id in record.unresolved_frame_value_ids
                            if int(value_id) in current_frame_ids
                        ),
                    )
                was_unresolved = record.resolution == "unresolved"
                callee_records = callee_callers.get(
                    str(record.callee_symbol), ()
                )
                callee_outputs = (
                    () if callee is None
                    else emit_outputs(record.callee_symbol, callee)
                )
                callee_aggregate_outputs = (
                    tuple((callee_outputs[0].accounting or {}).get(
                        "ssa_aggregate_outputs", ()
                    ))
                    if len(callee_outputs) == 1 else ()
                )
                if (
                    was_unresolved
                    and callee is not None
                    and record.result_bindings
                    and not callee_outputs
                ):
                    # A raise-boundary call: the callee records its only
                    # authored output as a structural shortfall (exception
                    # construction has no operator yet), and the caller never
                    # consumes the bound result -- the object exists only to
                    # be raised, which has no repository-SSA representation.
                    # Executing the call preserves authored execution (the
                    # tokenizer's tell() and offset arithmetic still run);
                    # fabricating a result would not. Dropping the dead
                    # bindings lets the ordinary void-call machinery resolve
                    # it. The abort semantics of raise itself remain a
                    # declared gap, recorded on the caller so an artifact
                    # audit sees exactly which callsites fall through.
                    shortfall_ids = {
                        int(item[0])
                        for item in callee.metadata.get(
                            "structural_output_shortfalls", ()
                        )
                    }
                    bound_caller_ids = {
                        int(caller_id)
                        for _callee_id, caller_id in record.result_bindings
                    }
                    consumed = set(map(int, caller.metadata.get(
                        "source_output_value_ids", ()
                    )))
                    source_graph = source_graphs_by_symbol.get(
                        str(record.caller)
                    )
                    if source_graph is not None:
                        # Calls are linked before every graph consumer has
                        # necessarily become an SSA instruction.  Treat an
                        # authored graph edge as consumption too; otherwise a
                        # perfectly live numerical call result can look like a
                        # dead raise-only object and have its exact result
                        # binding erased.  This is especially visible in an
                        # adjoint chain where bw_sum feeds bw_mul.
                        consumed.update(
                            int(value_id)
                            for value_id in bound_caller_ids
                            if value_id in source_graph
                            and (
                                source_graph.out_degree(value_id) > 0
                                or value_id in getattr(
                                    source_graph, "graph", {}
                                ).get("roots", ())
                            )
                        )
                    for block in caller.blocks.values():
                        for instruction in block.instrs:
                            consumed.update(
                                int(argument.id)
                                for argument in instruction.args
                            )
                    if (
                        {
                            int(callee_id)
                            for callee_id, _caller_id
                            in record.result_bindings
                        } <= shortfall_ids
                        and not (bound_caller_ids & consumed)
                    ):
                        record = replace(record, result_bindings=())
                        noted = tuple(caller.metadata.get(
                            "raise_boundary_callsites", ()
                        ))
                        entry = (
                            int(record.callsite_id),
                            str(record.callee_symbol),
                        )
                        if entry not in noted:
                            caller.metadata["raise_boundary_callsites"] = (
                                noted + (entry,)
                            )
                callee_record_table = all_record_tables.get(
                    str(record.callee_symbol)
                )
                caller_record_table = all_record_tables.get(
                    str(record.caller)
                )
                if (
                    callee_record_table is not None
                    and caller_record_table is None
                ):
                    caller_record_table = all_record_tables.setdefault(
                        str(record.caller), SSARecordTable()
                    )
                # A callee record can itself become physical during an inner
                # call-linking round. Materialize the corresponding caller
                # record at that moment rather than requiring it to have
                # existed during initial call discovery.
                if (
                    callee_record_table is not None
                    and caller_record_table is not None
                ):
                    for callee_id, caller_id in record.result_bindings:
                        callee_result_record = (
                            callee_record_table.records.get(int(callee_id))
                        )
                        if (
                            callee_result_record is None
                            or int(caller_id) in caller_record_table.records
                            or any(
                                field.sequence_id is not None
                                or field.record_id is not None
                                for field in (
                                    () if callee_result_record is None
                                    else callee_result_record.fields
                                )
                            )
                        ):
                            continue
                        live_result_map: dict[int, int] = {}
                        mapped_fields = []
                        callee_values = function_values(callee)
                        for field in callee_result_record.fields:
                            mapped_ids = []
                            for callee_value_id in map(int, field.value_ids):
                                caller_value_id = live_result_map.get(
                                    callee_value_id
                                )
                                if caller_value_id is None:
                                    caller_value_id = next_value_id
                                    next_value_id += 1
                                    source = callee_values.get(
                                        callee_value_id,
                                        SSAValue(
                                            callee_value_id,
                                            dtype=field.dtype,
                                        ),
                                    )
                                    value = clone_value(
                                        source,
                                        caller_value_id,
                                        accounting={
                                            "returned_record_storage": str(
                                                record.callee_symbol
                                            ),
                                            "callsite_id": int(
                                                record.callsite_id
                                            ),
                                            "late_record_surface": True,
                                        },
                                    )
                                    caller.args.append(value)
                                    values[caller_value_id] = value
                                    live_result_map[callee_value_id] = (
                                        caller_value_id
                                    )
                                mapped_ids.append(caller_value_id)
                            mapped_fields.append(SSARecordFieldDescriptor(
                                field.name,
                                field.storage,
                                storage_identity=field.storage_identity,
                                value_ids=tuple(mapped_ids),
                                sequence_id=field.sequence_id,
                                record_id=field.record_id,
                                offset=field.offset,
                                dtype=field.dtype,
                                writable=field.writable,
                            ))
                        caller_record_table.register(SSARecordDescriptor(
                            int(caller_id),
                            str(callee_result_record.identity),
                            tuple(mapped_fields),
                        ))
                        result_storage_bindings.update(live_result_map)
                # Record surfaces can become physical after initial call
                # discovery (for example, once a schema constructor's
                # defaulted fields and loop-carried values are recovered).
                # Refresh the result map from the live record tables on every
                # linking fixed-point pass. Stable storage identities, not a
                # stale discovery-time snapshot or source-local numeric ids,
                # prove which caller slot receives each callee field.
                if (
                    callee_record_table is not None
                    and caller_record_table is not None
                ):
                    for callee_id, caller_id in record.result_bindings:
                        callee_result_record = (
                            callee_record_table.records.get(int(callee_id))
                        )
                        caller_result_record = (
                            caller_record_table.records.get(int(caller_id))
                        )
                        if (
                            callee_result_record is None
                            or caller_result_record is None
                        ):
                            continue
                        caller_fields = {
                            str(field.storage_identity): field
                            for field in caller_result_record.fields
                        }
                        for callee_field in callee_result_record.fields:
                            caller_field = caller_fields.get(str(
                                callee_field.storage_identity
                            ))
                            if (
                                caller_field is None
                                or len(callee_field.value_ids)
                                != len(caller_field.value_ids)
                            ):
                                continue
                            result_storage_bindings.update(zip(
                                map(int, callee_field.value_ids),
                                map(int, caller_field.value_ids),
                            ))
                record_return_layouts = dict(
                    () if callee is None else callee.metadata.get(
                        "record_return_layouts", ()
                    )
                )
                live_record_result_map = {
                    int(field_id): int(result_storage_bindings[field_id])
                    for callee_id, _caller_id in record.result_bindings
                    for field_id in record_return_layouts.get(
                        int(callee_id), ()
                    )
                    if int(field_id) in result_storage_bindings
                }
                live_result_slots = set(live_record_result_map.values())
                if live_result_slots and callee is not None:
                    callee_values = {
                        int(argument.id): argument for argument in callee.args
                    }
                    refreshed_bindings = []
                    for callee_id, kind, source in record.frame_bindings:
                        source_id = (
                            int(source)
                            if str(kind) in {
                                "caller_value", "caller_alias", "caller_storage"
                            }
                            else None
                        )
                        if (
                            str(kind) == "caller_storage"
                            and source_id in live_result_slots
                            and live_record_result_map.get(int(callee_id))
                            != source_id
                        ):
                            argument = callee_values.get(
                                int(callee_id), SSAValue(int(callee_id))
                            )
                            replacement = clone_value(
                                argument,
                                next_value_id,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        record.callee_symbol
                                    ),
                                    "callsite_id": int(record.callsite_id),
                                    "split_from_result_storage": source_id,
                                },
                            )
                            next_value_id += 1
                            caller.args.append(replacement)
                            values[int(replacement.id)] = replacement
                            refreshed_bindings.append((
                                int(callee_id), "caller_storage",
                                int(replacement.id),
                            ))
                            changed = True
                        else:
                            refreshed_bindings.append((callee_id, kind, source))
                    record = replace(
                        record, frame_bindings=tuple(refreshed_bindings)
                    )
                if callee is not None:
                    callee_values = {
                        int(argument.id): argument for argument in callee.args
                    }
                    storage_identity_by_value = {}
                    if callee_record_table is not None:
                        for descriptor in callee_record_table.records.values():
                            for field in descriptor.fields:
                                for value_id in field.value_ids:
                                    storage_identity_by_value[int(value_id)] = (
                                        str(field.storage_identity)
                                    )
                    owner_by_slot = {}
                    slot_by_owner = {}
                    distinct_bindings = []
                    for callee_id, kind, source in record.frame_bindings:
                        if str(kind) != "caller_storage":
                            distinct_bindings.append((callee_id, kind, source))
                            continue
                        source_id = int(source)
                        storage_identity = storage_identity_by_value.get(
                            int(callee_id)
                        )
                        owner = (
                            ("record", storage_identity)
                            if storage_identity is not None
                            else ("value", int(callee_id))
                        )
                        first_owner = owner_by_slot.setdefault(source_id, owner)
                        if first_owner == owner:
                            distinct_bindings.append((callee_id, kind, source))
                            continue
                        replacement_id = slot_by_owner.get((source_id, owner))
                        if replacement_id is None:
                            argument = callee_values.get(
                                int(callee_id), SSAValue(int(callee_id))
                            )
                            replacement = clone_value(
                                argument,
                                next_value_id,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        record.callee_symbol
                                    ),
                                    "callsite_id": int(record.callsite_id),
                                    "split_from_unproven_alias": source_id,
                                },
                            )
                            next_value_id += 1
                            caller.args.append(replacement)
                            values[int(replacement.id)] = replacement
                            replacement_id = int(replacement.id)
                            slot_by_owner[(source_id, owner)] = replacement_id
                            changed = True
                        distinct_bindings.append((
                            int(callee_id), "caller_storage", replacement_id,
                        ))
                    record = replace(
                        record, frame_bindings=tuple(distinct_bindings)
                    )
                physical_result_bindings = []
                for callee_id, caller_id in record.result_bindings:
                    layout = tuple(record_return_layouts.get(
                        int(callee_id), ()
                    ))
                    if layout:
                        physical_result_bindings.extend(
                            (int(field_id), int(result_storage_bindings[field_id]))
                            for field_id in layout
                            if int(field_id) in result_storage_bindings
                        )
                    else:
                        physical_result_bindings.append((
                            int(callee_id), int(caller_id)
                        ))
                physical_result_bindings = tuple(physical_result_bindings)
                result_binding = (
                    physical_result_bindings[0]
                    if len(physical_result_bindings) == 1 else None
                )
                returns_structural_record = (
                    bool(record.result_bindings)
                    and callee_record_table is not None
                    and caller_record_table is not None
                    and all(
                        int(callee_id) in callee_record_table.records
                        and int(caller_id) in caller_record_table.records
                        for callee_id, caller_id in record.result_bindings
                    )
                )
                forwarded_aggregate = (
                    not record.result_bindings
                    and len(callee_outputs) > 1
                    and int(record.callsite_id) in set(map(
                        int,
                        caller.metadata.get("source_output_value_ids", ()),
                    ))
                )
                bound_aggregate_outputs = ()
                if (
                    len(record.result_bindings) == 1
                    and len(callee_aggregate_outputs) > 1
                ):
                    aggregate_id = int(record.result_bindings[0][1])
                    candidate_graphs = []
                    if caller_graph is not None:
                        candidate_graphs.append(caller_graph)
                    candidate_graphs.extend(
                        graph for graph in source_graphs_by_symbol.values()
                        if graph is not caller_graph
                    )
                    for aggregate_graph in candidate_graphs:
                        if aggregate_id not in aggregate_graph:
                            continue
                        aggregate_node = aggregate_graph.nodes[aggregate_id]
                        aggregate_attributes = (
                            aggregate_node.get("attributes") or {}
                        )
                        callee_ref = aggregate_attributes.get("callee_ref")
                        if (
                            record.callee_reference is not None
                            and callee_ref is not None
                            and int(callee_ref)
                            != int(record.callee_reference)
                        ):
                            continue
                        projections = []
                        projection_ids = set(map(
                            int, aggregate_graph.successors(aggregate_id)
                        ))
                        projection_ids.update(
                            int(child_id)
                            for child_id, _role
                            in aggregate_node.get(
                                "children", ()
                            )
                        )
                        projection_ids.update(
                            int(node_id)
                            for node_id, data
                            in aggregate_graph.nodes(data=True)
                            if any(
                                int(parent_id) == aggregate_id
                                and str(role) == "base"
                                for parent_id, role in data.get("parents", ())
                            )
                        )
                        for projection_id in projection_ids:
                            projection = aggregate_graph.nodes[projection_id]
                            if str(
                                projection.get("op")
                                or projection.get("type")
                                or ""
                            ).casefold() != "indexed":
                                continue
                            projection_index = (
                                projection.get("attributes") or {}
                            ).get("gradient_result_index")
                            if projection_index is None:
                                continue
                            projections.append((
                                int(projection_index), int(projection_id)
                            ))
                        projections.sort()
                        if tuple(
                            index for index, _node_id in projections
                        ) == tuple(range(len(callee_aggregate_outputs))):
                            bound_aggregate_outputs = tuple(
                                node_id for _index, node_id in projections
                            )
                            break
                    if not bound_aggregate_outputs:
                        downstream_projections = {
                            tuple(map(
                                int,
                                instruction.attributes.get("output_ids", ()),
                            ))
                            for block in caller.blocks.values()
                            for instruction in block.instrs
                            if instruction.op in {"Call", "call"}
                            and any(
                                int(argument.id) == aggregate_id
                                for argument in instruction.args
                            )
                            and len(tuple(
                                instruction.attributes.get("output_ids", ())
                            )) == len(callee_aggregate_outputs)
                        }
                        if len(downstream_projections) == 1:
                            bound_aggregate_outputs = next(iter(
                                downstream_projections
                            ))
                # Source output identities describe authored intent, but they
                # are not a native call result until lowering has retained a
                # physical SSA output.  Treating metadata alone as a result
                # made call linking index an empty ``callee_outputs`` tuple and
                # silently crossed precisely the unresolved object/tensor
                # boundary this table exists to preserve.
                returns_value = (
                    len(physical_result_bindings) == 1
                    and len(callee_outputs) == 1
                    and not callee_aggregate_outputs
                )
                returns_bound_aggregate = (
                    len(physical_result_bindings) == 1
                    and len(callee_aggregate_outputs) > 1
                    and len(bound_aggregate_outputs)
                    == len(callee_aggregate_outputs)
                )
                returns_aggregate = (
                    len(physical_result_bindings) > 1
                    and len(callee_outputs) == len(physical_result_bindings)
                ) or forwarded_aggregate
                returns_physical_result = (
                    returns_value
                    or returns_bound_aggregate
                    or returns_aggregate
                )
                returns_void = (
                    not record.result_bindings and not callee_outputs
                )
                eligible = (
                    was_unresolved
                    and callee is not None
                    and not record.unresolved_frame_value_ids
                    and (
                        returns_value
                        or returns_bound_aggregate
                        or returns_aggregate
                        or returns_void
                        or returns_structural_record
                    )
                    and record.decomposition != "requires_loop_instance_pool"
                    # Repository SSA calls bind linkable symbols, not
                    # recursively materialized function bodies. Requiring all
                    # of a callee's own calls to be resolved first imposes a
                    # false topological order and deadlocks every legitimate
                    # recursive/SCC call graph (tape traversal is one). Each
                    # occurrence is validated by its own frame and result
                    # contract; the module completeness audit catches any
                    # genuinely unresolved member after this fixed point.
                    and int(callee.metadata.get(
                        "source_conditional_count", 0
                    )) == int(callee.metadata.get(
                        "lowered_conditional_count", 0
                    ))
                )
                eligibility_reasons = tuple(filter(None, (
                    "not_pending" if not was_unresolved else None,
                    "missing_callee" if callee is None else None,
                    "unresolved_frame" if record.unresolved_frame_value_ids else None,
                    "unmaterialized_result" if not (
                        returns_value
                        or returns_bound_aggregate
                        or returns_aggregate
                        or returns_void
                        or returns_structural_record
                    ) else None,
                    "loop_instance_pool_required" if (
                        record.decomposition == "requires_loop_instance_pool"
                    ) else None,
                    "conditional_surface_incomplete" if (
                        callee is not None
                        and int(callee.metadata.get("source_conditional_count", 0))
                        != int(callee.metadata.get("lowered_conditional_count", 0))
                    ) else None,
                )))
                call_argument_failure = None
                binding_by_callee = {
                    int(value_id): (str(kind), source)
                    for value_id, kind, source in record.frame_bindings
                }
                # A scheduled marker inside a loop/branch has already bound
                # authored caller ids to the exact resident values at that
                # lexical point (for example a loop target's current indexed
                # load).  Frame records retain the stable authored ids; use
                # this plan-owned correlation when building the native call
                # instead of asking the function-wide value table for a
                # pre-loop spelling that may not exist.
                scheduled_sources: dict[int, SSAValue] = {}
                marker = next((
                    instruction
                    for block in caller.blocks.values()
                    for instruction in block.instrs
                    if instruction.attributes.get("plan_callsite_marker")
                    and int(instruction.attributes.get(
                        "plan_callsite_id", -1
                    )) == int(record.callsite_id)
                ), None)
                if marker is not None:
                    scheduled_sources = {
                        int(caller_id): argument
                        for (caller_id, _callee_id), argument in zip(
                            record.argument_bindings, marker.args
                        )
                    }
                if eligible:
                    call_arguments = []
                    constants = []
                    instance_pool = constructor_instance_pools.get((
                        str(caller_symbol), int(record.callsite_id)
                    ))
                    pooled_argument_ids = {}
                    pooled_setup = []
                    if instance_pool is not None:
                        target_loop_id = int(record.enclosing_loop_ids[-1])
                        induction = next((
                            instruction.res
                            for block in caller.blocks.values()
                            for instruction in block.instrs
                            if instruction.op == "Phi"
                            and instruction.res is not None
                            and instruction.attributes.get("source_name")
                            == f"iteration_{target_loop_id}"
                        ), None)
                        if induction is None:
                            eligible = False
                        else:
                            destination_sequence_id = int(
                                instance_pool["destination_sequence_id"]
                            )
                            for block in caller.blocks.values():
                                for instruction in block.instrs:
                                    if (
                                        instruction.attributes.get(
                                            "ssa_sequence_operation"
                                        ) in {"append", "add"}
                                        and int(instruction.attributes.get(
                                            "sequence_id", -1
                                        )) == destination_sequence_id
                                        and instruction.args
                                        and int(instruction.args[-1].id)
                                        == int(instance_pool["receiver_id"])
                                    ):
                                        # A list/set of records stores the
                                        # pool row handle. Replace only the
                                        # authored inserted-value operand;
                                        # another ABI argument may legally
                                        # share its local numeric id.
                                        instruction.args[-1] = induction
                            for field_spec in instance_pool["fields"]:
                                pool = field_spec["pool"]
                                callee_sequence = field_spec["callee_sequence"]
                                row_offset = SSAValue(
                                    next_value_id, dtype="int"
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "Mul",
                                    [
                                        induction,
                                        values[int(pool.row_stride_value_id)],
                                    ],
                                    row_offset,
                                    attributes={
                                        "binding": "record_instance_pool_row"
                                    },
                                ))
                                pointer_sources = {
                                    **{
                                        int(callee_id): (
                                            int(source_id), row_offset
                                        )
                                        for callee_id, source_id in zip(
                                            callee_sequence.column_value_ids,
                                            pool.column_value_ids,
                                        )
                                    },
                                    int(callee_sequence.length_address_id): (
                                        int(pool.length_value_id), induction
                                    ),
                                    **(
                                        {
                                            int(callee_sequence.status_address_id): (
                                                int(pool.status_value_id),
                                                induction,
                                            )
                                        }
                                        if (
                                            callee_sequence.status_address_id
                                            is not None
                                            and pool.status_value_id is not None
                                        ) else {}
                                    ),
                                    **(
                                        {
                                            int(callee_sequence.live_flags_value_id): (
                                                int(pool.live_flags_value_id),
                                                row_offset,
                                            )
                                        }
                                        if (
                                            callee_sequence.live_flags_value_id
                                            is not None
                                            and pool.live_flags_value_id is not None
                                        ) else {}
                                    ),
                                }
                                pooled_argument_ids[
                                    int(callee_sequence.capacity_value_id)
                                ] = values[int(pool.row_stride_value_id)]
                                for callee_id, (source_id, offset) in (
                                    pointer_sources.items()
                                ):
                                    pointer = SSAValue(
                                        next_value_id,
                                        dtype=values[int(source_id)].dtype,
                                        accounting={
                                            "record_instance_pool_pointer": True
                                        },
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "GetElementPtr",
                                        [values[int(source_id)], offset],
                                        pointer,
                                        attributes={
                                            "binding": "record_instance_pool"
                                        },
                                    ))
                                    pooled_argument_ids[callee_id] = pointer
                            for scalar_spec in instance_pool.get(
                                "scalar_fields", ()
                            ):
                                scalar_base = SSAValue(
                                    next_value_id, dtype="int"
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "Mul",
                                    [
                                        induction,
                                        values[int(
                                            scalar_spec["stride_value_id"]
                                        )],
                                    ],
                                    scalar_base,
                                    attributes={
                                        "binding": (
                                            "record_instance_pool_scalar_row"
                                        )
                                    },
                                ))
                                scalar_index = scalar_base
                                if int(scalar_spec["offset"]):
                                    offset_value = SSAValue(
                                        next_value_id, dtype="int"
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "Const", [], offset_value,
                                        attributes={
                                            "value": int(
                                                scalar_spec["offset"]
                                            )
                                        },
                                    ))
                                    scalar_index = SSAValue(
                                        next_value_id, dtype="int"
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "Add",
                                        [scalar_base, offset_value],
                                        scalar_index,
                                        attributes={
                                            "binding": (
                                                "record_instance_pool_scalar_offset"
                                            )
                                        },
                                    ))
                                pointer = SSAValue(
                                    next_value_id,
                                    dtype=values[int(
                                        scalar_spec["arena_value_id"]
                                    )].dtype,
                                    accounting={
                                        "record_instance_pool_pointer": True
                                    },
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "GetElementPtr",
                                    [
                                        values[int(
                                            scalar_spec["arena_value_id"]
                                        )],
                                        scalar_index,
                                    ],
                                    pointer,
                                    attributes={
                                        "binding": (
                                            "record_instance_pool_scalar"
                                        )
                                    },
                                ))
                                for callee_id in scalar_spec[
                                    "callee_value_ids"
                                ]:
                                    pooled_argument_ids[int(callee_id)] = pointer
                    scalar_source_transforms = {
                        int(value_id): (str(source_name), str(transform))
                        for value_id, source_name, transform
                        in callee.metadata.get("scalar_source_transforms", ())
                    }
                    callee_sequence_table = all_sequence_tables.get(
                        str(record.callee_symbol)
                    )
                    sequence_source_transforms = tuple(
                        (
                            int(sequence_id), str(source_name), str(transform)
                        )
                        for sequence_id, source_name, transform
                        in callee.metadata.get(
                            "sequence_source_transforms", ()
                        )
                    )
                    callee_argument_positions = {
                        int(argument.id): index
                        for index, argument in enumerate(callee.args)
                    }
                    for argument in callee.args:
                        if int(argument.id) in pooled_argument_ids:
                            call_arguments.append(
                                pooled_argument_ids[int(argument.id)]
                            )
                            continue
                        scalar_transform = scalar_source_transforms.get(
                            int(argument.id)
                        )
                        if (
                            scalar_transform is not None
                            and scalar_transform[1] in {
                                "materialized_length", "sequence_length",
                            }
                            and callee_sequence_table is not None
                        ):
                            source_name = scalar_transform[0]
                            source_sequence_id = next((
                                sequence_id
                                for sequence_id, candidate_source, transform
                                in sequence_source_transforms
                                if candidate_source == source_name
                                and transform == "row_count"
                            ), None)
                            source_sequence = (
                                None if source_sequence_id is None
                                else callee_sequence_table.by_id(
                                    int(source_sequence_id)
                                )
                            )
                            length_position = (
                                None if source_sequence is None
                                else callee_argument_positions.get(int(
                                    source_sequence.length_address_id
                                ))
                            )
                            if (
                                length_position is not None
                                and length_position < len(call_arguments)
                            ):
                                derived_length = SSAValue(
                                    next_value_id,
                                    dtype=str(argument.dtype or "int64"),
                                    accounting={
                                        "linked_scalar_source_transform": (
                                            scalar_transform[1]
                                        ),
                                        "source_name": source_name,
                                    },
                                )
                                next_value_id += 1
                                constants.append(Instr(
                                    "Load",
                                    [call_arguments[int(length_position)]],
                                    derived_length,
                                    attributes={
                                        "binding": (
                                            "linked_scalar_source_transform"
                                        ),
                                        "source_transform": (
                                            scalar_transform[1]
                                        ),
                                    },
                                ))
                                call_arguments.append(derived_length)
                                continue
                        binding = binding_by_callee.get(int(argument.id))
                        if binding is None:
                            eligible = False
                            break
                        kind, source = binding
                        if kind in {
                            "caller_value", "caller_alias", "caller_storage"
                        }:
                            value = scheduled_sources.get(int(source))
                            if value is None:
                                value = resolve_call_feed(
                                    int(source), constants
                                )
                            if value is None and kind == "caller_storage":
                                # A structural-record cleanup may remove a
                                # shapeless argument whose numeric id happens
                                # to alias a physical frame slot.  The binding
                                # still proves that slot belongs to this call,
                                # so restore the callee-shaped storage value
                                # rather than treating it as a Python input.
                                value = clone_value(
                                    argument,
                                    int(source),
                                    accounting={
                                        "linked_call_frame_storage": str(
                                            record.callee_symbol
                                        ),
                                        "callsite_id": int(
                                            record.callsite_id
                                        ),
                                    },
                                )
                                caller.args.append(value)
                                values[int(source)] = value
                            if value is None:
                                call_argument_failure = (
                                    f"missing_{kind}:{int(source)}"
                                )
                                eligible = False
                                break
                            call_arguments.append(value)
                        elif kind in {"default_literal", "caller_literal"}:
                            value = SSAValue(
                                next_value_id,
                                dtype=argument.dtype,
                                shape=argument.shape,
                            )
                            next_value_id += 1
                            constants.append(Instr(
                                "Const", [], value,
                                attributes={"value": source},
                            ))
                            call_arguments.append(value)
                        else:
                            call_argument_failure = f"unsupported_binding:{kind}"
                            eligible = False
                            break
                if eligible:
                    aliased_return_argument_index = None
                    result_frame_sync: list[Instr] = []
                    if returns_value:
                        _callee_result_id, caller_result_id = result_binding
                        callee_output = callee_outputs[0]
                        # A directly returned mutable aggregate (for example
                        # ``return bytes(out)``) is represented by the same
                        # repository value as the callee's sequence column.
                        # Its storage already crosses the call frame as an
                        # inout argument; manufacturing a second scalar/array
                        # result makes native backends pass one actual more
                        # than the callee declares.  Preserve the semantic
                        # caller result id for scheduling, but make the Call
                        # result alias the exact frame argument that owns the
                        # returned storage.
                        aliased_return_argument_index = next((
                            index
                            for index, argument in enumerate(callee.args)
                            if int(argument.id) == int(_callee_result_id)
                            and index < len(call_arguments)
                            and binding_by_callee.get(int(argument.id), (None,))[0]
                            in {
                                "caller_value", "caller_alias", "caller_storage"
                            }
                        ), None)
                        if aliased_return_argument_index is not None:
                            result = call_arguments[
                                int(aliased_return_argument_index)
                            ]
                            callee_sequence_table = all_sequence_tables.get(
                                str(record.callee_symbol)
                            )
                            caller_sequence_table = all_sequence_tables.get(
                                str(record.caller)
                            )
                            callee_result_sequence = (
                                None if callee_sequence_table is None
                                else callee_sequence_table.by_id(
                                    int(_callee_result_id)
                                )
                            )
                            caller_result_sequence = (
                                None if caller_sequence_table is None
                                else caller_sequence_table.by_id(
                                    int(caller_result_id)
                                )
                            )
                            if (
                                callee_result_sequence is not None
                                and caller_result_sequence is not None
                            ):
                                length_argument_index = next((
                                    index
                                    for index, argument in enumerate(callee.args)
                                    if int(argument.id) == int(
                                        callee_result_sequence.length_address_id
                                    )
                                ), None)
                                caller_length_address = values.get(int(
                                    caller_result_sequence.length_address_id
                                ))
                                if (
                                    length_argument_index is not None
                                    and length_argument_index < len(call_arguments)
                                    and caller_length_address is not None
                                ):
                                    returned_length = SSAValue(
                                        next_value_id, dtype="int64",
                                        accounting={
                                            "linked_sequence_result_length": True,
                                            "callsite_id": int(
                                                record.callsite_id
                                            ),
                                        },
                                    )
                                    next_value_id += 1
                                    result_frame_sync.extend((
                                        Instr(
                                            "Load",
                                            [call_arguments[
                                                int(length_argument_index)
                                            ]],
                                            returned_length,
                                            attributes={
                                                "binding": (
                                                    "linked_sequence_result_length"
                                                )
                                            },
                                        ),
                                        Instr(
                                            "Store",
                                            [
                                                returned_length,
                                                caller_length_address,
                                            ],
                                            None,
                                            attributes={
                                                "binding": (
                                                    "linked_sequence_result_length"
                                                )
                                            },
                                        ),
                                    ))
                        else:
                            result = values.get(int(caller_result_id), SSAValue(
                                int(caller_result_id),
                                dtype=callee_output.dtype,
                                shape=callee_output.shape,
                            ))
                        # The caller-side placeholder may predate source-call
                        # linking and therefore carry no useful type.  A
                        # resolved call's physical result contract is the
                        # callee output itself; copy it onto the retained SSA
                        # value instead of letting backend defaults silently
                        # turn predicates into floating-point values.
                        result.dtype = callee_output.dtype
                        result.shape = tuple(callee_output.shape)
                        result.device = callee_output.device
                        result.accounting = {
                            **dict(result.accounting or {}),
                            **dict(callee_output.accounting or {}),
                        }
                    elif returns_bound_aggregate:
                        _callee_result_id, caller_result_id = result_binding
                        result = values.get(
                            int(caller_result_id),
                            SSAValue(int(caller_result_id)),
                        )
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "ssa_aggregate_outputs": bound_aggregate_outputs,
                        }
                    elif returns_aggregate:
                        caller_result_id = int(record.callsite_id)
                        if caller_result_id in produced_ids:
                            # The call-site's own AST node id coincides with
                            # a value some OTHER, already-existing
                            # instruction already produces -- drawn from an
                            # unrelated numbering source (e.g. a
                            # required-source-value pulled out of a
                            # different call's aggregate output via
                            # `source_output_id`). Adopting it here would
                            # give two different instructions the same SSA
                            # identity, which is exactly the class of bug
                            # the freshening pass later in this function
                            # cannot safely repair (it renames a colliding
                            # `.res` in place but never rewrites the other
                            # instructions that already reference the old
                            # id by number). Allocate a genuinely fresh id
                            # for this call's own aggregate result instead.
                            caller_result_id = next_value_id
                            next_value_id += 1
                        result = values.get(
                            caller_result_id,
                            SSAValue(
                                caller_result_id,
                                accounting={
                                    "ssa_aggregate_outputs": tuple(
                                        (
                                            int(caller_id)
                                            for _callee_id, caller_id
                                            in physical_result_bindings
                                        ) if physical_result_bindings else (
                                            int(value.id)
                                            for value in callee_outputs
                                        )
                                    )
                                },
                            ),
                        )
                        produced_ids.add(caller_result_id)
                    else:
                        caller_result_id = (
                            int(record.result_bindings[0][1])
                            if returns_structural_record else None
                        )
                        result = None
                    native_call = Instr(
                        "Call", call_arguments, result,
                        attributes={
                            "callee": record.callee_symbol,
                            "source_linked": True,
                            "plan_callsite_id": record.callsite_id,
                            "callee_reference": record.callee_reference,
                            **({
                                "ssa_output_argument": int(
                                    aliased_return_argument_index
                                ),
                                "result_aliases_frame": True,
                                "semantic_result_id": int(caller_result_id),
                            } if aliased_return_argument_index is not None else {}),
                            **({
                                "result_convention": "ssa.aggregate",
                                "output_ids": tuple(
                                    bound_aggregate_outputs
                                    if returns_bound_aggregate else (
                                        int(caller_id)
                                        for _callee_id, caller_id
                                        in physical_result_bindings
                                    )
                                ),
                            } if (
                                returns_bound_aggregate or returns_aggregate
                            ) else {}),
                        },
                    )
                    aggregate_unpack = []
                    if returns_aggregate and result is not None:
                        callee_outputs_by_id = {
                            int(value.id): value for value in callee_outputs
                        }
                        for output_index, (callee_id, caller_id) in enumerate(
                            physical_result_bindings
                        ):
                            index_value = SSAValue(next_value_id, dtype="int")
                            next_value_id += 1
                            address = SSAValue(next_value_id, dtype="ptr")
                            next_value_id += 1
                            caller_node = caller_graph.nodes.get(
                                int(caller_id), {}
                            )
                            caller_tensor = caller_node.get("tensor") or {}
                            output = values.get(
                                int(caller_id),
                                SSAValue(
                                    int(caller_id),
                                    dtype=caller_tensor.get("dtype"),
                                    shape=tuple(
                                        caller_tensor.get("shape") or ()
                                    ),
                                ),
                            )
                            callee_output = callee_outputs_by_id.get(
                                int(callee_id)
                            )
                            if callee_output is not None:
                                # PlanCall result bindings are the exact type
                                # correlation.  The caller graph describes
                                # semantic source shape, but the callee's
                                # physical output owns the repository-SSA ABI.
                                output.dtype = callee_output.dtype
                                output.shape = tuple(callee_output.shape)
                                output.device = callee_output.device
                                output.accounting = {
                                    **dict(output.accounting or {}),
                                    **dict(callee_output.accounting or {}),
                                    "ssa_call_result_from": (
                                        str(record.callee_symbol),
                                        int(callee_id),
                                    ),
                                }
                            aggregate_unpack.extend((
                                Instr(
                                    "Const", [], index_value,
                                    attributes={"value": int(output_index)},
                                ),
                                Instr(
                                    "GetElementPtr",
                                    [result, index_value],
                                    address,
                                    attributes={
                                        "aggregate_index": int(output_index)
                                    },
                                ),
                                Instr(
                                    "Load", [address], output,
                                    attributes={
                                        "aggregate_index": int(output_index),
                                        "source_output_id": int(caller_id),
                                    },
                                ),
                            ))
                            values[int(caller_id)] = output
                    native_sequence = [
                        *constants, native_call, *result_frame_sync,
                        *aggregate_unpack
                    ]
                    # A source-linked call inside a loop is scheduled by the
                    # reducer's lexical call anchor within that exact loop
                    # compartment.  Its eventual result consumer may live at
                    # loop exit or function Ret and is therefore not a valid
                    # execution anchor.
                    inserted = replace_at_callsite_marker(
                        record, native_sequence
                    ) or insert_at_loop_anchor(
                        record, native_sequence
                    )
                    if returns_physical_result:
                        consumed_result_ids = (
                            {
                                int(caller_id)
                                for _callee_id, caller_id
                                in physical_result_bindings
                            } | {int(caller_result_id)}
                            if returns_aggregate
                            else {int(caller_result_id)}
                        )
                        if not inserted:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if any(
                                        int(value.id) in consumed_result_ids
                                        for value in instruction.args
                                    ):
                                        block.instrs[index:index] = native_sequence
                                        inserted = True
                                        break
                                if inserted:
                                    break
                    else:
                        if not inserted and record.enclosing_loop_ids:
                            target_loop_id = int(record.enclosing_loop_ids[-1])
                            header_name = next((
                                block.name
                                for block in caller.blocks.values()
                                if any(
                                    instruction.op == "Phi"
                                    and instruction.attributes.get("source_name")
                                    == f"iteration_{target_loop_id}"
                                    for instruction in block.instrs
                                )
                            ), None)
                            body_name = None
                            if header_name is not None:
                                header = caller.blocks[header_name]
                                branch = next((
                                    instruction for instruction in header.instrs
                                    if instruction.op == "CondBr"
                                ), None)
                                if branch is not None:
                                    body_name = branch.attributes.get(
                                        "true_target"
                                    )
                            body = caller.blocks.get(str(body_name))
                            if body is not None and instance_pool is not None:
                                destination_sequence_id = int(
                                    instance_pool["destination_sequence_id"]
                                )
                                insertion_index = next((
                                    index
                                    for index, instruction in enumerate(
                                        body.instrs
                                    )
                                    if instruction.attributes.get(
                                        "ssa_sequence_operation"
                                    ) in {"append", "add"}
                                    and int(instruction.attributes.get(
                                        "sequence_id", -1
                                    )) == destination_sequence_id
                                ), None)
                                if insertion_index is not None:
                                    # The constructor is the authored value
                                    # expression of this append/add. Initialize
                                    # its pool row before publishing the handle
                                    # into the containing sequence.
                                    body.instrs[insertion_index:insertion_index] = [
                                        *constants, *pooled_setup, native_call
                                    ]
                                    inserted = True
                            if (
                                not inserted
                                and
                                body is not None
                                and body.instrs
                                and body.instrs[-1].op in {
                                    "Br", "br", "Branch", "branch"
                                }
                            ):
                                body.instrs[-1:-1] = [
                                    *constants, *pooled_setup, native_call
                                ]
                                inserted = True
                        anchor_value_id = call_anchor_value_ids.get((
                            str(caller_symbol), int(record.callsite_id)
                        ))
                        if not inserted and anchor_value_id is not None:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if (
                                        instruction.res is not None
                                        and int(instruction.res.id)
                                        == int(anchor_value_id)
                                    ):
                                        block.instrs[index:index] = [
                                            *constants, native_call
                                        ]
                                        inserted = True
                                        break
                                if inserted:
                                    break
                        if not inserted and not record.enclosing_loop_ids:
                            for block in caller.blocks.values():
                                if (
                                    block.instrs
                                    and block.instrs[-1].op in {
                                        "Ret", "ret", "Return", "return"
                                    }
                                ):
                                    block.instrs[-1:-1] = [
                                        *constants, native_call
                                    ]
                                    inserted = True
                                    break
                        if (
                            not inserted
                            and not record.result_bindings
                            and (
                                int(record.callsite_id),
                                str(record.callee_symbol),
                            ) in tuple(caller.metadata.get(
                                "raise_boundary_callsites", ()
                            ))
                        ):
                            # A raise-boundary void call inside a loop: no
                            # callsite marker, no loop anchor, and no result
                            # consumer to anchor by -- the result is dead by
                            # construction. The one dominance-correct anchor
                            # left is its own argument's producer: place the
                            # call immediately after the last instruction
                            # producing one of its arguments (the error
                            # message chain, in the same conditional arm).
                            argument_ids = {
                                int(argument.id)
                                for argument in call_arguments
                            }
                            for block in caller.blocks.values():
                                position = None
                                for index, instruction in enumerate(
                                    block.instrs
                                ):
                                    if (
                                        instruction.res is not None
                                        and int(instruction.res.id)
                                        in argument_ids
                                    ):
                                        position = index
                                if position is not None:
                                    block.instrs[
                                        position + 1:position + 1
                                    ] = [*constants, native_call]
                                    inserted = True
                                    break
                    source_output_ids = tuple(map(
                        int,
                        caller.metadata.get("source_output_value_ids", ()),
                    ))
                    produced_results = {
                        int(caller_id): values[int(caller_id)]
                        for _callee_id, caller_id in record.result_bindings
                        if int(caller_id) in values
                    }
                    caller_record_table = all_record_tables.get(
                        caller_symbol
                    )
                    if caller_record_table is not None:
                        for source_output_id in source_output_ids:
                            if source_output_id in (
                                caller_record_table.records
                            ):
                                produced_results.setdefault(
                                    source_output_id,
                                    SSAValue(
                                        source_output_id,
                                        accounting={
                                            "structural_record_result": True,
                                            "callsite_id": int(
                                                record.callsite_id
                                            ),
                                        },
                                    ),
                                )
                    if (
                        returns_value
                        or returns_bound_aggregate
                        or forwarded_aggregate
                    ) and caller_result_id is not None and result is not None:
                        produced_results[int(caller_result_id)] = result
                    authored_results = {
                        value_id: produced_results[value_id]
                        for value_id in source_output_ids
                        if value_id in produced_results
                    }
                    if authored_results:
                        # A function whose body is solely ``return callee(...)``
                        # has no ordinary consumer instruction to anchor the
                        # call: control lowering emitted an empty Ret because
                        # PlanCall is linked afterward.  The same applies to an
                        # unpacked multi-result call: materializing the Call and
                        # its aggregate projections does not retroactively add
                        # those projections to Ret.  The source-output ledger is
                        # the exact authored order, so publish every produced
                        # result there whether the call already found an
                        # ordinary insertion anchor or must use Ret itself.
                        for block in caller.blocks.values():
                            if (
                                block.instrs
                                and block.instrs[-1].op in {
                                    "Ret", "ret", "Return", "return"
                                }
                            ):
                                if not inserted:
                                    block.instrs[-1:-1] = native_sequence
                                returned = {
                                    int(argument.id): argument
                                    for argument in block.instrs[-1].args
                                }
                                returned.update(authored_results)
                                block.instrs[-1].args = [
                                    returned[value_id]
                                    for value_id in source_output_ids
                                    if value_id in returned
                                ]
                                inserted = True
                                break
                    if (
                        not inserted
                        and returns_physical_result
                        and caller_result_id is not None
                    ):
                        # A source-call chain can have no materialized direct
                        # consumer yet: the next PlanCall is itself pending.
                        # Anchor the producer at the first downstream result
                        # that the scheduled SSA already consumes. This keeps
                        # dependency order without turning an intermediate
                        # source-call result into a host ABI argument.
                        anchor = downstream_anchor(int(caller_result_id))
                        if anchor is not None:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if any(
                                        int(argument.id) == int(anchor)
                                        for argument in instruction.args
                                    ):
                                        block.instrs[index:index] = [
                                            *constants, native_call
                                        ]
                                        inserted = True
                                        break
                                if inserted:
                                    break
                    if (
                        not inserted
                        and returns_physical_result
                        and record.enclosing_loop_ids
                    ):
                        target_loop_id = int(record.enclosing_loop_ids[-1])
                        header = next((
                            block
                            for block in caller.blocks.values()
                            if any(
                                instruction.op == "Phi"
                                and instruction.attributes.get("source_name")
                                == f"iteration_{target_loop_id}"
                                for instruction in block.instrs
                            )
                        ), None)
                        branch = (
                            None if header is None else next((
                                instruction for instruction in header.instrs
                                if instruction.op == "CondBr"
                            ), None)
                        )
                        body = (
                            None if branch is None else caller.blocks.get(str(
                                branch.attributes.get("true_target")
                            ))
                        )
                        if body is not None:
                            insertion_index = next((
                                index
                                for index, instruction in enumerate(body.instrs)
                                if instruction.attributes.get(
                                    "ssa_sequence_operation"
                                ) in {"append", "add", "store"}
                            ), None)
                            if insertion_index is None and body.instrs:
                                insertion_index = (
                                    len(body.instrs) - 1
                                    if body.instrs[-1].op in {
                                        "Br", "br", "Branch", "branch"
                                    }
                                    else len(body.instrs)
                                )
                            if insertion_index is not None:
                                body.instrs[insertion_index:insertion_index] = [
                                    *native_sequence
                                ]
                                inserted = True
                        if not inserted:
                            preceding_calls = []
                            for candidate_block in caller.blocks.values():
                                for candidate in candidate_block.instrs:
                                    candidate_callsite = candidate.attributes.get(
                                        "plan_callsite_id"
                                    )
                                    if (
                                        candidate.op in {"Call", "call"}
                                        and candidate_callsite is not None
                                        and int(candidate_callsite)
                                        < int(record.callsite_id)
                                    ):
                                        preceding_calls.append((
                                            int(candidate_callsite),
                                            candidate_block,
                                        ))
                            if preceding_calls:
                                _callsite, candidate_block = max(
                                    preceding_calls,
                                    key=lambda item: item[0],
                                )
                                insertion_index = len(candidate_block.instrs)
                                if (
                                    candidate_block.instrs
                                    and candidate_block.instrs[-1].op in {
                                        "Br", "br", "Branch", "branch",
                                        "Ret", "ret", "Return", "return",
                                    }
                                ):
                                    insertion_index -= 1
                                candidate_block.instrs[
                                    insertion_index:insertion_index
                                ] = native_sequence
                                inserted = True
                    if (
                        not inserted
                        and returns_physical_result
                        and not record.enclosing_loop_ids
                    ):
                        for block in caller.blocks.values():
                            if (
                                block.instrs
                                and block.instrs[-1].op in {
                                    "Ret", "ret", "Return", "return"
                                }
                            ):
                                block.instrs[-1:-1] = [
                                    *constants, native_call
                                ]
                                inserted = True
                                break
                    if inserted:
                        if returns_physical_result:
                            if aliased_return_argument_index is not None:
                                # Consumers were lowered against the authored
                                # call-result identity.  Once the call is
                                # placed, redirect those uses to the proven
                                # frame-owned storage.  Identity matching is
                                # sufficient here: a repository function has
                                # one SSA producer per value id, and the call
                                # linker has just established their exact
                                # result correlation.
                                for block in caller.blocks.values():
                                    for instruction in block.instrs:
                                        if instruction is native_call:
                                            continue
                                        instruction.args = [
                                            result
                                            if int(argument.id)
                                            == int(caller_result_id)
                                            else argument
                                            for argument in instruction.args
                                        ]
                            caller.args = [
                                value for value in caller.args
                                if int(value.id) != int(caller_result_id)
                            ]
                            values[int(caller_result_id)] = result
                        record = replace(record, resolution="native_call")
                        diagnostics = dict(caller.metadata.get(
                            "unresolved_call_diagnostics", {}
                        ))
                        diagnostics.pop(int(record.callsite_id), None)
                        if diagnostics:
                            caller.metadata[
                                "unresolved_call_diagnostics"
                            ] = diagnostics
                        else:
                            caller.metadata.pop(
                                "unresolved_call_diagnostics", None
                            )
                        changed = True
                if record.resolution == "unresolved":
                    diagnostics = dict(caller.metadata.get(
                        "unresolved_call_diagnostics", {}
                    ))
                    diagnostics[int(record.callsite_id)] = {
                        "callee": str(record.callee_symbol),
                        "reasons": (
                            *eligibility_reasons,
                            *((call_argument_failure,)
                              if call_argument_failure else ()),
                            *(
                                ("insertion_point_missing",)
                                if eligible else ()
                            ),
                        ),
                        "callee_output_count": len(callee_outputs),
                        "physical_result_count": len(
                            physical_result_bindings
                        ),
                        "semantic_result_count": len(
                            record.result_bindings
                        ),
                        "returns_structural_record": bool(
                            returns_structural_record
                        ),
                    }
                    caller.metadata["unresolved_call_diagnostics"] = diagnostics
                rebuilt_records.append(record)
            call_records[caller_symbol] = rebuilt_records

        # A call resolved in this round may have created physical fields for
        # a record-valued public result. Expand that Ret before the next round
        # so callers observe the callee's new aggregate surface as part of the
        # same dependency fixed point.
        for function_name, record_table in all_record_tables.items():
            function = all_functions.get(function_name)
            if function is None:
                continue
            current_values = function_values(function)
            layouts = dict(function.metadata.get(
                "record_return_layouts", ()
            ))
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {
                        "Ret", "ret", "Return", "return"
                    }:
                        continue
                    expanded = []
                    changed_return = False
                    for argument in instruction.args:
                        returned_record = record_table.records.get(
                            int(argument.id)
                        )
                        if returned_record is None:
                            expanded.append(argument)
                            continue
                        layout = tuple(
                            int(value_id)
                            for field in returned_record.fields
                            for value_id in field.value_ids
                            if int(value_id) in current_values
                        )
                        if not layout:
                            expanded.append(argument)
                            continue
                        carried = dict(
                            function.metadata.get("carried_port_values")
                            or {}
                        )
                        expanded.extend(
                            carried.get(
                                int(value_id), current_values[value_id]
                            )
                            for value_id in layout
                        )
                        layouts[int(returned_record.record_id)] = layout
                        changed_return = True
                    if changed_return:
                        instruction.args = expanded
                        changed = True
            if layouts:
                function.metadata["record_return_layouts"] = tuple(
                    layouts.items()
                )

    # Object/call-frame discovery precedes some ordinary SSA-producing passes.
    # Both phases allocate monotonically within the values visible at the
    # time, so a synthetic projection/index can otherwise reuse an authored
    # argument or output id materialized later.  This is target-neutral: two
    # distinct SSAValue objects may not own the same integer identity.
    #
    # Preserve arguments and authored/named outputs. Freshen only the other
    # result objects; every operand holds the object itself, so its edges,
    # ordering, types, and scheduling remain unchanged and no call/record ABI
    # identity needs rewriting.
    for function_name, function in all_functions.items():
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        ]
        occupied = {
            int(argument.id) for argument in function.args
        } | {
            int(instruction.res.id) for instruction in instructions
        }
        canonical: dict[int, Any] = {}
        for argument in function.args:
            canonical.setdefault(int(argument.id), argument)
        for output in emit_outputs(function_name, function):
            canonical.setdefault(int(output.id), output)
        for instruction in instructions:
            result_id = int(instruction.res.id)
            if int(instruction.attributes.get(
                "source_output_id", -1
            )) == result_id:
                canonical.setdefault(result_id, instruction.res)
        for instruction in instructions:
            canonical.setdefault(int(instruction.res.id), instruction.res)

        next_value_id = 1 + max(occupied, default=0)
        freshened: dict[int, int] = {}
        seen_objects: set[int] = set()
        for instruction in instructions:
            result = instruction.res
            object_id = id(result)
            if object_id in seen_objects:
                continue
            seen_objects.add(object_id)
            old_id = int(result.id)
            if canonical[old_id] is result:
                continue
            while next_value_id in occupied:
                next_value_id += 1
            result.id = next_value_id
            occupied.add(next_value_id)
            freshened[old_id] = next_value_id
            next_value_id += 1
        if freshened:
            function.metadata["freshened_synthetic_value_ids"] = tuple(
                sorted(freshened.items())
            )

    # A native Call is an equality constraint between each caller operand and
    # its callee parameter.  Settle that constraint in repository SSA so every
    # backend receives the same dtype and dynamic-rank facts.  An explicit
    # ABI/physical type on the formal remains authoritative; otherwise the
    # authored caller occurrence replaces a default/unaccounted formal type.
    changed_call_types = True
    while changed_call_types:
        changed_call_types = False
        for caller in all_functions.values():
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op not in {"Call", "call"}
                        or instruction.attributes.get("tensor_operation")
                    ):
                        continue
                    callee = all_functions.get(str(
                        instruction.attributes.get("callee") or ""
                    ))
                    if callee is None:
                        continue
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_rank = max(
                            len(tuple(actual.shape or ())),
                            int((actual.accounting or {}).get(
                                "program_abi_rank", 0
                            )),
                            int((actual.accounting or {}).get(
                                "ssa_call_rank", 0
                            )),
                        )
                        formal_rank = max(
                            len(tuple(formal.shape or ())),
                            int((formal.accounting or {}).get(
                                "program_abi_rank", 0
                            )),
                            int((formal.accounting or {}).get(
                                "ssa_call_rank", 0
                            )),
                        )
                        call_rank = max(actual_rank, formal_rank)
                        for value, rank in (
                            (actual, actual_rank), (formal, formal_rank)
                        ):
                            if rank == call_rank or call_rank == 0:
                                continue
                            value.accounting = {
                                **dict(value.accounting or {}),
                                "ssa_call_rank": call_rank,
                            }
                            changed_call_types = True

                        actual_dtype = str(actual.dtype or "")
                        formal_dtype = str(formal.dtype or "")
                        formal_accounting = dict(formal.accounting or {})
                        formal_is_physical = bool(
                            formal_accounting.get("physical_dtype")
                            or formal_accounting.get("program_abi_storage")
                        )
                        formal_is_contracted = bool(
                            formal_is_physical
                            or formal_accounting.get("ssa_call_dtype")
                        )
                        actual_is_exact_result = bool(
                            (actual.accounting or {}).get(
                                "ssa_call_result_from"
                            )
                        )
                        actual_is_link_storage = bool(
                            (actual.accounting or {}).get(
                                "returned_record_storage"
                            )
                            or (actual.accounting or {}).get(
                                "linked_call_frame_storage"
                            )
                        )
                        if (
                            actual_is_exact_result
                            and not formal_is_physical
                            and actual_dtype
                            and actual_dtype != "unknown"
                            and formal_dtype != actual_dtype
                        ):
                            # A PlanCall result binding correlates the callee's
                            # physical output with this caller value exactly.
                            # It outranks a dtype previously inferred onto the
                            # consumer formal, but never an explicit physical
                            # or program-ABI declaration.
                            formal.dtype = actual.dtype
                            formal.accounting = {
                                **formal_accounting,
                                "ssa_call_dtype": actual_dtype,
                                "ssa_call_result_source": tuple(
                                    (actual.accounting or {})[
                                        "ssa_call_result_from"
                                    ]
                                ),
                            }
                            changed_call_types = True
                        elif (
                            formal_is_contracted
                            and actual_is_link_storage
                            and formal_dtype
                            and formal_dtype != "unknown"
                            and actual_dtype != formal_dtype
                        ):
                            actual.dtype = formal.dtype
                            actual.accounting = {
                                **dict(actual.accounting or {}),
                                "ssa_call_dtype": formal_dtype,
                            }
                            changed_call_types = True
                        elif (
                            actual_dtype
                            and actual_dtype != "unknown"
                            and actual_dtype != formal_dtype
                            and not formal_is_contracted
                        ):
                            formal.dtype = actual.dtype
                            formal.accounting = {
                                **formal_accounting,
                                "ssa_call_dtype": actual_dtype,
                            }
                            changed_call_types = True
                        elif (
                            formal_dtype
                            and formal_dtype != "unknown"
                            and actual_dtype in {"", "unknown"}
                        ):
                            actual.dtype = formal.dtype
                            changed_call_types = True

    # Argument equality reaches its fixed point before result projection.
    # Projecting results inside that bidirectional loop lets provisional
    # consumer types feed back into their own producers and oscillate.  The
    # callee output ABI is now settled, so copy it outward once through the
    # exact aggregate result bindings, then update immediate consumer formals.
    exact_result_values: set[int] = set()
    for caller in all_functions.values():
        caller_values = function_values(caller)
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op not in {"Call", "call"}
                    or instruction.attributes.get("result_convention")
                    != "ssa.aggregate"
                ):
                    continue
                callee = all_functions.get(str(
                    instruction.attributes.get("callee") or ""
                ))
                output_ids = tuple(map(
                    int, instruction.attributes.get("output_ids", ())
                ))
                if callee is None or not output_ids:
                    continue
                callee_outputs = tuple(emit_outputs(callee.name, callee))
                if len(output_ids) != len(callee_outputs):
                    continue
                for caller_id, callee_output in zip(output_ids, callee_outputs):
                    caller_output = caller_values.get(caller_id)
                    if caller_output is None:
                        continue
                    caller_output.dtype = callee_output.dtype
                    caller_output.shape = tuple(callee_output.shape)
                    caller_output.device = callee_output.device
                    caller_output.accounting = {
                        **dict(caller_output.accounting or {}),
                        "ssa_call_result_from": (
                            str(callee.name), int(callee_output.id)
                        ),
                    }
                    exact_result_values.add(id(caller_output))
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee = all_functions.get(str(
                    instruction.attributes.get("callee") or ""
                ))
                if callee is None:
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    if id(actual) not in exact_result_values:
                        continue
                    formal_accounting = dict(formal.accounting or {})
                    if (
                        formal_accounting.get("physical_dtype")
                        or formal_accounting.get("program_abi_storage")
                    ):
                        continue
                    formal.dtype = actual.dtype
                    formal.shape = tuple(actual.shape)
                    formal.device = actual.device
                    formal.accounting = {
                        **formal_accounting,
                        "ssa_call_dtype": str(actual.dtype or "unknown"),
                        "ssa_call_result_source": tuple(
                            (actual.accounting or {})[
                                "ssa_call_result_from"
                            ]
                        ),
                    }

    # Linking a callee's own calls can expand its physical argument frame
    # after an incoming native Call was materialized in an earlier fixed-point
    # round. Refresh those already-emitted call operands from the final call
    # records so callsite and callee ABIs cannot drift by dependency order.
    for caller_symbol, records in call_records.items():
        caller = all_functions.get(caller_symbol)
        if caller is None:
            continue
        caller_values = function_values(caller)
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_records = all_record_tables.get(caller_symbol)

        def final_frame_value(source_id: int) -> SSAValue | None:
            source_id = int(source_id)
            value = caller_values.get(source_id)
            if value is not None or caller_graph is None:
                return value
            data = caller_graph.nodes.get(source_id, {})
            if str(
                data.get("op") or data.get("type") or ""
            ).casefold() != "getattr":
                return None
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) in {"value", "object", "base", "operand"}
            ), None)
            record = (
                None
                if caller_records is None or receiver_id is None
                else caller_records.records.get(receiver_id)
            )
            attribute = str((data.get("attributes") or {}).get(
                "attribute", ""
            ))
            field = (
                None if record is None else next((
                    field for field in record.fields
                    if str(field.name) == attribute
                ), None)
            )
            if field is None or len(field.value_ids) != 1:
                return None
            return caller_values.get(int(field.value_ids[0]))

        next_value_id = 1 + max(caller_values, default=0)
        for record in records:
            if record.resolution != "native_call":
                continue
            callee = all_functions.get(str(record.callee_symbol))
            if callee is None:
                continue
            binding_by_callee = {
                int(value_id): (str(kind), source)
                for value_id, kind, source in record.frame_bindings
            }
            call_site = next((
                (block, index, instruction)
                for block in caller.blocks.values()
                for index, instruction in enumerate(block.instrs)
                if instruction.op in {"Call", "call"}
                and instruction.attributes.get("source_linked")
                and instruction.attributes.get("plan_callsite_id") is not None
                and int(instruction.attributes["plan_callsite_id"])
                == int(record.callsite_id)
                and str(instruction.attributes.get("callee"))
                == str(record.callee_symbol)
            ), None)
            if call_site is None:
                continue
            block, index, instruction = call_site
            refreshed = []
            constants = []
            complete = True
            for argument in callee.args:
                binding = binding_by_callee.get(int(argument.id))
                if binding is None:
                    complete = False
                    break
                kind, source = binding
                if kind in {
                    "caller_value", "caller_alias", "caller_storage"
                }:
                    value = final_frame_value(int(source))
                    if value is None:
                        complete = False
                        break
                    exact_result_source = (argument.accounting or {}).get(
                        "ssa_call_result_from"
                    )
                    value_accounting = dict(value.accounting or {})
                    if (
                        exact_result_source
                        and not value_accounting.get("physical_dtype")
                        and not value_accounting.get("program_abi_storage")
                    ):
                        # The finalized frame binding is the planner's exact
                        # identity correlation, not a new type-inference
                        # opportunity.  Refreshing a late-created record slot
                        # must carry the already-settled callee result ABI with
                        # it; otherwise a provisional scalar default survives
                        # only because this operand was materialized after the
                        # call-type fixed point.
                        value.dtype = argument.dtype
                        value.shape = tuple(argument.shape)
                        value.device = argument.device
                        value.accounting = {
                            **value_accounting,
                            "ssa_call_result_from": tuple(
                                exact_result_source
                            ),
                            "ssa_call_dtype": str(
                                argument.dtype or "unknown"
                            ),
                        }
                    refreshed.append(value)
                    continue
                if kind in {"default_literal", "caller_literal"}:
                    if isinstance(source, FunctionReference):
                        complete = False
                        break
                    value = SSAValue(
                        next_value_id,
                        dtype=argument.dtype,
                        shape=argument.shape,
                    )
                    next_value_id += 1
                    constants.append(Instr(
                        "Const", [], value, attributes={"value": source},
                    ))
                    caller_values[int(value.id)] = value
                    refreshed.append(value)
                    continue
                complete = False
                break
            if not complete:
                continue
            if constants:
                block.instrs[index:index] = constants
            instruction.args = refreshed

    # Native call linking can create a caller-owned record descriptor and its
    # aggregate-unpack values after the earlier source-output recovery pass.
    # Finalize every such public record now so Ret exposes physical fields,
    # never the conceptual Python record handle.
    for function_name, function in all_functions.items():
        available = set(function_values(function))
        available.update(map(
            int, function.metadata.get("lowered_source_value_ids", ())
        ))
        value_aliases = {
            int(alias): int(source)
            for alias, source in dict(
                function.metadata.get("value_aliases", {})
            ).items()
        }
        changed_aliases = True
        while changed_aliases:
            changed_aliases = False
            for alias, source in value_aliases.items():
                if source in available and alias not in available:
                    available.add(alias)
                    changed_aliases = True
        graph = source_graphs_by_symbol.get(function_name)
        record_table = all_record_tables.get(function_name)
        if graph is not None and record_table is not None:
            for node_id, data in graph.nodes(data=True):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() != "getattr":
                    continue
                receiver_id = next((
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {
                        "value", "object", "base", "operand"
                    }
                ), None)
                record = (
                    None if receiver_id is None
                    else record_table.records.get(receiver_id)
                )
                attribute = str((data.get("attributes") or {}).get(
                    "attribute", ""
                ))
                field = (
                    None if record is None else next((
                        field for field in record.fields
                        if str(field.name) == attribute
                    ), None)
                )
                if (
                    field is not None
                    and field.value_ids
                    and all(int(value_id) in available
                            for value_id in field.value_ids)
                ):
                    available.add(int(node_id))
            # Fixed aggregates disappear as runtime objects once a record
            # field correlates their ordered leaves. Mark the authored tuple
            # identity satisfied only when every leaf is physically present;
            # this keeps semantic accounting exact without manufacturing an
            # aggregate slot merely to silence the frontier.
            changed_aggregates = True
            while changed_aggregates:
                changed_aggregates = False
                for node_id, data in graph.nodes(data=True):
                    attributes = dict(data.get("attributes") or {})
                    if attributes.get("aggregate_kind") not in {
                        "tuple", "list",
                    }:
                        continue
                    leaves = tuple(map(
                        int,
                        attributes.get("aggregate_leaf_value_ids") or (),
                    ))
                    value_id = int(data.get("value_id", node_id))
                    if (
                        leaves
                        and all(leaf in available for leaf in leaves)
                        and value_id not in available
                    ):
                        available.add(value_id)
                        changed_aggregates = True
        shortfalls = tuple(
            row for row in function.metadata.get(
                "structural_output_shortfalls", ()
            )
            if int(row[0]) not in available
        )
        if shortfalls:
            function.metadata["structural_output_shortfalls"] = shortfalls
        else:
            function.metadata.pop("structural_output_shortfalls", None)
        unresolved_required = tuple(
            row for row in function.metadata.get(
                "unresolved_required_source_values", ()
            )
            if int(row[0]) not in available
        )
        if unresolved_required:
            function.metadata[
                "unresolved_required_source_values"
            ] = unresolved_required
        else:
            function.metadata.pop(
                "unresolved_required_source_values", None
            )
    for function_name, record_table in all_record_tables.items():
        function = all_functions.get(function_name)
        if function is None:
            continue
        values = function_values(function)
        layouts = dict(function.metadata.get("record_return_layouts", ()))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                expanded = []
                changed_return = False
                for argument in instruction.args:
                    record = record_table.records.get(int(argument.id))
                    if record is None:
                        expanded.append(argument)
                        continue
                    layout = tuple(
                        int(value_id)
                        for field in record.fields
                        for value_id in field.value_ids
                        if int(value_id) in values
                    )
                    if not layout:
                        expanded.append(argument)
                        continue
                    # A component standing at a LoopResult port means the
                    # carried phi; the raw field value is the port's
                    # unwritten slot.
                    carried = dict(
                        function.metadata.get("carried_port_values") or {}
                    )
                    expanded.extend(
                        carried.get(int(value_id), values[value_id])
                        for value_id in layout
                    )
                    layouts[int(record.record_id)] = layout
                    changed_return = True
                if changed_return:
                    instruction.args = expanded
        if layouts:
            function.metadata["record_return_layouts"] = tuple(
                layouts.items()
            )

    # A constructed-record result is a compile-time correlation once every
    # consumer has been rewritten to its physical field arenas or pool handle.
    # Remove only the shapeless conceptual receiver argument; a sequence
    # capacity or other physical ABI value may legitimately share the same
    # source-local numeric id and must remain.
    for function_name, function in all_functions.items():
        record_table = all_record_tables.get(function_name)
        record_ids = set(
            () if record_table is None else map(int, record_table.records)
        )
        source_graph = source_graphs_by_symbol.get(function_name)
        if source_graph is not None:
            identities = source_graph.graph.get("identity_table") or {}
            for parameter_name in (
                source_graph.graph.get("parameter_record_abi") or {}
            ):
                record_ids.update(map(
                    int, identities.get(str(parameter_name), ())
                ))
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        function.args = [
            argument for argument in function.args
            if not (
                int(argument.id) in record_ids
                and int(argument.id) not in consumed_ids
                and argument.dtype is None
                and not argument.shape
                and not argument.accounting
            )
        ]
        unique_arguments = {}
        for argument in function.args:
            existing = unique_arguments.get(int(argument.id))
            if existing is None:
                unique_arguments[int(argument.id)] = argument
                continue
            existing_physical = bool(
                existing.dtype is not None
                or existing.shape
                or existing.accounting
            )
            argument_physical = bool(
                argument.dtype is not None
                or argument.shape
                or argument.accounting
            )
            if argument_physical and not existing_physical:
                unique_arguments[int(argument.id)] = argument
        function.args = list(unique_arguments.values())

    # The cleanup above deliberately removes conceptual record handles from
    # final physical signatures.  Some incoming Calls were refreshed before
    # that cleanup, so reconcile them once more from the exact call-frame
    # contract.  This is not positional trimming: each surviving formal is
    # rebound by its callee-local SSA identity, preserving ordinary SSA flow
    # when structural OOP values disappear from the native ABI.
    for caller_symbol, records in call_records.items():
        caller = all_functions.get(caller_symbol)
        if caller is None:
            continue
        caller_values = function_values(caller)
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_record_table = all_record_tables.get(caller_symbol)
        next_value_id = 1 + max(caller_values, default=0)

        def cleaned_frame_value(source_id: int) -> SSAValue | None:
            source_id = int(source_id)
            value = caller_values.get(source_id)
            if value is not None or caller_graph is None:
                return value
            data = caller_graph.nodes.get(source_id, {})
            if str(
                data.get("op") or data.get("type") or ""
            ).casefold() != "getattr":
                return None
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) in {"value", "object", "base", "operand"}
            ), None)
            descriptor = (
                None
                if caller_record_table is None or receiver_id is None
                else caller_record_table.records.get(receiver_id)
            )
            attribute = str((data.get("attributes") or {}).get(
                "attribute", ""
            ))
            field = (
                None if descriptor is None else next((
                    item for item in descriptor.fields
                    if str(item.name) == attribute
                ), None)
            )
            if field is None or len(field.value_ids) != 1:
                return None
            return caller_values.get(int(field.value_ids[0]))

        for record in records:
            if record.resolution != "native_call":
                continue
            callee = all_functions.get(str(record.callee_symbol))
            if callee is None:
                continue
            binding_by_callee = {
                int(value_id): (str(kind), source)
                for value_id, kind, source in record.frame_bindings
            }
            call_site = next((
                (block, index, instruction)
                for block in caller.blocks.values()
                for index, instruction in enumerate(block.instrs)
                if instruction.op in {"Call", "call"}
                and instruction.attributes.get("source_linked")
                and instruction.attributes.get("plan_callsite_id") is not None
                and int(instruction.attributes["plan_callsite_id"])
                == int(record.callsite_id)
                and str(instruction.attributes.get("callee"))
                == str(record.callee_symbol)
            ), None)
            if call_site is None:
                continue
            block, index, instruction = call_site
            refreshed = []
            constants = []
            for argument in callee.args:
                binding = binding_by_callee.get(int(argument.id))
                if binding is None:
                    refreshed = []
                    break
                kind, source = binding
                if kind in {
                    "caller_value", "caller_alias", "caller_storage"
                }:
                    value = cleaned_frame_value(int(source))
                    if value is None:
                        refreshed = []
                        break
                    refreshed.append(value)
                elif kind in {"default_literal", "caller_literal"}:
                    if isinstance(source, FunctionReference):
                        refreshed = []
                        break
                    value = SSAValue(
                        next_value_id,
                        dtype=argument.dtype,
                        shape=argument.shape,
                    )
                    next_value_id += 1
                    constants.append(Instr(
                        "Const", [], value, attributes={"value": source},
                    ))
                    caller_values[int(value.id)] = value
                    refreshed.append(value)
                else:
                    refreshed = []
                    break
            if len(refreshed) != len(callee.args):
                continue
            if constants:
                block.instrs[index:index] = constants
            instruction.args = refreshed

    # A table lookup on a keyed mapping walks the mapping's own declared
    # vectors.  Its descriptor was built during lowering from anonymous
    # storage -- (keys, values, length, capacity) fresh arguments -- because
    # the slots exist only after record materialization and call-frame
    # linking.  Every frame is linked now, so bind them: keys/values/length
    # are the owner's parts, and a caller-supplied mapping is always exactly
    # full, so capacity IS the length -- the same value fills both formal
    # positions.  Both formals must therefore agree with that one value's
    # real width (int64, matching keys/query below), not the generic
    # scalar-arena default: declaring either as int32 while the caller's
    # actual keyed-field length is int64 is a real Fortran ABI mismatch, not
    # a cosmetic one, since a shared value can only have one true width.
    # The status cell stays an ordinary frame-allocated scalar.
    _keyed_helper_dtypes = (
        ("int64", None), ("float64", None), ("int64", (1,)),
        ("int64", None), ("int", (1,)), ("int64", None),
    )
    for function in all_functions.values():
        parts_by_owner: dict[str, dict[str, Any]] = {}
        key_identity_owners: set[str] = set()
        for value in function.args:
            accounting = value.accounting or {}
            if (
                accounting.get("program_abi_value_identity") == "key"
                and accounting.get("program_abi_field") is not None
            ):
                key_identity_owners.add(str(
                    accounting["program_abi_field"]
                ))
            owner_name = accounting.get("program_abi_keyed_owner")
            part_name = accounting.get("program_abi_keyed_part")
            if owner_name is None or part_name is None:
                continue
            parts_by_owner.setdefault(str(owner_name), {})[
                str(part_name)
            ] = value
        for owner_name in key_identity_owners:
            parts = parts_by_owner.get(owner_name)
            if parts is not None and "keys" in parts:
                parts["values"] = parts["keys"]
        if not parts_by_owner:
            continue
        replaced_storage_ids: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                owner_name = instruction.attributes.get("keyed_lookup_owner")
                if owner_name is None or len(instruction.args) < 6:
                    continue
                parts = parts_by_owner.get(str(owner_name))
                if parts is None or any(
                    name not in parts
                    for name in ("length", "keys", "values")
                ):
                    continue
                replaced_storage_ids.update(
                    int(argument.id) for argument in instruction.args[:4]
                )
                instruction.args[0] = parts["keys"]
                instruction.args[1] = parts["values"]
                instruction.args[2] = parts["length"]
                instruction.args[3] = parts["length"]
                helper = all_functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if helper is not None:
                    typed: dict[int, str] = {}
                    for argument, (dtype, shape) in zip(
                        helper.args, _keyed_helper_dtypes
                    ):
                        if argument.dtype in {None, "unknown", "None"}:
                            argument.dtype = dtype
                        if shape is not None and not tuple(
                            argument.shape or ()
                        ):
                            argument.shape = shape
                        typed[int(argument.id)] = str(argument.dtype)
                    # The body holds its own SSAValue instances for the same
                    # ids; retype them too, and give each Load the element
                    # type of the span it reads.
                    span_element = {
                        int(helper.args[0].id): "int64",
                        int(helper.args[1].id): "float64",
                    }
                    address_element: dict[int, str] = {}
                    for helper_block in helper.blocks.values():
                        for helper_instruction in helper_block.instrs:
                            for value in (
                                *helper_instruction.args,
                                *((helper_instruction.res,)
                                  if helper_instruction.res is not None
                                  else ()),
                            ):
                                refined = typed.get(int(value.id))
                                if refined is not None and value.dtype in {
                                    None, "unknown", "None",
                                }:
                                    value.dtype = refined
                            if (
                                helper_instruction.op == "GetElementPtr"
                                and helper_instruction.res is not None
                                and helper_instruction.args
                            ):
                                element = span_element.get(
                                    int(helper_instruction.args[0].id)
                                )
                                if element is not None:
                                    address_element[
                                        int(helper_instruction.res.id)
                                    ] = element
                            if (
                                helper_instruction.op == "Load"
                                and helper_instruction.res is not None
                                and helper_instruction.args
                                and helper_instruction.res.dtype in {
                                    None, "unknown", "None",
                                }
                            ):
                                element = address_element.get(
                                    int(helper_instruction.args[0].id)
                                )
                                if element is not None:
                                    helper_instruction.res.dtype = element
        if not replaced_storage_ids:
            continue
        still_consumed = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        dropped_positions = [
            position
            for position, value in enumerate(function.args)
            if int(value.id) in replaced_storage_ids
            and int(value.id) not in still_consumed
        ]
        if not dropped_positions:
            continue
        original_arity = len(function.args)
        function.args = [
            value
            for position, value in enumerate(function.args)
            if position not in set(dropped_positions)
        ]
        # A formal exists only together with the operand every caller feeds
        # it.  Dropping the formal alone leaves each call site one operand
        # too long, and the public-span origin walk skips calls whose arity
        # disagrees -- silently severing every span reached through this
        # function for every caller above it.
        function_symbol = next(
            (
                candidate_symbol
                for candidate_symbol, candidate in all_functions.items()
                if candidate is function
            ),
            None,
        )
        if function_symbol is None:
            continue
        for caller in all_functions.values():
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op != "Call"
                        or str(
                            instruction.attributes.get("callee") or ""
                        ) != function_symbol
                        or len(instruction.args) != original_arity
                    ):
                        continue
                    instruction.args = [
                        argument
                        for position, argument in enumerate(instruction.args)
                        if position not in set(dropped_positions)
                    ]

    # Reconcile public source provenance after the linked-frame fixed point.
    # Storage may be allocated during initial discovery or during a later
    # callee-growth round; doing this once over the final call records makes
    # the exact ``body -> self`` binding authoritative regardless of which
    # round minted the slot. Conflicting correlations remain unresolved.
    for caller_symbol, records in call_records.items():
        caller = all_functions.get(str(caller_symbol))
        caller_graph = source_graphs_by_symbol.get(str(caller_symbol))
        if caller is None:
            continue
        correlated_names: dict[int, set[str]] = {}
        for record in records:
            callee_symbol = str(record.callee_symbol or "")
            callee = all_functions.get(callee_symbol)
            if callee is None:
                continue
            aliases = _linked_authored_parameter_aliases(
                caller,
                callee,
                caller_graph,
                source_graphs_by_symbol.get(callee_symbol),
                record.argument_bindings,
                all_record_tables.get(str(caller_symbol)),
                all_record_tables.get(callee_symbol),
            )
            callee_arguments = {
                int(argument.id): argument for argument in callee.args
            }
            caller_arguments_by_id = {
                int(argument.id): argument for argument in caller.args
            }
            frame_map = {
                int(callee_id): int(caller_id)
                for callee_id, kind, caller_id in record.frame_bindings
                if str(kind) in {
                    "caller_storage", "caller_value", "caller_alias",
                }
            }
            # A linked physical frame is not ABI-complete until its aggregate
            # descriptors travel with it. Propagate only descriptors whose
            # every resident member has an exact frame binding.
            callee_sequences = all_sequence_tables.get(callee_symbol)
            caller_sequences = all_sequence_tables.setdefault(
                str(caller_symbol), SSASequenceTable()
            )
            if callee_sequences is not None:
                for descriptor in callee_sequences.sequences.values():
                    required_ids = {
                        int(descriptor.sequence_id),
                        *map(int, descriptor.column_value_ids),
                        int(descriptor.length_address_id),
                        int(descriptor.capacity_value_id),
                        *((int(descriptor.status_address_id),)
                          if descriptor.status_address_id is not None else ()),
                        *((int(descriptor.live_flags_value_id),)
                          if descriptor.live_flags_value_id is not None else ()),
                    }
                    pool = descriptor.child_table_pool
                    if pool is not None:
                        required_ids.update({
                            *map(int, pool.column_value_ids),
                            int(pool.length_value_id),
                            int(pool.capacity_value_id),
                            int(pool.row_stride_value_id),
                            *((int(pool.status_value_id),)
                              if pool.status_value_id is not None else ()),
                            *((int(pool.live_flags_value_id),)
                              if pool.live_flags_value_id is not None else ()),
                        })
                    if not required_ids.issubset(frame_map):
                        continue
                    mapped_columns = tuple(
                        frame_map[int(value_id)]
                        for value_id in descriptor.column_value_ids
                    )
                    if not any(
                        (caller_arguments_by_id.get(value_id).accounting or {})
                        .get("program_abi_parameter")
                        for value_id in mapped_columns
                        if caller_arguments_by_id.get(value_id) is not None
                    ):
                        # Private linked scratch remains workspace. A source
                        # sequence needs descriptor propagation so the outer
                        # caller can initialize its length/capacity contract.
                        continue
                    caller_sequences.register(SSASequenceDescriptor(
                        sequence_id=frame_map[int(descriptor.sequence_id)],
                        column_value_ids=mapped_columns,
                        length_address_id=frame_map[int(
                            descriptor.length_address_id
                        )],
                        capacity_value_id=frame_map[int(
                            descriptor.capacity_value_id
                        )],
                        status_address_id=(
                            None if descriptor.status_address_id is None else
                            frame_map[int(descriptor.status_address_id)]
                        ),
                        column_dtypes=tuple(descriptor.column_dtypes),
                        key_columns=tuple(descriptor.key_columns),
                        live_flags_value_id=(
                            None if descriptor.live_flags_value_id is None else
                            frame_map[int(descriptor.live_flags_value_id)]
                        ),
                        capacity_policy=descriptor.capacity_policy,
                        writable=bool(descriptor.writable),
                        child_table_pool=map_child_pool(pool, frame_map),
                    ))
            callee_records = all_record_tables.get(callee_symbol)
            caller_records = all_record_tables.setdefault(
                str(caller_symbol), SSARecordTable()
            )
            record_map = {
                int(callee_id): int(caller_id)
                for caller_id, callee_id in record.argument_bindings
            }
            if callee_records is not None:
                for descriptor in callee_records.records.values():
                    caller_record_id = record_map.get(int(
                        descriptor.record_id
                    ))
                    if caller_record_id is None:
                        continue
                    fields = []
                    for field in descriptor.fields:
                        if (
                            any(int(value_id) not in frame_map
                                for value_id in field.value_ids)
                            or field.sequence_id is not None
                            and int(field.sequence_id) not in frame_map
                            or field.record_id is not None
                            and int(field.record_id) not in record_map
                        ):
                            fields = []
                            break
                        fields.append(SSARecordFieldDescriptor(
                            name=field.name,
                            storage=field.storage,
                            storage_identity=field.storage_identity,
                            value_ids=tuple(
                                frame_map[int(value_id)]
                                for value_id in field.value_ids
                            ),
                            sequence_id=(
                                None if field.sequence_id is None else
                                frame_map[int(field.sequence_id)]
                            ),
                            record_id=(
                                None if field.record_id is None else
                                record_map[int(field.record_id)]
                            ),
                            offset=field.offset,
                            dtype=field.dtype,
                            writable=field.writable,
                        ))
                    if fields:
                        caller_records.register(SSARecordDescriptor(
                            caller_record_id,
                            str(descriptor.identity),
                            tuple(fields),
                        ))
            if not aliases:
                continue
            for callee_id, kind, caller_id in record.frame_bindings:
                if str(kind) != "caller_storage":
                    continue
                callee_argument = callee_arguments.get(int(callee_id))
                parameter_name = (
                    None if callee_argument is None else
                    (callee_argument.accounting or {}).get(
                        "program_abi_parameter"
                    )
                )
                mapped_name = aliases.get(str(parameter_name))
                if mapped_name is not None:
                    correlated_names.setdefault(
                        int(caller_id), set()
                    ).add(str(mapped_name))
        caller_arguments = {
            int(argument.id): argument for argument in caller.args
        }
        for value_id, names in correlated_names.items():
            if len(names) != 1 or value_id not in caller_arguments:
                continue
            value = caller_arguments[value_id]
            accounting = dict(value.accounting or {})
            if accounting.get("program_abi_parameter") is None:
                continue
            value.accounting = {
                **accounting,
                "program_abi_parameter": next(iter(names)),
                "linked_parameter_provenance": "exact_argument_binding",
            }

    # A declared record field keeps its storage identity across the call frame.
    # The contract states `height` as a rank-2 span, but a callee's formal
    # parameter was built before that contract was materialized, so it arrived
    # as an untyped scalar and every address into it became unresolvable. The
    # binding the caller already computed is the exact carrier: walk each call's
    # argument positions and give the callee's parameter the same field
    # identity. Nothing is inferred from names, and no field is invented -- an
    # argument only inherits what its own caller was already declared to hold.
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Call":
                    continue
                callee = all_functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if callee is None or len(callee.args) != len(instruction.args):
                    continue
                for fed, formal in zip(instruction.args, callee.args):
                    accounting = dict(fed.accounting or {})
                    if not accounting.get("program_abi_storage"):
                        continue
                    if (formal.accounting or {}).get("program_abi_storage"):
                        continue
                    # A keyed mapping's slot ids name values in the caller's
                    # own frame. The callee materializes its own slots from the
                    # same contract, so carrying these across would point at
                    # whatever happens to hold those ids there.
                    for frame_local in (
                        "program_abi_keyed_length",
                        "program_abi_keyed_keys",
                        "program_abi_keyed_values",
                    ):
                        accounting.pop(frame_local, None)
                    formal.accounting = {
                        **dict(formal.accounting or {}), **accounting,
                    }
                    if formal.dtype in {None, "unknown"} and fed.dtype:
                        formal.dtype = fed.dtype
                    # The declared rank travels in the field identity, not in
                    # `shape`. Only the rank is known here -- the extents are
                    # measured from the real buffer at call time -- and `shape`
                    # is the repository's *static* element-count contract, so
                    # naming symbolic axes there would corrupt every buffer
                    # size and block copy derived from it.

    # A keyed mapping's slot ids name values in one frame. Several passes copy
    # field accounting between frames, so verify the correlation still resolves
    # where it is stated and drop it where it does not. A mapping that names no
    # slots is simply unresolved here -- honest, and refusable by a backend --
    # whereas one naming ids this frame never defined would silently address
    # whatever else happens to hold them.
    for function in all_functions.values():
        frame_values = {int(value.id) for value in function.args}
        frame_values.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for value in function.args:
            accounting = dict(value.accounting or {})
            if accounting.get("program_abi_storage") != "keyed":
                continue
            slots = [
                accounting.get("program_abi_keyed_length"),
                accounting.get("program_abi_keyed_keys"),
                accounting.get("program_abi_keyed_values"),
            ]
            if all(
                slot is not None and int(slot) in frame_values
                for slot in slots
            ):
                continue
            for frame_local in (
                "program_abi_keyed_length",
                "program_abi_keyed_keys",
                "program_abi_keyed_values",
            ):
                accounting.pop(frame_local, None)
            value.accounting = accounting

    # A schema-token scalar is the one-element case of the compiler's token
    # chains.  Translate adjacent authored string constants through the exact
    # ordered vocabulary carried by the physical field.  This is deliberately
    # not ``string_token`` hashing: the receipt is the reversible encoder and
    # its one-based index is collision-free within that declared vocabulary.
    vocabulary_lowerings = []
    for function_symbol, function in all_functions.items():
        producers = {
            int(instruction.res.id): instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Eq", "Ne"} or len(
                    instruction.args
                ) != 2:
                    continue
                for token_value, literal_value in (
                    (instruction.args[0], instruction.args[1]),
                    (instruction.args[1], instruction.args[0]),
                ):
                    vocabulary = tuple(map(str, (
                        (token_value.accounting or {}).get(
                            "program_abi_token_vocabulary"
                        ) or ()
                    )))
                    literal = producers.get(int(literal_value.id))
                    if (
                        not vocabulary
                        or literal is None
                        or literal.op != "string_token"
                    ):
                        continue
                    text = literal.attributes.get("text")
                    if not isinstance(text, str) or text not in vocabulary:
                        continue
                    encoded = vocabulary.index(text) + 1
                    literal.op = "Const"
                    literal.attributes = {
                        "value": encoded,
                        "program_abi_vocabulary_token": text,
                        "program_abi_vocabulary": vocabulary,
                    }
                    if literal.res is not None:
                        literal.res.dtype = "int64"
                    if token_value.dtype in {None, "unknown", "float64"}:
                        token_value.dtype = "int64"
                    vocabulary_lowerings.append({
                        "function": str(function_symbol),
                        "field": (token_value.accounting or {}).get(
                            "program_abi_field"
                        ),
                        "token": text,
                        "encoded": encoded,
                    })
                    break
    if vocabulary_lowerings:
        module_metadata["program_abi_vocabulary_lowerings"] = tuple(
            vocabulary_lowerings
        )

    # Call-frame linking can be the first point at which a provisional region
    # result becomes a physical formal.  Reconcile that late surface with the
    # authored graph before final identities/DCE: a scalar cast feeding a
    # source-linked call is an internal producer, never a caller-supplied ABI
    # slot.  Earlier structural recovery cannot claim this case because the
    # placeholder does not exist in ``function.args`` until linking finishes.
    # Reuse the exact SSAValue object already held by consumers so every use
    # keeps its deterministic identity while acquiring one real definition.
    for function_symbol, function in all_functions.items():
        graph = source_graphs_by_symbol.get(str(function_symbol))
        if graph is None or not function.blocks:
            continue
        required_ids = set(map(
            int, function.metadata.get("required_source_value_ids", ())
        ))
        authored_parameter_ids = {
            int(value_id)
            for _name, value_id in function.metadata.get(
                "parameter_names", ()
            )
        }
        values = function_values(function)
        produced_ids = {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        recovered = []
        for placeholder in tuple(function.args):
            value_id = int(placeholder.id)
            accounting = dict(placeholder.accounting or {})
            result_source = accounting.get("ssa_call_result_from")
            if (
                value_id not in required_ids
                or value_id in authored_parameter_ids
                or not result_source
                or value_id in produced_ids
                or accounting.get("program_abi_storage")
                or accounting.get("compiler_frame_storage")
                or accounting.get("linked_call_frame_storage")
            ):
                continue
            data = graph.nodes.get(value_id, {})
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            if operation not in {"int", "float", "bool"}:
                continue
            operands = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).startswith("arg:")
            )
            if len(operands) != 1:
                continue
            operand = values.get(operands[0])
            # Hoisting is exact only for an authored formal: it dominates all
            # control paths and scalar conversion has no side effects.  A
            # locally produced operand needs lexical placement evidence and
            # is deliberately left to the ordinary frontier instead.
            if operand is None or operand not in function.args:
                continue
            function.args = [
                argument for argument in function.args
                if int(argument.id) != value_id
            ]
            target_dtype = {
                "int": "int64", "float": "float64", "bool": "bool",
            }[operation]
            placeholder.dtype = target_dtype
            placeholder.accounting = {
                key: value
                for key, value in accounting.items()
                if key != "ssa_call_result_from"
            }
            placeholder.accounting.update({
                "recovered_structural_source": operation,
                "source_value_id": int(operand.id),
            })
            entry = next(iter(function.blocks.values()))
            insertion_index = 0
            while (
                insertion_index < len(entry.instrs)
                and entry.instrs[insertion_index].op in {"Const", "const"}
            ):
                insertion_index += 1
            entry.instrs.insert(insertion_index, Instr(
                "Cast", [operand], placeholder,
                attributes={
                    "structural_operation": operation,
                    "target_dtype": target_dtype,
                    "late_call_feed_recovery": True,
                },
            ))
            produced_ids.add(value_id)
            recovered.append((value_id, int(operand.id), operation))
        if recovered:
            function.metadata["recovered_late_call_feeds"] = tuple(recovered)

    # Literal construction is pure. Source ingestion intentionally retains
    # strings, empty tuples, debug labels, and optional markers long enough
    # for structural planning. After call frames and public returns are fixed,
    # an unconsumed literal is dead program metadata. Remove it once here so
    # all backends receive the same instruction stream instead of inventing
    # four different nonnumeric-constant policies.
    for function in all_functions.values():
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        for block in function.blocks.values():
            block.instrs = [
                instruction
                for instruction in block.instrs
                if not (
                    instruction.op in {"Const", "const"}
                    and not instruction.args
                    and "value" in instruction.attributes
                    and instruction.res is not None
                    and int(instruction.res.id) not in consumed_ids
                )
            ]

    # -- storage-formal ABI declaration -----------------------------------
    # The caller-provides-workspace design (the LAPACK WORK-array move: the
    # callee's scratch is allocated above and threaded down, so the hot path
    # never allocates) appends storage formals at several linking sites. A
    # convention only works DECLARED: an undeclared workspace formal is a
    # parameter no caller can name, size, or fill -- the exact defect the
    # scorecard's nested-calls journey pinned. Stamp the declaration into
    # each function's metadata in one complete post-pass, so every consumer
    # -- the native shell that leases the top of the chain from the heap,
    # the Python materializer that realizes it as a local allocation, the
    # self-check that validates parity -- reads one authoritative record.
    # ``dynamic`` marks entries whose extent is not compile-time known;
    # those are the heap participants.
    for function in all_functions.values():
        # Every physical member of a resident sequence is an intentional
        # caller-provided frame slot. Local lists/bytearrays need the same
        # declared workspace ABI as storage passed through a linked call; an
        # anonymous formal is never an acceptable way to smuggle that arena
        # across the native boundary. The deterministic SSA identity remains
        # unchanged -- this adds only its physical ABI role.
        sequence_table = all_sequence_tables.get(str(function.name))
        arguments_by_id = {
            int(argument.id): argument for argument in function.args
        }
        if sequence_table is not None:
            for sequence_id, descriptor in sorted(
                sequence_table.sequences.items()
            ):
                members = (
                    *descriptor.column_value_ids,
                    descriptor.length_address_id,
                    descriptor.capacity_value_id,
                    *((descriptor.status_address_id,)
                      if descriptor.status_address_id is not None else ()),
                    *((descriptor.live_flags_value_id,)
                      if descriptor.live_flags_value_id is not None else ()),
                )
                for member_index, value_id in enumerate(members):
                    argument = arguments_by_id.get(int(value_id))
                    if argument is None:
                        continue
                    accounting = dict(argument.accounting or {})
                    if not any(accounting.get(key) not in {None, ""} for key in (
                        "program_abi_parameter",
                        "linked_call_frame_storage",
                        "returned_record_storage",
                    )):
                        argument.accounting = {
                            **accounting,
                            "compiler_frame_storage": str(function.name),
                            "compiler_frame_sequence_id": int(sequence_id),
                            "compiler_frame_member": int(member_index),
                        }
        declared_storage = []
        for argument in function.args:
            accounting = dict(argument.accounting or {})
            owner = (
                accounting.get("linked_call_frame_storage")
                or accounting.get("returned_record_storage")
                or accounting.get("compiler_frame_storage")
            )
            if not owner:
                continue
            shape = tuple(
                int(extent) for extent in (argument.shape or ())
                if isinstance(extent, int) or str(extent).isdigit()
            )
            declared_storage.append({
                "value_id": int(argument.id),
                "dtype": str(argument.dtype or "float64"),
                "shape": shape,
                "callee": str(owner),
                "callsite_id": accounting.get("callsite_id"),
                "dynamic": len(shape) != len(tuple(argument.shape or ())),
            })
        if declared_storage:
            function.metadata["storage_formals"] = tuple(declared_storage)

    # Constant-exponent Pow becomes Mul/Div here, once, so every backend
    # inherits the reduction -- six of the seven have no optimizer behind
    # them. This must run AFTER region carving, structural-output recovery
    # and value pruning: run earlier, the rewrite orphans the exponent
    # constant, the region liveness pass then prunes it, and the caller's
    # recovered output binding references a value that no longer exists.
    # Here the module is final, so a constant that lost its last arithmetic
    # consumer still materializes wherever an output ledger names it.
    from .ir_identities import (
        drop_dead_pure_region_calls,
        reduce_constant_exponent_pow,
    )

    reduce_constant_exponent_pow(all_functions)
    # Catalogue 2.2, completeness-motivated: a pure region whose projected
    # results nothing reads (the materialized comprehension ``range``) must
    # not force a backend to spell code nothing runs.
    drop_dead_pure_region_calls(all_functions)

    # Sequence descriptors are the canonical physical ABI.  A source value
    # may have been captured provisionally by a numerical region before its
    # resident sequence schema was known, producing another SSAValue object
    # with the same deterministic identity but a generic float dtype.  Reconcile
    # every occurrence by identity after linking so backend declarations and
    # call frames cannot disagree with the sequence table.
    for function_name, function in all_functions.items():
        table = all_sequence_tables.get(function_name)
        if table is None:
            continue
        canonical_dtypes = {
            int(value_id): str(dtype)
            for descriptor in table.sequences.values()
            for value_id, dtype in zip(
                descriptor.column_value_ids, descriptor.column_dtypes
            )
            if dtype not in {None, "", "unknown"}
        }
        if not canonical_dtypes:
            continue
        values = [*function.args]
        for block in function.blocks.values():
            for instruction in block.instrs:
                values.extend(instruction.args)
                if instruction.res is not None:
                    values.append(instruction.res)
        for value in values:
            dtype = canonical_dtypes.get(int(value.id))
            if dtype is not None:
                value.dtype = dtype

    # Sequence helper calls are physical ABI operations.  Their row argument
    # often inlines the producer expression, so updating only the Call's
    # SSAValue occurrence leaves the producer spelling at its provisional
    # integer width.  Correlate by deterministic value identity and apply the
    # helper formal's declared element type to every occurrence in the caller.
    for function in all_functions.values():
        occurrences: dict[int, list[SSAValue]] = {}
        for value in function.args:
            occurrences.setdefault(int(value.id), []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in instruction.args:
                    occurrences.setdefault(int(value.id), []).append(value)
                if instruction.res is not None:
                    occurrences.setdefault(
                        int(instruction.res.id), []
                    ).append(instruction.res)
        for block in function.blocks.values():
            for instruction in block.instrs:
                if not instruction.attributes.get("ssa_sequence_operation"):
                    continue
                callee = all_functions.get(str(
                    instruction.attributes.get("callee") or ""
                ))
                if callee is None:
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    dtype = str(formal.dtype or "")
                    if dtype in {"", "unknown"}:
                        continue
                    for occurrence in occurrences.get(int(actual.id), ()):
                        occurrence.dtype = dtype

    lowered_module = IRModule(
            all_functions,
            **(
                {"function_table": source_function_table}
                if source_function_table is not None else {}
            ),
            **({"class_table": class_table} if class_table is not None else {}),
            tensor_tables=all_tensor_tables,
            sequence_tables=all_sequence_tables,
            record_tables=all_record_tables,
            reference_tables=all_reference_tables,
            call_table={
                caller: tuple(records)
                for caller, records in call_records.items()
            },
            machine_control_table=(
                SSAMachineControlTable(tuple(machine_control_links))
            ),
            machine_indirect_table=(
                SSAMachineIndirectTable(tuple(machine_indirect_links))
            ),
            metadata=module_metadata,
        )
    # Precision is one vertical compiler feature: the frontend names widened
    # arithmetic, the repository SSA proves exact reductions and materialises
    # limbs, and destinations consume the resulting contract.  Running this
    # transaction at the completed-module seam keeps every backend on the same
    # lowered program and lets it repair the canonical call ABI after formals
    # grow.  Modules without precision operations are left byte-for-byte alone.
    from .ir_identities import apply_precision_pipeline
    apply_precision_pipeline(lowered_module)

    return (
        lowered_module,
        {
            name: emit_outputs(name, function)
            for name, function in all_functions.items()
        },
        tuple(export_symbols),
    )


def lower_class_surface_to_ssa(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Public target-neutral entry to the repository whole-object lowering.

    The implementation historically lived beside the Fortran shell because
    that was its first consumer.  Its product is nevertheless an ordinary
    :class:`IRModule`: class definitions, record/sequence/reference tables,
    method functions, and explicit call records.  Generic compiler and
    visualization paths use this entry rather than discarding that object
    geometry at the numerical-precompile boundary.
    """

    return _class_surface_ssa_program(
        compilation,
        artifact_name,
        tensor_ssa_reference=tensor_ssa_reference,
    )


def _emit_class_surface_module(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Emit the reusable whole-object SSA program as one Fortran module."""

    from .ssa_fortran_backend import emit_module

    ssa_module, outputs, export_symbols = _class_surface_ssa_program(
        compilation,
        artifact_name,
        tensor_ssa_reference=tensor_ssa_reference,
    )
    if ssa_module is None:
        return None, ()
    emitted = emit_module(
        ssa_module,
        name=f"{artifact_name}_fortran",
        outputs=outputs,
        # A library exports its whole surface: keep and export every method and
        # region function, not just the ones one nominal entry reaches.
        extra_roots=tuple(ssa_module.functions),
    )
    if not emitted.complete:
        raise FortranEmissionError(
            "class surface could not emit hierarchical object program: "
            + "; ".join(item.format() for item in emitted.shortfalls)
        )
    return emitted, export_symbols


def _normalize_top_level_guard_returns(
    tree: ast.Module,
    target_names: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Give selected functions one exit without inventing control semantics.

    Repository control regions are owned by branch bodies.  A top-level guard
    whose body contains only an early ``return`` therefore has no numerical
    region to own and historically disappeared before SSA control lowering.
    For compiler-bootstrap targets, rewrite only that canonical guard form to
    assignments to a private result followed by one final return.  ProcessGraph
    can then retain both arms as ordinary nested control.  The authored source
    and its hash remain the public compilation receipt; this is a deterministic
    compiler-complementary AST normalization recorded separately below.
    """

    requested = {str(name) for name in target_names if str(name)}
    if not requested:
        return ()
    receipts: list[dict[str, Any]] = []

    def selected(qualified_name: str, simple_name: str) -> bool:
        return qualified_name in requested or simple_name in requested

    def normalize_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualified_name: str,
    ) -> None:
        if not selected(qualified_name, node.name) or not node.body:
            return
        if not isinstance(node.body[-1], ast.Return):
            return
        terminal = node.body[-1]
        if terminal.value is None:
            return

        occupied = {
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }
        result_name = "__turing_single_exit_result"
        suffix = 0
        while result_name in occupied:
            suffix += 1
            result_name = f"__turing_single_exit_result_{suffix}"

        guard_lines: list[int] = []

        def result_assignment(statement: ast.Return) -> ast.Assign:
            assignment = ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())],
                value=statement.value,
            )
            return ast.copy_location(assignment, statement)

        def nest(statements: list[ast.stmt]) -> list[ast.stmt] | None:
            for index, statement in enumerate(statements[:-1]):
                if not (
                    isinstance(statement, ast.If)
                    and not statement.orelse
                    and statement.body
                    and isinstance(statement.body[-1], ast.Return)
                    and statement.body[-1].value is not None
                ):
                    continue
                tail = nest(statements[index + 1 :])
                if tail is None:
                    tail = [
                        *statements[index + 1 : -1],
                        result_assignment(statements[-1]),
                    ]
                guarded_return = statement.body[-1]
                rewritten = ast.If(
                    test=statement.test,
                    body=[
                        *statement.body[:-1],
                        result_assignment(guarded_return),
                    ],
                    orelse=tail,
                )
                ast.copy_location(rewritten, statement)
                guard_lines.append(int(getattr(statement, "lineno", 0)))
                return [*statements[:index], rewritten]
            return None

        rewritten_body = nest(list(node.body))
        if rewritten_body is None:
            return
        final_return = ast.copy_location(
            ast.Return(value=ast.Name(id=result_name, ctx=ast.Load())),
            terminal,
        )
        node.body = [*rewritten_body, final_return]
        receipts.append({
            "function": qualified_name,
            "result_name": result_name,
            "guard_count": len(guard_lines),
            "source_lines": tuple(sorted(guard_lines)),
        })

    def walk_scope(statements: Iterable[ast.stmt], prefix: str = "") -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                walk_scope(statement.body, qualified)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                normalize_function(statement, qualified)
                # Nested definitions retain their authored lexical identity.
                walk_scope(statement.body, f"{qualified}.<locals>")

    walk_scope(tree.body)
    if receipts:
        ast.fix_missing_locations(tree)
    return tuple(receipts)


def _lower_resolved_process_graph_deployment(
    graph: Any,
    entrypoint: str | None,
    *,
    dependency_seeds: tuple[str, ...] = (),
    selected_function_references: tuple[int, ...] | None = None,
    activation_function_references: tuple[int, ...] | None = None,
    name: str | None = None,
    runtime_closure_only: bool = True,
    tensor_ssa_reference: Any = None,
    linked_source_region_ssa: Mapping[
        tuple[str, ...], tuple[Any, Mapping[str, Any], Mapping[str, Any]]
    ] | None = None,
    subdivision_request: Mapping[str, Any] | None = None,
    allow_authored_source_callees: bool = False,
    progress: Callable[[str], None] | None = None,
):
    """Lower an already reduced ProcessGraph through the canonical backend path.

    ``selected_function_references`` is the bootstrap boundary: it names one
    dependency-closed compilation unit from a recorded plan.  Supplying it
    prevents the deployment planner from recursively instantiating unrelated
    FunctionTable members, while retaining their authored graph
    representations for later units and source fallback.
    """

    from types import SimpleNamespace

    from .glsl_deployment_strategy import strategize_shell_deployment
    from .shell_reference_tables import (
        build_class_navigation_table,
        build_map_dependency_regions,
    )

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    whole_source = entrypoint is None and selected_function_references is None
    deployment_graph = graph
    if selected_function_references is not None:
        selected = tuple(dict.fromkeys(map(int, selected_function_references)))
        if not selected:
            raise ValueError("a resolved ProcessGraph unit cannot be empty")
        selected_names = tuple(
            str(graph.function_table.entry(reference).qualified_name)
            for reference in selected
        )
        activation = tuple(dict.fromkeys(map(
            int,
            activation_function_references or selected,
        )))
        if not set(activation).issubset(selected):
            raise ValueError(
                "resolved ProcessGraph activation references must belong to "
                "the retained runtime unit closure"
            )
        activation_names = tuple(
            str(graph.function_table.entry(reference).qualified_name)
            for reference in activation
        )
        graph.G.graph["compile_targets"] = activation_names
        map_ir = dict(graph.G.graph.get("map_ir") or {})
        map_ir["dependency_regions"] = {
            "runtime": selected,
            "mapped": (),
            "retained": selected,
            "map_only": (),
            "bindings": (),
        }
        graph.G.graph["map_ir"] = map_ir
        if allow_authored_source_callees:
            selected_set = set(selected)
            for entry in graph.function_table:
                if int(entry.reference.address) in selected_set:
                    continue
                if entry.metadata.get("host_repository_ssa_complete"):
                    continue
                entry.metadata["implementation_kind"] = (
                    "authored-source-fallback"
                )
        # The resolved project graph is a source catalogue, not another
        # executable body of the selected function.  Give deployment an empty
        # module shell carrying the same FunctionTable and metadata; it will
        # instantiate only the selected function shells below.  This is the
        # key memory/linearity cut: the worker retains authored graphs for
        # linking and fallback without scheduling all catalogue nodes again.
        from .process_graph_fusion import extract_clean_process_subgraph

        deployment_graph = extract_clean_process_subgraph(graph, ())
        runtime_closure_only = True
        entrypoint = activation_names[0]
    elif runtime_closure_only and not whole_source:
        dependency_regions = build_map_dependency_regions(
            graph,
            str(entrypoint),
            extra_seeds=dependency_seeds,
        )
        map_ir = dict(graph.G.graph.get("map_ir") or {})
        map_ir["dependency_regions"] = {
            "runtime": dependency_regions.runtime,
            "mapped": dependency_regions.mapped,
            "retained": dependency_regions.retained,
            "map_only": dependency_regions.map_only,
            "bindings": dependency_regions.bindings,
        }
        graph.G.graph["map_ir"] = map_ir

    report("ssa-source: selecting complete control/operator deployment")
    deployment_type = strategize_shell_deployment(
        deployment_graph,
        backend="fortran",
        runtime_closure_only=(runtime_closure_only and not whole_source),
    )
    report("ssa-source: instantiating complete control/operator deployment")
    deployment = deployment_type(profiling=False, shell_language="glsl")
    deployment.prepare_complete_catalogue = whole_source
    report("ssa-source: validating resolved ProcessGraph call topology")
    deployment.compile_process_graph(prepare_ephemerals=False)
    if subdivision_request is not None:
        from .glsl_deployment_strategy import (
            _structural_region_program_from_subgraph,
            _walk_planned_shells,
        )

        subdivision_kind = str(
            subdivision_request.get("kind") or "loop-control-owner"
        )
        requested_regions = tuple(sorted(map(
            int, subdivision_request.get("region_indices") or (),
        )))
        requested_names = set(map(
            str, subdivision_request.get("qualified_names") or (),
        ))
        requested_references = tuple(map(
            int, subdivision_request.get("function_references") or (),
        ))
        if subdivision_kind == "function-shell":
            if len(requested_references) != 1:
                raise ValueError(
                    "function-shell subdivision requires exactly one "
                    "function reference"
                )
            requested_reference = int(requested_references[0])
            target = deployment.function_shells.get(requested_reference)
            if target is None:
                deployment_reference = deployment.process_graph.G.graph.get(
                    "function_ref"
                )
                if deployment_reference == requested_reference:
                    target = deployment
            if target is None:
                raise ValueError(
                    "function-shell subdivision cannot find its selected "
                    f"shell reference {requested_reference}"
                )
            target_graph = target.process_graph.G
            target_name = str(
                target_graph.graph.get("qualified_name")
                or target_graph.graph.get("function_name")
                or ""
            )
            requested_regions = tuple(range(len(target.dispatch_subgraphs)))
            report(
                "ssa-source: extracting deterministic function-shell "
                f"subdivision {target_name} regions {requested_regions}"
            )
            return {
                region_index: _structural_region_program_from_subgraph(
                    target.dispatch_subgraphs[region_index]
                )
                for region_index in requested_regions
            }

        requested_loop = int(subdivision_request["loop_node_id"])
        matches = []
        for target in _walk_planned_shells(
            deployment, include_function_registry=True,
        ):
            target_graph = target.process_graph.G
            target_name = str(
                target_graph.graph.get("qualified_name")
                or target_graph.graph.get("function_name")
                or ""
            )
            if requested_names and target_name not in requested_names and not any(
                name.endswith("." + target_name) for name in requested_names
            ):
                continue
            reduction = next((
                item for item in target.loop_shader_reductions
                if int(item.loop_node_id) == requested_loop
            ), None)
            if reduction is None:
                continue
            owned_regions = set(map(int, reduction.region_indices))
            if not set(requested_regions).issubset(owned_regions):
                raise ValueError(
                    "subdivision request names regions outside its loop "
                    f"owner: requested={requested_regions!r} "
                    f"owned={tuple(sorted(owned_regions))!r}"
                )
            matches.append((target_name, target))
        if len(matches) != 1:
            raise ValueError(
                "subdivision request must resolve exactly one planned shell; "
                f"matched={tuple(name for name, _target in matches)!r}"
            )
        target_name, target = matches[0]
        report(
            "ssa-source: extracting deterministic subdivision integral "
            f"{target_name} loop {requested_loop} regions {requested_regions}"
        )
        return {
            region_index: _structural_region_program_from_subgraph(
                target.dispatch_subgraphs[region_index]
            )
            for region_index in requested_regions
        }
    report("ssa-source: planning complete control/operator graph")
    deployment.prepare_graph_precompile(
        progress=report,
        structural_ssa_only=True,
    )
    compilation = SimpleNamespace(
        deployment=deployment,
        class_navigation=build_class_navigation_table(graph),
    )
    report("ssa-source: lowering full planned source to repository SSA")
    artifact_name = _identifier(str(name or entrypoint or "whole_source"))
    module, outputs, exports = _class_surface_ssa_program(
        compilation,
        artifact_name,
        tensor_ssa_reference=tensor_ssa_reference,
    )
    if linked_source_region_ssa:
        from .precompile_to_ssa import link_verified_source_region_integrals

        region_link_receipts = link_verified_source_region_integrals(
            module, outputs, linked_source_region_ssa,
        )
        linked_count = sum(
            receipt.get("status") == "linked"
            for receipt in region_link_receipts
        )
        if linked_count:
            report(
                f"ssa-source: linked {linked_count} verified source region(s)"
            )
    return module, outputs, exports


def extract_resolved_process_graph_subdivision_programs(
    graph: Any,
    integral: Mapping[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> Mapping[int, Any]:
    """Extract exact planner regions without lowering the blocked owner."""

    references = tuple(map(int, integral.get("function_references") or ()))
    if not references:
        raise ValueError("subdivision integral has no function reference")
    return _lower_resolved_process_graph_deployment(
        graph,
        None,
        selected_function_references=references,
        activation_function_references=references,
        name="subdivision_integral",
        runtime_closure_only=True,
        subdivision_request=integral,
        # Every subdivision is deliberately smaller than its recorded
        # dependency closure. Calls outside the cut remain authored source;
        # the child is never allowed to make them implicit native launches.
        allow_authored_source_callees=True,
        progress=progress,
    )


def lower_resolved_process_graph_unit_to_ssa(
    graph: Any,
    function_references: Iterable[int],
    *,
    linked_repository_ssa: Mapping[
        int, tuple[Any, str, Mapping[str, Any]]
    ] | None = None,
    authored_dependency_references: Iterable[int] = (),
    name: str | None = None,
    tensor_ssa_reference: Any = None,
    allow_function_shell_cut: bool = False,
    progress: Callable[[str], None] | None = None,
):
    """Compile one exact unit from a serialized post-reduction ProcessGraph."""

    from .compilation_units import record_compilation_unit_plan

    selected = tuple(sorted(set(map(int, function_references))))
    plan = record_compilation_unit_plan(graph)
    matching = tuple(
        unit for unit in plan.units
        if tuple(sorted(unit.function_references)) == selected
    )
    if len(matching) == 1:
        selected_unit = matching[0]
        selected_names = tuple(selected_unit.qualified_names)
        selected_record = selected_unit.to_mapping()
    elif allow_function_shell_cut and len(selected) == 1:
        selected_unit = None
        selected_names = (
            str(graph.function_table.entry(selected[0]).qualified_name),
        )
        selected_record = {
            "qualified_names": list(selected_names),
            "function_references": list(selected),
            "dependency_units": [],
            "recursive": False,
            "kind": "authored-function-shell-cut",
        }
    else:
        raise ValueError(
            "selected references must exactly match one recorded "
            f"compilation unit; got {selected!r}"
        )
    for reference, (linked_module, linked_root, linked_outputs) in dict(
        linked_repository_ssa or {}
    ).items():
        entry = graph.function_table.entry(int(reference))
        entry.metadata.update({
            "host_ssa_module": linked_module,
            "host_ssa_root": str(linked_root),
            "host_ssa_outputs": dict(linked_outputs),
            "host_repository_ssa_complete": True,
            "host_machine_state_complete": False,
            "host_ssa_blockers": (),
            "host_ssa_hard_blockers": (),
            "host_ssa_legalization_shortfalls": (),
            "host_ssa_unresolved_dependencies": (),
            "implementation_kind": "linked-repository-ssa",
            "implementation_variants": ("repository-ssa",),
        })
    module, outputs, exports = _lower_resolved_process_graph_deployment(
        graph,
        selected_names[0],
        selected_function_references=tuple(dict.fromkeys((
            *selected, *map(int, authored_dependency_references),
        ))),
        activation_function_references=selected,
        name=name,
        runtime_closure_only=True,
        tensor_ssa_reference=tensor_ssa_reference,
        allow_authored_source_callees=allow_function_shell_cut,
        progress=progress,
    )
    module.metadata["compilation_unit_plan"] = plan.to_mapping()
    module.metadata["compiled_process_graph_unit"] = selected_record
    return module, outputs, exports


def lower_ast_source_to_ssa(
    source: str,
    entrypoint: str | None = None,
    *,
    python_bindings: Mapping[str, Any] | None = None,
    external_class_field_aggregate_kinds: Mapping[
        tuple[str, str], str
    ] | None = None,
    dependency_seeds: tuple[str, ...] = (),
    retain: Any = (),
    tensor_code_references: Mapping[str, Callable[..., Any]] | None = None,
    tensor_ssa_reference: Any = None,
    name: str | None = None,
    runtime_closure_only: bool = True,
    progress: Callable[[str], None] | None = None,
    boundary_namespace: Any = None,
    source_language: str = "python",
    extraction_contract: Any = None,
    linked_process_graphs: Mapping[str, Any] | None = None,
    linked_repository_ssa: Mapping[
        str, tuple[Any, str] | tuple[Any, str, Mapping[str, Any]]
    ] | None = None,
    linked_source_region_ssa: Mapping[
        tuple[str, ...], tuple[Any, Mapping[str, Any], Mapping[str, Any]]
    ] | None = None,
    compilation_unit_plan_sink: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    resolved_process_graph_sink: Callable[[Any], None] | None = None,
    stop_after_compilation_unit_plan: bool = False,
):
    """Ingest one complete authored program and lower it directly to SSA.

    This is the explicit non-projecting compiler entry point.  It preserves
    source control, ordinary arithmetic, tensor operations, calls, memory and
    returns through ProcessGraph planning and repository SSA.  It never
    captures, constructs, validates, or projects a numerical ``FusedProgram``
    and it does not execute the submitted program.

    The returned tuple is ``(IRModule, outputs, exports)``.  Target emission is
    deliberately separate so callers can inspect the complete SSA program
    before choosing Fortran or another backend.
    """

    from .compiler_bootstrap_runtime import (
        activate_registered_compiler_bootstraps,
    )

    bootstrap_activations = activate_registered_compiler_bootstraps()
    if progress is not None and bootstrap_activations:
        progress(
            "bootstrap: activated "
            f"{len(bootstrap_activations)} registered compiler deployment(s)"
        )

    import contextlib
    import io
    from types import SimpleNamespace

    from ..common.tensors.accelerator_backends.aot_compile import (
        _source_dependency_is_not_tensor_primitive,
    )
    from ..common.tensors.topological_reducer import (
        reduce_abstract_tensor_topology,
    )
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .glsl_deployment_strategy import (
        _lower_python_scalar_intrinsics,
        _resolve_grounded_method_references,
        strategize_shell_deployment,
    )
    from .loop_interchange import interchange_reduction_loops
    from .shell_reference_tables import (
        build_class_navigation_table,
        build_map_dependency_regions,
    )
    from .work_contract import active_contract

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    authored_source_sha256 = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    work_contract = active_contract()
    interchange = interchange_reduction_loops(
        source, licensed=bool(work_contract.inexact_identities),
    )
    source = interchange.source
    transformed_source_sha256 = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    if interchange.decisions:
        report(
            "ssa-source: loop interchange considered "
            f"{len(interchange.decisions)} reduction nest(s)"
        )
    tree = ast.parse(source)
    # No selected root is the canonical whole-source mode.  It deliberately
    # disables runtime-closure pruning so module statements, every authored
    # definition, and their configured dependency domains remain eligible.
    compile_targets = (
        () if entrypoint is None else (str(entrypoint), *map(str, dependency_seeds))
    )
    whole_source = entrypoint is None
    single_exit_receipts = _normalize_top_level_guard_returns(
        tree, compile_targets,
    )
    if single_exit_receipts:
        report(
            "ssa-source: normalized return-only guards in "
            f"{len(single_exit_receipts)} selected function(s)"
        )
    inferred_record_views = _authored_dataclass_record_views(tree)
    inferred_record_schemas = _authored_complete_record_schemas(tree)
    inferred_sequence_record_views = _authored_sequence_record_views(
        tree, inferred_record_schemas,
    )
    linked_repository_ssa = {
        str(function_name): (
            value[0], str(value[1]), dict(value[2]) if len(value) > 2 else {}
        )
        for function_name, value in dict(linked_repository_ssa or {}).items()
    }
    if linked_repository_ssa:
        linked_definitions = []

        def register_linkable_definition(node, qualified_name):
            linked_definitions.append((qualified_name, node))
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    register_linkable_definition(
                        member, f"{qualified_name}.<locals>.{member.name}",
                    )

        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                register_linkable_definition(statement, statement.name)
            elif isinstance(statement, ast.ClassDef):
                for member in statement.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        register_linkable_definition(
                            member, f"{statement.name}.{member.name}",
                        )
        for qualified_name, node in linked_definitions:
            linked = linked_repository_ssa.get(qualified_name)
            if linked is None:
                continue
            linked_module, linked_root, linked_outputs = linked
            node._linked_repository_ssa_module = linked_module
            node._linked_repository_ssa_root = linked_root
            node._linked_repository_ssa_outputs = linked_outputs
    extraction_policy = extraction_contract
    if extraction_policy is None:
        # The work contract may embed the whole extraction policy; a
        # per-call argument still wins. None from both preserves the
        # historical (gate-disabled) behavior.
        extraction_policy = work_contract.extraction
    if extraction_policy is not None:
        from .extraction_contract import ExtractionContract
        if isinstance(extraction_policy, (str, os.PathLike)):
            extraction_policy = ExtractionContract(extraction_policy)
        elif not hasattr(extraction_policy, "decide"):
            raise TypeError(
                "extraction_contract must be a path or ExtractionContract"
            )
    graph = ProcessGraph(
        materialize_memory=False,
        boundary_namespace=boundary_namespace,
        source_language=source_language,
    )
    linked_process_graphs = {
        str(function_name): function_graph
        for function_name, function_graph in dict(
            linked_process_graphs or {}
        ).items()
    }
    if linked_process_graphs:
        from .process_graph_function_linking import link_process_graph_functions

        report("ssa-source: registering authored ProcessGraph functions")
        link_process_graph_functions(graph, linked_process_graphs)
    graph.python_bindings = dict(python_bindings or {})
    graph.G.graph["external_class_field_aggregate_kinds"] = dict(
        external_class_field_aggregate_kinds or {}
    )
    report("ssa-source: building complete ProcessGraph source closure")
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            tree,
            resolve_unresolved_parents=True,
            parent_include=(
                extraction_policy
                if extraction_policy is not None
                else _source_dependency_is_not_tensor_primitive
            ),
            pursuit_roots=(
                tuple(dict.fromkeys(compile_targets))
                if runtime_closure_only and not whole_source else None
            ),
            tensor_code_references=dict(tensor_code_references or {}),
            retain=retain,
            progress=report,
        )
    if (
        (extraction_policy is not None and (
            extraction_policy.program_abi.records
            or extraction_policy.program_abi.values
        ))
        or inferred_record_views
        or inferred_record_schemas
        or inferred_sequence_record_views
    ):
        # Type the physical Python/native boundary before topology reduction.
        # This is declarative ABI information only: it does not instantiate a
        # Python object, infer a convenient shape, or authorize new source
        # pursuit. Every pursued function receives only the record bindings
        # whose function/parameter rules explicitly match the contract.
        for entry in graph.function_table:
            function_graph = getattr(getattr(entry, "graph", None), "G", None)
            if function_graph is None:
                continue
            function_name = str(
                function_graph.graph.get("function_name") or entry.name
            )
            method_owner = function_graph.graph.get("method_owner")
            qualified_function_name = (
                f"{method_owner}.{function_name}"
                if method_owner else function_name
            )
            records = (
                {}
                if extraction_policy is None else
                extraction_policy.program_abi.records_for_function(
                    function_name,
                    method_owner=function_graph.graph.get("method_owner"),
                    parameters=(
                        function_graph.graph.get("function_parameters") or ()
                    ),
                )
            )
            parameters = set(map(
                str, function_graph.graph.get("function_parameters") or ()
            ))
            selected = {
                parameter: dict(record)
                for parameter, record in inferred_record_views.get(
                    qualified_function_name, {}
                ).items()
                if parameter in parameters
            }
            selected.update({
                parameter: record.receipt()
                for parameter, record in records.items()
                if parameter in parameters
            })
            if selected:
                function_graph.graph["parameter_record_abi"] = selected
            selected_sequence_records = copy.deepcopy(dict(
                inferred_sequence_record_views.get(
                    qualified_function_name, {}
                )
            ))
            if selected_sequence_records:
                function_graph.graph["parameter_sequence_record_abi"] = (
                    selected_sequence_records
                )
            values = (
                {}
                if extraction_policy is None else
                extraction_policy.program_abi.values_for_function(
                    function_name
                )
            )
            selected_values = {
                parameter: binding.receipt()
                for parameter, binding in values.items()
                if parameter in parameters
            }
            if selected_values:
                function_graph.graph["parameter_value_abi"] = selected_values
        program_abi_receipt = (
            {"records": {}, "bindings": (), "values": ()}
            if extraction_policy is None else
            extraction_policy.program_abi.receipt()
        )
        # Explicit contracts remain authoritative. Source-derived schemas fill
        # only classes whose complete dataclass layout is present in this
        # authored compilation unit, making their constructors structural SSA
        # without adding project-specific entries to the extraction contract.
        program_abi_receipt["records"] = {
            **inferred_record_schemas,
            **dict(program_abi_receipt.get("records") or {}),
        }
        graph.G.graph["program_abi"] = program_abi_receipt
    graph.G.graph["compile_targets"] = tuple(dict.fromkeys(compile_targets))
    for intrinsic_graph in (
        graph,
        *(getattr(entry, "graph", None) for entry in graph.function_table),
    ):
        if getattr(intrinsic_graph, "G", None) is not None:
            _lower_python_scalar_intrinsics(intrinsic_graph)
    report("ssa-source: reducing source topology")
    reduce_abstract_tensor_topology(graph)
    for intrinsic_graph in (
        graph,
        *(getattr(entry, "graph", None) for entry in graph.function_table),
    ):
        if getattr(intrinsic_graph, "G", None) is not None:
            _lower_python_scalar_intrinsics(intrinsic_graph)
    if (
        (extraction_policy is not None and (
            extraction_policy.program_abi.records
            or extraction_policy.program_abi.values
        ))
        or inferred_record_views
        or inferred_record_schemas
        or inferred_sequence_record_views
    ):
        # Reduction extracts fresh per-function graphs from the complete
        # source graph. Reattach the declarative ABI to those canonical
        # graphs before structural specialization and hierarchy planning;
        # attaching it only to the pre-reduction discovery graphs leaves
        # method receivers untyped during exactly the pass that decides
        # schema guards and optional-field branches.
        for entry in graph.function_table:
            function_graph = getattr(getattr(entry, "graph", None), "G", None)
            if function_graph is None:
                continue
            function_name = str(
                function_graph.graph.get("function_name") or entry.name
            )
            method_owner = function_graph.graph.get("method_owner")
            qualified_function_name = (
                f"{method_owner}.{function_name}"
                if method_owner else function_name
            )
            parameters = set(map(
                str, function_graph.graph.get("function_parameters") or ()
            ))
            records = (
                {}
                if extraction_policy is None else
                extraction_policy.program_abi.records_for_function(
                    function_name,
                    method_owner=function_graph.graph.get("method_owner"),
                    parameters=parameters,
                )
            )
            selected = {
                parameter: dict(record)
                for parameter, record in inferred_record_views.get(
                    qualified_function_name, {}
                ).items()
                if parameter in parameters
            }
            selected.update({
                parameter: record.receipt()
                for parameter, record in records.items()
                if parameter in parameters
            })
            if selected:
                function_graph.graph["parameter_record_abi"] = selected
            selected_sequence_records = copy.deepcopy(dict(
                inferred_sequence_record_views.get(
                    qualified_function_name, {}
                )
            ))
            if selected_sequence_records:
                function_graph.graph["parameter_sequence_record_abi"] = (
                    selected_sequence_records
                )
            values = (
                {}
                if extraction_policy is None else
                extraction_policy.program_abi.values_for_function(
                    function_name
                )
            )
            selected_values = {
                parameter: binding.receipt()
                for parameter, binding in values.items()
                if parameter in parameters
            }
            if selected_values:
                function_graph.graph["parameter_value_abi"] = selected_values
    if linked_process_graphs:
        # Reduction may rewrite call nodes and dependency graphs. Reapply the
        # idempotent function-table link before planning so the direct SSA path
        # never substitutes Python capture or a FusedProgram for the authored
        # cross-language function.
        from .process_graph_function_linking import link_process_graph_functions

        report("ssa-source: resolving authored ProcessGraph calls")
        link_process_graph_functions(graph, linked_process_graphs)
    # Hierarchy construction consumes the canonical post-reduction function
    # graphs, not necessarily the graph instance later visited by recursive
    # shell specialization. Resolve ABI-declared method receivers here, after
    # their record contracts were reattached and before the compilation-unit
    # and call plans snapshot the graph. The operation is idempotent and still
    # requires an exact/unique receiver class.
    for function_entry in graph.function_table:
        function_graph = getattr(function_entry, "graph", None)
        if getattr(function_graph, "G", None) is not None:
            _resolve_grounded_method_references(function_graph)
    from .compilation_units import record_compilation_unit_plan

    report("ssa-source: dividing resolved project into compilation units")
    compilation_unit_plan = record_compilation_unit_plan(graph)
    if compilation_unit_plan_sink is not None:
        compilation_unit_plan_sink(compilation_unit_plan.to_mapping())
    if resolved_process_graph_sink is not None:
        resolved_process_graph_sink(graph)
    if stop_after_compilation_unit_plan:
        # Planning is a first-class bootstrap product. The caller requested
        # the exact post-reduction cut and must not pay for, or accidentally
        # claim, deployment/SSA work beyond that boundary.
        return None, {}, ()
    artifact_name = _identifier(str(name or entrypoint or "whole_source"))
    module, outputs, exports = _lower_resolved_process_graph_deployment(
        graph,
        entrypoint,
        dependency_seeds=dependency_seeds,
        name=artifact_name,
        runtime_closure_only=runtime_closure_only,
        tensor_ssa_reference=tensor_ssa_reference,
        linked_source_region_ssa=linked_source_region_ssa,
        progress=progress,
    )
    decision_records = tuple({
        "identity": str(decision.identity),
        "function": decision.function,
        "line": int(decision.line),
        "interchanged": bool(decision.interchanged),
        "reasons": tuple(map(str, decision.reasons)),
    } for decision in interchange.decisions)
    receipt = {
        "schema": "turing.loop-interchange.v2",
        "contract": str(work_contract.name),
        "licensed": bool(work_contract.inexact_identities),
        "authored_source_sha256": authored_source_sha256,
        "transformed_source_sha256": transformed_source_sha256,
        "changed": authored_source_sha256 != transformed_source_sha256,
        "decisions": decision_records,
    }
    module.metadata["loop_interchange"] = receipt
    module.metadata["single_exit_guard_normalization"] = (
        single_exit_receipts
    )
    module.metadata["compilation_unit_plan"] = compilation_unit_plan.to_mapping()
    if extraction_policy is not None:
        extraction_boundaries = tuple(
            dict(item)
            for item in graph.G.graph.get("extraction_boundary_calls", ())
        )
        materialized_identities: dict[str, int] = {}
        for function in module.functions.values():
            for record in function.metadata.get(
                "extraction_materializations", ()
            ):
                identity = record.get("extraction_identity")
                if identity is not None:
                    materialized_identities[str(identity)] = (
                        materialized_identities.get(str(identity), 0) + 1
                    )
            for block in function.blocks.values():
                for instruction in block.instrs:
                    identity = instruction.attributes.get(
                        "extraction_identity"
                    )
                    if identity is not None:
                        materialized_identities[str(identity)] = (
                            materialized_identities.get(str(identity), 0) + 1
                        )
        remaining = dict(materialized_identities)
        unmaterialized_boundaries = []
        boundary_transformations = []
        shell_contexts = [
            dict(item)
            for item in graph.G.graph.get("shell_file_contexts", ())
        ]
        used_shell_contexts: set[int] = set()
        for boundary in extraction_boundaries:
            contract = dict(boundary.get("extraction_contract") or {})
            identity = str(contract.get("identity") or "")
            if identity and remaining.get(identity, 0) > 0:
                remaining[identity] -= 1
                continue
            replacement = next((
                (index, context)
                for index, context in enumerate(shell_contexts)
                if index not in used_shell_contexts
                and str(context.get("identity") or "") == identity
            ), None)
            if replacement is None:
                unmaterialized_boundaries.append(boundary)
                continue
            context_index, context = replacement
            used_shell_contexts.add(context_index)
            boundary_transformations.append({
                "source_identity": identity,
                "source_rule": contract.get("rule_id"),
                "transformation": "python-file-context-to-shell-file-region",
                "scope": context.get("scope"),
                "shell_operation_identities": tuple(
                    map(str, context.get("operation_identities", ()))
                ),
            })
        unresolved_call_records = tuple(
            {
                "caller": str(record.caller),
                "callsite_id": int(record.callsite_id),
                "callee": str(record.callee_symbol or record.callee_name),
            }
            for records in module.call_table.values()
            for record in records
            if record.resolution == "unresolved"
        )
        module.metadata["extraction_boundary_accounting"] = {
            "occurrences": extraction_boundaries,
            "materialized_identity_counts": materialized_identities,
            "boundary_transformations": tuple(boundary_transformations),
            "unmaterialized": tuple(unmaterialized_boundaries),
            "unresolved_call_records": unresolved_call_records,
            "repository_ssa_complete": not (
                unmaterialized_boundaries or unresolved_call_records
            ),
        }
        module.metadata["extraction_contract"] = {
            "fingerprint": str(getattr(extraction_policy, "fingerprint", "")),
            "path": str(getattr(extraction_policy, "path", "")),
            "decisions": list(extraction_policy.receipts()),
        }
        shell_requests = {}
        native_reference_plans = {}
        for boundary in extraction_boundaries:
            contract = dict(boundary.get("extraction_contract") or {})
            parameters = dict(contract.get("parameters") or {})
            capability = parameters.get("shell_capability")
            if capability is None:
                continue
            attributes = {
                key: parameters[key]
                for key in ("execution", "shell_abi")
                if parameters.get(key) is not None
            }
            previous = shell_requests.setdefault(str(capability), attributes)
            if previous != attributes:
                raise ValueError(
                    f"conflicting {capability!r} shell boundary declarations"
                )
            if str(contract.get("action") or "") == "use_native":
                identity = str(contract.get("identity") or "")
                if not identity:
                    raise ValueError(
                        "native extraction boundary lacks an exact identity"
                    )
                plan = {
                    "identity": identity,
                    "module": contract.get("module"),
                    "qualname": contract.get("qualname"),
                    "classification": contract.get("classification"),
                    "loader": parameters.get("loader"),
                    "symbol_resolution": parameters.get(
                        "symbol_resolution"
                    ),
                    "callbacks": parameters.get("callbacks"),
                    "execution": parameters.get("execution"),
                    "shell_capability": parameters.get(
                        "shell_capability"
                    ),
                    "shell_abi": parameters.get("shell_abi"),
                    "external_domain": parameters.get("external_domain"),
                }
                previous_plan = native_reference_plans.setdefault(
                    identity, plan
                )
                if previous_plan != plan:
                    raise ValueError(
                        "conflicting native boundary ABI declarations for "
                        f"{identity!r}"
                    )
        if shell_requests:
            from .shell_io import (
                ShellIOManifest,
                ShellIORequest,
                attach_shell_io_metadata,
            )

            module.metadata = attach_shell_io_metadata(
                module.metadata,
                ShellIOManifest(requests=tuple(
                    ShellIORequest.create(capability, attributes=attributes)
                    for capability, attributes in sorted(shell_requests.items())
                )),
            )
            if shell_contexts:
                shell_io_metadata = dict(module.metadata.get("shell_io") or {})
                shell_io_metadata["boundary_plan_schema"] = (
                    "turing.shell-boundary-plan.v1"
                )
                shell_io_metadata["boundary_plans"] = tuple(shell_contexts)
                module.metadata["shell_io"] = shell_io_metadata
            if native_reference_plans:
                shell_io_metadata = dict(module.metadata.get("shell_io") or {})
                shell_io_metadata["external_reference_plan_schema"] = (
                    "turing.shell-external-reference-plan.v1"
                )
                shell_io_metadata["external_reference_plans"] = tuple(
                    native_reference_plans[identity]
                    for identity in sorted(native_reference_plans)
                )
                module.metadata["shell_io"] = shell_io_metadata
    for decision in decision_records:
        function = module.functions.get(
            f"{artifact_name}__{_identifier(decision['function'])}"
        )
        if function is not None:
            function.metadata.setdefault(
                "loop_interchange_decisions", ()
            )
            function.metadata["loop_interchange_decisions"] = (
                *function.metadata["loop_interchange_decisions"], decision,
            )
    return module, outputs, exports


lower_ast_source_to_ssa.__canonical_source_compiler__ = True


def compile_ast_fortran_c_shell(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    directory: str | Path,
    *,
    python_bindings: Mapping[str, Any] | None = None,
    output_names: tuple[str, ...] | list[str] | None = None,
    state_feedback: Mapping[str, str] | None = None,
    display: Mapping[str, Any] | None = None,
    name: str | None = None,
    standalone: bool = True,
    progress: Callable[[str], None] | None = None,
    checkpoint: bool | str | Path = False,
    mutable_parameters: tuple[str, ...] | list[str] | set[str] = (),
    retain_card_program: bool = True,
    compilation: Any | None = None,
    library: bool = False,
    dependency_seeds: tuple[str, ...] = (),
    retain: Any = (),
    tensor_code_references: Mapping[str, Callable[..., Any]] | None = None,
    tensor_ssa_reference: Any = None,
    runtime_closure_only: bool = False,
    trace: bool = False,
) -> FortranCShellExecutable:
    """Compile Python AST through the registered Fortran target and C shell.

    ``library=True`` builds a shared library (.dll/.so) of the compiled section
    -- the section exported for other programs to link against -- instead of a
    standalone C-shell executable. See ``compile_fortran_module_c_shell``.

    This is the application-neutral native entrypoint.  It accepts authored
    Python, runs the ordinary ProcessGraph/AOT compiler, projects that
    compiler's public numerical program, and only then selects Fortran.
    Dotted aggregate feed names such as ``state.u`` are resolved from the
    caller's object without flattening or copying its arena in Python source.

    ``compilation`` lets a caller that already ran the whole-program no-bake
    ``compile_ast_aot`` (e.g. to first release the backend-neutral dual-IR
    checkpoint) hand that exact ``AOTCompilation`` in, so the Fortran shell
    runs the already-produced dual IR instead of compiling the program a
    second time.
    """

    from .compiler_entrypoints import warn_legacy_source_compiler

    warn_legacy_source_compiler("compile_ast_fortran_c_shell")

    from ..common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
        project_public_numerical_program,
    )
    from .machine_targets import get_target
    from .shell_io import (
        ShellIOBinding,
        ShellIOCapability,
        ShellIOManifest,
        ShellIORequest,
        attach_shell_io,
    )

    compilation = compilation if compilation is not None else compile_ast_aot(
        source,
        entrypoint,
        dict(feeds),
        trace=trace,
        backend="c",
        precompile_only=True,
        bake_mode="whole_program",
        python_bindings=dict(python_bindings or {}),
        progress=progress,
        checkpoint=checkpoint,
        mutable_parameters=tuple(mutable_parameters),
        dependency_seeds=tuple(dependency_seeds),
        retain=retain,
        tensor_code_references=tensor_code_references,
        require_planned_shells=library,
        runtime_closure_only=runtime_closure_only,
        # A whole-object library lowers every method's complete local
        # ControlProgram/operator graph directly.  Captured-region hierarchy
        # projection is a numerical optimization with an independent marker
        # catalogue; allowing it to run here can discard structural call
        # placement before the direct SSA lowerer sees the program.
        project_captured_hierarchy=not library,
    )
    hierarchical_outputs = dict(compilation.public_output_value_ids)
    hierarchical_inputs = dict(compilation.public_input_value_ids)
    # ``public_output_value_ids`` contains only hierarchy terminals that the
    # numerical capture marked device-resident.  A source return can instead
    # be a structural/control value (a callee result or an object-field load)
    # and is still a real compiled-program output.  Recover every declared
    # return from the authoritative identity history so the complete
    # ControlProgram/region SSA path emits it; never force such a return
    # through ``project_public_numerical_program`` merely because capture did
    # not classify it as a numerical terminal.
    for output_name in getattr(compilation, "function_outputs", ()):
        history = tuple(
            getattr(compilation, "identity_table", {}).get(output_name, ())
        )
        if history:
            hierarchical_outputs.setdefault(
                str(output_name), int(history[-1])
            )
    # A later field read after a source-ordered write consumes the stored
    # value even when the ProcessGraph has only the receiver edge on GetAttr.
    # Recover that memory dependency explicitly for public returns.  This is
    # the same field-slot ordering rule used by whole-object lowering, applied
    # to an ordinary function receiving a record parameter.
    source_process_graph = getattr(
        getattr(compilation, "deployment", None), "process_graph", None
    )
    source_function_table = getattr(source_process_graph, "function_table", None)
    if source_function_table is not None:
        try:
            source_entry = source_function_table.entry(entrypoint)
        except KeyError:
            source_entry = None
        function_graph = getattr(getattr(source_entry, "graph", None), "G", None)
        if function_graph is not None:
            for output_name, output_id in tuple(hierarchical_outputs.items()):
                if int(output_id) not in function_graph:
                    continue
                output_data = function_graph.nodes[int(output_id)]
                if output_data.get("type") != "GetAttr":
                    continue
                attribute = (output_data.get("attributes") or {}).get("attribute")
                receiver = next((
                    int(parent)
                    for parent, role in (output_data.get("parents") or ())
                    if str(role) in {"value", "object"}
                ), None)
                stored_value = None
                for node_id in sorted(
                    function_graph.nodes, key=lambda value: int(value)
                ):
                    if int(node_id) >= int(output_id):
                        break
                    data = function_graph.nodes[node_id]
                    if (
                        data.get("type") not in {"SetAttr", "setattr"}
                        or (data.get("attributes") or {}).get("attribute")
                        != attribute
                    ):
                        continue
                    parents = tuple(data.get("parents") or ())
                    target = next((
                        int(parent) for parent, role in parents
                        if str(role) in {"object", "value"}
                    ), None)
                    value = next((
                        int(parent) for parent, role in parents
                        if str(role) == "value"
                    ), None)
                    if target == receiver and value is not None:
                        stored_value = value
                if stored_value is not None:
                    hierarchical_outputs[output_name] = int(stored_value)
    if output_names is not None and hierarchical_outputs:
        names = tuple(map(str, output_names))
        if set(hierarchical_outputs) <= set(names):
            pass
        elif len(names) != len(hierarchical_outputs):
            raise ValueError(
                f"received {len(names)} output names for "
                f"{len(hierarchical_outputs)} hierarchical outputs"
            )
        else:
            hierarchical_outputs = {
                output_name: value_id
                for output_name, value_id in zip(
                    names, hierarchical_outputs.values()
                )
            }

    # ``entrypoint`` is a Python-qualified source name and may contain dots
    # (for example ``ProcessGraph.build_from_ast``).  The artifact name is
    # also the prefix for every emitted Fortran procedure and C symbol, where
    # dots and the other Python punctuation are illegal.  Sanitize once at
    # this boundary so module names, intra-module calls, API symbols, and file
    # names all use the same spelling.
    artifact_name = _identifier(str(name or entrypoint))
    module = None

    # Whole-object library build: emit every planned method as its own export
    # via the non-numeric control-sections path. A class has no program-level
    # return surface, so it never reaches the numeric emission below -- and it
    # must not, because that path projects and validates a numerical program the
    # object does not have. This early return skips all of the single-entry
    # native-input/card machinery, which does not apply to a multi-method
    # library.
    if library:
        class_module, export_symbols = _emit_class_surface_module(
            compilation,
            artifact_name,
            tensor_ssa_reference=tensor_ssa_reference,
        )
        if class_module is not None:
            if progress is not None:
                progress(
                    f"emitted object surface {artifact_name}: "
                    f"exports {list(export_symbols)}"
                )
            return compile_fortran_module_c_shell(
                class_module,
                {},
                directory,
                entrypoint=export_symbols[0],
                name=artifact_name,
                standalone=standalone,
                library=True,
            )
        raise FortranEmissionError(
            "whole-object library compilation produced no planned method "
            "surface; refusing to substitute a numerical projection"
        )

    if hierarchical_outputs and compilation.region_programs:
        from .hierarchical_plan import PlanCall, PlanClosure
        from .precompile_to_ssa import lower_control_sections_to_ssa
        from .ssa_fortran_backend import emit_module

        runtime_value_meta: dict[int, tuple[tuple[int, ...], str]] = {}
        for source_name, value_id in hierarchical_inputs.items():
            root, *attributes = str(source_name).split(".")
            if root not in feeds:
                continue
            runtime_value = feeds[root]
            try:
                for attribute in attributes:
                    runtime_value = getattr(runtime_value, attribute)
                array = np.asarray(runtime_value)
            except (AttributeError, TypeError, ValueError):
                continue
            runtime_value_meta[int(value_id)] = (
                tuple(map(int, array.shape)), str(array.dtype)
            )

        def apply_runtime_value_meta(closure: Any) -> Any:
            if not isinstance(closure, PlanClosure):
                return closure
            shape_records = {
                int(value_id): (tuple(shape), str(dtype))
                for value_id, shape, dtype in closure.value_shapes
            }
            for value_id in (
                *closure.captures,
                *(
                    value_id
                    for item in closure.items
                    if hasattr(item, "inputs")
                    for value_id in (*item.inputs, *item.outputs)
                ),
            ):
                if int(value_id) in runtime_value_meta:
                    shape_records[int(value_id)] = runtime_value_meta[int(value_id)]
            shape_preserving = {
                "Add", "Sub", "Mul", "Div", "Pow", "Mod",
                "add", "sub", "mul", "div", "pow", "mod",
            }
            changed = True
            while changed:
                changed = False
                for item in closure.items:
                    if (
                        getattr(item, "opcode", None) not in shape_preserving
                        or not getattr(item, "outputs", ())
                    ):
                        continue
                    shaped_inputs = [
                        shape_records[int(value_id)]
                        for value_id in getattr(item, "inputs", ())
                        if int(value_id) in shape_records
                        and shape_records[int(value_id)][0]
                    ]
                    if not shaped_inputs:
                        continue
                    propagated = max(
                        shaped_inputs,
                        key=lambda record: (len(record[0]), record[0]),
                    )
                    for value_id in item.outputs:
                        previous = shape_records.get(int(value_id))
                        if previous != propagated:
                            shape_records[int(value_id)] = propagated
                            changed = True
            rebuilt_items = tuple(
                apply_runtime_value_meta(item)
                if isinstance(item, PlanClosure)
                else replace(item, callee=apply_runtime_value_meta(item.callee))
                if isinstance(item, PlanCall)
                else item
                for item in closure.items
            )
            return replace(
                closure,
                items=rebuilt_items,
                value_shapes=tuple(
                    (value_id, shape, dtype)
                    for value_id, (shape, dtype) in shape_records.items()
                ),
            )

        hierarchy_plan = apply_runtime_value_meta(
            getattr(compilation, "hierarchy_plan", None)
        )

        identity_table = {
            **dict(compilation.identity_table),
            **{
                source_name: (int(value_id),)
                for source_name, value_id in hierarchical_inputs.items()
            },
            **{
                source_name: (int(value_id),)
                for source_name, value_id in hierarchical_outputs.items()
            },
        }
        lowered_module, lowering_shortfalls, lowered_outputs = (
            lower_control_sections_to_ssa(
                compilation.shell_control_program,
                hierarchy_plan=hierarchy_plan,
                control_name=artifact_name,
                identity_table=identity_table,
                function_outputs=tuple(hierarchical_outputs),
                function_parameters=tuple(hierarchical_inputs),
                tensor_ssa_reference=tensor_ssa_reference,
            )
        )
        if lowering_shortfalls:
            raise FortranEmissionError(
                "complete hierarchical AST program has SSA shortfalls: "
                + "; ".join(
                    f"{item.name} ({item.reason})"
                    for item in lowering_shortfalls
                )
            )
        module = emit_module(
            lowered_module,
            name=f"{artifact_name}_fortran",
            outputs=lowered_outputs,
            extra_roots=tuple(lowered_module.functions) if library else (),
        )
        if not module.complete:
            raise FortranEmissionError(
                "Fortran target could not emit hierarchical AST program: "
                + "; ".join(item.format() for item in module.shortfalls)
            )

    program = project_public_numerical_program(compilation)
    if module is None and output_names is not None:
        names = tuple(map(str, output_names))
        if len(names) != len(program.outputs):
            metadata = program.meta or {}
            output_summary = tuple(
                (
                    output_name,
                    tuple(getattr(metadata.get(value_id), "shape", ()) or ()),
                )
                for output_name, value_id in program.outputs.items()
            )
            declared = {
                output_name: tuple(
                    compilation.identity_table.get(output_name, ())
                )
                for output_name in compilation.function_outputs
            }
            available = {
                *map(int, program.feeds),
                *(int(step.result_id) for step in program.steps),
                *map(int, program.outputs.values()),
            }
            raise ValueError(
                f"received {len(names)} output names for "
                f"{len(program.outputs)} compiled outputs; "
                f"declared={declared!r}; "
                f"declared_available={{{', '.join(f'{key!r}: {tuple(value in available for value in values)!r}' for key, values in declared.items())}}}; "
                f"first={output_summary[:16]!r}; last={output_summary[-16:]!r}"
            )
        program = replace(
            program,
            outputs={
                output_name: value_id
                for output_name, value_id in zip(
                    names, program.outputs.values()
                )
            },
        )
    if module is None:
        emitted = get_target("fortran").emit(program, name=artifact_name)
        if not emitted.complete or emitted.module is None:
            raise FortranEmissionError(
                "Fortran target could not emit compiled AST program: "
                + "; ".join(emitted.shortfalls)
            )
        module = emitted.module

    # Hierarchical lowering can promote a region-private feed into the public
    # control ABI after ``parameter_names`` was recorded.  Its SSA identity is
    # still present in the graph identity table, so restore the authored feed
    # name here.  Stateful shells can then declare feedback by program name
    # instead of depending on an unstable ``t<ID>`` spelling.
    feed_names_by_value_id: dict[int, str] = {}
    ambiguous_feed_ids: set[int] = set()
    candidate_feed_names = {
        str(name)
        for name in compilation.identity_table
        if str(name).split(".", 1)[0] in feeds
    }
    candidate_feed_names.update(map(str, feeds))
    for feed_name in candidate_feed_names:
        for value_id in compilation.identity_table.get(feed_name, ()):
            value_id = int(value_id)
            previous = feed_names_by_value_id.get(value_id)
            if previous is not None and previous != str(feed_name):
                ambiguous_feed_ids.add(value_id)
            else:
                feed_names_by_value_id[value_id] = str(feed_name)
    if feed_names_by_value_id:
        entry_points = []
        for described_entry in module.api.entry_points:
            described_parameters = []
            for parameter in described_entry.parameters:
                source_name = parameter.source_name
                if source_name is None and parameter.name.startswith("t"):
                    try:
                        value_id = int(parameter.name[1:])
                    except ValueError:
                        value_id = -1
                    if value_id not in ambiguous_feed_ids:
                        source_name = feed_names_by_value_id.get(value_id)
                described_parameters.append(
                    replace(parameter, source_name=source_name)
                )
            entry_points.append(replace(
                described_entry,
                parameters=tuple(described_parameters),
            ))
        module = replace(
            module,
            api=replace(module.api, entry_points=tuple(entry_points)),
        )
    if display is not None:
        options = dict(display)
        channels = tuple(
            map(str, options.pop("channels", ("red", "green", "blue")))
        )
        if channels != ("red", "green", "blue"):
            raise ValueError(
                "native rgb_f64_planar display requires red, green, blue"
            )
        manifest = ShellIOManifest(
            requests=(ShellIORequest.create(
                ShellIOCapability.DISPLAY,
                attributes={
                    "pixel_format": "rgb_f64_planar",
                    **options,
                },
            ),),
            bindings=tuple(
                ShellIOBinding(
                    f"display.{channel}", artifact_name, channel
                )
                for channel in channels
            ),
        )
        module = replace(module, api=attach_shell_io(module.api, manifest))

    def resolve_source_name(source_name: str) -> Any:
        if source_name in feeds:
            return feeds[source_name]
        root, *attributes = source_name.split(".")
        if root not in feeds:
            raise KeyError(source_name)
        value = feeds[root]
        for attribute in attributes:
            value = getattr(value, attribute)
        return value

    native_inputs: dict[str, Any] = {}
    public_input_names_by_id = {
        int(value_id): str(source_name)
        for source_name, value_id in (
            compilation.public_input_value_ids or {}
        ).items()
    }

    def resolve_compiled_value(value_id: int) -> Any:
        visited = set()
        current = int(value_id)
        while current not in visited:
            visited.add(current)
            if current in compilation.region_feed_values:
                return compilation.region_feed_values[current]
            source_name = public_input_names_by_id.get(current)
            if source_name is not None:
                return resolve_source_name(source_name)
            alias = compilation.hierarchical_value_aliases.get(current)
            if alias is None:
                break
            current = int(alias)
        raise KeyError(value_id)

    # A library build has no run harness and no initial state, so it needs no
    # concrete input values -- the section's parameters stay symbolic library
    # arguments. Skip resolving native inputs entirely.
    entry = module.api.entry_point(artifact_name)
    if not library:
        for parameter in entry.parameters:
            if parameter.role != "input":
                continue
            source_name = str(parameter.source_name or parameter.name)
            try:
                native_inputs[source_name] = resolve_source_name(source_name)
            except (AttributeError, KeyError) as error:
                if parameter.name.startswith("t"):
                    try:
                        value_id = int(parameter.name[1:])
                    except ValueError:
                        value_id = -1
                    try:
                        native_inputs[source_name] = resolve_compiled_value(
                            value_id
                        )
                    except (AttributeError, KeyError):
                        pass
                    else:
                        continue
                raise ValueError(
                    f"compiled input {source_name!r} ({parameter.shape!r}) "
                    "has no value in feeds or the captured region cache; "
                    f"endpoint={compilation.hierarchical_value_diagnostics.get(value_id)!r}"
                ) from error
    resolved_feedback = dict(state_feedback or {})
    abi_source_names = {
        str(parameter.source_name or parameter.name)
        for parameter in entry.parameters
        if parameter.role != "extent"
    }

    def canonical_hierarchy_value(value_id: int) -> int:
        visited = set()
        current = int(value_id)
        while current not in visited:
            visited.add(current)
            alias = compilation.hierarchical_value_aliases.get(current)
            if alias is None:
                return current
            current = int(alias)
        return current

    for input_name, output_name in tuple(resolved_feedback.items()):
        if output_name in abi_source_names or input_name not in abi_source_names:
            continue
        input_id = compilation.public_input_value_ids.get(input_name)
        output_id = compilation.public_output_value_ids.get(output_name)
        if output_id is None:
            history = tuple(compilation.identity_table.get(output_name, ()))
            output_id = history[-1] if history else None
        if (
            input_id is not None
            and output_id is not None
            and canonical_hierarchy_value(int(input_id))
            == canonical_hierarchy_value(int(output_id))
        ):
            # The declared function output is the same preallocated arena as
            # its input.  Fortran correctly publishes one inout ABI parameter
            # rather than allocating a copy-only output.  Point feedback at
            # that shared slot so the C shell preserves the alias contract.
            resolved_feedback[input_name] = input_name
    if retain_card_program:
        from .parametric_card_program import build_parametric_card_program

        card_public_inputs = dict(hierarchical_inputs)
        if not card_public_inputs:
            for parameter in entry.parameters:
                if parameter.role != "input" or not parameter.name.startswith("t"):
                    continue
                try:
                    value_id = int(parameter.name[1:])
                except ValueError:
                    continue
                card_public_inputs[
                    str(parameter.source_name or parameter.name)
                ] = value_id
        card_public_outputs = dict(hierarchical_outputs)
        if not card_public_outputs:
            card_public_outputs = {
                str(output_name): int(value_id)
                for output_name, value_id in program.outputs.items()
            }
        card_program = build_parametric_card_program(
            compilation,
            feedback=resolved_feedback,
            public_inputs=card_public_inputs,
            public_outputs=card_public_outputs,
        )
        module = replace(
            module,
            api=replace(
                module.api,
                metadata={
                    **dict(module.api.metadata or {}),
                    "card_program": card_program.to_mapping(),
                },
            ),
        )
    return compile_fortran_module_c_shell(
        module,
        native_inputs,
        directory,
        entrypoint=artifact_name,
        state_feedback=resolved_feedback,
        name=artifact_name,
        standalone=standalone,
        library=library,
        trace=trace,
    )


__all__ = [
    "FortranCShellExecutable",
    "compile_fortran_module_c_shell",
    "compile_ast_fortran_c_shell",
    "emit_fortran_c_shell_source",
    "lower_ast_source_to_ssa",
]
