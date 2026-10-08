"""Authored record ABI: annotation field receipts, record schemas and row layouts, sequence record views, dataclass record views (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
import copy
from typing import (
    Any,
    Iterable,
    Mapping,
)

from .authored_parameter_abi import _current_authored_parameter_annotations


def _authored_annotation_field_receipt(
    annotation: ast.AST,
    python_bindings: Mapping[str, Any] | None = None,
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
            receipt = _authored_annotation_field_receipt(
                concrete[0], python_bindings,
            )
            if receipt is not None:
                return {**dict(receipt), "optional": True}

    # A frontend ``Tensor`` annotation states an abstract tensor value.  Its
    # native storage is the repository's span ABI; neither the frontend class
    # nor an opaque object handle survives AST ingestion.  The annotation does
    # not state logical extents, so keep the physical flat-span rank here and
    # let ordinary tensor shape SSA carry the logical rank and extents.
    from ..transmogrifier.graph.node_special_cases import (
        tensor_annotation_identity,
    )

    tensor_identity = tensor_annotation_identity(annotation, python_bindings)
    if tensor_identity is not None:
        return {
            "storage": "span",
            "dtype": "unknown",
            "rank": 1,
            "mutable": False,
            "python_type": str(tensor_identity),
        }

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
        receipt = _authored_annotation_field_receipt(
            annotation.slice, python_bindings,
        )
        return (
            None if receipt is None
            else {**dict(receipt), "optional": True}
        )
    if container == "Callable":
        # A callable is an authored native reference, not an opaque Python
        # object and not a scalar guessed from its eventual use.  Keep the
        # exact signature on the source-derived record receipt so later call
        # planning can correlate the field with the function-table identity
        # stored by the constructor.  ``reference`` is the repository's
        # existing ABI storage class for an externally materialized address.
        return {
            "storage": "reference",
            "dtype": "int64",
            "rank": 0,
            "mutable": False,
            "callable_signature": ast.unparse(annotation),
        }
    if container in {"dict", "Dict", "Mapping", "MutableMapping"}:
        columns = (
            tuple(annotation.slice.elts)
            if isinstance(annotation.slice, ast.Tuple)
            else (annotation.slice,)
        )
        if len(columns) != 2:
            return None
        key_annotation, value_annotation = columns
        key_name = (
            key_annotation.id
            if isinstance(key_annotation, ast.Name)
            else key_annotation.attr
            if isinstance(key_annotation, ast.Attribute)
            else ""
        )
        key_encoding = {
            "str": "string_token",
            "int": "integer_identity",
        }.get(str(key_name))
        if key_encoding is None:
            return None
        value_optional = False
        if isinstance(value_annotation, ast.BinOp) and isinstance(
            value_annotation.op, ast.BitOr
        ):
            concrete = tuple(
                member
                for member in (value_annotation.left, value_annotation.right)
                if not (
                    isinstance(member, ast.Constant)
                    and member.value is None
                )
                and not (
                    isinstance(member, ast.Name) and member.id == "None"
                )
            )
            if len(concrete) != 1:
                return None
            value_annotation = concrete[0]
            value_optional = True
        value_name = (
            value_annotation.id
            if isinstance(value_annotation, ast.Name)
            else value_annotation.attr
            if isinstance(value_annotation, ast.Attribute)
            else ""
        )
        from ..transmogrifier.graph.node_special_cases import (
            tensor_annotation_identity,
        )

        value_is_tensor = tensor_annotation_identity(
            value_annotation, python_bindings,
        ) is not None
        value_dtype = {
            "bool": "bool", "int": "int64", "float": "float64",
            "str": "int64",
        }.get(str(value_name))
        value_record = None
        if value_is_tensor:
            value_dtype = "unknown"
        elif value_dtype is None and value_name:
            value_dtype = "int64"
            value_record = str(value_name)
        if value_dtype is None:
            return None
        return {
            "storage": "keyed",
            "dtype": value_dtype,
            "rank": 1,
            "mutable": container in {"dict", "Dict", "MutableMapping"},
            "key_encoding": key_encoding,
            **({
                "value_tensor": True,
                "value_python_type": "AbstractTensor",
            } if value_is_tensor else {}),
            **({"value_record": value_record} if value_record else {}),
            **({"optional": True} if value_optional else {}),
        }
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
    """Derive exact ``Sequence[Record]`` binding row contracts.

    The sequence is caller-owned columnar storage; its loop target is a row
    correlation, never a Python object crossing the native ABI. Only complete
    declared record schemas are admitted, so every projected field has a
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
        def register(binding_name: str, annotation: ast.AST | None, *, mutable: bool) -> None:
            identity = element_record(annotation)
            if identity is None:
                return
            schema = dict(schemas[identity])
            views.setdefault(qualified_name, {})[str(binding_name)] = {
                "identity": str(schema.get("identity") or identity),
                "fields": copy.deepcopy(dict(schema.get("fields") or {})),
                "aggregate_kind": "tuple",
                "mutable": bool(mutable),
                "source_derived": True,
            }

        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        ):
            register(argument.arg, argument.annotation, mutable=False)

        class LocalAnnotations(ast.NodeVisitor):
            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if isinstance(node.target, ast.Name):
                    register(node.target.id, node.annotation, mutable=True)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

        visitor = LocalAnnotations()
        for statement in definition.body:
            visitor.visit(statement)

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(statement.name, statement)
        elif isinstance(statement, ast.ClassDef):
            # ``self.rows: list[Row] = []`` is the receiver-field form of the
            # same authored contract.  Publish it on each method that actually
            # reads that field; the method graph's GetAttr identity then owns
            # the columnar row view.  Restricting this to explicit annotated
            # assignments and complete record schemas avoids inferring an
            # element type from append values or runtime contents.
            receiver_sequence_fields: dict[str, str] = {}
            initializer = next((
                member for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == "__init__"
            ), None)
            if initializer is not None:
                for node in ast.walk(initializer):
                    if not (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                    ):
                        continue
                    identity = element_record(node.annotation)
                    if identity is not None:
                        receiver_sequence_fields[
                            str(node.target.attr)
                        ] = identity
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = f"{statement.name}.{member.name}"
                    visit(qualified_name, member)
                    used_fields = {
                        str(node.attr)
                        for node in ast.walk(member)
                        if (
                            isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "self"
                        )
                    }
                    for field_name in sorted(
                        used_fields & set(receiver_sequence_fields)
                    ):
                        identity = receiver_sequence_fields[field_name]
                        schema = dict(schemas[identity])
                        views.setdefault(qualified_name, {})[
                            field_name
                        ] = {
                            "identity": str(
                                schema.get("identity") or identity
                            ),
                            "fields": copy.deepcopy(dict(
                                schema.get("fields") or {}
                            )),
                            "aggregate_kind": "tuple",
                            "mutable": True,
                            "source_derived": True,
                        }
    return views


def _unwrap_optional_annotation(expression: ast.AST) -> ast.AST:
    """``X | None`` and ``Optional[X]`` declare the same row as ``X``."""

    while True:
        if (
            isinstance(expression, ast.BinOp)
            and isinstance(expression.op, ast.BitOr)
        ):
            sides = (expression.left, expression.right)
            kept = tuple(
                side for side in sides
                if not (isinstance(side, ast.Constant) and side.value is None)
            )
            if len(kept) == 1:
                expression = kept[0]
                continue
            return expression
        if (
            isinstance(expression, ast.Subscript)
            and (
                (isinstance(expression.value, ast.Name)
                 and expression.value.id == "Optional")
                or (isinstance(expression.value, ast.Attribute)
                    and expression.value.attr == "Optional")
            )
        ):
            expression = expression.slice
            continue
        return expression


def _authored_row_layout(
    annotation: Any,
    repository_records: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str | None, int], ...] | None:
    """Read a fixed tuple row's physical layout from a container annotation.

    ``list[tuple[float, Metrics, tuple[str, ...]]]`` declares one row of
    three authored columns.  A column naming a repository record is not one
    physical column: the record has no physical identity of its own, so its
    row is its member fields, exactly as ``_record_row_physical_columns``
    lays out a whole-record row.  Each entry is ``(kind, identity, width)``
    with ``kind`` one of ``"scalar"``, ``"record"``, ``"value"``.
    """

    if not isinstance(annotation, str) or not annotation.strip():
        return None
    try:
        outer = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return None
    outer = _unwrap_optional_annotation(outer)
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
    if len(columns) < 2:
        return None
    scalar_names = {"bool", "int", "float", "str"}
    layout: list[tuple[str, str | None, int]] = []
    for column in columns:
        spelling = (
            column.id if isinstance(column, ast.Name)
            else column.attr if isinstance(column, ast.Attribute)
            else ""
        )
        if spelling in scalar_names:
            layout.append(("scalar", None, 1))
            continue
        matches = tuple(
            (str(identity), record)
            for identity, record in repository_records.items()
            if str(identity) == spelling
            or str(identity).rsplit(".", 1)[-1] == spelling
        )
        if spelling and len(matches) == 1:
            identity, record = matches[0]
            width = len(_record_row_physical_columns(record))
            if width:
                layout.append(("record", identity, width))
                continue
        layout.append(("value", None, 1))
    return tuple(layout)


def _sequence_row_record_slots(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> dict[int, tuple[tuple[int, str, int], ...]]:
    """Record-typed columns of each declared fixed-width row.

    Maps a resident sequence id to ``(authored column, identity, width)``
    triples.  A row insertion names the record by its semantic identity at
    that column; native-call linking, which proves the record's physical
    member fields, expands it in place (``ssa_deferred_record_slots``).
    """

    declared = {int(item[0]) for item in sequence_declarations}
    identities = graph_obj.graph.get("identity_table") or {}
    repository_records = dict(
        (graph_obj.graph.get("program_abi") or {}).get("records") or {}
    )
    result: dict[int, tuple[tuple[int, str, int], ...]] = {}
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        layout = _authored_row_layout(annotation, repository_records)
        if layout is None or not any(kind == "record" for kind, _i, _w in layout):
            continue
        slots = tuple(
            (position, str(identity), int(width))
            for position, (kind, identity, width) in enumerate(layout)
            if kind == "record"
        )
        # Every declared resident in the binding's history is the same
        # authored row: the parameter, its ``[]`` replacement and the merge
        # a mutation resolves to.
        for value_id in identities.get(str(parameter_name), ()):
            if int(value_id) in declared:
                result[int(value_id)] = slots
    return result


def _record_row_physical_columns(
    record: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return the exact flat ABI columns for one declared record row."""

    columns: list[tuple[str, str]] = []
    for field_name, receipt_value in dict(record.get("fields") or {}).items():
        receipt = dict(receipt_value)
        storage = str(receipt.get("storage") or "scalar")
        dtype = str(receipt.get("dtype") or "unknown")
        if storage == "keyed":
            columns.extend((
                (f"{field_name}.length", "int64"),
                (f"{field_name}.keys", "int64"),
                (f"{field_name}.values", dtype),
            ))
            continue
        if bool(receipt.get("optional")):
            columns.append((f"{field_name}.__present", "bool"))
        fixed_length = receipt.get("fixed_length")
        if storage == "span" and fixed_length is not None:
            columns.extend(
                (str(field_name), dtype)
                for _ in range(int(fixed_length))
            )
        else:
            columns.append((str(field_name), dtype))
    return tuple(columns)


def _record_row_sequence_columns(
    record: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Logical SoA columns for one record row, retaining shaped fields."""

    columns: list[tuple[str, str, tuple[int, ...]]] = []
    for field_name, receipt_value in dict(record.get("fields") or {}).items():
        receipt = dict(receipt_value)
        dtype = str(receipt.get("dtype") or "unknown")
        if bool(receipt.get("optional")):
            columns.append((f"{field_name}.__present", "bool", ()))
        storage = str(receipt.get("storage") or "scalar")
        if storage == "span":
            shape = receipt.get("shape")
            if shape is None and receipt.get("fixed_length") is not None:
                shape = (int(receipt["fixed_length"]),)
            columns.append((
                str(field_name), dtype,
                () if shape is None else tuple(map(int, shape)),
            ))
        else:
            columns.append((str(field_name), dtype, ()))
    return tuple(columns)


def _sequence_record_storage_columns(
    record: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Columns of a record sequence in its declared representation."""

    if bool(record.get("program_abi_table_row")):
        return _record_row_sequence_columns(record)
    return tuple(
        (name, dtype, ())
        for name, dtype in _record_row_physical_columns(record)
    )


def _graph_sequence_record_abi(graph_obj: Any) -> dict[object, Mapping[str, Any]]:
    """Resolve local/parameter sequence annotations against repository ABI."""

    from .identity_concordance import publish_program_abi_graph_identities

    # Sequence ABI discovery runs after ProgramABI schemas have been
    # enriched in place.  Republish from that authoritative boundary even
    # when the graph and schema container identities themselves are stable.
    publish_program_abi_graph_identities(graph_obj, force=True)

    resolved = copy.deepcopy(dict(
        graph_obj.graph.get("sequence_record_abi")
        or graph_obj.graph.get("parameter_sequence_record_abi")
        or {}
    ))
    repository_records = dict(
        (graph_obj.graph.get("program_abi") or {}).get("records") or {}
    )
    identities = graph_obj.graph.get("identity_table") or {}

    def schema_named(identity: object) -> tuple[str, Mapping[str, Any]] | None:
        matches = tuple(
            (str(name), record)
            for name, record in repository_records.items()
            if str(name) == str(identity)
            or str(record.get("identity") or "") == str(identity)
        )
        return matches[0] if len(matches) == 1 else None

    for node_id, data in graph_obj.nodes(data=True):
        row_identity = (data.get("attributes") or {}).get(
            "program_abi_sequence_row_identity"
        )
        target = None if row_identity is None else schema_named(row_identity)
        if target is not None:
            resolved[int(data.get("value_id", node_id))] = {
                **copy.deepcopy(dict(target[1])),
                "identity": str(target[1].get("identity") or target[0]),
                "aggregate_kind": "tuple",
                "mutable": False,
                "program_abi_table_row": True,
            }
    for binding_name, annotation in dict(
        graph_obj.graph.get("type_annotations") or {}
    ).items():
        if str(binding_name) in resolved or not isinstance(annotation, str):
            continue
        try:
            outer = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(outer, ast.Subscript):
            continue
        container = (
            outer.value.id if isinstance(outer.value, ast.Name)
            else outer.value.attr if isinstance(outer.value, ast.Attribute)
            else ""
        )
        if container not in {"list", "tuple", "Sequence", "Iterable"}:
            continue
        element = outer.slice
        element_name = (
            element.id if isinstance(element, ast.Name)
            else element.attr if isinstance(element, ast.Attribute)
            else ""
        )
        matches = tuple(
            (str(identity), record)
            for identity, record in repository_records.items()
            if str(identity) == str(element_name)
            or str(identity).rsplit(".", 1)[-1] == str(element_name)
        )
        if len(matches) != 1:
            continue
        identity, record = matches[0]
        resolved[str(binding_name)] = {
            **copy.deepcopy(dict(record)),
            "identity": identity,
            "aggregate_kind": "tuple",
            "mutable": str(binding_name) not in set(map(
                str, graph_obj.graph.get("function_parameters") or ()
            )),
            "source_derived": True,
        }
    return resolved


def _sequence_record_binding_value_ids(
    graph_obj: Any, binding: object,
) -> tuple[int, ...]:
    """Values owned by one authored-name or exact graph-value row receipt."""

    if isinstance(binding, int):
        return (int(binding),)
    identities = graph_obj.graph.get("identity_table") or {}
    return tuple(dict.fromkeys((
        *map(int, identities.get(str(binding), ())),
        *(
            int(data.get("value_id", node_id))
            for node_id, data in graph_obj.nodes(data=True)
            if str(data.get("op") or data.get("type") or "").casefold()
            == "getattr"
            and str((data.get("attributes") or {}).get("attribute"))
            == str(binding)
        ),
    )))


def _row_column_identity(
    declared_records: Any, parameter_name: str, leaf_path: str,
) -> str:
    """The storage identity of a keyed field's pooled row column:
    ``<declared owner record identity>.<field>[].<leaf>``."""

    owner = dict(dict(declared_records or {}).get(str(parameter_name)) or {})
    return f"{owner.get('identity') or parameter_name}.{leaf_path}"


def _record_field_incoming_slot(
    symbol: str, parameter: str, field_name: str, cells: Any,
) -> int | None:
    """The incoming slot of a written-but-never-read mutable scalar field
    of a declared record parameter, by its ``record_field_incoming_slot``
    row (minted once, NOVEL(NESTED_RECORD_PART, the write's field-state
    cells)).  ``None`` when no cell names the write."""

    from .concordance_declarations import (
        NESTED_RECORD_PART, RECORD_FIELD_INCOMING_SLOT, SHELL_HANDOFF,
        SSA_VALUE,
    )
    from .identity_concordance import (
        Derived, Mode, Ref, current_identity_book,
    )
    from .precompile_to_ssa import _mint_ssa_id

    book = current_identity_book()
    row = (str(symbol), str(parameter), str(field_name))
    existing = book.page(RECORD_FIELD_INCOMING_SLOT).latest(row)
    if existing is not None:
        return int(existing)
    operands = tuple(cell for cell in (cells or ()) if isinstance(cell, Ref))
    if not operands:
        return None
    scope = (str(symbol), "record_field_incoming_slot")
    value_id = _mint_ssa_id(
        book, scope, NESTED_RECORD_PART, operands, stage=SHELL_HANDOFF,
    )
    book.post(
        RECORD_FIELD_INCOMING_SLOT, row, int(value_id), stage=SHELL_HANDOFF,
        provenance=Derived((book.latest_ref(SSA_VALUE, (scope, value_id)),)),
        mode=Mode.CONCORD,
    )
    return int(value_id)


def _annotated_parameter_record_identity(
    function_identity: str, parameter: str,
) -> str | None:
    """The contract record an annotated parameter's class is, by book rows.

    ``Rules._set_momentum(item: Body)``: the ``parameter_annotation`` row
    names ``Body`` in its module; Python resolves that name to the module's
    ``class_declaration`` row ``(module, "Body")``; the class's
    ``source_span`` cell is the one the contract's
    ``source_record_class_concordance`` row (``rhworld.Body``) derives
    from.  The joined identity is posted as a ``parameter_record_class``
    row DERIVED from those three cells and returned; ``None`` when any
    link is absent.  A class with a span but no ``class_declaration`` row
    gets one, DERIVED from that span, as ``_class_schema_from_ast`` posts
    them.  No identity is compared by spelling.
    """

    from .concordance_declarations import (
        CLASS_DECLARATION, ClassFact, INGESTION, PARAMETER_ANNOTATION,
        PARAMETER_RECORD_CLASS, SOURCE_RECORD_CLASS, SOURCE_SPAN,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from ..transmogrifier.graph.graph_express2 import source_span_row

    book = current_identity_book()
    annotations = book.pages.get(PARAMETER_ANNOTATION.name)
    if annotations is None:
        return None
    rows = [
        row for row in annotations.rows()
        if row[1] == str(function_identity) and row[2] == str(parameter)
    ]
    if len(rows) != 1:
        return None
    annotation_row = rows[0]
    module = annotation_row[0]
    spelling = str(annotations.latest(annotation_row) or "")
    if not spelling.isidentifier():
        return None
    record_classes = book.pages.get(SOURCE_RECORD_CLASS.name)
    class_row = (module, spelling)
    class_cell = book.latest_ref(CLASS_DECLARATION, class_row)
    span_cells: list = []
    if class_cell is not None:
        span_cells = [
            source for source, _stage in book.edges_into(class_cell)
            if source.page.name == SOURCE_SPAN.name
        ]
    elif record_classes is not None:
        # The class was ingested as a contract record but never described
        # by the map IR: post its declaration at the class definition.
        for row in record_classes.rows():
            definition = record_classes.latest(row)
            span_row = (
                None if not isinstance(definition, ast.ClassDef)
                else source_span_row(None, definition)
            )
            if span_row is None or tuple(span_row[:2]) != class_row:
                continue
            span_cell = book.latest_ref(SOURCE_SPAN, span_row)
            if span_cell is None:
                continue
            class_cell = book.post(
                CLASS_DECLARATION, class_row,
                ClassFact(str(definition.name), ()),
                stage=INGESTION, provenance=Derived((span_cell,)),
                mode=Mode.CONCORD,
            )
            span_cells = [span_cell]
            break
    if class_cell is None or len(span_cells) != 1:
        return None
    # The span ROW is the construct (``post_source_span``: a later stage's
    # revision is a new cell of the same row), so the class declaration and
    # the contract's record class may derive from different cells of it.
    span_row = span_cells[0].row
    record_cells = list(dict.fromkeys(
        target
        for column, _fact in book.page(SOURCE_SPAN).history(span_row)
        for target, _stage in book.edges_out_of(
            type(span_cells[0])(span_cells[0].page, span_row, column)
        )
        if target.page.name == SOURCE_RECORD_CLASS.name
    ))
    if len(record_cells) != 1:
        return None
    identity = str(record_cells[0].row[0])
    row = (module, str(function_identity), str(parameter))
    if book.page(PARAMETER_RECORD_CLASS).latest(row) != identity:
        book.post(
            PARAMETER_RECORD_CLASS, row, identity, stage=INGESTION,
            provenance=Derived((
                book.latest_ref(PARAMETER_ANNOTATION, annotation_row),
                class_cell, record_cells[0],
            )),
            mode=Mode.CONCORD,
        )
    return identity


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
