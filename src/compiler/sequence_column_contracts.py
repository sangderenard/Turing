"""Sequence column contracts: declared column dtypes, key-column integrality, sequence column dtype and record identity contracts (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
from typing import (
    Any,
    Iterable,
    Mapping,
)

from .authored_record_abi import (
    _authored_row_layout,
    _graph_sequence_record_abi,
    _record_row_physical_columns,
    _sequence_record_binding_value_ids,
    _sequence_record_storage_columns,
)
from .authored_parameter_abi import (
    _authored_sequence_annotation_contract,
    _current_authored_parameter_annotations,
)
from .resident_sequence_materialization import (
    _sequence_append_call_mutations,
    _source_mapping_mutations,
    _source_sequence_snapshots,
)


_UNTYPED = {"", "unknown", "none", "None"}


_INTEGRAL_DTYPES = {
    "int", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool",
}


def _key_columns_integral(descriptor: Any) -> bool:
    """Whether every column this descriptor calls a key is integral.

    A key column is an index; a float cannot be one, and a string key is
    an fnv1a-64 token that float64's 53-bit mantissa silently rounds.  A
    descriptor failing this is invalid on its own terms, independently of
    which pass built it -- which is what lets a consumer refuse it without
    having to know the right answer.
    """
    dtypes = tuple(getattr(descriptor, "column_dtypes", ()) or ())
    for column in getattr(descriptor, "key_columns", ()) or ():
        if int(column) >= len(dtypes):
            return False
        if str(dtypes[int(column)]) not in _INTEGRAL_DTYPES:
            return False
    return True


def _declared_column_dtype(value: Any, default: str) -> str:
    """The dtype a sequence column actually holds, declaration first.

    Two failures this replaces, both seen on the real dt-system build:

    ``str(value.dtype or default)`` only falls back when dtype is None.
    The compiler's "I could not work it out" sentinel is the STRING
    ``"unknown"``, which is truthy, so it sailed through and a column
    declared int64 was published as ``unknown``.

    Worse, a key column that had been inferred from its neighbours came
    out ``float64``.  A string-token key is an fnv1a-**64** value and
    float64 carries only 53 bits exactly, so every token above 2**53 is
    silently rounded -- the key then never matches its own lookup and the
    channel goes missing rather than failing loudly.  Inference must not
    be able to decide this: ``physical_dtype`` is the declared contract
    the ABI already recorded beside the value, so it is consulted first
    and the categorical default (int64 for a key column) stands behind it.
    """
    accounting = dict(getattr(value, "accounting", None) or {})
    declared = accounting.get("physical_dtype")
    if declared is not None and str(declared) not in _UNTYPED:
        return str(declared)
    inferred = getattr(value, "dtype", None)
    if inferred is not None and str(inferred) not in _UNTYPED:
        return str(inferred)
    return str(default)


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
    repository_records = dict(
        (graph_obj.graph.get("program_abi") or {}).get("records") or {}
    )
    sequence_record_by_value: dict[int, Mapping[str, Any]] = {}
    for binding_name, sequence_record in _graph_sequence_record_abi(
        graph_obj
    ).items():
        for value_id in _sequence_record_binding_value_ids(
            graph_obj, binding_name,
        ):
            sequence_record_by_value[int(value_id)] = sequence_record
    for parameter_name, record in dict(
        graph_obj.graph.get("parameter_record_abi") or {}
    ).items():
        parameter_ids = set(map(
            int, identities.get(str(parameter_name), ())
        ))
        if not parameter_ids:
            continue
        for field_name, field in dict(record.get("fields") or {}).items():
            if str(field.get("storage") or "") != "table":
                continue
            dtypes = tuple(
                str(column.get("dtype") or "unknown")
                for column in field.get("columns") or ()
            )
            for node_id, data in graph_obj.nodes(data=True):
                attributes = data.get("attributes") or {}
                value_id = int(data.get("value_id", node_id))
                if (
                    value_id in declared
                    and str(data.get("type") or data.get("op") or "").casefold()
                    == "getattr"
                    and str(attributes.get("attribute") or "")
                    == str(field_name)
                    and any(
                        int(parent) in parameter_ids
                        and str(role) in {
                            "value", "object", "base", "receiver",
                        }
                        for parent, role in data.get("parents") or ()
                    )
                    and len(dtypes) == declared[value_id]
                ):
                    contracts[value_id] = dtypes
    for node_id, data in graph_obj.nodes(data=True):
        value_id = int(data.get("value_id", node_id))
        if value_id not in declared or str(
            data.get("type") or data.get("op") or ""
        ).casefold() != "getattr":
            continue
        receiver_id = next((
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in {"value", "object", "base", "receiver"}
            and int(parent) in graph_obj
        ), None)
        if receiver_id is None:
            continue
        receiver = graph_obj.nodes[receiver_id]
        if str(receiver.get("type") or receiver.get("op") or "").casefold() not in {
            "indexed", "load",
        }:
            continue
        row_source = next((
            int(parent)
            for parent, role in receiver.get("parents") or ()
            if str(role) == "base" and int(parent) in graph_obj
        ), None)
        record = (
            None if row_source is None
            else sequence_record_by_value.get(row_source)
        )
        attribute = str((data.get("attributes") or {}).get("attribute") or "")
        field = (
            None if record is None
            else dict(record.get("fields") or {}).get(attribute)
        )
        if not isinstance(field, Mapping) or str(
            field.get("storage") or ""
        ) != "table":
            continue
        dtypes = tuple(
            str(column.get("dtype") or "unknown")
            for column in field.get("columns") or ()
        )
        if len(dtypes) == declared[value_id]:
            contracts[value_id] = dtypes
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        sequence_id = next((
            int(value_id)
            for value_id in identities.get(str(parameter_name), ())
            if int(value_id) in declared
        ), None)
        if sequence_id is None:
            continue
        layout = _authored_row_layout(annotation, repository_records)
        if layout is not None and any(
            kind == "record" for kind, _identity, _width in layout
        ):
            row_dtypes: list[str] = []
            for kind, identity, _width in layout:
                if kind == "record":
                    record = next(
                        record for candidate, record
                        in repository_records.items()
                        if str(candidate) == identity
                    )
                    row_dtypes.extend(
                        dtype for _name, dtype
                        in _record_row_physical_columns(record)
                    )
                else:
                    row_dtypes.append("unknown")
            if len(row_dtypes) == declared[int(sequence_id)]:
                contracts[int(sequence_id)] = tuple(row_dtypes)
            continue
        contract = _authored_sequence_annotation_contract(annotation)
        if contract is None:
            continue
        _policy, _column_count, _writable, dtypes = contract
        if len(dtypes) == declared[int(sequence_id)]:
            contracts[int(sequence_id)] = tuple(map(str, dtypes))
    for parameter_name, record in _graph_sequence_record_abi(graph_obj).items():
        row_columns = _sequence_record_storage_columns(record)
        dtypes = tuple(dtype for _name, dtype, _shape in row_columns)
        if not dtypes:
            continue
        sequence_id = next((
            int(value_id)
            for value_id in _sequence_record_binding_value_ids(
                graph_obj, parameter_name,
            )
            if int(value_id) in declared
        ), None)
        if (
            sequence_id is not None
            and len(dtypes) == declared[int(sequence_id)]
        ):
            contracts[int(sequence_id)] = dtypes
            from .identity_concordance import (
                commit_sequence_row_layout,
                shape_scope_of,
            )

            commit_sequence_row_layout(
                shape_scope_of(graph_obj),
                int(sequence_id),
                tuple(shape for _name, _dtype, shape in row_columns),
                dtypes,
                source="ProgramABI sequence row record",
            )
    for binding_name, annotation in dict(
        graph_obj.graph.get("type_annotations") or {}
    ).items():
        if not isinstance(annotation, str) or not annotation.strip():
            continue
        try:
            outer = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            continue
        history = tuple(map(int, identities.get(str(binding_name), ())))
        sequence_id = next(
            (value_id for value_id in history if value_id in declared), None
        )
        direct_contract = _authored_sequence_annotation_contract(annotation)
        if direct_contract is not None and sequence_id is not None:
            _policy, _column_count, _writable, dtypes = direct_contract
            if len(dtypes) == declared[sequence_id]:
                contracts[int(sequence_id)] = tuple(map(str, dtypes))
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
        attrs = data.get("attributes") or {}
        explicit_dtypes = tuple(map(
            str, attrs.get("sequence_column_dtypes", ())
        ))
        if (
            explicit_dtypes
            and len(explicit_dtypes) == declared[value_id]
            and value_id not in contracts
        ):
            contracts[value_id] = explicit_dtypes
        if attrs.get("mapping_key_dtype") and attrs.get("mapping_value_dtype") and declared[value_id] == 2:
            contracts[value_id] = (str(attrs["mapping_key_dtype"]), str(attrs["mapping_value_dtype"]))
        if items is None and isinstance(data.get("expr_obj"), ast.Dict):
            try:
                items = tuple(ast.literal_eval(data["expr_obj"]).items())
            except (ValueError, TypeError):
                pass
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
    mapping_mutations = _source_mapping_mutations(graph_obj)
    for mutation in mapping_mutations:
        destination = int(mutation.sequence_value_id)
        if mutation.operator != "update" or destination in contracts:
            continue
        dtypes = tuple(literal_dtype(
            (graph_obj.nodes.get(int(value), {}).get("attributes") or {}).get("value")
        ) for value in mutation.argument_value_ids)
        if len(dtypes) == 2 and None not in dtypes:
            contracts[destination] = dtypes
    # A snapshot shares its source's row schema, but owns different storage.
    # Infer homogeneous scalar append rows together, never from only the
    # first write (which could choose a Boolean schema before a float row).
    append_dtypes: dict[int, set[str | None]] = {}
    for mutation in _sequence_append_call_mutations(graph_obj):
        destination = int(mutation.sequence_value_id)
        if mutation.operator != "append" or declared.get(destination) != 1:
            continue
        if len(mutation.argument_value_ids) != 1:
            continue
        value = graph_obj.nodes.get(int(mutation.argument_value_ids[0]), {})
        expression = value.get("expr_obj")
        literal = expression.value if isinstance(expression, ast.Constant) else (value.get("attributes") or {}).get("value")
        append_dtypes.setdefault(destination, set()).add(literal_dtype(literal))
    for destination, dtypes in append_dtypes.items():
        if destination not in contracts and len(dtypes) == 1 and None not in dtypes:
            contracts[destination] = (next(iter(dtypes)),)
    copies = [mutation for mutation in (*mapping_mutations, *_source_sequence_snapshots(graph_obj))
              if mutation.operator == "replace"]
    declaration_specs = {
        int(sequence_id): (str(policy), int(column_count))
        for sequence_id, policy, column_count, _writable
        in sequence_declarations
    }
    materialized_copies: list[tuple[int, int]] = []
    for node_id, data in graph_obj.nodes(data=True):
        destination = int(data.get("value_id", node_id))
        attributes = data.get("attributes") or {}
        sources = tuple(map(
            int, attributes.get(
                "materialized_source_value_ids", ()
            )
        ))
        if (
            destination in declaration_specs
            and len(sources) == 1
            and sources[0] in declaration_specs
            and declaration_specs[destination] == declaration_specs[sources[0]]
        ):
            materialized_copies.append((destination, sources[0]))
        if (
            data.get("type") == "Phi"
            and attributes.get("aggregate_kind")
            and destination in declaration_specs
        ):
            for parent_id, _role in data.get("parents", ()):
                parent_id = int(parent_id)
                if (
                    parent_id in declaration_specs
                    and declaration_specs[destination]
                    == declaration_specs[parent_id]
                ):
                    materialized_copies.append((destination, parent_id))
    changed = True
    while changed:
        changed = False
        for mutation in copies:
            source = contracts.get(int(mutation.argument_value_ids[0]))
            destination = int(mutation.sequence_value_id)
            if source is not None and destination in declared and destination not in contracts:
                contracts[destination] = source
                changed = True
            source_id = int(mutation.argument_value_ids[0])
            expression = graph_obj.nodes.get(source_id, {}).get("expr_obj")
            if (source_id not in contracts and destination in contracts
                    and isinstance(expression, ast.Dict) and not expression.keys):
                # An empty constructor has no conflicting row type. Its copy
                # consumer supplies the storage schema, not runtime contents.
                contracts[source_id] = contracts[destination]
                changed = True
        for destination, source_id in materialized_copies:
            source = contracts.get(source_id)
            destination_contract = contracts.get(destination)
            if source is not None and destination_contract is None:
                contracts[destination] = source
                changed = True
            elif destination_contract is not None and source is None:
                contracts[source_id] = destination_contract
                changed = True
    return contracts


def _sequence_record_identity_contracts(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> dict[int, str]:
    """Correlate resident sequence IDs with declared record-row identities."""

    declared = {int(item[0]) for item in sequence_declarations}
    identities = graph_obj.graph.get("identity_table") or {}
    records = _graph_sequence_record_abi(graph_obj)
    abi_identities = tuple(map(
        str, dict((graph_obj.graph.get("program_abi") or {}).get("records") or {})
    ))
    result: dict[int, str] = {}
    for binding_name, record in records.items():
        binding_ids = _sequence_record_binding_value_ids(
            graph_obj, binding_name,
        )
        sequence_id = next((
            int(value_id)
            for value_id in binding_ids
            if int(value_id) in declared
        ), None)
        if sequence_id is None:
            continue
        identity = str(record.get("identity") or "")
        canonical = tuple(
            candidate for candidate in abi_identities
            if candidate == identity or candidate.rsplit(".", 1)[-1] == identity
        )
        if len(canonical) == 1:
            identity = canonical[0]
        if identity:
            result[int(sequence_id)] = identity
    return result
