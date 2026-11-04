"""Concorded physical defaults of canonical native tensor constructors.

The authored ``dtype=None`` is only an omission proof. Physical defaults come
from selected constructor source declarations or typed repository SSA stores.
"""

from __future__ import annotations

from typing import Any

from .concordance_declarations import (
    INGESTION, NUMERICAL_CONSTRUCTOR_CODE_REFERENCE,
    NUMERICAL_CONSTRUCTOR_DEFAULT, PLANNER_HIERARCHY,
    REPOSITORY_KERNEL_DEFINITION, REPOSITORY_KERNEL_VALUE,
    SOURCE_CALLABLE_SIGNATURE, SOURCE_CALL_PARAMETER_DEFAULT,
    SOURCE_INTRINSIC_ADMISSION, SOURCE_PARAMETER_DEFAULT,
    PROCESS_GRAPH_OPERANDS, SSA_VALUE, SpecializationFact,
    SpecializationSource,
)
from .identity_concordance import Derived, Mode, Novel, Ref, current_identity_book
from .concordance_declarations import INGEST_SOURCE


_CONSTRUCTORS = frozenset({"zeros", "ones", "empty", "full"})


def selected_literal_constructor_source(reference):
    """Ingest the selected reference's literal constructor before source planning."""

    from ..transmogrifier.graph.graph_express2 import (
        ingest_callable_signature_by_reference,
    )

    constructor = getattr(reference, "compile_time_constructors", {}).get(
        "ensure_tensor"
    ) if reference is not None else None
    if constructor is None:
        return None
    book = current_identity_book()
    signature = ingest_callable_signature_by_reference(constructor)
    if signature is None:
        return None
    schema = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
        signature.row, signature.column,
    )[2]
    defaults = tuple(cell for name, _kind, _position, cell in schema
                     if name == "dtype" and isinstance(cell, Ref)
                     and cell.page == SOURCE_PARAMETER_DEFAULT)
    if len(defaults) != 1:
        return None
    default = defaults[0]
    default_fact = book.page(SOURCE_PARAMETER_DEFAULT).cell(
        default.row, default.column,
    )
    if (not isinstance(default_fact, SpecializationFact)
            or default_fact.source != SpecializationSource.DEFAULT
            or not isinstance(default_fact.value, str)):
        return None
    row = (reference.source_identity, "ensure_tensor")
    declaration = book.latest_ref(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, row)
    if declaration is None:
        declaration = book.post(
            NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, row, (signature,),
            stage=INGESTION, provenance=Derived((signature,)),
            mode=Mode.CONCORD,
        )
    elif book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE).cell(
            declaration.row, declaration.column) != (signature,):
        return None
    return declaration, signature, default


def source_literal_constructor_default(graph, node_id):
    """Read selected physical dtype for an exact numeric tuple/list argument."""

    from .concordance_declarations import (
        PROCESS_GRAPH_OPERANDS, SOURCE_AGGREGATE_LAYOUT,
        SOURCE_INTRINSIC_ADMISSION,
    )
    from .concorded_literals import MISSING, literal, origin_cell
    from ..common.tensors.topological_reducer import node_identity_cell

    book = current_identity_book()
    selected = graph.G.graph.get("selected_literal_constructor_source")
    if not (isinstance(selected, tuple) and len(selected) == 3
            and all(isinstance(cell, Ref) for cell in selected)):
        return None, ()
    declaration, signature, default = selected
    if (declaration.page != NUMERICAL_CONSTRUCTOR_CODE_REFERENCE
            or declaration.row[1] != "ensure_tensor"
            or book.page(declaration.page).cell(
                declaration.row, declaration.column) != (signature,)):
        return None, ()
    default_fact = book.page(SOURCE_PARAMETER_DEFAULT).cell(
        default.row, default.column,
    )
    if (not isinstance(default_fact, SpecializationFact)
            or default_fact.source != SpecializationSource.DEFAULT
            or not isinstance(default_fact.value, str)):
        return None, ()
    data = graph.G.nodes.get(int(node_id))
    if data is None:
        return None, ()
    call = node_identity_cell(graph, int(node_id), book=book)
    try:
        origin = origin_cell(book, call)
    except ValueError:
        return None, ()
    admission = book.latest_ref(SOURCE_INTRINSIC_ADMISSION, (origin,))
    if admission is None:
        return None, ()
    admitted_op, decision, _authored_signature = book.page(
        SOURCE_INTRINSIC_ADMISSION,
    ).cell(admission.row, admission.column)
    if not (admitted_op == "ensure_tensor"
            and isinstance(decision, dict)
            and decision.get("action") == "intrinsic"
            and (decision.get("parameters") or {}).get("lowering_namespace")
                == "abstract_tensor"):
        return None, ()
    payloads = tuple(int(parent) for parent, role in data.get("parents") or ()
                     if str(role).casefold() in {"arg:0", "value"}
                     and int(parent) in graph.G)
    if len(payloads) != 1:
        return None, ()
    payload = node_identity_cell(graph, payloads[0], book=book)
    operand = book.latest_ref(PROCESS_GRAPH_OPERANDS, call.row)
    if (operand is None or (payload, "arg:0") not in book.page(
            PROCESS_GRAPH_OPERANDS,
    ).cell(operand.row, operand.column)):
        return None, ()
    scope = graph.G.graph.get("source_aggregate_layout_scope")
    layout = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (scope, payloads[0]))
    if layout is None:
        return None, ()
    layout_fact = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
        layout.row, layout.column,
    )
    if (layout_fact.aggregate_kind not in {"tuple", "list"}
            or not layout_fact.members):
        return None, ()
    cells = [call, payload, operand, admission, layout,
             declaration, signature, default]
    for member in layout_fact.members:
        value, literal_cells = literal(graph, int(member.value_id))
        if value is MISSING or type(value) not in {bool, int, float}:
            return None, ()
        cells.extend((member.value_cell, *member.source_cells, *literal_cells))
    return default_fact.value, tuple(dict.fromkeys(cells))


def _stored_output_values(function: Any) -> tuple[Any, tuple[Any, ...]] | None:
    """Follow only explicit pointer-preserving SSA edges from named output."""

    position = function.metadata.get("ssa_output_argument")
    if not isinstance(position, int) or position < 0 or position >= len(function.args):
        return None
    output = function.args[position]
    if ("out", int(output.id)) not in function.metadata.get("named_outputs", ()):
        return None
    pointers = {int(output.id)}
    stores = []
    instructions = [instruction for block in function.blocks.values()
                    for instruction in block.instrs]
    changed = True
    while changed:
        changed = False
        for instruction in instructions:
            if (str(instruction.op) in {"GetElementPtr", "Cast"}
                    and instruction.res is not None and instruction.args
                    and str(instruction.res.dtype) == "ptr"
                    and int(instruction.args[0].id) in pointers
                    and int(instruction.res.id) not in pointers):
                pointers.add(int(instruction.res.id))
                changed = True
    for instruction in instructions:
        if str(instruction.op) != "Store":
            continue
        if len(instruction.args) != 2:
            return None
        value, pointer = instruction.args
        if int(pointer.id) in pointers:
            stores.append(value)
    if not stores or any(value.dtype != stores[0].dtype for value in stores):
        return None
    return output, tuple(stores)


def constructor_default_dtype(graph: Any, node_id: int) -> tuple[str | None, tuple[Ref, ...]]:
    """Return a proved native physical default, or no proof.

    Source semantic dtype remains ``None``.  Explicit/unknown dtype operands,
    absent source signature proof, and unsupported SSA output aliases refuse.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
        post_repository_kernel_identity,
    )
    from .ssa_record_return_state import function_scope_of
    from .concorded_literals import origin_cell
    from .process_graph_facts import operands

    data = graph.G.nodes.get(int(node_id))
    if data is None:
        return None, ()
    operation = str(data.get("op") or data.get("type") or "")
    if operation not in _CONSTRUCTORS:
        return None, ()
    book = current_identity_book()
    call = node_identity_cell(graph, int(node_id), book=book)
    attributes = data.get("attributes") or {}
    default = (attributes.get("source_parameter_default_cells") or {}).get("dtype")
    if not isinstance(default, Ref) or default.page != SOURCE_CALL_PARAMETER_DEFAULT:
        return None, ()
    if default.row[1] != "dtype":
        return None, ()
    try:
        same_call = origin_cell(book, default.row[0]) == origin_cell(book, call)
    except ValueError:
        same_call = False
    if not same_call:
        return None, ()
    admission = book.latest_ref(SOURCE_INTRINSIC_ADMISSION, (
        origin_cell(book, call),
    ))
    if admission is None:
        return None, ()
    admitted_op, decision, admitted_signature = book.page(
        SOURCE_INTRINSIC_ADMISSION,
    ).cell(admission.row, admission.column)
    if not (admitted_op == operation
            and isinstance(decision, dict)
            and decision.get("action") == "intrinsic"
            and (decision.get("parameters") or {}).get("lowering_namespace")
                == "abstract_tensor"):
        return None, ()
    default_causes = {source for source, _stage in book.edges_into(default)}
    if default.row[0] not in default_causes:
        return None, ()
    default_fact = book.page(SOURCE_CALL_PARAMETER_DEFAULT).cell(
        default.row, default.column,
    )
    if not (isinstance(default_fact, tuple) and len(default_fact) == 2
            and default_fact[0].page == SOURCE_PARAMETER_DEFAULT
            and default_fact[1].page == SOURCE_CALLABLE_SIGNATURE
            and set(default_fact) <= default_causes):
        return None, ()
    authored_default, signature = default_fact
    public_symbol = signature.row[0]
    if (signature != admitted_signature
            or decision.get("module") != public_symbol.row[0]
            or decision.get("qualname") != public_symbol.row[1]
            or decision.get("identity") != ".".join(public_symbol.row[:2])):
        return None, ()
    default_value = book.page(SOURCE_PARAMETER_DEFAULT).cell(
        authored_default.row, authored_default.column,
    )
    schema = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
        signature.row, signature.column,
    )[2]
    dtype_formals = tuple(item for item in schema
                          if item[0] == "dtype" and item[3] == authored_default)
    if (not isinstance(default_value, SpecializationFact)
            or default_value.source != SpecializationSource.DEFAULT
            or default_value.value is not None
            or len(dtype_formals) != 1):
        return None, ()
    dtype_position = dtype_formals[0][2]
    current_edges, operand_cell = operands(graph, int(node_id))
    if (operand_cell.page != PROCESS_GRAPH_OPERANDS
            or call not in {source for source, _stage
                            in book.edges_into(operand_cell)}
            or any(role == "kw:dtype"
                   or (dtype_position is not None
                       and role == f"arg:{dtype_position}")
                   for _parent, role in current_edges)):
        return None, ()

    reference = c_backend_repository_ssa_reference()
    operation_reference = reference.operation(operation)
    if (operation_reference is None
            or operation_reference.canonical_op != operation
            or len(operation_reference.entrypoints) != 1):
        return None, ()
    entrypoint = operation_reference.entrypoints[0]
    function = reference.module.functions[entrypoint]
    output_stores = _stored_output_values(function)
    if output_stores is None:
        return None, ()
    output, stored_values = output_stores
    if not post_repository_kernel_identity(function, book):
        return None, ()
    root_row = ("llvm_repository", entrypoint)
    root = book.latest_ref(REPOSITORY_KERNEL_DEFINITION, root_row)
    origins = function.metadata["llvm_repository_kernel"][1]
    output_position = origins[int(output.id)][0]
    output_cell = book.latest_ref(REPOSITORY_KERNEL_VALUE, (root_row, output_position))
    scope = function_scope_of(function)
    stored_cells = tuple(book.latest_ref(SSA_VALUE, (scope, int(value.id)))
                         for value in stored_values)
    if root is None or output_cell is None or any(cell is None for cell in stored_cells):
        return None, ()
    dtype = str(stored_values[0].dtype)
    if not dtype or dtype in {"None", "ptr"}:
        return None, ()
    declaration_row = (reference.source_identity, operation)
    declaration = book.latest_ref(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, declaration_row)
    declaration_fact = tuple(operation_reference.entrypoints)
    if declaration is None:
        declaration = book.post(
            NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, declaration_row,
            declaration_fact, stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
    elif book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE).cell(
            declaration.row, declaration.column) != declaration_fact:
        return None, ()
    fact = (operation, default, admission, operand_cell,
            declaration, root, output_cell, stored_cells, dtype)
    causes = tuple(dict.fromkeys((call, default, authored_default, signature,
                                   admission, operand_cell, declaration, root, output_cell,
                                   *stored_cells)))
    row = (call,)
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, row)
    if receipt is None:
        receipt = book.post(
            NUMERICAL_CONSTRUCTOR_DEFAULT, row, fact,
            stage=PLANNER_HIERARCHY, provenance=Derived(causes),
            mode=Mode.CONCORD,
        )
    elif book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).cell(
            receipt.row, receipt.column) != fact:
        return None, ()
    return dtype, (receipt, *causes)


__all__ = ["constructor_default_dtype"]


def _ssa_canonical_origin(book, scope, value_id):
    """Read one canonical cause through this value's booked SSA revisions."""

    from .concordance_declarations import CANONICAL_VALUE

    row = (scope, int(value_id))
    latest = book.latest_ref(SSA_VALUE, row)
    pending = [] if latest is None else [latest]
    seen = set()
    origins = []
    while pending:
        cell = pending.pop(0)
        if cell in seen:
            continue
        seen.add(cell)
        for parent, _stage in book.edges_into(cell):
            if parent.page == CANONICAL_VALUE:
                origins.append((cell, parent))
            elif parent.page == SSA_VALUE and parent.row == row:
                pending.append(parent)
    canonical = {parent for _cell, parent in origins}
    # The deepest direct source is the booked SSA birth. Later revisions may
    # repeat that same canonical cause without becoming a new birth.
    return origins[-1] if len(canonical) == 1 else (None, None)


def literal_constructor_default(function, instruction, constant, reference, *, book):
    """Prove a selected compile-time tensor constructor from its source literal.

    The literal's Python element type is not the selected backend's storage
    type.  The latter comes from the ingested constructor's default argument.
    """

    from .concordance_declarations import (
        CANONICAL_VALUE, PROVEN_LITERAL, SSA_VALUE,
    )
    from .concorded_literals import origin_cell
    from .ssa_record_return_state import function_scope_of
    from ..transmogrifier.graph.graph_express2 import (
        ingest_callable_signature_by_reference,
    )

    constructor = reference.compile_time_constructors.get("ensure_tensor")
    if constructor is None or len(instruction.args) != 2 or instruction.res is None:
        return None
    if constant.res is None or constant.res.id != instruction.args[1].id:
        return None
    payload = constant.attributes.get("value")
    if not isinstance(payload, (tuple, list)):
        return None
    scope = function_scope_of(function)

    result_ssa, call = _ssa_canonical_origin(
        book, scope, instruction.res.id,
    )
    payload_ssa, payload_canonical = _ssa_canonical_origin(
        book, scope, constant.res.id,
    )
    if any(cell is None for cell in (
        result_ssa, call, payload_ssa, payload_canonical,
    )):
        return None
    operand = book.latest_ref(PROCESS_GRAPH_OPERANDS, call.row)
    if operand is None or (payload_canonical, "arg:0") not in book.page(
        PROCESS_GRAPH_OPERANDS,
    ).cell(operand.row, operand.column):
        return None
    provenance = []
    for row in book.page(PROVEN_LITERAL).rows():
        cell = book.latest_ref(PROVEN_LITERAL, row)
        if cell is None or book.page(PROVEN_LITERAL).cell(
            cell.row, cell.column,
        ) != payload:
            continue
        members = tuple(source for source, _stage in book.edges_into(cell)
                        if source.page == CANONICAL_VALUE)
        if any(origin_cell(book, member)
               == origin_cell(book, payload_canonical) for member in members):
            provenance.append(cell)
    if len(provenance) != 1:
        return None
    literal = provenance[0]
    if current_identity_book() is not book:
        return None
    signature = ingest_callable_signature_by_reference(constructor)
    if signature is None:
        return None
    schema = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
        signature.row, signature.column,
    )[2]
    defaults = tuple(cell for name, _kind, _position, cell in schema
                     if name == "dtype" and isinstance(cell, Ref)
                     and cell.page == SOURCE_PARAMETER_DEFAULT)
    if len(defaults) != 1:
        return None
    default = defaults[0]
    default_fact = book.page(SOURCE_PARAMETER_DEFAULT).cell(
        default.row, default.column,
    )
    if (not isinstance(default_fact, SpecializationFact)
            or default_fact.source != SpecializationSource.DEFAULT
            or not isinstance(default_fact.value, str)):
        return None
    dtype = default_fact.value
    declaration_row = (reference.source_identity, "ensure_tensor")
    declaration_fact = (signature,)
    declaration = book.latest_ref(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE,
                                  declaration_row)
    if declaration is None:
        declaration = book.post(
            NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, declaration_row,
            declaration_fact, stage=INGESTION,
            provenance=Derived((signature,)), mode=Mode.CONCORD,
        )
    elif book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE).cell(
            declaration.row, declaration.column) != declaration_fact:
        return None
    fact = ("ensure_tensor", payload_ssa, result_ssa, operand, literal,
            signature, default, declaration, dtype)
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, (call,))
    if receipt is None:
        receipt = book.post(
            NUMERICAL_CONSTRUCTOR_DEFAULT, (call,), fact,
            stage=PLANNER_HIERARCHY,
            provenance=Derived((call, payload_canonical, payload_ssa,
                                result_ssa, operand, literal, signature,
                                default, declaration)), mode=Mode.CONCORD,
        )
    elif book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).cell(
            receipt.row, receipt.column) != fact:
        return None
    return tuple(payload), dtype, receipt


__all__.append("literal_constructor_default")
