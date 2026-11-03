"""Read published source evidence for exact source values.

The resolver does not discover methods or publish facts.  A missing answer is
an explicit demand on the earlier source evidence pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .concordance_declarations import (
    CALLABLE_IDENTITY, CANONICAL_VALUE, CELL_SET, DESCRIPTOR_CELL, FUNCTION_ADDRESS, INGESTION_VALUE,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_CLOSED_AGGREGATE_LENGTHS,
    SOURCE_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_CALL_ACTUAL_FORMAL, SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS,
    SEQUENCE_CONTRACT_CONCORDANCE, SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_MAPPING_GET_READ,
    SOURCE_INT_RANGE, SOURCE_PYTHON_IDENTITY,
    SOURCE_RESOURCE_ADMISSION_DEMAND, SOURCE_VALUE_CLASS, SSA_VALUE,
    VALUE_TYPE_ABI, SSAValueFact, ValueTypeABIFact,
    SourceAggregateLayout, SourceAggregateMember, SourceClosedAggregateLengths,
    SourceClosedClassAlternatives,
)
from .identity_concordance import Ref, current_identity_book


@dataclass(frozen=True)
class UnresolvedSourceDemand:
    scope: Any
    value_id: int
    facet: str
    missing: str
    subject_cell: Ref | None = None
    alternatives: tuple[int, ...] = ()
    proof_cells: tuple[Ref, ...] = ()
    control_scope: Ref | None = None


@dataclass(frozen=True)
class CallableEvidence:
    scope: Any
    value_id: int
    address: int
    subject_cell: Ref
    callable_cell: Ref
    address_cell: Ref
    proof_cells: tuple[Ref, ...]
    ordered_inputs: tuple[tuple[str, Ref], ...]
    control_scope: Ref | UnresolvedSourceDemand


@dataclass(frozen=True)
class ClassEvidence:
    scope: Any
    value_id: int
    class_identity: str
    precision_limbs: int
    subject_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Any
    class_cell: Ref | None = None


@dataclass(frozen=True)
class ClassAlternativesEvidence:
    scope: Any
    value_id: int
    alternatives: tuple[tuple[str, int], tuple[str, int]]
    subject_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Ref


@dataclass(frozen=True)
class LengthEvidence:
    scope: Any
    value_id: int
    aggregate_kind: str
    length: int
    subject_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Any


@dataclass(frozen=True)
class MemberEvidence:
    scope: Any
    value_id: int
    index: int
    member_value_id: int
    member_cell: Ref
    layout_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Any


@dataclass(frozen=True)
class PhysicalLayoutEvidence:
    scope: Any
    value_id: int
    physical_dtype: str
    storage: str
    rank: int
    shape: tuple
    arena: bool
    device: Any
    subject_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Any


@dataclass(frozen=True)
class IntegerDomainEvidence:
    scope: Any
    value_id: int
    lower: int
    upper: int
    subject_cell: Ref
    proof_cells: tuple[Ref, ...]
    control_scope: Any


def _source_cells(book: Any, subject: Ref) -> tuple[Ref, ...]:
    """The exact published predecessors of one concordance cell."""

    return tuple(dict.fromkeys(parent for parent, _stage in book.edges_into(subject)))


def require_integer_domain(scope: Any, value_id: int, *, book: Any = None):
    """Read one exact source integer interval in its authored or copied scope."""

    book = current_identity_book() if book is None else book
    value_id = int(value_id)
    subject = book.latest_ref(SOURCE_INT_RANGE, (scope, value_id))
    if subject is None:
        return UnresolvedSourceDemand(
            scope, value_id, "integer_domain", "missing_source_int_range")
    interval = book.page(SOURCE_INT_RANGE).cell(subject.row, subject.column)
    if (not isinstance(interval, tuple) or len(interval) != 2
            or any(type(bound) is not int for bound in interval)
            or interval[0] > interval[1]):
        return UnresolvedSourceDemand(
            scope, value_id, "integer_domain", "invalid_source_int_range",
            subject)
    proof = _source_cells(book, subject)
    if not proof:
        return UnresolvedSourceDemand(
            scope, value_id, "integer_domain", "missing_integer_domain_provenance",
            subject)
    return IntegerDomainEvidence(
        scope, value_id, int(interval[0]), int(interval[1]),
        subject, proof, scope,
    )


def _descriptor_span_proof(book: Any, subject: Ref, fact: SSAValueFact) -> tuple[Ref, ...]:
    """Certify the exact typed one-cell sequence descriptor mint and revision.

    Program: the generated extent argument of root(xs: list[float]) calling
    helper(xs) in test_source_evidence_span_layout. Its SSA row has no ABI
    arm, but the descriptor mint and span revision each name the same exact
    sequence contract. The C array decision must read that path, rather than
    resident extent accounting.
    """

    if (fact.physical_dtype != "int64" or fact.storage != "span"
            or fact.rank != 1 or tuple(fact.shape) != (1,)):
        return ()
    revision_sources = _source_cells(book, subject)
    previous = tuple(cell for cell in revision_sources
                     if cell.page == SSA_VALUE and cell.row == subject.row)
    sequence = tuple(cell for cell in revision_sources
                     if cell.page == SEQUENCE_CONTRACT_CONCORDANCE)
    if len(previous) != 1 or len(sequence) != 1:
        return ()
    minted = book.page(SSA_VALUE).cell(previous[0].row, previous[0].column)
    mint = book.mint_of(previous[0])
    if (not isinstance(minted, SSAValueFact)
            or minted.physical_dtype != "int64" or tuple(minted.shape) != (1,)
            or mint is None or mint[0] != DESCRIPTOR_CELL
            or len(mint[1]) != 1 or mint[1][0].page != CELL_SET):
        return ()
    cell_set = mint[1][0]
    causes = _source_cells(book, cell_set)
    owner = tuple(cell for cell in causes
                  if cell.page == SSA_VALUE and cell.row[0] == subject.row[0])
    if (len(causes) != 2 or len(owner) != 1
            or sequence[0] not in causes
            or sequence[0].row[-1] != owner[0].row[-1]):
        return ()
    return tuple(dict.fromkeys((previous[0], sequence[0], cell_set, *causes)))


def require_physical_layout(scope: Any, value_id: int, *, book: Any = None):
    """Read a posted scalar or exact ABI-backed span physical layout.

    The SSA_VALUE row owns the physical fields. A span additionally requires
    its exact VALUE_TYPE_ABI view in the same SSA predecessor chain.
    Source class and semantic dtype cannot fill an absent physical field.
    The returned scope is the exact SSA function scope that owns the row.
    """

    book = current_identity_book() if book is None else book
    value_id = int(value_id)
    subject = book.latest_ref(SSA_VALUE, (scope, value_id))
    if subject is None:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "missing_ssa_value")
    fact = book.page(SSA_VALUE).cell(subject.row, subject.column)
    if not isinstance(fact, SSAValueFact):
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "invalid_ssa_value", subject)
    if not isinstance(fact.physical_dtype, str) or not fact.physical_dtype:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "missing_physical_dtype", subject)
    if fact.storage is None:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "missing_physical_storage", subject)
    if fact.rank is None:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "missing_physical_rank", subject)
    if fact.storage == "scalar" and fact.rank == 0 and tuple(fact.shape) == ():
        sources = _source_cells(book, subject)
        if not sources:
            mint = book.mint_of(subject)
            if mint is not None:
                transform, operands = mint
                if len(operands) == transform.arity and operands:
                    sources = tuple(dict.fromkeys((
                        *operands,
                        *(cell for operand in operands
                          for cell in _source_cells(book, operand)),
                    )))
        if not sources:
            return UnresolvedSourceDemand(
                scope, value_id, "physical_layout", "missing_physical_provenance",
                subject)
        return PhysicalLayoutEvidence(
            scope, value_id, fact.physical_dtype, fact.storage, fact.rank,
            tuple(fact.shape), fact.arena, fact.device, subject, sources, scope,
        )
    if fact.storage != "span" or fact.rank <= 0:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "unsupported_physical_layout",
            subject)
    # A revision may add arena ownership after the formal's ABI birth. Walk
    # only same-row SSA predecessors; a matching class or dtype elsewhere
    # cannot supply the formal's physical view.
    pending, seen, proof, abi_cells = [subject], set(), [], []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for predecessor in _source_cells(book, current):
            proof.append(predecessor)
            if predecessor.page == SSA_VALUE and predecessor.row == subject.row:
                pending.append(predecessor)
            elif predecessor.page == VALUE_TYPE_ABI:
                abi_cells.append(predecessor)
    if not abi_cells:
        descriptor_proof = _descriptor_span_proof(book, subject, fact)
        if descriptor_proof:
            return PhysicalLayoutEvidence(
                scope, value_id, fact.physical_dtype, fact.storage, fact.rank,
                tuple(fact.shape), fact.arena, fact.device, subject,
                descriptor_proof, scope,
            )
    if len(set(abi_cells)) != 1:
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "missing_unique_span_abi",
            subject, proof_cells=tuple(dict.fromkeys(proof)))
    abi_cell = abi_cells[0]
    abi = book.page(VALUE_TYPE_ABI).cell(abi_cell.row, abi_cell.column)
    arms = (tuple(abi.physical_alternatives)
            if isinstance(abi, ValueTypeABIFact) else ())
    views = {(arm.storage, arm.dtype, tuple(arm.shape), arm.rank)
             for arm in arms}
    # _octets' authored bytes span has one unknown extent. VALUE_TYPE_ABI
    # encodes it as (-1,), while the SSA formal retains no static shape and
    # posts rank=1. The dynamic ABI cell proves these are the same physical
    # span; no partial or conflicting shape is admitted by this equivalence.
    dynamic_unknown_extent = (
        len(arms) == 1
        and arms[0].metadata_state == "dynamic"
        and arms[0].storage == "span"
        and arms[0].dtype == fact.physical_dtype
        and arms[0].rank == fact.rank
        and tuple(arms[0].shape) == (-1,) * fact.rank
        and tuple(fact.shape) == ()
    )
    if (views != {("span", fact.physical_dtype, tuple(fact.shape), fact.rank)}
            and not dynamic_unknown_extent):
        return UnresolvedSourceDemand(
            scope, value_id, "physical_layout", "span_abi_disagreement",
            subject, proof_cells=tuple(dict.fromkeys(proof)))
    return PhysicalLayoutEvidence(
        scope, value_id, fact.physical_dtype, fact.storage, fact.rank,
        tuple(fact.shape), fact.arena, fact.device, subject,
        tuple(dict.fromkeys((abi_cell, *proof))), scope,
    )


def require_class(
    scope: Any, value_id: int, *, control_scope: Ref | None = None,
    book: Any = None,
):
    """Read a global class, or one class valid at a guarded present-arm Call.

    The control-scoped query names the exact SOURCE_GUARDED_MAPPING_PAYLOAD
    cell. Its row owns both the optional get result and the borrowed Call.
    No class is published for the optional result outside that arm.
    """

    book = current_identity_book() if book is None else book
    value_id = int(value_id)
    if control_scope is not None:
        from .source_guarded_payload import candidate_facts

        if (not isinstance(control_scope, Ref)
                or control_scope.page != SOURCE_GUARDED_MAPPING_PAYLOAD
                or control_scope.row[:2] != (scope, value_id)
                or book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD,
                                   control_scope.row) != control_scope):
            return UnresolvedSourceDemand(
                scope, value_id, "class", "missing_exact_guarded_scope",
                control_scope if isinstance(control_scope, Ref) else None)
        guard_fact = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).cell(
            control_scope.row, control_scope.column)
        if not isinstance(guard_fact, tuple) or len(guard_fact) < 2:
            return UnresolvedSourceDemand(
                scope, value_id, "class", "invalid_guarded_payload",
                control_scope)
        read, stored = guard_fact[:2]
        if (not isinstance(read, Ref) or read.page != SOURCE_MAPPING_GET_READ
                or not isinstance(stored, Ref)
                or (control_scope.row, guard_fact)
                not in candidate_facts(book, scope, value_id, read)):
            return UnresolvedSourceDemand(
                scope, value_id, "class", "missing_guarded_payload_proof",
                control_scope)
        read_fact = book.page(SOURCE_MAPPING_GET_READ).cell(
            read.row, read.column)
        if (read_fact[5] != stored or read_fact[7].row[-1] != value_id):
            return UnresolvedSourceDemand(
                scope, value_id, "class", "guarded_payload_identity_disagreement",
                control_scope)
        subject = read_fact[7]
        store_scope = read_fact[1].row[0]
        class_cell = book.latest_ref(
            SOURCE_VALUE_CLASS, (store_scope, stored.row[-1]))
        if class_cell is None:
            return UnresolvedSourceDemand(
                scope, value_id, "class", "missing_stored_payload_class",
                subject)
        fact = book.page(SOURCE_VALUE_CLASS).cell(
            class_cell.row, class_cell.column)
        if (not isinstance(fact, tuple) or len(fact) < 3
                or not isinstance(fact[0], str) or not fact[0]
                or int(fact[1]) != 1
                or fact[2] != "source class construction"):
            return UnresolvedSourceDemand(
                scope, value_id, "class", "invalid_stored_payload_class",
                subject)
        class_sources = _source_cells(book, class_cell)
        if not class_sources:
            return UnresolvedSourceDemand(
                scope, value_id, "class", "missing_class_provenance",
                subject)
        proof = tuple(dict.fromkeys((control_scope, read, stored,
                                     class_cell, *class_sources)))
        return ClassEvidence(
            scope, value_id, fact[0], 1, subject, proof,
            control_scope, class_cell)

    subject = book.latest_ref(SOURCE_VALUE_CLASS, (scope, value_id))
    if subject is None:
        return UnresolvedSourceDemand(scope, value_id, "class", "missing_source_value_class")
    fact = book.page(SOURCE_VALUE_CLASS).cell(subject.row, subject.column)
    if not (isinstance(fact, tuple) and len(fact) >= 2
            and isinstance(fact[0], str) and fact[0] and int(fact[1]) > 0):
        return UnresolvedSourceDemand(scope, value_id, "class", "invalid_source_value_class", subject)
    sources = _source_cells(book, subject)
    if not sources:
        return UnresolvedSourceDemand(scope, value_id, "class", "missing_class_provenance", subject)
    return ClassEvidence(scope, value_id, fact[0], int(fact[1]),
                         subject, sources, scope, subject)


def require_class_alternatives(
    scope: Any, value_id: int, *, control_scope: Ref | None = None,
    book: Any = None,
):
    """Read closed semantic IfExp classes only at their authored control.

    This does not establish a native tagged representation or global class.
    """

    book = current_identity_book() if book is None else book

    def domain(source: Ref, seen: frozenset[Ref] = frozenset()):
        """Follow only published call/formal links to an authored arm page."""

        if not isinstance(source, Ref) or source in seen:
            return None
        if source.page == SOURCE_CLOSED_CLASS_ALTERNATIVES:
            fact = book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cell(
                source.row, source.column)
            return ((fact, (source,))
                    if isinstance(fact, SourceClosedClassAlternatives)
                    else None)
        if source.page not in {
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
        }:
            return None
        cells = book.page(source.page).cell(source.row, source.column)
        if not (isinstance(cells, tuple) and len(cells) == 5
                and all(isinstance(cell, Ref) for cell in cells)):
            return None
        nested = domain(cells[2], seen | {source})
        return None if nested is None else (
            nested[0], tuple(dict.fromkeys((source, *cells, *nested[1]))),
        )

    value_id = int(value_id)
    subject = book.latest_ref(SOURCE_CLOSED_CLASS_ALTERNATIVES,
                              (scope, value_id))
    if subject is None:
        formal_page = book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES)
        formal_rows = tuple(row for row in formal_page.scope_rows(scope)
                            if int(row[1]) == value_id)
        if formal_rows:
            binding_page = book.page(SOURCE_CALL_ACTUAL_FORMAL)
            origin_scope = book.read_through.get(scope, (scope, None))[0]
            reached = tuple(
                row for row in binding_page.rows()
                if int(row[3]) == value_id
                and (binding := binding_page.latest(row)) is not None
                and isinstance(binding[4], Ref)
                and binding[4].row[0] == origin_scope
            )
            proven = {(row[2], row[3]) for row in formal_rows}
            inventory = {(row[0], row[1]) for row in reached}
            first = book.latest_ref(
                SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES, formal_rows[0])
            if proven != inventory:
                return UnresolvedSourceDemand(
                    scope, value_id, "class_alternatives",
                    "incomplete_formal_closed_class_domain", first)
            facts = tuple(formal_page.cell(
                (cell := book.latest_ref(
                    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES, row)).row,
                cell.column,
            ) for row in formal_rows)
            resolved = tuple(domain(fact[2]) for fact in facts)
            if (any(item is None for item in resolved)
                    or len({item[0].alternatives for item in resolved}) != 1
                    or len({fact[4] for fact in facts}) != 1):
                return UnresolvedSourceDemand(
                    scope, value_id, "class_alternatives",
                    "conflicting_formal_closed_class_domain", first)
            formal_binding = facts[0][4]
            if control_scope is not None and control_scope != formal_binding:
                return UnresolvedSourceDemand(
                    scope, value_id, "class_alternatives",
                    "missing_exact_formal_class_alternative_owner", first,
                    proof_cells=(formal_binding,), control_scope=control_scope)
            proof = tuple(dict.fromkeys((
                *(book.latest_ref(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
                                  row) for row in formal_rows),
                *(cell for fact in facts for cell in fact),
                *(cell for item in resolved for cell in item[1]),
                *(cell for item in resolved for cell in (
                    item[0].control_cell, item[0].operand_cell,
                    *item[0].arm_value_cells, *item[0].arm_class_cells)),
            )))
            return ClassAlternativesEvidence(
                scope, value_id, resolved[0][0].alternatives, first,
                proof, formal_binding)
        # A reached call owns its own control boundary. The callee IfExp
        # control is retained as provenance, not treated as a caller guard.
        rows = tuple(row for row in book.page(
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).rows()
            if row[:2] == (scope, value_id))
        if len(rows) != 1:
            return UnresolvedSourceDemand(
                scope, value_id, "class_alternatives",
                "missing_closed_class_alternatives")
        subject = book.latest_ref(
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES, rows[0])
        output, returned, closed, reached, call = book.page(
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cell(
                subject.row, subject.column)
        resolved = domain(closed)
        if resolved is None:
            return UnresolvedSourceDemand(
                scope, value_id, "class_alternatives",
                "invalid_closed_class_alternatives", subject)
        fact, domain_proof = resolved
        if control_scope != call:
            return UnresolvedSourceDemand(
                scope, value_id, "class_alternatives",
                "missing_exact_call_class_alternative_control", subject,
                proof_cells=(call,), control_scope=control_scope)
        proof = tuple(dict.fromkeys((
            output, returned, closed, reached, call, *domain_proof,
            fact.control_cell, fact.operand_cell,
            *fact.arm_value_cells, *fact.arm_class_cells,
            *_source_cells(book, subject),
        )))
        return ClassAlternativesEvidence(
            scope, value_id, fact.alternatives, subject, proof, call)
    fact = book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cell(
        subject.row, subject.column)
    if not isinstance(fact, SourceClosedClassAlternatives):
        return UnresolvedSourceDemand(
            scope, value_id, "class_alternatives",
            "invalid_closed_class_alternatives", subject)
    if control_scope != fact.control_cell:
        return UnresolvedSourceDemand(
            scope, value_id, "class_alternatives",
            "missing_exact_class_alternative_control", subject,
            proof_cells=(fact.control_cell,), control_scope=control_scope)
    proof = tuple(dict.fromkeys((fact.control_cell, fact.operand_cell,
                                 *fact.arm_value_cells, *fact.arm_class_cells,
                                 *_source_cells(book, subject))))
    return ClassAlternativesEvidence(
        scope, value_id, fact.alternatives, subject, proof,
        fact.control_cell)


def _tuple_layout(scope: Any, value_id: int, book: Any, facet: str):
    """Read an immutable tuple layout; a missing merge is a publication gap."""

    subject = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (scope, value_id))
    if subject is None:
        return UnresolvedSourceDemand(scope, value_id, facet, "missing_aggregate_layout")
    layout = book.page(SOURCE_AGGREGATE_LAYOUT).cell(subject.row, subject.column)
    if not isinstance(layout, SourceAggregateLayout):
        return UnresolvedSourceDemand(scope, value_id, facet, "invalid_aggregate_layout", subject)
    if layout.aggregate_kind != "tuple":
        return UnresolvedSourceDemand(scope, value_id, facet, "unsupported_exact_length_kind", subject)
    if any(not isinstance(member, SourceAggregateMember)
           or not isinstance(member.value_cell, Ref)
           or not member.source_cells
           or any(not isinstance(cell, Ref) for cell in member.source_cells)
           for member in layout.members):
        return UnresolvedSourceDemand(scope, value_id, facet, "missing_member_provenance", subject)
    sources = _source_cells(book, subject)
    if not sources:
        return UnresolvedSourceDemand(scope, value_id, facet, "missing_layout_provenance", subject)
    if bool(layout.conditional_cell) != bool(layout.arm_layout_cells) or (
        layout.conditional_cell is not None and (
            not isinstance(layout.conditional_cell, Ref)
            or len(layout.arm_layout_cells) != 2
            or any(not isinstance(cell, Ref) for cell in layout.arm_layout_cells)
        )
    ):
        return UnresolvedSourceDemand(scope, value_id, facet, "missing_control_arm_provenance", subject)
    # Copy-on-read scope projections may expose only the origin layout edge.
    # The immutable layout fact itself retains the exact arm/member cells.
    evidence = tuple(dict.fromkeys((
        *sources,
        *(layout.arm_layout_cells or ()),
        *((layout.conditional_cell,) if layout.conditional_cell is not None else ()),
        *(member.value_cell for member in layout.members),
        *(cell for member in layout.members for cell in member.source_cells),
    )))
    return layout, subject, evidence


def require_length(scope: Any, value_id: int, *, book: Any = None):
    """Return exact tuple length or the published closed arm domain."""

    book = current_identity_book() if book is None else book
    value_id = int(value_id)
    result = _tuple_layout(scope, value_id, book, "length")
    if isinstance(result, UnresolvedSourceDemand):
        if result.missing != "missing_aggregate_layout":
            return result
        domain_cell = book.latest_ref(
            SOURCE_CLOSED_AGGREGATE_LENGTHS, (scope, value_id),
        )
        subject_cell = domain_cell
        formal_proof: tuple[Ref, ...] = ()
        if domain_cell is None:
            formal_page = book.page(SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS)
            formal_rows = tuple(
                row for row in formal_page.scope_rows(scope)
                if int(row[1]) == value_id
            )
            if formal_rows:
                binding_page = book.page(SOURCE_CALL_ACTUAL_FORMAL)
                origin_scope = book.read_through.get(scope, (scope, None))[0]
                reached_bindings = tuple(
                    row for row in binding_page.rows()
                    if int(row[3]) == value_id
                    and (fact := binding_page.latest(row)) is not None
                    and isinstance(fact[4], Ref)
                    and fact[4].row[0] == origin_scope
                )
                proven_bindings = {
                    (row[2], row[3]) for row in formal_rows
                }
                all_bindings = {
                    (row[0], row[1]) for row in reached_bindings
                }
                if proven_bindings != all_bindings or len(formal_rows) != 1:
                    return UnresolvedSourceDemand(
                        scope, value_id, "length",
                        "incomplete_formal_closed_length_domain",
                        book.latest_ref(
                            SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS,
                            formal_rows[0],
                        ),
                    )
                formal_row = formal_rows[0]
                subject_cell = book.latest_ref(
                    SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS, formal_row,
                )
                binding_cell, call_domain_cell, domain_cell, output_cell = (
                    formal_page.cell(subject_cell.row, subject_cell.column)
                )
                formal_proof = (
                    subject_cell, binding_cell, call_domain_cell,
                    output_cell,
                )
        if domain_cell is None:
            return result
        domain = book.page(SOURCE_CLOSED_AGGREGATE_LENGTHS).cell(
            domain_cell.row, domain_cell.column,
        )
        if not (
            isinstance(domain, SourceClosedAggregateLengths)
            and domain.aggregate_kind == "tuple"
            and len(domain.lengths) == len(domain.arm_layout_cells) == 2
            and isinstance(domain.conditional_cell, Ref)
            and all(isinstance(cell, Ref) for cell in domain.arm_layout_cells)
        ):
            return UnresolvedSourceDemand(
                scope, value_id, "length", "invalid_closed_length_domain",
                domain_cell,
            )
        arm_layouts = tuple(
            book.page(SOURCE_AGGREGATE_LAYOUT).cell(cell.row, cell.column)
            for cell in domain.arm_layout_cells
        )
        if any(
            not isinstance(layout, SourceAggregateLayout)
            or layout.aggregate_kind != "tuple"
            or len(layout.members) != int(length)
            for layout, length in zip(arm_layouts, domain.lengths, strict=True)
        ):
            return UnresolvedSourceDemand(
                scope, value_id, "length", "closed_length_arm_disagreement",
                domain_cell,
            )
        proof = tuple(dict.fromkeys((
            *formal_proof,
            *_source_cells(book, domain_cell),
            domain.conditional_cell,
            *domain.arm_layout_cells,
            *(member.value_cell for layout in arm_layouts for member in layout.members),
            *(cell for layout in arm_layouts for member in layout.members
              for cell in member.source_cells),
        )))
        return UnresolvedSourceDemand(
            scope, value_id, "length",
            "closed_length_alternatives_require_native_layout", subject_cell,
            tuple(sorted(set(map(int, domain.lengths)))), proof,
            domain.conditional_cell,
        )
    layout, subject, sources = result
    return LengthEvidence(scope, value_id, "tuple", len(layout.members),
                          subject, sources, layout.conditional_cell or scope)


def require_member(scope: Any, value_id: int, index: int, *, book: Any = None):
    """Return one exact tuple member identity and its published sources."""

    book = current_identity_book() if book is None else book
    value_id, index = int(value_id), int(index)
    result = _tuple_layout(scope, value_id, book, "member")
    if isinstance(result, UnresolvedSourceDemand):
        return result
    layout, subject, sources = result
    if index < 0 or index >= len(layout.members):
        return UnresolvedSourceDemand(scope, value_id, "member", "member_index_out_of_bounds", subject)
    member = layout.members[index]
    if (not isinstance(member, SourceAggregateMember)
            or not isinstance(member.value_cell, Ref)
            or not member.source_cells
            or any(not isinstance(cell, Ref) for cell in member.source_cells)):
        return UnresolvedSourceDemand(scope, value_id, "member", "missing_member_provenance", subject)
    return MemberEvidence(scope, value_id, index, int(member.value_id),
                          member.value_cell, subject, tuple(member.source_cells),
                          layout.conditional_cell or scope)


def require_callable(
    graph: Any, scope: Any, value_id: int, *, book: Any = None,
) -> CallableEvidence | UnresolvedSourceDemand:
    """Resolve one graph call from the already-published concordance cells.

    ``scope`` and ``value_id`` are the canonical query key.  For a bound
    resource call, the demand or admitted identity supplies exact ordered
    input cells.  This first slice requires one receiver operand.  A later
    query cannot create a class, callable, or control proof here.
    """

    from ..common.tensors.topological_reducer import (
        _source_numeric_scope, node_identity_cell,
    )

    book = current_identity_book() if book is None else book
    nodes = getattr(graph, "G", graph)
    value_id = int(value_id)

    def unresolved(missing: str, subject: Ref | None = None):
        return UnresolvedSourceDemand(scope, value_id, "callable", missing, subject)

    if scope != _source_numeric_scope(graph):
        return unresolved("scope_mismatch")
    # A post-reduction graph may have canonically relabeled the call vertex.
    # Resource admission has already published its exact original call key and
    # ordered inputs on SOURCE_PYTHON_IDENTITY; read that cell instead of
    # reconstructing the old vertex from a spelling or source coordinate.
    identity_cell = book.latest_ref(SOURCE_PYTHON_IDENTITY, (scope, value_id))
    descriptor = (None if identity_cell is None else
                  book.page(SOURCE_PYTHON_IDENTITY).cell(
                      identity_cell.row, identity_cell.column,
                  ))
    admitted_call = (isinstance(descriptor, tuple)
                     and descriptor[:1] == ("concorded_context_call",))
    if admitted_call and len(descriptor) != 5:
        return unresolved("call_identity_proof_disagreement")
    demands = []
    demand_page = book.page(SOURCE_RESOURCE_ADMISSION_DEMAND)
    canonical = book.latest_ref(
        CANONICAL_VALUE,
        (nodes.graph.get('lexical_read_scope'), value_id),
    ) if hasattr(nodes, 'graph') else None
    canonical_sources = (
        set(_source_cells(book, canonical)) if canonical is not None else set()
    )
    direct_subject = None
    if value_id in nodes:
        try:
            direct_subject = node_identity_cell(graph, value_id, book=book)
        except ValueError:
            pass
    for row in demand_page.rows():
        if row[0] != scope:
            continue
        demand_cell = book.latest_ref(SOURCE_RESOURCE_ADMISSION_DEMAND, row)
        _with_cell, activity, calls = demand_page.cell(
            demand_cell.row, demand_cell.column,
        )
        for call_cell, _receiver, ordered_inputs in calls:
            if (isinstance(call_cell, Ref)
                    and call_cell.page in {INGESTION_VALUE, CANONICAL_VALUE}
                    and (call_cell == direct_subject
                         or call_cell in canonical_sources
                         or (call_cell.page == INGESTION_VALUE
                             and book.latest_ref(INGESTION_VALUE, (
                                 call_cell.row[0], value_id,
                             )) == call_cell))):
                demands.append((call_cell, activity, tuple(ordered_inputs)))
    if len(demands) > 1:
        return unresolved("multiple_resource_demand_owners")
    demand = demands[0] if demands else None
    if value_id in nodes:
        try:
            subject = node_identity_cell(graph, value_id, book=book)
        except ValueError:
            return unresolved("missing_subject_identity")
    elif demand is not None:
        subject = demand[0]
    elif admitted_call:
        represented = {descriptor[1], descriptor[2], *tuple(descriptor[3]),
                       *(cell for _role, cell in descriptor[4])}
        subjects = tuple(parent for parent, _stage in book.edges_into(identity_cell)
                         if parent.page in {INGESTION_VALUE, CANONICAL_VALUE}
                         and parent not in represented)
        if len(subjects) != 1:
            return unresolved("missing_unique_subject_identity")
        subject = subjects[0]
    else:
        return unresolved("missing_call_value")
    callable_cell = book.latest_ref(CALLABLE_IDENTITY, (scope, value_id))
    if callable_cell is None:
        return unresolved("missing_callable_identity", subject)
    address = book.page(CALLABLE_IDENTITY).cell(
        callable_cell.row, callable_cell.column,
    )
    sources = tuple(parent for parent, _stage in book.edges_into(callable_cell))
    address_cells = tuple(parent for parent in sources
                          if parent.page == FUNCTION_ADDRESS
                          and book.page(FUNCTION_ADDRESS).cell(
                              parent.row, parent.column,
                          ) == address)
    if len(address_cells) != 1:
        return unresolved("missing_unique_function_address", subject)
    proofs = tuple(dict.fromkeys(parent for parent in sources
                                if parent != subject and parent not in address_cells))
    if not proofs:
        return unresolved("missing_callable_proof", subject)
    if admitted_call and (descriptor[1] != callable_cell
                          or descriptor[2] != address_cells[0]
                          or tuple(descriptor[3]) != proofs):
        return unresolved("call_identity_proof_disagreement", subject)

    if demand is not None:
        inputs = demand[2]
        if admitted_call and tuple(descriptor[4]) != inputs:
            return unresolved("call_input_proof_disagreement", subject)
    elif admitted_call:
        inputs = tuple(descriptor[4])
    else:
        parents = tuple(nodes.nodes[value_id].get("parents") or ())
        if any(int(parent) not in nodes for parent, _role in parents):
            return unresolved("missing_input_value", subject)
        try:
            inputs = tuple((str(role), node_identity_cell(graph, int(parent), book=book))
                           for parent, role in parents)
        except ValueError:
            return unresolved("missing_input_identity", subject)
    if sum(role == "operand" for role, _cell in inputs) != 1:
        return unresolved("missing_unique_receiver", subject)
    if any(not isinstance(cell, Ref) for _role, cell in inputs):
        return unresolved("missing_input_identity", subject)
    return CallableEvidence(
        scope, value_id, int(address), subject, callable_cell,
        address_cells[0], proofs, inputs,
        demand[1] if demand is not None else UnresolvedSourceDemand(
            scope, value_id, "control_scope", "missing_control_scope", subject,
        ),
    )
