"""Native exception admission is decided from ingested node identities.

No AST evaluation, source-coordinate matching, or source rewrite supplies
exception control after conversion. Unsupported source nodes retain a typed,
concorded refusal until a real exceptional-control graph is available.
"""
from .concordance_declarations import INGESTION_VALUE, NATIVE_EXCEPTION_CONTROL_GAP, REDUCTION, SOURCE_RESOURCE_SCOPE
from .identity_concordance import Derived, Mode, current_identity_book


class MissingResourceControlTopology(ValueError):
    pass


class MissingExceptionControlTopology(ValueError):
    def __init__(self, cell, kind):
        self.cell = cell
        super().__init__(f"native {kind} requires concorded exceptional control and cleanup edges")


def require_exception_control(graph, node):
    from ..common.tensors.topological_reducer import node_identity_cell
    if node not in graph.G:
        return
    book = current_identity_book()
    cell = node_identity_cell(graph, node)
    # This admission check runs while source statements are converted. At
    # this point their identities are the already-posted INGESTION_VALUEs.
    if cell.page != INGESTION_VALUE:
        return
    fact = book.page(INGESTION_VALUE).cell(cell.row, cell.column)
    if fact.type == "With":
        gap = book.post(NATIVE_EXCEPTION_CONTROL_GAP, (cell,),
                        (fact.type, ("resource_enter", "resource_exit", "cleanup")),
                        stage=REDUCTION, provenance=Derived((cell,)), mode=Mode.CONCORD)
        raise MissingResourceControlTopology(
            f"native resource scope requires concorded entry, exit and cleanup edges: {gap}")
    if fact.type == "Try" and book.latest_ref(SOURCE_RESOURCE_SCOPE, (cell,)) is not None:
        # Its source conversion declares entry/body/cleanup control arms;
        # it is admitted as a ResourceScopeBlock, not a Python exception call.
        return
    if fact.type not in {"Try", "TryStar", "Raise"}:
        return
    gap = book.post(NATIVE_EXCEPTION_CONTROL_GAP, (cell,),
                    (fact.type, ("exceptional_successor", "handler", "cleanup")),
                    stage=REDUCTION, provenance=Derived((cell,)), mode=Mode.CONCORD)
    raise MissingExceptionControlTopology(gap, fact.type)
