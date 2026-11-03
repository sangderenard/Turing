"""Convert source evaluation and control ownership at the ingestion boundary.

This module is the AST writer. Its products contain only exact source cells,
typed activity points and graph edges. Readers live in control_activity.py.
"""
import ast

from .concordance_declarations import (
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_BINDING,
    SOURCE_ACTIVITY_CALL, SOURCE_ACTIVITY_CHILD, REDUCTION, SOURCE_ACTIVITY_ARM_RELATION,
    NATIVE_EXCEPTION_CONTROL_GAP, SOURCE_SPAN, SOURCE_TERMINAL_RETURN,
)
from .identity_concordance import Derived, Mode, current_identity_book


class UnadmittedSourceControlTopology(ValueError):
    """Source control reached reduction without an admitted graph contract."""

    def __init__(self, kind, gap):
        self.kind = str(kind)
        self.gap = gap
        super().__init__(
            f"source {self.kind} reached reduction without admitted control topology: {gap}"
        )


def ingest_control_activity(graph, definition, scope):
    from ..transmogrifier.graph.graph_express2 import post_source_span
    book = current_identity_book()

    def source(node):
        cell = post_source_span(None, node)
        if cell is None:
            raise ValueError('source control activity has no ingested occurrence')
        return cell

    def point(node, phase, owners):
        occurrence = source(node)
        return book.post(SOURCE_ACTIVITY_NODE, (scope, occurrence, phase),
                         (type(node).__name__, tuple(owners)), stage=REDUCTION,
                         provenance=Derived((occurrence,)), mode=Mode.CONCORD)

    def edge(before, after, relation='flow', proofs=()):
        return book.post(SOURCE_ACTIVITY_EDGE, (scope, before, after, relation), True,
                         stage=REDUCTION, provenance=Derived((before, after, *proofs)), mode=Mode.CONCORD)

    def sequence(nodes, before, after, owners, relation='flow'):
        tail = before
        reaches_successor = True
        for node in nodes:
            first, last = visit(node, owners)
            if not reaches_successor:
                continue
            edge(tail, first, relation)
            relation = 'flow'
            # Program: `if book is None: return 0; with book.borrowed(...)`.
            # The Return value reaches Return.exit, but that exit cannot flow
            # through If.exit into the following With. Converted resource
            # Try/With scopes keep their existing cleanup route; their owner
            # cells identify them exactly, so this narrow rule leaves them.
            protected = any(
                book.page(SOURCE_SPAN).cell(owner.row, owner.column).kind
                in {'Try', 'TryStar', 'With', 'AsyncWith'}
                for owner, _arm in owners
            )
            if isinstance(node, ast.Return) and not protected:
                occurrence = source(node)
                book.post(
                    SOURCE_TERMINAL_RETURN, (scope, occurrence),
                    (first, last, tuple(owners)), stage=REDUCTION,
                    provenance=Derived((occurrence, first, last,
                                        *(owner for owner, _arm in owners))),
                    mode=Mode.CONCORD,
                )
                reaches_successor = False
            else:
                tail = last
        if reaches_successor:
            edge(tail, after, relation)

    def visit(node, owners=(), root=False):
        enter, leave = point(node, 'enter', owners), point(node, 'exit', owners)
        occurrence = source(node)
        if isinstance(node, ast.Call):
            book.post(SOURCE_ACTIVITY_CALL, (scope, occurrence),
                      (len(node.args), tuple(keyword.arg for keyword in node.keywords)), stage=REDUCTION,
                      provenance=Derived((enter, leave)), mode=Mode.CONCORD)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            tail = enter
            for index, member in enumerate(node.elts):
                first, last = visit(member, owners)
                book.post(SOURCE_ACTIVITY_CHILD, (scope, enter, index), (first, last), stage=REDUCTION,
                          provenance=Derived((enter, first, last)), mode=Mode.CONCORD)
                edge(tail, first); tail = last
            edge(tail, leave)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if root:
                body = (node.body,) if isinstance(node, ast.Lambda) else node.body
                arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                             *((node.args.vararg,) if node.args.vararg is not None else ()),
                             *((node.args.kwarg,) if node.args.kwarg is not None else ()))
                sequence((*arguments, *body), enter, leave, owners)
            else:
                # Defining a function evaluates defaults/decorators, not its
                # worker body. That body gets its own function control graph.
                children = (*node.args.defaults, *(x for x in node.args.kw_defaults if x is not None),
                            *getattr(node, 'decorator_list', ()))
                sequence(children, enter, leave, owners)
        elif isinstance(node, (ast.If, ast.IfExp)):
            first, last = visit(node.test, owners)
            select = point(node, 'select', owners)
            edge(enter, first); edge(last, select)
            for arm in ('body', 'orelse'):
                children = getattr(node, arm)
                if isinstance(node, ast.IfExp):
                    children = (children,)
                sequence(children, select, leave, (*owners, (occurrence, arm)), arm)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            header = node.test if isinstance(node, ast.While) else node.iter
            header_arm = 'test' if isinstance(node, ast.While) else 'iter'
            first, last = visit(header, (*owners, (occurrence, header_arm)))
            select = point(node, 'select', owners)
            edge(enter, first); edge(last, select)
            body_enter = point(node, 'body_enter', (*owners, (occurrence, 'body')))
            body_exit = point(node, 'body_exit', (*owners, (occurrence, 'body')))
            edge(select, body_enter, 'body')
            children = tuple(node.body)
            if not isinstance(node, ast.While):
                children = (node.target, *children)
            sequence(children, body_enter, body_exit, (*owners, (occurrence, 'body')))
            # A loop's recurrence is explicit, not flattened into an authored
            # order between different workers or iterations. Scheduling the
            # static region excludes this declared backedge only.
            edge(body_exit, first if isinstance(node, ast.While) else select, 'repeat')
            sequence(node.orelse, select, leave, (*owners, (occurrence, 'orelse')), 'exhausted')
            edge(body_exit, leave, 'iteration_extent')
        elif isinstance(node, ast.BoolOp):
            tail = enter
            arm = 'body' if isinstance(node.op, ast.And) else 'orelse'
            for index, value in enumerate(node.values):
                first, last = visit(value, owners if index == 0 else (*owners, (occurrence, arm)))
                edge(tail, first, 'flow' if index == 0 else arm)
                edge(last, leave, 'short_circuit' if index + 1 < len(node.values) else 'flow')
                tail = last
        elif isinstance(node, ast.Try) and getattr(node, '_turing_resource_contract', None) is not None:
            if node.handlers or node.orelse:
                raise ValueError('resource control conversion cannot supply exception handlers')
            body_enter = point(node, 'body_enter', (*owners, (occurrence, 'body')))
            body_exit = point(node, 'body_exit', (*owners, (occurrence, 'body')))
            cleanup_enter = point(node, 'cleanup_enter', (*owners, (occurrence, 'finalbody')))
            edge(enter, body_enter, 'resource_entry')
            sequence(node.body, body_enter, body_exit, (*owners, (occurrence, 'body')))
            arm_relation = book.post(SOURCE_ACTIVITY_ARM_RELATION,
                (scope, occurrence, 'body', 'finalbody'), 'resource_cleanup', stage=REDUCTION,
                provenance=Derived((node._turing_resource_contract, body_exit, cleanup_enter)), mode=Mode.CONCORD)
            edge(body_exit, cleanup_enter, 'resource_cleanup', (arm_relation,))
            sequence(node.finalbody, cleanup_enter, leave, (*owners, (occurrence, 'finalbody')))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # This pass records source activity only.  Resource admission is
            # decided after expression reduction has written each context
            # call's exact callable and input identities to the concordance.
            contexts = tuple(
                child
                for item in node.items
                for child in (
                    item.context_expr,
                    *((item.optional_vars,) if item.optional_vars is not None else ()),
                )
            )
            body_enter = point(node, 'body_enter', (*owners, (occurrence, 'body')))
            sequence(contexts, enter, body_enter, owners)
            sequence(node.body, body_enter, leave, (*owners, (occurrence, 'body')))
        elif isinstance(node, (ast.Try, ast.TryStar, ast.Raise)):
            requirements = ('exceptional_successor', 'handler', 'cleanup')
            gap = book.post(
                NATIVE_EXCEPTION_CONTROL_GAP, (source(node),),
                (type(node).__name__, requirements), stage=REDUCTION,
                provenance=Derived((enter, leave)), mode=Mode.CONCORD,
            )
            raise UnadmittedSourceControlTopology(type(node).__name__, gap)
        else:
            if isinstance(node, ast.Call):
                children = (node.func, *node.args, *(keyword.value for keyword in node.keywords))
            elif isinstance(node, ast.Assign):
                children = (node.value, *node.targets)
            elif isinstance(node, ast.AnnAssign):
                children = (*(() if node.value is None else (node.value,)), node.target)
            elif isinstance(node, ast.Dict):
                children = tuple(child for key, value in zip(node.keys, node.values)
                                 for child in (key, value) if child is not None)
            else:
                children = tuple(child for child in ast.iter_child_nodes(node)
                                 if isinstance(child, (ast.expr, ast.stmt, ast.comprehension)))
            sequence(children, enter, leave, owners)
        return enter, leave

    visit(definition, root=True)
    # Join existing node registrations through their exact source-span
    # provenance. Some registered operators already carry lowered payloads,
    # so expr_obj is not an authoritative ingestion reader even here.
    from .concordance_declarations import INGESTION_VALUE
    for row in tuple(book.page(INGESTION_VALUE).rows()):
        identity = book.latest_ref(INGESTION_VALUE, row)
        occurrences = tuple(cell for cell, _stage in book.edges_into(identity)
                            if cell.page == SOURCE_SPAN)
        if len(occurrences) == 1:
            bind_activity(identity, scope, occurrences[0])


def bind_activity(identity, scope, occurrence):
    book = current_identity_book()
    enter = book.latest_ref(SOURCE_ACTIVITY_NODE, (scope, occurrence, 'enter'))
    leave = book.latest_ref(SOURCE_ACTIVITY_NODE, (scope, occurrence, 'exit'))
    if enter is None or leave is None:
        return None
    return book.post(SOURCE_ACTIVITY_BINDING, (scope, identity), (enter, leave),
                     stage=REDUCTION, provenance=Derived((identity, enter, leave)), mode=Mode.CONCORD)
