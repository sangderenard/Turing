import ast
import contextlib
import io

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    SOURCE_PYTHON_IDENTITY,
    SOURCE_SEQUENCE_MUTATION,
    SOURCE_VALUE_CLASS,
)
from src.compiler.identity_concordance import (
    _identity_bases_of,
    begin_identity_book,
    end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_nonloop_field_append_retains_exact_len_argument_lineage():
    source = """\
class Box:
    def __init__(self):
        self.items = []
        self.offsets = []

    def record(self):
        self.offsets.append(len(self.items))

def root():
    box = Box()
    box.record()
    return 0
"""
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)

        mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
        identity_page = book.page(SOURCE_PYTHON_IDENTITY)
        class_page = book.page(SOURCE_VALUE_CLASS)
        candidates = []
        for row in mutation_page.rows():
            fact = mutation_page.latest(row)
            if not (
                isinstance(fact, tuple)
                and len(fact) == 5
                and fact[1] == "append"
                and fact[4] == "sequence"
            ):
                continue
            argument_ids = tuple(map(int, fact[3]))
            if len(argument_ids) != 1:
                continue
            argument_id = argument_ids[0]
            len_receipts = []
            for identity_row in identity_page.rows():
                identity = identity_page.latest(identity_row)
                if not identity or identity[0] != "builtins.len":
                    continue
                identity_ref = book.latest_ref(
                    SOURCE_PYTHON_IDENTITY, identity_row,
                )
                if argument_id in {
                    base.row[1]
                    for base in _identity_bases_of(
                        book, identity_ref, argument_id,
                    )
                    if base.page.name == "canonical_value"
                }:
                    len_receipts.append(identity_row)
            if len_receipts:
                candidates.append((row, fact, argument_id, len_receipts))

        assert len(candidates) == 1
        mutation_row, fact, argument_id, len_receipts = candidates[0]
        assert fact[2] == "duplicates"
        assert len(len_receipts) == 1
        class_fact = class_page.latest((mutation_row[0], argument_id))
        assert class_fact is not None and class_fact[0] == "builtins.int"
    finally:
        end_identity_book(token)


def test_nonloop_local_alias_append_retains_exact_field_receiver():
    source = """\
class Box:
    def __init__(self):
        self.items = []
        self.offsets = []

    def record(self):
        offsets = self.offsets
        offsets.append(len(self.items))

def root():
    box = Box()
    box.record()
    return 0
"""
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)

        mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
        class_page = book.page(SOURCE_VALUE_CLASS)
        identity_page = book.page(SOURCE_PYTHON_IDENTITY)
        candidates = []
        for row in mutation_page.rows():
            fact = mutation_page.latest(row)
            if not (
                isinstance(fact, tuple)
                and len(fact) == 5
                and fact[1] == "append"
                and fact[4] == "sequence"
                and len(fact[3]) == 1
            ):
                continue
            argument_id = int(fact[3][0])
            receiver_id = int(fact[0])
            identity_refs = [
                book.latest_ref(SOURCE_PYTHON_IDENTITY, identity_row)
                for identity_row in identity_page.rows()
                if (identity := identity_page.latest(identity_row))
                and identity[0] == "builtins.len"
            ]
            len_arguments = [
                ref for ref in identity_refs
                if ref is not None and any(
                    base.page.name == "canonical_value"
                    and base.row[1] == argument_id
                    for base in _identity_bases_of(
                        book, ref, argument_id,
                    )
                )
            ]
            if len(len_arguments) != 1:
                continue
            class_fact = class_page.latest((row[0], argument_id))
            if class_fact is None or class_fact[0] != "builtins.int":
                continue
            candidates.append((row, fact, receiver_id))

        assert len(candidates) == 1
        mutation_row, fact, receiver_id = candidates[0]
        # The receiver is the resolved field-read identity, not the local
        # spelling `offsets` and not the raw Call node id.
        assert receiver_id != mutation_row[1]
        mutation_ref = book.latest_ref(
            SOURCE_SEQUENCE_MUTATION, mutation_row,
        )
        receiver_bases = _identity_bases_of(
            book, mutation_ref, receiver_id,
        )
        assert any(
            base.page == CANONICAL_VALUE
            for base in receiver_bases
        )
    finally:
        end_identity_book(token)


def test_nonloop_indexed_sequence_store_is_recorded_as_a_writer():
    source = """\
class Box:
    def __init__(self):
        self.offsets = [0]

    def record(self, value: int):
        self.offsets[0] = value

def root():
    box = Box()
    box.record(1)
    return 0
"""
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)

        mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
        setitem_rows = [
            (row, mutation_page.latest(row))
            for row in mutation_page.rows()
            if isinstance(mutation_page.latest(row), tuple)
            and mutation_page.latest(row)[1] == "setitem"
            and any(
                base.page == CANONICAL_VALUE
                for base in _identity_bases_of(
                    book,
                    book.latest_ref(SOURCE_SEQUENCE_MUTATION, row),
                    int(row[1]),
                )
            )
        ]
        assert len(setitem_rows) == 1
        _row, fact = setitem_rows[0]
        assert fact[2] == "duplicates"
        assert fact[3]  # index and stored value are both exact operands
        assert fact[4] == "sequence"
    finally:
        end_identity_book(token)


def test_nonloop_unknown_sequence_call_and_delete_are_receipts():
    source = """\
class Box:
    def __init__(self):
        self.offsets = [0]

    def record(self):
        self.offsets.copy()
        del self.offsets[0]

def root():
    box = Box()
    box.record()
    return 0
"""
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)

        mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
        operations = {
            mutation_page.latest(row)[1]
            for row in mutation_page.rows()
            if isinstance(mutation_page.latest(row), tuple)
        }
        assert "unclassified_call:copy" in operations
        assert "delitem" in operations
    finally:
        end_identity_book(token)


def test_sort_keyword_writer_matches_loop_mutation_receipt():
    source = """\
def root(key):
    values = [-1, 2]
    for index in range(2):
        values.sort(key=key)
    return 0
"""
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)

        mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
        sort_facts = [
            mutation_page.latest(row)
            for row in mutation_page.rows()
            if isinstance(mutation_page.latest(row), tuple)
            and mutation_page.latest(row)[1] == "sort"
        ]
        # The source call writer and loop-state writer must concord on the
        # exact keyword operand, which may itself be a function value.
        assert sort_facts
        assert all(len(fact[3]) == 1 for fact in sort_facts)
        # The same AST occurrence is present in ingestion and canonical
        # scopes, whose value IDs intentionally differ. Reduction completing
        # without a concordance conflict checks the same-scope writer join.
    finally:
        end_identity_book(token)
