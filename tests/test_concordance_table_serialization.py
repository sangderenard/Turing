import pickle
from dataclasses import replace

from src.compiler.concordance_declarations import INGESTION, SEQUENCE_DESCRIPTOR
from src.compiler.identity_concordance import CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book
from src.transmogrifier.ssa import (
    SSACallRecord, SSACallTable, SSARecordTable, SSASequenceDescriptor, SSASequenceTable, SSAStructTable,
    SSAUnionTable, SSAValueUnionTable, IRModule,
)


def test_all_table_views_restore_on_their_exact_concordance_owner():
    book, token = begin_identity_book()
    try:
        tables = [kind(owner="same", book=book) for kind in (
            SSARecordTable, SSASequenceTable, SSAValueUnionTable,
            SSAStructTable, SSAUnionTable, SSACallTable,
        )]
        original_owners = tuple(table.owner for table in tables)
        original_clock = book.clock[0]
        saved = pickle.dumps((book, tables))
    finally:
        end_identity_book(token)
    ambient, token = begin_identity_book()
    try:
        ambient_clock = ambient.clock[0]
        restored_book, restored_tables = pickle.loads(saved)
        assert all(table.book is restored_book for table in restored_tables)
        assert tuple(table.owner for table in restored_tables) == original_owners
        assert restored_book.clock[0] == original_clock
        assert ambient.clock[0] == ambient_clock
    finally:
        end_identity_book(token)


def test_nonempty_views_and_call_record_refs_remain_live_after_restore():
    book, token = begin_identity_book()
    try:
        descriptor = SSASequenceDescriptor(7, (8,), 9, 10, column_dtypes=("int64",))
        sequences = SSASequenceTable({7: descriptor}, owner="owner", book=book)
        calls = SSACallTable({"caller": (SSACallRecord(
            "caller", 11, None, "callee", "callee", argument_bindings=((8, 8),),
        ),)}, owner="calls", mutable=True, book=book)
        source = book.latest_ref(SEQUENCE_DESCRIPTOR, (sequences.owner, 7))
        call_source = calls["caller"][0].cell
        restored_book, restored_sequences, restored_calls = pickle.loads(pickle.dumps((book, sequences, calls)))
        assert restored_sequences.book is restored_calls.book is restored_book
        assert restored_calls.mutable
        assert restored_calls["caller"][0].cell == call_source
        list_book, call_list = pickle.loads(pickle.dumps((book, calls["caller"])))
        assert call_list._page.book is list_book
        assert call_list[0].cell == call_source
        assert restored_book.latest_ref(SEQUENCE_DESCRIPTOR, (sequences.owner, 7)) == source
        newer = replace(descriptor, column_dtypes=("uint64",))
        restored_book.post(SEQUENCE_DESCRIPTOR, (sequences.owner, 7), newer,
                           stage=INGESTION, provenance=Derived((source,)), mode=Mode.REVISE)
        assert restored_sequences.sequences[7] == newer
        assert sequences.sequences[7] == descriptor
    finally:
        end_identity_book(token)


def test_concordance_refuses_equal_descriptor_copy_on_another_book():
    book, token = begin_identity_book()
    try:
        descriptor = SSASequenceDescriptor(7, (8,), 9, 10, column_dtypes=("int64",))
        original = SSASequenceTable({7: descriptor}, owner="owner", book=book)
        module = IRModule({}, sequence_tables={"body": original}, metadata={"identity_book": book})
        assert not CorrelationTable._table_member_findings(module)
    finally:
        end_identity_book(token)
    other, token = begin_identity_book()
    try:
        copied = SSASequenceTable({7: descriptor}, owner="owner", book=other)
        module.sequence_tables["body"] = copied
        assert dict(copied.sequences) == dict(original.sequences)
        findings = CorrelationTable._table_member_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == "concordance-table-owner-disagreement"
    finally:
        end_identity_book(token)
