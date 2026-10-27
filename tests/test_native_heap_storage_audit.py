"""The descriptor, native storage view and resident cells are one contract."""
from types import SimpleNamespace
import pytest
from tests.test_fortran_heap_cell_concordance import _forwarded_pool
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, CorrelationTable


def _module(function, table, book):
    return SimpleNamespace(functions={function.name: function},
                           sequence_tables={function.name: table}, metadata={"identity_book": book})


def test_forwarded_native_pool_needs_no_duplicate_resident_labels():
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        assert CorrelationTable._table_member_findings(_module(function, table, book)) == []
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("poison", ["dtype", "element", "owner", "owner_role", "cell_role"])
def test_native_pool_audit_refuses_conflicting_resident_views(poison):
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        member = next(value for value in function.args if value.id == 14)
        if poison == "dtype":
            member.dtype = "float64"
        elif poison == "element":
            member.accounting = {"heap_element_dtype": "float64"}
        elif poison == "owner":
            member.accounting = {"heap_owner_value_id": 999}
        elif poison == "owner_role":
            member.accounting = {"heap_owner": True}
        else:
            member = next(value for value in function.args if value.id == 13)
            member.accounting = {"heap_element_dtype": "uint64", "heap_owner_value_id": 13}
        findings = CorrelationTable._table_member_findings(_module(function, table, book))
        assert any(item.kind == "native-heap-storage-disagreement" and item.value_id == member.id for item in findings)
    finally:
        end_identity_book(token)


def test_owned_unknown_element_layout_is_audit_visible():
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book, child_dtype="unknown")
        findings = CorrelationTable._table_member_findings(_module(function, table, book))
        assert any("resolved exact descriptor" in item.detail for item in findings)
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("poison", ["source", "missing_member", "element", "owner_link"])
def test_native_storage_tree_poison_is_audit_visible(monkeypatch, poison):
    from dataclasses import replace
    from src.compiler import structural_concordance
    from src.compiler.concordance_declarations import SSA_VALUE
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        view, ref = structural_concordance.storage_structure("sequence", table.owner, 10, book=book)
        children = dict(view.children)
        native = children["native_storage"]
        members = dict(native.children)
        if poison == "source":
            native = replace(native, source=book.latest_ref(SSA_VALUE, ("forwarded", 18)))
        elif poison == "missing_member":
            del members[14]
            native = replace(native, children=tuple(members.items()))
        else:
            cell = members[14]
            detail = dict(cell.children)
            if poison == "element":
                detail["element"] = replace(detail["element"], dtype="float64", physical_dtype="float64")
            else:
                detail["owner"] = replace(detail["owner"], value_id=999)
            members[14] = replace(cell, children=tuple(detail.items()))
            native = replace(native, children=tuple(members.items()))
        children["native_storage"] = native
        poisoned = replace(view, children=tuple(children.items()))
        monkeypatch.setattr(structural_concordance, "storage_structure", lambda *args, **kwargs: (poisoned, ref))
        findings = CorrelationTable._table_member_findings(_module(function, table, book))
        assert any(item.kind == "native-heap-storage-disagreement" for item in findings)
    finally:
        end_identity_book(token)


def test_one_heap_cell_cannot_belong_to_two_descriptor_contexts():
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSAChildTablePoolDescriptor, SSASequenceCapacityPolicy
    from src.compiler.concordance_declarations import SSA_VALUE, INGESTION
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        table.register(SSASequenceDescriptor(
            20, (20,), 21, 22, column_dtypes=("int64",),
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=23,
            child_table_pool=SSAChildTablePoolDescriptor(
                0, (14,), 25, 26, 27, column_dtypes=("uint64",), key_columns=(),
            ),
        ), sources=(book.latest_ref(SSA_VALUE, ("forwarded", 18)),), stage=INGESTION)
        findings = CorrelationTable._table_member_findings(_module(function, table, book))
        assert any(item.kind == "native-heap-storage-disagreement" and "one native identity" in item.detail for item in findings)
    finally:
        end_identity_book(token)
