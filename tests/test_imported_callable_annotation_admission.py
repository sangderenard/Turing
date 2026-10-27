"""Imported annotation records use the ordinary declaration/identity path."""
import ast
import importlib.util
import sys
from types import SimpleNamespace
import pytest
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, CorrelationTable, Derived, Mode
from src.compiler.concordance_declarations import (
    CLASS_DECLARATION, CLASS_FIELD_DECLARATION, PARAMETER_ANNOTATION,
    FUNCTION_RETURN_ANNOTATION, STATIC_SYMBOL_DEFINITION, INGESTION,
)
from src.transmogrifier.graph.graph_express2 import (
    _source_ast_definition, _annotate_visual_source_owners,
    _source_owner_annotation_declaration_shells,
)


def _module(tmp_path, monkeypatch, name):
    path = tmp_path / (name + ".py")
    path.write_text("""from __future__ import annotations
class Record:
    optional_value: int | None = None
    def unused(self):
        raise RuntimeError('not executable source')
def read(record: "Record"):
    return record.optional_value
def result() -> "Record":
    return None
def scalar(value: int, flag: bool, values: list[float]) -> float:
    return float(value)
""")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _admit(module, function, identities):
    target = getattr(module, function)
    definition = _source_ast_definition(target)
    definition._python_source_identity = (module.__name__, target.__qualname__)
    _annotate_visual_source_owners(definition, module_name=module.__name__)
    return _source_owner_annotation_declaration_shells(
        target, source_definition=definition, record_identities=identities,
    )


def test_quoted_parameter_and_return_keep_exact_callable_class_field_causes(tmp_path, monkeypatch):
    module = _module(tmp_path, monkeypatch, "annotation_import_left")
    book, token = begin_identity_book()
    try:
        identity = module.Record.__module__ + "." + module.Record.__qualname__
        shells = _admit(module, "read", (identity,))
        assert shells == ()  # declaration facts are admitted; no second graph class identity
        _admit(module, "result", (identity,))
        for authority, row in ((PARAMETER_ANNOTATION, (module.__name__, "read", "record")),
                               (FUNCTION_RETURN_ANNOTATION, (module.__name__, "result"))):
            ref = book.latest_ref(authority, row)
            causes = {cell for cell, _ in book.edges_into(ref)}
            declaration = next(cell for cell in causes if cell.page == CLASS_DECLARATION)
            callable_ref = next(cell for cell in causes if cell.page == STATIC_SYMBOL_DEFINITION)
            assert declaration.row == (module.__name__, module.Record.__qualname__)
            assert callable_ref in {cell for cell, _ in book.edges_into(declaration)}
            field = book.latest_ref(CLASS_FIELD_DECLARATION, (*declaration.row, "optional_value"))
            assert declaration in {cell for cell, _ in book.edges_into(field)}
        assert CorrelationTable._annotation_class_admission_findings(SimpleNamespace(metadata={"identity_book": book})) == []
    finally:
        end_identity_book(token)


def test_same_named_imported_class_cannot_satisfy_another_modules_contract(tmp_path, monkeypatch):
    left = _module(tmp_path, monkeypatch, "annotation_import_left")
    right = _module(tmp_path, monkeypatch, "annotation_import_right")
    book, token = begin_identity_book()
    try:
        identity = left.Record.__module__ + "." + left.Record.__qualname__
        assert _admit(right, "read", (identity,)) == ()
        page = book.pages.get(CLASS_DECLARATION.name)
        assert page is None or not tuple(page.rows())
    finally:
        end_identity_book(token)


def test_audit_refuses_foreign_class_added_to_an_annotation(tmp_path, monkeypatch):
    left = _module(tmp_path, monkeypatch, "annotation_import_left")
    right = _module(tmp_path, monkeypatch, "annotation_import_right")
    book, token = begin_identity_book()
    try:
        for module in (left, right):
            _admit(module, "read", (module.Record.__module__ + "." + module.Record.__qualname__,))
        right_ref = book.latest_ref(CLASS_DECLARATION, (right.__name__, right.Record.__qualname__))
        row = (left.__name__, "read", "record")
        book.post(PARAMETER_ANNOTATION, row, book.page(PARAMETER_ANNOTATION).latest(row),
                  stage=INGESTION, provenance=Derived((right_ref,)), mode=Mode.CONCORD)
        findings = CorrelationTable._annotation_class_admission_findings(SimpleNamespace(metadata={"identity_book": book}))
        assert any(item.kind == "annotation-class-admission-disagreement" for item in findings)
    finally:
        end_identity_book(token)


def test_real_dt_imported_optional_field_has_declaration_before_planning(monkeypatch):
    from pathlib import Path
    from src.compiler.identity_concordance import current_identity_book
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from llvm_dt_system import dt_system_contract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    class Captured(Exception):
        pass
    def capture(graph):
        book = current_identity_book()
        fields = book.page(CLASS_FIELD_DECLARATION)
        rows = tuple(row for row in fields.rows()
                     if row[0] == "src.common.dt_system.dt_controller" and row[-1] == "energy_exchange_fraction")
        assert len(rows) == 1
        field = book.latest_ref(CLASS_FIELD_DECLARATION, rows[0])
        declarations = tuple(ref for ref, _ in book.edges_into(field) if ref.page == CLASS_DECLARATION)
        assert len(declarations) == 1 and declarations[0].row == rows[0][:2]
        assert CorrelationTable._annotation_class_admission_findings(SimpleNamespace(metadata={"identity_book": book})) == []
        raise Captured
    source = "from src.common.dt_system.dt_controller import _propose_dt_pen\ndef root(metrics, targets):\n    return _propose_dt_pen(metrics, targets, 1.0, None, 0.5)\n"
    with pytest.raises(Captured):
        lower_ast_source_to_ssa(source, "root", extraction_contract=dt_system_contract("root", (), 1, 1),
                                python_bindings={"AbstractTensor": AbstractTensor},
                                tensor_ssa_reference=c_backend_repository_ssa_reference(),
                                resolved_process_graph_sink=capture, progress=lambda _: None, progress_bars=False)


def test_builtin_annotations_do_not_require_record_declarations(tmp_path, monkeypatch):
    module = _module(tmp_path, monkeypatch, "annotation_import_scalar")
    book, token = begin_identity_book()
    try:
        identity = module.Record.__module__ + "." + module.Record.__qualname__
        assert _admit(module, "scalar", (identity,)) == ()
        page = book.pages.get(CLASS_DECLARATION.name)
        assert page is None or not tuple(page.rows())
        assert CorrelationTable._annotation_class_admission_findings(SimpleNamespace(metadata={"identity_book": book})) == []
    finally:
        end_identity_book(token)


def test_audit_refuses_field_declaration_without_its_admitted_class_cause(tmp_path, monkeypatch):
    module = _module(tmp_path, monkeypatch, "annotation_import_unlinked")
    book, token = begin_identity_book()
    try:
        _admit(module, "read", (module.Record.__module__ + "." + module.Record.__qualname__,))
        row = (module.__name__, "read", "record")
        annotation = book.latest_ref(PARAMETER_ANNOTATION, row)
        declaration = next(ref for ref, _ in book.edges_into(annotation) if ref.page == CLASS_DECLARATION)
        fact = book.page(CLASS_FIELD_DECLARATION).latest((*declaration.row, "optional_value"))
        book.post(CLASS_FIELD_DECLARATION, (*declaration.row, "unlinked"), fact,
                  stage=INGESTION, provenance=Derived((annotation,)), mode=Mode.CONCORD)
        findings = CorrelationTable._annotation_class_admission_findings(SimpleNamespace(metadata={"identity_book": book}))
        assert any("unlinked" in item.detail for item in findings)
    finally:
        end_identity_book(token)
