from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    end_identity_book,
    materializing_binding_kind,
)


def test_materializing_binding_resolution_preserves_history_and_settles_audit():
    book, token = begin_identity_book()
    try:
        page = book.page("argument_binding")
        row = ("helper", 2, "binding")
        page.set(row, 10, ("caller_storage", 1))
        page.set(row, 11, ("caller_value", 1))

        assert materializing_binding_kind(
            book, "helper", 2, 1, "caller_value"
        ) == "caller_storage"
        assert page.history(row) == (
            (10, ("caller_storage", 1)),
            (11, ("caller_value", 1)),
        )
        assert book.page("argument_binding_resolution").latest(
            ("helper", 2, 1)
        ) == (
            "caller_storage", "caller_value",
            "materializing_binding_kind",
        )

        module = type("Module", (), {
            "metadata": {"identity_book": book},
        })()
        assert CorrelationTable._binding_kind_findings(module) == []
    finally:
        end_identity_book(token)


def test_precision_operator_concordance_reports_changed_source_identity():
    book, token = begin_identity_book()
    try:
        page = book.page("source_precision_operator_concordance")
        row = (("law", "section"), 27)
        page.set(row, 0, ("Exp", 19, "Precision", 2))
        page.set(row, 1, ("Log", 19, "Precision", 2))
        module = type("Module", (), {
            "metadata": {"identity_book": book},
        })()

        findings = CorrelationTable._source_precision_operator_findings(module)

        assert len(findings) == 1
        assert findings[0].kind == "source-precision-operator-disagreement"
        assert findings[0].function == str(("law", "section"))
        assert findings[0].value_id == 27
    finally:
        end_identity_book(token)
