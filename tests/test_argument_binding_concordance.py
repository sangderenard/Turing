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
