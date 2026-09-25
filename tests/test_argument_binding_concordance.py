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


def test_audit_reads_post_ssa_numeric_concordance_pages():
    book, token = begin_identity_book()
    try:
        feed = book.page("exact_region_feed_dtype")
        feed.set(("feed", "root", 37), 0, ("bool",))
        feed.set(("feed", "root", 37), 1, ("float64",))
        channel = book.page("precision_channel_shape_concordance")
        channel.set(("wide", 4), 0, ((2,), (2, 2), 2))
        channel.set(("wide", 4), 1, ((), (2,), 2))
        phi = book.page("single_input_phi_descriptor_concordance")
        phi.set(("wide", 9), 0, (4, "float64", (2,)))
        phi.set(("wide", 9), 1, (5, "float64", ()))
        module = type("Module", (), {
            "metadata": {"identity_book": book},
        })()

        findings = CorrelationTable._post_ssa_numeric_identity_findings(
            module
        )

        assert {finding.kind for finding in findings} == {
            "exact-region-feed-dtype-disagreement",
            "precision-channel-shape-disagreement",
            "single-input-phi-descriptor-disagreement",
        }
        assert {(finding.function, finding.value_id) for finding in findings} == {
            ("root", 37), ("wide", 4), ("wide", 9),
        }
    finally:
        end_identity_book(token)
