from src.compiler.identity_concordance import (
    CorrelationTable,
    argument_binding_history,
    begin_identity_book,
    end_identity_book,
    materializing_binding_kind,
)


def test_materializing_binding_resolution_preserves_history_and_settles_audit():
    from src.compiler.concordance_declarations import (
        ARGUMENT_BINDING, CANONICAL_VALUE, FRAME_BINDING, INGESTION,
        INGEST_SOURCE, ArgumentBindingFact,
    )
    from src.compiler.identity_concordance import Derived, Mode, Novel
    book, token = begin_identity_book()
    try:
        formal = book.post(
            CANONICAL_VALUE, (("helper", 0), 2), ("formal",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        source = book.post(
            CANONICAL_VALUE, (("caller", 0), 1), ("value",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        revised_source = book.post(
            CANONICAL_VALUE, (("caller", 0), 3), ("revised value",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        page = book.page("argument_binding")
        row = ("helper", "caller", 2, 7)
        book.post(
            ARGUMENT_BINDING, row, ArgumentBindingFact("caller_storage", 1),
            stage=FRAME_BINDING,
            provenance=Derived((formal, source)), mode=Mode.CONCORD,
        )
        book.post(
            ARGUMENT_BINDING, row, ArgumentBindingFact("caller_value", 1),
            stage=FRAME_BINDING,
            provenance=Derived((formal, revised_source)), mode=Mode.REVISE,
        )

        assert materializing_binding_kind(
            book, "helper", 2, 1, "caller_value", caller_symbol="caller"
        ) == "caller_storage"
        assert page.history(row) == (
            (0, ArgumentBindingFact("caller_storage", 1)),
            (1, ArgumentBindingFact("caller_value", 1)),
        )
        assert book.page("argument_binding_resolution").latest(
            ("helper", "caller", 2, 1)
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
    from src.compiler.concordance_declarations import (
        CANONICAL_VALUE, INGESTION, INGEST_SOURCE, PRECISION_IDENTITIES,
        SOURCE_PRECISION_OPERATOR,
    )
    from src.compiler.identity_concordance import Derived, Mode, Novel
    book, token = begin_identity_book()
    try:
        row = (("law", "section"), 27)
        value = book.post(
            CANONICAL_VALUE, row, ("value",), stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        operand = book.post(
            CANONICAL_VALUE, (("law", "section"), 19), ("operand",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        revised_operand = book.post(
            CANONICAL_VALUE, (("law", "section"), 20), ("revised operand",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        book.post(
            SOURCE_PRECISION_OPERATOR, row,
            ("Exp", 19, "Precision", 2), stage=PRECISION_IDENTITIES,
            provenance=Derived((value, operand)), mode=Mode.CONCORD,
        )
        book.post(
            SOURCE_PRECISION_OPERATOR, row,
            ("Log", 19, "Precision", 2), stage=PRECISION_IDENTITIES,
            provenance=Derived((value, revised_operand)), mode=Mode.REVISE,
        )
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


def test_argument_binding_identity_includes_caller_and_local_callsite():
    class Page:
        def __init__(self, rows=()):
            self._history = {row: ((0, fact),) for row, fact in rows}

        def rows(self):
            return tuple(self._history)

        def scope_rows(self, scope):
            return tuple(
                row for row in self._history if str(row[0]) == str(scope)
            )

        def history(self, row):
            return self._history.get(row, ())

        def latest(self, row):
            history = self.history(row)
            return None if not history else history[-1][1]

    binding_page = Page((
        (("helper", "caller_a", 2, 7), ("caller_storage", 1)),
        (("helper", "caller_b", 2, 7), ("caller_value", 1)),
    ))
    resolution_a = (
        "caller_storage", "caller_value", "materializing_binding_kind",
    )
    resolution_page = Page((
        (("helper", "caller_a", 2, 1), resolution_a),
    ))
    class Book:
        pages = {
            "argument_binding": binding_page,
            "argument_binding_resolution": resolution_page,
        }

        def page(self, name):
            return self.pages[name]

    book = Book()
    assert argument_binding_history(binding_page, "helper", 2) == (
        (("caller_a", 7), ("caller_storage", 1)),
        (("caller_b", 7), ("caller_value", 1)),
    )
    assert materializing_binding_kind(
        book, "helper", 2, 1, "caller_value",
        caller_symbol="caller_a",
    ) == "caller_storage"
    assert materializing_binding_kind(
        book, "helper", 2, 1, "caller_value",
        caller_symbol="caller_b",
    ) == "caller_value"
    module = type("Module", (), {
        "metadata": {"identity_book": book},
    })()
    assert CorrelationTable._binding_kind_findings(module) == []


def test_audit_reads_post_ssa_numeric_concordance_pages():
    from src.compiler.concordance_declarations import (
        CANONICAL_VALUE, EXACT_REGION_FEED_DTYPE, PHYSICAL_CALL_INPUT_ADAPTATION,
        PRECISION_CARRY, PRECISION_CHANNEL_SHAPE, SINGLE_INPUT_PHI_DESCRIPTOR,
        SINGLE_INPUT_PHI_SETTLEMENT, INGESTION, INGEST_SOURCE,
    )
    from src.compiler.identity_concordance import Derived, Mode, Novel
    book, token = begin_identity_book()
    try:
        def identity(scope, value_id):
            return book.post(
                CANONICAL_VALUE, (scope, value_id), ("value",),
                stage=INGESTION,
                provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
            )

        feed_row = ("root", 37)
        feed_cell = identity("root", 37)
        feed_other = identity("root", 38)
        book.post(
            EXACT_REGION_FEED_DTYPE, feed_row, ("bool",),
            stage=PHYSICAL_CALL_INPUT_ADAPTATION,
            provenance=Derived((feed_cell,)), mode=Mode.CONCORD,
        )
        book.post(
            EXACT_REGION_FEED_DTYPE, feed_row, ("float64",),
            stage=PHYSICAL_CALL_INPUT_ADAPTATION,
            provenance=Derived((feed_other,)), mode=Mode.REVISE,
        )
        channel_row = ("wide", 4)
        channel_value = identity("wide", 4)
        channel_operand = identity("wide", 5)
        book.post(
            PRECISION_CHANNEL_SHAPE, channel_row, ((2,), (2, 2), 2),
            stage=PRECISION_CARRY,
            provenance=Derived((channel_value, channel_operand)),
            mode=Mode.CONCORD,
        )
        book.post(
            PRECISION_CHANNEL_SHAPE, channel_row, ((), (2,), 2),
            stage=PRECISION_CARRY,
            provenance=Derived((channel_value, identity("wide", 6))),
            mode=Mode.REVISE,
        )
        phi_row = ("wide", 9)
        phi_result = identity("wide", 9)
        phi_source = identity("wide", 4)
        book.post(
            SINGLE_INPUT_PHI_DESCRIPTOR, phi_row, (4, "float64", (2,)),
            stage=SINGLE_INPUT_PHI_SETTLEMENT,
            provenance=Derived((phi_result, phi_source)), mode=Mode.CONCORD,
        )
        book.post(
            SINGLE_INPUT_PHI_DESCRIPTOR, phi_row, (5, "float64", ()),
            stage=SINGLE_INPUT_PHI_SETTLEMENT,
            provenance=Derived((phi_result, identity("wide", 5))),
            mode=Mode.REVISE,
        )
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
