"""A later unequal keyed store preserves the earlier exact mapping member."""

from types import SimpleNamespace

from src.compiler.concordance_declarations import (
    SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_MAPPING_GET_READ,
    SOURCE_MAPPING_GET_KEY_VERSION_DEMAND, REDUCTION,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, end_identity_book,
)
from tests.test_source_mapping_get_copy_state_demand import (
    SECOND_PUBLISH_STORE, _reduced,
)


SAME_KEY = SECOND_PUBLISH_STORE.replace(
    'module.metadata["other"]', 'module.metadata["identity_book"]',
)
UNKNOWN_KEY = SECOND_PUBLISH_STORE.replace(
    'def publish(module: Module):', 'def publish(module: Module, key: str):',
).replace(
    'module.metadata["other"]', 'module.metadata[key]',
).replace(
    '    publish(module)', '    publish(module, name)',
)
EQUAL_CROSS_CLASS_KEY = SECOND_PUBLISH_STORE.replace(
    '"identity_book"', '1',
).replace('"other"', 'True')


def test_later_equal_key_selects_latest_store():
    book, token, graph = _reduced(SAME_KEY)
    try:
        page = book.page(SOURCE_MAPPING_GET_READ)
        row = next(iter(page.rows()))
        fact = page.latest(row)
        assert len(fact) == 19
        assert fact[1].row[-1] == 9
        assert not CorrelationTable._source_mapping_get_read_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)


def test_python_equal_int_and_bool_keys_select_latest_store():
    book, token, _graph = _reduced(EQUAL_CROSS_CLASS_KEY)
    try:
        page = book.page(SOURCE_MAPPING_GET_READ)
        row = next(iter(page.rows()))
        fact = page.latest(row)
        assert len(fact) == 19
        assert fact[1].row[-1] == 9
    finally:
        end_identity_book(token)


def test_unknown_later_key_names_missing_equality_and_refuses_earlier_value():
    book, token, graph = _reduced(UNKNOWN_KEY)
    try:
        assert len(tuple(book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).rows())) == 1
        assert not tuple(book.page(SOURCE_MAPPING_GET_READ).rows())
        demands = book.page(SOURCE_MAPPING_GET_KEY_VERSION_DEMAND)
        row = next(iter(demands.rows()))
        fact = demands.latest(row)
        assert fact[0] == 'later_store_key_equality_unproved'
        assert not CorrelationTable._source_mapping_get_key_version_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)


def test_audit_rejects_swapped_skipped_key_cell():
    book, token, graph = _reduced(SECOND_PUBLISH_STORE)
    try:
        page = book.page(SOURCE_MAPPING_GET_READ)
        row = next(iter(page.rows()))
        original = book.latest_ref(SOURCE_MAPPING_GET_READ, row)
        fact = page.cell(original.row, original.column)
        selected_state, hops = fact[19]
        wrong_hop = (*hops[0][:6], fact[3])
        book.post(SOURCE_MAPPING_GET_READ, row,
                  (*fact[:19], (selected_state, (wrong_hop,))),
                  stage=REDUCTION,
                  provenance=Derived((original, fact[3])),
                  mode=Mode.REVISE)
        assert CorrelationTable._source_mapping_get_read_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)
