"""A reverse helper metadata edge cites the changed matching forward edge."""

from src.compiler import identity_concordance as ic
from tests.test_linked_byte_slices import (
    test_linked_split_has_three_typed_fresh_byte_slices as exercise_linked_slice_fixture,
)


def test_reverse_append_slice_metadata_cites_new_forward_revision(
    tmp_path, monkeypatch,
):
    original = ic.record_shape_transformation
    observed = []

    def checked(source_scope, source_id, target_scope, target_id, **kwargs):
        book = ic.current_identity_book()
        candidate = None
        if (
            "append_slice" in str(source_scope)
            and kwargs["operation"] == "ssa_metadata_transport"
        ):
            suffix = (str(kwargs["stage"]), str(kwargs["operation"]),
                      str(kwargs["role"]))
            reverse_row = (target_scope, target_id, source_scope, source_id,
                           *suffix)
            forward_row = (source_scope, source_id, target_scope, target_id,
                           *suffix)
            previous = book.latest_ref(ic.SHAPE_EDGE_PAGE, reverse_row)
            forward = book.latest_ref(ic.SHAPE_EDGE_PAGE, forward_row)
            new = (ic.shape_transformation_state(kwargs["source_state"]),
                   ic.shape_transformation_state(kwargs["target_state"]))
            if (
                previous is not None and forward is not None
                and book.page(ic.SHAPE_EDGE_PAGE).cell(
                    previous.row, previous.column,
                ) != new
                and book.stamp_of(forward) > book.stamp_of(previous)
                and book.page(ic.SHAPE_EDGE_PAGE).cell(
                    forward.row, forward.column,
                ) == (new[1], new[0])
            ):
                candidate = forward
        result = original(source_scope, source_id, target_scope, target_id,
                          **kwargs)
        if candidate is not None:
            revised = book.latest_ref(ic.SHAPE_EDGE_PAGE, reverse_row)
            assert candidate in {
                source for source, _stage in book.edges_into(revised)
            }
            observed.append((source_scope, source_id))
        return result

    monkeypatch.setattr(ic, "record_shape_transformation", checked)
    exercise_linked_slice_fixture(tmp_path)
    assert observed
