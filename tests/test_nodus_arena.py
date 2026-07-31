import struct

import pytest

from src.common.tensors.accelerator_backends import nodus_arena as na


@pytest.fixture(scope="module")
def arena():
    try:
        return na.arena()
    except na.NodusArenaUnavailable as error:
        pytest.skip(str(error))


def test_the_transport_needs_no_array_library():
    """The point of this bridge is that tensor transport carries no
    dependency of its own -- if it ever imports numpy or torch, the reason it
    exists has been lost."""

    source = (
        na.__file__.replace(".pyc", ".py")
        if na.__file__
        else None
    )
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import numpy" not in text
    assert "import torch" not in text


def test_create_describe_and_destroy(arena):
    handle = arena.create(na.F64, (3, 4))
    try:
        desc = arena.describe(handle)
        assert desc.rank == 2
        assert [desc.shape[i] for i in range(desc.rank)] == [3, 4]
        assert desc.element_count == 12
        assert desc.element_bytes == 8
        assert desc.total_bytes == 96
    finally:
        arena.destroy(handle)


def test_values_round_trip_exactly(arena):
    values = [i * 1.5 for i in range(12)]
    handle = arena.from_values(na.F64, (3, 4), values)
    try:
        assert arena.to_values(handle) == tuple(values)
    finally:
        arena.destroy(handle)


def test_a_mapped_window_writes_through_to_the_arena(arena):
    """A view is the arena's own storage, not a copy of it."""

    handle = arena.from_values(na.F64, (4,), [0.0, 0.0, 0.0, 0.0])
    try:
        with arena.map(handle) as view:
            block = view.as_ctypes()
            block[0] = 3.25
            block[3] = -7.5
        assert arena.to_values(handle) == (3.25, 0.0, 0.0, -7.5)
    finally:
        arena.destroy(handle)


def test_zero_clears_the_payload(arena):
    handle = arena.from_values(na.F64, (4,), [1.0, 2.0, 3.0, 4.0])
    try:
        arena.zero(handle)
        assert arena.to_values(handle) == (0.0, 0.0, 0.0, 0.0)
    finally:
        arena.destroy(handle)


def test_partial_write_leaves_the_rest_alone(arena):
    handle = arena.from_values(na.F64, (4,), [1.0, 2.0, 3.0, 4.0])
    try:
        arena.write_bytes(handle, struct.pack("<d", 9.0), byte_offset=8)
        assert arena.to_values(handle) == (1.0, 9.0, 3.0, 4.0)
    finally:
        arena.destroy(handle)


def test_a_write_past_the_payload_is_clamped_not_overrun(arena):
    handle = arena.create(na.F64, (2,))
    try:
        written = arena.write_bytes(handle, b"\x00" * 64)
        assert written == 16
    finally:
        arena.destroy(handle)


def test_integer_tensors_carry_their_own_element_size(arena):
    handle = arena.from_values(na.I32, (5,), [1, -2, 3, -4, 5])
    try:
        desc = arena.describe(handle)
        assert desc.element_bytes == 4
        assert desc.total_bytes == 20
        assert arena.to_values(handle) == (1, -2, 3, -4, 5)
    finally:
        arena.destroy(handle)


def test_rank_above_the_abi_maximum_is_refused(arena):
    with pytest.raises(na.NodusArenaError):
        arena.create(na.F64, (1,) * (na.MAX_RANK + 1))


def test_allocation_reports_where_the_tensor_lives(arena):
    """A compiled kernel addresses arena memory directly rather than through
    a mapping, so the address and extent have to be answerable."""

    handle = arena.create(na.F64, (8,))
    try:
        address, size, _bucket = arena.allocation(handle)
        assert address != 0
        assert size >= 64
    finally:
        arena.destroy(handle)


def test_freed_spans_return_to_the_arena(arena):
    """Leases must come back on destroy; an arena that only grows is the
    failure this allocator exists to avoid."""

    before = arena.stats().active_leases
    handles = [arena.create(na.F64, (1024,)) for _ in range(4)]
    assert arena.stats().active_leases == before + 4
    for handle in handles:
        arena.destroy(handle)
    assert arena.stats().active_leases == before


def test_the_arena_reserves_virtual_address_space(arena):
    stats = arena.stats()
    assert stats.reserve_bytes > 0
    assert stats.span_nodes_capacity > 0
