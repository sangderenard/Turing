from __future__ import annotations

import os
import sys

import pytest

from src.compiler.compressed_tiles import CompressedTable, DiskTile, DiskTileStore, DiskPackedIndex, DiskTileDirectory


def test_disk_directory_has_fixed_residency_and_exact_mutable_headers(tmp_path):
    store = DiskTileStore(tmp_path)
    record = store.write(b"same independently compressed data" * 50, 7)
    directory = DiskTileDirectory(store, tmp_path)
    for index in range(20000):
        directory.append(record, kind=index % 2)
    directory.flush()
    assert directory.resident_bytes() < 70 << 10
    for index in (0, 19000, 819, 19999, 1601):
        assert directory[index].decode() == record.decode()
        assert directory.kinds[index] == index % 2
        assert directory.row_lengths[index] == 7
    directory[819] = None
    directory.row_lengths[819] = 3
    directory.kinds[819] = 1
    directory.append(None, kind=1, rows=2)
    assert directory[19999] == record  # forces window eviction
    assert directory[819] is None
    assert directory.row_lengths[819] == 3 and directory.kinds[819] == 1
    assert directory[20000] is None and directory.row_lengths[20000] == 2
    assert directory.resident_bytes() < 70 << 10


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 17

    def __eq__(self, other):
        return type(other) is CollisionKey and self.value == other.value


class IdentityKey:
    """Hashes and compares by identity: no decoded copy equals it."""


def test_lookup_decodes_one_tile_not_whole_table():
    original = {("scope", i): ("repeated fact " * 40 + str(i), i) for i in range(4096)}
    table = CompressedTable(original.items(), tile_rows=32)
    before = table.stats()
    assert table[("scope", 2047)] == original[("scope", 2047)]
    after = table.stats()
    assert after.decoded_tiles - before.decoded_tiles == 1
    assert after.tiles == 128
    assert after.ratio > 10
    assert len(table) == len(original)
    assert all(not isinstance(value, dict) for value in table.__dict__.values())
    assert table.stats().resident_bytes == after.resident_bytes


def test_collision_and_absent_key_are_equality_checked():
    original = {CollisionKey(i): i * 2 for i in range(75)}
    table = CompressedTable(original.items(), tile_rows=8)
    for i in (0, 34, 74):
        assert table[CollisionKey(i)] == i * 2
    with pytest.raises(KeyError):
        table[CollisionKey(999)]
    before = table.stats().decoded_tiles
    with pytest.raises(KeyError):
        table["different hash"]
    assert table.stats().decoded_tiles == before


def test_operators_stream_without_mutable_view(monkeypatch):
    table = CompressedTable(((i, i * i) for i in range(1000)), tile_rows=25)
    def forbidden(*args, **kwargs):
        raise AssertionError("operator requested whole mutable table")
    monkeypatch.setattr(CompressedTable, "mutable", forbidden)
    before = table.stats().decoded_tiles
    selected = table.select(lambda k, v: k % 10 == 0)
    assert table.stats().decoded_tiles - before == 40
    mapped = selected.map_values(lambda k, v: v + 1)
    assert list(mapped.items()) == [(i, i * i + 1) for i in range(0, 1000, 10)]
    assert table.reduce(lambda result, k, v: result + v, 0) == sum(i*i for i in range(1000))
    assert list(table) == list(range(1000))


def test_pressure_target_reports_achievable_and_impossible_limits():
    table = CompressedTable(((i, "abcde" * 300) for i in range(4096)))
    initial = table.stats()
    fit = table.fit_budget(initial.resident_bytes + 2000, reserved_bytes=1000)
    assert fit.met
    assert fit.available_bytes == initial.resident_bytes + 1000
    failed = table.fit_budget(5000, reserved_bytes=5000)
    assert not failed.met and failed.target_ratio == float("inf")
    assert table.stats().resident_bytes <= initial.resident_bytes
    assert table[123] == "abcde" * 300


def test_incompressible_data_is_not_expanded_and_large_fact_is_reported():
    data = os.urandom(128 * 1024)
    table = CompressedTable((("one", data),), tile_rows=1)
    stats = table.stats()
    assert stats.payload_bytes <= stats.raw_bytes
    assert stats.largest_raw_tile > len(data)
    assert table["one"] == data
    assert not table.fit_budget(1024).met


def test_empty_and_configuration_validation():
    table = CompressedTable(())
    assert len(table) == 0 and list(table.items()) == []
    assert table.reduce(lambda *args: 0, 19) == 19
    assert table.fit_budget(table.stats().resident_bytes).met
    for kwargs in ({"tile_rows": 0}, {"tile_rows": True}, {"level": 10}):
        with pytest.raises(ValueError):
            CompressedTable((), **kwargs)


def test_point_update_reencodes_only_one_tile(monkeypatch):
    table = CompressedTable(((i, i*i) for i in range(256)), tile_rows=16)
    before = table.stats().decoded_tiles
    updated = table.with_value(100, -5)
    assert table.stats().decoded_tiles - before == 1
    assert updated[100] == -5 and table[100] == 10000
    assert sum(a is b for a, b in zip(table._tiles, updated._tiles)) == 15
    assert updated._hashes is table._hashes
    assert list(updated) == list(table)
    with pytest.raises(KeyError):
        table.with_value(999, 1)


def test_duplicate_and_identity_sensitive_keys_fail_closed():
    with pytest.raises(ValueError, match="duplicate keys"):
        CompressedTable(((1, "first"), (1, "second")), tile_rows=1)
    # A key with no row encoding is refused before its tile is written; one
    # that decodes unequal is refused by the round trip.
    with pytest.raises(TypeError, match="no row encoding"):
        CompressedTable(((object(), "identity-only key"),))
    with pytest.raises(TypeError, match="hash/equality"):
        CompressedTable(((IdentityKey(), "identity-only key"),))


def test_disk_tier_moves_tiles_and_indexes_and_lookup_remains_local(tmp_path, monkeypatch):
    table = CompressedTable(((i, "repeat " * 100 + str(i)) for i in range(8192)))
    before = table.stats()
    after = table.offload_to_disk(directory=str(tmp_path))
    assert after.disk_tiles == after.tiles and after.disk_bytes > 0
    assert after.resident_bytes < before.resident_bytes / 3
    assert isinstance(table._hashes, DiskPackedIndex)
    assert isinstance(table._ordinals, DiskPackedIndex)
    assert all(isinstance(tile, DiskTile) for tile in table._tiles)
    assert list(tmp_path.iterdir())
    decoded = table.stats().decoded_tiles
    assert table[4000] == "repeat " * 100 + "4000"
    assert table.stats().decoded_tiles - decoded == 1
    def forbidden(*args, **kwargs):
        raise AssertionError("disk-tier operator unfurled the table")
    monkeypatch.setattr(CompressedTable, "mutable", forbidden)
    selected = table.select(lambda k,v: k % 2000 == 0)
    assert list(selected) == [0,2000,4000,6000,8000]
    updated = table.with_value(4000, "new")
    assert updated[4000] == "new" and table[4000].endswith("4000")


def test_cold_table_and_packed_indices_page_their_physical_headers(tmp_path):
    table = CompressedTable(((i, "data" * 25 + str(i)) for i in range(8192)), tile_rows=8)
    table.offload_to_disk(directory=tmp_path)
    assert isinstance(table._tiles, DiskTileDirectory)
    assert isinstance(table._hashes._tiles, DiskTileDirectory)
    assert isinstance(table._ordinals._tiles, DiskTileDirectory)
    assert table._tiles.resident_bytes() < 70 << 10
    updated = table.with_value(4100, "isolated")
    assert isinstance(updated._tiles, DiskTileDirectory)
    assert updated._tiles is not table._tiles
    assert table[4100].endswith("4100") and updated[4100] == "isolated"
    assert updated._hashes is table._hashes
    assert table[0] == updated[0] and table[8191] == updated[8191]


def test_partial_disk_failure_preserves_every_value(tmp_path, monkeypatch):
    table = CompressedTable(((i, str(i)) for i in range(100)), tile_rows=10)
    store = DiskTileStore(str(tmp_path))
    original = store._append
    def failing(*arguments):
        if store.writes == 3:
            raise OSError("disk full")
        return original(*arguments)
    monkeypatch.setattr(store, "_append", failing)
    with pytest.raises(OSError, match="disk full"):
        table.offload_to_disk(store=store)
    assert table.stats().disk_tiles == 3
    assert list(table.items()) == [(i,str(i)) for i in range(100)]


def test_disk_shared_view_lifetime_and_corruption_detection(tmp_path):
    import gc
    from pathlib import Path
    table = CompressedTable(((i,i*i) for i in range(100)))
    table.offload_to_disk(directory=str(tmp_path))
    path = Path(table._disk_store.path)
    view = table.with_value(20, -1)
    del table
    gc.collect()
    assert path.exists() and view[21] == 441
    del view
    gc.collect()
    assert not path.exists()
    # Damage only this test-owned tile file, never a compiler/user artifact.
    store = DiskTileStore(str(tmp_path))
    tile = store.write(b"abc" * 1000, 1)
    with open(store.tile_path(tile), "r+b") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError):
        tile.decode()
