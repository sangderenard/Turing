"""Resident compressed tables with a stronger-compression disk tile tier.

The index is two packed uint64 arrays (hash, insertion ordinal). Hashes locate
candidates; equality against the decoded key is ALWAYS authoritative, including
collisions. Iteration retains at most one decoded tile internally. Callers can
of course retain the values they request. A tile contains at most ``tile_rows``
entries; a single large fact is indivisible, so this is not a byte/RSS cap.

Tables are read-only snapshots. Writers must explicitly obtain a mutable view;
ordinary reads, select/map/reduce and budget recompression never do so. Pickle
is internal same-process storage, never an untrusted interchange format. A
codec supplied by the owner preserves registry references in concordance facts.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import io
import lzma
import os
import pickle
import shutil
import struct
import sys
import tempfile
import weakref
from threading import RLock
from typing import Any, Callable
import zlib


def _close_tile_directory(path):
    shutil.rmtree(path, ignore_errors=True)


class DiskTileStore:
    """Independently replaceable compressed artifacts owned by one tile set.

    Each record is independently compressed. No table/page pickle is rebuilt.
    LZMA2 uses a dictionary bounded to 1 MiB, rather than preset 9's large
    compression workspace precisely when RAM is scarce. Small records that
    compress better with zlib9 retain that encoding instead.
    """

    def __init__(self, directory=None):
        directory = directory or os.environ.get("TURING_IDENTITY_SPILL_DIR") or None
        self.path = tempfile.mkdtemp(prefix="identity_tiles_", dir=directory)
        self._lock = RLock()
        self._next_record = 0
        self.reads = self.writes = self.bytes_written = 0
        self._finalizer = weakref.finalize(self, _close_tile_directory, self.path)

    def tile_path(self, tile_or_offset):
        offset = (
            tile_or_offset.offset
            if isinstance(tile_or_offset, DiskTile)
            else int(tile_or_offset)
        )
        return os.path.join(self.path, f"{offset:016x}.tile")

    def write(self, data, rows, *, heavy=True):
        dictionary = min(1 << 20, max(1 << 16, len(data)))
        # Stronger than the RAM tier, with bounded encoder working memory.
        strong = lzma.compress(data, filters=[{
            "id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": dictionary,
        }]) if heavy else None
        fast = zlib.compress(data, 9)
        codec, payload = ("lzma", strong) if strong is not None and len(strong) < len(fast) else ("zlib", fast)
        if len(data) < len(payload):
            codec, payload = "raw", data
        return self._append(payload, len(data), rows, codec, zlib.crc32(data))

    def copy_strong_tile(self, tile):
        """Relocate an already strong-compressed tile without encoding it again.

        Validate the original decoded bytes before committing its header.
        This is an explicit single-tile copy operation; storage maintenance
        never invokes it for unaffected tiles. Workspace is bounded to one tile.
        """
        if tile.codec != "lzma":
            raise ValueError("only an already strong-compressed tile can be copied")
        tile.decode()  # Retain the compaction corruption/length check.
        with open(tile.store.tile_path(tile), "rb") as stream:
            payload = stream.read()
        with tile.store._lock:
            tile.store.reads += 1
        if len(payload) != tile.stored_bytes:
            raise OSError("truncated compressed tile")
        return self._append(payload, tile.raw_bytes, tile.rows, tile.codec, tile.checksum)

    def _append(self, payload, raw_bytes, rows, codec, checksum):
        with self._lock:
            offset = self._next_record
            self._next_record += 1
        path = self.tile_path(offset)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{offset:016x}.", suffix=".tmp", dir=self.path,
        )
        try:
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError("short write of compressed tile")
                stream.flush()
            os.replace(temporary, path)
        finally:
            try:
                os.remove(temporary)
            except OSError:
                pass
        with self._lock:
            self.writes += 1
            self.bytes_written += len(payload)
        return DiskTile(self, offset, len(payload), raw_bytes, rows, codec, checksum)

    def read(self, tile):
        with open(self.tile_path(tile), "rb") as stream:
            payload = stream.read()
        with self._lock:
            self.reads += 1
        if len(payload) != tile.stored_bytes:
            raise OSError("truncated compressed tile")
        if tile.codec == "lzma":
            data = lzma.decompress(payload, memlimit=16 << 20)
        elif tile.codec == "zlib":
            data = zlib.decompress(payload)
        else:
            data = payload
        if len(data) != tile.raw_bytes or zlib.crc32(data) != tile.checksum:
            raise ValueError("disk tile checksum/length disagrees with its header")
        return data

    def discard(self, tile):
        """Release exactly one superseded artifact without moving survivors."""
        if not isinstance(tile, DiskTile) or tile.store is not self:
            raise ValueError("cannot discard a tile owned by another store")
        try:
            os.remove(self.tile_path(tile))
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class DiskTile:
    store: DiskTileStore
    offset: int
    stored_bytes: int
    raw_bytes: int
    rows: int
    codec: str
    checksum: int

    def decode(self):
        return self.store.read(self)


class DiskTileDirectory:
    """The existing physical tile headers in a fixed-window disk index.

    No per-tile Python objects are retained. Fixed-width addresses permit
    random reads/writes without a resident offset table; one 64KiB window is
    cached. Kind/row columns are views of this same entry, not parallel ledgers.
    Payloads live as independent DiskTileStore artifacts with per-tile
    compression/CRC. This temporary index and its artifacts are not a restart
    format.
    """
    _entry = struct.Struct("<QQQIBIB6x")
    window_items = (64 << 10) // _entry.size
    _codecs = ("raw", "zlib", "lzma")

    def __init__(self, store, directory=None, *, window_items=None, exclusive=False):
        if window_items is not None:
            if type(window_items) is not int or not 0 < window_items <= type(self).window_items:
                raise ValueError("tile-directory window exceeds its fixed RAM allowance")
            self.window_items = window_items
        self.store = store
        self.exclusive = bool(exclusive)
        self._file = tempfile.TemporaryFile(dir=directory)
        self._count = 0
        self._window_index = -1
        self._window = bytearray()
        self._dirty = False
        self._persisted_extent = 0
        self.kinds = _DirectoryColumn(self, 6)
        self.row_lengths = _DirectoryColumn(self, 3)

    def __len__(self):
        return self._count

    def flush(self):
        if self._dirty:
            self._file.seek(self._window_index * self.window_items * self._entry.size)
            if self._file.write(self._window) != len(self._window):
                raise OSError("short write of tile directory")
            self._file.flush()
            self._persisted_extent = max(self._persisted_extent, self._file.tell())
            self._dirty = False

    def _position(self, index):
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        window, position = divmod(index, self.window_items)
        if window != self._window_index:
            self.flush()
            self._file.seek(window * self.window_items * self._entry.size)
            data = self._file.read(self.window_items * self._entry.size)
            if (window * self.window_items * self._entry.size < self._persisted_extent
                    and len(data) != self.window_items * self._entry.size):
                raise OSError("truncated tile directory")
            self._window = bytearray(data.ljust(self.window_items * self._entry.size, b"\0"))
            self._window_index = window
        return position * self._entry.size

    def _fields(self, index):
        position = self._position(index)
        return self._entry.unpack_from(self._window, position)

    def _set_fields(self, index, fields):
        position = self._position(index)
        self._entry.pack_into(self._window, position, *fields)
        self._dirty = True

    def __getitem__(self, index):
        offset, stored, raw, rows, codec, checksum, kind = self._fields(index)
        if codec == 0:
            return None
        return DiskTile(self.store, offset, stored, raw, rows, self._codecs[codec - 1], checksum)

    def __setitem__(self, index, record):
        old = self._fields(index)
        previous = self[index]
        if record is None:
            fields = (0, 0, 0, old[3], 0, 0, old[6])
        else:
            if not isinstance(record, DiskTile) or record.store is not self.store:
                raise ValueError("tile directory needs a record in its own payload store")
            fields = (record.offset, record.stored_bytes, record.raw_bytes, record.rows,
                      self._codecs.index(record.codec) + 1, record.checksum, old[6])
        self._set_fields(index, fields)
        if (
            self.exclusive and previous is not None
            and (record is None or previous.offset != record.offset)
        ):
            self.store.discard(previous)

    def append(self, record, *, kind=0, rows=0):
        index = self._count
        self._count += 1
        try:
            self._set_fields(index, (0, 0, 0, rows, 0, 0, kind))
            if record is not None:
                self[index] = record
        except BaseException:
            self._count -= 1
            raise

    def __iter__(self):
        for index in range(self._count):
            yield self[index]

    def resident_bytes(self):
        return (sys.getsizeof(self) + sys.getsizeof(self.__dict__)
                + sys.getsizeof(self._window) + sys.getsizeof(self._file)
                + sys.getsizeof(self.kinds) + sys.getsizeof(self.row_lengths))


class _DirectoryColumn:
    __slots__ = ("directory", "column")
    def __init__(self, directory, column):
        self.directory, self.column = directory, column

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, index):
        return self.directory._fields(index)[self.column]

    def __setitem__(self, index, value):
        fields = list(self.directory._fields(index))
        fields[self.column] = value
        self.directory._set_fields(index, fields)


class DiskPackedIndex:
    """Random-access uint64 index in independently compressed disk chunks.

    One last-read 8 KiB chunk is retained, not a reconstructed index. Bisect
    uses the same exact hash/ordinal values as the original packed arrays.
    """
    chunk_items = 1024

    def __init__(self, values, store):
        self._count = len(values)
        chunks = (self._count + self.chunk_items - 1) // self.chunk_items
        self._tiles = DiskTileDirectory(store, window_items=min(max(1, chunks),
                                                               DiskTileDirectory.window_items))
        for start in range(0, self._count, self.chunk_items):
            self._tiles.append(store.write(values[start:start + self.chunk_items].tobytes(),
                                           min(self.chunk_items, self._count - start)))
        self._tiles.flush()
        self._cache_index = -1
        self._cache = b""
        self._lock = RLock()

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        tile, position = divmod(index, self.chunk_items)
        with self._lock:
            if tile != self._cache_index:
                self._cache = self._tiles[tile].decode()
                self._cache_index = tile
            return int.from_bytes(self._cache[position * 8:(position + 1) * 8], sys.byteorder)

    def resident_bytes(self):
        return (sys.getsizeof(self) + sys.getsizeof(self.__dict__)
                + self._tiles.resident_bytes() + sys.getsizeof(self._cache)
                + sys.getsizeof(self._lock))

    def disk_bytes(self):
        return sum(t.stored_bytes for t in self._tiles)


@dataclass(frozen=True, slots=True)
class CompressedTile:
    payload: bytes
    raw_bytes: int
    rows: int
    compressed: bool

    @classmethod
    def encode(cls, data: bytes, rows: int, level: int) -> CompressedTile:
        compressed = zlib.compress(data, level)
        if len(compressed) < len(data):
            return cls(compressed, len(data), rows, True)
        return cls(data, len(data), rows, False)

    def decode(self) -> bytes:
        data = zlib.decompress(self.payload) if self.compressed else self.payload
        if len(data) != self.raw_bytes:
            raise ValueError("compressed tile length disagrees with its header")
        return data


@dataclass(frozen=True, slots=True)
class CompressionStats:
    rows: int
    tiles: int
    raw_bytes: int
    payload_bytes: int
    resident_bytes: int
    largest_raw_tile: int
    decoded_tiles: int
    disk_tiles: int = 0
    disk_bytes: int = 0

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(1, self.resident_bytes)


@dataclass(frozen=True, slots=True)
class BudgetFit:
    available_bytes: int
    resident_bytes: int
    target_ratio: float
    achieved_ratio: float
    met: bool


class CompressedTable(Mapping):
    """Ordered mapping with compressed values AND keys, accessed per tile.

    A lookup is O(log N + collision candidates * tile_rows), not a full scan.
    Missing hashes decode nothing. ``items``/``values`` scan O(N), decoding each
    tile once. No cache of decoded Python objects grows with table size.
    """

    def __init__(
        self, items: Iterable[tuple[Any, Any]], *, tile_rows: int = 128,
        level: int = 1, dispatch_table: dict | None = None,
        loader: Callable[[bytes], Any] = pickle.loads,
    ) -> None:
        if type(tile_rows) is not int or tile_rows <= 0:
            raise ValueError("tile_rows must be a positive integer")
        if type(level) is not int or not 0 <= level <= 9:
            raise ValueError("compression level must be an integer in [0, 9]")
        self.tile_rows = tile_rows
        self.level = level
        self._dispatch = dispatch_table
        self._loader = loader
        self._lock = RLock()
        self._decoded_tiles = 0
        self._tiles: list[CompressedTile] = []
        hashes = array("Q")
        chunk: list[tuple[Any, Any]] = []
        for key, value in items:
            hashes.append(hash(key) & ((1 << 64) - 1))
            chunk.append((key, value))
            if len(chunk) == tile_rows:
                self._append(chunk, level)
                chunk.clear()
        if chunk:
            self._append(chunk, level)
        self._count = len(hashes)
        # Only construction needs a sort workspace; no key/value graph is kept
        # in the index. Stable ordinals are the existing insertion identity.
        order = sorted(range(self._count), key=hashes.__getitem__)
        self._hashes = array("Q", (hashes[i] for i in order))
        self._ordinals = array("Q", order)
        # The constructor consumes unique mapping entries. Refuse duplicates,
        # including equal keys spanning tiles, rather than violate Mapping.
        start = 0
        while start < self._count:
            stop = bisect_right(self._hashes, self._hashes[start], lo=start)
            if stop - start > 1:
                seen = set()
                last_tile, entries = -1, ()
                for index in range(start, stop):
                    tile, row = divmod(self._ordinals[index], self.tile_rows)
                    if tile != last_tile:
                        entries = self._entries(tile)
                        last_tile = tile
                    key = entries[row][0]
                    if key in seen:
                        raise ValueError("compressed table received duplicate keys")
                    seen.add(key)
            start = stop

    def _append(self, chunk: list, level: int) -> None:
        self._tiles.append(self._pack(chunk, level))

    def _pack(self, chunk, level):
        stream = io.BytesIO()
        pickler = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
        if self._dispatch is not None:
            pickler.dispatch_table = self._dispatch
        pickler.dump(tuple(chunk))
        data = stream.getvalue()
        restored = self._loader(data)
        for (original, _), (key, _) in zip(chunk, restored):
            if hash(original) != hash(key) or original != key:
                raise TypeError("compressed key does not preserve hash/equality on round trip")
        return CompressedTile.encode(data, len(chunk), level)

    def _entries(self, tile: int) -> tuple:
        with self._lock:
            record = self._tiles[tile]
            self._decoded_tiles += 1
        entries = self._loader(record.decode())
        if len(entries) != record.rows:
            raise ValueError("compressed tile row count disagrees with its header")
        return entries

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, key: Any) -> Any:
        _tile, row, entries = self._lookup(key)
        return entries[row][1]

    def _lookup(self, key):
        code = hash(key) & ((1 << 64) - 1)
        begin = bisect_left(self._hashes, code)
        end = bisect_right(self._hashes, code)
        last_tile = -1
        entries = ()
        for index in range(begin, end):
            tile, row = divmod(self._ordinals[index], self.tile_rows)
            if tile != last_tile:
                entries = self._entries(tile)
                last_tile = tile
            candidate, value = entries[row]
            if candidate == key:
                return tile, row, entries
        raise KeyError(key)

    def with_value(self, key: Any, value: Any) -> CompressedTable:
        """Functional point update: encode one tile, share the untouched ones.

        Keys/order/index stay unchanged. Unsupported new keys refuse. No
        mutable whole-table view, no decode of unrelated tiles.
        """
        tile, row, entries = self._lookup(key)
        changed = list(entries)
        changed[row] = (changed[row][0], value)
        replacement = self._pack(changed, self.level)
        result = object.__new__(type(self))
        result.__dict__ = dict(self.__dict__)
        result._lock = RLock()
        result._decoded_tiles = 0
        with self._lock:
            if isinstance(self._tiles, DiskTileDirectory):
                # Functional update still owns a separate physical directory;
                # clone its scalar headers on disk, never list(N headers) in
                # RAM. Only the changed payload is decoded/re-encoded.
                directory = DiskTileDirectory(self._tiles.store,
                                               window_items=self._tiles.window_items)
                for record in self._tiles:
                    directory.append(record)
                directory[tile] = self._tiles.store.write(replacement.decode(), replacement.rows,
                                                         heavy=False)
                directory.flush()
                result._tiles = directory
            else:
                result._tiles = list(self._tiles)
                result._tiles[tile] = replacement
        return result

    def __iter__(self) -> Iterator:
        for tile in range(len(self._tiles)):
            for key, _value in self._entries(tile):
                yield key

    def items(self) -> Iterator:
        for tile in range(len(self._tiles)):
            yield from self._entries(tile)

    def values(self) -> Iterator:
        for _key, value in self.items():
            yield value

    def select(self, predicate: Callable[[Any, Any], bool]) -> CompressedTable:
        return self._derived((k, v) for k, v in self.items() if predicate(k, v))

    def map_values(self, operator: Callable[[Any, Any], Any]) -> CompressedTable:
        return self._derived((k, operator(k, v)) for k, v in self.items())

    def reduce(self, operator: Callable[[Any, Any, Any], Any], initial: Any) -> Any:
        result = initial
        for key, value in self.items():
            result = operator(result, key, value)
        return result

    def _derived(self, items: Iterable) -> CompressedTable:
        return type(self)(items, tile_rows=self.tile_rows, level=self.level,
                          dispatch_table=self._dispatch, loader=self._loader)

    def mutable(self) -> dict:
        """Explicit promotion for a writer; never used by read operators."""
        return dict(self.items())

    def stats(self) -> CompressionStats:
        with self._lock:
            resident = (sys.getsizeof(self) + sys.getsizeof(self.__dict__)
                        + (self._tiles.resident_bytes() if isinstance(self._tiles, DiskTileDirectory)
                           else sys.getsizeof(self._tiles)) + sys.getsizeof(self._lock))
            resident += sum(index.resident_bytes() if isinstance(index, DiskPackedIndex)
                            else sys.getsizeof(index) for index in (self._hashes, self._ordinals))
            if not isinstance(self._tiles, DiskTileDirectory):
                resident += sum(sys.getsizeof(t) + (sys.getsizeof(t.payload)
                                                   if isinstance(t, CompressedTile) else 0)
                                for t in self._tiles)
            disk_bytes = sum(t.stored_bytes for t in self._tiles if isinstance(t, DiskTile))
            disk_bytes += sum(index.disk_bytes() for index in (self._hashes, self._ordinals)
                              if isinstance(index, DiskPackedIndex))
            return CompressionStats(
                self._count, len(self._tiles), sum(t.raw_bytes for t in self._tiles),
                sum(len(t.payload) for t in self._tiles if isinstance(t, CompressedTile)), resident,
                max((t.raw_bytes for t in self._tiles), default=0), self._decoded_tiles,
                sum(isinstance(t, DiskTile) for t in self._tiles), disk_bytes,
            )

    def offload_to_disk(self, *, store=None, target_bytes=0, directory=None):
        """Move resident tiles AND packed indexes, retaining the same accessors.

        Each tile is committed only after its file write succeeds. A failure
        may leave a mixed RAM/disk table, never a lost tile. Functional copies
        keep shared disk records alive until the last reader releases them.
        """
        if type(target_bytes) is not int or target_bytes < 0:
            raise ValueError("target_bytes must be a nonnegative integer")
        if not self._tiles or self.stats().resident_bytes <= target_bytes:
            return self.stats()
        store = store or self.__dict__.get("_disk_store") or DiskTileStore(directory)
        self._disk_store = store
        with self._lock:
            resident = self.stats().resident_bytes
            for index, tile in enumerate(self._tiles):
                if resident <= target_bytes:
                    break
                if isinstance(tile, DiskTile):
                    continue
                disk = store.write(tile.decode(), tile.rows)
                self._tiles[index] = disk
                resident -= sys.getsizeof(tile) + sys.getsizeof(tile.payload) - sys.getsizeof(disk)
            if resident > target_bytes:
                for name in ("_hashes", "_ordinals"):
                    index = getattr(self, name)
                    if not isinstance(index, DiskPackedIndex) and len(index):
                        setattr(self, name, DiskPackedIndex(index, store))
            if (not isinstance(self._tiles, DiskTileDirectory)
                    and all(isinstance(tile, DiskTile) and tile.store is store for tile in self._tiles)):
                directory = DiskTileDirectory(store, directory,
                    window_items=min(max(1, len(self._tiles)), DiskTileDirectory.window_items))
                for tile in self._tiles:
                    directory.append(tile)
                directory.flush()
                self._tiles = directory
        return self.stats()

    def fit_budget(self, limit_bytes: int, *, reserved_bytes: int = 0) -> BudgetFit:
        """Target the actual headroom, returning failure if data won't fit.

        Recompress one encoded tile at a time at increasing effort, keeping a
        replacement only when smaller. Never retain a second whole-table copy.
        ``resident_bytes`` measures owned buffers/index/header, not interpreter
        allocator slack or memory retained by consumers/codec/registry.
        """
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError("limit_bytes must be a nonnegative integer")
        if type(reserved_bytes) is not int or reserved_bytes < 0:
            raise ValueError("reserved_bytes must be a nonnegative integer")
        available = max(0, limit_bytes - reserved_bytes)
        stats = self.stats()
        for level in (3, 6, 9):
            if stats.resident_bytes <= available:
                break
            if level <= self.level:
                continue
            with self._lock:
                for index, tile in enumerate(self._tiles):
                    if isinstance(tile, DiskTile):
                        continue
                    candidate = CompressedTile.encode(tile.decode(), tile.rows, level)
                    if len(candidate.payload) < len(tile.payload):
                        self._tiles[index] = candidate
                self.level = level
            stats = self.stats()
        return BudgetFit(
            available, stats.resident_bytes,
            stats.raw_bytes / available if available else float("inf"),
            stats.ratio, stats.resident_bytes <= available,
        )
