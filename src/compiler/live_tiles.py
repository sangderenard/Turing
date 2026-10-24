"""Writable compiler storage over the existing compressed RAM/disk tiles.

NetworkX remains the graph implementation; SSA/control objects remain their
original objects. Only their container/attribute ownership changes. References
are encoded by the arena, not recursively pickled into independent copies.
An escaping container view is a stable handle, never a stale decoded payload.
"""
from __future__ import annotations

from collections import OrderedDict
from array import array
from collections.abc import MutableMapping, MutableSequence, MutableSet, Set
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import is_dataclass
from dataclasses import fields as dataclass_fields
import gc
import io
import operator
import os
import pickle
import sys
import tempfile
from threading import RLock
import time
import weakref

from .compressed_tiles import CompressedTile, DiskTile, DiskTileStore, DiskTileDirectory


_ACTIVE_ARENA = ContextVar("compiler_live_tile_arena", default=None)
_ACTIVE_REGISTRAR = ContextVar("compiler_live_root_registrar", default=None)
_MEMO_SOURCES = object()
_PLAIN_ATOM_TYPES = frozenset({type(None), str, bytes, int, float, complex, bool})
_PLAIN_VALUE_TYPES = frozenset({type(None), str, bytes, int, float, complex,
                               bool, dict, list, tuple, set, frozenset})


def _value_fact(kind):
    """The immutable value vocabulary of the SAME concordance tile codec."""
    return (kind.__module__ in {"src.compiler.concordance_declarations",
                                "src.compiler.identity_concordance"}
            and kind.__name__ not in {"Page", "Stage", "Transform", "Reason"}
            and is_dataclass(kind) and kind.__dataclass_params__.frozen)


@dataclass(frozen=True)
class LiveTileStats:
    records: int
    tiles: int
    hot_tiles: int
    hot_peak: int
    resident_encoded_bytes: int
    disk_bytes: int
    disk_tiles: int
    compressions: int
    reloads: int
    identity_headers: int
    owned_header_bytes: int = 0
    catalogued_headers: int = 0
    decoded_bytes: int = 0
    decoded_peak_bytes: int = 0
    pressure_level: int = 0
    routing_reloads: int = 0
    payload_reloads: int = 0
    retained_index_tiles: int = 0
    payload_hot_tiles: int = 0
    route_hits: int = 0
    route_misses: int = 0
    retained_index_bytes: int = 0
    payload_decoded_bytes: int = 0
    payload_hot_peak: int = 0
    payload_cache_hits: int = 0
    payload_evictions: int = 0
    parallel_compression_jobs: int = 0
    compression_seconds: float = 0.0
    compression_workers: int = 1


@dataclass(slots=True)
class _DecodedSize:
    """Size measurements for the currently decoded rows, discarded on eviction."""
    total: int
    rows: array


_EMPTY_ROUTE = (1 << 64) - 1


class _RouteIndex:
    """Compact resident hash-to-bucket accelerator for an authoritative trie."""

    def __init__(self):
        self.codes = array("Q", [0]) * 8
        self.handles = array("Q", [_EMPTY_ROUTE]) * 8
        self.count = 0

    @staticmethod
    def _slot(code, mask):
        return (code ^ (code >> 33) ^ (code >> 17)) & mask

    def get(self, code):
        mask = len(self.handles) - 1
        slot = self._slot(code, mask)
        while self.handles[slot] != _EMPTY_ROUTE:
            if self.codes[slot] == code:
                return self.handles[slot]
            slot = (slot + 1) & mask
        return None

    def set(self, code, handle):
        if (self.count + 1) * 10 >= len(self.handles) * 7:
            self._resize(len(self.handles) * 2)
        self._insert(code, handle)

    def _insert(self, code, handle):
        mask = len(self.handles) - 1
        slot = self._slot(code, mask)
        while self.handles[slot] != _EMPTY_ROUTE:
            if self.codes[slot] == code:
                self.handles[slot] = handle
                return
            slot = (slot + 1) & mask
        self.codes[slot], self.handles[slot] = code, handle
        self.count += 1

    def _resize(self, capacity):
        old_codes, old_handles = self.codes, self.handles
        self.codes = array("Q", [0]) * capacity
        self.handles = array("Q", [_EMPTY_ROUTE]) * capacity
        self.count = 0
        for code, handle in zip(old_codes, old_handles):
            if handle != _EMPTY_ROUTE:
                self._insert(code, handle)

    def resident_bytes(self):
        return sys.getsizeof(self) + sys.getsizeof(self.codes) + sys.getsizeof(self.handles)


class LiveTileArena:
    """One shared decoded arena, coarse first and bounded only under pressure.

    A tile contains at most tile_rows storage records. Maps split before a
    bucket exceeds bucket_rows; sequences use independently indexed chunks.
    Opaque semantic leaves stay exact and are counted, never silently copied.
    Level zero keeps every touched tile decoded. Pressure progressively enables
    the configured bounds; those still do not bound allocator slack or a giant
    indivisible leaf.
    """
    def __init__(self, *, tile_rows=64, hot_tiles=8, bucket_rows=64, directory=None,
                 decoded_budget_bytes=128 << 20, compression_workers=None):
        if min(tile_rows, hot_tiles, bucket_rows) <= 0:
            raise ValueError("live tile bounds must be positive")
        self.tile_rows, self.hot_limit, self.bucket_rows = tile_rows, hot_tiles, bucket_rows
        self.base_payload_limit = hot_tiles
        self.payload_limit = hot_tiles
        self.directory = directory
        if decoded_budget_bytes <= 0:
            raise ValueError("decoded tile budget must be positive")
        self.base_decoded_budget_bytes = decoded_budget_bytes
        self.decoded_budget_bytes = decoded_budget_bytes
        self.compression_workers = max(
            1, int(compression_workers if compression_workers is not None
                   else min(4, os.cpu_count() or 1)))
        self._decoded_sizes = {}
        self._decoded_bytes = self._decoded_peak = 0
        self._retained_decoded_bytes = self._payload_decoded_bytes = 0
        self._payload_hot_count = 0
        self._payload_hot_peak = 0
        self._tiles = []
        self._hot = OrderedDict()
        self._dirty = set()
        self._count = 0
        self._refs = weakref.WeakValueDictionary()
        self._reference_count = 0
        self._classes = []
        self._class_ids = {}
        # Identity descriptors contain only scalar handles, never payload
        # objects. Separate allocation pages prevent a data tile's decoder
        # from recursively trying to decode its own identity catalogue.
        self._cursors = {}
        self._page_kinds = array("B")
        self._row_lengths = array("I")
        self._leaf_refs = {}
        self._route_indexes = {}
        self.route_hits = self.route_misses = 0
        self._lock = RLock()
        self._store = None
        self._live_disk_bytes = 0
        self.disk_mode = False
        # 0: fully decoded fast path; 1: LRU tail backed by disk;
        # 2: bounded decoded front with compressed-RAM backing;
        # 3: one decoded payload tile plus retained search metadata, disk backing.
        self.pressure_level = 0
        self.compressions = self.reloads = self.hot_peak = 0
        self.routing_reloads = self.payload_reloads = 0
        self.payload_cache_hits = self.payload_evictions = 0
        self.parallel_compression_jobs = 0
        self.compression_seconds = 0.0
        self.categories = set()
        self.pressure_probe = None
        self._operations = 0
        self._inside_probe = False

    def _pressure(self):
        self._operations += 1
        if self._operations % 128 == 0 and self.pressure_probe is not None and not self._inside_probe:
            self._inside_probe = True
            try:
                self.pressure_probe("live storage access/allocation")
            finally:
                self._inside_probe = False

    def _dump(self, records):
        stream = io.BytesIO()
        owner = self
        class Writer(pickle.Pickler):
            def persistent_id(self, value):
                # The one-expression forced-disk source repro made 17.8M
                # visits here. Ordinary ints/strings/container records need
                # no identity lookup; checking ABC-backed tiled classes first
                # spent most encoding time dispatching __instancecheck__.
                kind = type(value)
                if kind in _PLAIN_VALUE_TYPES:
                    return None
                if kind is _StorageLink:
                    return None
                if isinstance(value, (TiledDict, TiledList, TiledTuple, TiledSet)):
                    if value._arena is owner:
                        return ("live", value._ref)
                if isinstance(value, TiledObject):
                    state = object.__getattribute__(value, "__dict__")
                    if state.get("_tile_arena") is owner:
                        return ("live", state["_tile_ref"])
                # Concordance facts are immutable value records, already
                # serialized by value in the existing book tile codec. Only
                # their registry Page/Stage/Transform/Reason leaves are exact
                # shared identities. Do not pin millions of fact payloads.
                if _value_fact(kind):
                    return None
                # Non-container semantic leaves must preserve actual identity:
                # AST objects, registries, tensor descriptors, imported bindings.
                owner._leaf_refs[id(value)] = value
                return ("leaf", id(value))
        Writer(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(records)
        return stream.getvalue()

    def _load(self, data):
        owner = self
        class Reader(pickle.Unpickler):
            def persistent_load(self, key):
                if key[0] == "live":
                    return owner.resolve_reference(key[1])
                if key[0] == "leaf":
                    return owner._leaf_refs[key[1]]
                raise ValueError("unknown compiler tile reference")
        return Reader(io.BytesIO(data)).load()

    def _replace_record(self, index, record):
        directory = isinstance(self._tiles, DiskTileDirectory)
        previous = self._tiles[index]
        self._tiles[index] = record
        if (
            not directory and isinstance(previous, DiskTile)
            and (not isinstance(record, DiskTile) or previous.offset != record.offset)
        ):
            previous.store.discard(previous)
        return previous

    def _encode_batch(self, indexes, *, disk=None):
        """Serialize, compress concurrently, and publish bounded eviction batches."""
        if disk is None:
            disk = self.disk_mode or self.pressure_level in (1, 3)
        candidates = [
            index for index in dict.fromkeys(indexes)
            if index in self._dirty or self._tiles[index] is None
        ]
        if not candidates:
            return
        store = self._disk_store() if disk else None

        def encode(job):
            index, data, rows = job
            record = (
                store.write(data, rows, heavy=False)
                if store is not None
                else CompressedTile.encode(data, rows, 1)
            )
            return index, record

        started = time.perf_counter()
        workers = min(self.compression_workers, len(candidates))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="compiler-compress",
        ) as pool:
            for start in range(0, len(candidates), workers):
                # Serialization discovers semantic leaf identities and remains
                # ordered under the arena lock. Retain at most one raw tile per
                # worker, then compress that whole eviction batch together.
                jobs = []
                for index in candidates[start:start + workers]:
                    records = self._hot[index]
                    jobs.append((index, self._dump(records), len(records)))
                for index, record in pool.map(encode, jobs):
                    previous = self._replace_record(index, record)
                    if isinstance(record, DiskTile):
                        self._live_disk_bytes += record.stored_bytes - (
                            previous.stored_bytes if isinstance(previous, DiskTile) else 0
                        )
                    self._dirty.discard(index)
                    self.compressions += 1
        self.parallel_compression_jobs += len(candidates)
        self.compression_seconds += time.perf_counter() - started

    def _encode(self, index, *, disk=None):
        self._encode_batch((index,), disk=disk)

    def _disk_store(self):
        if self._store is None:
            self._store = DiskTileStore(self.directory)
        return self._store

    def reference(self, value):
        handle = self.allocate(self._descriptor(value), header=True)
        self._remember(handle, value)
        self._reference_count += 1
        return handle

    def _remember(self, handle, value):
        if isinstance(value, TiledTuple):
            # CPython tuples cannot be weak-referenced. A collectable anchor
            # cycle retains the same tuple while it is externally alive,
            # without turning the catalogue into a strong owner of every row.
            anchor = _TupleAnchor(value)
            value._anchor = anchor
            self._refs[handle] = anchor
        else:
            self._refs[handle] = value

    def _descriptor(self, value):
        if isinstance(value, TiledDict):
            return ("dict", value._root,
                    -1 if value._order is None else value._order._ref,
                    value._count, value._version)
        if isinstance(value, TiledList):
            return ("list", -1 if value._first is None else value._first,
                    -1 if value._chunks is None else value._chunks._ref,
                    value._count)
        if isinstance(value, TiledTuple):
            return ("tuple", value._sequence._ref)
        if isinstance(value, TiledSet):
            return ("set", value._members._ref)
        kind = type(value)
        class_id = self._class_ids.get(kind)
        if class_id is None:
            class_id = len(self._classes)
            self._classes.append(kind)
            self._class_ids[kind] = class_id
        state = object.__getattribute__(value, "__dict__")
        return ("object", class_id, state["_tile_fields"]._ref)

    def save_header(self, value):
        self.write(value._ref, self._descriptor(value))

    def resolve_reference(self, handle):
        with self._lock:
            value = self._refs.get(handle)
            if value is not None:
                return value.value if isinstance(value, _TupleAnchor) else value
            descriptor = self.read(handle)
            tag = descriptor[0]
            if tag == "dict":
                _, root, order, count, version = descriptor
                value = dict.__new__(TiledDict)
                value._arena, value._root = self, root
                value._order, value._count, value._version, value._ref = None, count, version, handle
                self._remember(handle, value)
                if order != -1:
                    value._order = self.resolve_reference(order)
            elif tag == "list":
                _, first, chunks, count = descriptor
                value = list.__new__(TiledList)
                value._arena, value._first = self, None if first == -1 else first
                value._chunks, value._count, value._ref = None, count, handle
                self._remember(handle, value)
                if chunks != -1:
                    value._chunks = self.resolve_reference(chunks)
            elif tag == "tuple":
                value = tuple.__new__(TiledTuple)
                value._sequence = self.resolve_reference(descriptor[1])
                value._arena, value._ref = self, handle
                self._remember(handle, value)
            elif tag == "set":
                value = object.__new__(TiledSet)
                value._arena, value._ref = self, handle
                self._remember(handle, value)
                value._members = self.resolve_reference(descriptor[1])
            elif tag == "object":
                kind = self._classes[descriptor[1]]
                value = object.__new__(kind)
                state = object.__getattribute__(value, "__dict__")
                state.update(_tile_arena=self, _tile_ref=handle)
                self._remember(handle, value)
                state["_tile_fields"] = self.resolve_reference(descriptor[2])
            else:
                raise ValueError("invalid compiler identity descriptor")
            return value

    def _trim(self, except_index=None):
        if self.pressure_level == 0:
            return
        hot_count = self._payload_hot_count
        decoded_bytes = self._payload_decoded_bytes
        victims = []
        for index in self._hot:
            if index == except_index or self._page_kinds[index]:
                continue
            if (
                hot_count <= self.payload_limit
                and decoded_bytes <= self.decoded_budget_bytes
            ):
                break
            victims.append(index)
            hot_count -= 1
            decoded_bytes -= self._decoded_sizes[index].total
        if not victims:
            return
        self._encode_batch(victims)
        for index in victims:
            self._drop_hot(index)

    def _page(self, index):
        if index not in self._hot:
            record = self._tiles[index]
            self._hot[index] = ([None] * self._row_lengths[index] if record is None
                                else self._load(record.decode()))
            self._measure_page(index)
            self.reloads += 1
            if self._page_kinds[index]:
                self.routing_reloads += 1
            else:
                self.payload_reloads += 1
        elif not self._page_kinds[index]:
            self.payload_cache_hits += 1
        self._hot.move_to_end(index)
        self._trim(except_index=index)
        self.hot_peak = max(self.hot_peak, len(self._hot))
        return self._hot[index]

    def _set_size(self, index, size):
        current = self._decoded_sizes[index]
        delta = size - current.total
        self._decoded_bytes += delta
        if self._page_kinds[index]:
            self._retained_decoded_bytes += delta
        else:
            self._payload_decoded_bytes += delta
        current.total = size
        self._decoded_peak = max(self._decoded_peak, self._decoded_bytes)

    def _measure_page(self, index):
        rows = self._hot[index]
        sizes = array("Q", (_owned_row_bytes(value) for value in rows))
        size = sys.getsizeof(rows) + sum(sizes) + sys.getsizeof(sizes)
        previous = self._decoded_sizes.get(index)
        delta = size - (0 if previous is None else previous.total)
        self._decoded_bytes += delta
        if self._page_kinds[index]:
            self._retained_decoded_bytes += delta
        else:
            self._payload_decoded_bytes += delta
            if previous is None:
                self._payload_hot_count += 1
                self._payload_hot_peak = max(self._payload_hot_peak, self._payload_hot_count)
        self._decoded_sizes[index] = _DecodedSize(size, sizes)
        self._decoded_peak = max(self._decoded_peak, self._decoded_bytes)

    def _drop_hot(self, index):
        del self._hot[index]
        size = self._decoded_sizes.pop(index).total
        self._decoded_bytes -= size
        if self._page_kinds[index]:
            self._retained_decoded_bytes -= size
        else:
            self._payload_decoded_bytes -= size
            self._payload_hot_count -= 1
            if self.pressure_level > 0:
                self.payload_evictions += 1

    def cached_route(self, owner_handle, code):
        if self.pressure_level == 0:
            return None
        with self._lock:
            index = self._route_indexes.get(owner_handle)
            handle = None if index is None else index.get(code)
            if handle is None:
                self.route_misses += 1
            else:
                self.route_hits += 1
            return handle

    def remember_route(self, owner_handle, code, handle):
        if self.pressure_level > 0:
            with self._lock:
                self._route_indexes.setdefault(owner_handle, _RouteIndex()).set(code, handle)

    def invalidate_routes(self, owner_handle):
        with self._lock:
            self._route_indexes.pop(owner_handle, None)

    def allocate(self, value, *, header=False):
        self._pressure()
        with self._lock:
            cursor = self._cursors.get(header)
            if cursor is None or cursor[1] == self.tile_rows:
                index = len(self._tiles)
                position = 0
                if isinstance(self._tiles, DiskTileDirectory):
                    self._tiles.append(None, kind=int(header))
                else:
                    self._tiles.append(None)
                    self._page_kinds.append(int(header))
                    self._row_lengths.append(0)
                self._hot[index] = []
                self._measure_page(index)
            else:
                index, position = cursor
            rows = self._page(index)
            measured = self._decoded_sizes[index]
            before = sys.getsizeof(rows) + sys.getsizeof(measured.rows)
            size = _owned_row_bytes(value)
            rows.append(value)
            measured.rows.append(size)
            self._set_size(index, measured.total + size + sys.getsizeof(rows)
                           + sys.getsizeof(measured.rows) - before)
            self._row_lengths[index] += 1
            self._dirty.add(index)
            handle = index * self.tile_rows + position
            self._cursors[header] = (index, position + 1)
            self._count += 1
            return handle

    def read(self, handle):
        self._pressure()
        with self._lock:
            index, position = divmod(handle, self.tile_rows)
            value = self._page(index)[position]
            # An individually oversized tile need not remain cached once its
            # requested row has been selected. The caller holds only that row.
            self._trim()
            return value

    def write(self, handle, value):
        self._pressure()
        with self._lock:
            index, position = divmod(handle, self.tile_rows)
            rows = self._page(index)
            measured = self._decoded_sizes[index]
            before = measured.rows[position]
            size = _owned_row_bytes(value)
            rows[position] = value
            # The old writer re-traversed the entire previous bounded row
            # just to subtract the same measurement made at its last write.
            # Keep that scalar only while this row is decoded; eviction drops
            # it with the payload. This is cache accounting, not graph identity.
            measured.rows[position] = size
            self._set_size(index, measured.total + size - before)
            self._dirty.add(index)
            self._trim()

    def manage(self, value, memo=None, *, progress=None):
        # Exact scalar atoms need neither container dispatch nor retention in
        # the conversion memo. Subclasses still follow the ordinary path.
        if type(value) in _PLAIN_ATOM_TYPES:
            return memo.get(id(value), value) if memo is not None else value
        if isinstance(value, (TiledDict, TiledList, TiledTuple, TiledSet)):
            return value
        if isinstance(value, TiledObject):
            self.attach(value, memo=memo)
            return value
        memo = {} if memo is None else memo
        memo.setdefault(_MEMO_SOURCES, []).append(value)
        if id(value) in memo:
            return memo[id(value)]
        if type(value) is dict:
            result = TiledDict(self)
            memo[id(value)] = result
            items = value.items()
            if progress is not None:
                from .shell_telemetry import compile_iter
                key, label = progress
                items = compile_iter(
                    items, key, label=label, total=len(value),
                )
            result._adopt_items(
                [(key, self.manage(member, memo)) for key, member in items]
            )
            return result
        if type(value) is tuple and len(value) <= self.bucket_rows:
            # Immutable small records need no mutable container header. Their
            # object edges still use the arena's exact-reference codec.
            members = tuple(self.manage(member, memo) for member in value)
            return value if all(a is b for a, b in zip(value, members)) else members
        if type(value) in {list, tuple}:
            result = TiledList(self)
            view = TiledTuple(result) if type(value) is tuple else result
            memo[id(value)] = view
            members = value
            if progress is not None:
                from .shell_telemetry import compile_iter
                key, label = progress
                members = compile_iter(
                    members, key, label=label, total=len(value),
                )
            result._adopt_members([self.manage(member, memo) for member in members])
            return view
        if type(value) is set:
            result = TiledSet(self)
            memo[id(value)] = result
            members = value
            if progress is not None:
                from .shell_telemetry import compile_iter
                key, label = progress
                members = compile_iter(
                    members, key, label=label, total=len(value),
                )
            for member in members:
                result.add(self.manage(member, memo))
            return result
        if _value_fact(type(value)):
            # Derived(cells=(...)) was one giant decoded pickle row, counted
            # as just56 bytes. Page/Stage/Transform/Reason identities remain
            # exact leaves; immutable fact fields use the same paged sequence
            # protocol as ordinary data, retaining the original fact class.
            result = object.__new__(type(value))
            memo[id(value)] = result
            changed = False
            for field in dataclass_fields(value):
                previous = getattr(value, field.name)
                member = self.manage(previous, memo)
                changed |= member is not previous
                object.__setattr__(result, field.name, member)
            if not changed:
                memo[id(value)] = value
                return value
            # Frozen Ref's cached hash remains valid: field values are equal.
            state = getattr(value, "__dict__", {})
            for name, member in state.items():
                if name not in result.__dict__:
                    object.__setattr__(result, name, member)
            return result
        return value

    def attach(self, value, *, memo=None):
        state = object.__getattribute__(value, "__dict__")
        if "_tile_arena" in state:
            return value
        # Install the handle before following edges: self/recursive references
        # belong to the SAME object, not a reconstructed compiler graph.
        fields = TiledDict(self)
        previous = dict(state)
        state.clear()
        state.update(_tile_arena=self, _tile_fields=fields)
        state["_tile_ref"] = self.reference(value)
        try:
            # A dispatch plan and its generated class share node_locations.
            # Independent field adoptions made that ONE mutable map into two
            # equal but divergent writers. Follow all owned fields with the
            # caller's same adoption memo, including nested compiler objects.
            memo = {} if memo is None else memo
            for name, member in previous.items():
                fields._set(name, self.manage(member, memo))
        except BaseException:
            state.clear()
            state.update(previous)
            raise
        self.categories.add(type(value).__module__ + "." + type(value).__name__)
        return value

    def flush(self, *, disk=None, preserve_routing=False):
        with self._lock:
            indexes = tuple(self._hot)
            self._encode_batch(indexes, disk=disk)
            for index in indexes:
                if not (preserve_routing and self._page_kinds[index]):
                    self._drop_hot(index)
        return self.stats()

    def offload_cold_tail(self):
        """Enter the first pressure tier and disk-back only the LRU tail.

        The newest ``hot_limit`` tiles within ``decoded_budget_bytes`` remain
        decoded. No stage boundary calls this while the compile is below its
        RSS budget, so the ordinary path pays no codec churn.
        """
        with self._lock:
            self.pressure_level = max(self.pressure_level, 1)
            hot_count = self._payload_hot_count
            decoded_bytes = self._payload_decoded_bytes
            victims = []
            for index in self._hot:
                if self._page_kinds[index]:
                    continue
                if hot_count <= 1:
                    break
                if (
                    hot_count <= self.payload_limit
                    and decoded_bytes <= self.decoded_budget_bytes
                ):
                    break
                victims.append(index)
                hot_count -= 1
                decoded_bytes -= self._decoded_sizes[index].total
            self._encode_batch(victims, disk=True)
            for index in victims:
                self._drop_hot(index)
        return self.stats()

    def compress_resident_ram(self):
        """Enter the second tier: compressed RAM plus a bounded decoded front."""
        with self._lock:
            self.pressure_level = max(self.pressure_level, 2)
            self.disk_mode = False
            self.flush(disk=False, preserve_routing=True)
        return self.stats()

    def expand_decoded_front(self, decoded_budget_bytes):
        """Use available RAM as a larger cache over the same backing store.

        Expansion is demand-driven: this does not decode a tile merely to
        fill memory.  A subsequently accessed payload remains hot, avoiding
        repeated decode/evict cycles while the process has budget headroom.
        """

        with self._lock:
            if self.pressure_level == 0:
                return self.stats()
            target = max(
                self.base_decoded_budget_bytes,
                int(decoded_budget_bytes),
            )
            if target > self.decoded_budget_bytes:
                self.decoded_budget_bytes = target
                self.payload_limit = max(
                    self.payload_limit, len(self._tiles),
                )
        return self.stats()

    def resize_decoded_front(self, decoded_budget_bytes):
        """Resize the disposable decoded cache without changing its backing."""

        with self._lock:
            if self.pressure_level == 0:
                return self.stats()
            self.decoded_budget_bytes = max(
                self.base_decoded_budget_bytes,
                int(decoded_budget_bytes),
            )
            self.payload_limit = (
                max(self.base_payload_limit, len(self._tiles))
                if self.decoded_budget_bytes > self.base_decoded_budget_bytes
                else (1 if self.pressure_level >= 3
                      else self.base_payload_limit)
            )
            self._trim()
        return self.stats()

    def contract_decoded_front(self):
        """Return an expanded cache to this tier's declared working set."""

        with self._lock:
            self.decoded_budget_bytes = self.base_decoded_budget_bytes
            self.payload_limit = (
                1 if self.pressure_level >= 3
                else self.base_payload_limit
            )
            self._trim()
        return self.stats()

    def offload_to_disk(self):
        """Enter the final tier: disk backing and one decoded payload tile.

        Every non-disk or dirty tile is written once.  Serialization remains
        ordered under the arena lock because it discovers exact semantic leaf
        identities.  Independent bounded-memory codec jobs run in the pool in
        batches of at most ``compression_workers``.  With each codec dictionary
        capped at 1 MiB this is O(B + T) work.  Serialization is necessarily
        ordered and remains O(B + T); after each bounded batch is serialized,
        its codec portion has O(B/W + S + T) span for B payload bytes, T
        tile-directory entries, W workers and largest tile S.  Workspace is at
        most W raw tiles plus W bounded codec dictionaries.
        Compact identity headers and mutable-map route indexes remain resident;
        actual cache-missing payload accesses cost O(A*S) codec bytes in the
        worst case for A accesses and largest payload tile S.
        """
        with self._lock:
            self.pressure_level = 3
            self.disk_mode = True
            self.payload_limit = 1
            store = self._disk_store()
            candidates = (
                index for index, record in enumerate(self._tiles)
                if index in self._dirty or not isinstance(record, DiskTile)
            )
            job_count = 0
            with ThreadPoolExecutor(
                max_workers=self.compression_workers,
                thread_name_prefix="compiler-compress",
            ) as pool:
                started = time.perf_counter()
                while True:
                    batch = []
                    for index in candidates:
                        previous = self._tiles[index]
                        dirty = index in self._dirty or previous is None
                        source = self._dump(self._hot[index]) if dirty else previous
                        batch.append((index, source, self._row_lengths[index], previous, dirty))
                        if len(batch) == self.compression_workers:
                            break
                    if not batch:
                        break

                    def write(job):
                        index, source, rows, previous, dirty = job
                        data = source if isinstance(source, bytes) else source.decode()
                        return index, store.write(data, rows), previous, dirty

                    # pool.map has only this bounded batch to submit; neither
                    # futures nor raw payload arguments can grow with the arena.
                    for index, record, previous, dirty in pool.map(write, batch):
                        self._replace_record(index, record)
                        self._live_disk_bytes += record.stored_bytes - (
                            previous.stored_bytes if isinstance(previous, DiskTile) else 0)
                        if dirty:
                            self._dirty.discard(index)
                            self.compressions += 1
                        if index in self._hot and not self._page_kinds[index]:
                            self._drop_hot(index)
                        job_count += 1
            if job_count:
                self.parallel_compression_jobs += job_count
                self.compression_seconds += time.perf_counter() - started
            if not isinstance(self._tiles, DiskTileDirectory):
                directory = DiskTileDirectory(
                    store, self.directory, exclusive=True,
                )
                for index, record in enumerate(self._tiles):
                    directory.append(record, kind=self._page_kinds[index], rows=self._row_lengths[index])
                directory.flush()
                self._tiles = directory
                self._page_kinds, self._row_lengths = directory.kinds, directory.row_lengths
        return self.stats()

    def _raw_load(self, data):
        owner = self
        class Reader(pickle.Unpickler):
            def persistent_load(self, key):
                if key[0] == "live":
                    return _ManagedLink(key[1])
                if key[0] == "leaf":
                    leaf = owner._leaf_refs[key[1]]
                    # NEWOBJ requires the actual class, but a payload's opaque
                    # leaf can be a marker. No compiler objects are revived by
                    # the collector merely to find their storage edges.
                    return leaf if isinstance(leaf, type) else _OpaqueLink(key[1])
                raise ValueError("unknown compiler tile reference")
        return Reader(io.BytesIO(data)).load()

    def collect_unused(self):
        """Mark/sweep the actual storage graph from still-live handle views.

        Drain decoded caches first so they do not become artificial roots.
        Mark bits and the traversal frontier are disk-backed bounded buffers;
        no second semantic/provenance ledger or whole-graph working copy.
        Missing roots are never inferred from names, SSA ids or refcounts.

        The configured tile, mark-window and codec dictionary sizes are fixed,
        so the operational bound is O(B + V + E): encoded bytes, reachable
        records and storage edges.  Parameterizing the maximum decoded tile
        size S exposes the locality caveat: the one-tile DFS cache may decode
        on every cross-tile visit, making the strict worst case
        O(V*S + E + B). The sweep remains O(B + V); dead and superseded
        tiles are reclaimed individually without relocating live tiles.
        """
        from .shell_telemetry import compile_substage, compile_count, compile_count_close, compile_iter

        with self._lock:
            previous_probe, self.pressure_probe = self.pressure_probe, None
            try:
                # Drop decoded generations while determining reachability so
                # cached payload objects cannot become artificial roots. The
                # retained search pages are eagerly restored after the sweep.
                retained_pages = tuple(index for index, kind in enumerate(self._page_kinds)
                                       if kind)
                compile_substage("storage collection: flush decoded generations")
                self.flush()
                compile_substage("storage collection: Python cycle collection")
                gc.collect()
                roots = tuple(self._refs.keys())
                leaves = set()
                # Class objects are codec vocabulary, not graph payloads.
                leaves.update(identity for identity, leaf in self._leaf_refs.items()
                              if isinstance(leaf, type))
                reclaimed = 0
                compile_substage("storage collection: trace live handles")
                traced = 0
                compile_count("storage-mark", 0, label="Storage handles traced")
                with _DiskMarks(self.directory) as marked, _DiskFrontier(self.directory) as pending:
                    def edge(handle):
                        if handle != -1 and marked.mark(handle):
                            pending.push(handle)
                    for handle in roots:
                        edge(handle)
                    cache_index, cache = -1, None
                    while pending:
                        handle = pending.pop()
                        traced += 1
                        if traced % 1024 == 0:
                            compile_count("storage-mark", traced)
                        index, position = divmod(handle, self.tile_rows)
                        if index != cache_index:
                            encoded = self._tiles[index]
                            cache = ([None] * self._row_lengths[index] if encoded is None
                                     else self._raw_load(encoded.decode()))
                            cache_index = index
                        record = cache[position]
                        if record is None:
                            raise ValueError("live compiler handle points to reclaimed storage")
                        if self._page_kinds[index] == 1:
                            tag = record[0]
                            if tag == "dict":
                                edge(record[1]); edge(record[2])
                            elif tag == "list":
                                edge(record[1]); edge(record[2])
                            elif tag in {"tuple", "set"}:
                                edge(record[1])
                            elif tag == "object":
                                edge(record[2])
                            continue
                        if isinstance(record, tuple) and len(record) == 2:
                            tag, payload = record
                            if tag == "branch":
                                edge(payload[1]); edge(payload[2])
                            elif tag == "collision" and payload[1] is not None:
                                edge(payload[1])
                        work = [record]
                        while work:
                            value = work.pop()
                            if isinstance(value, (_StorageLink, _ManagedLink)):
                                edge(value.handle)
                            elif isinstance(value, _OpaqueLink):
                                leaves.add(value.identity)
                            elif type(value) is dict:
                                for key, member in value.items():
                                    work.extend((key, member))
                            elif type(value) in {tuple, list, set, frozenset}:
                                work.extend(value)
                            elif is_dataclass(value):
                                work.extend(getattr(value, field.name) for field in dataclass_fields(value))
                    # No mutation until marking succeeds. Sweep one encoded
                    # tile at a time; its surviving rows keep the SAME handles.
                    compile_count("storage-mark", traced)
                    compile_count_close("storage-mark")
                    compile_substage("storage collection: sweep unreachable records")
                    live_route_indexes = set()
                    for index, encoded in compile_iter(enumerate(self._tiles), "storage-sweep",
                            label="Storage tiles swept", total=len(self._tiles)):
                        if encoded is None:
                            continue
                        rows = self._raw_load(encoded.decode())
                        keep = [marked.contains(index * self.tile_rows + p) for p in range(len(rows))]
                        if self._page_kinds[index] == 1:
                            live_route_indexes.update(
                                index * self.tile_rows + position
                                for position, (row, live) in enumerate(zip(rows, keep))
                                if live and isinstance(row, tuple) and row and row[0] == "dict"
                            )
                        dead = sum(row is not None and not live for row, live in zip(rows, keep))
                        if not dead:
                            continue
                        reclaimed += dead
                        if not any(keep):
                            self._replace_record(index, None)
                            if isinstance(encoded, DiskTile):
                                self._live_disk_bytes -= encoded.stored_bytes
                            continue
                        # Decode actual references only in a partially live
                        # tile, then release unmarked rows before re-encoding.
                        self._hot[index] = [self._restore_raw(v) if live else None
                                            for v, live in zip(rows, keep)]
                        self._measure_page(index)
                        self._dirty.add(index)
                        self._encode(index)
                        self._drop_hot(index)
                self._route_indexes = {
                    handle: route for handle, route in self._route_indexes.items()
                    if handle in live_route_indexes
                }
                removed = len(self._leaf_refs) - len(leaves)
                self._leaf_refs = {identity: self._leaf_refs[identity] for identity in leaves}
                compile_substage("storage collection: release dead object headers")
                gc.collect()
                compile_substage("storage collection: restore retained search pages")
                for index in compile_iter(retained_pages, "storage-index-restore", label="Search pages restored"):
                    if index < len(self._tiles) and self._tiles[index] is not None:
                        self._page(index)
                return {"reclaimed_records": reclaimed, "reclaimed_leaves": removed,
                        "live_roots": len(roots)}
            finally:
                self.pressure_probe = previous_probe

    def _restore_raw(self, value):
        # Restore only marked rows. Loading the whole original tile would
        # revive dead headers, including ones an earlier sweep reclaimed.
        if isinstance(value, _ManagedLink):
            return self.resolve_reference(value.handle)
        if isinstance(value, _OpaqueLink):
            return self._leaf_refs[value.identity]
        kind = type(value)
        if kind is dict:
            return {self._restore_raw(k): self._restore_raw(v) for k, v in value.items()}
        if kind in {list, tuple, set, frozenset}:
            return kind(self._restore_raw(v) for v in value)
        if kind is not _StorageLink and is_dataclass(value) and not isinstance(value, type):
            result = object.__new__(kind)
            for field in dataclass_fields(value):
                object.__setattr__(result, field.name, self._restore_raw(getattr(value, field.name)))
            return result
        return value

    def stats(self):
        encoded = sum(sys.getsizeof(t) + sys.getsizeof(t.payload)
                      for t in self._tiles if isinstance(t, CompressedTile))
        disk_bytes = disk_count = 0
        for record in self._tiles:
            if isinstance(record, DiskTile):
                disk_bytes += record.stored_bytes
                disk_count += 1
        route_bytes = (sys.getsizeof(self._route_indexes)
                       + sum(route.resident_bytes() for route in self._route_indexes.values()))
        headers = (sys.getsizeof(self._refs) + sys.getsizeof(self._refs.data)
                   + sys.getsizeof(self._leaf_refs) + sys.getsizeof(self._decoded_sizes)
                   + sys.getsizeof(self._classes) + sys.getsizeof(self._class_ids)
                   + route_bytes)
        headers += sum(sys.getsizeof(measured) for measured in self._decoded_sizes.values())
        if isinstance(self._tiles, DiskTileDirectory):
            headers += self._tiles.resident_bytes()
        else:
            headers += (sys.getsizeof(self._tiles) + sys.getsizeof(self._page_kinds)
                        + sys.getsizeof(self._row_lengths))
            headers += sum(sys.getsizeof(t) for t in self._tiles if isinstance(t, DiskTile))
        headers += sum(sys.getsizeof(identity) for identity in self._leaf_refs)
        scalar_seen = set()
        def scalar(value):
            if value is None or id(value) in scalar_seen:
                return 0
            scalar_seen.add(id(value))
            return sys.getsizeof(value)
        for cached in self._refs.values():
            value = cached.value if isinstance(cached, _TupleAnchor) else cached
            if isinstance(value, (TiledDict, TiledList, TiledObject, TiledTuple, TiledSet)):
                headers += sys.getsizeof(value)
                if isinstance(value, TiledList):
                    headers += sum(scalar(v) for v in (value._first, value._count, value._ref))
                elif isinstance(value, TiledDict):
                    headers += sum(scalar(v) for v in (value._root, value._count, value._version, value._ref))
                elif isinstance(value, TiledObject):
                    state = object.__getattribute__(value, "__dict__")
                    headers += sys.getsizeof(state) + scalar(state.get("_tile_ref"))
        return LiveTileStats(self._count, len(self._tiles), len(self._hot), self.hot_peak,
                             encoded, disk_bytes, disk_count,
                             self.compressions, self.reloads, len(self._refs), headers,
                             self._reference_count, self._decoded_bytes, self._decoded_peak,
                             self.pressure_level, self.routing_reloads, self.payload_reloads,
                             sum(bool(self._page_kinds[index]) for index in self._hot),
                             self._payload_hot_count, self.route_hits, self.route_misses,
                             self._retained_decoded_bytes + route_bytes,
                             self._payload_decoded_bytes,
                             self._payload_hot_peak, self.payload_cache_hits,
                             self.payload_evictions, self.parallel_compression_jobs,
                             self.compression_seconds, self.compression_workers)


def arena_for_book(book):
    """Return the one live arena already carried by a concordance book."""
    arenas = {
        id(owner): owner
        for page in dict.values(book.pages)
        if (owner := page.__dict__.get("_live_tile_owner")) is not None
    }
    if len(arenas) > 1:
        raise RuntimeError("one identity book is backed by multiple live tile arenas")
    return next(iter(arenas.values()), None)


def live_storage_report(book):
    """Current physical-storage telemetry, including inherited arenas."""
    arena = arena_for_book(book)
    if arena is None:
        return {"source_and_ssa_paged": False, "storage_pressure_level": 0}
    stats = arena.stats()
    source_and_ssa = any(
        category == "ProcessGraph.NetworkX"
        or category.endswith(".ControlProgram")
        or category.endswith(".Instr")
        for category in arena.categories
    )
    return {
        "source_and_ssa_paged": source_and_ssa,
        "storage_pressure_level": stats.pressure_level,
        "storage_disk_tiles": stats.disk_tiles,
        "storage_disk_bytes": stats.disk_bytes,
        "storage_decoded_bytes": stats.decoded_bytes,
        "storage_payload_hot_tiles": stats.payload_hot_tiles,
        "storage_payload_cache_hits": stats.payload_cache_hits,
        "storage_payload_reloads": stats.payload_reloads,
        "storage_payload_evictions": stats.payload_evictions,
        "storage_routing_reloads": stats.routing_reloads,
        "storage_route_hits": stats.route_hits,
        "storage_route_misses": stats.route_misses,
        "storage_retained_index_bytes": stats.retained_index_bytes,
        "storage_compression_jobs": stats.parallel_compression_jobs,
        "storage_compression_seconds": stats.compression_seconds,
        "storage_compression_workers": stats.compression_workers,
    }


class _TupleAnchor:
    __slots__ = ("value", "__weakref__")
    def __init__(self, value):
        self.value = value


def _owned_row_bytes(value):
    """Conservative owned bytes for one bounded raw row, not foreign heaps."""
    work, seen, size = [value], set(), 0
    while work:
        item = work.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        kind = type(item)
        if kind is dict:
            for k, v in item.items():
                work.extend((k, v))
        elif kind in {list, tuple, set, frozenset}:
            work.extend(item)
        elif _value_fact(kind):
            state = getattr(item, "__dict__", None)
            if state is not None:
                work.append(state)
            else:
                work.extend(getattr(item, field.name) for field in dataclass_fields(item))
    return size


@dataclass(frozen=True, slots=True)
class _StorageLink:
    handle: int


@dataclass(frozen=True, slots=True)
class _OpaqueLink:
    identity: int


@dataclass(frozen=True, slots=True)
class _ManagedLink:
    handle: int


class _DiskMarks:
    """External reachability bitmap with one 64KiB read/write window."""
    window_bytes = 64 << 10
    def __init__(self, directory):
        self.file = tempfile.TemporaryFile(dir=directory)
        self.index, self.window, self.dirty = -1, bytearray(), False

    def _window(self, handle):
        byte, bit = divmod(handle, 8)
        index, position = divmod(byte, self.window_bytes)
        if index != self.index:
            self._flush()
            self.file.seek(index * self.window_bytes)
            self.window = bytearray(self.file.read(self.window_bytes).ljust(self.window_bytes, b"\0"))
            self.index = index
        return position, 1 << bit

    def _flush(self):
        if self.dirty:
            self.file.seek(self.index * self.window_bytes)
            self.file.write(self.window)
            self.dirty = False

    def mark(self, handle):
        position, mask = self._window(handle)
        if self.window[position] & mask:
            return False
        self.window[position] |= mask
        self.dirty = True
        return True

    def contains(self, handle):
        position, mask = self._window(handle)
        return bool(self.window[position] & mask)

    def __enter__(self): return self
    def __exit__(self, *error): self.file.close()


class _DiskFrontier:
    """External DFS work stack; at most4096 uint64 handles in RAM."""
    block_items = 4096
    def __init__(self, directory):
        self.file = tempfile.TemporaryFile(dir=directory)
        self.buffer, self.blocks = array("Q"), 0

    def push(self, handle):
        self.buffer.append(handle)
        if len(self.buffer) == self.block_items:
            self.file.seek(self.blocks * self.block_items * 8)
            self.file.write(self.buffer.tobytes())
            self.blocks += 1
            self.buffer = array("Q")

    def pop(self):
        if not self.buffer:
            self.blocks -= 1
            self.file.seek(self.blocks * self.block_items * 8)
            self.buffer.frombytes(self.file.read(self.block_items * 8))
        return self.buffer.pop()

    def __bool__(self): return bool(self.buffer) or self.blocks > 0
    def __enter__(self): return self
    def __exit__(self, *error): self.file.close()


class TiledDict(dict, MutableMapping):
    """Ordered mutable mapping; hash-trie buckets and order are separate tiles.

    The dict superclass preserves existing isinstance contracts. ALL ordinary
    methods are overridden; explicit dict.__setitem__ bypass is unsupported.
    Values can escape eviction because they are handles into this same arena.
    Under pressure, a compact resident uint64 hash-to-bucket index accelerates
    the authoritative trie. Equality remains in the decoded bucket; collisions
    therefore retain Python mapping semantics without repeated routing reads.
    """
    __slots__ = ("_arena", "_root", "_order", "_count", "_version", "_ref", "__weakref__")

    def __init__(self, arena):
        self._arena = arena
        self._root = arena.allocate(("leaf", {}))
        self._order = None
        self._count = self._version = 0
        self._ref = arena.reference(self)

    def _bucket(self, key):
        code = hash(key) & ((1 << 64) - 1)
        cached = self._arena.cached_route(self._ref, code)
        handle = self._root if cached is None else cached
        collision_head = None
        next_bit = 0
        while True:
            tag, payload = self._arena.read(handle)
            if tag == "leaf":
                self._arena.remember_route(self._ref, code, handle)
                return handle, payload, next_bit
            if tag == "collision":
                if collision_head is None:
                    collision_head = handle
                bucket, following = payload
                if key in bucket or following is None:
                    self._arena.remember_route(self._ref, code, collision_head)
                    return handle, bucket, 64
                handle = following
                continue
            bit, left, right = payload
            handle = right if (code >> bit) & 1 else left
            next_bit = bit + 1

    def _replace_bucket(self, handle, items):
        tag, payload = self._arena.read(handle)
        self._arena.write(handle, ("collision", (items, payload[1]))
                          if tag == "collision" else ("leaf", items))

    def _split(self, handle, items, bit=0):
        if len(items) <= self._arena.bucket_rows:
            self._replace_bucket(handle, items)
            if bit < 64:
                for key in items:
                    self._arena.remember_route(
                        self._ref, hash(key) & ((1 << 64) - 1), handle)
            return
        # Equal full hashes cannot be separated by another trie bit. Keep
        # equality authoritative in bounded overflow buckets, rather than
        # silently placing every colliding key in one decoded RAM record.
        if bit == 64:
            tag, payload = self._arena.read(handle)
            following = payload[1] if tag == "collision" else None
            rows, entries = self._arena.bucket_rows, list(items.items())
            # One newly inserted entry after ``_set``; any number after a
            # bulk adoption: every segment stays bounded, built tail first.
            for start in range(rows * ((len(entries) - 1) // rows), 0, -rows):
                following = self._arena.allocate(("collision", (
                    dict(entries[start:start + rows]), following)))
            self._arena.write(handle, ("collision", (dict(entries[:rows]), following)))
            # _bucket cached the collision-chain HEAD before returning this
            # possibly later segment. Preserve that head for every equal full
            # hash; caching this local handle would skip earlier candidates.
            return
        # Skip bits shared by every key: no empty 64-level chain for identical
        # hashes, and no redundant branch levels on later insertions.
        codes = [hash(key) & ((1 << 64) - 1) for key in items]
        differences = 0
        for code in codes[1:]:
            differences |= code ^ codes[0]
        differences >>= bit
        if not differences:
            self._split(handle, items, 64)
            return
        bit += (differences & -differences).bit_length() - 1
        left, right = {}, {}
        for key, value in items.items():
            (right if (hash(key) >> bit) & 1 else left)[key] = value
        a, b = self._child(left, bit + 1), self._child(right, bit + 1)
        self._arena.write(handle, ("branch", (bit, a, b)))

    def _child(self, items, bit):
        # A child that fits one bucket is born holding its entries: one
        # allocation instead of an empty leaf rewritten (and re-measured).
        if len(items) > self._arena.bucket_rows:
            handle = self._arena.allocate(("leaf", {}))
            self._split(handle, items, bit)
            return handle
        handle = self._arena.allocate(("leaf", items))
        if bit < 64:
            for key in items:
                self._arena.remember_route(
                    self._ref, hash(key) & ((1 << 64) - 1), handle)
        return handle

    def _set(self, key, value):
        with self._arena._lock:
            handle, bucket, bit = self._bucket(key)
            # A nested allocation can evict this generation. Never mutate a
            # previously returned raw bucket AFTER that allocation.
            fresh = dict(bucket)
            exists = key in fresh
            ordinal = fresh[key][1] if exists else -1
            if not exists:
                if self._order is not None:
                    ordinal = len(self._order)
                    self._order._append((True, key))
                self._count += 1
                self._version += 1
            fresh[key] = (value, ordinal)
            if self._order is None and len(fresh) > self._arena.bucket_rows:
                self._order = TiledList(self._arena)
                for position, item in enumerate(fresh):
                    self._order._append((True, item))
                    fresh[item] = (fresh[item][0], position)
            self._split(handle, fresh, bit)
            self._arena.save_header(self)

    def _adopt_items(self, items):
        """Fill this new, empty mapping from ``(key, managed value)`` pairs
        with distinct keys, in order.

        Adopting a 1.56M-entry identity page key by key routed every insert
        to a random trie tile; once the tiles were cold each insert decoded,
        re-measured and re-encoded a whole tile (about 20 ms per entry, the
        managed tire's ``concordance_mint`` page ran for hours).  The trie a
        sequence of ``_set`` calls builds is the one ``_split`` builds from
        all the entries at once, so build it once: its buckets are allocated
        depth first and fill tiles in order.  The insertion order and each
        entry's ordinal are the ones ``_set`` would have recorded.
        """
        with self._arena._lock:
            if self._count:
                raise ValueError("bulk adoption needs an empty mapping")
            if not items:
                return
            ordered = len(items) > self._arena.bucket_rows
            if ordered:
                self._order = TiledList(self._arena)
                self._order._adopt_members([(True, key) for key, _value in items])
            fresh = {
                key: (value, position if ordered else -1)
                for position, (key, value) in enumerate(items)
            }
            if len(fresh) != len(items):
                raise ValueError("bulk adoption needs distinct keys")
            self._count = len(fresh)
            self._version += len(fresh)
            self._split(self._root, fresh, 0)
            self._arena.save_header(self)

    def __setitem__(self, key, value):
        self._set(key, self._arena.manage(value))

    def __getitem__(self, key):
        return self._bucket(key)[1][key][0]

    def __delitem__(self, key):
        handle, bucket, _ = self._bucket(key)
        fresh = dict(bucket)
        ordinal = fresh[key][1]
        del fresh[key]
        self._replace_bucket(handle, fresh)
        if self._order is not None:
            # The old ordered-map writer scanned and shifted every later key
            # on each deletion, making N removals quadratic. The ordinal is
            # part of the existing bucket entry; tombstone its exact slot.
            self._order[ordinal] = (False, None)
        self._count -= 1
        self._version += 1
        if self._count == 0:
            self._root = self._arena.allocate(("leaf", {}))
            # Removing/rebuilding loop edges empties each adjacency map.
            # A retained route into its OLD root made reinsertion visible to
            # lookup but invisible to iteration/ancestry, which then deleted
            # the still-live Add operands of `for i in range(2): x += 1.0`.
            # Root replacement retires this map's routes, just as clear does.
            self._arena.invalidate_routes(self._ref)
            self._order = None
        elif self._order is not None and len(self._order) > 2 * self._count + self._arena.bucket_rows:
            self._compact_order()
        self._arena.save_header(self)

    def _compact_order(self):
        # Reclaim tombstones amortized, streaming the same insertion order.
        # No list of all keys and no second key->ordinal index is needed.
        replacement = TiledList(self._arena)
        for present, key in self._order:
            if not present:
                continue
            handle, bucket, _ = self._bucket(key)
            fresh = dict(bucket)
            fresh[key] = (fresh[key][0], len(replacement))
            self._replace_bucket(handle, fresh)
            replacement._append((True, key))
        self._order = replacement

    def __iter__(self):
        version = self._version
        keys = (self._order if self._order is not None
                else ((True, key) for key in tuple(self._arena.read(self._root)[1])))
        for present, key in keys:
            if version != self._version:
                raise RuntimeError("dictionary changed size during iteration")
            if present:
                yield key

    def __reversed__(self):
        version = self._version
        keys = (reversed(self._order) if self._order is not None
                else ((True, key) for key in reversed(tuple(self._arena.read(self._root)[1]))))
        for present, key in keys:
            if version != self._version:
                raise RuntimeError("dictionary changed size during iteration")
            if present:
                yield key

    def __len__(self):
        return self._count

    def __contains__(self, key):
        return key in self._bucket(key)[1]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    keys = MutableMapping.keys
    values = MutableMapping.values
    items = MutableMapping.items
    update = MutableMapping.update
    def setdefault(self, key, default=None):
        # MutableMapping.setdefault returns the RAW default after __setitem__.
        # Here insertion adopts a managed container, so that raw default is
        # no longer the writer. E.g. loop_control_site_bindings.setdefault(...,
        # {})[span] = fact must mutate the authoritative tiled map immediately.
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return self[key]
    pop = MutableMapping.pop
    def popitem(self):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self))
        value = self[key]
        del self[key]
        return key, value
    def clear(self):
        if not self._count:
            return
        self._root = self._arena.allocate(("leaf", {}))
        self._arena.invalidate_routes(self._ref)
        self._order, self._count = None, 0
        self._version += 1
        self._arena.save_header(self)

    def copy(self):
        return dict(self.items())

    def __repr__(self):
        return repr(self.copy())

    def __eq__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        if len(self) != len(other):
            return False
        for key, value in self.items():
            if key not in other:
                return False
            member = other[key]
            if value is not member and value != member:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __or__(self, other):
        return self.copy() | other

    def __ror__(self, other):
        return other | self.copy()

    def __ior__(self, other):
        self.update(other)
        return self

    def __reduce_ex__(self, protocol):
        return dict, (self.copy(),)


def _sequence_compare(left, right, expected_type, relation):
    """Builtin lexicographic semantics, with no whole-sequence work buffer."""
    if not isinstance(right, expected_type):
        return NotImplemented
    for a, b in zip(left, right):
        if a is b or a == b:
            continue
        return relation(a, b)
    return relation(len(left), len(right))


class TiledList(list, MutableSequence):
    """Mutable sequence indexed by independent bounded chunks."""
    __slots__ = ("_arena", "_first", "_chunks", "_count", "_ref", "__weakref__")

    def __init__(self, arena):
        self._arena, self._first, self._chunks, self._count = arena, None, None, 0
        self._ref = arena.reference(self)

    def _handle(self, chunk):
        return self._first if chunk == 0 else self._chunks[chunk - 1].handle

    def __len__(self):
        return self._count

    def _position(self, index):
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return divmod(index, self._arena.bucket_rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        chunk, position = self._position(index)
        return self._arena.read(self._handle(chunk))[position]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            indices = range(*index.indices(self._count))
            # Consume the iterable before mutation (including self-aliases),
            # but stage it in the same paged store, not an unbounded raw list.
            values = TiledList(self._arena)
            values.extend(value)
            if index.step not in (None, 1):
                if len(indices) != len(values):
                    raise ValueError("extended slice size mismatch")
                for i, member in zip(indices, values):
                    self[i] = member
            else:
                start, stop, _ = index.indices(self._count)
                replacement = TiledList(self._arena)
                for i in range(start):
                    replacement._append(self[i])
                for member in values:
                    replacement._append(member)
                for i in range(max(start, stop), self._count):
                    replacement._append(self[i])
                self._replace_sequence(replacement)
            return
        member = self._arena.manage(value)
        chunk, position = self._position(index)
        values = list(self._arena.read(self._handle(chunk)))
        values[position] = member
        self._arena.write(self._handle(chunk), values)

    def _append(self, value):
        position = self._count % self._arena.bucket_rows
        if position == 0:
            handle = self._arena.allocate([])
            if self._count == 0:
                self._first = handle
            else:
                if self._chunks is None:
                    self._chunks = TiledDict(self._arena)
                self._chunks[self._count // self._arena.bucket_rows - 1] = _StorageLink(handle)
        handle = self._handle(self._count // self._arena.bucket_rows)
        values = list(self._arena.read(handle))
        values.append(value)
        self._arena.write(handle, values)
        self._count += 1
        self._arena.save_header(self)

    def _adopt_members(self, values):
        """Fill this new, empty list from managed ``values``, one write per
        chunk (``_append`` rewrites its chunk once per member).  The chunks
        and their links are the ones the same ``_append`` calls make."""
        if self._count:
            raise ValueError("bulk adoption needs an empty list")
        rows, links = self._arena.bucket_rows, []
        with self._arena._lock:
            for start in range(0, len(values), rows):
                handle = self._arena.allocate(list(values[start:start + rows]))
                if start == 0:
                    self._first = handle
                else:
                    links.append((start // rows - 1, self._arena.manage(
                        _StorageLink(handle))))
                self._count += len(values[start:start + rows])
            if links:
                # The chunk index is itself adopted whole: a long list's
                # links set one by one thrash the same cold trie tiles.
                self._chunks = TiledDict(self._arena)
                self._chunks._adopt_items(links)
            self._arena.save_header(self)

    def append(self, value):
        self._append(self._arena.manage(value))

    def insert(self, index, value):
        index = max(0, min(self._count, index if index >= 0 else self._count + index))
        self.append(value)
        for i in range(self._count - 1, index, -1):
            self[i] = self[i - 1]
        self[index] = value

    def __delitem__(self, index):
        if isinstance(index, slice):
            removed = range(*index.indices(self._count))
            if not removed:
                return
            replacement = TiledList(self._arena)
            for i, member in enumerate(self):
                if i not in removed:
                    replacement._append(member)
            self._replace_sequence(replacement)
            return
        chunk, position = self._position(index)
        start = chunk * self._arena.bucket_rows + position
        for i in range(start, self._count - 1):
            self[i] = self[i + 1]
        handle = self._handle((self._count - 1) // self._arena.bucket_rows)
        last = list(self._arena.read(handle))
        last.pop()
        self._arena.write(handle, last)
        self._count -= 1
        if not last:
            if self._count == 0:
                self._first = None
            else:
                del self._chunks[self._count // self._arena.bucket_rows - 1]
        self._arena.save_header(self)

    def _replace_sequence(self, replacement):
        self._first, self._chunks, self._count = (replacement._first, replacement._chunks,
                                                  replacement._count)
        self._arena.save_header(self)

    def __iter__(self):
        # Do not retain a decoded page across yield: another graph/SSA accessor
        # can evict it while this iterator is suspended.
        index = 0
        while index < self._count:
            yield self[index]
            index += 1

    def __reversed__(self):
        # list.__reversed__ reads the empty superclass storage directly. Book
        # copy-on-read resolves revisions newest-first with reversed(columns);
        # inheriting it made all as-of lexical binding reads falsely absent.
        for index in range(self._count - 1, -1, -1):
            yield self[index]

    __contains__ = MutableSequence.__contains__
    index = MutableSequence.index
    count = MutableSequence.count
    extend = MutableSequence.extend
    pop = MutableSequence.pop
    remove = MutableSequence.remove
    def clear(self):
        self._first = self._chunks = None
        self._count = 0
        self._arena.save_header(self)
    reverse = MutableSequence.reverse

    def sort(self, *, key=None, reverse=False):
        # Stable external merge sort. Only one bounded run is sorted in RAM;
        # keys are evaluated once and merge intermediates use this same arena.
        # The old sorted(self) unfurled the entire sequence at the boundary.
        current = TiledList(self._arena)
        run, run_bytes = [], 0
        width = max(1, self._arena.bucket_rows)
        # Equal-sized runs are needed by the merge schedule. The byte limit
        # below may shorten them; record each run boundary in a paged sequence.
        boundaries = TiledList(self._arena)
        def emit_run():
            for entry in sorted(run, key=lambda entry: entry[0], reverse=reverse):
                current._append(entry)
            boundaries._append(len(current))
        for member in self:
            entry = (member if key is None else key(member), member)
            size = _owned_row_bytes(entry)
            if run and (len(run) == width or run_bytes + size > self._arena.decoded_budget_bytes // 4):
                emit_run()
                run.clear()
                run_bytes = 0
            run.append(entry)
            run_bytes += size
        if run:
            emit_run()
        while len(boundaries) > 1:
            merged = TiledList(self._arena)
            next_boundaries = TiledList(self._arena)
            start = 0
            for boundary in range(0, len(boundaries), 2):
                middle = boundaries[boundary]
                stop = boundaries[boundary + 1] if boundary + 1 < len(boundaries) else middle
                left, right = start, middle
                while left < middle and right < stop:
                    a, b = current[left], current[right]
                    take_right = a[0] < b[0] if reverse else b[0] < a[0]
                    merged._append(b if take_right else a)
                    if take_right: right += 1
                    else: left += 1
                for position in range(left, middle):
                    merged._append(current[position])
                for position in range(right, stop):
                    merged._append(current[position])
                next_boundaries._append(len(merged))
                start = stop
            current, boundaries = merged, next_boundaries
        replacement = TiledList(self._arena)
        for _sort_key, member in current:
            replacement._append(member)
        self._replace_sequence(replacement)

    def copy(self):
        return list(self)

    def __repr__(self):
        return repr(list(self))

    def __eq__(self, other):
        return _sequence_compare(self, other, list, operator.eq)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return _sequence_compare(self, other, list, operator.lt)

    def __le__(self, other):
        return _sequence_compare(self, other, list, operator.le)

    def __gt__(self, other):
        return _sequence_compare(self, other, list, operator.gt)

    def __ge__(self, other):
        return _sequence_compare(self, other, list, operator.ge)

    def __add__(self, other):
        return list(self) + other

    def __radd__(self, other):
        return other + list(self)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __mul__(self, count):
        return list(self) * count

    __rmul__ = __mul__

    def __imul__(self, count):
        self[:] = list(self) * count
        return self

    def __reduce_ex__(self, protocol):
        return list, (list(self),)


class TiledTuple(tuple):
    """Immutable tuple-compatible view of a chunked sequence."""
    def __new__(cls, sequence):
        result = tuple.__new__(cls)
        result._sequence, result._arena = sequence, sequence._arena
        result._ref = sequence._arena.reference(result)
        return result

    def __len__(self):
        return len(self._sequence)

    def __iter__(self):
        return iter(self._sequence)

    def __getitem__(self, index):
        result = self._sequence[index]
        return tuple(result) if isinstance(index, slice) else result

    def __repr__(self):
        return repr(tuple(self))

    def __eq__(self, other):
        return _sequence_compare(self, other, tuple, operator.eq)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return _sequence_compare(self, other, tuple, operator.lt)

    def __le__(self, other):
        return _sequence_compare(self, other, tuple, operator.le)

    def __gt__(self, other):
        return _sequence_compare(self, other, tuple, operator.gt)

    def __ge__(self, other):
        return _sequence_compare(self, other, tuple, operator.ge)

    def __hash__(self):
        return hash(tuple(self))

    def __contains__(self, item):
        return item in self._sequence

    def __add__(self, other):
        return tuple(self) + other

    def __radd__(self, other):
        return other + tuple(self)

    def __mul__(self, count):
        return tuple(self) * count

    __rmul__ = __mul__
    index = TiledList.index
    count = TiledList.count

    def __reduce_ex__(self, protocol):
        return tuple, (tuple(self),)


class TiledSet(MutableSet):
    """The mutable set protocol over the same bounded hash-trie storage."""
    __slots__ = ("_arena", "_members", "_ref", "__weakref__")
    def __init__(self, arena, values=()):
        self._arena, self._members = arena, TiledDict(arena)
        self._ref = arena.reference(self)
        self.update(values)

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return iter(self._members)

    def __contains__(self, value):
        if isinstance(value, set):
            value = frozenset(value)
        return value in self._members

    def add(self, value):
        self._members[value] = None

    def discard(self, value):
        if isinstance(value, set):
            value = frozenset(value)
        self._members.pop(value, None)

    def remove(self, value):
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def pop(self):
        if not self:
            raise KeyError("pop from an empty set")
        value = next(iter(self))
        self.discard(value)
        return value

    def clear(self):
        self._members.clear()

    def update(self, *others):
        for other in others:
            for value in other:
                self.add(value)

    def intersection_update(self, *others):
        admitted = tuple(other if isinstance(other, Set) else TiledSet(self._arena, other)
                         for other in others)
        replacement = TiledDict(self._arena)
        for value in self:
            if all(value in other for other in admitted):
                replacement[value] = None
        self._members = replacement
        self._arena.save_header(self)

    def difference_update(self, *others):
        for other in others:
            if other is self:
                self.clear()
                continue
            for value in other:
                self.discard(value)

    def symmetric_difference_update(self, other):
        if other is self:
            self.clear()
            return
        other = other if isinstance(other, Set) else TiledSet(self._arena, other)
        for value in other:
            self.discard(value) if value in self else self.add(value)

    def copy(self):
        return TiledSet(self._arena, self)

    def union(self, *others):
        result = self.copy()
        result.update(*others)
        return result

    def intersection(self, *others):
        return TiledSet(self._arena, (v for v in self if all(v in other for other in others)))

    def difference(self, *others):
        return TiledSet(self._arena, (v for v in self if not any(v in other for other in others)))

    def symmetric_difference(self, other):
        result = self.copy()
        result.symmetric_difference_update(other)
        return result

    def __or__(self, other):
        return self.union(other) if isinstance(other, Set) else NotImplemented

    __ror__ = __or__

    def __and__(self, other):
        return self.intersection(other) if isinstance(other, Set) else NotImplemented

    __rand__ = __and__

    def __sub__(self, other):
        return self.difference(other) if isinstance(other, Set) else NotImplemented

    def __rsub__(self, other):
        return TiledSet(self._arena, (v for v in other if v not in self)) if isinstance(other, Set) else NotImplemented

    def __xor__(self, other):
        return self.symmetric_difference(other) if isinstance(other, Set) else NotImplemented

    __rxor__ = __xor__
    __eq__, __ne__ = MutableSet.__eq__, MutableSet.__ne__
    __lt__, __le__ = MutableSet.__lt__, MutableSet.__le__
    __gt__, __ge__ = MutableSet.__gt__, MutableSet.__ge__
    def __ior__(self, other):
        if not isinstance(other, Set): return NotImplemented
        self.update(other)
        return self

    def __iand__(self, other):
        if not isinstance(other, Set): return NotImplemented
        self.intersection_update(other)
        return self

    def __isub__(self, other):
        if not isinstance(other, Set): return NotImplemented
        self.difference_update(other)
        return self

    def __ixor__(self, other):
        if not isinstance(other, Set): return NotImplemented
        self.symmetric_difference_update(other)
        return self
    isdisjoint = MutableSet.isdisjoint

    def issubset(self, other):
        other = other if isinstance(other, Set) else TiledSet(self._arena, other)
        return all(value in other for value in self)

    def issuperset(self, other):
        return all(value in self for value in other)

    def __repr__(self):
        return repr(set(self))

    def __reduce_ex__(self, protocol):
        return set, (set(self),)


class TiledObject:
    """Original compiler object/identity; its fields become a writable tile map."""
    __slots__ = ()
    __live_storage_root__ = False
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original = cls.__dict__.get("__post_init__")
        if original is not None:
            def initialize(self, *args, **kwargs):
                original(self, *args, **kwargs)
                TiledObject.__post_init__(self)
            cls.__post_init__ = initialize

    def __post_init__(self):
        arena = _ACTIVE_ARENA.get()
        if arena is not None:
            arena.attach(self)
        else:
            registrar = _ACTIVE_REGISTRAR.get()
            if registrar is not None and type(self).__live_storage_root__:
                registrar(self)

    def __getattribute__(self, name):
        if name in {"_tile_arena", "_tile_fields", "_tile_ref", "__class__"}:
            return object.__getattribute__(self, name)
        state = object.__getattribute__(self, "__dict__")
        fields = state.get("_tile_fields")
        if fields is not None:
            if name == "__dict__":
                return fields
            # The amount-property repro returned7 rather than14 because the
            # tiled writer bypassed the original property's setter, and a
            # paged field then shadowed its getter. Python data descriptors
            # remain the authoritative writer/reader, ahead of instance data.
            descriptor = _data_descriptor(type(self), name)
            if descriptor is not None and hasattr(type(descriptor), "__get__"):
                return object.__getattribute__(self, name)
            try:
                return fields[name]
            except KeyError:
                pass
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        state = object.__getattribute__(self, "__dict__")
        fields = state.get("_tile_fields")
        if fields is not None and name not in {"_tile_arena", "_tile_fields", "_tile_ref"}:
            if _data_descriptor(type(self), name) is not None:
                object.__setattr__(self, name, value)
            else:
                fields[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        state = object.__getattribute__(self, "__dict__")
        if "_tile_fields" in state:
            if _data_descriptor(type(self), name) is not None:
                object.__delattr__(self, name)
            else:
                try:
                    del state["_tile_fields"][name]
                except KeyError:
                    raise AttributeError(name) from None
        else:
            object.__delattr__(self, name)

    def __getstate__(self):
        return dict(self.__dict__.items())

    def __setstate__(self, state):
        object.__getattribute__(self, "__dict__").update(state)


def _data_descriptor(kind, name):
    descriptor = getattr(kind, name, None)
    protocol = type(descriptor)
    return descriptor if hasattr(protocol, "__set__") or hasattr(protocol, "__delete__") else None


def live_digraph(arena=None):
    """NetworkX with its documented dictionary factories; no surrogate graph."""
    import networkx as nx
    arena = arena or _ACTIVE_ARENA.get()
    if arena is None:
        graph = nx.DiGraph()
        registrar = _ACTIVE_REGISTRAR.get()
        if registrar is not None:
            registrar(graph)
        return graph
    return LiveDiGraph(arena=arena)


def adopt_digraph(graph, arena):
    """Move an existing ordinary NetworkX graph into the live arena in place."""
    if isinstance(graph, LiveDiGraph):
        graph._tile_owner = arena
        arena.attach(graph)
        return graph
    import networkx as nx
    if not isinstance(graph, nx.DiGraph):
        return graph
    if type(graph) is nx.DiGraph:
        # Late adoption previously changed the maps but kept plain factories.
        # NetworkX.add_node installed an empty dict, our map adopted a COPY,
        # and its following attr_dict.update wrote only the discarded original.
        # finalize_graph_with_outputs therefore minted an empty Store node.
        # Keep this exact NetworkX object and use the existing live factories
        # for subsequent node/edge births, plus their existing snapshot codec.
        graph.__class__ = LiveDiGraph
        graph._tile_owner = arena
        state = object.__getattribute__(graph, "__dict__")
        for name in _NETWORKX_CACHED_VIEWS:
            state.pop(name, None)
    # NetworkX's documented map attributes are the authority. Their setters
    # invalidate cached views, and outer tiled maps adopt all future inner maps.
    # Successor and predecessor entries own the SAME mutable edge record.
    # One memo across both maps must preserve that alias during conversion.
    memo = {}
    graph.graph = arena.manage(graph.graph, memo)
    graph._node = arena.manage(graph._node, memo)
    graph._adj = arena.manage(graph._adj, memo)
    graph._succ = graph._adj
    graph._pred = arena.manage(graph._pred, memo)
    if isinstance(graph, LiveDiGraph):
        arena.attach(graph)
    arena.categories.add("ProcessGraph.NetworkX")
    return graph


import networkx as nx


class LiveDiGraph(nx.DiGraph, TiledObject):
    """The normal NetworkX graph, with compile-wide writable dict factories."""
    # DiGraph remains the solid base, allowing an ordinary graph to adopt the
    # same storage class in place without replacing its identity or layout.
    __slots__ = ()
    def __init__(self, incoming_graph_data=None, *, arena=None, **attr):
        self._tile_owner = arena or _ACTIVE_ARENA.get()
        super().__init__(incoming_graph_data, **attr)
        if self._tile_owner is not None:
            self._tile_owner.categories.add("ProcessGraph.NetworkX")
            self._tile_owner.attach(self)

    def __getattribute__(self, name):
        # NetworkX cached views are derived from its authoritative maps. Keep
        # them only with the live graph header, not as opaque leaves in the
        # arena: OutEdgeView holds _graph and would pin every cold graph.
        if name in _NETWORKX_CACHED_VIEWS:
            state = object.__getattribute__(self, "__dict__")
            cache = state.setdefault("_tile_views", {})
            if name not in cache:
                cache[name] = _NETWORKX_CACHED_VIEWS[name].func(self)
            return cache[name]
        return TiledObject.__getattribute__(self, name)

    def _mapping(self):
        return {} if self._tile_owner is None else TiledDict(self._tile_owner)

    def __setattr__(self, name, value):
        TiledObject.__setattr__(self, name, value)
        if name in {"_adj", "_succ", "_pred", "_node"}:
            # NetworkX invalidates its __dict__ cache in these descriptors.
            # Our views intentionally live with the weak graph header instead;
            # rebinding must invalidate that cache at the SAME writer too.
            state = object.__getattribute__(self, "__dict__")
            state.get("_tile_views", {}).clear()

    node_dict_factory = _mapping
    node_attr_dict_factory = _mapping
    adjlist_outer_dict_factory = _mapping
    adjlist_inner_dict_factory = _mapping
    edge_attr_dict_factory = _mapping
    graph_attr_dict_factory = _mapping

    def __reduce_ex__(self, protocol):
        # Explicit export is a snapshot, never a pickle of active file handles.
        return _restore_digraph, (dict(self.graph.items()), list(self.nodes(data=True)),
                                  list(self.edges(data=True)))


_NETWORKX_CACHED_VIEWS = {
    name: getattr(nx.DiGraph, name) for name in
    ("nodes", "edges", "adj", "succ", "pred", "in_edges", "out_edges",
     "degree", "in_degree", "out_degree")
}


def _restore_digraph(metadata, nodes, edges):
    graph = live_digraph()
    graph.graph.update(metadata)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def tile_identity_page(page, arena):
    """Tile the authoritative book tables too, including writable hot pages.

    The construction clock and registry are NOT cloned. A write uses the same
    page._stamp and the same book clock; no whole-page promotion is necessary.
    """
    if page.__dict__.get("_live_tile_owner") is arena:
        return
    from .identity_spill import SPILLABLE_ATTRS
    page.__dict__["_live_tile_owner"] = arena
    for name in SPILLABLE_ATTRS:
        # A page is only the outer unit.  Its tables can contain millions of
        # entries, so the page-level 0/N counter can stay unchanged for a long
        # time. Report the actual existing entry loop through the compiler's
        # normal nested bar.
        value = getattr(page, name)
        managed = arena.manage(
            value,
            progress=(
                "adopt-identity-table-entries",
                f"Identity page {page.name}: {name} entries",
            ),
        )
        object.__setattr__(page, name, managed)
    arena.categories.add("IdentityBook.writable-tables")
