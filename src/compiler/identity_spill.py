"""Cold-page spill of the identity book.

The book grows monotonically through a compile and is then kept whole on
``module.metadata["identity_book"]`` for the native system's life.  Most of
it is read by no stage after the one that wrote it (``page_lifecycle`` says
which).  This module moves such a page -- and the edge rows it OWNS -- out of
RAM into the book's spill file, and brings it back when anything touches it.

THE UNIT.  A page ``P`` spills as a set of *parts*, each one segment of the
spill file:

``"page"``
    ``P``'s own tables: ``cells``, ``stamps``, ``columns`` (the indexes
    ``scopes``, ``row_columns`` and ``column_positions`` are derived from
    them on the way back, exactly as ``IdentityPage._stamp`` built them).
    The page keeps a ``spill`` marker -- its ``Segment`` (offset, sizes,
    cell and row counts) -- and its six table attributes are DELETED from the
    instance, so the first read of any of them (``cells``, ``stamps``,
    ``columns``, ``scopes``, ``row_columns``, ``column_positions``) goes to
    ``IdentityPage.__getattr__``, which reads the segment back.  Every read
    api of the page (``cell``, ``latest``, ``history``, ``rows``, ``spans``,
    ``scope_rows``, the detectors, the viewer, ``render_identity_book``)
    reaches the tables through those attributes, so none of them needs to
    know the page can be away.  A write reloads first
    (``IdentityBook._ensure_owner_whole``).

``"concordance_edge"``, ``"concordance_dependents"``, ``"concordance_mint"``,
``"concordance_unsourced"``
    The rows of the book's private edge pages that ``P`` owns: an edge row is
    owned by its TARGET cell's page, a dependents row by its SOURCE cell's
    page, a mint row by its target's page, an unsourced row by the page it
    tags.  They are cut out of the live private page (which stays
    resident, holding every other owner's rows) and written as a partition.
    ``ordinals`` records, for each cut cell, its ABSOLUTE position in the
    private page's insertion order (positions are never reused: nothing is
    ever deleted), so a partition merges back into exactly the order the
    unspilled page had -- the log of a spilled book lists the edge pages in
    the order an unspilled book does.

THE FILE.  One append-only file per book, one lzma stream per segment
(``identity_log_lzma_preset``, the identity log's preset and framing), with
an index ``(part, owner) -> Segment(offset, stored bytes, ...)`` kept on the
book and mirrored line by line to ``<file>.index``.  The payload of a stream is
a pickle, not rendered lines: a log line is ``repr`` of a fact and cannot
bring the fact back.  Registered pages, stages, transforms and reasons are
written as references to the registry's own objects so a reloaded ``Ref``
points at the same ``Page`` the registry holds.  The file is removed when the
book is collected.

WHAT IS NEVER DONE.  Nothing is deleted from the book: a spilled page's
cells, stamps and edges are all in its segments and every one of them is
readable through the ordinary api.  A page whose tables something else is
holding a reference to (``sys.getrefcount``: a caller kept ``page.cells``),
or whose facts do not pickle, is REFUSED -- left in RAM, with a ``refuse``
receipt -- rather than risk a writer holding a stale table.  Pages the
spill machinery itself reads and writes are never spilled (``NEVER_SPILLED``).

RECEIPTS.  Every spill, reload and refusal is a ``page_spill_receipt`` row on
the book (``concordance_declarations``), DERIVED from the ``compile_policy``
cell that licensed it, so the decision is on the book.  A *borrow*
(``IdentityBook.borrowed``: the whole-book scanners -- the unsourced detector
and the log -- reading one spilled page and letting it go) changes nothing on
the book and posts nothing: posting while a scan walks the book would change
what the scan sees.
"""
from __future__ import annotations

import copyreg
import heapq
import io
import lzma
import os
import pickle
import sys
import tempfile
import weakref
from array import array
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from typing import Any, Iterable

#: The six tables of an ``IdentityPage`` that a spill deletes from the
#: instance; reading any of them reloads the page.
SPILLABLE_ATTRS = (
    "cells", "stamps", "columns", "scopes", "row_columns", "column_positions",
)
SPILL_DIR_ENV = "TURING_IDENTITY_SPILL_DIR"

#: The part naming a page's own tables (the other parts are private page names).
PAGE = "page"
#: The private edge pages whose rows are owned by the page their cells name.
PARTITIONED = (
    "concordance_edge", "concordance_dependents", "concordance_mint",
    "concordance_unsourced",
)
#: Pages the machinery reads or writes while it spills or reloads (and the
#: scope bookkeeping every post consults): resident for the book's life.
NEVER_SPILLED = frozenset({
    "compile_policy", "memory_release_receipt", "page_spill_receipt",
    "scope_registry", "scope_origin", "shape_scope_function",
    *PARTITIONED,
})


def never_spilled(book: Any, name: str) -> bool:
    return name in NEVER_SPILLED or name in book.registry.private_pages


def owner_of_row(row: Any) -> str | None:
    """The page that owns a row of a private edge page: the page its first
    element (the row's scope) names -- ``(page, row, column)`` for a cell key,
    the page's name itself for an unsourced row."""

    if type(row) is not tuple or not row:
        return None
    scope = row[0]
    if type(scope) is str:
        return scope
    if type(scope) is tuple and scope and type(scope[0]) is str:
        return scope[0]
    return None


@dataclass(frozen=True)
class Segment:
    """One lzma stream in the spill file: a page's own tables (``kind ==
    "page"``) or one owner's partition of a private page (``kind`` its name,
    ``ordinals`` the cut cells' absolute positions in that page's order)."""

    kind: str
    owner: str
    offset: int
    stored: int           # bytes in the spill file (compressed)
    raw: int              # bytes of the pickle stream (uncompressed)
    cells: int
    rows: int
    ordinals: Any = None


# ------------------------------------------------------------------ pickling
_LOAD_REGISTRY: Any = None


def _registry_object(kind: str, name: str) -> Any:
    registry = _LOAD_REGISTRY
    table = {
        "page": registry.pages, "stage": registry.stages,
        "transform": registry.transforms, "reason": registry.reasons,
    }[kind]
    return table[name]


def _dispatch_table(registry: Any) -> dict:
    from .identity_concordance import Page, Reason, Stage, Transform

    def reducer(kind: str, table: dict):
        def reduce(obj: Any):
            if table.get(obj.name) is obj:
                return _registry_object, (kind, obj.name)
            return object.__reduce_ex__(obj, pickle.HIGHEST_PROTOCOL)
        return reduce

    return {
        Page: reducer("page", registry.pages),
        Stage: reducer("stage", registry.stages),
        Transform: reducer("transform", registry.transforms),
        Reason: reducer("reason", registry.reasons),
    }


class _CountingWriter:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.count = 0

    def write(self, data: Any) -> int:
        self.count += len(data)
        return self.stream.write(data)


class BookSpill:
    """The book's spill file, its index and its bookkeeping."""

    def __init__(self, registry: Any, directory: str | None = None) -> None:
        directory = directory or os.environ.get(SPILL_DIR_ENV) or None
        descriptor, self.path = tempfile.mkstemp(
            prefix="identity_book_spill_", suffix=".xz.segments",
            dir=directory,
        )
        os.close(descriptor)
        self.index_path = self.path + ".index"
        self.registry = registry
        #: ``(part, owner)`` -> the latest segment written for it.
        self.index: dict[tuple[str, str], Segment] = {}
        #: owner -> the parts of it that are out of RAM right now.
        self.detached: dict[str, set[str]] = {}
        #: owner -> why it can never be spilled (its facts do not serialise):
        #: it is not asked again.
        self.refused: dict[str, str] = {}
        #: owner -> why it is held right now (a reader keeps a table of it):
        #: asked again at the next boundary, receipted once per holding.
        self.held: dict[str, str] = {}
        #: Receipts posted so far: the next receipt's ordinal.
        self.receipts = 0
        #: owner -> (its latest spill receipt Ref, the policy trigger it
        #: derives from): a reload is derived from both.
        self.spill_cells: dict[str, tuple[Any, str]] = {}
        #: trigger -> the ``compile_policy`` Ref declared for it.
        self.policy_cells: dict[str, Any] = {}
        #: Counters, for a reader of the book that wants the totals.
        self.spills = self.reloads = self.refusals = self.borrows = 0
        #: The budget a ``"budget"`` spill declared its policy cell with.
        self.budget_bytes: int | None = None
        self._finalizer = weakref.finalize(
            self, _remove_files, self.path, self.index_path,
        )

    # ------------------------------------------------------------------ file
    def size(self) -> int:
        return os.path.getsize(self.path)

    def truncate(self, size: int) -> None:
        with open(self.path, "r+b") as handle:
            handle.truncate(size)

    def write(
        self, kind: str, owner: str, payload: Any, *, cells: int, rows: int,
        ordinals: Any = None,
    ) -> Segment:
        from .identity_concordance import identity_log_lzma_preset

        with open(self.path, "r+b") as handle:
            handle.seek(0, os.SEEK_END)
            offset = handle.tell()
            try:
                stream = lzma.LZMAFile(
                    handle, "wb", preset=identity_log_lzma_preset(),
                )
                counted = _CountingWriter(stream)
                try:
                    pickler = pickle.Pickler(
                        counted, protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    pickler.dispatch_table = _dispatch_table(self.registry)
                    pickler.dump(payload)
                finally:
                    stream.close()
                handle.flush()
                stored = handle.tell() - offset
            except BaseException:
                handle.truncate(offset)
                raise
        segment = Segment(
            kind, owner, offset, stored, counted.count, cells, rows, ordinals,
        )
        self.index[(kind, owner)] = segment
        with open(self.index_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{kind}\t{owner}\t{offset}\t{stored}\t{counted.count}\t"
                f"{cells}\t{rows}\n"
            )
        return segment

    def read(self, segment: Segment) -> Any:
        global _LOAD_REGISTRY
        with open(self.path, "rb") as handle:
            handle.seek(segment.offset)
            data = handle.read(segment.stored)
        previous, _LOAD_REGISTRY = _LOAD_REGISTRY, self.registry
        try:
            with lzma.LZMAFile(io.BytesIO(data)) as stream:
                return pickle.Unpickler(stream).load()
        finally:
            _LOAD_REGISTRY = previous


def _remove_files(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------ the book
def ensure_declared(book: Any) -> None:
    """Declare, on the book's registry, what a receipt needs (a no-op for the
    module registry, which has it all)."""

    from .concordance_declarations import (
        COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, MEMORY_REGULATION,
        PAGE_SPILL_RECEIPT, POLICY_DECLARATION,
    )

    registry = book.registry
    if registry.pages.get(PAGE_SPILL_RECEIPT.name) == PAGE_SPILL_RECEIPT:
        return
    for page in (COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, PAGE_SPILL_RECEIPT):
        registry.declare_page(
            page.name, page.row_fields, page.fact_type,
            rows_level=page.rows_level,
        )
    registry.declare_stage(MEMORY_REGULATION.name)
    registry.declare_transform(POLICY_DECLARATION.name, POLICY_DECLARATION.arity)


def spill_store(book: Any) -> BookSpill:
    spill = book.__dict__.get("_spill")
    if spill is None:
        ensure_declared(book)
        spill = book._spill = BookSpill(book.registry)
    return spill


def _policy_cell(book: Any, trigger: str, budget_bytes: int | None) -> Any:
    from .concordance_declarations import (
        COMPILE_POLICY, MEMORY_REGULATION, POLICY_DECLARATION,
        SPILL_COLD_PAGES_POLICY,
    )
    from .identity_concordance import Mode, Novel

    spill = book._spill
    ref = spill.policy_cells.get(trigger)
    if ref is None:
        if trigger == "budget":
            row, fact = ("memory_budget_bytes",), str(budget_bytes)
        elif trigger == "policy":
            row, fact = SPILL_COLD_PAGES_POLICY, "true"
        else:
            row, fact = ("spill_explicit",), "true"
        book.post(
            COMPILE_POLICY, row, fact, stage=MEMORY_REGULATION,
            provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
        )
        ref = spill.policy_cells[trigger] = book.latest_ref(COMPILE_POLICY, row)
    return ref


def _post_receipt(
    book: Any, action: str, owner: str, segments: Iterable[Segment],
    trigger: str, boundary: str, *, policy_trigger: str,
    budget_bytes: int | None = None, cells: int | None = None,
) -> Any:
    """Post one ``page_spill_receipt`` row, DERIVED from the policy cell (and,
    for a reload, from the page's latest spill receipt)."""

    from .concordance_declarations import (
        MEMORY_REGULATION, PAGE_SPILL_RECEIPT, PageSpill,
    )
    from .identity_concordance import Derived, Mode

    spill = book._spill
    segments = tuple(segments)
    saved = (book._stamp_override, book._materialising)
    # A receipt posted from inside a materialisation (a read through a fork
    # that reloaded a page) is an event of its own: it ticks the clock and
    # does not take the fork's stamp.
    book._stamp_override = None
    book._materialising = None
    try:
        sources = [_policy_cell(book, policy_trigger, budget_bytes)]
        if action == "reload" and owner in spill.spill_cells:
            sources.append(spill.spill_cells[owner][0])
        offsets = tuple(segment.offset for segment in segments)
        fact = PageSpill(
            action, owner, tuple(segment.kind for segment in segments),
            offsets[0] if offsets else -1, offsets,
            sum(segment.stored for segment in segments),
            sum(segment.raw for segment in segments),
            sum(segment.cells for segment in segments)
            if cells is None else cells,
            sum(segment.rows for segment in segments),
            trigger, boundary,
        )
        row = (owner, spill.receipts)
        spill.receipts += 1
        ref = book.post(
            PAGE_SPILL_RECEIPT, row, fact, stage=MEMORY_REGULATION,
            provenance=Derived(tuple(sources)), mode=Mode.CONCORD,
        )
    finally:
        book._stamp_override, book._materialising = saved
    if action == "spill":
        spill.spill_cells[owner] = (ref, policy_trigger)
    return ref


# --------------------------------------------------------- private partitions
def _detached_ordinals(page: Any) -> list:
    detached = page.__dict__.get("_detached")
    if not detached:
        return []
    return sorted(chain.from_iterable(
        segment.ordinals for segment in detached.values()
    ))


def _extract(page: Any, owners: set) -> dict[str, tuple[list, list, list, list]]:
    """One pass over a private page's cells in insertion order: for each
    cell owned by one of ``owners``, its absolute ordinal, key, fact, stamp.

    The absolute ordinal of a resident cell is its index in the page's whole
    insertion order, detached cells included: walking the resident cells, the
    positions held by detached cells are skipped."""

    detached = _detached_ordinals(page)
    count = len(detached)
    cells = page.__dict__["cells"]
    stamps = page.__dict__["stamps"]
    found = {owner: ([], [], [], []) for owner in owners}
    position = 0
    index = 0
    for key, fact in cells.items():
        while index < count and detached[index] == position:
            position += 1
            index += 1
        owner = owner_of_row(key[0])
        if owner in found:
            ordinals, keys, facts, stamp_list = found[owner]
            ordinals.append(position)
            keys.append(key)
            facts.append(fact)
            stamp_list.append(stamps[key])
        position += 1
    return found


def _reindex_in_place(page: Any) -> None:
    """Rebuild ``scopes`` and ``row_columns`` of a page from its cells, in
    the dicts it already has (a holder of the dicts keeps a live view)."""

    state = page.__dict__
    positions = state["column_positions"]
    row_columns = state["row_columns"]
    scopes = state["scopes"]
    row_columns.clear()
    scopes.clear()
    for row, column in state["cells"]:
        row_columns.setdefault(row, []).append(column)
        if type(row) is tuple and row:
            scopes.setdefault(row[0], {}).setdefault(row, None)
    for columns in row_columns.values():
        if len(columns) > 1:
            columns.sort(key=positions.__getitem__)


def _replace_cells(page: Any, items: Iterable[tuple[Any, Any, Any]]) -> None:
    """Make a private page hold exactly ``items`` (key, fact, stamp), in that
    order, in its existing dicts."""

    state = page.__dict__
    cells, stamps = state["cells"], state["stamps"]
    new_cells: dict = {}
    new_stamps: dict = {}
    for key, fact, stamp in items:
        if key in new_cells:
            raise RuntimeError(
                f"{page.name}: row {key!r} is both resident and spilled"
            )
        new_cells[key] = fact
        new_stamps[key] = stamp
    cells.clear()
    cells.update(new_cells)
    stamps.clear()
    stamps.update(new_stamps)
    del new_cells, new_stamps
    _reindex_in_place(page)


def _drop_owners(page: Any, owners: set) -> None:
    cells = page.__dict__["cells"]
    stamps = page.__dict__["stamps"]
    keep = [
        (key, fact, stamps[key]) for key, fact in cells.items()
        if owner_of_row(key[0]) not in owners
    ]
    _replace_cells(page, keep)


# ----------------------------------------------------------------- spilling
@dataclass
class SpillResult:
    spilled: tuple = ()
    refused: tuple = ()
    cells_dropped: int = 0
    receipts: tuple = ()


def spill_pages(
    book: Any, names: Iterable[str], *, trigger: str = "explicit",
    budget_bytes: int | None = None, boundary: str = "",
) -> SpillResult:
    """Spill the named pages (and the edge rows they own) out of RAM.

    ``trigger``: ``"policy"`` (the work contract's ``spill_cold_pages``),
    ``"budget"`` (a boundary over ``memory_budget_bytes``) or ``"explicit"``.
    A name that is not a spillable page, is already away, holds nothing, or
    was declined before is skipped; a page that cannot be dropped safely is
    refused (receipt ``refuse``)."""

    spill = spill_store(book)
    if budget_bytes is not None:
        spill.budget_bytes = budget_bytes
    pages = book.pages
    candidates: list[str] = []
    refused: list[tuple[str, str]] = []
    for name in dict.fromkeys(names):
        page = dict.get(pages, name)
        if page is None or never_spilled(book, name):
            continue
        state = page.__dict__
        if state.get("spill") is not None or "cells" not in state:
            continue
        if name in spill.refused:
            continue
        if not state["cells"]:
            continue    # a page with no cell owns no edge either
        held = [
            attribute for attribute in SPILLABLE_ATTRS
            if sys.getrefcount(state[attribute]) > 2
        ]
        if held:
            reason = f"held: {', '.join(held)} referenced outside the page"
            if spill.held.get(name) != reason:
                spill.held[name] = reason
                refused.append((name, reason))
            continue
        spill.held.pop(name, None)
        candidates.append(name)

    wanted = set(candidates)
    extracted: dict[str, dict] = {}
    for part in PARTITIONED:
        private = dict.get(pages, part)
        if private is not None and wanted and private.__dict__["cells"]:
            extracted[part] = _extract(private, wanted)

    committed: list[tuple[str, list[Segment]]] = []
    for name in candidates:
        page = pages[name]
        state = page.__dict__
        start = spill.size()
        segments: list[Segment] = []
        try:
            segments.append(spill.write(
                PAGE, name,
                {"cells": state["cells"], "stamps": state["stamps"],
                 "columns": state["columns"]},
                cells=len(state["cells"]), rows=len(state["row_columns"]),
            ))
            for part, per_owner in extracted.items():
                ordinals, keys, facts, stamps = per_owner[name]
                if not keys:
                    continue
                segments.append(spill.write(
                    part, name,
                    {"ordinals": ordinals, "keys": keys, "facts": facts,
                     "stamps": stamps},
                    cells=len(keys), rows=len({key[0] for key in keys}),
                    ordinals=array("Q", ordinals),
                ))
        except Exception as error:  # a fact that does not serialise
            spill.truncate(start)
            reason = f"unserialisable: {type(error).__name__}: {error}"
            spill.refused[name] = reason
            refused.append((name, reason))
            continue
        committed.append((name, segments))

    dropped = 0
    receipts: list[Any] = []
    names_by_part: dict[str, set] = {}
    for name, segments in committed:
        page = pages[name]
        state = page.__dict__
        page_segment = segments[0]
        for attribute in SPILLABLE_ATTRS:
            del state[attribute]
        state["spill"] = page_segment
        detached = spill.detached.setdefault(name, set())
        detached.add(PAGE)
        dropped += page_segment.cells
        for segment in segments[1:]:
            names_by_part.setdefault(segment.kind, set()).add(name)
            detached.add(segment.kind)
            dropped += segment.cells
    for part, owners in names_by_part.items():
        private = dict.get(pages, part)
        _drop_owners(private, owners)
        table = private.__dict__.setdefault("_detached", {})
        for name, segments in committed:
            for segment in segments[1:]:
                if segment.kind == part:
                    table[name] = segment
    del extracted
    if names_by_part:
        book._refresh_page_table()

    for name, segments in committed:
        receipts.append(_post_receipt(
            book, "spill", name, segments, trigger, boundary,
            policy_trigger=trigger, budget_bytes=budget_bytes,
        ))
        spill.spills += 1
    for name, reason in refused:
        receipts.append(_post_receipt(
            book, "refuse", name, (), reason, boundary,
            policy_trigger=trigger, budget_bytes=budget_bytes, cells=0,
        ))
        spill.refusals += 1
    return SpillResult(
        tuple(name for name, _ in committed), tuple(refused), dropped,
        tuple(receipts),
    )


# ---------------------------------------------------------------- restoring
def _load_page(book: Any, owner: str) -> Segment:
    page = dict.get(book.pages, owner)
    state = page.__dict__
    segment = state.pop("spill")
    payload = book._spill.read(segment)
    columns = payload["columns"]
    state["cells"] = payload["cells"]
    state["stamps"] = payload["stamps"]
    state["columns"] = columns
    state["column_positions"] = {
        column: position for position, column in enumerate(columns)
    }
    state["row_columns"] = {}
    state["scopes"] = {}
    _reindex_in_place(page)
    return segment


def _merge_partitions(book: Any, part: str, owners: Iterable[str]) -> list[Segment]:
    """Merge the named owners' partitions of private page ``part`` back into
    it, restoring the insertion order the unspilled page had."""

    page = dict.get(book.pages, part)
    table = page.__dict__["_detached"]
    owners = [owner for owner in owners if owner in table]
    if not owners:
        return []
    everything = _detached_ordinals(page)
    state = page.__dict__
    stamps = state["stamps"]
    count = len(everything)

    def resident():
        position = 0
        index = 0
        for key, fact in state["cells"].items():
            while index < count and everything[index] == position:
                position += 1
                index += 1
            yield position, key, fact, stamps[key]
            position += 1

    segments = [table.pop(owner) for owner in owners]
    streams = [resident()]
    for segment in segments:
        payload = book._spill.read(segment)
        streams.append(zip(
            payload["ordinals"], payload["keys"], payload["facts"],
            payload["stamps"],
        ))
        del payload
    merged = heapq.merge(*streams, key=itemgetter(0))
    _replace_cells(page, ((key, fact, stamp) for _, key, fact, stamp in merged))
    if not table:
        del state["_detached"]
        book._refresh_page_table()
    return segments


def restore(book: Any, wanted: dict[str, Iterable[str]], reason: str) -> int:
    """Bring parts of owners back into RAM.  ``wanted``: owner -> the parts to
    load (``None`` for all of the owner's parts that are away).  One reload
    receipt per owner.  Returns the cells restored."""

    spill = book.__dict__.get("_spill")
    if spill is None:
        return 0
    loaded: dict[str, list[Segment]] = {}
    by_part: dict[str, list[str]] = {}
    for owner, parts in wanted.items():
        away = spill.detached.get(owner)
        if not away:
            continue
        parts = away if parts is None else set(parts) & away
        for part in sorted(parts, key=lambda item: item != PAGE):
            if part == PAGE:
                loaded.setdefault(owner, []).append(_load_page(book, owner))
            else:
                by_part.setdefault(part, []).append(owner)
    for part, owners in by_part.items():
        for owner, segment in zip(
            owners, _owner_segments(book, part, owners),
        ):
            loaded.setdefault(owner, []).append(segment)
    restored = 0
    for owner, segments in loaded.items():
        away = spill.detached[owner]
        for segment in segments:
            away.discard(segment.kind)
        if not away:
            del spill.detached[owner]
    for owner, segments in loaded.items():
        spill.reloads += 1
        restored += sum(segment.cells for segment in segments)
        _post_receipt(
            book, "reload", owner, segments, reason, "",
            policy_trigger=spill.spill_cells.get(owner, (None, "explicit"))[1],
            budget_bytes=spill.budget_bytes,
        )
    return restored


def _owner_segments(book: Any, part: str, owners: list[str]) -> list[Segment]:
    page = dict.get(book.pages, part)
    segments = [page.__dict__["_detached"][owner] for owner in owners]
    _merge_partitions(book, part, owners)
    return segments


def restore_all(book: Any, reason: str) -> int:
    spill = book.__dict__.get("_spill")
    if spill is None or not spill.detached:
        return 0
    return restore(book, {owner: None for owner in tuple(spill.detached)}, reason)


def make_whole(book: Any, part: str, reason: str = "access") -> int:
    """Reload every owner's partition of private page ``part``."""

    page = dict.get(book.pages, part)
    if page is None:
        return 0
    table = page.__dict__.get("_detached")
    if not table:
        return 0
    return restore(book, {owner: {part} for owner in tuple(table)}, reason)
