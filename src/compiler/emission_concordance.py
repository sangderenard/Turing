"""Emission as the last layer of the identity book (plan 100, part B).

Every backend that prints a ``Function`` posts what it printed beside the
text it already builds: one ``emission_unit`` row per unit of text, DERIVED
from the identity cells of the SSA values the unit spells (and the function's
``emission_function`` cell); one ``emission_function`` row per (function,
backend); one ``emission_artifact`` row per artifact part (module text ->
source file -> compile command -> library).  The emitter appends to its own
lists exactly as before and hands the SAME strings to the recorder, so the
emitted text is byte-identical with and without a book
(``tools/compiler_probes/probe_emission_chain.py`` checks it).

The book.  Backends run after ``end_identity_book``.  The only book an
emitter may post to is the one the compile attached to the module
(``module.metadata["identity_book"]``).  ``identity_book(module)`` falls back
to ``current_identity_book()`` when nothing is attached, which after the
compile closed mints a detached book and loses every post silently; this
module therefore reads the attached book directly and never calls either.  A
module with no attached book (or a detached one) posts nothing; the first such
emission in the process says so once on stderr (``no_book_at_emission``).

Row keys.  ``emission_unit`` and ``emission_function`` are keyed by the
function's SYMBOL, not by ``function_scope_of(function)``: a planned region
function carries its root's control scope (its values are the root's graph
ids, so they share one ``ssa_value`` scope), and two functions on one key
would interleave their unit ordinals.  Value cells are still read under
``function_scope_of``.  ``FUNCTION_SCOPE`` (plan 90 N1) is not declared on
this tree; the function's own root is the ``cell_set`` row ``(scope, 0)``
the control builder posts (``_function_root_cell`` posts the same row on
first need when the lowering never named it), and the ``emission_function``
row derives from it.

This module must not import the reducer (backends import it).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Iterable

from .concordance_declarations import (
    ARTIFACT_BUILD,
    BLOCK_ORIGIN_UNROUTED,
    CONTROL_VALUE_BINDING,
    EMISSION_ARTIFACT,
    EMISSION_FUNCTION,
    EMISSION_UNIT,
    FUNCTION_TEXT_PENDING,
    NO_FUNCTION_SCOPE,
    UNIT_ELIDED,
    VALUE_WITHOUT_IDENTITY_CELL,
    ArtifactFact,
    ArtifactPart,
    Backend,
    EmittedUnit,
    FunctionEmission,
    UnitKind,
)
from .identity_concordance import (
    ConcordanceRefusal,
    Derived,
    Mode,
    Ref,
    Unresolved,
    Unsourced,
)

_NO_BOOK_REPORTED = [False]


def emission_book(module: Any, what: str = "emission") -> Any:
    """The book the compile attached to ``module``, or None.

    Never ``identity_book(module)`` / ``current_identity_book()``: after the
    compile closed either would hand back a fresh detached book.  A detached
    book attached by hand (a probe's no-book run) counts as no book."""

    metadata = getattr(module, "metadata", None)
    book = None if metadata is None else metadata.get("identity_book")
    if book is None or getattr(book, "detached", False):
        if not _NO_BOOK_REPORTED[0]:
            _NO_BOOK_REPORTED[0] = True
            print(
                f"[emission] {what}: the module carries no attached identity "
                "book; emission posts nothing (no_book_at_emission)",
                file=sys.stderr,
            )
        return None
    return book


def value_cell(book: Any, function: Any, value: Any) -> Ref | None:
    """The identity cell of one SSA value (an ``SSAValue`` or an int id).

    ``ssa_value`` under ``function_scope_of(function)`` first (step 5/6:
    minted values and adopted graph ids); then, for a graph id the builder
    bound without adopting, its ``control_value_binding`` cell."""

    from .ssa_record_return_state import function_scope_of, ssa_value_identity_cell

    value_id = getattr(value, "id", value)
    if value_id is None or isinstance(value_id, bool) or not isinstance(value_id, int):
        return None
    cell = ssa_value_identity_cell(function, value_id, book=book)
    if cell is not None:
        return cell
    return book.latest_ref(
        CONTROL_VALUE_BINDING, (function_scope_of(function), int(value_id)),
    )


def unit_kind_of(op: Any) -> UnitKind:
    """The unit kind an instruction's own text is, by its declared op."""

    name = str(op)
    if name in {"Br", "br", "CondBr", "condbr"}:
        return UnitKind.BRANCH
    if name in {"Ret", "ret", "Return", "return"}:
        return UnitKind.RETURN
    if name in {"Call", "call"}:
        return UnitKind.CALL
    return UnitKind.STATEMENT


def _sha256(data: Any) -> tuple[str, int]:
    payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _distinct(cells: Iterable[Any]) -> tuple[Ref, ...]:
    return tuple(dict.fromkeys(cell for cell in cells if isinstance(cell, Ref)))


class _Span:
    """A cursor over one emitter list (``body``, ``entry_lines``): ``take``
    posts the lines appended since the cursor as one unit.  ``open`` names
    the instruction whose lines follow; ``close`` posts them (the emitter's
    ``continue`` paths all return to the loop top, where ``close`` runs)."""

    __slots__ = ("recorder", "lines", "at", "pending")

    def __init__(self, recorder: "EmissionRecorder", lines: list) -> None:
        self.recorder = recorder
        self.lines = lines
        self.at = len(lines)
        self.pending: tuple | None = None

    def take(
        self, kind: UnitKind, *, result: Any = None, args: Iterable = (),
        spelling: str = "", instruction: Any = None,
    ) -> Ref | None:
        if len(self.lines) <= self.at:
            return None
        text = "\n".join(self.lines[self.at:])
        self.at = len(self.lines)
        return self.recorder.unit(
            kind, text, result=result, args=args, spelling=spelling,
            instruction=instruction,
        )

    def open(self, instruction: Any) -> None:
        self.close()
        result = getattr(instruction, "res", None)
        self.pending = (
            unit_kind_of(getattr(instruction, "op", "")),
            result,
            tuple(getattr(instruction, "args", ()) or ()),
            "" if result is None else f"t{int(result.id)}",
            instruction,
        )

    def take_pending(self, kind: UnitKind | None = None, args: Iterable | None = None) -> Ref | None:
        """Post the open instruction's lines so far (under ``kind`` / ``args``
        when given) and keep the instruction open for what follows."""
        if self.pending is None:
            return self.take(kind or UnitKind.STATEMENT, args=args or ())
        pending_kind, result, pending_args, spelling, instruction = self.pending
        return self.take(
            kind or pending_kind, result=result,
            args=pending_args if args is None else args,
            spelling=spelling, instruction=instruction,
        )

    def close(self) -> Ref | None:
        if self.pending is None:
            return None
        ref = self.take_pending()
        self.pending = None
        return ref


class EmissionRecorder:
    """Posts one function's emission for one backend (plan 100, 4.3).

    ``book`` None: every method is a no-op that still counts, so the unit
    ordinals and counts are the same with and without a book."""

    def __init__(
        self, book: Any, function: Any, backend: Backend, *, stage: Any,
        symbol: str | None = None,
    ) -> None:
        self.book = book
        self.function = function
        self.backend = backend
        self.stage = stage
        self.key = str(function.name)
        self.symbol = str(symbol if symbol is not None else function.name)
        self.function_cell: Ref | None = None
        self.units: list[Ref] = []
        self.count = 0
        self.unsourced = 0
        self.unsourced_values: list[int] = []
        self.instruction_units: dict[int, Ref] = {}
        self.finished: Ref | None = None

    # ---------------------------------------------------------------- posts
    def _post(self, page: Any, row: tuple, fact: Any, provenance: Any) -> Ref:
        book = self.book
        latest = book.latest_ref(page, row)
        mode = Mode.CONCORD
        if latest is not None:
            incumbent = book.pages[page.name].cells.get((row, latest.column))
            if incumbent != fact:
                # Re-emission of the same function changed its text (another
                # root, another entry): a revision caused by the new
                # ``emission_function`` cell it derives from.
                mode = Mode.REVISE
        return book.post(
            page, row, fact, stage=self.stage, provenance=provenance, mode=mode,
        )

    def header(
        self, text: str, *, args: Iterable = (), spelling: str = "",
    ) -> Ref | None:
        """Post ``emission_function`` pending, then the FUNCTION_HEADER unit
        DERIVED from it and the formals' cells."""

        if self.book is not None:
            from .ssa_record_return_state import function_scope_of

            book = self.book
            row = (self.key, self.backend)
            pending = Unresolved(FUNCTION_TEXT_PENDING)
            latest = book.latest_ref(EMISSION_FUNCTION, row)
            incumbent = (
                None if latest is None
                else book.pages[EMISSION_FUNCTION.name].cells.get(
                    (row, latest.column)
                )
            )
            if incumbent == pending:
                # A header posted again with no finish in between (an
                # emission that stopped early): the pending cell stands.
                self.function_cell = latest
            else:
                # The lowering's root row, posted on first need by the same
                # helper module-level minters share (a straight-line lowering
                # that never named it has none yet).
                from .precompile_to_ssa import _function_root_cell

                root = _function_root_cell(book, function_scope_of(self.function))
                self.function_cell = book.post(
                    EMISSION_FUNCTION, row, pending, stage=self.stage,
                    provenance=(
                        Derived((root,)) if root is not None
                        else Unsourced(NO_FUNCTION_SCOPE)
                    ),
                    mode=Mode.REVISE,
                )
        return self.unit(
            UnitKind.FUNCTION_HEADER, text, args=args,
            spelling=spelling or self.symbol,
        )

    def unit(
        self, kind: UnitKind, text: str, *, result: Any = None,
        args: Iterable = (), spelling: str = "", extra_cells: Iterable = (),
        instruction: Any = None,
    ) -> Ref | None:
        """One unit of emitted text.  DERIVED from the function cell, the
        result's and every argument's identity cell, and ``extra_cells``;
        ``Unsourced(value_without_identity_cell)`` when some value has no
        cell; a BLOCK_LABEL is ``Unsourced(block_origin_unrouted)`` until
        ``ssa_block`` (plan 100, 2.6) is on the tree."""

        ordinal = self.count
        self.count += 1
        if self.book is None:
            return None
        values = ([result] if result is not None else []) + list(args or ())
        cells: list[Any] = [self.function_cell, *extra_cells]
        missing: list[int] = []
        for value in values:
            cell = value_cell(self.book, self.function, value)
            if cell is None:
                missing.append(int(getattr(value, "id", value)))
            else:
                cells.append(cell)
        if missing:
            provenance: Any = Unsourced(VALUE_WITHOUT_IDENTITY_CELL)
            self.unsourced += 1
            self.unsourced_values.extend(missing)
        elif kind is UnitKind.BLOCK_LABEL:
            provenance = Unsourced(BLOCK_ORIGIN_UNROUTED)
            self.unsourced += 1
        elif not _distinct(cells):
            # No header was posted before this unit (a programming error at
            # the emitter would raise here; the recorder says it instead).
            provenance = Unsourced(NO_FUNCTION_SCOPE)
            self.unsourced += 1
        else:
            provenance = Derived(_distinct(cells))
        ref = self._post(
            EMISSION_UNIT, (self.key, self.backend, ordinal),
            EmittedUnit(kind, str(text), str(spelling)), provenance,
        )
        self.units.append(ref)
        if instruction is not None:
            self.instruction_units.setdefault(id(instruction), ref)
        return ref

    def elided(self, instruction: Any, *, binding: Any = None) -> Ref | None:
        """An instruction the emitter skips because another unit spells it
        (an aggregate projection bound by its call): a row of its own,
        ``Unresolved(unit_elided, read=(the binding instruction's unit,))``."""

        ordinal = self.count
        self.count += 1
        if self.book is None:
            return None
        read = _distinct((self.instruction_units.get(id(binding)),))
        result = getattr(instruction, "res", None)
        cells: list[Any] = [self.function_cell, *read]
        cell = None if result is None else value_cell(self.book, self.function, result)
        if cell is not None:
            cells.append(cell)
        sources = _distinct(cells)
        ref = self._post(
            EMISSION_UNIT, (self.key, self.backend, ordinal),
            Unresolved(UNIT_ELIDED, read),
            Derived(sources) if sources else Unsourced(NO_FUNCTION_SCOPE),
        )
        self.units.append(ref)
        self.instruction_units.setdefault(id(instruction), ref)
        return ref

    def span(self, lines: list) -> _Span:
        return _Span(self, lines)

    def finish(self, text: str) -> Ref | None:
        """REVISE ``emission_function`` with the unit count and the text's
        hash, DERIVED from every unit cell of this emission."""

        if self.book is None or self.function_cell is None:
            return None
        digest, _length = _sha256(text)
        sources = _distinct(self.units) or (self.function_cell,)
        self.finished = self.book.post(
            EMISSION_FUNCTION, (self.key, self.backend),
            FunctionEmission(self.symbol, self.count, digest),
            stage=self.stage, provenance=Derived(sources), mode=Mode.REVISE,
        )
        return self.finished


def emission_recorder(
    book: Any, function: Any, backend: Backend, *, stage: Any,
    symbol: str | None = None,
) -> EmissionRecorder:
    return EmissionRecorder(book, function, backend, stage=stage, symbol=symbol)


def post_artifact_part(
    book: Any, artifact: str, backend: Backend, part: Any, *, data: Any,
    location: Iterable = (), sources: Iterable = (), reason: Any = None,
    stage: Any = ARTIFACT_BUILD,
) -> Ref | None:
    """One ``emission_artifact`` row: ``(artifact, backend, part)`` ->
    ``ArtifactFact(sha256 of data, byte length, location)``, REVISE.

    DERIVED from ``sources`` when every one is a cell; ``Unsourced(reason)``
    when some source is None (``reason`` names the missing hop).  A rebuild
    that names the same source cells and no newer one is not a new fact the
    api admits: the previous row stands and is returned."""

    if book is None:
        return None
    sources = tuple(sources)
    digest, length = _sha256(data)
    fact = ArtifactFact(digest, length, tuple(location))
    cells = _distinct(sources)
    if cells and all(isinstance(cell, Ref) for cell in sources):
        provenance: Any = Derived(cells)
    else:
        provenance = Unsourced(reason or VALUE_WITHOUT_IDENTITY_CELL)
    row = (str(artifact), backend, part)
    latest = book.latest_ref(EMISSION_ARTIFACT, row)
    try:
        return book.post(
            EMISSION_ARTIFACT, row, fact, stage=stage,
            provenance=provenance, mode=Mode.REVISE,
        )
    except ConcordanceRefusal:
        if latest is None:
            raise
        return latest


def path_location(path: Any) -> tuple:
    return tuple(Path(path).parts)


class ArtifactEmission:
    """What an artifact carries from emission to build: the book, the
    backend, and the MODULE_TEXT / BUFFER_ORDER cells its later parts derive
    from (plan 100, 4.2: SOURCE_FILE <- MODULE_TEXT, COMPILE_COMMAND <-
    SOURCE_FILE + PIECE_FILE, LIBRARY <- COMPILE_COMMAND)."""

    __slots__ = ("book", "backend", "module_text", "buffer_order")

    def __init__(
        self, book: Any, backend: Backend, module_text: Ref | None,
        buffer_order: Ref | None = None,
    ) -> None:
        self.book = book
        self.backend = backend
        self.module_text = module_text
        self.buffer_order = buffer_order

    def build(
        self, artifact: str, *, source_text: str, source_path: Any,
        command: Iterable, library_path: Any, pieces: Iterable = (),
        variant: str | None = None, extra_sources: Iterable = (),
    ) -> Ref | None:
        """Post SOURCE_FILE, PIECE_FILE per linked piece, COMPILE_COMMAND
        and LIBRARY for one build of this artifact.  ``variant`` suffixes
        the part labels (``standalone``); ``extra_sources`` are further
        (label, text, path, source cells) files the command compiles (the
        standalone host)."""

        from .concordance_declarations import PIECE_ARTIFACT_UNROUTED

        book = self.book
        if book is None:
            return None

        def part(member: ArtifactPart, detail: Any = None) -> Any:
            label: tuple = (member,)
            if detail is not None:
                label += (detail,)
            if variant is not None:
                label += (variant,)
            return label[0] if len(label) == 1 else label

        source = post_artifact_part(
            book, artifact, self.backend, part(ArtifactPart.SOURCE_FILE),
            data=source_text, location=path_location(source_path),
            sources=(self.module_text,), reason=FUNCTION_TEXT_PENDING,
        )
        compiled = [source]
        for label, text, path, cells in extra_sources:
            compiled.append(post_artifact_part(
                book, artifact, self.backend,
                part(ArtifactPart.SOURCE_FILE, label), data=text,
                location=path_location(path), sources=tuple(cells),
                reason=VALUE_WITHOUT_IDENTITY_CELL,
            ))
        for symbol, text, path in pieces:
            # A linked LLVM piece was emitted (if at all) on the book of the
            # compile that built it, not on this one.
            compiled.append(post_artifact_part(
                book, artifact, self.backend,
                part(ArtifactPart.PIECE_FILE, str(symbol)), data=text,
                location=path_location(path) if path is not None else (),
                sources=(None,), reason=PIECE_ARTIFACT_UNROUTED,
            ))
        command = tuple(map(str, command))
        compile_command = post_artifact_part(
            book, artifact, self.backend, part(ArtifactPart.COMPILE_COMMAND),
            data="\0".join(command), location=command,
            sources=tuple(compiled),
        )
        library = Path(library_path)
        return post_artifact_part(
            book, artifact, self.backend, part(ArtifactPart.LIBRARY),
            data=library.read_bytes() if library.is_file() else b"",
            location=path_location(library), sources=(compile_command,),
        )


__all__ = [
    "ArtifactEmission",
    "EmissionRecorder",
    "emission_book",
    "emission_recorder",
    "path_location",
    "post_artifact_part",
    "unit_kind_of",
    "value_cell",
]
