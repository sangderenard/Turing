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
emission in the process says so once on stderr (``no_book_at_emission``), and
every MODULE_TEXT it would have posted is kept on ``module.metadata
["emission_gaps"]`` (artifact, backend, text hash, length, emitter) so that
``replay_emission_gaps`` can post each one as ``Unsourced(NO_BOOK_AT_EMISSION)``
once a book is attached.  The gap is not posted on a detached ambient book:
reaching one needs ``current_identity_book()``, which emission never calls.

Scopes that are not lowerings.  ``EmissionRecorder(scope_cells=...)`` derives
the ``emission_function`` row from the given cells instead of the lowering's
``cell_set`` root: a tensor reference kernel imported from authored LLVM text
(its callers' ``emission_function`` rows, its call instructions' operand
cells, and the kernel text's own ``emission_artifact`` KERNEL_SOURCE root) and
a native loop wrapper (the wrapped root's ``emission_function`` row).  With
``local_values=True`` (the imported kernels) a value with no identity cell is
the kernel body's own: the unit derives from the kernel's scope cell, and the
recorder counts it in ``local_values``.

Kernel and library text pulled into an LLVM module by symbol is one
``KERNEL_TEXT`` unit per symbol (``EmissionRecorder.kernel_texts``), DERIVED
from the symbol's origin row (``kernel_source_cell``: NOVEL
``AUTHORED_KERNEL_TEXT`` root for authored text, the piece's
``Unsourced(PIECE_ARTIFACT_UNROUTED)`` row for a linked piece), every unit
whose text calls the symbol, and the cells of the call instructions that name
it.

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
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .concordance_declarations import (
    ARTIFACT_BUILD,
    AUTHORED_KERNEL_TEXT,
    BLOCK_ORIGIN_UNROUTED,
    CONTROL_VALUE_BINDING,
    EMISSION_ARTIFACT,
    EMISSION_FUNCTION,
    EMISSION_UNIT,
    FUNCTION_OUTPUT,
    FUNCTION_PARAMETER,
    FUNCTION_TEXT_PENDING,
    NATIVE_LOOP_VALUE,
    NATIVE_LOOP_WRAPPER_VALUE,
    NO_BOOK_AT_EMISSION,
    NO_FUNCTION_SCOPE,
    PIECE_ARTIFACT_UNROUTED,
    PROGRAM_ABI_FIELD_SLOT,
    SSA_VALUE,
    UNIT_ELIDED,
    VALUE_WITHOUT_IDENTITY_CELL,
    ArtifactFact,
    ArtifactPart,
    Backend,
    EmittedUnit,
    FunctionEmission,
    LayoutBuffer,
    LayoutKind,
    NativeLoopValue,
    ProgramAbiSlot,
    ProgramAbiSlotRole,
    UnitKind,
)
from .identity_concordance import (
    ConcordanceRefusal,
    Derived,
    Mode,
    Novel,
    Ref,
    Unresolved,
    Unsourced,
)

_NO_BOOK_REPORTED = [False]

#: ``module.metadata`` key of the MODULE_TEXT posts an unbooked emission
#: could not make (see ``replay_emission_gaps``).
EMISSION_GAPS = "emission_gaps"

#: An LLVM call site's callee symbol, as the backends' own closure scans
#: spell it (``_emit_repository_call_module`` / the single-block lane).
_SYMBOL_REFERENCE = re.compile(r"@([A-Za-z_$.-][\w$.-]*)\s*\(")


def emission_book(module: Any, what: str = "emission") -> Any:
    """The book the compile attached to ``module``, or None.

    Never ``identity_book(module)`` / ``current_identity_book()``: with no
    book attached the first falls through to the second, which after the
    compile closed hands back a fresh detached book.  A detached book attached
    by hand (a probe's no-book run) counts as no book."""

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


def replay_emission_gaps(module: Any) -> tuple[Ref, ...]:
    """Post every MODULE_TEXT an unbooked emission of ``module`` kept on its
    metadata, now that a book is attached: one ``emission_artifact`` row
    ``(artifact, backend, (MODULE_TEXT, NO_BOOK_AT_EMISSION.name))`` per gap,
    ``Unsourced(NO_BOOK_AT_EMISSION)``, the fact the text's hash and length.
    The replayed gaps leave the metadata; with no book attached nothing is
    posted and the gaps stay."""

    metadata = getattr(module, "metadata", None)
    if metadata is None or not metadata.get(EMISSION_GAPS):
        return ()
    book = emission_book(module, "replay_emission_gaps")
    if book is None:
        return ()
    posted: list[Ref] = []
    for artifact, backend, digest, length, what in tuple(metadata[EMISSION_GAPS]):
        posted.append(book.post(
            EMISSION_ARTIFACT,
            (str(artifact), backend, (ArtifactPart.MODULE_TEXT, NO_BOOK_AT_EMISSION.name)),
            ArtifactFact(digest, length, (str(what),)),
            stage=ARTIFACT_BUILD, provenance=Unsourced(NO_BOOK_AT_EMISSION),
            mode=Mode.REVISE,
        ))
    metadata[EMISSION_GAPS] = []
    return tuple(posted)


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


def ssa_block_cell(book: Any, function: Any, label: Any) -> Ref | None:
    """The ``ssa_block`` cell of ``function``'s block ``label`` (plan 100,
    2.6), or None.  Always with the attached ``book``: without it the
    reader would fall back to the ambient one, which emission never reads."""

    from .ssa_record_return_state import ssa_block_identity_cell

    if book is None or label is None or function is None:
        return None
    return ssa_block_identity_cell(function, str(label), book=book)


def kernel_source_cell(
    book: Any, symbol: str, text: str, *, origin: Iterable = (),
    stage: Any = None,
) -> Ref | None:
    """The origin row of authored kernel / library text pulled in by symbol:
    ``emission_artifact (symbol, LLVM_MODULE, KERNEL_SOURCE)``, a NOVEL
    ``AUTHORED_KERNEL_TEXT`` root (the text is authored input, as a source
    span is), ``location`` = ``origin`` (the table it was read from).  Every
    backend that spells the kernel derives from this one row: the LLVM text
    pulled in by symbol, and the C function imported from the same text."""

    if book is None:
        return None
    from .concordance_declarations import EMISSION_LLVM

    digest, length = _sha256(text)
    fact = ArtifactFact(digest, length, tuple(map(str, origin)))
    row = (str(symbol), Backend.LLVM_MODULE, ArtifactPart.KERNEL_SOURCE)
    latest = book.latest_ref(EMISSION_ARTIFACT, row)
    if latest is not None and book.pages[EMISSION_ARTIFACT.name].latest(row) == fact:
        return latest
    return book.post(
        EMISSION_ARTIFACT, row, fact, stage=stage or EMISSION_LLVM,
        provenance=Novel(AUTHORED_KERNEL_TEXT, ()),
        mode=Mode.REVISE if latest is not None else Mode.CONCORD,
    )


def piece_source_cell(
    book: Any, artifact: str, piece: str, text: str, *, stage: Any = None,
) -> Ref | None:
    """A linked LLVM piece's text inside ``artifact``: the piece was emitted
    (if at all) on the book of the compile that built it, so its row here is
    ``Unsourced(PIECE_ARTIFACT_UNROUTED)``, as ``ArtifactEmission.build``
    posts it for C."""

    from .concordance_declarations import EMISSION_LLVM

    return post_artifact_part(
        book, artifact, Backend.LLVM_MODULE, (ArtifactPart.PIECE_FILE, str(piece)),
        data=text, sources=(None,), reason=PIECE_ARTIFACT_UNROUTED,
        stage=stage or EMISSION_LLVM,
    )


def call_demands(
    book: Any, functions: Iterable, symbols: Iterable[str],
) -> dict[str, tuple[Ref, ...]]:
    """symbol -> the cells of the call instructions (in ``functions``) whose
    declared ``callee`` is that symbol: the result's cell, else the
    operands' cells."""

    wanted = {str(symbol) for symbol in symbols}
    found: dict[str, list[Ref]] = {}
    if book is None:
        return {}
    for function in functions:
        for block in function.blocks.values():
            for instruction in block.instrs:
                callee = (getattr(instruction, "attributes", None) or {}).get("callee")
                if callee is None or str(callee) not in wanted:
                    continue
                cells = []
                result = getattr(instruction, "res", None)
                cell = None if result is None else value_cell(book, function, result)
                if cell is not None:
                    cells.append(cell)
                else:
                    cells.extend(
                        value_cell(book, function, argument)
                        for argument in instruction.args
                    )
                found.setdefault(str(callee), []).extend(cells)
    return {symbol: _distinct(cells) for symbol, cells in found.items()}


def calling_units(recorders: Iterable, symbol: str) -> tuple[Ref, ...]:
    """Every unit (of ``recorders``) whose text calls ``@symbol(``."""

    symbol = str(symbol)
    return _distinct(
        ref
        for recorder in recorders
        for ref, text in recorder.texts
        if symbol in _SYMBOL_REFERENCE.findall(text)
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
        spelling: str = "", instruction: Any = None, block: Any = None,
        block_cell: Ref | None = None, extra_cells: Iterable = (),
    ) -> Ref | None:
        if len(self.lines) <= self.at:
            return None
        text = "\n".join(self.lines[self.at:])
        self.at = len(self.lines)
        return self.recorder.unit(
            kind, text, result=result, args=args, spelling=spelling,
            instruction=instruction, block=block, block_cell=block_cell,
            extra_cells=extra_cells,
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
        symbol: str | None = None, key: Any = None,
        scope_cells: Iterable | None = None, local_values: bool = False,
    ) -> None:
        self.book = book
        self.function = function
        self.backend = backend
        self.stage = stage
        self.key = str(function.name) if key is None else key
        self.symbol = str(symbol if symbol is not None else function.name)
        # Not a lowering: the function row derives from these cells (see the
        # module docstring, "Scopes that are not lowerings").
        self.scope_cells = None if scope_cells is None else _distinct(scope_cells)
        self.local_values = 0 if local_values else None
        self.function_cell: Ref | None = None
        self.units: list[Ref] = []
        self.texts: list[tuple[Ref, str]] = []
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

    def open_function(self) -> Ref | None:
        """Post ``emission_function`` pending (once per emission) and return
        its cell.  ``header`` calls it; an emitter calls it early for a
        function whose cell another function's scope derives from before
        that function's own header is reached (a kernel's later caller)."""

        if self.book is None:
            return None
        if self.function_cell is not None:
            return self.function_cell
        book = self.book
        row = (self.key, self.backend)
        pending = Unresolved(FUNCTION_TEXT_PENDING)
        latest = book.latest_ref(EMISSION_FUNCTION, row)
        incumbent = (
            None if latest is None
            else book.pages[EMISSION_FUNCTION.name].cells.get((row, latest.column))
        )
        if incumbent == pending:
            # Posted pending already with no finish in between (an emission
            # that stopped early, or ``open_function`` called ahead of the
            # header): the pending cell stands.
            self.function_cell = latest
            return latest
        if self.scope_cells is not None:
            sources = self.scope_cells
        else:
            # The lowering's root row, posted on first need by the same
            # helper module-level minters share (a straight-line lowering
            # that never named it has none yet).
            from .precompile_to_ssa import _function_root_cell
            from .ssa_record_return_state import function_scope_of

            sources = _distinct((
                _function_root_cell(book, function_scope_of(self.function)),
            ))
        self.function_cell = book.post(
            EMISSION_FUNCTION, row, pending, stage=self.stage,
            provenance=(
                Derived(sources) if sources else Unsourced(NO_FUNCTION_SCOPE)
            ),
            mode=Mode.REVISE,
        )
        return self.function_cell

    def header(
        self, text: str, *, args: Iterable = (), spelling: str = "",
        extra_cells: Iterable = (),
    ) -> Ref | None:
        """Post ``emission_function`` pending, then the FUNCTION_HEADER unit
        DERIVED from it and the formals' cells."""

        self.open_function()
        return self.unit(
            UnitKind.FUNCTION_HEADER, text, args=args,
            spelling=spelling or self.symbol, extra_cells=extra_cells,
        )

    def unit(
        self, kind: UnitKind, text: str, *, result: Any = None,
        args: Iterable = (), spelling: str = "", extra_cells: Iterable = (),
        instruction: Any = None, block: Any = None,
        block_cell: Ref | None = None,
    ) -> Ref | None:
        """One unit of emitted text.  DERIVED from the function cell, the
        result's and every argument's identity cell, and ``extra_cells``;
        ``Unsourced(value_without_identity_cell)`` when some value has no
        cell (in a ``local_values`` scope such a value is the scope's own and
        is counted, not unsourced).  A BLOCK_LABEL derives from
        ``block_cell`` (a block the emitter itself authors: the wrapper's
        entry) or from the ``ssa_block`` cell of ``block``;
        ``Unsourced(block_origin_unrouted)`` while neither exists."""

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
        label_cell = None
        if kind is UnitKind.BLOCK_LABEL:
            label_cell = block_cell or ssa_block_cell(
                self.book, self.function, block,
            )
            if label_cell is None and self.local_values is not None:
                # A block of the imported kernel text: no control lowering
                # made it, the kernel scope did.
                label_cell = self.function_cell
            cells.append(label_cell)
        if missing and self.local_values is not None:
            # The imported kernel body's own values: no pass lowers them,
            # their identity is the kernel scope the function cell carries.
            self.local_values += len(missing)
            missing = []
        if missing:
            provenance: Any = Unsourced(VALUE_WITHOUT_IDENTITY_CELL)
            self.unsourced += 1
            self.unsourced_values.extend(missing)
        elif kind is UnitKind.BLOCK_LABEL and label_cell is None:
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
        self.texts.append((ref, str(text)))
        if instruction is not None:
            self.instruction_units.setdefault(id(instruction), ref)
        return ref

    def revise_unit(
        self, ordinal: int, text: str, *, sources: Iterable, stage: Any,
    ) -> Ref | None:
        """A later pass rewrote the text of unit ``ordinal`` (``_annotate_
        noalias`` adds ``noalias`` to a define line): REVISE the same row with
        the text that ships, DERIVED from the unit's previous cell and
        ``sources`` (what justified the rewrite), under the rewriting pass's
        ``stage`` -- the rewrite is its own edge, the count is unchanged."""

        if self.book is None:
            return None
        row = (self.key, self.backend, int(ordinal))
        previous = self.book.latest_ref(EMISSION_UNIT, row)
        if previous is None:
            return None
        fact = self.book.pages[EMISSION_UNIT.name].latest(row)
        if not isinstance(fact, EmittedUnit) or fact.text == text:
            return previous
        ref = self.book.post(
            EMISSION_UNIT, row, EmittedUnit(fact.kind, str(text), fact.spelling),
            stage=stage, provenance=Derived(_distinct((previous, *sources))),
            mode=Mode.REVISE,
        )
        self.units = [ref if unit == previous else unit for unit in self.units]
        self.texts = [
            (ref, str(text)) if unit == previous else (unit, old)
            for unit, old in self.texts
        ]
        return ref

    def kernel_texts(
        self, texts: Mapping[str, str], *, origins: Mapping[str, Ref | None],
        demands: Mapping[str, Iterable[Ref]] = {}, scan: Iterable = (),
    ) -> dict[str, Ref | None]:
        """One KERNEL_TEXT unit per symbol of kernel / library text pulled
        into this module by symbol (plan 100, 4.3), posted under this
        recorder's row (the root's), in an order where every kernel that
        calls another is posted first (ties by symbol).  Each derives from
        its origin row (``origins``), the cells of the call instructions that
        name it (``demands``), and every unit -- of ``scan``'s recorders,
        this one, and the kernels already posted -- whose text calls it."""

        symbols = sorted(texts)
        callers: dict[str, set[str]] = {symbol: set() for symbol in symbols}
        for symbol in symbols:
            for called in _SYMBOL_REFERENCE.findall(texts[symbol]):
                if called in callers and called != symbol:
                    callers[called].add(symbol)
        order: list[str] = []
        remaining = list(symbols)
        while remaining:
            ready = [s for s in remaining if callers[s] <= set(order)]
            chosen = ready[0] if ready else remaining[0]
            order.append(chosen)
            remaining.remove(chosen)
        references: dict[str, list[Ref]] = {}
        if self.book is not None:
            for recorder in (*scan, self):
                for ref, text in recorder.texts:
                    for called in _SYMBOL_REFERENCE.findall(text):
                        references.setdefault(called, []).append(ref)
        posted: dict[str, Ref | None] = {}
        for symbol in order:
            ref = self.unit(
                UnitKind.KERNEL_TEXT, texts[symbol], spelling=symbol,
                extra_cells=(
                    origins.get(symbol), *demands.get(symbol, ()),
                    *references.get(symbol, ()),
                    *(posted.get(caller) for caller in sorted(callers[symbol])),
                ),
            )
            posted[symbol] = ref
        return posted

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
    symbol: str | None = None, key: Any = None,
    scope_cells: Iterable | None = None, local_values: bool = False,
) -> EmissionRecorder:
    return EmissionRecorder(
        book, function, backend, stage=stage, symbol=symbol, key=key,
        scope_cells=scope_cells, local_values=local_values,
    )


def imported_kernel_scope(
    book: Any, module: Any, kernel: str, callers: Iterable,
    recorder_for: Any, *, stage: Any,
) -> tuple[Ref, ...]:
    """The scope cells of a tensor reference kernel the module imported from
    authored LLVM text (its function declares ``llvm_argument_names``, which
    only ``import_llvm_to_repository_ssa`` writes): every calling function's
    ``emission_function`` cell (``recorder_for(caller).open_function()``,
    opened ahead of its header when the caller is emitted later), the cells
    of each call instruction's operands, and the kernel text's own
    KERNEL_SOURCE root."""

    if book is None:
        return ()
    cells: list[Any] = []
    for caller, instruction in callers:
        cells.append(recorder_for(caller).open_function())
        cells.extend(
            value_cell(book, module.functions[caller], argument)
            for argument in instruction.args
        )
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_function,
    )

    try:
        text = extract_llvm_function(str(kernel))
    except KeyError:
        text = None
    if text is not None:
        cells.append(kernel_source_cell(
            book, str(kernel), text,
            origin=(extract_llvm_function.__module__, "LLVM_SSA_MODULE"),
            stage=stage,
        ))
    return _distinct(cells)


def is_imported_kernel(function: Any) -> bool:
    """A function ``import_llvm_to_repository_ssa`` made from authored LLVM
    text (it alone declares ``llvm_argument_names``)."""

    return (getattr(function, "metadata", None) or {}).get(
        "llvm_argument_names"
    ) is not None


def kernel_callers(module: Any, names: Iterable[str]) -> dict[str, list]:
    """kernel name -> [(calling function, call instruction)], over ``names``
    in order, for every call whose declared ``callee`` is an imported
    kernel among ``names``."""

    names = tuple(names)
    kernels = {name for name in names if is_imported_kernel(module.functions[name])}
    found: dict[str, list] = {name: [] for name in kernels}
    for name in names:
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                callee = (getattr(instruction, "attributes", None) or {}).get("callee")
                if callee is not None and str(callee) in kernels:
                    found[str(callee)].append((name, instruction))
    return found


def post_artifact_part(
    book: Any, artifact: str, backend: Backend, part: Any, *, data: Any,
    location: Iterable = (), sources: Iterable = (), reason: Any = None,
    stage: Any = ARTIFACT_BUILD, module: Any = None, what: str = "",
) -> Ref | None:
    """One ``emission_artifact`` row: ``(artifact, backend, part)`` ->
    ``ArtifactFact(sha256 of data, byte length, location)``, REVISE.

    DERIVED from ``sources`` when every one is a cell; ``Unsourced(reason)``
    when some source is None (``reason`` names the missing hop).  A rebuild
    that names the same source cells and no newer one is not a new fact the
    api admits: the previous row stands and is returned.

    No book: nothing is posted; a MODULE_TEXT of ``module`` is kept on its
    metadata for ``replay_emission_gaps``."""

    if book is None:
        metadata = getattr(module, "metadata", None)
        if metadata is not None and part is ArtifactPart.MODULE_TEXT:
            digest, length = _sha256(data)
            gaps = metadata.get(EMISSION_GAPS)
            if not isinstance(gaps, list):
                gaps = metadata[EMISSION_GAPS] = []
            gaps.append((str(artifact), backend, digest, length, str(what)))
        return None
    digest, length = _sha256(data)
    return _post_revision(
        book, EMISSION_ARTIFACT, (str(artifact), backend, part),
        ArtifactFact(digest, length, tuple(location)), sources,
        reason or VALUE_WITHOUT_IDENTITY_CELL, stage,
    )


def _post_revision(
    book: Any, page: Any, row: tuple, fact: Any, sources: Iterable,
    reason: Any, stage: Any,
) -> Ref:
    """REVISE one row, DERIVED from ``sources`` when every one is a cell and
    ``Unsourced(reason)`` otherwise.  A rebuild that names the same source
    cells and no newer one is not a new fact the api admits: the previous
    row stands and is returned."""

    sources = tuple(sources)
    cells = _distinct(sources)
    if cells and all(isinstance(cell, Ref) for cell in sources):
        provenance: Any = Derived(cells)
    else:
        provenance = Unsourced(reason)
    latest = book.latest_ref(page, row)
    try:
        return book.post(
            page, row, fact, stage=stage, provenance=provenance,
            mode=Mode.REVISE,
        )
    except ConcordanceRefusal:
        if latest is None:
            raise
        return latest


def _declared_count(shape: Any) -> int | None:
    """The element count of a static declared shape; None when the shape is
    empty or carries a non-static extent."""

    shape = tuple(shape or ())
    if not shape or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in shape
    ):
        return None
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def post_program_abi_field_slots(
    book: Any, root: Any, entry: str, backend: Backend, *,
    buffer_order: Iterable[int],
    buffer_dtypes: Iterable[str], buffer_shapes: Iterable,
    buffer_order_cell: Ref | None, stage: Any,
) -> dict[tuple, Ref]:
    """Post one ``program_abi_field_slot`` row per ProgramABI slot of the
    entry ``root`` (the host-facing layout, book rows first).

    A slot is ``(parameter, field, role)``: a record field of a declared
    parameter, or a bare parameter (``field`` None, named by the root's
    ``parameter_names``).  A lowered root can carry several formals for one
    slot: the direct ProgramABI slot and callsite-forwarded aliases used
    while assembling nested regions.  The ABI rule, here and nowhere else on
    the host side: the written slot is the resident, then a direct slot over
    a callsite-forwarded one, then the earlier formal.  Argument order never
    decides which buffer a host reads back.

    DERIVED(the resident's and each alias's ``ssa_value`` cell, a bare
    parameter's ``function_parameter`` cell, ``buffer_order_cell``);
    ``Unsourced(value_without_identity_cell)`` when the resident has no
    ``ssa_value`` cell.  Descriptors that are not physical storage
    (``is_structural_abi_value``) and returned-record slots, whose record is
    a call result and not a parameter, name no slot."""

    if book is None:
        return {}
    from .ssa_record_return_state import function_scope_of
    from .ssa_storage_requirements import is_structural_abi_value

    order = tuple(int(value_id) for value_id in buffer_order)
    dtypes = tuple(buffer_dtypes)
    shapes = tuple(buffer_shapes)
    index_of = {value_id: index for index, value_id in enumerate(order)}
    scope = function_scope_of(root)
    parameter_names = {
        int(value_id): str(name)
        for name, value_id in root.metadata.get("parameter_names", ())
    }
    groups: dict[tuple, list] = {}
    for formal in root.args:
        if is_structural_abi_value(root, formal):
            continue
        accounting = dict(formal.accounting or {})
        if accounting.get("returned_record_storage") is not None:
            continue
        field = accounting.get("program_abi_field")
        parameter = accounting.get("program_abi_parameter")
        if field is None:
            if parameter is None:
                parameter = parameter_names.get(int(formal.id))
            if parameter is None:
                continue
        elif parameter is None:
            continue
        role = (
            ProgramAbiSlotRole.PRESENCE
            if accounting.get("program_abi_optional_presence")
            else ProgramAbiSlotRole.PAYLOAD
        )
        groups.setdefault(
            (str(parameter), None if field is None else str(field), role), [],
        ).append(formal)

    def priority(formal: Any) -> tuple[int, int]:
        accounting = dict(formal.accounting or {})
        return (
            int(bool(accounting.get("program_abi_field_written"))),
            int(accounting.get("callsite_id") is None),
        )

    posted: dict[tuple, Ref] = {}
    for key, candidates in groups.items():
        parameter, field, role = key
        resident = max(candidates, key=priority)
        resident_id = int(resident.id)
        aliases = tuple(
            int(formal.id) for formal in candidates if formal is not resident
        )
        cells = [
            value_cell(book, root, formal_id)
            for formal_id in (resident_id, *aliases)
        ]
        declared = None
        if cells[0] is not None:
            declared = book.pages[SSA_VALUE.name].latest(cells[0].row)
        shape = tuple(
            getattr(declared, "shape", None) or resident.shape or ()
        )
        accounting = dict(resident.accounting or {})
        storage = accounting.get("program_abi_storage")
        count = _declared_count(shape)
        if count is None and not shape and storage == "scalar":
            count = 1
        index = index_of.get(resident_id)
        capacity = None
        if index is not None:
            capacity = _declared_count(shapes[index]) or (
                1 if not tuple(shapes[index] or ()) else None
            )
        elif count is not None:
            capacity = count
        dtype = (
            str(dtypes[index]) if index is not None
            else str(getattr(declared, "dtype", None) or resident.dtype or "")
        )
        fact = ProgramAbiSlot(
            resident_id, index, dtype, shape, count, capacity,
            bool(accounting.get("program_abi_field_written")),
            None if storage is None else str(storage), aliases,
        )
        extra = []
        if field is None:
            extra.append(book.latest_ref(FUNCTION_PARAMETER, (scope, parameter)))
        row = (str(entry), backend, parameter, field, role)
        ref = _post_revision(
            book, PROGRAM_ABI_FIELD_SLOT, row, fact,
            (*cells, *extra, buffer_order_cell),
            VALUE_WITHOUT_IDENTITY_CELL, stage,
        )
        recorded = book.pages[PROGRAM_ABI_FIELD_SLOT.name].latest(row)
        if recorded != fact:
            raise ConcordanceRefusal(
                f"program_abi_field_slot {row!r}: the book holds {recorded!r} "
                f"and nothing new justifies {fact!r}"
            )
        posted[key] = ref
    return posted


def post_api_contract(
    book: Any, root: Any, entry: str, backend: Backend, *,
    batch: int | None, buffer_order: Iterable[int],
    buffer_dtypes: Iterable[str], buffer_shapes: Iterable,
    buffer_order_cell: Ref | None, unit_cells: Iterable, stage: Any,
) -> Ref | None:
    """Post the entry's API_CONTRACT: ``(entry, backend, API_CONTRACT)`` ->
    ``ArtifactFact`` whose ``location`` is ``(entry, batch, buffers)``, one
    ``LayoutBuffer`` per public buffer in ``void **buffers`` order.

    DERIVED(``unit_cells``: the wrapper's FUNCTION_HEADER and FORMAL units,
    the root's ``function_output`` cells, BUFFER_ORDER, and the
    ``program_abi_field_slot`` row of every buffer a slot names).  A buffer's
    parameter and field are its resident slot row's; a buffer no row names is
    ``OTHER`` and carries no name.  ``batch`` is the cells per column the
    host declared (payload: no declared column role yields it), None when
    none was declared."""

    if book is None:
        return None
    from ..transmogrifier.dtype_layout import DTYPES
    from .ssa_record_return_state import function_scope_of

    order = tuple(int(value_id) for value_id in buffer_order)
    dtypes = tuple(buffer_dtypes)
    shapes = tuple(buffer_shapes)
    page = book.pages.get(PROGRAM_ABI_FIELD_SLOT.name)
    resident: dict[int, tuple] = {}
    slot_cells: list[Ref | None] = []
    for row in () if page is None else page.scope_rows(str(entry)):
        if row[1] != backend:
            continue
        slot = page.latest(row)
        if slot.buffer_index is not None and order[slot.buffer_index] == slot.value_id:
            resident[slot.value_id] = (row, slot)
            slot_cells.append(book.latest_ref(PROGRAM_ABI_FIELD_SLOT, row))
    buffers = []
    for index, value_id in enumerate(order):
        held = resident.get(value_id)
        shape_count = _declared_count(shapes[index]) or 1
        itemsize = int(DTYPES[str(dtypes[index])].byte_size)
        if held is None:
            buffers.append(LayoutBuffer(
                None, None, None, index, str(dtypes[index]), shape_count,
                shape_count, itemsize, LayoutKind.OTHER, False,
            ))
            continue
        row, slot = held
        parameter, field, role = row[2], row[3], row[4]
        buffers.append(LayoutBuffer(
            parameter, field, role, index, str(dtypes[index]),
            slot.count if slot.count is not None else shape_count,
            slot.capacity if slot.capacity is not None else shape_count,
            itemsize,
            LayoutKind.SCALAR if field is None else LayoutKind.FIELD,
            slot.written,
        ))
    outputs = book.pages.get(FUNCTION_OUTPUT.name)
    output_cells = () if outputs is None else tuple(
        book.latest_ref(FUNCTION_OUTPUT, row)
        for row in outputs.scope_rows(function_scope_of(root))
    )
    location = (str(entry), None if batch is None else int(batch), tuple(buffers))
    row = (str(entry), backend, ArtifactPart.API_CONTRACT)
    return _post_revision(
        book, EMISSION_ARTIFACT, row,
        ArtifactFact(*_sha256(repr(location)), location),
        (*unit_cells, *output_cells, buffer_order_cell, *slot_cells),
        VALUE_WITHOUT_IDENTITY_CELL, stage,
    )


def api_contract(book: Any, entry: str, backend: Backend) -> tuple | None:
    """``(entry, batch, buffers)`` of the entry's API_CONTRACT row on
    ``book``, or None when it was never posted."""

    page = book.pages.get(EMISSION_ARTIFACT.name)
    if page is None:
        return None
    fact = page.latest((str(entry), backend, ArtifactPart.API_CONTRACT))
    return None if fact is None else fact.location


def program_abi_field_slots(
    module: Any, entry: str, backend: Backend,
) -> dict[tuple, ProgramAbiSlot]:
    """``(parameter, field, role) -> ProgramAbiSlot`` of the entry ``entry``
    emitted by ``backend``, read from the ``program_abi_field_slot`` rows on
    the module's attached book.  Loud when there is none: a host layout is a
    book fact, and an artifact emitted without a book has not published
    one."""

    metadata = getattr(module, "metadata", None)
    book = None if metadata is None else metadata.get("identity_book")
    page = None if book is None else book.pages.get(PROGRAM_ABI_FIELD_SLOT.name)
    rows = () if page is None else tuple(
        row for row in page.scope_rows(str(entry)) if row[1] == backend
    )
    if not rows:
        raise RuntimeError(
            f"no program_abi_field_slot rows for entry {entry!r} "
            f"({backend.value}): the module "
            "carries no attached identity book, or the entry was never "
            "emitted on it"
        )
    return {(row[2], row[3], row[4]): page.latest(row) for row in rows}


def path_location(path: Any) -> tuple:
    return tuple(Path(path).parts)


class ArtifactEmission:
    """What an artifact carries from emission to build: the book, the
    backend, and the MODULE_TEXT / BUFFER_ORDER cells its later parts derive
    from (plan 100, 4.2: SOURCE_FILE <- MODULE_TEXT, COMPILE_COMMAND <-
    SOURCE_FILE + PIECE_FILE, LIBRARY <- COMPILE_COMMAND)."""

    __slots__ = (
        "book", "backend", "module_text", "buffer_order", "root",
        "function_cell", "api_contract",
    )

    def __init__(
        self, book: Any, backend: Backend, module_text: Ref | None,
        buffer_order: Ref | None = None, *, root: Any = None,
        function_cell: Ref | None = None, api_contract: Ref | None = None,
    ) -> None:
        self.book = book
        self.backend = backend
        self.module_text = module_text
        self.buffer_order = buffer_order
        # The entry's API_CONTRACT cell: what a host-facing file made from
        # the layout (``<entry>_layout.h``) derives from.
        self.api_contract = api_contract
        # The root ``Function`` whose values the public slots are, and its
        # finished ``emission_function`` cell (the entry): what a native
        # loop wrapper of this artifact derives from.
        self.root = root
        self.function_cell = function_cell

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


_DEFINED_REGISTER = re.compile(r"^\s*(%[\w.$-]+)\s*=", re.MULTILINE)
_USED_REGISTER = re.compile(r"%[\w.$-]+")


class NativeLoopEmission:
    """Posts the text ``with_native_sgd_loop`` / ``with_native_adam_loop``
    add around an emitted LLVM artifact (plan 100, 4.1 item 2 / 4.3).

    The book is the wrapped artifact's (``artifact.emission.book``); without
    one every method is a no-op that still counts.  Rows: the wrapper's
    ``emission_function`` / ``emission_unit`` are keyed ``(symbol, loop)``
    (the wrapper may reuse the wrapped entry's symbol) and derive from the
    wrapped root's ``emission_function`` cell and the wrapped MODULE_TEXT;
    each id the wrapper mints for its own buffers is a NOVEL
    ``native_loop_value`` row, transform ``NATIVE_LOOP_WRAPPER_VALUE``,
    operand the wrapped root's ``emission_function`` cell.  The wrapper's
    blocks are authored here: their labels derive from the wrapper's function
    cell.  A body unit derives from the units that define the registers it
    reads (the wrapper's own def-use, in the text it wrote)."""

    def __init__(self, wrapped: Any, loop: Any, symbol: str) -> None:
        from .concordance_declarations import EMISSION_LLVM

        self.wrapped = wrapped
        book = None if wrapped is None else wrapped.book
        if book is not None and wrapped.function_cell is None:
            book = None
        self.book = book
        self.backend = Backend.LLVM_MODULE if wrapped is None else wrapped.backend
        self.loop = loop
        self.symbol = str(symbol)
        self.key = (self.symbol, loop)
        self.values: dict[int, Ref | None] = {}
        self.defined: dict[str, Ref] = {}
        self.recorder = EmissionRecorder(
            book, None if wrapped is None else wrapped.root,
            self.backend, stage=EMISSION_LLVM, symbol=self.symbol,
            key=self.key,
            scope_cells=(
                None if book is None
                else (wrapped.function_cell, wrapped.module_text)
            ),
        )

    def value(self, value_id: int, role: str, parameter: int | None = None) -> Ref | None:
        """The NOVEL row of one id the wrapper minted for its own buffer."""

        if self.book is None:
            self.values[int(value_id)] = None
            return None
        from .concordance_declarations import EMISSION_LLVM

        row = (self.key, int(value_id))
        fact = NativeLoopValue(str(role), None if parameter is None else int(parameter))
        latest = self.book.latest_ref(NATIVE_LOOP_VALUE, row)
        ref = self.book.post(
            NATIVE_LOOP_VALUE, row, fact, stage=EMISSION_LLVM,
            provenance=Novel(NATIVE_LOOP_WRAPPER_VALUE, (self.wrapped.function_cell,)),
            mode=Mode.REVISE if latest is not None else Mode.CONCORD,
        )
        self.values[int(value_id)] = ref
        return ref

    def _note(self, ref: Ref | None, text: str) -> Ref | None:
        if ref is not None:
            for register in _DEFINED_REGISTER.findall(text):
                self.defined[register] = ref
        return ref

    def _readers(self, text: str) -> tuple:
        defined = set(_DEFINED_REGISTER.findall(text))
        return tuple(
            self.defined[register]
            for register in dict.fromkeys(_USED_REGISTER.findall(text))
            if register in self.defined and register not in defined
        )

    def header(self, text: str) -> Ref | None:
        return self.recorder.header(text, spelling=self.symbol)

    def renamed(self, text: str, once_name: str) -> Ref | None:
        """The wrapped entry's define line, renamed internal: a rewrite of
        the wrapped module text, posted as a unit of the wrapper."""

        return self.recorder.unit(
            UnitKind.FUNCTION_HEADER, text, spelling=str(once_name),
            extra_cells=(None if self.wrapped is None else self.wrapped.module_text,),
        )

    def formal(
        self, lines: Iterable[str], *, values: Iterable = (),
        minted: Iterable[int] = (), spelling: str = "",
    ) -> Ref | None:
        """One public slot (or slot group) the wrapper loads: ``values`` are
        the wrapped root's public value ids, ``minted`` the wrapper's own."""

        text = "\n".join(lines)
        return self._note(self.recorder.unit(
            UnitKind.FORMAL, text, args=tuple(int(v) for v in values),
            spelling=spelling,
            extra_cells=tuple(self.values.get(int(v)) for v in minted),
        ), text)

    def lines(self, lines: list, start: int = 0, stop: int | None = None) -> None:
        """Post ``lines[start:stop]`` block by block: a label line is a
        BLOCK_LABEL of a block this wrapper authored; each block's body is a
        STATEMENT and its terminator a BRANCH / RETURN; the closing brace a
        RETURN."""

        recorder = self.recorder
        group: list[str] = []

        def flush() -> None:
            if not group:
                return
            terminator = None
            if group[-1].lstrip().startswith(("br ", "ret ")) or group[-1].strip() == "ret void":
                terminator = group.pop()
            if group:
                text = "\n".join(group)
                self._note(recorder.unit(
                    UnitKind.STATEMENT, text, extra_cells=self._readers(text),
                ), text)
            if terminator is not None:
                kind = UnitKind.RETURN if terminator.lstrip().startswith("ret") else UnitKind.BRANCH
                recorder.unit(kind, terminator, extra_cells=self._readers(terminator))
            group.clear()

        for line in lines[start:stop]:
            if line == "}":
                flush()
                recorder.unit(UnitKind.RETURN, line)
            elif line.endswith(":") and not line.startswith(" "):
                flush()
                recorder.unit(
                    UnitKind.BLOCK_LABEL, line, spelling=line[:-1],
                    block_cell=recorder.function_cell,
                )
            else:
                group.append(line)
        flush()

    def kernel_declaration(self, symbol: str, text: str, origin: Iterable) -> None:
        """A library declaration the wrapper adds by symbol (``llvm.sqrt``)."""

        self.recorder.kernel_texts(
            {str(symbol): text},
            origins={str(symbol): kernel_source_cell(
                self.book, str(symbol), text, origin=origin,
            )},
        )

    def artifact(
        self, name: str, llvm_ir: str, buffer_order: Iterable[int],
    ) -> "ArtifactEmission | None":
        """Finish the wrapper's function row and post the wrapped artifact's
        MODULE_TEXT (from the wrapped one's and the wrapper's function row)
        and BUFFER_ORDER (from the wrapped one's and every minted value)."""

        from .concordance_declarations import EMISSION_LLVM

        recorder = self.recorder
        text = "\n".join(text for _ref, text in recorder.texts)
        finished = recorder.finish(text)
        if self.book is None:
            return None
        module_text = post_artifact_part(
            self.book, name, self.backend, ArtifactPart.MODULE_TEXT,
            data=llvm_ir, sources=(self.wrapped.module_text, finished),
            reason=FUNCTION_TEXT_PENDING, stage=EMISSION_LLVM,
        )
        order = tuple(int(value) for value in buffer_order)
        buffer_cell = post_artifact_part(
            self.book, name, self.backend, ArtifactPart.BUFFER_ORDER,
            data=repr(order), location=order,
            sources=(
                self.wrapped.buffer_order,
                *(self.values[value] for value in order if value in self.values),
            ),
            reason=VALUE_WITHOUT_IDENTITY_CELL, stage=EMISSION_LLVM,
        )
        return ArtifactEmission(
            self.book, self.backend, module_text, buffer_cell,
            root=self.wrapped.root, function_cell=finished,
        )


__all__ = [
    "ArtifactEmission",
    "api_contract",
    "EMISSION_GAPS",
    "EmissionRecorder",
    "NativeLoopEmission",
    "call_demands",
    "emission_book",
    "emission_recorder",
    "imported_kernel_scope",
    "is_imported_kernel",
    "kernel_callers",
    "kernel_source_cell",
    "path_location",
    "piece_source_cell",
    "post_api_contract",
    "post_artifact_part",
    "post_program_abi_field_slots",
    "program_abi_field_slots",
    "replay_emission_gaps",
    "ssa_block_cell",
    "unit_kind_of",
    "value_cell",
]
