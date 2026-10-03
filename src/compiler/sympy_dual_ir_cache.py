"""Persistent content-addressed cache for SymPy programs and repository IR.

The cache has two deliberately separate layers:

``solved-equations``
    The result of an expensive symbolic solve/reduction.  Callers supply the
    semantic solve record and a zero-argument producer.

``dual-ir``
    The backend-neutral repository compilation produced from a canonical
    equation program.  Backend emission is intentionally not part of this
    layer, so C, LLVM, WebAssembly, and GPU backends all reuse the same result.

Only compiler-owned objects are unpickled, from a repository-local cache whose
identity includes the implementation digest.  Writes use the existing atomic
AOT checkpoint store; corrupt or stale entries are misses and are rebuilt.

Compiler provenance (2026-10-03).  The key digests the symbolic modules
only, so a ``dual-ir`` entry built by ANOTHER compiler tree was served as
current; it hid the phasing regression
(``docs/concordance_census/CONTINUATION_phasing_regression.md``).  The two
layers that hold compiler output (``dual-ir``, ``symbolic-program``) now
store a ``CompiledPayload``: the value plus the piece cache's compiler record
(``native_law_kernels.PieceCompilerRecord``: a content digest per ``src.*``
module loaded when the build finished), in ONE file, so the record and the
value are published by one atomic replace and can never be paired with
another build's.  The record is a record, not a key: a load compares it with
the sources on disk (``piece_staleness``).  A stale entry is rebuilt by
default; ``serve_stale=True`` or ``TURING_SYMPY_DUAL_IR_SERVE_STALE=1``
serves it.  Every lookup posts on the active book: a
``symbolic_cache_staleness`` row for a stale entry (decision ``rebuilt`` /
``served`` / ``rebuild_failed: <Error>``) and a ``symbolic_cache_build`` row
with the compiler record of the value returned (DERIVED from the staleness
row when it is that row's rebuild).  ``solved-equations`` holds a SymPy
solve keyed on the producer's own source files; it carries no compiler
output and is not provenanced.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from src.common.tensors.accelerator_backends.aot_checkpoint import (
    AOTCheckpointStore,
)


_CACHE_SCHEMA = "turing-sympy-dual-ir-v1"
_T = TypeVar("_T")


def _disabled() -> bool:
    return os.environ.get("TURING_DISABLE_SYMPY_DUAL_IR_CACHE", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _serve_stale_configured() -> bool:
    return os.environ.get("TURING_SYMPY_DUAL_IR_SERVE_STALE", "").casefold() in {
        "1", "true", "yes", "on",
    }


#: Layers whose value is compiler output; they carry a compiler record.
PROVENANCED_LAYERS = frozenset({"dual-ir", "symbolic-program"})


@dataclass(frozen=True)
class CompiledPayload:
    """One stored compilation and the compiler that built it, one file.

    ``compiler`` is a ``PieceCompilerRecord``; ``piece_staleness`` reads it
    from this object as it reads it from an ``LLVMPiece``."""

    value: Any
    compiler: Any


def _same_or_revise(book: Any, page: Any, row: tuple, fact: Any):
    """CONCORD while the row's latest fact is this one (a repeat writes no
    cell); REVISE when it differs (a rebuild after a staleness row, or one
    entry met again in the same process after another build)."""
    from .identity_concordance import Mode

    entries = book.page(page).history(row)
    if entries and entries[-1][1] != fact:
        return Mode.REVISE
    return Mode.CONCORD


def post_symbolic_cache_event(layer: str, identity: str, *, built: Any = None,
                              stale: tuple[str, ...] | None = None,
                              stale_record: Any = None,
                              decision: str = "rebuilt") -> Any:
    """Post one provenanced-layer lookup on the active book.

    ``stale`` (changed modules) posts the staleness row, a root
    (``symbolic_cache_stale_check``); ``built`` (a compiler record) posts the
    build row, DERIVED from that staleness row when the build is its
    rebuild, else a root (``symbolic_cache_build``).  Returns the build row's
    Ref (the staleness row's when nothing was built)."""
    from .concordance_declarations import (
        SYMBOLIC_CACHE, SYMBOLIC_CACHE_BUILD, SYMBOLIC_CACHE_BUILD_TRANSFORM,
        SYMBOLIC_CACHE_STALE_CHECK, SYMBOLIC_CACHE_STALENESS,
        PieceBuildFact, PieceStalenessFact,
    )
    from .identity_concordance import Derived, Novel, current_identity_book

    book = current_identity_book()
    row = (str(layer), str(identity))
    cause = None
    if stale is not None:
        fact = PieceStalenessFact(
            str(decision), tuple(stale), getattr(stale_record, "digest", None))
        cause = book.post(
            SYMBOLIC_CACHE_STALENESS, row, fact, stage=SYMBOLIC_CACHE,
            provenance=Novel(SYMBOLIC_CACHE_STALE_CHECK, ()),
            mode=_same_or_revise(book, SYMBOLIC_CACHE_STALENESS, row, fact))
    if built is None:
        return cause
    fact = PieceBuildFact(built.digest, built.modules)
    return book.post(
        SYMBOLIC_CACHE_BUILD, row, fact, stage=SYMBOLIC_CACHE,
        provenance=(Derived((cause,)) if cause is not None
                    else Novel(SYMBOLIC_CACHE_BUILD_TRANSFORM, ())),
        mode=_same_or_revise(book, SYMBOLIC_CACHE_BUILD, row, fact))


def _configured_root() -> Path | None:
    value = os.environ.get("TURING_SYMPY_DUAL_IR_CACHE_DIR")
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True, slots=True)
class SympyCacheResult:
    """One cache lookup, including enough state for build diagnostics."""

    value: Any
    identity: str
    layer: str
    hit: bool
    status: str
    #: The compiler record of the returned value (provenanced layers only).
    compiler: Any = None
    #: Modules changed since the stored entry's build, found on load (empty
    #: for a current entry or a first build).
    stale: tuple[str, ...] = ()


class SympyDualIRCache:
    """Cache solved SymPy programs and their shared repository IR separately."""

    def __init__(
        self,
        implementation: str,
        *,
        root: str | Path | None = None,
        enabled: bool | None = None,
        serve_stale: bool | None = None,
    ) -> None:
        self.implementation = str(implementation)
        self.root = Path(root).expanduser().resolve() if root else _configured_root()
        self.enabled = not _disabled() if enabled is None else bool(enabled)
        self.serve_stale = (
            _serve_stale_configured() if serve_stale is None else bool(serve_stale))

    def get_or_compute(
        self,
        layer: str,
        record: Mapping[str, Any],
        compute: Callable[[], _T],
    ) -> SympyCacheResult:
        """Load ``layer`` or atomically persist the value produced on a miss."""

        semantic_record = {
            "sympy_cache_schema": _CACHE_SCHEMA,
            "layer": str(layer),
            **dict(record),
        }
        store = AOTCheckpointStore(semantic_record, root=self.root)
        if not self.enabled:
            return SympyCacheResult(
                compute(), store.identity, str(layer), False, "disabled",
            )
        value = store.load(str(layer), self.implementation)
        if str(layer) in PROVENANCED_LAYERS:
            return self._provenanced(store, str(layer), value, compute)
        if value is not None:
            return SympyCacheResult(
                value, store.identity, str(layer), True, store.last_load_status,
            )
        miss_status = store.last_load_status
        value = compute()
        store.store(str(layer), self.implementation, value)
        return SympyCacheResult(
            value, store.identity, str(layer), False, miss_status,
        )

    def _provenanced(self, store: Any, layer: str, loaded: Any,
                     compute: Callable[[], _T]) -> SympyCacheResult:
        """A compiler-output layer: compare the stored compiler record with
        the sources on disk, rebuild a stale entry (serve it on opt-in),
        post the staleness and build rows, publish value + record as one
        file."""
        from .native_law_kernels import piece_staleness, route_compiler_record

        identity = store.identity
        stale: tuple[str, ...] | None = None
        stale_record = None
        status = store.last_load_status
        if loaded is not None:
            if isinstance(loaded, CompiledPayload):
                stale_record = loaded.compiler
                changed = piece_staleness(loaded)
            else:
                # Written before records existed: its compiler is unknown,
                # which is not the same as current (``<no compiler record>``).
                changed = piece_staleness(None)
            if not changed:
                post_symbolic_cache_event(layer, identity, built=loaded.compiler)
                return SympyCacheResult(
                    loaded.value, identity, layer, True, status,
                    compiler=loaded.compiler,
                )
            if self.serve_stale:
                post_symbolic_cache_event(
                    layer, identity, stale=changed, stale_record=stale_record,
                    decision="served")
                print(
                    f"[sympy-dual-ir] SERVING STALE {layer} {identity[:16]}: "
                    f"{len(changed)} module(s) changed since its build: "
                    + ", ".join(changed[:8]),
                    file=sys.stderr, flush=True,
                )
                return SympyCacheResult(
                    getattr(loaded, "value", loaded), identity, layer, True,
                    f"stale-served: {len(changed)} module(s) changed",
                    compiler=stale_record, stale=tuple(changed),
                )
            stale = tuple(changed)
            status = f"stale: {len(changed)} module(s) changed"
        try:
            value = compute()
        except BaseException as error:
            if stale is not None:
                post_symbolic_cache_event(
                    layer, identity, stale=stale, stale_record=stale_record,
                    decision=f"rebuild_failed: {type(error).__name__}")
            raise
        # Which compiler built it, recorded when the build finished and
        # stored in the same file as the value (one os.replace publishes
        # both; a reader never pairs a value with another build's record).
        compiler = route_compiler_record()
        store.store(layer, self.implementation, CompiledPayload(value, compiler))
        post_symbolic_cache_event(
            layer, identity, built=compiler, stale=stale,
            stale_record=stale_record, decision="rebuilt")
        return SympyCacheResult(
            value, identity, layer, False, status, compiler=compiler,
            stale=stale or (),
        )

    def solved_equations(
        self,
        record: Mapping[str, Any],
        solve: Callable[[], _T],
    ) -> SympyCacheResult:
        """Cache a solved/reduced equation-set program before IR lowering."""

        return self.get_or_compute("solved-equations", record, solve)

    def dual_ir(
        self,
        record: Mapping[str, Any],
        lower: Callable[[], _T],
    ) -> SympyCacheResult:
        """Cache repository IR independently of every eventual backend."""

        return self.get_or_compute("dual-ir", record, lower)


__all__ = [
    "CompiledPayload", "PROVENANCED_LAYERS", "SympyCacheResult",
    "SympyDualIRCache", "post_symbolic_cache_event",
]
