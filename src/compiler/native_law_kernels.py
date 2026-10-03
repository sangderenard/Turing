"""Native stand-ins for the sympy laws' AbstractTensor stages.

One program, three stages: sympy law -> AbstractTensor stage (materialised by
the compiler from the law's SSA) -> native.  A stand-in is a callable with the
eager stage's signature that lowers that stage through
``lower_ast_source_to_ssa`` under an ExtractionContract whose feeds carry the
batch shape, emits a backend kernel, and calls it on whole columns.  No AOT
capture, no lane loops: the batch axis is declared on the contract and the
kernel is one call over the whole batch.

The eager run opts in with ``TURING_LAW_NATIVE=llvm``.  Each stand-in lowers
lazily the first time it meets a batch length, caches the kernel on disk keyed
by the stage source and batch (a later launch loads the DLL and skips the
lowering), and keeps the eager stage as its fallback for any law or batch the
backend cannot yet carry (reported once, never silent).  Outputs are read by
NAME through the lowering's ``named_outputs`` record, so a CSE-shared return
(fewer Ret operands than named outputs) maps correctly.

Environment:
  TURING_LAW_NATIVE        backend name ("llvm"); unset = eager stages only
  TURING_LAW_NATIVE_LAWS   comma list of law names, or "all" (default)
  TURING_LAW_NATIVE_SKIP   comma list never lowered (default: the configured
                           vehicle body, whose lowering takes minutes)
  TURING_LAW_NATIVE_CHECK  "1": compare the first native call of each law
                           against the eager stage and report the error
  TURING_LAW_NATIVE_CACHE  kernel cache directory
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACTS = _ROOT / "extraction_contracts"
_CACHE_VERSION = "v4"


def _log(message: str) -> None:
    print(f"[native-law] {message}", file=sys.stderr, flush=True)


def native_backend() -> str:
    return os.environ.get("TURING_LAW_NATIVE", "").strip().lower()


def _law_selected(name: str) -> bool:
    wanted = os.environ.get("TURING_LAW_NATIVE_LAWS", "all").strip()
    skipped = {
        item.strip() for item in os.environ.get(
            "TURING_LAW_NATIVE_SKIP", "abstract_ui_vehicle_step").split(",")
        if item.strip()
    }
    if name in skipped:
        return False
    if wanted.lower() == "all":
        return True
    return name in {item.strip() for item in wanted.split(",")}


def cache_root() -> Path:
    configured = os.environ.get("TURING_LAW_NATIVE_CACHE")
    root = Path(configured) if configured else (
        Path(tempfile.gettempdir()) / "turing_native_laws")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Compiler identity of a built piece.
#
# A piece cache keyed by its equations alone survives compiler fixes: the
# batch-4 propellant-supply piece kept a scalar branch on lane 0 after
# ssa_python_materializer learned to spell Select as ``where`` (2026-10-03,
# docs/concordance_census/CONTINUATION_batch_piecewise_lane0.md).  The build
# therefore RECORDS which compiler made it -- a content digest per ``src.*``
# module loaded when the build finished -- on the piece and as one book row.
# It is a record, not a cache key: a load compares it with the sources on
# disk and names the modules that changed.
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[1]
#: Per-process content digests by absolute path: each source file is hashed
#: at most once per process, so the check on every cached load is a dict read.
_SOURCE_DIGESTS: dict[str, str | None] = {}
#: Per-process verdicts by record (frozen, hashable): the same record is
#: checked once per process.
_STALENESS: dict[Any, tuple[str, ...]] = {}


@dataclass(frozen=True)
class PieceCompilerRecord:
    """The compiler a piece was built by: ``(module, path relative to the
    turing root, sha256 of its content)`` for every ``src.*`` module loaded
    when the build finished (the route's import closure, plus whatever else
    the building process had loaded -- a superset can only over-report
    staleness, never hide it), and one digest over all of them."""

    digest: str
    modules: tuple[tuple[str, str, str], ...]


def _source_digest(path: Path) -> str | None:
    key = str(path)
    if key not in _SOURCE_DIGESTS:
        try:
            _SOURCE_DIGESTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            _SOURCE_DIGESTS[key] = None
    return _SOURCE_DIGESTS[key]


def _combined_digest(modules) -> str:
    return hashlib.sha256("\n".join(
        f"{name} {relative} {digest}" for name, relative, digest in modules
    ).encode("utf-8")).hexdigest()


def route_compiler_record() -> PieceCompilerRecord:
    """The compiler identity of a build finishing now in this process."""
    modules = []
    for name, module in sorted(sys.modules.items()):
        if module is None or not (name == "src" or name.startswith("src.")):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(_SRC_ROOT.parent).as_posix()
        except ValueError:
            continue  # an ``src`` package that is not this compiler's
        digest = _source_digest(path)
        if digest is not None:
            modules.append((name, relative, digest))
    modules = tuple(modules)
    return PieceCompilerRecord(_combined_digest(modules), modules)


def piece_staleness(piece: Any) -> tuple[str, ...]:
    """The modules whose source differs from the compiler that built
    ``piece``; empty when the piece is current.  A piece with no record
    (built before records existed) is stale by name: its compiler is
    unknown, which is not the same as current."""
    record = getattr(piece, "compiler", None)
    if not isinstance(record, PieceCompilerRecord):
        return ("<no compiler record>",)
    verdict = _STALENESS.get(record)
    if verdict is None:
        root = _SRC_ROOT.parent
        verdict = _STALENESS[record] = tuple(
            name for name, relative, digest in record.modules
            if _source_digest(root / relative) != digest
        )
    return verdict


@dataclass(frozen=True)
class PieceBuildFact:
    """Book fact of one piece build: the compiler record it carries."""

    compiler_digest: str
    module_digests: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class PieceStalenessFact:
    """Book fact of a cached piece found stale on load, and what was done:
    ``"rebuilt"`` (default) or ``"served"`` (opt-in)."""

    decision: str
    changed_modules: tuple[str, ...]
    recorded_digest: str | None


_PIECE_VOCABULARY: dict[str, Any] = {}


def _piece_vocabulary() -> dict[str, Any]:
    """The declared pages/stage/transforms the piece cache posts with
    (declarations are idempotent; this lane owns these names)."""
    if not _PIECE_VOCABULARY:
        from .identity_concordance import (
            RowField, RowFieldKind as K, declare_page, declare_stage,
            declare_transform)

        row = (RowField("piece", K.NAME), RowField("batch", K.INDEX),
               RowField("equations_key", K.LABEL))
        _PIECE_VOCABULARY.update(
            build_page=declare_page("piece_build", row, PieceBuildFact),
            stale_page=declare_page("piece_staleness", row, PieceStalenessFact),
            stage=declare_stage("piece_cache"),
            build=declare_transform("piece_build", 0),
            stale_check=declare_transform("piece_stale_check", 0),
        )
    return _PIECE_VOCABULARY


def post_piece_book(directory: Any, piece_id: str, batch: int, key: str, *,
                    built: Any = None, stale: tuple[str, ...] | None = None,
                    stale_record: Any = None, decision: str = "rebuilt") -> Any:
    """Post one piece-cache event on its own book and write the book beside
    the piece (``<piece_id>.book.log``, or ``.stale.book.log`` for a served
    stale piece).

    ``stale`` (changed modules) posts the staleness row as a root
    (``piece_stale_check``); ``built`` (the new piece) posts the build row,
    DERIVED from that staleness row when the build is its rebuild, else a
    root (``piece_build``).  Returns the book."""
    from .identity_concordance import (
        Derived, Mode, Novel, begin_identity_book, end_identity_book,
        render_identity_book)

    vocabulary = _piece_vocabulary()
    row = (str(piece_id), int(batch), str(key))
    book, token = begin_identity_book()
    try:
        cause = None
        if stale is not None:
            cause = book.post(
                vocabulary["stale_page"], row,
                PieceStalenessFact(
                    str(decision), tuple(stale),
                    getattr(stale_record, "digest", None)),
                stage=vocabulary["stage"],
                provenance=Novel(vocabulary["stale_check"], ()),
                mode=Mode.CONCORD)
        if built is not None:
            record = built.compiler
            book.post(
                vocabulary["build_page"], row,
                PieceBuildFact(record.digest, record.modules),
                stage=vocabulary["stage"],
                provenance=(Derived((cause,)) if cause is not None
                            else Novel(vocabulary["build"], ())),
                mode=Mode.CONCORD)
    finally:
        end_identity_book(token)
    name = f"{piece_id}.book.log" if built is not None else f"{piece_id}.stale.book.log"
    Path(directory, name).write_text(render_identity_book(book), encoding="utf-8")
    return book


def batch_contract(entry: str, argument_names: tuple[str, ...], batch: int):
    """The extraction contract that declares every law input as a batch span."""

    from .extraction_contract import ExtractionContract

    values = [{
        "function": entry, "parameter": name, "storage": "span",
        "dtype": "float64", "rank": 1, "shape": [int(batch)],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    } for name in argument_names]
    return ExtractionContract(
        _CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(
        {"records": {}, "bindings": [], "values": values}
    ).with_execution_file(_CONTRACTS / "vehicle_full_native_execution.yaml")


@dataclass
class LawKernel:
    """One compiled batch kernel of a law, callable on flat float64 columns."""

    law: str
    batch: int
    backend: str
    artifact: Any
    argument_names: tuple[str, ...]
    argument_ids: tuple[int, ...]
    output_ids: dict[str, int]
    #: Outputs the law reduces to a constant. They have no value id -- a
    #: literal is never a region output -- so they are carried here and
    #: served as a column of that value. Absent them, one constant zero
    #: refuses a law of 146 outputs.
    constant_outputs: dict = field(default_factory=dict)
    calls: int = 0
    seconds: float = 0.0
    #: The compiler that built it (``PieceCompilerRecord``); a record from
    #: before compiler records loads with None, which is stale.
    compiler: Any = None
    #: ``<version dir>/<dll>`` relative to the kernel's cache directory.
    library: str = ""

    def __call__(self, columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        from .ssa_llvm_backend import prepare_artifact_execution

        started = time.perf_counter()
        execution = prepare_artifact_execution(self.artifact, {
            value_id: columns[name]
            for value_id, name in zip(self.argument_ids, self.argument_names)
        })
        execution.run()
        results = {
            name: execution.buffers[value_id]
            for name, value_id in self.output_ids.items()
        }
        for name, value in self.constant_outputs.items():
            results[name] = np.full(self.batch, value, dtype=np.float64)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return results


@dataclass
class LLVMPiece:
    """A compiled law as a plain positional Python callable.

    This is the object authored Python calls when it wants a law: columns in
    ``argument_names`` order in, a tuple in ``output_names`` order out.  It
    runs the LLVM artifact eagerly, and it DECLARES itself to the source
    compiler -- an ``artifact`` that is an ``LLVMFunctionArtifact`` plus the
    ``argument_ids`` / ``output_ids`` that map the call to the artifact's
    buffers -- so a lowering that meets it at a call site lowers the call as
    an in-C call to the same symbol instead of ingesting Python.
    """

    artifact: Any
    argument_names: tuple[str, ...]
    argument_ids: tuple[int, ...]
    output_names: tuple[str, ...]
    output_ids: dict[str, int]
    constant_outputs: dict = field(default_factory=dict)
    batch: int = 1
    #: The repository SSA the artifact was emitted from -- module, entry
    #: symbol and outputs -- which is what a lowering links at a call site
    #: to know the piece's exact signature without re-lowering its body.
    module: Any = None
    entry: str | None = None
    outputs: Any = None
    #: The authored Python the piece was lowered from.  A lowering that
    #: links the piece ingests this def for the call's signature and arity
    #: only; its body is never lowered again -- the link supplies the SSA.
    source: str | None = None
    #: The compiler that built this piece (``PieceCompilerRecord``), stamped
    #: by ``piece_from_law``; None on a piece built before records existed.
    compiler: Any = None
    #: Runtime binding made by the instantiation hook: the prepared execution
    #: and the exact spans it was prepared against.  Lives for the state's
    #: lifetime, is never persisted, and is absent on a freshly loaded piece.
    _execution: Any = field(default=None, repr=False, compare=False)
    _bound: Any = field(default=None, repr=False, compare=False)
    #: Output names the instantiation placed in the containing system's own
    #: spans (``instantiate(..., outputs=)``); runtime binding like the two
    #: above, never persisted.
    in_place: tuple = field(default=(), repr=False, compare=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_execution"] = None
        state["_bound"] = None
        state["in_place"] = ()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("compiler", None)
        self.__dict__.setdefault("_execution", None)
        self.__dict__.setdefault("_bound", None)
        self.__dict__.setdefault("in_place", ())

    def instantiate(self, columns, outputs=None):
        """The instantiation hook: prepare this piece once, against its spans.

        ``columns`` maps each argument name to the span the containing system
        will hand this piece every round.  The artifact's public ABI -- the
        output buffers, the pointer table, the per-dtype scalar arena -- is
        allocated here, once, with the inputs aliasing the given spans.  A
        column that would have to be copied to become a contiguous float64
        span is refused: aliasing is the point of instantiating, and a hidden
        copy would be exactly the per-round marshalling this removes.

        ``outputs`` (optional) maps output names to the spans the containing
        system has decided those outputs land in -- the state's own column
        views -- so the kernel stores into them in place.  Which outputs may
        land in place is the containing system's decision (it knows its read
        discipline); this piece only refuses what its ABI cannot honour.  An
        output whose buffer is also one of its INPUT buffers (one SSA value
        filling an input and an output, e.g. ``dt_prev_next`` = ``dt``), or
        whose buffer fills more than one declared output, or whose declared
        extent is not the span's, keeps its own buffer.  The names
        that did land in place are ``in_place``.
        """
        from .ssa_llvm_backend import prepare_artifact_execution

        bound = []
        for name in self.argument_names:
            given = columns[name]
            span = np.asarray(given, dtype=np.float64)
            if span is not given or (span.ndim and not span.flags.c_contiguous):
                raise TypeError(
                    f"{self.artifact.name}: column {name!r} is not a contiguous "
                    "float64 span; the piece cannot alias it")
            bound.append(span)
        feeds = {value_id: span for value_id, span in zip(self.argument_ids, bound)}
        shapes = {
            int(value_id): tuple(shape or ())
            for value_id, shape in zip(self.artifact.buffer_order,
                                       self.artifact.buffer_shapes)
        }
        sharing = {}
        for value_id in self.output_ids.values():
            sharing[int(value_id)] = sharing.get(int(value_id), 0) + 1
        in_place = []
        for name, given in dict(outputs or {}).items():
            value_id = self.output_ids.get(name)
            if value_id is None or int(value_id) in feeds:
                continue
            if sharing[int(value_id)] > 1:
                # One SSA value fills several declared outputs; placing it in
                # one column would make the others read that column, which a
                # later write may change before they are read.
                continue
            span = np.asarray(given, dtype=np.float64)
            if span is not given or (span.ndim and not span.flags.c_contiguous):
                raise TypeError(
                    f"{self.artifact.name}: output {name!r} span is not a "
                    "contiguous float64 span; the piece cannot store into it")
            if tuple(span.shape) != shapes.get(int(value_id)):
                continue
            feeds[int(value_id)] = span
            in_place.append(name)
        self._execution = prepare_artifact_execution(self.artifact, feeds)
        self._bound = tuple(bound)
        self.in_place = tuple(in_place)
        return self._execution

    @classmethod
    def from_kernel(cls, kernel: LawKernel) -> "LLVMPiece":
        return cls(
            kernel.artifact, kernel.argument_names, kernel.argument_ids,
            tuple(kernel.output_ids) + tuple(kernel.constant_outputs),
            dict(kernel.output_ids), dict(kernel.constant_outputs), kernel.batch,
        )

    def save(self, path) -> None:
        """Persist the piece as one file: artifact, ids, names, SSA and source."""
        import pickle

        with open(path, "wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path) -> "LLVMPiece":
        import pickle

        with open(path, "rb") as stream:
            piece = pickle.load(stream)
        if not isinstance(piece, cls):
            raise TypeError(f"{path}: not an LLVMPiece")
        return piece

    def __call__(self, *columns):
        from .ssa_llvm_backend import prepare_artifact_execution

        if len(columns) != len(self.argument_names):
            raise TypeError(
                f"{self.artifact.name}: takes {len(self.argument_names)} "
                f"columns, got {len(columns)}")
        if self._execution is not None and all(
            column is span for column, span in zip(columns, self._bound)
        ):
            # Instantiated against exactly these spans: the ABI is already
            # bound, the round only runs.  The output buffers are the same
            # arrays every call; the spelled step assigns them into the
            # state's spans immediately.
            execution = self._execution
        else:
            # Standalone use, or columns other than the instantiated spans:
            # the historical per-call preparation.
            execution = prepare_artifact_execution(self.artifact, {
                value_id: np.asarray(column, dtype=np.float64)
                for value_id, column in zip(self.argument_ids, columns)
            })
        execution.run()
        return tuple(
            execution.buffers[self.output_ids[name]] if name in self.output_ids
            else np.full(self.batch, self.constant_outputs[name], dtype=np.float64)
            for name in self.output_names
        )


def _lower_law(compilation: Any, law: str, batch: int, backend: str,
               precision_policy: Any = None) -> LawKernel:
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference)
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    metadata = compilation.function.metadata
    argument_names = tuple(metadata["argument_names"])
    output_names = tuple(metadata["output_names"])
    # ``precision_policy`` (optional): the AbstractTensor stage is produced
    # with its measured precision sections written in (precision_policy.py),
    # and those Precision sections lower through apply_precision_pipeline
    # like any authored Precision code.
    source = symbolic_abstract_tensor_source(compilation, "tick", precision_policy)
    from src.common.tensors.extended_precision import Precision

    lowered = lower_ast_source_to_ssa(
        source, "tick",
        python_bindings={"AbstractTensor": AbstractTensor, "Precision": Precision},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=f"{law}_batched", runtime_closure_only=True,
        extraction_contract=batch_contract("tick", argument_names, batch),
    )
    module = lowered[0] if isinstance(lowered, tuple) else lowered.module
    entry = next(name for name in module.functions if name.endswith("__tick"))
    function = module.functions[entry]
    # The stage returns one expression per declared output, but a CSE-shared
    # law returns the same temporary for several outputs, and the lowering's
    # ``named_outputs`` record lists each temporary once.  Read the stage's
    # own return tuple to pair every declared output with its temporary.
    import ast as _ast

    stage_ast = _ast.parse(source)
    stage_function = next(
        node for node in stage_ast.body
        if isinstance(node, _ast.FunctionDef) and node.name == "tick")
    return_node = next(
        node for node in _ast.walk(stage_function) if isinstance(node, _ast.Return))
    return_value = return_node.value
    returned = (
        list(return_value.elts) if isinstance(return_value, _ast.Tuple)
        else [return_value])
    returned_names = [
        node.id if isinstance(node, _ast.Name) else None for node in returned]
    # A TEMPORARY BOUND TO A LITERAL IS NOT AN UNLOWERED OUTPUT.
    # Literals never become region outputs, so a stage line like `t109 = 0`
    # leaves `t109` out of `named_outputs` entirely and the check below
    # reads that absence as a failure to lower. Measured on the vehicle
    # body: exactly one output of 146 --
    # `wheel_gyroscopic_reaction_torque_z`, whose temporary is the literal
    # 0 -- and that single constant refused the whole law. It is why this
    # law sits in TURING_LAW_NATIVE_SKIP; the reason recorded there, that
    # its lowering "takes minutes", is true but is not what stopped it.
    literal_of_temporary = {
        node.targets[0].id: float(node.value.value)
        for node in stage_function.body
        if isinstance(node, _ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], _ast.Name)
        and isinstance(node.value, _ast.Constant)
        and isinstance(node.value.value, (int, float))
        and not isinstance(node.value.value, bool)}
    if len(returned_names) != len(output_names):
        raise RuntimeError(
            f"{law}: stage returns {len(returned_names)} values, the law "
            f"declares {len(output_names)} outputs")
    id_of_temporary = {
        str(temporary): int(value_id)
        for temporary, value_id in tuple(function.metadata.get("named_outputs") or ())
    }
    # A constant output is served as a constant whether or not the lowering
    # also lists it: once the control function materialises its own literal
    # for a returned temporary, ``named_outputs`` names it and the kernel
    # exposes it as a rank-0 buffer -- one cell, not a column, so a
    # per-row reader would index past it.  The literal's value is the
    # output; the buffer is an ABI fact about the kernel.
    constant_outputs = {
        output: literal_of_temporary[temporary]
        for output, temporary in zip(output_names, returned_names)
        if temporary is not None and temporary in literal_of_temporary}
    unresolved = [
        output for output, temporary in zip(output_names, returned_names)
        if output not in constant_outputs
        and (temporary is None or temporary not in id_of_temporary)]
    if unresolved:
        raise RuntimeError(
            f"{law}: outputs without a lowered value: {unresolved[:5]}")
    if constant_outputs:
        _log(f"{law}: {len(constant_outputs)} output(s) are constants, served "
             f"as such: {sorted(constant_outputs)[:4]}")
    output_ids = {
        output: id_of_temporary[temporary]
        for output, temporary in zip(output_names, returned_names)
        if output not in constant_outputs
    }
    argument_ids = tuple(int(value.id) for value in function.args)
    if len(argument_ids) != len(argument_names):
        raise RuntimeError(
            f"{law}: entry takes {len(argument_ids)} values, the law "
            f"declares {len(argument_names)} arguments")
    if backend != "llvm":
        raise RuntimeError(f"{law}: backend {backend!r} stand-in not wired yet")
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    artifact = emit_ssa_function_to_llvm(module, entry)
    if not artifact.complete:
        raise RuntimeError(
            f"{law}: LLVM shortfalls: "
            + "; ".join(s.reason for s in artifact.shortfalls[:3]))
    exposed = set(int(value_id) for value_id in artifact.buffer_order)
    missing = [name for name, value_id in output_ids.items() if value_id not in exposed]
    if missing:
        raise RuntimeError(f"{law}: outputs not exposed by the kernel ABI: {missing[:5]}")
    artifact = compile_artifact(artifact)
    return LawKernel(
        law=law, batch=batch, backend=backend, artifact=artifact,
        argument_names=argument_names, argument_ids=argument_ids,
        output_ids=output_ids, constant_outputs=constant_outputs,
    )


def _cache_key(compilation: Any, law: str, batch: int, backend: str) -> str:
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    source = symbolic_abstract_tensor_source(compilation, "tick")
    digest = hashlib.sha256(
        f"{_CACHE_VERSION}|{backend}|{batch}|{source}".encode("utf-8")).hexdigest()
    return digest[:20]


def law_kernel(compilation: Any, law: str, batch: int, backend: str, *,
               serve_stale: bool | None = None) -> LawKernel:
    """Compiled kernel for (law, batch): from the disk cache or lowered now.

    The key is the stage source (``_cache_key``), so a kernel survives a
    compiler edit.  The kernel therefore carries the compiler that built it
    (``PieceCompilerRecord``, as an ``LLVMPiece`` does) and a load compares
    it with the sources on disk (``piece_staleness``): a stale kernel, or one
    with no record, is rebuilt by default; ``serve_stale=True`` or
    ``TURING_LAW_NATIVE_SERVE_STALE=1`` serves it.  Each event is posted on
    the piece-cache book beside the kernel (``post_piece_book``: staleness
    row, build row DERIVED from it when it is its rebuild).

    Publish is atomic: the DLL is built in a private ``.build-*`` directory,
    renamed to an immutable ``v-<sha256(dll)[:16]>`` version directory (a
    loaded DLL is never overwritten), and ``kernel.pkl`` -- which names its
    version directory -- is written to a unique temporary and ``os.replace``d.
    """

    import shutil
    import uuid

    from .ssa_llvm_backend import compile_artifact

    key = _cache_key(compilation, law, batch, backend)
    directory = cache_root() / law / f"{backend}_b{batch}_{key}"
    record = directory / "kernel.pkl"
    if serve_stale is None:
        serve_stale = os.environ.get("TURING_LAW_NATIVE_SERVE_STALE", "").casefold() in {
            "1", "true", "yes", "on"}
    stale: tuple[str, ...] | None = None
    stale_record = None
    if record.is_file():
        cached = None
        try:
            with record.open("rb") as handle:
                cached = pickle.load(handle)
            version = getattr(cached, "library", "") or ""
            library = directory / version if version else None
            if library is None or not library.is_file():
                # A record from before version directories names no DLL of
                # its own; its compiler is unknown.
                cached.compiler = None
            else:
                cached.artifact.library_path = library
                cached.artifact._entry = None
        except Exception as error:  # an unreadable record is rebuilt
            _log(f"{law}: cache record unreadable ({type(error).__name__}), rebuilding")
            cached = None
        if cached is not None:
            changed = piece_staleness(cached)
            if not changed:
                _log(f"{law}: batch {batch} kernel from cache {directory.name}")
                return cached
            stale, stale_record = tuple(changed), getattr(cached, "compiler", None)
            if serve_stale and getattr(cached, "library", ""):
                post_piece_book(directory, law, batch, key, stale=stale,
                                stale_record=stale_record, decision="served")
                _log(f"{law}: SERVING STALE batch {batch} kernel {directory.name}: "
                     f"{len(stale)} module(s) changed: {', '.join(stale[:8])}")
                return cached
            _log(f"{law}: batch {batch} kernel stale ({len(stale)} module(s) changed: "
                 f"{', '.join(stale[:4])}), rebuilding")
    started = time.time()
    directory.mkdir(parents=True, exist_ok=True)
    build = directory / f".build-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        kernel = _lower_law(compilation, law, batch, backend)
        build.mkdir(parents=True, exist_ok=True)
        compile_artifact(kernel.artifact, directory=build, optimization="O2")
        # Which compiler built it: recorded when the build finished.
        kernel.compiler = route_compiler_record()
        built = Path(kernel.artifact.library_path)
        version = directory / (
            "v-" + hashlib.sha256(built.read_bytes()).hexdigest()[:16])
        if version.exists():
            # Another builder published the identical DLL; take it.
            shutil.rmtree(build, ignore_errors=True)
        else:
            try:
                os.replace(build, version)
            except OSError:
                if not version.exists():
                    raise
                shutil.rmtree(build, ignore_errors=True)
        kernel.library = f"{version.name}/{built.name}"
        kernel.artifact.library_path = version / built.name
        entry = kernel.artifact._entry
        kernel.artifact._entry = None
        temporary = directory / f"kernel.pkl.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                pickle.dump(kernel, handle)
            os.replace(temporary, record)
        finally:
            temporary.unlink(missing_ok=True)
            kernel.artifact._entry = entry
    except BaseException as error:
        shutil.rmtree(build, ignore_errors=True)
        if stale is not None:
            post_piece_book(directory, law, batch, key, stale=stale,
                            stale_record=stale_record,
                            decision=f"rebuild_failed: {type(error).__name__}")
        raise
    post_piece_book(directory, law, batch, key, built=kernel, stale=stale,
                    stale_record=stale_record, decision="rebuilt")
    _log(f"{law}: batch {batch} lowered+compiled ({backend}) in "
         f"{time.time() - started:.1f}s -> {version}")
    return kernel


class NativeLawStage:
    """A law's stage executed by its batch kernel, eager stage as fallback."""

    def __init__(self, law: str, compilation: Any, fallback: Callable, backend: str):
        metadata = compilation.function.metadata
        self.law = law
        self.compilation = compilation
        self.fallback = fallback
        self.backend = backend
        self.argument_names = tuple(metadata["argument_names"])
        self.output_names = tuple(metadata["output_names"])
        self.kernels: dict[int, LawKernel | None] = {}
        self.checked = False
        self.__name__ = law

    def _kernel(self, batch: int) -> LawKernel | None:
        if batch not in self.kernels:
            try:
                self.kernels[batch] = law_kernel(
                    self.compilation, self.law, batch, self.backend)
            except Exception as error:
                self.kernels[batch] = None
                _log(f"{self.law}: batch {batch} stays on the eager stage: "
                     f"{type(error).__name__}: {str(error)[:300]}")
        return self.kernels[batch]

    def __call__(self, *arguments):
        from src.common.tensors import AbstractTensor

        if len(arguments) != len(self.argument_names):
            raise TypeError(
                f"{self.law} takes {len(self.argument_names)} arguments, "
                f"got {len(arguments)}")
        arrays = [
            np.asarray(getattr(value, "data", value), dtype=np.float64)
            for value in arguments
        ]
        shape = np.broadcast_shapes(*(array.shape for array in arrays))
        batch = int(np.prod(shape)) if shape else 1
        kernel = self._kernel(batch)
        if kernel is None:
            return self.fallback(*arguments)
        columns = {
            name: np.ascontiguousarray(np.broadcast_to(array, shape).reshape(batch))
            for name, array in zip(self.argument_names, arrays)
        }
        produced = kernel(columns)
        results = []
        for name in self.output_names:
            value = np.asarray(produced[name], dtype=np.float64)
            if value.size == batch:
                value = value.reshape(shape).copy()
            elif value.size == 1:
                value = np.broadcast_to(value.reshape(()), shape).copy()
            else:
                raise RuntimeError(
                    f"{self.law}: output {name} has {value.size} elements "
                    f"for batch {batch}")
            results.append(AbstractTensor.tensor(value))
        if (os.environ.get("TURING_LAW_NATIVE_CHECK", "").strip() == "1"
                and not self.checked):
            self.checked = True
            self._check(arguments, results, shape)
        return tuple(results)

    def _check(self, arguments, results, shape) -> None:
        expected = self.fallback(*arguments)
        worst = 0.0
        scale = 0.0
        for got, want in zip(results, expected):
            got_array = np.asarray(getattr(got, "data", got), dtype=np.float64)
            want_array = np.broadcast_to(
                np.asarray(getattr(want, "data", want), dtype=np.float64), shape)
            worst = max(worst, float(np.max(np.abs(got_array - want_array))))
            scale = max(scale, float(np.max(np.abs(want_array))))
        _log(f"{self.law}: first native call vs eager stage: max_abs={worst:.3e} "
             f"scale={scale:.3g} rel={worst / max(scale, 1e-300):.2e} "
             f"shape={tuple(shape)}")


_STAGES: list[NativeLawStage] = []


def bind_native_stand_ins(
    bindings: dict[str, Any], compilations: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace selected law bindings with native stand-ins when opted in."""

    from src.common.tensors.source_realization import deployed_with_authored_fallback

    backend = native_backend()
    if not backend:
        return bindings
    armed = []
    for name, compilation in compilations.items():
        if name not in bindings or not _law_selected(name):
            continue
        stage = NativeLawStage(name, compilation, bindings[name], backend)
        # Cached eager bindings must still expose authored source when a
        # whole-program compiler enters the standard realization context.
        bindings[name] = deployed_with_authored_fallback(bindings[name], stage)
        _STAGES.append(stage)
        armed.append(name)
    _log(f"{backend} stand-ins armed for: {armed}")
    return bindings


def native_law_report() -> dict[str, dict[str, float]]:
    """Per-law native call counts and kernel seconds (for run heartbeats)."""

    report: dict[str, dict[str, float]] = {}
    for stage in _STAGES:
        for batch, kernel in stage.kernels.items():
            key = f"{stage.law}@{batch}"
            if kernel is None:
                report[key] = {"calls": 0, "seconds": 0.0, "native": 0}
            else:
                report[key] = {
                    "calls": kernel.calls, "seconds": kernel.seconds, "native": 1}
    return report
