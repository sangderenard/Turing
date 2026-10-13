"""Compose authored Python and LLVM-compiled laws into one native module.

One automatic process, no per-scenario assembly:

  1. Each law is lowered on its own to an LLVM piece (``piece_from_law``):
     SymPy -> AbstractTensor source -> repository SSA -> LLVM, compiled,
     and wrapped as an ``LLVMPiece`` -- a plain positional Python callable
     that also declares its artifact and buffer ids to the compiler.
  2. The system is authored as ordinary Python that calls the pieces.  It
     runs as Python (the pieces run their DLLs eagerly), and it lowers as a
     whole program: at every piece call site the source compiler notices the
     LLVM callee and links its SSA for the signature; the C lane then emits
     an extern call to the piece's symbol and links its LLVM module in
     unchanged (``compose_native_package``).
  3. The result is a C module with the public ``void entry(void **buffers,
     long long *extents)`` ABI -- every buffer caller-owned, so any host
     reads state by its own copies -- or, through ``compile_standalone``, a
     Python-free executable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.common.tensors import AbstractTensor

from .native_law_kernels import LLVMPiece, batch_contract, route_compiler_record


def piece_from_law(compilation: Any, law: str, batch: int, *,
                   directory: str | Path | None = None,
                   optimization: str = "O2",
                   resolved_process_graph_sink=None,
                   externals: Mapping[str, Any] | None = None) -> LLVMPiece:
    """Lower one compiled SymPy law to an LLVM piece, on its own.

    ``resolved_process_graph_sink`` is handed to ``lower_ast_source_to_ssa``
    unchanged: a viewer that draws the lowering's process graph receives the
    graph of this very lowering instead of lowering the source a second time.

    The law's declared externals (its undefined Functions,
    ``metadata["external_functions"]``) are bound as runtime-slot leaves
    specialized at their callsites (``external_functions.externals_for_law``).
    ``externals`` optionally names an implementation per external (an
    ``LLVMPiece`` the leaf takes its ABI from, a host callable, or an
    ``ExternalFunction``); the slots are filled at load with
    ``external_functions.bind_external_slots``.

    One compiled program is one book: the leaves and the law are lowered on
    the program's book (``symbolic_equation_compiler.compilation_program_
    book``), which holds its symbolic rows; each lowered external call is
    posted there DERIVED from its symbolic callsite rows
    (``external_functions.post_lowered_external_calls``), and the emitters
    derive the slot-call units from those rows.
    """

    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    from .external_functions import externals_for_law, post_lowered_external_calls
    from .symbolic_equation_compiler import compilation_program_book

    metadata = compilation.function.metadata
    argument_names = tuple(metadata["argument_names"])
    output_names = tuple(metadata["output_names"])
    book = compilation_program_book(compilation)
    compilation, source, declared_externals = externals_for_law(
        compilation, law, batch, externals, book=book)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, law,
        python_bindings={"AbstractTensor": AbstractTensor, **declared_externals},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=law, runtime_closure_only=True,
        extraction_contract=batch_contract(law, argument_names, batch),
        resolved_process_graph_sink=resolved_process_graph_sink,
        identity_book=book,
    )
    post_lowered_external_calls(module, compilation, source, book)
    entry = exports[0]
    return _piece_from_lowered_law(compilation, law, batch, source,
                                   module, outputs, entry, directory, optimization)


def _piece_from_lowered_law(compilation, law, batch, source,
                            module, outputs, entry, directory, optimization):
    """Publish the existing piece ABI from an accepted compiler result."""
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    metadata = compilation.function.metadata
    argument_names = tuple(metadata["argument_names"])
    output_names = tuple(metadata["output_names"])
    function = module.functions[entry]
    artifact = compile_artifact(
        emit_ssa_function_to_llvm(module, entry),
        directory=directory, optimization=optimization,
    )
    parameter_ids = {
        str(name): int(value_id)
        for name, value_id in dict(function.metadata.get("parameter_names") or ()).items()
    }
    named = {
        str(temporary): int(value_id)
        for temporary, value_id in dict(function.metadata.get("named_outputs") or ()).items()
    }
    stage = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == law)
    returned = next(node for node in ast.walk(stage) if isinstance(node, ast.Return)).value
    returned_names = [
        node.id if isinstance(node, ast.Name) else None
        for node in (returned.elts if isinstance(returned, ast.Tuple) else [returned])]
    literals = {
        node.targets[0].id: float(node.value.value)
        for node in stage.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool)}
    output_ids: dict[str, int] = {}
    constant_outputs: dict[str, float] = {}
    for output, temporary in zip(output_names, returned_names):
        if temporary in named:
            output_ids[output] = named[temporary]
        elif temporary in literals:
            constant_outputs[output] = literals[temporary]
        else:
            raise RuntimeError(f"{law}: output {output!r} has no lowered value")
    return LLVMPiece(
        artifact, argument_names, tuple(parameter_ids[name] for name in argument_names),
        output_names, output_ids, constant_outputs, batch,
        module=module, entry=entry, outputs=outputs, source=source,
        # Which compiler built it: recorded, not a cache key (a load
        # compares it with the sources on disk; piece_staleness).
        compiler=route_compiler_record(),
    )


def pieces_from_equations(
    laws: Mapping[str, Sequence[Any]], batch: int, *, directory: str | Path,
    source_compilation: str = "auto", partitioned_options: Mapping[str, Any] | None = None,
    optimization: str = "O2", progress=None,
) -> dict[str, LLVMPiece]:
    """Supply the existing law-piece schedule; every submission lowers in auto.

    Laws keep their own symbolic program books and piece ABIs. The public
    compiler schedules the supplied sources, then each complete source compile
    uses its own deployment regions. No replacement source partition is made.
    """
    from .identity_concordance import begin_identity_book, end_identity_book
    from .symbolic_equation_compiler import compile_sympy_equations
    from .external_functions import externals_for_law, post_lowered_external_calls
    from .fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not laws:
        return {}
    prepared, requests = {}, []
    for law, equations in laws.items():
        book, token = begin_identity_book()
        try:
            if progress is not None:
                progress(f"symbolic ingestion: {law}")
            compilation = compile_sympy_equations(tuple(equations), name=law)
            compilation, source, external_bindings = externals_for_law(
                compilation, law, batch, book=book,
            )
            prepared[law] = (compilation, source, book)
            requests.append({"label": law, "source": source, "entrypoint": law, "options": {
                "python_bindings": {"AbstractTensor": AbstractTensor, **external_bindings},
                "identity_book": book,
                "tensor_ssa_reference": c_backend_repository_ssa_reference(),
                "runtime_closure_only": True, "name": law,
                "extraction_contract": batch_contract(law, tuple(compilation.function.metadata["argument_names"]), batch),
            }})
        finally:
            end_identity_book(token)
    schedule = "parallel" if source_compilation in {"auto", "units"} else source_compilation
    results = lower_ast_source_to_ssa(requests, source_compilation=schedule,
                                    partitioned_options=dict(partitioned_options or {}))
    pieces = {}
    from .shell_telemetry import active_rule_batch_progress, compile_substage
    from time import monotonic
    compile_substage("native law-piece publication")
    for law, (module, outputs, exports) in zip(prepared, results):
        started = monotonic()
        compilation, source, book = prepared[law]
        if len(exports) != 1:
            raise ValueError(f"law {law!r} must retain its one authored export: {exports!r}")
        _book, token = begin_identity_book(book)
        try:
            post_lowered_external_calls(module, compilation, source, book)
            pieces[law] = _piece_from_lowered_law(
                compilation, law, batch, source, module, outputs, exports[0], root / law, optimization,
            )
        finally:
            end_identity_book(token)
        rule_progress = active_rule_batch_progress()
        if rule_progress is not None:
            rule_progress.complete(law, monotonic() - started + sum(
                row["seconds"] for row in module.metadata["compile_profile"]["stages"]))
    return pieces


def compose_native_package(source: str, entry: str, pieces: Mapping[str, LLVMPiece],
                           argument_names: Sequence[str], batch: int, *,
                           directory: str | Path, name: str | None = None,
                           optimization: str = "O2", link: str = "static"):
    """Lower authored Python that calls LLVM pieces into one compiled C module.

    ``source`` is the program as you would run it in Python; ``pieces`` binds
    the names it calls.  Returns the compiled ``CModuleArtifact``: call
    ``prepare_execution`` with feeds keyed by ``parameter_ids`` for the
    public buffers, or ``compile_standalone`` for a Python-free executable.
    """

    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .ssa_c_backend import emit_ssa_module_to_c

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, entry,
        python_bindings={"AbstractTensor": AbstractTensor, **dict(pieces)},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=name or entry, runtime_closure_only=True,
        extraction_contract=batch_contract(entry, tuple(argument_names), batch),
    )
    artifact = emit_ssa_module_to_c(module, exports[0])
    if not artifact.complete:
        raise RuntimeError(
            f"{entry}: C emission shortfalls: "
            + "; ".join(f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
    artifact = artifact.compile(directory, optimization=optimization, link=link)
    root = module.functions[exports[0]]
    return NativePackage(
        artifact=artifact,
        entry=exports[0],
        parameter_ids={
            str(k): int(v) for k, v in dict(root.metadata.get("parameter_names") or ()).items()},
    )


@dataclass(frozen=True)
class NativePackage:
    """A compiled system: the C module plus the public buffer ids by name."""

    artifact: Any
    entry: str
    parameter_ids: Mapping[str, int]

    @property
    def library_path(self) -> Path:
        return self.artifact.library_path

    @property
    def buffer_order(self) -> tuple[int, ...]:
        return tuple(self.artifact.buffer_order)

    @property
    def linked_llvm(self):
        return tuple(self.artifact.linked_llvm)

    def prepare_execution(self, feeds: Mapping[int, Any], **kwargs):
        return self.artifact.prepare_execution(feeds, **kwargs)

    def feeds(self, columns: Mapping[str, Any]) -> dict[int, Any]:
        """Public buffer feeds from columns named as the authored parameters."""
        return {self.parameter_ids[name]: columns[name] for name in self.parameter_ids}

    def compile_standalone(self, directory, feeds, **kwargs):
        return self.artifact.compile_standalone(directory, feeds, **kwargs)
