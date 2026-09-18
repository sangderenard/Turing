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

from .native_law_kernels import LLVMPiece, batch_contract


def piece_from_law(compilation: Any, law: str, batch: int, *,
                   directory: str | Path | None = None,
                   optimization: str = "O2") -> LLVMPiece:
    """Lower one compiled SymPy law to an LLVM piece, on its own."""

    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    metadata = compilation.function.metadata
    argument_names = tuple(metadata["argument_names"])
    output_names = tuple(metadata["output_names"])
    source = symbolic_abstract_tensor_source(compilation, law)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, law,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=law, runtime_closure_only=True,
        extraction_contract=batch_contract(law, argument_names, batch),
    )
    entry = exports[0]
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
    )


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
