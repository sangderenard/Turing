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
from dataclasses import dataclass
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
    calls: int = 0
    seconds: float = 0.0

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
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return results


def _lower_law(compilation: Any, law: str, batch: int, backend: str) -> LawKernel:
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference)
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    metadata = compilation.function.metadata
    argument_names = tuple(metadata["argument_names"])
    output_names = tuple(metadata["output_names"])
    source = symbolic_abstract_tensor_source(compilation, "tick")
    lowered = lower_ast_source_to_ssa(
        source, "tick",
        python_bindings={"AbstractTensor": AbstractTensor},
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
    if len(returned_names) != len(output_names):
        raise RuntimeError(
            f"{law}: stage returns {len(returned_names)} values, the law "
            f"declares {len(output_names)} outputs")
    id_of_temporary = {
        str(temporary): int(value_id)
        for temporary, value_id in tuple(function.metadata.get("named_outputs") or ())
    }
    unresolved = [
        output for output, temporary in zip(output_names, returned_names)
        if temporary is None or temporary not in id_of_temporary]
    if unresolved:
        raise RuntimeError(
            f"{law}: outputs without a lowered value: {unresolved[:5]}")
    output_ids = {
        output: id_of_temporary[temporary]
        for output, temporary in zip(output_names, returned_names)
    }
    argument_ids = tuple(int(value.id) for value in function.args)
    if len(argument_ids) != len(argument_names):
        raise RuntimeError(
            f"{law}: entry takes {len(argument_ids)} values, the law "
            f"declares {len(argument_names)} arguments")
    if backend != "llvm":
        raise RuntimeError(f"{law}: backend {backend!r} stand-in not wired yet")
    from .ssa_llvm_backend import emit_ssa_function_to_llvm

    artifact = emit_ssa_function_to_llvm(module, entry)
    if not artifact.complete:
        raise RuntimeError(
            f"{law}: LLVM shortfalls: "
            + "; ".join(s.reason for s in artifact.shortfalls[:3]))
    exposed = set(int(value_id) for value_id in artifact.buffer_order)
    missing = [name for name, value_id in output_ids.items() if value_id not in exposed]
    if missing:
        raise RuntimeError(f"{law}: outputs not exposed by the kernel ABI: {missing[:5]}")
    return LawKernel(
        law=law, batch=batch, backend=backend, artifact=artifact,
        argument_names=argument_names, argument_ids=argument_ids,
        output_ids=output_ids,
    )


def _cache_key(compilation: Any, law: str, batch: int, backend: str) -> str:
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    source = symbolic_abstract_tensor_source(compilation, "tick")
    digest = hashlib.sha256(
        f"{_CACHE_VERSION}|{backend}|{batch}|{source}".encode("utf-8")).hexdigest()
    return digest[:20]


def law_kernel(compilation: Any, law: str, batch: int, backend: str) -> LawKernel:
    """Compiled kernel for (law, batch): from the disk cache or lowered now."""

    from .ssa_llvm_backend import compile_artifact

    key = _cache_key(compilation, law, batch, backend)
    directory = cache_root() / law / f"{backend}_b{batch}_{key}"
    record = directory / "kernel.pkl"
    if record.is_file():
        try:
            with record.open("rb") as handle:
                kernel = pickle.load(handle)
            libraries = sorted(directory.glob("*.dll"))
            if libraries:
                kernel.artifact.library_path = libraries[0]
                kernel.artifact._entry = None
                _log(f"{law}: batch {batch} kernel from cache {directory.name}")
                return kernel
        except Exception as error:  # a stale or foreign record is rebuilt
            _log(f"{law}: cache record unreadable ({type(error).__name__}), rebuilding")
    started = time.time()
    kernel = _lower_law(compilation, law, batch, backend)
    directory.mkdir(parents=True, exist_ok=True)
    compile_artifact(kernel.artifact, directory=directory, optimization="O2")
    _log(f"{law}: batch {batch} lowered+compiled ({backend}) in "
         f"{time.time() - started:.1f}s -> {directory}")
    try:
        entry = kernel.artifact._entry
        kernel.artifact._entry = None
        with record.open("wb") as handle:
            pickle.dump(kernel, handle)
        kernel.artifact._entry = entry
    except Exception as error:
        _log(f"{law}: kernel record not cached ({type(error).__name__}: {error})")
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

    backend = native_backend()
    if not backend:
        return bindings
    armed = []
    for name, compilation in compilations.items():
        if name not in bindings or not _law_selected(name):
            continue
        stage = NativeLawStage(name, compilation, bindings[name], backend)
        bindings[name] = stage
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
