"""Precompile and execute the shared AbstractTensor backend torture matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from .artifact_cache import RepositoryArtifactCache, repository_cache_root
from .tensor_torture import (
    TortureTier,
    capture_torture_case,
    eager_backend_result,
    tensor_torture_cases,
)


BACKENDS = (
    "raw_numpy",
    "numpy",
    "torch",
    "torch_cuda",
    "nodus",
    "c_jit",
    "glsl_jit",
    "llvm_jit",
    "fortran",
    "c_aot",
    "glsl_aot",
    "python_numpy",
    "python_torch",
    "python_nodus",
)
DEFAULT_TIERS = (
    TortureTier.ISOLATED,
    TortureTier.GRAB_BAG,
    TortureTier.ADVANCED,
)


@dataclass(frozen=True)
class TortureMatrixRow:
    case: str
    tier: str
    backend: str
    status: str
    compile_ns: int
    execute_ns: int
    shell_ns: int
    device_ns: int
    cache_hit: bool | None
    max_abs_error: float | None
    error: str | None


def _compare(case, actual) -> float:
    expected = case.numpy_reference()
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"output names differ: {sorted(actual)} != {sorted(expected)}"
        )
    largest = 0.0
    rtol = max(case.rtol, 5.0e-5)
    atol = max(case.atol, 5.0e-5)
    for name, expected_value in expected.items():
        actual_value = np.asarray(actual[name])
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{case.name}:{name}",
        )
        finite = np.isfinite(actual_value) & np.isfinite(expected_value)
        if finite.any():
            if (
                np.issubdtype(actual_value.dtype, np.bool_)
                or np.issubdtype(expected_value.dtype, np.bool_)
            ):
                difference = np.not_equal(
                    actual_value[finite], expected_value[finite]
                ).astype(np.float64)
            else:
                difference = np.abs(
                    actual_value[finite] - expected_value[finite]
                )
            largest = max(
                largest,
                float(np.max(difference)),
            )
    return largest


def _run_one(
    case,
    backend: str,
    *,
    cache: RepositoryArtifactCache,
    compile_only: bool,
    trig_solver: str,
    trig_epsilon: float | None,
    llvm_profile: Any | None = None,
) -> TortureMatrixRow:
    compile_started = perf_counter_ns()
    execute_ns = 0
    shell_ns = 0
    device_ns = 0
    cache_hit = None
    max_abs_error = None
    try:
        if backend == "raw_numpy":
            compiled = None
            compile_ns = perf_counter_ns() - compile_started
            execute_started = perf_counter_ns()
            actual = case.numpy_reference()
            execute_ns = perf_counter_ns() - execute_started
        elif backend in {"numpy", "torch", "torch_cuda", "nodus"}:
            device = None
            if backend == "numpy":
                from ..numpy_backend import NumPyTensorOperations

                backend_type = NumPyTensorOperations
            elif backend == "nodus":
                from .nodus_backend import NodusTensorOperations

                backend_type = NodusTensorOperations
            else:
                from ..torch_backend import PyTorchTensorOperations, torch

                if torch is None:
                    raise RuntimeError("PyTorch is not installed")
                backend_type = PyTorchTensorOperations
                if backend == "torch_cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available for torch_cuda")
                    device = "cuda"
            compiled = None
            compile_ns = perf_counter_ns() - compile_started
            execute_started = perf_counter_ns()
            actual = eager_backend_result(case, backend_type, device=device)
            execute_ns = perf_counter_ns() - execute_started
        elif backend in {"c_aot", "glsl_aot"}:
            # AOT precompiler route: ProcessGraph.build_from_ast ->
            # reduce_abstract_tensor_topology -> strategize_shell_deployment
            # -> compile_process_graph/capture_fused_programs, not the
            # tape-walking JIT backends above. See aot_compile.py and
            # docs/PIPELINE_STAGE_DISAMBIGUATION.md.
            if case.ast_source is None or case.ast_entrypoint is None:
                raise ValueError(
                    f"{case.name} has no ast_source; not expressible for "
                    "the AOT precompiler route"
                )
            from .aot_compile import compile_ast_aot

            aot_backend = "glsl" if backend == "glsl_aot" else "c"
            compiled = None
            compile_ns = perf_counter_ns() - compile_started
            execute_started = perf_counter_ns()
            result = compile_ast_aot(
                case.ast_source,
                case.ast_entrypoint,
                case.inputs,
                backend=aot_backend,
            )
            execute_ns = perf_counter_ns() - execute_started
            (_, value), = result.outputs.items()
            value = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            actual = {next(iter(case.numpy_reference())): value}
        elif backend in {"python_numpy", "python_torch", "python_nodus"}:
            # The tape-walking JIT route (see program_order.py's module
            # docstring: this is a different product from the AOT
            # ast=/c_aot/glsl_aot route above, not the same pipeline). The
            # exact recorded AbstractTensor program is captured as a
            # FusedProgram, then that numeric IR is lowered to a real
            # ast.Module in the requested dialect
            # (fused_program_python_backend) instead of compiling C/GLSL/
            # LLVM -- elementwise-only, so non-elementwise cases (matmul,
            # reshape, reductions, where, stack/cat) fail here by design.
            from ....compiler.fused_program_python_backend import (
                compile_single_region_python,
            )
            from ..fused_ir import ordered_feed_ids
            from .c_primitive_program import compile_elementwise_tape

            dialect = {
                "python_numpy": "numpy",
                "python_torch": "torch",
                "python_nodus": "abstract_tensor",
            }[backend]
            captured = capture_torture_case(case)
            output_name = next(iter(captured.outputs))
            capture = compile_elementwise_tape(captured.tape, dict(captured.outputs))
            feed_names = {
                id(tensor): name for name, tensor in captured.inputs.items()
            }
            compiled_program = compile_single_region_python(
                capture.program,
                feed_names,
                dialect=dialect,
                abstract_tensor_backend="nodus" if dialect == "abstract_tensor" else None,
            )
            compile_ns = perf_counter_ns() - compile_started
            if compile_only:
                return TortureMatrixRow(
                    case=case.name,
                    tier=case.tier.value,
                    backend=backend,
                    status="compiled",
                    compile_ns=compile_ns,
                    execute_ns=0,
                    shell_ns=0,
                    device_ns=0,
                    cache_hit=None,
                    max_abs_error=None,
                    error=None,
                )
            execute_started = perf_counter_ns()
            if dialect == "numpy":
                call_inputs = {
                    name: np.asarray(value) for name, value in case.inputs.items()
                }
            elif dialect == "torch":
                import torch

                call_inputs = {
                    name: torch.as_tensor(np.asarray(value))
                    for name, value in case.inputs.items()
                }
            else:
                from ..abstraction import AbstractTensor

                call_inputs = {
                    name: AbstractTensor.tensor(np.asarray(value))
                    for name, value in case.inputs.items()
                }
            parameters = tuple(
                feed_names[feed_id] for feed_id in ordered_feed_ids(capture.program)
            )
            result = compiled_program.callable(
                *(call_inputs[name] for name in parameters)
            )
            execute_ns = perf_counter_ns() - execute_started
            if hasattr(result, "detach"):
                result_value = result.detach().cpu().numpy()
            else:
                result_value = getattr(result, "data", result)
            actual = {output_name: np.asarray(result_value)}
        else:
            captured = capture_torture_case(case)
            if backend == "c_jit":
                from .c_jit_backend import compile_torture_case_to_c

                compiled = compile_torture_case_to_c(captured, cache=cache)
            elif backend == "glsl_jit":
                from .glsl_backend import require_gl_context
                from .glsl_jit_backend import compile_torture_case_to_glsl

                require_gl_context()
                compiled = compile_torture_case_to_glsl(
                    captured, cache=cache
                )
            elif backend == "llvm_jit":
                from .llvm_jit_backend import compile_torture_case_to_llvm

                compiled = compile_torture_case_to_llvm(
                    captured,
                    cache=cache,
                    llvm_profile=llvm_profile,
                    trig_solver=trig_solver,
                    trig_epsilon=trig_epsilon,
                )
            elif backend == "fortran":
                from .fortran_jit_backend import compile_torture_case_to_fortran

                compiled = compile_torture_case_to_fortran(captured, cache=cache)
            else:
                raise ValueError(f"unknown torture backend {backend!r}")
            compile_ns = perf_counter_ns() - compile_started
            cache_artifact = getattr(
                compiled,
                "cache_artifact",
                getattr(compiled, "source_artifact", None),
            )
            cache_hit = bool(cache_artifact.hit)
            if compile_only:
                return TortureMatrixRow(
                    case=case.name,
                    tier=case.tier.value,
                    backend=backend,
                    status="compiled",
                    compile_ns=compile_ns,
                    execute_ns=0,
                    shell_ns=0,
                    device_ns=0,
                    cache_hit=cache_hit,
                    max_abs_error=None,
                    error=None,
                )
            execute_started = perf_counter_ns()
            execution = compiled.execute(case.inputs)
            execute_ns = perf_counter_ns() - execute_started
            actual = execution.outputs
            shell_ns = execution.profile.shell_ns
            device_ns = execution.profile.device_ns
        max_abs_error = _compare(case, actual)
        return TortureMatrixRow(
            case=case.name,
            tier=case.tier.value,
            backend=backend,
            status="passed",
            compile_ns=compile_ns,
            execute_ns=execute_ns,
            shell_ns=shell_ns,
            device_ns=device_ns,
            cache_hit=cache_hit,
            max_abs_error=max_abs_error,
            error=None,
        )
    except BaseException as error:
        return TortureMatrixRow(
            case=case.name,
            tier=case.tier.value,
            backend=backend,
            status="failed",
            compile_ns=perf_counter_ns() - compile_started,
            execute_ns=execute_ns,
            shell_ns=shell_ns,
            device_ns=device_ns,
            cache_hit=cache_hit,
            max_abs_error=max_abs_error,
            error=f"{type(error).__name__}: {error}",
        )


def _resolve_llvm_profile(name: str):
    """Map the CLI name onto an optimization profile."""

    from ....compiler.llvm_optimizing_pipeline import (
        REFERENCE_PROFILE,
        tuned_host_profile,
    )

    return REFERENCE_PROFILE if name == "reference" else tuned_host_profile()


def run_torture_matrix(
    *,
    backends: Iterable[str] = BACKENDS,
    tiers: Iterable[TortureTier | str] = DEFAULT_TIERS,
    case_names: Iterable[str] | None = None,
    compile_only: bool = False,
    trig_solver: str = "lut",
    trig_epsilon: float | None = None,
    cache: RepositoryArtifactCache | None = None,
    llvm_profile: Any | None = None,
) -> tuple[TortureMatrixRow, ...]:
    selected_backends = tuple(backends)
    unknown = set(selected_backends) - set(BACKENDS)
    if unknown:
        raise ValueError(f"unknown torture backends: {sorted(unknown)}")
    selected_tiers = {TortureTier(tier) for tier in tiers}
    selected_names = None if case_names is None else set(case_names)
    cases = tuple(
        case
        for case in tensor_torture_cases(
            include_large=TortureTier.LARGE in selected_tiers
        )
        if case.tier in selected_tiers
        and (selected_names is None or case.name in selected_names)
    )
    if selected_names is not None:
        missing = selected_names - {case.name for case in cases}
        if missing:
            raise ValueError(f"unknown or excluded torture cases: {sorted(missing)}")
    cache = cache or RepositoryArtifactCache()
    return tuple(
        _run_one(
            case,
            backend,
            cache=cache,
            compile_only=compile_only,
            trig_solver=trig_solver,
            trig_epsilon=trig_epsilon,
            llvm_profile=llvm_profile,
        )
        for case in cases
        for backend in selected_backends
    )


def format_torture_matrix(rows: Iterable[TortureMatrixRow]) -> str:
    lines = []
    for row in rows:
        timing = (
            f"compile={row.compile_ns / 1e6:.3f}ms "
            f"execute={row.execute_ns / 1e6:.3f}ms"
        )
        if row.backend in {"c_jit", "glsl_jit", "llvm_jit", "fortran"}:
            timing += (
                f" shell={row.shell_ns / 1e6:.3f}ms"
                f" device={row.device_ns / 1e6:.3f}ms"
                f" cache={'hit' if row.cache_hit else 'miss'}"
            )
        detail = f" error={row.error}" if row.error else ""
        lines.append(
            f"{row.tier:9} {row.case:28} {row.backend:10} "
            f"{row.status:8} {timing}{detail}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKENDS,
        dest="backends",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=[tier.value for tier in TortureTier],
        dest="tiers",
    )
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--trig-solver",
        choices=("lut", "continuous"),
        default="lut",
    )
    parser.add_argument("--trig-epsilon", type=float)
    parser.add_argument(
        "--llvm-profile",
        choices=("tuned", "reference"),
        default="tuned",
        help=(
            "LLVM codegen settings. 'tuned' runs -O3 with the host CPU named, "
            "noalias asserted, and the host's preferred vector width; "
            "'reference' is the unoptimized opt=0 path, kept as a "
            "differential baseline."
        ),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    rows = run_torture_matrix(
        backends=args.backends or BACKENDS,
        tiers=args.tiers or DEFAULT_TIERS,
        case_names=args.case_names,
        compile_only=args.compile_only,
        trig_solver=args.trig_solver,
        trig_epsilon=args.trig_epsilon,
        llvm_profile=_resolve_llvm_profile(args.llvm_profile),
    )
    print(format_torture_matrix(rows))
    report_path = args.json
    if report_path is None:
        report_path = repository_cache_root() / "last-torture-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failures = sum(row.status == "failed" for row in rows)
    print(
        f"summary: rows={len(rows)} failures={failures} "
        f"report={report_path}"
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKENDS",
    "DEFAULT_TIERS",
    "TortureMatrixRow",
    "format_torture_matrix",
    "run_torture_matrix",
]
