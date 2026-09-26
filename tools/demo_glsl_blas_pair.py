"""Build and run standalone source-vs-intrinsic GLSL GEMM deployments.

The generated executables own hidden SDL2 OpenGL contexts. Python is only the
build/measurement driver; neither executable links or calls Python or Turing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.deployment_lowering import ComputeDispatchLimits
from src.compiler.glsl_blas_deployment import build_gemm_deployment_pair


def _active_limits() -> tuple[ComputeDispatchLimits, str]:
    from src.common.tensors.accelerator_backends import glsl_backend
    from src.common.tensors.accelerator_backends.glsl_backend import (
        release_gl_context,
        require_gl_context,
    )

    info = require_gl_context()
    limits = glsl_backend._compute_limits()
    result = ComputeDispatchLimits(
        max_group_count=limits.max_group_count,
        max_group_size=limits.max_group_size,
        max_invocations=limits.max_invocations,
    )
    device = f"{info.get('vendor', '')} {info.get('renderer', '')}".strip()
    release_gl_context()
    return result, device


def _sdl_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
    else:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        path = Path(pygame.__file__).resolve().parent / "SDL2.dll"
    if not path.is_file():
        raise FileNotFoundError(f"SDL2 runtime not found: {path}")
    return path


def _compile(shell: Path, executable: Path) -> None:
    command = [
        sys.executable, "-m", "ziglang", "cc", "-O3",
        str(shell), "-o", str(executable),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def _run(executable: Path, shader: Path, sdl: Path, output: Path) -> dict:
    completed = subprocess.run(
        [str(executable), str(shader), str(sdl), str(output)],
        cwd=str(executable.parent), capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
    line = next(
        line for line in reversed(completed.stdout.splitlines())
        if line.strip().startswith("{")
    )
    return json.loads(line)


def run_comparison(
    m: int,
    n: int,
    k: int,
    *,
    output: Path,
    sdl2: str | None = None,
    warmups: int = 3,
    iterations: int = 20,
    build_only: bool = False,
    variants: tuple[str, ...] = ("source_algorithm", "glslblas_gemm"),
) -> dict:
    """Build and optionally run the standalone GLSL GEMM comparison.

    This is shared by the focused GLSL CLI and the end-to-end tiled deployment
    demo so both commands measure the exact same generated shader products.
    """

    unknown = set(variants) - {"source_algorithm", "glslblas_gemm"}
    if unknown:
        raise ValueError(f"unknown GLSL GEMM variants: {sorted(unknown)}")
    limits, device = _active_limits()
    deployments = build_gemm_deployment_pair(
        m, n, k, limits=limits,
        warmup_dispatches=warmups,
        measured_dispatches=iterations,
    )
    root = output.resolve()
    records = []
    sdl = None if build_only else _sdl_path(sdl2)
    outputs = []
    for deployment in deployments:
        if deployment.variant not in variants:
            continue
        written = deployment.write(root / deployment.variant)
        executable = written.shell_path.with_suffix(".exe")
        _compile(written.shell_path, executable)
        record = {
            "variant": deployment.variant,
            "manifest": str(written.manifest_path),
            "shader": str(written.shader_path),
            "executable": str(executable),
        }
        if sdl is not None:
            result = executable.parent / "C.bin"
            record["measurement"] = _run(
                executable, written.shader_path, sdl, result,
            )
            outputs.append((deployment.variant, result))
        records.append(record)

    report = {
        "schema": "turing.glsl-blas-comparison.v1",
        "device": device,
        "shape": {"m": m, "n": n, "k": k},
        "deployments": records,
    }
    if len(outputs) == 2:
        baseline = np.fromfile(outputs[0][1], dtype=np.float32)
        specialized = np.fromfile(outputs[1][1], dtype=np.float32)
        difference = np.abs(baseline - specialized)
        report["equivalence"] = {
            "elements": int(baseline.size),
            "max_abs": float(difference.max(initial=0.0)),
            "allclose": bool(np.allclose(
                baseline, specialized, rtol=2e-5, atol=2e-5,
            )),
        }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=Path("build/glsl-blas-pair"),
    )
    parser.add_argument("--sdl2")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    report = run_comparison(
        args.m, args.n, args.k,
        output=args.output,
        sdl2=args.sdl2,
        warmups=args.warmups,
        iterations=args.iterations,
        build_only=args.build_only,
    )
    report_path = report.pop("report_path")
    print(json.dumps(report, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
