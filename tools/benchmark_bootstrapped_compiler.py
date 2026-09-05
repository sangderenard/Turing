"""Compare the authored Python compiler with the live bootstrapped compiler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SAMPLE = """\
def sample(values: list[float], scale: float) -> float:
    total = 0.0
    for value in values:
        total = total + value * scale
    return total
"""


def _one_compile() -> dict[str, Any]:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.deployment_lowering import (
        ComputeDispatchLimits,
        select_deployment_strategy,
    )

    module, outputs, exports = lower_ast_source_to_ssa(SAMPLE, "sample")
    deployment = select_deployment_strategy(
        backend="webgpu",
        execution_class="shader-compute",
        work=4096,
        compute_limits=ComputeDispatchLimits(
            (65535, 65535, 65535), (1024, 1024, 64), 1024,
        ),
    )
    functions = []
    instruction_count = 0
    for function in module.functions.values():
        functions.append(str(function.name))
        instruction_count += sum(
            len(block.instrs) for block in function.blocks.values()
        )
    return {
        "functions": sorted(functions),
        "instruction_count": instruction_count,
        "output_count": len(outputs),
        "exports": sorted(map(str, exports)),
        "deployment": deployment.as_record(),
    }


def _worker(repetitions: int) -> int:
    from src.compiler.compiler_bootstrap_runtime import (
        activate_registered_compiler_bootstraps,
        compiler_bootstrap_registry_state,
        compiler_bootstrap_runtime_state,
    )

    import_started = time.perf_counter()
    activations = activate_registered_compiler_bootstraps()
    activation_seconds = time.perf_counter() - import_started
    samples = []
    signature = None
    for _ in range(repetitions):
        started = time.perf_counter()
        current = _one_compile()
        samples.append(time.perf_counter() - started)
        if signature is None:
            signature = current
        elif signature != current:
            raise RuntimeError("compiler sample changed across repetitions")
    print(json.dumps({
        "schema": "turing.bootstrapped-compiler-benchmark-worker.v1",
        "process_id": os.getpid(),
        "activation_seconds": activation_seconds,
        "compile_seconds": samples,
        "first_compile_seconds": samples[0],
        "warm_compile_median_seconds": statistics.median(samples[1:] or samples),
        "signature": signature,
        "activations": [item.to_mapping() for item in activations],
        "registry": compiler_bootstrap_registry_state(),
        "runtime_routes": list(compiler_bootstrap_runtime_state()),
    }, sort_keys=True), flush=True)
    return 0


def _run_lane(
    label: str,
    registry: Path,
    repetitions: int,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["TURING_COMPILER_BOOTSTRAP_REGISTRY"] = str(registry.resolve())
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.benchmark_bootstrapped_compiler",
        "--worker",
        "--repetitions",
        str(repetitions),
    ]
    print(f"{label}: starting fresh compiler process", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"{label} compiler failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(
            f"{label} compiler emitted no benchmark receipt:\n{completed.stdout}"
        )
    result = json.loads(lines[-1])
    result["label"] = label
    result["fresh_process_seconds"] = elapsed
    if completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()
    print(
        f"{label}: activation={result['activation_seconds']:.6f}s "
        f"first={result['first_compile_seconds']:.6f}s "
        f"warm-median={result['warm_compile_median_seconds']:.6f}s "
        f"process={elapsed:.6f}s",
        flush=True,
    )
    return result


def compare_results(
    python_result: dict[str, Any],
    bootstrapped_result: dict[str, Any],
) -> dict[str, Any]:
    equivalent = python_result["signature"] == bootstrapped_result["signature"]
    python_warm = float(python_result["warm_compile_median_seconds"])
    bootstrap_warm = float(bootstrapped_result["warm_compile_median_seconds"])
    return {
        "equivalent": equivalent,
        "warm_seconds_difference": bootstrap_warm - python_warm,
        "warm_speedup": python_warm / bootstrap_warm if bootstrap_warm else None,
        "first_compile_seconds_difference": (
            float(bootstrapped_result["first_compile_seconds"])
            - float(python_result["first_compile_seconds"])
        ),
        "activation_seconds_difference": (
            float(bootstrapped_result["activation_seconds"])
            - float(python_result["activation_seconds"])
        ),
        "activation_plus_first_seconds_difference": (
            float(bootstrapped_result["activation_seconds"])
            + float(bootstrapped_result["first_compile_seconds"])
            - float(python_result["activation_seconds"])
            - float(python_result["first_compile_seconds"])
        ),
        "fresh_process_seconds_difference": (
            float(bootstrapped_result["fresh_process_seconds"])
            - float(python_result["fresh_process_seconds"])
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same sample through source-only and live bootstrapped "
            "compiler processes"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("build/compiler-bootstrap-registry.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("build/compiler-bootstrap-benchmark.json"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    if arguments.worker:
        return _worker(arguments.repetitions)

    with tempfile.TemporaryDirectory(prefix="turing-python-compiler-") as raw:
        absent_registry = Path(raw) / "absent-registry.json"
        python_result = _run_lane(
            "authored-python", absent_registry, arguments.repetitions,
        )
    bootstrapped_result = _run_lane(
        "receipt-gated-bootstrap", arguments.registry, arguments.repetitions,
    )
    comparison = compare_results(python_result, bootstrapped_result)
    report = {
        "schema": "turing.bootstrapped-compiler-benchmark.v1",
        "sample": SAMPLE,
        "repetitions": arguments.repetitions,
        "authored_python": python_result,
        "bootstrapped": bootstrapped_result,
        "comparison": comparison,
    }
    if not comparison["equivalent"]:
        raise RuntimeError("bootstrapped compiler changed the sample result")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(
        "difference: "
        f"warm={comparison['warm_seconds_difference']:+.6f}s, "
        f"speedup={comparison['warm_speedup']:.3f}x, "
        f"first={comparison['first_compile_seconds_difference']:+.6f}s, "
        f"activation+first="
        f"{comparison['activation_plus_first_seconds_difference']:+.6f}s, "
        f"fresh-process={comparison['fresh_process_seconds_difference']:+.6f}s",
        flush=True,
    )
    native_calls = sum(
        int(item.get("post_activation_native_calls") or 0)
        for item in bootstrapped_result["runtime_routes"]
    )
    fallback_calls = sum(
        int(item.get("post_activation_fallback_calls") or 0)
        for item in bootstrapped_result["runtime_routes"]
    )
    print(
        f"bootstrapped routes: native={native_calls}, fallback={fallback_calls}, "
        f"accepted_deployments={len(bootstrapped_result['runtime_routes'])}",
        flush=True,
    )
    print(f"equivalent: yes; report: {arguments.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
