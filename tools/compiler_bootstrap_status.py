"""Inspect and exercise the live receipt-gated compiler bootstrap runtime.

This tool is deliberately read-only unless ``--write`` is supplied.  A native
artifact is reported as accepted only when it is pinned by the current
registry; loose verification receipts are evidence, not installed compiler
coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence


STATE_SCHEMA = "turing.compiler-bootstrap-graph-status.v1"


def _read_json(path: Path) -> dict[str, Any]:
    # The supervisor publishes with os.replace.  Retrying also makes the tool
    # useful with older state writers which may not have done so atomically.
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.02)
    raise AssertionError("unreachable")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_registry_entries(registry_path: Path) -> tuple[
    list[dict[str, Any]], list[dict[str, str]]
]:
    if not registry_path.is_file():
        return [], [{"product": "", "reason": "registry is absent"}]
    registry = _read_json(registry_path)
    if registry.get("schema") != "turing.compiler-bootstrap-registry.v1":
        return [], [{
            "product": "",
            "reason": "unsupported compiler bootstrap registry schema",
        }]
    accepted = []
    refused = []
    for record in registry.get("products") or ():
        product = Path(str(record.get("product") or "")).resolve()
        manifest_path = product / "manifest.json"
        try:
            if not manifest_path.is_file():
                raise ValueError("manifest is absent")
            if _sha256(manifest_path) != str(record.get("manifest_sha256") or ""):
                raise ValueError("manifest digest differs from registry")
            manifest = _read_json(manifest_path)
            source = Path(str(manifest.get("source") or "")).resolve()
            if not source.is_file():
                raise ValueError("authored source is absent")
            if _sha256(source) != str(record.get("source_sha256") or ""):
                raise ValueError("authored source digest differs from registry")
            for installable in record.get("installable") or ():
                receipt_path = product / str(
                    installable.get("verification_receipt") or ""
                )
                if not receipt_path.is_file():
                    raise ValueError(
                        f"verification receipt is absent: {receipt_path}"
                    )
                receipt = _read_json(receipt_path)
                if receipt.get("status") != "verified":
                    raise ValueError(
                        f"receipt is not verified: {receipt_path}"
                    )
                accepted.append({
                    "source": source.as_posix(),
                    "qualified_name": str(installable["qualified_name"]),
                    "activation_adapter": str(
                        installable.get("activation_adapter") or ""
                    ),
                    "product": product.as_posix(),
                    "verification_receipt": receipt_path.as_posix(),
                })
        except (KeyError, OSError, TypeError, ValueError) as error:
            refused.append({
                "product": product.as_posix(),
                "reason": f"{type(error).__name__}: {error}",
            })
    return accepted, refused


def _verified_receipt_count(root: Path) -> int:
    count = 0
    if not root.is_dir():
        return count
    for path in root.rglob("native-verification.json"):
        try:
            count += _read_json(path).get("status") == "verified"
        except (OSError, TypeError, ValueError):
            continue
    return count


def build_status(
    bootstrap_root: str | Path,
    registry_path: str | Path,
) -> dict[str, Any]:
    root = Path(bootstrap_root).resolve()
    state_path = root / "bootstrap-state.json"
    state = _read_json(state_path)
    registry = Path(registry_path).resolve()
    accepted, refused = _accepted_registry_entries(registry)
    accepted_keys = {
        (record["source"], record["qualified_name"]) for record in accepted
    }
    usage = {
        (str(record.get("source") or ""), str(record.get("qualified_name") or "")): {
            "call_count": int(record.get("call_count") or 0),
            "inclusive_seconds": float(record.get("inclusive_seconds") or 0.0),
        }
        for record in (state.get("compiler_usage") or {}).get("records") or ()
    }

    batches = []
    call_counts = {"accepted_native": 0, "attempted_frontier": 0, "pending": 0}
    for queue_index, record in enumerate(state.get("sources") or ()):
        source = str(record.get("source") or "")
        attempts = int(record.get("attempts") or 0)
        entries = []
        for name in record.get("entries") or ():
            key = (source, str(name))
            status = (
                "accepted_native" if key in accepted_keys else
                "attempted_frontier" if attempts else "pending"
            )
            call_counts[status] += 1
            entries.append({
                "qualified_name": str(name),
                "status": status,
                "observed_usage": usage.get(key, {
                    "call_count": 0, "inclusive_seconds": 0.0,
                }),
            })
        statuses = {entry["status"] for entry in entries}
        batches.append({
            "queue_index": queue_index,
            "source": source,
            "batch_index": int(record.get("batch_index") or 0),
            "attempts": attempts,
            "pending_deep_retry": bool(record.get("pending_deep_retry")),
            "dependency_batch_indices": list(
                record.get("dependency_batch_indices") or ()
            ),
            "status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
            "entries": entries,
        })

    waves = list(state.get("waves") or ())
    wave_counts: dict[str, int] = {}
    for wave in waves:
        key = str(wave.get("status") or "unknown")
        wave_counts[key] = wave_counts.get(key, 0) + 1
    generation = int(state.get("generation") or 0)
    current_wave = root / "waves" / f"generation_{generation:05d}"
    in_progress = current_wave.is_dir() and not (
        current_wave / "wave-result.json"
    ).is_file()
    total_calls = sum(call_counts.values())
    return {
        "schema": STATE_SCHEMA,
        "bootstrap_root": root.as_posix(),
        "state_path": state_path.as_posix(),
        "registry_path": registry.as_posix(),
        "state": {
            "status": str(state.get("status") or "unknown"),
            "generation": generation,
            "active_generation_in_progress": in_progress,
            "sweep": int(state.get("sweep") or 0),
            "cursor": int(state.get("cursor") or 0),
            "wave_counts": dict(sorted(wave_counts.items())),
            "hard_failure": state.get("hard_failure"),
        },
        "graph": {
            "total_batches": len(batches),
            "total_authored_calls": total_calls,
            "call_counts": call_counts,
            "accepted_percent": (
                100.0 * call_counts["accepted_native"] / total_calls
                if total_calls else 0.0
            ),
            "attempted_percent": (
                100.0 * (
                    call_counts["accepted_native"]
                    + call_counts["attempted_frontier"]
                ) / total_calls if total_calls else 0.0
            ),
            "batches": batches,
        },
        "native": {
            "accepted": accepted,
            "registry_refusals": refused,
            "verified_receipts_under_bootstrap_root": _verified_receipt_count(root),
            "warning": (
                "verified receipts not pinned by the current registry are not "
                "accepted compiler deployments"
            ),
        },
        "usage_census": {
            "generation": (state.get("compiler_usage") or {}).get("generation"),
            "observed_callables": len(usage),
        },
    }


def _exercise(report: dict[str, Any]) -> None:
    from src.compiler.compiler_bootstrap_runtime import (
        activate_registered_compiler_bootstraps,
        compiler_bootstrap_registry_state,
        compiler_bootstrap_runtime_state,
    )

    activations = activate_registered_compiler_bootstraps()
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, outputs, exports = lower_ast_source_to_ssa(
        "def bootstrap_probe(value: float) -> float:\n"
        "    return value * 2.0 + 1.0\n",
        "bootstrap_probe",
    )
    # Exercise the real compiler caller of today's accepted deployment.  This
    # is not a direct DLL probe: select_deployment_strategy reaches the
    # installed plan_compute_dispatch through its ordinary compiler edge.
    from src.compiler.deployment_lowering import (
        ComputeDispatchLimits,
        select_deployment_strategy,
    )

    deployment = select_deployment_strategy(
        backend="webgpu",
        execution_class="shader-compute",
        work=1024,
        compute_limits=ComputeDispatchLimits(
            (65535, 65535, 65535), (1024, 1024, 64), 1024,
        ),
    )
    registry_state = compiler_bootstrap_registry_state()
    runtime_routes = list(compiler_bootstrap_runtime_state())
    if registry_state["failures"]:
        raise RuntimeError(
            "registered compiler bootstrap activation was refused: "
            f"{registry_state['failures']}"
        )
    expected = {
        item["qualified_name"] for item in report["native"]["accepted"]
    }
    loaded = {item["qualified_name"] for item in runtime_routes}
    if expected != loaded:
        raise RuntimeError(
            "fresh compiler did not load its complete accepted native set: "
            f"expected={sorted(expected)!r}, loaded={sorted(loaded)!r}"
        )
    report["exercise"] = {
        "status": "compiled-with-registered-bootstrap",
        "new_activations": [item.to_mapping() for item in activations],
        "registry": registry_state,
        "runtime_routes": runtime_routes,
        "ssa_function_count": len(module.functions),
        "output_count": len(outputs),
        "exports": sorted(map(str, exports)),
        "deployment": deployment.as_record(),
    }


def _print_human(report: dict[str, Any]) -> None:
    state = report["state"]
    graph = report["graph"]
    native = report["native"]
    calls = graph["call_counts"]
    running = "yes" if state["active_generation_in_progress"] else "no"
    print(
        f"bootstrap: {state['status']} generation={state['generation']} "
        f"sweep={state['sweep']} active_wave={running}"
    )
    hard_failure = state.get("hard_failure") or {}
    chief = hard_failure.get("chief_failure") or {}
    if chief:
        print(
            "chief failure: "
            f"{', '.join(chief.get('qualified_names') or ['<unknown>'])}: "
            f"{chief.get('error_type')}: {chief.get('error')}"
        )
        print(f"  evidence: {chief.get('artifact')}")
    print(
        f"graph: {graph['total_authored_calls']} calls in "
        f"{graph['total_batches']} batches; "
        f"{calls['accepted_native']} accepted native "
        f"({graph['accepted_percent']:.3f}%), "
        f"{calls['attempted_frontier']} attempted/frontier, "
        f"{calls['pending']} pending"
    )
    print(
        f"waves: " + ", ".join(
            f"{name}={count}" for name, count in state["wave_counts"].items()
        )
    )
    print(
        f"evidence: {native['verified_receipts_under_bootstrap_root']} verified "
        "receipts under this run; only registry-pinned entries count as accepted"
    )
    if native["accepted"]:
        print("accepted native compiler calls:")
        for item in native["accepted"]:
            print(
                f"  {item['qualified_name']} [{item['activation_adapter']}] "
                f"<- {item['product']}"
            )
    else:
        print("accepted native compiler calls: none")
    if native["registry_refusals"]:
        print("registry refusals:")
        for item in native["registry_refusals"]:
            print(f"  {item['product']}: {item['reason']}")
    pending = [
        entry
        for batch in graph["batches"]
        for entry in batch["entries"]
        if entry["status"] == "pending"
    ]
    pending.sort(key=lambda item: (
        -item["observed_usage"]["inclusive_seconds"],
        -item["observed_usage"]["call_count"],
        item["qualified_name"],
    ))
    print("hottest pending observed compiler calls:")
    for item in pending[:10]:
        usage = item["observed_usage"]
        print(
            f"  {item['qualified_name']}: calls={usage['call_count']} "
            f"inclusive={usage['inclusive_seconds']:.3f}s"
        )
    if "exercise" in report:
        exercise = report["exercise"]
        print(
            f"exercise: {exercise['status']}; "
            f"ssa_functions={exercise['ssa_function_count']}; "
            f"runtime_deployments={len(exercise['runtime_routes'])}"
        )
        for item in exercise["runtime_routes"]:
            print(
                f"  {item['qualified_name']}: verified="
                f"{item['verification_status']} native_calls="
                f"{item['runtime_native_calls']} fallback_calls="
                f"{item['runtime_fallback_calls']} post_activation_native="
                f"{item['post_activation_native_calls']} "
                f"post_activation_fallback="
                f"{item['post_activation_fallback_calls']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report exact accepted/native coverage of compiler bootstrap",
    )
    parser.add_argument(
        "--bootstrap-root", type=Path,
        default=Path("build/compiler-exponential-bootstrap"),
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("build/compiler-bootstrap-registry.json"),
    )
    parser.add_argument(
        "--exercise", action="store_true",
        help="activate the registry and lower a small function to SSA",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--write", type=Path,
        help="also write the complete per-batch graph report as JSON",
    )
    arguments = parser.parse_args(argv)
    report = build_status(arguments.bootstrap_root, arguments.registry)
    if arguments.exercise:
        _exercise(report)
    if arguments.write is not None:
        arguments.write.parent.mkdir(parents=True, exist_ok=True)
        arguments.write.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
