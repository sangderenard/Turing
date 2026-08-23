"""Bootstrap the compiler through joined, fresh-process generations.

Each generation compiles one automatically selected compiler source catalogue.
Its bounded workers all terminate before the generation publishes a result.
The supervisor then starts a fresh Python process, which loads the newly
published verified-product registry before compiling the next catalogue. A
timed-out unit is not immediately repeated by the subdivision crawler.  Once
the bounded wave has done the configured minimum amount of work, that unit is
retried alone, without a time ceiling but with the same memory ceiling, in the
next fresh process.  It is retried again only after the registry advances.  A
complete deterministic sweep with no registry, verified-region, or normalized
frontier change is the terminal fixed point.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

from src.compiler.project_compilation_product import (
    DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    compile_project_bootstrap_creep,
    discover_authored_calls,
    authored_call_dependencies,
)


STATE_SCHEMA = "turing.exponential-compiler-bootstrap-state.v2"
LEGACY_STATE_SCHEMA = "turing.exponential-compiler-bootstrap-state.v1"
WAVE_SCHEMA = "turing.exponential-compiler-bootstrap-wave.v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unused_wave_root(root: Path, generation: int) -> Path:
    """Preserve an interrupted generation and select a fresh attempt path."""

    base = root / "waves" / f"generation_{generation:05d}"
    if not base.exists():
        return base
    attempt = 1
    while True:
        candidate = root / "waves" / (
            f"generation_{generation:05d}_attempt_{attempt:03d}"
        )
        if not candidate.exists():
            return candidate
        attempt += 1


def discover_compiler_catalogues(source_root: str | Path) -> list[dict[str, Any]]:
    """Discover nonempty authored-call catalogues in deterministic cheap-first order."""

    root = Path(source_root).resolve()
    records = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            calls = discover_authored_calls(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        if not calls:
            continue
        records.append({
            "source": path.as_posix(),
            "source_sha256": _sha256(path),
            "source_bytes": len(source.encode("utf-8")),
            "authored_call_count": len(calls),
            "attempts": 0,
            "last_outcome_sha256": None,
            "last_outcomes": {},
            "seed_product": None,
            "pending_deep_retry": [],
            "deep_retry_attempted": False,
            "last_deep_retry_registry_sha256": None,
        })
    records.sort(key=lambda record: (
        int(record["source_bytes"]),
        int(record["authored_call_count"]),
        str(record["source"]),
    ))
    return records


def _authored_call_weights(source: str) -> dict[str, tuple[int, int]]:
    """Estimate exact authored-body size without compiling or partitioning it."""

    indexed: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def add_function(node, qualified_name: str) -> None:
        indexed[qualified_name] = node
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(
                    statement,
                    f"{qualified_name}.<locals>.{statement.name}",
                )

    for statement in ast.parse(source).body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(statement, statement.name)
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_function(member, f"{statement.name}.{member.name}")
    lines = source.splitlines(keepends=True)
    return {
        name: (
            sum(len(line.encode("utf-8")) for line in lines[
                int(node.lineno) - 1:int(node.end_lineno)
            ]),
            sum(1 for _child in ast.walk(node)),
        )
        for name, node in indexed.items()
    }


def discover_compiler_work_batches(
    source_root: str | Path, *, batch_size: int,
) -> list[dict[str, Any]]:
    """Build deterministic dependency-first, smallest-ready compiler batches."""

    width = int(batch_size)
    if width < 1:
        raise ValueError("compiler work batch size must be positive")
    per_source = []
    root = Path(source_root).resolve()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            calls = discover_authored_calls(source)
            if not calls:
                continue
            weights = _authored_call_weights(source)
            names = tuple(call.qualified_name for call in calls)
            dependencies = authored_call_dependencies(source, names)
        except (OSError, SyntaxError, UnicodeError):
            continue
        remaining = set(names)
        emitted: set[str] = set()
        ordered = []
        while remaining:
            ready = [
                name for name in remaining
                if set(dependencies.get(name, ())) <= emitted
            ]
            # A recursive SCC has no ready member; the project worker retains
            # the exact authored cycle, so select its smallest stable member.
            candidates = ready or list(remaining)
            selected = min(candidates, key=lambda name: (
                *weights.get(name, (len(source.encode("utf-8")), 0)), name,
            ))
            ordered.append(selected)
            emitted.add(selected)
            remaining.remove(selected)
        batches = [
            tuple(ordered[offset:offset + width])
            for offset in range(0, len(ordered), width)
        ]
        per_source.append((
            min(weights.get(name, (len(source.encode("utf-8")), 0)) for name in names),
            path,
            source,
            weights,
            batches,
        ))
    source_batches = []
    for _minimum, path, source, weights, batches in per_source:
        source_digest = _sha256(path)
        batch_records = []
        for batch_index, entries in enumerate(batches):
            batch_records.append({
                "source": path.as_posix(),
                "source_sha256": source_digest,
                "source_bytes": len(source.encode("utf-8")),
                "entries": list(entries),
                "batch_index": batch_index,
                "authored_call_count": len(entries),
                "estimated_authored_bytes": sum(weights[name][0] for name in entries),
                "estimated_ast_nodes": sum(weights[name][1] for name in entries),
                "attempts": 0,
                "last_outcome_sha256": None,
                "last_outcomes": {},
                "seed_product": None,
                "pending_deep_retry": [],
                "deep_retry_attempted": False,
                "last_deep_retry_registry_sha256": None,
            })
        source_batches.append(batch_records)
    # Each source is a widening chain: its next batch becomes eligible only
    # after the preceding (smaller/dependency-earlier) batch has published.
    # Across those chains, always select the smallest eligible batch.  This
    # is deterministic best-first traversal without sacrificing the join
    # boundary between a leaf batch and the wider work that can consume it.
    records = []
    available = [
        (batch_records, 0)
        for batch_records in source_batches if batch_records
    ]
    while available:
        selected_position = min(range(len(available)), key=lambda position: (
            int(available[position][0][available[position][1]][
                "estimated_authored_bytes"
            ]),
            int(available[position][0][available[position][1]][
                "estimated_ast_nodes"
            ]),
            str(available[position][0][available[position][1]]["source"]),
            int(available[position][0][available[position][1]]["batch_index"]),
        ))
        batch_records, batch_index = available[selected_position]
        records.append(batch_records[batch_index])
        next_index = batch_index + 1
        if next_index == len(batch_records):
            available.pop(selected_position)
        else:
            available[selected_position] = (batch_records, next_index)
    return records


def _normalized_outcome(manifest: dict[str, Any]) -> dict[str, Any]:
    rounds = list(manifest.get("rounds") or ())
    final = dict(rounds[-1]) if rounds else {}
    frontier = []
    for raw in final.get("creep_frontier") or ():
        record = dict(raw)
        frontier.append({
            key: record.get(key)
            for key in (
                "qualified_name", "status", "action", "error_type",
                "control_frontier_action", "unresolved_call_count",
                "unmaterialized_extraction_boundaries",
            )
            if key in record
        })
    native_frontier = [{
        "qualified_name": str(record.get("qualified_name") or ""),
        "status": str(record.get("status") or ""),
        "reason": str(record.get("reason") or ""),
    } for record in final.get("native_verification_frontier") or ()]
    subdivisions = [{
        "qualified_name": str(record.get("qualified_name") or ""),
        "status": str(record.get("status") or ""),
        "verified_product_count": int(
            record.get("verified_product_count") or 0
        ),
        "fixed_point_count": int(record.get("fixed_point_count") or 0),
    } for record in final.get("process_graph_creeps") or ()]
    return {
        "status": str(manifest.get("status") or ""),
        "installed_qualified_names": sorted(map(
            str, manifest.get("installed_qualified_names") or (),
        )),
        "installed_source_regions": sorted(
            tuple(map(str, chain))
            for chain in manifest.get("installed_source_regions") or ()
        ),
        "unit_counts": dict(sorted(
            dict(final.get("unit_counts") or {}).items()
        )),
        "creep_frontier": frontier,
        "native_verification_frontier": native_frontier,
        "process_graph_creeps": subdivisions,
    }


def _outcome_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        _normalized_outcome(manifest),
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _wave_worker(arguments: argparse.Namespace) -> int:
    from src.compiler.compiler_bootstrap_runtime import (
        activate_registered_compiler_bootstraps,
        compiler_bootstrap_registry_path,
    )

    wave_root = arguments.output.resolve()
    wave_root.mkdir(parents=True, exist_ok=True)
    registry = compiler_bootstrap_registry_path()
    registry_before = _sha256(registry) if registry.is_file() else None
    activations = activate_registered_compiler_bootstraps()
    active_products = tuple(dict.fromkeys(
        Path(activation.product).resolve() for activation in activations
    ))
    started = time.perf_counter()
    try:
        manifest = compile_project_bootstrap_creep(
            arguments.source,
            wave_root / "product",
            entries=arguments.entry or None,
            expand_entry_dependencies=False,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * 1024 ** 3)
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * 1024 ** 3
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb == 0
                else int(arguments.max_worker_gb * 1024 ** 3)
            ),
            unit_timeout_seconds=(
                None if arguments.unit_timeout_seconds == 0
                else arguments.unit_timeout_seconds
            ),
            extraction_contract=arguments.extraction_contract,
            bootstrap_products=active_products,
            seed_product=arguments.seed_product,
            crawl_timed_out_units=False,
            max_rounds=1,
            progress=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True,
            ),
        )
        registry_after = _sha256(registry) if registry.is_file() else None
        round_manifest_path = (
            wave_root / "product" / "round_000" / "manifest.json"
        )
        round_manifest = (
            json.loads(round_manifest_path.read_text(encoding="utf-8"))
            if round_manifest_path.is_file() else {}
        )
        timed_out_entries = sorted({
            str(unit.get("qualified_name") or "")
            for unit in round_manifest.get("units") or ()
            if unit.get("error_type") == "ResourceLimitExceeded"
            and "elapsed time" in str(unit.get("error") or "")
            and unit.get("qualified_name")
        })
        result = {
            "schema": WAVE_SCHEMA,
            "status": "complete",
            "generation": int(arguments.generation),
            "process_id": os.getpid(),
            "source": arguments.source.resolve().as_posix(),
            "source_sha256": _sha256(arguments.source.resolve()),
            "entries": list(arguments.entry),
            "workers_joined": True,
            "mode": "deep-retry" if arguments.deep_retry else "bounded",
            "elapsed_seconds": time.perf_counter() - started,
            "registry_before_sha256": registry_before,
            "registry_after_sha256": registry_after,
            "registry_changed": registry_before != registry_after,
            "active_products": [path.as_posix() for path in active_products],
            "product": (wave_root / "product").as_posix(),
            "seed_product": (
                wave_root / "product" / "round_000"
            ).as_posix(),
            "timed_out_entries": timed_out_entries,
            "outcome": _normalized_outcome(manifest),
            "outcome_sha256": _outcome_sha256(manifest),
        }
    except Exception as error:
        result = {
            "schema": WAVE_SCHEMA,
            "status": "failed",
            "generation": int(arguments.generation),
            "process_id": os.getpid(),
            "source": arguments.source.resolve().as_posix(),
            "entries": list(arguments.entry),
            "workers_joined": True,
            "mode": "deep-retry" if arguments.deep_retry else "bounded",
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "registry_before_sha256": registry_before,
        }
    _atomic_json(wave_root / "wave-result.json", result)
    print(json.dumps({
        "stage": "generation_exit",
        **{key: result.get(key) for key in (
            "generation", "process_id", "source", "status",
            "workers_joined", "registry_changed", "outcome_sha256", "mode",
        )},
    }, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 1


def _initial_state(arguments: argparse.Namespace) -> dict[str, Any]:
    sources = discover_compiler_work_batches(
        arguments.source_root, batch_size=arguments.jobs,
    )
    if not sources:
        raise ValueError(
            f"no authored compiler catalogues beneath {arguments.source_root}"
        )
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "source_root": arguments.source_root.resolve().as_posix(),
        "generation": 0,
        "sweep": 0,
        "cursor": 0,
        "batch_size": int(arguments.jobs),
        "sweep_progress": False,
        "sources": sources,
        "waves": [],
    }


def _migrate_legacy_state(
    state: dict[str, Any], arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Expand v1 file-wide work into v2 call batches without losing evidence."""

    legacy_by_source = {
        str(record.get("source") or ""): dict(record)
        for record in state.get("sources") or ()
    }
    sources = discover_compiler_work_batches(
        arguments.source_root, batch_size=arguments.jobs,
    )
    for record in sources:
        legacy = legacy_by_source.get(str(record["source"]))
        if legacy is not None and legacy.get("seed_product"):
            record["seed_product"] = str(legacy["seed_product"])
    return {
        **state,
        "schema": STATE_SCHEMA,
        "status": "running",
        "cursor": 0,
        "batch_size": int(arguments.jobs),
        "sweep_progress": False,
        "sources": sources,
        "migration": {
            "from": LEGACY_STATE_SCHEMA,
            "kind": "deterministic-authored-call-batches",
            "preserved_wave_count": len(state.get("waves") or ()),
        },
    }


def _supervise(arguments: argparse.Namespace) -> int:
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "bootstrap-state.json"
    lock_path = root / "supervisor.lock"
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"another exponential bootstrap owns {lock_path}"
        ) from error
    os.close(descriptor)
    try:
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else _initial_state(arguments)
        )
        if state.get("schema") == LEGACY_STATE_SCHEMA:
            state = _migrate_legacy_state(state, arguments)
            _atomic_json(state_path, state)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported exponential bootstrap state schema")
        while (
            state["status"] == "running"
            and int(state["generation"]) < int(arguments.max_generations)
            and int(state["sweep"]) < int(arguments.max_sweeps)
        ):
            sources = state["sources"]
            cursor = int(state["cursor"])
            source_record = sources[cursor]
            source = Path(str(source_record["source"])).resolve()
            current_source_hash = _sha256(source)
            if current_source_hash != str(source_record["source_sha256"]):
                state["sources"] = discover_compiler_work_batches(
                    arguments.source_root, batch_size=arguments.jobs,
                )
                state["cursor"] = 0
                state["batch_size"] = int(arguments.jobs)
                state["sweep_progress"] = True
                state["catalogue_refresh"] = {
                    "generation": int(state["generation"]),
                    "changed_source": source.as_posix(),
                    "reason": "authored-source-sha256-changed",
                }
                _atomic_json(state_path, state)
                print(json.dumps({
                    "stage": "catalogue_refresh",
                    **state["catalogue_refresh"],
                    "batch_count": len(state["sources"]),
                }, sort_keys=True), flush=True)
                continue
            generation = int(state["generation"])
            wave_root = _unused_wave_root(root, generation)
            command = [
                str(sys.executable), "-m",
                "tools.bootstrap_compiler_exponentially",
                "--wave-worker",
                "--generation", str(generation),
                "--source", str(source),
                "--output", str(wave_root),
                "--jobs", str(arguments.jobs),
                "--worker-reserve-gb", str(arguments.worker_reserve_gb),
                "--max-worker-gb", str(arguments.max_worker_gb),
                "--unit-timeout-seconds", str(arguments.unit_timeout_seconds),
                "--extraction-contract", str(arguments.extraction_contract),
            ]
            if arguments.max_total_gb is not None:
                command.extend(("--max-total-gb", str(arguments.max_total_gb)))
            if source_record.get("seed_product"):
                command.extend((
                    "--seed-product", str(source_record["seed_product"]),
                ))
            pending_deep_retry = tuple(map(
                str, source_record.get("pending_deep_retry") or (),
            ))
            mode = "deep-retry" if pending_deep_retry else "bounded"
            selected_entries = (
                pending_deep_retry if pending_deep_retry else
                tuple(map(str, source_record.get("entries") or ()))
            )
            if pending_deep_retry:
                timeout_index = command.index("--unit-timeout-seconds") + 1
                command[timeout_index] = "0"
                command.append("--deep-retry")
            for qualified_name in selected_entries:
                command.extend(("--entry", qualified_name))
            print(json.dumps({
                "stage": "generation_start",
                "generation": generation,
                "sweep": int(state["sweep"]),
                "source_index": cursor,
                "source_count": len(sources),
                "source": source.as_posix(),
                "entries": list(selected_entries),
                "mode": mode,
            }, sort_keys=True), flush=True)
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            result_path = wave_root / "wave-result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file() else {
                    "schema": WAVE_SCHEMA,
                    "status": "failed",
                    "generation": generation,
                    "source": source.as_posix(),
                    "workers_joined": True,
                    "error_type": "WorkerExit",
                    "error": f"generation exited with {completed.returncode}",
                }
            )
            last_outcomes = dict(source_record.get("last_outcomes") or {})
            previous_outcome = last_outcomes.get(mode)
            current_outcome = result.get("outcome_sha256")
            progressed = bool(
                result.get("registry_changed")
                or (
                    current_outcome is not None
                    and current_outcome != previous_outcome
                )
            )
            state["sweep_progress"] = bool(
                state.get("sweep_progress") or progressed
            )
            source_record["attempts"] = int(source_record.get("attempts") or 0) + 1
            if current_outcome is not None:
                source_record["last_outcome_sha256"] = str(current_outcome)
                last_outcomes[mode] = str(current_outcome)
                source_record["last_outcomes"] = last_outcomes
            if result.get("seed_product"):
                source_record["seed_product"] = str(result["seed_product"])
            scheduled_deep_retry = False
            if mode == "deep-retry":
                source_record["pending_deep_retry"] = []
                source_record["deep_retry_attempted"] = True
                source_record["last_deep_retry_registry_sha256"] = (
                    result.get("registry_before_sha256")
                )
            else:
                timed_out_entries = list(
                    result.get("timed_out_entries") or ()
                )
                registry_revision = result.get("registry_after_sha256")
                if (
                    timed_out_entries
                    and float(result.get("elapsed_seconds") or 0.0)
                    >= float(arguments.minimum_compile_seconds_before_widening)
                    and (
                        not source_record.get("deep_retry_attempted")
                        or registry_revision
                        != source_record.get(
                            "last_deep_retry_registry_sha256"
                        )
                    )
                ):
                    source_record["pending_deep_retry"] = timed_out_entries
                    scheduled_deep_retry = True
            state["waves"].append({
                "generation": generation,
                "source": source.as_posix(),
                "status": str(result.get("status") or "failed"),
                "process_id": result.get("process_id"),
                "workers_joined": bool(result.get("workers_joined")),
                "progressed": progressed,
                "registry_changed": bool(result.get("registry_changed")),
                "outcome_sha256": current_outcome,
                "mode": mode,
                "scheduled_deep_retry": scheduled_deep_retry,
                "result": result_path.as_posix(),
            })
            state["generation"] = generation + 1
            if not scheduled_deep_retry:
                cursor += 1
            if cursor == len(sources):
                if not state.get("sweep_progress"):
                    state["status"] = "fixed-point"
                    state["fixed_point"] = {
                        "kind": "complete-sweep-without-progress",
                        "sweep": int(state["sweep"]),
                    }
                else:
                    state["sweep"] = int(state["sweep"]) + 1
                    state["sweep_progress"] = False
                cursor = 0
            state["cursor"] = cursor
            _atomic_json(state_path, state)
        if state["status"] == "running":
            state["status"] = "frontier"
            state["fixed_point"] = {
                "kind": (
                    "maximum-generations"
                    if int(state["generation"]) >= int(arguments.max_generations)
                    else "maximum-sweeps"
                ),
            }
            _atomic_json(state_path, state)
        print(json.dumps({
            "stage": "supervisor_exit",
            "status": state["status"],
            "generations": int(state["generation"]),
            "sweeps": int(state["sweep"]),
            "state": state_path.as_posix(),
        }, sort_keys=True), flush=True)
        return 0 if state["status"] == "fixed-point" else 1
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("src/compiler"),
        help="compiler package to discover automatically (default: src/compiler)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-total-gb", type=float, default=None)
    parser.add_argument("--worker-reserve-gb", type=float, default=4.0)
    parser.add_argument("--max-worker-gb", type=float, default=4.0)
    parser.add_argument("--unit-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-generations", type=int, default=32768)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument(
        "--minimum-compile-seconds-before-widening",
        type=float,
        default=30.0,
        help=(
            "minimum completed bounded work before scheduling an unlimited-"
            "time parent retry (default: 30 seconds)"
        ),
    )
    parser.add_argument(
        "--extraction-contract", type=Path,
        default=DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    )
    parser.add_argument("--wave-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--generation", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed-product", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--entry", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--deep-retry", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.jobs < 1:
        parser.error("--jobs must be positive")
    if arguments.minimum_compile_seconds_before_widening < 0:
        parser.error(
            "--minimum-compile-seconds-before-widening cannot be negative"
        )
    if arguments.wave_worker:
        if arguments.source is None:
            parser.error("wave worker requires --source")
        return _wave_worker(arguments)
    return _supervise(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
