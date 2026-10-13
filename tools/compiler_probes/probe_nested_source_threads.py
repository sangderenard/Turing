"""Real odd-source stress through parallel submissions and deployment auto.

No synthetic PlanClosures, no worker sleeps/barriers, no alternate compiler.
Observations wrap the existing scheduler and region lowerer for this process
only. Native parity uses the existing crash-isolated graph-program worker.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from contextvars import ContextVar
import dataclasses
import functools
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE_OWNER = ContextVar("nested_source_probe_owner", default="")


def source_case(index: int, islands: int, operations: int) -> tuple[str, str]:
    """Full outer functions plus separately authored, independent islands."""
    name = f"nightmare_{index}"
    lines = []
    for island in range(islands):
        helper = f"island_{index}_{island}"
        lines += [f"def {helper}(value, selector):",
                  f"    left = value + {island + 1}.0",
                  f"    right = value - {island + 2}.0",
                  "    held = left",
                  "    if selector % 2 == 0:"]
        for step in range(operations):
            target = "left" if step % 2 == 0 else "right"
            other = "right" if step % 2 == 0 else "left"
            # Alternating residents, variadic temporary minting and repeated
            # shared inputs, without dependencies on any other island.
            lines.append(f"        {target} = ({target} * 0.999 + {other} * 0.001) + 0.03125")
        lines += ["    else:"]
        for step in range(operations):
            target = "right" if step % 2 == 0 else "left"
            other = "left" if step % 2 == 0 else "right"
            lines.append(f"        {target} = ({target} * 0.998 + {other} * 0.002) - 0.015625")
        lines += ["    parked = right", "    left, right = right, left",
                  "    for turn in range(3):",
                  "        if turn == 1:", "            continue",
                  "        left = left + right * 0.125",
                  "        if left > 1000.0:", "            break",
                  "    if value < -2.0:", "        folded = min(left, right, held)",
                  "    else:", "        folded = max(left, right, parked)",
                  "    return folded + held * 0.25", ""]
    lines += [f"def {name}(x, turns):", "    total = 0.0"]
    for island in range(islands):
        lines += [f"    area_{island} = island_{index}_{island}(x + {island}.0 * 0.125, turns)"]
    # No island consumes another one's result. Only this final reduction joins.
    lines += ["    total = " + " + ".join(f"area_{island}" for island in range(islands)),
              "    if turns > 2:", "        total = total - x * 0.5",
              "    else:", "        total = total + x * 0.5", "    return total", ""]
    return name, "\n".join(lines)


class Observation:
    """Measure real active work; never extend a worker's lifetime for overlap."""
    def __init__(self):
        self.lock = threading.RLock()
        self.sources, self.regions, self.worker_ids = {}, {}, set()
        self.peak_sources = self.peak_regions = self.peak_combined = 0
        self.peak_parallel_sources_with_inner = 0
        self.region_calls = 0
        self.rows = []
        self.started = time.perf_counter()
        self.cpu_start = time.process_time()
        self.peak_rss = 0
        self.peak_cpu_cores = 0.0
        self.stop = threading.Event()

    def _snapshot(self, event, owner, kind):
        owners = {row["owner"] for row in self.regions.values()}
        active_ids = set(self.sources) | set(self.regions)
        self.peak_sources = max(self.peak_sources, len(self.sources))
        self.peak_regions = max(self.peak_regions, len(self.regions))
        self.peak_combined = max(self.peak_combined, len(active_ids))
        self.peak_parallel_sources_with_inner = max(self.peak_parallel_sources_with_inner, len(owners))
        self.rows.append({"time": time.perf_counter() - self.started, "event": event,
            "kind": kind, "owner": owner, "thread_id": threading.get_ident(),
            "sources": len(self.sources), "regions": len(self.regions),
            "combined": len(active_ids), "inner_source_owners": sorted(owners)})

    def wrap(self, function, *, owner, kind):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            label = owner(*args, **kwargs)
            token = SOURCE_OWNER.set(label) if kind == "source" else None
            thread = threading.get_ident()
            table = self.sources if kind == "source" else self.regions
            with self.lock:
                self.worker_ids.add(thread)
                table[thread] = {"owner": label}
                self.region_calls += int(kind == "region")
                self._snapshot("start", label, kind)
            try:
                return function(*args, **kwargs)
            finally:
                with self.lock:
                    self._snapshot("end", label, kind)
                    del table[thread]
                if token is not None:
                    SOURCE_OWNER.reset(token)
        return observed

    def sample(self):
        from src.compiler.shell_telemetry import process_rss_bytes
        previous_wall, previous_cpu = time.perf_counter(), time.process_time()
        while not self.stop.wait(0.5):
            wall, cpu = time.perf_counter(), time.process_time()
            cores = (cpu - previous_cpu) / (wall - previous_wall)
            with self.lock:
                self.peak_rss = max(self.peak_rss, process_rss_bytes())
                self.peak_cpu_cores = max(self.peak_cpu_cores, cores)
            previous_wall, previous_cpu = wall, cpu

    def receipt(self):
        wall = time.perf_counter() - self.started
        return {"wall_seconds": wall, "process_cpu_seconds": time.process_time() - self.cpu_start,
            "average_cpu_cores": (time.process_time() - self.cpu_start) / wall,
            "peak_sampled_cpu_cores": self.peak_cpu_cores,
            "peak_sampled_rss_bytes": self.peak_rss,
            "peak_source_jobs": self.peak_sources, "peak_inner_region_jobs": self.peak_regions,
            "peak_combined_worker_threads": self.peak_combined,
            "peak_sources_simultaneously_inside_inner_work": self.peak_parallel_sources_with_inner,
            "distinct_worker_threads": len(self.worker_ids), "region_calls": self.region_calls}


def verify_native(result, source, entry, directory):
    import numpy as np
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
    from src.compiler.repository_ssa_dispatch import plan_repository_ssa_dispatch
    module, outputs, exports = result
    report = concordance_report(module)
    if not report.splitlines()[0].endswith(", 0 finding(s)"):
        raise AssertionError(report)
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name, entry_name=entry)
    if not artifact.complete:
        raise AssertionError([shortfall.format() for shortfall in artifact.shortfalls])
    deployment = plan_repository_ssa_dispatch(module, backend="llvm").as_manifest()
    from src.compiler.deployment_lowering import deployment_profile
    runtime = {"backend": "llvm",
        "native_pool_launches_emitted": "turing_pool_deploy" in artifact.llvm_ir,
        "deployment_boundary_comments": artifact.llvm_ir.count("deployment boundary"),
        "runtime_thread_execution_verified": False,
        "registered_executor": deployment_profile("llvm").executor,
        "reason": "This receipt verifies native numerical results, not runtime worker overlap. "
                  "LLVM module text alone does not describe the registered host-pool executor.",
        "deployment_plan": deployment}
    artifact = compile_artifact(artifact, directory=directory / "native", optimization="O0")
    artifact = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
        _validation_error_reset=None, _validation_error_take=None)
    bindings = dict(root.metadata["parameter_names"])
    namespace = {}
    exec(compile(source, entry, "exec"), namespace)
    checks = []
    for case, (x, turns) in enumerate(((-3.5, 0), (0.0, 1), (2.0, 3))):
        expected = namespace[entry](x, turns)
        payload = directory / f"native-case-{case}.pkl"
        with payload.open("wb") as stream:
            pickle.dump((artifact, {}, ("program_outputs",), {"inputs": {
                int(bindings["x"]): np.array(x, dtype=np.float64),
                int(bindings["turns"]): np.array(turns, dtype=np.int64)},
                "outputs": tuple(int(value.id) for value in outputs[root.name])}), stream)
        child = subprocess.run([sys.executable, "-u", str(ROOT / "tools/compiler_probes/graph_kernel_native_worker.py"),
            str(payload)], text=True, capture_output=True)
        if child.returncode:
            raise RuntimeError(child.stdout + child.stderr)
        actual = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
            if line.startswith("GRAPH_PROGRAM_OUTPUTS=")))[0]
        # Explicit O0 arithmetic, no reassociation or numerical tolerance.
        if actual != expected:
            raise AssertionError({"entry": entry, "x": x, "turns": turns, "actual": actual, "expected": expected})
        checks.append({"x": x, "turns": turns, "actual": actual, "expected": expected})
    return {"entry": entry, "native_checks": checks, "audit": report,
            "native_runtime": runtime,
            "source_compilation": module.metadata["source_compilation"],
            "profile": module.metadata["compile_profile"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-schedule", choices=("parallel", "serial"), default="parallel",
                        help="Supplied-source scheduling; each source still uses inner auto.")
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--inner-jobs", type=int, default=5)
    parser.add_argument("--islands", type=int, default=8)
    parser.add_argument("--operations", type=int, default=24)
    parser.add_argument("--budget-gb", type=float, default=2.0)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if min(args.sources, args.inner_jobs, args.islands, args.operations, args.budget_gb) <= 0:
        parser.error("all sizes and memory budget must be positive")
    os.chdir(ROOT)
    from src.compiler import graph_work, hierarchical_plan
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import SourceListCompilationError
    from src.compiler.shell_telemetry import compile_progress, compile_rule_batch, system_available_memory_bytes
    from src.compiler.work_contract import active_contract, scoped_work_contract
    budget = int(args.budget_gb * 2**30)
    available = system_available_memory_bytes()
    if available is not None and available < budget + (2 << 30):
        raise RuntimeError("insufficient available RAM for the chosen bounded stress run")
    root = Path(tempfile.mkdtemp(prefix="nested_thread_nightmare_", dir=ROOT / "build"))
    print("STRESS_DIRECTORY=" + str(root), flush=True)
    cases = [source_case(index, args.islands, args.operations) for index in range(args.sources)]
    values = []
    for parameter, dtype, python_type in (("x", "float64", "float"), ("value", "float64", "float"),
                                           ("turns", "int64", "int"), ("selector", "int64", "int")):
        values.append({"function": "*", "parameter": parameter, "storage": "scalar", "dtype": dtype,
                       "rank": 0, "shape": [], "python_type": "builtins." + python_type})
    contract = ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml").with_program_abi(
        {"records": {}, "bindings": [], "values": values})
    requests = []
    for name, source in cases:
        path = root / f"{name}.py"
        # Generated source is an artifact of this reproducible fixture.
        path.write_text(source, encoding="utf-8")
        requests.append({"label": name, "source_path": path, "entrypoint": name,
                         "options": {"extraction_contract": contract, "name": name}})
    observation = Observation()
    original_dispatch = graph_work.bounded_compiler_work
    def dispatch(jobs, work, publish, **kwargs):
        if kwargs.get("label") == "source submissions":
            work = observation.wrap(work, owner=lambda index: cases[index][0], kind="source")
        return original_dispatch(jobs, work, publish, **kwargs)
    original_region = hierarchical_plan.plan_region_to_ssa_instrs
    tracked_region = observation.wrap(original_region, owner=lambda *a, **k: SOURCE_OWNER.get(), kind="region")
    result_records, failures = [], []
    compile_measurement = None
    sampler = threading.Thread(target=observation.sample, name="stress-measurement", daemon=True)
    sampler.start()
    try:
        with ExitStack() as scope:
            scope.enter_context(scoped_work_contract(dataclasses.replace(active_contract(),
                memory_budget_bytes=budget, memory_elastic_available_percent=0)))
            display = scope.enter_context(compile_progress(not args.headless, collect_profile=True,
                renderer="rich", prefix="[compiler] "))
            scope.enter_context(compile_rule_batch(len(cases), names=[name for name, _ in cases]))
            graph_work.bounded_compiler_work = dispatch
            hierarchical_plan.plan_region_to_ssa_instrs = tracked_region
            scope.callback(setattr, graph_work, "bounded_compiler_work", original_dispatch)
            scope.callback(setattr, hierarchical_plan, "plan_region_to_ssa_instrs", original_region)
            try:
                results = lower_ast_source_to_ssa(requests, source_compilation=args.source_schedule, partitioned_options={
                    "jobs": args.sources, "inner_jobs": args.inner_jobs,
                    "max_total_resident_bytes": budget, "max_worker_memory_bytes": budget,
                    "system_reserve_bytes": 2 << 30,
                    "worker_resident_reservation_bytes": 64 << 20, "thread_working_bytes": 8 << 20})
            except SourceListCompilationError as error:
                failures.extend(error.failures)
                results = error.results
            # CPU/wall/RSS below concern this process's compilation, not
            # time waiting for crash-isolated native-verification children.
            observation.stop.set()
            sampler.join()
            compile_measurement = observation.receipt()
            for (entry, source), result in zip(cases, results):
                if result is None:
                    continue
                try:
                    if args.skip_native:
                        record = {"entry": entry, "source_compilation": result[0].metadata["source_compilation"],
                                  "profile": result[0].metadata["compile_profile"]}
                    else:
                        display.substage("native parity " + entry)
                        directory = root / entry
                        directory.mkdir(exist_ok=True)
                        record = verify_native(result, source, entry, directory)
                except Exception as error:
                    import traceback
                    failures.append({"label": entry, "type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc()})
                    display.log(failures[-1]["traceback"])
                    continue
                result_records.append(record)
                from src.compiler.shell_telemetry import active_rule_batch_progress
                active_rule_batch_progress().complete(entry, sum(row["seconds"] for row in record["profile"]["stages"]))
    except Exception as error:
        import traceback
        failures.append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        print(failures[-1]["traceback"], flush=True)
    finally:
        observation.stop.set()
        sampler.join()
        receipt = {"schema": "turing.nested-source-thread-stress.v1", "settings": vars(args),
            "logical_cpus": os.cpu_count(), "python": sys.version,
            "initial_available_bytes": available, "measurement": compile_measurement or observation.receipt(),
            "results": result_records, "failures": failures, "events": observation.rows}
        with (root / "receipt.json").open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
        print("STRESS_MEASUREMENT=" + json.dumps(receipt["measurement"]), flush=True)
        print("STRESS_THREAD_TARGET=" + json.dumps({
            "potential_compiler_worker_threads": (args.sources if args.source_schedule == "parallel" else 1) * (1 + args.inner_jobs),
            "achieved_peak_worker_threads": receipt["measurement"]["peak_combined_worker_threads"],
            "cpu_saturation_is_separate_from_worker_overlap": True}), flush=True)
        print(f"STRESS_RESULT completed={len(result_records)}/{len(cases)} failures={len(failures)} receipt={root / 'receipt.json'}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
