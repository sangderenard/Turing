"""Time the existing untraced N4 profiler with the installed native analysis.

Uses the preserved serial source route, four graph workers/eight resident
regions and a fresh TEMP/TMP output directory. No deadline, source edits,
stale-asset acceptance or alternative compiler. Does not execute orbital physics.
The existing full-main self-lowering may run concurrently; report that timing
confound, never terminate it to make this benchmark look isolated.

--graph-workers 1 explicitly tests the preserved serial graph lane instead
of the native graph-foundation bootstrap. Report this changed configuration.

--native-copier requires the installed whole original copier and counts native
batches separately from pre-execution subclass fallbacks. No build on this run.
--fresh-pieces gives this process its own empty law-cache root, so a concurrent
comparison cannot reuse or overwrite pieces being built by the other run.
"""
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("TURING_PROBE_EXTERNAL_ROOT", str(ROOT)))
for entry in (EXTERNAL_ROOT.parent / "engine_toy", EXTERNAL_ROOT / "examples", ROOT):
    sys.path.insert(0, str(entry))


def main():
    import json
    import os
    import tempfile
    import time

    workers = int(sys.argv[sys.argv.index("--graph-workers") + 1]) if "--graph-workers" in sys.argv else 4
    os.environ["TURING_GRAPH_WORKERS"] = str(workers)
    os.environ["TURING_GRAPH_RESIDENT_REGIONS"] = "8"
    started = time.perf_counter()
    directory = Path(tempfile.mkdtemp(prefix="profile_memory_n4_native_storage_", dir=ROOT / "build"))
    os.environ["TEMP"] = os.environ["TMP"] = str(directory)
    tempfile.tempdir = None
    print("N4_SUBMISSION=" + json.dumps({
        "laws": 4, "trace": False, "deadline_seconds": None,
        "source_compilation": "serial", "graph_workers": workers,
        "resident_regions": 8, "directory": str(directory),
        "concurrent_full_main_job": None, "target_execution": False,
        "copier": "native" if '--native-copier' in sys.argv else "original_python",
        "fresh_pieces": '--fresh-pieces' in sys.argv,
    }), flush=True)

    from src.compiler.native_compiler_accelerators import load_storage_effects_accelerator
    provider = load_storage_effects_accelerator()
    if provider is None:
        raise RuntimeError("no matching installed native storage-effects unit; not benchmarking a silent fallback")
    print("N4_NATIVE_ASSET=" + str(provider.library_path), flush=True)
    # Observe the existing public entry in this diagnostic process only.
    # Stale law builds otherwise use its extremely verbose default callback.
    # Preserve their lowering; stream real stage progress without per-AST-node
    # prints (no output redirection, log file, tracing or hidden compiler).
    from src.compiler import fortran_c_shell
    print("N4_COMPILER_SOURCE=" + fortran_c_shell.__file__, flush=True)
    public_lowerer = fortran_c_shell.lower_ast_source_to_ssa
    last_default_progress = [0.0]

    def default_progress(message):
        now = time.perf_counter()
        if now - last_default_progress[0] >= 20 or str(message).startswith("ssa-source:"):
            print(f"[N4 +{now - started:.1f}s] {message}", file=sys.__stdout__, flush=True)
            last_default_progress[0] = now

    def observed_public_lowerer(*args, **kwargs):
        if kwargs.get("progress") is None:
            kwargs["progress"] = default_progress
        return public_lowerer(*args, **kwargs)

    fortran_c_shell.lower_ast_source_to_ssa = observed_public_lowerer
    native_run = provider.run
    native = {"calls": 0, "value_facts": 0, "seconds_including_transfers": 0.0}

    def counted(*args, **kwargs):
        before = time.perf_counter()
        result = native_run(*args, **kwargs)
        native["seconds_including_transfers"] += time.perf_counter() - before
        native["calls"] += 1
        native["value_facts"] += len(result[0] if isinstance(result, tuple) else result)
        if native["calls"] == 1:
            print("N4_NATIVE_ANALYSIS_FIRST_CALL=" + json.dumps(native),
                  file=sys.__stdout__, flush=True)
        return result

    provider.run = counted
    copier_stats = None
    if '--native-copier' in sys.argv:
        from src.compiler.native_compiler_accelerators import load_metadata_copy_accelerator
        copier_provider = load_metadata_copy_accelerator()
        if copier_provider is None:
            raise RuntimeError('no matching installed whole metadata copier; refusing silent serial benchmark')
        print('N4_COPIER_ASSET=' + str(copier_provider.library_path), flush=True)
        copier_run = copier_provider.run
        copier_stats = {'batches': 0, 'native_batches': 0, 'fallback_batches': 0,
                        'fallback_roots': 0, 'roots': 0, 'largest_batch_roots': 0,
                        'seconds_including_boundaries': 0.0}
        def counted_copier(values, *, copier):
            before = time.perf_counter()
            values = tuple(values)
            previous_fallbacks = copier_stats['fallback_roots']
            def observed_fallback(value):
                copier_stats['fallback_roots'] += 1
                return copier(value)
            result = copier_run(values, copier=observed_fallback)
            copier_stats['seconds_including_boundaries'] += time.perf_counter() - before
            copier_stats['batches'] += 1
            copier_stats['roots'] += len(values)
            copier_stats['largest_batch_roots'] = max(copier_stats['largest_batch_roots'], len(values))
            copier_stats['fallback_batches' if copier_stats['fallback_roots'] != previous_fallbacks
                         else 'native_batches'] += 1
            if copier_stats['batches'] == 1:
                print('N4_COPIER_FIRST_BATCH=' + json.dumps(copier_stats), file=sys.__stdout__, flush=True)
            return result
        copier_provider.run = counted_copier
    else:
        from src.compiler.native_compiler_accelerators import METADATA_COPY, compiler_accelerators
        if compiler_accelerators.resolve(METADATA_COPY).tier != 'python':
            raise RuntimeError('original-copier benchmark unexpectedly has a native provider')
        print('N4_COPIER_MODE=original_python; current compiler fixes retained', flush=True)
    import llvm_dt_system as system
    from tools.compiler_probes import profile_memory_n2 as probe
    if '--fresh-pieces' in sys.argv:
        import honorary_engine_equation_catalogue as honorary
        honorary._LLVM_LAW_CACHE_DIR = directory / 'law_cache'
        print('N4_ISOLATED_LAW_CACHE=' + str(honorary._LLVM_LAW_CACHE_DIR), flush=True)
    original_lowered_system = system.lowered_system
    phases = {}

    def observed_lowered_system(*args, **kwargs):
        kwargs["source_compilation"] = "serial"
        original_progress = kwargs.get("progress")
        last_print = [0.0]
        before = time.perf_counter()
        phases["combined_begin_seconds"] = before - started

        def progress(message):
            if original_progress is not None:
                original_progress(message)
            now = time.perf_counter()
            if "complete; functions=" in str(message):
                phases["combined_ssa_seconds"] = now - before
            if now - last_print[0] >= 20 or str(message).startswith("ssa-source:"):
                print(f"[N4 +{now - started:.1f}s] {message}", file=sys.__stdout__, flush=True)
                last_print[0] = now

        kwargs["progress"] = progress
        try:
            return original_lowered_system(*args, **kwargs)
        finally:
            phases["combined_seconds"] = time.perf_counter() - before

    system.lowered_system = observed_lowered_system
    try:
        result = probe.main([str(Path(probe.__file__)), "4"])
    except BaseException as error:
        print("N4_FAILED=" + json.dumps({
            "seconds": time.perf_counter() - started, "type": type(error).__name__,
            "message": str(error), "native_analysis": native, "native_copier": copier_stats, "phases": phases,
        }), flush=True)
        raise
    print("N4_TIMING=" + json.dumps({
        "seconds": time.perf_counter() - started, "exit_code": result,
        "native_analysis": native, "native_copier": copier_stats, "phases": phases, "directory": str(directory),
        "concurrent_full_main_job": None,
    }), flush=True)
    return result


if __name__ == "__main__":
    if "--child" in sys.argv:
        raise SystemExit(main())
    result = subprocess.run([sys.executable, "-u", __file__, "--child", *sys.argv[1:]])
    raise SystemExit(result.returncode)
