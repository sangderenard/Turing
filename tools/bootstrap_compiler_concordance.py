"""Grow the incumbent book/concordance compartment through SSA composition.

The ordinary source compiler resolves the call graph once, then dependency
units retain SSA bodies for composition into progressively larger enclosures.
Each invocation persists one stage; the bootstrap coordinator restarts before
the next. The final library still owns the complete module surface.
A produced library is not installed until its persistent-state operator ABI
has passed semantic verification. Failed/partial compilation is not native
book activation.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


_REPOSITORY_SSA_PHASE = 'provenance-repository-ssa'


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_checkpoint(store, phase, implementation, directory, artifact, loader):
    """Load a checkpoint only while its separately published artifact agrees."""
    descriptor = store.load(phase, implementation)
    if not isinstance(descriptor, dict):
        return None
    if descriptor.get('artifact') != artifact:
        store.last_load_status = 'miss: external artifact name mismatch'
        return None
    path = Path(directory) / artifact
    try:
        actual_sha256 = _file_sha256(path)
    except OSError as error:
        store.last_load_status = f'miss: {type(error).__name__}: {error}'
        return None
    if actual_sha256 != descriptor.get('sha256'):
        store.last_load_status = 'miss: external artifact digest mismatch'
        return None
    try:
        return loader(path)
    except (
        OSError, ValueError, TypeError, EOFError, ImportError, AttributeError,
        pickle.UnpicklingError,
    ) as error:
        store.last_load_status = f'miss: {type(error).__name__}: {error}'
        return None


def build_compartment(directory, *, lower_only=False, grow_stage=True, jobs=None,
                      max_stage_units=None, schedule_options=None,
                      stop_after_failure=True, progress=print):
    worker_jobs = 12 if jobs is None else int(jobs)
    if worker_jobs < 1:
        raise ValueError('provenance worker count must be positive')
    # Fail-fast stages remain intentionally small: once a worker reports a
    # blocker, only the already-admitted peers drain.  A collecting run has a
    # different policy.  With no explicit stage limit, admit the whole plan
    # while `jobs` continues to bound concurrency, so one invocation reports
    # every independent failure instead of five at a time.
    stage_units = (
        worker_jobs
        if max_stage_units is None and stop_after_failure
        else max_stage_units
    )
    scheduling = dict(schedule_options or {})
    allowed = {'max_total_resident_bytes', 'minimum_total_resident_bytes',
               'system_reserve_bytes', 'adaptive_memory_ceiling',
               'worker_resident_reservation_bytes', 'max_worker_memory_bytes',
               'worker_elastic_available_percent'}
    if scheduling.keys() - allowed:
        raise ValueError('unsupported provenance scheduling options')
    scheduling = {'adaptive_memory_ceiling': True, 'max_total_resident_bytes': 10 << 30,
                  **scheduling}
    from src.common.tensors.accelerator_backends.aot_checkpoint import (
        AOTCheckpointStore, callable_digest,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _atomic_json, _dump_pickle_artifact, _dump_resolved_process_graph,
        _load_pickle_artifact, compiler_toolchain_fingerprint,
        authored_compilation_growth_plan,
    )
    from src.compiler.work_contract import (
        active_contract, expansion_policy, scoped_work_contract, with_expansion_policy,
    )
    root = Path(__file__).resolve().parents[1]
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_path = root / 'src/compiler/identity_concordance.py'
    source = source_path.read_text(encoding='utf-8')
    source_module = 'src.compiler.identity_concordance'
    python_package = 'src.compiler'
    policy = with_expansion_policy(active_contract(), {
        'unroll_depth': 0, 'inline_depth': 0, 'custom_version_depth': 0,
    })
    contract = (
        ExtractionContract(
            root / 'extraction_contracts/program_extraction.yaml'
        )
        .with_execution_file(
            root / 'extraction_contracts/vehicle_full_native_execution.yaml'
        )
        .with_sources(((
            'src.compiler.concordance_declarations',
            root / 'src/compiler/concordance_declarations.py',
        ),))
    )
    receipt = {
        'schema': 'turing.compiler-concordance-compartment.v1',
        'source': source_path.as_posix(),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'python_cache_tag': sys.implementation.cache_tag,
        'compiler_toolchain': compiler_toolchain_fingerprint(),
        'scope': 'whole-module', 'entrypoint': None,
        'required_native_work': [
            'post-concord-revise', 'row-and-scope-search', 'history-search',
            'forward-and-reverse-provenance', 'correlation-and-audit-search',
            'complete-log-formatting', 'compressed-log-writing',
        ],
        'expansion_policy': expansion_policy(policy),
        'execution_contract': contract.execution.receipt(),
        'stage': 'source', 'installed': False,
    }
    started = time.perf_counter()

    checkpoint_input = {
        key: value for key, value in receipt.items()
        if key not in {'stage', 'installed'}
    }
    checkpoint_store = AOTCheckpointStore(checkpoint_input, destination)
    checkpoint_implementation = callable_digest(
        build_compartment, _load_external_checkpoint, _file_sha256,
    )
    receipt['checkpoint_identity'] = checkpoint_store.identity

    def publish():
        receipt['elapsed_seconds'] = time.perf_counter() - started
        _atomic_json(destination / 'compartment.json', receipt)

    def plan_sink(value):
        _atomic_json(destination / 'process-graph-units.json', value)

    def graph_sink(value):
        # Existing exact post-reduction checkpoint, before deployment's cost.
        path = destination / 'resolved-process-graph.pkl'
        temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        try:
            with temporary.open('wb') as stream:
                _dump_resolved_process_graph(value, stream)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        receipt['saved_resolved_process_graph_sha256'] = _file_sha256(path)
        publish()

    def report(message):
        # Keep stage progress visible without per-AST-node terminal rendering.
        if (str(message).startswith('[ast-parent]')
                or str(message).startswith('[graph-build')
                or 'upward search' in str(message) or 'downward descent' in str(message)):
            return
        if progress is not None:
            progress(message)

    def growth_report(event):
        if progress is not None:
            progress(event)

    repository_artifact = 'repository-ssa.pkl'
    resumed = _load_external_checkpoint(
        checkpoint_store, _REPOSITORY_SSA_PHASE, checkpoint_implementation,
        destination, repository_artifact, _load_pickle_artifact,
    )
    if resumed is not None:
        try:
            module, outputs, exports = resumed
            receipt.update(
                stage='repository-ssa', functions=len(module.functions),
                exports=list(exports), resumed_from=_REPOSITORY_SSA_PHASE,
                checkpoint_status=checkpoint_store.last_load_status,
                resumable_stage='repository-ssa',
            )
        except (TypeError, ValueError, AttributeError):
            resumed = None
            receipt['checkpoint_status'] = 'miss: invalid repository SSA payload'
    else:
        receipt['checkpoint_status'] = checkpoint_store.last_load_status
    receipt['checkpoint_resume_status'] = receipt['checkpoint_status']
    publish()
    try:
        if resumed is None:
            with scoped_work_contract(policy):
                if grow_stage:
                    from src.compiler.project_compilation_product import (
                        compile_partitioned_source, compile_resolved_process_graph_plan,
                        load_composed_source_units, changed_compiler_toolchain_files,
                    )
                    growth_root = destination / 'growth'
                    growth_root.mkdir(parents=True, exist_ok=True)
                    triage_path = growth_root / 'source-complexity.json'
                    if not triage_path.exists():
                        triage = authored_compilation_growth_plan(source)
                        triage_path.write_text(json.dumps(triage, indent=2, sort_keys=True) + '\n')
                        growth_report({'stage': 'provenance_complexity_triage',
                                       'receipt': str(triage_path), 'unit_count': len(triage['units'])})
                    receipt['source_complexity'] = str(triage_path)
                    plan_path = growth_root / 'inputs/plan.json'
                    if not plan_path.exists():
                        planned = compile_partitioned_source(
                            source, growth_root, entrypoint=None, extraction_contract=contract,
                            ssa_composition=True, plan_only=True,
                            source_options={'source_module': source_module,
                                            'python_package': python_package,
                                            'runtime_closure_only': False,
                                            'identity_log_level': 'summary'},
                            progress=growth_report,
                        )
                    else:
                        plan = json.loads(plan_path.read_text(encoding='utf-8'))
                        if changed_compiler_toolchain_files(plan['compiler_toolchain']):
                            raise ValueError('provenance growth plan is stale; use a fresh output directory')
                    # A completed interior supplies SSA to its larger enclosure.
                    # No Python/native installation occurs until the complete
                    # bootstrap module has been emitted and verified.
                    product = compile_resolved_process_graph_plan(
                        growth_root / 'resolved-process-graph.pkl', plan_path,
                        growth_root / 'product', jobs=worker_jobs, max_stage_units=stage_units,
                        stop_after_failure=bool(stop_after_failure),
                        **scheduling,
                        progress=growth_report,
                    )
                    receipt['native_growth'] = product
                    if product.get('stopped_after_failure'):
                        receipt['stage'] = 'native-growth-failed'
                        publish()
                        return receipt
                    if product['pending_unit_indices']:
                        receipt['stage'] = 'native-growth-stage-persisted'
                        publish()
                        return receipt
                    module, outputs, exports = load_composed_source_units(product)
                else:
                    module, outputs, exports = lower_ast_source_to_ssa(
                        source, None, source_module=source_module,
                        python_package=python_package,
                        name='compiler_identity_concordance', runtime_closure_only=False,
                        extraction_contract=contract, progress=report, identity_log_level='summary',
                        compilation_unit_plan_sink=plan_sink, resolved_process_graph_sink=graph_sink,
                    )
            artifact_path = destination / repository_artifact
            _dump_pickle_artifact(artifact_path, (module, outputs, exports))
            checkpoint_store.store(_REPOSITORY_SSA_PHASE, checkpoint_implementation, {
                'artifact': repository_artifact,
                'sha256': _file_sha256(artifact_path),
            })
            receipt.update(
                stage='repository-ssa', functions=len(module.functions),
                exports=list(exports), checkpoint_status='stored',
                resumable_stage='repository-ssa',
            )
            publish()
        if lower_only:
            return receipt
        with scoped_work_contract(policy):
            # This existing emitter exports an entire library surface as bind(C)
            # roots in ONE module; no per-operator library/dispatcher is created.
            from src.compiler.ssa_fortran_backend import emit_module, compile_module
            emitted = emit_module(module, name='compiler_identity_concordance',
                                  outputs=outputs, extra_roots=exports, progress=report)
            emitted.write(destination)
            if not emitted.complete:
                receipt['shortfalls'] = [repr(item) for item in emitted.shortfalls]
                raise ValueError('whole concordance compartment has native emission shortfalls')
            library = compile_module(emitted, directory=destination)
            receipt.update(stage='native-built-unverified', library=library.as_posix())
            publish()
            return receipt
    except Exception as error:
        receipt.update(stage='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        publish()
        raise


def _one_line(value, *, limit=320):
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _stage_failure_rows(receipt, directory):
    growth = receipt.get('native_growth') or {}
    admitted = (
        growth.get('blocking_failure_unit_indices')
        if growth.get('stopped_after_failure')
        else growth.get('stage_admitted_unit_indices')
    )
    admitted = None if admitted is None else {int(index) for index in admitted}
    root = Path(directory).resolve() / 'growth' / 'product' / 'units'
    rows = []
    for record in growth.get('units') or ():
        index = int(record.get('unit_index', -1))
        if admitted is not None and index not in admitted:
            continue
        if record.get('status') not in {'failed', 'partial'}:
            continue
        unit = record.get('unit') or {}
        names = record.get('qualified_names') or unit.get('qualified_names') or ()
        stage = record.get('stage') or {}
        compiler = stage.get('compiler') or {}
        phase = (stage.get('phase') or compiler.get('stage_label')
                 or compiler.get('stage_key') or 'worker')
        unit_root = root / f'unit_{index:03d}'
        detail = next((path for path in (
            unit_root / 'failure.json', unit_root / 'unit.json',
        ) if path.is_file()), unit_root / 'failure.json')
        rows.append({
            'unit_index': index,
            'qualified_name': ', '.join(map(str, names)) or '<unnamed>',
            'phase': str(phase),
            'error_type': str(record.get('error_type') or record.get('status')),
            'reason': _one_line(record.get('error') or record.get('message')
                                or record.get('status')),
            'receipt': detail,
        })
    return sorted(rows, key=lambda row: row['unit_index'])


def _print_outcome(receipt, directory, *, emit=print):
    path = Path(directory).resolve() / 'compartment.json'
    growth = receipt.get('native_growth') or {}
    if not growth:
        emit(f"provenance stage {receipt.get('stage', 'unknown')}; receipt: {path}")
        return
    counts = growth.get('counts') or {}
    completed = len(growth.get('units') or ())
    pending = len(growth.get('pending_unit_indices') or ())
    total = completed + pending
    count_text = ', '.join(
        f"{name}={int(counts.get(name, 0))}"
        for name in ('verified', 'compiled-unverified', 'partial', 'failed')
    )
    emit(
        f"provenance stage complete: admitted={growth.get('stage_admitted_units', 0)}, "
        f"peak-workers={growth.get('concurrent_workers_peak', 0)}, "
        f"cumulative={completed}/{total}, pending={pending}; {count_text}"
    )
    failures = _stage_failure_rows(receipt, directory)
    if failures:
        label = ('blocking failures' if growth.get('stopped_after_failure')
                 else 'failures from this stage')
        emit(f"{label} ({len(failures)}):")
        for row in failures:
            emit(
                f"  - unit {row['unit_index']:03d} | {row['qualified_name']} | "
                f"{row['phase']} | {row['error_type']}: {row['reason']}"
            )
            emit(f"    receipt: {row['receipt']}")
    emit(f"stage receipt: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/compiler-concordance'))
    parser.add_argument('--lower-only', action='store_true', help='persist complete repository SSA without native emission')
    parser.add_argument('--grow-stage', action='store_true',
                        help='compile a worker stage of complexity-ranked interiors and resume its frozen plan')
    parser.add_argument('--whole-module', action='store_true',
                        help='use the original indivisible whole-module lowering')
    parser.add_argument('--jobs', type=int, default=12, help='maximum concurrent isolated workers (default: 12)')
    parser.add_argument('--stage-units', type=int, help='units admitted before draining and restarting (default: jobs)')
    parser.add_argument(
        '--continue-after-failure', action='store_true',
        help='admit the complete scheduled stage and report every failed/partial unit',
    )
    parser.add_argument('--schedule-json', default='{}', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.grow_stage and args.whole_module:
        parser.error('--grow-stage and --whole-module are mutually exclusive')
    from src.compiler.shell_telemetry import compile_progress
    display = compile_progress(None, prefix='provenance-core: ')

    def present(message):
        if isinstance(message, Mapping):
            if message.get('stage') == 'resolved_worker_log':
                # Worker stdout contains complete JSON receipts and tracebacks.
                # They remain in worker.log/failure.json; snapshots drive the
                # live lanes and the compact rollup reports failures afterward.
                return
            display.compilation_event(message)
        else:
            display.log(str(message))

    with display:
        receipt = build_compartment(
            args.output, lower_only=args.lower_only,
            grow_stage=not args.whole_module, jobs=args.jobs,
            max_stage_units=args.stage_units,
            stop_after_failure=not args.continue_after_failure,
            schedule_options=json.loads(args.schedule_json), progress=present,
        )
    _print_outcome(receipt, args.output, emit=lambda message: print(message, flush=True))
    return 1 if (receipt.get('native_growth') or {}).get('stopped_after_failure') else 0


if __name__ == '__main__':
    raise SystemExit(main())
