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
import hashlib
import json
from pathlib import Path
import time
import traceback


def build_compartment(directory, *, lower_only=False, grow_stage=True, jobs=None,
                      max_stage_units=None, schedule_options=None, progress=print):
    worker_jobs = 12 if jobs is None else int(jobs)
    if worker_jobs < 1:
        raise ValueError('provenance worker count must be positive')
    stage_units = worker_jobs if max_stage_units is None else max_stage_units
    scheduling = dict(schedule_options or {})
    allowed = {'max_total_resident_bytes', 'minimum_total_resident_bytes',
               'system_reserve_bytes', 'adaptive_memory_ceiling',
               'worker_resident_reservation_bytes', 'max_worker_memory_bytes',
               'worker_elastic_available_percent'}
    if scheduling.keys() - allowed:
        raise ValueError('unsupported provenance scheduling options')
    scheduling = {'adaptive_memory_ceiling': True, 'max_total_resident_bytes': 10 << 30,
                  **scheduling}
    import src.compiler.identity_concordance as incumbent
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _dump_pickle_artifact, _dump_resolved_process_graph, compiler_toolchain_fingerprint,
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
    policy = with_expansion_policy(active_contract(), {
        'unroll_depth': 0, 'inline_depth': 0, 'custom_version_depth': 0,
    })
    contract = ExtractionContract(root / 'extraction_contracts/program_extraction.yaml').with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    receipt = {
        'schema': 'turing.compiler-concordance-compartment.v1',
        'source': source_path.as_posix(),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
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

    def publish():
        receipt['elapsed_seconds'] = time.perf_counter() - started
        path = destination / 'compartment.json'
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    def plan_sink(value):
        (destination / 'process-graph-units.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

    def graph_sink(value):
        # Existing exact post-reduction checkpoint, before deployment's cost.
        with (destination / 'resolved-process-graph.pkl').open('wb') as stream:
            _dump_resolved_process_graph(value, stream)

    def report(message):
        # Keep stage progress visible without per-AST-node terminal rendering.
        if (str(message).startswith('[ast-parent]')
                or str(message).startswith('[graph-build')
                or 'upward search' in str(message) or 'downward descent' in str(message)):
            return
        if progress is not None:
            progress(message)

    def growth_report(event):
        report(json.dumps(event, sort_keys=True))

    publish()
    try:
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
                        python_bindings=vars(incumbent), ssa_composition=True, plan_only=True,
                        source_options={'source_module': incumbent.__name__,
                                        'python_package': incumbent.__package__,
                                        'runtime_closure_only': False,
                                        'identity_log_level': 'summary'},
                        progress=growth_report,
                    )
                    plan = planned['plan']
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
                    **scheduling,
                    progress=growth_report,
                )
                receipt['native_growth'] = product
                if product['pending_unit_indices']:
                    receipt['stage'] = 'native-growth-stage-persisted'
                    publish()
                    return receipt
                module, outputs, exports = load_composed_source_units(product)
            else:
                module, outputs, exports = lower_ast_source_to_ssa(
                    source, None, source_module=incumbent.__name__,
                    python_package=incumbent.__package__, python_bindings=vars(incumbent),
                    name='compiler_identity_concordance', runtime_closure_only=False,
                    extraction_contract=contract, progress=report, identity_log_level='summary',
                    compilation_unit_plan_sink=plan_sink, resolved_process_graph_sink=graph_sink,
                )
            receipt.update(stage='repository-ssa', functions=len(module.functions), exports=list(exports))
            _dump_pickle_artifact(destination / 'repository-ssa.pkl', (module, outputs, exports))
            publish()
            if lower_only:
                return receipt
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
    parser.add_argument('--schedule-json', default='{}', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.grow_stage and args.whole_module:
        parser.error('--grow-stage and --whole-module are mutually exclusive')
    receipt = build_compartment(args.output, lower_only=args.lower_only, grow_stage=not args.whole_module,
                                jobs=args.jobs, max_stage_units=args.stage_units,
                                schedule_options=json.loads(args.schedule_json), progress=lambda message: print(message, flush=True))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
