"""Compile the complete incumbent book/concordance module as one compartment.

This uses the ordinary whole-source compiler election, never a list of tiny
accessor accelerators. Calls among book operators remain in the same module.
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


def build_compartment(directory, *, lower_only=False, progress=print):
    import src.compiler.identity_concordance as incumbent
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _dump_pickle_artifact, _dump_resolved_process_graph, compiler_toolchain_fingerprint,
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

    publish()
    try:
        with scoped_work_contract(policy):
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
    args = parser.parse_args(argv)
    receipt = build_compartment(args.output, lower_only=args.lower_only,
                                progress=lambda message: print(message, flush=True))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
