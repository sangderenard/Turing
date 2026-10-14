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


def build_compartment(directory, *, lower_only=False, progress=print):
    import src.compiler.identity_concordance as incumbent
    from src.common.tensors.accelerator_backends.aot_checkpoint import (
        AOTCheckpointStore, callable_digest,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _atomic_json, _dump_pickle_artifact, _dump_resolved_process_graph,
        _load_pickle_artifact, compiler_toolchain_fingerprint,
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
                module, outputs, exports = lower_ast_source_to_ssa(
                    source, None, source_module=incumbent.__name__,
                    python_package=incumbent.__package__, python_bindings=vars(incumbent),
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
