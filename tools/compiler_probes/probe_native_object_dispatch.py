"""Small public-source object/ctypes probe; no timeout or production install."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
    from src.compiler.shell_external_references import ExternalReferenceValueCodec
    from src.compiler.ssa_object_context import lower_object_context
    from src.compiler.ssa_llvm_backend import (compile_artifact,
        emit_ssa_function_to_llvm, prepare_artifact_execution)
    from src.transmogrifier.ssa import SSAObjectContext, SSASequenceDescriptor, SSASequenceCapacityPolicy
    mint = GLOBAL_MONOTONIC_IDS.mint
    owner = mint()
    def table(dtypes, keys):
        columns = tuple(mint() for _ in dtypes)
        return SSASequenceDescriptor(columns[0], columns, mint(), mint(),
            status_address_id=mint(), column_dtypes=dtypes, key_columns=keys,
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=owner)
    scope = SSAObjectContext('probe.dispatch.context', table(('opaque_ref', 'int64'), (0,)),
        table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(),
        ('builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
         'builtins.frozenset', 'builtins.object'))
    contract = ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': 'value',
         'storage': 'reference', 'python_type': 'builtins.object'}]}).with_execution_file(
        ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')
    started = time.perf_counter()
    module, _, exports = lower_ast_source_to_ssa('def root(value):\n    return tuple(value)\n',
        'root', name='object_dispatch_probe', extraction_contract=contract,
        native_object_context=scope, progress=lambda _: None, progress_bars=False)
    directory = Path(tempfile.mkdtemp(prefix='turing-object-dispatch-'))
    native = compile_artifact(emit_ssa_function_to_llvm(module, exports[0]), directory=directory / 'source')
    release_module = lower_object_context(scope, prefix='dispatch_release')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'dispatch_release_release'),
        directory=directory / 'release')
    root = module.functions[exports[0]]
    input_id = dict(root.metadata['parameter_names'])['value']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    print(f'BUILD_SECONDS {time.perf_counter() - started:.6f}', flush=True)
    leaf = object()
    for size in (0, 17, 257, 4097):
        start = time.perf_counter()
        original = [leaf] * size
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        imported = time.perf_counter() - start
        try:
            feeds = {input_id: np.array([boundary.roots[0]], dtype=np.int64)}
            start = time.perf_counter()
            invocation = boundary.bind_execution(native, root, feeds)
            prepared = time.perf_counter() - start
            start = time.perf_counter()
            invocation.run()
            stamped = time.perf_counter() - start
            start = time.perf_counter()
            result, = boundary.export((int(invocation.buffers[output].item()),))
            exported = time.perf_counter() - start
            assert len(result) == size and all(item is leaf for item in result)
            print(f'STAMP n={size} calls={invocation.native_invocations} import_ms={imported*1000:.6f} '
                  f'bind_ms={prepared*1000:.6f} native_ms={stamped*1000:.6f} export_ms={exported*1000:.6f}', flush=True)
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()
    # Same tuple -> same tuple is a no-allocation control, isolating dispatch
    # setup from row copying/growth. Both lanes call the SAME compiled source.
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, ((leaf,),))
    try:
        feeds = {input_id: np.array([boundary.roots[0]], dtype=np.int64)}
        invocation = boundary.bind_execution(native, root, feeds)
        invocation.run()
        samples = 1000
        start = time.perf_counter()
        for _ in range(samples):
            prepare_artifact_execution(native, {**feeds, **boundary.buffers}).run()
        fresh = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(samples):
            invocation.run()
        reused = time.perf_counter() - start
        assert int(invocation.buffers[output].item()) == boundary.roots[0]
        print(f'DISPATCH samples={samples} prepare_each_us={fresh*1e6/samples:.6f} '
              f'reuse_us={reused*1e6/samples:.6f} ratio={fresh/reused:.3f}', flush=True)
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()
    print(f'ARTIFACT_DIRECTORY {directory}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
