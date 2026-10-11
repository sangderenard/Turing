"""Compile the ENTIRE incumbent metadata copier, without a Python callback.

The source is taken from the production extractor, not reimplemented. Its
memo is an explicit boundary input because one extraction shares it across
graph metadata and every node. No deadline and no production route mutation.
"""
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def authored_program():
    from src.compiler.native_compiler_accelerators import metadata_copy_source
    return metadata_copy_source()


def main():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

    # Opt-in physical ABI for the unchanged production source. The context
    # schema is the incumbent SSA descriptor, not a replacement copier.
    context_options = {}
    if '--object-context' in sys.argv[1:]:
        from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
        from src.transmogrifier.ssa import (SSAObjectContext,
            SSASequenceDescriptor, SSASequenceCapacityPolicy)
        mint = GLOBAL_MONOTONIC_IDS.mint
        owner = mint()
        def table(dtypes, keys):
            columns = tuple(mint() for _ in dtypes)
            return SSASequenceDescriptor(columns[0], columns, mint(), mint(),
                status_address_id=mint(), column_dtypes=dtypes, key_columns=keys,
                capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=owner)
        context_options['native_object_context'] = SSAObjectContext('compiler.metadata.context',
            table(('opaque_ref', 'int64', 'int64'), (0,)),
            table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(),
            ('builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
             'builtins.frozenset', 'builtins.object'),
            type_vocabulary=('builtins.dict', 'builtins.list', 'builtins.tuple',
                'builtins.set', 'builtins.frozenset', 'builtins.object'))

    # Native id(value) is the admitted stable integer handle. Its memo uses
    # the incumbent integer-key/opaque-value table ABI, not reference-equality
    # object-member keys. This only declares the lowering probe's boundary;
    # it does not import/export a Python memo or change the production copier.
    contract = ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml")
    contract = contract.with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": "metadata_copy_program", "parameter": name,
         "storage": "keyed" if name == "metadata_memo" and context_options else "reference",
         "mutable": True,
         **({"dtype": "opaque_ref", "key_encoding": "integer_identity"}
            if name == "metadata_memo" and context_options else {}),
         "python_type": "builtins.object" if name == "value" else "builtins.dict"}
        for name in ("value", "metadata_memo")
    ]}).with_execution_file(ROOT / "extraction_contracts/vehicle_full_native_execution.yaml")
    source = authored_program()
    print("EXACT_METADATA_COPIER_SUBMISSION: " + str(len(source.splitlines()))
          + " authored lines; full-native, no Python callbacks", flush=True)
    started = time.perf_counter()
    if '--control-writes' in sys.argv:
        from tools.compiler_probes.observe_copier_contract_batch import trace
        sys.settrace(trace)
    try:
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "metadata_copy_program", name="compiler_metadata_copy",
            python_bindings={}, extraction_contract=contract, runtime_closure_only=True,
            progress=lambda message: print(str(message), flush=True)
            if str(message).startswith("ssa-source:") else None, progress_bars=False,
            **context_options,
        )
        if '--details' in sys.argv[1:]:
            for function in module.functions.values():
                for block in function.blocks.values():
                    for instruction in block.instrs:
                        if instruction.op == 'Call' and not instruction.attributes.get('callee'):
                            print('UNBOUND_CALL', function.name, block.name, repr(instruction), flush=True)
            def emission_trace(frame, event, argument):
                if frame.f_code.co_name != '_emit_repository_call_module':
                    return None
                if event == 'line' and frame.f_lineno == 4435:
                    print('EMISSION_REFUSAL', frame.f_locals.get('name'), repr(frame.f_locals.get('instruction')), flush=True)
                return emission_trace
            sys.settrace(emission_trace)
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        if '--details' in sys.argv[1:]:
            sys.settrace(None)
        print("LLVM_COMPLETE: " + repr((artifact.complete, artifact.shortfalls)), flush=True)
        return 0 if artifact.complete else 2
    except Exception as error:
        print("EXACT_METADATA_COPIER_BLOCKER: " + type(error).__name__ + ": " + str(error),
              flush=True)
        traceback.print_exc(file=sys.stdout)
        return 2
    finally:
        sys.settrace(None)
        print("ELAPSED: " + str(time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
