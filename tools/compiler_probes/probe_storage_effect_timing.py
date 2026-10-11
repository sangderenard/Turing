"""Original Python vs installed compiler component; no build or DT program."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import json
    from statistics import median
    from time import perf_counter
    from src.compiler import fortran_c_shell
    from src.compiler.native_compiler_accelerators import load_storage_effects_accelerator
    from src.compiler.ssa_call_input_adapters import _infer_pointer_storage, _read_only_feed, _read_only_module_inputs
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    def forbidden(*_args, **_kwargs):
        raise AssertionError('this benchmark must LOAD the compiler component, not compile it')
    fortran_c_shell.lower_ast_source_to_ssa = forbidden
    started = perf_counter()
    provider = load_storage_effects_accelerator()
    if provider is None:
        raise RuntimeError('no matching installed storage-effect component')
    print('PRECOMPILED_LOAD_SECONDS=' + str(perf_counter() - started), flush=True)

    def build(function_count, formal_count, chain_length):
        functions = {}
        for number in range(function_count):
            args = [SSAValue(index, 'ptr') for index in range(formal_count)]
            index = SSAValue(formal_count, 'int64')
            instructions = []
            latest = list(args)
            value_id = formal_count + 1
            for _ in range(chain_length):
                for position in range(formal_count):
                    address = SSAValue(value_id, 'ptr')
                    value_id += 1
                    instructions.append(Instr('GetElementPtr', [latest[position], index], address))
                    latest[position] = address
            for position in range(formal_count):
                loaded = SSAValue(value_id, 'i64' if position % 2 else 'double')
                value_id += 1
                instructions.append(Instr('Load', [latest[position]], loaded))
            # Pass every storage to the next exact callee formal. The last
            # function writes only its first destination; the remaining
            # formals stay read-only along the whole call chain.
            if number + 1 < function_count:
                instructions.append(Instr('Call', latest, None,
                    attributes={'callee': f'f{number + 1}'}))
            else:
                instructions.append(Instr('Store', [index, latest[0]], None))
            instructions.append(Instr('Ret', [], None))
            functions[f'f{number}'] = Function(f'f{number}', args,
                {'entry': BasicBlock('entry', instructions)})
        return functions

    def timed(run, repetitions):
        samples = []
        for _ in range(5):
            started = perf_counter()
            for _ in range(repetitions):
                run()
            samples.append((perf_counter() - started) / repetitions)
        return median(samples)

    for function_count, formal_count, chain_length in ((1, 4, 4), (8, 8, 8), (24, 16, 12)):
        functions = build(function_count, formal_count, chain_length)
        identities, inputs = _read_only_module_inputs(functions)
        # Actual caller asks about formals, not every defined temporary.
        queries = tuple((function, value.id) for function in functions.values() for value in function.args)
        formals = tuple(value for function in functions.values() for value in function.args)
        def reset():
            for value in formals:
                value.accounting = {}
        def storage_receipt():
            return {(function.name, int(value.id)): dict(value.accounting or {})
                    for function in functions.values() for value in function.args}
        def python_run():
            reset()
            _infer_pointer_storage(functions)
            return ({(function.name, value): _read_only_feed(function, value, functions)
                     for function, value in queries}, storage_receipt())
        def native_run():
            reset()
            effects, storage = provider.run(functions, include_storage_types=True)
            _infer_pointer_storage(functions, storage)
            return effects, storage_receipt()
        def python_storage_run():
            reset()
            _infer_pointer_storage(functions)
            return storage_receipt()
        def python_effects_run():
            return {(function.name, value): _read_only_feed(function, value, functions)
                    for function, value in queries}
        expected = python_run()
        actual = native_run()
        assert all(actual[0][identity] == answer for identity, answer in expected[0].items())
        assert actual[1] == expected[1]
        python_seconds = timed(python_run, 2)
        native_seconds = timed(native_run, 4)
        storage_seconds = timed(python_storage_run, 2)
        effects_seconds = timed(python_effects_run, 2)
        print('COMPILER_EFFECT_TIMING=' + json.dumps({
            'functions': function_count, 'formal_queries': len(queries), 'ssa_values': len(identities),
            'instructions': inputs['instruction_count'], 'operand_occurrences': inputs['use_count'],
            'python_ms': python_seconds * 1000, 'native_ms': native_seconds * 1000,
            'python_pointer_storage_ms': storage_seconds * 1000,
            'python_read_only_ms': effects_seconds * 1000,
            'saved_ms': (python_seconds - native_seconds) * 1000,
            'speedup': python_seconds / native_seconds, 'native_calls_per_pass': 1,
            'native_includes': 'input packing, complete native analysis, final table unpacking',
            'native_computed_all_values': True,
            'analyses': ['original read-only queries', 'original pointer-storage inference'],
        }), flush=True)


if __name__ == '__main__':
    if '--child' in sys.argv:
        main()
    else:
        raise SystemExit(subprocess.run([sys.executable, '-u', __file__, '--child'], timeout=35).returncode)
