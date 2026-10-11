"""Complete native compiler analysis, not a per-value proposal/callback."""
import dataclasses
import copy
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys

from src.compiler.native_compiler_accelerators import (
    READ_ONLY_MODULE, CompilerAcceleratorRegistry, _python_read_only_module,
    compile_and_register_storage_effects_accelerator,
)
from src.compiler.ssa_call_input_adapters import _infer_pointer_storage, _read_only_feed, _read_only_module_inputs
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def modules():
    result = [({}, {})]
    # A pure recursive SCC remains read-only; an alias's DEFINITION must
    # not poison its base address. Add a write in the second case to prove
    # propagation crosses both aliases and the recursive call edge.
    for writable in (False, True):
        a, index, address = SSAValue(0, 'ptr'), SSAValue(1, 'int64'), SSAValue(2, 'ptr')
        b, other_index = SSAValue(0, 'ptr'), SSAValue(1, 'int64')
        first = Function('first', [a, index], {'entry': BasicBlock('entry', [
            Instr('GetElementPtr', [a, index], address),
            Instr('Call', [address, index], None, attributes={'callee': 'second'}),
        ])})
        second = Function('second', [b, other_index], {'entry': BasicBlock('entry', [
            Instr('Call', [b, other_index], None, attributes={'callee': 'first'}),
            *([Instr('Store', [other_index, b], None)] if writable else []),
        ])})
        functions = {'first': first, 'second': second}
        identities, _ = _read_only_module_inputs(functions)
        expected = {identity: _read_only_feed(functions[identity[0]], identity[1], functions) for identity in identities}
        assert expected[('first', 0)] is not writable
        assert expected[('first', 1)] is True
        result.append((functions, expected))
    rng = random.Random(89301)
    for trial in range(32):
        functions = {}
        for number in range(5):
            args = [SSAValue(index, 'ptr' if index < 3 else 'int64') for index in range(4)]
            values = list(args)
            instructions = []
            for position in range(8):
                base = rng.choice(values)
                value = SSAValue(4 + position, 'ptr')
                if rng.random() < .6:
                    instructions.append(Instr('GetElementPtr', [base, args[3]], value))
                else:
                    instructions.append(Instr('View', [base], value))
                values.append(value)
            if trial % 3:
                instructions.append(Instr('Store', [args[3], rng.choice(values)], None))
            if number < 4 or trial % 2:
                instructions.append(Instr('Call', [rng.choice(values) for _ in range(4)], None,
                    attributes={'callee': f'f{(number + 1) % 5}'}))
            instructions.append(Instr('Call', [args[3], args[0]], None,
                attributes={'callee': 'external', 'ssa_output_argument': 1}))
            instructions.append(Instr('Ret', [], None))
            functions[f'f{number}'] = Function(f'f{number}', args, {'entry': BasicBlock('entry', instructions)})
        header_args = [SSAValue(0, 'ptr'), SSAValue(1, 'ptr')]
        functions['header'] = Function('header', header_args, {}, metadata={'native_interface': {
            'read_only_arguments': {0: True, 1: False},
        }})
        root = functions['f0']
        root.blocks['entry'].instrs.insert(0, Instr('Call', root.args[:2], None, attributes={'callee': 'header'}))
        identities, _inputs = _read_only_module_inputs(functions)
        expected = {identity: _read_only_feed(functions[identity[0]], identity[1], functions)
                    for identity in identities}
        result.append((functions, expected))
    return result


def test_same_authored_whole_module_effect_program_matches_existing_analyzer():
    for functions, expected in modules():
        assert _python_read_only_module(functions) == expected


def pointer_modules():
    rng = random.Random(34098)
    result = []
    for trial in range(24):
        args = [SSAValue(index, 'ptr', accounting={'retained': index}) for index in range(6)]
        if trial % 3 == 0:
            args[0].accounting['physical_dtype'] = 'float64'
        index_value = SSAValue(6, 'int64')
        values = [*args, index_value]
        instructions = []
        next_id = 7
        for _ in range(32):
            base = rng.choice(values[:6] if trial % 2 else values)
            value = SSAValue(next_id, 'ptr')
            next_id += 1
            op = rng.choice(('GetElementPtr', 'GEP', 'getelementptr', 'PtrCast', 'View'))
            instructions.append(Instr(op, [base, index_value], value))
            values.append(value)
            dtype = rng.choice((None, 'double', 'float', 'i64', 'i32', 'i1', 'ptr', 'unknown'))
            loaded = SSAValue(next_id, dtype)
            next_id += 1
            instructions.append(Instr('Load', [value], loaded))
            if rng.random() < .4:
                instructions.append(Instr('Store', [loaded, base], None))
        # Reverse alias order exercises convergence, not one lexical sweep.
        if trial % 2:
            instructions.reverse()
        function = Function('pointer', args, {'entry': BasicBlock('entry', instructions)})
        result.append({'pointer': function})

    # One exact type, no evidence, conflict, pointer-only veto, and View veto.
    args = [SSAValue(i, 'ptr') for i in range(5)]
    view = SSAValue(5, 'ptr')
    instructions = [Instr('Load', [args[0]], SSAValue(6, 'i64')),
        Instr('Load', [args[0]], SSAValue(7, 'int64')),
        Instr('Load', [args[2]], SSAValue(8, 'float64')),
        Instr('Load', [args[2]], SSAValue(9, 'int32')),
        Instr('Load', [args[3]], SSAValue(10, 'ptr')),
        Instr('View', [args[4]], view), Instr('Load', [view], SSAValue(11, 'float64'))]
    result.append({'exact': Function('exact', args, {'entry': BasicBlock('entry', instructions)})})

    # Callee memory evidence MUST NOT flow back through a call for storage
    # inference, even though that edge participates in read-only analysis.
    a, b, loaded = SSAValue(0, 'ptr'), SSAValue(0, 'ptr'), SSAValue(1, 'float64')
    result.append({'caller': Function('caller', [a], {'entry': BasicBlock('entry', [
        Instr('Call', [a], None, attributes={'callee': 'callee'})])}),
        'callee': Function('callee', [b], {'entry': BasicBlock('entry', [Instr('Load', [b], loaded)])})})

    # Accepted imported effects override its body, but the existing local
    # storage inference still reads typed memory operations in that body.
    pointer, loaded = SSAValue(0, 'ptr'), SSAValue(1, 'int32')
    result.append({'import': Function('import', [pointer], {'entry': BasicBlock('entry', [
        Instr('Load', [pointer], loaded)])}, metadata={'native_interface': {'read_only_arguments': {0: True}}})})
    return result


def pointer_cases():
    cases = []
    for functions in pointer_modules():
        oracle = copy.deepcopy(functions)
        _infer_pointer_storage(oracle)
        expected_storage = {(function.name, int(value.id)): dict(value.accounting or {})
                            for function in oracle.values() for value in function.args}
        identities, _ = _read_only_module_inputs(functions)
        effects = {identity: _read_only_feed(functions[identity[0]], identity[1], functions) for identity in identities}
        cases.append((functions, effects, expected_storage))
    return cases


def test_fused_eager_pointer_storage_is_exactly_the_existing_inference():
    for functions, effects, expected_storage in pointer_cases():
        actual, storage = _python_read_only_module(functions, include_storage_types=True)
        assert actual == effects
        _infer_pointer_storage(functions, storage)
        assert {(function.name, int(value.id)): dict(value.accounting or {})
                for function in functions.values() for value in function.args} == expected_storage


def test_our_compiler_compiles_whole_analysis_and_installs_it_in_real_serial_lowering(tmp_path):
    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(READ_ONLY_MODULE, _python_read_only_module, name='test:whole-module-eager')
    provider = compile_and_register_storage_effects_accelerator(
        tmp_path / 'native', registry=registry, progress=lambda _message: None,
    )
    artifact = dataclasses.replace(provider.metadata['artifact'], emission=None, _entry=None,
        _library=None, _validation_error_reset=None, _validation_error_take=None)
    payload = tmp_path / 'effects.pkl'
    worker = Path(__file__).resolve().parents[1] / 'tools/compiler_probes/graph_kernel_native_worker.py'
    cases = [*modules(), *pointer_cases()]
    for names, arrays, timeout in ((('storage_effect_modules',), cases, 15),
                                  (('storage_effect_serial',), {'installation': tmp_path / 'native'}, 20)):
        payload.write_bytes(pickle.dumps((artifact, provider.metadata['abi']['parameters'], names, arrays)))
        completed = subprocess.run([sys.executable, '-u', str(worker), str(payload)],
                                   capture_output=True, text=True, timeout=timeout)
        print(completed.stdout, end='', flush=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr[-3000:]
        if names == ('storage_effect_modules',):
            receipt = json.loads(next(line.split('=', 1)[1] for line in completed.stdout.splitlines()
                                     if line.startswith('STORAGE_EFFECT_NATIVE=')))
            assert receipt == {'modules': len(cases), 'native_calls': len(cases) - 1, 'exact_all_values': True}
        else:
            receipt = json.loads(next(line.split('=', 1)[1] for line in completed.stdout.splitlines()
                                     if line.startswith('STORAGE_EFFECT_REAL=')))
            assert receipt['whole_module_calls'] > 0 and receipt['value_facts'] > 100
            assert receipt['loaded_without_compilation']
