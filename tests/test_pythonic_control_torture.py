"""Small authored programs must preserve effects, not just lower successfully.

These cases deliberately separate collection population from the enclosing
guard: correlating those booleans hid both flattened and duplicated loops.
The oracle is exact literal cardinality plus the number of authored appends.
"""
import pickle
import subprocess
import sys

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


CASES = {
    "literal": ("""
def root():
    values = [10.0, 20.0]
    values.append(30.0)
    return len(values)
""", (({}, 3), ({}, 3))),
    "literal_payload": ("""
def root():
    values = [10.0, 20.0]
    values.append(30.0)
    return values[0] + values[1] + values[2]
""", (({}, 60.0), ({}, 60.0))),
    "literal_inside_while": ("""
def root():
    while True:
        values = [10.0, 20.0]
        values.append(30.0)
        break
    return len(values)
""", (({}, 3), ({}, 3))),
    "guarded_rows": ("""
def root(guard, populated):
    failures = []
    lines = [10.0]
    while True:
        if populated:
            failures.append((1.0, 2.0, (3.0,)))
        if guard:
            for index, (dt_f, metrics, reasons) in enumerate(failures, 1):
                lines.append(20.0)
        break
    return len(lines)
""", tuple((dict(guard=guard, populated=populated), 1 + int(guard and populated))
             for guard, populated in ((True, False), (False, True), (True, True),
                                      (False, False), (True, True), (True, False)))),
}

_source, _trials = CASES["guarded_rows"]
_append = "            failures.append((1.0, 2.0, (3.0,)))"
CASES["guarded_rows_two"] = (_source.replace(_append, _append + "\n" + _append),
    tuple((inputs, 1 + 2 * int(inputs["guard"] and inputs["populated"])) for inputs, _ in _trials))
CASES["guarded_rows_seventeen"] = (_source.replace(_append, "\n".join([_append] * 17)),
    tuple((inputs, 1 + 17 * int(inputs["guard"] and inputs["populated"])) for inputs, _ in _trials))
CASES["guarded_splice"] = (_source.replace("enumerate(failures, 1)",
    "enumerate((*failures, (4.0, 5.0, (6.0,))), 1)"),
    tuple((inputs, 1 + int(inputs["guard"]) * (1 + int(inputs["populated"]))) for inputs, _ in _trials))

CASES["mixed_numeric_rows"] = ("""
def root(seed, guard):
    values = []
    total = seed
    for i in range(32):
        total = total + i * 0.125
        if guard:
            values.append(total)
    for value in values:
        total = total + value * 0.5
    return total
""", tuple((dict(seed=0.125, guard=guard),
            0.125 + 0.125 * 32 * 31 / 2 +
            (0.5 * (32 * 0.125 + 0.125 * 32 * 31 * 33 / 6) if guard else 0))
           for guard in (False, True, False, True)))

CASES["integer_while_counter"] = ("""
def root(limit):
    iters = 0
    while iters < limit:
        iters += 1
    return iters
""", tuple((dict(limit=limit), limit) for limit in (0, 1, 7, 0)))
_counter_source, _counter_trials = CASES['integer_while_counter']
CASES['folded_integer_seed'] = (_counter_source.replace('while iters < limit:', 'while limit > 0:')
    .replace('        iters += 1', '        limit -= 1\n        iters += 1'), _counter_trials)

CASES['first_read_augmented_branch'] = ('''
def root(flags, mask, add):
    if add & mask:
        flags &= ~mask
    return flags | add
''', tuple((dict(flags=flags, mask=mask, add=add), expected)
          for flags, mask, add, expected in ((15, 6, 2, 11), (15, 6, 1, 15),
                                            (8, 6, 2, 10), (15, 6, 2, 11))))
CASES['first_read_augmented_loop'] = ('''
def root(value, count):
    for index in range(count):
        value += 2
    return value
''', tuple((dict(value=value, count=count), value + 2 * count)
          for value, count in ((5, 0), (5, 1), (5, 7), (5, 0))))

CASES['unary_plus_return'] = ('''
def root(value):
    return +value
''', tuple((dict(value=value), value) for value in (0.125, -3.0, 0.0, 0.125)))

for _index, (_xs, _ys, _flag) in enumerate((((), (), True), ((1, 2), (), True),
                                          ((), (3, 4), True), ((1, 2), (3, 4), False))):
    CASES[f'generator_target_scope_{_index}'] = (f'''
def root():
    xs = {_xs!r}
    ys = {_ys!r}
    flag = {_flag!r}
    tagged = tuple(int(address) for address in xs)
    total = 0
    for address in ys:
        total = total + address
    if flag:
        return total + len(tagged)
    return total
''', (({}, sum(_ys) + (len(_xs) if _flag else 0)),) * 2)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ("c", "llvm", "fortran"))
def test_pythonic_control_native_effects(tmp_path, case, backend):
    source, trials = CASES[case]
    policy = ExtractionContract("extraction_contracts/program_extraction.yaml").with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml").with_program_abi({
        "bindings": [], "values": [
            {"function": "root", "parameter": name, "storage": "scalar",
             "dtype": "bool" if isinstance(value, bool) else "int64" if isinstance(value, int) else "float64",
             "rank": 0, "python_type": "builtins.bool" if isinstance(value, bool) else "builtins.int" if isinstance(value, int) else "builtins.float"}
            for name, value in trials[0][0].items()
        ]})
    module, outputs, exports = lower_ast_source_to_ssa(source, "root", name="pythonic_" + case,
        extraction_contract=policy, progress=lambda _: None)
    function = module.functions[exports[0]]
    from src.compiler.identity_concordance import concordance_report
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    if backend == "c":
        artifact = emit_ssa_module_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "native")
    else:
        from src.compiler.ssa_fortran_backend import emit_module
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        emitted = emit_module(module, name="pythonic_" + case, outputs=outputs)
        assert emitted.complete, emitted.shortfalls
        built = compile_fortran_module_c_shell(emitted, {}, tmp_path / "native",
            entrypoint=function.name, library=True)
        artifact = (built.executable_path, emitted.api.entry_point(function.name), emitted.api)
    proof = tmp_path / "proof.pkl"
    proof.write_bytes(pickle.dumps((artifact, dict(function.metadata["parameter_names"]),
                                  outputs[function.name][0].id, trials, backend)))
    # A native fault cannot take down pytest, and repeated calls reuse the
    # same artifact/buffers. There is no timeout or interpreter fallback.
    invocation = subprocess.run([sys.executable, "-u", "-c", """
import pickle, sys
import numpy as np
artifact, parameters, result, trials, backend = pickle.load(open(sys.argv[1], 'rb'))
feeds = {parameters[name]: np.array([value], dtype=np.bool_ if isinstance(value, bool) else np.int64 if isinstance(value, int) else np.float64)
         for name, value in trials[0][0].items()}
if backend == 'c':
    execution = artifact.prepare_execution(feeds)
elif backend == 'llvm':
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    execution = prepare_artifact_execution(artifact, feeds)
else:
    from src.compiler.ssa_fortran_backend import FortranCoreNative
    execution = FortranCoreNative(*artifact).prepare_execution(feeds)
failures = []
for inputs, expected in trials:
    for name, value in inputs.items():
        execution.buffers[parameters[name]][0] = value
    execution.run()
    actual = execution.buffers[result].item()
    print('PYTHONIC_NATIVE', backend, inputs, 'actual', actual, 'expected', expected, flush=True)
    if actual != expected:
        failures.append((inputs, actual, expected))
    if backend == 'fortran' and inputs.get('populated'):
        # Cardinality alone used to hide bool((3.0,)) replacing nested data.
        # Read the authored failures arena through its published identity.
        api = artifact[2]
        for descriptor in api.metadata['sequence_tables'][artifact[1].name]:
            if 'failures' not in descriptor['source_names']:
                continue
            length = int(execution.buffers[descriptor['length_address_id']][0])
            assert length > 0
            for column, expected_element in zip(descriptor['column_value_ids'], (1.0, 2.0, 3.0), strict=True):
                np.testing.assert_array_equal(execution.buffers[column][:length], np.full(length, expected_element))
        print('PYTHONIC_NATIVE nested row payload preserved', flush=True)
assert not failures, failures
""", str(proof)], capture_output=True, text=True)
    print(invocation.stdout, end="")
    assert invocation.returncode == 0, invocation.stdout + invocation.stderr
