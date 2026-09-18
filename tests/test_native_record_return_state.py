"""A returned record must publish the field state at its return edge."""

import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import (
    normalize_declared_scalar_record_shapes,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    SSARecordDescriptor,
    SSARecordFieldDescriptor,
    SSARecordFieldStorage,
    SSARecordTable,
    SSAValue,
)


def test_declared_scalar_record_shape_wins_after_result_refinement():
    value = SSAValue(7, dtype="float64", shape=(1, 1, 1))
    function = Function("root", [value], {
        "entry": BasicBlock("entry", []),
    })
    module = IRModule({"root": function})
    module.record_tables["root"] = SSARecordTable(records={
        4: SSARecordDescriptor(4, "Metrics", (
            SSARecordFieldDescriptor(
                "advanced_dt", SSARecordFieldStorage.SCALAR,
                storage_identity="Metrics.advanced_dt",
                value_ids=(7,), dtype="float64",
            ),
        )),
    })

    receipts = normalize_declared_scalar_record_shapes(module)

    assert value.shape == ()
    assert receipts == ({
        "function": "root",
        "value_id": 7,
        "prior_shape": (1, 1, 1),
        "record_fields": ((4, "advanced_dt", "Metrics.advanced_dt"),),
        "priority": "declared_record_storage",
        "tie_policy": "incumbent",
    },)
    assert normalize_declared_scalar_record_shapes(module) == ()


def test_child_record_conditional_write_reaches_return(tmp_path):
    source = '''
def keep(state):
    return bool(state.flag), state

def root(state, change):
    before, result = keep(state)
    observed = bool(result.flag)
    if change:
        result.flag = False
    return before, observed, result
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'State': {'identity': 'State', 'fields': {
            'flag': {'storage': 'scalar', 'dtype': 'bool', 'rank': 0, 'mutable': True},
        }}},
        'bindings': [{'function': '*', 'parameter': 'state', 'record': 'State'}],
        'values': [{'function': 'root', 'parameter': 'change', 'storage': 'scalar',
                    'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}],
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='record_return_state', extraction_contract=policy,
    )
    root = module.functions['record_return_state__root']
    (tmp_path / 'module.pkl').write_bytes(pickle.dumps(module))
    names = dict(root.metadata['parameter_names'])
    assert 'change' in names, 'Conditional field write and its guard disappeared'
    flag = next(argument.id for argument in root.args
                if argument.accounting.get('program_abi_field') == 'flag')
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, flag, names['change'],
                                   tuple(value.id for value in outputs[root.name]), source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
from types import SimpleNamespace
import numpy as np
artifact, flag, change, output, source = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
for initial in (False, True):
    for enabled in (False, True):
        before, observed, record = namespace['root'](SimpleNamespace(flag=initial), enabled)
        execution = artifact.prepare_execution({
            flag: np.array([initial], dtype=np.bool_),
            change: np.array([enabled], dtype=np.bool_),
        }).run()
        actual = tuple(execution.buffers[value].item() for value in output)
        assert actual == (before, observed, record.flag), (initial, enabled, actual, before, record.flag)
''', str(saved)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
