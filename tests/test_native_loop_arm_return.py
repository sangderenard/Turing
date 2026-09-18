"""An effectful return inside a loop must leave from its lexical arm."""

import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_self_check import run_all


def test_native_loop_return_dominates_and_does_not_repeat_mutation(tmp_path):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    source = '''
class Material:
    def mutate(self):
        self.state[0] = self.state[0] + 1.0
        return self.state[0]

def root(material, enabled, limit):
    turns = 0
    while turns < limit:
        material.state[1] = material.state[1] + 1.0
        if enabled:
            result = material.mutate()
            return result
        turns = turns + 1
    return material.state[0]
'''
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': [
            {'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
             'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'},
            {'function': 'root', 'parameter': 'limit', 'storage': 'scalar',
             'dtype': 'int64', 'rank': 0, 'python_type': 'builtins.int'},
        ],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, 'root', name='loop_arm_return', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    assert run_all(module) == []
    root = module.functions['loop_arm_return__root']
    names = dict(root.metadata['parameter_names'])
    state = next(argument.id for argument in root.args
                 if argument.accounting.get('program_abi_field') == 'state')
    returned = next(instruction.args[0].id for block in root.blocks.values()
                    for instruction in block.instrs if instruction.op == 'Ret')
    (tmp_path / 'repository-ssa.pkl').write_bytes(pickle.dumps(module))
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native', optimization='O0')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, names, state, returned)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
artifact, names, state_id, output_id = pickle.load(open(sys.argv[1], 'rb'))
for enabled, limit, expected, visits in (
        (True, 3, 3.0, 1), (False, 3, 2.0, 3),
        (True, 0, 2.0, 0), (False, 5, 2.0, 5)):
    execution = artifact.prepare_execution({state_id: [2.0, 9.0],
        names['enabled']: enabled, names['limit']: limit}).run()
    assert execution.buffers[state_id].tolist() == [expected, 9.0 + visits]
    assert execution.buffers[output_id].item() == expected
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
