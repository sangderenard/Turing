import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_record_method_call_retains_runtime_conditional(tmp_path):
    source = '''
class Material:
    def mutate(self):
        self.state[0] = 7.0

def root(material, enabled):
    if enabled:
        material.mutate()
    return material.state[0]
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': [{'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
                    'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, 'root', name='call_only', extraction_contract=policy,
    )
    root = module.functions['call_only__root']
    branches = [instruction for block in root.blocks.values()
                for instruction in block.instrs if instruction.op == 'CondBr']
    assert len(branches) == 1
    enabled = dict(root.metadata['parameter_names'])['enabled']
    assert any(argument.id == enabled for argument in root.args)
    calls = [(name, instruction) for name, block in root.blocks.items()
             for instruction in block.instrs if instruction.op == 'Call'
             and 'mutate' in str(instruction.attributes.get('callee', ''))]
    assert len(calls) == 1
    assert calls[0][0] != 'entry'
    state = next(argument.id for argument in root.args
                 if argument.accounting.get('program_abi_field') == 'state')
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'conditional_method')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, state, enabled)))
    # Execute native code under a subprocess bound; test both outcomes with
    # fresh state so an unconditional or missing method call cannot pass.
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, state_id, enabled_id = pickle.load(stream)
for enabled in (False, True):
    feeds = {state_id: np.array([2.0, 3.0]),
             enabled_id: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    state = np.asarray(result.buffers[state_id]).reshape(-1).tolist()
    assert state == [7.0 if enabled else 2.0, 3.0], (enabled, state)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_conditional_capture_precedes_later_mutation_natively(tmp_path):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference

    source = '''
class Material:
    def read(self):
        return self.state[0]

def root(material, enabled):
    saved = material.read() if enabled else -1.0
    material.state[0] = 7.0
    return saved
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': [{'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
                    'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}],
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='capture_order', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['capture_order__root']
    enabled = dict(root.metadata['parameter_names'])['enabled']
    state = next(argument.id for argument in root.args
                 if argument.accounting.get('program_abi_field') == 'state')
    assert len(root.args) == 2
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'capture_order')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, state, enabled, outputs[root.name][0].id)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, state_id, enabled_id, result_id = pickle.load(stream)
for enabled in (False, True):
    feeds = {state_id: np.array([2.0, 3.0]),
             enabled_id: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    actual = float(np.asarray(result.buffers[result_id]).reshape(-1)[0])
    assert actual == (2.0 if enabled else -1.0), (enabled, actual)
    assert np.array_equal(result.buffers[state_id], [7.0, 3.0])
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
