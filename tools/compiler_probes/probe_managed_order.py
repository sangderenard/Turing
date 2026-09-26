import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler import glsl_deployment_strategy as strategy
original_plan = strategy._build_shell_hierarchy_plan
def inspect_plan(shell):
    if shell.process_graph.G.graph.get('function_name') == 'root':
        print('ROOT GRAPH')
        for node_id, data in shell.process_graph.G.nodes(data=True):
            print(node_id, data.get('op'), getattr(data.get('expr_obj'), 'lineno', None),
                  data.get('parents'), data.get('attributes'))
    return original_plan(shell)
strategy._build_shell_hierarchy_plan = inspect_plan

def advance(material, callback):
    callback(material)
    material.count = material.count + 1
    return 5.0

source = '''
def increment(material):
    material.telemetry[0] = material.telemetry[0] + 1.0

def root(material):
    material.telemetry = material.telemetry * 0.0
    before = material.count
    result = advance(material, increment)
    material.telemetry[1] = material.count - before
    return result, material.telemetry
'''
policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
    'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
    'records': {'Material': {'identity': 'Material', 'fields': {
        'telemetry': {'storage':'span','dtype':'float64','rank':1,'shape':[2],'mutable':True},
        'count': {'storage':'scalar','dtype':'int64','mutable':True}}}},
    'bindings': [{'function':'*','parameter':'material','record':'Material'}], 'values':[]})
module, outputs, exports = lower_ast_source_to_ssa(source, 'root', name='order_probe',
    python_bindings={'advance': advance},
    extraction_contract=policy, tensor_ssa_reference=c_backend_repository_ssa_reference())
for name, function in module.functions.items():
    if name.startswith('order_probe'):
        print(name)
        for block in function.blocks.values():
            for inst in block.instrs:
                print(inst.op, [v.id for v in inst.args], None if inst.res is None else inst.res.id,
                      str(inst.attributes)[:180])
