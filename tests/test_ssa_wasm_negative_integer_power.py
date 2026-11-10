"""Direct negative integer power lowering needs a declared f64 temporary."""
import base64
import json
import shutil
import subprocess
import pytest
from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue

@pytest.mark.parametrize('power', [-2, -3, -4, -8, 3, 8])
def test_integer_power_binary_executes(power):
    x, exponent, result = (SSAValue(i, 'float64') for i in range(3))
    f = Function('power_probe', [x], {'entry': BasicBlock('entry', [
        Instr('Const', [], exponent, attributes={'constant': float(power)}),
        Instr('Pow', [x, exponent], result), Instr('Ret', [result], None),
    ])}, metadata={'argument_names': ('x',), 'output_names': ('result',)})
    artifact = emit_ssa_function_to_wasm(IRModule({f.name: f}), f.name, work_contract='deploy')
    assert artifact.complete, artifact.shortfalls
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for real Wasm execution')
    script = '''(async()=>{const {instance}=await WebAssembly.instantiate(Buffer.from(process.argv[1],'base64'));const a=new Float64Array(instance.exports.memory.buffer);const out=[];for(const x of [-2,-.5,.5,2,3]){a[Number(process.argv[2])/8]=x;instance.exports.power_probe(0);out.push(a[Number(process.argv[3])/8]);}console.log(JSON.stringify(out));})().catch(e=>{console.error(e);process.exit(1)});'''
    completed = subprocess.run([node, '-e', script, base64.b64encode(artifact.binary).decode(), str(artifact.input_offsets[0]), str(artifact.output_offsets[0])], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == pytest.approx([x ** power for x in [-2,-.5,.5,2,3]], rel=1e-14)
