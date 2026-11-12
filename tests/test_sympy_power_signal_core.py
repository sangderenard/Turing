"""SymPy power provenance reaches the existing scalar signal core."""
import sympy
import pytest
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.ir_identities import lower_transcendentals_to_signal_cores, two_product_flavor_scope
from src.compiler.identity_concordance import concordance_report


def test_sympy_live_power_preserves_origin_and_uses_signal_core():
    x, y, z = sympy.symbols("pressure_ratio gamma result")
    compiled = compile_sympy_equations([sympy.Eq(z, x ** ((y-1)/y))], name="sympy_power_signal_core")
    power = next(i for b in compiled.function.blocks.values() for i in b.instrs if i.op == "Pow")
    assert power.attributes["source_type"] == "Pow"
    assert power.attributes["source_language"] == "sympy"
    result_id = power.res.id
    from src.compiler.identity_concordance import _ACTIVE_IDENTITY_BOOK
    token = _ACTIVE_IDENTITY_BOOK.set(compiled.module.metadata["identity_book"])
    try:
        with two_product_flavor_scope("split"):
            receipt = lower_transcendentals_to_signal_cores(compiled.module.functions, mode="scientific")
    finally:
        _ACTIVE_IDENTITY_BOOK.reset(token)
    assert receipt["lowered"][compiled.function.name] == {"Pow": 1}
    instructions = [i for b in compiled.function.blocks.values() for i in b.instrs]
    assert not any(i.op in {"Pow", "Log", "Exp", "Fma"} for i in instructions)
    assert any(i.res is not None and i.res.id == result_id for i in instructions)
    assert {i.attributes.get("error_code") for i in instructions if i.op == "Call"} == {92, 93, 95}
    assert "0 finding(s)" in concordance_report(compiled.module).splitlines()[0]


def test_unproven_pow_origin_still_refused():
    from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock
    x, y, z = (SSAValue(i, "float64") for i in range(3))
    f = Function("unknown_power", [x,y], {"entry": BasicBlock("entry", [Instr("Pow", [x,y], z, attributes={"source_type":"Pow"}), Instr("Ret", [z], None)])})
    with pytest.raises(ValueError, match="neither a math.pow"):
        lower_transcendentals_to_signal_cores({f.name:f}, mode="scientific")


def test_wasm_backend_rule_uses_core_without_mutating_shared_symbolic_ir():
    from src.compiler.backend_identities import apply_backend_identities
    x,y,z = sympy.symbols("x y z")
    compiled = compile_sympy_equations([sympy.Eq(z,x**y)], name="sympy_wasm_power_rule")
    result = apply_backend_identities(compiled.module, {}, backend="wasm", licensed_inexact=False)
    assert len(compiled.function.blocks) == 1
    assert any(i.op == "Pow" for i in compiled.function.blocks["entry"].instrs)
    lowered = result.module.functions[compiled.function.name]
    assert len(lowered.blocks) == 7
    assert lowered.metadata["transcendental_pipeline"]["lowered"] == {"Pow":1}
    decision = next(d for d in result.decisions if d.identity == "wasm_scalar_power_signal_cores")
    assert decision.applied and decision.exact
    assert decision.before_sha256 != decision.after_sha256
    assert "0 finding(s)" in concordance_report(result.module).splitlines()[0]


@pytest.mark.parametrize("pneumatic", [False, True])
def test_direct_wasm_executes_signal_power_and_named_faults(tmp_path, pneumatic):
    import json, shutil, subprocess
    import mpmath
    from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for real Wasm execution")
    x,y,z = sympy.symbols("x y z")
    expression=x**((y-1)/y) if pneumatic else x**y
    compiled = compile_sympy_equations([sympy.Eq(z,expression)], name="canonical_pow")
    artifact = emit_ssa_function_to_wasm(compiled.module, compiled.function.name, work_contract="deploy")
    assert artifact.complete, artifact.shortfalls
    wasm = tmp_path / "power.wasm"
    wasm.write_bytes(artifact.binary)
    cases = [[1,2/7],[2,2/7],[100,2/7],[1e8,.37],[.0001,-.5],[-2,3],
             [3,34],[2,-1074],[2,-1075],[2,1023],[1.0000000000000002,12345],
             [0,0],[-0.0,3],[-2,-3],[.5,1000]]
    if pneumatic:
        cases=[[1,1.4],[2,1.4],[100,1.4],[1e8,1.63],[1e-100,1.2],[1e100,1.001]]
    with mpmath.workprec(250):
        expected = [float(mpmath.power(mpmath.mpf(a),mpmath.mpf((b-1)*(1/b) if pneumatic else b))) for a,b in cases]
    if not pneumatic:
        expected[12] = -0.0
    payload={"inputs":artifact.input_offsets,"output":artifact.output_offsets[0],
             "cases":cases,"expected":expected,"pneumatic":pneumatic}
    data=tmp_path / "cases.json"
    data.write_text(json.dumps(payload),encoding="utf-8")
    script = r"""const fs=require('fs');(async()=>{
const bytes=fs.readFileSync(process.argv[1]),p=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const module=new WebAssembly.Module(bytes),imports=WebAssembly.Module.imports(module);
if(imports.some(i=>i.module==='math'))throw Error('Numerical host import');
if(imports.length!==1||imports[0].module!=='turing'||imports[0].name!=='turing_validation_error')throw Error('Missing named fault ABI');
let fault=0;const {instance}=await WebAssembly.instantiate(bytes,{turing:{turing_validation_error:code=>{fault=code;}}});
const a=new Float64Array(instance.exports.memory.buffer);
const run=(x,y)=>{fault=0;a[p.inputs[0]/8]=x;a[p.inputs[1]/8]=y;a[p.output/8]=987654321;instance.exports.canonical_pow(0);return a[p.output/8];};
p.cases.forEach(([x,y],i)=>{const actual=run(x,y);if(fault||!Object.is(actual,p.expected[i]))throw Error(JSON.stringify({x,y,actual,expected:p.expected[i],fault}));});
if(!p.pneumatic){for(const [x,y,code] of [[0,-1,92],[-2,.5,93],[1e308,2,95]]){const actual=run(x,y);if(fault!==code||actual!==987654321)throw Error('Fault publication '+JSON.stringify({x,y,actual,fault,code}));}
for(const [x,y,expected] of [[NaN,0,1],[Infinity,-1,0],[-Infinity,3,-Infinity],[-0,-3,null]]){const actual=run(x,y);if(expected===null){if(fault!==92)throw Error('signed zero fault');}else if(fault||!Object.is(actual,expected))throw Error('special power');}}
})().catch(e=>{console.error(e);process.exit(1)});"""
    completed=subprocess.run([node,'-e',script,str(wasm),str(data)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_cached_pneumatic_power_keeps_identity_book(tmp_path):
    from src.compiler.backend_identities import apply_backend_identities
    x,y,z = sympy.symbols("pressure_ratio gamma result")
    equations = [sympy.Eq(z,x**((y-1)/y))]
    compile_sympy_equations(equations,name="cached_pneumatic_power")
    compiled = compile_sympy_equations(equations,name="cached_pneumatic_power")
    assert compiled.cache_hit
    assert "identity_book" in compiled.module.metadata
    result = apply_backend_identities(compiled.module,{},backend="wasm",licensed_inexact=True)
    assert "0 finding(s)" in concordance_report(result.module).splitlines()[0]


def test_worker_binds_canonical_fault_abi_for_all_three_modules(tmp_path):
    import shutil, subprocess
    from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
    from src.compiler.state_loop_deployment import emit_javascript_physics_worker
    from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule
    node=shutil.which("node")
    if not node:
        pytest.skip("Node required for real Wasm execution")
    value=SSAValue(0,"float64")
    function=Function("fault_probe",[],{"entry":BasicBlock("entry",[
        Instr("Call",[],None,attributes={"callee":"turing_validation_error","error_code":92}),
        Instr("Const",[],value,attributes={"constant":42.0}),Instr("Ret",[value],None)])})
    artifact=emit_ssa_function_to_wasm(IRModule({function.name:function}),function.name)
    assert artifact.complete,artifact.shortfalls
    wasm=tmp_path / "fault.wasm";wasm.write_bytes(artifact.binary)
    worker=tmp_path / "worker.js";worker.write_text(emit_javascript_physics_worker(),encoding="utf-8")
    script=r"""const fs=require('fs'),vm=require('vm');(async()=>{
const bytes=fs.readFileSync(process.argv[2]);const instances=[];const stop=Error('three real instantiations');
const context={Math,performance,console,WebAssembly:{instantiate:async(binary,imports)=>{
const made=await WebAssembly.instantiate(binary,imports);instances.push(made.instance);
if(instances.length===3)throw stop;return made;}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
try{await context.onmessage({data:{type:'init',wasm:bytes,abi:{},vehicleWasm:{wasm:bytes},contactWasm:{wasm:bytes}}});}catch(e){if(e!==stop)throw e;}
if(instances.length!==3)throw Error('Missing worker binding');
for(const instance of instances){let caught=false;try{instance.exports.fault_probe(0);}catch(e){if(e.code!==92)throw e;caught=true;}if(!caught)throw Error('Silenced fault');}
})().catch(e=>{console.error(e);process.exit(1)});"""
    completed=subprocess.run([node,'-e',script,str(worker),str(wasm)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_id_issuer_reserves_only_restored_minted_serials():
    from src.compiler.monotonic_ids import MonotonicIdSource
    from src.compiler.id_space import compose, MINTED, TENSOR, serial_of
    issuer=MonotonicIdSource(first_id=10)
    issuer.reserve_existing([None, compose(100,TENSOR),compose(25,MINTED)])
    assert serial_of(issuer.mint()) == 26
    issuer.reserve_existing([compose(1,MINTED)])
    assert serial_of(issuer.mint()) == 27


def test_restored_symbolic_book_can_mint_in_fresh_process(tmp_path):
    import pickle, subprocess, sys
    x,y,z=sympy.symbols("pressure_ratio gamma result")
    compiled=compile_sympy_equations([sympy.Eq(z,x**((y-1)/y))],name="restored_pneumatic_power")
    saved=tmp_path / "module.pickle"
    saved.write_bytes(pickle.dumps(compiled.module))
    script="""import pickle,sys
from src.compiler.backend_identities import apply_backend_identities
from src.compiler.identity_concordance import concordance_report
with open(sys.argv[1],'rb') as stream: module=pickle.load(stream)
result=apply_backend_identities(module,{},backend='wasm',licensed_inexact=True)
report=concordance_report(result.module)
print(report)
assert '0 finding(s)' in report.splitlines()[0]
"""
    completed=subprocess.run([sys.executable,'-c',script,str(saved)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
