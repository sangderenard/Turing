import ast
import pytest
from src.compiler.precision_policy import ShaderPrecisionPolicy
from src.compiler.ssa_python_materializer import materialize_function_body
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

def source(dtype='float32', policy=None):
    value=SSAValue(1,dtype=dtype)
    f=Function('literal',[],{'entry':BasicBlock('entry',[Instr('Const',[],value,attributes={'constant':0.1}),Instr('Ret',[value],None)])})
    body,_=materialize_function_body(f,shader_precision=policy)
    return body

def test_shader_default_is_float32_and_native_unchanged():
    assert ShaderPrecisionPolicy().default_dtype=='float32'
    native=source()
    assert isinstance(native[0],ast.Assign)
    shader=source(policy=ShaderPrecisionPolicy())
    assert isinstance(shader[0],ast.AnnAssign)
    assert shader[0].annotation.value=='float32'
    assert shader[0].value.value==native[0].value.value==0.1

def test_explicit_float64_is_preserved_not_silently_downgraded():
    assert source('float64',ShaderPrecisionPolicy())[0].annotation.value=='float64'
    assert source('float32',ShaderPrecisionPolicy(literal_overrides={1:'float64'}))[0].annotation.value=='float64'

def test_default_only_resolves_unstamped_floating_literals():
    p=ShaderPrecisionPolicy()
    assert p.literal_dtype(1,None,0.1)=='float32'
    assert p.literal_dtype(1,'int64',7) is None
    assert p.literal_dtype(1,None,7) is None
    assert p.literal_dtype(1,None,True) is None
    assert ShaderPrecisionPolicy(default_dtype='float64').literal_dtype(1,None,0.1)=='float64'

@pytest.mark.parametrize('kwargs',[{'default_dtype':'float16'},{'literal_overrides':{1:'int32'}}])
def test_unsupported_precision_is_rejected(kwargs):
    with pytest.raises(ValueError):ShaderPrecisionPolicy(**kwargs)

@pytest.mark.parametrize('dtype,complete',[('float32',True),('float64',False)])
def test_typed_shader_literals_reach_normal_backend_without_narrowing(dtype,complete):
    from pathlib import Path
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_webgpu_backend import emit_module
    body=source(dtype,ShaderPrecisionPolicy())
    function=ast.FunctionDef(name='literal',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    text=ast.unparse(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])))
    module,outputs,exports=lower_ast_source_to_ssa(text,'literal',name='shader_literal',extraction_contract=Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml',progress_bars=False)
    root=exports[0]
    assert outputs[root][0].dtype==dtype
    emitted=emit_module(module,outputs={root:outputs[root]})
    assert emitted.complete is complete
    if not complete:
        assert any('float64 has no WebGPU core equivalent' in s.reason for s in emitted.shortfalls)
