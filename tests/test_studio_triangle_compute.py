import json,subprocess,shutil,hashlib
import pytest
from examples.meowdoku.studio_triangle_compute import build,studio_source
from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa

@pytest.mark.skipif(shutil.which('node') is None,reason='Node executes original-law Wasm')
def test_original_studio_law_wasm_and_webgpu_artifacts(tmp_path):
    result,a,w=build(tmp_path,count=8)
    assert result.module.metadata['identity_book']
    manifest=json.loads((tmp_path/'studio_hit_tri.json').read_text())
    for kind,digest in manifest['artifacts'].items():
        assert hashlib.sha256((tmp_path/('studio_hit_tri.'+kind)).read_bytes()).hexdigest()==digest
    assert 'array<bool>' not in w.source
    assert 'array<u32>' in w.source
    assert 'bitcast<u32>(v_' not in w.source
    assert 'Relative float32 parallel test' in studio_source()
    # Includes backface, outside, parallel, behind, degenerate, miniature,
    # and accepted tolerance just outside the mathematical triangle edge.
    cases=[
      ([.25,.25,-3],[0,0,1],[0,0,0],[1,0,0],[0,1,0],1,3),
      ([.25,.25,3],[0,0,-1],[0,0,0],[1,0,0],[0,1,0],1,3),
      ([4,4,-3],[0,0,1],[0,0,0],[1,0,0],[0,1,0],0,0),
      ([.25,.25,-3],[1,0,0],[0,0,0],[1,0,0],[0,1,0],0,0),
      ([.25,.25,3],[0,0,1],[0,0,0],[1,0,0],[0,1,0],0,0),
      ([0,0,-3],[0,0,1],[0,0,0],[0,0,0],[0,1,0],0,0),
      ([.000025,.000025,-3],[0,0,1],[0,0,0],[.0001,0,0],[0,.0001,0],1,3),
      ([-.000005,.25,-3],[0,0,1],[0,0,0],[1,0,0],[0,1,0],1,3),
    ]
    script='''const fs=require('fs');WebAssembly.instantiate(fs.readFileSync(process.argv[1])).then(({instance:i})=>{let out=[];for(const c of JSON.parse(process.argv[2])){i.exports.hit_tri(...c.slice(0,5).flat(),16,24);out.push([Number(new BigInt64Array(i.exports.memory.buffer,16,1)[0]),new Float64Array(i.exports.memory.buffer,24,1)[0]]);}console.log(JSON.stringify(out));});'''
    actual=json.loads(subprocess.check_output(['node','-e',script,str(tmp_path/'studio_hit_tri.wasm'),json.dumps(cases)],text=True))
    for got,case in zip(actual,cases):
        assert got[0]==case[5]
        if got[0]:assert got[1]==pytest.approx(case[6],abs=1e-6)


def test_pure_frontend_refuses_loop():
    with pytest.raises(ValueError):
        lower_glsl_function_to_ssa('float loop(float x){for(int i=0;i<3;i++){x=x+1.0;}return x;}',function_name='loop')


@pytest.mark.parametrize('source',[
    'float f(float x){return float(1/2);}',
    'float f(float x){return float(-1);}',
    'float f(float x){return float(int(x));}',
    'float f(float x){if(x>0.0) return x;}',
    'float f(float x){if(x>0.0) {return x;}}',
])
def test_pure_frontend_refuses_unimplemented_semantics(source):
    with pytest.raises(ValueError):
        lower_glsl_function_to_ssa(source,function_name='f')


@pytest.mark.parametrize('source',[
    'float f(float x){\n#if SOMETHING\nreturn 1.0;\n#else\nreturn 2.0;\n#endif\n}',
    '#define EPS -1-2\nfloat f(float x){return x;}',
    '#define EPS 1e\nfloat f(float x){return x;}',
    '#undef EPS\nfloat f(float x){return x;}',
])
def test_pure_frontend_refuses_unhandled_preprocessor(source):
    with pytest.raises(ValueError):
        lower_glsl_function_to_ssa(source,function_name='f')
