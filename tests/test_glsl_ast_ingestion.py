import ast

from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast


def test_glsl_parser_can_build_inspectable_python_ast_without_glsl_ops():
    result = lower_glsl_source_to_ast(
        """
        uniform float x;
        uniform float lo;
        uniform float hi;
        uniform float alpha;
        out float color;
        void main() {
            float bounded = clamp(x, lo, hi);
            color = mix(bounded, lo, alpha);
        }
        """
    )

    assert result.complete, result.shortfall_report()
    dumped = ast.dump(result.module, include_attributes=False)
    assert "clamp" not in dumped
    assert "mix" not in dumped
    assert "IfExp" in dumped
    assert "BinOp" in dumped

    namespace = {}
    exec(compile(result.module, "<glsl-ast>", "exec"), namespace)
    assert namespace["main"](2.0, 0.0, 1.0, 0.25) == 0.75


def test_ast_path_uses_canonical_existing_call_names():
    result = lower_glsl_source_to_ast(
        """
        in float x;
        out float color;
        void main() { color = sin(x) + sqrt(x); }
        """
    )

    calls = [
        node.func.id
        for node in ast.walk(result.module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert result.complete, result.shortfall_report()
    assert sorted(calls) == ["sin", "sqrt"]


def test_ast_path_reports_webgl_only_calls_without_placeholder_nodes():
    result = lower_glsl_source_to_ast(
        """
        uniform float source;
        out float color;
        void main() { color = texture(source, 0.0); }
        """
    )

    assert not result.complete
    assert "texture" not in {
        node.func.id
        for node in ast.walk(result.module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

def test_existing_ast_reader_loops_and_continue():
    result = lower_glsl_source_to_ast("""uniform int n; out int answer;
    void main() { int total = 0;
      for (int i=0; i<n; ++i) { if (i == 2) continue; total += i; }
      answer = total;
    }""")
    assert result.complete, result.shortfall_report()
    ns = {}; exec(compile(result.module, '<source>', 'exec'), ns)
    assert ns['main'](5) == 8
    assert ns['main'](0) == 0


def test_existing_ast_reader_preserves_short_circuit_index_guard():
    result = lower_glsl_source_to_ast("""uniform int n; uniform int values[2]; out int answer;
    void main() { int x=0;
      if (n == 0 || values[n] == 3) { x=1; } else { x=2; }
      answer=x;
    }""")
    assert result.complete, result.shortfall_report()
    ns = {}; exec(compile(result.module, '<source>', 'exec'), ns)
    assert ns['main'](0, []) == 1
    assert ns['main'](1, [0, 0]) == 2


def test_counter_for_continue_does_not_rewrite_nested_loop_continue():
    result = lower_glsl_source_to_ast('''uniform int n; out int answer;
    void main() { int total=0;
      for(int i=0;i<n;i++) { int j=0; while(j<2){ j+=1; if(j==1) continue; total+=i; } }
      answer=total;
    }''')
    assert result.complete, result.shortfall_report()
    ns={}; exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main'](4)==6


def test_ast_reader_loop_reaches_sanctioned_source_compiler():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    result=lower_glsl_source_to_ast('''uniform int n; out int answer;
      void main(){int total=0;int k=0;while(k<n){total+=k;k+=1;}answer=total;}''')
    assert result.complete,result.shortfall_report()
    root=Path(__file__).resolve().parents[1]
    contract=ExtractionContract(root/'extraction_contracts/program_extraction.yaml').with_program_abi({
      'records':{},'bindings':[],'values':[{'function':'main','parameter':'n',
      'storage':'scalar','dtype':'int32','rank':0,'shape':[],'python_type':'int'}]
    }).with_execution_file(root/'extraction_contracts/vehicle_full_native_execution.yaml')
    module,outputs,exports=lower_ast_source_to_ssa(ast.unparse(result.module),'main',
      extraction_contract=contract,progress_bars=False)
    assert exports and all(name in module.functions for name in exports)
    assert any(instruction.op == 'CondBr'
      for fn in module.functions.values() for block in fn.blocks.values()
      for instruction in block.instrs)

def test_ast_reader_lowers_glsl_block_scopes():
    # The inner x is a different variable; the outer x is unchanged after
    # the block.  Two loops may each declare their own counter.
    result=lower_glsl_source_to_ast('''out int answer; void main(){
      int x=1; {int x=2; x=x+10;} int t=0;
      for(int i=0;i<3;++i){t+=i;} for(int i=0;i<2;++i){t+=i*10;}
      answer=x*100+t;}''')
    assert result.complete,result.shortfall_report()
    ns={};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main']()==113


def test_ast_reader_local_scalar_array_is_a_typed_list():
    result=lower_glsl_source_to_ast("""out float answer; void main(){
      float a[2]; a[0]=1.5; answer=a[0]+a[1];}""")
    assert result.complete,result.shortfall_report()
    declaration=next(node for node in ast.walk(result.module)
        if isinstance(node,ast.AnnAssign) and getattr(node.target,'id','')=='a')
    assert ast.unparse(declaration.annotation)=="list['float32']"
    ns={};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main']()==1.5


def test_ast_reader_vector_compare_reflect_refract_and_increments():
    import math
    result=lower_glsl_source_to_ast("""uniform float eta; out float answer;
      void main(){
        int stack[4]; int sp=0; stack[sp++]=7; stack[sp++]=9; int top=stack[--sp];
        vec3 i=normalize(vec3(1.0,-1.0,0.0)); vec3 n=vec3(0.0,1.0,0.0);
        vec3 r=reflect(i,n); vec3 t=refract(i,n,eta);
        bool below=all(lessThan(vec2(r.x,t.y),vec2(1.0,0.0)));
        answer=float(top)*1000.0+float(sp)*100.0+(below?10.0:0.0)+t.x+r.y;
      }""")
    assert result.complete,result.shortfall_report()
    ns={'__name__':__name__,'sqrt':math.sqrt};exec(compile(result.module,'<source>','exec'),ns)
    s=1/math.sqrt(2.0)
    def expect(eta):
        c=-s; k=1-eta*eta*(1-c*c)
        tx=0.0 if k<0 else eta*s; ty=0.0 if k<0 else eta*-s-(eta*c+math.sqrt(k))*1.0
        below=(s<1.0) and (ty<0.0)
        return 9*1000+1*100+(10.0 if below else 0.0)+tx+s
    for eta in (0.5,1.0/1.5,2.0):
        assert ns['main'](eta)==expect(eta)


def test_ast_reader_hex_literals_and_bitwise_assignments():
    result=lower_glsl_source_to_ast("""uniform uint s; out uint answer; void main(){
      uint x=s; x^=x*0x9e3779b9u; x&=0x00ffffffu; x|=0x10u; x<<=1u; answer=x;}""")
    assert result.complete,result.shortfall_report()
    ns={};exec(compile(result.module,'<source>','exec'),ns)
    x=3; x^=x*0x9e3779b9; x&=0x00ffffff; x|=0x10; x<<=1
    assert ns['main'](3)==x


def test_ast_reader_keeps_source_helpers_as_normal_calls():
    result=lower_glsl_source_to_ast('''
      int twice(int x){ return x*2; }
      uniform int n; out int answer;
      void main(){ answer=twice(n)+twice(3); }
    ''')
    assert result.complete,result.shortfall_report()
    assert [fn.name for fn in result.module.body]==['twice','main']
    ns={};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main'](4)==14


def test_ast_reader_inout_helper_copies_out_to_the_caller():
    # GLSL copies inout/out parameters back on return.  The helper returns
    # (result, outs...) and the call site unpacks them before the consumer.
    result=lower_glsl_source_to_ast('''
      bool update(inout float x){x+=1.0;return true;}
      void reset(out float y, in float z){y=z*2.0;}
      out float answer;void main(){float x=1.0;float y;
        bool ok=update(x);reset(y,x);answer=ok?x+y:0.0;}
    ''')
    assert result.complete,result.shortfall_report()
    ns={};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main']()==6.0


def test_ast_reader_vectors_and_structs_are_dataclass_records():
    result=lower_glsl_source_to_ast('''
      struct Ray { vec3 origin; vec3 direction; float t; };
      vec3 at(Ray r){ return r.origin + r.direction * r.t; }
      uniform float s; out float answer;
      void main(){
        Ray r = Ray(vec3(1.0, 2.0, 3.0), normalize(vec3(0.0, 3.0, 4.0)), s);
        vec3 p = at(r);
        vec2 q = p.zy;
        q *= 2.0;
        p.xz = q;
        answer = dot(p, vec3(1.0)) + length(cross(vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0)));
      }
    ''')
    assert result.complete,result.shortfall_report()
    records=[node.name for node in result.module.body if isinstance(node, ast.ClassDef)]
    assert set(records)=={'vec3','Ray','vec2'}
    dumped=ast.dump(result.module, include_attributes=False)
    for spelling in ('normalize','cross','dot',"'zy'","'xz'"):
        assert spelling not in dumped
    import math
    # A namespace without __name__ would create these record classes under
    # the ``builtins`` module, leaking them into later compiler runs.
    ns={'__name__':__name__,'sqrt': math.sqrt};exec(compile(result.module,'<source>','exec'),ns)
    # r.t = 5: p = (1, 2+3, 3+4) = (1, 5, 7); q = (7, 5) * 2 = (14, 10);
    # p.xz = q -> (14, 5, 10); dot with 1s = 29; |x cross y| = 1.
    assert ns['main'](5.0)==30.0
    fields={node.name:[(f.target.id, ast.unparse(f.annotation)) for f in node.body]
            for node in result.module.body if isinstance(node, ast.ClassDef)}
    assert fields['vec3']==[('x',"'float32'"),('y',"'float32'"),('z',"'float32'")]
    assert fields['Ray']==[('origin','vec3'),('direction','vec3'),('t',"'float32'")]


def test_ast_reader_vector_operand_is_evaluated_once():
    # A per-component expansion must not repeat its operand's evaluation.
    result=lower_glsl_source_to_ast('''
      vec3 make(float a){ return vec3(a, a + 1.0, a + 2.0); }
      uniform float s; out float answer;
      void main(){ vec3 v = make(s) * 2.0; answer = v.x + v.y + v.z; }
    ''')
    assert result.complete,result.shortfall_report()
    main=result.module.body[-1]
    calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and getattr(n.func,'id','')=='make']
    assert len(calls)==1
    ns={'__name__':__name__};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main'](1.0)==12.0


def test_ast_reader_conditional_operand_statements_run_only_when_selected():
    # make() counts its calls through an inout parameter.  GLSL evaluates
    # the right of && and one arm of ?: only when selected, so the count
    # shows exactly which operands ran.
    result=lower_glsl_source_to_ast('''
      vec3 make(float a, inout int calls){ calls += 1; return vec3(a); }
      uniform float s; out float answer;
      void main(){
        int calls = 0;
        bool hit = s > 0.0 && dot(make(s, calls), vec3(1.0)) > 1.0;
        float v = hit ? make(2.0, calls).x : 0.5;
        answer = v * 10.0 + float(calls);
      }
    ''')
    assert result.complete,result.shortfall_report()
    ns={'__name__':__name__};exec(compile(result.module,'<source>','exec'),ns)
    assert ns['main'](-1.0)==5.0    # neither make() ran
    assert ns['main'](0.2)==6.0     # && ran make(); dot = 0.6, not a hit
    assert ns['main'](1.0)==22.0    # both ran; v = 2

def test_ast_reader_preserves_declared_scalar_width_annotations():
    result=lower_glsl_source_to_ast('''uniform float value;out float result;
      void main(){float local=value*2.0;int counter=1;result=local;}''')
    assert result.complete,result.shortfall_report()
    fn=result.module.body[-1]
    assert fn.args.args[0].annotation.value=='float32'
    annotations={node.target.id:node.annotation.value for node in ast.walk(fn)
      if isinstance(node,ast.AnnAssign) and not node.target.id.startswith('__glsl_literal_')}
    assert annotations=={'local':'float32','counter':'i32'}


def test_source_annotations_resolve_through_shared_layout_authority():
    from src.transmogrifier.dtype_layout import source_annotation_dtype
    assert source_annotation_dtype('int')=='int64'
    assert source_annotation_dtype('float')=='float64'
    assert source_annotation_dtype("'float32'")=='float32'
    assert source_annotation_dtype("'i32'")=='int32'
    assert source_annotation_dtype('uint32')=='uint32'
    assert source_annotation_dtype('list[float]') is None

def test_glsl_float_literals_and_arithmetic_stay_f32_in_sanctioned_ssa():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    source='''uniform float value;out float answer;
      void main(){float x=value*2.0;answer=x;}'''
    result=lower_glsl_source_to_ast(source)
    assert result.complete,result.shortfall_report()
    root=Path(__file__).resolve().parents[1]
    contract=ExtractionContract(root/'extraction_contracts/program_extraction.yaml').with_program_abi({
      'records':{},'bindings':[],'values':[{'function':'main','parameter':'value',
      'storage':'scalar','dtype':'float32','rank':0,'shape':[],'python_type':'float'}]
    }).with_execution_file(root/'extraction_contracts/vehicle_full_native_execution.yaml')
    module,_,_=lower_ast_source_to_ssa(source,'main',source_language='glsl',
      extraction_contract=contract,progress_bars=False)
    products=[ins for fn in module.functions.values() for block in fn.blocks.values()
      for ins in block.instrs if ins.op=='Mul']
    assert products
    assert all(ins.res.dtype=='float32' and all(arg.dtype=='float32' for arg in ins.args)
      for ins in products)
    assert module.metadata["glsl_source_ingestion"]["source_sha256"]
    from src.compiler.concordance_declarations import SOURCE_SPAN
    book=module.metadata["identity_book"]
    page=book.page(SOURCE_SPAN)
    token_rows=[row for row in page.rows() if page.latest(row).kind=="GLSLToken"]
    assert token_rows
    assert any(tuple(book.edges_out_of(book.latest_ref(SOURCE_SPAN,row))) for row in token_rows)


def test_glsl_source_evidence_keeps_token_declarations_after_comments():
    from src.compiler.glsl_ast_ingestion import retain_glsl_source_evidence
    from src.transmogrifier.graph.graph_express2 import SourceDefinition
    source='uniform float x;out float y;void main(){ /* note */ y=x+2.0;}'
    result=lower_glsl_source_to_ast(source)
    assert result.complete,result.shortfall_report()
    retain_glsl_source_evidence(result.module,source)
    constants=[n for n in ast.walk(result.module) if isinstance(n,ast.Constant) and n.value==2.0]
    assert constants
    assert all(isinstance(cell,SourceDefinition) for cell in constants[0]._turing_source_cells)
    assert constants[0]._turing_source_cells[-1].fact.kind=='GLSLToken'


def test_glsl_source_correspondence_rejects_changed_literal_location():
    import pytest
    from src.compiler.glsl_ast_ingestion import retain_glsl_source_evidence
    source='out float y;void main(){y=2.0;}'
    result=lower_glsl_source_to_ast(source)
    node=next(n for n in ast.walk(result.module) if isinstance(n,ast.Constant) and n.value==2.0)
    node._glsl_source_location=(1,1,'2.0')
    with pytest.raises(ValueError,match='correspondence'):
        retain_glsl_source_evidence(result.module,source)
