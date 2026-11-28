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


def _declared_literals(function):
    return {node.target.id: (node.annotation.value,
                             ast.literal_eval(ast.unparse(node.value)))
            for node in function.body
            if isinstance(node, ast.AnnAssign)
            and node.target.id.startswith('__glsl_literal_')}


def test_negated_int_literal_is_one_declared_int32_literal_unsigned_keeps_operator():
    result = lower_glsl_source_to_ast('''uniform float x;out int a;out uint b;
      void main(){a = x > 0.0 ? 1 : -1; b = -1u;}''')
    assert result.complete, result.shortfall_report()
    fn = result.module.body[-1]
    literals = _declared_literals(fn)
    selected = next(node for node in ast.walk(fn) if isinstance(node, ast.IfExp))
    arms = [selected.body, selected.orelse]
    # Both arms of ``? 1 : -1`` are declared int32 literals (no UnaryOp).
    assert all(isinstance(arm, ast.Name) for arm in arms)
    assert [literals[arm.id] for arm in arms] == [
        ('int32', 1), ('int32', -1)]
    # ``-1u`` wraps: the unsigned literal keeps its negation operator.
    negations = [node for node in ast.walk(fn)
                 if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                 and isinstance(node.operand, ast.Name)]
    assert len(negations) == 1
    assert literals[negations[0].operand.id] == ('uint32', 1)


def test_scalar_conversion_and_return_are_held_at_the_glsl_width():
    result = lower_glsl_source_to_ast('''uniform float x;out int y;
      int pick(float v){ if (v < 0.0) return -1; return int(v) - 2; }
      void main(){ y = pick(x); }''')
    assert result.complete, result.shortfall_report()
    pick = next(node for node in result.module.body
                if isinstance(node, ast.FunctionDef) and node.name == 'pick')
    held = {node.target.id: (node.annotation.value, ast.unparse(node.value))
            for node in ast.walk(pick) if isinstance(node, ast.AnnAssign)
            and not node.target.id.startswith('__glsl_literal_')}
    # ``int(v)`` is an int32 conversion, not the Python int class.
    assert ('i32', 'int(v)') in held.values()
    # Every returned value is a temporary declared with the result type.
    returns = [node for node in ast.walk(pick) if isinstance(node, ast.Return)]
    assert len(returns) == 2
    assert all(isinstance(node.value, ast.Name) and held[node.value.id][0] == 'i32'
               for node in returns)


def test_declared_int_return_and_folded_constants_keep_glsl_widths_on_llvm():
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    # The Studio response shader's int widths: a declared ``int`` return
    # computed as ``int(...) - 2`` or ``-1``, its caller's ``int`` local, a
    # conditional whose arm folds (``2 * 3``) beside ``-1``, and a folded
    # float constant.  Before: "authored local scalar conflicts with an
    # existing physical ABI" / "conditional union arm changes its physical
    # ABI" (int32 against int64).
    source = '''#version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Vals { float vals[]; };
    layout(std430, binding = 1) buffer Outs { int outs[]; };
    int pick(float x) { if (x < 0.0) return -1; return int(x) - 2; }
    void main() {
        int m = pick(vals[0]);
        int s = m + (vals[0] > 0.0 ? 2 * 3 : -1);
        float f = vals[0] * (0.5 * 4.0);
        outs[0] = s + int(f);
    }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    root = Path(__file__).resolve().parents[1]
    span = {'function': '*', 'storage': 'span', 'rank': 1,
            'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {**span, 'parameter': 'vals', 'dtype': 'float32'},
        {**span, 'parameter': 'outs', 'dtype': 'int32'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _outputs, exports = lower_ast_source_to_ssa(
        ast.unparse(result.module), 'main', extraction_contract=contract,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False)
    phis = [ins for fn in module.functions.values() for block in fn.blocks.values()
            for ins in block.instrs if ins.op == 'Phi']
    assert phis and all(arg.dtype == 'int32' for ins in phis for arg in ins.args)
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    for x, expected in ((5.7, 3 + 6 + 11), (-1.0, -1 - 1 - 2), (2.0, 0 + 6 + 4)):
        vals = np.array([x], dtype=np.float32)
        outs = np.zeros(1, dtype=np.int32)
        prepare_artifact_execution(
            artifact, {names['vals']: vals, names['outs']: outs}).run()
        assert int(outs[0]) == expected


def test_mixed_width_atomics_fan_one_body_per_element_type_at_depth_zero():
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.work_contract import (
        active_contract, scoped_work_contract, with_expansion_policy,
    )
    # The Studio response shader at expansion depth 0: ``atomicAdd`` on a
    # ``uint`` buffer and on an ``int`` buffer are both
    # ``AbstractTensor.atomic_add``.  Before: the one generic body received a
    # uint32 and an int32 span for ``self`` ("native union alternative has
    # conflicting physical ProgramABI"), and even a single call lost its
    # declared literal widths (the shell's dict folded ``index``/``value``
    # with no ``planner_specialization`` row: "exact integer domain").
    # Now: one body per element type (the fanned copy), widths kept.
    source = '''#version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer UWords { uint uwords[]; };
    layout(std430, binding = 1) buffer IWords { int iwords[]; };
    void main() {
        uint a = atomicAdd(uwords[0], 1u);
        int b = atomicAdd(iwords[0], 2);
        uwords[1] = a;
        iwords[1] = b;
    }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    root = Path(__file__).resolve().parents[1]
    span = {'function': '*', 'storage': 'span', 'rank': 1,
            'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {**span, 'parameter': 'uwords', 'dtype': 'uint32'},
        {**span, 'parameter': 'iwords', 'dtype': 'int32'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    depth_zero = with_expansion_policy(active_contract(), {
        'inline_depth': 0, 'custom_version_depth': 0, 'unroll_depth': 0,
    })
    with scoped_work_contract(depth_zero):
        module, _outputs, exports = lower_ast_source_to_ssa(
            ast.unparse(result.module), 'main', extraction_contract=contract,
            python_bindings={'AbstractTensor': AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            runtime_closure_only=True, progress_bars=False)
    bodies = [name for name in module.functions
              if 'atomic_add' in name and '__planned_region' not in name]
    assert len(bodies) == 2, bodies
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    uwords = np.array([0xFFFFFFFF, 0], dtype=np.uint32)
    iwords = np.array([5, 0], dtype=np.int32)
    prepare_artifact_execution(
        artifact, {names['uwords']: uwords, names['iwords']: iwords}).run()
    assert uwords.tolist() == [0, 0xFFFFFFFF]
    assert iwords.tolist() == [7, 5]


def test_uint_conditional_compared_to_uint_literal_stays_uint32_on_llvm():
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    # The Studio response shader's studio_response_copy termination code:
    # ``(... ? 2u : ...) == 0u ? 8u : ...``.  Before: the conditional's
    # contract named only its class (builtins.int), whose int64 default
    # was seeded over its uint32 arms, so the comparison's region was fed
    # int64 from the uint32 Phi ("incompatible final physical call
    # inputs"; in the whole shader "native union alternative has
    # conflicting physical ProgramABI" uint32 != int64).
    source = '''#version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Vals { float vals[]; };
    layout(std430, binding = 1) buffer Outs { uint outs[]; };
    void main() {
        float z = vals[0];
        outs[0] = (z < 0.0 ? 2u : 3u) == 2u ? 8u : 1u;
    }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    root = Path(__file__).resolve().parents[1]
    span = {'function': '*', 'storage': 'span', 'rank': 1,
            'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {**span, 'parameter': 'vals', 'dtype': 'float32'},
        {**span, 'parameter': 'outs', 'dtype': 'uint32'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _outputs, exports = lower_ast_source_to_ssa(
        ast.unparse(result.module), 'main', extraction_contract=contract,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False)
    phis = [ins for fn in module.functions.values() for block in fn.blocks.values()
            for ins in block.instrs if ins.op == 'Phi']
    assert phis and all(arg.dtype == 'uint32' for ins in phis for arg in ins.args)
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    for z, expected in ((-1.0, 8), (0.5, 1), (float('nan'), 1)):
        vals = np.array([z], dtype=np.float32)
        outs = np.zeros(1, dtype=np.uint32)
        prepare_artifact_execution(
            artifact, {names['vals']: vals, names['outs']: outs}).run()
        assert int(outs[0]) == expected


def test_float_returned_through_inout_tuple_stays_float32_on_llvm():
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    # The Studio response shader's rand01 -> cosine_dir: a float result of an
    # inout helper (returned with its uint state) feeds ``max(0.0, 1.0 - u)``.
    # Before: ``float(...)`` was described float64 by its class, the formal
    # ``u: 'float32'`` and ``1.0 - u`` followed, and the conditional's arms
    # refused ("conditional union arm changes its physical ABI").
    source = '''#version 430
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Seeds { uint seeds[]; };
    layout(std430, binding = 1) buffer Outs { float outs[]; };
    float rand01(inout uint s) {
        s = s * 1664525u + 1013904223u;
        return float(s & 0x00ffffffu) / 16777215.0;
    }
    float f(float u) { return max(0.0, 1.0 - u); }
    void main() {
        uint rng = seeds[0];
        outs[0] = f(rand01(rng));
        seeds[0] = rng;
    }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    root = Path(__file__).resolve().parents[1]
    span = {'function': '*', 'storage': 'span', 'rank': 1,
            'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {**span, 'parameter': 'seeds', 'dtype': 'uint32'},
        {**span, 'parameter': 'outs', 'dtype': 'float32'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _outputs, exports = lower_ast_source_to_ssa(
        ast.unparse(result.module), 'main', extraction_contract=contract,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False)
    widths = {getattr(value, 'dtype', None)
              for fn in module.functions.values() for block in fn.blocks.values()
              for ins in block.instrs for value in (*ins.args, ins.res)}
    assert 'float64' not in widths
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    for seed in (0, 12345, 4000000000):
        seeds = np.array([seed], dtype=np.uint32)
        outs = np.zeros(1, dtype=np.float32)
        prepare_artifact_execution(
            artifact, {names['seeds']: seeds, names['outs']: outs}).run()
        state = np.uint32((seed * 1664525 + 1013904223) % (1 << 32))
        unit = np.float32(state & 0x00ffffff) / np.float32(16777215.0)
        expected = max(np.float32(0.0), np.float32(1.0) - unit)
        assert int(seeds[0]) == int(state)
        # Within one float32 ulp: the LLVM lane evaluates a float32 region
        # in double and rounds at the store.
        assert abs(float(outs[0]) - float(expected)) <= float(np.spacing(expected))


def test_declared_float32_formal_converts_a_float64_actual_on_llvm():
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    # The Studio response shader's cosine_dir(float u, ...): a declared
    # ``u: 'float32'`` formal reached by a float64 actual.  Before: the
    # formal took the actual's float64, so ``1.0 - u`` was float64 beside the
    # float32 literal arm of ``max(0.0, 1.0 - u)`` ("conditional union arm
    # changes its physical ABI").  The declared width governs; the actual
    # converts at the call.
    source = '''
from src.common.tensors.abstraction import AbstractTensor
def f(u: 'float32') -> 'float32':
    zero: 'float32' = 0.0
    one: 'float32' = 1.0
    held: 'float32' = zero if zero > one - u else one - u
    return held
def main(x: float, out: AbstractTensor):
    out[0] = f(x)
'''
    root = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        root / 'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({'records': {}, 'bindings': [], 'values': [
        {'function': 'main', 'parameter': 'x', 'storage': 'scalar',
         'dtype': 'float64', 'rank': 0, 'shape': [], 'python_type': 'builtins.float'},
        {'function': '*', 'parameter': 'out', 'storage': 'span', 'dtype': 'float32',
         'rank': 1, 'python_type': 'src.common.tensors.abstraction.AbstractTensor'},
    ]}).with_execution_file(
        root / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, 'main', extraction_contract=contract,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress_bars=False)
    callee_widths = {getattr(value, 'dtype', None)
                     for name, fn in module.functions.items() if '__f__' in name
                     for block in fn.blocks.values()
                     for ins in block.instrs for value in (*ins.args, ins.res)}
    assert 'float64' not in callee_widths
    entry = exports[0]
    names = dict(module.functions[entry].metadata['parameter_names'])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    for x in (0.25, 1.5, 1.0 / 3.0):
        out = np.zeros(1, dtype=np.float32)
        prepare_artifact_execution(artifact, {names['x']: x, names['out']: out}).run()
        u = np.float32(x)
        expected = max(np.float32(0.0), np.float32(1.0) - u)
        assert float(out[0]) == float(expected)
