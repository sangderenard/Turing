"""Pure GLSL function ingestion into existing scalar SSA.

Vector scalarization and predicated early returns add no numerical opcodes.
Only side-effect-free arithmetic functions are accepted; storage, loops and
unknown statements fail closed. Predication may evaluate inactive arithmetic,
but inactive values cannot escape through the selected return/out values.
"""
import re
from .glsl_source_ingestion import (_SSABuilder,_ExpressionParser,_Token,_tokenize,_without_comments,GLSLSSALoweringResult)
from .glsl_source_tables import GLSL_BINARY_TO_SSA
from ..transmogrifier.ssa_registry import Handler
from ..transmogrifier.ssa import Instr

class _VectorBuilder(_SSABuilder):
    def emit(self,handler,args=(),**kw):
        vectors=[a for a in args if isinstance(a,tuple)]
        if vectors:
            n=len(vectors[0])
            if any(len(a)!=n for a in vectors):raise ValueError('GLSL vector width mismatch')
            return tuple(self.emit(handler,[a[i] if isinstance(a,tuple) else a for a in args],**kw) for i in range(n))
        kw.setdefault('dtype', 'float32')
        if kw.get('dtype') is None: kw['dtype']='float32'
        return super().emit(handler,args,**kw)
    def call(self,token,args):
        helper=getattr(self,'pure_helpers',{}).get(token.text)
        if helper is not None:
            names,expression=helper
            if len(names)!=len(args):raise ValueError('Pure helper arity mismatch')
            for kind,arg in zip(self.helper_types[token.text],args):
                if (kind=='float' and isinstance(arg,tuple)) or (kind!='float' and (not isinstance(arg,tuple) or len(arg)!=int(kind[-1]))):raise ValueError('Pure helper argument type mismatch')
            if token.text in getattr(self,'helper_stack',[]):raise ValueError('Recursive helper unsupported')
            prior=self.bindings.copy();self.helper_stack=getattr(self,'helper_stack',[])+[token.text]
            try:
                self.bindings=dict(zip(names,args))
                for name,arg in zip(names,args):
                    if isinstance(arg,tuple):
                        for axis,value in zip('xyzw',arg):
                            scalar=name+'__helper_'+axis
                            self.bindings[scalar]=value
                            expression=re.sub(r'\b'+re.escape(name)+r'\.'+axis+r'\b',scalar,expression)
                return _ExpressionParser(_tokenize(expression,start_line=1),self).parse()
            finally:
                self.bindings=prior;self.helper_stack.pop()
        if token.text in {'int','uint','double','bool'}:raise ValueError('Non-float scalar conversion is outside this pure-function subset')
        if token.text=='abs':return self.emit(Handler.Abs,args)
        if token.text=='length' and isinstance(args[0],tuple):
            d=self.call(_Token('dot',1,1),[args[0],args[0]])
            value=self.fresh('float32');self.instructions.append(Instr('Sqrt',[d],value));return value
        if token.text in {'dot','cross'}:
            if len(args)!=2 or not all(isinstance(a,tuple) for a in args) or len(args[0])!=len(args[1]):raise ValueError('Expected equal vectors')
            a,b=args
            if token.text=='cross':
                if len(a)!=3:raise ValueError('cross requires vec3')
                return tuple(self.emit(Handler.Sub,[self.emit(Handler.Mul,[a[j],b[k]]),self.emit(Handler.Mul,[a[k],b[j]])]) for j,k in [(1,2),(2,0),(0,1)])
            products=[self.emit(Handler.Mul,[x,y]) for x,y in zip(a,b)];total=products[0]
            for p in products[1:]:total=self.emit(Handler.Add,[total,p])
            return total
        return super().call(token,args)

def lower_glsl_function_to_ssa(source,*,function_name, struct_fields=None, pure_helpers=None):
    """Lower a named pure float function with scalar/vec2-4 inputs and out floats.

    Source is the actual original function plus numeric #defines. Unsupported
    syntax raises ValueError rather than publishing a partial executable.
    """
    clean=_without_comments(source)
    defines={}
    number=r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fF]?'
    uncommented=re.sub(r'/\*.*?\*/','',source,flags=re.S)
    uncommented=re.sub(r'//[^\n]*','',uncommented)
    for directive in re.findall(r'^[ \t]*#[^\n]*',uncommented,re.M):
        m=re.fullmatch(r'\s*#define\s+(\w+)\s+('+number+r')\s*',directive)
        if not m or m.group(1) in defines:raise ValueError('Only unique numeric #defines are supported')
        defines[m.group(1)]=m.group(2)
    clean=re.sub(r'^\s*#.*$','',clean,flags=re.M)
    for name,value in defines.items():clean=re.sub(r'\b'+name+r'\b',value,clean)
    return_kind='bool' if re.search(r'\bbool\s+'+re.escape(function_name)+r'\s*\(',clean) else 'float32'
    if struct_fields:
        for kind,fields in struct_fields.items():
            for m in list(re.finditer(r'\b'+re.escape(kind)+r'\s+(\w+)',clean)):
                variable=m.group(1)
                clean=clean.replace(m.group(0),', '.join(f'{ft} {variable}_{fn}' for fn,ft in fields.items()))
                for fn in fields:clean=clean.replace(variable+'.'+fn+'.xyz',variable+'_'+fn)
    match=re.search(r'\b(?:float|bool)\s+'+re.escape(function_name)+r'\s*\(([^)]*)\)\s*\{',clean)
    if not match:raise ValueError('Expected named float function')
    b=_VectorBuilder(function_name);outs=[]
    b.pure_helpers={};b.helper_types={}
    for helper_name,helper_source in (pure_helpers or {}).items():
        h=_without_comments(helper_source).strip()
        hm=re.fullmatch(r'float\s+'+re.escape(helper_name)+r'\s*\(([^)]*)\)\s*\{\s*return\s+([^;{}]+);\s*\}',h)
        if not hm:raise ValueError('Only single-expression pure float helpers supported')
        names=[];kinds=[]
        for parameter in hm.group(1).split(','):
            pm=re.fullmatch(r'\s*(float|vec[234])\s+(\w+)\s*',parameter)
            if not pm:raise ValueError('Unsupported pure helper parameter')
            names.append(pm.group(2));kinds.append(pm.group(1))
        tokens=_tokenize(hm.group(2),start_line=1)
        if any(re.fullmatch(r'\d+[uU]?',t.text) for t in tokens):raise ValueError('Integer helper arithmetic unsupported')
        b.pure_helpers[helper_name]=(names,hm.group(2));b.helper_types[helper_name]=kinds
    for part in match.group(1).split(','):
        m=re.fullmatch(r'\s*(?:(in|out|inout)\s+)?(float|vec[234])\s+(\w+)\s*',part)
        if not m:raise ValueError('Unsupported function parameter '+part)
        qualifier,kind,name=m.groups()
        if qualifier in {'out','inout'}:
            if kind!='float':raise ValueError('Only scalar out parameters supported')
            outs.append(name)
            if qualifier=='inout':b.argument(name,'float')
            else:b.bindings[name]=b.constant(_Token('0.0',1,1))
        elif kind=='float':b.argument(name,'float')
        else:b.bindings[name]=tuple(b.argument(name+'_'+axis,'float') for axis in 'xyzw'[:int(kind[-1])])
    body=clean[match.end():];depth=1;end=0
    for end,ch in enumerate(body):
        depth+=(ch=='{')-(ch=='}')
        if depth==0:break
    if depth:raise ValueError('Unclosed function')
    body=body[:end]
    true=b.constant(_Token('true',1,1));false=b.constant(_Token('false',1,1));active=true
    returned=b.constant(_Token('false' if return_kind=='bool' else '0.0',1,1))
    def expr(s):
        tokens=_tokenize(s.strip(),start_line=1)
        if any(re.fullmatch(r'\d+[uU]?',t.text) for t in tokens):raise ValueError('Integer arithmetic is outside this float-only GLSL subset')
        return _ExpressionParser(tokens,b).parse()
    def select(cond,x,y):return b.emit(Handler.Select,[cond,x,y],dtype=x.dtype if not isinstance(x,tuple) else 'float32')
    def statement(s,mask):
        nonlocal active,returned
        s=s.strip()
        if s.startswith('return '):
            returned=select(mask,expr(s[7:]),returned)
            active=b.emit(Handler.LAnd,[active,b.emit(Handler.LNot,[mask],dtype='bool')],dtype='bool');return
        s=re.sub(r'^(float|vec[234])\s+','',s)
        bits=s.split('=')
        if len(bits)<2 or any(not re.fullmatch(r'\s*\w+\s*',n) for n in bits[:-1]):raise ValueError('Unsupported pure assignment '+s)
        value=expr(bits[-1])
        for name in reversed(bits[:-1]):
            name=name.strip();prior=b.bindings.get(name)
            # Only top-level locals or existing bindings in terminating guards.
            if mask is not active and prior is None:raise ValueError('Guard-local declarations are unsupported')
            b.bindings[name]=value if prior is None else select(mask,value,prior)
    pos=0;last_top_return=False
    while pos<len(body):
        while pos<len(body) and body[pos].isspace():pos+=1
        if pos>=len(body):break
        if body.startswith('if',pos) and re.match(r'if\s*\(',body[pos:]):
            last_top_return=False
            start=body.index('(',pos)+1;depth=1;i=start
            while depth:
                depth+=(body[i]=='(')-(body[i]==')');i+=1
            condition=expr(body[start:i-1]);mask=b.emit(Handler.LAnd,[active,condition],dtype='bool')
            while body[i].isspace():i+=1
            if body[i]=='{':
                end=body.find('}',i)
                if end<0 or '{' in body[i+1:end]:raise ValueError('Nested blocks unsupported')
                statements=body[i+1:end].split(';');pos=end+1
            else:
                end=body.index(';',i);statements=[body[i:end]];pos=end+1
            statements=[s.strip() for s in statements if s.strip()]
            if not statements or not statements[-1].startswith("return ") or any(s.startswith("return ") for s in statements[:-1]):raise ValueError("Only terminating return guards supported")
            for s in statements:statement(s,mask)
        else:
            end=body.index(';',pos);text=body[pos:end].strip();statement(text,active);last_top_return=text.startswith('return ');pos=end+1
    if not last_top_return:raise ValueError('A final unconditional return is required')
    b.outputs=[('return',returned)]+[(name,b.bindings[name]) for name in outs]
    module=b.finish()
    _publish_source(module,source+"\n"+"\n".join((pure_helpers or {}).values()),function_name)
    return GLSLSSALoweringResult(module,())



def _publish_source(module,source,name):
    # Real front-end provenance: the original GLSL text owns every generated
    # scalar operation. This is not a hand-authored-SSA fixture or replacement law.
    from .identity_concordance import Novel, Derived, Mode, begin_identity_book, end_identity_book
    from .concordance_declarations import (SSA_VALUE, SSAValueFact, SSAValueOrigin, GLSL_PURE_FUNCTION_SOURCE, GLSLPureFunctionSource, GLSL_FUNCTION_INGESTION, AUTHORED_GLSL_FUNCTION, SSA_BLOCK, SSABlockKind)
    from .ssa_record_return_state import function_scope_of
    book,token=begin_identity_book()
    try:
        f=module.functions[name];scope=function_scope_of(f)
        root=book.post(GLSL_PURE_FUNCTION_SOURCE,(scope,),GLSLPureFunctionSource(name,source),stage=GLSL_FUNCTION_INGESTION,provenance=Novel(AUTHORED_GLSL_FUNCTION,()),mode=Mode.CONCORD)
        refs={}
        for a in f.args:
            refs[a.id]=book.post(SSA_VALUE,(scope,a.id),SSAValueFact(a.dtype,tuple(a.shape),SSAValueOrigin.MINTED),stage=GLSL_FUNCTION_INGESTION,provenance=Derived((root,)),mode=Mode.CONCORD)
        for label,block in f.blocks.items():
            book.post(SSA_BLOCK,(scope,name,label),SSABlockKind.ENTRY,stage=GLSL_FUNCTION_INGESTION,provenance=Derived((root,)),mode=Mode.CONCORD)
            for i in block.instrs:
                if i.res is not None:
                    v=i.res
                    refs[v.id]=book.post(SSA_VALUE,(scope,v.id),SSAValueFact(v.dtype,tuple(v.shape),SSAValueOrigin.MINTED),stage=GLSL_FUNCTION_INGESTION,provenance=Derived(tuple([root]+[refs[a.id] for a in i.args])),mode=Mode.CONCORD)
        module.metadata['identity_book']=book
    finally:end_identity_book(token)


