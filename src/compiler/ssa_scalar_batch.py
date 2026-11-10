"""Counted, single-entry batch invocation of an existing scalar SSA function.

No numerical law is copied. Each loop iteration calls the original function
through its declared result identities. Columns are typed spans, not JS calls.
"""
from .identity_concordance import Derived, Mode
from .concordance_declarations import (SCALAR_BATCH_CONTRACT, SCALAR_BATCH_LOWERING,
    ScalarBatchContract, SSA_VALUE, SSAValueFact, SSAValueOrigin, SSA_BLOCK,
    SSABlockKind, SSA_CONTROL_EDGE)
from .ssa_record_return_state import function_scope_of
from ..transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function


def add_scalar_batch(module, callee_name, *, capacity, name=None, shared_argument_ids=()):
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('Batch capacity must be positive')
    callee=module.functions[callee_name]
    name=name or callee_name+'_batch'
    if name in module.functions: raise ValueError('Batch name already exists')
    returns=[i.args for b in callee.blocks.values() for i in b.instrs if i.op=='Ret']
    if len(returns)!=1: raise ValueError('Batch wrapper requires one declared return')
    outputs=returns[0]
    shared=set(shared_argument_ids)
    if not shared <= {v.id for v in callee.args}: raise ValueError('Unknown shared formal identity')
    per_item=[v for v in callee.args if v.id not in shared]
    if any(v.shape or v.dtype not in {'float32','float64','i32','int32','bool','int64'} for v in [*per_item,*outputs]):
        raise ValueError('Batch wrapper accepts only declared scalar numerical values')
    book=module.metadata['identity_book'];scope_callee=function_scope_of(callee)
    # Callee value cells are the authoritative contract, including return IDs.
    refs=[book.current_ref(SSA_VALUE,(scope_callee,v.id)) for v in [*callee.args,*outputs]]
    n=0
    def fresh(dtype,shape=(),span=False):
        nonlocal n
        v=SSAValue(n,dtype,shape,accounting={'physical_dtype':dtype,**({'program_abi_storage':'span','program_abi_rank':1} if span else {})});n+=1;return v
    def storage_dtype(v): return 'int32' if v.dtype in {'bool','i32'} else v.dtype
    count=fresh('int32')
    inputs=[]
    for v in callee.args:
        if v.id in shared:
            clone=fresh(v.dtype,tuple(v.shape));clone.accounting=dict(v.accounting or {})
            inputs.append(clone)
        else: inputs.append(fresh(storage_dtype(v),(capacity,),True))
    destinations=[fresh(storage_dtype(v),(capacity,),True) for v in outputs]
    zero=fresh('int32');one=fresh('int32');limit=fresh('int32');valid=fresh('bool');within=fresh('bool');allowed=fresh('bool')
    index=fresh('int32');advanced=fresh('int32');condition=fresh('bool')
    entry=[Instr('Const',[],zero,attributes={'value':0}),Instr('Const',[],one,attributes={'value':1}),Instr('Const',[],limit,attributes={'value':capacity}),Instr('Ge',[count,zero],valid),Instr('Le',[count,limit],within),Instr('LAnd',[valid,within],allowed),Instr('CondBr',[allowed],None,attributes={'true_target':'header','false_target':'exit'})]
    header=[Instr('Phi',[zero,advanced],index,attributes={'incoming_blocks':('entry','body')}),Instr('Lt',[index,count],condition),Instr('CondBr',[condition],None,attributes={'true_target':'body','false_target':'exit'})]
    body=[];actuals=[]
    for formal,column in zip(callee.args,inputs):
        if formal.id in shared:
            actuals.append(column);continue
        p=fresh('ptr');v=fresh(storage_dtype(formal))
        body += [Instr('GetElementPtr',[column,index],p),Instr('Load',[p],v)]
        actuals.append(v)
    aggregate=fresh('ssa.aggregate')
    body.append(Instr('Call',actuals,aggregate,attributes={'callee':callee_name,'output_ids':tuple(v.id for v in outputs)}))
    for position,(original,column) in enumerate(zip(outputs,destinations)):
        j=fresh('int32');p=fresh('ptr');v=fresh(storage_dtype(original));q=fresh('ptr')
        body += [Instr('Const',[],j,attributes={'value':position}),Instr('GetElementPtr',[aggregate,j],p,attributes={'source_output_id':original.id}),Instr('Load',[p],v),Instr('GetElementPtr',[column,index],q),Instr('Store',[v,q],None)]
    body += [Instr('Add',[index,one],advanced),Instr('Br',[],None,attributes={'target':'header'})]
    blocks={'entry':BasicBlock('entry',entry),'header':BasicBlock('header',header),'body':BasicBlock('body',body),'exit':BasicBlock('exit',[Instr('Ret',[],None)])}
    f=Function(name,[count,*inputs,*destinations],blocks);scope=function_scope_of(f)
    root=book.post(SCALAR_BATCH_CONTRACT,(scope,),ScalarBatchContract(callee_name,capacity,tuple(v.id for v in callee.args),tuple(v.id for v in outputs),tuple(sorted(shared))),stage=SCALAR_BATCH_LOWERING,provenance=Derived(tuple(refs)),mode=Mode.CONCORD)
    block_refs={}
    for label,kind in [('entry',SSABlockKind.ENTRY),('header',SSABlockKind.LOOP_HEADER),('body',SSABlockKind.LOOP_BODY),('exit',SSABlockKind.LOOP_EXIT)]:
        block_refs[label]=book.post(SSA_BLOCK,(scope,name,label),kind,stage=SCALAR_BATCH_LOWERING,provenance=Derived((root,)),mode=Mode.CONCORD)
    for label,block in blocks.items():
        terminal=block.instrs[-1]
        targets=[('flow',terminal.attributes['target'])] if terminal.op=='Br' else ([('body',terminal.attributes['true_target']),('orelse',terminal.attributes['false_target'])] if terminal.op=='CondBr' else [])
        block.successors.extend(target for _,target in targets)
        for arm,target in targets:
            book.post(SSA_CONTROL_EDGE,(scope,name,block_refs[label],block_refs[target],arm),True,stage=SCALAR_BATCH_LOWERING,provenance=Derived((root,block_refs[label],block_refs[target])),mode=Mode.CONCORD)
    values={v.id:v for v in f.args}
    values.update({i.res.id:i.res for b in blocks.values() for i in b.instrs if i.res is not None})
    for v in values.values():
        a=v.accounting or {}
        book.post(SSA_VALUE,(scope,v.id),SSAValueFact(v.dtype,tuple(v.shape),SSAValueOrigin.MINTED,physical_dtype=a.get('physical_dtype'),storage=a.get('program_abi_storage'),rank=a.get('program_abi_rank')),stage=SCALAR_BATCH_LOWERING,provenance=Derived((root,)),mode=Mode.CONCORD)
    module.functions[name]=f
    return f,tuple(v.id for v in inputs),tuple(v.id for v in destinations)


