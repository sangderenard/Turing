import pytest
from src.compiler.backend_identities import apply_backend_identities
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def fixture():
    aggregate = SSAValue(10, "ssa.aggregate")
    index = SSAValue(11, "int64")
    pointer = SSAValue(12, "ptr")
    result = SSAValue(13, "int32")
    instructions = [
        Instr("Call", [], aggregate, attributes={"callee":"child", "output_ids":(21,22)}),
        Instr("Const", [], index, attributes={"value":1}),
        Instr("GetElementPtr", [aggregate,index], pointer,
              attributes={"aggregate_index":1,"source_output_id":22}),
        Instr("Load", [pointer], result), Instr("Ret",[result],None),
    ]
    child_values = [SSAValue(21,"int32"),SSAValue(22,"int32")]
    child = Function("child",[],{"entry":BasicBlock("entry",[
        Instr("Const",[],child_values[0],attributes={"value":2}),
        Instr("Const",[],child_values[1],attributes={"value":3}),
        Instr("Ret",child_values,None)])})
    module = IRModule({"main":Function("main",[],{"entry":BasicBlock("entry",instructions)}),"child":child})
    return module,{"main":(result,)},instructions


def test_exact_static_aggregate_index_elision_preserves_original():
    module,outputs,instructions=fixture()
    result=apply_backend_identities(module,outputs,backend="webgpu",licensed_inexact=False)
    assert result.decisions[-1].identity=="webgpu_static_aggregate_index_elision"
    assert result.decisions[-1].exact and result.decisions[-1].applied
    assert result.decisions[-1].before_sha256!=result.decisions[-1].after_sha256
    emitted=result.module.functions["main"].blocks["entry"].instrs
    assert not any(i.op=="Const" and i.res.id==11 for i in emitted)
    assert len(next(i for i in emitted if i.op=="GetElementPtr").args)==1
    assert len(instructions[2].args)==2 and instructions[1].res.dtype=="int64"


@pytest.mark.parametrize("mutation",["data_use","return","published_output","mismatch","unknown_owner","out_of_bounds","duplicate_definition","wrong_projection","wrong_type"])
def test_real_i64_and_unproven_indexes_are_not_elided(mutation):
    module,outputs,instructions=fixture()
    index=instructions[1].res
    if mutation=="data_use":instructions.insert(-1,Instr("Add",[index,index],SSAValue(30,"int64")))
    elif mutation=="return":instructions[-1].args.append(index)
    elif mutation=="published_output":outputs["main"]+=(index,)
    elif mutation=="mismatch":instructions[2].attributes["aggregate_index"]=0
    elif mutation=="unknown_owner":instructions[0].attributes["callee"]="missing"
    elif mutation=="out_of_bounds":instructions[0].attributes["output_ids"]=(21,)
    elif mutation=="duplicate_definition":instructions.insert(0,Instr("Const",[],SSAValue(11,"int64"),attributes={"value":1}))
    elif mutation=="wrong_projection":instructions[2].attributes["source_output_id"]=21
    elif mutation=="wrong_type":instructions[0].res.dtype="ptr"
    result=apply_backend_identities(module,outputs,backend="webgpu",licensed_inexact=False)
    assert not any(d.identity=="webgpu_static_aggregate_index_elision" for d in result.decisions)
    assert any(i.op=="Const" and i.res.id==11 and i.res.dtype=="int64"
               for i in result.module.functions["main"].blocks["entry"].instrs)


def test_identity_does_not_change_other_backends():
    module,outputs,_=fixture()
    result=apply_backend_identities(module,outputs,backend="wasm",licensed_inexact=False)
    assert not any(d.identity=="webgpu_static_aggregate_index_elision" for d in result.decisions)


@pytest.mark.parametrize("arity,index,complete",[(1,0,True),(2,1,True),(1,1,False),(1,-1,False)])
def test_webgpu_projection_matches_the_callee_return_shape(arity,index,complete):
    from src.compiler.ssa_webgpu_backend import _FunctionEmitter
    result=SSAValue(10,"ssa.aggregate")
    pointer=SSAValue(11,"ptr")
    value=SSAValue(12,"int32")
    function=Function("main",[],{"entry":BasicBlock("entry",[
        Instr("Call",[],result,attributes={"callee":"child"}),
        Instr("GetElementPtr",[result],pointer,attributes={"aggregate_index":index}),
        Instr("Load",[pointer],value),Instr("Ret",[value],None),
    ])})
    emitter=_FunctionEmitter(function,outputs=(value,),loop_records=(),
        callees={"child":tuple(SSAValue(20+i,"int32") for i in range(arity))})
    source,shortfalls=emitter.emit()
    assert (not shortfalls)==complete
    if complete:
        assert ("v_10[" in source)==(arity>1)
