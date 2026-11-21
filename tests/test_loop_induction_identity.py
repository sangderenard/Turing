"""A historical induction binding is valid only at a use inside its loop."""
from dataclasses import replace
from tests.test_owned_record_primitive_elements import lower
from tests.test_owned_record_return_source import SOURCE
from src.compiler.concordance_declarations import CONTROL_BLOCK, CONTROL_PROGRAM, PHI_LOOP_HEADER, LOOP_SCOPE
from src.compiler.ssa_record_return_state import ssa_value_identity_cell
from src.compiler.source_loop_induction_identity import loop_induction_sources


def test_exact_induction_binding_rejects_other_loop_and_outside_scope():
    module,name,_=lower(SOURCE)
    book=module.metadata['identity_book']; function=module.functions[name]
    arguments={}
    for row in book.page(CONTROL_BLOCK).rows():
        cell=book.latest_ref(CONTROL_BLOCK,row)
        for target,operation,args,effect,*rest in book.page(cell.page).cell(cell.row,cell.column).sequence_mutations:
            if operation=='append':arguments[effect.row[-1]]=args[0]
    uses=[]
    for label,block in function.blocks.items():
        for op in block.instrs:
            if op.op!='Call' or op.attributes.get('ssa_sequence_operation')!='append':continue
            actual=ssa_value_identity_cell(function,op.args[-1].id,book=book)
            mint=book.mint_of(actual)
            if mint is not None and mint[0]==PHI_LOOP_HEADER:
                source=arguments[op.attributes['source_effect_node_id']]
                proof=loop_induction_sources(module,name,actual,source,label)
                assert proof is not None
                uses.append((actual,source,label,proof))
    assert len(uses)==2
    for actual,source,label,proof in uses:
        boundary=next(cell for cell in proof if cell.page==LOOP_SCOPE)
        header,latch,exit_block=book.page(boundary.page).cell(boundary.row,boundary.column)
        assert loop_induction_sources(module,name,actual,source,header) is None
        assert loop_induction_sources(module,name,actual,source,exit_block) is None
        for other_actual,other_source,other_label,_ in uses:
            if other_source!=source:
                assert loop_induction_sources(module,name,actual,other_source,label) is None
                assert loop_induction_sources(module,name,actual,source,other_label) is None
        phi=next(op for op in function.blocks[header].instrs if op.res is not None and op.res.id==actual.row[-1])
        old_incoming=phi.attributes['incoming_blocks']
        try:
            phi.attributes['incoming_blocks']=(latch,latch)
            assert loop_induction_sources(module,name,actual,source,label) is None
        finally:phi.attributes['incoming_blocks']=old_incoming
        assert loop_induction_sources(module,'not_the_function',actual,source,label) is None
        old=phi.op
        try:
            phi.op='Const'
            assert loop_induction_sources(module,name,actual,source,label) is None
        finally:phi.op=old
        program=next(cell for cell in proof if cell.page==CONTROL_PROGRAM)
        page=book.page(program.page); key=(program.row,program.column); original=page.cells[key]
        try:
            page.cells[key]=replace(original,projected_iterable_bindings=())
            assert loop_induction_sources(module,name,actual,source,label) is None
        finally:page.cells[key]=original
        assert loop_induction_sources(module,name,actual,source,label)==proof


def test_nested_loop_keeps_outer_binding_but_not_inner_after_exit():
    source_text = """from dataclasses import dataclass
@dataclass
class Result:
    items: list[int]
def root(n: int) -> Result:
    out = []
    for i in range(n):
        for j in range(n):
            out.append(i)
            out.append(j)
        out.append(i)
    return Result(items=out)
"""
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    module, _, exports = lower_ast_source_to_ssa(source_text, 'root',
        extraction_contract=ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _:None,progress_bars=False)
    name=exports[0]
    book=module.metadata['identity_book']; function=module.functions[name]
    arguments={}
    for row in book.page(CONTROL_BLOCK).rows():
        cell=book.latest_ref(CONTROL_BLOCK,row)
        for target,operation,args,effect,*rest in book.page(cell.page).cell(cell.row,cell.column).sequence_mutations:
            if operation=='append':arguments[effect.row[-1]]=args[0]
    uses=[]
    for label,block in function.blocks.items():
        for op in block.instrs:
            if op.op!='Call' or op.attributes.get('ssa_sequence_operation')!='append':continue
            actual=ssa_value_identity_cell(function,op.args[-1].id,book=book)
            source=arguments[op.attributes['source_effect_node_id']]
            proof=loop_induction_sources(module,name,actual,source,label)
            assert proof is not None
            uses.append((actual,source,label,proof))
    assert len(uses)==3
    inner=next(use for use in uses if sum(other[1]==use[1] for other in uses)==1)
    outer_uses=[use for use in uses if use[1]!=inner[1]]
    outside_inner=next(use for use in outer_uses if use[2]!=inner[2])
    assert loop_induction_sources(module,name,inner[0],inner[1],outside_inner[2]) is None
