"""Borrowed component roles and native recipes are immutable birth evidence."""
from dataclasses import replace
from pathlib import Path
import pytest
from src.compiler.precompile_to_ssa import _ControlSSABuilder
from src.compiler.source_borrowed_record_spans import borrowed_record_string_view
from src.compiler.concordance_declarations import BORROWED_RECORD_SPAN_ROLE
from tests.test_borrowed_record_string_copy import SOURCE,lower_before_owned_record_boundary,ExtractionContract

def _with_builder(monkeypatch, check):
    class Captured(BaseException):pass
    old=_ControlSSABuilder._lower_sequence_mutation_body
    def inspect(builder,mutation,*args,**kwargs):
        if mutation.operator=='append':
            sid,=mutation.argument_value_ids
            assert borrowed_record_string_view(builder,sid)
            check(builder,sid)
            raise Captured
        return old(builder,mutation,*args,**kwargs)
    monkeypatch.setattr(_ControlSSABuilder,'_lower_sequence_mutation_body',inspect)
    with pytest.raises(Captured):
        lower_before_owned_record_boundary(SOURCE,'root',extraction_contract=ExtractionContract(
            Path(__file__).parents[1]/'extraction_contracts/program_extraction.yaml'),
            progress=lambda _:None,progress_bars=False)

@pytest.mark.parametrize('change',['glyph-length','capacity-stride','capacity-alias','wrong-binding','op-only'])
def test_current_role_or_recipe_cannot_replace_birth(monkeypatch,change):
    def check(builder,sid):
        view=builder.sequence_descriptors[sid];pool,handle=builder.child_table_selections[sid]
        key=next(k for k,p in builder.borrowed_record_field_pools.items() if p==pool)
        defs={op.res.id:op for b in builder.blocks.values() for op in b.instrs if op.res is not None}
        if change in ('glyph-length','op-only'):
            altered=replace(pool,column_value_ids=(pool.length_value_id,),length_value_id=pool.column_value_ids[0]) if change=='glyph-length' else pool
            ops=[(defs[view.column_value_ids[0]],0,pool.length_value_id),(defs[view.length_address_id],0,pool.column_value_ids[0])]
        elif change=='capacity-stride':
            altered=replace(pool,capacity_value_id=pool.row_stride_value_id,row_stride_value_id=pool.capacity_value_id)
            offset=defs[defs[view.column_value_ids[0]].args[1].id]
            ops=[(offset,1,pool.capacity_value_id),(defs[view.capacity_value_id],0,pool.capacity_value_id)]
        else:
            altered=replace(pool,capacity_value_id=pool.row_stride_value_id) if change=='capacity-alias' else pool
            ops=[]
        monkeypatch.setitem(builder.borrowed_record_field_pools,key,altered)
        monkeypatch.setitem(builder.child_table_selections,sid,(altered,handle))
        for op,pos,new in ops:
            args=list(op.args);args[pos]=builder.ssa_value_objects[new];monkeypatch.setattr(op,'args',args)
        if change=='wrong-binding':
            from src.compiler.concordance_declarations import CONTROL_VALUE_BINDING,SSA_VALUE
            book=builder._book();cell=book.latest_ref(CONTROL_VALUE_BINDING,(builder._scope(),sid))
            page=book.page(CONTROL_VALUE_BINDING);fact=page.cell(cell.row,cell.column)
            monkeypatch.setitem(page.cells,(cell.row,cell.column),replace(fact,ssa_value=book.latest_ref(SSA_VALUE,(builder._scope(),view.length_address_id))))
        assert borrowed_record_string_view(builder,sid) is None
    _with_builder(monkeypatch,check)

@pytest.mark.parametrize('change',['missing','forged-parent','forged-projection','missing-cause'])
def test_birth_role_requires_exact_source(monkeypatch,change):
    def check(builder,sid):
        book=builder._book()
        proof=borrowed_record_string_view(builder,sid)
        role=next(c for c in proof if c.page==BORROWED_RECORD_SPAN_ROLE and c.row[2]=='column')
        page=book.page(BORROWED_RECORD_SPAN_ROLE);fact=page.cell(role.row,role.column)
        if change=='missing':
            old=book.latest_ref
            monkeypatch.setattr(book,'latest_ref',lambda p,r:None if p==BORROWED_RECORD_SPAN_ROLE and r==role.row else old(p,r))
        elif change=='missing-cause':
            old=book.edges_into
            monkeypatch.setattr(book,'edges_into',lambda c:tuple(e for e in old(c) if e[0]!=fact.parent_cell) if c==role else old(c))
        else:
            altered=replace(fact,parent_cell=role.row[1]) if change=='forged-parent' else replace(fact,projection=(*fact.projection[:3],'Node.wrong',*fact.projection[4:]))
            monkeypatch.setitem(page.cells,(role.row,role.column),altered)
        assert borrowed_record_string_view(builder,sid) is None
    _with_builder(monkeypatch,check)

@pytest.mark.parametrize('change',['missing-mint','disconnected-lineage','duplicate-operands'])
def test_recipe_requires_current_connected_mint(monkeypatch,change):
    def check(builder,sid):
        from src.compiler.concordance_declarations import SSA_VALUE
        book=builder._book();view=builder.sequence_descriptors[sid]
        cell=book.latest_ref(SSA_VALUE,(builder._scope(),view.column_value_ids[0]))
        if change in ('missing-mint','disconnected-lineage'):
            old_mint=book.mint_of
            monkeypatch.setattr(book,'mint_of',lambda c:None if c.page==SSA_VALUE and c.row==cell.row else old_mint(c))
            if change=='disconnected-lineage':
                old_edges=book.edges_into
                monkeypatch.setattr(book,'edges_into',lambda c:tuple(e for e in old_edges(c) if e[0].page!=SSA_VALUE or e[0].row!=cell.row) if c.page==SSA_VALUE and c.row==cell.row else old_edges(c))
        else:
            op=next(op for b in builder.blocks.values() for op in b.instrs if op.res is not None and op.res.id==view.column_value_ids[0])
            monkeypatch.setattr(op,'args',[op.args[0],op.args[0]])
        assert borrowed_record_string_view(builder,sid) is None
    _with_builder(monkeypatch,check)
