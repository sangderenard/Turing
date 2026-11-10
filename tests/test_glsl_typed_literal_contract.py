"""Explicit source integer widths survive the existing compiler and audit."""
from pathlib import Path
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book, EDGE_PAGE
from src.compiler.concordance_declarations import SSA_CONSTANT_MATERIALIZATION, ANNOTATION_DECLARATION, VALUE_TYPE_ABI

@pytest.fixture(scope='module')
def product():
    root=Path(__file__).resolve().parents[1]
    contract=ExtractionContract(root/'extraction_contracts/program_extraction.yaml').with_program_abi({
      'records':{},'bindings':[],'values':[{'function':'main','parameter':'n','storage':'scalar','dtype':'int32','rank':0,'shape':[],'python_type':'int'}]
    }).with_execution_file(root/'extraction_contracts/vehicle_full_native_execution.yaml')
    module,_,exports=lower_ast_source_to_ssa(
      'uniform int n; out int answer; void main(){int total=0;int k=0;while(k<n){total+=k;k+=1;}answer=total;}',
      'main',source_language='glsl',extraction_contract=contract,progress_bars=False)
    book=identity_book(module)
    page=book.page(SSA_CONSTANT_MATERIALIZATION)
    receipts=[book.latest_ref(SSA_CONSTANT_MATERIALIZATION,row) for row in page.rows()
              if page.latest(row)[1]=='int32' and any(c.page==ANNOTATION_DECLARATION for c,_ in book.edges_into(book.latest_ref(SSA_CONSTANT_MATERIALIZATION,row)))]
    assert receipts
    assert CorrelationTable.build(module).findings(module)==[]
    return module,book,receipts[0],exports[0]

def test_original_glsl_loop_has_strict_typed_receipts(product):
    module,book,ref,_=product
    assert CorrelationTable._constant_materialization_findings(module)==[]
    assert any(c.page==VALUE_TYPE_ABI for c,_ in book.edges_into(ref))

@pytest.mark.parametrize('dtype,literal',[('int64','0'),('int32',str(1<<31)),('int32',str(-(1<<31)-1))])
def test_source_width_rejects_forged_materialization(product,dtype,literal):
    module,book,ref,_=product
    page=book.page(SSA_CONSTANT_MATERIALIZATION);old=page.cell(ref.row,ref.column)
    page.cells[(ref.row,ref.column)]=('Const',dtype,literal)
    try: assert CorrelationTable._constant_materialization_findings(module)
    finally: page.cells[(ref.row,ref.column)]=old

@pytest.mark.parametrize('missing',[ANNOTATION_DECLARATION,VALUE_TYPE_ABI])
def test_source_width_requires_both_direct_causes(product,monkeypatch,missing):
    module,book,ref,_=product
    causes={c for c,_ in book.edges_into(ref) if c.page==missing}
    assert causes
    edges=book.page(EDGE_PAGE); original=edges.scope_rows
    monkeypatch.setattr(edges,'scope_rows',lambda scope:tuple(e for e in original(scope)
      if not(scope==ref.key and any(e[1]==c.key for c in causes))))
    assert CorrelationTable._constant_materialization_findings(module)

def test_unannotated_python_integer_domain_stays_int64():
    root=Path(__file__).resolve().parents[1]
    contract=ExtractionContract(root/'extraction_contracts/program_extraction.yaml').with_program_abi({
      'records':{},'bindings':[],'values':[{'function':'main','parameter':'n','storage':'scalar','dtype':'int64','rank':0,'shape':[],'python_type':'int'}]
    }).with_execution_file(root/'extraction_contracts/vehicle_full_native_execution.yaml')
    module,_,_=lower_ast_source_to_ssa('def main(n: int):\n    return n + 1\n','main',extraction_contract=contract,progress_bars=False)
    assert CorrelationTable.build(module).findings(module)==[]
    assert all(i.res.dtype=='int64' for f in module.functions.values() for b in f.blocks.values()
               for i in b.instrs if i.op=='Const' and type(i.attributes.get('value')) is int)


def test_original_glsl_int32_loop_executes_in_wasm(product):
    import numpy as np
    from src.compiler.ssa_wasm_backend import (
        emit_ssa_module_to_wasm_core, prepare_wasm_core_execution,
    )
    module, _book, _receipt, entry = product
    artifact = emit_ssa_module_to_wasm_core(module, entry)
    assert artifact.complete, artifact.shortfalls
    function = module.functions[entry]
    outputs = [value.id for block in function.blocks.values()
               for instruction in block.instrs if instruction.op == "Ret"
               for value in instruction.args]
    assert len(outputs) == 1
    for n in (0, 1, 2, 5, 12, 31):
        execution = prepare_wasm_core_execution(
            artifact, {function.args[0].id: np.array([n], dtype=np.int32)},
        )
        try:
            execution.run()
            assert execution.buffers[outputs[0]].tolist() == [n * (n - 1) // 2]
        finally:
            execution.close()
    assert CorrelationTable.build(module).findings(module) == []


def test_original_glsl_loop_uses_normal_webgpu_backend(product):
    from src.compiler.ssa_webgpu_backend import emit_module
    module, _book, _receipt, entry = product
    outputs = tuple(value for block in module.functions[entry].blocks.values()
                    for instruction in block.instrs if instruction.op == "Ret"
                    for value in instruction.args)
    from src.compiler.backend_identities import apply_backend_identities
    prepared = apply_backend_identities(module, {entry: outputs}, backend="webgpu", licensed_inexact=False)
    assert CorrelationTable.build(prepared.module).findings(prepared.module) == []
    artifact = emit_module(module, name=entry, count=32, outputs={entry: outputs})
    assert artifact.complete, artifact.shortfalls
    assert "loop {" in artifact.source
    assert "array<i32, 2>" in artifact.source
    assert "-> bool" in artifact.source
    assert CorrelationTable.build(module).findings(module) == []
