"""Lifetime is an exact storage birth proof, never an escape authorization."""
from dataclasses import replace
from pathlib import Path
import pytest
from src.compiler import source_sequence_lifetime as lifetime
from src.compiler.concordance_declarations import SequenceLifetimeKind, SEQUENCE_STORAGE_LIFETIME

@pytest.fixture(scope="module")
def module():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    source = """def root(n: int, xs: list[int]):
    a=[]
    for i in range(n):
        a.append(i)
    b=[x for x in xs]
    c=list({x for x in xs})
    return len(a)+len(b)+len(c)
"""
    result, _, names = lower_ast_source_to_ssa(source, "root", extraction_contract=ExtractionContract(str(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")), progress=lambda _:None, progress_bars=False)
    return result, names[0]

def receipts(module):
    m,name=module
    return [value for sid in m.sequence_tables[name].sequences
            if (value:=lifetime.validated_sequence_lifetime(m,name,sid)) is not None]

def test_real_mixed_births_and_borrowed_parameter(module):
    m,name=module
    found=receipts(module)
    assert lifetime.sequence_lifetime_findings(m)==()
    assert sum(f.kind is SequenceLifetimeKind.BORROWED for _,f in found)==1
    local=[f for _,f in found if f.kind is SequenceLifetimeKind.LOCAL_NEW]
    assert any(len(f.birth_cells)==1 for f in local)  # authored list display
    assert any(len(f.birth_cells)==4 for f in local)  # exact comprehension publication
    # Reads continue to accept immutable birth revisions after descriptor refinements.
    assert all(lifetime.validated_sequence_lifetime(m,name,c.row[1]) for c,_ in found)

def forged(monkeypatch,book,cell,fact):
    original_fact=lifetime._fact
    original_edges=book.edges_into
    monkeypatch.setattr(lifetime,"_fact",lambda b,c:fact if b is book and c==cell else original_fact(b,c))
    monkeypatch.setattr(book,"edges_into",lambda c:tuple((x,None) for x in lifetime._causes(fact)) if c==cell else original_edges(c))

@pytest.mark.parametrize("change,match",[("source","source differs"),("contract","wrong descriptor contract"),("owner","member mint"),("data","member mint")])
def test_forged_identity_even_with_matching_causes(module,monkeypatch,change,match):
    m,_=module;book=m.metadata["identity_book"]
    local=[(c,f) for c,f in receipts(module) if f.kind is SequenceLifetimeKind.LOCAL_NEW]
    (cell,fact),(_,other)=local[:2]
    changes={"source":{"source_cell":other.source_cell,"birth_cells":other.birth_cells},
             "contract":{"contract_cell":other.contract_cell},
             "owner":{"owner_cell":other.owner_cell},
             "data":{"column_cells":other.column_cells}}
    forged(monkeypatch,book,cell,replace(fact,**changes[change]))
    with pytest.raises(ValueError,match=match):lifetime._validate(book,cell)

def test_borrowed_cannot_be_blessed_by_unrelated_local_source(module,monkeypatch):
    m,_=module;book=m.metadata["identity_book"]
    found=receipts(module)
    cell,fact=next((c,f) for c,f in found if f.kind is SequenceLifetimeKind.BORROWED)
    _,local=next((c,f) for c,f in found if f.kind is SequenceLifetimeKind.LOCAL_NEW)
    forged(monkeypatch,book,cell,replace(fact,kind=SequenceLifetimeKind.LOCAL_NEW,
        source_cell=local.source_cell,birth_cells=local.birth_cells,owner_cell=local.owner_cell))
    with pytest.raises(ValueError,match="source differs"):lifetime._validate(book,cell)

def test_adopted_receipt_inherits_exact_original(module,monkeypatch):
    m,_=module;book=m.metadata["identity_book"]
    original,fact=next((c,f) for c,f in receipts(module) if f.kind is SequenceLifetimeKind.LOCAL_NEW)
    cell=type(original)(original.page,original.row,original.column+1000)
    adopted=replace(fact,kind=SequenceLifetimeKind.ADOPTED,birth_cells=(),parent_cell=original)
    forged(monkeypatch,book,cell,adopted)
    assert lifetime._validate(book,cell)==adopted


def test_adoption_cycle_refused(module,monkeypatch):
    m,_=module;book=m.metadata["identity_book"]
    original,fact=next((c,f) for c,f in receipts(module) if f.kind is SequenceLifetimeKind.LOCAL_NEW)
    cell=type(original)(original.page,original.row,original.column+1000)
    adopted=replace(fact,kind=SequenceLifetimeKind.ADOPTED,birth_cells=(),parent_cell=cell)
    forged(monkeypatch,book,cell,adopted)
    with pytest.raises(ValueError,match="cyclic lifetime"):
        lifetime._validate(book,cell)


def test_current_descriptor_mismatch_refused(module,monkeypatch):
    m,name=module;cell,fact=next((c,f) for c,f in receipts(module) if f.kind is SequenceLifetimeKind.LOCAL_NEW)
    table=m.sequence_tables[name];original=table.sequences[cell.row[1]]
    monkeypatch.setitem(table.sequences,cell.row[1],replace(original,heap_owner_value_id=None))
    with pytest.raises(ValueError,match="current descriptor|current lifetime descriptor"):lifetime.validated_sequence_lifetime(m,name,cell.row[1])


@pytest.mark.parametrize("source", [
    "def helper(xs: list[int]):\n    return len(xs)\ndef root(xs: list[int]):\n    a=[x for x in xs]\n    return helper(a)\n",
    "def root(n: int):\n    a=[]\n    for i in range(n):\n        a.append(i)\n    return n\n",
    "def root(n: int):\n    a=[]\n    b=a\n    for i in range(n):\n        b.append(i)\n    return len(a)\n",
    "def root(xs: list[int], n: int):\n    xs.append(n)\n    return len(xs)\n",
], ids=["specialized-helper", "unused-local-result", "local-alias-mutation", "borrowed-input-mutation"])
def test_live_lifetime_scopes_pass_global_concordance(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.identity_concordance import CorrelationTable
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", extraction_contract=ExtractionContract(str(
            Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")),
        progress=lambda _:None, progress_bars=False)
    assert tuple(module.metadata["identity_book"].page(SEQUENCE_STORAGE_LIFETIME).rows())
    assert lifetime.sequence_lifetime_findings(module) == ()
    assert CorrelationTable.build(module).findings(module) == []


@pytest.mark.parametrize("expression,parameters", [("a", ("a",)), ("a+b", ("a", "b"))])
def test_explicit_span_signature_lifetime_uses_declaration_mint(expression,parameters):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.concordance_declarations import PARAMETER_ABI_DECLARATION
    contract=ExtractionContract(str(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")).with_program_abi({
        "values":[{"function":"root","parameter":p,"storage":"span","dtype":"int64","rank":1,"python_type":"builtins.bytes"} for p in parameters],
        "records":{}, "bindings":[]})
    signature=", ".join(p+": bytes" for p in parameters)
    m,_,names=lower_ast_source_to_ssa(f"def root({signature}) -> bytes:\n    return {expression}\n", "root", extraction_contract=contract, progress=lambda _:None, progress_bars=False)
    book=m.metadata["identity_book"]
    borrowed=[f for _,f in receipts((m,names[0])) if f.kind is SequenceLifetimeKind.BORROWED]
    assert len(borrowed)==len(parameters)
    for f in borrowed:
        kind,binding=f.birth_cells
        assert book.mint_of(kind)==(PARAMETER_ABI_DECLARATION,(binding,))
    assert CorrelationTable.build(m).findings(m)==[]


@pytest.fixture(scope="module")
def declared_bytes_module():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    contract=ExtractionContract(str(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")).with_program_abi({
        "values":[{"function":"root","parameter":p,"storage":"span","dtype":"int64","rank":1,"python_type":"builtins.bytes"} for p in ("a", "b")],
        "records":{}, "bindings":[]})
    m,_,names=lower_ast_source_to_ssa("def root(a: bytes, b: bytes) -> bytes:\n    return a+b\n", "root", extraction_contract=contract, progress=lambda _:None, progress_bars=False)
    return m,names[0]


@pytest.mark.parametrize("change", ["swapped-binding", "foreign-mint-operand"])
def test_declared_signature_requires_its_exact_binding(declared_bytes_module,monkeypatch,change):
    m,_=declared_bytes_module;book=m.metadata["identity_book"]
    borrowed=[(c,f) for c,f in receipts(declared_bytes_module) if f.kind is SequenceLifetimeKind.BORROWED]
    (cell,fact),(_,foreign)=borrowed
    kind,binding=fact.birth_cells
    foreign_binding=foreign.birth_cells[1]
    if change=="swapped-binding":
        forged(monkeypatch,book,cell,replace(fact,birth_cells=(kind,foreign_binding)))
    else:
        from src.compiler.concordance_declarations import PARAMETER_ABI_DECLARATION
        original=book.mint_of
        monkeypatch.setattr(book,"mint_of",lambda c:(PARAMETER_ABI_DECLARATION,(foreign_binding,)) if c==kind else original(c))
    with pytest.raises(ValueError,match="signature binding"):
        lifetime._validate(book,cell)
