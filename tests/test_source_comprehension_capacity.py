"""Source-bound proof rejects altered identities, cardinality and placement."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT, ControlBlockFact,
    ControlBlockKind, Placement, Arm, ROOT, NATIVE_OBJECT_SOURCE,
    PROCESS_GRAPH_OPERANDS, SOURCE_COMPREHENSION_CAPACITY_BOUND,
    SourceComprehensionCapacityBound, INGESTION, INGEST_SOURCE, LOOP_RESULT_PORT,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode
from src.compiler.source_comprehension_capacity import _validate, _sources, source_comprehension_capacity_findings

@pytest.fixture
def proof():
    book, token = begin_identity_book()
    scope = ("comprehension-test",)
    def post(page, row, value, causes=()):
        return book.post(page, row, value, stage=INGESTION,
            provenance=Derived(tuple(causes)) if causes else Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    cells = {i:post(CANONICAL_VALUE,(scope,i),("Input","input",str(i))) for i in (1,2,3,4,5,6)}
    destination = book.post(CANONICAL_VALUE,(scope,7),("LoopResult","loopresult","result"), stage=INGESTION, provenance=Novel(LOOP_RESULT_PORT,(cells[2],)),mode=Mode.CONCORD)
    publication=post(NATIVE_OBJECT_SOURCE,(scope,3,"publication"), (("projected_argument_value_ids",(4,5)),("source_argument_value_ids",(6,))), (cells[3],cells[6],cells[4],cells[5]))
    operands=post(PROCESS_GRAPH_OPERANDS,(scope,2),((cells[1],"iterable"),),(cells[2],cells[1]))
    mutation=(destination,"update",(cells[4],cells[5]),cells[2],"unique","mapping_items",publication)
    control=post(CONTROL_BLOCK,(scope,ControlBlockKind.LOOP,cells[2]),ControlBlockFact(ControlBlockKind.LOOP,None,(),(),(),None,("i","0","__iterable_extent_1__","1","lt","alap"),sequence_mutations=(mutation,)),(cells[2],destination,cells[4],cells[5],publication))
    placement=post(CONTROL_BLOCK_PLACEMENT,(scope,control),Placement(ROOT,Arm.ROOT_SEQUENCE,0),(control,))
    fact=SourceComprehensionCapacityBound(destination,cells[1],cells[2],cells[3],publication,operands,control,placement)
    try: yield book,fact,cells,post
    finally:end_identity_book(token)


def test_exact_one_row_bound(proof):
    book,fact,_,post=proof
    assert _validate(book,fact)==fact
    post(SOURCE_COMPREHENSION_CAPACITY_BOUND,(fact.destination_cell.row[0],7),fact,_sources(fact))
    assert source_comprehension_capacity_findings(SimpleNamespace(metadata={"identity_book":book}))==()

@pytest.mark.parametrize("field,other",[("source_cell",4),("loop_cell",5),("materializer_cell",6),("destination_cell",1)])
def test_identity_substitution_refused(proof,field,other):
    book,fact,cells,_=proof
    with pytest.raises(ValueError):_validate(book,replace(fact,**{field:cells[other]}))


def test_missing_bound_causes_refused(proof):
    book,fact,_,post=proof
    post(SOURCE_COMPREHENSION_CAPACITY_BOUND,(fact.destination_cell.row[0],7),fact,(fact.destination_cell,))
    assert source_comprehension_capacity_findings(SimpleNamespace(metadata={"identity_book":book}))

@pytest.mark.parametrize("change",["extra", "bulk", "nested", "repeat"])
def test_changed_control_refuses_prior_bound(proof,change):
    book,fact,_,post=proof
    old=book.page(CONTROL_BLOCK).cell(fact.control_cell.row,fact.control_cell.column)
    # A changed current revision invalidates the exact prior bound, regardless
    # of whether its old publication is still present somewhere in history.
    if change=="extra":new=replace(old,sequence_mutations=old.sequence_mutations*2)
    elif change=="bulk":
        mutation=list(old.sequence_mutations[0]);mutation[2]=mutation[2][:1]
        new=replace(old,sequence_mutations=(tuple(mutation),))
    elif change=="repeat":new=replace(old,extra=("i","0","__iterable_extent_1__","0","lt","alap"))
    else:
        book.post(CONTROL_BLOCK_PLACEMENT,fact.placement_cell.row,Placement(fact.control_cell,Arm.BODY,0),stage=INGESTION,provenance=Derived((fact.placement_cell,)),mode=Mode.REVISE)
        with pytest.raises(ValueError):_validate(book,fact)
        return
    revised=book.post(CONTROL_BLOCK,fact.control_cell.row,new,stage=INGESTION,provenance=Derived((fact.control_cell,)),mode=Mode.REVISE)
    with pytest.raises(ValueError):_validate(book,fact)
    placement=post(CONTROL_BLOCK_PLACEMENT,(fact.destination_cell.row[0],revised),Placement(ROOT,Arm.ROOT_SEQUENCE,0),(revised,))
    with pytest.raises(ValueError):_validate(book,replace(fact,control_cell=revised,placement_cell=placement))


def test_alias_write_invalidates_bound(proof):
    book,fact,cells,post=proof
    scope=fact.destination_cell.row[0]
    alias=post(CANONICAL_VALUE,(scope,8),("Name","name","alias"),(fact.destination_cell,))
    mutation=(alias,"update",(cells[4],cells[5]),cells[2],"unique","mapping_items",fact.publication_cell)
    other=post(CONTROL_BLOCK,(scope,ControlBlockKind.SEQUENCE_MUTATION,alias),ControlBlockFact(ControlBlockKind.SEQUENCE_MUTATION,None,(),(),(),None,(),sequence_mutations=(mutation,)),(alias,cells[2],cells[4],cells[5],fact.publication_cell))
    post(CONTROL_BLOCK_PLACEMENT,(scope,other),Placement(ROOT,Arm.ROOT_SEQUENCE,1),(other,))
    with pytest.raises(ValueError,match="additional publications"):_validate(book,fact)


def test_real_dict_comprehension_source_receipt(monkeypatch):
    from pathlib import Path
    from src.compiler import control_source
    from src.compiler.source_comprehension_capacity import publish_source_comprehension_capacity
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    contexts=[]
    original=control_source.post_control_rewrite
    def capture(graph,control,*args,**kwargs):
        result=original(graph,control,*args,**kwargs)
        contexts.append((graph,control))
        return result
    monkeypatch.setattr(control_source,"post_control_rewrite",capture)
    source=("from dataclasses import dataclass\n@dataclass\nclass Result:\n    registry:dict[int,int]\n"
            "def root(xs:list[int])->Result:\n    entries={x:x*2 for x in xs if x>0}\n"
            + "    return Result(entries)\n")
    module,_,_=lower_ast_source_to_ssa(source,"root",extraction_contract=ExtractionContract(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml"),progress=lambda _:None,progress_bars=False)
    book=module.metadata["identity_book"]
    seen=set();published=[]
    for graph,control in reversed(contexts):
        graph=getattr(graph,"G",graph)
        if graph is None:continue
        scope=graph.graph.get("lexical_read_scope")
        if scope in seen:continue
        seen.add(scope)
        published.extend(publish_source_comprehension_capacity(graph,control,book=book))
    assert len(published)==1
    assert source_comprehension_capacity_findings(module)==()

@pytest.mark.parametrize('parent_kind', [ControlBlockKind.CONDITIONAL, ControlBlockKind.LOOP, ControlBlockKind.WHILE])
def test_separate_publication_must_be_guarded_not_repeated(proof, parent_kind):
    from src.compiler.source_comprehension_capacity import _publication_path
    book,fact,cells,post=proof
    scope=fact.destination_cell.row[0]
    loop_fact=book.page(CONTROL_BLOCK).cell(fact.control_cell.row,fact.control_cell.column)
    mutation=loop_fact.sequence_mutations[0]
    loop=book.post(CONTROL_BLOCK,fact.control_cell.row,replace(loop_fact,sequence_mutations=()),
        stage=INGESTION,provenance=Derived((fact.control_cell,)),mode=Mode.REVISE)
    placement=post(CONTROL_BLOCK_PLACEMENT,(scope,loop),Placement(ROOT,Arm.ROOT_SEQUENCE,0),(loop,))
    parent=post(CONTROL_BLOCK,(scope,parent_kind,cells[5]),ControlBlockFact(parent_kind,None,(),(),(),None,()),(cells[5],))
    post(CONTROL_BLOCK_PLACEMENT,(scope,parent),Placement(loop,Arm.BODY,0),(parent,loop))
    publication=post(CONTROL_BLOCK,(scope,ControlBlockKind.SEQUENCE_MUTATION,cells[3]),
        ControlBlockFact(ControlBlockKind.SEQUENCE_MUTATION,None,(),(),(),None,(),sequence_mutations=(mutation,)),
        (fact.destination_cell,cells[2],cells[4],cells[5],fact.publication_cell))
    post(CONTROL_BLOCK_PLACEMENT,(scope,publication),Placement(parent,Arm.BODY,0),(publication,parent))
    revised=replace(fact,control_cell=loop,placement_cell=placement)
    if parent_kind is not ControlBlockKind.CONDITIONAL:
        with pytest.raises(ValueError,match='repeatable'):
            _publication_path(book,scope,publication,loop)
        return
    path=_publication_path(book,scope,publication,loop)
    with pytest.raises(ValueError,match='placement proof'):_validate(book,revised)
    assert _validate(book,replace(revised,publication_path=path))
    # Moving the guarded publication into loop-else cannot establish a per-row bound.
    old=book.latest_ref(CONTROL_BLOCK_PLACEMENT,(scope,parent))
    book.post(CONTROL_BLOCK_PLACEMENT,(scope,parent),Placement(loop,Arm.ORELSE,0),
        stage=INGESTION,provenance=Derived((old,)),mode=Mode.REVISE)
    with pytest.raises(ValueError):_validate(book,replace(revised,publication_path=path))
