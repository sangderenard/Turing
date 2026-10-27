"""Sequence row Loads agree with their exact ABI and materialized column."""
import pytest
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, SSA_VALUE, SOURCE_VALUE_CLASS, SOURCE_AGGREGATE_LAYOUT,
    VALUE_TYPE_ABI, SourceAggregateLayout, SourceAggregateMember,
    ValueTypeABIFact, ValueTypeABIAlternative, SSAValueFact, SSAValueOrigin,
    INGESTION, INGEST_SOURCE,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode, CorrelationTable
from src.transmogrifier.ssa import SSAValue, Function, BasicBlock, Instr, IRModule, SSASequenceTable, SSASequenceDescriptor


def _fixture(book, *, descriptor_dtype='int64', load_dtype='int64', abi_dtype='bool',
             physical=None, nominal_members=False, result_class='builtins.int', foreign_classes=False):
    scope = ('source',0)
    canonical = book.post(CANONICAL_VALUE, (scope,2), ('sequence',), stage=INGESTION,
                          provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    target = book.post(CANONICAL_VALUE, (scope,1), ('loop target',), stage=INGESTION,
                       provenance=Derived((canonical,)), mode=Mode.CONCORD)
    sources = [canonical]
    if nominal_members:
        aggregate = book.post(CANONICAL_VALUE, (scope,100), ('aggregate',), stage=INGESTION,
                              provenance=Derived((canonical,)), mode=Mode.CONCORD)
        members = []
        for value_id in (20,21):
            value = book.post(CANONICAL_VALUE, (scope,value_id), ('member',), stage=INGESTION,
                              provenance=Derived((aggregate,)), mode=Mode.CONCORD)
            class_source = value
            if foreign_classes:
                class_source = book.post(CANONICAL_VALUE, (('foreign',0),value_id), ('other member',),
                                         stage=INGESTION, provenance=Derived((canonical,)), mode=Mode.CONCORD)
            cls = book.post(SOURCE_VALUE_CLASS, (('classes',0),value_id), ('builtins.bool',1,'exact fixture member'),
                            stage=INGESTION, provenance=Derived((class_source,)), mode=Mode.CONCORD)
            sources.append(cls)
            members.append(SourceAggregateMember(value_id,value,(value,)))
        book.post(SOURCE_AGGREGATE_LAYOUT, (scope,100), SourceAggregateLayout('list',tuple(members)),
                  stage=INGESTION, provenance=Derived((aggregate,*(member.value_cell for member in members))), mode=Mode.CONCORD)
        sources.append(aggregate)
    abi = ValueTypeABIFact(('builtins.list',),(),(ValueTypeABIAlternative(
        'builtins.list','span',abi_dtype,(),1,(),tuple(sources)),))
    book.post(VALUE_TYPE_ABI, (scope,2), abi, stage=INGESTION,
              provenance=Derived(tuple(sources)), mode=Mode.CONCORD)
    source = SSAValue(2,physical or descriptor_dtype,(2,))
    constant = SSAValue(10,'ssa.aggregate')
    index, address, loaded = SSAValue(11,'int64'), SSAValue(12,'ptr'), SSAValue(1,load_dtype)
    for value in (source,constant,index,address,loaded):
        book.post(SSA_VALUE, ('root',value.id),
                  SSAValueFact(value.dtype,value.shape,SSAValueOrigin.ADOPTED_GRAPH_ID,
                               physical_dtype=physical if value.id in (2,10) else None),
                  stage=INGESTION, provenance=Derived((target if value.id==1 else canonical,)), mode=Mode.CONCORD)
    result_abi = ValueTypeABIFact((result_class,),(),(ValueTypeABIAlternative(
        result_class,'scalar',load_dtype,(),0,(),(target,)),))
    book.post(VALUE_TYPE_ABI, (scope,1), result_abi, stage=INGESTION,
              provenance=Derived((target,)), mode=Mode.CONCORD)
    function = Function('root',[source],{'entry':BasicBlock('entry',[
        Instr('Const',[],constant,attributes={'value':(False,True),'binding':'static_iterable','source_value_id':2}),
        Instr('Const',[],index,attributes={'value':0}),
        Instr('GetElementPtr',[constant,index],address),Instr('Load',[address],loaded),Instr('Ret',[loaded],None),
    ])},metadata={'native_value_type_abi':{1:result_abi}})
    table = SSASequenceTable(owner='root',book=book)
    table.register(SSASequenceDescriptor(2,(2,),30,31,column_dtypes=(descriptor_dtype,)),
                   sources=(canonical,),stage=INGESTION)
    return IRModule({'root':function},sequence_tables={'root':table},metadata={'identity_book':book})


def test_static_bool_column_and_int64_loop_load_disagree_with_exact_abi():
    book, token = begin_identity_book()
    try:
        module = _fixture(book)
        kinds = {finding.kind for finding in CorrelationTable._sequence_descriptor_findings(module)}
        assert 'sequence-column-abi-disagreement' in kinds
        assert 'static-iterable-load-type-disagreement' in kinds
    finally:end_identity_book(token)


def test_explicit_post_load_cast_is_not_an_implicit_width_change():
    book, token = begin_identity_book()
    try:
        module = _fixture(book,descriptor_dtype='bool',load_dtype='bool',result_class='builtins.bool')
        block = module.functions['root'].blocks['entry']
        loaded = block.instrs[-1].args[0]
        converted = SSAValue(40,'int64')
        block.instrs[-1:-1] = [Instr('Cast',[loaded],converted)]
        block.instrs[-1].args = [converted]
        assert not CorrelationTable._sequence_descriptor_findings(module)
    finally:end_identity_book(token)


def test_declared_physical_materialization_preserves_packed_representation():
    book, token = begin_identity_book()
    try:
        module = _fixture(book,descriptor_dtype='float64',load_dtype='float64',physical='float64')
        assert not CorrelationTable._sequence_descriptor_findings(module)
    finally:end_identity_book(token)


def test_equal_numeric_id_in_foreign_scope_is_not_the_source_abi():
    book, token = begin_identity_book()
    try:
        module = _fixture(book,abi_dtype='int64')
        foreign = book.post(CANONICAL_VALUE, (('foreign',0),2), ('unrelated',), stage=INGESTION,
                            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        fact = ValueTypeABIFact(('builtins.list',),(),(ValueTypeABIAlternative(
            'builtins.list','span','bool',(),1,(),(foreign,)),))
        book.post(VALUE_TYPE_ABI, foreign.row, fact,stage=INGESTION,
                  provenance=Derived((foreign,)),mode=Mode.CONCORD)
        assert not CorrelationTable._sequence_descriptor_findings(module)
    finally:end_identity_book(token)


def test_exact_member_class_disagreement_survives_matching_physical_types():
    book, token = begin_identity_book()
    try:
        module = _fixture(book,descriptor_dtype='bool',load_dtype='bool',nominal_members=True)
        kinds = {finding.kind for finding in CorrelationTable._sequence_descriptor_findings(module)}
        assert kinds == {'static-iterable-element-class-disagreement'}
    finally:end_identity_book(token)


def test_foreign_equal_member_ids_do_not_prove_nominal_element_class():
    book, token = begin_identity_book()
    try:
        module = _fixture(book,descriptor_dtype='bool',load_dtype='bool',nominal_members=True,foreign_classes=True)
        assert not CorrelationTable._sequence_descriptor_findings(module)
    finally:end_identity_book(token)


def test_product_carrier_does_not_define_its_member_column_width():
    from dataclasses import replace
    book, token = begin_identity_book()
    try:
        module = _fixture(book,descriptor_dtype='bool',load_dtype='bool',nominal_members=True,result_class='builtins.bool')
        scope = ('source',0)
        layout = book.page(SOURCE_AGGREGATE_LAYOUT).latest((scope,100))
        source = book.latest_ref(CANONICAL_VALUE,(scope,2))
        layout_cell = book.post(SOURCE_AGGREGATE_LAYOUT,(scope,2),layout,stage=INGESTION,
                                provenance=Derived((source,)),mode=Mode.CONCORD)
        old = book.latest_ref(VALUE_TYPE_ABI,(scope,2))
        fact = book.page(VALUE_TYPE_ABI).cell(old.row,old.column)
        carrier = replace(fact,physical_alternatives=tuple(replace(arm,dtype='int64') for arm in fact.physical_alternatives))
        book.post(VALUE_TYPE_ABI,old.row,carrier,stage=INGESTION,
                  provenance=Derived((old,layout_cell)),mode=Mode.REVISE)
        assert not CorrelationTable._sequence_descriptor_findings(module)
    finally:end_identity_book(token)
