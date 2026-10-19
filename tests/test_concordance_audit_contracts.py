"""Audit overlays preserve the program's already-declared helper ABI."""
from pathlib import Path
import runpy
import pytest
from src.compiler.identity_concordance import concordance_report
from src.compiler.identity_concordance import CorrelationTable


def test_record_handle_alias_is_owned_but_not_a_physical_operand():
    from src.transmogrifier.ssa import (
        SSAValue, Instr, Function, BasicBlock, IRModule,
        SSARecordDescriptor, SSARecordFieldDescriptor, SSARecordTable,
    )
    leaf = SSAValue(0, "float64")
    function = Function("root", [leaf], {
        "entry": BasicBlock("entry", [Instr("Ret", [leaf], None)]),
    }, metadata={"value_aliases": {6: 5, 8: 7}})
    module = IRModule({"root": function})
    module.record_tables["root"] = SSARecordTable(records={
        5: SSARecordDescriptor(5, "Metrics", (
            SSARecordFieldDescriptor("value", "scalar", value_ids=(0,), dtype="float64"),
        )),
    })
    findings = CorrelationTable.build(module).findings(module)
    missing = [finding.value_id for finding in findings if finding.kind == "alias-target-missing"]
    assert missing == [8]
    function.blocks["entry"].instrs = [Instr("Ret", [SSAValue(5, "unknown")], None)]
    findings = CorrelationTable.build(module).findings(module)
    assert any(finding.kind == "operand-never-written" and finding.value_id == 5 for finding in findings)


@pytest.mark.parametrize('unresolved_side', ['target', 'source', 'both'])
def test_capture_audit_reports_unresolved_binding_instead_of_crashing(unresolved_side):
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Unresolved, NEW
    from src.compiler.concordance_declarations import BindingFact
    from src.transmogrifier.ssa import IRModule
    book, token = begin_identity_book()
    try:
        fact = BindingFact(7, True, (), '')
        unresolved = Unresolved(NEW)
        book.page('name_binding').set(('scope', 'target', 0), 0,
                                      unresolved if unresolved_side in {'target', 'both'} else fact)
        book.page('name_binding').set(('scope', 'source', 0), 0,
                                      unresolved if unresolved_side in {'source', 'both'} else fact)
        book.page('name_binding_capture').set(('scope', 'target', 0), 0, ('source', 0, None))
        module = IRModule({}, metadata={'identity_book': book})
        findings = CorrelationTable._name_binding_capture_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == 'name-binding-capture-disagreement'
        assert 'Unresolved' in findings[0].detail
    finally:
        end_identity_book(token)


def test_value_type_abi_ignores_an_unbased_dense_id_from_another_scope():
    """A dense integer is not enough to associate two concordance rows."""
    from src.compiler.concordance_declarations import (
        CANONICAL_VALUE,
        INGESTION,
        INGEST_SOURCE,
        LINKED_VALUE_ABI_SETTLEMENT,
        PARAMETER_ANNOTATION,
        REDUCTION,
        SOURCE_VALUE_CLASS,
        VALUE_TYPE_ABI,
        ValueTypeABIAlternative,
        ValueTypeABIFact,
    )
    from src.compiler.identity_concordance import (
        Derived,
        Mode,
        Novel,
        begin_identity_book,
        end_identity_book,
    )
    from src.transmogrifier.ssa import IRModule

    book, token = begin_identity_book()
    try:
        value_id = 7
        unrelated_scope = ("unrelated", 0)
        abi_scope = ("accepted", 0)
        annotation_row = ("unrelated", "function", "value")
        annotation_cell = book.post(
            PARAMETER_ANNOTATION,
            annotation_row,
            "int",
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        book.post(
            SOURCE_VALUE_CLASS,
            (unrelated_scope, value_id),
            ("builtins.int", 1, "unrelated dense id"),
            stage=REDUCTION,
            provenance=Derived((annotation_cell,)),
            mode=Mode.CONCORD,
        )
        canonical_row = (abi_scope, value_id)
        canonical_cell = book.post(
            CANONICAL_VALUE,
            canonical_row,
            ("value",),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        fact = ValueTypeABIFact(("builtins.int",), (), (
            ValueTypeABIAlternative(
                "builtins.int", "scalar", "int64", (), 0,
                (("primitive_class_abi", "int"),),
                (canonical_cell,),
            ),
        ))
        book.post(
            VALUE_TYPE_ABI,
            canonical_row,
            fact,
            stage=LINKED_VALUE_ABI_SETTLEMENT,
            provenance=Derived((canonical_cell,)),
            mode=Mode.REVISE,
        )
        module = IRModule({}, metadata={"identity_book": book})

        assert CorrelationTable._value_type_abi_findings(module) == []

        book.post(
            SOURCE_VALUE_CLASS,
            (abi_scope, value_id),
            ("builtins.int", 1, "same-scope declaration"),
            stage=REDUCTION,
            provenance=Derived((annotation_cell,)),
            mode=Mode.CONCORD,
        )
        findings = CorrelationTable._value_type_abi_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == "value-type-abi-disagreement"
    finally:
        end_identity_book(token)


def test_value_type_abi_accepts_direct_post_ssa_value_identity():
    """A transformed generation is identified by its exact SSA value cell."""
    from src.compiler.concordance_declarations import (
        CONTROL_SSA_FINISH,
        INGEST_SOURCE,
        LINKED_VALUE_ABI_SETTLEMENT,
        SSA_VALUE,
        VALUE_TYPE_ABI,
        SSAValueFact,
        SSAValueOrigin,
        ValueTypeABIAlternative,
        ValueTypeABIFact,
    )
    from src.compiler.identity_concordance import (
        Derived,
        Mode,
        Novel,
        begin_identity_book,
        end_identity_book,
    )
    from src.transmogrifier.ssa import IRModule

    book, token = begin_identity_book()
    try:
        value_id = 2305843010213693986
        ssa_scope = "control"
        abi_scope = ("lexical", 0)
        value_cell = book.post(
            SSA_VALUE,
            (ssa_scope, value_id),
            SSAValueFact("int64", (), SSAValueOrigin.MINTED),
            stage=CONTROL_SSA_FINISH,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        fact = ValueTypeABIFact(("builtins.int",), (), (
            ValueTypeABIAlternative(
                "builtins.int", "scalar", "int64", (), 0,
                (("primitive_class_abi", "int"),), (value_cell,),
            ),
        ))
        book.post(
            VALUE_TYPE_ABI,
            (abi_scope, value_id),
            fact,
            stage=LINKED_VALUE_ABI_SETTLEMENT,
            provenance=Derived((value_cell,)),
            mode=Mode.REVISE,
        )
        module = IRModule({}, metadata={"identity_book": book})

        assert CorrelationTable._value_type_abi_findings(module) == []
    finally:
        end_identity_book(token)


@pytest.mark.parametrize('already_removed', [False, True])
def test_removed_binding_retires_its_capture_with_the_same_provenance(already_removed):
    from types import SimpleNamespace
    import networkx as nx
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Unresolved
    from src.compiler.concordance_declarations import BindingFact, BINDING_VERSION_REMOVED
    from src.compiler.planner_concordance_posts import _post_identity_table_mutation
    from src.transmogrifier.ssa import IRModule
    book, token = begin_identity_book()
    try:
        graph = SimpleNamespace(G=nx.DiGraph())
        graph.G.graph['lexical_read_scope'] = ('scope', 0)
        scope = ('scope', 0)
        for name in ('target', 'source'):
            book.page('name_binding').set((scope, name, 0), 0,
                Unresolved(BINDING_VERSION_REMOVED) if already_removed
                else BindingFact(310, True, (), ''))
        row = (scope, 'target', 0)
        book.page('canonical_value').set((scope, 310), 0, ('removed',))
        book.page('name_binding_capture').set(row, 0, ('source', 0, None))
        _post_identity_table_mutation(graph, removed=(310,))
        retired = book.page('name_binding_capture').latest(row)
        assert isinstance(retired, Unresolved) and retired.reason == BINDING_VERSION_REMOVED
        assert retired.read
        module = IRModule({}, metadata={'identity_book': book})
        assert CorrelationTable._name_binding_capture_findings(module) == []
        history = book.page('name_binding_capture').history(row)
        _post_identity_table_mutation(graph, removed=(310,))
        assert book.page('name_binding_capture').history(row) == history
    finally:
        end_identity_book(token)


@pytest.mark.parametrize('declare_root_types', [False, True])
def test_controller_audit_preserves_existing_restore_type_contract(declare_root_types):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/audit_identity_concordance.py'))
    module = namespace['_lower_controller'](declare_root_types=declare_root_types)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    # The whole actual controller was still included, not replaced by a stub.
    assert any('pi_update' in name for name in module.functions)
    assert any('update_dt_max' in name for name in module.functions)


@pytest.mark.parametrize('case', ['energy', 'toplevel'])
def test_audit_has_no_undefined_values_or_unposted_minted_ids(case):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/audit_identity_concordance.py'))
    module = namespace['_lower_' + case]()
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    _, identities = CorrelationTable.build(module)._unsourced_worklist(module)
    assert identities == [], report
    if case == 'toplevel':
        assert any('distribution_override' in name for name in module.functions)
        assert any('_energy_time_limit' in name for name in module.functions)
        root = next(function for name, function in module.functions.items() if name.endswith('__root'))
        assert any(instruction.op == 'CondBr' for block in root.blocks.values()
                   for instruction in block.instrs)
