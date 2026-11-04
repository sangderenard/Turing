"""A call-only tensor seed keeps the exact authored span ABI on its SSA row."""

from dataclasses import replace
from types import SimpleNamespace

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, INGESTION, INGEST_SOURCE, SSA_VALUE, VALUE_TYPE_ABI,
    ValueTypeABIAlternative, ValueTypeABIFact,
)
from src.compiler.control_source import ControlProgram, SequenceBlock
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, Novel, Ref,
    begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
from src.compiler.ssa_record_return_state import function_scope_of


def test_singleton_span_seed_refuses_erased_physical_fields():
    book, token = begin_identity_book()
    try:
        lexical = ("seed_span_audit", 0)
        canonical = book.post(
            CANONICAL_VALUE, (lexical, 2), ("Input", "octets"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        arm = ValueTypeABIAlternative(
            "AbstractTensor", "span", "int64", (), 1,
            (("program_abi", "octets"),), (canonical,),
            metadata_state="dynamic",
        )
        abi_cell = book.post(
            VALUE_TYPE_ABI, (lexical, 2),
            ValueTypeABIFact((arm.identity,), ((arm.identity, 0),), (arm,)),
            stage=INGESTION, provenance=Derived((canonical,)),
            mode=Mode.CONCORD,
        )
        function, shortfalls = lower_control_program_to_ssa(
            ControlProgram(SequenceBlock(())),
            function_name="seed_span_audit", lexical_read_scope=lexical,
            value_name_histories={"octets": (2,)},
            parameter_names=("octets",), region_value_meta={},
        )
        assert shortfalls == ()
        module = SimpleNamespace(
            functions={function.name: function},
            metadata={"identity_book": book},
        )
        findings = CorrelationTable._singleton_span_parameter_seed_findings
        assert findings(module) == []
        cell = book.latest_ref(SSA_VALUE, (function_scope_of(function), 2))
        original = book.page(SSA_VALUE).cell(cell.row, cell.column)
        book.page(SSA_VALUE).cells[(cell.row, cell.column)] = replace(
            original, physical_dtype=None, storage=None, rank=None,
        )
        assert [finding.kind for finding in findings(module)] == [
            "singleton-span-parameter-seed-disagreement"]
        book.page(SSA_VALUE).cells[(cell.row, cell.column)] = original
        recorded_abi = book.page(VALUE_TYPE_ABI).cell(
            abi_cell.row, abi_cell.column)
        book.page(VALUE_TYPE_ABI).cells[(abi_cell.row, abi_cell.column)] = replace(
            recorded_abi, physical_alternatives=(replace(
                arm, source_cells=(Ref(CANONICAL_VALUE, (lexical, 999), 0),),
            ),),
        )
        assert [finding.kind for finding in findings(module)] == [
            "singleton-span-parameter-seed-disagreement"]
    finally:
        end_identity_book(token)
