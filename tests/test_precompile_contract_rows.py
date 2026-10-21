"""Precompile producers post their physical facts and shape contracts through
``book.post``; accounting is then a view of the posted fact.

Each test reaches one construct of ``precompile_to_ssa`` whose producer used
to write ``program_abi_storage``/``program_abi_rank`` into accounting with no
post on the value's ``ssa_value`` row (audit ``ssa-physical-unposted``), or
wrote a ``tensor_shape_concordance`` row past the closed latch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGESTION,
    INGEST_SOURCE,
    SSA_VALUE,
    TENSOR_SHAPE_CONCORDANCE,
    VALUE_TYPE_ABI,
    SSAValueFact,
    SSAValueOrigin,
    ValueTypeABIAlternative,
    ValueTypeABIFact,
)
from src.compiler.control_source import (
    ControlProgram,
    LoopBlock,
    SequenceBlock,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable,
    Derived,
    Mode,
    Novel,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.compiler.precompile_to_ssa import (
    _ControlSSABuilder,
    lower_control_program_to_ssa,
    lower_control_sections_to_ssa,
)
from src.compiler.ssa_record_return_state import function_scope_of
from src.common.tensors.fused_ir import Meta
from src.transmogrifier.ssa import IRModule, SSAValue

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / (
    "program_extraction.yaml"
)

_UNSOURCED = {"unsourced-fact", "unsourced-identity"}


@pytest.fixture(autouse=True)
def _isolated_book():
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)


def _lower(source, entry, **kwargs):
    return lower_ast_source_to_ssa(
        source, entry,
        extraction_contract=ExtractionContract(CONTRACT),
        progress=lambda _message: None, progress_bars=False, **kwargs,
    )[0]


def _book_of(module):
    return (module.metadata or {}).get("identity_book") or (
        current_identity_book()
    )


def _physical_findings(module, *, function=None):
    """Findings the producers in this file are answerable for: unposted or
    disagreeing storage/rank, and anything unsourced."""

    found = []
    for finding in CorrelationTable.build(module).findings(module):
        if function is not None and finding.function != function:
            continue
        if finding.kind in _UNSOURCED:
            found.append(finding)
        elif finding.kind == "ssa-physical-unposted" and (
            "has no storage" in finding.detail
            or "has no rank" in finding.detail
        ):
            found.append(finding)
        elif (
            finding.kind == "ssa-publication-type-disagreement"
            and "accounting" in finding.detail
            and ("storage" in finding.detail or "rank" in finding.detail)
        ):
            found.append(finding)
    return found


def _fact(book, function, value):
    ref = book.latest_ref(SSA_VALUE, (function_scope_of(function), int(value.id)))
    assert ref is not None, f"no ssa_value row for {int(value.id)}"
    fact = book.page(SSA_VALUE).cell(ref.row, ref.column)
    assert isinstance(fact, SSAValueFact)
    return fact


def _assert_accounting_follows_fact(book, function, value, storage, rank=None):
    fact = _fact(book, function, value)
    assert fact.storage == storage
    if rank is not None:
        assert fact.rank == rank
    assert (value.accounting or {}).get("program_abi_storage") == fact.storage
    return fact


def _formals(function, key):
    return [value for value in function.args if key in (value.accounting or {})]


# --- site: native union parameter (shared arm / distinct arms) -------------


def test_union_parameter_shared_arm_posts_its_physical_view():
    module = _lower(
        "def f(x: int | None, n: int) -> int:\n"
        "    if x is None:\n"
        "        return n\n"
        "    return x + n\n",
        "f",
    )
    function = next(iter(module.functions.values()))
    book = _book_of(module)
    (shared,) = _formals(function, "native_value_union_shared_parameter")
    _assert_accounting_follows_fact(book, function, shared, "scalar", 0)
    assert _physical_findings(module) == []


def test_union_parameter_distinct_arms_post_their_physical_views():
    scope = ("distinct_union_parameter",)
    cell = current_identity_book().post(
        SSA_VALUE, (scope, 1),
        SSAValueFact("int64", (), SSAValueOrigin.MINTED),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )
    physical = (
        ValueTypeABIAlternative(
            "builtins.int", "scalar", "int64", (), 0,
            (("primitive_class_abi", "int"),), (cell,),
        ),
        ValueTypeABIAlternative(
            "builtins.float", "scalar", "float64", (), 0,
            (("primitive_class_abi", "float"),), (cell,),
        ),
    )
    fact = ValueTypeABIFact(
        tuple(arm.identity for arm in physical),
        (("builtins.int", 0), ("builtins.float", 1)), physical,
    )
    current_identity_book().post(
        VALUE_TYPE_ABI, (scope, 1), fact,
        stage=INGESTION, provenance=Derived((cell,)), mode=Mode.CONCORD,
    )

    function, shortfalls = lower_control_program_to_ssa(
        ControlProgram(SequenceBlock(())),
        function_name="distinct_union_parameter",
        lexical_read_scope=scope,
        region_value_meta={1: Meta((), "int64")},
        value_name_histories={"x": (1,)},
        parameter_names=("x",),
        native_value_type_abi={1: fact},
    )

    assert shortfalls == ()
    book = current_identity_book()
    arms = _formals(function, "native_value_union_alternative")
    assert len(arms) == 2
    for payload in arms:
        _assert_accounting_follows_fact(book, function, payload, "scalar", 0)
        ref = book.latest_ref(
            SSA_VALUE, (function_scope_of(function), int(payload.id)),
        )
        # The mint cites the ``value_type_abi`` row it read the arm from.
        assert book.mint_of(ref) is not None
    module = IRModule({function.name: function})
    assert [
        finding for finding in CorrelationTable.build(module).findings(module)
        if finding.kind == "ssa-physical-unposted"
        and ("has no storage" in finding.detail or "has no rank" in finding.detail)
    ] == []


# --- site: native reference handle cast -----------------------------------


def test_reference_key_cast_posts_scalar_rank_zero_at_the_mint():
    from src.transmogrifier.ssa import SSASequenceDescriptor

    builder = _ControlSSABuilder(
        ControlProgram(SequenceBlock(())),
        function_name="reference_table_query",
        first_value_id=100,
        region_callees={},
        region_signatures={},
    )
    builder.external_values[7] = SSAValue(
        7, dtype="opaque_ref",
        accounting={"native_exact_type": "builtins.str"},
    )
    descriptor = SSASequenceDescriptor(
        sequence_id=20,
        column_value_ids=(21, 22),
        length_address_id=23,
        capacity_value_id=24,
        status_address_id=25,
        column_dtypes=("int64", "int64"),
        key_columns=(0,),
    )

    (handle,) = builder._table_query_values(7, descriptor)

    book = current_identity_book()
    ref = book.latest_ref(SSA_VALUE, (builder._scope(), int(handle.id)))
    fact = book.page(SSA_VALUE).cell(ref.row, ref.column)
    assert (fact.storage, fact.rank) == ("scalar", 0)
    assert handle.accounting["program_abi_storage"] == fact.storage
    assert handle.accounting["program_abi_rank"] == fact.rank
    # Declared at the mint: Novel, not a later rewrite.
    assert book.mint_of(ref) is not None


# --- site: union tag column projections -----------------------------------


def test_union_tag_column_projections_post_span_rank_one():
    scope = ("union_tag_columns",)
    target_id = 14
    cell = current_identity_book().post(
        SSA_VALUE, (scope, target_id),
        SSAValueFact("opaque_ref", (), SSAValueOrigin.MINTED),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )
    physical = (
        ValueTypeABIAlternative(
            "builtins.int", "scalar", "int64", (), 0,
            (("primitive_class_abi", "int"),), (cell,),
        ),
        ValueTypeABIAlternative(
            "builtins.float", "scalar", "float64", (), 0,
            (("primitive_class_abi", "float"),), (cell,),
        ),
    )
    fact = ValueTypeABIFact(
        tuple(arm.identity for arm in physical),
        (("builtins.int", 0), ("builtins.float", 1)), physical,
    )
    current_identity_book().post(
        VALUE_TYPE_ABI, (scope, target_id), fact,
        stage=INGESTION, provenance=Derived((cell,)), mode=Mode.CONCORD,
    )
    builder = _ControlSSABuilder(
        ControlProgram(SequenceBlock(())),
        function_name="union_tag_columns",
        first_value_id=100,
        region_callees=None,
        region_signatures=None,
        lexical_read_scope=scope,
        native_value_type_abi={target_id: fact},
    )

    source, _fact = builder._native_type_tag_sequence_column(
        20, 1, target_id,
    )

    book = current_identity_book()
    columns = [source, *(
        builder.variant_type_payload_columns[(20, 1, arm.identity)]
        for arm in physical
    )]
    assert all(column is not None for column in columns)
    for column in columns:
        ref = book.latest_ref(SSA_VALUE, (builder._scope(), int(column.id)))
        row = book.page(SSA_VALUE).cell(ref.row, ref.column)
        assert (row.storage, row.rank) == ("span", 1)
        assert column.accounting["program_abi_storage"] == row.storage
        assert column.accounting["program_abi_rank"] == row.rank


# --- site: field-slot receiver columns ------------------------------------


def test_field_slot_columns_post_their_physical_view():
    handle = -734_921
    module, shortfalls, _ = lower_control_sections_to_ssa(
        ControlProgram(SequenceBlock(())),
        identity_table={"self": (5,)},
        self_value_id=5,
        field_ops=(("write", 11, 0), ("write", 12, 1)),
        field_const_sources={
            11: {
                "ssa_reference_identity": "autograd.tape",
                "reference_kind": "static-python",
                "reference_handle": handle,
                "host_resident": True,
            },
            12: 2.5,
        },
        field_count=2,
        field_names=("_tape", "gain"),
        record_identity="AbstractTensor",
        record_field_dtypes={"gain": "float64"},
        record_field_contracts={
            "_tape": {"storage": "reference", "mutable": True},
            "gain": {
                "storage": "scalar", "dtype": "float64", "mutable": True,
            },
        },
    )

    assert shortfalls == ()
    function = module.functions["planned_control"]
    book = current_identity_book()
    by_field = {
        value.accounting["program_abi_field"]: value
        for value in function.args
        if "program_abi_field" in (value.accounting or {})
    }
    _assert_accounting_follows_fact(book, function, by_field["_tape"], "reference", 0)
    _assert_accounting_follows_fact(book, function, by_field["gain"], "scalar", 0)
    assert _physical_findings(module) == []


# --- site: planned region formal of a resident sequence arena -------------


def test_region_sequence_arena_formal_posts_span_rank():
    module = _lower(
        "def f(a: list[float], n: int) -> float:\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        j = 0\n"
        "        while j < n:\n"
        "            a[j] = a[j] + 1.0\n"
        "            j = j + 1\n"
        "        i = i + 1\n"
        "    return a[0]\n",
        "f",
    )
    book = _book_of(module)
    seen = 0
    for function in module.functions.values():
        for formal in _formals(function, "sequence_row_shape"):
            if (formal.accounting or {}).get("program_abi_storage") != "span":
                continue
            fact = _fact(book, function, formal)
            assert fact.storage == "span"
            assert fact.rank == formal.accounting["program_abi_rank"]
            seen += 1
    assert seen
    assert _physical_findings(module) == []


# --- site: keyed Tensor lookup --------------------------------------------


def test_keyed_tensor_lookup_posts_span_rank_derived_from_its_contract():
    from tests.test_sequence_contract_concordance import (
        c_backend_repository_ssa_reference,
    )

    module = lower_ast_source_to_ssa(
        "from dataclasses import dataclass\n"
        "from typing import Mapping\n"
        "import torch\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Reading:\n"
        "    voltage_v: Mapping[str, torch.Tensor]\n"
        "\n"
        "def sample(reading: Reading, key: str):\n"
        "    selected = reading.voltage_v[key]\n"
        "    return selected.real.sum()\n",
        "sample",
        name="keyed_tensor_rows",
        extraction_contract=ExtractionContract(CONTRACT),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )[0]
    root = module.functions["keyed_tensor_rows__sample"]
    book = _book_of(module)
    row_base = next(
        instruction.res
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "keyed_tensor_row_base"
    )
    fact = _assert_accounting_follows_fact(book, root, row_base, "span")
    contract_row = (
        root.metadata["tensor_shape_concordance_scope"], int(row_base.id),
    )
    assert fact.rank == book.page(TENSOR_SHAPE_CONCORDANCE).latest(
        contract_row
    )["program_abi_rank"]
    contract_ref = book.latest_ref(TENSOR_SHAPE_CONCORDANCE, contract_row)
    assert book.mint_of(contract_ref) is None
    assert book.edges_into(contract_ref)
    assert _physical_findings(module, function=root.name) == []


# --- site: loop resident shape contract (was a raw page write) -------------


def test_loop_resident_shape_contract_is_posted_derived():
    scope = ("loop_resident", 0)
    book = current_identity_book()
    canonical = {}
    for value_id in (1, 2):
        book.post(
            CANONICAL_VALUE, (scope, value_id),
            ("Input", "input", str(value_id)),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        canonical[value_id] = book.latest_ref(CANONICAL_VALUE, (scope, value_id))
    control = ControlProgram(LoopBlock(
        "i", "0", "3", "1", SequenceBlock(()),
        carried_aliases=((2, 1),), result_ports=((3, 1, 2),),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        lexical_read_scope=scope,
        region_signatures={},
        region_value_meta={
            1: Meta((4,), "float64"), 2: Meta((4,), "float64"),
        },
        value_aliases={2: 1},
    )

    assert shortfalls == ()
    page = book.page(TENSOR_SHAPE_CONCORDANCE)
    control_scope = function.metadata["tensor_shape_concordance_scope"]
    rows = [row for row in page.rows() if row[0] == control_scope]
    assert {row[1] for row in rows} == {1, 2}
    for row in rows:
        contract = page.latest(row)
        assert contract["source"] == "concorded-loop-resident-meta"
        assert (contract["program_abi_storage"], contract["program_abi_rank"]) == (
            "span", 1,
        )
        ref = book.latest_ref(TENSOR_SHAPE_CONCORDANCE, row)
        # Derived through ``book.post`` from the value's canonical cell.
        assert book.mint_of(ref) is None
        assert canonical[row[1]] in {
            source for source, _stage in book.edges_into(ref)
        }
    module = IRModule({function.name: function})
    assert [
        finding for finding in CorrelationTable.build(module).findings(module)
        if finding.kind in _UNSOURCED
    ] == []
