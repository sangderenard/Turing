"""Intrinsic default signatures retain the public wrapper and authored source."""

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.concordance_declarations import (
    INGESTION, SOURCE_CALLABLE_SIGNATURE, SOURCE_CALL_PARAMETER_DEFAULT,
    SOURCE_INTRINSIC_ADMISSION,
    SOURCE_PARAMETER_DEFAULT, SOURCE_SPAN,
    STATIC_SYMBOL_DEFINITION, SpecializationFact, SpecializationSource,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book,
    current_identity_book, end_identity_book,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import hierarchical_plan
from src.transmogrifier.graph.graph_express2 import (
    ProcessGraph, ingest_callable_signature_by_reference,
)


def _plain_required_and_default(value, offset=7):
    return value + offset


def test_wrapped_zeros_signature_reaches_authored_default():
    book, token = begin_identity_book()
    try:
        signature = ingest_callable_signature_by_reference(AbstractTensor.zeros)
        assert signature is not None
        assert signature.page == SOURCE_CALLABLE_SIGNATURE
        definition, chain, formals = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
            signature.row, signature.column,
        )
        assert signature.row[0].page == STATIC_SYMBOL_DEFINITION
        assert definition.page == SOURCE_SPAN
        assert definition.row[:2] == (
            "src.common.tensors.abstraction_methods.creation", "zeros",
        )
        assert len(chain) == 2
        assert chain[0][:2] == chain[-1][:2] == definition.row[:2]
        assert chain[0][2] != chain[-1][2]
        defaults = {name: default for name, _kind, _index, default in formals}
        assert defaults["size"] is None
        for name, expected in (
            ("dtype", None), ("device", None), ("cls", None),
            ("requires_grad", False), ("tape", None),
        ):
            default = defaults[name]
            assert default is not None and default.page == SOURCE_PARAMETER_DEFAULT
            assert book.page(SOURCE_PARAMETER_DEFAULT).cell(
                default.row, default.column,
            ) == SpecializationFact(expected, SpecializationSource.DEFAULT)
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_callable_signature_findings(module) == []
        assert CorrelationTable._source_intrinsic_admission_findings(module) == []
        assert CorrelationTable._source_parameter_default_findings(module) == []
    finally:
        end_identity_book(token)


def test_plain_signature_has_no_default_for_required_formal():
    book, token = begin_identity_book()
    try:
        signature = ingest_callable_signature_by_reference(
            _plain_required_and_default,
        )
        assert signature is not None
        definition, chain, formals = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
            signature.row, signature.column,
        )
        assert len(chain) == 1
        assert definition.row[1] == "_plain_required_and_default"
        assert formals[0][0] == "value" and formals[0][3] is None
        assert formals[1][:3] == ("offset", "POSITIONAL_OR_KEYWORD", 1)
        assert formals[1][3] is not None
        assert book.latest_ref(
            SOURCE_PARAMETER_DEFAULT, (definition.row[:2], "value"),
        ) is None
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_callable_signature_findings(module) == []
    finally:
        end_identity_book(token)


def test_live_default_must_equal_authored_default():
    original = _plain_required_and_default.__defaults__
    book, token = begin_identity_book()
    try:
        _plain_required_and_default.__defaults__ = (11,)
        assert ingest_callable_signature_by_reference(
            _plain_required_and_default,
        ) is None
        assert tuple(book.page(SOURCE_CALLABLE_SIGNATURE).rows()) == ()
    finally:
        _plain_required_and_default.__defaults__ = original
        end_identity_book(token)


def test_intrinsic_pursuit_boundary_keeps_zeros_signature():
    book, token = begin_identity_book()
    try:
        policy = ExtractionContract(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts/program_extraction.yaml"
        )
        graph = ProcessGraph(materialize_memory=False)
        authored = ast.parse(
            "def root(like, count: int):\n"
            "    return AbstractTensor.zeros((count,), cls=type(like))\n"
        )
        call = authored.body[0].body[0].value
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                authored,
                resolve_unresolved_parents=True,
                parent_include=policy,
                parent_bindings={"AbstractTensor": AbstractTensor},
                pursuit_roots=("root",),
            )
        signatures = tuple(book.page(SOURCE_CALLABLE_SIGNATURE).rows())
        assert len(signatures) == 1
        signature = book.latest_ref(SOURCE_CALLABLE_SIGNATURE, signatures[0])
        assert call._intrinsic_signature_cell == signature
        admissions = tuple(book.page(SOURCE_INTRINSIC_ADMISSION).rows())
        assert len(admissions) == 1
        assert book.page(SOURCE_INTRINSIC_ADMISSION).latest(
            admissions[0],
        ) == ("zeros", call._extraction_contract, signature)
        _definition, chain, formals = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
            signature.row, signature.column,
        )
        assert chain[0][:2] == (
            "src.common.tensors.abstraction_methods.creation", "zeros",
        )
        assert next(default for name, _kind, _index, default in formals
                    if name == "dtype") is not None
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_callable_signature_findings(module) == []
        assert CorrelationTable._source_intrinsic_admission_findings(module) == []
    finally:
        end_identity_book(token)


def test_signature_row_refuses_wrapper_digest_substitution():
    book, token = begin_identity_book()
    try:
        signature = ingest_callable_signature_by_reference(AbstractTensor.zeros)
        definition, chain, formals = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
            signature.row, signature.column,
        )
        forged_chain = ((chain[0][0], chain[0][1], "wrong-digest"), *chain[1:])
        causes = tuple(source for source, _stage in book.edges_into(signature))
        with pytest.raises(ValueError):
            book.post(
                SOURCE_CALLABLE_SIGNATURE, signature.row,
                (definition, forged_chain, formals),
                stage=INGESTION, provenance=Derived(causes), mode=Mode.CONCORD,
            )
    finally:
        end_identity_book(token)


@pytest.mark.parametrize(
    ("call", "has_dtype_default"),
    [
        ("AbstractTensor.zeros((count,), cls=type(like))", True),
        ("AbstractTensor.zeros((count,), dtype='float64', cls=type(like))", False),
    ],
)
def test_full_source_intrinsic_default_at_region_boundary(
    monkeypatch, call, has_dtype_default,
):
    class ReachedZerosRegion(Exception):
        pass

    observed = []
    original = hierarchical_plan.plan_region_to_ssa_instrs

    def inspect_region(region, **kwargs):
        if any(getattr(item, "opcode", None) == "zeros"
               for item in region.items):
            book = current_identity_book()
            rows = tuple(book.page(SOURCE_CALL_PARAMETER_DEFAULT).rows())
            dtype_rows = tuple(row for row in rows if row[1] == "dtype")
            assert bool(dtype_rows) is has_dtype_default
            module = SimpleNamespace(metadata={"identity_book": book})
            assert CorrelationTable._source_callable_signature_findings(module) == []
            assert CorrelationTable._source_intrinsic_admission_findings(module) == []
            assert CorrelationTable._source_call_parameter_default_findings(module) == []
            observed.append((len(rows), len(dtype_rows)))
            raise ReachedZerosRegion
        return original(region, **kwargs)

    monkeypatch.setattr(
        hierarchical_plan, "plan_region_to_ssa_instrs", inspect_region,
    )
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "like", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(ReachedZerosRegion):
        lower_ast_source_to_ssa(
            f"def root(like, count):\n    return {call}\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="intrinsic_signature_default", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
    assert len(observed) == 1
