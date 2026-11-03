"""Authored defaults are source facts used by planner formal bindings."""

import ast
import contextlib
import inspect
import io
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.common.tensors.abstraction import AbstractTensor
from src.compiler.concordance_declarations import (
    PLANNER_HIERARCHY, PLANNER_SPECIALIZATION_PAGE,
    SOURCE_PARAMETER_DEFAULT, SOURCE_SPAN,
    SpecializationFact, SpecializationSource,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book,
    end_identity_book,
)
from src.compiler.planner_concordance_posts import (
    _formal_identity_cells, _post_copy_planner_specializations,
    _post_planner_specialization,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


@pytest.fixture
def source_book():
    book, token = begin_identity_book()
    try:
        yield book
    finally:
        end_identity_book(token)


def _reduced_function(source: str, name: str):
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
    return graph.function_table.entry(name).graph


def test_positional_and_keyword_only_defaults_are_concorded(source_book):
    callee = _reduced_function(
        "def f(x, y=None, z=3, *, required, flag=False, absent=None):\n"
        "    return x\n",
        "f",
    )
    scope = tuple(callee.G.graph["authored_function_identity"])
    page = source_book.page(SOURCE_PARAMETER_DEFAULT)
    assert {row[1] for row in page.rows() if row[0] == scope} == {
        "y", "z", "flag", "absent",
    }
    for name, value in (("y", None), ("z", 3),
                        ("flag", False), ("absent", None)):
        ref = source_book.latest_ref(SOURCE_PARAMETER_DEFAULT, (scope, name))
        assert page.cell(ref.row, ref.column) == SpecializationFact(
            value, SpecializationSource.DEFAULT,
        )
        causes = tuple(source for source, _stage
                       in source_book.edges_into(ref))
        assert len(causes) == 3
        assert all(source.page == SOURCE_SPAN for source in causes)
        assert ref in _formal_identity_cells(callee, name)
    for name in ("x", "required"):
        assert source_book.latest_ref(
            SOURCE_PARAMETER_DEFAULT, (scope, name),
        ) is None
        assert not any(source.page == SOURCE_PARAMETER_DEFAULT
                       for source in _formal_identity_cells(callee, name))
    _post_copy_planner_specializations(
        callee, {"y": "stale graph default"}, {},
    )
    planned_row = (tuple(callee.G.graph["lexical_read_scope"]), "y")
    planned = source_book.latest_ref(PLANNER_SPECIALIZATION_PAGE, planned_row)
    assert source_book.page(PLANNER_SPECIALIZATION_PAGE).cell(
        planned.row, planned.column,
    ) == SpecializationFact(None, SpecializationSource.DEFAULT)
    authored = source_book.latest_ref(SOURCE_PARAMETER_DEFAULT, (scope, "y"))
    assert authored in {
        source for source, _stage in source_book.edges_into(planned)
    }
    module = SimpleNamespace(metadata={"identity_book": source_book})
    assert CorrelationTable._source_parameter_default_findings(module) == []


def test_planner_default_cites_authored_default_and_a_missing_cause_refuses(
    source_book,
):
    callee = _reduced_function(
        "def g(x, offset=4, *, enabled=False):\n"
        "    return x + offset\n",
        "g",
    )
    scope = tuple(callee.G.graph["authored_function_identity"])
    default = source_book.latest_ref(
        SOURCE_PARAMETER_DEFAULT, (scope, "offset"),
    )
    causes = _formal_identity_cells(callee, "offset")
    assert default in causes
    specialization = _post_planner_specialization(
        callee, "offset", SpecializationFact(4, SpecializationSource.DEFAULT),
        causes,
    )
    assert specialization is not None
    assert default in {
        source for source, _stage in source_book.edges_into(specialization)
    }

    source_spans = tuple(
        source for source, _stage in source_book.edges_into(default)
        if source.page == SOURCE_SPAN
    )
    source_book.post(
        SOURCE_PARAMETER_DEFAULT, (scope, "forged"),
        SpecializationFact(4, SpecializationSource.DEFAULT),
        stage=PLANNER_HIERARCHY,
        provenance=Derived(source_spans[:2]), mode=Mode.CONCORD,
    )
    module = SimpleNamespace(metadata={"identity_book": source_book})
    findings = CorrelationTable._source_parameter_default_findings(module)
    assert len(findings) == 1
    assert findings[0].kind == "source-parameter-default"


def test_lambda_defaults_keep_full_authored_scope(source_book):
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(
            "choose = lambda x, y=None, *, flag=False: x\n"
        ))
        reduce_abstract_tensor_topology(graph)
    page = source_book.page(SOURCE_PARAMETER_DEFAULT)
    rows = tuple(page.rows())
    assert {row[1] for row in rows} == {"y", "flag"}
    assert all(len(row[0]) == 3 for row in rows)
    assert {row[1]: page.latest(row).value for row in rows} == {
        "y": None, "flag": False,
    }
    module = SimpleNamespace(metadata={"identity_book": source_book})
    assert CorrelationTable._source_parameter_default_findings(module) == []


def test_omitted_call_argument_uses_authored_default_in_full_pipeline():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def choose(x: int, offset=3):\n"
        "    return x + offset\n"
        "def root(x: int):\n"
        "    return choose(x)\n",
        "root", name="authored_default_call",
        extraction_contract=contract,
        progress=lambda _message: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    rows = tuple(row for row in book.page(SOURCE_PARAMETER_DEFAULT).rows()
                 if row[1] == "offset" and row[0][1] == "choose")
    assert len(rows) == 1
    assert book.page(SOURCE_PARAMETER_DEFAULT).latest(rows[0]) == (
        SpecializationFact(3, SpecializationSource.DEFAULT)
    )
    assert CorrelationTable._source_parameter_default_findings(module) == []


def test_creation_zeros_authored_defaults_exist_at_source_ingestion(source_book):
    raw = inspect.unwrap(AbstractTensor.zeros)
    source = textwrap.dedent(inspect.getsource(raw))
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
    page = source_book.page(SOURCE_PARAMETER_DEFAULT)
    rows = tuple(row for row in page.rows() if row[0][1] == "zeros")
    assert {row[1] for row in rows} == {
        "dtype", "device", "cls", "requires_grad", "tape",
    }
    expected = {
        "dtype": None, "device": None, "cls": None,
        "requires_grad": False, "tape": None,
    }
    for row in rows:
        assert page.latest(row) == SpecializationFact(
            expected[row[1]], SpecializationSource.DEFAULT,
        )
    module = SimpleNamespace(metadata={"identity_book": source_book})
    assert CorrelationTable._source_parameter_default_findings(module) == []


