"""The orbital transfer sympy set as the compiler's quality benchmark.

The set is ``Orbit.stable_orbit_transfer_solution(Orbit.symbolic_orbit('1'),
Orbit.symbolic_orbit('2'))`` exactly as ``graph_express2_tests`` builds it.
The compiler must take it AS WRITTEN.  The builder, the per-law route and
the work list live in ``tools/compiler_probes/probe_orbital_transfer.py``;
this file only asserts them.

Per law (entry/side, plus the raw Equalities): ``compile_sympy_equations``
-> ``piece_from_law`` (``lower_ast_source_to_ssa``, LLVM emitted and
compiled) -> C emitted and compiled -> both native lanes match the sympy
reference within 1e-12 relative.  A law the open work list still blocks is
``xfail(strict=True)`` with that work item as the reason: a fix flips it to
XPASS, which fails, which forces the mark (and the probe's ``LAW_BLOCKERS``
entry) to be removed.

    python -m pytest tests/test_orbital_transfer_compile.py -q
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import sympy as sp

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

import probe_orbital_transfer as orbital  # noqa: E402

PROGRAM = orbital.build_program()
LAWS = orbital.benchmark_laws(PROGRAM)


def _case(law):
    items = orbital.LAW_BLOCKERS.get(law)
    marks = () if not items else (
        pytest.mark.xfail(strict=True, reason=orbital.blocker_reason(items)),)
    return pytest.param(law, id=law, marks=marks)


@pytest.mark.parametrize("law", [_case(law) for law in LAWS])
def test_law_compiles_and_matches_reference(law):
    failures: list = []
    piece = orbital.run_law(law, LAWS[law], failures,
                            orbital.reference_bindings(PROGRAM))
    assert piece is not None and not failures, "\n".join(
        f"{failure.stage}: {failure.frame}: {failure.error[:400]}"
        for failure in failures)


def test_work_list_is_the_set_of_xfail_reasons():
    assert set(orbital.LAW_BLOCKERS) <= set(LAWS)
    assert not any(" | " in text for text in orbital.WORK_ITEMS.values())
    reasons = {
        reason
        for items in orbital.LAW_BLOCKERS.values()
        for reason in orbital.blocker_reason(items).split(" | ")}
    expected = {f"work item {item}: {text}" for item, text in orbital.WORK_ITEMS.items()}
    assert reasons == expected


def test_matrix_equality_declares_one_output_per_component_on_the_book():
    """Work item 1 at the identity: the raw ``initial_condition`` Equality,
    ``Matrix([r1(0), r2(0), r3(0)]) = r_start``, as written."""
    from src.compiler.concordance_declarations import (
        SYMBOLIC_EQUATION, SYMBOLIC_EQUATION_OUTPUT, SymbolicOutputForm,
    )
    from src.compiler.identity_concordance import current_identity_book
    from src.compiler.symbolic_equation_compiler import (
        compile_sympy_equations, symbolic_program_scope,
    )

    name = "orbital_initial_condition_raw"
    compilation = compile_sympy_equations([PROGRAM["initial_condition"]], name=name)
    metadata = compilation.function.metadata
    outputs = [f"{name}_0_{row}_0" for row in range(3)]
    columns = [f"r_start_{row}_0" for row in range(3)]
    assert list(metadata["output_names"]) == outputs
    assert [row[3] for row in metadata["symbolic_outputs"]] == ["residual"] * 3
    assert metadata["matrix_element_inputs"] == tuple(
        (column, "r_start", (3, 1), (row, 0)) for row, column in enumerate(columns))
    assert set(columns) <= set(metadata["argument_names"])

    book = current_identity_book()
    program = symbolic_program_scope(compilation, name)
    equation = book.latest_ref(SYMBOLIC_EQUATION, (program, 0))
    assert equation is not None
    for row, output in enumerate(outputs):
        cell = book.latest_ref(SYMBOLIC_EQUATION_OUTPUT, (program, output))
        assert cell is not None
        fact = book.page(SYMBOLIC_EQUATION_OUTPUT).latest((program, output))
        assert (fact.equation_index, fact.component, fact.form) == (
            0, (row, 0), SymbolicOutputForm.RESIDUAL)
        assert [source for source, _stage in book.edges_into(cell)] == [equation]
