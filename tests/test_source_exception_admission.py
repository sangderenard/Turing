"""An authored catch must not disappear from a complete native artifact."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import CompilationSubdivisionRequired
from src.compiler.concordance_declarations import (
    CONTROL_BLOCK, ControlBlockKind,
    CONTROL_VALUE_BINDING, BindingKind,
    NAME_BINDING, SOURCE_EXCEPTION_ADMISSION, SOURCE_EXCEPTION_SITE_STATE,
    SOURCE_EXCEPTION_STRUCTURAL_RETIREMENT,
    SOURCE_EXCEPTION_TRANSFER, SOURCE_EXCEPTION_HANDLER_CLASS,
    SOURCE_EXCEPTION_INERT_CONSTRUCTOR,
    SOURCE_EXCEPTION_SSA_UNREACHABLE_EXIT,
    SOURCE_EXCEPTION_STATUS_ROUTE, NATIVE_STATUS_CLASS,
    SOURCE_EXCEPTION_REGION_SITE,
    SOURCE_EXCEPTION_REGION_STATE,
    SOURCE_EXCEPTION_SSA_CHECKPOINT,
    SOURCE_EXCEPTION_SSA_RETURN_FINALIZER,
    SOURCE_EXCEPTION_FINALIZER_BINDING,
    SOURCE_EXCEPTION_FINALIZER_INPUT,
    SOURCE_EXCEPTION_FINALIZER_OUTPUT,
    SOURCE_EXCEPTION_SSA_FINALIZER_OUTPUT,
    SOURCE_EXCEPTION_SSA_NORMAL_FINALIZER_INPUT,
    SOURCE_EXCEPTION_SSA_LOOP_EXIT_FINALIZER,
    SOURCE_EXCEPTION_SSA_FINALIZER_INPUT,
    SOURCE_EXCEPTION_SSA_ROUTE,
    SOURCE_EXCEPTION_SSA_CONTROL_CHECKPOINT,
    SOURCE_EXCEPTION_SSA_CONTROL_ROUTE,
    SOURCE_EXCEPTION_SSA_HANDLER_ARM,
    SOURCE_EXCEPTION_SSA_HANDLER_STATE,
    SOURCE_EXCEPTION_SSA_CONTINUATION_ARM,
    SOURCE_EXCEPTION_SSA_CONTINUATION_PHI,
    SOURCE_EXCEPTION_SSA_HANDLER_ENTRY,
    SOURCE_EXCEPTION_SSA_HANDLER_PHI,
    SOURCE_EXCEPTION_HANDLER_ENTRY,
    SOURCE_EXCEPTION_HANDLER_PHI,
    SOURCE_EXCEPTION_HANDLER_ARM_VALUE,
    SOURCE_EXCEPTION_CONTINUATION_PHI,
    SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Unresolved, begin_identity_book, end_identity_book,
)
from src.compiler.native_faults import FLOAT_TO_INT_NAN, FLOAT_TO_INT_INFINITY
from src.compiler.native_faults import (
    NativeAssertionError, NativeOverflowError, NativeValueError,
)


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _lower(source: str, name: str, *, complete=False, **kwargs):
    def lower():
        return lower_ast_source_to_ssa(
            source, "root", name=name, extraction_contract=CONTRACT,
            progress=lambda _message: None, progress_bars=False,
            **kwargs,
        )

    if complete or not any(
        isinstance(node, (ast.Try, ast.TryStar, ast.Raise, ast.Assert))
        for node in ast.walk(ast.parse(source))
    ):
        return lower()

    # These control-shape cases stop at the SSA boundary so they can inspect
    # the exact planned control tree and lower it with bounded region inputs.
    # The production path is exercised separately with complete=True.
    from src.compiler import fortran_c_shell

    def planned(compilation, *_args, **_kwargs):
        raise CompilationSubdivisionRequired(
            "bounded exception control planning probe",
            boundaries=getattr(
                compilation.deployment, "_source_exception_boundaries", (),
            ),
        )

    with patch.object(fortran_c_shell, "_class_surface_ssa_program", planned):
        return lower()


def test_structural_fold_retires_exact_rejected_raise():
    source = """\
def root(x: int) -> int:
    if isinstance(x, int):
        return x
    raise TypeError("bad")
"""
    module, _outputs, _exports = _lower(source, "root", complete=True)
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_EXCEPTION_STRUCTURAL_RETIREMENT)
    rows = tuple(page.rows())
    assert rows
    assert all(page.latest(row).transfer == row[1] for row in rows)
    assert not CorrelationTable._source_exception_control_coverage_findings(
        module
    )
    assert not CorrelationTable._source_exception_admission_findings(module)


def test_authored_catch_without_exceptional_control_refuses(monkeypatch):
    # Before admission, the C artifact for NaN returned INT64_MIN instead of
    # running the ValueError handler and returning 2.
    source = """\
def root(x: float) -> int:
    try:
        y = int(x)
    except ValueError:
        y = 2
    return y
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import NativeTryBlock

    overlaid = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            overlaid.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as error:
            _lower(source, "exception_admission_catch", identity_book=book)
        page = book.page(SOURCE_EXCEPTION_ADMISSION)
        row, = tuple(page.rows())
        source_cell, = row
        obligation = page.latest(row)
        assert isinstance(obligation, Unresolved)
        assert obligation.read == (source_cell,)
        receipt = book.latest_ref(SOURCE_EXCEPTION_ADMISSION, row)
        assert {cell for cell, _stage in book.edges_into(receipt)} == {
            source_cell
        }
        findings = CorrelationTable._source_exception_admission_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        assert len(findings) == 1
        assert "no executable exceptional control" in findings[0].detail
        try_blocks = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        assert len(try_blocks) == 1
        assert len(overlaid) == 1
        assert any(
            isinstance(block, NativeTryBlock)
            for block in overlaid[0].root.blocks
        )
        try_fact = book.page(CONTROL_BLOCK).latest(try_blocks[0])
        assert len(try_fact.carried) == 2
        assert try_fact.carried[1].page == SOURCE_EXCEPTION_CONTINUATION_PHI
        assert len(try_fact.sites) == 1
        assert try_fact.sites[0].page == SOURCE_EXCEPTION_TRANSFER
        region_rows = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        assert len(region_rows) == 1
        region_site = book.page(SOURCE_EXCEPTION_REGION_SITE).latest(
            region_rows[0]
        )
        assert region_site.transfer == try_fact.sites[0]
        assert region_site.canonical_site.row[1] == region_rows[0][2]
        assert region_site.deployment_member.row == region_rows[0]
        transfer_cell = region_site.transfer
        nan_route = book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest((
            *transfer_cell.row, FLOAT_TO_INT_NAN,
        ))
        infinity_route = book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest((
            *transfer_cell.row, FLOAT_TO_INT_INFINITY,
        ))
        assert nan_route.selected_handler is not None
        assert infinity_route.selected_handler is None
        arm_page = book.page(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE)
        arm_rows = tuple(sorted(arm_page.rows(), key=lambda row: row[2]))
        assert len(arm_rows) == 2
        arm_facts = tuple(arm_page.latest(row) for row in arm_rows)
        assert tuple(fact.arm_kind for fact in arm_facts) == (
            "success", "handler",
        )
        assert all(
            fact.source_phi == try_fact.carried[1]
            and fact.canonical_phi.row[1] == row[1]
            for row, fact in zip(arm_rows, arm_facts)
        )
        assert arm_facts[0].canonical_value != arm_facts[1].canonical_value
        source_module = SimpleNamespace(metadata={"identity_book": book})
        for check in (
            CorrelationTable._source_exception_site_state_findings,
            CorrelationTable._source_exception_transfer_findings,
            CorrelationTable._source_exception_region_site_findings,
            CorrelationTable._source_exception_handler_class_findings,
            CorrelationTable._source_exception_handler_entry_findings,
            CorrelationTable._source_exception_handler_phi_findings,
            CorrelationTable._source_exception_continuation_phi_findings,
            CorrelationTable._source_exception_continuation_arm_value_findings,
            CorrelationTable._source_exception_status_route_findings,
        ):
            assert check(source_module) == []
        from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

        lowered, shortfalls = lower_control_program_to_ssa(
            overlaid[0], function_name="exception_checkpoint_probe",
            first_value_id=1000,
            lexical_read_scope=try_blocks[0][0],
            planning_scope=region_rows[0][0],
            region_callees={region_rows[0][1]: "numerical_region_0"},
            region_signatures={region_rows[0][1]: (
                (), (int(arm_facts[0].canonical_value.row[1]),),
            )},
            constant_value_ids=(
                int(arm_facts[1].canonical_value.row[1]),
            ),
        )
        assert not shortfalls, shortfalls
        assert any(
            instruction.attributes.get("callee")
            == "turing_validation_error_take"
            for block in lowered.blocks.values()
            for instruction in block.instrs
        )
        assert CorrelationTable._source_exception_ssa_checkpoint_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        route_rows = tuple(book.page(SOURCE_EXCEPTION_SSA_ROUTE).rows())
        nan_route_row, = tuple(
            row for row in route_rows if row[3] == FLOAT_TO_INT_NAN
        )
        native_route = book.page(SOURCE_EXCEPTION_SSA_ROUTE).latest(
            nan_route_row
        )
        assert native_route.selected_handler == nan_route.selected_handler
        assert CorrelationTable._source_exception_ssa_route_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_ssa_handler_entry_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        continuation_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).rows())
        continuation = book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).latest(continuation_row)
        assert continuation.source_phi == try_fact.carried[1]
        assert len(continuation.incoming_arms) == 2
        assert tuple(
            book.page(SOURCE_EXCEPTION_SSA_CONTINUATION_ARM).latest(
                cell.row
            ).source_arm
            for cell in continuation.incoming_arms
        ) == tuple(
            book.latest_ref(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE, row)
            for row in arm_rows
        )
        join_block = lowered.blocks[continuation.join_block.row[2]]
        join_phi, = tuple(
            instruction for instruction in join_block.instrs
            if instruction.attributes.get("binding")
            == "native_exception_continuation_phi"
        )
        assert tuple(arg.id for arg in join_phi.args) == tuple(
            book.page(SOURCE_EXCEPTION_SSA_CONTINUATION_ARM).latest(
                cell.row
            ).incoming_value.row[1]
            for cell in continuation.incoming_arms
        )
        assert all(
            join_block.name in lowered.blocks[
                book.page(SOURCE_EXCEPTION_SSA_CONTINUATION_ARM).latest(
                    cell.row
                ).exit_block.row[2]
            ].successors
            for cell in continuation.incoming_arms
        )
        assert CorrelationTable._source_exception_ssa_continuation_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)
    boundary, = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert boundary["kind"] == "source-exception-control-owner"
    assert "Ref('ingestion_value'" in boundary["source_try_cell"]


def test_authored_finally_without_exceptional_control_refuses():
    source = """\
def root(x: float) -> int:
    y = 0
    try:
        y = int(x)
    finally:
        y = y + 1
    return y
"""
    with pytest.raises(CompilationSubdivisionRequired) as error:
        _lower(source, "exception_admission_finally")
    boundary, = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert boundary["kind"] == "source-exception-control-owner"


def test_protected_call_carries_its_exact_live_handler_binding():
    source = """\
def root(x: float, z: int) -> int:
    try:
        y = int(x)
    except ValueError:
        y = z
    return y
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_live_binding", identity_book=book)
        site_row, = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        state_row = (*site_row, "z")
        state = book.page(SOURCE_EXCEPTION_REGION_STATE).latest(state_row)
        site = book.page(SOURCE_EXCEPTION_REGION_SITE).latest(site_row)
        transfer = book.page(SOURCE_EXCEPTION_TRANSFER).latest(
            site.transfer.row
        )
        assert state.region_site == book.latest_ref(
            SOURCE_EXCEPTION_REGION_SITE, site_row,
        )
        assert state.source_binding == dict(transfer.entry_bindings)["z"]
        assert state.canonical_value.row[0] == site.canonical_site.row[0]
        assert CorrelationTable._source_exception_region_state_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_standalone_authored_raise_refuses_wrong_native_class():
    # Before this obligation, the complete native artifact reported code 3.
    # validation_fault(3) is RuntimeError, not the authored ValueError.
    source = """\
def root(x: int) -> int:
    raise ValueError("bad")
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as error:
            _lower(source, "exception_standalone_raise", identity_book=book)
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        )
        raise_transfer, = tuple(
            transfer for transfer in transfers
            if transfer.source_kind == "raise"
        )
        route_row, = tuple(book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows())
        route = book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(route_row)
        assert route.selected_handler is None
        assert route.status_class.page == NATIVE_STATUS_CLASS
        source_cell = raise_transfer.source
        admission = book.page(SOURCE_EXCEPTION_ADMISSION).latest(
            (source_cell,)
        )
        assert isinstance(admission, Unresolved)
        assert admission.read == (source_cell,)
        receipt = book.latest_ref(SOURCE_EXCEPTION_ADMISSION, (source_cell,))
        assert {cell for cell, _stage in book.edges_into(receipt)} == {
            source_cell,
        }
        findings = CorrelationTable._source_exception_admission_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        assert len(findings) == 1
        assert "no executable exceptional control" in findings[0].detail
    finally:
        end_identity_book(token)
    boundary, = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert boundary["kind"] == "source-exception-check-owner"
    assert boundary["source_check_cell"] == repr(source_cell)


def test_every_retained_try_has_an_admission_obligation():
    source = """\
def root(x: float) -> int:
    try:
        y = int(x)
    except ValueError:
        y = 2
    try:
        z = int(x)
    finally:
        y = y + 1
    return y + z
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as error:
            _lower(source, "exception_admission_all_tries", identity_book=book)
        rows = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
        assert len(rows) == 2
        assert all(
            isinstance(book.page(SOURCE_EXCEPTION_ADMISSION).latest(row),
                       Unresolved)
            for row in rows
        )
        assert CorrelationTable._source_exception_region_state_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_continuation_arm_value_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)
    boundaries = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert {item["source_try_cell"] for item in boundaries} == {
        repr(row[0]) for row in rows
    }


def test_exception_site_cites_binding_after_preceding_write():
    source = """\
def root(x: float) -> int:
    y = 3
    try:
        y = y + 1
        z = int(x)
    except ValueError:
        return y
    return z
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_site_binding", identity_book=book)
        sites = tuple(
            row for row in book.page(SOURCE_EXCEPTION_SITE_STATE).rows()
            if book.page(SOURCE_EXCEPTION_SITE_STATE).latest(row).action
            == "exception"
        )
        assert len(sites) == 1
        site = book.page(SOURCE_EXCEPTION_SITE_STATE).latest(sites[0])
        assert site.required_names == ("y",)
        binding, = (cell for name, cell in site.bindings if name == "y")
        versions = tuple(
            row for row in book.page(NAME_BINDING).rows()
            if row[0] == sites[0][0] and row[1] == "y"
        )
        assert len(versions) >= 2
        assert binding.row == max(versions, key=lambda row: row[2])
        transfer = book.page(SOURCE_EXCEPTION_TRANSFER).latest(sites[0])
        assert transfer.entry_bindings == site.bindings
        assert transfer.catch_try == site.scopes[-1][0]
        state_rows = tuple(
            row for row in book.page(SOURCE_EXCEPTION_REGION_STATE).rows()
            if row[3] == "y"
        )
        assert len(state_rows) == 1
        state = book.page(SOURCE_EXCEPTION_REGION_STATE).latest(state_rows[0])
        assert state.source_binding == binding
        assert CorrelationTable._source_exception_region_state_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        entries = tuple(book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).rows())
        assert len(entries) == 1
        entry = book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).latest(entries[0])
        assert entry.required_names == ("y",)
        assert entry.incoming_bindings == ((
            "y", ((entry.candidate_transfers[0], binding),),
        ),)
    finally:
        end_identity_book(token)


def test_handler_raise_routes_through_inner_finalizer_to_outer_try():
    source = """\
def root(x: int) -> int:
    try:
        try:
            raise ValueError("first")
        except ValueError:
            raise TypeError("second")
        finally:
            x = x + 1
    except TypeError:
        return 7
    return x
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_nested_transfer", identity_book=book)
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        )
        from_handler = tuple(
            transfer for transfer in transfers
            if transfer.source_kind == "raise"
            and len(transfer.scopes) == 2
            and transfer.scopes[-1][1] == "handler"
        )
        assert len(from_handler) == 1
        transfer, = from_handler
        assert transfer.catch_try == transfer.scopes[0][0]
        assert transfer.unwind_finally == (transfer.scopes[-1][0],)
        classes = tuple(
            book.page(SOURCE_EXCEPTION_HANDLER_CLASS).latest(row)
            for row in book.page(SOURCE_EXCEPTION_HANDLER_CLASS).rows()
        )
        assert len(classes) == 2
        assert all(
            len(fact.declarations) == 1
            and fact.declarations[0][1] is not None
            for fact in classes
        )
        try_facts = tuple(
            book.page(CONTROL_BLOCK).latest(row)
            for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        assert len(try_facts) == 2
        # Each authored ``raise E(...)`` has a fallible constructor Call and
        # a subsequent Raise; both sites have their own transfer cell.
        assert sorted(
            sum(site.page == SOURCE_EXCEPTION_TRANSFER
                for site in fact.sites)
            for fact in try_facts
        ) == [2, 2]
        assert sum(
            site.page == SOURCE_EXCEPTION_SITE_STATE
            for fact in try_facts for site in fact.sites
        ) == 1
    finally:
        end_identity_book(token)


def test_handler_phi_joins_distinct_exception_site_bindings(monkeypatch):
    source = """\
def root(x: float) -> int:
    y = 0
    try:
        y = 1
        a = int(x)
        y = 2
        b = int(x)
    except ValueError:
        return y
    return y + a + b
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    overlaid = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            overlaid.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_handler_phi", identity_book=book)
        rows = tuple(book.page(SOURCE_EXCEPTION_HANDLER_PHI).rows())
        assert len(rows) == 1
        phi = book.page(SOURCE_EXCEPTION_HANDLER_PHI).latest(rows[0])
        assert phi.name == "y"
        assert len(phi.arms) == 2
        assert phi.arms[0][1] != phi.arms[1][1]
        phi_cell = book.latest_ref(SOURCE_EXCEPTION_HANDLER_PHI, rows[0])
        try_rows = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        assert len(try_rows) == 1
        assert phi_cell in book.page(CONTROL_BLOCK).latest(
            try_rows[0]
        ).carried
        arm_rows = tuple(sorted(
            book.page(SOURCE_EXCEPTION_HANDLER_ARM_VALUE).rows(),
            key=lambda row: row[2],
        ))
        assert len(arm_rows) == 2
        arm_facts = tuple(
            book.page(SOURCE_EXCEPTION_HANDLER_ARM_VALUE).latest(row)
            for row in arm_rows
        )
        assert tuple(fact.source_binding for fact in arm_facts) == tuple(
            binding for _transfer, binding in phi.arms
        )
        assert arm_facts[0].canonical_value != arm_facts[1].canonical_value
        site_rows = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        assert len(site_rows) == 2
        assert len({row[:2] for row in site_rows}) == 2
        assert CorrelationTable._source_exception_region_site_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_handler_arm_value_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert len(overlaid) == 1
        state_rows = tuple(book.page(SOURCE_EXCEPTION_REGION_STATE).rows())
        # These two literal versions are already present in the source
        # graph. Give direct control lowering the same literal IDs the
        # public compiler would supply after region planning.
        literal_ids = tuple(
            int(book.page(SOURCE_EXCEPTION_REGION_STATE).latest(
                row
            ).canonical_value.row[1])
            for row in state_rows if row[3] == "y"
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            overlaid[0], function_name="exception_handler_phi_probe",
            first_value_id=1000,
            lexical_read_scope=try_rows[0][0],
            planning_scope=site_rows[0][0],
            region_callees={
                row[1]: f"numerical_region_{row[1]}" for row in site_rows
            },
            region_signatures={row[1]: ((), ()) for row in site_rows},
            constant_value_ids=literal_ids,
        )
        assert all(
            shortfall.name != "native_try_dispatch"
            for shortfall in shortfalls
        ), shortfalls
        assert len(tuple(book.page(SOURCE_EXCEPTION_SSA_CHECKPOINT).rows())) == 2
        route_rows = tuple(book.page(SOURCE_EXCEPTION_SSA_ROUTE).rows())
        source_route_rows = tuple(
            row for row in book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows()
            if book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(
                row
            ).selected_handler is not None
        )
        assert len(route_rows) == len(source_route_rows)
        assert {row[3] for row in route_rows} == {
            row[2] for row in source_route_rows
        }
        assert {row[1:3] for row in route_rows} == {
            row[1:] for row in site_rows
        }
        assert all(
            book.page(SOURCE_EXCEPTION_SSA_ROUTE).latest(
                row
            ).selected_handler == arm_facts[0].source_handler
            for row in route_rows
        )
        assert CorrelationTable._source_exception_ssa_route_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        incoming_rows = tuple(book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_ARM
        ).rows())
        assert len(incoming_rows) == len(route_rows)
        state_rows = tuple(book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_STATE
        ).rows())
        assert len(state_rows) == len(route_rows)
        assert {
            row[:4]: book.page(SOURCE_EXCEPTION_SSA_HANDLER_STATE).latest(
                row
            ).incoming_value
            for row in state_rows
        } == {
            row[:4]: book.page(SOURCE_EXCEPTION_SSA_HANDLER_ARM).latest(
                row
            ).incoming_value
            for row in incoming_rows
        }
        assert CorrelationTable._source_exception_ssa_handler_state_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert {
            row[1:3]: book.page(SOURCE_EXCEPTION_SSA_HANDLER_ARM).latest(
                row
            ).source_arm
            for row in incoming_rows
        } == {
            site_row[1:]: book.latest_ref(
                SOURCE_EXCEPTION_HANDLER_ARM_VALUE, arm_row,
            )
            for site_row, arm_row in zip(site_rows, arm_rows)
        }
        assert CorrelationTable._source_exception_ssa_handler_arm_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        entry_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_ENTRY
        ).rows())
        entry = book.page(SOURCE_EXCEPTION_SSA_HANDLER_ENTRY).latest(entry_row)
        phi_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_PHI
        ).rows())
        native_phi = book.page(SOURCE_EXCEPTION_SSA_HANDLER_PHI).latest(
            phi_row
        )
        assert native_phi.handler_entry == book.latest_ref(
            SOURCE_EXCEPTION_SSA_HANDLER_ENTRY, entry_row,
        )
        assert native_phi.source_phi == phi_cell
        assert tuple(
            book.page(SOURCE_EXCEPTION_SSA_HANDLER_ARM).latest(
                cell.row
            ).incoming_value.row[1]
            for cell in native_phi.incoming_arms
        ) == tuple(
            instruction.args[index].id
            for instruction in lowered.blocks[
                entry.entry_block.row[2]
            ].instrs
            if instruction.attributes.get("binding")
            == "native_exception_handler_phi"
            for index in range(len(instruction.args))
        )
        assert all(
            entry.entry_block.row[2] in lowered.blocks[
                book.page(SOURCE_EXCEPTION_SSA_ROUTE).latest(
                    route_cell.row
                ).match_block.row[2]
            ].successors
            for route_cell in entry.incoming_routes
        )
        assert CorrelationTable._source_exception_ssa_handler_entry_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_handler_body_reads_restored_failure_site_state(monkeypatch):
    source = """\
def root(x: float) -> float:
    try:
        y = int(x)
    except ValueError:
        y = x + 1.0
    return y
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    overlaid = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            overlaid.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_handler_body", identity_book=book)
        assert len(overlaid) == 1
        site_row, = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        state_row, = tuple(book.page(SOURCE_EXCEPTION_REGION_STATE).rows())
        assert state_row[3] == "x"
        input_id = int(book.page(SOURCE_EXCEPTION_REGION_STATE).latest(
            state_row
        ).canonical_value.row[1])
        outputs = {
            book.page(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE).latest(
                row
            ).arm_kind: int(book.page(
                SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE
            ).latest(row).canonical_value.row[1])
            for row in book.page(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE).rows()
        }
        try_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            overlaid[0], function_name="exception_handler_body_probe",
            first_value_id=1000,
            lexical_read_scope=try_row[0], planning_scope=site_row[0],
            region_callees={
                0: "numerical_region_0", 1: "numerical_region_1",
            },
            region_signatures={
                0: ((input_id,), (outputs["handler"],)),
                1: ((input_id,), (outputs["success"],)),
            },
        )
        assert not shortfalls, shortfalls
        handler_entry_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_ENTRY
        ).rows())
        handler_entry = book.page(
            SOURCE_EXCEPTION_SSA_HANDLER_ENTRY
        ).latest(handler_entry_row)
        assert any(
            instruction.attributes.get("callee") == "numerical_region_0"
            for instruction in lowered.blocks[
                handler_entry.entry_block.row[2]
            ].instrs
        )
        continuation_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).rows())
        continuation = book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).latest(continuation_row)
        assert len(continuation.incoming_arms) == 2
        assert continuation.join_block.row[2] in lowered.blocks[
            handler_entry.entry_block.row[2]
        ].successors
        assert CorrelationTable._source_exception_ssa_continuation_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        binding = book.page(CONTROL_VALUE_BINDING).latest((
            handler_entry_row[0], input_id,
        ))
        assert binding.kind is BindingKind.RESTORED
        source_entry_row, = tuple(
            row for row in book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).rows()
            if book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).latest(
                row
            ).handler == handler_entry.source_handler
        )
        binding_cell = book.latest_ref(CONTROL_VALUE_BINDING, (
            handler_entry_row[0], input_id,
        ))
        assert book.latest_ref(
            SOURCE_EXCEPTION_HANDLER_ENTRY, source_entry_row,
        ) in {
            cause for cause, _stage in book.edges_into(binding_cell)
        }
        assert all(
            book.latest_ref(SOURCE_EXCEPTION_SSA_HANDLER_STATE, (
                *route_cell.row, "x",
            )) in {
                cause for cause, _stage in book.edges_into(binding_cell)
            }
            for route_cell in handler_entry.incoming_routes
        )
        assert CorrelationTable._source_exception_ssa_handler_state_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_finally_reads_post_try_join_before_its_region_runs(monkeypatch):
    source = """\
def root(x: float) -> float:
    y = 0.0
    try:
        y = int(x)
    except ValueError:
        y = 2.0
    finally:
        y = y + 1.0
    return y
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((
                program, args[0], tuple(kwargs["dispatch_subgraphs"]),
            ))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_finally_join", identity_book=book)
        (program, graph, subgraphs), = observed
        site_row, = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        arm_facts = tuple(
            book.page(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE).latest(row)
            for row in sorted(
                book.page(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE).rows(),
                key=lambda row: row[2],
            )
        )
        assert tuple(fact.arm_kind for fact in arm_facts) == (
            "success", "handler",
        )
        phi_id = int(arm_facts[0].canonical_phi.row[1])
        try_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            program, function_name="exception_finally_join_probe",
            first_value_id=1000,
            lexical_read_scope=try_row[0], planning_scope=site_row[0],
            region_callees={
                0: "numerical_region_0", 1: "numerical_region_1",
            },
            region_signatures={
                0: ((0,), (int(arm_facts[0].canonical_value.row[1]),)),
                1: ((phi_id,), tuple(map(
                    int, subgraphs[1].G.graph["deployment_nodes"],
                ))),
            },
            constant_value_ids=constants,
        )
        assert not shortfalls, shortfalls
        continuation_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).rows())
        continuation = book.page(
            SOURCE_EXCEPTION_SSA_CONTINUATION_PHI
        ).latest(continuation_row)
        join = lowered.blocks[continuation.join_block.row[2]]
        finalizer_call, = tuple(
            instruction for instruction in join.instrs
            if instruction.attributes.get("callee") == "numerical_region_1"
        )
        assert finalizer_call.args[0].id == continuation.result_value.row[1]
        unmatched = tuple(
            instruction
            for name, block in lowered.blocks.items()
            if name != join.name
            for instruction in block.instrs
            if instruction.attributes.get("callee")
            == "numerical_region_1"
        )
        assert len(unmatched) == 1
        input_rows = tuple(book.page(
            SOURCE_EXCEPTION_SSA_FINALIZER_INPUT
        ).rows())
        assert len(input_rows) == 1
        restored = book.page(
            SOURCE_EXCEPTION_SSA_FINALIZER_INPUT
        ).latest(input_rows[0])
        assert unmatched[0].args[0].id == restored.incoming_value.row[1]
        assert CorrelationTable._source_exception_finalizer_path_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("source", ["""\
def root(x: int) -> int:
    try:
        raise ValueError("bad")
    except ValueError:
        return 2
""", """\
def root(x: int) -> int:
    y = 2
    try:
        raise ValueError("bad")
    except ValueError:
        return y
"""])
def test_regionless_authored_raise_keeps_try_control(monkeypatch, source):
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import (
        LoopControlBlock, NativeAssertBlock, NativeTryBlock,
    )

    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0]))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_regionless_raise", identity_book=book)
        (program, graph), = observed
        retained_try, = tuple(
            child for child in program.root.blocks
            if isinstance(child, NativeTryBlock)
        )
        assert retained_try.protected_transfer_cells
        raise_id, = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Raise)
        )
        handler_id = retained_try.handlers[0].source_handler_node_id
        try_id = retained_try.source_try_node_id
        assert graph.G.has_edge(raise_id, try_id)
        assert graph.G.has_edge(handler_id, try_id)
        assert graph.G.edges[raise_id, try_id]["role"] == "body"
        assert graph.G.edges[handler_id, try_id]["role"] == "handlers"
        transfer_cell, = tuple(
            cell for cell in retained_try.protected_transfer_cells
            if book.page(SOURCE_EXCEPTION_TRANSFER).cell(
                cell.row, cell.column,
            ).source_kind == "raise"
        )
        transfer = book.page(SOURCE_EXCEPTION_TRANSFER).cell(
            transfer_cell.row, transfer_cell.column,
        )
        assert transfer.source_kind == "raise"
        raise_block, = tuple(
            child for child in retained_try.body.blocks
            if isinstance(child, NativeAssertBlock)
        )
        assert raise_block.source_node_id == raise_id
        assert raise_block.transfer_cell == transfer_cell
        raise_control, = tuple(
            book.page(CONTROL_BLOCK).latest(row)
            for row in book.page(CONTROL_BLOCK).rows()
            if book.page(CONTROL_BLOCK).latest(row).kind
            is ControlBlockKind.NATIVE_ASSERT
        )
        assert raise_control.sites == (transfer_cell,)
        assert not CorrelationTable._source_exception_assert_control_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        assert not CorrelationTable._source_exception_status_route_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        route_row, = tuple(
            row for row in book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows()
            if row[:2] == transfer_cell.row
        )
        route = book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(route_row)
        assert route.owner == transfer.catch_try
        handler_fact, = tuple(
            book.page(SOURCE_EXCEPTION_HANDLER_CLASS).latest(row)
            for row in book.page(SOURCE_EXCEPTION_HANDLER_CLASS).rows()
            if book.page(SOURCE_EXCEPTION_HANDLER_CLASS).latest(row).owner
            == transfer.catch_try
        )
        assert route.selected_handler == handler_fact.handler
        assert any(
            isinstance(child, LoopControlBlock)
            and child.action == "return"
            and child.return_site_cell is not None
            for child in retained_try.handlers[0].body.blocks
        )
        assert program.region_indices == ()
        from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

        try_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            program, function_name="exception_regionless_raise_probe",
            first_value_id=1000, lexical_read_scope=try_row[0],
            planning_scope=graph.G.graph.get("planning_scope"),
            constant_value_ids=constants,
        )
        assert lowered.blocks
        assert any(item.name == "native_try_dispatch" for item in shortfalls)
        checkpoint, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_CONTROL_CHECKPOINT
        ).rows())
        control_route, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_CONTROL_ROUTE
        ).rows())
        assert checkpoint[1] == transfer_cell
        assert control_route[1:] == (transfer_cell, int(route_row[2]))
        assert CorrelationTable._source_exception_ssa_control_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_ssa_handler_entry_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_return_in_try_reaches_finally_return_before_function_exit(monkeypatch):
    source = """\
def root(x: float) -> int:
    try:
        y = int(x)
        return 1
    finally:
        return 2
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0], tuple(kwargs["dispatch_subgraphs"])))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as refusal:
            _lower(source, "exception_finally_override", identity_book=book)
        assert observed, str(refusal.value)
        (program, graph, subgraphs), = observed
        try_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            program, function_name="exception_finally_override_probe",
            first_value_id=1000, lexical_read_scope=try_row[0],
            planning_scope=graph.G.graph.get("planning_scope"),
            constant_value_ids=constants,
        )
        assert any(
            shortfall.name == "native_try_dispatch" for shortfall in shortfalls
        )
        returns = tuple(
            (block.name, instruction)
            for block in lowered.blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("source_control") == "return"
        )
        assert len(returns) >= 2
        pending_row, = tuple(book.page(
            SOURCE_EXCEPTION_SSA_RETURN_FINALIZER
        ).rows())
        pending = book.page(SOURCE_EXCEPTION_SSA_RETURN_FINALIZER).latest(
            pending_row
        )
        assert pending.before_block.row[2] == "entry"
        assert returns[0][1].attributes["return_site_cell"] != pending.return_site
        assert returns[1][1].attributes["return_site_cell"] == pending.return_site
        assert not any(
            pending.after_block.row[2] in block.successors
            for block in lowered.blocks.values()
        )
        assert CorrelationTable._source_exception_ssa_return_finalizer_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_handler_return_runs_finalizer_with_exceptional_entry_binding(monkeypatch):
    source = """\
def root(x: float) -> float:
    y = 0.0
    try:
        y = int(x)
    except ValueError:
        return 1.0
    finally:
        y = y + 1.0
    return y
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0], tuple(kwargs["dispatch_subgraphs"])))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_handler_finally_return", identity_book=book)
        (program, graph, subgraphs), = observed
        site_row, = tuple(book.page(SOURCE_EXCEPTION_REGION_SITE).rows())
        assert len(program.region_indices) == 2, program.region_indices
        finalizer_node, = tuple(subgraphs[1].G.graph["deployment_nodes"])
        try_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        state_constants = tuple(
            int(book.page(SOURCE_EXCEPTION_REGION_STATE).latest(
                row
            ).canonical_value.row[1])
            for row in book.page(SOURCE_EXCEPTION_REGION_STATE).rows()
        )
        lowered, shortfalls = lower_control_program_to_ssa(
                program,
                function_name="exception_handler_finally_return_probe",
                first_value_id=1000, lexical_read_scope=try_row[0],
                planning_scope=site_row[0],
                region_signatures={
                    0: ((0,), (int(site_row[2]),)),
                    1: (tuple(map(
                        int, graph.G.predecessors(finalizer_node)
                    )), (int(finalizer_node),)),
                },
                constant_value_ids=(*constants, *state_constants),
        )
        assert lowered.blocks
        assert any(item.name == "return" for item in shortfalls)
        source_rows = tuple(book.page(
            SOURCE_EXCEPTION_FINALIZER_BINDING
        ).rows())
        canonical_rows = tuple(book.page(
            SOURCE_EXCEPTION_FINALIZER_INPUT
        ).rows())
        ssa_rows = tuple(book.page(
            SOURCE_EXCEPTION_SSA_FINALIZER_INPUT
        ).rows())
        assert len(source_rows) == len(canonical_rows) == 1
        assert len(ssa_rows) == 2
        canonical = book.page(
            SOURCE_EXCEPTION_FINALIZER_INPUT
        ).latest(canonical_rows[0])
        for row in ssa_rows:
            ssa_fact = book.page(
                SOURCE_EXCEPTION_SSA_FINALIZER_INPUT
            ).latest(row)
            assert ssa_fact.source_input == book.latest_ref(
                SOURCE_EXCEPTION_FINALIZER_INPUT, canonical_rows[0],
            )
            assert ssa_fact.canonical_incoming == dict(
                canonical.site_values
            )[ssa_fact.exit_site]
        assert CorrelationTable._source_exception_finalizer_path_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_explicit_raise_status_uses_first_authored_matching_handler():
    source = """\
def root(x: int) -> int:
    try:
        raise ValueError("bad")
    except TypeError:
        return 1
    except Exception:
        return 2
    except ValueError:
        return 3
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_status_first_match", identity_book=book)
        rows = tuple(book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows())
        raise_rows = tuple(
            row for row in rows
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row[:2]).source_kind
            == "raise"
        )
        assert len(raise_rows) == 1
        route = book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(raise_rows[0])
        assert route.status_class.page == NATIVE_STATUS_CLASS
        assert len(route.examined_handlers) == 2
        matched = book.page(SOURCE_EXCEPTION_HANDLER_CLASS).latest(
            route.examined_handlers[-1].row
        )
        assert matched.ordinal == 1
        assert route.selected_handler == matched.handler
        assert route.matched_declaration == matched.declarations[0][1]
        assert CorrelationTable._source_exception_status_route_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_raise_in_handler_uses_outer_try_body_as_catch_owner(monkeypatch):
    source = """\
def root(x: int) -> int:
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise TypeError("outer")
    except TypeError:
        return 1
"""
    from src.compiler import glsl_deployment_strategy as strategy
    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0]))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_nested_raise_owner", identity_book=book)
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            == "raise"
        )
        assert len(transfers) == 2
        raise_controls = tuple(
            book.page(CONTROL_BLOCK).latest(row)
            for row in book.page(CONTROL_BLOCK).rows()
            if book.page(CONTROL_BLOCK).latest(row).kind
            is ControlBlockKind.NATIVE_ASSERT
        )
        assert len(raise_controls) == 2
        inner_raise, = tuple(
            transfer for transfer in transfers
            if len(transfer.scopes) == 2
            and transfer.scopes[-1][1] == "body"
        )
        handler_raise, = tuple(
            transfer for transfer in transfers
            if len(transfer.scopes) == 2
            and transfer.scopes[-1][1] == "handler"
        )
        assert inner_raise.catch_try == inner_raise.scopes[-1][0]
        assert handler_raise.catch_try == handler_raise.scopes[0][0]
        assert handler_raise.catch_try != inner_raise.catch_try
        transfer_cell, = tuple(
            book.latest_ref(SOURCE_EXCEPTION_TRANSFER, row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            == handler_raise
        )
        routes = tuple(
            book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(row)
            for row in book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows()
            if book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(
                row
            ).transfer == transfer_cell
        )
        assert len(routes) == 1
        assert routes[0].owner == handler_raise.catch_try
        assert routes[0].selected_handler is not None
        (program, graph), = observed
        from src.compiler.control_source import NativeTryBlock, SequenceBlock
        outer_blocks = (
            program.root.blocks
            if isinstance(program.root, SequenceBlock) else (program.root,)
        )
        outer_try, = tuple(
            block for block in outer_blocks
            if isinstance(block, NativeTryBlock)
        )
        assert any(
            isinstance(block, NativeTryBlock)
            for block in outer_try.body.blocks
        )
        from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
        try_row = next(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        lowered, _shortfalls = lower_control_program_to_ssa(
            program, function_name="exception_nested_raise_probe",
            first_value_id=1000, lexical_read_scope=try_row[0],
            planning_scope=graph.G.graph.get("planning_scope"),
            constant_value_ids=constants,
        )
        assert lowered.blocks
        assert any(
            row[1] == transfer_cell
            for row in book.page(SOURCE_EXCEPTION_SSA_CONTROL_ROUTE).rows()
        )
    finally:
        end_identity_book(token)


def test_inner_try_miss_has_explicit_outer_handler_edge(monkeypatch):
    source = """\
def root(x: int) -> int:
    try:
        try:
            raise ValueError("inner")
        except TypeError:
            return 1
    except ValueError:
        return 2
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.concordance_declarations import (
        SOURCE_EXCEPTION_ROUTE_STEP, SOURCE_EXCEPTION_SSA_ESCALATION_ROUTE,
        SOURCE_EXCEPTION_SSA_PROPAGATION,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    observed = []
    original = strategy._overlay_control_or_require_subdivision

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0]))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_outer_handler_edge", identity_book=book)
        transfer_cell, = tuple(
            book.latest_ref(SOURCE_EXCEPTION_TRANSFER, row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            == "raise"
        )
        source_route_row, = tuple(
            row for row in book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows()
            if row[:2] == transfer_cell.row
        )
        assert book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(
            source_route_row,
        ).selected_handler is None
        first = book.page(SOURCE_EXCEPTION_ROUTE_STEP).latest(
            (*source_route_row, 0),
        )
        assert first.selected_handler is None
        assert first.successor is not None
        second = book.page(SOURCE_EXCEPTION_ROUTE_STEP).cell(
            first.successor.row, first.successor.column,
        )
        assert second.selected_handler is not None
        assert CorrelationTable._source_exception_route_step_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        (program, graph), = observed
        try_row = next(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        constants = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if str(data.get("op")).casefold() == "const"
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            program, function_name="exception_outer_handler_edge_probe",
            first_value_id=1000, lexical_read_scope=try_row[0],
            planning_scope=graph.G.graph.get("planning_scope"),
            constant_value_ids=constants,
        )
        assert lowered.blocks
        assert any(item.name == "native_try_dispatch" for item in shortfalls)
        assert book.page(SOURCE_EXCEPTION_SSA_ESCALATION_ROUTE).latest((
            "exception_outer_handler_edge_probe@control:0",
            transfer_cell, int(source_route_row[2]), 1,
        )) is not None
        assert any(
            row[1] == transfer_cell
            for row in book.page(SOURCE_EXCEPTION_SSA_PROPAGATION).rows()
        )
        assert CorrelationTable._source_exception_ssa_propagation_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_ssa_escalation_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_ssa_handler_entry_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_raise_divides_try_arm_before_later_source_work(monkeypatch):
    source = """\
def root(x: int) -> int:
    try:
        y = x + 1
        raise ValueError("stop")
        z = x + 2
    except ValueError:
        return y
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import (
        NativeAssertBlock, NativeTryBlock, StatementBlock,
    )

    observed = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append((program, args[0]))
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_raise_frontier", identity_book=book)
        (program, graph), = observed
        retained_try, = tuple(
            block for block in program.root.blocks
            if isinstance(block, NativeTryBlock)
        )
        raise_positions = tuple(
            index for index, block in enumerate(retained_try.body.blocks)
            if isinstance(block, NativeAssertBlock)
        )
        assert len(raise_positions) == 1
        assert any(
            isinstance(block, StatementBlock)
            and block.lines[0].startswith("__scheduled_region_")
            for block in retained_try.body.blocks[:raise_positions[0]]
        )
        raise_block = retained_try.body.blocks[raise_positions[0]]
        transfer = book.page(SOURCE_EXCEPTION_TRANSFER).cell(
            raise_block.transfer_cell.row,
            raise_block.transfer_cell.column,
        )
        assert transfer.source_kind == "raise"
    finally:
        end_identity_book(token)


def test_raise_is_a_region_frontier_for_reachable_later_work(monkeypatch):
    source = """\
def root(x: int) -> int:
    try:
        if x:
            raise ValueError("stop")
        y = x + 2
    except ValueError:
        y = 1
    return y
"""
    from src.compiler import glsl_deployment_strategy as strategy

    graphs = []
    original_overlay = strategy._overlay_control_or_require_subdivision

    def observe_overlay(*args, **kwargs):
        program = original_overlay(*args, **kwargs)
        if kwargs.get("try_controls"):
            graphs.append(args[0])
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", observe_overlay,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_raise_frontier_live", identity_book=book)
        graph, = graphs
        raise_id, = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Raise)
        )
        later_nodes = tuple(
            int(node_id) for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.BinOp)
            and getattr(data["expr_obj"], "lineno", None) == 5
        )
        assert later_nodes
        keys = strategy._control_partition_keys(graph, (), later_nodes)
        assert all(raise_id in keys[node_id][6] for node_id in later_nodes)
        assert any(
            row[1] is ControlBlockKind.NATIVE_ASSERT
            for row in book.page(CONTROL_BLOCK).rows()
        )
    finally:
        end_identity_book(token)


def test_assert_and_raise_share_concorded_exception_check():
    source = """\
def root(x: int) -> int:
    try:
        assert x > 0
        raise ValueError("bad")
    except AssertionError:
        return 1
    except ValueError:
        return 2
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_assert_raise_check", identity_book=book)
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            in {"assert", "raise"}
        )
        assert {transfer.source_kind for transfer in transfers} == {
            "assert", "raise",
        }
        checks = tuple(
            book.page(CONTROL_BLOCK).latest(row)
            for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_ASSERT
        )
        assert len(checks) == 2
        assert {check.predicate is None for check in checks} == {True, False}
        assert not CorrelationTable._source_exception_assert_control_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        assert not CorrelationTable._source_exception_status_route_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        for transfer in transfers:
            transfer_cell, = tuple(
                book.latest_ref(SOURCE_EXCEPTION_TRANSFER, row)
                for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
                if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row) == transfer
            )
            routes = tuple(
                book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(row)
                for row in book.page(SOURCE_EXCEPTION_STATUS_ROUTE).rows()
                if book.page(SOURCE_EXCEPTION_STATUS_ROUTE).latest(
                    row
                ).transfer == transfer_cell
            )
            assert len(routes) == 1
            assert routes[0].selected_handler is not None
    finally:
        end_identity_book(token)


def test_authored_assert_without_try_refuses_unimplemented_status_exit(
    monkeypatch,
):
    source = """\
def root(x: int) -> int:
    assert x > 0
    return x
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import NativeAssertBlock
    observed = []
    original = strategy._overlay_control_or_require_subdivision

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_unhandled_assert", identity_book=book)
        assert any(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            == "assert"
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        )
        assert any(
            book.page(CONTROL_BLOCK).latest(row).kind
            is ControlBlockKind.NATIVE_ASSERT
            for row in book.page(CONTROL_BLOCK).rows()
        )
        assert CorrelationTable._source_exception_control_coverage_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert any(
            isinstance(block, NativeAssertBlock)
            for block in observed[-1].root.blocks
        )
        assert any(
            isinstance(book.page(SOURCE_EXCEPTION_ADMISSION).latest(row),
                       Unresolved)
            for row in book.page(SOURCE_EXCEPTION_ADMISSION).rows()
        )
    finally:
        end_identity_book(token)


def test_root_raise_control_lowers_to_ssa_fault_edge(monkeypatch):
    source = """\
def root() -> int:
    raise ValueError()
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.concordance_declarations import ExceptionAdmissionFact
    from src.transmogrifier.ssa_registry import Handler

    observed = []
    original = strategy._overlay_control_or_require_subdivision

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )
    book, token = begin_identity_book()
    try:
        module, _outputs, exports = _lower(
            source, "exception_root_raise_ssa", complete=True,
            identity_book=book,
        )
        control_row, = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_ASSERT
        )
        assert observed
        assert control_row[0]
        lowered = module.functions[exports[0]]
        assert any(
            instruction.op in {Handler.CondBr.value, Handler.Br.value}
            for block in lowered.blocks.values()
            for instruction in block.instrs
        )
        assert all(
            isinstance(book.page(SOURCE_EXCEPTION_ADMISSION).latest(row),
                       ExceptionAdmissionFact)
            for row in book.page(SOURCE_EXCEPTION_ADMISSION).rows()
        )
        assert CorrelationTable._source_exception_admission_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert any(
            instruction.attributes.get("callee") == "turing_validation_error"
            for block in lowered.blocks.values()
            for instruction in block.instrs
        )
        assert CorrelationTable._source_exception_ssa_control_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_root_assert_predicate_precedes_production_ssa_fault_edge():
    source = """\
def root(x: int) -> int:
    assert x > 0
    return x
"""
    from src.compiler.concordance_declarations import ExceptionAdmissionFact
    from src.transmogrifier.ssa_registry import Handler

    book, token = begin_identity_book()
    try:
        module, _outputs, exports = _lower(
            source, "exception_root_assert_ssa", complete=True,
            identity_book=book,
        )
        function = module.functions[exports[0]]
        entry_ops = tuple(instruction.op for instruction in (
            function.blocks["entry"].instrs
        ))
        assert entry_ops.index(Handler.Call.value) < entry_ops.index(
            Handler.CondBr.value
        )
        assert any(
            instruction.op == Handler.CondBr.value
            for block in function.blocks.values()
            for instruction in block.instrs
        )
        assert any(
            instruction.op == Handler.Trap.value
            for block in function.blocks.values()
            for instruction in block.instrs
        )
        assert any(
            instruction.attributes.get("callee") == "turing_validation_error"
            for block in function.blocks.values()
            for instruction in block.instrs
        )
        admission_row, = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
        assert isinstance(
            book.page(SOURCE_EXCEPTION_ADMISSION).latest(admission_row),
            ExceptionAdmissionFact,
        )
        assert CorrelationTable._source_exception_admission_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("kind", ["raise", "assert"])
def test_root_exception_control_executes_on_native_backends(
    backend, kind, tmp_path,
):
    source = (
        "def root(x: int) -> int:\n"
        "    if x <= 0:\n        raise ValueError()\n"
        "    return x\n"
        if kind == "raise" else
        "def root(x: int) -> int:\n    assert x > 0\n    return x\n"
    )
    module, _outputs, exports = _lower(
        source, f"exception_native_{kind}_{backend}", complete=True,
    )
    function = module.functions[exports[0]]
    parameters = {
        dict(function.metadata["parameter_names"])["x"]:
            np.array([1], dtype=np.int64),
    }
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda inputs: artifact.prepare_execution(inputs).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda inputs: prepare_artifact_execution(artifact, inputs).run()
    assert run(parameters).buffers
    parameter, = tuple(parameters)
    with pytest.raises(
        NativeValueError if kind == "raise" else NativeAssertionError
    ):
        run({parameter: np.array([0], dtype=np.int64)})


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_handler_raise_leaves_loop_on_native_backends(backend, tmp_path):
    source = """\
def root(count: int, value: float) -> int:
    result = 0
    for index in range(count):
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError() from error
    return result
"""
    module, _outputs, exports = _lower(
        source, f"exception_handler_loop_{backend}", complete=True,
    )
    function = module.functions[exports[0]]
    parameters = dict(function.metadata["parameter_names"])
    returns = tuple(
        instruction for basic in function.blocks.values()
        for instruction in basic.instrs if str(instruction.op) == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    output = int(returns[0].args[0].id)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda count, value: artifact.prepare_execution({
            parameters["count"]: np.array([count], dtype=np.int64),
            parameters["value"]: np.array([value], dtype=np.float64),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda count, value: prepare_artifact_execution(artifact, {
            parameters["count"]: np.array([count], dtype=np.int64),
            parameters["value"]: np.array([value], dtype=np.float64),
        }).run()
    assert int(run(2, 1.5).buffers[output].reshape(-1)[0]) == 1
    with pytest.raises(NativeValueError):
        run(2, float("nan"))


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_unconditional_root_raise_has_empty_native_buffer_abi(
    backend, tmp_path,
):
    module, _outputs, exports = _lower(
        "def root() -> int:\n    raise ValueError()\n",
        f"exception_unconditional_raise_{backend}", complete=True,
    )
    export = exports[0]
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda: artifact.prepare_execution({}).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda: prepare_artifact_execution(artifact, {}).run()
    with pytest.raises(NativeValueError):
        run()


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("with_finally", [False, True], ids=["no-finally", "numeric-finally"])
def test_production_try_dispatch_matches_native_status(
    backend, with_finally, tmp_path,
):
    source = """\
def root(x: float) -> int:
{initial}
    try:
        y = int(x)
    except ValueError:
        y = 2
{finalbody}
    return y
""".format(
        initial="    y = 0" if with_finally else "",
        finalbody="    finally:\n        y = y + 1" if with_finally else "",
    )
    module, _outputs, exports = _lower(
        source, f"exception_try_dispatch_{backend}", complete=True,
    )
    function = module.functions[exports[0]]
    book = module.metadata["identity_book"]
    assert book.page(SOURCE_EXCEPTION_SSA_CHECKPOINT).rows()
    from src.compiler.concordance_declarations import ExceptionTryAdmissionFact
    admission_rows = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
    assert len(admission_rows) == 1
    assert isinstance(
        book.page(SOURCE_EXCEPTION_ADMISSION).latest(admission_rows[0]),
        ExceptionTryAdmissionFact,
    )
    parameter = dict(function.metadata["parameter_names"])["x"]
    returns = tuple(
        instruction for basic in function.blocks.values()
        for instruction in basic.instrs if str(instruction.op) == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    output = int(returns[0].args[0].id)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda value: artifact.prepare_execution({
            parameter: np.array([value], dtype=np.float64),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda value: prepare_artifact_execution(artifact, {
            parameter: np.array([value], dtype=np.float64),
        }).run()
    assert int(run(1.9).buffers[output].reshape(-1)[0]) == (2 if with_finally else 1)
    assert int(run(float("nan")).buffers[output].reshape(-1)[0]) == (3 if with_finally else 2)
    with pytest.raises(NativeOverflowError):
        run(float("inf"))


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_production_try_return_matches_native_status(backend, tmp_path):
    source = """\
def root(x: float) -> int:
    try:
        return int(x)
    except ValueError:
        return 2
"""
    module, _outputs, exports = _lower(
        source, f"exception_try_return_{backend}", complete=True,
    )
    function = module.functions[exports[0]]
    book = module.metadata["identity_book"]
    admission_rows = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
    assert len(admission_rows) == 1
    admission = book.page(SOURCE_EXCEPTION_ADMISSION).latest(
        admission_rows[0]
    )
    assert len(admission.abrupt_exit_proofs) == 2
    parameter = dict(function.metadata["parameter_names"])["x"]
    returns = tuple(
        instruction for basic in function.blocks.values()
        for instruction in basic.instrs if str(instruction.op) == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    output = int(returns[0].args[0].id)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda value: artifact.prepare_execution({
            parameter: np.array([value], dtype=np.float64),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda value: prepare_artifact_execution(artifact, {
            parameter: np.array([value], dtype=np.float64),
        }).run()
    assert int(run(1.9).buffers[output].reshape(-1)[0]) == 1
    assert int(run(float("nan")).buffers[output].reshape(-1)[0]) == 2
    with pytest.raises(NativeOverflowError):
        run(float("inf"))


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("case", ["pending-return", "handler-return"])
def test_production_return_crosses_numeric_finally(
    backend, case, tmp_path,
):
    if case == "pending-return":
        source = """\
def root(x: int) -> int:
    try:
        return x
    finally:
        x = x + 1
"""
        dtype = np.int64
        examples = ((3, 3), (-2, -2))
    else:
        source = """\
def root(x: float) -> float:
    y = 0.0
    try:
        y = int(x)
    except ValueError:
        return 1.0
    finally:
        y = y + 1.0
    return y
"""
        dtype = np.float64
        examples = ((1.9, 2.0), (float("nan"), 1.0))
    module, _outputs, exports = _lower(
        source, f"exception_{case}_{backend}", complete=True,
    )
    function = module.functions[exports[0]]
    book = module.metadata["identity_book"]
    admission_row, = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
    admission = book.page(SOURCE_EXCEPTION_ADMISSION).latest(admission_row)
    assert len(admission.abrupt_exit_proofs) == 1
    pending_rows = tuple(book.page(
        SOURCE_EXCEPTION_SSA_RETURN_FINALIZER
    ).rows())
    assert len(pending_rows) == 1
    pending = book.page(SOURCE_EXCEPTION_SSA_RETURN_FINALIZER).latest(
        pending_rows[0]
    )
    assert pending.finalizer_outputs
    if case == "handler-return":
        normal_rows = tuple(book.page(
            SOURCE_EXCEPTION_SSA_NORMAL_FINALIZER_INPUT
        ).rows())
        assert len(normal_rows) == 1
        normal = book.page(
            SOURCE_EXCEPTION_SSA_NORMAL_FINALIZER_INPUT
        ).latest(normal_rows[0])
        planned_row, = tuple(book.page(
            SOURCE_EXCEPTION_FINALIZER_INPUT
        ).rows())
        planned = book.page(SOURCE_EXCEPTION_FINALIZER_INPUT).latest(
            planned_row
        )
        assert normal.canonical_input == planned.normal_value
        assert normal.incoming_value is not None
        assert CorrelationTable._source_exception_finalizer_path_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    assert CorrelationTable._source_exception_ssa_return_finalizer_findings(
        SimpleNamespace(metadata={"identity_book": book})
    ) == []
    parameter = dict(function.metadata["parameter_names"])["x"]
    returns = tuple(
        instruction for basic in function.blocks.values()
        for instruction in basic.instrs if str(instruction.op) == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    output = int(returns[0].args[0].id)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda value: artifact.prepare_execution({
            parameter: np.array([value], dtype=dtype),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda value: prepare_artifact_execution(artifact, {
            parameter: np.array([value], dtype=dtype),
        }).run()
    for value, expected in examples:
        assert float(run(value).buffers[output].reshape(-1)[0]) == expected
    if case == "handler-return":
        with pytest.raises(NativeOverflowError):
            run(float("inf"))


@pytest.mark.parametrize("source", [
    """\
def root(x: int) -> int:
    try:
        raise ValueError()
    except ValueError:
        return 2
    finally:
        x = x + 1
""",
    """\
def root(x: int) -> int:
    try:
        assert x > 0
        return x
    except AssertionError:
        return 2
    finally:
        x = x + 1
""",
    """\
def root(x: int) -> int:
    try:
        try:
            raise ValueError()
        except TypeError:
            return 1
    except ValueError:
        return 2
""",
], ids=["handled-raise-finally", "assert-finally",
        "nested-propagation"])
@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_production_exception_control_executes(source, backend, tmp_path):
    book, token = begin_identity_book()
    try:
        module, _outputs, exports = _lower(
            source, f"exception_production_matrix_{backend}",
            complete=True, identity_book=book,
        )
        assert book.page(SOURCE_EXCEPTION_SSA_CONTROL_CHECKPOINT).rows() or (
            book.page(SOURCE_EXCEPTION_SSA_CHECKPOINT).rows()
        )
        assert CorrelationTable._source_exception_control_coverage_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        if "except TypeError" in source:
            assert len(book.page(SOURCE_EXCEPTION_SSA_UNREACHABLE_EXIT).rows()) == 2
        function = module.functions[exports[0]]
        parameter = dict(function.metadata["parameter_names"]).get("x")
        returns = tuple(
            instruction for basic in function.blocks.values()
            for instruction in basic.instrs if str(instruction.op) == "Ret"
        )
        assert len(returns) == 1 and len(returns[0].args) == 1
        output = int(returns[0].args[0].id)
        if backend == "c":
            from src.compiler.ssa_c_backend import emit_ssa_to_c

            artifact = emit_ssa_to_c(module, function.name)
            assert artifact.complete, artifact.shortfalls
            artifact.compile(tmp_path / "c")
            run = lambda value: artifact.prepare_execution(
                {parameter: np.array([value], dtype=np.int64)}
                if parameter is not None else {}
            ).run()
        else:
            from src.compiler.ssa_llvm_backend import (
                compile_artifact, emit_ssa_function_to_llvm,
                prepare_artifact_execution,
            )

            emitted = emit_ssa_function_to_llvm(module, function.name)
            assert not emitted.shortfalls, emitted.shortfalls
            artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
            run = lambda value: prepare_artifact_execution(
                artifact,
                {parameter: np.array([value], dtype=np.int64)}
                if parameter is not None else {},
            ).run()
        expect_positive = 3 if "assert x > 0" in source else 2
        assert int(run(0).buffers[output].reshape(-1)[0]) == 2
        assert int(run(3).buffers[output].reshape(-1)[0]) == expect_positive
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("nested", [False, True])
def test_exact_builtin_raise_has_only_raise_transfer(nested):
    source = (
        "def root(x: int) -> int:\n"
        "    try:\n"
        "        try:\n"
        "            raise ValueError()\n"
        "        except TypeError:\n"
        "            return 1\n"
        "    except ValueError:\n"
        "        return 2\n"
        if nested else
        "def root(x: int) -> int:\n"
        "    try:\n"
        "        raise ValueError()\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    finally:\n"
        "        x = x + 1\n"
    )
    book, token = begin_identity_book()
    try:
        _lower(
            source, f"exact_builtin_raise_{nested}", complete=True,
            identity_book=book,
        )
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        )
        assert tuple(fact.source_kind for fact in transfers) == ("raise",)
        assert len(tuple(book.page(
            SOURCE_EXCEPTION_INERT_CONSTRUCTOR
        ).rows())) == 1
        assert book.page(SOURCE_EXCEPTION_SSA_CONTROL_CHECKPOINT).rows()
        assert CorrelationTable._source_exception_inert_constructor_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_route_step_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_message_constructor_keeps_its_separate_call_transfer():
    source = """\
def root(x: int) -> int:
    try:
        raise ValueError("bad")
    except ValueError:
        return 2
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(
                source, "message_constructor_transfer", complete=True,
                identity_book=book,
            )
        kinds = tuple(sorted(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        ))
        assert kinds == ("call", "raise")
    finally:
        end_identity_book(token)


def test_finally_arithmetic_dtype_cites_exact_try_phi_arms():
    from src.compiler.concordance_declarations import (
        PLAN_REGION_VALUE_DTYPE, SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE,
    )

    source = """\
def root(x: float) -> int:
    y = 0
    try:
        y = int(x)
    except ValueError:
        y = 2
    finally:
        y = y + 1
    return y
"""
    book, token = begin_identity_book()
    try:
        _lower(
            source, "exception_finally_dtype_evidence",
            complete=True, identity_book=book,
        )
        arms = {
            book.latest_ref(SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE, row)
            for row in book.page(
                SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE
            ).rows()
        }
        assert len(arms) == 2
        assert any(
            book.page(PLAN_REGION_VALUE_DTYPE).latest(row) == "int64"
            and arms.issubset({
                cause for cause, _stage in book.edges_into(
                    book.latest_ref(PLAN_REGION_VALUE_DTYPE, row)
                )
            })
            for row in book.page(PLAN_REGION_VALUE_DTYPE).rows()
        )
    finally:
        end_identity_book(token)


def test_nested_assert_is_in_dual_ir_control_half(monkeypatch):
    source = """\
def root(x: int) -> int:
    try:
        if x > 0:
            assert x < 10
            return 1
        return 2
    except AssertionError:
        return 3
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import (
        ConditionalBlock, NativeAssertBlock, NativeTryBlock,
    )
    observed = []
    original = strategy._overlay_control_or_require_subdivision

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_nested_assert_control", identity_book=book)
        assert_cells = tuple(
            book.latest_ref(SOURCE_EXCEPTION_TRANSFER, row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
            if book.page(SOURCE_EXCEPTION_TRANSFER).latest(row).source_kind
            == "assert"
        )
        assert len(assert_cells) == 1
        fork_scope = next(
            row[0] for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
            and "|fork" in str(row[0])
        )
        assert_checks = tuple(
            book.page(CONTROL_BLOCK).latest(row)
            for row in book.page(CONTROL_BLOCK).rows()
            if row[0] == fork_scope
            if book.page(CONTROL_BLOCK).latest(row).kind
            is ControlBlockKind.NATIVE_ASSERT
        )
        assert len(assert_checks) == 1
        assert assert_checks[0].sites == assert_cells
        assert CorrelationTable._source_exception_control_coverage_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        program = observed[-1]
        from src.compiler.control_source import SequenceBlock

        def conditionals(block):
            if isinstance(block, ConditionalBlock):
                yield block
            if isinstance(block, SequenceBlock):
                for child in block.blocks:
                    yield from conditionals(child)

        outer_try, = tuple(
            block for block in program.root.blocks
            if isinstance(block, NativeTryBlock)
        )
        conditional, = tuple(
            conditionals(outer_try.body)
        )
        assert any(
            isinstance(block, NativeAssertBlock)
            for block in conditional.body.blocks
        )
    finally:
        end_identity_book(token)


def test_try_and_root_assert_in_conditional_control_arms(monkeypatch):
    source = """\
def root(x: int) -> int:
    if x > 0:
        try:
            assert x < 10
        except AssertionError:
            return 1
    else:
        assert x < 0
    return x
"""
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import (
        ConditionalBlock, NativeAssertBlock, NativeTryBlock, SequenceBlock,
    )
    observed = []
    original = strategy._overlay_control_or_require_subdivision

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_conditional_arms", identity_book=book)
        program = observed[-1]

        def children(block):
            if isinstance(block, SequenceBlock):
                return block.blocks
            return (block,)

        outer, = tuple(
            block for block in children(program.root)
            if isinstance(block, ConditionalBlock)
        )
        assert any(isinstance(block, NativeTryBlock)
                   for block in children(outer.body))
        assert any(isinstance(block, NativeAssertBlock)
                   for block in children(outer.orelse))
        assert CorrelationTable._source_exception_control_coverage_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
    finally:
        end_identity_book(token)


def test_cast_without_authored_try_still_lowers():
    source = """\
def root(x: float) -> int:
    return int(x)
"""
    module, _outputs, exports = _lower(source, "exception_admission_plain")
    assert exports[0] in module.functions


@pytest.mark.parametrize("action", ["break", "continue"])
@pytest.mark.parametrize("guarded", [False, True])
def test_try_loop_exit_has_one_source_owned_control_edge(
    action, guarded, monkeypatch,
):
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.control_source import (
        ConditionalBlock, LoopBlock, LoopControlBlock, NativeTryBlock,
        SequenceBlock, WhileBlock,
    )
    from src.compiler.precompile_to_ssa import _ControlSSABuilder

    body = (
        f"            if x > 1:\n                {action}\n"
        "            x = x - 1\n"
        if guarded else
        f"            x = x - 1\n            {action}\n"
    )
    source = f"""\
def root(x: int) -> int:
    while x > 0:
        try:
{body.rstrip()}
        finally:
            x = x - 1
    return x
"""
    observed = []
    finished = []
    original = strategy._overlay_control_or_require_subdivision
    original_finish = _ControlSSABuilder.finish

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if kwargs.get("try_controls"):
            observed.append(program)
        return program

    monkeypatch.setattr(
        strategy, "_overlay_control_or_require_subdivision", capture,
    )

    def capture_finish(builder, *args, **kwargs):
        result = original_finish(builder, *args, **kwargs)
        finished.append(result[0])
        return result

    monkeypatch.setattr(_ControlSSABuilder, "finish", capture_finish)
    book, token = begin_identity_book()
    try:
        _lower(
            source, f"exception_try_{action}_edge", complete=True,
            identity_book=book,
        )
        program, = observed

        def visit(block):
            yield block
            if isinstance(block, SequenceBlock):
                arms = block.blocks
            elif isinstance(block, ConditionalBlock):
                arms = (block.body, block.orelse)
            elif isinstance(block, (LoopBlock, WhileBlock)):
                arms = (block.body, *block.terminal_controls)
            elif isinstance(block, NativeTryBlock):
                arms = (
                    block.body, *(handler.body for handler in block.handlers),
                    block.orelse, block.finalbody,
                )
            else:
                arms = ()
            for arm in arms:
                if arm is not None:
                    yield from visit(arm)

        exits = tuple(
            block for block in visit(program.root)
            if isinstance(block, LoopControlBlock)
            and block.action == action
        )
        assert len(exits) == 1
        site = exits[0].exception_exit_site_cell
        assert site is not None
        tries = tuple(
            block for block in visit(program.root)
            if isinstance(block, NativeTryBlock)
        )
        assert len(tries) == 1
        assert site in tries[0].abrupt_exit_site_cells
        assert exits[0] in tuple(visit(tries[0].body))
        output_rows = tuple(
            row for row in book.page(
                SOURCE_EXCEPTION_FINALIZER_OUTPUT
            ).rows() if row[2] == site
        )
        assert len(output_rows) == 1
        ssa_rows = tuple(
            row for row in book.page(
                SOURCE_EXCEPTION_SSA_LOOP_EXIT_FINALIZER
            ).rows() if row[1] == site
        )
        assert len(ssa_rows) == 1
        admission_rows = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
        assert len(admission_rows) == 1
        admission = book.page(SOURCE_EXCEPTION_ADMISSION).latest(
            admission_rows[0]
        )
        assert admission.abrupt_exit_proofs == ((
            site,
            book.latest_ref(SOURCE_EXCEPTION_SSA_LOOP_EXIT_FINALIZER,
                            ssa_rows[0]),
        ),)
        receipt = book.page(
            SOURCE_EXCEPTION_SSA_LOOP_EXIT_FINALIZER
        ).latest(ssa_rows[0])
        assert len(receipt.output_values) == 1
        versions = tuple(
            book.page(SOURCE_EXCEPTION_SSA_FINALIZER_OUTPUT).latest(row)
            for row in book.page(
                SOURCE_EXCEPTION_SSA_FINALIZER_OUTPUT
            ).rows()
            if row[4] == book.page(
                SOURCE_EXCEPTION_FINALIZER_OUTPUT
            ).latest(output_rows[0]).canonical_output.row[1]
        )
        assert any(version.exit_site == site for version in versions)
        assert any(version.exit_site is None for version in versions)
        if guarded:
            assert len({version.ssa_value for version in versions}) == 2
        assert CorrelationTable._source_exception_ssa_finalizer_output_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert CorrelationTable._source_exception_ssa_loop_exit_finalizer_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        exit_blocks = tuple(
            basic for function in finished
            for basic in function.blocks.values()
            if any(instruction.attributes.get("source_control") == action
                   for instruction in basic.instrs)
        )
        assert len(exit_blocks) == 1
        instructions = exit_blocks[0].instrs
        assert str(instructions[-1].op) == "Br"
        assert any(str(instruction.op) == "Call"
                   for instruction in instructions[:-1])
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize(
    "action,guarded,expected",
    [("break", False, 1), ("continue", False, -1),
     ("break", True, 2), ("continue", True, -1)],
)
def test_try_loop_exit_finalizer_executes_on_correct_edge(
    backend, action, guarded, expected, tmp_path,
):
    body = (
        f"            if x > 1:\n                {action}\n"
        "            x = x - 1\n"
        if guarded else
        f"            x = x - 1\n            {action}\n"
    )
    source = f"""\
def root(x: int) -> int:
    while x > 0:
        try:
{body.rstrip()}
        finally:
            x = x - 1
    return x
"""
    module, _outputs, exports = _lower(
        source, f"exception_loop_{action}_{guarded}_{backend}",
        complete=True,
    )
    function = module.functions[exports[0]]
    parameter = dict(function.metadata["parameter_names"])["x"]
    returns = tuple(
        instruction for basic in function.blocks.values()
        for instruction in basic.instrs if str(instruction.op) == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    output = int(returns[0].args[0].id)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda value: artifact.prepare_execution({
            parameter: np.array([value], dtype=np.int64),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda value: prepare_artifact_execution(artifact, {
            parameter: np.array([value], dtype=np.int64),
        }).run()
    assert int(run(3).buffers[output].reshape(-1)[0]) == expected
    assert int(run(0).buffers[output].reshape(-1)[0]) == 0
