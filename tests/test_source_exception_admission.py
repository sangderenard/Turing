"""An authored catch must not disappear from a complete native artifact."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import CompilationSubdivisionRequired
from src.compiler.concordance_declarations import (
    CONTROL_BLOCK, ControlBlockKind,
    NAME_BINDING, SOURCE_EXCEPTION_ADMISSION, SOURCE_EXCEPTION_SITE_STATE,
    SOURCE_EXCEPTION_TRANSFER, SOURCE_EXCEPTION_HANDLER_CLASS,
    SOURCE_EXCEPTION_STATUS_ROUTE, NATIVE_STATUS_CLASS,
    SOURCE_EXCEPTION_REGION_SITE,
    SOURCE_EXCEPTION_REGION_STATE,
    SOURCE_EXCEPTION_SSA_CHECKPOINT,
    SOURCE_EXCEPTION_SSA_ROUTE,
    SOURCE_EXCEPTION_SSA_HANDLER_ARM,
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


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _lower(source: str, name: str, **kwargs):
    return lower_ast_source_to_ssa(
        source, "root", name=name, extraction_contract=CONTRACT,
        progress=lambda _message: None, progress_bars=False,
        **kwargs,
    )


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
            region_signatures={region_rows[0][1]: ((), ())},
        )
        assert any(
            shortfall.name == "native_try_dispatch" for shortfall in shortfalls
        )
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
    assert boundary["kind"] == "source-raise-control-owner"
    assert boundary["source_raise_cell"] == repr(source_cell)


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
        sites = tuple(book.page(SOURCE_EXCEPTION_SITE_STATE).rows())
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
        assert sorted(len(fact.sites) for fact in try_facts) == [2, 2]
        assert all(
            site.page == SOURCE_EXCEPTION_TRANSFER
            for fact in try_facts for site in fact.sites
        )
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
        assert any(
            shortfall.name == "native_try_dispatch" for shortfall in shortfalls
        )
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


def test_cast_without_authored_try_still_lowers():
    source = """\
def root(x: float) -> int:
    return int(x)
"""
    module, _outputs, exports = _lower(source, "exception_admission_plain")
    assert exports[0] in module.functions
