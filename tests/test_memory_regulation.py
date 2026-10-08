"""Memory regulation of a compile, proven on the data structures (no lowering).

Every test here drives the regulator, the caches and the book directly with
small hand-built inputs; none of them compiles anything.
"""
from __future__ import annotations

import gc
from types import SimpleNamespace

import networkx as nx
import pytest

from src.compiler import memory_regulation as regulation
from src.compiler.concordance_declarations import (
    COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, MemoryRelease,
)
from src.compiler.identity_concordance import IdentityBook, IdentityPage
from src.compiler.work_contract import (
    PRESETS, WorkContract, active_contract, set_active_contract,
)


def _regulator(book, budget, readings, monkeypatch, items):
    """A regulator whose resident-set reads come from ``readings`` (one per
    call, the last repeated) and whose release order is ``items``."""

    sequence = list(readings)

    def read():
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]

    monkeypatch.setattr(regulation, "process_rss_bytes", read)
    monkeypatch.setattr(regulation, "RELEASE_ORDER", tuple(items))
    return regulation.MemoryRegulator(book, budget)


def test_under_budget_posts_nothing_and_leaves_the_book_unchanged(monkeypatch):
    book = IdentityBook()
    called = []
    regulator = _regulator(
        book, 1000, [10, 20, 30], monkeypatch,
        [("never", lambda: called.append("never") or 1)],
    )
    for label in ("a", "b", "c"):
        regulator.boundary(label)
    assert called == []
    assert [label for label, _ in regulator.history] == ["a", "b", "c"]
    assert MEMORY_RELEASE_RECEIPT.name not in book.pages
    assert COMPILE_POLICY.name not in book.pages


def test_no_budget_never_releases(monkeypatch):
    book = IdentityBook()
    called = []
    regulator = _regulator(
        book, None, [10**12], monkeypatch,
        [("never", lambda: called.append("never") or 1)],
    )
    regulator.boundary("a")
    assert called == []
    assert MEMORY_RELEASE_RECEIPT.name not in book.pages


def test_over_budget_releases_in_declared_order_until_under_and_posts_receipts(
    monkeypatch,
):
    book = IdentityBook()
    log = []
    # Readings, in the order boundary() takes them: the boundary reading,
    # then (before, after) around each release it makes.
    readings = [500, 500, 400, 400, 90, 90]
    regulator = _regulator(
        book, 100, readings, monkeypatch,
        [
            ("first", lambda: log.append("first") or 7),
            ("second", lambda: log.append("second") or 3),
            ("third", lambda: log.append("third") or 1),
        ],
    )
    assert regulator.boundary("stage x") == 90
    # Stops at the item that brought it under budget; "third" never ran.
    assert log == ["first", "second"]
    page = book.pages[MEMORY_RELEASE_RECEIPT.name]
    rows = page.rows()
    assert rows == (("stage x", 0), ("stage x", 1))
    assert page.latest(rows[0]) == MemoryRelease("first", 500, 400, 100, 7)
    assert page.latest(rows[1]) == MemoryRelease("second", 400, 90, 100, 3)
    # Each receipt is DERIVED from the policy cell, which states the budget.
    policy = book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",))
    assert book.pages[COMPILE_POLICY.name].latest(
        ("memory_budget_bytes",)
    ) == "100"
    for ordinal in (0, 1):
        ref = book.latest_ref(MEMORY_RELEASE_RECEIPT, ("stage x", ordinal))
        assert [source for source, _stage in book.edges_into(ref)] == [policy]


def test_release_ordinals_continue_across_boundaries(monkeypatch):
    book = IdentityBook()
    regulator = _regulator(
        book, 100, [500, 500, 50, 500, 500, 50], monkeypatch,
        [("only", lambda: 1)],
    )
    regulator.boundary("one")
    regulator.boundary("two")
    assert book.pages[MEMORY_RELEASE_RECEIPT.name].rows() == (
        ("one", 0), ("two", 1),
    )


def test_work_contract_budget_is_validated_and_defaults_to_none():
    assert all(preset.memory_budget_bytes is None for preset in PRESETS.values())
    assert WorkContract(
        "t", register_reuse=False, inexact_identities=False,
        contract_multiply_add=False, memory_budget_bytes=1 << 30,
    ).memory_budget_bytes == 1 << 30
    for bad in (0, -5, True, "8GB", 1.5):
        with pytest.raises(ValueError):
            WorkContract(
                "t", register_reuse=False, inexact_identities=False,
                contract_multiply_add=False, memory_budget_bytes=bad,
            )


def test_environment_budget_overrides_without_renaming_the_contract(monkeypatch):
    set_active_contract(None)
    monkeypatch.delenv("TURING_WORK_CONTRACT", raising=False)
    monkeypatch.delenv("TURING_POW_INEXACT", raising=False)
    monkeypatch.delenv("TURING_FMA_CONTRACT", raising=False)
    monkeypatch.delenv("TURING_MEMORY_BUDGET_BYTES", raising=False)
    assert active_contract() is PRESETS["develop"]
    monkeypatch.setenv("TURING_MEMORY_BUDGET_BYTES", str(3 << 30))
    contract = active_contract()
    assert contract.memory_budget_bytes == 3 << 30
    assert contract.name == "develop"


def test_dependency_level_cache_holds_live_graphs_only():
    from src.compiler import region_scheduling as scheduling

    graph = SimpleNamespace(G=nx.DiGraph([(0, 1), (1, 2)]))
    key = id(graph.G)
    assert scheduling._dependency_levels(graph) == {0: 0, 1: 1, 2: 2}
    assert key in scheduling._DEPENDENCY_LEVEL_CACHE
    del graph
    gc.collect()
    assert key not in scheduling._DEPENDENCY_LEVEL_CACHE


def test_release_dependency_levels_drops_recomputable_tables():
    from src.compiler import region_scheduling as scheduling

    graph = SimpleNamespace(G=nx.DiGraph([(0, 1)]))
    scheduling._dependency_levels(graph)
    assert scheduling.release_dependency_levels() >= 1
    assert not scheduling._DEPENDENCY_LEVEL_CACHE
    # Recomputed identically on the next ask.
    assert scheduling._dependency_levels(graph) == {0: 0, 1: 1}


def test_polymorphic_formal_scan_memo_holds_live_pages_only():
    from src.compiler import glsl_deployment_strategy as strategy

    page = IdentityPage("formal_shape")
    key = id(page)
    assert strategy._owner_has_polymorphic_formal(page, "f") is False
    assert key in strategy._POLYMORPHIC_FORMAL_SCANS
    del page
    gc.collect()
    assert key not in strategy._POLYMORPHIC_FORMAL_SCANS


def test_compile_end_releases_only_the_state_keyed_to_its_own_book():
    from src.compiler import glsl_deployment_strategy as strategy

    mine, other = IdentityBook(), IdentityBook()
    shells = strategy._CALLSITE_SHELL_TYPE_CACHE
    children = strategy._CHILD_SIGNATURE_RESULTS
    shells[(1, id(mine), 10, ())] = object
    shells[(1, id(mine), 11, ())] = object
    shells[(1, id(other), 10, ())] = object
    children[id(mine)] = (mine, {"a": 1, "b": 2, "c": 3})
    children[id(other)] = (other, {"x": 1})
    try:
        released = strategy.release_compile_planning_state(mine)
        assert released == {
            "callsite_shell_types": 2, "child_signature_results": 3,
        }
        assert [key for key in shells if key[1] == id(mine)] == []
        assert (1, id(other), 10, ()) in shells
        assert id(mine) not in children
        assert id(other) in children
    finally:
        for key in [key for key in shells if key[1] in (id(mine), id(other))]:
            del shells[key]
        children.pop(id(mine), None)
        children.pop(id(other), None)


def test_a_cell_key_is_one_tuple_shared_by_cells_and_stamps():
    page = IdentityPage("p")
    page.set(("scope", 1), 0, "fact")
    (cell_key,) = page.cells
    (stamp_key,) = page.stamps
    assert cell_key is stamp_key
    assert page.latest(("scope", 1)) == "fact"
    page.revise(("scope", 1), "next")
    assert [column for column, _ in page.history(("scope", 1))] == [0, 1]


def test_stage_boundary_is_a_noop_outside_a_regulated_compile():
    assert regulation._ACTIVE.get() is None
    regulation.stage_boundary("nothing is open")
    regulation.stage_end("nothing is open")


def test_nested_regulation_releases_only_at_the_outermost_compile(monkeypatch):
    collected = []
    monkeypatch.setattr(regulation.gc, "collect", lambda: collected.append(1) or 0)
    outer_book, inner_book = IdentityBook(), IdentityBook()
    outer, outer_token, outer_owns = regulation.begin_regulation(outer_book)
    inner, inner_token, inner_owns = regulation.begin_regulation(inner_book)
    assert (outer_owns, inner_owns) == (True, False)
    regulation.end_regulation(
        inner, inner_token, releases_book_state=inner_owns,
    )
    assert collected == []
    # The outer regulation is the active one again.
    assert regulation._ACTIVE.get() is outer
    regulation.end_regulation(
        outer, outer_token, releases_book_state=outer_owns,
    )
    assert collected == [1]
    assert regulation._ACTIVE.get() is None


def test_a_compile_resuming_a_callers_book_keeps_that_books_planning_state():
    book = IdentityBook()
    regulator, token, releases = regulation.begin_regulation(
        book, owned_book=False,
    )
    assert releases is False
    regulation.end_regulation(regulator, token, releases_book_state=releases)


def test_released_dispatch_region_copies_are_gone_and_reads_fail_loudly():
    from src.compiler.glsl_deployment_strategy import (
        ReleasedDispatchRegions, release_dispatch_region_copies,
    )

    class Shell:
        dispatch_subgraphs = ("r0", "r1", "r2")
        deep_compilers = ("c0",)
        ephemeral_callables = ()
        process_graph = "kept"

    class Other(Shell):
        pass

    first, second, third = Shell(), Shell(), Other()
    first.deep_compilers = ("instance override", "x")   # an instance override
    released = release_dispatch_region_copies([first, second, third])
    # Counted once per owner: the Shell class and the Other subclass (which
    # inherits, so owns nothing) are not double counted; the instance
    # override is its own owner.
    assert released == {
        "dispatch_subgraphs": 3, "deep_compilers": 1 + 2,
        "ephemeral_callables": 0,
    }
    for shell in (first, second, third):
        assert isinstance(shell.dispatch_subgraphs, ReleasedDispatchRegions)
        for read in (len, iter, bool, lambda held: held[0]):
            with pytest.raises(RuntimeError, match="released"):
                read(shell.dispatch_subgraphs)
        assert shell.process_graph == "kept"
    # Releasing again lets go of nothing.
    assert release_dispatch_region_copies([first, second, third]) == {
        "dispatch_subgraphs": 0, "deep_compilers": 0,
        "ephemeral_callables": 0,
    }
