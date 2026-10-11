"""Bounded, seconds-sized proofs of the integrated dispatch work path."""
import ast
import dataclasses
from pathlib import Path
import threading
import re
from types import SimpleNamespace

import networkx as nx
import pytest

from src.compiler.graph_work import (
    RegionProjectionSequence, RegionTopologyPool, bounded_schedule_regions,
    dependency_snapshot, publish_schedule, resident_region,
    schedule_dependency_snapshot,
)
from src.compiler.identity_concordance import IdentityBook
from src.compiler.work_contract import PRESETS, active_contract, set_active_contract


def _region(offset=0):
    graph = nx.DiGraph()
    expression = ast.parse("x + 1", mode="eval").body
    graph.add_node(offset, parents=[], children=[(offset + 1, "value")], expr_obj=expression,
                   tensor={"shape": [4], "dtype": "float64"})
    graph.add_node(offset + 1, parents=[(offset, "value")], children=[])
    graph.add_edge(offset, offset + 1, role="value")
    graph.graph["deployment_nodes"] = (offset, offset + 1)
    return SimpleNamespace(G=graph, levels={offset: 0, offset + 1: 1},
                           scheduler=SimpleNamespace(G=graph, levels_asap={}, levels_alap={}))


def test_many_region_headers_hold_only_bounded_topologies_and_preserve_ast_identity():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        regions = []
        expressions = []
        for index in range(12):
            region = _region(index * 2)
            expressions.append(region.G.nodes[index * 2]["expr_obj"])
            regions.append(pool.add(region))
        assert len(pool.resident) == pool.stats["resident_peak"] == 1
        assert pool.stats["spills"] == 11
        # Cached NetworkX views and nested data reads remain valid after eviction.
        view = regions[0].G.nodes
        for index, region in enumerate(regions):
            assert tuple(nx.lexicographical_topological_sort(region.G)) == (index * 2, index * 2 + 1)
            assert region.G.nodes[index * 2]["expr_obj"] is expressions[index]
            assert dict(region.levels) == {index * 2: 0, index * 2 + 1: 1}
        assert view[0]["tensor"]["shape"] == [4]
        assert pool.stats["resident_peak"] == 1
        assert pool.stats["reloads"] >= 12
    finally:
        paths = (pool.store.path, pool.store.index_path)
        pool.close()
        assert all(not Path(path).exists() for path in paths)


def test_nested_writes_require_a_lease_and_survive_spill_reload():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        first, second = pool.add(_region()), pool.add(_region(2))
        with pytest.raises(RuntimeError, match="write lease"):
            first.G.nodes[0]["tensor"]["shape"].append(7)
        with resident_region(first):
            escaped = first.G.nodes[0]
            escaped["tensor"]["shape"].append(7)
            escaped["tensor"]["dtype"] = "int64"
        with pytest.raises(RuntimeError, match="write lease"):
            escaped["tensor"]["dtype"] = "float32"
        with pytest.raises(RuntimeError, match="write lease"):
            escaped["tensor"]["shape"].append(9)
        assert second.G.number_of_nodes() == 2  # evict the dirty first region
        assert first.G.nodes[0]["tensor"] == {"shape": [4, 7], "dtype": "int64"}
    finally:
        pool.close()


def test_a_pinned_region_refuses_over_admission_without_losing_data():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        region = pool.add(_region())
        with resident_region(region):
            with pytest.raises(RuntimeError, match="active write leases"):
                pool.add(_region(2))
        assert dict(region.levels) == {0: 0, 1: 1}
    finally:
        pool.close()


def test_escaped_write_view_cannot_mutate_a_new_residency_generation():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        first, second = pool.add(_region()), pool.add(_region(2))
        with resident_region(first):
            escaped = first.G.nodes[0]["tensor"]
        assert second.G.number_of_nodes() == 2  # expire first's old generation
        with resident_region(first):
            with pytest.raises(RuntimeError, match="expired residency generation"):
                escaped["dtype"] = "int32"
            first.G.nodes[0]["tensor"]["dtype"] = "int64"
        assert first.G.nodes[0]["tensor"]["dtype"] == "int64"
    finally:
        pool.close()


def test_cyclic_metadata_container_identity_survives_spill_reload_and_write_leases():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        first = _region()
        items = []
        pair = (items,)
        items.append(pair)
        first.G.nodes[0]["cycle"] = pair
        pool.add(first)
        pool.add(_region(2))
        with resident_region(first):
            restored = first.G.nodes[0]["cycle"]
            assert restored[0][0] is restored
    finally:
        pool.close()


def test_set_attributes_preserve_type_and_refuse_escaped_writes():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        first = _region()
        first.G.nodes[0]["flags"] = {"read"}
        pool.add(first)
        with pytest.raises(RuntimeError, match="write lease"):
            first.G.nodes[0]["flags"].add("write")
        with resident_region(first):
            escaped = first.G.nodes[0]["flags"]
            escaped.add("observed")
        with pytest.raises(RuntimeError, match="write lease"):
            escaped.add("write")
        pool.add(_region(2))
        assert isinstance(first.G.nodes[0]["flags"], set)
        assert first.G.nodes[0]["flags"] == {"read", "observed"}
    finally:
        pool.close()


def test_in_lease_assignments_preserve_current_container_alias_identity():
    pool = RegionTopologyPool(IdentityBook(), 1)
    try:
        first = pool.add(_region())
        with resident_region(first):
            tensor = first.G.nodes[0]["tensor"]
            first.G.nodes[1]["shared"] = tensor
            assert first.G.nodes[1]["shared"] is tensor
            tensor["dtype"] = "int64"
            assert first.G.nodes[1]["shared"]["dtype"] == "int64"
        pool.add(_region(2))
        with resident_region(first):
            assert first.G.nodes[0]["tensor"] is first.G.nodes[1]["shared"]
    finally:
        pool.close()


def test_stale_schedule_proposal_is_rejected_before_publication():
    region = _region()
    snapshot = dependency_snapshot(region, "asap")
    proposal = schedule_dependency_snapshot(snapshot)
    original = dict(region.levels)
    region.G.nodes[1]["parents"] = []
    with pytest.raises(RuntimeError, match="stale"):
        publish_schedule(region, snapshot, proposal)
    assert region.levels == original
    assert "compartment_schedule" not in region.G.graph


@pytest.mark.parametrize("preference", ["asap", "alap"])
def test_worker_proposal_is_the_existing_scheduler_exactly(preference):
    from src.transmogrifier.ilpscheduler import ILPScheduler
    region = _region()
    region.G.add_node(9, parents=[], children=[])
    snapshot = dependency_snapshot(region, preference)
    proposal = schedule_dependency_snapshot(snapshot)
    assert proposal.levels == ILPScheduler(region).compute_levels(preference, "dependency")
    publish_schedule(region, snapshot, proposal)
    assert tuple(n for _level, nodes in proposal.schedule for n in nodes) == (0, 9, 1)


def test_ordered_publication_and_bound_include_completed_results(monkeypatch):
    # This test deliberately substitutes only the worker operation: it proves
    # the admission mechanism, not native CPU parallelism (proven separately).
    from src.compiler import graph_work
    from src.compiler.native_compiler_accelerators import compiler_accelerators
    first_release, second_finished = threading.Event(), threading.Event()
    scheduled = graph_work.schedule_dependency_snapshot
    coordinator = threading.get_ident()
    produced, published, worker_ids = [], [], set()

    def operation(snapshot):
        worker_ids.add(threading.get_ident())
        if snapshot.records[0][0] == 0:
            assert second_finished.wait(5)
            # The generator cannot run beyond the two admitted items while
            # ordinal zero is waiting, even though ordinal one has completed.
            assert produced == [0, 1]
            first_release.set()
        else:
            second_finished.set()
        return scheduled(snapshot)

    monkeypatch.setattr(graph_work, "schedule_dependency_snapshot", operation)
    original_resolve = compiler_accelerators.resolve
    monkeypatch.setattr(compiler_accelerators, "resolve",
                        lambda semantic: dataclasses.replace(original_resolve(semantic), tier="native"))

    def regions():
        for index in range(9):
            produced.append(index)
            region = _region(index * 2)
            yield index, region, dependency_snapshot(region, "asap")

    def finish(index, *_args):
        assert threading.get_ident() == coordinator
        published.append(index)

    bounded_schedule_regions(regions(), workers=2, admission_limit=2, finish=finish)
    assert first_release.is_set()
    assert published == list(range(9))
    assert len(worker_ids) == 2 and coordinator not in worker_ids


def test_projection_constructors_are_lazy_and_keep_per_region_identity():
    calls = []
    sequence = RegionProjectionSequence(20, lambda index: calls.append(index) or object())
    assert len(sequence) == 20 and not calls
    first = sequence[3]
    assert sequence[3] is first and calls == [3]
    assert sequence[-1] is sequence[19]
    assert calls == [3, 19]


@pytest.mark.parametrize("field", ["graph_resident_regions", "graph_workers"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_work_contract_refuses_unhonorable_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(PRESETS["develop"], **{field: value})


def _lower_small(calls, *, resident_regions, workers=1):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    repo = Path(__file__).resolve().parents[1]
    contracts = repo / "extraction_contracts"
    contract = ExtractionContract(contracts / "program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [{
            "function": "root", "parameter": "x", "storage": "span",
            "dtype": "float64", "rank": 1, "shape": [4],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    }).with_execution_file(contracts / "vehicle_full_native_execution.yaml")
    source = "def law(x):\n    return (x + 1.0) * (x - 1.0)\n\ndef root(x):\n"
    for index in range(calls):
        source += f"    y{index} = law(x + {float(index)!r})\n"
    source += "    return " + " + ".join(f"y{i}" for i in range(calls)) + "\n"
    from src.compiler import work_contract
    previous = work_contract._active
    set_active_contract(dataclasses.replace(PRESETS["develop"],
                        graph_resident_regions=resident_regions, graph_workers=workers))
    try:
        return lower_ast_source_to_ssa(source, "root", name="graph_work_proof",
            extraction_contract=contract, runtime_closure_only=True, python_bindings={},
            progress=lambda message: None)
    finally:
        set_active_contract(previous)


def _alpha_llvm(text):
    # Independent books draw distinct IDs from the process-global monotonic
    # issuer. Compare strict SSA alpha-equivalence, not numeric tolerances or
    # changed opcodes/constants. Local identifiers are scoped to each function.
    names = {}
    lines = []
    for line in text.splitlines():
        if line.startswith("define "):
            names = {}
        def rename(match):
            name = match.group(0)
            return names.setdefault(name, f"%alpha.{len(names)}")
        lines.append(re.sub(r"%[-a-zA-Z$._0-9]+", rename, line))
    return "\n".join(lines)


def test_public_entry_spills_and_reloads_with_exact_emitted_parity():
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    eager, _outputs, _exports = _lower_small(4, resident_regions=None)
    paged, _outputs, _exports = _lower_small(4, resident_regions=1)
    report = concordance_report(paged)
    assert report.splitlines()[0].endswith(", 0 finding(s)"), report
    eager_ir = emit_ssa_function_to_llvm(eager, "graph_work_proof__root")
    paged_ir = emit_ssa_function_to_llvm(paged, "graph_work_proof__root")
    assert eager_ir.complete and paged_ir.complete
    assert _alpha_llvm(paged_ir.llvm_ir) == _alpha_llvm(eager_ir.llvm_ir)
    stats = paged.metadata["graph_work"]
    assert stats["resident_peak"] == stats["resident_limit"] == 1
    assert stats["spills"] > 0
    # This source-SSA path reads only region headers; no pointless payload
    # reload is required. Actual reload/mutation is proven by the pool micros.
    assert stats["reloads"] == 0
    assert stats["admitted"] == stats["published"] > 4
    assert stats["resident_now"] == 0
    assert not stats["source_and_ssa_paged"]


def test_concordance_rejects_a_worker_receipt_disagreeing_with_its_region():
    from src.compiler.identity_concordance import CorrelationTable
    module, _, _ = _lower_small(2, resident_regions=1)
    assert CorrelationTable._graph_work_publication_findings(module) == []
    page = module.metadata["identity_book"].page("graph_work_publication")
    row = page.rows()[0]
    fact = page.latest(row)
    page.set(row, 100, dataclasses.replace(fact,
        schedule_preference="alap" if fact.schedule_preference == "asap" else "asap"))
    findings = CorrelationTable._graph_work_publication_findings(module)
    assert len(findings) == 1 and findings[0].kind == "graph-work-publication-disagreement"


def test_public_entry_native_workers_publish_on_coordinator_with_paged_parity(tmp_path, monkeypatch):
    from itertools import chain, islice
    from src.compiler import graph_work
    from src.compiler.identity_concordance import IdentityBook, concordance_report
    from src.compiler.native_compiler_accelerators import (
        TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators, compile_and_register_topology_accelerator,
    )
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    # Build through our compiler, before the program's book opens. The source
    # entry also supports this bootstrap itself; retaining this provider lets
    # the micro force and observe two real native calls at the integration seam.
    provider = compile_and_register_topology_accelerator(tmp_path)
    eager, _, _ = _lower_small(4, resident_regions=None)
    coordinator = threading.get_ident()
    worker_ids, publication_ids = set(), set()
    phase = dict(barrier=None, count=0)
    lock = threading.Lock()
    native = provider.run
    bounded = graph_work.bounded_schedule_regions
    post = IdentityBook.post

    def observed_native(graph, **kwargs):
        if threading.current_thread().name.startswith("compiler-graph"):
            worker_ids.add(threading.get_ident())
            with lock:
                phase["count"] += 1
                barrier = phase["barrier"] if phase["count"] <= 2 else None
            if barrier is not None:
                barrier.wait(5)
        return native(graph, **kwargs)

    def observed_work(regions, **kwargs):
        iterator = iter(regions)
        heads = list(islice(iterator, kwargs["admission_limit"]))
        phase.update(barrier=threading.Barrier(2) if len(heads) == 2 else None, count=0)
        return bounded(chain(heads, iterator), **kwargs)

    def observed_post(self, *args, **kwargs):
        publication_ids.add(threading.get_ident())
        assert threading.get_ident() == coordinator
        return post(self, *args, **kwargs)

    monkeypatch.setattr(provider, "run", observed_native)
    monkeypatch.setattr(graph_work, "bounded_schedule_regions", observed_work)
    monkeypatch.setattr(IdentityBook, "post", observed_post)
    try:
        paged, _, _ = _lower_small(4, resident_regions=2, workers=2)
    finally:
        compiler_accelerators.unregister_native(TOPOLOGICAL_ORDER)
        compiler_accelerators.unregister_native(DEPENDENCY_LEVELS)
    assert len(worker_ids) == 2 and coordinator not in worker_ids
    assert publication_ids == {coordinator}
    assert concordance_report(paged).splitlines()[0].endswith(", 0 finding(s)")
    assert _alpha_llvm(emit_ssa_function_to_llvm(paged, "graph_work_proof__root").llvm_ir) == (
        _alpha_llvm(emit_ssa_function_to_llvm(eager, "graph_work_proof__root").llvm_ir)
    )
    stats = paged.metadata["graph_work"]
    assert stats["worker_threads"] >= 2
    assert stats["outstanding_peak"] == stats["admission_limit"] == 2
    assert stats["resident_peak"] <= 2
    assert stats["spills"] > 0 and stats["reloads"] == 0
    assert stats["live_topology_peak"] <= 2
