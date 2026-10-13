"""Small real mutable storage tests; no full orbital compile as a diagnostic."""
import ast
from dataclasses import dataclass
import pickle
import sys
from threading import current_thread
import time

import networkx as nx
import pytest

from src.compiler.compressed_tiles import DiskTileStore
from src.compiler.live_tiles import (
    LiveTileArena, TiledDict, TiledList, TiledObject, _ACTIVE_ARENA, live_digraph,
)


def arena(tmp_path):
    owner = LiveTileArena(tile_rows=4, hot_tiles=1, bucket_rows=4, directory=tmp_path)
    owner.compress_resident_ram()  # Most storage tests deliberately force the bounded tier.
    return owner


def test_empty_map_reinsertion_routes_to_its_current_root(tmp_path):
    owner = arena(tmp_path)
    values = owner.manage({2: "first"})
    assert values[2] == "first"  # Retain this hash's route to the old root.
    del values[2]
    owner.flush()
    values[2] = "second"
    owner.flush()
    assert list(values) == [2]
    assert values[2] == "second"
    assert len(values) == 1


def test_rebuilt_graph_edges_remain_visible_to_ancestry(tmp_path):
    owner = arena(tmp_path)
    graph = live_digraph(owner)
    graph.add_edges_from(((1, 2), (2, 3)))
    graph.remove_edges_from(tuple(graph.edges))
    owner.flush()
    graph.add_edges_from(((1, 2), (2, 3)))
    owner.flush()
    assert set(graph.edges) == {(1, 2), (2, 3)}
    assert nx.ancestors(graph, 3) == {1, 2}
    assert graph.has_edge(1, 2)


def test_storage_collection_reports_its_actual_traversal_and_compaction(tmp_path):
    from types import SimpleNamespace
    from src.compiler.shell_telemetry import CompileProgress
    owner = arena(tmp_path)
    values = owner.manage({i: {"value": i} for i in range(24)})
    owner.offload_to_disk()
    messages = []
    layer = CompileProgress(enabled=False, collect_profile=True, heartbeat=None)
    layer.owner_display = SimpleNamespace(log=messages.append)
    with layer:
        layer.stage("compiler-finalization")
        owner.collect_unused()
    for phase in ("trace live handles", "sweep unreachable records", "restore retained search pages"):
        assert any(f"storage collection: {phase}: begin" in line for line in messages)
    assert any("storage compaction: compress live tiles: begin" in line for line in messages)
    work = layer.summary()["work"]["compiler-finalization"]
    assert work["storage-mark"]["done"] > 0
    assert work["storage-sweep"]["done"] == work["storage-sweep"]["total"]
    assert work["storage-compaction"]["done"] == work["storage-compaction"]["total"]
    assert values[17]["value"] == 17


def test_pressure_ladder_stays_fully_decoded_until_explicit_escalation(tmp_path):
    owner = LiveTileArena(tile_rows=2, hot_tiles=1, bucket_rows=1,
                          decoded_budget_bytes=256, directory=tmp_path)
    values = [owner.manage({"value": index, "payload": "x" * 200})
              for index in range(8)]

    fast = owner.stats()
    assert fast.pressure_level == 0
    assert fast.hot_tiles > owner.hot_limit
    assert fast.compressions == 0
    assert [value["value"] for value in values] == list(range(8))

    tail = owner.offload_cold_tail()
    assert tail.pressure_level == 1
    assert tail.payload_hot_tiles == 1
    assert tail.hot_tiles == tail.retained_index_tiles + 1
    assert tail.disk_tiles > 0

    ram = owner.compress_resident_ram()
    assert ram.pressure_level == 2
    assert ram.payload_hot_tiles == 0
    assert ram.hot_tiles == ram.retained_index_tiles
    assert [value["value"] for value in values] == list(range(8))

    minimum = owner.offload_to_disk()
    assert minimum.pressure_level == 3
    assert owner.hot_limit == 1
    assert owner.payload_limit == 1
    assert minimum.payload_hot_tiles == 0
    assert minimum.hot_tiles == minimum.retained_index_tiles
    assert [value["value"] for value in values] == list(range(8))


def test_compressed_backing_can_expand_and_contract_its_decoded_front(tmp_path):
    owner = LiveTileArena(
        tile_rows=2, hot_tiles=1, bucket_rows=1,
        decoded_budget_bytes=256, directory=tmp_path,
    )
    values = [owner.manage({"value": index, "payload": "x" * 200})
              for index in range(8)]
    owner.compress_resident_ram()

    owner.expand_decoded_front(4096)
    assert owner.pressure_level == 2
    assert owner.decoded_budget_bytes == 4096
    assert [value["value"] for value in values] == list(range(8))
    expanded = owner.stats()
    assert expanded.payload_hot_tiles > owner.base_payload_limit

    contracted = owner.contract_decoded_front()
    assert owner.decoded_budget_bytes == owner.base_decoded_budget_bytes
    assert owner.payload_limit == owner.base_payload_limit
    assert contracted.payload_hot_tiles <= owner.base_payload_limit


def test_maximum_tier_uses_bounded_parallel_compression_jobs(tmp_path, monkeypatch):
    codec_threads = set()
    write = DiskTileStore.write

    def observed_write(self, data, rows, *, heavy=True):
        if heavy:
            codec_threads.add(current_thread().name)
        return write(self, data, rows, heavy=heavy)

    monkeypatch.setattr(DiskTileStore, "write", observed_write)
    owner = LiveTileArena(
        tile_rows=2, hot_tiles=2, bucket_rows=2, directory=tmp_path,
        compression_workers=2,
    )
    values = [owner.manage({"value": index, "payload": "x" * 4000})
              for index in range(24)]
    owner.compress_resident_ram()
    stats = owner.offload_to_disk()

    assert stats.disk_tiles > 0
    assert stats.parallel_compression_jobs > 1
    assert stats.compression_workers == 2
    assert stats.compression_seconds >= 0.0
    assert codec_threads
    assert all(name.startswith("compiler-compress") for name in codec_threads)
    assert [value["value"] for value in values] == list(range(24))


def test_maximum_pressure_retains_mutable_search_routing(tmp_path):
    owner = LiveTileArena(tile_rows=2, hot_tiles=1, bucket_rows=1,
                          decoded_budget_bytes=512, directory=tmp_path)
    mapping = owner.manage({index: index * 3 for index in range(64)})
    owner.offload_to_disk()
    before = owner.stats().reloads
    assert [mapping[index] for index in range(64)] == [index * 3 for index in range(64)]
    first_pass = owner.stats().reloads - before
    before = owner.stats().reloads
    hits = owner.stats().route_hits
    assert [mapping[index] for index in range(64)] == [index * 3 for index in range(64)]
    second_pass = owner.stats().reloads - before

    # The first pass discovers authoritative trie routes. Every later lookup
    # goes directly to its bucket: at most one payload reload per key, never a
    # repeated chain of branch-page reads.
    stats = owner.stats()
    assert stats.route_hits - hits == 64
    assert second_pass <= 64
    assert first_pass > second_pass

    for index in range(64, 96):
        mapping[index] = index * 3
    hits, misses = owner.stats().route_hits, owner.stats().route_misses
    assert [mapping[index] for index in range(96)] == [index * 3 for index in range(96)]
    stats = owner.stats()
    assert stats.route_hits - hits == 96
    assert stats.route_misses == misses


def test_mutable_nested_views_survive_eviction_and_disk(tmp_path):
    owner = arena(tmp_path)
    table = owner.manage({i: {"parents": [(i - 1, "operand")], "value": i}
                          for i in range(30)})
    parents = table[12]["parents"]
    owner.flush()
    parents.append((99, "carry"))
    owner.offload_to_disk()
    assert table[12]["parents"] is parents
    assert list(parents) == [(11, "operand"), (99, "carry")]
    table[12]["value"] = 42
    table[88] = {"value": 88}
    del table[2]
    owner.flush()
    assert list(table) == [i for i in range(30) if i != 2] + [88]
    assert table[12]["value"] == 42
    assert owner.stats().payload_hot_tiles <= 1
    assert owner.stats().disk_tiles > 0
    assert owner.stats().resident_encoded_bytes == 0


def test_setdefault_returns_the_actual_adopted_writer(tmp_path):
    owner = arena(tmp_path)
    root = TiledDict(owner)
    root.setdefault("sites", {})["break"] = {"value": 17}
    root.setdefault("parents", []).append((2, "operand"))
    owner.offload_to_disk()
    assert root["sites"]["break"]["value"] == 17
    assert list(root["parents"]) == [(2, "operand")]
    assert root.setdefault("sites", {}) is root["sites"]


def test_mutable_sets_publish_writes_after_clean_page_eviction(tmp_path):
    owner = arena(tmp_path)
    root = owner.manage({"sites_seen": set(), "expected_sites": {20, 24}})
    owner.offload_to_disk()
    root["sites_seen"].add(20)
    owner.flush()
    root["sites_seen"].add(24)
    owner.flush()
    assert set(root["expected_sites"]) - root["sites_seen"] == set()
    sites = root["sites_seen"]
    assert set(sites) == {20, 24} and frozenset(sites) == frozenset({20, 24})
    assert sites == {20, 24} and {20, 24} == sites
    assert sites & {20} == {20}
    assert {20, 25} - sites == {25}
    sites ^= {20, 25}
    assert sites == {24, 25}
    sites.clear()
    owner.flush()
    assert not root["sites_seen"]


def test_exact_semantic_leaf_and_container_cycles(tmp_path):
    owner = arena(tmp_path)
    leaf = ast.parse("x + 1").body[0].value
    original = {"expr_obj": leaf}
    original["self"] = original
    table = owner.manage(original)
    owner.offload_to_disk()
    assert table["self"] is table
    assert table["expr_obj"] is leaf
    leaf._receipt = 19
    owner.flush()
    assert table["expr_obj"]._receipt == 19


def test_cold_identity_headers_are_paged_not_strongly_pinned(tmp_path):
    import gc
    import weakref
    owner = arena(tmp_path)
    root = owner.manage({i: {"data": list(range(12))} for i in range(300)})
    held = root[17]
    weak = weakref.ref(root[23])
    owner.offload_to_disk()
    owner.flush()
    gc.collect()
    assert weak() is None
    assert owner.stats().identity_headers < 20
    assert root[17] is held
    assert list(root[23]["data"]) == list(range(12))
    first = root[23]
    owner.flush()
    assert root[23] is first


def test_networkx_real_algorithms_and_edge_aliases(tmp_path):
    owner = arena(tmp_path)
    graph = live_digraph(owner)
    graph.add_nodes_from((i, {"parents": [], "children": []}) for i in range(25))
    graph.add_edges_from((i, i + 1, {"role": "flow"}) for i in range(24))
    assert isinstance(graph, nx.DiGraph)
    assert isinstance(graph.nodes[3], dict)
    edge = graph._succ[3][4]
    assert graph._pred[4][3] is edge
    owner.offload_to_disk()
    graph.nodes[3]["parents"].append((2, "operand"))
    edge["role"] = "changed"
    assert graph._pred[4][3]["role"] == "changed"
    assert list(nx.topological_sort(graph)) == list(range(25))
    assert nx.descendants(graph, 20) == {21, 22, 23, 24}
    assert list(nx.strongly_connected_components(graph))[0] == {24}
    graph.remove_node(8)
    graph.add_edge(7, 9)
    assert list(nx.topological_sort(graph)) == [i for i in range(25) if i != 8]
    copy = graph.copy()
    assert list(copy.nodes(data=True)) == list(graph.nodes(data=True))
    assert list(copy.edges(data=True)) == list(graph.edges(data=True))


def test_networkx_map_rebinding_invalidates_cached_views(tmp_path):
    owner = arena(tmp_path)
    graph = live_digraph(owner)
    graph.add_edge(0, 1)
    old = {name: getattr(graph, name) for name in
           ("nodes", "edges", "adj", "succ", "pred", "in_edges", "out_edges",
            "degree", "in_degree", "out_degree")}
    # NetworkX's own descriptors are the writers, including _adj/_succ aliasing.
    replacement = nx.DiGraph([(10, 11), (11, 12)])
    maps = owner.manage((replacement._succ, replacement._pred, replacement._node))
    graph._succ, graph._pred, graph._node = maps
    owner.offload_to_disk()
    assert graph._adj is graph._succ
    for name in old:
        assert getattr(graph, name) is not old[name], name
        assert list(getattr(graph, name)) == list(getattr(replacement, name)), name
    assert graph._pred[11][10] is graph._succ[10][11]


def test_decoded_concordance_facts_count_their_owned_fields(tmp_path):
    from src.compiler.identity_concordance import Derived, Ref, Page
    from src.compiler.live_tiles import _owned_row_bytes
    page = Page("tile_owned_fact_test", (), str)
    fact = Derived(tuple(Ref(page, (index, str(index) + "value" * 300), 0) for index in range(50)))
    # These are serialized value records, not opaque leaves. Their tuples,
    # instance dictionaries and strings therefore belong to the decoded tile.
    assert _owned_row_bytes(fact) > 50000
    owner = LiveTileArena(tile_rows=4, hot_tiles=100, bucket_rows=4,
                          decoded_budget_bytes=32000, directory=tmp_path)
    root = owner.manage({"fact": fact})
    owner.offload_to_disk()
    assert root["fact"] == fact
    assert owner.stats().decoded_bytes <= owner.decoded_budget_bytes


def test_byte_measurements_are_cache_local_and_writes_do_not_rescan_old_rows(tmp_path, monkeypatch):
    from src.compiler import live_tiles
    owner = LiveTileArena(tile_rows=4, hot_tiles=20, bucket_rows=4,
                          decoded_budget_bytes=32000, directory=tmp_path)
    handle = owner.allocate(("old", {"a": "payload" * 100}))
    measured = live_tiles._owned_row_bytes
    visits = []
    def observe(value):
        visits.append(value)
        return measured(value)
    monkeypatch.setattr(live_tiles, "_owned_row_bytes", observe)
    replacement = ("new", {"b": "payload" * 100})
    owner.write(handle, replacement)
    assert visits == [replacement]
    assert owner.read(handle) == replacement
    owner.flush()
    assert not owner._decoded_sizes and owner.stats().decoded_bytes == 0
    assert owner.read(handle) == replacement
    assert owner.stats().decoded_bytes <= owner.decoded_budget_bytes


def test_cold_graphs_and_cached_views_do_not_pin_whole_headers(tmp_path):
    import gc
    import weakref
    owner = arena(tmp_path)
    root = TiledDict(owner)
    references = []
    for i in range(30):
        graph = live_digraph(owner)
        graph.add_edge(i, i + 1, role="operand")
        assert list(graph.nodes) == [i, i + 1]
        assert list(graph.edges) == [(i, i + 1)]
        root[i] = graph
        references.append(weakref.ref(graph))
    del graph
    owner.offload_to_disk()
    owner.flush()
    gc.collect()
    assert not any(reference() is not None for reference in references)
    assert not any(isinstance(leaf, nx.Graph) for leaf in owner._leaf_refs.values())
    held = root[12]
    assert root[12] is held
    assert list(held.edges) == [(12, 13)]
    assert held.nodes is held.nodes
    held.add_edge(13, 14)
    owner.flush()
    assert list(root[12].nodes) == [12, 13, 14]


def test_reachability_releases_dead_records_and_opaque_leaves(tmp_path):
    import gc
    import weakref
    owner = arena(tmp_path)
    live = owner.manage({"keep": list(range(100))})
    source = ast.parse("discarded + 1")
    source_ref = weakref.ref(source)
    dead = owner.manage({"expr_obj": source, "huge": ["dead " * 400 for _ in range(100)]})
    owner.offload_to_disk()
    before = owner.stats().disk_bytes
    del dead, source
    gc.collect()
    receipt = owner.collect_unused()
    gc.collect()
    assert receipt["reclaimed_records"] > 0
    assert receipt["reclaimed_leaves"] > 0
    assert source_ref() is None
    assert owner.stats().disk_bytes < before
    assert list(live["keep"]) == list(range(100))
    live["after"] = {"new": 44}
    owner.flush()
    assert live["after"]["new"] == 44


def test_reachability_preserves_shared_cycles_and_mixed_tiles(tmp_path):
    import gc
    owner = arena(tmp_path)
    live = TiledDict(owner)
    live["self"] = live
    for i in range(80):
        discarded = owner.manage({"unused": [i, i + 1]})
        live[i] = {"used": i, "cycle": live}
    del discarded
    owner.offload_to_disk()
    owner.collect_unused()
    gc.collect()
    assert live["self"] is live
    for i in range(80):
        assert live[i]["used"] == i
        assert live[i]["cycle"] is live


def test_disk_escalation_pages_the_existing_header_directory(tmp_path):
    from src.compiler.compressed_tiles import DiskTileDirectory
    owner = LiveTileArena(tile_rows=4, hot_tiles=8, bucket_rows=4, directory=tmp_path)
    root = owner.manage({i: {"value": i} for i in range(300)})
    owner.offload_to_disk()
    assert isinstance(owner._tiles, DiskTileDirectory)
    assert owner._page_kinds is owner._tiles.kinds
    assert owner._row_lengths is owner._tiles.row_lengths
    for i in (0, 149, 299):
        assert root[i]["value"] == i
    root[301] = {"value": 301}
    root[149]["value"] = -149
    owner.flush()
    owner.collect_unused()
    assert root[301]["value"] == 301 and root[149]["value"] == -149
    assert owner._tiles.resident_bytes() < 70 << 10


def test_sequence_edits_and_tuple_contracts(tmp_path):
    owner = arena(tmp_path)
    values = owner.manage(list(range(20)))
    truth = list(range(20))
    for target in (values, truth):
        target.insert(4, 98)
        target[8:11] = [99, 100]
        target[::3] = [-1] * len(target[::3])
        del target[3:6]
        target.reverse()
    owner.offload_to_disk()
    assert values == truth and truth == values
    assert list(reversed(values)) == list(reversed(truth))
    assert values + [4] == truth + [4]
    assert [4] + values == [4] + truth
    shape = owner.manage((2, 3, 4))
    assert isinstance(shape, tuple)
    assert tuple(shape) == (2, 3, 4)
    assert hash(shape) == hash((2, 3, 4))
    assert (1,) + shape == (1, 2, 3, 4)
    assert pickle.loads(pickle.dumps(shape)) == (2, 3, 4)
    assert pickle.loads(pickle.dumps(values)) == truth


def test_sequence_ordering_reads_tiles_not_empty_builtin_storage(tmp_path):
    owner = arena(tmp_path)
    for eager in ([1, 2, 3, 4, 5], (1, 2, 3, 4, 5)):
        values = owner.manage(eager)
        owner.offload_to_disk()
        smaller = type(eager)((1, 2, 3, 4, 4))
        larger = type(eager)((1, 2, 3, 4, 6))
        assert values > smaller and smaller < values
        assert values < larger and larger > values
        assert values >= eager and values <= eager


class CollidingKey:
    def __init__(self, number):
        self.number = number

    def __hash__(self):
        return 3

    def __eq__(self, other):
        return isinstance(other, CollidingKey) and self.number == other.number


def test_hash_collisions_do_not_make_unbounded_decoded_buckets(tmp_path):
    owner = arena(tmp_path)
    table = owner.manage({CollidingKey(i): i for i in range(60)})
    owner.offload_to_disk()
    for i in range(60):
        assert table[CollidingKey(i)] == i
    table[CollidingKey(13)] = 99
    del table[CollidingKey(28)]
    table[CollidingKey(80)] = 80
    assert [k.number for k in table] == [i for i in range(60) if i != 28] + [80]
    assert table[CollidingKey(13)] == 99
    for index in range(len(owner._tiles)):
        for record in tuple(owner._page(index)):
            if isinstance(record, tuple) and len(record) == 2:
                tag, payload = record
                if tag == "leaf":
                    assert len(payload) <= owner.bucket_rows
                elif tag == "collision":
                    assert len(payload[0]) <= owner.bucket_rows


def test_original_object_is_shared_after_field_eviction(tmp_path):
    @dataclass
    class Item(TiledObject):
        number: int
        args: list
    owner = arena(tmp_path)
    token = _ACTIVE_ARENA.set(owner)
    try:
        value = Item(3, [])
        consumer = Item(4, [value, value])
    finally:
        _ACTIVE_ARENA.reset(token)
    owner.offload_to_disk()
    assert consumer.args[0] is value and consumer.args[1] is value
    value.number = 17
    owner.flush()
    assert consumer.args[1].number == 17
    assert isinstance(value.__dict__, TiledDict)


def test_failed_disk_write_keeps_existing_and_dirty_data(tmp_path, monkeypatch):
    owner = arena(tmp_path)
    values = owner.manage(list(range(20)))
    owner.flush()
    store = owner._disk_store()
    write = store.write
    count = 0
    def fail(data, rows, **options):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("fixture disk full")
        return write(data, rows, **options)
    monkeypatch.setattr(store, "write", fail)
    with pytest.raises(OSError, match="disk full"):
        owner.offload_to_disk()
    monkeypatch.setattr(store, "write", write)
    assert list(values) == list(range(20))
    values[9] = 99
    owner.flush()
    assert values[9] == 99


def test_original_object_data_descriptors_keep_their_writer(tmp_path):
    class Record(TiledObject):
        def __init__(self):
            self._amount = 4
            self.__post_init__()

        @property
        def amount(self): return self._amount

        @amount.setter
        def amount(self, value): self._amount = value * 2

        @amount.deleter
        def amount(self): del self._amount

    owner = arena(tmp_path)
    token = _ACTIVE_ARENA.set(owner)
    try:
        value = Record()
        value.amount = 7
        owner.offload_to_disk()
        assert value.amount == 14
        value.__dict__["amount"] = 999
        assert value.amount == 14  # data descriptor wins over instance fields
        del value.amount
        with pytest.raises(AttributeError):
            _ = value.amount
    finally:
        _ACTIVE_ARENA.reset(token)


def test_ordered_map_deletion_never_shifts_the_order_sequence(tmp_path, monkeypatch):
    owner = LiveTileArena(tile_rows=16, hot_tiles=32, bucket_rows=16, directory=tmp_path)
    values = owner.manage({i: i * 2 for i in range(400)})
    expected = {i: i * 2 for i in range(400)}
    def forbidden(*args):
        raise AssertionError("ordered deletion attempted a linear key-list removal")
    monkeypatch.setattr(TiledList, "remove", forbidden)
    for i in range(350):
        del values[i]
        del expected[i]
    values[11] = 22
    expected[11] = 22
    owner.offload_to_disk()
    assert list(values.items()) == list(expected.items())
    assert list(reversed(values)) == list(reversed(expected))
    assert len(values._order) <= 2 * len(values) + owner.bucket_rows


@pytest.mark.parametrize("reverse", [False, True])
def test_sort_is_stable_key_once_and_never_sorts_the_whole_sequence(tmp_path, monkeypatch, reverse):
    import builtins
    from src.compiler import live_tiles
    owner = LiveTileArena(tile_rows=16, hot_tiles=32, bucket_rows=16, directory=tmp_path)
    source = [(i % 7, i) for i in reversed(range(200))]
    values = owner.manage(source)
    owner.offload_to_disk()
    calls = []
    def sort_key(value):
        calls.append(value[1])
        return value[0]
    def bounded_sort(run, **kwargs):
        assert len(run) <= owner.bucket_rows
        return builtins.sorted(run, **kwargs)
    monkeypatch.setattr(live_tiles, "sorted", bounded_sort, raising=False)
    values.sort(key=sort_key, reverse=reverse)
    assert list(values) == builtins.sorted(source, key=lambda value: value[0], reverse=reverse)
    assert calls == [member[1] for member in source]
    owner.collect_unused()
    assert list(values) == builtins.sorted(source, key=lambda value: value[0], reverse=reverse)


def test_fresh_source_compilation_forced_disk(tmp_path, monkeypatch):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
    )
    monkeypatch.setenv("TURING_COMPRESS_COLD_PAGES", "1")
    monkeypatch.setenv("TURING_SPILL_COLD_PAGES", "0")
    monkeypatch.setenv("TURING_MEMORY_BUDGET_BYTES", "1")
    monkeypatch.setenv("TURING_LIVE_TILE_ROWS", "8")
    monkeypatch.setenv("TURING_LIVE_RESIDENT_TILES", "2")
    monkeypatch.setenv("TURING_IDENTITY_SPILL_DIR", str(tmp_path))
    source = "def root(x):\n    y = x * x + 1.0\n    return y\n"
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="live_tile_source",
        extraction_contract=ExtractionContract("extraction_contracts/program_extraction.yaml"),
        runtime_closure_only=True, progress=lambda message: None,
    )
    owner = object.__getattribute__(module, "__dict__")["_tile_arena"]
    stats = owner.flush()
    assert stats.disk_tiles > 0 and stats.hot_tiles == 0
    assert stats.pressure_level == 3
    assert stats.resident_encoded_bytes == 0
    assert stats.retained_index_bytes > 0
    assert any(name.endswith(".ProcessGraphGLSLDeployment") for name in owner.categories)
    assert any(name.endswith(".ControlProgram") for name in owner.categories)
    assert any(name.endswith(".Instr") for name in owner.categories)
    assert "ProcessGraph.NetworkX" in owner.categories
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert not artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path / "native", optimization="O3")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["x"]
    namespace = {}
    exec(source, namespace)
    for x in (-2.5, 0.0, 3.25):
        execution = prepare_artifact_execution(artifact, {parameter: x}).run()
        actual = execution.buffers[outputs[root.name][0].id].reshape(-1)[0]
        assert actual == namespace["root"](x)


def test_identity_mapping_values_are_not_reinterpreted(tmp_path):
    owner = arena(tmp_path)
    source = {"parameter": [1, 2, 3], "other": [6, 7]}
    table = owner.manage(source)
    owner.offload_to_disk()
    assert list(table["parameter"]) == source["parameter"]
    assert list(table["other"]) == source["other"]
    graph = live_digraph(owner)
    graph.graph["identity_table"] = table
    graph.graph["program_abi"] = {"records": {"X": {"identity": "X"}}}
    owner.flush()
    assert list(graph.graph["identity_table"]["parameter"]) == [1, 2, 3]


def test_repeated_short_lived_records_do_not_alias_recycled_ids(tmp_path):
    owner = arena(tmp_path)
    root = TiledDict(owner)
    for i in range(500):
        root[i] = {"parents": [(i, "role")], "identity_table": {"x": (i,)}}
        if i % 29 == 0:
            owner.flush()
        for j in range(max(0, i - 3), i + 1):
            assert list(root[j]["parents"]) == [(j, "role")]
            assert root[j]["identity_table"]["x"] == (j,)
    owner.offload_to_disk()
    for i in range(500):
        assert list(root[i]["parents"]) == [(i, "role")]
        assert root[i]["identity_table"]["x"] == (i,)


def test_live_graph_footprint_includes_headers_and_remains_accessible(tmp_path):
    def footprint(value, seen):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            size += sum(footprint(k, seen) + footprint(v, seen) for k,v in value.items())
        elif isinstance(value, (list, tuple)):
            size += sum(footprint(v, seen) for v in value)
        return size
    eager = nx.DiGraph()
    for i in range(800):
        eager.add_node(i, op="Add", value_id=i, dtype="float64", shape=(),
                       parents=[(i - 1, "operand")], children=[(i + 1, "operand")],
                       attributes={"source_value_id": i, "role": "numerical", "region": i // 20},
                       source_span={"lineno": i + 1, "col_offset": 0})
    eager.add_edges_from((i, i + 1, {"role":"operand"}) for i in range(799))
    raw = footprint((eager._node, eager._succ, eager._pred), set())
    owner = LiveTileArena(tile_rows=32, hot_tiles=2, bucket_rows=32, directory=tmp_path)
    started = time.perf_counter()
    graph = live_digraph(owner)
    graph.add_nodes_from(eager.nodes(data=True))
    graph.add_edges_from(eager.edges(data=True))
    ram = owner.flush()
    disk = owner.offload_to_disk()
    total_ram = ram.resident_encoded_bytes + ram.owned_header_bytes
    print(f"LIVE_GRAPH_FOOTPRINT raw={raw} ram_tiles_and_headers={total_ram} "
          f"disk_headers={disk.owned_header_bytes} disk_payload={disk.disk_bytes} "
          f"hot_peak={disk.hot_peak} hot_limit=2 elapsed={time.perf_counter()-started:.3f}s")
    assert total_ram < raw
    assert disk.owned_header_bytes < total_ram
    assert graph.nodes[613]["attributes"]["source_value_id"] == 613
    assert nx.has_path(graph, 0, 799)


def test_authored_metadata_copier_keeps_source_identities_during_eviction(tmp_path):
    from src.compiler.native_compiler_accelerators import metadata_copy_source
    namespace = {}
    exec(metadata_copy_source(), namespace)
    owner = arena(tmp_path)
    source = owner.manage([{ "parents": [(i, "role")], "shape": ((i,),)} for i in range(100)])
    owner.offload_to_disk()
    memo = {}
    copy = namespace["metadata_copy_program"](source, memo)
    assert copy == list(source)
    assert memo[-id(source)] is source
    assert all(-identity in memo for identity in memo if identity > 0)


def test_writable_book_tiles_preserve_clock_provenance_and_registry(tmp_path):
    from src.compiler.identity_concordance import IdentityBook, Registry, Novel, Mode, Derived, RowField, RowFieldKind
    from src.compiler.live_tiles import tile_identity_page
    registry = Registry()
    declared = registry.declare_page("live_page", (RowField("scope", RowFieldKind.SCOPE),
                                                   RowField("id", RowFieldKind.VALUE_ID)), object)
    stage = registry.declare_stage("live_stage")
    origin = registry.declare_transform("live_origin", 0)
    book = IdentityBook(registry=registry)
    owner = arena(tmp_path)
    token = _ACTIVE_ARENA.set(owner)
    try:
        for i in range(100):
            ref = book.post(declared, ("scope", i), (i, "repeated " * 200), stage=stage,
                            provenance=Novel(origin, ()), mode=Mode.CONCORD)
        initial_clock = book.clock[0]
        page = book.page(declared)
        assert page.clock is book.clock
        owner.offload_to_disk()
        derived = book.post(declared, ("scope", 100), "derived", stage=stage,
                            provenance=Derived((ref,)), mode=Mode.CONCORD)
        assert book.clock[0] == initial_clock + 1
        assert book.edges_into(derived)[0][0].page is declared
        assert book.mint_of(ref)[0] is origin
        assert page.cells[(ref.row, ref.column)][0] == 99
        assert isinstance(page.cells, TiledDict)
        owner.flush()
        assert page.history(("scope", 100))[-1][1] == "derived"
        assert page._resolved(("scope", 99), book.clock[0]) is not None
        restored = pickle.loads(pickle.dumps(book))
        assert restored.page(restored.registry.pages[declared.name]).latest(("scope", 100)) == "derived"
    finally:
        _ACTIVE_ARENA.reset(token)


def test_explicit_api_export_handles_tile_backed_metadata(tmp_path):
    import yaml
    from src.compiler.compiled_program_api import CompiledProgramAPI
    owner = arena(tmp_path)
    metadata = owner.manage({"requirements": {"requests": [{"optional": False}], "slots": (1, 2)}})
    owner.offload_to_disk()
    api = CompiledProgramAPI("fixture", "fortran", None, metadata=metadata)
    assert yaml.safe_load(api.to_yaml())["metadata"] == {
        "requirements": {"requests": [{"optional":False}], "slots": [1,2]}}


def test_compaction_reclaims_superseded_records_without_whole_rehydrate(tmp_path):
    import os
    import gc
    owner = arena(tmp_path)
    table = owner.manage({i: "repeat " * 100 for i in range(80)})
    owner.offload_to_disk()
    for version in range(10):
        table[20] = (version, "updated repeated text " * 100)
        owner.flush()
    old_path = owner._store.path
    before = os.path.getsize(old_path)
    owner.compact_disk()
    gc.collect()
    assert not os.path.exists(old_path)
    assert os.path.getsize(owner._store.path) < before
    assert table[20] == (9, "updated repeated text " * 100)
    assert table[79] == "repeat " * 100
    assert owner.stats().payload_hot_tiles == 1


def test_pressure_can_escalate_inside_storage_writes_without_stage_boundary(tmp_path, monkeypatch):
    from src.compiler import memory_regulation as regulation
    from src.compiler.identity_concordance import IdentityBook
    from src.compiler.work_contract import PRESETS, set_active_contract, active_contract
    from dataclasses import replace
    contract = replace(PRESETS["develop"], compress_cold_pages=True, spill_cold_pages=False,
                       live_tile_rows=4, live_resident_tiles=2, memory_budget_bytes=100)
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 50)
    previous = active_contract()
    set_active_contract(contract)
    try:
        book = IdentityBook()
        regulator, token, release = regulation.begin_regulation(book)
        try:
            assert regulator.live_arena is None
            monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 200)
            regulator._last_pressure_read = 0
            graph = live_digraph()
            graph.add_nodes_from((i, {"payload": i}) for i in range(200))
            regulator.pressure_checkpoint("fixture pressure")
            assert regulator.live_arena.disk_mode
            assert regulator.live_arena.stats().disk_tiles > 0
            assert graph.nodes[187]["payload"] == 187
        finally:
            regulation.end_regulation(regulator, token, releases_book_state=release)
    finally:
        set_active_contract(previous)


def test_decoded_cache_obeys_byte_limit_not_only_tile_count(tmp_path):
    owner = LiveTileArena(tile_rows=4, hot_tiles=1000, bucket_rows=4,
                          decoded_budget_bytes=32000, directory=tmp_path)
    table = owner.manage({i: str(i) + "payload " * 300 for i in range(200)})
    owner.offload_to_disk()
    for i in range(200):
        assert table[i].startswith(str(i))
        assert owner.stats().payload_decoded_bytes <= owner.decoded_budget_bytes
    assert owner.stats().payload_hot_tiles < owner.hot_limit


def test_failed_compile_setup_restores_active_storage_context(monkeypatch):
    from dataclasses import replace
    from src.compiler import memory_regulation as regulation, live_tiles
    from src.compiler.identity_concordance import IdentityBook
    from src.compiler.work_contract import PRESETS, active_contract, set_active_contract
    previous_contract = active_contract()
    previous_regulator = regulation._ACTIVE.get()
    previous_arena = _ACTIVE_ARENA.get()
    set_active_contract(replace(PRESETS["develop"], compress_cold_pages=True,
                               spill_cold_pages=False))
    def fail(page, owner):
        raise OSError("fixture tile setup failure")
    monkeypatch.setattr(live_tiles, "tile_identity_page", fail)
    regulator = token = None
    try:
        book = IdentityBook()
        regulator, token, release = regulation.begin_regulation(book)
        monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 20 << 30)
        with pytest.raises(OSError, match="tile setup failure"):
            regulator.pressure_checkpoint("fixture activation")
        assert regulation._ACTIVE.get() is regulator
        assert _ACTIVE_ARENA.get() is previous_arena
        assert "_storage_pressure_activated" not in book.__dict__
    finally:
        if regulator is not None and regulation._ACTIVE.get() is regulator:
            regulator.budget_bytes = None
            regulation.end_regulation(regulator, token, releases_book_state=False)
        set_active_contract(previous_contract)
