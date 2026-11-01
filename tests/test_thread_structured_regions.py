"""Structured authored workers retain dual IR through standalone C lowering."""
import subprocess

import networkx as nx
import numpy as np
import pytest

from src.compiler.concordance_declarations import (
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_THREAD_REGION_MEMBER,
    SOURCE_THREAD_REGION_EDGE, SOURCE_THREAD_REGION_CONTROL, SOURCE_CONTROL_RENDEZVOUS,
)
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from tests.test_thread_dual_ir_sync_contract import _compile, BARRIER_SOURCE


STRUCTURED_SOURCE = '''import threading
def root(n: int):
    shared = [0, 0]
    def worker(index: int):
        total = 0
        i = 0
        try:
            while i < 4:
                if i == 1:
                    total += 12 // (i - n)
                else:
                    total += i
                i += 1
        except ZeroDivisionError:
            total += 50
        finally:
            total += 100
        shared[index] = total
    a = threading.Thread(target=worker, args=(0,))
    b = threading.Thread(target=worker, args=(1,))
    a.start()
    b.start()
    a.join()
    b.join()
    return shared[0] * 1000 + shared[1]
'''


CONDITIONAL_SOURCE = '''import threading
def root(flag: int):
    shared = [0]
    def worker():
        shared[0] = 23
    if flag > 0:
        task = threading.Thread(target=worker)
        task.start()
        task.join()
    else:
        shared[0] = 7
    return shared[0]
'''


@pytest.fixture(scope='module')
def structured():
    return _compile(STRUCTURED_SOURCE)


def test_nested_region_owners_and_repeat_edges_are_copied(structured):
    module, graphs = structured
    book = module.metadata['identity_book']
    region, = graphs['root'].G.graph['thread_regions']
    scope = region.control_graph.G.graph['thread_region_scope']
    members = book.page(SOURCE_THREAD_REGION_MEMBER)
    for token in region.worker_tokens:
        ports = region.worker_activity_nodes(token)
        assert ports
        sources = {members.latest((scope, node))[0]: node for node in ports}
        assert any(book.page(SOURCE_ACTIVITY_NODE).cell(point.row, point.column)[1] for point in sources)
        repeats = [row for row in book.page(SOURCE_ACTIVITY_EDGE).rows()
                   if row[1] in sources and row[2] in sources and row[3] == 'repeat']
        assert repeats
        for row in repeats:
            source_edge = book.latest_ref(SOURCE_ACTIVITY_EDGE, row)
            matches = [fact for edge in book.page(SOURCE_THREAD_REGION_EDGE).scope_rows(scope)
                       if edge[1] == 'control'
                       for fact in [book.page(SOURCE_THREAD_REGION_EDGE).latest(edge)]
                       if source_edge in fact[1] and (fact[0] == 'repeat' or 'repeat' in fact[0])]
            assert matches
        assert all(book.latest_ref(SOURCE_THREAD_REGION_CONTROL, (scope, node)) is not None
                   for node in region.worker_nodes(token)
                   if region.control_graph.G.nodes[node]['attributes'].get('control_projection') is not None)
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('preference', ['asap', 'alap'])
def test_structured_workers_schedule_without_cross_worker_order(structured, preference):
    module, graphs = structured
    region, = graphs['root'].G.graph['thread_regions']
    before = tuple(tuple(graph.G.edges) for graph in (region.dependency_graph, region.control_graph))
    first, second = region.worker_tokens
    for half, levels in zip(('dependency', 'control'), region.schedule(preference)):
        graph = region.scheduling_graph(half)
        assert all(levels[a] < levels[b] for a, b in graph.edges)
        left, right = set(region.worker_nodes(first)), set(region.worker_nodes(second))
        assert not any(right & nx.descendants(graph, node) for node in left)
        assert not any(left & nx.descendants(graph, node) for node in right)
    assert before == tuple(tuple(graph.G.edges) for graph in (region.dependency_graph, region.control_graph))
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('n', [1, 5])
def test_nested_workers_execute_as_standalone_c(structured, tmp_path, n):
    module, _graphs = structured
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    parameter = dict(module.functions['root__root'].metadata['parameter_names'])['n']
    executable = artifact.compile_standalone(tmp_path / str(n), {parameter: np.array([n], dtype=np.int64)})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    namespace = {}
    exec(STRUCTURED_SOURCE, namespace)
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += values.nbytes
        if role == 'output':
            outputs.extend(values.tolist())
    assert outputs == [namespace['root'](n)]


def test_two_authored_barrier_phases_use_separate_rendezvous():
    source = BARRIER_SOURCE.replace('        gate.wait()', '        gate.wait()\n        gate.wait()')
    source = source.replace('    gate.wait()\n    first.join()', '    gate.wait()\n    gate.wait()\n    first.join()')
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    book = module.metadata['identity_book']
    scope = region.control_graph.G.graph['thread_region_scope']
    rows = tuple(book.page(SOURCE_CONTROL_RENDEZVOUS).scope_rows(scope))
    assert {row[2] for row in rows} == {0, 1}
    assert all(len(book.page(SOURCE_CONTROL_RENDEZVOUS).latest(row)[0]) == 2 for row in rows)
    for preference in ('asap', 'alap'):
        region.schedule(preference)
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('source', [BARRIER_SOURCE, STRUCTURED_SOURCE])
def test_region_reader_requires_no_diagnostic_ast(monkeypatch, source):
    import src.compiler.thread_regions as implementation
    original = implementation.ingest_thread_regions
    visited = []

    def without_diagnostics(graph):
        saved = []
        for entry in graph.function_table:
            body = getattr(entry, 'graph', None)
            if body is None:
                continue
            for data in (body.G.graph, *(data for _, data in body.G.nodes(data=True))):
                for key in ('expr_obj', 'authored_expr_obj', 'function_body', 'source_control_records', 'source_span'):
                    if key in data:
                        saved.append((data, key, data.pop(key)))
        try:
            result = original(graph)
            visited.extend(result)
            return result
        finally:
            for data, key, value in saved:
                data[key] = value

    monkeypatch.setattr(implementation, 'ingest_thread_regions', without_diagnostics)
    module, graphs = _compile(source)
    assert visited and graphs['root'].G.graph['thread_regions']
    visited[0].schedule('alap')
    assert '0 finding(s)' in concordance_report(module)


def test_repeated_join_keeps_each_completion_boundary():
    source = BARRIER_SOURCE.replace('    first.join()', '    first.join()\n    first.join()')
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    token, = region.worker_tokens
    root_ref = graphs['root'].G.graph['function_ref']
    joins = [node for node, data in graphs['root'].G.nodes(data=True)
             if (data.get('attributes', {}).get('dispatch_operation') or {}).get('operation') == 'thread_join']
    assert len(joins) == 2
    for join in joins:
        boundary = region.member_ids[(None, root_ref, join)]
        assert region.control_graph.G.nodes[boundary]['attributes']['dispatch_worker_token'] == token
        assert all(region.control_graph.G.has_edge(node, boundary) for node in region.worker_nodes(token))
    region.schedule('asap')
    assert '0 finding(s)' in concordance_report(module)


def test_loop_factory_retains_iteration_and_worker_contexts():
    source = '''import threading
def root():
    shared = [0, 0, 0]
    def worker(index: int):
        shared[index] = index + 11
    i = 0
    while i < 3:
        task = threading.Thread(target=worker, args=(i,))
        task.start()
        task.join()
        i += 1
    return shared[0] * 10000 + shared[1] * 100 + shared[2]
'''
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    assert len(region.worker_tokens) == 1  # one factory site; runtime instances are distinct
    assert any(data['attributes'].get('source_kind') == 'While'
               for _, data in region.control_graph.G.nodes(data=True))
    for preference in ('asap', 'alap'):
        region.schedule(preference)
    assert '0 finding(s)' in concordance_report(module)


def test_conditional_lifecycle_retains_exclusive_arm_ownership():
    source = CONDITIONAL_SOURCE
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    token, = region.worker_tokens
    root_ref = graphs['root'].G.graph['function_ref']
    create = region.activations[0][0]
    rid = region.member_ids[(None, root_ref, create)]
    projection = region.control_graph.G.nodes[rid]['attributes']['control_projection']
    book = module.metadata['identity_book']
    _program, _binding, enter, _leave = book.page(SOURCE_THREAD_REGION_CONTROL).cell(projection.row, projection.column)
    assert region.control_graph.G.nodes[enter]['attributes']['control_owners'][-1][1] == 'body'
    assert region.control_graph.G.nodes[rid]['attributes']['dispatch_worker_token'] == token
    for preference in ('asap', 'alap'):
        region.schedule(preference)
    assert '0 finding(s)' in concordance_report(module)


def test_barrier_does_not_enroll_an_unrelated_worker():
    source = '''import threading
def root():
    gate = threading.Barrier(2)
    shared = [0]
    def participant():
        gate.wait()
    def other():
        shared[0] = 17
    a = threading.Thread(target=participant)
    b = threading.Thread(target=other)
    a.start()
    b.start()
    gate.wait()
    a.join()
    b.join()
    return shared[0]
'''
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    other_ref = graphs['other'].G.graph['function_ref']
    other = next(token for token, (_create, _start, _join, target)
                 in zip(region.worker_tokens, region.activations) if target == other_ref)
    releases = [node for node, data in region.control_graph.G.nodes(data=True)
                if data['type'] == 'ControlBarrierRelease']
    release, = releases
    for half in ('dependency', 'control'):
        graph = region.scheduling_graph(half)
        assert all(not nx.has_path(graph, node, release) and not nx.has_path(graph, release, node)
                   for node in region.worker_nodes(other))
    region.schedule('alap')
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('flag', [-1, 1])
def test_conditional_worker_executes_the_selected_arm_in_c(tmp_path, flag):
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    module, _graphs = _compile(CONDITIONAL_SOURCE)
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    parameter = dict(module.functions['root__root'].metadata['parameter_names'])['flag']
    executable = artifact.compile_standalone(tmp_path / str(flag), {parameter: np.array([flag], dtype=np.int64)})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += values.nbytes
        if role == 'output':
            outputs.extend(values.tolist())
    namespace = {}
    exec(CONDITIONAL_SOURCE, namespace)
    assert outputs == [namespace['root'](flag)]


def test_missing_barrier_participant_does_not_become_a_global_barrier():
    module, graphs = _compile(BARRIER_SOURCE.replace('threading.Barrier(2)', 'threading.Barrier(3)'))
    region, = graphs['root'].G.graph['thread_regions']
    with pytest.raises(ValueError, match='barrier participant set or phase is unresolved'):
        region.schedule('alap')
    assert '0 finding(s)' in concordance_report(module)


def test_for_worker_keeps_iteration_regions_and_exception_effects():
    source = '''import threading
def root():
    shared = [0]
    def worker():
        total = 0
        for i in range(4):
            try:
                total = total + 12 // (i - 1)
            except ZeroDivisionError:
                total = total + 50
            finally:
                total = total + 1
        shared[0] = total
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return shared[0]
'''
    module, graphs = _compile(source)
    region, = graphs['root'].G.graph['thread_regions']
    token, = region.worker_tokens
    assert any(region.control_graph.G.nodes[node]['attributes']['source_kind'] == 'For'
               for node in region.worker_activity_nodes(token))
    for preference in ('asap', 'alap'):
        region.schedule(preference)
    assert '0 finding(s)' in concordance_report(module)
