"""Real source ingestion, configured graph scheduling and standalone C."""
import subprocess

import numpy as np
import pytest

from src.compiler.concordance_declarations import (
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_POLICY,
    SOURCE_ACTIVITY_SCHEDULE,
)
from src.compiler.control_activity import ActivitySchedule, conditional_memberships, binding
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
from tests.test_source_c_threads import CASES
from tests.test_source_ssa_synchronization import CONTROL_OWNERSHIP_SOURCE, CONTRACT


@pytest.fixture(scope='module')
def compilations():
    results = {}
    for name, source in (('threads', CASES['two_workers']), ('control', CONTROL_OWNERSHIP_SOURCE)):
        for preference in ('asap', 'alap'):
            captured = []
            module, _, _ = lower_ast_source_to_ssa(
                source, 'root', extraction_contract=CONTRACT, schedule_preference=preference,
                resolved_process_graph_sink=captured.append,
                progress=lambda _: None, progress_bars=False,
            )
            graphs = {entry.graph.G.graph['function_name']: entry.graph
                      for entry in captured[0].function_table if getattr(entry, 'graph', None) is not None}
            results[name, preference] = module, graphs
    return results


def test_configurations_preserve_the_same_ingested_activity_edges(compilations):
    for name in ('threads', 'control'):
        snapshots = []
        for preference in ('asap', 'alap'):
            module, _graphs = compilations[name, preference]
            book = module.metadata['identity_book']
            snapshots.append(frozenset(book.page(SOURCE_ACTIVITY_EDGE).rows()))
            assert set(book.page(SOURCE_ACTIVITY_POLICY).latest(row)
                       for row in book.page(SOURCE_ACTIVITY_POLICY).rows()) == {preference}
            for row in book.page(SOURCE_ACTIVITY_SCHEDULE).rows():
                assert row[1] == preference
                levels = dict(book.page(SOURCE_ACTIVITY_SCHEDULE).latest(row)[0])
                for scope, before, after, relation in snapshots[-1]:
                    assert before.row[0] == after.row[0] == scope
                    if scope == row[0] and relation not in {'repeat', 'unadmitted'}:
                        assert levels[before] < levels[after]
            assert '0 finding(s)' in concordance_report(module)
        assert snapshots[0] == snapshots[1]


def test_worker_bodies_are_not_expanded_into_parent_activity(compilations):
    module, graphs = compilations['threads', 'asap']
    book = module.metadata['identity_book']
    root = ActivitySchedule(graphs['root'], book=book)
    root_occurrences = {cell.row[1] for cell in root.levels}
    for name in ('first', 'second'):
        graph = graphs[name]
        worker = ActivitySchedule(graph, book=book)
        assert worker.scope != root.scope
        for node in graph.G:
            contract, _cell = binding(graph, node, book=book)
            if contract is not None:
                kind = book.page(SOURCE_ACTIVITY_NODE).cell(contract[0].row, contract[0].column)[0]
                if kind not in {'FunctionDef', 'AsyncFunctionDef'}:
                    assert contract[0].row[1] not in root_occurrences


def test_branch_membership_reader_uses_no_ast_metadata(compilations):
    module, graphs = compilations['control', 'alap']
    graph = graphs['root']
    book = module.metadata['identity_book']
    before = conditional_memberships(graph, book=book)
    assert before
    saved = []
    for data in (graph.G.graph, *(data for _, data in graph.G.nodes(data=True))):
        for key in ('expr_obj', 'authored_expr_obj', 'function_body', 'source_control_records', 'source_span'):
            if key in data:
                saved.append((data, key, data.pop(key)))
    try:
        assert conditional_memberships(graph, book=book) == before
        # This verifies the migrated reader, not the still-unconverted
        # legacy conditional/loop builders elsewhere in the compiler.
        schedule = ActivitySchedule(graph, book=book)
        assert schedule.levels
    finally:
        for data, key, value in saved:
            data[key] = value


@pytest.mark.parametrize('preference', ['asap', 'alap'])
def test_both_schedules_materialize_the_two_workers_as_standalone_c(compilations, tmp_path, preference):
    module, _graphs = compilations['threads', preference]
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / preference, {})
    process = subprocess.run([str(executable.executable_path), '2', '1'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += data.nbytes
        if role == 'output':
            outputs.extend(data.tolist())
    assert outputs == [2341]
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('preference', ['asap', 'alap'])
def test_sealed_unit_keeps_storage_fence_and_later_read_without_graph_ast(tmp_path, preference):
    from src.compiler.fortran_c_shell import lower_resolved_process_graph_unit_to_ssa
    source = '''from src.compiler.synchronization import Scope, MemorySemantics, memory_barrier
def root():
    shared = [3]
    memory_barrier(Scope.THREAD_REGION, MemorySemantics.RELEASE | 0, shared)
    return shared[0] + 9
'''
    captured = []
    lower_ast_source_to_ssa(
        source, 'root', extraction_contract=CONTRACT, resolved_process_graph_sink=captured.append,
        stop_after_compilation_unit_plan=True, progress=lambda _: None, progress_bars=False,
    )
    graph = captured[0]
    references = []
    for entry in graph.function_table:
        body = getattr(entry, 'graph', None)
        if body is None:
            continue
        if body.G.graph.get('function_name') == 'root':
            references.append(int(entry.reference.address))
        # The accepted source graph is the input. No surrogate graph or C
        # implementation is substituted for the authored storage operations.
        for data in (body.G.graph, *(data for _, data in body.G.nodes(data=True))):
            for key in ('expr_obj', 'authored_expr_obj', 'function_body', 'source_control_records', 'source_span'):
                data.pop(key, None)
    module, _, _ = lower_resolved_process_graph_unit_to_ssa(
        graph, tuple(references), schedule_preference=preference,
        progress=lambda _: None, progress_bars=False,
    )
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / ('sealed_' + preference), {})
    process = subprocess.run([str(executable.executable_path), '2', '1'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += data.nbytes
        if role == 'output':
            outputs.extend(data.tolist())
    assert outputs == [12]
    assert '0 finding(s)' in concordance_report(module)


def test_process_operations_are_control_ir_at_the_post_ingestion_cut():
    from src.compiler.concordance_declarations import (
        SOURCE_CONTROL_IR, SOURCE_CONTROL_OPERATOR, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT,
        DEPLOYMENT_REGION, ControlBlockKind,
    )
    captured = []
    lower_ast_source_to_ssa(
        CONTROL_OWNERSHIP_SOURCE, 'root', extraction_contract=CONTRACT,
        resolved_process_graph_sink=captured.append, stop_after_compilation_unit_plan=True,
        progress=lambda _: None, progress_bars=False,
    )
    graph = captured[0]
    book = graph.G.graph['compilation_unit_identity_book']
    root = next(entry.graph for entry in graph.function_table if getattr(entry, 'graph', None) is not None
                and entry.graph.G.graph.get('function_name') == 'root')
    scope = root.G.graph['lexical_read_scope']
    assert book.latest_ref(SOURCE_CONTROL_IR, (scope,)) is not None
    assert not tuple(book.page(DEPLOYMENT_REGION).rows())
    controls = []
    for row in book.page(SOURCE_CONTROL_OPERATOR).scope_rows(scope):
        program, control, placement, block = book.page(SOURCE_CONTROL_OPERATOR).latest(row)
        assert control.page == CONTROL_BLOCK
        assert book.page(CONTROL_BLOCK).cell(control.row, control.column).kind == ControlBlockKind.DISPATCH
        assert placement.page == CONTROL_BLOCK_PLACEMENT
        assert placement.row[1] == control
        assert block.operand_graph_cell is not None
        assert root.G.nodes[row[1]]['type'] != 'Call'
        controls.append((block.operation, book.page(CONTROL_BLOCK_PLACEMENT).cell(placement.row, placement.column)))
    assert [operation for operation, _placement in controls].count('sync_memory_barrier') == 2
    barriers = [placement for operation, placement in controls if operation == 'sync_memory_barrier']
    assert {placement.arm.value for placement in barriers} == {'body', 'orelse'}
    assert len({placement.parent for placement in barriers}) == 1
    assert book.page(CONTROL_BLOCK).cell(barriers[0].parent.row, barriers[0].parent.column).kind == ControlBlockKind.SOURCE_REGION


@pytest.mark.parametrize('preference', ['asap', 'alap'])
def test_worker_while_inside_try_uses_ingested_exception_nodes_and_runs_c(tmp_path, preference):
    from src.compiler.concordance_declarations import (
        SOURCE_EXCEPTION_INGESTION, SOURCE_EXCEPTION_NODE, SOURCE_EXCEPTION_OPERATION,
        SOURCE_CONTROL_OPERATOR,
    )
    source = '''import threading
def root(n: int) -> int:
    shared = [0]
    def worker():
        total = 0
        i = 0
        try:
            while i < 4:
                total = total + 12 // (i - n)
                i = i + 1
            else:
                total = total + 7
        except ZeroDivisionError:
            total = total + 50
        finally:
            total = total + 100
        shared[0] = total
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return shared[0]
'''
    captured = []
    module, _, _ = lower_ast_source_to_ssa(
        source, 'root', extraction_contract=CONTRACT, schedule_preference=preference,
        resolved_process_graph_sink=captured.append, progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata['identity_book']
    assert len(book.page(SOURCE_EXCEPTION_INGESTION).rows()) == 1
    operations = tuple(book.page(SOURCE_EXCEPTION_OPERATION).rows())
    assert operations
    for row in operations:
        source_node = row[0]
        conversion = book.latest_ref(SOURCE_EXCEPTION_NODE, (source_node,))
        assert conversion is not None
        assert book.page(SOURCE_EXCEPTION_NODE).cell(conversion.row, conversion.column).page == SOURCE_EXCEPTION_INGESTION
    early_operations = [book.page(SOURCE_CONTROL_OPERATOR).latest(row)[3]
                        for row in book.page(SOURCE_CONTROL_OPERATOR).rows()
                        if book.page(SOURCE_CONTROL_OPERATOR).latest(row)[3].operation.startswith('exception_')]
    assert early_operations
    assert all(block.exception_operation_cell is not None for block in early_operations)
    # Diagnostic AST can be retained, but the withdrawn synthetic scheduling
    # overrides must not return with exception ingestion.
    for entry in captured[0].function_table:
        if getattr(entry, 'graph', None) is None:
            continue
        for _, data in entry.graph.G.nodes(data=True):
            expression = data.get('expr_obj')
            assert not hasattr(expression, '_turing_exception_position')
            assert not hasattr(expression, '_turing_exception_completion')
    assert '0 finding(s)' in concordance_report(module)
    root = module.functions['root__root']
    parameter = dict(root.metadata['parameter_names'])['n']
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    namespace = {}
    exec(source, namespace)
    for n in (0, 2, 5):
        executable = artifact.compile_standalone(tmp_path / str(n), {parameter: np.array([n], dtype=np.int64)})
        process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                                 capture_output=True, text=True, timeout=10)
        assert process.returncode == 0, process.stderr
        raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
        for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
            count = int(np.prod(shape)) if shape else 1
            values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
            offset += values.nbytes
            if role == 'output':
                outputs.extend(values.tolist())
        assert outputs == [namespace['root'](n)]
    assert '0 finding(s)' in concordance_report(module)
